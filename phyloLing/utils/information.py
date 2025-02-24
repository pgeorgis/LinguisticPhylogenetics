from math import log
from statistics import mean

from constants import GAP_CH_DEFAULT, PAD_CH_DEFAULT
from phonAlign import Alignment
from utils.phoneme_map import PhonemeMap
from utils.sequence import (Ngram, PhonEnvNgram, get_phonEnv_weight,
                            pad_sequence)
from utils.utils import normalize_dict


def pointwise_mutual_info(p_joint, p_x, p_y):
    return log(p_joint / (p_x * p_y), 2)


def bayes_pmi(pX_given_Y, pX):
    # via Bayes Theorem : pmi = log( p(x,y) / ( p(x) * p(y) ) ) = log( p(x|y) / p(x) )
    return log(pX_given_Y / pX, 2)


def surprisal(p):
    try:
        return -log(p, 2)
    except ValueError:
        raise ValueError(f'Math Domain Error: cannot take the log of {p}')


def surprisal_to_prob(s):
    return 2**-s


def adaptation_surprisal(alignment: Alignment,
                         surprisal_map: PhonemeMap,
                         ngram_size=1,
                         phon_env=False,
                         normalize=True,
                         pad_ch=PAD_CH_DEFAULT,
                         gap_ch=GAP_CH_DEFAULT,
                         ):
    """Calculates the surprisal of an aligned sequence, given a dictionary of
    surprisal values for the sequence corresponcences"""

    if phon_env:
        if alignment.phon_env_alignment is None:
            alignment.phon_env_alignment = alignment.add_phon_env()
        alignment = alignment.phon_env_alignment
    else:
        alignment = alignment.alignment
    length = len(alignment)
    pad_n = ngram_size - 1
    if ngram_size > 1:
        alignment = [(pad_ch, pad_ch)] * pad_n + alignment + [(pad_ch, pad_ch)] * pad_n

    values = []
    for i in range(pad_n, length + pad_n):
        ngram = alignment[i:i + ngram_size]
        segs = list(zip(*ngram))
        seg1, seg2 = segs
        # Convert to form stored in correspondence dictionaries,
        # whereby unigrams are strings and larger ngrams are tuples
        # Ngrams with phon env context:
        # (ngram, phon_env), e.g. ('v', '#|S|<') or (('<#', 'v'), '#|S|<')
        ngram1 = Ngram(seg1)
        # NB: I think handling below is now deprecated/unneeded given recent fixes
        # if ngram1.is_boundary(pad_ch):
        #     seg1 = ngram1.undo()  # e.g. ('#>',)
        #     if isinstance(seg1, tuple) and len(seg1) > 2:  # e.g. ('<#', 'v', '#|S|<')
        #         # e.g. ((('<#', 'v'), '#|S|<'),) -> ('<#', 'v', '#|S|<') -> (('<#', 'v'), '#|S|<')
        #         seg1 = PhonEnvNgram(seg1).ngram_w_context
        # elif phon_env and ngram1.string != gap_ch:
        phon_env_ngram1 = None
        if phon_env and ngram1.string != gap_ch:
            if ngram1.is_boundary(pad_ch) and ngram1.size == 1:  # e.g. ('#>',)
                seg1 = ngram1.undo()
            else:
                phon_env_ngram1 = PhonEnvNgram(seg1)
                seg1 = phon_env_ngram1.ngram_w_context
        else:
            seg1 = ngram1.undo()
        seg2 = Ngram(seg2).undo()
        if phon_env and phon_env_ngram1 is not None:
            # Interpolate over all lower-order phon envs
            weights, estimates = [], []
            for subcontext in phon_env_ngram1.list_subcontexts():
                subcontext_ngram = (Ngram(phon_env_ngram1.ngram).undo(), subcontext)
                estimate = surprisal_map.get_value(subcontext_ngram, seg2)
                estimates.append(estimate)
                weights.append(get_phonEnv_weight(subcontext))
            total_weights = sum(weights)
            weights = [weight/total_weights for weight in weights]
            estimate = sum([weight * estimate for weight, estimate in zip(weights, estimates)])
            values.append(estimate)
        else:
            values.append(surprisal_map.get_value(seg1, seg2))

    if normalize:
        return mean(values)
    else:
        return values



def marginalize_over_phon_env(phon_env_corr_map: PhonemeMap,
                              doculect,
                              from_surprisal=False,
                              from_counts=True,
                              to_surprisal=False,
                              normalize=False,
                              ) -> PhonemeMap:
    """Converts a phon env correspondence map into a vanilla correspondence map by marginalizing over phon envs"""
    vanilla_corr_map = PhonemeMap(0)
    for phon_env_ngram in phon_env_corr_map.get_primary_keys():
        ngram1, _ = phon_env_ngram if len(phon_env_ngram) == 2 else (phon_env_ngram, '')
        ngram_size = Ngram(ngram1).size
        phon_env_ngrams_per_size = doculect.phon_env_ngrams[ngram_size]
        if sum(phon_env_ngrams_per_size.values()) == 0:
            doculect.list_ngrams(ngram_size, phon_env=True)
        count = phon_env_ngrams_per_size.get(phon_env_ngram, 0)
        if count == 0:
            continue
        for ngram2, val in phon_env_corr_map.get_primary_key_map(phon_env_ngram).items():
            if from_surprisal:
                val = surprisal_to_prob(val)
                count_ngram2 = val * count
            elif not from_counts:
                count_ngram2 = val * count
            else:
                assert from_counts
                count_ngram2 = val
            if count_ngram2 > 0:
                vanilla_corr_map.increment_value(ngram1, ngram2, count_ngram2)
    for ngram1 in vanilla_corr_map.get_primary_keys():
        ngram1_map = vanilla_corr_map.get_primary_key_map(ngram1)
        if normalize or to_surprisal:
            inner_map = normalize_dict(ngram1_map)
        else:
            inner_map = ngram1_map
        for ngram2, val in inner_map.items():
            if to_surprisal:
                vanilla_val = surprisal(val)
            else:
                vanilla_val = val
            vanilla_corr_map.set_value(ngram1, ngram2, vanilla_val)
    return vanilla_corr_map


def entropy(x) -> float:
    """x should be a dictionary with absolute counts"""
    total = sum(x.values())
    e: float = 0
    for i in x:
        p: float = x[i] / total
        if p > 0:
            e += p * surprisal(p)
    return e
