from math import log
from statistics import mean

from constants import GAP_CH_DEFAULT, PAD_CH_DEFAULT
from utils.sequence import Ngram, PhonEnvNgram, pad_sequence


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


def adaptation_surprisal(alignment,
                         surprisal_dict,
                         ngram_size=1,
                         phon_env=False,
                         normalize=True,
                         pad_ch=PAD_CH_DEFAULT,
                         gap_ch=GAP_CH_DEFAULT,
                         ):
    """Calculates the surprisal of an aligned sequence, given a dictionary of
    surprisal values for the sequence corresponcences"""

    # if type(alignment) is Alignment:
    #     length = alignment.length
    #     alignment = alignment.alignment
    # elif type(alignment) is list:
    #     length = len(alignment)
    # else:
    #     raise TypeError
    # TODO problem: this function needs to live in this script to avoid circular imports
    # but importing Alignment object from phonAlign.py would also cause a circular import
    # Temp solution: assume that if the alignment is not a list, it is an Alignment class object
    if isinstance(alignment, list):
        length = len(alignment)
    else:
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
        if phon_env and ngram1.string != gap_ch:
            if ngram1.is_boundary(pad_ch) and ngram1.size == 1:  # e.g. ('#>',)
                seg1 = ngram1.undo()
            else:
                seg1 = PhonEnvNgram(seg1).ngram_w_context
        else:
            seg1 = ngram1.undo()
        seg2 = Ngram(seg2).undo()
        values.append(surprisal_dict[seg1][seg2])

    if normalize:
        return mean(values)
    else:
        return values


def calculate_infocontent_of_word(seq, lang, ngram_size=3, pad_ch=PAD_CH_DEFAULT):
    if len(seq) < ngram_size:
        add_pad_n = ngram_size-len(seq)
        seq = pad_sequence(seq, pad_n=add_pad_n, pad_ch=pad_ch)
    pad_n = ngram_size - 1
    info_content = {}
    for i in range(pad_n, len(seq) - pad_n):
        if ngram_size == 1:
            unigram_count = lang.unigrams.get(Ngram(seq[i]).ngram, 0)
            gappy_count = sum(lang.unigrams.values())
            info_content_value = (seq[i], -log(unigram_count / gappy_count, 2))
            info_content[i] = info_content_value
        elif ngram_size == 2:
            bigram_counts = 0
            if i > 0:
                bigram_counts += lang.bigrams.get((seq[i - 1], seq[i]), 0)
            if i < len(seq) - 1:
                bigram_counts += lang.bigrams.get((seq[i], seq[i + 1]), 0)
            gappy_counts = 0
            if i > 0:
                gappy_counts += lang.gappy_bigrams.get((seq[i - 1], 'X'), 0)
            if i < len(seq) - 1:
                gappy_counts += lang.gappy_bigrams.get(('X', seq[i + 1]), 0)
            try:
                info_content_value = (seq[i], -log(bigram_counts / gappy_counts, 2))
            except ZeroDivisionError:
                breakpoint()
            info_content[i] = info_content_value
        else: # ngram_size = 3
            trigram_counts = 0
            trigram_counts += lang.trigrams.get((seq[i - 2], seq[i - 1], seq[i]), 0)
            trigram_counts += lang.trigrams.get((seq[i - 1], seq[i], seq[i + 1]), 0)
            trigram_counts += lang.trigrams.get((seq[i], seq[i + 1], seq[i + 2]), 0)
            gappy_counts = 0
            gappy_counts += lang.gappy_trigrams.get((seq[i - 2], seq[i - 1], 'X'), 0)
            gappy_counts += lang.gappy_trigrams.get((seq[i - 1], 'X', seq[i + 1]), 0)
            gappy_counts += lang.gappy_trigrams.get(('X', seq[i + 1], seq[i + 2]), 0)
            # TODO : needs smoothing
            try:
                info_content_value = (seq[i], -log(trigram_counts / gappy_counts, 2))
                info_content[i - 2] = info_content_value
            except ValueError:
                breakpoint()
        
    return info_content


def entropy(x) -> float:
    """x should be a dictionary with absolute counts"""
    total = sum(x.values())
    e: float = 0
    for i in x:
        p: float = x[i] / total
        if p > 0:
            e += p * surprisal(p)
    return e
