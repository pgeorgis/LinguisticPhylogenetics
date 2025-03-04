import logging
import os
import random
from collections import defaultdict
from functools import lru_cache
from itertools import product
from math import inf, log
from statistics import mean, stdev
from typing import Self

import numpy as np
from constants import (GAP_CH_DEFAULT, PAD_CH_DEFAULT,
                       PHONE_CORRELATORS_INDEX_KEY, SEG_JOIN_CH)
from nltk.translate import AlignedSent, IBMModel1, IBMModel2
from phonAlign import Alignment
from phonUtils.phonSim import phone_sim
from phonUtils.segment import Segment, _toSegment
from scipy.stats import norm
from utils.alignment import (calculate_alignment_costs,
                             needleman_wunsch_extended)
from utils.distance import Distance
from utils.doculect import Doculect
from utils.information import (marginalize_over_phon_env,
                               pointwise_mutual_info, surprisal)
from utils.logging import (log_phon_corr_iteration, write_alignments_log,
                           write_phon_corr_iteration_log,
                           write_phon_corr_report, write_phoneme_pmi_report,
                           write_phoneme_surprisal_report, write_sample_log)
from utils.phoneme_map import (PhonemeMap, average_corrs, average_nested_dicts,
                               reverse_corr_dict_map)
from utils.sequence import (Ngram, PhonEnvNgram, end_token,
                            filter_out_invalid_ngrams, get_phonEnv_weight,
                            pad_sequence, start_token)
from utils.utils import (balanced_resample, create_default_dict, default_dict,
                         segment_ranges)
from utils.wordlist import Wordlist, sort_wordlist

logging.basicConfig(level=logging.INFO, format='%(asctime)s phonCorr %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Designate phonetic feature distance for alignment purposes
PROSODIC_UNIT_LABELS = {'TONEME', 'SUPRASEGMENTAL'}

@lru_cache(maxsize=None)
def is_prosodic_unit(segment):
    segment = _toSegment(segment) if not isinstance(segment, Segment) else segment
    if segment.phone_class in PROSODIC_UNIT_LABELS:
        return True
    return False

def phone_dist(x, y, **kwargs):
    if x == y:
        return 0
    # Free-standing tonemes and suprasegmentals get distance of 0 to each other
    # (for alignment purposes only)
    if is_prosodic_unit(x) and is_prosodic_unit(y):
        return 0
    sim = phone_sim(x, y, **kwargs)
    if sim > 0:
        return log(sim)
    return -inf
PhoneFeatureDist = Distance(phone_dist, name='PhoneFeatureDist')
PHON_GOP = -1.2 # approximately corresponds to log(0.3), i.e. insert gap if less than 30% phonetic similarity

def fit_ibm_align_model(corpus: list[tuple],
                        iterations: int=5,
                        gap_ch: str=GAP_CH_DEFAULT,
                        ibm_model: int=2,
                        seed:int =None,
                        ):
    """Fits an IBM alignment model on a corpus of word pairs and
    returns the aligned corpus, fitted model, and translation table of correspondences.

    Args:
        corpus (list[tuple]): list of (Word, Word) tuple pairs
        iterations (int, optional): Number of iterations to run EM algorithm.
        gap_ch (str, optional): Gap character.
        ibm_model (int, optional): IBM model.
        seed (int, optional): Random seed.

    Raises:
        ValueError: If an invalid IBM model is specified.

    Returns:
        tuple: (corpus, fit_model, translation_table)
    """
    if ibm_model == 1:
        ibm_model = IBMModel1
    elif ibm_model == 2:
        ibm_model = IBMModel2
    else:
        raise ValueError(f"Invalid IBM model: {ibm_model}")

    # Set random seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    corpus = [AlignedSent(word1, word2) for word1, word2 in corpus]
    fit_model = ibm_model(corpus, iterations)
    translation_table = fit_model.translation_table
    for seg1 in translation_table:
        translation_table[seg1][gap_ch] = translation_table[seg1].get(None, 0)
        if None in translation_table[seg1]:
            del translation_table[seg1][None]
    # Align full corpus using the fitted IBM model
    fit_model.align_all(corpus)

    # Return aligned corpus, fit_model, translation_table
    return corpus, fit_model, translation_table


def postprocess_ibm_alignment(aligned_pair, remove_non_sequential_complex_alignments=True):
    # Postprocess start-end boundary alignments
    alignment = postprocess_boundary_alignments(aligned_pair)
    seq1 = aligned_pair.words
    seq2 = aligned_pair.mots

    # Initialize alignment dicts
    aligned_seq1 = defaultdict(lambda:[])
    aligned_seq2 = defaultdict(lambda:[])

    # Fill in aligned dicts
    for idx1, idx2 in alignment:
        if idx1 is not None:
            aligned_seq1[idx1].append(idx2)
        if idx2 is not None:
            aligned_seq2[idx2].append(idx1)

    # Add gaps
    for idx1, _ in enumerate(seq1):
        if idx1 not in aligned_seq1:
            aligned_seq1[idx1].append(None)
    for idx2, _ in enumerate(seq2):
        if idx2 not in aligned_seq2:
            aligned_seq2[idx2].append(None)

    if remove_non_sequential_complex_alignments:
        """IBM alignment can produce alignments with units aligned in multiple places, e.g.
        [(0, 0), (1, 1), (2, 9), (3, 3), (4, 4), (5, 7), (6, 4), (7, 9), (8, 10)]
        where idx 9 in seq2 is aligned to both idx 2 and idx 7 in seq1

        Disallow such complex/double alignments if they are not consecutive
        """
        for j, aligned_idxs in aligned_seq2.items():
            aligned_idxs.sort()
            if len(aligned_idxs) > 1:
                offset = 0
                n_aligned_idxs = len(aligned_idxs)
                aligned_idxs_copy = aligned_idxs[:]
                while offset < n_aligned_idxs - 1:
                    i = aligned_idxs_copy[offset]
                    i_next = aligned_idxs_copy[offset + 1]
                    if abs(i - i_next) > 1:
                        anchor = mean(aligned_idxs + [j])
                        # Find which of the two is more distant from anchor
                        i_next_distance = abs(i_next - anchor)
                        i_distance = abs(i - anchor)
                        # Choose to remove the later of the two if both equally distant
                        # (to avoid non-deterministic choice using max)
                        more_distant_i = i_next if i_next_distance >= i_distance else i
                        aligned_seq2[j].remove(more_distant_i)
                        if len(aligned_seq1[more_distant_i]) > 1:
                            aligned_seq1[more_distant_i].remove(j)
                        else:
                            aligned_seq1[more_distant_i] = [None]
                    offset += 1

    return aligned_seq1, aligned_seq2


def postprocess_boundary_alignments(aligned_pair):
    """Correct alignment of a start boundary with an end boundary."""
    seq1_len = len(aligned_pair.words)
    seq2_len = len(aligned_pair.mots)
    alignment = list(aligned_pair.alignment)
    if (0, seq2_len-1) in alignment:
        idx = alignment.index((0, seq2_len-1))
        alignment[idx] = (0, 0)
    elif (seq1_len-1, 0) in alignment:
        idx = alignment.index((seq1_len-1, 0))
        alignment[idx] = (0, 0)
    alignment.sort(key=lambda x: (x[0], -float('inf') if x[-1] is None else x[-1]))
    return alignment


def prune_corrs(corr_map: PhonemeMap, min_val: int=2, exc1: set=None, exc2: set=None):
    # Prune correspondences below a minimum count/probability threshold
    for seg1 in corr_map.get_primary_keys():
        if exc1 and Ngram(seg1).string in exc1:
            continue
        seg2_to_del = [
            seg2
            for seg2 in corr_map.get_secondary_keys(seg1)
            if corr_map.get_value_or_default(seg1, seg2, 0) < min_val
        ]
        for seg2 in seg2_to_del:
            if exc2 and Ngram(seg2).string in exc2:
                continue
            corr_map.delete_value(seg1, seg2)
    # Delete empty seg1 entries
    seg1_to_del = [
        seg1
        for seg1 in corr_map.get_primary_keys()
        if len(corr_map.get_secondary_keys(seg1)) == 0
    ]
    for seg1 in seg1_to_del:
        corr_map.delete_primary_key(seg1)
    return corr_map


def prune_extraneous_synonyms(wordlist: Wordlist,
                              alignments: list,
                              family_index: dict,
                              scores: list=None,
                              maximize_score: bool=True,
                              ):
    # Resolve synonyms: prune redundant/extraneous
    # If a concept has >1 words listed, we may end up with, e.g.
    # DE <Kopf> - NL <kop>
    # DE <Haupt> - NL <hoofd>
    # but also
    # DE <Kopf> - NL <hoofd>
    # DE <Haupt> - NL <kop>
    # If both languages have >1 qualifying words for a concept, only consider the best pairings, i.e. prune the extraneous pairs
    if scores is None:
        scores = [alignment.cost for alignment in alignments]
    assert len(wordlist.word_pairs) == len(alignments) == len(scores)

    def score_is_better(score1, score2):
        # maximize_score : if True, the optimum alignment has a score score
        #                  if False, the optimum alignment has the lowest score (lowest cost))
        if maximize_score:
            if score1 > score2:
                return True
            return False
        return score1 < score2

    # Count instances of concepts to check for concepts with >1 word pair
    concept_counts = defaultdict(lambda: 0)
    concept_indices = defaultdict(lambda: [])
    tied_indices = defaultdict(lambda: set())
    indices_to_prune = set()
    for q, pair in enumerate(wordlist.word_pairs):
        word1, word2 = pair
        concept = word1.concept
        concept_counts[concept] += 1
        concept_indices[concept].append(q)

    # Find optimal mappings for each concept with >1 word pair
    for concept, count in concept_counts.items():
        if count > 1:
            best_pairings = {}
            best_pairing_scores = {}
            for index in concept_indices[concept]:
                word1, word2 = wordlist.word_pairs[index]
                score = scores[index]  # TODO seems like this could benefit from a Wordpair object
                if word1 not in best_pairings or score_is_better(score, best_pairing_scores[word1]):
                    best_pairings[word1] = index, word2
                    best_pairing_scores[word1] = score
                elif score == best_pairing_scores[word1]:
                    # In case of tied correspondence-based scores, consider the phonetic distance too
                    from wordDist import phonological_dist
                    best_index, word2_best = best_pairings[word1]
                    best_alignment = alignments[best_index]
                    current_alignment = alignments[index]
                    # TODO check that these word objects input to phonological_dist are the right ones
                    phon_dist_best = phonological_dist(word1, word2_best, alignment=best_alignment, family_index=family_index)
                    phon_dist_current = phonological_dist(word1, word2, alignment=current_alignment, family_index=family_index)
                    if maximize_score:
                        phon_dist_best *= -1
                        phon_dist_current *= -1
                    best_score = best_pairing_scores[word1] + phon_dist_best
                    current_score = score + phon_dist_current
                    if best_score == current_score:
                        # If still no distance between the two pairs, use both as there is no good way to select one
                        tied_indices[concept].update({index, best_index})
                    elif score_is_better(current_score, best_score):
                        best_pairings[word1] = index, word2
                        best_pairing_scores[word1] = score

            # Now best_pairings contains the best mapping for each concept
            # based on the best (highest/lowest) scoring pair wrt to first language word
            best_indices = [index for index, _ in best_pairings.values()]
            for index in concept_indices[concept]:
                if index not in best_indices and index not in tied_indices[concept]:
                    indices_to_prune.add(index)

            # Now check for multiple l1 words mapped to the same l2 word
            # Choose only the best of these, 
            # unless scores are tied, in which case all are kept
            selected_word2 = [wordlist.word_pairs[index][-1] for index in best_indices]
            if len(set(selected_word2)) < len(best_indices):
                for word2 in set(selected_word2):
                    indices = [index for index in best_indices if wordlist.word_pairs[index][-1] == word2]
                    best_score = max(scores[x] for x in indices) if maximize_score else min(scores[x] for x in indices)
                    best_choices = set(x for x in indices if scores[x] == best_score)
                    indices_to_prune.update([index for index in indices if index not in best_choices])

    # Then prune all suboptimal word pair indices
    indices_to_prune = sorted(list(indices_to_prune), reverse=True)
    for index in indices_to_prune:
        wordlist.remove_by_index(idx=index)
        del alignments[index]

    return wordlist, alignments


class PhonCorrelator:
    def __init__(self,
                 lang1,
                 lang2,
                 wordlist=None,
                 gap_ch=GAP_CH_DEFAULT,
                 pad_ch=PAD_CH_DEFAULT,
                 seed=1,
                 log_outdir=None,
                 ):
        # Set Doculect objects
        self.lang1 = lang1
        self.lang1_name = lang1.name
        self.lang2 = lang2
        self.lang2_name = lang2.name

        # Alignment parameters
        self.gap_ch = gap_ch
        self.pad_ch = pad_ch
        self.seed = seed

        # Prepare wordlists: sort out same/different-meaning words and loanwords
        self.input_wordlist = tuple(wordlist) if wordlist is not None else wordlist  # used for initializing twin, needs to be None of wordlist input arg was also None
        self.wordlist = self.get_concept_list(wordlist)
        self.same_meaning = None
        self.diff_meaning = None
        self.loanwords = None
        self.samples = {}

        # PMI and surprisal results
        self.low_coverage_phones = None
        self.pmi_results: PhonemeMap = PhonemeMap()
        self.surprisal_results = create_default_dict(self.lang2.phoneme_entropy, 3)
        self.phon_env_surprisal_results = create_default_dict(self.lang2.phoneme_entropy, 3)
        
        # Non-cognate thresholds for calibration
        self.noncognate_thresholds: dict[(Distance, int, int), list] = defaultdict(list)

        # Logging output directories
        self.log_outdir = log_outdir if log_outdir else ""  # TODO revisit what default outdir path should be
        self.phon_corr_dir = os.path.join(self.log_outdir, self.lang1.path_name, self.lang2.path_name)
        os.makedirs(self.phon_corr_dir, exist_ok=True)
        self.align_log = create_default_dict(0, 2)

    def get_twin(self, phone_correlators_index) -> Self:
        """Retrieve the twin PhonCorrelator object for the reverse direction of the same language pair."""
        if self.lang1_name == self.lang2_name:
            return self, phone_correlators_index
        twin_correlator, phone_correlators_index = get_phone_correlator(
            self.lang2,
            self.lang1,
            phone_correlators_index=phone_correlators_index,
            wordlist=self.input_wordlist,
            seed=self.seed,
            log_outdir=self.log_outdir,
        )
        return twin_correlator, phone_correlators_index

    def reset_seed(self):
        random.seed(self.seed)

    def get_concept_list(self, wordlist=None):
        # If no wordlist is provided, by default use all concepts shared by the two languages
        if wordlist is None:
            wordlist = self.lang1.vocabulary.keys() & self.lang2.vocabulary.keys()

        # If a wordlist is provided, use only the concepts shared by both languages
        else:
            wordlist = set(wordlist) & self.lang1.vocabulary.keys() & self.lang2.vocabulary.keys()

        return wordlist

    def prepare_wordlists(self):
        # Check if wordlists were already previously generated, if so skip
        if self.wordlists_created() is True:
            return

        # Get lexical items in each language belonging to the specified wordlist
        l1_wordlist = [word for concept in self.wordlist for word in self.lang1.vocabulary[concept]]
        l2_wordlist = [word for concept in self.wordlist for word in self.lang2.vocabulary[concept]]

        # Get all unique combinations of L1 and L2 word
        # Sort the wordlists in order to ensure that random samples of same/different meaning pairs are reproducible
        all_wordpairs = sort_wordlist(product(l1_wordlist, l2_wordlist))

        # Sort out same-meaning from different-meaning word pairs, and loanwords
        same_meaning, diff_meaning, loanwords = [], [], []
        for pair in all_wordpairs:
            word1, word2 = pair
            concept1, concept2 = word1.concept, word2.concept
            if concept1 == concept2:
                if word1.loanword or word2.loanword:
                    loanwords.append(pair)
                else:
                    same_meaning.append(pair)
            else:
                diff_meaning.append(pair)

        # Set the three word type lists
        self.same_meaning = same_meaning
        self.diff_meaning = diff_meaning
        self.loanwords = loanwords
    
    def wordlists_created(self):
        """Returns True if wordlists have already been prepared, else false."""
        if any(_wordlist is None for _wordlist in (self.same_meaning, self.diff_meaning, self.loanwords)):
            return False
        return True

    def sample_wordlists(self, n_samples, sample_size, start_seed=None, log_outfile='samples.log'):
        # Take N samples of same- and different-meaning words
        if start_seed is None:
            start_seed = self.seed

        samples = self.samples
        new_samples = False
        if log_outfile:
            sample_logs = {}

        # Track how many times each index has been sampled
        same_meaning_count = np.zeros(len(self.same_meaning))
        diff_meaning_count = np.zeros(len(self.diff_meaning))
        diff_n = min(sample_size, len(self.diff_meaning))
        for sample_n in range(n_samples):
            seed_i = start_seed + sample_n

            # Skip previously saved samples
            if (seed_i, sample_size) in samples:
                continue

            # Draw new samples if not yet done
            new_samples = True

            # Initialize random number generator with seed
            rng = np.random.default_rng(seed_i)

            # Take balanced resampling of same-meaning words
            synonym_sample, same_meaning_count = balanced_resample(
                self.same_meaning, sample_size, same_meaning_count, rng
            )

            # Take a sample of different-meaning words, as large as the same-meaning set
            diff_sample, diff_meaning_count = balanced_resample(
                self.diff_meaning, diff_n, diff_meaning_count, rng
            )

            # Record samples
            samples[(seed_i, sample_size)] = (synonym_sample, diff_sample)

            # Log samples
            if log_outfile:
                same_meaning_sample_log = self.log_sample(
                    synonym_sample, sample_n, label="synonym", seed=seed_i
                )
                diff_meaning_sample_log = self.log_sample(
                    diff_sample, sample_n, label="different meaning", seed=seed_i
                )
                sample_logs[sample_n] = (same_meaning_sample_log, diff_meaning_sample_log)

        # Update dictionary of samples
        if new_samples:
            self.samples.update(samples)

        # Write sample log (only if new samples were drawn)
        if log_outfile and new_samples:
            sample_log_file = os.path.join(self.log_outdir, self.lang1.path_name, self.lang2.path_name, log_outfile)
            write_sample_log(sample_logs, sample_log_file)

        return samples

    def pad_wordlist(self, wordlist, pad_n=1):
        def _pad(seq):
            return pad_sequence(seq, pad_ch=self.pad_ch, pad_n=pad_n)
        return [map(_pad, word) for word in wordlist]

    def align_wordlist(self,
                       wordlist: Wordlist | list,
                       align_costs: PhonemeMap = None,
                       remove_uncompacted_padding: bool=True,
                       add_phon_dist: bool=True,
                       # phon_env=False,
                       **kwargs):
        """Returns a list of the aligned segments from the wordlists"""

        # Optionally add phone similarity measure between phone pairs to align costs/scores
        if add_phon_dist and align_costs.phon_dist_added is False:
            for ngram1, ngram2 in align_costs.get_key_pairs():
                ngram1 = Ngram(ngram1)
                ngram2 = Ngram(ngram2)

                # Remove gaps and boundaries
                gapless_ngram1 = ngram1.remove_boundaries(self.pad_ch).remove_gaps(self.gap_ch)
                gapless_ngram2 = ngram2.remove_boundaries(self.pad_ch).remove_gaps(self.gap_ch)

                # Skip computing phonetic similarity between gaps/boundaries with each other
                if gapless_ngram1.size == 0 and gapless_ngram2.size == 0:
                    continue
                # For full gaps/boundaries with segments, assign cost as GOP * length of segment sequence
                elif gapless_ngram1.size == 0 or gapless_ngram2.size == 0:
                    phon_align_cost = PHON_GOP * max(gapless_ngram1.size, gapless_ngram2.size)
                # Directly compute phonetic distance of two single phones
                elif gapless_ngram1.size == 1 and gapless_ngram2.size == 1:
                    phon_align_cost = PhoneFeatureDist.eval(gapless_ngram1.string, gapless_ngram2.string)
                # Else compute phonetic distance of the two aligned ngram sequences
                else:
                    phon_costs: PhonemeMap = calculate_alignment_costs(
                        gapless_ngram1.ngram,
                        gapless_ngram2.ngram,
                        cost_func=PhoneFeatureDist,
                        as_indices=False
                    )
                    phon_align_cost, _, _ = needleman_wunsch_extended(
                        gapless_ngram1.ngram,
                        gapless_ngram2.ngram,
                        align_cost=phon_costs,
                        gap_cost=PhonemeMap(),
                        default_gop=PHON_GOP,
                        maximize_score=True,
                        gap_ch=self.gap_ch,
                        allow_complex=False
                    )
                # Add phonetic alignment cost to align_costs dict storing PMI values
                base_align_cost = align_costs.get_value_or_default(ngram1.undo(), ngram2.undo(), 0)
                align_costs.set_value(ngram1.undo(), ngram2.undo(), base_align_cost + phon_align_cost)
            align_costs.phon_dist_added = True

        word_pairs = wordlist.word_pairs if isinstance(wordlist, Wordlist) else wordlist
        alignment_list = [
            Alignment(
                seq1=word1,
                seq2=word2,
                lang1=self.lang1,
                lang2=self.lang2,
                align_costs=align_costs,
                gap_ch=self.gap_ch,
                pad_ch=self.pad_ch,
                **kwargs
            )
            for word1, word2 in word_pairs
        ]

        if remove_uncompacted_padding:
            for alignment in alignment_list:
                alignment.remove_padding()

        # if phon_env:
        #     for alignment in alignment_list:
        #         alignment.phon_env_alignment = alignment.add_phon_env()

        return alignment_list

    def get_possible_ngrams(self, lang: Doculect, ngram_size: int, phon_env=False) -> set:
        # Iterate over all possible/attested ngrams
        # Only perform calculation for ngrams which have actually been observed/attested to
        # in the current dataset or which could have been observed (with gaps)
        attested = lang.list_ngrams(ngram_size, phon_env=phon_env)
        if phon_env:
            return attested
        else:
            all_ngrams = set(product(
                list(attested.keys()) + [self.gap_ch],
                repeat=ngram_size
            ))
            all_ngrams = set(Ngram(ngram).ngram for ngram in all_ngrams)
            return all_ngrams

    def fit_radial_ibm_model(self,
                             sample: Wordlist,
                             lang1: Doculect,
                             lang2: Doculect,
                             phon_env=False, # TODO add
                             ibm_model: int=2,
                             max_ngram_size: int=2,
                             min_corr: int=2,
                             seed: int=None,
                             ):
        """Fits IBM translation models on ngram sequences of varying sizes and aggregates the translation tables."""

        # Create "corpora" consisting of words segmented into unigrams or bigrams
        corpus = []
        ngram_sizes = list(range(1, max_ngram_size + 1))
        for word1, word2 in sample.word_pairs:
            segs1, segs2 = word1.segments, word2.segments
            # Optionally add phon env
            if phon_env:
                raise NotImplementedError
                env1, env2 = word1.phon_env, word2.phon_env
                segs1 = zip(segs1, env1)
                segs1 = zip(segs2, env2)
            for ngram_size_i in ngram_sizes:
                ngrams1 = word1.get_ngrams(size=ngram_size_i, pad_ch=self.pad_ch)
                ngrams1 = filter_out_invalid_ngrams(ngrams1, language=self.lang1)
                if ngram_size_i > 1:
                    ngrams1 = [SEG_JOIN_CH.join(ngram) for ngram in ngrams1]
                for ngram_size_j in ngram_sizes:
                    ngrams2 = word2.get_ngrams(size=ngram_size_j, pad_ch=self.pad_ch)
                    ngrams2 = filter_out_invalid_ngrams(ngrams2, language=self.lang2)
                    if ngram_size_j > 1:
                        ngrams2 = [SEG_JOIN_CH.join(ngram) for ngram in ngrams2]
                    corpus.append((ngrams1, ngrams2))
        corpus, _, _ = fit_ibm_align_model(
            corpus,
            gap_ch=self.gap_ch,
            ibm_model=ibm_model,
            seed=seed
        )

        # Create corr dicts in each direction from aligned segments
        # TODO add new function for this for cleanliness
        corr_dict_l1l2 = PhonemeMap(0)
        corr_dict_l2l1 = PhonemeMap(0)
        for aligned_pair in corpus:
            aligned_seq1, aligned_seq2 = postprocess_ibm_alignment(aligned_pair)
            for idx1, seq2corrs in aligned_seq1.items():
                seg_i = Ngram(aligned_pair.words[idx1]).undo()
                complex_idx2 = False
                if None not in seq2corrs:
                    seq2corrs = segment_ranges(seq2corrs)
                for idx2 in seq2corrs:
                    if idx2 is not None:
                        if isinstance(idx2, tuple):
                            complex_idx2 = True
                            start, end = idx2
                            seg_j = []
                            for x in range(start, end + 1):
                                if x == start or SEG_JOIN_CH not in aligned_pair.mots[x]:
                                    seg_j.append(Ngram(aligned_pair.mots[x]).undo())
                                else:
                                    seg_j.append(Ngram(aligned_pair.mots[x]).undo()[-1])
                            seg_j = Ngram(seg_j).ngram
                            #seg_j = Ngram(aligned_pair.mots[start:end+1]).ngram
                        else:
                            seg_j = Ngram(aligned_pair.mots[idx2]).undo()
                    else:
                        seg_j = self.gap_ch
                    corr_dict_l1l2.increment_value(seg_i, seg_j, 1)
                    if complex_idx2:
                        corr_dict_l2l1.increment_value(seg_j, seg_i, 1)
                        # for seg_j_j in seg_j:
                        #     corr_dict_l2l1.increment_value(seg_j_j, seg_i, -1)  # if this is added back, need to ensure the value doesn't become negative
            for idx2, seq1corrs in aligned_seq2.items():
                seg_j = Ngram(aligned_pair.mots[idx2]).undo()
                complex_idx1 = False
                if None not in seq1corrs:
                    seq1corrs = segment_ranges(seq1corrs)
                for idx1 in seq1corrs:
                    if idx1 is not None:
                        if isinstance(idx1, tuple):
                            complex_idx1 = True
                            start, end = idx1
                            seg_i = []
                            for x in range(start, end + 1):
                                if x == start or SEG_JOIN_CH not in aligned_pair.words[x]:
                                    seg_i.append(Ngram(aligned_pair.words[x]).undo())
                                else:
                                    seg_i.append(Ngram(aligned_pair.words[x]).undo()[-1])
                            seg_i = Ngram(seg_i).ngram
                            #seg_i = Ngram(aligned_pair.words[start:end+1]).ngram
                        else:
                            seg_i = Ngram(aligned_pair.words[idx1]).undo()
                    else:
                        seg_i = self.gap_ch
                    corr_dict_l2l1.increment_value(seg_j, seg_i, 1)
                    if complex_idx1:
                        corr_dict_l1l2.increment_value(seg_i, seg_j, 1)
                        # for seg_i_i in seg_i:
                        #     corr_dict_l1l2.increment_value(seg_i_i, seg_j, -1) # if this is added back, need to ensure the value doesn't become negative

        # Prune correspondences which occur fewer than min_corr times
        # with the exception of phones which occur fewer than min_corr times in the sample
        exc1, exc2 = sample.phones_below_min_corr(min_corr=min_corr, lang1=lang1, lang2=lang2)
        corr_dict_l1l2 = prune_corrs(corr_dict_l1l2, min_val=min_corr, exc1=exc1, exc2=exc2)
        corr_dict_l2l1 = prune_corrs(corr_dict_l2l1, min_val=min_corr, exc1=exc1, exc2=exc2)

        return corr_dict_l1l2, corr_dict_l2l1

    def joint_probs(self, conditional_counts: PhonemeMap, wordlist: Wordlist):
        """Converts a nested dictionary of conditional phoneme frequencies into a nested dictionary of joint probabilities.
        
        Args:
            conditional_counts (PhonemeMap): Nested dictionary of conditional frequencies.
            wordlist (Wordlist): Wordlist for searching ngram probabilities.

        Returns:
            PhonemeMap: Nested dictionary of joint phoneme probabilities.
        """
        joint_probabilities = PhonemeMap(0)

        # Aggregate total counts of seg1 and adjust conditional counts
        # by marginalizing over all unigrams of aligned units
        agg_seg1_totals = defaultdict(lambda: 0)
        adj_cond_counts = PhonemeMap(0)
        for seg1 in conditional_counts.get_primary_keys():
            seg1_ngram = Ngram(seg1)

            # Get the total occurrence of this segment/ngram
            seg1_totals = sum(conditional_counts.get_primary_key_map(seg1).values())

            # Update count of full higher-order lang1 ngram
            if seg1_ngram.size > 1:
                agg_seg1_totals[seg1] += seg1_totals

                # Update correspondence counts of full higher-order lang1 ngram
                # with full higher-order lang2 ngram and with all lang2 component unigrams
                for seg2, cond_val in conditional_counts.get_primary_key_map(seg1).items():
                    seg2_ngram = Ngram(seg2)
                    # full higher-order lang1 ngram with higher-order lang2 ngram
                    if seg2_ngram.size > 1:
                        adj_cond_counts.increment_value(seg1, seg2, cond_val)
                    # full higher-order lang1 ngram with lang2 component unigrams
                    for seg2_j in seg2_ngram.ngram:
                        adj_cond_counts.increment_value(seg1, seg2_j, cond_val)

            # Update aggregated counts of all component lang1 unigrams
            for seg1_i in seg1_ngram.ngram:
                agg_seg1_totals[seg1_i] += seg1_totals

                # Adjust correspondence counts for each component unigram in lang1
                # And the full ngram in lang2, as well as component unigrams of lang2 ngram
                for seg2, cond_val in conditional_counts.get_primary_key_map(seg1).items():
                    seg2_ngram = Ngram(seg2)

                    # Update the count for the full ngram in seg2
                    if seg2_ngram.size > 1:
                        adj_cond_counts.increment_value(seg1_i, seg2, cond_val)

                    # TODO also ngram sizes between N and 1, e.g. if a trigram, update component bigram counts
                    # # Update the counts for all lower-order ngrams until unigrams
                    # for n in range(seg2_ngram.size-1, 1, -1):
                    #     # Get all possible sub-ngrams of size `n` from `seg2`
                    #     for sub_ngram in generate_ngrams(seg2_ngram.ngram, ngram_size=n):
                    #         adj_cond_counts[seg1_i][sub_ngram] += cond_val

                    # Also update for each unigram component of seg2
                    for seg2_j in seg2_ngram.ngram:
                        adj_cond_counts.increment_value(seg1_i, seg2_j, cond_val)

        # Henceforth use the adjusted correspondence count dict
        conditional_counts = adj_cond_counts
        for seg1 in conditional_counts.get_primary_keys():
            seg1_ngram = Ngram(seg1)
            seg1_totals = agg_seg1_totals[seg1]
            for seg2 in conditional_counts.get_secondary_keys(seg1):
                seg2_ngram = Ngram(seg2)
                cond_count = conditional_counts.get_value_or_default(seg1, seg2, 0)
                cond_prob = cond_count / seg1_totals
                p_ind1 = wordlist.ngram_probability(seg1_ngram, lang=1)
                joint_prob = cond_prob * p_ind1
                joint_probabilities.set_value(seg1, seg2, joint_prob)
        return joint_probabilities

    def correspondence_counts(self,
                              alignment_list,
                              wordlist: Wordlist,
                              ngram_size=1,
                              min_corr=2,
                              exclude_null=True,
                              pad=False,
                             ) -> PhonemeMap:
        """Returns a dictionary of conditional phone counts, based on a list
        of Alignment objects.
        exclude_null : Bool; if True, does not consider aligned pairs including a null segment"""

        corr_counts = PhonemeMap(0)
        for alignment in alignment_list:
            if exclude_null:
                _alignment = alignment.remove_gaps()
            else:
                _alignment = alignment.alignment
            # Pad with at least one boundary position
            pad_n = 0
            if pad:
                pad_n = max(1, ngram_size - 1)
                _alignment = alignment.pad(
                    ngram_size,
                    alignment=_alignment,
                    pad_ch=self.pad_ch,
                    pad_n=pad_n
                )

            for i in range(ngram_size - 1, len(_alignment)):
                ngram = _alignment[i - (ngram_size - 1):i + 1]
                seg1, seg2 = list(zip(*ngram))
                corr_counts.increment_value(seg1, seg2, 1)

        if min_corr > 1:
            exc1, exc2 = wordlist.phones_below_min_corr(
                min_corr=min_corr,
                lang1=self.lang1,
                lang2=self.lang2
            )
            corr_counts = prune_corrs(
                corr_counts,
                min_val=min_corr,
                exc1=exc1,
                exc2=exc2
            )

        return corr_counts

    def phon_env_correspondence_probabilities(self, alignment_list: list[Alignment]) -> PhonemeMap:
        """Computes conditional correspondence probabilities for phoneme environments.

        Args:
            alignment_list (list): List of Alignment objects.

        Returns:
            PhonemeMap: PhonemeMap of conditional phoneme environment correspondence probabilities.
        """
        corr_counts = PhonemeMap(0)
        for alignment in alignment_list:
            phon_env_align = alignment.add_phon_env()
            for phon_env_seg1, seg2 in phon_env_align:
                corr_counts.increment_value(phon_env_seg1, seg2, 1)
        return corr_counts

    def phoneme_pmi(self,
                    conditional_counts: PhonemeMap,
                    l1: Doculect,
                    l2: Doculect,
                    wordlist: Wordlist) -> PhonemeMap:
        """Computes phoneme PMI based on conditional counts from a paired wordlist.

        Args:
            conditional_counts (PhonemeMap): Mapping of conditional correspondence probabilities in potential cognates.
            l1 (Doculect): First doculect/language.
            l2 (Doculect): Second doculect/language.
            wordlist (Wordlist, optional): _description_. Defaults to None.

        Returns:
            PhonemeMap: Mapping of phoneme pairs from the two languages with their PMI values.
        """

        # Convert conditional probabilities to joint probabilities
        joint_probabilities = self.joint_probs(conditional_counts, wordlist=wordlist)
        reversed_conditional_counts = reverse_corr_dict_map(conditional_counts)

        # Get set of all possible phoneme correspondences
        segment_pairs = set(
            [
                (seg1, seg2)
                for seg1 in l1.phonemes
                for seg2 in l2.phonemes
            ]
        )
        # Extend with any more complex ngram correspondences discovered
        segment_pairs.update(
            [
                (corr1, corr2)
                for corr1, corr2 in joint_probabilities.get_key_pairs()
                if corr1 not in l1.phonemes or corr2 not in l2.phonemes
            ]
        )
        # Extend with gaps in seq1, not included in joint_prob_dist (but in the seq2 nested dicts and therefore in reverse_cond_counts already)
        segment_pairs.update([(self.gap_ch, seg2) for seg2 in l2.phonemes])
        segment_pairs.update([(self.gap_ch, corr2)
                              for _, corr2 in joint_probabilities.get_key_pairs()
                              if corr2 not in l2.phonemes])
        # Extend with gap and pad tokens
        segment_pairs.update([
            (self.gap_ch, start_token(self.pad_ch)),
            (self.gap_ch, end_token(self.pad_ch))
        ])

        # Estimate probability of a gap in each language from conditional counts
        gap_counts1, gap_counts2 = 0, 0
        seg_counts1, seg_counts2 = 0, 0
        for corr1 in conditional_counts.get_primary_keys():
            corr2_dict = conditional_counts.get_primary_key_map(corr1)
            for corr2, _ in corr2_dict.items():
                if corr2 == self.gap_ch:
                    gap_counts2 += 1
                else:
                    seg_counts2 += 1
        for corr2 in reversed_conditional_counts.get_primary_keys():
            corr1_dict = reversed_conditional_counts.get_primary_key_map(corr2)
            if corr2 == self.gap_ch:
                gap_counts1 += sum(corr1_dict.values())
            else:
                seg_counts1 += sum(corr1_dict.values())
        gap_prob1 = gap_counts1 / (seg_counts1 + gap_counts1)
        gap_prob2 = gap_counts2 / (seg_counts2 + gap_counts2)

        # Calculate PMI for all phoneme pairs
        pmi_dict: PhonemeMap = PhonemeMap(0)
        for seg1, seg2 in segment_pairs:
            seg1_ngram = Ngram(seg1)
            seg2_ngram = Ngram(seg2)

            # Skip alignments of start boundary with end boundary tokens
            # (seg1, seg2) == (start_boundary, end_boundary)
            if seg1_ngram.is_boundary(self.pad_ch) and seg2_ngram.is_boundary(self.pad_ch):
                if seg1_ngram.size == seg2_ngram.size == 1:
                    if seg1_ngram.ngram != seg2_ngram.ngram:
                        continue

            # Calculation below gives a more precise probability specific to a certain subset of words,
            # which directly reflects shared coverage between l1 and l2.
            if not seg1_ngram.is_gappy(self.gap_ch):
                p_ind1 = wordlist.ngram_probability(seg1_ngram, lang=1)
            elif seg1_ngram.size == 1:
                p_ind1 = gap_prob1
            elif seg1_ngram.size > 1:
                gapless_seg1_ngram = seg1_ngram.remove_gaps(self.gap_ch)
                p_ind1 = wordlist.ngram_probability(gapless_seg1_ngram, lang=1)
            if not seg2_ngram.is_gappy(self.gap_ch):
                p_ind2 = wordlist.ngram_probability(seg2_ngram, lang=2)
            elif seg2_ngram.size == 1:
                p_ind2 = gap_prob2
            elif seg1_ngram.size > 1:
                gapless_seg2_ngram = seg2_ngram.remove_gaps(self.gap_ch)
                p_ind2 = wordlist.ngram_probability(gapless_seg2_ngram, lang=2)

            # Because we iterate over all possible phone pairs, when considering only the counts of phones/ngrams
            # within a specific wordlist, it could occur that the count is zero for a segment in that wordlist.
            # In that case, skip PMI calculation for this pair as there is insufficient data.
            if p_ind1 == 0 or p_ind2 == 0:
                continue

            p_ind = p_ind1 * p_ind2
            joint_prob = joint_probabilities.get_value_or_default(seg1, seg2, p_ind)
            if p_ind1 == 0:
                raise ValueError(f"Couldn't calculate independent probability of segment {seg1} in {self.lang1_name}")
            if p_ind2 == 0:
                raise ValueError(f"Couldn't calculate independent probability of segment {seg2} in {self.lang2_name}")
            # As long as the independent probabilities > 0, skip calculating PMI for segment pairs with 0 joint probability
            if joint_prob > 0:
                pmi_val = pointwise_mutual_info(joint_prob, p_ind1, p_ind2)

                # Add only non-zero PMI to dictionary
                if pmi_val != 0:
                    pmi_dict.set_value(seg1_ngram.undo(), seg2_ngram.undo(), pmi_val)
        return pmi_dict

    def compute_phone_corrs(self,
                            family_index,
                            p_threshold=0.1,
                            max_iterations=3,
                            n_samples=3,
                            sample_size=0.8,
                            min_corr=2,
                            max_ngram_size=2,
                            ngram_size=1, # TODO remove: this is a dummy variable added temporarily to enable surprisal calculation together
                            phon_env=False,
                            cumulative=False,
                            ):
        """Computes phone correspondences between two languages in the form of PMI and surprisal.

        Args:
            p_threshold (float, optional): Threshold for determining whether aligned word pairs qualify for next iteration of correspondence calculation. \
                Defaults to 0.1, which corresponds with a 90% chance that aligned word pairs with a given score do NOT belong to the non-cognate distribution.
            max_iterations (int, optional): Maximum number of iterations.
            n_samples (int, optional): Number of samples to draw.
            sample_size (float, optional): Sample size proportional to shared vocablary size.
            min_corr (int, optional): Minimum instances of a phone correspondence to be considered valid.
            max_ngram_size (int, optional): Maximum ngram size for radial IBM alignment.
            ngram_size (int, optional): Ngram size for surprisal.
            cumulative (bool, optional): Accumulate correspondence counts over iterations, continuing to consider alignments from earlier iterations.

        Returns:
            results (dict): Nested dictionary of PMI correspondences.
        """
        # Generate and prepare wordlists if not previously done
        self.prepare_wordlists()

        # Take a sample of same-meaning words, by default 80% of available same-meaning pairs
        sample_results: dict[int, PhonemeMap] = {}
        sample_size = round(len(self.same_meaning) * sample_size)
        # Take N samples of different-meaning words, perform PMI calibration, then average all of the estimates from the various samples
        iter_logs = defaultdict(lambda: [])
        sample_iterations = {}
        start_seed = self.seed
        if n_samples > 1:
            sample_dict = self.sample_wordlists(
                n_samples=n_samples,
                sample_size=sample_size,
                start_seed=start_seed,
                log_outfile="phone_corr_samples.log",
            )
            diff_meaning_sampled = set()
            for _, (_, diff_sample) in sample_dict.items():
                diff_meaning_sampled.update(diff_sample)
        else:
            diff_meaning_sampled = random.sample(self.diff_meaning, len(self.same_meaning))
            sample_dict = {
                (start_seed, len(self.same_meaning)): (
                    self.same_meaning, diff_meaning_sampled
                )
            }
        final_qualifying = set()
        #other_word_pairs = set()
        for key, sample in sample_dict.items():
            seed_i, _ = key
            sample_n = seed_i - start_seed
            synonym_sample, diff_sample = sample
            synonym_sample, diff_sample = map(sort_wordlist, [synonym_sample, diff_sample])
            synonym_sample_wordlist = Wordlist(synonym_sample)

            # At each following iteration N, re-align using the pmi_stepN as an
            # additional penalty, and then recalculate PMI
            iteration = 0
            PMI_iterations: dict[int, PhonemeMap] = {}
            qualifying_words = default_dict({iteration: synonym_sample_wordlist}, lmbda=[])
            disqualified_words = default_dict({iteration: diff_sample}, lmbda=[])
            if cumulative:
                all_cognate_alignments = []

            while iteration < max_iterations and qualifying_words[iteration] != qualifying_words[iteration - 1]:
                iteration += 1
                qual_prev_sample = qualifying_words[iteration - 1]
                reversed_qual_prev_sample = qual_prev_sample.reverse()

                if iteration == 1:
                    # Fit IBM translation/alignment model on ngrams of varying sizes
                    initial_corr_counts1, _ = self.fit_radial_ibm_model(
                        qual_prev_sample,
                        lang1=self.lang1,
                        lang2=self.lang2,
                        min_corr=min_corr,
                        max_ngram_size=max_ngram_size,
                        seed=seed_i,
                    )
                    initial_corr_counts2, _ = self.fit_radial_ibm_model(
                        reversed_qual_prev_sample,
                        lang1=self.lang2,
                        lang2=self.lang1,
                        min_corr=min_corr,
                        max_ngram_size=max_ngram_size,
                        seed=seed_i,
                    )

                    # Calculate initial PMI for all ngram pairs
                    pmi_dict_l1l2, pmi_dict_l2l1 = [
                        self.phoneme_pmi(
                            conditional_counts=initial_corr_counts1,
                            l1=self.lang1,
                            l2=self.lang2,
                            wordlist=qual_prev_sample,
                        ),
                        self.phoneme_pmi(
                            conditional_counts=initial_corr_counts2,
                            l1=self.lang2,
                            l2=self.lang1,
                            wordlist=reversed_qual_prev_sample,
                        )
                    ]

                    # Average together the PMI values from each direction
                    pmi_step_i = average_corrs(pmi_dict_l1l2, pmi_dict_l2l1)
                
                else:
                    pmi_step_i = PMI_iterations[iteration - 1]

                # Align the qualifying words of the previous step using initial PMI
                cognate_alignments = self.align_wordlist(
                    qual_prev_sample,
                    align_costs=pmi_step_i
                )

                # Add cognate alignments into running pool of alignments
                if cumulative:
                    all_cognate_alignments.extend(cognate_alignments)
                    cognate_alignments = all_cognate_alignments

                # Recalculate correspondence counts and PMI values
                # from these alignments alone, i.e. not using radial EM
                # Reason for recalculating is that using alignments we can be stricter:
                # impose minimum corr requirements and only consider actually aligned segments
                cognate_corr_counts = self.correspondence_counts(
                    cognate_alignments,
                    wordlist=qual_prev_sample,
                    exclude_null=False,
                    min_corr=min_corr,
                )
                PMI_iterations[iteration] = self.phoneme_pmi(
                    cognate_corr_counts,
                    l1=self.lang1,
                    l2=self.lang2,
                    wordlist=qual_prev_sample
                )

                # Align all same-meaning word pairs with recalculated PMI
                aligned_synonym_sample = self.align_wordlist(
                    synonym_sample,
                    align_costs=PMI_iterations[iteration],
                )

                # Align sample of different-meaning word pairs + non-cognates detected from previous iteration
                # disqualified_words[iteration-1] already contains both types
                noncognate_alignments = self.align_wordlist(
                    disqualified_words[iteration - 1],
                    align_costs=PMI_iterations[iteration],
                )

                # Score PMI for different meaning words and words disqualified in previous iteration
                noncognate_alignment_scores = []
                for alignment in noncognate_alignments:
                    length_normalized_score = alignment.cost / alignment.length
                    noncognate_alignment_scores.append(length_normalized_score)
                nc_mean = mean(noncognate_alignment_scores)
                nc_stdev = stdev(noncognate_alignment_scores)

                # Score same-meaning alignments against different-meaning alignments
                qualifying, disqualified = [], []
                qualifying_alignments = []
                qualified_PMI = []
                for q, pair in enumerate(synonym_sample):
                    alignment = aligned_synonym_sample[q]
                    length_normalized_score = alignment.cost / alignment.length

                    # Proportion of non-cognate word pairs which would have an alignment score at least as low as this word pair
                    pnorm = 1 - norm.cdf(length_normalized_score, loc=nc_mean, scale=nc_stdev)
                    if pnorm < p_threshold:
                        qualifying.append(pair)
                        qualifying_alignments.append(alignment)
                        qualified_PMI.append(length_normalized_score)
                    else:
                        disqualified.append(pair)
                        #other_word_pairs.add(pair)
                if len(qualifying) == 0:
                    logger.warning(f'All word pairs were disqualified in PMI iteration {iteration}')
                qualifying_wordlist = Wordlist(sort_wordlist(qualifying))
                qualifying_wordlist, qualifying_alignments = prune_extraneous_synonyms(
                    wordlist=qualifying_wordlist,
                    alignments=qualifying_alignments,
                    scores=qualified_PMI,
                    maximize_score=True,
                    family_index=family_index,
                )
                qualifying_words[iteration] = qualifying_wordlist
                disqualified_words[iteration] = disqualified + diff_sample

                # Log results of this iteration
                iter_log = log_phon_corr_iteration(
                    iteration=iteration,
                    qualifying_words=qualifying_words,
                    disqualified_words=disqualified_words,
                )
                iter_logs[sample_n].append(iter_log)

            # Log final set of qualifying/disqualified word pairs
            iter_logs[sample_n].append((qualifying_words[iteration], sort_wordlist(disqualified)))

            # Return and save the final iteration's PMI results
            results = PMI_iterations[max(PMI_iterations.keys())]
            sample_results[sample_n] = results
            sample_iterations[sample_n] = len(PMI_iterations) - 1
            final_qualifying.update(qualifying)

            self.reset_seed()

        # Average together the PMI estimations from each sample
        if n_samples > 1:
            results = average_nested_dicts(list(sample_results.values()))
        else:
            results = sample_results[0]

        # Realign final qualifying using averaged PMI values from all samples
        #other_word_pairs = other_word_pairs - final_qualifying
        #other_word_pairs.update(diff_meaning_sampled)
        final_qualifying = Wordlist(final_qualifying)
        reversed_final_qualifying = final_qualifying.reverse()
        #other_word_pairs = list(other_word_pairs)
        final_qualifying_alignments = self.align_wordlist(
            final_qualifying,
            align_costs=results,
        )
        final_qualifying, final_qualifying_alignments = prune_extraneous_synonyms(
            wordlist=final_qualifying,
            alignments=final_qualifying_alignments,
            maximize_score=True,
            family_index=family_index,
        )
        # final_other_alignments = self.align_wordlist(
        #     other_word_pairs,
        #     align_costs=results,
        # )

        # Compute phone surprisal
        self.compute_phone_surprisal(
            alignments=final_qualifying_alignments,
            wordlist=final_qualifying,
            phon_env=phon_env,
            min_corr=min_corr,
            ngram_size=ngram_size,
        )
        # Compute surprisal in opposite direction with reversed alignments
        twin, family_index[PHONE_CORRELATORS_INDEX_KEY] = self.get_twin(family_index[PHONE_CORRELATORS_INDEX_KEY])
        reversed_final_alignments = [alignment.reverse() for alignment in final_qualifying_alignments]
        twin.compute_phone_surprisal(
            alignments=reversed_final_alignments,
            wordlist=reversed_final_qualifying,
            phon_env=phon_env,
            min_corr=min_corr,
            ngram_size=ngram_size,
        )

        # Log all final alignments
        self.log_alignments(final_qualifying_alignments)
        #self.log_alignments(final_other_alignments)
        twin.log_alignments(reversed_final_alignments)
        #reversed_other_alignments = [alignment.reverse() for alignment in final_other_alignments]
        #twin.log_alignments(reversed_other_alignments)

        # Write the iteration log
        log_file = os.path.join(self.phon_corr_dir, 'iterations.log')
        write_phon_corr_iteration_log(iter_logs, log_file, n_same_meaning_pairs=len(self.same_meaning))

        # Write alignment log
        align_log_file = os.path.join(self.phon_corr_dir, 'alignments.log')
        write_alignments_log(self.align_log, align_log_file)

        # Save PMI results
        self.pmi_results = results
        twin.pmi_results = reverse_corr_dict_map(results)
        self.log_phoneme_pmi()

        return results

    def phoneme_surprisal(self,
                          correspondence_counts: PhonemeMap,
                          phon_env_corr_counts: PhonemeMap=None,
                          ngram_size=1,
                          ) -> PhonemeMap:
        # Set phon_env bool
        use_phon_env = True if phon_env_corr_counts is not None else False

        # Interpolation smoothing
        # Each ngram estimate will be weighted proportional to its size
        # Weight the estimate from a 2gram twice as much as 1gram, etc.
        ngram_weights = [i + 1 for i in range(ngram_size)]
        interpolation = defaultdict(lambda: PhonemeMap(0))
        for i in range(ngram_size, 0, -1):
            for ngram1, ngram2 in correspondence_counts.get_key_pairs():
                # Exclude correspondences with a fully null ngram, e.g. ('k', 'a') with ('-', '-')
                # Only needs to be done with ngram_size > 1
                if (self.gap_ch,) * (max(ngram_size, 2)) not in [ngram1, ngram2]:
                    interpolation[i].increment_value(
                        Ngram(ngram1).ngram,
                        Ngram(ngram2).ngram,
                        correspondence_counts.get_value_or_default(ngram1, ngram2, 0)
                    )

        # Add in phonological environment correspondences, e.g. ('l', '#S<') (word-initial 'l') with 'ʎ'
        max_ngram_size = ngram_size
        if use_phon_env:
            for ngram1 in phon_env_corr_counts.get_primary_keys():
                if self.gap_ch in ngram1 or self.pad_ch in ngram1:
                    continue  # skip before trying to compute PhonEnvNgram; gap_ch/pad_ch will raise an error in PhonEnvNgram
                phon_env_ngram = PhonEnvNgram(ngram1)
                if phon_env_ngram.is_gappy(self.gap_ch) or phon_env_ngram.is_boundary(self.pad_ch):
                    continue  # TODO verify that these should be skipped
                if phon_env_ngram.size > max_ngram_size:
                    max_ngram_size = phon_env_ngram.size
                for context in phon_env_ngram.list_subcontexts():
                    if phon_env_ngram.size > 1:
                        ngram1_context = (phon_env_ngram.ngram,) + (context,)
                    else:
                        ngram1_context = phon_env_ngram.ngram + (context,)
                    for ngram2 in phon_env_corr_counts.get_secondary_keys(ngram1):
                        # backward
                        interpolation['phon_env'].increment_value(
                            ngram1_context,
                            Ngram(ngram2).ngram,
                            phon_env_corr_counts.get_value_or_default(ngram1, ngram2, 0)
                        )

        # Get lists of possible ngrams in lang1 and lang2
        # lang2 ngram size fixed at 1, only trying to predict single phone; also not trying to predict its phon_env
        all_ngrams_lang2 = self.get_possible_ngrams(self.lang2, ngram_size=1, phon_env=False)
        all_ngrams_lang2.update(reverse_corr_dict_map(interpolation[1]).get_primary_keys())  # add any complex ngram corrs
        all_ngrams_lang1 = set()
        for size_i in range(1, max_ngram_size + 1):
            all_ngrams_lang1.update(self.get_possible_ngrams(
                lang=self.lang1,
                ngram_size=size_i,
                phon_env=use_phon_env
            ))
        # Add complex ngram corrs to all_ngrams_lang1
        if use_phon_env:
            all_ngrams_lang1.update(interpolation['phon_env'].get_primary_keys())
        else:
            all_ngrams_lang1.update(interpolation[1].get_primary_keys())

        oov_value = self.lang2.phoneme_entropy * ngram_size
        smoothed_surprisal = PhonemeMap(oov_value)
        for ngram1 in all_ngrams_lang1:
            if self.gap_ch in ngram1 or self.pad_ch in ngram1:
                continue # skip before trying to compute PhonEnvNgram; gap_ch/pad_ch will raise an error in PhonEnvNgram
            ngram1_obj = PhonEnvNgram(ngram1) if use_phon_env else Ngram(ngram1)
            if ngram1_obj.is_gappy(self.gap_ch):
                continue  # TODO verify that this should be skipped
            if use_phon_env:
                ngram1_phon_env_map = interpolation['phon_env'].get_primary_key_map(ngram1_obj.ngram_w_context)
                if ngram1_phon_env_map is None or sum(ngram1_phon_env_map.values()) == 0:
                    continue
            ngram1_interpolation_key = ngram1_obj.ngram if use_phon_env else ngram1
            ngram1_map = interpolation[i].get_primary_key_map(ngram1_interpolation_key)
            if ngram1_map is not None and sum(ngram1_map.values()) == 0:
                continue

            ngram1_key = ngram1_obj.undo()  # need to convert to dictionary form, whereby unigrams are strings and larger ngrams are tuples
            for ngram2 in all_ngrams_lang2:
                if interpolation[i].get_value_or_default(ngram1_interpolation_key, ngram2, 0) == 0:
                    continue
                estimates = []
                estimate_weights = ngram_weights.copy()
                for i in range(ngram_size, 0, -1):
                    ngram1_map_i = interpolation[i].get_primary_key_map(ngram1_interpolation_key)
                    if ngram1_map_i is not None:
                        estimates.append(
                            interpolation[i].get_value_or_default(ngram1_interpolation_key, ngram2, 0) / sum(ngram1_map_i.values())
                        )
                
                assert (len(estimate_weights) == len(estimates))
                # Add interpolation with phon env surprisal
                if use_phon_env:
                    if not (ngram1_obj.size == 1 and ngram1_obj.is_gappy(self.pad_ch)): # skip computing phon env probabilities with e.g. ('#>',)
                        ngram1_w_context = ngram1_obj.ngram_w_context
                        total_context_counts = sum(interpolation['phon_env'].get_primary_key_map(ngram1_w_context).values())
                        estimates.append(
                            interpolation['phon_env'].get_value_or_default(ngram1_w_context, ngram2, 0) / total_context_counts
                        )
                        # Weight each contextual estimate based on the size of the context
                        estimate_weights.append(get_phonEnv_weight(ngram1_obj.phon_env))

                assert (len(estimate_weights) == len(estimates))
                weight_sum = sum(estimate_weights)
                estimate_weights = [i / weight_sum for i in estimate_weights]
                smoothed = sum([estimate * weight for estimate, weight in zip(estimates, estimate_weights)])
                surprisal_value = surprisal(smoothed)
                # Skip saving surprisal values which exceed the oov_val (phoneme entropy of lang2)
                if surprisal_value <= oov_value:
                    ngram2_key = Ngram(ngram2).undo()  # need to convert to dictionary form, whereby unigrams are strings and larger ngrams are tuples
                    if use_phon_env:
                        if not (len(ngram1) == 1 and self.pad_ch in ngram1[0]):  # skip computing phon env probabilities with e.g. ('#>',)
                            smoothed_surprisal.set_value(
                                ngram1_obj.ngram_w_context,
                                ngram2_key,
                                surprisal_value
                            )
                    else:
                        smoothed_surprisal.set_value(
                            ngram1_key,
                            ngram2_key,
                            surprisal_value
                        )

        return smoothed_surprisal

    def compute_phone_surprisal(self,
                                alignments: list,
                                wordlist: Wordlist,
                                phon_env=False,
                                min_corr=2,
                                ngram_size=1, # TODO remove if not going to be developed further
                                ):
        """Computes phone surprisal from word pair alignments.

        Args:
            alignments (list): List of Alignment objects representing aligned word pairs.
            phon_env (bool, optional): Use phonological environment.
            min_corr (int, optional): Minimum instances of a phone correspondence to be considered valid.
            ngram_size (int, optional): Ngram size.

        Returns:
            (surprisal_results, phon_env_surprisal_results) (tuple): Tuple with nested dictionaries of surprisal results and optionally phonological environment surprisal results. \
                If phon_env is False, the latter will be None.
        """
        # Get correspondence probabilities from alignments
        corr_counts = self.correspondence_counts(
            alignment_list=alignments,
            wordlist=wordlist,
            min_corr=min_corr,
            exclude_null=False,
            #ngram_size=ngram_size,
        )

        # Optionally calculate phonological environment correspondence probabilities
        phon_env_corr_counts = None
        if phon_env:
            phon_env_corr_counts = self.phon_env_correspondence_probabilities(alignments)

        # Calculate surprisal based on correspondence probabilities
        # NB: Will be phon_env surprisal if phon_env=True and therefore phon_env_corr_counts != None
        surprisal_results = self.phoneme_surprisal(
            corr_counts,
            phon_env_corr_counts=phon_env_corr_counts
            #ngram_size=ngram_size
        )
        # Get vanilla surprisal (no phon env) from phon_env surprisal by marginalizing over phon_env
        if phon_env:
            phon_env_surprisal_results = surprisal_results.copy()
            surprisal_results = marginalize_over_phon_env(
                phon_env_surprisal_results,
                from_surprisal=True,
                to_surprisal=True,
                doculect=self.lang1,
            )

        # Save surprisal results
        self.surprisal_results[ngram_size] = surprisal_results
        if phon_env:
            self.phon_env_surprisal_results = phon_env_surprisal_results

        # Write phone correlation report based on surprisal results
        phon_corr_report = os.path.join(self.phon_corr_dir, 'phon_corr.tsv')
        self.write_phon_corr_report(surprisal_results, phon_corr_report, corr_type='surprisal')

        # Write surprisal logs
        self.log_phoneme_surprisal(phon_env=False, ngram_size=ngram_size)
        if phon_env:
            self.log_phoneme_surprisal(phon_env=True)


    def compute_noncognate_thresholds(self, eval_func, sample_size=None, seed=None):
        """Calculate non-synonymous word pair scores against which to calibrate synonymous word scores"""
        # Generate and prepare wordlists if not previously done
        self.prepare_wordlists()

        # Take a sample of different-meaning words, by default as large as the same-meaning set
        if sample_size is None:
            sample_size = len(self.same_meaning)
        else:
            sample_size = min(sample_size, len(self.diff_meaning))

        if (seed, sample_size) not in self.samples:
            _ = self.sample_wordlists(
                n_samples=1, 
                sample_size=sample_size,
                start_seed=seed,
                log_outfile="noncognate_thresholds_samples.log",
            )
        _, diff_sample = self.samples[(seed, sample_size)]
        noncognate_scores = []
        for pair in diff_sample:
            score = eval_func.eval(pair[0], pair[1])
            noncognate_scores.append(score)
        # Save results
        key = (eval_func, sample_size, seed)
        self.noncognate_thresholds[key] = noncognate_scores

        return noncognate_scores

    def log_phoneme_pmi(self, threshold=0.0001):
        write_phoneme_pmi_report(
            self.pmi_results,
            outfile=os.path.join(self.phon_corr_dir, 'phonPMI.tsv'),
            threshold=threshold,
        )

    def log_phoneme_surprisal(self, phon_env=True, ngram_size=1):
        if phon_env:
            outfile = os.path.join(self.phon_corr_dir, 'phonEnvSurprisal.tsv')
            surprisal_results = self.phon_env_surprisal_results
        else:
            outfile = os.path.join(self.phon_corr_dir, 'phonSurprisal.tsv')
            surprisal_results = self.surprisal_results[ngram_size]
        write_phoneme_surprisal_report(
            surprisal_results=surprisal_results,
            outfile=outfile,
            phon_env=phon_env,
            ngram_size=ngram_size,
        )

    def log_sample(self, sample, sample_n, label, seed=None):
        if seed is None:
            seed = self.seed
        sample = sorted(
            [
                f'[{word1.concept}] {word1.orthography} /{word1.ipa}/ - [{word2.concept}] {word2.orthography} /{word2.ipa}/'
                for word1, word2 in sample
            ]
        )
        sample_log = f'SAMPLE: {sample_n}\nSEED: {seed}\nLABEL: {label}\n'
        sample_log += '\n'.join(sample)
        return sample_log

    def log_alignments(self, alignments):
        for alignment in alignments:
            self.align_log[alignment.key] = alignment

    def write_phon_corr_report(self, corr, outfile, corr_type):
        write_phon_corr_report(
            corr=corr,
            corr_type=corr_type,
            outfile=outfile,
            lang1_name=self.lang1_name,
            lang2_name=self.lang2_name,
            gap_ch=self.gap_ch,
        )


def get_phone_correlator(lang1,
                         lang2,
                         phone_correlators_index,
                         wordlist=None,
                         log_outdir=None,
                         seed=1,
                         ):
    """Retrieve previously initialized PhonCorrelator or create a new instance if not yet initialized."""
    key = (lang1.name, lang2.name, wordlist, seed)
    if key not in phone_correlators_index:
        phone_correlators_index[key] = PhonCorrelator(
            lang1=lang1,
            lang2=lang2,
            wordlist=wordlist,
            #gap_ch=self.alignment_params.get('gap_ch', ALIGNMENT_PARAM_DEFAULTS['gap_ch']),
            #pad_ch=self.alignment_params.get('pad_ch', ALIGNMENT_PARAM_DEFAULTS['pad_ch']),
            seed=seed,
            log_outdir=log_outdir,
        )
    correlator = phone_correlators_index[key]
    return correlator, phone_correlators_index
