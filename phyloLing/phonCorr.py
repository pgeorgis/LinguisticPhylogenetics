import os
import random
import re
from collections import defaultdict
from functools import lru_cache
from itertools import product
from statistics import mean, stdev
from typing import Self

import numpy as np
from constants import (END_PAD_CH, GAP_CH_DEFAULT, NON_IPA_CH_DEFAULT,
                       PAD_CH_DEFAULT, SEG_JOIN_CH, START_PAD_CH)
from nltk.translate import AlignedSent, IBMModel1, IBMModel2
from phonAlign import Alignment, visual_align
from phonUtils.phonEnv import (phon_env_ngrams, relative_post_sonority,
                               relative_prev_sonority)
from phonUtils.segment import _toSegment
from scipy.stats import norm
from utils.information import (pointwise_mutual_info, surprisal,
                               surprisal_to_prob)
from utils.sequence import (Ngram, PhonEnvNgram, count_subsequences, end_token,
                            pad_sequence, start_token)
from utils.utils import (default_dict,
                         dict_tuplelist,
                         normalize_dict,
                         balanced_resample,
                         segment_ranges,
                         create_default_dict,
                         create_default_dict_of_dicts)


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
        iterations (int, optional): Number of iterations to run EM algorithm. Defaults to 5.
        gap_ch (str, optional): Gap character. Defaults to GAP_CH_DEFAULT.
        ibm_model (int, optional): IBM model. Defaults to 2.
        seed (int, optional): Random seed. Defaults to None.

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
        for j, i_s in aligned_seq2.items():
            i_s.sort()
            if len(i_s) > 1:
                i_i = 0
                i_s_len = len(i_s)
                i_s_copy = i_s[:]
                while i_i < i_s_len-1:
                    i = i_s_copy[i_i]
                    i_next = i_s_copy[i_i + 1]
                    if abs(i - i_next) > 1:
                        anchor = mean(i_s+[j])
                        
                        # Find which of the two is more distant from anchor
                        more_distant_i = max(i_next, i, key=lambda x: abs(x - anchor))
                        aligned_seq2[j].remove(more_distant_i)
                        if len(aligned_seq1[more_distant_i]) > 1:
                            aligned_seq1[more_distant_i].remove(j)
                        else:
                            aligned_seq1[more_distant_i] = [None]
                    i_i += 1
            
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


def sort_wordlist(wordlist):
    return sorted(wordlist, key=lambda x: (x[0].ipa, x[1].ipa, x[0].concept, x[1].concept))


def prune_corrs(corr_dict, min_val=2):
    # Prune correspondences below a minimum count/probability threshold
    for seg1 in corr_dict:
        seg2_to_del = [seg2 for seg2 in corr_dict[seg1] if corr_dict[seg1][seg2] < min_val]
        for seg2 in seg2_to_del:
            del corr_dict[seg1][seg2]
    # Delete empty seg1 entries
    seg1_to_del = [seg1 for seg1 in corr_dict if len(corr_dict[seg1]) < 1]
    for seg1 in seg1_to_del:
        del corr_dict[seg1]
    return corr_dict


def get_oov_val(corr_dict, oov_ch=NON_IPA_CH_DEFAULT):
    # Determine the (potentially smoothed) value for unseen ("out of vocabulary" [OOV]) correspondences
    # Check using an OOV/non-IPA character
    oov_val = corr_dict[oov_ch]

    # Then remove this character from the surprisal dictionary
    del corr_dict[oov_ch]

    return oov_val


def prune_oov_surprisal(surprisal_dict):
    # Prune correspondences with a surprisal value greater than OOV surprisal
    pruned = defaultdict(lambda: {})
    for seg1 in surprisal_dict:
        oov_val = get_oov_val(surprisal_dict[seg1])

        # Save values which are not equal to (less than) the OOV smoothed value
        for seg2 in surprisal_dict[seg1]:
            surprisal_val = surprisal_dict[seg1][seg2]
            if surprisal_val < oov_val:
                pruned[seg1][seg2] = surprisal_val

        # Set as default dict with OOV value as default
        pruned[seg1] = default_dict(pruned[seg1], lmbda=oov_val)

    return pruned, oov_val


def prune_extraneous_synonyms(wordlist, alignments, scores, prefer_small=False):
    # Resolve synonyms: prune redundant/extraneous
    # If a concept has >1 words listed, we may end up with, e.g.
    # DE <Kopf> - NL <kop>
    # DE <Haupt> - NL <hoofd>
    # but also
    # DE <Kopf> - NL <hoofd>
    # DE <Haupt> - NL <kop>
    # If both languages have >1 qualifying words for a concept, only consider the best pairings, i.e. prune the extraneous pairs
    concept_counts1, concept_counts2 = defaultdict(lambda: 0), defaultdict(lambda: 0)
    concept_indices1, concept_indices2 = defaultdict(lambda: []), defaultdict(lambda: [])
    indices_to_prune = set()
    for q, pair in enumerate(wordlist):
        word1, word2 = pair
        concept1, concept2 = word1.concept, word2.concept
        concept_counts1[concept1] += 1
        concept_counts2[concept2] += 1
        concept_indices1[concept1].append(q)
        concept_indices2[concept2].append(q)

    def prune_suboptimal_pairings(concept_counts, concept_indices, wordlist):
        for concept, count in concept_counts.items():
            if count > 1:
                best_pairings = {}
                best_pairing_scores = {}
                for q in concept_indices[concept]:
                    pair = wordlist[q]
                    word1, word2 = pair
                    score = scores[q]  # TODO seems like this could benefit from a Wordpair object
                    if prefer_small:  # consider a smaller score to be better, e.g. for surprisal
                        if word1 not in best_pairings or score < best_pairings[word1]:
                            best_pairings[word1] = q
                            best_pairing_scores[word1] = score
                    else:
                        if word1 not in best_pairings or score > best_pairing_scores[word1]:
                            best_pairings[word1] = q
                            best_pairing_scores[word1] = score
                # Now best_pairings contains the best mapping for each concept based on the best (highest/lowest) scoring pair
                for q in concept_indices[concept]:
                    if q not in best_pairings.values():
                        indices_to_prune.add(q)
    prune_suboptimal_pairings(concept_counts=concept_counts1, concept_indices=concept_indices1, wordlist=wordlist)
    reversed_wordlist = [(word2, word1) for word1, word2 in wordlist]
    prune_suboptimal_pairings(concept_counts=concept_counts2, concept_indices=concept_indices2, wordlist=reversed_wordlist)

    indices_to_prune = sorted(list(indices_to_prune), reverse=True)
    for index in indices_to_prune:
        del wordlist[index]
        del alignments[index]
    return wordlist, alignments


def average_corrs(corr_dict1, corr_dict2):
    """Average together values from nested dictionaries in opposite directions"""
    avg_corr = defaultdict(lambda: defaultdict(lambda: 0))
    for seg1 in corr_dict1:
        for seg2 in corr_dict1[seg1]:
            avg_corr[seg1][seg2] = mean([corr_dict1[seg1][seg2], corr_dict2[seg2][seg1]])
    for seg2 in corr_dict2:
        for seg1 in corr_dict2[seg2]:
            if seg2 not in avg_corr[seg1]:
                avg_corr[seg1][seg2] = mean([corr_dict1[seg1][seg2], corr_dict2[seg2][seg1]])
    return avg_corr


def average_nested_dicts(dict_list, default=0):
    corr1_all = set(corr1 for d in dict_list for corr1 in d)
    corr2_all = {corr1: set(corr2 for d in dict_list for corr2 in d[corr1]) for corr1 in corr1_all}
    results = defaultdict(lambda: defaultdict(lambda: 0))
    for corr1 in corr1_all:
        for corr2 in corr2_all[corr1]:
            vals = []
            for d in dict_list:
                vals.append(d.get(corr1, {}).get(corr2, default))
            if len(vals) > 0:
                results[corr1][corr2] = mean(vals)
    return results


def reverse_corr_dict(corr_dict):
    reverse = defaultdict(lambda: defaultdict(lambda: 0))
    for seg1 in corr_dict:
        for seg2 in corr_dict[seg1]:
            reverse[seg2][seg1] = corr_dict[seg1][seg2]
    return reverse


def ngram_count_word(ngram, word):
    count = 0
    for i in range(len(word) - len(ngram) + 1):
        if word[i:i + len(ngram)] == list(ngram):
            count += 1
    return count


def ngram_count_wordlist(ngram, seq_list):
    """Retrieve the count of an ngram of segments from a list of segment sequences"""
    count = 0
    for seq in seq_list:
        count += ngram_count_word(ngram, seq)
    return count


def ngram2log_format(ngram, phon_env=False):
    if phon_env:
        ngram, phon_env = ngram[:-1], ngram[-1]
        return (Ngram(ngram).string, phon_env)
    else:
        return Ngram(ngram).string


class Wordlist:
    def __init__(self, word_pairs, pad_n=1):
        self.pad_n = pad_n
        self.wordlist_lang1, self.wordlist_lang2 = zip(*word_pairs)
        self.seqs1, self.seqs2 = self.extract_seqs()
        self.seq_lens1, self.seq_lens2 = self.seq_lens()
        self.total_seq_len1, self.total_seq_len2 = self.total_lens()
        self.ngram_probs1, self.ngram_probs2 = {}, {}

    def extract_seqs(self):
        seqs1 = [word.segments for word in self.wordlist_lang1]
        seqs2 = [word.segments for word in self.wordlist_lang2]
        if self.pad_n > 0:
            seqs1 = [pad_sequence(seq, pad_ch=PAD_CH_DEFAULT, pad_n=self.pad_n) for seq in seqs1]
            seqs2 = [pad_sequence(seq, pad_ch=PAD_CH_DEFAULT, pad_n=self.pad_n) for seq in seqs2]
        return seqs1, seqs2

    def seq_lens(self):
        seq_lens1 = [len(seq) for seq in self.seqs1]
        seq_lens2 = [len(seq) for seq in self.seqs2]
        return seq_lens1, seq_lens2

    def total_lens(self):
        total_seq_len1 = sum(self.seq_lens1)
        total_seq_len2 = sum(self.seq_lens2)
        return total_seq_len1, total_seq_len2

    def ngram_probability(self, ngram, lang=1, normalize=True):
        # if not isinstance(ngram, [Ngram, PhonEnvNgram]):
        #     if PHON_ENV_REGEX.search(ngram):
        #         ngram = PhonEnvNgram(ngram)
        #     else:
        #         ngram = Ngram(ngram)
        assert isinstance(ngram, (Ngram, PhonEnvNgram))

        if lang == 1:
            seqs = self.seqs1
            seq_lens = self.seq_lens1
            total_seq_len = self.total_seq_len1
            saved = self.ngram_probs1
        elif lang == 2:
            seqs = self.seqs2
            seq_lens = self.seq_lens2
            total_seq_len = self.total_seq_len2
            saved = self.ngram_probs2
        else:
            raise ValueError

        if ngram.ngram in saved:
            return saved[ngram.ngram]

        else:
            count = ngram_count_wordlist(ngram.ngram, seqs)
            if normalize:
                if ngram.size > 1:
                    prob = count / sum([count_subsequences(length, ngram.size) for length in seq_lens])
                else:
                    prob = count / total_seq_len
            else:
                prob = count

        saved[ngram.ngram] = prob
        return prob


class PhonCorrelator:
    def __init__(self,
                 lang1,
                 lang2,
                 wordlist=None,
                 gap_ch=GAP_CH_DEFAULT,
                 pad_ch=PAD_CH_DEFAULT,
                 seed=1,
                 logger=None):
        # Set Language objects
        self.lang1 = lang1
        self.lang1_name = lang1.name
        self.lang2 = lang2
        self.lang2_name = lang2.name

        # Alignment parameters
        self.gap_ch = gap_ch
        self.pad_ch = pad_ch
        self.seed = seed

        # Prepare wordlists: sort out same/different-meaning words and loanwords
        self.wordlist = self.get_concept_list(wordlist)
        self.same_meaning, self.diff_meaning, self.loanwords = self.prepare_wordlists()
        self.samples = {}

        # PMI, ngrams, scored words
        self.pmi_dict: dict[str, dict[str, float]] = {}
        self.surprisal_dict: dict[str, dict[str, float]] = {}
        self.phon_env_surprisal_dict: dict[str, dict[str, float]] = {}
        self.complex_ngrams: dict[str, dict[str, float]] = {}
        self.reload_language_pair_data()
        self.scored_words = create_default_dict_of_dicts()

        # Logging
        self.set_log_dirs()
        self.align_log = create_default_dict(0, 3)
        self.logger = logger

    def reload_language_pair_data(self):
        self.pmi_dict = self.lang1.phoneme_pmi[self.lang2_name]
        self.surprisal_dict = self.lang1.phoneme_surprisal[self.lang2_name]
        self.phon_env_surprisal_dict = self.lang1.phon_env_surprisal[self.lang2_name]
        self.complex_ngrams = self.lang1.complex_ngrams[self.lang2_name]


    def get_twin(self) -> Self:
        """Retrieve the twin PhonCorrelator object for the reverse direction of the same language pair."""
        return self.lang2.get_phoneme_correlator(
            lang2=self.lang1,
            wordlist=tuple(self.wordlist),
            seed=self.seed
        )

    def reset_seed(self):
        random.seed(self.seed)

    def langs(self, l1=None, l2=None):
        if l1 is None:
            l1 = self.lang1
        if l2 is None:
            l2 = self.lang2
        return l1, l2

    def set_log_dirs(self):
        self.outdir = self.lang1.family.phone_corr_dir
        self.phon_corr_dir = os.path.join(self.outdir, self.lang1.path_name, self.lang2.path_name)
        os.makedirs(self.phon_corr_dir, exist_ok=True)

    def get_concept_list(self, wordlist=None):
        # If no wordlist is provided, by default use all concepts shared by the two languages
        if wordlist is None:
            wordlist = self.lang1.vocabulary.keys() & self.lang2.vocabulary.keys()

        # If a wordlist is provided, use only the concepts shared by both languages
        else:
            wordlist = set(wordlist) & self.lang1.vocabulary.keys() & self.lang2.vocabulary.keys()

        return wordlist

    def prepare_wordlists(self):
        # Get lexical items in each language belonging to the specified wordlist
        l1_wordlist = [word for concept in self.wordlist for word in self.lang1.vocabulary[concept]]
        l2_wordlist = [word for concept in self.wordlist for word in self.lang2.vocabulary[concept]]

        # Sort the wordlists in order to ensure that random samples of same/different meaning pairs are reproducible
        l1_wordlist = sorted(l1_wordlist, key=lambda x: (x.ipa, x.concept))
        l2_wordlist = sorted(l2_wordlist, key=lambda x: (x.ipa, x.concept))

        # Get all combinations of L1 and L2 words
        all_wordpairs = product(l1_wordlist, l2_wordlist)

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

        # Return a tuple of the three word type lists
        return same_meaning, diff_meaning, loanwords

    def sample_wordlists(self, n_samples, sample_size, start_seed=None, log_samples=True):
        # Take N samples of same- and different-meaning words
        if start_seed is None:
            start_seed = self.seed

        samples = self.samples
        new_samples = False
        if log_samples:
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

            # Log same-meaning sample
            if log_samples:
                sample_log = self.log_sample(synonym_sample, sample_n, seed=seed_i)
                sample_logs[sample_n] = sample_log

        # Update dictionary of samples
        if new_samples:
            self.samples.update(samples)

        # Write sample log (only if new samples were drawn)
        if log_samples and new_samples:
            sample_log_file = os.path.join(self.outdir, self.lang1.path_name, self.lang2.path_name, 'samples.log')
            self.write_sample_log(sample_logs, sample_log_file)

        return samples

    def pad_wordlist(self, wordlist, pad_n=1):
        def _pad(seq):
            return pad_sequence(seq, pad_ch=self.pad_ch, pad_n=pad_n)
        return [map(_pad, word) for word in wordlist]

    def align_wordlist(self,
                       wordlist,
                       align_costs=None,
                       remove_uncompacted_padding=True,
                       # phon_env=False,
                       **kwargs):
        """Returns a list of the aligned segments from the wordlists"""
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
            for word1, word2 in wordlist
        ]

        if remove_uncompacted_padding:
            for alignment in alignment_list:
                alignment.remove_padding()

        # if phon_env:
        #     for alignment in alignment_list:
        #         alignment.phon_env_alignment = alignment.add_phon_env()

        return alignment_list

    def get_possible_ngrams(self, lang, ngram_size, phon_env=False):
        # Iterate over all possible/attested ngrams
        # Only perform calculation for ngrams which have actually been observed/attested to
        # in the current dataset or which could have been observed (with gaps)
        if phon_env:
            attested = lang.list_ngrams(ngram_size, phon_env=True)
            phone_contexts = [(seg, env)
                              for seg in lang.phon_environments
                              for env in lang.phon_environments[seg]]
            all_ngrams = product(phone_contexts + [f'{self.pad_ch}{END_PAD_CH}', f'{START_PAD_CH}{self.pad_ch}', self.gap_ch], repeat=ngram_size)

        else:
            attested = lang.list_ngrams(ngram_size, phon_env=False)
            all_ngrams = product(list(lang.phonemes.keys()) + [f'{self.pad_ch}{END_PAD_CH}', f'{START_PAD_CH}{self.pad_ch}', self.gap_ch], repeat=ngram_size)

        gappy = set(ngram for ngram in all_ngrams if self.gap_ch in ngram)
        all_ngrams = gappy.union(attested.keys())
        all_ngrams = set(Ngram(ngram).ngram for ngram in all_ngrams)  # standardize forms
        return all_ngrams

    def radial_em(self,
                  sample,
                  normalize=True,
                  phon_env=False, # TODO add
                  ibm_model=2,
                  max_ngram_size=2,
                  min_corr=2,
                  seed=None,
                  ):
        """Fits EM IBM models on pairs of ngrams of varying sizes and aggregates the translation tables."""

        # Create "corpora" consisting of words segmented into unigrams or bigrams
        corpus = []
        ngram_sizes = list(range(1, max_ngram_size + 1))
        for word1, word2 in sample:
            segs1, segs2 = word1.segments, word2.segments
            # Optionally add phon env
            if phon_env:
                raise NotImplementedError
                env1, env2 = word1.phon_env, word2.phon_env
                segs1 = zip(segs1, env1)
                segs1 = zip(segs2, env2)
            for ngram_size_i in ngram_sizes:
                ngrams1 = word1.get_ngrams(size=ngram_size_i, pad_ch=self.pad_ch)
                if ngram_size_i > 1:
                    ngrams1 = [SEG_JOIN_CH.join(ngram) for ngram in ngrams1]
                for ngram_size_j in ngram_sizes:
                    ngrams2 = word2.get_ngrams(size=ngram_size_j, pad_ch=self.pad_ch)
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
        corr_dict_l1l2 = defaultdict(lambda: defaultdict(lambda:0))
        corr_dict_l2l1 = defaultdict(lambda: defaultdict(lambda:0))
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
                    corr_dict_l1l2[seg_i][seg_j] += 1
                    if complex_idx2:
                        corr_dict_l2l1[seg_j][seg_i] += 1
                        for seg_j_j in seg_j:
                            corr_dict_l2l1[seg_j_j][seg_i] -= 1
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
                    corr_dict_l2l1[seg_j][seg_i] += 1
                    if complex_idx1:
                        corr_dict_l1l2[seg_i][seg_j] += 1
                        for seg_i_i in seg_i:
                            corr_dict_l1l2[seg_i_i][seg_j] -= 1
        
        # Prune correspondences which occur fewer than min_corr times
        corr_dict_l1l2 = prune_corrs(corr_dict_l1l2, min_val=min_corr)
        corr_dict_l2l1 = prune_corrs(corr_dict_l2l1, min_val=min_corr)
        
        # Remove keys with 0 values
        # (would occur from adjusting complex correspondences in preceding loop)
        corr_l1l2_to_delete = [seg_i for seg_i, inner_dict in corr_dict_l1l2.items() if sum(inner_dict.values()) < 1]
        for seg_i in corr_l1l2_to_delete:
            del corr_dict_l1l2[seg_i]
        corr_l2l1_to_delete = [seg_j for seg_j, inner_dict in corr_dict_l2l1.items() if sum(inner_dict.values()) < 1]
        for seg_j in corr_l2l1_to_delete:
            del corr_dict_l2l1[seg_j]

        return corr_dict_l1l2, corr_dict_l2l1

    def joint_probs(self, conditional_counts, l1=None, l2=None):
        """Converts a nested dictionary of conditional frequencies into a nested dictionary of joint probabilities"""
        l1, l2 = self.langs(l1=l1, l2=l2)
        joint_prob_dist = defaultdict(lambda: {})
        for seg1 in conditional_counts:
            seg1_totals = sum(conditional_counts[seg1].values())
            for seg2 in conditional_counts[seg1]:
                cond_prob = conditional_counts[seg1][seg2] / seg1_totals
                p_ind1 = l1.ngram_probability(Ngram(seg1))
                joint_prob = cond_prob * p_ind1
                joint_prob_dist[seg1][seg2] = joint_prob
        return joint_prob_dist

    def correspondence_probs(self,
                             alignment_list,
                             ngram_size=1,
                             counts=False,
                             min_corr=2,
                             exclude_null=True,
                             pad=False,
                             ):
        """Returns a dictionary of conditional phone probabilities, based on a list
        of Alignment objects.
        counts : Bool; if True, returns raw counts instead of normalized probabilities;
        exclude_null : Bool; if True, does not consider aligned pairs including a null segment"""

        corr_counts = defaultdict(lambda: defaultdict(lambda: 0))
        for alignment in alignment_list:
            if exclude_null:
                _alignment = alignment.remove_gaps()
            else:
                _alignment = alignment.alignment
            # Pad with at least one boundary position
            pad_n = 0
            if pad:
                pad_n = max(1, ngram_size - 1)
                _alignment = alignment.pad(ngram_size,
                                           alignment=_alignment,
                                           pad_ch=self.pad_ch,
                                           pad_n=pad_n)

            for i in range(ngram_size - 1, len(_alignment)):
                ngram = _alignment[i - (ngram_size - 1):i + 1]
                seg1, seg2 = list(zip(*ngram))
                corr_counts[seg1][seg2] += 1

        if min_corr > 1:
            corr_counts = prune_corrs(corr_counts, min_val=min_corr)

        if not counts:
            for seg1 in corr_counts:
                corr_counts[seg1] = normalize_dict(corr_counts[seg1])

        return corr_counts

    def phon_env_corr_probs(self, alignment_list, counts=False, ngram_size=1):
        if ngram_size > 1:
            raise NotImplementedError

        corr_counts = defaultdict(lambda: defaultdict(lambda: 0))
        for alignment in alignment_list:
            phon_env_align = alignment.add_phon_env()
            for phon_env_seg1, seg2 in phon_env_align:
                corr_counts[phon_env_seg1][seg2] += 1
        if not counts:
            for seg1 in corr_counts:
                corr_counts[seg1] = normalize_dict(corr_counts[seg1])

        return corr_counts

    def phoneme_pmi(self, conditional_counts, l1=None, l2=None, wordlist=None):
        """
        conditional_probs : nested dictionary of conditional correspondence probabilities in potential cognates
        """
        l1, l2 = self.langs(l1=l1, l2=l2)
        # Convert conditional probabilities to joint probabilities
        joint_prob_dist = self.joint_probs(conditional_counts, l1=l1, l2=l2)
        reverse_cond_counts = reverse_corr_dict(conditional_counts)

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
                for corr1 in joint_prob_dist
                for corr2 in joint_prob_dist[corr1]
                if corr1 not in l1.phonemes or corr2 not in l2.phonemes
            ]
        )
        # Extend with gaps in seq1, not included in joint_prob_dist (but in the seq2 nested dicts and therefore in reverse_cond_counts already)
        segment_pairs.update([(self.gap_ch, seg2) for seg2 in l2.phonemes])
        segment_pairs.update([(self.gap_ch, corr2) 
                              for corr1 in joint_prob_dist
                              for corr2 in joint_prob_dist[corr1]
                              if corr2 not in l2.phonemes])
        # Extend with gap and pad tokens
        segment_pairs.update([(self.gap_ch, start_token(self.pad_ch)), (self.gap_ch, end_token(self.pad_ch))])

        # If using a specific wordlist, extract sequences in each language
        if wordlist:
            wordlist = Wordlist(wordlist, pad_n=1)

        # Estimate probability of a gap in each language from conditional counts
        gap_counts1, gap_counts2 = 0, 0
        seg_counts1, seg_counts2 = 0, 0
        for corr1, corr2_dict in conditional_counts.items():
            for corr2, val in corr2_dict.items():
                if corr2 == self.gap_ch:
                    gap_counts2 += 1
                else:
                    seg_counts2 += 1
        for corr2, corr1_dict in reverse_cond_counts.items():
            if corr2 == self.gap_ch:
                gap_counts1 += sum(corr1_dict.values())
            else:
                seg_counts1 += sum(corr1_dict.values())
        gap_prob1 = gap_counts1 / (seg_counts1 + gap_counts1)
        gap_prob2 = gap_counts2 / (seg_counts2 + gap_counts2)

        # Calculate PMI for all phoneme pairs
        pmi_dict = defaultdict(lambda: defaultdict(lambda: 0))
        for seg1, seg2 in segment_pairs:
            seg1_ngram = Ngram(seg1)
            seg2_ngram = Ngram(seg2)
            # Skip full boundary gap alignments
            if self.pad_ch in seg1 and self.pad_ch in seg2:  # (seg1, seg2) == (self.pad_ch, self.pad_ch)
                continue

            # Standard alignment pairs and boundary gap alignments with segments
            else:

                # Calculation below gives a more precise probability specific to a certain subset of words,
                # which directly reflects shared coverage between l1 and l2.
                # Else, using lang.ngram_probability will consider all words in the vocabulary
                if wordlist:
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

                else: # TODO consolidate this block with the above as much as possible
                    if not seg1_ngram.is_gappy(self.gap_ch):
                        p_ind1 = l1.ngram_probability(seg1_ngram)
                    elif seg1_ngram.size == 1:
                        p_ind1 = gap_prob1
                    elif seg1_ngram.size > 1:
                        gapless_seg1_ngram = seg1_ngram.remove_gaps(self.gap_ch)
                        p_ind1 = l1.ngram_probability(gapless_seg1_ngram)
                    if not seg2_ngram.is_gappy(self.gap_ch):
                        p_ind2 = l2.ngram_probability(seg2_ngram)
                    elif seg2_ngram.size == 1:
                        p_ind2 = gap_prob2
                    elif seg1_ngram.size > 1:
                        gapless_seg2_ngram = seg2_ngram.remove_gaps(self.gap_ch)
                        p_ind2 = l2.ngram_probability(gapless_seg2_ngram)

                p_ind = p_ind1 * p_ind2
                joint_prob = joint_prob_dist.get(seg1, {}).get(seg2, p_ind)
                if p_ind1 == 0:
                    raise ValueError(f"Couldn't calculate independent probability of segment {seg1} in {self.lang1_name}")
                if p_ind2 == 0:
                    raise ValueError(f"Couldn't calculate independent probability of segment {seg2} in {self.lang2_name}")
                # As long as the independent probabilities > 0, skip calculating PMI for segment pairs with 0 joint probability
                if joint_prob > 0:
                    pmi_val = pointwise_mutual_info(joint_prob, p_ind1, p_ind2)
                    
                    # Add only non-zero PMI to dictionary
                    if pmi_val != 0:
                        pmi_dict[seg1_ngram.undo()][seg2_ngram.undo()] = pmi_val
        
        return pmi_dict

    def compute_phone_corrs(self,
                            p_threshold=0.1,
                            max_iterations=3,
                            n_samples=3,
                            sample_size=0.8,
                            min_corr=2,
                            max_ngram_size=2,
                            ngram_size=1, # dummy variable added temporarily to enable surprisal calculation together
                            phon_env=False,
                            cumulative=False,
                            ):
        """
        Parameters
        ----------
        p_threshold : float, optional
            p-value threshold for words to qualify for PMI calculation in the next iteration. The default is 0.05.
        max_iterations : float, optional
            Maximum number of iterations. The default is 10.
        samples : int, optional
            Number of random samples to draw. The default is 5.
        cumulative : bool, optional
            Whether PMI accumulates over iterations, continuing to consider alignments from earlier iterations. The default is False.
        Returns
        -------
        results : collections.defaultdict
            Nested dictionary of phoneme PMI values.
        """
        # TODO update documentation
        if self.logger:
            self.logger.info(f'Computing phone correspondences: {self.lang1_name}-{self.lang2_name}...')

        # Take a sample of same-meaning words, by default 80% of available same-meaning pairs
        sample_results = {}
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
            )
        else:
            sample_dict = {
                (start_seed, len(self.same_meaning)): (
                    self.same_meaning, random.sample(self.diff_meaning, len(self.same_meaning))
                )
            }
        final_qualifying = set()
        for key, sample in sample_dict.items():
            seed_i, _ = key
            sample_n = seed_i - start_seed
            synonym_sample, diff_sample = sample
            synonym_sample, diff_sample = map(sort_wordlist, [synonym_sample, diff_sample])

            # At each following iteration N, re-align using the pmi_stepN as an
            # additional penalty, and then recalculate PMI
            iteration = 0
            PMI_iterations: dict[int, dict[str, dict]] = {}
            qualifying_words = default_dict({iteration: synonym_sample}, lmbda=[])
            disqualified_words = default_dict({iteration: diff_sample}, lmbda=[])
            if cumulative:
                all_cognate_alignments = []

            def score_pmi(alignment: Alignment, pmi_dict: dict[str, dict]):  # TODO use more sophisticated pmi_dist from wordDist.py or word adaptation surprisal or alignment cost measure within Alignment object
                alignment_tuples = alignment.alignment
                PMI_score = mean([pmi_dict.get(pair[0], {}).get(pair[1], 0) for pair in alignment_tuples])
                return PMI_score

            while iteration < max_iterations and qualifying_words[iteration] != qualifying_words[iteration - 1]:
                iteration += 1
                qual_prev_sample = qualifying_words[iteration - 1]
                reversed_qual_prev_sample = [(pair[-1], pair[0]) for pair in qual_prev_sample]

                # Perform EM algorithm and fit IBM model 1 on ngrams of varying sizes
                em_synonyms1, em_synonyms2 = self.radial_em(
                    qual_prev_sample,
                    min_corr=min_corr,
                    max_ngram_size=max_ngram_size,
                    seed=seed_i,
                )

                # Calculate initial PMI for all ngram pairs
                pmi_dict_l1l2, pmi_dict_l2l1 = [
                    self.phoneme_pmi(
                        conditional_counts=em_synonyms1,
                        l1=self.lang1,
                        l2=self.lang2,
                        wordlist=qual_prev_sample,
                    ),
                    self.phoneme_pmi(
                        conditional_counts=em_synonyms2,
                        l1=self.lang2,
                        l2=self.lang1,
                        wordlist=reversed_qual_prev_sample,
                    )
                ]

                # Average together the PMI values from each direction
                pmi_step_i = average_corrs(pmi_dict_l1l2, pmi_dict_l2l1)

                # Align the qualifying words of the previous step using initial PMI
                cognate_alignments = self.align_wordlist(
                    qual_prev_sample,
                    align_costs=pmi_step_i
                )

                # Add cognate alignments into running pool of alignments
                if cumulative:
                    all_cognate_alignments.extend(cognate_alignments)
                    cognate_alignments = all_cognate_alignments

                # Recalculate correspondence probabilities and PMI values
                # from these alignments alone, i.e. not using radial EM
                # Reason for recalculating is that using alignments we can be stricter:
                # impose minimum corr requirements and only consider actually aligned segments
                cognate_probs = self.correspondence_probs(
                    cognate_alignments,
                    exclude_null=False,
                    counts=True,
                    min_corr=min_corr,
                )
                PMI_iterations[iteration] = self.phoneme_pmi(
                    cognate_probs,
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
                noncognate_PMI = []
                for alignment in noncognate_alignments:
                    noncognate_PMI.append(score_pmi(alignment, pmi_dict=PMI_iterations[iteration]))
                nc_mean = mean(noncognate_PMI)
                nc_stdev = stdev(noncognate_PMI)

                # Score same-meaning alignments for overall PMI and calculate p-value
                # against different-meaning alignments
                qualifying, disqualified = [], []
                qualifying_alignments = []
                qualified_PMI = []
                for q, pair in enumerate(synonym_sample):
                    alignment = aligned_synonym_sample[q]
                    PMI_score = score_pmi(alignment, pmi_dict=PMI_iterations[iteration])

                    # Proportion of non-cognate word pairs which would have a PMI score at least as low as this word pair
                    pnorm = 1 - norm.cdf(PMI_score, loc=nc_mean, scale=nc_stdev)
                    if pnorm < p_threshold:
                        qualifying.append(pair)
                        qualifying_alignments.append(alignment)
                        qualified_PMI.append(PMI_score)
                    else:
                        disqualified.append(pair)
                        # disqualified_PMI.append(PMI_score)
                qualifying, qualifying_alignments = prune_extraneous_synonyms(
                    qualifying,
                    qualifying_alignments,
                    qualified_PMI
                )
                qualifying_words[iteration] = sort_wordlist(qualifying)
                if len(qualifying_words[iteration]) == 0:
                    self.logger.warning(f'All word pairs were disqualified in PMI iteration {iteration}')
                disqualified_words[iteration] = disqualified + diff_sample

                # Log results of this iteration
                iter_log = self.log_iteration(iteration, qualifying_words, disqualified_words)
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
        final_alignments = self.align_wordlist(
            final_qualifying,
            align_costs=results,
        )
        # Log final alignments
        self.log_alignments(final_alignments, self.align_log['PMI'])
        
        # Compute phone surprisal
        self.compute_phone_surprisal(
            final_alignments,
            phon_env=phon_env,
            min_corr=min_corr,
            ngram_size=ngram_size,
        )
        # Compute surprisal in opposite direction with reversed alignments
        twin = self.get_twin()
        reversed_final_alignments = [alignment.reverse() for alignment in final_alignments]
        twin.compute_phone_surprisal(
            reversed_final_alignments,
            phon_env=phon_env,
            min_corr=min_corr,
            ngram_size=ngram_size,
        )

        # Write the iteration log
        log_file = os.path.join(self.phon_corr_dir, 'iterations.log')
        self.write_iter_log(iter_logs, log_file)

        # Write alignment log
        align_log_file = os.path.join(self.phon_corr_dir, 'alignments.log')
        self.write_alignments_log(self.align_log['PMI'], align_log_file)

        # Save PMI results
        self.lang1.phoneme_pmi[self.lang2_name] = results
        self.lang2.phoneme_pmi[self.lang1_name] = reverse_corr_dict(results)
        self.lang1.complex_ngrams[self.lang2_name] = self.complex_ngrams
        reversed_complex_ngrams = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        for corr1 in self.complex_ngrams:
            for corr2, val in self.complex_ngrams[corr1].items():
                reversed_complex_ngrams[corr2][corr1] = val
        self.lang2.complex_ngrams[self.lang1_name] = reversed_complex_ngrams
        # self.lang1.phoneme_pmi[self.lang2]['thresholds'] = noncognate_PMI

        self.pmi_dict = results
        self.log_phoneme_pmi()

        return results

    def phoneme_surprisal(self,
                          correspondence_counts,
                          phon_env_corr_counts=None,
                          ngram_size=1,
                          weights=None,
                          # attested_only=True,
                          # alpha=0.2 # TODO conduct more formal experiment to select default alpha, or find way to adjust automatically; so far alpha=0.2 is best (at least on Romance)
                          ):
        # Set phon_env bool
        phon_env = False
        if phon_env_corr_counts is not None:
            phon_env = True

        # Interpolation smoothing
        if weights is None:
            # Each ngram estimate will be weighted proportional to its size
            # Weight the estimate from a 2gram twice as much as 1gram, etc.
            weights = [i + 1 for i in range(ngram_size)]
        interpolation = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        for i in range(ngram_size, 0, -1):
            for ngram1 in correspondence_counts:
                for ngram2 in correspondence_counts[ngram1]:
                    # Exclude correspondences with a fully null ngram, e.g. ('k', 'a') with ('-', '-')
                    # Only needs to be done with ngram_size > 1
                    if (self.gap_ch,) * (max(ngram_size, 2)) not in [ngram1, ngram2]:
                        # forward
                        # interpolation[i][ngram1[-i:]][ngram2[-1]] += correspondence_counts[ngram1][ngram2]

                        # backward
                        interpolation[i][Ngram(ngram1).ngram][Ngram(ngram2).ngram] += correspondence_counts[ngram1][ngram2]

        # Add in phonological environment correspondences, e.g. ('l', '#S<') (word-initial 'l') with 'ʎ'
        if phon_env:
            for ngram1 in phon_env_corr_counts:
                if ngram1 == self.gap_ch or self.pad_ch in ngram1:
                    continue  # TODO verify that these should be skipped
                phon_env_ngram = PhonEnvNgram(ngram1)
                contexts = phon_env_ngrams(phon_env_ngram.phon_env, exclude={'|S|'})
                for context in contexts:
                    if len(phon_env_ngram.ngram) > 1:
                        ngram1_context = (phon_env_ngram.ngram,) + (context,)
                    else:
                        ngram1_context = phon_env_ngram.ngram + (context,)
                    for ngram2 in phon_env_corr_counts[ngram1]:
                        # backward
                        interpolation['phon_env'][ngram1_context][Ngram(ngram2).ngram] += phon_env_corr_counts[ngram1][ngram2]

        # Get lists of possible ngrams in lang1 and lang2
        # lang2 ngram size fixed at 1, only trying to predict single phone; also not trying to predict its phon_env
        all_ngrams_lang2 = self.get_possible_ngrams(self.lang2, ngram_size=1, phon_env=False)
        all_ngrams_lang2.update(reverse_corr_dict(interpolation[i]).keys())  # add any complex ngram corrs
        all_ngrams_lang1 = self.get_possible_ngrams(
            lang=self.lang1,
            # phon_env ngrams fixed at size 1
            ngram_size=ngram_size if not phon_env else 1,
            phon_env=phon_env
        )
        # Add complex ngram corrs to all_ngrams_lang1
        if phon_env:
            for complex_ngram in self.complex_ngrams:
                if complex_ngram.size > 1:
                    if self.pad_ch in complex_ngram.ngram[0]:  # complex ngram alignment with start boundary
                        ngram_envs = set(ngram[-1] for ngram in all_ngrams_lang1 if ngram[:-1] == complex_ngram.ngram[1:] and re.search(r'#\|S', ngram[-1]))
                    elif self.pad_ch in complex_ngram.ngram[-1]:  # complex ngram alignment with end boundary
                        ngram_envs = set(ngram[-1] for ngram in all_ngrams_lang1 if ngram[:-1] == complex_ngram.ngram[:-1] and re.search(r'S\|#', ngram[-1]))
                    else:  # other complex ngram alignments
                        # get post environments of first segment and preceding environments of last segment in complex ngram
                        first_seg, last_seg = map(_toSegment, [complex_ngram.ngram[0], complex_ngram.ngram[-1]])
                        if complex_ngram.size == 2:
                            next_seg, penult_seg = last_seg, first_seg
                        else:
                            next_seg, penult_seg = map(_toSegment, [complex_ngram.ngram[1], complex_ngram.ngram[-2]])
                        rel_post_son = relative_post_sonority(first_seg, next_seg)
                        rel_prev_son = relative_prev_sonority(last_seg, penult_seg)
                        first_envs = set(ngram[-1].split('|')[0] for ngram in all_ngrams_lang1 if ngram[:-1] == complex_ngram.ngram[:-1] and re.search(rf'S\|{rel_post_son}', ngram[-1]))
                        last_envs = set(ngram[-1].split('|')[-1] for ngram in all_ngrams_lang1 if ngram[:-1] == complex_ngram.ngram[1:] and re.search(rf'{rel_prev_son}\|S', ngram[-1]))
                        ngram_envs = set(f'{first_env}|S|{last_env}' for first_env, last_env in product(first_envs, last_envs))

                    complex_envs = set()
                    for ngram_env in ngram_envs:
                        phon_env_ngram = PhonEnvNgram((complex_ngram.ngram, ngram_env))
                        complex_envs.add(phon_env_ngram.ngram_w_context)
                    all_ngrams_lang1.update(complex_envs)
        else:
            all_ngrams_lang1.update(interpolation[i].keys())

        smoothed_surprisal = defaultdict(lambda: defaultdict(lambda: self.lang2.phoneme_entropy * ngram_size))
        for ngram1 in all_ngrams_lang1:
            if self.gap_ch in ngram1:
                continue

            if phon_env:
                if not (len(ngram1) == 1 and self.pad_ch in ngram1[0]): # skip adding phon env to e.g. ('#>',)
                    ngram1_phon_env = PhonEnvNgram(ngram1)
                    ngram1 = ngram1_phon_env.ngram
                    if sum(interpolation['phon_env'][ngram1_phon_env.ngram_w_context].values()) == 0:
                        # TODO verify that these should be skipped
                        continue

            if sum(interpolation[i][ngram1].values()) == 0:
                continue

            undone_ngram1 = Ngram(ngram1).undo()  # need to convert to dictionary form, whereby unigrams are strings and larger ngrams are tuples
            for ngram2 in all_ngrams_lang2:
                if interpolation[i][ngram1].get(ngram2, 0) == 0:
                    continue

                ngram_weights = weights[:]
                # forward # TODO has not been updated since before addition of phon_env
                # estimates = [interpolation[i][ngram1[-i:]][ngram2] / sum(interpolation[i][ngram1[-i:]].values())
                #              if i > 1 else lidstone_smoothing(x=interpolation[i][ngram1[-i:]][ngram2],
                #                                               N=sum(interpolation[i][ngram1[-i:]].values()),
                #                                               d = len(self.lang2.phonemes) + 1)
                #              for i in range(ngram_size,0,-1)]
                # backward
                estimates = [interpolation[i][ngram1].get(ngram2, 0) / sum(interpolation[i][ngram1].values())
                             for i in range(ngram_size, 0, -1)]
                # estimates = [lidstone_smoothing(x=interpolation[i][ngram1[:i]].get(ngram2, 0),
                #                                 N=sum(interpolation[i][ngram1[:i]].values()),
                #                                 d = len(self.lang2.phonemes) + 1,
                #                                 # modification: I believe the d (vocabulary size) value should be every combination of phones from lang1 and lang2
                #                                 #d = n_ngram_pairs + 1,
                #                                 # updated mod: it should actually be the vocabulary GIVEN the phone of lang1, otherwise skewed by frequency of phone1
                #                                 #d = len(interpolation[i][ngram1[:i]]),
                #                                 alpha=alpha)
                #             for i in range(ngram_size,0,-1)]

                # add interpolation with phon_env surprisal
                if phon_env:
                    if not (len(ngram1) == 1 and self.pad_ch in ngram1[0]): # skip computing phon env probabilities with e.g. ('#>',)
                        ngram1_w_context = ngram1_phon_env.ngram_w_context
                        phonEnv_contexts = phon_env_ngrams(ngram1_phon_env.phon_env, exclude={'|S|'})
                        for context in phonEnv_contexts:
                            estimates.append(interpolation['phon_env'][ngram1_w_context].get(ngram2, 0) / sum(interpolation['phon_env'][ngram1_w_context].values()))
                            # estimates.append(lidstone_smoothing(x=interpolation['phon_env'][(ngram1_context,)].get(ngram2, 0),
                            #                                     N=sum(interpolation['phon_env'][(ngram1_context,)].values()),
                            #                                     d = len(self.lang2.phonemes) + 1,
                            #                                     alpha=alpha)
                            #                                     )
                            # Weight each contextual estimate based on the size of the context
                            ngram_weights.append(get_phonEnv_weight(context))

                weight_sum = sum(ngram_weights)
                ngram_weights = [i / weight_sum for i in ngram_weights]
                assert (len(ngram_weights) == len(estimates))
                smoothed = sum([estimate * weight for estimate, weight in zip(estimates, ngram_weights)])
                undone_ngram2 = Ngram(ngram2).undo()  # need to convert to dictionary form, whereby unigrams are strings and larger ngrams are tuples
                if phon_env:
                    if not (len(ngram1) == 1 and self.pad_ch in ngram1[0]):  # skip computing phon env probabilities with e.g. ('#>',)
                        smoothed_surprisal[ngram1_w_context][undone_ngram2] = surprisal(smoothed)
                else:
                    smoothed_surprisal[undone_ngram1][undone_ngram2] = surprisal(smoothed)

            # oov_estimates = [lidstone_smoothing(x=0, N=sum(interpolation[i][ngram1[:i]].values()),
            #                                  d = len(self.lang2.phonemes) + 1,
            #                                  alpha=alpha)
            #               for i in range(ngram_size,0,-1)]
            # if phon_env:
            #     for context in phonEnv_contexts:
            #         ngram1_context = ngram1_phon_env[0][:-1] + (context,)
            #         oov_estimates.append(lidstone_smoothing(x=0,
            #                                                 N=sum(interpolation['phon_env'][(ngram1_context,)].values()),
            #                                                 d = len(self.lang2.phonemes) + 1,
            #                                                 alpha=alpha)
            #         )
            # assert (len(ngram_weights) == len(oov_estimates))
            # smoothed_oov = max(surprisal(sum([estimate*weight for estimate, weight in zip(oov_estimates, ngram_weights)])), self.lang2.phoneme_entropy)
            smoothed_oov = self.lang2.phoneme_entropy

            if phon_env:
                if not (len(ngram1) == 1 and self.pad_ch in ngram1[0]):  # skip computing phon env probabilities with e.g. ('#>',)
                    smoothed_surprisal[ngram1_w_context] = default_dict(smoothed_surprisal[ngram1_w_context], lmbda=smoothed_oov)
            else:
                smoothed_surprisal[undone_ngram1] = default_dict(smoothed_surprisal[undone_ngram1], lmbda=smoothed_oov)

            # Prune saved surprisal values which exceed the phoneme entropy of lang2
            if phon_env:
                key = ngram1_w_context
            else:
                key = undone_ngram1
            to_prune = [ngram2 for ngram2 in smoothed_surprisal[key] if smoothed_surprisal[key][ngram2] > self.lang2.phoneme_entropy]
            for ngram_to_prune in to_prune:
                del smoothed_surprisal[key][ngram_to_prune]

        return smoothed_surprisal

    def compute_phone_surprisal(self,
                                alignments,
                                phon_env=False,
                                min_corr=2,
                                ngram_size=1, # TODO remove if not going to be developed further
                                ):
        """Computes phone surprisal based on aligned phone sequences.

        Args:
            alignments (list): list of Alignment objects representing aligned phone sequences 
            phon_env (bool, optional): Computes surprisal taking phonological environment into account. Defaults to False.

        Returns:
            (surprisal_results, phon_env_surprisal_results): tuple containing dictionaries of surprisal results.
            If phon_env=False, phon_env_surprisal_results will be None.
        """
        # Get correspondence probabilities from alignments
        corr_probs = self.correspondence_probs(
            alignments,
            counts=True,
            min_corr=min_corr,
            exclude_null=False,
            #ngram_size=ngram_size,
        )

        # Optionally calculate phonological environment correspondence probabilities
        phon_env_corr_counts = None
        if phon_env:
            phon_env_corr_counts = self.phon_env_corr_probs(
                alignments,
                counts=True,
                #ngram_size=ngram_size
            )
            
        # Calculate surprisal based on correspondence probabilities
        # NB: Will be phon_env surprisal if phon_env=True and therefore phon_env_corr_counts != None
        surprisal_results = self.phoneme_surprisal(
            corr_probs,
            phon_env_corr_counts=phon_env_corr_counts
            #ngram_size=ngram_size
        )
        # Get vanilla surprisal (no phon env) from phon_env surprisal by marginalizing over phon_env
        if phon_env:
            phon_env_surprisal_results = surprisal_results.copy()
            surprisal_results = self.marginalize_over_phon_env_surprisal(
                surprisal_results, 
                #ngram_size=ngram_size
            )
        
        # Save surprisal results
        self.lang1.phoneme_surprisal[self.lang2_name][ngram_size] = surprisal_results
        self.surprisal_dict[ngram_size] = surprisal_results
        if phon_env:
            self.lang1.phon_env_surprisal[self.lang2_name] = phon_env_surprisal_results
            self.phon_env_surprisal_dict = phon_env_surprisal_results
            
        # Write phone correlation report based on surprisal results
        phon_corr_report = os.path.join(self.phon_corr_dir, 'phon_corr.tsv')
        self.write_phon_corr_report(surprisal_results, phon_corr_report, type='surprisal')

        # Write surprisal logs
        self.log_phoneme_surprisal(phon_env=False, ngram_size=ngram_size)
        if phon_env:
            self.log_phoneme_surprisal(phon_env=True)
        
        if phon_env:
            return surprisal_results, phon_env_surprisal_results

        return surprisal_results, None

    def marginalize_over_phon_env_surprisal(self, phon_env_surprisal_dict, ngram_size=1):
        """Converts a phon env surprisal dictionary into a vanilla surprisal dictionary by marginalizing over phon envs"""
        surprisal_dict = defaultdict(lambda: defaultdict(lambda: 0))
        oov_vals = defaultdict(lambda: [])
        for phon_env_ngram in phon_env_surprisal_dict:
            if isinstance(phon_env_ngram, str):
                assert phon_env_ngram == self.gap_ch or self.pad_ch in phon_env_ngram
                ngram1 = phon_env_ngram
            elif isinstance(phon_env_ngram, tuple):
                assert len(phon_env_ngram) == 2
                ngram1, phon_env = phon_env_ngram
            else:
                raise TypeError
            count = self.lang1.phon_env_ngrams[ngram_size][phon_env_ngram]
            oov_val = get_oov_val(phon_env_surprisal_dict[phon_env_ngram])
            oov_vals[ngram1].append(oov_val)
            for ngram2, surprisal_val in phon_env_surprisal_dict[phon_env_ngram].items():
                if surprisal_val >= oov_val:
                    continue
                prob = surprisal_to_prob(surprisal_val)
                count_ngram2 = prob * count
                if count_ngram2 > 0:
                    surprisal_dict[ngram1][ngram2] += count_ngram2
        for ngram1 in surprisal_dict:
            inner_surprisal_dict = normalize_dict(surprisal_dict[ngram1])
            surprisal_dict[ngram1] = default_dict(inner_surprisal_dict, lmbda=mean(oov_vals[ngram1]))
            for ngram2, prob in surprisal_dict[ngram1].items():
                surprisal_dict[ngram1][ngram2] = surprisal(prob)
        outer_oov_val = get_oov_val(phon_env_surprisal_dict)
        surprisal_dict, oov_value = prune_oov_surprisal(default_dict(surprisal_dict, lmbda=outer_oov_val))
        return surprisal_dict

    def noncognate_thresholds(self, eval_func, sample_size=None, save=True, seed=None):
        """Calculate non-synonymous word pair scores against which to calibrate synonymous word scores"""

        # Take a sample of different-meaning words, by default as large as the same-meaning set
        if sample_size is None:
            sample_size = len(self.same_meaning)

        # Set random seed: may or may not be the default seed attribute of the PhonCorrelator class
        if not seed:
            seed = self.seed
        random.seed(seed)

        diff_sample = random.sample(self.diff_meaning, min(sample_size, len(self.diff_meaning)))
        noncognate_scores = []
        func_key = (eval_func, eval_func.hashable_kwargs)
        for pair in diff_sample:
            if pair in self.scored_words[func_key]:
                noncognate_scores.append(self.scored_words[func_key][pair])
            else:
                score = eval_func.eval(pair[0], pair[1])
                noncognate_scores.append(score)
                self.scored_words[func_key][pair] = score
        self.reset_seed()

        if save:
            key = (self.lang2_name, eval_func, sample_size, seed)
            self.lang1.noncognate_thresholds[key] = noncognate_scores

        return noncognate_scores

    def log_phoneme_pmi(self, outfile=None, threshold=0.0001, sep='\t'):
        # Save calculated PMI values to file
        if outfile is None:
            outfile = os.path.join(self.phon_corr_dir, 'phonPMI.tsv')

        # Save all segment pairs with non-zero PMI values to file
        # Skip extremely small decimals that are close to zero
        lines = []
        for seg1 in self.pmi_dict:
            for seg2 in self.pmi_dict[seg1]:
                pmi_val = round(self.pmi_dict[seg1][seg2], 3)
                if abs(pmi_val) > threshold:
                    line = [ngram2log_format(seg1), ngram2log_format(seg2), str(pmi_val)]
                    lines.append(line)
        # Sort PMI in descending order
        lines = sorted(lines, key=lambda x: x[-1], reverse=True)
        lines = '\n'.join([sep.join(line) for line in lines])

        with open(outfile, 'w') as f:
            header = sep.join(['Phone1', 'Phone2', 'PMI'])
            f.write(f'{header}\n{lines}')

    def log_phoneme_surprisal(self, outfile=None, sep='\t', phon_env=True, ngram_size=1):
        if outfile is None:
            if phon_env:
                outfile = os.path.join(self.phon_corr_dir, 'phonEnvSurprisal.tsv')
            else:
                outfile = os.path.join(self.phon_corr_dir, 'phonSurprisal.tsv')
        outdir = os.path.abspath(os.path.dirname(outfile))
        os.makedirs(outdir, exist_ok=True)

        if phon_env:
            surprisal_dict = self.phon_env_surprisal_dict
        else:
            surprisal_dict = self.surprisal_dict[ngram_size]

        lines = []
        surprisal_dict, oov_value = prune_oov_surprisal(surprisal_dict)
        oov_value = round(oov_value, 3)
        for seg1 in surprisal_dict:
            for seg2 in surprisal_dict[seg1]:
                if ngram_size > 1:
                    raise NotImplementedError  # TODO need to decide format for how to save/load larger ngrams from logs; previously they were separated by whitespace
                if phon_env:
                    seg1_str, phon_env = ngram2log_format(seg1, phon_env=True)
                else:
                    seg1_str = ngram2log_format(seg1, phon_env=False)
                lines.append([
                    seg1_str,
                    ngram2log_format(seg2, phon_env=False),  # phon_env only on seg1
                    str(round(surprisal_dict[seg1][seg2], 3)),
                    str(oov_value)
                    ]
                )
                if phon_env:
                    lines[-1].insert(1, phon_env)

        # Sort by phone1 (by phon env if relevant) and then by surprisal in ascending order
        if phon_env:
            lines = sorted(lines, key=lambda x: (x[0], x[1], x[3], x[2]), reverse=False)
        else:
            lines = sorted(lines, key=lambda x: (x[0], x[2], x[1]), reverse=False)
        lines = '\n'.join([sep.join(line) for line in lines])
        with open(outfile, 'w') as f:
            header = ['Phone1', 'Phone2', 'Surprisal', 'OOV_Smoothed']
            if phon_env:
                header.insert(1, "PhonEnv")
            header = sep.join(header)
            f.write(f'{header}\n{lines}')

    def log_sample(self, sample, sample_n, seed=None):
        if seed is None:
            seed = self.seed
        sample = sorted(
            [
                f'[{word1.concept}] {word1.orthography} /{word1.ipa}/ - {word2.orthography} /{word2.ipa}/'
                for word1, word2 in sample
            ]
        )
        sample_log = f'SAMPLE: {sample_n}\nSEED: {seed}\n'
        sample_log += '\n'.join(sample)
        return sample_log

    def write_sample_log(self, sample_logs, log_file):
        log_dir = os.path.abspath(os.path.dirname(log_file))
        os.makedirs(log_dir, exist_ok=True)
        content = '\n\n'.join([sample_logs[sample_n] for sample_n in range(len(sample_logs))])
        with open(log_file, 'w') as f:
            f.write(content)

    def log_iteration(self, iteration, qualifying_words, disqualified_words, method=None, same_meaning_alignments=None):
        iter_log = []
        if method == 'surprisal':
            assert same_meaning_alignments is not None

            def get_word_pairs(indices, lst):
                aligns = [lst[i] for i in indices]
                pairs = [(align.word1, align.word2) for align in aligns]
                return pairs

            qualifying = get_word_pairs(qualifying_words[iteration], same_meaning_alignments)
            prev_qualifying = get_word_pairs(qualifying_words[iteration - 1], same_meaning_alignments)
            disqualified = get_word_pairs(disqualified_words[iteration], same_meaning_alignments)
            prev_disqualified = get_word_pairs(disqualified_words[iteration - 1], same_meaning_alignments)
        else:
            qualifying = qualifying_words[iteration]
            prev_qualifying = qualifying_words[iteration - 1]
            disqualified = disqualified_words[iteration]
            prev_disqualified = disqualified_words[iteration - 1]
        iter_log.append(f'Iteration {iteration}')
        iter_log.append(f'\tQualified: {len(qualifying)}')
        iter_log.append(f'\tDisqualified: {len(disqualified)}')
        added = set(qualifying) - set(prev_qualifying)
        iter_log.append(f'\tAdded: {len(added)}')
        for word1, word2 in sort_wordlist(added):
            iter_log.append(f'\t\t{word1.orthography} /{word1.ipa}/ - {word2.orthography} /{word2.ipa}/')
        removed = set(disqualified) - set(prev_disqualified)
        iter_log.append(f'\tRemoved: {len(removed)}')
        for word1, word2 in sort_wordlist(removed):
            iter_log.append(f'\t\t{word1.orthography} /{word1.ipa}/ - {word2.orthography} /{word2.ipa}/')

        iter_log = '\n'.join(iter_log)

        return iter_log

    def write_iter_log(self, iter_logs, log_file):
        log_dir = os.path.abspath(os.path.dirname(log_file))
        os.makedirs(log_dir, exist_ok=True)
        with open(log_file, 'w') as f:
            f.write(f'Same meaning pairs: {len(self.same_meaning)}\n')
            for n in iter_logs:
                iter_log = '\n\n'.join(iter_logs[n][:-1])
                f.write(f'****SAMPLE {n+1}****\n')
                f.write(iter_log)
                final_qualifying, final_disqualified = iter_logs[n][-1]
                f.write('\n\nFinal qualifying:\n')
                for word1, word2 in sort_wordlist(final_qualifying):
                    f.write(f'\t\t{word1.orthography} /{word1.ipa}/ - {word2.orthography} /{word2.ipa}/\n')
                f.write('\nFinal disqualified:\n')
                for word1, word2 in sort_wordlist(final_disqualified):
                    f.write(f'\t\t{word1.orthography} /{word1.ipa}/ - {word2.orthography} /{word2.ipa}/\n')
                f.write('\n\n-------------------\n\n')

    def log_alignments(self, alignments, align_log):
        for alignment in alignments:
            key = f'/{alignment.word1.ipa}/ - /{alignment.word2.ipa}/'
            align_str = visual_align(alignment.alignment, gap_ch=alignment.gap_ch)
            align_log[key][align_str] += 1

    def write_alignments_log(self, alignment_log, log_file):
        sorted_alignment_keys = sorted(alignment_log.keys())
        with open(log_file, 'w') as f:
            for key in sorted_alignment_keys:
                f.write(f'{key}\n')
                sorted_alignments = dict_tuplelist(alignment_log[key])
                sorted_alignments.sort(key=lambda x: (x[-1], x[0]), reverse=True)
                for alignment, count in sorted_alignments:
                    freq = f'{count}/{sum(alignment_log[key].values())}'
                    f.write(f'[{freq}] {alignment}\n')
                f.write('\n-------------------\n\n')

    def write_phon_corr_report(self, corr, outfile, type, min_prob=0.05):
        lines = []
        corr, _ = prune_oov_surprisal(corr)
        l1_phons = sorted([p for p in corr if self.gap_ch not in p], key=lambda x: Ngram(x).string)
        for p1 in l1_phons:
            p2_candidates = corr[p1]
            if len(p2_candidates) > 0:
                p2_candidates = dict_tuplelist(p2_candidates, reverse=True)
                for p2, score in p2_candidates:
                    if type == 'surprisal':
                        prob = surprisal_to_prob(score)  # turn surprisal value into probability
                        if prob >= min_prob:
                            p1 = Ngram(p1).string
                            p2 = Ngram(p2).string
                            line = [p1, p2, str(round(prob, 3))]
                            lines.append(line)
                    else:
                        raise NotImplementedError  # not implemented for PMI
        # Sort by corr value, then by phone string if values are equal
        lines.sort(key=lambda x: (x[-1], x[0], x[1]))
        lines = ['\t'.join(line) for line in lines]
        header = '\t'.join([self.lang1_name, self.lang2_name, 'probability'])
        lines = '\n'.join(lines)
        content = '\n'.join([header, lines])
        with open(outfile, 'w') as f:
            f.write(f'{content}')


@lru_cache(maxsize=None)
def get_phonEnv_weight(phonEnv):
    # Weight contextual estimate based on the size of the context
    # #|S|< would have weight 3 because it considers the segment plus context on both sides
    # #|S would have weight 2 because it considers only the segment plus context on one side
    # #|S|<_l would have weight 4 because it the context on both sides, with two attributes of RHS context
    # #|S|<_ALVEOLAR_l would have weight 5 because it the context on both sides, with three attributes of RHS context
    prefix, base, suffix = phonEnv.split('|')
    weight = 1
    prefix = [p for p in prefix.split('_') if p != '']
    suffix = [s for s in suffix.split('_') if s != '']
    weight += len(prefix)
    weight += len(suffix)
    return weight
