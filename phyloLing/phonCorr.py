from collections import defaultdict
from functools import lru_cache
from itertools import product, combinations
from math import log
import os
import random
import re
from scipy.stats import norm
from statistics import mean, stdev
from phonAlign import Alignment, compatible_segments, visual_align
from phonUtils.segment import _toSegment
from phonUtils.phonEnv import relative_prev_sonority, relative_post_sonority
from constants import PHON_ENV_JOIN_CH, START_PAD_CH, END_PAD_CH, GAP_CH_DEFAULT, PAD_CH_DEFAULT
from utils.sequence import Ngram, flatten_ngram, count_subsequences, pad_sequence
from utils.information import surprisal, adaptation_surprisal, pointwise_mutual_info, bayes_pmi, surprisal_to_prob
from utils.utils import default_dict, normalize_dict, dict_tuplelist

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

def average_corrs(corr_dict1, corr_dict2):
    # Average together values from nested dictionaries in opposite directions
    avg_corr = defaultdict(lambda:defaultdict(lambda:0))
    for seg1 in corr_dict1:
        for seg2 in corr_dict1[seg1]:
            avg_corr[seg1][seg2] = mean([corr_dict1[seg1][seg2], corr_dict2[seg2][seg1]])
    return avg_corr

def average_nested_dicts(dict_list, default=0):
    corr1_all = set(corr1 for d in dict_list for corr1 in d)
    corr2_all = {corr1:set(corr2 for d in dict_list for corr2 in d[corr1]) for corr1 in corr1_all}
    results = defaultdict(lambda:defaultdict(lambda:0))
    for corr1 in corr1_all:
        for corr2 in corr2_all[corr1]:
            vals = []
            for d in dict_list:
                vals.append(d.get(corr1, {}).get(corr2, default))
            if len(vals) > 0:
                results[corr1][corr2] = mean(vals)
    return results

def reverse_corr_dict(corr_dict):
    reverse = defaultdict(lambda:defaultdict(lambda:0))
    for seg1 in corr_dict:
        for seg2 in corr_dict[seg1]:
            reverse[seg2][seg1] = corr_dict[seg1][seg2]
    return reverse

def ngram_count_word(ngram, word):
    count = 0
    for i in range(len(word) - len(ngram) + 1):
        if word[i:i+len(ngram)] == list(ngram):
            count += 1
    return count

def ngram_count_wordlist(ngram, seq_list):
    """Retrieve the count of an ngram of segments from a list of segment sequences"""
    count = 0
    for seq in seq_list:
        count += ngram_count_word(ngram, seq)
    return count

def ngram2str(ngram, phon_env=False):
    if phon_env:
        ngram, phon_env = ngram[:-1], ngram[-1]
        return PHON_ENV_JOIN_CH.join([Ngram(ngram).string, phon_env])
    else:
        return Ngram(ngram).string

class PhonCorrelator:
    def __init__(self, lang1, lang2, wordlist=None, gap_ch=GAP_CH_DEFAULT, pad_ch=PAD_CH_DEFAULT, seed=1, logger=None):        
        # Set Language objects
        self.lang1 = lang1
        self.lang2 = lang2
        
        # Alignment parameters
        self.gap_ch = gap_ch
        self.pad_ch = pad_ch
        self.seed = seed
        
        # Prepare wordlists: sort out same/different-meaning words and loanwords
        self.wordlist = self.get_concept_list(wordlist)
        self.same_meaning, self.diff_meaning, self.loanwords = self.prepare_wordlists()
        self.samples = {}
        
        # PMI, ngrams, scored words
        self.pmi_dict = self.lang1.phoneme_pmi[self.lang2]
        self.surprisal_dict = self.lang1.phoneme_surprisal[self.lang2]
        self.phon_env_surprisal_dict = self.lang1.phon_env_surprisal[self.lang2]
        self.total_unigrams = (sum([len(word1.segments) for word1, word2 in self.same_meaning]),
                               sum([len(word2.segments) for word1, word2 in self.same_meaning]))
        self.complex_ngrams = self.lang1.complex_ngrams[self.lang2]
        self.scored_words = defaultdict(lambda:{})
        
        # Logging
        self.outdir = self.lang1.family.phone_corr_dir
        self.pmi_log_dir, self.surprisal_log_dir = self.log_dirs()
        self.align_log = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0)))
        self.logger = logger
    
    def reset_seed(self):
        random.seed(self.seed)
        
    def langs(self, l1=None, l2=None):
        if l1 is None:
            l1 = self.lang1
        if l2 is None:
            l2 = self.lang2
        return l1, l2

    def log_dirs(self):
        pmi_log_dir = os.path.join(self.outdir, 'pmi', self.lang1.path_name, self.lang2.path_name)
        surprisal_log_dir = os.path.join(self.outdir, 'surprisal', self.lang1.path_name, self.lang2.path_name)
        os.makedirs(pmi_log_dir, exist_ok=True)
        os.makedirs(surprisal_log_dir, exist_ok=True)
        return pmi_log_dir, surprisal_log_dir
    
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
        
        samples = {}
        if log_samples:
            sample_logs = {}
        for sample_n in range(n_samples):
            seed_i = start_seed+sample_n
            rng = random.Random(seed_i)
            synonym_sample = rng.sample(self.same_meaning, sample_size)
             # Log sample
            if log_samples:
                sample_log = self._log_sample(synonym_sample, sample_n, seed=seed_i)
                sample_logs[sample_n] = sample_log
            # Take a sample of different-meaning words, as large as the same-meaning set
            diff_sample = rng.sample(self.diff_meaning, min(sample_size, len(self.diff_meaning)))
            samples[sample_n] = (synonym_sample, diff_sample)
        self.samples.update(samples)
        
        # Write sample log
        if log_samples:
            sample_log_file = os.path.join(self.outdir, 'samples', self.lang1.path_name, self.lang2.path_name, 'samples.log')
            self._write_sample_log(sample_logs, sample_log_file)
        
        return samples
    
    def pad_wordlist(self, wordlist, pad_n=1):
        def _pad(seq):
            return pad_sequence(seq, pad_ch=self.pad_ch, pad_n=pad_n)
        return [map(_pad, word) for word in wordlist]

    def align_wordlist(self,
                       wordlist,
                       added_penalty_dict=None,
                       complex_ngrams=None,
                       **kwargs):
        """Returns a list of the aligned segments from the wordlists"""
        alignment_list = [
            Alignment(
                seq1=word1, 
                seq2=word2,
                added_penalty_dict=added_penalty_dict,
                gap_ch=self.gap_ch,
                **kwargs
            )
            for word1, word2 in wordlist
        ] # TODO: tuple would be better than list if possible
        if complex_ngrams:
            compacted_alignments = self.compact_alignments(alignment_list, complex_ngrams)
            return compacted_alignments
        else:
            return alignment_list
    
    def compact_alignments(self, alignment_list, complex_ngrams):
        for alignment in alignment_list:
            alignment.compact_gaps(complex_ngrams)
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
            all_ngrams = product(phone_contexts+[f'{self.pad_ch}{END_PAD_CH}', f'{START_PAD_CH}{self.pad_ch}', self.gap_ch], repeat=ngram_size)
            
        else:
            attested = lang.list_ngrams(ngram_size, phon_env=False)
            all_ngrams = product(list(lang.phonemes.keys())+[f'{self.pad_ch}{END_PAD_CH}', f'{START_PAD_CH}{self.pad_ch}', self.gap_ch], repeat=ngram_size) 

        gappy = set(ngram for ngram in all_ngrams if self.gap_ch in ngram)
        all_ngrams = gappy.union(attested.keys())
        all_ngrams = set(Ngram(ngram).ngram for ngram in all_ngrams) # standardize forms
        return all_ngrams

    def radial_counts(self, wordlist, radius=1, normalize=True):
        """Checks the number of times that phones occur within a specified 
        radius of positions in their respective words from one another"""
        corr_dict = defaultdict(lambda:defaultdict(lambda:0))
        for word1, word2 in wordlist:
            segs1, segs2 = word1.segments, word2.segments
            for i in range(len(segs1)):
                seg1 = segs1[i]
                for j in range(max(0, i-radius), min(i+radius+1, len(segs2))):
                    seg2 = segs2[j]
                    
                    # Only count sounds which are compatible as corresponding
                    if compatible_segments(seg1, seg2):
                        corr_dict[seg1][seg2] += 1
                        
        if normalize:
            for seg1 in corr_dict:
                corr_dict[seg1] = normalize_dict(corr_dict[seg1])
        
        return corr_dict
    
    def joint_probs(self, conditional_counts, l1=None, l2=None):
        """Converts a nested dictionary of conditional frequencies into a nested dictionary of joint probabilities"""
        l1, l2 = self.langs(l1=l1, l2=l2)
        joint_prob_dist = defaultdict(lambda:{})
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
                             prune=None,
                             exclude_null=True,
                             compact_null=True,
                             pad=True,
                             ):
        """Returns a dictionary of conditional phone probabilities, based on a list
        of Alignment objects.
        counts : Bool; if True, returns raw counts instead of normalized probabilities;
        exclude_null : Bool; if True, does not consider aligned pairs including a null segment"""
        
        # Purpose of padding is to achieve null alignment compacting, so no need to pad if this is not being performed
        if compact_null is False:
            pad = False
        
        corr_counts = defaultdict(lambda:defaultdict(lambda:0))
        if compact_null:
            null_compacter = NullCompacter(corr_counts, 
                                           alignment_list, 
                                           ngram_size=ngram_size, 
                                           gap_ch=self.gap_ch,
                                           pad_ch=self.pad_ch,
                                           lang1=self.lang1, 
                                           lang2=self.lang2)
        for alignment in alignment_list:
            if exclude_null and compact_null:
                _alignment = alignment.alignment # do nothing, will get excluded on recursive call when compact_null=False
            elif exclude_null:
                _alignment = alignment.remove_gaps()
            else:
                _alignment = alignment.alignment
            # Pad with at least one boundary position
            pad_n = 0
            if pad:
                pad_n = max(1, ngram_size-1)
                _alignment = alignment.pad(ngram_size, 
                                        _alignment, 
                                        pad_ch=self.pad_ch, 
                                        pad_n=pad_n,
                )
                alignment.alignment = _alignment
                
            for i in range(pad_n, len(_alignment)):
                ngram = _alignment[i-min(pad_n,ngram_size-1):i+1]
                seg1, seg2 = list(zip(*ngram))
                corr_counts[seg1][seg2] += 1
                if compact_null:
                    null_compacter.compact_null(_alignment, i, ngram)
                    
        if compact_null:
            #complex_ngrams = null_compacter.combine_corrs()        
            # Prune any without at least 2 occurrences
            null_compacter.prune(min_val=2)
            self.complex_ngrams = null_compacter.select_valid_null_corrs()
            compacted_alignments = self.compact_alignments(alignment_list, self.complex_ngrams)
            adjusted_corrs = self.correspondence_probs(
                compacted_alignments, 
                ngram_size=ngram_size, 
                counts=counts, 
                prune=prune, 
                exclude_null=exclude_null, 
                compact_null=False,
                pad=False,
            )
            return adjusted_corrs
            
        if prune:
            corr_counts = prune_corrs(corr_counts, min_val=prune)

        if not counts:
            for seg1 in corr_counts:
                corr_counts[seg1] = normalize_dict(corr_counts[seg1])

        return corr_counts

    def phon_env_corr_probs(self, alignment_list, counts=False):
        # TODO currently works only with ngram_size=1 (I think this is fine?)
        corr_counts = defaultdict(lambda:defaultdict(lambda:0))
        for alignment in alignment_list:
            phon_env_align = alignment.add_phon_env()
            for seg_weight1, seg2 in phon_env_align:
                corr_counts[seg_weight1][seg2] += 1
        if not counts:
            for seg1 in corr_counts:
                corr_counts[seg1] = normalize_dict(corr_counts[seg1])
        
        return corr_counts
    
    def phoneme_pmi(self, conditional_counts, l1=None, l2=None):
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
            
        # Calculate PMI for all phoneme pairs
        pmi_dict = defaultdict(lambda:defaultdict(lambda:0))
        for seg1, seg2 in segment_pairs:
            seg1_ngram = Ngram(seg1)
            seg2_ngram = Ngram(seg2)
            # Skip full boundary gap alignments
            if self.pad_ch in seg1 and self.pad_ch in seg2: # (seg1, seg2) == (self.pad_ch, self.pad_ch)
                continue
            # Boundary gap alignments with segments
            elif self.pad_ch in seg1:
                assert self.pad_ch in seg2_ngram.string
                if re.search(rf'{self.pad_ch}{END_PAD_CH}', seg2_ngram.string):
                    #direction = 'END'
                    counts_seg2 = len([word2 for word1, word2 in self.same_meaning # TODO needs to account for sample, not just all of self.same_meaning
                                if tuple(word2.segments[-(seg2_ngram.size-1):]) == seg2[:-1]])
                else:
                    #direction = 'START'
                    counts_seg2 = len([word2 for word1, word2 in self.same_meaning # TODO needs to account for sample, not just all of self.same_meaning
                                if tuple(word2.segments[:seg2_ngram.size-1]) == seg2[1:]])
                p_seg2 = counts_seg2 / len(self.same_meaning)
                cond_prob = conditional_counts[seg1][seg2] / counts_seg2
                pmi_dict[seg1][seg2] = bayes_pmi(cond_prob, p_seg2)
            
            elif self.pad_ch in seg2:
                assert self.pad_ch in seg1_ngram.string
                if re.search(rf'{self.pad_ch}{END_PAD_CH}', seg1_ngram.string):
                    #direction = 'END'
                    counts_seg1 = len([word1 for word1, word2 in self.same_meaning # TODO needs to account for sample, not just all of self.same_meaning
                                if tuple(word1.segments[-(seg1_ngram.size-1):]) == seg1[:-1]])
                else:
                    #direction = 'START'
                    counts_seg1 = len([word1 for word1, word2 in self.same_meaning # TODO needs to account for sample, not just all of self.same_meaning
                                       if tuple(word1.segments[:seg1_ngram.size-1]) == seg1[1:]])
                p_seg1 = counts_seg1 / len(self.same_meaning)
                cond_prob = reverse_cond_counts[seg2][seg1] / counts_seg1
                pmi_dict[seg1][seg2] = bayes_pmi(cond_prob, p_seg1)
            
            # Standard alignment pairs
            else:
                p_ind1 = l1.ngram_probability(seg1_ngram)
                p_ind2 = l2.ngram_probability(seg2_ngram)
                p_ind = p_ind1 * p_ind2
                joint_prob = joint_prob_dist.get(seg1, {}).get(seg2, p_ind)
                pmi_dict[seg1_ngram.undo()][seg2_ngram.undo()] = pointwise_mutual_info(joint_prob, p_ind1, p_ind2)
            
        return pmi_dict

    def calc_phoneme_pmi(self, 
                         radius=1, # TODO make configurable
                         p_threshold=0.05,
                         p_step=0.02,
                         max_p=0.25,
                         samples=5, # TODO make configurable
                         sample_size=0.8, # TODO make configurable
                         cumulative=False,
                         log_iterations=True,
                         save=True):
        """
        Parameters
        ----------
        radius : int, optional
            Number of word positions forward and backward to check for initial correspondences. The default is 2.
        max_iterations : int, optional
            Maximum number of iterations. The default is 3.
        p_threshold : float, optional
            p-value threshold for words to qualify for PMI calculation in the next iteration. The default is 0.1.
        log_iterations : bool, optional
            Whether to log the results of each iteration. The default is False.
        save : bool, optional
            Whether to save the results to the Language class object's phoneme_pmi attribute. The default is True.
        Returns
        -------
        results : collections.defaultdict
            Nested dictionary of phoneme PMI values.
        """
        if self.logger:
            self.logger.info(f'Calculating phoneme PMI: {self.lang1.name}-{self.lang2.name}...')
            
        # Take a sample of same-meaning words, by default 80% of available same-meaning pairs
        sample_results = {}
        sample_size = round(len(self.same_meaning)*sample_size)
        # Take N samples of different-meaning words, perform PMI calibration, then average all of the estimates from the various samples
        iter_logs = defaultdict(lambda:[])
        max_iterations = max(round((max_p-p_threshold)/p_step), 2)
        sample_iterations = {}
        sample_dict = self.sample_wordlists(n_samples=samples, sample_size=sample_size, start_seed=self.seed, log_samples=log_iterations)
        for sample_n, sample in sample_dict.items():
            synonym_sample, diff_sample = sample
            
            # First step: calculate probability of phones co-occuring within within 
            # a set radius of positions within their respective words
            synonyms_radius1 = self.radial_counts(synonym_sample, radius, normalize=False)
            synonyms_radius2 = reverse_corr_dict(synonyms_radius1)
            for d in [synonyms_radius1, synonyms_radius2]:
                for seg1 in d:
                    d[seg1] = normalize_dict(d[seg1])
            pmi_step1 = [self.phoneme_pmi(conditional_counts=synonyms_radius1, l1=self.lang1, l2=self.lang2),
                        self.phoneme_pmi(conditional_counts=synonyms_radius2, l1=self.lang2, l2=self.lang1)]
            pmi_dict_l1l2, pmi_dict_l2l1 = pmi_step1
            
            # Average together the PMI values from each direction
            pmi_step1 = average_corrs(pmi_dict_l1l2, pmi_dict_l2l1)

            # At each following iteration N, re-align using the pmi_stepN as an 
            # additional penalty, and then recalculate PMI
            iteration = 0
            PMI_iterations = {iteration:pmi_step1}
            p_threshold_sample = p_threshold
            qualifying_words = default_dict({iteration:sort_wordlist(synonym_sample)}, l=[])
            disqualified_words = default_dict({iteration:diff_sample}, l=[])
            if cumulative:
                all_cognate_alignments = []
            #while (iteration < max_iterations) and (qualifying_words[iteration] != qualifying_words[iteration-1]):
            while (iteration < max_iterations) and (qualifying_words[iteration] not in [qualifying_words[i] for i in range(max(0,iteration-5),iteration)]):
            #while (iteration < max_iterations) and (nc_thresholds[iteration-1] not in [nc_thresholds[i] for i in range(max(0,iteration-2),iteration-1)]):
                iteration += 1

                # Align the qualifying words of the previous step using previous step's PMI
                cognate_alignments = self.align_wordlist(qualifying_words[iteration-1], added_penalty_dict=PMI_iterations[iteration-1], complex_ngrams=self.complex_ngrams)
                
                # Add these alignments into running pool of alignments
                if cumulative:
                    all_cognate_alignments.extend(cognate_alignments)
                    cognate_alignments = all_cognate_alignments
                
                # Calculate correspondence probabilities and PMI values from these alignments
                cognate_probs = self.correspondence_probs(cognate_alignments, exclude_null=True, counts=True)
                cognate_probs = default_dict({k[0]:{v[0]:cognate_probs[k][v] 
                                                    for v in cognate_probs[k]} 
                                            for k in cognate_probs}, l=defaultdict(lambda:0))
                PMI_iterations[iteration] = self.phoneme_pmi(cognate_probs)# , noncognate_probs)
                
                # Align all same-meaning word pairs
                aligned_synonym_sample = self.align_wordlist(synonym_sample, added_penalty_dict=PMI_iterations[iteration], complex_ngrams=self.complex_ngrams)
                # Align sample of different-meaning word pairs + non-cognates detected from previous iteration
                # disqualified_words[iteration-1] already contains both types
                noncognate_alignments = self.align_wordlist(disqualified_words[iteration-1], added_penalty_dict=PMI_iterations[iteration], complex_ngrams=self.complex_ngrams)

                # Score PMI for different meaning words and words disqualified in previous iteration
                noncognate_PMI = []
                for alignment in noncognate_alignments:
                    noncognate_PMI.append(mean([PMI_iterations[iteration][pair[0]][pair[1]] for pair in alignment.alignment]))
                nc_mean = mean(noncognate_PMI)
                nc_stdev = stdev(noncognate_PMI)
                
                # Score same-meaning alignments for overall PMI and calculate p-value
                # against different-meaning alignments
                qualifying, disqualified = [], []
                qualifying_alignments = []
                for i, item in enumerate(synonym_sample):
                    alignment = aligned_synonym_sample[i]
                    PMI_score = mean([PMI_iterations[iteration][pair[0]][pair[1]] for pair in alignment.alignment])
                    
                    # Proportion of non-cognate word pairs which would have a PMI score at least as low as this word pair
                    pnorm = 1 - norm.cdf(PMI_score, loc=nc_mean, scale=nc_stdev)
                    if pnorm < p_threshold_sample:
                        qualifying.append(item)
                        qualifying_alignments.append(alignment)
                    else:
                        disqualified.append(item)
                qualifying_words[iteration] = sort_wordlist(qualifying)
                if len(qualifying_words[iteration]) == 0:
                    self.logger.warning(f'All word pairs were disqualified in PMI iteration {iteration}')
                disqualified_words[iteration] = disqualified + diff_sample
                if p_threshold_sample+p_step <= max_p:
                    p_threshold_sample += p_step
                
                # Log results of this iteration
                if log_iterations:
                    iter_log = self._log_iteration(iteration, qualifying_words, disqualified_words)
                    iter_logs[sample_n].append(iter_log)
            
            # Log final set of qualifying/disqualified word pairs
            if log_iterations:
                iter_logs[sample_n].append((qualifying_words[iteration], sort_wordlist(disqualified)))
            
                # Log final alignments from which PMI was calculated
                self._log_alignments(qualifying_alignments, self.align_log['PMI'])
            
            # Return and save the final iteration's PMI results
            results = PMI_iterations[max(PMI_iterations.keys())]
            sample_results[sample_n] = results
            sample_iterations[sample_n] = len(PMI_iterations)-1
            # if self.logger:
            #     self.logger.debug(f'Sample {sample_n+1} converged after {iteration} iterations.')
            
            self.reset_seed()
        
        # Average together the PMI estimations from each sample
        if samples > 1:
            results = average_nested_dicts(list(sample_results.values()))
        else:
            results = sample_results[0]

        # Write the iteration log
        if log_iterations:
            self.logger.debug(f'{samples} sample(s) converged after {round(mean(sample_iterations.values()), 1)} iterations on average')
            log_file = os.path.join(self.pmi_log_dir, f'PMI_iterations.log')
            self._write_iter_log(iter_logs, log_file)
            
            # Write alignment log
            align_log_file = os.path.join(self.pmi_log_dir, 'PMI_alignments.log')
            self._write_alignments_log(self.align_log['PMI'], align_log_file)

        if save:
            self.lang1.phoneme_pmi[self.lang2] = results
            self.lang2.phoneme_pmi[self.lang1] = reverse_corr_dict(results)
            self.lang1.complex_ngrams[self.lang2] = self.complex_ngrams
            reversed_complex_ngrams = {val:set(key for key in self.complex_ngrams if val in self.complex_ngrams[key]) 
                                       for key in self.complex_ngrams for val in self.complex_ngrams[key]}
            self.lang2.complex_ngrams[self.lang1] = default_dict(reversed_complex_ngrams, l=[])
            # self.lang1.phoneme_pmi[self.lang2]['thresholds'] = noncognate_PMI
            
        self.pmi_dict = results
        self.log_phoneme_pmi()
        
        return results
    
    def phoneme_surprisal(self, 
                          correspondence_counts, 
                          phon_env_corr_counts=None, 
                          ngram_size=1, 
                          weights=None, 
                          #attested_only=True,
                          #alpha=0.2 # TODO conduct more formal experiment to select default alpha, or find way to adjust automatically; so far alpha=0.2 is best (at least on Romance)
                          ):
        # Interpolation smoothing
        if weights is None:
            # Each ngram estimate will be weighted proportional to its size
            # Weight the estimate from a 2gram twice as much as 1gram, etc.
            weights = [i+1 for i in range(ngram_size)]
        if phon_env_corr_counts is not None:
            phon_env = True
        else:
            phon_env = False
        interpolation = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0)))

        for i in range(ngram_size,0,-1):
            for ngram1 in correspondence_counts:
                for ngram2 in correspondence_counts[ngram1]:
                    # Exclude correspondences with a fully null ngram, e.g. ('k', 'a') with ('-', '-')
                    # Only needs to be done with ngram_size > 1
                    if (self.gap_ch,)*(max(ngram_size, 2)) not in [ngram1, ngram2]:
                        # forward
                        # interpolation[i][ngram1[-i:]][ngram2[-1]] += correspondence_counts[ngram1][ngram2]
                        
                        # backward
                        interpolation[i][Ngram(ngram1).ngram][Ngram(ngram2).ngram] += correspondence_counts[ngram1][ngram2]
        
        # Add in phonological environment correspondences, e.g. ('l', '#S<') (word-initial 'l') with 'ʎ' 
        if phon_env:
            for ngram1 in phon_env_corr_counts:
                if ngram1 == self.gap_ch:
                    continue
                phonEnv = ngram1[-1]
                phonEnv_contexts = phon_env_ngrams(phonEnv, exclude={'|S|'})
                for context in phonEnv_contexts:
                    ngram1_context = ngram1[:-1] + (context,)
                    if len(ngram1_context) > 2:
                        breakpoint()
                    for ngram2 in phon_env_corr_counts[ngram1]: # TODO where do these ngram2 come from?
                        #backward
                        interpolation['phon_env'][Ngram(ngram1_context).ngram][Ngram(ngram2).ngram] += phon_env_corr_counts[ngram1][ngram2]
        
        # Get lists of possible ngrams in lang1 and lang2
        # lang2 ngram size fixed at 1, only trying to predict single phone; also not trying to predict its phon_env 
        all_ngrams_lang2 = self.get_possible_ngrams(self.lang2, ngram_size=1, phon_env=False)
        all_ngrams_lang2.update(reverse_corr_dict(interpolation[i]).keys()) # add any complex ngram corrs
        all_ngrams_lang1 = self.get_possible_ngrams(
            lang=self.lang1,
            # phon_env ngrams fixed at size 1
            ngram_size=ngram_size if phon_env else 1,
            phon_env=phon_env
        )
        # Add complex ngram corrs to all_ngrams_lang1
        if phon_env:
            for complex_ngram in self.complex_ngrams:
                if complex_ngram.size > 1:
                    if self.pad_ch in complex_ngram.ngram[0]: # complex ngram alignment with start boundary
                        ngram_envs = [ngram for ngram in all_ngrams_lang1 if ngram[:-1] == complex_ngram.ngram[1:] and re.search(r'#\|S', ngram[-1])]
                    elif self.pad_ch in complex_ngram.ngram[-1]: # complex ngram alignment with end boundary
                        ngram_envs = [ngram for ngram in all_ngrams_lang1 if ngram[:-1] == complex_ngram.ngram[:-1] and re.search(r'S\|#', ngram[-1])]
                    else: # other complex ngram alignments
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
                        
                    complex_envs = [Ngram((complex_ngram.ngram, ngram_env[-1])).ngram for i, ngram_env in enumerate(ngram_envs)]
                    all_ngrams_lang1.update(complex_envs)
        else:
            all_ngrams_lang1.update(interpolation[i].keys())

        smoothed_surprisal = defaultdict(lambda:defaultdict(lambda:self.lang2.phoneme_entropy*ngram_size))
        for ngram1 in all_ngrams_lang1:
            if self.gap_ch in ngram1:
                continue

            if phon_env:
                ngram1_phon_env = ngram1[:]
                ngram1 = Ngram(ngram1[:-1]).ngram
                if sum(interpolation['phon_env'][ngram1_phon_env].values()) == 0:
                    continue
            if sum(interpolation[i][ngram1].values()) == 0:
                continue

            undone_ngram1 = Ngram(ngram1).undo() # need to convert to dictionary form, whereby unigrams are strings and larger ngrams are tuples
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
                             for i in range(ngram_size,0,-1)]
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
                    phonEnv = ngram1_phon_env[-1]
                    phonEnv_contexts = phon_env_ngrams(phonEnv, exclude={'|S|'})
                    for context in phonEnv_contexts:
                        ngram1_context = ngram1_phon_env[:-1] + (context,)
                        estimates.append(interpolation['phon_env'][ngram1_context].get(ngram2, 0) / sum(interpolation['phon_env'][ngram1_context].values()))
                        # estimates.append(lidstone_smoothing(x=interpolation['phon_env'][(ngram1_context,)].get(ngram2, 0), 
                        #                                     N=sum(interpolation['phon_env'][(ngram1_context,)].values()), 
                        #                                     d = len(self.lang2.phonemes) + 1,
                        #                                     alpha=alpha)
                        #                                     )
                        # Weight each contextual estimate based on the size of the context
                        ngram_weights.append(get_phonEnv_weight(context))

                weight_sum = sum(ngram_weights)
                ngram_weights = [i/weight_sum for i in ngram_weights]
                assert(len(ngram_weights) == len(estimates))
                smoothed = sum([estimate*weight for estimate, weight in zip(estimates, ngram_weights)])
                undone_ngram2 = Ngram(ngram2).undo() # need to convert to dictionary form, whereby unigrams are strings and larger ngrams are tuples
                if phon_env:
                    smoothed_surprisal[ngram1_phon_env][undone_ngram2] = surprisal(smoothed)
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
            #smoothed_oov = max(surprisal(sum([estimate*weight for estimate, weight in zip(oov_estimates, ngram_weights)])), self.lang2.phoneme_entropy)
            smoothed_oov = self.lang2.phoneme_entropy
            
            if phon_env:
                smoothed_surprisal[ngram1_phon_env] = default_dict(smoothed_surprisal[ngram1_phon_env], l=smoothed_oov)
            else:
                smoothed_surprisal[undone_ngram1] = default_dict(smoothed_surprisal[undone_ngram1], l=smoothed_oov)
        
            # Prune saved surprisal values which exceed the phoneme entropy of lang2
            if phon_env:
                key = ngram1_phon_env
            else:
                key = undone_ngram1
            to_prune = [ngram2 for ngram2 in smoothed_surprisal[key] if smoothed_surprisal[key][ngram2] > self.lang2.phoneme_entropy]
            for ngram_to_prune in to_prune:
                del smoothed_surprisal[key][ngram_to_prune]            

        return smoothed_surprisal
    
    def calc_phoneme_surprisal(self, radius=1, 
                               max_iterations=10, 
                               p_threshold=0.1,
                               ngram_size=2,
                               gold=False, # TODO add same with PMI?
                               log_iterations=True,
                               samples=5,
                               sample_size=0.8, # TODO make configurable
                               cumulative=False,
                               save=True):
        # METHOD
        # 1) Calculate phoneme PMI
        # 2) Use phoneme PMI to align 
        # 3) Iterate
        # Calculate phoneme PMI if not already done, for alignment purposes

        if len(self.pmi_dict) == 0:
            self.pmi_dict = self.calc_phoneme_pmi(radius=radius, 
                                                  max_iterations=max_iterations, 
                                                  p_threshold=p_threshold)
        
        if self.logger:
            self.logger.info(f'Calculating phoneme surprisal: {self.lang1.name}-{self.lang2.name}...')

        if not gold:
            # Take N samples of same and different-meaning words, perform surprisal calibration, then average all of the estimates from the various samples
            sample_size = round(len(self.same_meaning)*sample_size)
            sample_results = {}
            iter_logs = defaultdict(lambda:[])
            sample_iterations = {}
            sample_dict = self.sample_wordlists(n_samples=samples, sample_size=sample_size, start_seed=self.seed, log_samples=log_iterations)
            for sample_n, sample in sample_dict.items():
                same_sample, diff_sample = sample

                # Align same-meaning and different meaning word pairs using PMI values: 
                # the alignments will remain the same throughout iteration
                same_meaning_alignments = self.align_wordlist(same_sample, added_penalty_dict=self.pmi_dict, complex_ngrams=self.complex_ngrams)
                diff_meaning_alignments = self.align_wordlist(diff_sample, added_penalty_dict=self.pmi_dict, complex_ngrams=self.complex_ngrams)

                # At each iteration, re-calculate surprisal for qualifying and disqualified pairs
                # Then test each same-meaning word pair to see if if it meets the qualifying threshold
                iteration = 0
                surprisal_iterations = {}
                qualifying_words = default_dict({iteration:list(range(len(same_meaning_alignments)))}, l=[])
                disqualified_words = defaultdict(lambda:[])
                align_log = defaultdict(lambda:set())
                if cumulative:
                    all_cognate_alignments = []
                while (iteration < max_iterations) and (qualifying_words[iteration] != qualifying_words[iteration-1]):
                    iteration += 1
                    
                    # Calculate surprisal from the qualifying alignments of the previous iteration
                    cognate_alignments = [same_meaning_alignments[i] for i in qualifying_words[iteration-1]]
                    if cumulative:
                        all_cognate_alignments.extend(cognate_alignments)
                        cognate_alignments = all_cognate_alignments
                    surprisal_iterations[iteration] = self.phoneme_surprisal(self.correspondence_probs(cognate_alignments,
                                                                                                       counts=True,
                                                                                                       exclude_null=False,
                                                                                                       compact_null=False,
                                                                                                       ngram_size=ngram_size), 
                                                                             ngram_size=ngram_size)

                    # Retrieve the alignments of different-meaning and disqualified word pairs
                    # and calculate adaptation surprisal for them using new surprisal values
                    noncognate_alignments = diff_meaning_alignments + [same_meaning_alignments[i]
                                                                    for i in disqualified_words[iteration-1]]
                    noncognate_surprisal = [adaptation_surprisal(alignment, 
                                                                surprisal_dict=surprisal_iterations[iteration], 
                                                                normalize=True,
                                                                ngram_size=ngram_size,
                                                                pad_ch=self.pad_ch,
                                                                ) 
                                            for alignment in noncognate_alignments]
                    mean_nc_score = mean(noncognate_surprisal)
                    nc_score_stdev = stdev(noncognate_surprisal)
                    
                    # Normalize different-meaning pair surprisal scores by self-surprisal of word2
                    # for i in range(len(noncognate_alignments)):
                    #     alignment = noncognate_alignments[i]
                    #     word2 = ''.join([pair[1] for pair in alignment if pair[1] != self.gap_ch])
                    #     self_surprisal = self.lang2.self_surprisal(word2, segmented=False, normalize=False)
                    #     noncognate_surprisal[i] /= self_surprisal

                    # Score same-meaning alignments for surprisal and calculate p-value
                    # against different-meaning alignments
                    qualifying, disqualified = [], []
                    qualifying_alignments = []
                    for i, item in enumerate(same_sample):
                        alignment = same_meaning_alignments[i]
                        surprisal_score = adaptation_surprisal(alignment, 
                                                               surprisal_dict=surprisal_iterations[iteration], 
                                                               normalize=True,
                                                               ngram_size=ngram_size,
                                                               pad_ch=self.pad_ch,
                                                               )
                        
                        # Proportion of non-cognate word pairs which would have a surprisal score at least as low as this word pair
                        pnorm = norm.cdf(surprisal_score, loc=mean_nc_score, scale=nc_score_stdev)
                        if pnorm < p_threshold:
                            qualifying.append(i)
                            qualifying_alignments.append(alignment)
                        else:
                            disqualified.append(i)
                            
                    qualifying_words[iteration] = qualifying
                    if len(qualifying) == 0:
                        self.logger.warning(f'All word pairs were disqualified in surprisal iteration {iteration}')
                    disqualified_words[iteration] = disqualified
                        
                    # Log results of this iteration
                    if log_iterations:
                        iter_log = self._log_iteration(iteration, qualifying_words, disqualified_words, method='surprisal', same_meaning_alignments=same_meaning_alignments)
                        iter_logs[sample_n].append(iter_log)

                        # Log final alignments from which surprisal was calculated
                        self._log_alignments(qualifying_alignments, self.align_log['surprisal'])

                # Save final set of qualifying/disqualified word pairs
                if log_iterations:
                    iter_logs[sample_n].append(([same_sample[i] for i in qualifying], 
                                                [same_sample[i] for i in disqualified]))
                    
                cognate_alignments = [same_meaning_alignments[i] for i in qualifying_words[iteration]]
                sample_results[sample_n] = surprisal_iterations[iteration]
                sample_iterations[sample_n] = iteration
                self.reset_seed()
            
            # Average together the surprisal estimations from each sample
            if samples > 1:
                p1_all = set(p for sample_n in sample_results for p in sample_results[sample_n])
                p2_all = set(p2 for sample_n in sample_results for p1 in sample_results[sample_n] for p2 in sample_results[sample_n][p1])
                surprisal_results = defaultdict(lambda:defaultdict(lambda:0))
                for p1 in p1_all:
                    p1_dict = defaultdict(lambda:[])
                    for p2 in p2_all:
                        for sample_n in sample_results:
                            p1_dict[p2].append(sample_results[sample_n][p1][p2])
                        surprisal_results[p1][p2] = mean(p1_dict[p2])
                    # oov_values = []
                    # for sample_n in sample_results:
                    #     # Use non-IPA character <?> to retrieve OOV value from surprisal dict
                    #     oov_values.append(sample_results[sample_n][p1]['?'])
                    # surprisal_results[p1] = default_dict(surprisal_results[p1], l=mean(oov_values))
                    surprisal_results[p1] = default_dict(surprisal_results[p1], l=self.lang2.phoneme_entropy)
                    
            else:
                surprisal_results = sample_results[0]

        else: # gold : assumes wordlist is already coded by cognate; skip iteration and calculate surprisal directly 
            # Align same-meaning and different meaning word pairs using PMI values: 
            # the alignments will remain the same throughout iteration
            same_meaning_alignments = self.align_wordlist(self.same_meaning, added_penalty_dict=self.pmi_dict)
            
            # TODO need to confirm that when the gloss/concept is checked, it considers the possible cognate class ID, e.g. rain_1
            cognate_alignments = same_meaning_alignments
            surprisal_results = self.phoneme_surprisal(self.correspondence_probs(same_meaning_alignments,
                                                                                 counts=True,
                                                                                 exclude_null=False, 
                                                                                 compact_null=False, 
                                                                                 ngram_size=ngram_size), 
                                                                                 ngram_size=ngram_size)
        
        # Write the iteration log
        surprisal_ngram_log_dir = os.path.join(self.surprisal_log_dir, f'{ngram_size}-gram')
        if log_iterations and not gold:
            self.logger.debug(f'{samples} sample(s) converged after {round(mean(sample_iterations.values()), 1)} iterations on average')
            log_file = os.path.join(surprisal_ngram_log_dir, f'surprisal_iterations.log')
            self._write_iter_log(iter_logs, log_file)
                
        # Add phonological environment weights after final iteration
        phon_env_surprisal = self.phoneme_surprisal(
            self.correspondence_probs(cognate_alignments, counts=True, exclude_null=False, compact_null=False), 
            phon_env_corr_counts=self.phon_env_corr_probs(cognate_alignments, counts=True),
            ngram_size=ngram_size
            )
        
        # Return and save the final iteration's surprisal results
        if save:
            self.lang1.phoneme_surprisal[(self.lang2, ngram_size)] = surprisal_results
            self.lang1.phon_env_surprisal[self.lang2] = phon_env_surprisal
        self.surprisal_dict[ngram_size] = surprisal_results
        self.phon_env_surprisal_dict = phon_env_surprisal

        # Write logs
        # Alignments log
        align_log_file = os.path.join(surprisal_ngram_log_dir, 'surprisal_alignments.log')
        self._write_alignments_log(self.align_log['surprisal'], align_log_file)
        
        # Phone correlation report
        phon_corr_report = os.path.join(surprisal_ngram_log_dir, 'phon_corr.tsv')
        self._write_phon_corr_report(surprisal_results, phon_corr_report, type='surprisal')
        
        # Surprisal logs
        self.log_phoneme_surprisal(phon_env=False, ngram_size=ngram_size)
        self.log_phoneme_surprisal(phon_env=True)
        
        return surprisal_results, phon_env_surprisal
    
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
            key = (self.lang2, eval_func, sample_size, seed)
            self.lang1.noncognate_thresholds[key] = noncognate_scores
        
        return noncognate_scores

    def log_phoneme_pmi(self, outfile=None, threshold=0.0001, sep='\t'):
        # Save calculated PMI values to file
        if outfile is None:
            outfile = os.path.join(self.pmi_log_dir, 'phonPMI.tsv')

        # Save all segment pairs with non-zero PMI values to file
        # Skip extremely small decimals that are close to zero
        lines = []
        for seg1 in self.pmi_dict:
            for seg2 in self.pmi_dict[seg1]:
                pmi_val = self.pmi_dict[seg1][seg2]
                if abs(pmi_val) > threshold:
                    line = [ngram2str(seg1), ngram2str(seg2), str(pmi_val)]
                    lines.append(line)
        # Sort PMI in descending order
        lines = sorted(lines, key=lambda x:x[-1], reverse=True)
        lines = '\n'.join([sep.join(line) for line in lines])
        
        with open(outfile, 'w') as f:
            header = sep.join(['Phone1', 'Phone2', 'PMI'])
            f.write(f'{header}\n{lines}')
    
    def log_phoneme_surprisal(self, outfile=None, sep='\t', phon_env=True, ngram_size=1):
        if outfile is None:
            if phon_env:
                outfile = os.path.join(self.surprisal_log_dir, 'phonEnv', 'phonEnvSurprisal.tsv')
            else:
                surprisal_ngram_log_dir = os.path.join(self.surprisal_log_dir, f'{ngram_size}-gram')
                outfile = os.path.join(surprisal_ngram_log_dir, 'phonSurprisal.tsv')
        outdir = os.path.abspath(os.path.dirname(outfile))
        os.makedirs(outdir, exist_ok=True)
        
        if phon_env:
            surprisal_dict = self.phon_env_surprisal_dict
        else:
            surprisal_dict = self.surprisal_dict[ngram_size]
            
        lines = []
        for seg1 in surprisal_dict:
            # Determine the smoothed value for unseen ("out of vocabulary") correspondences
            # Check using a non-IPA character
            non_IPA = '<NON-IPA>'
            oov_smoothed = surprisal_dict[seg1][non_IPA]
            
            # Then remove this character from the surprisal dictionary
            del surprisal_dict[seg1][non_IPA]
            
            # Save values which are not equal to (less than) the OOV smoothed value
            for seg2 in surprisal_dict[seg1]:
                surprisal_val = surprisal_dict[seg1][seg2]
                if surprisal_val < oov_smoothed:
                    if ngram_size > 1:
                        breakpoint() # TODO need to decide format for how to save/load larger ngrams from logs; previously they were separated by whitespace
                    lines.append([ngram2str(seg1, phon_env=phon_env), 
                                  ngram2str(seg2, phon_env=False), # phon_env only on seg1  
                                  str(surprisal_val), 
                                  str(oov_smoothed)])
        
        # Sort surprisal in ascending order
        lines = sorted(lines, key=lambda x:x[2], reverse=False)
        lines = '\n'.join([sep.join(line) for line in lines])
        
        with open(outfile, 'w') as f:
            header = sep.join(['Phone1', 'Phone2', 'Surprisal', 'OOV_Smoothed'])
            f.write(f'{header}\n{lines}')

    def _log_sample(self, sample, sample_n, seed=None):
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
    
    def _write_sample_log(self, sample_logs, log_file):
        log_dir = os.path.abspath(os.path.dirname(log_file))
        os.makedirs(log_dir, exist_ok=True)
        content = '\n\n'.join([sample_logs[sample_n] for sample_n in range(len(sample_logs))])
        with open(log_file, 'w') as f:
            f.write(content)

    def _log_iteration(self, iteration, qualifying_words, disqualified_words, method=None, same_meaning_alignments=None):
        iter_log = []
        if method == 'surprisal':
            assert same_meaning_alignments is not None
            
            def get_word_pairs(indices, lst):
                aligns = [lst[i] for i in indices]
                pairs = [(align.word1, align.word2) for align in aligns]
                return pairs
            
            qualifying = get_word_pairs(qualifying_words[iteration], same_meaning_alignments)
            prev_qualifying = get_word_pairs(qualifying_words[iteration-1], same_meaning_alignments)
            disqualified = get_word_pairs(disqualified_words[iteration], same_meaning_alignments)
            prev_disqualified = get_word_pairs(disqualified_words[iteration-1], same_meaning_alignments)
        else:
            qualifying = qualifying_words[iteration]
            prev_qualifying = qualifying_words[iteration-1]
            disqualified = disqualified_words[iteration]
            prev_disqualified = disqualified_words[iteration-1]
        iter_log.append(f'Iteration {iteration}')
        iter_log.append(f'\tQualified: {len(qualifying)}')
        iter_log.append(f'\tDisqualified: {len(disqualified)}')
        added = set(qualifying) - set(prev_qualifying)
        iter_log.append(f'\tAdded: {len(added)}')
        for word1, word2 in added:
            iter_log.append(f'\t\t{word1.orthography} /{word1.ipa}/ - {word2.orthography} /{word2.ipa}/')
        removed = set(disqualified) - set(prev_disqualified)
        iter_log.append(f'\tRemoved: {len(removed)}')
        for word1, word2 in removed:
            iter_log.append(f'\t\t{word1.orthography} /{word1.ipa}/ - {word2.orthography} /{word2.ipa}/')
        
        iter_log = '\n'.join(iter_log)
        
        return iter_log
    
    def _write_iter_log(self, iter_logs, log_file):
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
                for word1, word2 in final_qualifying:
                    f.write(f'\t\t{word1.orthography} /{word1.ipa}/ - {word2.orthography} /{word2.ipa}/\n')
                f.write('\nFinal disqualified:\n')
                for word1, word2 in final_disqualified:
                    f.write(f'\t\t{word1.orthography} /{word1.ipa}/ - {word2.orthography} /{word2.ipa}/\n')
                f.write('\n\n-------------------\n\n')
    
    def _log_alignments(self, alignments, align_log):
        for alignment in alignments:
            key = f'/{alignment.word1.ipa}/ - /{alignment.word2.ipa}/'
            align_str = visual_align(alignment.alignment, gap_ch=alignment.gap_ch)
            align_log[key][align_str] += 1
    
    def _write_alignments_log(self, alignment_log, log_file):
        sorted_alignment_keys = sorted(alignment_log.keys())
        with open(log_file, 'w') as f:
            for key in sorted_alignment_keys:
                f.write(f'{key}\n')
                sorted_alignments = dict_tuplelist(alignment_log[key])
                sorted_alignments.sort(key=lambda x:(x[-1], x[0]), reverse=True)
                for alignment, count in sorted_alignments:
                    freq = f'{count}/{sum(alignment_log[key].values())}'#count/sum(alignment_log[key].values())
                    #f.write(f'[{round(freq, 2)}] {alignment}\n')
                    f.write(f'[{freq}] {alignment}\n')
                f.write('\n-------------------\n\n')
    
    def _write_phon_corr_report(self, corr, outfile, type, n=5):
        lines = []
        l1_phons = sorted([p for p in corr if p[0] != self.gap_ch], key=lambda x:Ngram(x).string)
        for p1 in l1_phons:
            p2_candidates = corr[p1]
            if len(p2_candidates) > 0:
                p2_candidates = dict_tuplelist(p2_candidates, reverse=True)[-n:]
                # Sort by corr value, then by phone string if values are equal
                p2_candidates.sort(key=lambda x:(x[-1], Ngram(x[0]).string))
                for p2, score in p2_candidates:
                    if type == 'surprisal':
                        if score < self.lang2.phoneme_entropy:
                            prob = surprisal_to_prob(score) # turn surprisal value into probability
                            p1 = Ngram(p1).string
                            p2 = Ngram(p2).string
                            line = '\t'.join([p1, p2, str(round(prob, 3))])
                            lines.append(line)
                    else:
                        raise NotImplementedError # not implemented for PMI
        header = '\t'.join([self.lang1.name, self.lang2.name, 'probability'])
        lines = '\n'.join(lines)
        content = '\n'.join([header, lines])
        with open(outfile, 'w') as f:
            f.write(f'{content}')

def phon_env_ngrams(phonEnv, exclude=set()):
    """Returns set of phonological environment strings of equal and lower order, 
    e.g. ">|S|#" -> ">|S", "S|#", ">|S|#"

    Args:
        phonEnv (str): Phonological environment string, e.g. ">S#"

    Returns:
        set: possible equal and lower order phonological environment strings
    """
    if re.search(r'.\|S\|.+', phonEnv):
        prefix, base, suffix = phonEnv.split('|')
        prefix = prefix.split('_')
        prefixes = set()
        for i in range(1, len(prefix)+1):
            for x in combinations(prefix, i):
                prefixes.add('_'.join(x))
        prefixes.add('')
        suffix = suffix.split('_')
        suffixes = set()
        for i in range(1, len(suffix)+1):
            for x in combinations(suffix, i):
                suffixes.add('_'.join(x))
        suffixes.add('')
        ngrams = set()
        for prefix in prefixes:
            for suffix in suffixes:
                ngrams.add(f'{prefix}|S|{suffix}')
    else:
        assert phonEnv == '|T|'
        ngrams = [phonEnv]
    
    if len(exclude) > 0:
        return [ngram for ngram in ngrams if ngram not in exclude]
    return ngrams

class NullCompacter:
    def __init__(self, corr_counts, alignments, lang1, lang2, gap_ch, pad_ch, ngram_size=1):
        self.gap_ch = gap_ch
        self.pad_ch = pad_ch
        self.gap_ngram = Ngram(self.gap_ch).ngram
        self.pad_ngrams = (Ngram(f'{self.pad_ch}{END_PAD_CH}').ngram, Ngram(f'{START_PAD_CH}{self.pad_ch}').ngram)
        self.ngram_size = ngram_size
        self.corr_counts = corr_counts
        self.alignments = alignments
        self.seqs1, self.seqs2 = self.extract_seqs()
        self.set_seq_lens()
        self.lang1 = lang1
        self.lang2 = lang2
        self.compacted_corr_counts = defaultdict(lambda:defaultdict(lambda:0))
        self.reversed_corr_counts = None
        self.reversed_compacted_corr_counts = None
        self.valid_corrs = defaultdict(lambda:[])
        
    def extract_seqs(self):
        seqs1 = [alignment.seq1 for alignment in self.alignments]
        seqs2 = [alignment.seq2 for alignment in self.alignments]
        return seqs1, seqs2

    def set_seq_lens(self):
        self.seqs1_lens = [len(seq) for seq in self.seqs1]
        self.seqs2_lens = [len(seq) for seq in self.seqs2]
        self.seqs1_len_total = sum(self.seqs1_lens)
        self.seqs2_len_total = sum(self.seqs2_lens)
    
    def reverse_corr_counts(self):
        if self.reversed_corr_counts is None:
            self.reversed_corr_counts = reverse_corr_dict(self.corr_counts)
        if self.reversed_compacted_corr_counts is None:
            self.reversed_compacted_corr_counts = reverse_corr_dict(self.compacted_corr_counts)  
        
    def compact_next_ngram(self, alignment, i, ngram, ngram_i, gap_index):
        opposite_index = gap_index-1
        next_ngram = alignment[i+1-(self.ngram_size-1):i+2]
        if self.gap_ch not in (next_ngram[ngram_i][gap_index], next_ngram[ngram_i][opposite_index]):
            gap_seg = next_ngram[ngram_i][gap_index]
            larger_ngram = flatten_ngram((ngram[ngram_i][opposite_index], next_ngram[ngram_i][opposite_index]))
            if gap_index == 0:
                self.compacted_corr_counts[gap_seg][larger_ngram] += 1
            else: # 1 
                self.compacted_corr_counts[larger_ngram][gap_seg] += 1     
                        
    def compact_prev_ngram(self, alignment, i, ngram, ngram_i, gap_index):
        opposite_index = gap_index-1
        prev_ngram = alignment[i-1-(self.ngram_size-1):i]
        if self.gap_ch not in (prev_ngram[ngram_i][gap_index], prev_ngram[ngram_i][opposite_index]):
            gap_seg = prev_ngram[ngram_i][gap_index]
            larger_ngram = flatten_ngram((prev_ngram[ngram_i][opposite_index], ngram[ngram_i][opposite_index]))
            if gap_index == 0:
                self.compacted_corr_counts[gap_seg][larger_ngram] += 1
            else: # 1 
                self.compacted_corr_counts[larger_ngram][gap_seg] += 1

    def compact_null(self, alignment, i, ngram):
        if any([self.gap_ch in ngram_i for ngram_i in ngram]):
            if self.ngram_size > 1:
                raise NotImplementedError # unsure if this will work for ngram size > 1
            else:
                ngram_i = 0
            gap_index = ngram[ngram_i].index(self.gap_ch)
            if i > 0 and i < len(alignment)-1: # medial (has preceding and following)
                self.compact_next_ngram(alignment, i, ngram, ngram_i, gap_index)
                self.compact_prev_ngram(alignment, i, ngram, ngram_i, gap_index)
            elif i > 0: # final (preceding only)
                self.compact_prev_ngram(alignment, i, ngram, ngram_i, gap_index)
            elif len(alignment) > 1: # initial with following ngram
                self.compact_next_ngram(alignment, i, ngram, ngram_i, gap_index)
        
    def eval_null_corr(self, larger_ngrams, gap_segs, direction):
        # Determine direction
        self.reverse_corr_counts()
        if direction == 'FORWARD':
            seqs, opp_seqs = self.seqs2, self.seqs1
            seq_lens, opp_seq_lens = self.seqs2_lens, self.seqs1_lens
            total_seq_len, opp_total_seq_len = self.seqs2_len_total, self.seqs1_len_total
            corr_counts = self.corr_counts
            compacted_corr_counts = self.compacted_corr_counts
            reversed_compacted_corr_counts = self.reversed_compacted_corr_counts
        elif direction == 'BACKWARD':
            seqs, opp_seqs = self.seqs1, self.seqs2
            seq_lens, opp_seq_lens = self.seqs1_lens, self.seqs2_lens
            total_seq_len, opp_total_seq_len = self.seqs1_len_total, self.seqs2_len_total
            corr_counts = self.reversed_corr_counts
            compacted_corr_counts = self.reversed_compacted_corr_counts
            reversed_compacted_corr_counts = self.compacted_corr_counts
        else:
            raise ValueError
        
        def gap_pmi(p_seg_given_gap, p_seg):
            # via Bayes Theorem : pmi = log( p(x,y) / ( p(x) * p(y) ) ) = log( p(x|y) / p(x) )
            # we can't calculate the probability of a gap, but we can calculate the conditional probability of a segment given a gap
            # this is just a helper function to clarify what the inputs should be by renaming input args with ref to gap/seg
            return bayes_pmi(p_seg_given_gap, p_seg)
        
        for gap_seg in gap_segs:
            gap_seg = Ngram(gap_seg)
            for larger_ngram in larger_ngrams:
                larger_ngram = Ngram(larger_ngram)
                comp_count = compacted_corr_counts[larger_ngram.raw][gap_seg.raw]
                
                if self.pad_ch in gap_seg.string:
                    # Special handling for boundary (start/end seq) gaps, marked with pad_ch, see details below
                    # Logic: PAD_CH corresponds to either the start or end of an alignment/segmented sequence
                    # Every sequence has exactly one of each, therefore the probability (irrespective whether beginning/end) is just the number of sequences divided by the total number of unigrams
                    gap_seg_prob = len(seqs) / total_seq_len
                # Calculation below gives a more precise probability specific to this set of alignments,  # TODO add this method with ngram_count_wordlist() in calculation of PMI elsewhere
                # which directly reflects shared coverage between l1 and l2
                # Else, using lang.phonemes will consider all words in the vocabulary
                elif gap_seg.size > 1: # gap seg is also a n>1-ngram
                    gap_seg_count = ngram_count_wordlist(gap_seg.ngram, seqs)
                    gap_seg_prob = gap_seg_count / sum([count_subsequences(l, gap_seg.size) for l in seq_lens])
                else:
                    gap_seg_prob = ngram_count_wordlist(gap_seg.ngram, seqs) / total_seq_len
                
                # Estimate PMI of larger/complex ngram correlation 
                if self.pad_ch in larger_ngram.string:
                    # Logic: PAD_CH corresponds to either the start or end of an alignment/segmented sequence
                    # Each word by definition has exactly 1 of these, but we are interested in the ones aligned with a particular sequence
                    # Probability of boundary gaps with complex ngrams can be calculated by counting the number of 
                    # words starting/ending with the sequence aligned to the opposite-language boundary gap
                    # and then via Bayes theorem, given this probability and the conditional probability, we can get the PMI
                    if re.search(rf'{self.pad_ch}{END_PAD_CH}', gap_seg.string): # ENDING BOUNDARY
                        larger_ngram_count = len([seq for seq in opp_seqs if tuple(seq[-gap_seg.size:]) == larger_ngram.ngram[:-1]])
                        larger_ngram_prob = larger_ngram_count / len(opp_seqs)
                        
                    else: # STARTING BOUNDARY
                        larger_ngram_count = len([seq for seq in opp_seqs if tuple(seq[:gap_seg.size]) == larger_ngram.ngram[1:]])
                        larger_ngram_prob = larger_ngram_count / len(opp_seqs)
                    
                    cond_count_complex = reversed_compacted_corr_counts[gap_seg.raw][larger_ngram.raw]
                    cond_prob_complex = cond_count_complex / larger_ngram_count
                    pmi_complex = bayes_pmi(cond_prob_complex, larger_ngram_prob)
                        
                else:
                    larger_ngram_count = ngram_count_wordlist(larger_ngram.ngram, opp_seqs)
                    larger_ngram_prob = larger_ngram_count / sum([count_subsequences(l, larger_ngram.size) for l in opp_seq_lens])
                    cond_prob_complex = comp_count/larger_ngram_count
                    joint_prob_complex = cond_prob_complex * larger_ngram_prob
                    pmi_complex = pointwise_mutual_info(joint_prob_complex, larger_ngram_prob, gap_seg_prob)
                
                # Estimate PMI of simpler ngram (unigram) correlation
                pmi_basic = 0
                if self.pad_ch in larger_ngram.string:
                    for unigram in larger_ngram.unigrams():
                        if unigram.ngram not in self.pad_ngrams:
                            cond_prob_basic = corr_counts[unigram.ngram][self.gap_ngram] / sum(corr_counts[unigram.ngram].values())
                            if cond_prob_basic > 0:
                                pmi_basic += max(0, bayes_pmi(cond_prob_basic, gap_seg_prob))
                    
                else:
                    for seg in larger_ngram.ngram:
                        seg = Ngram(seg)
                        seg_prob = ngram_count_wordlist(seg.ngram, opp_seqs) / opp_total_seq_len
                        count_seg_given_gap = corr_counts[self.gap_ngram].get(seg.ngram, 0)
                        if count_seg_given_gap > 0:
                            p_seg_given_gap = count_seg_given_gap / sum(corr_counts[self.gap_ngram].values())
                            if p_seg_given_gap > 0:
                                pmi_basic += max(0, gap_pmi(p_seg_given_gap, seg_prob))
                        count_gap_seg_given_seg = corr_counts[seg.ngram].get(gap_seg.ngram, 0)
                        if count_gap_seg_given_seg > 0:
                            p_gap_seg_given_seg = count_gap_seg_given_seg / sum(corr_counts[seg.ngram].values())
                            if p_gap_seg_given_seg > 0:
                                pmi_basic += max(0, bayes_pmi(p_gap_seg_given_seg, gap_seg_prob))
                                
                # Consider the compacted null alignment to be valid if its PMI is greater than that of the simpler ngram correlations
                if pmi_complex > pmi_basic:
                    if direction == 'FORWARD':
                        self.valid_corrs[larger_ngram].append(gap_seg)
                    else: # BACKWARD
                        self.valid_corrs[gap_seg].append(larger_ngram)
                        
    def select_valid_null_corrs(self):               
        for corr in self.compacted_corr_counts:
            corr_ngram = Ngram(corr, lang=self.lang1)
            if corr_ngram.size > 1: # e.g. ('s', 'k')
                self.eval_null_corr(
                    larger_ngrams=[corr],
                    gap_segs=self.compacted_corr_counts[corr],
                    direction='FORWARD',
                )        
                
            else: # str e.g. 'ʃ'
                self.eval_null_corr(
                    larger_ngrams=self.compacted_corr_counts[corr],
                    gap_segs=[corr],
                    direction='BACKWARD',
                )
                
        return self.valid_corrs

    def prune(self, min_val=2):
        self.compacted_corr_counts = prune_corrs(self.compacted_corr_counts, min_val=min_val)

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