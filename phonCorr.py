from collections import defaultdict
from auxFuncs import normalize_dict, default_dict, lidstone_smoothing, surprisal, adaptation_surprisal
from phonAlign import phone_align, compatible_segments, prosodic_env_alignment
from phonSim.phonSim import prosodic_environment_weight
from statistics import mean, stdev
import random
from itertools import product
from math import log

class PhonemeCorrDetector:
    def __init__(self, lang1, lang2, wordlist=None):
        self.lang1 = lang1
        self.lang2 = lang2
        self.same_meaning, self.diff_meaning, self.loanwords = self.prepare_wordlists(wordlist)
        self.pmi_dict = self.lang1.phoneme_pmi[self.lang2]
        # self.surprisal_dict = self.lang1.phoneme_surprisal[self.lang2]
    
    def prepare_wordlists(self, wordlist):
    
        # If no wordlist is provided, by default use all concepts shared by the two languages
        if wordlist is None:
            wordlist = [concept for concept in self.lang1.vocabulary 
                        if concept in self.lang2.vocabulary]
        
        # If a wordlist is provided, use only the concepts shared by both languages
        else:
            wordlist = [concept for concept in wordlist 
                        if concept in self.lang1.vocabulary 
                        if concept in self.lang2.vocabulary]
            
        # Get tuple (concept, orthography, IPA, segmented IPA) for each word entry
        l1_wordlist = [(concept, entry[0], entry[1], entry[2]) 
                       for concept in wordlist for entry in self.lang1.vocabulary[concept]]
        l2_wordlist = [(concept, entry[0], entry[1], entry[2]) 
                       for concept in wordlist for entry in self.lang2.vocabulary[concept]]
        
        # Get all combinations of L1 and L2 words
        all_wordpairs = product(l1_wordlist, l2_wordlist)
        
        # Sort out same-meaning from different-meaning word pairs, and loanwords
        same_meaning, diff_meaning, loanwords = [], [], []
        for pair in all_wordpairs:
            l1_entry, l2_entry = pair
            gloss1, gloss2 = l1_entry[0], l2_entry[0]
            if gloss1 == gloss2:
                if list(l1_entry[1:]) in self.lang1.loanwords[gloss1]:
                    loanwords.append(pair)
                elif list(l2_entry[1:]) in self.lang2.loanwords[gloss2]:
                    loanwords.append(pair)
                else:
                    same_meaning.append(pair)
            else:
                diff_meaning.append(pair)
        
        # Return a tuple of the three word type lists
        return same_meaning, diff_meaning, loanwords
    
    
    def align_wordlist(self, wordlist, 
                       align_function=phone_align, **kwargs):
        """Returns a list of the aligned segments from the wordlists"""
        return [align_function(pair[0][-1], pair[1][-1], **kwargs)
                for pair in wordlist]
    
    
    def correspondence_probs(self, alignment_list, ngram_size=1,
                             counts=False, exclude_null=True):
        """Returns a dictionary of conditional phone probabilities, based on a list
        of alignments.
        counts : Bool; if True, returns raw counts instead of normalized probabilities;
        exclude_null : Bool; if True, does not consider aligned pairs including a null segment"""
        corr_counts = defaultdict(lambda:defaultdict(lambda:0))
        for alignment in alignment_list:
            if exclude_null:
                alignment = [pair for pair in alignment if '-' not in pair]
            if ngram_size > 1:
                pad_n = ngram_size - 1
                alignment = [('# ', '# ')]*pad_n + alignment + [('# ', '# ')]*pad_n
                
            for i in range(ngram_size-1, len(alignment)):
                ngram = alignment[i-(ngram_size-1):i+1]
                segs = list(zip(*ngram))
                seg1, seg2 = segs
                corr_counts[seg1][seg2] += 1
        if not counts:
            for seg1 in corr_counts:
                corr_counts[seg1] = normalize_dict(corr_counts[seg1])
        return corr_counts


    def prosodic_env_corr_probs(self, alignment_list, counts=False):
        corr_counts = defaultdict(lambda:defaultdict(lambda:0))
        for alignment in alignment_list:
            pros_env_align = prosodic_env_alignment(alignment)
            for seg_weight1, seg2 in pros_env_align:
                corr_counts[seg_weight1][seg2] += 1
        if not counts:
            for seg1 in corr_counts:
                corr_counts[seg1] = normalize_dict(corr_counts[seg1])
        
        return corr_counts
                          
    
    def radial_counts(self, wordlist, radius=2, normalize=True):
        """Checks the number of times that phones occur within a specified 
        radius of positions in their respective words from one another"""
        corr_dict = defaultdict(lambda:defaultdict(lambda:0))
        
        for item in wordlist:
            segs1, segs2 = item[0][3], item[1][3]
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
    
    def reverse_corr_dict(self, corr_dict):
        reverse = defaultdict(lambda:defaultdict(lambda:0))
        for seg1 in corr_dict:
            for seg2 in corr_dict[seg1]:
                reverse[seg2][seg1] = corr_dict[seg1][seg2]
        return reverse

    def phoneme_pmi(self, dependent_probs,
                    independent_probs=None,
                    l1=None, l2=None):
        """
        dependent_probs : nested dictionary of conditional correspondence probabilities in potential cognates
        independent_probs : None, or nested dictionary of conditional correspondence probabilities in non-cognates
        """
        if l1 is None:
            l1 = self.lang1
        if l2 is None:
            l2 = self.lang2

        # If no independent probabilities are specified, 
        # use product of phoneme probabilities by default
        if independent_probs is None:
            independent_probs = defaultdict(lambda:defaultdict(lambda:0))
            for phoneme1 in l1.phonemes:
                for phoneme2 in l2.phonemes:
                    independent_probs[phoneme1][phoneme2] = l1.phonemes[phoneme1] * l2.phonemes[phoneme2]

        # Calculate joint probabilities from conditional probabilities
        for corr_dict in [dependent_probs, independent_probs]:
            for seg1 in corr_dict:
                seg1_totals = sum(corr_dict[seg1].values())
                for seg2 in corr_dict[seg1]:
                    cond_prob = corr_dict[seg1][seg2] / seg1_totals
                    joint_prob = cond_prob * l1.phonemes[seg1]
                    corr_dict[seg1][seg2] = joint_prob
                    
        # Get set of all possible phoneme correspondences
        segment_pairs = set([(seg1, seg2)
                         for corr_dict in [dependent_probs, independent_probs]
                         for seg1 in corr_dict 
                         for seg2 in corr_dict[seg1]])
            
        # Calculate PMI for all phoneme pairs
        pmi_dict = defaultdict(lambda:defaultdict(lambda:0))
        for segment_pair in segment_pairs:
            seg1, seg2 = segment_pair
            p_ind = l1.phonemes[seg1] * l2.phonemes[seg2]
            cognate_prob = dependent_probs[seg1].get(seg2, p_ind)
            noncognate_prob = independent_probs[seg1].get(seg2, p_ind)
            pmi_dict[seg1][seg2] = log(cognate_prob/noncognate_prob)
        
        return pmi_dict


    def calc_phoneme_pmi(self, radius=2, max_iterations=10,
                          p_threshold=0.1,
                          seed=1, 
                          print_iterations=False, save=True):
        """
        Parameters
        ----------
        radius : int, optional
            Number of word positions forward and backward to check for initial correspondences. The default is 2.
        max_iterations : int, optional
            Maximum number of iterations. The default is 3.
        p_threshold : float, optional
            p-value threshold for words to qualify for PMI calculation in the next iteration. The default is 0.05.
        seed : int, optional
            Random seed for drawing a sample of different meaning word pairs. The default is 1.
        print_iterations : bool, optional
            Whether to print the results of each iteration. The default is False.
        save : bool, optional
            Whether to save the results to the Language class object's phoneme_pmi attribute. The default is True.
        Returns
        -------
        results : collections.defaultdict
            Nested dictionary of phoneme PMI values.
        """
        
        random.seed(seed)
        # Take a sample of different-meaning words, as large as the same-meaning set
        sample_size = len(self.same_meaning)
        diff_sample = random.sample(self.diff_meaning, min(sample_size, len(self.diff_meaning)))

        # First step: calculate probability of phones co-occuring within within 
        # a set radius of positions within their respective words
        # synonyms_radius = self.radial_counts(self.same_meaning, radius)
        
        
        # new
        synonyms_radius1 = self.radial_counts(self.same_meaning, radius, normalize=False)
        synonyms_radius2 = self.reverse_corr_dict(synonyms_radius1)
        for d in [synonyms_radius1, synonyms_radius2]:
            for seg1 in d:
                d[seg1] = normalize_dict(d[seg1])
        pmi_step1 = [self.phoneme_pmi(dependent_probs=synonyms_radius1, l1=self.lang1, l2=self.lang2),
                     self.phoneme_pmi(dependent_probs=synonyms_radius2, l1=self.lang2, l2=self.lang1)]
        pmi_dict_l1l2, pmi_dict_l2l1 = pmi_step1
        
        # Average together the PMI values from each direction
        pmi_step1 = defaultdict(lambda:defaultdict(lambda:0))
        for seg1 in pmi_dict_l1l2:
            for seg2 in pmi_dict_l1l2[seg1]:
                pmi_step1[seg1][seg2] = mean([pmi_dict_l1l2[seg1][seg2], pmi_dict_l2l1[seg2][seg1]])
        
        
        # At each following iteration N, re-align using the pmi_stepN as an 
        # additional penalty, and then recalculate PMI
        iteration = 0
        PMI_iterations = {iteration:pmi_step1}
        qualifying_words = default_dict({iteration:sorted(self.same_meaning)}, l=[])
        disqualified_words = default_dict({iteration:diff_sample}, l=[])
        while (iteration < max_iterations) and (qualifying_words[iteration] != qualifying_words[iteration-1]):
            iteration += 1

            # Align the qualifying words of the previous step using previous step's PMI
            cognate_alignments = self.align_wordlist(qualifying_words[iteration-1], 
                                                     added_penalty_dict=PMI_iterations[iteration-1],
                                                     segmented=True)
            
            # Align the sample of different meaning and non-qualifying words again using previous step's PMI
            noncognate_alignments = self.align_wordlist(disqualified_words[iteration-1],
                                                        added_penalty_dict=PMI_iterations[iteration-1],
                                                        segmented=True)
            
            # Calculate correspondence probabilities and PMI values from these alignments
            cognate_probs = self.correspondence_probs(cognate_alignments)
            cognate_probs = default_dict({k[0]:{v[0]:cognate_probs[k][v] 
                                                for v in cognate_probs[k]} 
                                          for k in cognate_probs}, l=defaultdict(lambda:0))
            # noncognate_probs = self.correspondence_probs(noncognate_alignments)
            PMI_iterations[iteration] = self.phoneme_pmi(cognate_probs)# , noncognate_probs)
            
            # Align all same-meaning word pairs
            all_alignments = self.align_wordlist(self.same_meaning, 
                                                 added_penalty_dict=PMI_iterations[iteration-1],
                                                 segmented=True)

            # Score PMI for different meaning words and words disqualified in previous iteration
            noncognate_PMI = []
            for alignment in noncognate_alignments:
                noncognate_PMI.append(mean([PMI_iterations[iteration][pair[0]][pair[1]] 
                                            for pair in alignment]))
            # nc_mean = mean(noncognate_PMI)
            # nc_stdev = stdev(noncognate_PMI)
            
            # Score same-meaning alignments for overall PMI and calculate p-value
            # against different-meaning alignments
            qualifying, disqualified = [], []
            for i in range(len(self.same_meaning)):
                alignment = all_alignments[i]
                item = self.same_meaning[i]
                PMI_score = mean([PMI_iterations[iteration][pair[0]][pair[1]] 
                                for pair in alignment])
                
                # pnorm = norm.cdf(PMI_score, loc=nc_mean, scale=nc_stdev)
                # p_value = 1 - pnorm
                p_value = (len([score for score in noncognate_PMI if score >= PMI_score])+1) / (len(noncognate_PMI)+1)
                if p_value < p_threshold:
                    qualifying.append(item)
                else:
                    disqualified.append(item)
            qualifying_words[iteration] = sorted(qualifying)
            disqualified_words[iteration] = disqualified + diff_sample
            
            # Print results of this iteration
            if print_iterations:
                print(f'Iteration {iteration}')
                print(f'\tQualified: {len(qualifying)}')
                added = [item for item in qualifying_words[iteration]
                         if item not in qualifying_words[iteration-1]]
                for item in added:
                    word1, word2 = item[0][1], item[1][1]
                    ipa1, ipa2 = item[0][2], item[1][2]
                    print(f'\t\t{word1} /{ipa1}/ - {word2} /{ipa2}/')
                
                print(f'\tDisqualified: {len(disqualified)}')
                removed = [item for item in qualifying_words[iteration-1]
                         if item not in qualifying_words[iteration]]
                for item in removed:
                    word1, word2 = item[0][1], item[1][1]
                    ipa1, ipa2 = item[0][2], item[1][2]
                    print(f'\t\t{word1} /{ipa1}/ - {word2} /{ipa2}/')
                    
        # Return and save the final iteration's PMI results
        results = PMI_iterations[max(PMI_iterations.keys())]
        if save:
            self.lang1.phoneme_pmi[self.lang2] = results
            self.lang2.phoneme_pmi[self.lang1] = self.reverse_corr_dict(results)
            # self.lang1.phoneme_pmi[self.lang2]['thresholds'] = noncognate_PMI
        self.pmi_dict = results
        
        return results

    
    def noncognate_thresholds(self, eval_func, seed=1, save=True):
        #eval func is tuple (function, {kwarg:value})
        """Calculate non-synonymous word pair scores against which to calibrate synonymous word scores"""
        
        random.seed(seed)

        # Take a sample of different-meaning words, by default as large as the same-meaning set
        sample_size = len(self.same_meaning)
        diff_sample = random.sample(self.diff_meaning, min(sample_size, len(self.diff_meaning)))
        noncognate_word_forms = [((item[0][2], self.lang1), (item[1][2], self.lang2)) for item in diff_sample]
        noncognate_scores = [eval_func[0](pair[0], pair[1], **eval_func[1]) for pair in noncognate_word_forms]
        
        if save:
            self.lang1.noncognate_thresholds[(self.lang2, (eval_func[0], tuple(eval_func[1].items())))] = noncognate_scores
        
        return noncognate_scores
        
    
    def phoneme_surprisal(self, correspondence_counts, ngram_size=1, weights=None,
                          attested_only=True):
        # Interpolation smoothing
        if weights is None:
            weights = [1/ngram_size for i in range(ngram_size)]
        interpolation = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0)))
        
        for i in range(ngram_size,0,-1):
            for ngram1 in correspondence_counts:
                for ngram2 in correspondence_counts[ngram1]:
                    # Exclude correspondences with a fully null ngram, e.g. ('k', 'a') with ('-', '-')
                    # Only needs to be done with ngram_size > 1
                    if ('-',)*(max(ngram_size, 2)) not in [ngram1, ngram2]:
                        # forward
                        # interpolation[i][ngram1[-i:]][ngram2[-1]] += correspondence_counts[ngram1][ngram2]
                        
                        # backward
                        interpolation[i][ngram1[:i]][ngram2[0]] += correspondence_counts[ngram1][ngram2]
            
        
        smoothed_surprisal = defaultdict(lambda:defaultdict(lambda:self.lang2.phoneme_entropy*ngram_size))
        
        # Iterate over all possible ngrams
        all_ngrams = list(product(list(self.lang1.phonemes.keys())+['# ', '-'], repeat=ngram_size))
        
        # Only perform calculation for ngrams which have actually been observed 
        # in the current dataset or which could have been observed (with gaps)
        if attested_only:
            attested = [tuple(ngram.split()) if type(ngram) == str else ngram for ngram in self.lang1.list_ngrams(ngram_size)]
            gappy = [ngram for ngram in all_ngrams if '-' in ngram]
            all_ngrams = set(attested + gappy)
            
        for ngram1 in all_ngrams:
            for ngram2 in list(self.lang2.phonemes.keys())+['# ', '-']: 
                # forward
                # estimates = [interpolation[i][ngram1[-i:]][ngram2] / sum(interpolation[i][ngram1[-i:]].values())
                #              if i > 1 else lidstone_smoothing(x=interpolation[i][ngram1[-i:]][ngram2], 
                #                                               N=sum(interpolation[i][ngram1[-i:]].values()), 
                #                                               d = len(self.lang2.phonemes) + 1)  
                #              for i in range(ngram_size,0,-1)]
                
                # backward
                estimates = [lidstone_smoothing(x=interpolation[i][ngram1[:i]].get(ngram2, 0), 
                                                 N=sum(interpolation[i][ngram1[:i]].values()), 
                                                 d = len(self.lang2.phonemes) + 1,
                                                 alpha=0.1) 
                              for i in range(ngram_size,0,-1)] 
                
                
                
                smoothed = sum([estimate*weight for estimate, weight in zip(estimates, weights)])
                smoothed_surprisal[ngram1][ngram2] = surprisal(smoothed)
                
            oov_estimates = [lidstone_smoothing(x=0, N=sum(interpolation[i][ngram1[:i]].values()), 
                                             d = len(self.lang2.phonemes) + 1,
                                             alpha=0.1) 
                          for i in range(ngram_size,0,-1)]
            smoothed_oov = surprisal(sum([estimate*weight for estimate, weight in zip(oov_estimates, weights)]))
            smoothed_surprisal[ngram1] = default_dict(smoothed_surprisal[ngram1], l=smoothed_oov)
                
        return smoothed_surprisal
    
    def calc_phoneme_surprisal(self, radius=2, 
                               max_iterations=10, 
                               p_threshold=0.1,
                               ngram_size=2,
                               print_iterations=False,
                               seed=1,
                               save=True):
        # METHOD
        # 1) Calculate phoneme PMI
        # 2) Use phoneme PMI to align 
        # 3) Iterate
        
        random.seed(seed)
        # Take a sample of different-meaning words, as large as the same-meaning set
        sample_size = len(self.same_meaning)
        diff_sample = random.sample(self.diff_meaning, min(sample_size, len(self.diff_meaning)))
        
        # Calculate phoneme PMI if not already done, for alignment purposes
        if len(self.pmi_dict) == 0:
            self.pmi_dict = self.calc_phoneme_pmi(radius=radius, 
                                                  max_iterations=max_iterations, 
                                                  p_threshold=p_threshold, 
                                                  seed=seed)
        
        # Align same-meaning and different meaning word pairs using PMI values: 
        # the alignments will remain the same throughout iteration
        same_meaning_alignments = self.align_wordlist(self.same_meaning,
                                                      added_penalty_dict=self.pmi_dict,
                                                      segmented=True)
        diff_meaning_alignments = self.align_wordlist(diff_sample,
                                                      added_penalty_dict=self.pmi_dict,
                                                      segmented=True)
        
        # At each iteration, re-calculate surprisal for qualifying and disqualified pairs
        # Then test each same-meaning word pair to see if if it meets the qualifying threshold
        iteration = 0
        surprisal_iterations = {}
        qualifying_words = default_dict({iteration:list(range(len(same_meaning_alignments)))}, l=[])
        disqualified_words = defaultdict(lambda:[])
        while (iteration < max_iterations) and (qualifying_words[iteration] != qualifying_words[iteration-1]):
            iteration += 1
            
            # Calculate surprisal from the qualifying alignments of the previous iteration
            cognate_alignments = [same_meaning_alignments[i] for i in qualifying_words[iteration-1]]
            surprisal_iterations[iteration] = self.phoneme_surprisal(self.correspondence_probs(cognate_alignments,
                                                                                               counts=True,
                                                                                               exclude_null=False, 
                                                                                               ngram_size=ngram_size), 
                                                                     ngram_size=ngram_size)
                                                                     
            # Retrieve the alignments of different-meaning and disqualified word pairs
            # and calculate adaptation surprisal for them using new surprisal values
            noncognate_alignments = diff_meaning_alignments + [same_meaning_alignments[i]
                                                               for i in disqualified_words[iteration-1]]
            noncognate_surprisal = [adaptation_surprisal(alignment, 
                                                         surprisal_dict=surprisal_iterations[iteration], 
                                                         normalize=True,
                                                         ngram_size=ngram_size) 
                                    for alignment in noncognate_alignments]
            
            # Normalize different-meaning pair surprisal scores by self-surprisal of word2
            # for i in range(len(noncognate_alignments)):
            #     alignment = noncognate_alignments[i]
            #     word2 = ''.join([pair[1] for pair in alignment if pair[1] != '-'])
            #     self_surprisal = self.lang2.self_surprisal(word2, segmented=False, normalize=False)
            #     noncognate_surprisal[i] /= self_surprisal

            # Score same-meaning alignments for surprisal and calculate p-value
            # against different-meaning alignments
            qualifying, disqualified = [], []
            for i in range(len(self.same_meaning)):
                alignment = same_meaning_alignments[i]
                item = self.same_meaning[i]
                surprisal_score = adaptation_surprisal(alignment, 
                                                       surprisal_dict=surprisal_iterations[iteration], 
                                                       normalize=True,
                                                       ngram_size=ngram_size)
                segs2 = item[1][3]
                # self_surprisal = self.lang2.self_surprisal(segs2, segmented=True, normalize=False)
                # surprisal_score /= self_surprisal
                p_value = (len([score for score in noncognate_surprisal if score <= surprisal_score])+1) / (len(noncognate_surprisal)+1)
                if p_value < p_threshold:
                    qualifying.append(i)
                else:
                    disqualified.append(i)
            qualifying_words[iteration] = qualifying
            disqualified_words[iteration] = disqualified
            
            # Print results of this iteration
            if print_iterations:
                print(f'Iteration {iteration}')
                print(f'\tQualified: {len(qualifying)}')
                added = [self.same_meaning[i] for i in qualifying_words[iteration]
                         if i not in qualifying_words[iteration-1]]
                for item in added:
                    word1, word2 = item[0][1], item[1][1]
                    ipa1, ipa2 = item[0][2], item[1][2]
                    print(f'\t\t{word1} /{ipa1}/ - {word2} /{ipa2}/')
                
                print(f'\tDisqualified: {len(disqualified)}')
                removed = [self.same_meaning[i] for i in qualifying_words[iteration-1] 
                           if i not in qualifying_words[iteration]]
                for item in removed:
                    word1, word2 = item[0][1], item[1][1]
                    ipa1, ipa2 = item[0][2], item[1][2]
                    print(f'\t\t{word1} /{ipa1}/ - {word2} /{ipa2}/')
        
        # Return and save the final iteration's surprisal results
        results = surprisal_iterations[iteration]
        if save:
            self.lang1.phoneme_surprisal[(self.lang2, ngram_size)] = results
        self.surprisal_dict = results
        
        return results