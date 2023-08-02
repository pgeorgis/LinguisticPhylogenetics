from collections import defaultdict
from auxFuncs import normalize_dict, default_dict, lidstone_smoothing, surprisal, adaptation_surprisal
from phonAlign import Alignment, compatible_segments, phon_env_ngrams
from statistics import mean
import random
from itertools import product
from math import log

class PhonemeCorrDetector:
    def __init__(self, lang1, lang2, wordlist=None, seed=1):
        self.lang1 = lang1
        self.lang2 = lang2
        self.same_meaning, self.diff_meaning, self.loanwords = self.prepare_wordlists(wordlist)
        self.pmi_dict = self.lang1.phoneme_pmi[self.lang2]
        # self.surprisal_dict = self.lang1.phoneme_surprisal[self.lang2]
        self.scored_words = defaultdict(lambda:{})
        self.seed = seed
    
    def prepare_wordlists(self, wordlist):
    
        # If no wordlist is provided, by default use all concepts shared by the two languages
        if wordlist is None:
            wordlist = self.lang1.vocabulary.keys() & self.lang2.vocabulary.keys()
        
        # If a wordlist is provided, use only the concepts shared by both languages
        else:
            wordlist = set(wordlist) & self.lang1.vocabulary.keys() & self.lang2.vocabulary.keys()
            
        # Get tuple (concept, orthography, IPA, segmented IPA) for each word entry
        # TODO create Wordlist class to handle this better
        l1_wordlist = [word for concept in wordlist for word in self.lang1.vocabulary[concept]]
        l2_wordlist = [word for concept in wordlist for word in self.lang2.vocabulary[concept]]
        
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
    
    
    def align_wordlist(self, 
                       wordlist,
                       added_penalty_dict=None,
                       **kwargs):
        """Returns a list of the aligned segments from the wordlists"""
        return [Alignment(
            seq1=word1, 
            seq2=word2,
            added_penalty_dict=added_penalty_dict,
            **kwargs
            ) for word1, word2 in wordlist] # TODO: tuple would be better than list if possible
    
    
    def correspondence_probs(self, alignment_list, ngram_size=1,
                             counts=False, exclude_null=True):
        """Returns a dictionary of conditional phone probabilities, based on a list
        of Alignment objects.
        counts : Bool; if True, returns raw counts instead of normalized probabilities;
        exclude_null : Bool; if True, does not consider aligned pairs including a null segment"""
        corr_counts = defaultdict(lambda:defaultdict(lambda:0))
        for alignment in alignment_list:
            if exclude_null:
                _alignment = alignment.remove_gaps()
            else:
                _alignment = alignment.alignment
            if ngram_size > 1:
                _alignment = _alignment.pad(ngram_size, _alignment)
                
            for i in range(ngram_size-1, len(_alignment)):
                ngram = _alignment[i-(ngram_size-1):i+1]
                seg1, seg2 = list(zip(*ngram))
                corr_counts[seg1][seg2] += 1
        if not counts:
            for seg1 in corr_counts:
                corr_counts[seg1] = normalize_dict(corr_counts[seg1])

        return corr_counts


    def phon_env_corr_probs(self, alignment_list, counts=False):
        # TODO currently works only with ngram_size=1 (I think this is fine?hh)
        corr_counts = defaultdict(lambda:defaultdict(lambda:0))
        for alignment in alignment_list:
            phon_env_align = alignment._add_phon_env()
            for seg_weight1, seg2 in phon_env_align:
                corr_counts[seg_weight1][seg2] += 1
        if not counts:
            for seg1 in corr_counts:
                corr_counts[seg1] = normalize_dict(corr_counts[seg1])
        
        return corr_counts
                          
    
    def radial_counts(self, wordlist, radius=2, normalize=True):
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
                          samples=10,
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
        print_iterations : bool, optional
            Whether to print the results of each iteration. The default is False.
        save : bool, optional
            Whether to save the results to the Language class object's phoneme_pmi attribute. The default is True.
        Returns
        -------
        results : collections.defaultdict
            Nested dictionary of phoneme PMI values.
        """
        # First step: calculate probability of phones co-occuring within within 
        # a set radius of positions within their respective words
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
        
        sample_results = {}
        # Take a sample of same-meaning words, by default 70% of available same-meaning pairs
        sample_size = round(len(self.same_meaning)*0.7)
        # Take N samples of different-meaning words, perform PMI calibration, then average all of the estimates from the various samples
        for sample_n in range(samples):
            random.seed(self.seed+sample_n)
            synonym_sample = random.sample(self.same_meaning, sample_size)
            # Take a sample of different-meaning words, as large as the same-meaning set
            diff_sample = random.sample(self.diff_meaning, min(sample_size, len(self.diff_meaning)))

            # At each following iteration N, re-align using the pmi_stepN as an 
            # additional penalty, and then recalculate PMI
            iteration = 0
            PMI_iterations = {iteration:pmi_step1}
            
            def _sort_wordlist(wordlist):
                return sorted(wordlist, key=lambda x: (x[0].ipa, x[1].ipa))

            qualifying_words = default_dict({iteration:_sort_wordlist(synonym_sample)}, l=[])
            disqualified_words = default_dict({iteration:diff_sample}, l=[])
            while (iteration < max_iterations) and (qualifying_words[iteration] != qualifying_words[iteration-1]):
                iteration += 1

                # Align the qualifying words of the previous step using previous step's PMI
                cognate_alignments = self.align_wordlist(qualifying_words[iteration-1], added_penalty_dict=PMI_iterations[iteration-1])
                
                # Align the sample of different meaning and non-qualifying words again using previous step's PMI
                noncognate_alignments = self.align_wordlist(disqualified_words[iteration-1], added_penalty_dict=PMI_iterations[iteration-1])
                
                # Calculate correspondence probabilities and PMI values from these alignments
                cognate_probs = self.correspondence_probs(cognate_alignments)
                cognate_probs = default_dict({k[0]:{v[0]:cognate_probs[k][v] 
                                                    for v in cognate_probs[k]} 
                                            for k in cognate_probs}, l=defaultdict(lambda:0))
                # noncognate_probs = self.correspondence_probs(noncognate_alignments)
                PMI_iterations[iteration] = self.phoneme_pmi(cognate_probs)# , noncognate_probs)
                
                # Align all same-meaning word pairs
                all_alignments = self.align_wordlist(synonym_sample, added_penalty_dict=PMI_iterations[iteration-1])

                # Score PMI for different meaning words and words disqualified in previous iteration
                noncognate_PMI = []
                for alignment in noncognate_alignments:
                    noncognate_PMI.append(mean([PMI_iterations[iteration][pair[0]][pair[1]] for pair in alignment.alignment]))
                # nc_mean = mean(noncognate_PMI)
                # nc_stdev = stdev(noncognate_PMI)
                
                # Score same-meaning alignments for overall PMI and calculate p-value
                # against different-meaning alignments
                qualifying, disqualified = [], []
                for i, item in enumerate(synonym_sample):
                    alignment = all_alignments[i]
                    PMI_score = mean([PMI_iterations[iteration][pair[0]][pair[1]] for pair in alignment.alignment])
                    
                    # pnorm = norm.cdf(PMI_score, loc=nc_mean, scale=nc_stdev)
                    # p_value = 1 - pnorm
                    p_value = (len([score for score in noncognate_PMI if score >= PMI_score])+1) / (len(noncognate_PMI)+1)
                    if p_value < p_threshold:
                        qualifying.append(item)
                    else:
                        disqualified.append(item)
                qualifying_words[iteration] = _sort_wordlist(qualifying)
                disqualified_words[iteration] = disqualified + diff_sample
                
                # Print results of this iteration
                if print_iterations:
                    raise NotImplementedError('needs to be updated using logger.debug AND Word class objects') # TODO
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
            sample_results[sample_n] = results
        
        # Average together the PMI estimations from each sample
        if samples > 1:
            p1_all = set(p for sample_n in sample_results for p in sample_results[sample_n])
            p2_all = set(p2 for sample_n in sample_results for p1 in sample_results[sample_n] for p2 in sample_results[sample_n][p1])
            results = defaultdict(lambda:defaultdict(lambda:0))
            for p1 in p1_all:
                p1_dict = defaultdict(lambda:[])
                for p2 in p2_all:
                    for sample_n in sample_results:
                        p1_dict[p2].append(sample_results[sample_n][p1][p2])
                    results[p1][p2] = mean(p1_dict[p2])
        else:
            results = sample_results[0]

        if save:
            self.lang1.phoneme_pmi[self.lang2] = results
            self.lang2.phoneme_pmi[self.lang1] = self.reverse_corr_dict(results)
            # self.lang1.phoneme_pmi[self.lang2]['thresholds'] = noncognate_PMI
        self.pmi_dict = results
        
        return results

    
    def noncognate_thresholds(self, eval_func, sample_size=None, save=True, seed=None):
        """Calculate non-synonymous word pair scores against which to calibrate synonymous word scores"""
        
        # Set random seed: may or may not be the default seed attribute of the PhonemeCorrDetector class
        if not seed:
            seed = self.seed
        random.seed(seed)

        # Take a sample of different-meaning words, by default as large as the same-meaning set
        if sample_size is None:
            sample_size = len(self.same_meaning)
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
        
        if save:
            self.lang1.noncognate_thresholds[(self.lang2, eval_func)] = noncognate_scores
        
        return noncognate_scores
        
    def get_possible_ngrams(self, lang, ngram_size, phon_env=False):
        # Iterate over all possible/attested ngrams
        # Only perform calculation for ngrams which have actually been observed/attested to 
        # in the current dataset or which could have been observed (with gaps)
        if phon_env:
            attested = set(tuple(ngram.split()) 
                           if type(ngram) == str else ngram 
                           for ngram in lang.list_ngrams(ngram_size, phon_env=True))
            phone_contexts = [(seg, env) 
                              for seg in lang.phon_environments 
                              for env in lang.phon_environments[seg]]
            all_ngrams = product(phone_contexts+['# ', '-'], repeat=ngram_size)
            
        else:
            attested = set(tuple(ngram.split()) 
                           if type(ngram) == str else ngram 
                           for ngram in lang.list_ngrams(ngram_size, phon_env=False))
            all_ngrams = product(list(lang.phonemes.keys())+['# ', '-'], repeat=ngram_size)

        gappy = set(ngram for ngram in all_ngrams if '-' in ngram)
        all_ngrams = attested.union(gappy)
        return all_ngrams
    
    def phoneme_surprisal(self, 
                          correspondence_counts, 
                          phon_env_corr_counts=None, 
                          ngram_size=1, 
                          weights=None, 
                          attested_only=True,
                          alpha=0.2): # TODO conduct more formal experiment to select default alpha, or find way to adjust automatically
                          #so far alpha=0.2 is best (at least on Romance)
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
                    if ('-',)*(max(ngram_size, 2)) not in [ngram1, ngram2]:
                        # forward
                        # interpolation[i][ngram1[-i:]][ngram2[-1]] += correspondence_counts[ngram1][ngram2]
                        
                        # backward
                        interpolation[i][ngram1[:i]][ngram2[0]] += correspondence_counts[ngram1][ngram2]
        
        # Add in phonological environment correspondences, e.g. ('l', '#S<') (word-initial 'l') with 'ʎ' 
        if phon_env:
            for ngram1 in phon_env_corr_counts:
                if ngram1 == '-':
                    continue

                phonEnv = ngram1[-1]
                phonEnv_contexts = set(context for context in phon_env_ngrams(phonEnv) if context != 'S')
                for context in phonEnv_contexts:
                    ngram1_context = ngram1[:-1] + (context,)

                    for ngram2 in phon_env_corr_counts[ngram1]: # TODO where do these ngram2 come from?

                        #backward
                        interpolation['phon_env'][(ngram1_context,)][ngram2] += phon_env_corr_counts[ngram1][ngram2]
        
        smoothed_surprisal = defaultdict(lambda:defaultdict(lambda:self.lang2.phoneme_entropy*ngram_size))
        all_ngrams_lang1 = self.get_possible_ngrams(self.lang1, ngram_size=ngram_size, phon_env=False)
        # lang2 ngram size fixed at 1, only trying to predict single phone; also not trying to predict phon_env 
        all_ngrams_lang2 = self.get_possible_ngrams(self.lang2, ngram_size=1, phon_env=False) 
        all_ngrams_lang2 = [ngram[0] for ngram in all_ngrams_lang2]
        
        if phon_env:
            # phon_env ngrams fixed at size 1
            all_ngrams_lang1 = self.get_possible_ngrams(self.lang1, ngram_size=1, phon_env=True)

        for ngram1 in all_ngrams_lang1:
            # if phon_env:
            #     ngram1 = ngram1[0]
            #     if ngram_size > 1:
            #         raise NotImplementedError('TODO: does ngram1[0] work with ngram_size > 1 here? if so, remove this error')
            # else:
            if ngram1[0][0] == '-':
                continue

            if phon_env:
                ngram1_phon_env = ngram1[:][:i]
                ngram1 = tuple(i[0] for i in ngram1)
                if sum(interpolation['phon_env'][ngram1_phon_env].values()) == 0:
                    continue

            if sum(interpolation[i][ngram1[:i]].values()) == 0:
                    continue

            for ngram2 in all_ngrams_lang2:
                if interpolation[i][ngram1[:i]].get(ngram2, 0) == 0:
                    continue

                ngram_weights = weights[:]
                # forward # TODO has not been updated since before addition of phon_env
                # estimates = [interpolation[i][ngram1[-i:]][ngram2] / sum(interpolation[i][ngram1[-i:]].values())
                #              if i > 1 else lidstone_smoothing(x=interpolation[i][ngram1[-i:]][ngram2], 
                #                                               N=sum(interpolation[i][ngram1[-i:]].values()), 
                #                                               d = len(self.lang2.phonemes) + 1)  
                #              for i in range(ngram_size,0,-1)]
                # backward
                estimates = [lidstone_smoothing(x=interpolation[i][ngram1[:i]].get(ngram2, 0), 
                                                N=sum(interpolation[i][ngram1[:i]].values()), 
                                                d = len(self.lang2.phonemes) + 1,
                                                # modification: I believe the d (vocabulary size) value should be every combination of phones from lang1 and lang2
                                                #d = n_ngram_pairs + 1,
                                                # updated mod: it should actually be the vocabulary GIVEN the phone of lang1, otherwise skewed by frequency of phone1
                                                #d = len(interpolation[i][ngram1[:i]]),
                                                alpha=alpha)
                            for i in range(ngram_size,0,-1)]
                
                # add interpolation with phon_env surprisal
                if phon_env:
                    phonEnv = ngram1_phon_env[-1][-1]
                    phonEnv_contexts = set(context for context in phon_env_ngrams(phonEnv) if context != 'S')
                    for context in phonEnv_contexts:
                        ngram1_context = ngram1_phon_env[0][:-1] + (context,)
                        estimates.append(lidstone_smoothing(x=interpolation['phon_env'][(ngram1_context,)].get(ngram2, 0), 
                                                            N=sum(interpolation['phon_env'][(ngram1_context,)].values()), 
                                                            d = len(self.lang2.phonemes) + 1,
                                                            alpha=alpha)
                                                            )
                        # Weight each contextual estimate based on the size of the context
                        # #S< would have weight 3 because it considers the segment plus context on both sides
                        # #S would have weight 2 because it considers only the segment plus context on one side 
                        ngram_weights.append(len(context))

                weight_sum = sum(ngram_weights)
                ngram_weights = [i/weight_sum for i in ngram_weights]
                assert(len(ngram_weights) == len(estimates))
                smoothed = sum([estimate*weight for estimate, weight in zip(estimates, ngram_weights)])
                if phon_env:
                    smoothed_surprisal[ngram1_phon_env][ngram2] = surprisal(smoothed)
                else:
                    smoothed_surprisal[ngram1][ngram2] = surprisal(smoothed)

            oov_estimates = [lidstone_smoothing(x=0, N=sum(interpolation[i][ngram1[:i]].values()), 
                                             d = len(self.lang2.phonemes) + 1,
                                             alpha=alpha) 
                          for i in range(ngram_size,0,-1)]
            if phon_env:
                for context in phonEnv_contexts:
                    ngram1_context = ngram1_phon_env[0][:-1] + (context,)
                    oov_estimates.append(lidstone_smoothing(x=0, 
                                                            N=sum(interpolation['phon_env'][(ngram1_context,)].values()), 
                                                            d = len(self.lang2.phonemes) + 1,
                                                            alpha=alpha)
                    )
            assert (len(ngram_weights) == len(oov_estimates))
            smoothed_oov = surprisal(sum([estimate*weight for estimate, weight in zip(oov_estimates, weights)]))
            
            if phon_env:
                smoothed_surprisal[ngram1_phon_env] = default_dict(smoothed_surprisal[ngram1_phon_env], l=smoothed_oov)
            else:
                smoothed_surprisal[ngram1] = default_dict(smoothed_surprisal[ngram1], l=smoothed_oov)
                
        return smoothed_surprisal
    
    def calc_phoneme_surprisal(self, radius=2, 
                               max_iterations=10, 
                               p_threshold=0.1,
                               ngram_size=2,
                               gold=False, # TODO add same with PMI?
                               print_iterations=False,
                               samples=10,
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

        if not gold:
            # Take N samples of same and different-meaning words, perform surprisal calibration, then average all of the estimates from the various samples
            sample_size = round(len(self.same_meaning)*0.7)
            sample_results = {}
            for sample_n in range(samples):
                random.seed(self.seed+sample_n)
                same_sample = random.sample(self.same_meaning, sample_size)
                # Take a sample of different-meaning words, as large as the same-meaning set
                diff_sample = random.sample(self.diff_meaning, min(sample_size, len(self.diff_meaning)))
                diff_meaning_alignments = self.align_wordlist(diff_sample, added_penalty_dict=self.pmi_dict)

                # Align same-meaning and different meaning word pairs using PMI values: 
                # the alignments will remain the same throughout iteration
                same_meaning_alignments = self.align_wordlist(same_sample, added_penalty_dict=self.pmi_dict)

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
                    for i, item in enumerate(same_sample):
                        alignment = same_meaning_alignments[i]
                        surprisal_score = adaptation_surprisal(alignment, 
                                                            surprisal_dict=surprisal_iterations[iteration], 
                                                            normalize=True,
                                                            ngram_size=ngram_size)
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
                    if print_iterations: # TODO change this to loglevel=DEBUG
                        print(f'Iteration {iteration}')
                        print(f'\tQualified: {len(qualifying)}')
                        added = [same_sample[i] for i in qualifying_words[iteration]
                                if i not in qualifying_words[iteration-1]]
                        for item in added:
                            word1, word2 = item[0][1], item[1][1]
                            ipa1, ipa2 = item[0][2], item[1][2]
                            print(f'\t\t{word1} /{ipa1}/ - {word2} /{ipa2}/')
                        
                        print(f'\tDisqualified: {len(disqualified)}')
                        removed = [same_sample[i] for i in qualifying_words[iteration-1] 
                                if i not in qualifying_words[iteration]]
                        for item in removed:
                            word1, word2 = item[0][1], item[1][1]
                            ipa1, ipa2 = item[0][2], item[1][2]
                            print(f'\t\t{word1} /{ipa1}/ - {word2} /{ipa2}/')
                
                cognate_alignments = [same_meaning_alignments[i] for i in qualifying_words[iteration]]
                sample_results[sample_n] = surprisal_iterations[iteration]
            
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
                    oov_values = []
                    for sample_n in sample_results:
                        # Use non-IPA character <?> to retrieve OOV value from surprisal dict
                        oov_values.append(sample_results[sample_n][p1]['?'])
                    surprisal_results[p1] = default_dict(surprisal_results[p1], l=mean(oov_values))
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
                                                                                 ngram_size=ngram_size), 
                                                                                 ngram_size=ngram_size)
        
        # Add phonological environment weights after final iteration
        phon_env_surprisal = self.phoneme_surprisal(
            self.correspondence_probs(cognate_alignments, counts=True), 
            phon_env_corr_counts=self.phon_env_corr_probs(cognate_alignments, counts=True),
            ngram_size=ngram_size
            )
        
        # Return and save the final iteration's surprisal results
        if save:
            self.lang1.phoneme_surprisal[(self.lang2, ngram_size)] = surprisal_results
            self.lang1.phon_env_surprisal[(self.lang2, ngram_size)] = phon_env_surprisal
        self.surprisal_dict = surprisal_results
        
        return surprisal_results, phon_env_surprisal