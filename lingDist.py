from collections import defaultdict
from functools import lru_cache
import random
from statistics import mean, stdev, StatisticsError
from scipy.stats import norm
from auxFuncs import dist_to_sim
from wordDist import Z_dist


def binary_cognate_sim(lang1, lang2, clustered_cognates,
                       exclude_synonyms=True):
    """Calculates the proportion of shared cognates between the two languages.
    lang1   :   Language class object
    lang2   :   Language class object
    clustered_cognates  :   nested dictionary of words organized by concepts and cognate classes
    exclude_synonyms    :   Bool, default = True
        if True, calculation is based on concepts rather than cognate IDs 
        (i.e. maximum score = 1 for each concept, regardless of how many forms
         or cognate IDs there are for the concept)"""
    raise NotImplementedError('update for Word class objects') # TODO
    sims = {}
    total_cognate_ids = 0
    for concept in clustered_cognates:
        shared, not_shared = 0, 0
        l1_words, l2_words = 0, 0
        for cognate_id in clustered_cognates[concept]:
            langs_with_form = set(entry.split('/')[0].strip() 
                               for entry in clustered_cognates[concept][cognate_id])
            if lang1.name in langs_with_form:
                l1_words += 1
                if lang2.name in langs_with_form:
                    l2_words += 1
                    shared += 1
                else:
                    not_shared += 1
            elif lang2.name in langs_with_form:
                l2_words += 1
                not_shared += 1
        
        if (l1_words > 0) and (l2_words > 0):
            if exclude_synonyms:
                sims[concept] = min(shared, 1) if shared > 0 else 0
                total_cognate_ids += 1
            else:
                sims[concept] = max(shared, 0)
                total_cognate_ids += shared + not_shared
    
    return sum(sims.values()) / total_cognate_ids

@lru_cache(maxsize=None)
def get_calibration_params(lang1, lang2, eval_func, seed, sample_size):
    # Get the non-synonymous word pair scores against which to calibrate the synonymous word scores
    key = (lang2, eval_func, sample_size, seed)
    if len(lang1.noncognate_thresholds[key]) > 0:
        noncognate_scores = lang1.noncognate_thresholds[key]
    else:
        correlator = lang1.get_phoneme_correlator(lang2)
        noncognate_scores = correlator.noncognate_thresholds(eval_func, seed=seed, sample_size=sample_size)
    
    # Transform distance scores into similarity scores
    if not eval_func.sim:
        noncognate_scores = map(dist_to_sim, noncognate_scores)
    
    # Calculate mean and standard deviation from this sample distribution
    mean_nc_score = mean(noncognate_scores)
    nc_score_stdev = stdev(noncognate_scores)
    
    # Save calibration parameters
    return mean_nc_score, nc_score_stdev


def cognate_sim(lang1, 
                lang2, 
                clustered_cognates,
                eval_func, 
                exclude_synonyms=True, # TODO improve exclude_synonyms
                calibrate=True,
                min_similarity=0,
                clustered_id=None, # TODO incorporate or remove
                seed=1,
                n_samples=50,
                sample_size=0.8,
                logger=None):
    
    # Get list of shared concepts between the two languages
    shared_concepts = list(clustered_cognates.keys() & lang1.vocabulary.keys() & lang2.vocabulary.keys())
    if len(shared_concepts) == 0:
        raise StatisticsError(f'Error: no shared concepts found between {lang1.name} and {lang2.name}!')

    # Random forest-like sampling
    # Take N samples of the available concepts of size K
    # Calculate the cognate sim for each sample, then average together
    group_scores = {}
    if n_samples > 1:
        random.seed(seed)
        # Set default sample size to 80% of shared concepts
        sample_n = round(sample_size*len(shared_concepts))
        # Create N samples of size K
        concept_groups = {n: set(random.choices(shared_concepts, k=sample_n)) for n in range(n_samples)}

        # Ensure that every shared concept is in at least one of the groups
        # If not, add to smallest (if equal sizes then add to one at random)
        for concept in shared_concepts:
            if concept not in concept_groups.values():
                smallest_group = min(concept_groups.keys(), key=lambda x: len(concept_groups[x]))
                concept_groups[smallest_group].add(concept)
    else:
        concept_groups = {0: shared_concepts}

    # Score all shared concepts in advance, then calculate similarity based on the N samples of concepts
    scored_pairs = defaultdict(lambda:{})
    for concept in shared_concepts:
        for cognate_id in clustered_cognates[concept]:
            l1_words = list(filter(lambda word: word.language == lang1, clustered_cognates[concept][cognate_id]))
            l2_words = list(filter(lambda word: word.language == lang2, clustered_cognates[concept][cognate_id]))
            for l1_word in l1_words:
                for l2_word in l2_words:
                    score = eval_func.eval(l1_word, l2_word)
                    
                    # Transform distance into similarity
                    if not eval_func.sim:
                        score = dist_to_sim(score)
                    
                    # Save in scored_pairs
                    scored_pairs[cognate_id][(l1_word, l2_word)] = score

    for n, group in concept_groups.items():
        sims = {}
        group_size = len(group)
        for concept in group:
            concept_sims = {}
            l1_wordcount, l2_wordcount = 0, 0
                
            for cognate_id in clustered_cognates[concept]:                
                l1_wordcount += len(set(pair[0] for pair in scored_pairs[cognate_id]))
                l1_wordcount += len(set(pair[1] for pair in scored_pairs[cognate_id]))
                concept_sims.update(scored_pairs[cognate_id])
                        
            if len(concept_sims) > 0:
                if exclude_synonyms:
                    sims[concept] = max(concept_sims.values())
                else:
                    sims[concept] = mean(concept_sims.values())
                
            else:
                if (l1_wordcount > 0) and (l2_wordcount > 0):
                    sims[concept] = 0        
        
        if len(sims) == 0:
            continue
            
        # Get the non-synonymous word pair scores against which to calibrate the synonymous word scores
        if calibrate:
            # TODO is this necessary to recalculate per group?
            mean_nc_score, nc_score_stdev = get_calibration_params(lang1, lang2, eval_func, seed+n, group_size)
        
        # Apply minimum similarity and calibration
        for concept, score in sims.items():
            
            # Calibrate score against scores of non-synonymous word pairs
            # pnorm: proportion of values from a normal distribution with
            # mean and standard deviation defined by those of the sample
            # of non-synonymous word pair scores, which are lower than
            # a particular value (score)
            # e.g. pnorm = 0.99 = 99% of values from the distribution
            # of non-synonymous word pair scores are lower than the given score
            # The higher this value, the more confident we can be that
            # the given score does not come from that distribution, 
            # i.e. that it is truly a cognate
            if calibrate:
                pnorm = norm.cdf(score, loc=mean_nc_score, scale=nc_score_stdev)
                score *= pnorm
            
            sims[concept] = score if score >= min_similarity else 0
        
        group_scores[n] = mean(sims.values())
    
    score = mean(group_scores.values())

    if logger:
        logger.debug(f'Similarity of {lang1.name} and {lang2.name}: {round(score, 3)}')
    
    return score
    
    
def Z_score_dist(lang1, 
                 lang2, 
                 eval_func,
                #eval func was tuple (function, {kwarg:value}), now is Distance class object
                 concept_list=None, 
                 exclude_synonyms=True,
                 seed=1,
                 **kwargs):
    if concept_list is None:
        concept_list = [concept for concept in lang1.vocabulary 
                        if concept in lang2.vocabulary]
    else:
        concept_list = [concept for concept in concept_list 
                        if concept in lang1.vocabulary 
                        if concept in lang2.vocabulary]
    
    # Generate a dictionary of word form pairs
    word_forms = {concept:[((word1.ipa, lang1), (word2.ipa, lang2)) 
                           for word1 in lang1.vocabulary[concept] 
                           for word2 in lang2.vocabulary[concept]] 
                  for concept in concept_list}
    
    # Score the word form pairs according to the specified function
    scores = {concept:[eval_func.eval(pair[0], pair[1]) for pair in word_forms[concept]] 
              for concept in word_forms}
    
    # Get the non-synonymous word pair scores against which to calibrate the synonymous word scores
    if len(lang1.noncognate_thresholds[(lang2, eval_func)]) > 0:
        noncognate_scores = lang1.noncognate_thresholds[(lang2, eval_func)]
    else:
        correlator = lang1.get_phoneme_correlator(lang2)
        noncognate_scores = correlator.noncognate_thresholds(eval_func)
    nc_len = len(noncognate_scores)
        
    # Calculate the p-values for the synonymous word pairs against non-synonymous word pairs
    if eval_func.sim:
        p_values = {concept:[(len([nc_score for nc_score in noncognate_scores if nc_score >= score])+1) / (nc_len+1) 
                             for score in scores[concept]] 
                    for concept in scores}
        
    else:
        p_values = {concept:[(len([nc_score for nc_score in noncognate_scores if nc_score <= score])+1) / (nc_len+1) 
                             for score in scores[concept]] 
                    for concept in scores}
   
    # Exclude synonyms if specified
    if exclude_synonyms:
        p_values = [min(p_values[concept]) for concept in p_values]
    else:
        p_values = [p for concept in p_values for p in p_values[concept]]
    
    
    return Z_dist(p_values)    
