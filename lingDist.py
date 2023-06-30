from auxFuncs import euclidean_dist
from phonCorr import PhonemeCorrDetector
from wordSim import Z_dist
from statistics import mean, stdev, StatisticsError
from math import e
from scipy.stats import norm
import random


def binary_cognate_sim(lang1, lang2, clustered_cognates,
                       exclude_synonyms=True):
    """Calculates the proportion of shared cognates between the two languages.
    lang1   :   Language class object
    lang2   :   Language class object
    clustered_cognates  :   nested dictionary of concepts and cognate IDs with their forms
    exclude_synonyms    :   Bool, default = True
        if True, calculation is based on concepts rather than cognate IDs 
        (i.e. maximum score = 1 for each concept, regardless of how many forms
         or cognate IDs there are for the concept)"""
        
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


calibration_params = {}
def cognate_sim(lang1, lang2, clustered_cognates,
                eval_func, eval_sim, exclude_synonyms=True,
                #eval func is tuple (function, {kwarg:value})
                calibrate=True,
                min_similarity=0,
                clustered_id=None,
                seed=1,
                random_forest=True,
                n_trees=200,
                forest_size=None,
                **kwargs): # TODO **kwargs isn't used but causes an error if it's not here
    
    # Get list of shared concepts between the two languages
    shared_concepts = [concept for concept in clustered_cognates if concept in lang1.vocabulary if concept in lang2.vocabulary]
    if len(shared_concepts) == 0:
        raise StatisticsError(f'Error: no shared concepts found between {lang1.name} and {lang2.name}!')

    # Random forest approximation
    # Take N samples of the available concepts of size K
    # Calculate the cognate sim for each sample, then average together
    concept_groups = {}
    group_scores = {}
    if random_forest:
        random.seed(seed)
        # Set default forest size to 70% of shared concepts
        if not forest_size:
            forest_size = round(0.7*len(shared_concepts))
        for n in range(n_trees):
            concept_groups[n] = random.choices(shared_concepts, k=forest_size)
        # Ensure that every shared concept is in at least one of the groups
        # If not, add to smallest (if equal sizes then add to one at random)
        for concept in shared_concepts:
            present = False
            for n, group in concept_groups.items():
                if concept in group:
                    present = True
                    continue
            if not present:
                smallest = min(concept_groups.keys(), key=lambda x: len(concept_groups[x]))
                concept_groups[smallest].append(concept)
    else:
        concept_groups[1] = shared_concepts

    # TODO some of this calculation is redundant, should not be repeated per concept_group
    for n, group in concept_groups.items():
        sims = {}
        for concept in group:
            concept_sims = {}
            l1_wordcount, l2_wordcount = 0, 0
                
            for cognate_id in clustered_cognates[concept]:
                items = [entry.split('/') for entry in clustered_cognates[concept][cognate_id]]
                items = [(item[0].strip(), item[1]) for item in items]
                l1_words = [item[1] for item in items if item[0] == lang1.name]
                l2_words = [item[1] for item in items if item[0] == lang2.name]
                l1_wordcount += len(l1_words)
                l2_wordcount += len(l2_words)
                for l1_word in l1_words:
                    for l2_word in l2_words:
                        score = eval_func[0]((l1_word, lang1), (l2_word, lang2), **eval_func[1])
                        
                        # Transform distance into similarity
                        if not eval_sim:
                            score = e**-score
                        
                        concept_sims[(l1_word, l2_word)] = score
                        
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
            
        # Get the non-synonymous word pair scores against which to 
        # calibrate the synonymous word scores
        if calibrate:
            
            try:
                # Try to load previously calculated calibration parameters
                mean_nc_score, nc_score_stdev = calibration_params[(lang1, lang2, (eval_func[0], tuple(eval_func[1].items())))]
                
            except KeyError:
                if len(lang1.noncognate_thresholds[(lang2, (eval_func[0], tuple(eval_func[1].items())))]) > 0:
                    noncognate_scores = lang1.noncognate_thresholds[(lang2, (eval_func[0], tuple(eval_func[1].items())))]
                else:
                    kwargs = eval_func[1]
                    noncognate_scores = PhonemeCorrDetector(lang1, lang2).noncognate_thresholds(eval_func)
                # nc_len = len(noncognate_scores)
                
                # Transform distance scores into similarity scores
                if not eval_sim:
                    noncognate_scores = [e**-score for score in noncognate_scores]
                
                # Calculate mean and standard deviation from this sample distribution
                mean_nc_score = mean(noncognate_scores)
                nc_score_stdev = stdev(noncognate_scores)
                
                # Save calibration parameters
                calibration_params[(lang1, lang2, (eval_func[0], tuple(eval_func[1].items())))] = mean_nc_score, nc_score_stdev
        
        # Apply minimum similarity and calibration
        for concept in sims:
            score = sims[concept]
            
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
        
        mean_sim = mean(sims.values())

        group_scores[n] = mean_sim
    
    return mean(group_scores.values())
            

# TODO: update this function if necessary
def weighted_cognate_sim(lang1, lang2, 
                         clustered_cognates, 
                         eval_funcs, eval_sims, weights=None,
                         exclude_synonyms=True, 
                         **kwargs):
    if weights is None:
        weights = [1/len(eval_funcs) for i in range(len(eval_funcs))]
    sim_score = 0
    for eval_func, eval_sim, weight in zip(eval_funcs, eval_sims, weights):
        sim_score += (cognate_sim(lang1, lang2, clustered_cognates=clustered_cognates, 
                                  eval_func=eval_func, eval_sim=eval_sim, **kwargs) * weight)
    return sim_score
            
    
def hybrid_cognate_dist(lang1, lang2,
                       clustered_cognates,
                       eval_funcs, eval_sims,
                       exclude_synonyms=True,
                       **kwargs):
    scores = []
    for eval_func, eval_sim in zip(eval_funcs, eval_sims):
        measure = cognate_sim(lang1, lang2, clustered_cognates, 
                              eval_func=eval_func, eval_sim=eval_sim,
                              exclude_synonyms=exclude_synonyms)
        scores.append(1-measure)
    return euclidean_dist(scores)
    
    
def Z_score_dist(lang1, lang2, eval_func, eval_sim,
                #eval func is tuple (function, {kwarg:value})
                 concept_list=None, exclude_synonyms=True,
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
    scores = {concept:[eval_func[0](pair[0], pair[1], **eval_func[1]) for pair in word_forms[concept]] 
              for concept in word_forms}
    
    # Get the non-synonymous word pair scores against which to calibrate the synonymous word scores
    if len(lang1.noncognate_thresholds[(lang2, (eval_func[0], tuple(eval_func[1].items())))]) > 0:
        noncognate_scores = lang1.noncognate_thresholds[(lang2, (eval_func[0], tuple(eval_func[1].items())))]
    else:
        noncognate_scores = PhonemeCorrDetector(lang1, lang2).noncognate_thresholds(eval_func)
    nc_len = len(noncognate_scores)
        
    
    # Calculate the p-values for the synonymous word pairs against non-synonymous word pairs
    if eval_sim:
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


# DONE:
# 1) Binary cognate similarity (with/without synonyms)
# 2) Phonetic word similarity (basic/advanced, with and without feature weighting)
# 3) PMI similarity/distance
# 4) Surprisal distance
# 5) Hybrid distance
# 6) Z-score distance
    
    
    