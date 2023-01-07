from auxFuncs import euclidean_dist
from phonCorr import PhonemeCorrDetector
from wordSim import Z_dist
from statistics import mean, stdev, StatisticsError
from math import e
from scipy.stats import norm


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
            if exclude_synonyms == True:
                sims[concept] = min(shared, 1) if shared > 0 else 0
                total_cognate_ids += 1
            else:
                sims[concept] = max(shared, 0)
                total_cognate_ids += shared + not_shared
    
    return sum(sims.values()) / total_cognate_ids


cognate_sims = {}
calibration_params = {}
def cognate_sim(lang1, lang2, clustered_cognates,
                eval_func, eval_sim, exclude_synonyms=True,
                calibrate=False,
                min_similarity=0,
                clustered_id=None,
                return_score_dict=False,
                **kwargs):
    if (lang1, lang2, clustered_id, eval_func, eval_sim, exclude_synonyms, min_similarity) in cognate_sims:
        #Try to retrieve previously calculated similarity dictionary
        sims = cognate_sims[(lang1, lang2, clustered_id, eval_func, eval_sim, exclude_synonyms, min_similarity)]
    
    else:
        sims = {}
        for concept in clustered_cognates:
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
                        score = eval_func((l1_word, lang1), (l2_word, lang2), **kwargs)
                        
                        #Transform distance into similarity
                        if eval_sim == False:
                            score = e**-score
                        
                        concept_sims[(l1_word, l2_word)] = score
                        
            if len(concept_sims) > 0:
                if exclude_synonyms == True:
                    sims[concept] = max(concept_sims.values())
                else:
                    sims[concept] = mean(concept_sims.values())
                
            else:
                if (l1_wordcount > 0) and (l2_wordcount > 0):
                    sims[concept] = 0        
        
        if len(sims) == 0:
            print(f'Error: no shared concepts found between {lang1.name} and {lang2.name}!')
            raise StatisticsError
            
        #Save score dictionary
        cognate_sims[(lang1, lang2, clustered_id, eval_func, eval_sim, exclude_synonyms, min_similarity)] = sims
        
    #Get the non-synonymous word pair scores against which to 
    #calibrate the synonymous word scores
    if calibrate == True:
        
        try:
            #Try to load previously calculated calibration parameters
            mean_nc_score, nc_score_stdev = calibration_params[(lang1, lang2, eval_func)]
            
        except KeyError:
            if len(lang1.noncognate_thresholds[(lang2, eval_func)]) > 0:
                noncognate_scores = lang1.noncognate_thresholds[(lang2, eval_func)]
            else:
                noncognate_scores = PhonemeCorrDetector(lang1, lang2).noncognate_thresholds(eval_func, **kwargs)
            #nc_len = len(noncognate_scores)
            
            #Transform distance scores into similarity scores
            if eval_sim == False:
                noncognate_scores = [e**-score for score in noncognate_scores]
            
            #Calculate mean and standard deviation from this sample distribution
            mean_nc_score = mean(noncognate_scores)
            nc_score_stdev = stdev(noncognate_scores)
            
            #Save calibration parameters
            calibration_params[(lang1, lang2, eval_func)] = mean_nc_score, nc_score_stdev
    
    #Apply minimum similarity and calibration
    for concept in sims:
        score = sims[concept]
        
        #Calibrate score against scores of non-synonymous word pairs
        #pnorm: proportion of values from a normal distribution with
        #mean and standard deviation defined by those of the sample
        #of non-synonymous word pair scores, which are lower than
        #a particular value (score)
        #e.g. pnorm = 0.99 = 99% of values from the distribution
        #of non-synonymous word pair scores are lower than the given score
        #The higher this value, the more confident we can be that
        #the given score does not come from that distribution, 
        #i.e. that it is truly a cognate
        if calibrate == True:
            pnorm = norm.cdf(score, loc=mean_nc_score, scale=nc_score_stdev)
            score *= pnorm
        
        sims[concept] = score if score >= min_similarity else 0
    
    if return_score_dict == True:
        return sims 
    
    else:
        mean_sim = mean(sims.values())

        return mean_sim
            

def weighted_cognate_sim(lang1, lang2, 
                         clustered_cognates, 
                         eval_funcs, eval_sims, weights=None,
                         exclude_synonyms=True, 
                         return_score_dict=False,
                         **kwargs):
    if weights == None:
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
    

#%%

def Z_score_dist(lang1, lang2, eval_func, eval_sim,
                 concept_list=None, exclude_synonyms=True,
                 seed=1,
                 **kwargs):
    if concept_list == None:
        concept_list = [concept for concept in lang1.vocabulary 
                        if concept in lang2.vocabulary]
    else:
        concept_list = [concept for concept in concept_list 
                        if concept in lang1.vocabulary 
                        if concept in lang2.vocabulary]
    
    #Generate a dictionary of word form pairs
    word_forms = {concept:[((entry1[1], lang1), (entry2[1], lang2)) 
                           for entry1 in lang1.vocabulary[concept] 
                           for entry2 in lang2.vocabulary[concept]] 
                  for concept in concept_list}
    
    #Score the word form pairs according to the specified function
    scores = {concept:[eval_func(pair[0], pair[1], **kwargs) for pair in word_forms[concept]] 
              for concept in word_forms}
    
    #Get the non-synonymous word pair scores against which to calibrate the synonymous word scores
    if len(lang1.noncognate_thresholds[(lang2, eval_func)]) > 0:
        noncognate_scores = lang1.noncognate_thresholds[(lang2, eval_func)]
    else:
        noncognate_scores = PhonemeCorrDetector(lang1, lang2).noncognate_thresholds(eval_func, **kwargs)
    nc_len = len(noncognate_scores)
        
    
    #Calculate the p-values for the synonymous word pairs against non-synonymous word pairs
    if eval_sim == True:
        p_values = {concept:[(len([nc_score for nc_score in noncognate_scores if nc_score >= score])+1) / (nc_len+1) 
                             for score in scores[concept]] 
                    for concept in scores}
        
        
    else:
        p_values = {concept:[(len([nc_score for nc_score in noncognate_scores if nc_score <= score])+1) / (nc_len+1) 
                             for score in scores[concept]] 
                    for concept in scores}
   
    #Exclude synonyms if specified
    if exclude_synonyms == True:
        p_values = [min(p_values[concept]) for concept in p_values]
    else:
        p_values = [p for concept in p_values for p in p_values[concept]]
    
    
    return Z_dist(p_values)    


#DONE:
#1) Binary cognate similarity (with/without synonyms)
#2) Phonetic word similarity (basic/advanced, with and without feature weighting)
#3) PMI similarity/distance
#4) Surprisal distance
#5) Hybrid distance
#6) Z-score distance
    
    
    