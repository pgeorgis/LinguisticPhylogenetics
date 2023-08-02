from collections import defaultdict
from math import sqrt, log, e, exp
from statistics import mean
from asjp import ipa2asjp
from nltk import edit_distance
import re
from phonSim.phonSim import consonants, vowels, glides, nasals, palatal, suprasegmental_diacritics, strip_diacritics
from phonSim.phonSim import phone_sim, get_sonority, max_sonority, prosodic_environment_weight
from auxFuncs import Distance, strip_ch, euclidean_dist, adaptation_surprisal
from phonAlign import Alignment, add_phon_env, reverse_alignment, get_alignment_iter
from phonCorr import PhonemeCorrDetector

def prepare_alignment(word1, word2, **kwargs):
    """Prepares pairwise alignment of two Word objects. 
    If the language of both words is specified, phoneme PMI is additionally used for the alignment, else only phonetic similarity. 
    If phonetic PMI has not been previously calculated, it will be calculated automatically here.

    Args:
        word1 (Word): first Word object to align
        word2 (Word): second Word object to align

    Returns:
        Alignment: aligned object of the two words
    """

    # If language is specified for both words, incorporate their phoneme PMI for the alignment
    lang1, lang2 = word1.language, word2.language
    if lang1 is not None and lang2 is not None:
        
        # Check whether phoneme PMI has been calculated for this language pair
        # If not, then calculate it; if so, then retrieve it
        if len(lang1.phoneme_pmi[lang2]) > 0:
            pmi_dict = lang1.phoneme_pmi[lang2]
        else:
            pmi_dict = PhonemeCorrDetector(lang1, lang2).calc_phoneme_pmi()
        
        # Align the phonetic sequences with phonetic similarity and phoneme PMI
        alignment = Alignment(word1, word2, added_penalty_dict=pmi_dict, **kwargs)
        
    # Perform phonetic alignment without PMI support
    else:
        alignment = Alignment(word1, word2, **kwargs)
    
    return alignment


def phonetic_dist(word1, word2=None, sim_func=phone_sim, **kwargs):
    """Calculates phonetic distance of an alignment without weighting by 
    segment type, position, etc.
    
    Input:
    ("word1", Lang1) : tuples of an IPA string and a Language class object
        
    or
        
    "word1" : IPA strings
    
    If word2 is not provided, word1 is assumed to be an alignment.""" # TODO this can be simplified using my new classes
    
    # Calculate or retrieve the alignment
    if word2:
        alignment = prepare_alignment(word1, word2)
    else:
        alignment = word1
    
    # Calculate the phonetic similarity of each aligned segment
    # Gap alignments receive a score of 0
    phone_sims = [sim_func(pair[0], pair[1], **kwargs) 
                  if alignment.gap_ch not in pair else 0 
                  for pair in alignment.alignment]
    
    # Return the mean distance
    return 1 - mean(phone_sims)


# TODO name of function below to be confirmed
def phonological_dist(word1, word2=None, # TODO word2=None is weird, could instead require either two words or a single alignment as input 
              sim_func=phone_sim,
              penalize_infocontent=False, 
              penalize_sonority=True,
              context_reduction=True, penalty_discount=2,
              prosodic_env_scaling=True,
              total_sim=False, # TODO confirm that this is the better default; I think averaging is required to normalize for different word lengths
              **kwargs):
    """Calculates phonological distance between two words on the basis of 
    the phonetic similarity of aligned segments and phonological deletion penalties.
    No weighting by segment type, position, etc.
    
    word1 : string (first word), list (alignment of two words), 
            or tuple (first word, second language)
    word2 : string (second word) or tuple (second word, second language)"""
    
    # If word2 is None, we assume word1 argument is actually an aligned word pair # TODO this can be simplified with Word/Alignment classes
    # Otherwise, align the two words
    if word2:
        alignment = prepare_alignment(word1, word2)
    else:
        alignment = word1
    gap_ch = alignment.gap_ch
    length = alignment.length
    alignment = alignment.alignment
    
    # Get list of penalties
    penalties = []
    for i, pair in enumerate(alignment):
        seg1, seg2 = pair
        
        # If the pair is a gap-aligned segment, assign the penalty 
        # based on the sonority and information content (if available) of the deleted segment
        if gap_ch in pair:
            penalty = 1
            if seg1 == gap_ch:
                deleted_segment = seg2
                index = len([alignment[j][1] for j in range(i) if alignment[j][1] != gap_ch])
                
            else:
                deleted_segment = seg1
                index = len([alignment[j][0] for j in range(i) if alignment[j][0] != gap_ch])
            
            if penalize_sonority:
                sonority = get_sonority(deleted_segment)
                sonority_penalty = 1-(sonority/(max_sonority+1))
                penalty *= sonority_penalty
            
            deleted_index = pair.index(deleted_segment)
            gap_index = deleted_index-1
            
            # Lessen the penalty under certain circumstances
            if context_reduction:
                stripped_deleted = strip_diacritics(deleted_segment)
                if i > 0:
                    previous_seg = alignment[i-1][gap_index]
                    # 1) If the deleted segment is a nasal and the corresponding 
                    # precending segment was nasalized
                    if stripped_deleted in nasals:
                        if '̃' in previous_seg: # check for nasalization diacritic:
                            penalty /= penalty_discount
                    
                    # 2) If the deleted segment is a palatal glide (j, ɥ, i̯, ɪ̯),  
                    # and the corresponding preceding segment was palatalized
                    # or is a palatal consonant
                    elif strip_diacritics(deleted_segment, excepted=['̯']) in {'j', 'ɥ', 
                                                                                'i̯', 'ɪ̯'}:
                        if strip_diacritics(previous_seg)[0] in palatal:
                            penalty /= penalty_discount
                        elif ('ʲ' in previous_seg) or ('ᶣ' in previous_seg):
                            penalty /= penalty_discount
                            
                    # 3) If the deleted segment is a high rounded/labial glide
                    # and the corresponding preceding segment was labialized
                    elif strip_diacritics(deleted_segment, excepted=['̯']) in {'w', 'ʍ', 'ʋ', 
                                                                                'u', 'ʊ', 
                                                                                'y', 'ʏ'}:
                        if ('ʷ' in previous_seg) or ('ᶣ' in previous_seg):
                            penalty /= penalty_discount
                    
                    # 4) If the deleted segment is /h, ɦ/ and the corresponding 
                    # preceding segment was aspirated or breathy
                    elif stripped_deleted in {'h', 'ɦ'}:
                        if ('ʰ' in previous_seg) or ('ʱ' in previous_seg) or ('̤' in previous_seg):
                            penalty /= penalty_discount
                    
                        # Or if the following corresponding segment is breathy or
                        # pre-aspirated
                        else:
                            try:
                                next_seg = alignment[i+1][gap_index]
                                if ('̤' in next_seg) or (next_seg[0] in {'ʰ', 'ʱ'}):
                                    penalty /= penalty_discount
                            except IndexError:
                                pass
                        
                    # 5) If the deleted segment is a rhotic approximant /ɹ, ɻ/
                    # and the corresponding preceding segment was rhoticized
                    elif stripped_deleted in {'ɹ', 'ɻ'}:
                        if (strip_diacritics(previous_seg) == 'ɚ') or ('˞' in previous_seg):
                            penalty /= penalty_discount
                    
                    # 6) If the deleted segment is a glottal stop and the corresponding
                    # preceding segment was glottalized (or creaky?)
                    elif stripped_deleted == 'ʔ':
                        if 'ˀ' in previous_seg:
                            penalty /= penalty_discount
                    
                    
                # 6) if the deleted segment is part of a long/geminate segment represented as double (e.g. /tt/ rather than /tː/), 
                # where at least one part of the geminate has been aligned
                # Method: check if the preceding or following pair contained the deleted segment at deleted_index, aligned to something other than the gap character
                # Check following pair
                double = False
                try:
                    nxt_pair = alignment[i+1]
                    if gap_ch not in nxt_pair:
                        if nxt_pair[deleted_index] == deleted_segment:
                            double = True
                            
                            # Eliminate the penalty altogether if the gemination
                            # is simply transcribed differently (see example below)
                            if ('ː' in nxt_pair[gap_index]) or ('ˑ' in nxt_pair[gap_index]):
                                penalty = 0
                        
                except IndexError:
                    pass
                
                # Check preceding pair
                if i > 0:
                    prev_pair = alignment[i-1]
                    if gap_ch not in prev_pair:
                        if prev_pair[deleted_index] == deleted_segment:
                            double = True
                        
                            # Eliminate the penalty altogether in the case of 
                            # an alignment like: [('t', 'tː'), ('t', '-')]
                            # where the gemination is simply transcribed differently
                            if ('ː' in prev_pair[gap_index]) or ('ˑ' in prev_pair[gap_index]):
                                penalty = 0
                            
                if double:
                    penalty /= penalty_discount
            
            # TODO: is this right?
            if prosodic_env_scaling:
                # Discount deletion penalty according to prosodic sonority 
                # environment (based on List, 2012)
                deleted_i = sum([1 for j in range(i+1) if alignment[j][deleted_index] != gap_ch])-1
                segment_list = [alignment[j][deleted_index] 
                                for j in range(length)
                                if alignment[j][deleted_index] != gap_ch]
                prosodic_env_weight = prosodic_environment_weight(segment_list, deleted_i)
                penalty /= sqrt(abs(prosodic_env_weight-7)+1)
            
            
            # Add the final penalty to penalty list
            penalties.append(penalty)
        
        # Otherwise take the penalty as the phonetic distance between the aligned segments
        else:
            distance = 1 - sim_func(seg1, seg2, **kwargs)
            penalties.append(distance)
    
    if total_sim:
        word_dist = sum(penalties)
    else:
        # Euclidean distance of all penalties (= distance per dimension of the word)
        # normalized by square root of number of dimensions
        word_dist = euclidean_dist(penalties) / sqrt(len(penalties))
        
    return word_dist

# TODO name of this function TBD
def segmental_word_dist(word1, word2=None, 
                       c_weight=0.5, v_weight=0.3, syl_weight=0.2):
    """Calculates the phonetic similarity of an aligned word pair according to
    weighted average similarity of consonantal segments, vocalic segments, and 
    syllable structure"""
    
    assert round(sum([c_weight, v_weight, syl_weight]),1) == 1.0
    
    # If word2 is None, we assume word1 argument is actually an aligned word pair
    # Otherwise, align the two words # TODO this can be simplified using Word/Alignment classes
    if word2:
        alignment = prepare_alignment(word1, word2)
    else:
        alignment = word1
    length = alignment.length
    alignment = alignment.alignment
    
    # Iterate through pairs of alignment:
    # Add fully consonant pairs to c_list, fully vowel/glide pairs to v_list
    # (ignore pairs of matched non-glide consonants with vowels)
    # and create syllable structure string for each word
    c_pairs, v_pairs = [], []
    syl_structure1, syl_structure2 = [], []
    for pair in alignment:
        strip_pair = (strip_diacritics(pair[0])[-1], strip_diacritics(pair[1])[-1])
        if (strip_pair[0] in consonants) and (strip_pair[1] in consonants):
            c_pairs.append(pair)
            syl_structure1.append('C')
            syl_structure2.append('C')
        elif (strip_pair[0] in vowels+glides) and (strip_pair[1] in vowels+glides):
            v_pairs.append(pair)
            syl_structure1.append('V')
            syl_structure2.append('V')
        else:
            if strip_pair[0] in consonants:
                syl_structure1.append('C')
            elif strip_pair[0] in vowels:
                syl_structure1.append('V')
            if strip_pair[1] in consonants:
                syl_structure2.append('C')
            elif strip_pair[1] in vowels:
                syl_structure2.append('V')
    
    # Count numbers of consonants and vowels: take the larger count to account
    # for possibly deleted consonants or vowels
    N_c = max(syl_structure1.count('C'), syl_structure2.count('C'))
    N_v = max(syl_structure1.count('V'), syl_structure2.count('V'))
    
    # Consonant score: mean phonetic similarity of all matched consonantal pairs, divided by number of consonant segments
    try:
        c_score = sum([phone_sim(pair[0], pair[1]) for pair in c_pairs]) / N_c
    except ZeroDivisionError:
        c_score = 1
    
    # Vowel score: sum of phonetic similarity of all matched vowel pairs, divided by number of vowel segments
    try:
        v_score = sum([phone_sim(pair[0], pair[1]) for pair in v_pairs]) / N_v
    except ZeroDivisionError:
        v_score = 1
    
    # Syllable score: length-normalized Levenshtein distance of syllable structure strings
    syl_structure1, syl_structure2 = ''.join(syl_structure1), ''.join(syl_structure2)
    syl_score = edit_distance(syl_structure1, syl_structure2) / length
    
    # Final score: weighted sum of each component score as distance
    return (c_weight * (1-c_score)) + (v_weight * (1-v_score)) + (syl_weight * syl_score)


def mutual_surprisal(word1, word2, ngram_size=1, phon_env=True, **kwargs):
    lang1 = word1.language
    lang2 = word2.language
    
    # Check whether phoneme PMI has been calculated for this language pair
    # Otherwise calculate from scratch # TODO use helper function
    if len(lang1.phoneme_pmi[lang2]) > 0:
        pmi_dict = lang1.phoneme_pmi[lang2]
    else:
        pmi_dict = PhonemeCorrDetector(lang1, lang2).calc_phoneme_pmi(**kwargs)
    
    # Calculate phoneme surprisal if not already done # TODO use helper function
    if len(lang1.phoneme_surprisal[(lang2, ngram_size)]) == 0:
        # TODO add a logging message so that we know surprisal is being calculated (again) -- maybe best within phonCorr.py
        surprisal_dict_l1l2 = PhonemeCorrDetector(lang1, lang2).calc_phoneme_surprisal(ngram_size=ngram_size, **kwargs)
    if len(lang2.phoneme_surprisal[(lang1, ngram_size)]) == 0:
        surprisal_dict_l2l1 = PhonemeCorrDetector(lang2, lang1).calc_phoneme_surprisal(ngram_size=ngram_size, **kwargs)
    
    # Generate alignments in each direction: alignments need to come from PMI
    alignment = Alignment(word1, word2, added_penalty_dict=pmi_dict, phon_env=phon_env)
    gap_ch = alignment.gap_ch
    forward_alignment = get_alignment_iter(alignment, phon_env=phon_env)
    # Reverse alignment: needs to be reversed, then phological environment added 
    rev_alignment = add_phon_env(reverse_alignment(alignment, phon_env=False), gap_ch=gap_ch)

    # Calculate the word-adaptation surprisal in each direction
    # (note: alignment needs to be reversed to run in second direction)
    if phon_env:
        sur_dict1 = lang1.phon_env_surprisal[(lang2, ngram_size)]
        sur_dict2 = lang2.phon_env_surprisal[(lang1, ngram_size)]
    else:
        sur_dict1 = lang1.phoneme_surprisal[(lang2, ngram_size)]
        sur_dict2 = lang2.phoneme_surprisal[(lang1, ngram_size)]
    WAS_l1l2 = adaptation_surprisal(forward_alignment,
                                    surprisal_dict=sur_dict1,
                                    ngram_size=ngram_size,
                                    normalize=False)
    if ngram_size > 1:
        breakpoint() # TODO issue is possibly that the ngram size of 2 is not actually in the dict keys also including phon env, just has phon_env OR 2gram in separate dicts... 
        raise NotImplementedError
    WAS_l2l1 = adaptation_surprisal(rev_alignment, 
                                    surprisal_dict=sur_dict2,
                                    ngram_size=ngram_size,
                                    normalize=False)

    # Calculate self-surprisal values in each direction
    self_surprisal1 = lang1.self_surprisal(word1, normalize=False)
    self_surprisal2 = lang2.self_surprisal(word2, normalize=False)

    # Weight surprisal values by self-surprisal/information content value of corresponding segment
    # Segments with greater information content weighted more heavily
    def weight_by_self_surprisal(alignment, WAS, self_surprisal, gap_ch=gap_ch):
        self_info = sum([self_surprisal[j][-1] for j in self_surprisal])
        weighted_WAS = []
        adjust = 0
        if type(alignment) is Alignment:
            alignment = alignment.alignment
        for i, pair in enumerate(alignment):
            if pair[0][0] != gap_ch: # TODO: is the second 0 index necessary here? I think pair[0] would work exactly the same
                # TODO I wonder if skipping the gaps had something to do with the former success?
                weight = self_surprisal[(i-adjust)][-1] / self_info # TODO: this adjust method might not work if the gap is at the beginning of the alignment
                weighted = weight * WAS[i]
                weighted_WAS.append(weighted)
            else:
                adjust += 1
        return weighted_WAS
    weighted_WAS_l1l2 = weight_by_self_surprisal(alignment, WAS_l1l2, self_surprisal1)
    weighted_WAS_l2l1 = weight_by_self_surprisal(rev_alignment, WAS_l2l1, self_surprisal2)
    
    # Return and save the average of these two values
    score = mean([mean(weighted_WAS_l1l2), mean(weighted_WAS_l2l1)])
    # TODO Treat surprisal values as distances and compute euclidean distance over these, then take average
    # score = mean([euclidean_dist(weighted_WAS_l1l2), euclidean_dist(weighted_WAS_l2l1)])

    return score


def pmi_dist(word1, word2, sim2dist=True, alpha=0.5, **kwargs):
    lang1 = word1.language
    lang2 = word2.language
    
    # Check whether phoneme PMI has been calculated for this language pair
    # Otherwise calculate from scratch # TODO create a function or something for this since this requirement appears in multiple places
    if len(lang1.phoneme_pmi[lang2]) > 0:
        pmi_dict = lang1.phoneme_pmi[lang2]
    else:
        pmi_dict = PhonemeCorrDetector(lang1, lang2).calc_phoneme_pmi(**kwargs)
        
    # Align the words with PMI
    alignment = Alignment(word1, word2, added_penalty_dict=pmi_dict)
    
    # Calculate PMI scores for each aligned pair
    PMI_values = [pmi_dict[pair[0]][pair[1]] for pair in alignment.alignment]

    # # TODO: Weight by information content per segment
    # info_content1 = lang1.calculate_infocontent(word1, segmented=False)
    # info_content2 = lang2.calculate_infocontent(word2, segmented=False)
    # def weight_by_info_content(alignment, PMI_vals, info1, info2):
    #     self_info1 = sum([info1[j][-1] for j in info1])
    #     self_info2 = sum([info2[j][-1] for j in info2])
    #     weighted_PMI = []
    #     #adjust1, adjust2 = 0, 0
    #     for i, pair in enumerate(alignment.alignment):
    #         breakpoint()
    #         # if pair[0] == '-':
    #         #     adjust1 += 1
    #         # elif pair[1] == '-':
    #         #     adjust2 += 1
    #         if pair[0] == '-':
    #             pass
    #         else:
    #             pass
    #         if pair[-1] == '-':
    #             pass
    #         else:
    #             pass

            
    #         # Take the information content value of each segment within the respective word
    #         # Divide this by the total info content of the word to calculate the proportion of info content constituted by the segment
    #         # Average together the info contents of each aligned segment
    #         # Weight by the averaged values
    #         weight1 = info1[i-adjust1][-1] / self_info1
    #         try:
    #             weight2 = info2[i-adjust2][-1] / self_info2
    #         except KeyError:
    #             breakpoint()
    #         weight = mean([weight1, weight2])
    #         weighted = weight * PMI_vals[i]
    #         weighted_PMI.append(weighted)
    #     return weighted_PMI
    
    # PMI_values = weight_by_info_content(alignment, PMI_values, info_content1, info_content2)
    PMI_score = mean(PMI_values) 
    
    if sim2dist:
        PMI_dist = exp(-max(PMI_score, 0)**alpha)
        return PMI_dist
    
    else:
        return PMI_score


def levenshtein_dist(word1, word2, normalize=True, asjp=True):
    word1 = word1.ipa
    word2 = word2.ipa
    # TODO create helper function for these fixes
    fixes = {'ᵐ':'m', # bilabial prenasalization diacritic
             '̈':'', # central diacritic
             }
    for f in fixes:
        word1 = re.sub(f, fixes[f], word1)
        word2 = re.sub(f, fixes[f], word2)
        
    if asjp:
        word1 = strip_ch(ipa2asjp(word1), ["~"])        
        word2 = strip_ch(ipa2asjp(word2), ["~"])
        
    LevDist = edit_distance(word1, word2)
    if normalize:
        LevDist /= max(len(word1), len(word2))
        
    return LevDist


def hybrid_dist(word1, word2, funcs:dict, weights=None)->float:
    """Calculates a hybrid distance of multiple distance or similarity functions

    Args:
        word1 (phyloLing.Word): first Word object
        word2 (phyloLing.Word): second Word object
        funcs (iterable): iterable of Distance class objects
    Returns:
        float: hybrid similarity measure
    """
    scores = []
    if weights is None:
        weights = [1/len(funcs) for i in range(len(funcs))]
    else:
        assert len(weights) == len(funcs)
    assert round(sum(weights)) == 1.0
    for func, weight in zip(funcs, weights):
        if weight == 0:
            continue
        func_sim = func.sim
        score = func.eval(word1, word2)
        if func_sim:
            score = 1 - score
        
        # Distance weighting concept: if a distance is weighted with a higher coefficient relative to another distance,
        # it is as if that dimension is more impactful 
        scores.append(score*weight)

    #score = euclidean_dist(scores)
    score = sum(scores)
    
    return score

# Initialize distance functions as Distance objects
LevenshteinDist = Distance(
    func=levenshtein_dist,
    name='LevenshteinDist',
    cluster_threshold=0.73)
PhoneticDist = Distance(
    func=phonetic_dist,
    name='PhoneticDist') # TODO name TBD
SegmentalDist = Distance(
    func=segmental_word_dist,
    name='SegmentalDist') # TODO name TBD
PhonologicalDist = Distance(
    func=phonological_dist,
    name='PhonologicalDist', # TODO name TBD
    cluster_threshold=0.16 # TODO cluster_threshold needs to be recalibrated; this value was from when it was a similarity function
) 
PMIDist = Distance(
    func=pmi_dist,
    name='PMIDist',
    cluster_threshold=0.36)
SurprisalDist = Distance(
    func=mutual_surprisal, 
    name='SurprisalDist',
    cluster_threshold=0.74, # TODO cluster_threshold needs to be recalibrated; this value was from when it was a similarity function
    ngram_size=1)
# Note: Hybrid distance needs to be defined in classifyLangs.py or else we can't set the parameters of the component functions based on command line args

# Z SCORE
def Z_score(p_values):
    neg_log_p = [-log(p) for p in p_values]
    return (sum(neg_log_p) - len(p_values)) / sqrt(len(p_values))

def Z_max(n_concepts):
    return ((n_concepts * -log(1/((n_concepts**2)-n_concepts+1))) - n_concepts) / sqrt(n_concepts)

def Z_min(n_concepts):
    return (n_concepts * -log(1) - n_concepts) / sqrt(n_concepts)

def Z_dist(p_values):
    N = len(p_values)
    Zmax = Z_max(N)
    Zmin = Z_min(N)
    Zscore = Z_score(p_values)
    return (Zmax - Zscore) / (Zmax - Zmin)