from collections import defaultdict
from math import sqrt, log, e, exp
from statistics import mean
from asjp import ipa2asjp
from nltk import edit_distance
import re
from phonSim.phonSim import consonants, vowels, glides, nasals, palatal, suprasegmental_diacritics, strip_diacritics
from phonSim.phonSim import phone_sim, get_sonority, max_sonority, prosodic_environment_weight
from auxFuncs import strip_ch, euclidean_dist, surprisal, adaptation_surprisal
from phonAlign import phone_align, reverse_alignment
from phonCorr import PhonemeCorrDetector

def prepare_alignment(item1, item2, **kwargs):
    """Prepares alignment of two items, either:
        ("word1", Lang1) : tuples of an IPA string and a Language class object
        
        or
        
        "word1" : IPA strings
        
    If the language class objects are provided, phoneme PMI is additionally used
    for the alignment, otherwise only phonetic similarity.
    
    If phonetic PMI has not been previously calculated, it will be calculated
    automatically here."""
    
    # Check the structure of the input, whether tuples or strings
    if (type(item1) == tuple and type(item2) == tuple):
        word1, lang1 = item1
        word2, lang2 = item2
    
    else:
        word1, word2 = item1, item2
        lang1, lang2 = None, None
        
    # If language input has been given, incorporate their phoneme PMI for the alignment
    if (lang1, lang2) != (None, None):
        
        # Check whether phoneme PMI has been calculated for this language pair
        # If not, then calculate it; if so, then retrieve it
        if len(lang1.phoneme_pmi[lang2]) > 0:
            pmi_dict = lang1.phoneme_pmi[lang2]
        else:
            pmi_dict = PhonemeCorrDetector(lang1, lang2).calc_phoneme_pmi()
        
        # Align the phonetic sequences with phonetic similarity and phoneme PMI
        alignment = phone_align(word1, word2, added_penalty_dict=pmi_dict, **kwargs)
        
    # Perform phonetic alignment without PMI support
    else:
        alignment = phone_align(word1, word2, **kwargs)
    
    return alignment


def basic_word_sim(word1, word2=None, sim_func=phone_sim, **kwargs):
    """Calculates phonetic similarity of an alignment without weighting by 
    segment type, position, etc.
    
    Input:
    ("word1", Lang1) : tuples of an IPA string and a Language class object
        
    or
        
    "word1" : IPA strings
    
    If word2 is not provided, word1 is assumed to be an alignment."""
    
    # Calculate or retrieve the alignment
    if word2:
        alignment = prepare_alignment(word1, word2)
    else:
        alignment = word1
    
    # Calculate the phonetic similarity of each aligned segment
    # Gap alignments receive a score of 0
    phone_sims = [sim_func(pair[0], pair[1], **kwargs) 
                  if '-' not in pair else 0 
                  for pair in alignment]
    
    # Return the mean similarity
    return mean(phone_sims)


calculated_word_sims = {}
def word_sim(word1, word2=None, 
              sim_func=phone_sim,
              penalize_infocontent=False, 
              penalize_sonority=True,
              context_reduction=True, penalty_discount=2,
              prosodic_env_scaling=True,
              total_sim=False,
              **kwargs):
    """Calculates phonetic similarity of an alignment without weighting by 
    segment type, position, etc.
    
    word1 : string (first word), list (alignment of two words), 
            or tuple (first word, second language)
    word2 : string (second word) or tuple (second word, second language)"""
    

    # If word2 is None, we assume word1 argument is actually an aligned word pair
    # Otherwise, align the two words
    if word2:
        alignment = prepare_alignment(word1, word2)
    else:
        alignment = word1
    
    if (tuple(alignment), sim_func, penalize_sonority, 
       context_reduction, prosodic_env_scaling, total_sim) in calculated_word_sims: 
        return calculated_word_sims[(tuple(alignment), sim_func, penalize_sonority, 
                                     context_reduction, prosodic_env_scaling, total_sim)] 
    else:
        # Get list of penalties
        penalties = []
        for i in range(len(alignment)):
            pair = alignment[i]
            seg1, seg2 = pair
            
            # If the pair is a gap-aligned segment, assign the penalty 
            # based on the sonority and information content (if available) of the deleted segment
            if '-' in pair:
                penalty = 1
                if seg1 == '-':
                    deleted_segment = seg2
                    index = len([alignment[j][1] for j in range(i) if alignment[j][1] != '-'])
                    
                else:
                    deleted_segment = seg1
                    index = len([alignment[j][0] for j in range(i) if alignment[j][0] != '-'])
                
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
                        if '-' not in nxt_pair:
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
                        if '-' not in prev_pair:
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
                    deleted_i = sum([1 for j in range(i+1) if alignment[j][deleted_index] != '-'])-1
                    segment_list = [alignment[j][deleted_index] 
                                    for j in range(len(alignment))
                                    if alignment[j][deleted_index] != '-']
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
            word_dist = mean(penalties)
        
        # Return as similarity: e**-distance = 1/(e**distance)
        word_sim = e**-word_dist
        
        # Save the calculated score
        if word2:
            calculated_word_sims[(tuple(alignment), sim_func, 
                                 penalize_sonority, 
                                 context_reduction, prosodic_env_scaling, 
                                 total_sim)] = word_sim
            
        return word_sim


def segmental_word_sim(word1, word2=None, 
                       c_weight=0.5, v_weight=0.3, syl_weight=0.2):
    """Calculates the phonetic similarity of an aligned word pair according to
    weighted average similarity of consonantal segments, vocalic segments, and 
    syllable structure"""
    
    assert round(sum([c_weight, v_weight, syl_weight]),1) == 1.0
    
    # If word2 is None, we assume word1 argument is actually an aligned word pair
    # Otherwise, align the two words
    if word2:
        alignment = prepare_alignment(word1, word2)
    else:
        alignment = word1
    
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
    syl_score = 1 - (edit_distance(syl_structure1, syl_structure2) / len(alignment))
    
    # Final score: weighted sum of each component score
    return (c_weight * c_score) + (v_weight * v_score) + (syl_weight * syl_score)


combined_surprisal_dicts = {}
scored_WAS = {}
def mutual_surprisal(pair1, pair2, ngram_size=1, **kwargs):
    if (pair1, pair2, ngram_size) in scored_WAS:
        return scored_WAS[(pair1, pair2, ngram_size)]
    
    else:
        word1, lang1 = pair1
        word2, lang2 = pair2
        
        # Remove suprasegmental and other ignored characters
        diacritics_to_remove = lang1.ch_to_remove.union(lang2.ch_to_remove)
        word1 = strip_ch(word1, diacritics_to_remove)
        word2 = strip_ch(word2, diacritics_to_remove)
        
        # Calculate combined phoneme PMI if not already done
        # pmi_dict = combine_PMI(lang1, lang2, **kwargs)
        
        # Check whether phoneme PMI has been calculated for this language pair
        # Otherwise calculate from scratch
        if len(lang1.phoneme_pmi[lang2]) > 0:
            pmi_dict = lang1.phoneme_pmi[lang2]
        else:
            pmi_dict = PhonemeCorrDetector(lang1, lang2).calc_phoneme_pmi(**kwargs)
        
        # Generate alignments in each direction: alignments need to come from PMI
        alignment = phone_align(word1, word2, added_penalty_dict=pmi_dict)
        
        # Calculate phoneme surprisal if not already done
        if len(lang1.phoneme_surprisal[(lang2, ngram_size)]) == 0:
            surprisal_dict_l1l2 = PhonemeCorrDetector(lang1, lang2).calc_phoneme_surprisal(ngram_size=ngram_size, **kwargs)
        if len(lang2.phoneme_surprisal[(lang1, ngram_size)]) == 0:
            surprisal_dict_l2l1 = PhonemeCorrDetector(lang2, lang1).calc_phoneme_surprisal(ngram_size=ngram_size, **kwargs)
            
        # Calculate the word-adaptation surprisal in each direction
        # (note: alignment needs to be reversed to run in second direction)
        WAS_l1l2 = adaptation_surprisal(alignment, 
                                        surprisal_dict=lang1.phoneme_surprisal[(lang2, ngram_size)],
                                        ngram_size=ngram_size,
                                        normalize=False)
        WAS_l2l1 = adaptation_surprisal(reverse_alignment(alignment), 
                                        surprisal_dict=lang2.phoneme_surprisal[(lang1, ngram_size)],
                                        ngram_size=ngram_size,
                                        normalize=False)

        # Calculate self-surprisal values in each direction
        self_surprisal1 = lang1.self_surprisal(word1, segmented=False, normalize=False) 
        self_surprisal2 = lang2.self_surprisal(word2, segmented=False, normalize=False) 
        
        # Divide WAS by self-surprisal
        WAS_l1l2 /= self_surprisal2
        WAS_l2l1 /= self_surprisal1
        
        # Return and save the average of these two values
        mean_WAS = mean([WAS_l1l2, WAS_l2l1])
        scored_WAS[(pair1, pair2, ngram_size)] = mean_WAS
        return mean_WAS

surprisal_sims = {}
def surprisal_sim(pair1, pair2, ngram_size=1, **kwargs):
    try:
        return surprisal_sims[(pair1, pair2, ngram_size)] 
    except KeyError:
        score = e**-(mutual_surprisal(pair1, pair2, ngram_size=ngram_size, **kwargs))
        surprisal_sims[(pair1, pair2, ngram_size)] = score
        return score


def phonetic_surprisal(pair1, pair2, surprisal_dict=None, normalize=True, ngram_size=1):
    lang1, lang2 = pair1[1], pair2[1]
    alignment = prepare_alignment(pair1, pair2)
    
    # If no surprisal dictionary is specified, use the standard L1-->L2/L2-->L1 phoneme surprisal
    if surprisal_dict is None:
        assert lang1 and lang2
        
        # Calculate phoneme surprisal if not already done
        if len(lang1.phoneme_surprisal[(lang2, ngram_size)]) == 0:
            surprisal_dict_l1l2 = PhonemeCorrDetector(lang1, lang2).calc_phoneme_surprisal(ngram_size=ngram_size)
        if len(lang2.phoneme_surprisal[(lang1, ngram_size)]) == 0:
            surprisal_dict_l2l1 = PhonemeCorrDetector(lang2, lang1).calc_phoneme_surprisal(ngram_size=ngram_size)
        
        # Take surprisal value as average of surprisal from each direction
        surprisal_values = [mean([lang1.phoneme_surprisal[(lang2, ngram_size)][tuple(pair[0].split())][pair[1]],
                                  lang2.phoneme_surprisal[(lang1, ngram_size)][tuple(pair[1].split())][pair[0]]]) 
                            for pair in alignment]
        
    else:
        surprisal_values = [surprisal_dict[pair[0]][pair[1]] for pair in alignment]        
    
    # Calculate phonetic distance of aligned segments, with distance to gap = 1
    phon_dists = [1 - phone_sim(pair[0], pair[1]) if '-' not in pair else 0 for pair in alignment]
    
    # Calculate phonetic surprisal as phonetic distance * surprisal
    phon_surprisal = [phon_dists[i]*surprisal_values[i] for i in range(len(alignment))]
    
    return e**-sum(phon_surprisal)
    

combined_PMI_dicts = {}
def combine_PMI(lang1, lang2, **kwargs):
    # Return already calculated dictionary if possible
    if (lang1, lang2) in combined_PMI_dicts:
        return combined_PMI_dicts[(lang1, lang2)]

    # Otherwise calculate from scratch
    if len(lang1.phoneme_pmi[lang2]) > 0:
        pmi_dict_l1l2 = lang1.phoneme_pmi[lang2]
    else:
        pmi_dict_l1l2 = PhonemeCorrDetector(lang1, lang2).calc_phoneme_pmi(**kwargs)
    if len(lang2.phoneme_pmi[lang1]) > 0:
        pmi_dict_l2l1 = lang2.phoneme_pmi[lang1]
    else:
        pmi_dict_l2l1 = PhonemeCorrDetector(lang2, lang1).calc_phoneme_pmi(**kwargs)
        
    # Average together the PMI values from each direction
    pmi_dict = defaultdict(lambda:defaultdict(lambda:0))
    for seg1 in pmi_dict_l1l2:
        for seg2 in pmi_dict_l1l2[seg1]:
            pmi_dict[seg1][seg2] = mean([pmi_dict_l1l2[seg1][seg2], pmi_dict_l2l1[seg2][seg1]])
    combined_PMI_dicts[(lang1, lang2)] = pmi_dict
    combined_PMI_dicts[(lang2, lang1)] = pmi_dict
    return pmi_dict



scored_word_pmi = {}
def pmi_dist(pair1, pair2, sim2dist=True, alpha=0.5, **kwargs):
    if (pair1, pair2, sim2dist) in scored_word_pmi:
        return scored_word_pmi[(pair1, pair2, sim2dist)]
    
    else:
        word1, lang1 = pair1
        word2, lang2 = pair2
        
        # Remove suprasegmental diacritics
        diacritics_to_remove = list(suprasegmental_diacritics) + ['̩', '̍', ' ']
        word1 = strip_ch(word1, diacritics_to_remove)
        word2 = strip_ch(word2, diacritics_to_remove)
        
        # Calculate PMI in both directions if not already done, otherwise retrieve the dictionaries
        # pmi_dict = combine_PMI(lang1, lang2, **kwargs)
        
        # Check whether phoneme PMI has been calculated for this language pair
        # Otherwise calculate from scratch
        if len(lang1.phoneme_pmi[lang2]) > 0:
            pmi_dict = lang1.phoneme_pmi[lang2]
        else:
            pmi_dict = PhonemeCorrDetector(lang1, lang2).calc_phoneme_pmi(**kwargs)
            
        # Align the words with PMI
        alignment = phone_align(word1, word2, added_penalty_dict=pmi_dict)
        
        # Calculate PMI scores for each aligned pair
        PMI_values = [pmi_dict[pair[0]][pair[1]] for pair in alignment]
        PMI_score = mean(PMI_values) 
        
        if sim2dist:
            PMI_dist = exp(-max(PMI_score, 0)**alpha)
            scored_word_pmi[(pair1, pair2, sim2dist)] = PMI_dist
            return PMI_dist
        
        else:
            scored_word_pmi[(pair1, pair2, sim2dist)] = PMI_score
            return PMI_score

def LevenshteinDist(word1, word2, normalize=True, asjp=True):
    if type(word1) == tuple:
        word1 = word1[0]
    if type(word2) == tuple:
        word2 = word2[0]
        
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
        
hybrid_scores = {}
def hybrid_dist(pair1:tuple, pair2:tuple, funcs:dict, func_sims)->float:
    """Uses the euclidean distance to calculate the hybrid distance of multiple distance or similarity functions

    Args:
        pair1 (tuple): (word, Language) pair
        pair2 (tuple): (word, Language) pair
        funcs (dict): dictionary of functions with their kwargs {function:{arg:val}}
        func_sims (iterable): list of Boolean values representing whether each function is a similarity function

    Returns:
        float: hybrid similarity measure
    """
    # Try to retrieve previously calculated value if possible
    if (pair1, pair2, tuple(funcs.keys())) in hybrid_scores:
        return hybrid_scores[(pair1, pair2, tuple(funcs.keys()))]
    
    scores = []
    for func, func_sim in zip(funcs, func_sims):
        kwargs = funcs[func]
        score = func(pair1, pair2, **kwargs)
        if func_sim:
            score = 1 - score
        scores.append(score)
    
    score = euclidean_dist(scores)
    hybrid_scores[(pair1, pair2, tuple(funcs.keys()))] = score
    
    return score
        
def hybrid_sim(pair1, pair2, funcs, func_sims, **kwargs):
    hybrid_d = hybrid_dist(pair1, pair2, funcs=funcs, func_sims=func_sims)
    hybrid_sim = e**-(hybrid_d)
    return hybrid_sim
    
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
    
    
