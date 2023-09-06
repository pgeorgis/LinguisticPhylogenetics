from math import sqrt, log, exp
import re
from statistics import mean
from asjp import ipa2asjp
from nltk import edit_distance
from PhoneticSimilarity.initPhoneData import consonants, vowels, glides, nasals, palatal, alveolopalatal, postalveolar
from PhoneticSimilarity.ipaTools import strip_diacritics
from PhoneticSimilarity.segment import _toSegment
from PhoneticSimilarity.phonSim import phone_sim
from auxFuncs import Distance, sim_to_dist, strip_ch, euclidean_dist, adaptation_surprisal
from phonAlign import Alignment, get_alignment_iter


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
            correlator = lang1.get_phoneme_correlator(lang2)
            pmi_dict = correlator.calc_phoneme_pmi()
        
        # Align the phonetic sequences with phonetic similarity and phoneme PMI
        alignment = Alignment(word1, word2, added_penalty_dict=pmi_dict, **kwargs)
        
    # Perform phonetic alignment without PMI support
    else:
        alignment = Alignment(word1, word2, **kwargs)
    
    return alignment


def phonetic_dist(word1, word2=None, phone_sim_func=phone_sim, **kwargs):
    """Calculates phonetic distance of an alignment without weighting by 
    segment type, position, etc.

    Args:
        word1 (phyloLing.Word): first Word object, or an Alignment object
        word2 (phyloLing.Word): second Word object, or an Alignment object. Defaults to None.
        phone_sim_func (phone_sim, optional): Phonetic similarity function. Defaults to phone_sim.
    """
    
    #If word2 is not provided, word1 is assumed to be an alignment.""" # TODO this should not be done
    
    # Calculate or retrieve the alignment
    if word2:
        alignment = prepare_alignment(word1, word2)
    else:
        alignment = word1
    
    # Calculate the phonetic similarity of each aligned segment
    # Gap alignments receive a score of 0
    phone_sims = [phone_sim_func(pair[0], pair[1], **kwargs) 
                  if alignment.gap_ch not in pair else 0 
                  for pair in alignment.alignment]
    
    # Return the mean distance
    return 1 - mean(phone_sims)


def prosodic_environment_weight(segments, i):
    """Returns the relative prosodic environment weight of a segment within a word, based on List (2012)"""
    
    seg = _toSegment(segments[i])

    # Word-initial segments
    if i == 0:
        # Word-initial consonants: weight 7
        if seg.phone_class in ('CONSONANT', 'GLIDE'):
            return 7
        
        # Word-initial vowels: weight 6
        elif seg.phone_class in ('VOWEL', 'DIPHTHONG'):
            return 6
        
        # Word-initial tonemes # TODO is this valid?
        else:
            return 0
    
    # Word-final segments
    elif i == len(segments)-1:
        
        # Word-final consonants: weight 2
        if seg.phone_class in ('CONSONANT', 'GLIDE'):
            return 2
        
        # Word-final vowels: weight 1
        elif seg.phone_class in ('VOWEL', 'DIPHTHONG'):
            return 1
        
        # Word-final tonemes: weight 0
        else:
            return 0
    
    # Word-medial segments
    else:
        prev_segment, segment_i, next_segment = segments[i-1], segments[i], segments[i+1]
        prev_segment, segment_i, next_segment = map(_toSegment, [prev_segment, segment_i, next_segment])
        prev_sonority = prev_segment.sonority
        sonority_i = segment_i.sonority 
        next_sonority = next_segment.sonority
        
        # Sonority peak: weight 3
        if prev_sonority <= sonority_i >= next_sonority:
            return 3
        
        # Descending sonority: weight 4
        elif prev_sonority >= sonority_i >= next_sonority:
            return 4
        
        # Ascending sonority: weight 5
        else:
            return 5
        
        # TODO: what if the sonority is all the same? add tests to ensure that all of these values are correct
        # TODO: sonority of free-standing vowels (and consonants)?: would assume same as word-initial


def phonological_dist(word1, 
                      word2=None,
                      sim_func=phone_sim,
                      penalize_sonority=True,
                      max_sonority=16,
                      context_reduction=True, 
                      penalty_discount=2,
                      prosodic_env_scaling=True,
                      total_dist=False, # TODO confirm that this is the better default; I think averaging is required to normalize for different word lengths
                      **kwargs):
    """Calculates phonological distance between two words on the basis of the phonetic similarity of aligned segments and phonological deletion penalties.
    No weighting by segment type, position, etc.

    Args:
        word1 (phyloLing.Word): first Word object, or an Alignment object
        word2 (phyloLing.Word): second Word object, or an Alignment object. Defaults to None. # TODO word2=None is weird, could instead require either two words or a single alignment as input  
        sim_func (_type_, optional): Phonetic similarity function. Defaults to phone_sim.
        penalize_sonority (bool, optional): Penalizes deletions according to sonority of the deleted segment. Defaults to True.
        max_sonority (int, optional): Maximum sonority value. Defaults to 16 as defined in PhoneticSimilarity submodule.
        context_reduction (bool, optional): Reduces deletion penalties if certain phonological context conditions are met. Defaults to True.
        penalty_discount (int, optional): Value by which deletion penalties are divided if reduction conditions are met. Defaults to 2.
        prosodic_env_scaling (bool, optional): Reduces deletion penalties according to prosodic environment strength (List, 2012). Defaults to True.
        total_dist (bool, optional): Computes phonological distance as the sum of all penalties. Defaults to False.

    Returns:
        float: phonological distance value
    """
    
    # If word2 is None, we assume word1 argument is actually an aligned word pair
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
                deleted_segment = _toSegment(seg2)
                
            else:
                deleted_segment = _toSegment(seg1)
                
            if penalize_sonority:
                sonority = deleted_segment.sonority
                sonority_penalty = 1-(sonority/(max_sonority+1))
                penalty *= sonority_penalty
            
            deleted_index = pair.index(deleted_segment.segment)
            gap_index = deleted_index-1
            
            # Lessen the penalty if certain phonological conditions are met
            if context_reduction:
                if i > 0 and alignment[i-1][gap_index] != gap_ch:
                    previous_seg = _toSegment(alignment[i-1][gap_index])
                    # 1) If the deleted segment is a nasal and the corresponding 
                    # precending segment was (pre)nasalized
                    if deleted_segment.stripped in nasals:
                        if previous_seg.features['nasal'] == 1:
                            penalty /= penalty_discount
                    
                    # 2) If the deleted segment is a palatal glide (j, ɥ) or high front vowel (i, ɪ, y, ʏ),  
                    # and the corresponding preceding segment was palatalized
                    # or is a palatal, alveolopalatal, or postalveolar consonant
                    elif re.search(r'[jɥiɪyʏ]', deleted_segment.base):
                        if previous_seg.base in palatal.union(alveolopalatal).union(postalveolar):
                            penalty /= penalty_discount
                        elif re.search(r'[ʲᶣ]', previous_seg.segment):
                            penalty /= penalty_discount
                            
                    # 3) If the deleted segment is a high rounded/labial glide
                    # and the corresponding preceding segment was labialized
                    elif re.search(r'[wʋuʊyʏ]', deleted_segment.base):
                        if re.search(r'[ʷᶣ]', previous_seg.segment):
                            penalty /= penalty_discount
                    
                    # 4) If the deleted segment is /h, ɦ/ and the corresponding 
                    # preceding segment was aspirated or breathy
                    elif re.search(r'[hɦ]', deleted_segment.base):
                        if re.search(r'[ʰʱ̤]', previous_seg.segment):
                            penalty /= penalty_discount
                    
                        # Or if the following corresponding segment is breathy or pre-aspirated
                        else:
                            try:
                                if re.search(r'[ʰʱ̤]', alignment[i+1][gap_index]):
                                    penalty /= penalty_discount
                            except IndexError:
                                pass
                        
                    # 5) If the deleted segment is a rhotic approximant /ɹ, ɻ/
                    # and the corresponding preceding segment was rhoticized
                    elif re.search(r'[ɹɻ]', deleted_segment.base):
                        if re.search(r'[ɚɝ˞]', previous_seg.segment):
                            penalty /= penalty_discount
                    
                    # 6) If the deleted segment is a glottal stop and the corresponding
                    # preceding segment was glottalized or creaky
                    elif deleted_segment.base == 'ʔ':
                        if re.search(r'[ˀ̰]', previous_seg.segment):
                            penalty /= penalty_discount
                    
                    
                # 7) If the deleted segment is part of a long/geminate segment transcribed as double (e.g. /tt/ rather than /tː/), 
                # where at least one part of the geminate has been aligned
                # Method: check if the preceding or following pair contained the deleted segment at deleted_index, aligned to something other than the gap character
                # Check following pair
                double = False
                try:
                    nxt_pair = alignment[i+1]
                    if gap_ch not in nxt_pair and nxt_pair[deleted_index] == deleted_segment:
                        double = True
                        
                        # Eliminate the penalty altogether if the length is simply transcribed with a diacritic
                        if re.search(r'[ːˑ]', nxt_pair[gap_index]):
                            penalty = 0
                        
                except IndexError:
                    pass
                
                # Check preceding pair
                if i > 0:
                    prev_pair = alignment[i-1]
                    if gap_ch not in prev_pair and prev_pair[deleted_index] == deleted_segment:
                        double = True
                    
                        # Eliminate the penalty altogether in the case of 
                        # an alignment like: [('t', 'tː'), ('t', '-')]
                        # where the length/gemination is simply transcribed differently
                        if re.search(r'[ːˑ]', prev_pair[gap_index]):
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
    
    if total_dist:
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
        pmi_dict = lang1.get_phoneme_correlator(lang2).calc_phoneme_pmi(**kwargs)

    # Calculate phoneme surprisal if not already done # TODO use helper function
    if len(lang1.phoneme_surprisal[(lang2, ngram_size)]) == 0:
        correlator1 = lang1.get_phoneme_correlator(lang2)
        # TODO add a logging message so that we know surprisal is being calculated (again) -- maybe best within phonCorr.py
        correlator1.calc_phoneme_surprisal(ngram_size=ngram_size, **kwargs)
    if len(lang2.phoneme_surprisal[(lang1, ngram_size)]) == 0:
        correlator2 = lang2.get_phoneme_correlator(lang1)
        correlator2.calc_phoneme_surprisal(ngram_size=ngram_size, **kwargs)
    
    # Generate alignments in each direction: alignments need to come from PMI
    alignment = Alignment(word1, word2, added_penalty_dict=pmi_dict, phon_env=phon_env)
    forward_alignment = get_alignment_iter(alignment, phon_env=phon_env)
    rev_alignment = alignment._reverse()
    backward_alignment = get_alignment_iter(rev_alignment, phon_env=phon_env)

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
        # the way to get around this is:
        # calculate the 2gram, then get the 2gram's phon_env equivalent
        # interpolate the probability/surprisal of the 2gram with that of the phon_env equivalent 
        raise NotImplementedError
    WAS_l2l1 = adaptation_surprisal(backward_alignment, 
                                    surprisal_dict=sur_dict2,
                                    ngram_size=ngram_size,
                                    normalize=False)

    # Calculate self-surprisal values in each direction
    self_surprisal1 = lang1.self_surprisal(word1, normalize=False)
    self_surprisal2 = lang2.self_surprisal(word2, normalize=False)

    # Weight surprisal values by self-surprisal/information content value of corresponding segment
    # Segments with greater information content weighted more heavily
    # Normalize by phoneme entropy
    def weight_by_self_surprisal(alignment, WAS, self_surprisal, normalize_by):
        self_info = sum([self_surprisal[j][-1] for j in self_surprisal])
        weighted_WAS = []
        seq_map1 = alignment.seq_map[0]
        for i, pair in enumerate(alignment.alignment):
            if seq_map1[i] is not None:
                weight = self_surprisal[seq_map1[i]][-1] / self_info
                normalized = WAS[i] / normalize_by
                weighted = weight * normalized
                weighted_WAS.append(weighted)
        return weighted_WAS
    weighted_WAS_l1l2 = weight_by_self_surprisal(alignment, WAS_l1l2, self_surprisal1, normalize_by=lang2.phoneme_entropy)
    weighted_WAS_l2l1 = weight_by_self_surprisal(rev_alignment, WAS_l2l1, self_surprisal2, normalize_by=lang1.phoneme_entropy)
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
        correlator = lang1.get_phoneme_correlator(lang2)
        pmi_dict = correlator.calc_phoneme_pmi(**kwargs)
        
    # Align the words with PMI
    alignment = Alignment(word1, word2, added_penalty_dict=pmi_dict)
    
    # Calculate PMI scores for each aligned pair
    PMI_values = [pmi_dict[pair[0]][pair[1]] for pair in alignment.alignment]

    # Weight by information content per segment
    def weight_by_info_content(alignment, PMI_vals):
        word1, word2 = alignment.word1, alignment.word2
        info_content1 = word1.getInfoContent()
        info_content2 = word2.getInfoContent()
        total_info1 = sum([info_content1[j][-1] for j in info_content1])
        total_info2 = sum([info_content2[j][-1] for j in info_content2])
        seq_map1, seq_map2 = alignment.seq_map
        weighted_PMI = []
        for i, pair in enumerate(alignment.alignment):
            # Take the information content value of each segment within the respective word
            # Divide this by the total info content of the word to calculate the proportion of info content constituted by the segment
            if seq_map1[i] is not None:
                weight1 = info_content1[seq_map1[i]][-1] / total_info1
            else:
                weight1 = None
            
            if seq_map2[i] is not None:
                weight2 = info_content2[seq_map2[i]][-1] / total_info2
            else:
                weight2 = None
            
            # Average together the info contents of each aligned segment
            if weight1 is None:
                weight = weight2
            elif weight2 is None:
                weight = weight1
            else:
                weight = mean([weight1, weight2])

            # Weight by the averaged values
            weighted = weight * PMI_vals[i]
            weighted_PMI.append(weighted)

        return weighted_PMI

    PMI_values = weight_by_info_content(alignment, PMI_values)
    PMI_score = mean(PMI_values) 
    
    if sim2dist:
        #PMI_dist = exp(-max(PMI_score, 0)**alpha)
        return sim_to_dist(PMI_score, alpha)
    
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