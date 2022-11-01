from nwunschAlign import best_alignment
from phonSim import *

#WORD-LEVEL PHONETIC COMPARISON AND ALIGNMENT
def compatible_segments(seg1, seg2):
    """Returns True if the two segments are either:
        two consonants
        two vowels
        a vowel and a sonorant consonant (nasals, liquids, glides)
        two tonemes
    Else returns False"""
    strip_seg1, strip_seg2 = map(strip_diacritics, [seg1, seg2])
    if strip_seg1[0] in consonants:
        if strip_seg2[0] in consonants:
            return True
        elif strip_seg2[0] in vowels:
            if phone_id(seg1)['sonorant'] == 1:
                return True
            else:
                return False
        else:
            return False
    elif strip_seg1[0] in vowels:
        if strip_seg2[0] in vowels:
            return True
        elif strip_seg2[0] in consonants:
            if phone_id(seg2)['sonorant'] == 1:
                return True
            else:
                return False
        else:
            return False
    #Tonemes
    else: 
        if strip_seg2[0] in tonemes:
            return True
        else:
            return False

def align_costs(seq1, seq2, 
                dist_func, sim=False, 
                **kwargs):
    alignment_costs = {}
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            seq1_i, seq2_j = seq1[i], seq2[j]
            cost = dist_func(seq1_i, seq2_j, **kwargs)
            
            #If similarity function, turn into distance and ensure it is negative
            if sim == True:
                if cost > 0:
                    cost = math.log(cost)
                else:
                    cost = -math.inf
            
            alignment_costs[(i, j)] = cost
    return alignment_costs


def phone_align(word1, word2, 
                dist_func=phone_sim, sim=True,
                gop=-0.7,
                added_penalty_dict=None,
                segmented=False,
                **kwargs):
    """Align segments of word1 with segments of word2 according to Needleman-
    Wunsch algorithm, with costs determined by phonetic and sonority similarity;
    If segmented == False, the words are first segmented before being aligned.
    GOP = -1.22 by default, determined by cross-validation on gold alignments."""
    if segmented == False:        
        segments1, segments2 = segment_word(word1), segment_word(word2)
    else:
        segments1, segments2 = word1, word2  
    
    #Combine base distances from distance function with additional penalties, if specified
    if added_penalty_dict != None:
        def added_penalty_dist(seq1, seq2, **kwargs):
            added_penalty = added_penalty_dict[seq1][seq2]
            base_dist = dist_func(seq1, seq2, **kwargs)
            #If similarity function, turn into distance and ensure it is negative
            if sim == True:
                base_dist = -(1 - base_dist)
            return base_dist + added_penalty
        
        alignment_costs = align_costs(segments1, segments2, 
                                      dist_func=added_penalty_dist, sim=False, 
                                      **kwargs)
    
    #Otherwise calculate alignment costs for each segment pair using only the base distance function
    else:
        alignment_costs = align_costs(segments1, segments2, 
                                      dist_func=phone_sim, sim=sim, 
                                      **kwargs)
    
    #Calculate best alignment using Needleman-Wunsch algorithm
    best = best_alignment(SEQUENCE_1=segments1, SEQUENCE_2=segments2,
                          SCORES_DICT=alignment_costs, GAP_SCORE=gop)
    return best


def visual_align(alignment):
    """Renders list of aligned segment pairs as an easily interpretable
    alignment string, with <∅> representing null segments,
    e.g.:
    visual_align([('z̪', 'ɡ'),('vʲ', 'v'),('ɪ', 'j'),('-', 'ˈa'),('z̪', 'z̪'),('d̪', 'd'),('ˈa', 'a')])
    = 'z̪-ɡ / vʲ-v / ɪ-j / ∅-ˈa / z̪-z̪ / d̪-d / ˈa-a' """
    a = []
    for item in alignment:
        if '-' not in item:
            a.append(f'{item[0]}-{item[1]}')
        else:
            if item[0] == '-':
                a.append(f'∅-{item[1]}')
            else:
                a.append(f'{item[0]}-∅')
    return ' / '.join(a)


def undo_visual_align(visual_alignment):
    """Reverts a visual alignment to a list of tuple segment pairs"""
    seg_pairs = visual_alignment.split(' / ')
    seg_pairs = [tuple(pair.split('-')) for pair in seg_pairs]
    return seg_pairs
    

def reverse_alignment(alignment):
    """Flips the alignment, e.g.:
        reverse_alignment([('s', 's̪'), ('o', 'ɔ'), ('l', 'l'), ('-', 'ɛ'), ('-', 'j')])
        = [('s̪', 's'), ('ɔ', 'o'), ('l', 'l'), ('ɛ', '-'), ('j', '-')]"""
    reverse = []
    for pair in alignment:
        reverse.append((pair[1], pair[0]))
    return reverse