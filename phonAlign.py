from math import log, inf
import re
from itertools import combinations
from nwunschAlign import best_alignment
from phonSim.phonSim import consonants, vowels, tonemes, phone_id, strip_diacritics, segment_ipa, phone_sim
from phonSim.phonSim import phonEnvironment #, prosodic_environment_weight
from auxFuncs import Distance

AlignmentPhoneSim = Distance(
    func=phone_sim,
    sim=True,
    name='AlignmentPhoneSim'
)

class Alignment:
    def __init__(self, 
                 seq1, seq2, 
                 cost_func=AlignmentPhoneSim, 
                 added_penalty_dict=None,
                 gap_ch='-',
                 gop=-0.7,
                 #segmented=False,
                 **kwargs
                 ):
        if type(cost_func) is not Distance:
            raise TypeError(f'Expected cost_func to be a Distance class object, was {type(cost_func)}')
        
        # TODO use Word class objects so we can access word.segments in order to have language-specific segmentation of diphthongs
        # if we use Word class objects, we can more easily/accurately distinguish between segmented and non-segmented sequences
        segmented1 = type(seq1) != str
        segmented2 = type(seq2) != str
        assert segmented1 == segmented2
        segmented = segmented1

        self.seq1 = seq1 if segmented else segment_ipa(seq1)
        self.seq2 = seq2 if segmented else segment_ipa(seq2)
        self.word1 = seq1 if not segmented else ''.join(seq1)
        self.word2 = seq1 if not segmented else ''.join(seq2)
        self.gap_ch = gap_ch
        self.gop = gop
        self.cost_func = cost_func
        self.added_penalty_dict = added_penalty_dict
        self.kwargs = kwargs
        self.alignment = self.align()
        self.length = len(self.alignment)

    def calculate_alignment_costs(self, cost_func):
        """Calculates pairwise alignment costs for phone sequences using a specified cost function.

        Args:
            cost_func (Distance): cost function used for computing pairwise alignment costs 

        Returns:
            dict: dictionary of pairwise alignment costs by sequence indices
        """
        alignment_costs = {}
        for i, seq1_i in enumerate(self.seq1):
            for j, seq2_j in enumerate(self.seq2):
                cost = cost_func.eval(seq1_i, seq2_j, **self.kwargs)
                
                # If similarity function, turn into distance and ensure it is negative # TODO add into Distance object
                if cost_func.sim:
                    if cost > 0:
                        cost = log(cost)
                    else:
                        cost = -inf

                alignment_costs[(i, j)] = cost

        return alignment_costs


    def align(self):
        """Align segments of word1 with segments of word2 according to Needleman-
        Wunsch algorithm, with costs determined by phonetic and sonority similarity;
        If not segmented, the words are first segmented before being aligned.
        GOP = -0.7 by default, determined by cross-validation on dataset of gold cognate alignments."""
        
        # Combine base distances from distance function with additional penalties, if specified
        if self.added_penalty_dict: # TODO this could be a separate class method
            def added_penalty_dist(seq1, seq2, **kwargs):
                added_penalty = self.added_penalty_dict[seq1][seq2]
                base_dist = self.cost_func.eval(seq1, seq2, **kwargs)
                # If similarity function, turn into distance and ensure it is negative # TODO add into Distance object
                if self.cost_func.sim:
                    base_dist = -(1 - base_dist)
                return base_dist + added_penalty
            
            AddedPenaltyDist = Distance(func=added_penalty_dist, **self.kwargs)
            alignment_costs = self.calculate_alignment_costs(AddedPenaltyDist)
        
        # Otherwise calculate alignment costs for each segment pair using only the base distance function
        else:
            alignment_costs = self.calculate_alignment_costs(self.cost_func)
        
        # Calculate least costly alignment using Needleman-Wunsch algorithm
        best = best_alignment(SEQUENCE_1=self.seq1, 
                              SEQUENCE_2=self.seq2,
                              SCORES_DICT=alignment_costs, 
                              GAP_SCORE=self.gop)
        
        return best # TODO add ability to return best N alignments


    def reverse_alignment(self):
        """Flips the alignment, e.g.:
            reverse_alignment([('s', 's̪'), ('o', 'ɔ'), ('l', 'l'), ('-', 'ɛ'), ('-', 'j')])
            = [('s̪', 's'), ('ɔ', 'o'), ('l', 'l'), ('ɛ', '-'), ('j', '-')]"""
        reverse = []
        for pair in self.alignment:
            reverse.append((pair[1], pair[0]))
        return reverse


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
    # Tonemes
    else: 
        if strip_seg2[0] in tonemes:
            return True
        else:
            return False


# TODO: should the other functions below also be incorporated into Alignment class methods?
def visual_align(alignment, gap_ch='-'):
    """Renders list of aligned segment pairs as an easily interpretable
    alignment string, with <∅> representing null segments,
    e.g.:
    visual_align([('z̪', 'ɡ'),('vʲ', 'v'),('ɪ', 'j'),('-', 'ˈa'),('z̪', 'z̪'),('d̪', 'd'),('ˈa', 'a')])
    = 'z̪-ɡ / vʲ-v / ɪ-j / ∅-ˈa / z̪-z̪ / d̪-d / ˈa-a' """
    a = []
    for item in alignment:
        if gap_ch not in item:
            a.append(f'{item[0]}-{item[1]}')
        else:
            if item[0] == gap_ch:
                a.append(f'∅-{item[1]}')
            else:
                a.append(f'{item[0]}-∅')
    return ' / '.join(a)


def undo_visual_align(visual_alignment, gap_ch='-'):
    """Reverts a visual alignment to a list of tuple segment pairs"""
    seg_pairs = visual_alignment.split(' / ')
    seg_pairs = [tuple(pair.split(gap_ch)) for pair in seg_pairs]
    return seg_pairs


def phon_env_alignment(alignment, word2=False, env_func=phonEnvironment, gap_ch='-'):
    """Adds the phonological environment value of segments to an alignment
    e.g. phon_env_alignment([('m', 'm'), ('j', '-'), ('ɔu̯', 'ɪ'), ('l̥', 'l'), ('k', 'ç')])
    = [(('m', '#S<'), 'm'), (('j', '<S<'), '-'), (('ɔu̯', '<S>'), 'ɪ'), (('l̥', '>S>'), 'l'), (('k', '>S#'), 'ç')]
    """
    word1_aligned, word2_aligned = tuple(zip(*alignment))
    word1_aligned = list(word1_aligned)
    segs1 = tuple([seg for seg in word1_aligned if seg != gap_ch])
    if word2:
        segs2 = tuple([seg for seg in word2_aligned if seg != gap_ch])
    gap_count1, gap_count2 = 0, 0

    def add_phon_env(word_aligned, segs, i, gap_count, gap_ch=gap_ch):
        if word_aligned[i] == gap_ch:
            gap_count += 1
            # TODO so gaps are skipped?
        
        else:
            seg_index = i - gap_count
            phonEnv = env_func(segs, seg_index)
            word_aligned[i] = word_aligned[i], phonEnv
        
        return word1_aligned, gap_count

    for i in range(len(word1_aligned)):
        word1_aligned, gap_count1 = add_phon_env(word1_aligned, segs1, i, gap_count1)
        
        # Do the same for word2 (not done by default)
        if word2:
            word2_aligned, gap_count2 = add_phon_env(word2_aligned, segs2, i, gap_count2)
    
    return zip(word1_aligned, word2_aligned)

def phon_env_ngrams(phonEnv):
    """Returns set of phonological environment strings of equal and lower order, 
    e.g. ">S#" -> ">S", "S#", ">S#"

    Args:
        phonEnv (str): Phonological environment string, e.g. ">S#"

    Returns:
        set: possible equal and lower order phonological environment strings
    """
    assert re.search(r'.+S.+', phonEnv)
    prefix = set(re.findall(r'[^S](?=.*S)', phonEnv))
    prefixes = set()
    for i in range(1, len(prefix)+1):
        for x in combinations(prefix, i):
            prefixes.add(''.join(x))
    prefixes.add('')
    suffix = set(re.search(r'(?<=S).+', phonEnv).group())
    suffixes = set()
    for i in range(1, len(suffix)+1):
        for x in combinations(suffix, i):
            suffixes.add(''.join(x))
    suffixes.add('')
    ngrams = set()
    for prefix in prefixes:
        for suffix in suffixes:
            ngrams.add(f'{prefix}S{suffix}')
    return ngrams