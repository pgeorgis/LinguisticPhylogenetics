from math import log, inf
from collections.abc import Iterable
from nwunschAlign import best_alignment
from PhoneticSimilarity.segment import _toSegment
from PhoneticSimilarity.phonSim import phone_sim
from PhoneticSimilarity.phonEnv import get_phon_env
from auxFuncs import Distance, validate_class
import phyloLing # need Language and Word classes from phyloLing.py but cannot import them directly here because it will cause circular imports

AlignmentPhoneSim = Distance(
    func=phone_sim,
    sim=True,
    name='AlignmentPhoneSim'
)

class Alignment:
    def __init__(self, 
                 seq1, seq2,
                 lang1=None, 
                 lang2=None,
                 cost_func=AlignmentPhoneSim, 
                 added_penalty_dict=None,
                 gap_ch='-',
                 gop=-0.7, # TODO possibly need to recalibrate
                 n_best=1,
                 phon_env=False,
                 **kwargs
                 ):
        """Produces a pairwise alignment of two phone sequences. 

        Args:
            seq1 (phyloLing.Word or str): first phone sequence
            seq2 (phyloLing.Word or str): second phone sequence
            lang1 (phyloLing.Language, optional): Language of seq1. Defaults to None.
            lang2 (phyloLing.Language, optional): Language of seq2. Defaults to None.
            cost_func (Distance, optional): Cost function used for minimizing overall alignment cost. Defaults to AlignmentPhoneSim.
            added_penalty_dict (dict, optional): Dictionary of additional penalties to combine with cost_func. Defaults to None.
            gap_ch (str, optional): Gap character. Defaults to '-'.
            gop (float, optional): Gap opening penalty. Defaults to -0.7.
            n_best (int, optional): Number of best (least costly) alignments to return. Defaults to 1.
            phon_env (Bool, optional): Adds phonological environment to alignment. Defaults to False.
        """
        
        # Verify that input arguments are of the correct types
        self.validate_args(seq1, seq2, lang1, lang2, cost_func)
    
        # Prepare the input sequences for alignment
        self.seq1, self.word1 = self.prepare_seq(seq1, lang1)
        self.seq2, self.word2 = self.prepare_seq(seq2, lang2)

        # Designate alignment parameters
        self.gap_ch = gap_ch
        self.gop = gop
        self.cost_func = cost_func
        self.added_penalty_dict = added_penalty_dict
        self.kwargs = kwargs

        # Perform alignment
        self.n_best = self.align(n_best)
        self.alignment = self.n_best[0][0]

        # Map aligned pairs to respective sequence indices
        self.seq_map = self.map_to_seqs()

        # Save length and cost of single best alignment
        self.cost = self.n_best[0][-1]
        self.length = len(self.alignment)
        
        # Phonological environment alignment
        self.phon_env = phon_env
        if self.phon_env:
            self.phon_env_alignment = self._add_phon_env()
        else:
            self.phon_env_alignment = None


    def validate_args(self, seq1, seq2, lang1, lang2, cost_func):
        """Verifies that all input arguments are of the correct types"""

        validate_class((cost_func,), (Distance,))
        validate_class((seq1,), ((phyloLing.Word, str),))
        validate_class((seq2,), ((phyloLing.Word, str),))
        for lang in (lang1, lang2):
            if lang: # skip if None
                validate_class((lang,), (phyloLing.Language,))


    def prepare_seq(self, seq, lang):
        if isinstance(seq, phyloLing.Word):
            word1 = seq
        elif isinstance(seq, str):
            word1 = phyloLing.Word(seq, language=lang)
        
        return word1.segments, word1


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


    def align(self, n_best=1):
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
        
        # Calculate least costly N best alignment(s) using Needleman-Wunsch algorithm
        best = best_alignment(SEQUENCE_1=self.seq1, 
                              SEQUENCE_2=self.seq2,
                              SCORES_DICT=alignment_costs, 
                              GAP_SCORE=self.gop,
                              N_BEST=n_best)
        
        return best
    

    def remove_gaps(self, alignment=None):
        """Returns the alignment without gap-aligned positions.

        Returns:
            list: gap-free alignment list # TODO change to tuple later if possible
        """
        if alignment is None:
            alignment = self.alignment
        return [pair for pair in alignment if self.gap_ch not in pair] 
    

    def pad(self, ngram_size, alignment=None):
        if alignment is None:
            alignment = self.alignment
        pad_n = max(0, ngram_size-1)
        return [('# ', '# ')]*pad_n + alignment + [('# ', '# ')]*pad_n
    

    def map_to_seqs(self):
        """Maps aligned pair indices to their respective sequence indices
        e.g.
        self.alignment = [('ʃ', 's'), ('ˈa', 'ˈɛ'), ('p', '-'), ('t', 't'), ('e', '-')]
        map1 = {0:0, 1:1, 2:2, 3:3, 4:4}
        map2 = {0:0, 1:1, 2:None, 3:3, 4:None}
        """
        map1, map2 = {}, {}
        adjust1, adjust2 = 0, 0
        for i, pair in enumerate(self.alignment):
            seg1, seg2 = pair
            if seg1 == self.gap_ch:
                map1[i] = None
                adjust1 += 1
            else:
                map1[i] = i-adjust1
            if seg2 == self.gap_ch:
                map2[i] = None
                adjust2 += 1
            else:
                map2[i] = i-adjust2

        return map1, map2


    def _add_phon_env(self, env_func=get_phon_env):
        self.phon_env = True
        return add_phon_env(self.alignment,
                            env_func=env_func, 
                            gap_ch=self.gap_ch,
                            segs1=self.seq1)
    

    def _reverse(self):
        return ReversedAlignment(self)
        

class ReversedAlignment(Alignment):
    def __init__(self, alignment):
        validate_class((alignment,), (Alignment,))
        self.seq1 = alignment.seq2
        self.seq2 = alignment.seq1
        self.word1 = alignment.word2
        self.word2 = alignment.word1
        self.gap_ch = alignment.gap_ch
        self.gop = alignment.gop
        self.cost_func = alignment.cost_func
        self.added_penalty_dict = alignment.added_penalty_dict
        self.kwargs = alignment.kwargs
        self.n_best = [(reverse_alignment(alignment_n), cost) for alignment_n, cost in alignment.n_best]
        self.alignment = self.n_best[0][0]

        # Map aligned pairs to respective sequence indices
        self.seq_map = tuple(reversed(alignment.seq_map))

        # Save length and cost of single best alignment
        self.cost = self.n_best[0][-1]
        self.length = len(self.alignment)
        
        # Phonological environment alignment
        self.phon_env = alignment.phon_env
        if self.phon_env:
            self.phon_env_alignment = super()._add_phon_env()
        else:
            self.phon_env_alignment = None



def compatible_segments(seg1, seg2):
    """Determines whether a pair of segments are compatible for alignment. 
    Returns True if the two segments are either:
        two consonants
        two vowels
        a vowel and a sonorant consonant (nasals, liquids, glides)
        two tonemes
    Else returns False"""
    seg1, seg2 = map(_toSegment, [seg1, seg2])
    phone_class1, phone_class2 = seg1.phone_class, seg2.phone_class
    if phone_class1 == 'CONSONANT' and phone_class2 in ('CONSONANT', 'GLIDE'):
        return True
    elif phone_class1 in ('VOWEL', 'DIPHTHONG', 'GLIDE') and phone_class2 in ('VOWEL', 'DIPHTHONG', 'GLIDE'):
        return True
    elif phone_class1 == 'TONEME' and phone_class2 == 'TONEME':
        return True
    elif phone_class1 in ('VOWEL', 'DIPHTHONG', 'GLIDE') and seg2.features['sonorant'] == 1:
        return True
    elif seg1.features['sonorant'] == 1 and phone_class2 in ('VOWEL', 'DIPHTHONG', 'GLIDE'):
        return True
    else:
        return False


def add_phon_env(alignment,
                 env_func=get_phon_env, 
                 gap_ch='-',
                 segs1=None):
    """Adds the phonological environment value of segments to an alignment
    e.g. 
    [('m', 'm'), ('j', '-'), ('ɔu̯', 'ɪ'), ('l̥', 'l'), ('k', 'ç')]) 
    becomes
    [(('m', '#S<'), 'm'), (('j', '<S<'), '-'), (('ɔu̯', '<S>'), 'ɪ'), (('l̥', '>S>'), 'l'), (('k', '>S#'), 'ç')]
    """
    word1_aligned, word2_aligned = tuple(zip(*alignment))
    word1_aligned = list(word1_aligned) # TODO use as tuple if possible, but this might disrupt some behavior elsewhere if lists are expected
    if not segs1:
        segs1 = tuple([seg for seg in word1_aligned if seg != gap_ch])
    gap_count1, gap_count2 = 0, 0

    def _add_phon_env(word_aligned, segs, i, gap_count, gap_ch):
        if word_aligned[i] == gap_ch:
            gap_count += 1
            # TODO so gaps are skipped?
        else:
            seg_index = i - gap_count
            phonEnv = env_func(segs, seg_index)
            word_aligned[i] = word_aligned[i], phonEnv
        
        return word1_aligned, gap_count

    for i, seg in enumerate(word1_aligned):
        word1_aligned, gap_count1 = _add_phon_env(word1_aligned, segs1, i, gap_count1, gap_ch)

    # TODO use as tuple if possible, but this might disrupt some behavior elsewhere if lists are expected
    return list(zip(word1_aligned, word2_aligned))


def get_alignment_iter(alignment, phon_env=False):
    # Handle class of input alignment: should be either an Alignment object or an iterable
    validate_class((alignment,), (Alignment, Iterable))
    if isinstance(alignment, Alignment):
        if phon_env:
            alignment = alignment.phon_env_alignment
        else:
            alignment = alignment.alignment
    
    return alignment


def reverse_alignment(alignment, phon_env=False):
    """Flips the alignment, e.g.:
        reverse_alignment([('s', 's̪'), ('o', 'ɔ'), ('l', 'l'), ('-', 'ɛ'), ('-', 'j')])
        = [('s̪', 's'), ('ɔ', 'o'), ('l', 'l'), ('ɛ', '-'), ('j', '-')]"""
    
    alignment = get_alignment_iter(alignment, phon_env)

    # TODO use tuple if possible, see above TODO comments
    return [(pair[1], pair[0]) for pair in alignment]


def visual_align(alignment, gap_ch='-', null='∅', phon_env=False):
    """Renders list of aligned segment pairs as an easily interpretable
    alignment string, with <∅> representing null segments,
    e.g.:
    visual_align([('z̪', 'ɡ'),('vʲ', 'v'),('ɪ', 'j'),('-', 'ˈa'),('z̪', 'z̪'),('d̪', 'd'),('ˈa', 'a')])
    = 'z̪-ɡ / vʲ-v / ɪ-j / ∅-ˈa / z̪-z̪ / d̪-d / ˈa-a' """

    if isinstance(alignment, Alignment) and gap_ch != alignment.gap_ch:
        raise ValueError(f'Gap character "{gap_ch}" does not match gap character of alignment "{alignment.gap_ch}"')
    
    alignment = get_alignment_iter(alignment, phon_env)
    if phon_env:
        raise NotImplementedError('TODO needs to be updated for phon_env') # TODO
    a = []
    for item in alignment:
        if gap_ch not in item:
            a.append(f'{item[0]}-{item[1]}')
        else:
            if item[0] == gap_ch:
                a.append(f'{null}-{item[1]}')
            else:
                a.append(f'{item[0]}-{null}')
    return ' / '.join(a)


def undo_visual_align(visual_alignment, gap_ch='-'):
    """Reverts a visual alignment to a list of tuple segment pairs"""
    seg_pairs = visual_alignment.split(' / ')
    seg_pairs = [tuple(pair.split(gap_ch)) for pair in seg_pairs]
    return seg_pairs
