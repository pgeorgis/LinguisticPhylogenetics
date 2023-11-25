from math import log, inf
from collections.abc import Iterable
from nwunschAlign import best_alignment
from phonUtils.segment import _toSegment, consonants
from phonUtils.phonSim import phone_sim
from phonUtils.phonEnv import get_phon_env
from auxFuncs import Distance, validate_class, flatten_ngram
import phyloLing # need Language and Word classes from phyloLing.py but cannot import them directly here because it will cause circular imports

class Ngram:
    def __init__(self, ngram, lang=None, seg_sep='_'):
        self.raw = ngram
        self.ngram = self.get_ngram(ngram, seg_sep)
        self.string = seg_sep.join(self.ngram)
        self.size = len(self.ngram)
        self.lang = lang
    
    @staticmethod
    def get_ngram(ngram, seg_sep='_'):
        if isinstance(ngram, str):
            return tuple(ngram.split(seg_sep))
        elif isinstance(ngram, Ngram):
            return ngram.ngram
        elif isinstance(ngram, tuple):
            return flatten_ngram(ngram)
        else:
            return flatten_ngram(tuple(ngram))
    
    def __str__(self):
        return self.string

def compatible_segments(seg1, seg2):
    """Determines whether a pair of segments are compatible for alignment. 
    Returns True if the two segments are either:
        two consonants
        two vowels
        a vowel and a sonorant consonant (nasals, liquids, glides)
        two tonemes/suprasegmentals
    Else returns False"""
    seg1, seg2 = map(_toSegment, [seg1, seg2])
    phone_class1, phone_class2 = seg1.phone_class, seg2.phone_class
    if phone_class1 in ('TONEME', 'SUPRASEGMENTAL') and phone_class2 in ('TONEME', 'SUPRASEGMENTAL'):
        return True
    elif phone_class1 in ('TONEME', 'SUPRASEGMENTAL'):
        return False
    elif phone_class2 in ('TONEME', 'SUPRASEGMENTAL'):
        return False
    
    if phone_class1 == 'CONSONANT' and phone_class2 in ('CONSONANT', 'GLIDE'):
        return True
    elif phone_class1 in ('VOWEL', 'DIPHTHONG', 'GLIDE') and phone_class2 in ('VOWEL', 'DIPHTHONG', 'GLIDE'):
        return True
    elif phone_class1 in ('VOWEL', 'DIPHTHONG', 'GLIDE') and seg2.features['sonorant'] == 1:
        return True
    elif phone_class1 in ('VOWEL', 'DIPHTHONG', 'GLIDE') and seg2.features['syllabic'] == 1:
        return True
    elif seg1.features['sonorant'] == 1 and phone_class2 in ('VOWEL', 'DIPHTHONG', 'GLIDE'):
        return True
    elif seg1.features['syllabic'] == 1 and phone_class2 in ('VOWEL', 'DIPHTHONG', 'GLIDE'):
        return True
    else:
        return False


def phon_alignment_cost(seg1, seg2, phon_func=phone_sim):
    # sim = phon_func(seg1, seg2)
    # if sim > 0:
    #     dist = log(sim)
    # else:
    #     dist = 1
    # if not compatible_segments(seg1, seg2):
    #     return phon_dist + 0.5
    # else:
    #     return phon_dist
    if seg1 == seg2:
        return 0
    elif compatible_segments(seg1, seg2):
        ph_sim = phon_func(seg1, seg2)
        if ph_sim > 0:
            return log(ph_sim)
        else:
            return -0.1
    else:
        # ph_sim = phon_func(seg1, seg2)
        # if ph_sim > 0:
        #     return log(ph_sim)
        # else:
        #     return -inf
        return -inf


AlignmentCost = Distance(
    func=phon_alignment_cost,
    sim=False,
    name='AlignmentCost'
)

class Alignment:
    def __init__(self, 
                 seq1, seq2,
                 lang1=None, 
                 lang2=None,
                 cost_func=AlignmentCost, 
                 added_penalty_dict=None,
                 gap_ch='-',
                 gop=-0.3, # TODO possibly need to recalibrate **
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
        self.alignment_costs, self.n_best = self.align(n_best)
        self.alignment = self.n_best[0][0]

        # Map aligned pairs to respective sequence indices
        self.seq_map = self.map_to_seqs()

        # Save length and cost of single best alignment
        self.cost = self.n_best[0][-1]
        self.length = len(self.alignment)
        
        # Phonological environment alignment
        self.phon_env = phon_env
        if self.phon_env:
            self.phon_env_alignment = self.add_phon_env()
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
                else:
                    return min(base_dist, -base_dist) + added_penalty
            
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
        
        return alignment_costs, best
    
    def remove_gaps(self, alignment=None):
        """Returns the alignment without gap-aligned positions.

        Returns:
            list: gap-free alignment list # TODO change to tuple later if possible
        """
        if alignment is None:
            alignment = self.alignment
        return [pair for pair in alignment if self.gap_ch not in pair]
    
    def gaps(self):
        """Returns a list of Gap objects in the alignment.

        Returns:
            list: list of Gap class objects
        """
        return [Gap(self, i) for i, pair in enumerate(self.alignment) if self.gap_ch in pair]

    def merge_align_positions(self, indices, new_index=None):
        indices.sort()
        if new_index is None:
            new_index = indices[0]
            
        to_merge = [(index, self.alignment[index]) for index in indices]
        merged = [[], []]
        for i, pair in to_merge:
            if self.gap_ch in pair:
                gap = Gap(self, i)
                merged[gap.seg_i].append(gap.segment)
            else:
                seg1, seg2 = pair
                merged[0].append(seg1)
                merged[1].append(seg2)
        merged = tuple([tuple(item) if len(item) > 1 else item[0] for item in merged])
        
        indices.reverse()
        #merged_alignment = deepcopy(self.alignment)
        for index in indices:
            del self.alignment[index]
        self.alignment.insert(new_index, merged)
        self.length = len(self.alignment)

    def compact_gaps(self, complex_ngrams):
        l1_bigrams = [ngram for ngram in complex_ngrams if ngram.size > 1]
        l2_bigrams = [nested_ngram for ngram in complex_ngrams 
                      for nested_ngram in complex_ngrams[ngram] if nested_ngram.size > 1]
        l1_bigram_segs = set(seg for bigram in l1_bigrams for seg in bigram.ngram)
        l2_bigram_segs = set(seg for bigram in l2_bigrams for seg in bigram.ngram)
         
        gaps = self.gaps()
        gaps.reverse()
        for gap in gaps:
            slices = gap.bigram_slices()
            # If the gap is in the first position, that means lang1 has a single seg where lang2 has two segs
            # gap.segment is the segment this gap is aligned to
            # gap.segment should be part of l2_bigram_segs 
            if gap.gap_i == 0 and gap.segment in l2_bigram_segs:
                for bigram in l2_bigrams:
                    if gap.segment in bigram.ngram:
                        gap_aligned_corr, seg_aligned_corr = gap.bigram_aligned_segs(bigram.ngram)
                        unigrams = [uni_corr for uni_corr in complex_ngrams if bigram in complex_ngrams[uni_corr]]
                        for unigram_corr in unigrams:
                            for start_i, end_i in slices:
                                if self.alignment[start_i:end_i] in (
                                    [(gap.gap_ch, gap_aligned_corr), (unigram_corr.string, seg_aligned_corr)],
                                    [(unigram_corr.string, seg_aligned_corr), (gap.gap_ch, gap_aligned_corr)],
                                ):
                                    self.merge_align_positions(indices=list(range(start_i,end_i)))
                                    break

            # If gap is not in first position, this means that lang1 has two segs where lang2 has one seg
            # gap.segment should be part of l1_bigram_segs
            elif gap.gap_i != 0 and gap.segment in l1_bigram_segs:
                for bigram in l1_bigrams:
                    if gap.segment in bigram.ngram:
                        gap_aligned_corr, seg_aligned_corr = gap.bigram_aligned_segs(bigram.ngram)
                        for unigram_corr in complex_ngrams[bigram]:                   
                            for start_i, end_i in slices:
                                if self.alignment[start_i:end_i] in (
                                    [(gap_aligned_corr, gap.gap_ch), (seg_aligned_corr, unigram_corr.string)],
                                    [(seg_aligned_corr, unigram_corr.string), (gap_aligned_corr, gap.gap_ch)],
                                ):
                                    self.merge_align_positions(indices=list(range(start_i,end_i)))
                                    break
        
        # Update sequence map and length for compacted alignment
        self.seq_map = self.map_to_seqs()
        self.length = len(self.alignment)

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
        adjust_gap1, adjust_gap2 = 0, 0
        adjust_complex1, adjust_complex2 = 0, 0
        for i, pair in enumerate(self.alignment):
            seg1, seg2 = pair
            if seg1 == self.gap_ch:
                map1[i] = None
                adjust_gap1 += 1
            else:
                map1[i] = [i-adjust_gap1+adjust_complex1]
                ngram = Ngram(seg1)
                if ngram.size > 1:
                    adjust_complex1 += ngram.size-1
                    for n in range(map1[i][0]+1, min(map1[i][0]+ngram.size, len(self.alignment)-1)):
                        map1[i].append(n)
                    
            if seg2 == self.gap_ch:
                map2[i] = None
                adjust_gap2 += 1
            else:
                map2[i] = [i-adjust_gap2+adjust_complex2]
                ngram = Ngram(seg2)
                if ngram.size > 1:
                    adjust_complex2 += ngram.size-1
                    for n in range(map2[i][0]+1, min(map2[i][0]+ngram.size, len(self.alignment)-1)):
                        map2[i].append(n)

        return map1, map2

    def add_phon_env(self, env_func=get_phon_env):
        self.phon_env = True
        return add_phon_env(self.alignment,
                            env_func=env_func, 
                            gap_ch=self.gap_ch,
                            segs1=self.seq1)

    def reverse(self):
        return ReversedAlignment(self)
    
    def __str__(self):
        return visual_align(self.alignment, gap_ch=self.gap_ch, phon_env=self.phon_env)


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
            self.phon_env_alignment = super().add_phon_env()
        else:
            self.phon_env_alignment = None

class AlignedPair:
    def __init__(self, alignment, index):
        self.alignment = alignment
        self.gap_ch = self.alignment.gap_ch
        self.index = index
        self.pair = self.alignment.alignment[self.index]
    
    def prev_pair(self):
        if self.index > 0:
            return AlignedPair(self.alignment, self.index-1)
        else:
            return None
        
    def next_pair(self):
        try:
            return AlignedPair(self.alignment, self.index+1)
        except IndexError:
            return None
    
    def context(self):
        return self.prev_pair(), self.next_pair()
    
    def is_gap(self):
        return self.gap_ch in self.pair
        

class Gap(AlignedPair):
    def __init__(self, alignment, index):
        super().__init__(alignment, index)
        self.gap_i = self.pair.index(self.gap_ch)
        self.seg_i = abs(self.gap_i - 1)
        self.segment = self.pair[self.seg_i]
    
    def bigram_slices(self):
        slices = [
                # e.g. kt ~ ʧ
                # ('-', 'k'), ('ʧ', 't') or ('k', '-'), ('t', 'ʧ')
                (self.index, self.index+2), 
                # ('ʧ', 'k'), ('-', 't') or ('k', 'ʧ'), ('t', '-')
                (self.index-1, self.index+1)
        ]
        return slices
    
    def bigram_aligned_segs(self, bigram):
        """Given a bigram tuple that the gap forms a larger alignment unit with, 
        this function identifies the bigram segment aligned to the gap 
        and the bigram segment aligned to another segment"""
        gap_aligned_corr_i = bigram.index(self.segment)
        seg_aligned_corr_i = abs(gap_aligned_corr_i-1)
        gap_aligned_corr = bigram[gap_aligned_corr_i]
        seg_aligned_corr = bigram[seg_aligned_corr_i]
        return gap_aligned_corr, seg_aligned_corr  

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

    def add_phon_env(word_aligned, segs, i, gap_count, gap_ch):
        if word_aligned[i] == gap_ch:
            gap_count += 1
            # TODO so gaps are skipped?
        else:
            seg_index = i - gap_count
            phonEnv = env_func(segs, seg_index)
            word_aligned[i] = word_aligned[i], phonEnv
        
        return word1_aligned, gap_count

    for i, seg in enumerate(word1_aligned):
        word1_aligned, gap_count1 = add_phon_env(word1_aligned, segs1, i, gap_count1, gap_ch)

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
    for pair in alignment:
        pair = list(pair)
        for i, seg in enumerate(pair):
            if isinstance(seg, tuple):
                pair[i] = ' '.join(seg)
            
        seg1, seg2 = pair
        if gap_ch not in pair:
            a.append(f'{seg1}-{seg2}')
        else:
            if seg1 == gap_ch:
                a.append(f'{null}-{seg2}')
            else:
                a.append(f'{seg1}-{null}')
                
    return ' / '.join(a)


def undo_visual_align(visual_alignment, gap_ch='-'):
    """Reverts a visual alignment to a list of tuple segment pairs"""
    seg_pairs = visual_alignment.split(' / ')
    seg_pairs = [tuple(pair.split(gap_ch)) for pair in seg_pairs]
    return seg_pairs
