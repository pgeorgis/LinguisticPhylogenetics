import importlib
import logging
import sys
from collections import defaultdict
from collections.abc import Iterable
from itertools import zip_longest
from math import inf, log
from statistics import mean

import numpy as np
from constants import (END_PAD_CH, GAP_CH_DEFAULT, NULL_CH_DEFAULT,
                       PAD_CH_DEFAULT, SEG_JOIN_CH, START_PAD_CH)
from nwunschAlign import best_alignment
from phonUtils.phonEnv import get_phon_env
from phonUtils.phonSim import phone_sim
from phonUtils.segment import _toSegment
from utils.distance import Distance
from utils.information import calculate_infocontent_of_word
from utils.sequence import (Ngram, PhonEnvNgram, end_token, flatten_ngram,
                            pad_sequence, start_token)
from utils.utils import combine_dicts, validate_class

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def needleman_wunsch_extended(seq1, seq2, align_cost, gap_cost, default_gop, gap_ch=GAP_CH_DEFAULT, maximize_score=False):
    """
    Align two sequences with a modified Needleman-Wunsch algorithm, allowing for flexible alignments.
    
    Args:
        seq1: list of elements in sequence 1.
        seq2: list of elements in sequence 2.
        align_cost: Dictionary where the keys are tuples of sub-sequences to align, 
                    and the values are the associated cost.
                    For example: align_cost[(('A',), ('G', 'C'))] gives the cost of aligning 'A' with 'GC'.
        gap_cost: Dictionary where the keys are tuples representing gaps and sub-sequences,
                  and the values are the associated gap cost.
                  For example: gap_cost[('A', None)] gives the cost of aligning 'A' to a gap.
                  
    Returns:
        alignment_score: The optimal alignment score.
        aligned_seq1, aligned_seq2: The two aligned sequences (as lists).
    """
    n = len(seq1)
    m = len(seq2)
    worst_score = -inf if maximize_score else inf

    # Initialize score and traceback matrices
    dp = np.zeros((n + 1, m + 1))
    traceback = np.empty((n + 1, m + 1), dtype=object)
    
    def seq2ngram(seq):
        return Ngram(seq).undo()
    
    def score_is_better(score1, score2):
        # maximize_score : if True, the optimum alignment has a score score
        #                  if False, the optimum alignment has the lowest score (lowest cost))
        if maximize_score:
            if score1 > score2:
                return True
            return False
        return score1 < score2
    
    # Fill first row and column with gap penalties
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + gap_cost.get((seq2ngram(seq1[:i]), gap_ch), default_gop)
        traceback[i][0] = (1, 0)  # Indicates seq1 gaps
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + gap_cost.get((gap_ch, seq2ngram(seq2[:j])), default_gop)
        traceback[0][j] = (0, 1)  # Indicates seq2 gaps
    
    # Fill the dp matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best_score = worst_score
            best_move = None
            
            # Align one unit from seq1 to one or more from seq2
            for k in range(1, i + 1):
                for l in range(1, j + 1):
                    cost = align_cost.get((seq2ngram(seq1[i-k:i]), seq2ngram(seq2[j-l:j])), worst_score)
                    score = dp[i-k][j-l] + cost
                    if score_is_better(score, best_score):
                        best_score = score
                        best_move = (k, l)
            
            # Align seq1 to a gap
            for k in range(1, i + 1):
                cost = gap_cost.get((seq2ngram(seq1[i-k:i]), gap_ch), worst_score)
                score = dp[i-k][j] + cost
                if score_is_better(score, best_score):
                    best_score = score
                    best_move = (k, 0)
            
            # Align seq2 to a gap
            for l in range(1, j + 1):
                cost = gap_cost.get((gap_ch, seq2ngram(seq2[j-l:j])), worst_score)
                score = dp[i][j-l] + cost
                if score_is_better(score, best_score):
                    best_score = score
                    best_move = (0, l)
            
            dp[i][j] = best_score
            traceback[i][j] = best_move  # Store the move in traceback
    
    # Backtrack to find the alignment
    aligned_seq1, aligned_seq2 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        move = traceback[i][j]
        if move[0] > 0 and move[1] > 0:
            # Align subsequences of both seq1 and seq2
            aligned_seq1.append(seq2ngram(seq1[i-move[0]:i]))
            aligned_seq2.append(seq2ngram(seq2[j-move[1]:j]))
            i -= move[0]
            j -= move[1]
        elif move[0] > 0:
            # Align subsequence of seq1 to a gap
            aligned_seq1.append(seq2ngram(seq1[i-move[0]:i]))
            aligned_seq2.append(seq2ngram([gap_ch]))
            i -= move[0]
        else:
            # Align subsequence of seq2 to a gap
            aligned_seq1.append(seq2ngram([gap_ch]))
            aligned_seq2.append(seq2ngram(seq2[j-move[1]:j]))
            j -= move[1]
    
    # Reverse the aligned sequences to be in correct order
    aligned_seq1.reverse()
    aligned_seq2.reverse()
    alignment = list(zip(aligned_seq1, aligned_seq2))

    return dp[n][m], alignment


def compatible_segments(seg1, seg2):
    """Determines whether a pair of segments are compatible for alignment.
    Returns True if the two segments are either:
        two consonants
        two vowels
        a vowel and a sonorant (nasals, liquids, glides) and/or syllabic consonant
        two tonemes/suprasegmentals
    Else returns False"""
    seg1, seg2 = map(_toSegment, [seg1, seg2])
    phone_class1, phone_class2 = seg1.phone_class, seg2.phone_class

    # Tonemes/suprasegmentals can only be aligned with tonemes/suprasegmentals
    if phone_class1 in ('TONEME', 'SUPRASEGMENTAL') and phone_class2 in ('TONEME', 'SUPRASEGMENTAL'):
        return True
    elif phone_class1 in ('TONEME', 'SUPRASEGMENTAL'):
        return False
    elif phone_class2 in ('TONEME', 'SUPRASEGMENTAL'):
        return False

    # Consonants can always be aligned with consonants and glides
    if phone_class1 == 'CONSONANT' and phone_class2 in ('CONSONANT', 'GLIDE'):
        return True
    elif phone_class2 == 'CONSONANT' and phone_class1 in ('CONSONANT', 'GLIDE'):
        return True

    # Vowels/diphthongs/glides can always be aligned with one another
    elif phone_class1 in ('VOWEL', 'DIPHTHONG', 'GLIDE') and phone_class2 in ('VOWEL', 'DIPHTHONG', 'GLIDE'):
        return True

    # Sonorant and syllabic consonants can be aligned with vowels/diphthongs
    elif seg1.features['sonorant'] == 1 and phone_class2 in ('VOWEL', 'DIPHTHONG', 'GLIDE'):
        return True
    elif seg1.features['syllabic'] == 1 and phone_class2 in ('VOWEL', 'DIPHTHONG', 'GLIDE'):
        return True
    elif phone_class1 in ('VOWEL', 'DIPHTHONG', 'GLIDE') and seg2.features['sonorant'] == 1:
        return True
    elif phone_class1 in ('VOWEL', 'DIPHTHONG', 'GLIDE') and seg2.features['syllabic'] == 1:
        return True
    else:
        return False


def to_unigram_alignment(bigram, fillvalue=GAP_CH_DEFAULT):
    unigrams = [[] for _ in range(len(bigram))]
    for i, pos in enumerate(bigram):
        if isinstance(pos, str):
            unigrams[i].append(pos)
        elif isinstance(pos, tuple):
            unigrams[i].extend(pos)

    return list(zip_longest(*unigrams, fillvalue=fillvalue))


def unigram_complex_alignment_mismatches(complex_alignment, unigram_alignment, gap_ch=GAP_CH_DEFAULT):
    """Compares a complex ngram alignment with a unigram alignment and returns the indices
    of the complex alignment where the aligned pairs don't match the unigram alignment."""

    def increment_unigram_i(unigram_alignment, unigram_i, complex_ngram, complex_gap_left, complex_gap_right):
        if unigram_i >= len(unigram_alignment)-1:
            return 0, complex_gap_left, complex_gap_right
        left, right = complex_ngram.pair
        final_unigrams = (Ngram(left).ngram[-1], Ngram(right).ngram[-1])
        gap_i = None
        total_gaps = complex_gap_left + complex_gap_right
        if complex_ngram.is_gappy:
            gap = Gap([complex_ngram.pair], 0)
            gap_i = gap.gap_i
            total_gaps += 1


        incr_left, incr_right = 0, 0
        if gap_i != 0:
            while unigram_alignment[unigram_i - complex_gap_left + incr_left][0] != final_unigrams[0]:
                incr_left += 1
                if unigram_i - complex_gap_left + incr_left >= len(unigram_alignment):
                    # final_unigrams[0] not found in unigram alignment search scope,
                    # which means that starting index was too high because alignment component is behind current unigram_i
                    # set incr_left to -1 and stop iteration
                    incr_left = -1
                    break
        else:
            complex_gap_left += 1

        if gap_i != 1:
            while unigram_alignment[unigram_i - complex_gap_right + incr_right][-1] != final_unigrams[-1]:
                incr_right += 1
                if unigram_i - complex_gap_right + incr_right >= len(unigram_alignment):
                    # final_unigrams[-1] not found in unigram alignment search scope,
                    # which means that starting index was too high because alignment component is behind current unigram_i
                    # set incr_right to -1 and stop iteration
                    incr_right = -1
                    break
        else:
            complex_gap_right += 1

        incr_unigram = max(0, min(incr_left - complex_gap_left + 1, incr_right - complex_gap_right + 1))
        if complex_gap_left == complex_gap_right:
            complex_gap_left = 0
            complex_gap_right = 0

        return incr_unigram, complex_gap_left, complex_gap_right

    mismatch_indices = []
    unigram_i = 0
    complex_gap_left, complex_gap_right = 0, 0
    unigram_len = len(unigram_alignment)
    for complex_i, complex_ngram in enumerate(complex_alignment):
        # Skip if unigram indices already reached end
        # In this case there should only be end boundary alignments left in complex if indices are correctly incremented
        if unigram_i >= unigram_len:
            mismatch_indices.append(complex_i)
            unigram_i += 1
            continue

        aligned_ngram = AlignedPair(complex_alignment, complex_i)
        next_unigram_pos = AlignedPair(unigram_alignment, unigram_i)
        if complex_ngram in unigram_alignment[unigram_i:]:
            # "complex" ngram is just a unigram matching a unigram in the unigram alignment
            if complex_i < len(complex_alignment)-1:
                # if not the final complex alignment position
                # check that the same unigram doesn't occur later in the unigram alignment
                if complex_ngram in unigram_alignment[unigram_i+1:]:
                    # if it does, determine whether the current complex ngram position corresponds with this unigram or the later one
                    breakpoint()
                    bp = 1
                # else accept the current match
                else:
                    unigram_i += 1
                    if next_unigram_pos.is_gappy and not aligned_ngram.is_gappy:
                        incr, complex_gap_left, complex_gap_right = increment_unigram_i(
                            unigram_alignment,
                            unigram_i,
                            aligned_ngram,
                            complex_gap_left,
                            complex_gap_right,
                        )
                        unigram_i += incr
                        breakpoint()
                        bp = 1.5

            else: # if it is the final complex alignment position, accept the match
                unigram_i += 1
        else:
            mismatch_indices.append(complex_i)
            if aligned_ngram.is_complex:
                # complex alignment pair
                # Increment unigrams_i by max of 2 or an increment indicating
                # the number of positions until all component segments were found in unigram alignment
                incr, complex_gap_left, complex_gap_right = increment_unigram_i(
                    unigram_alignment,
                    unigram_i,
                    aligned_ngram,
                    complex_gap_left,
                    complex_gap_right,
                )
                unigram_i += incr

            else:
                # unigram alignment pair not in unigrams alignment
                incr, complex_gap_left, complex_gap_right = increment_unigram_i(
                    unigram_alignment,
                    unigram_i,
                    aligned_ngram,
                    complex_gap_left,
                    complex_gap_right,
                )
                unigram_i += incr
    return mismatch_indices


PhonSim = Distance(
    func=phone_sim,
    sim=True,
    name='PhonSim'
)

def phon_alignment_cost(seg1, seg2, phon_func=phone_sim):
    if seg1 == seg2:
        return 0
    #elif compatible_segments(seg1, seg2):
    ngram1, ngram2 = map(lambda x: flatten_ngram(Ngram(x).ngram), [seg1, seg2])
    # TODO would be better to somehow recursively align segments and take phon sim of just aligned segs rather than taking mean of all combinations

    ph_sims = []
    for i, ngram_seg1 in enumerate(ngram1):
        ngram_seg1 = Ngram(ngram_seg1)
        for j, ngram_seg2 in enumerate(ngram2):
            ngram_seg2 = Ngram(ngram_seg2)
            if ngram_seg1.is_boundary() and not (ngram_seg2.is_boundary() or ngram_seg2.is_gappy()):
                # seg1 is boundary, seg2 is not boundary or gap
                if ngram_seg1.size > 1:
                    # seg1 is complex boundary, seg2 is not boundary or gap
                    ph_sims.append(phon_alignment_cost(ngram1[i], ngram2[j], phon_func=phon_func))
                else:
                    # seg1 is unigram boundary, seg2 is not boundary or gap
                    ph_sims.append(0) # TODO would be better to quantify similarity of segment to gap via sonority or features
            elif ngram_seg1.is_gappy() and not (ngram_seg2.is_boundary() or ngram_seg2.is_gappy()):
                # seg1 is gap, seg2 is not boundary or gap
                ph_sims.append(0) # TODO would be better to quantify similarity of segment to gap via sonority or features
            elif ngram_seg1.is_boundary():
                # seg1 is boundary, seg2 is a boundary or gap
                if ngram_seg2.is_gappy():
                    # seg1 is boundary, seg2 is gap
                    if ngram_seg1.size > 1:
                        # seg1 is complex boundary, seg2 is gap
                        ph_sims.append(phon_alignment_cost(ngram1[i], ngram2[j], phon_func=phon_func))
                    else:
                        # seg1 is unigram boundary, seg2 is gap
                        ph_sims.append(0)
                else:
                    # seg1 is boundary, seg2 is boundary
                    if ngram_seg1.size == 1 and ngram_seg2.size == 1:
                        # both seg1 and seg2 are unigram boundaries
                        # score = 1 if same direction boundary, else -inf
                        if ngram1[i] == ngram2[j]:
                            ph_sims.append(1)
                        else:
                            ph_sims.append(-inf)
                    else:
                        # at least one of the two is a complex boundary
                        ph_sims.append(phon_alignment_cost(ngram1[i], ngram2[j], phon_func=phon_func))
            elif ngram_seg1.is_gappy():
                # seg1 is gap, seg2 is a boundary or gap
                if ngram_seg2.size > 1:
                    # seg1 is gap, seg2 is a complex boundary
                    ph_sims.append(phon_alignment_cost(ngram1[i], ngram2[j], phon_func=phon_func))
                else:
                    # seg1 is gap, seg2 is unigram boundary
                    ph_sims.append(0)
            # seg1 is not boundary or gap
            elif ngram_seg2.is_boundary():
                # seg1 is not boundary or gap, seg2 is boundary
                if ngram_seg2.size > 1:
                    # seg1 is not boundary or gap, seg2 is complex boundary
                    ph_sims.append(phon_alignment_cost(ngram1[i], ngram2[j], phon_func=phon_func))
                else:
                    # seg1 is not boundary or gap, seg2 is unigram boundary
                    ph_sims.append(0) # TODO would be better to quantify similarity of segment to gap via sonority or features
            elif ngram_seg2.is_gappy():
                # seg1 is not boundary or gap, seg2 is gap
                ph_sims.append(0) # TODO would be better to quantify similarity of segment to gap via sonority or features
            else:
                # neither seg1 nor seg2 is a boundary or gap
                ph_sims.append(phon_func(ngram1[i], ngram2[j]))

    ph_sim = mean(ph_sims)
    if ph_sim > 0:
        return log(ph_sim)
    else:
        return log(0.00001) * len(ph_sims) # currently no segment pairs have a phone_sim score this low, so will always be well below any segment score


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
                 gap_ch=GAP_CH_DEFAULT,
                 gop=-10,
                 pad_ch=PAD_CH_DEFAULT,
                 n_best=1,
                 phon_env=False,
                 **kwargs
                 ):
        f"""Produces a pairwise alignment of two phone sequences.

        Args:
            seq1 (phyloLing.Word or str): first phone sequence
            seq2 (phyloLing.Word or str): second phone sequence
            lang1 (phyloLing.Language, optional): Language of seq1. Defaults to None.
            lang2 (phyloLing.Language, optional): Language of seq2. Defaults to None.
            cost_func (Distance, optional): Cost function used for minimizing overall alignment cost. Defaults to AlignmentPhoneSim.
            added_penalty_dict (dict, optional): Dictionary of additional penalties to combine with cost_func. Defaults to None.
            gap_ch (str, optional): Gap character. Defaults to '{GAP_CH_DEFAULT}'.
            gop (float, optional): Gap opening penalty. Defaults to -0.7.
            n_best (int, optional): Number of best (least costly) alignments to return. Defaults to 1.
            phon_env (Bool, optional): Adds phonological environment to alignment. Defaults to False.
        """

        # Verify that input arguments are of the correct types
        self.validate_args(seq1, seq2, lang1, lang2, cost_func)

        # Prepare the input sequences for alignment
        self.seq1, self.word1 = self.prepare_seq(seq1, lang1)
        self.seq2, self.word2 = self.prepare_seq(seq2, lang2)
        
        # Set languages
        self.lang1 = lang1
        self.lang2 = lang2

        # Designate alignment parameters
        self.gap_ch = gap_ch
        self.gop = gop
        self.pad_ch = pad_ch
        self.start_boundary_token = f'{START_PAD_CH}{self.pad_ch}'
        self.end_boundary_token = f'{self.pad_ch}{END_PAD_CH}'
        self.cost_func = cost_func
        self.added_penalty_dict = added_penalty_dict
        self.kwargs = kwargs

        # Perform alignment
        self.alignment_costs, self.n_best = self.align(n_best)
        self.alignment = self.n_best[0][0][:]

        # Save length and cost of single best alignment
        self.cost = self.n_best[0][-1]
        self.length = len(self.alignment)
        self.original_length = self.length

        # Map aligned pairs to respective sequence indices
        self.seq_map = self.map_to_seqs()

        # Phonological environment alignment
        self.phon_env = phon_env
        if self.phon_env:
            self.phon_env_alignment = self.add_phon_env()
        else:
            self.phon_env_alignment = None

    def validate_args(self, seq1, seq2, lang1, lang2, cost_func):
        """Verifies that all input arguments are of the correct types"""
        phyloLing = importlib.import_module('phyloLing')
        validate_class((cost_func,), (Distance,))
        validate_class((seq1,), ((phyloLing.Word, str),))
        validate_class((seq2,), ((phyloLing.Word, str),))
        for lang in (lang1, lang2):
            if lang:  # skip if None
                validate_class((lang,), (phyloLing.Language,))

    def prepare_seq(self, seq, lang):
        phyloLing = importlib.import_module('phyloLing')
        if isinstance(seq, phyloLing.Word):
            word1 = seq
        elif isinstance(seq, str):
            word1 = phyloLing.Word(seq, language=lang)

        return word1.segments, word1

    def calculate_alignment_costs(self, cost_func, seq1=None, seq2=None):
        """Calculates pairwise alignment costs for phone sequences using a specified cost function.

        Args:
            cost_func (Distance): cost function used for computing pairwise alignment costs

        Returns:
            dict: dictionary of pairwise alignment costs by sequence indices
        """
        if seq1 is None:
            seq1 = self.seq1
        if seq2 is None:
            seq2 = self.seq2
        alignment_costs = {}
        for i, seq1_i in enumerate(seq1):
            for j, seq2_j in enumerate(seq2):
                cost = cost_func.eval(seq1_i, seq2_j, **self.kwargs)

                # If similarity function, turn into distance and ensure it is negative # TODO add into Distance object
                if cost_func.sim:
                    if cost > 0:
                        cost = log(cost)
                    else:
                        cost = -inf

                alignment_costs[(i, j)] = cost
                alignment_costs[(seq1_i, seq2_j)] = cost

        return alignment_costs

    def calculate_gap_costs(self, cost_func, seq1=None):
        if seq1 is None:
            seq1 = self.seq1
        # if seq2 is None:
        #     seq2 = self.seq2
        gap_costs = {}
        for i, seq1_i in enumerate(seq1):
            cost = cost_func.eval(seq1_i, self.gap_ch, **self.kwargs)
            gap_costs[(seq1_i, self.gap_ch)] = cost
        # for j, seq2_j in enumerate(seq2):
        #     cost = cost_func.eval(self.gap_ch, seq2_j, **self.kwargs)
        #     gap_costs[(self.gap_ch, seq2_j)] = cost
        return gap_costs

    def added_penalty_dist(self, seq1, seq2, **kwargs):
        assert self.added_penalty_dict is not None
        added_penalty = self.added_penalty_dict[seq1][seq2]
        base_dist = self.cost_func.eval(seq1, seq2, **kwargs)
        # If similarity function, turn into distance and ensure it is negative # TODO add into Distance object
        if self.cost_func.sim:
            base_dist = -(1 - base_dist)
            return base_dist + added_penalty
        else:
            return min(base_dist, -base_dist) + added_penalty


    def align(self, n_best=1): # TODO update description
        """Align segments of word1 with segments of word2 according to Needleman-
        Wunsch algorithm, with costs determined by phonetic and sonority similarity;
        If not segmented, the words are first segmented before being aligned.
        GOP = -1 by default, determined by cross-validation on dataset of gold cognate alignments."""

        # Combine base distances from distance function with additional penalties, if specified
        if self.added_penalty_dict:
            cost_func = Distance(func=self.added_penalty_dist, **self.kwargs)
        # Otherwise calculate alignment costs for each segment pair using only the base distance function
        else:
            cost_func = self.cost_func
        get_ngram_alignment_costs = lambda seq1, seq2: self.calculate_alignment_costs(cost_func, seq1=seq1, seq2=seq2)
        get_gap_costs = lambda seq1, seq2: self.calculate_gap_costs(cost_func, seq1=seq1) # seq2=seq2

        # Pad unigram sequences
        padded1 = pad_sequence(self.seq1, pad_ch=self.pad_ch, pad_n=1)
        padded2 = pad_sequence(self.seq2, pad_ch=self.pad_ch, pad_n=1)

        #Generate bigrams
        bigrams_seq1 = self.word1.get_ngrams(size=2, pad_ch=self.pad_ch)
        bigrams_seq2 = self.word2.get_ngrams(size=2, pad_ch=self.pad_ch)

        # Get alignment costs for ngram pairs of each size
        bigram_scores = get_ngram_alignment_costs(bigrams_seq1, bigrams_seq2)
        unigram_scores = get_ngram_alignment_costs(padded1, padded2)
        bigram1_unigram2_scores = get_ngram_alignment_costs(bigrams_seq1, padded2)
        bigram2_unigram1_scores = get_ngram_alignment_costs(padded1, bigrams_seq2)

        # Get gap costs for each ngram size pair
        bigram_gap_costs = get_gap_costs(bigrams_seq1, bigrams_seq2)
        unigram_gap_costs = get_gap_costs(padded1, padded2)
        bigram1_unigram2_gap_costs = get_gap_costs(bigrams_seq1, padded2)
        bigram2_unigram1_gap_costs = get_gap_costs(padded1, bigrams_seq2)

        # Combine alignment and gap costs from bigrams and unigrams
        combined_align_costs = combine_dicts(
            bigram_scores,
            unigram_scores,
            bigram1_unigram2_scores,
            bigram2_unigram1_scores,
        )
        combined_gap_costs = combine_dicts(
            bigram_gap_costs,
            unigram_gap_costs,
            bigram1_unigram2_gap_costs,
            bigram2_unigram1_gap_costs,
        )
        complex_alignment_score, complex_alignment = needleman_wunsch_extended(
            seq1=padded1, 
            seq2=padded2,
            align_cost=combined_align_costs,
            gap_cost=combined_gap_costs,
            gap_ch=GAP_CH_DEFAULT,
            default_gop=self.gop,
            maximize_score=True,
        )
        # complex_alignment = self.compact_boundary_gaps(complex_alignment)
        #print(visual_align(complex_alignment))

        # TODO current tasks
        # still bug: in compact_boundary_gaps (Romanian-Ligurian)
        # missing final /u/
        # (Pdb) complex_alignment
        # [('-', '<#'), (('<#', 'a'), ('i', 'n')), (('n', 'ˈe'), 'ˈe'), ('l', ('l', '#>')), (('u', '#>'), '-')]
        # (Pdb) self.compact_boundary_gaps(complex_alignment)
        # [(('<#', 'a'), ('i', 'n', '<#')), (('n', 'ˈe'), 'ˈe'), (('l', '#>'), ('l', '#>'))]
        # Compact boundary gap alignments
        # TODO: possibly compact before? or both before and after?
        # TODO: possibly save unigram, bigram, and complex alignments

        alignment_costs = {
            "2-2": bigram_scores,
            "2-1": bigram1_unigram2_scores,
            "1-2": bigram2_unigram1_scores,
            "1-1": unigram_scores,
        }

        return alignment_costs, [(complex_alignment, )] # TODO simplify output format

    def compact_boundary_gaps(self, complex_alignment):
        # Add compacting of boundary gap alignment in situations like:
        # (('ˈa', '#>'), ('ˈɐ̃', 'w̃')), ('-', '#>') -> (('ˈa', '#>'), ('ˈɐ̃', 'w̃', '#>')) (Catalan/Portuguese)

        # Do nothing if the alignment consists of a single unit
        # or if there are unmatched boundary tokens (which could occur if realigning a subsequence)
        if len(complex_alignment) == 1:
            return complex_alignment
        else:
            flat = flatten_ngram(complex_alignment)

        if flat.count(end_token()) == 2:
            last_ngram = Ngram(complex_alignment[-1])
            if last_ngram.is_gappy(self.gap_ch) and last_ngram.is_boundary(self.pad_ch):
                penult_ngram = Ngram(complex_alignment[-2])
                end_boundary_gap = complex_alignment.pop()
                end_boundary_gap = Gap([end_boundary_gap], 0, gap_ch=self.gap_ch)
                if penult_ngram.is_boundary(self.pad_ch):
                    final_complex = [[], []]
                    final_complex[end_boundary_gap.gap_i].extend([x for x in Ngram(complex_alignment[-1][end_boundary_gap.gap_i]).ngram if x != self.end_boundary])
                    final_complex[end_boundary_gap.seg_i].extend([x for x in Ngram(complex_alignment[-1][end_boundary_gap.seg_i]).ngram if x != self.end_boundary])
                    final_complex[end_boundary_gap.seg_i].append(self.end_boundary_token)
                    complex_alignment = complex_alignment[:-1]
                    complex_alignment.append( (Ngram(final_complex[0]).undo(), Ngram(final_complex[-1]).undo()) )

                else:
                    # Iterate backwards to find the index which contains the other boundary token
                    j = 1 # start at 1 because j=0 is already implicitly checked in order to even enter this else block
                    while not Ngram(complex_alignment[-1-j]).is_boundary(self.pad_ch):
                        j += 1
                    penult_ngram = Ngram(complex_alignment[-1-j])
                    final_complex = [[], []]
                    for k in range(-1-j, 0):
                        final_complex[end_boundary_gap.gap_i].extend([x for x in Ngram(complex_alignment[k][end_boundary_gap.gap_i]).ngram if x not in (self.gap_ch, self.end_boundary_token)])
                        final_complex[end_boundary_gap.seg_i].extend([x for x in Ngram(complex_alignment[k][end_boundary_gap.seg_i]).ngram if x not in (self.gap_ch, self.end_boundary_token)])
                    final_complex[end_boundary_gap.seg_i].extend([x for x in Ngram(end_boundary_gap.pair).ngram if x != self.gap_ch])
                    final_complex[end_boundary_gap.gap_i].append(self.end_boundary_token)
                    complex_alignment = complex_alignment[:-1-j]
                    complex_alignment.append( (tuple(final_complex[0]), Ngram(final_complex[-1]).undo()) )

        if flat.count(start_token()) == 2:
            first_ngram = Ngram(complex_alignment[0])
            if first_ngram.is_gappy(self.gap_ch) and first_ngram.is_boundary(self.pad_ch):
                next_ngram = Ngram(complex_alignment[1])
                if next_ngram.is_boundary(self.pad_ch):
                    # Penultimate ngram is a complex ngram alignment
                    start_boundary_gap = complex_alignment[0]
                    start_boundary_gap = Gap([start_boundary_gap], 0, gap_ch=self.gap_ch)
                    initial_complex = [[], []]
                    if next_ngram.size == 2 or first_ngram.size == 2: # aligned unigram
                        initial_complex[start_boundary_gap.gap_i].extend([x for x in Ngram(complex_alignment[1][start_boundary_gap.gap_i]).ngram if x not in (self.gap_ch, self.start_boundary_token)])
                        initial_complex[start_boundary_gap.seg_i].extend([x for x in Ngram(complex_alignment[1][start_boundary_gap.seg_i]).ngram if x not in (self.gap_ch, self.start_boundary_token)])
                        initial_complex[start_boundary_gap.seg_i].extend([x for x in Ngram(start_boundary_gap.pair).ngram if x != self.gap_ch])
                        initial_complex[start_boundary_gap.gap_i].insert(0, self.start_boundary_token)
                        complex_alignment = [(Ngram(initial_complex[0]).undo(), Ngram(initial_complex[-1]).undo())] + complex_alignment[2:]
                    else:
                        breakpoint()
                        bp = 4
                        initial_complex[start_boundary_gap.gap_i].extend([x for x in complex_alignment[1][start_boundary_gap.gap_i] if x != self.gap_ch])
                        initial_complex[start_boundary_gap.seg_i].extend([x for x in complex_alignment[0][start_boundary_gap.seg_i] if x != self.gap_ch])
                        initial_complex[start_boundary_gap.seg_i].insert(0, self.start_boundary_token)
                        complex_alignment[0] = (tuple(initial_complex[0]), tuple(initial_complex[-1]))

        return complex_alignment

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
        # merged_alignment = deepcopy(self.alignment)
        for index in indices:
            del self.alignment[index]
        self.alignment.insert(new_index, merged)
        self.length = len(self.alignment)


    def compact_gaps(self, complex_ngrams, simple_ngrams):
        # Step 1: Extract bigrams and their aligned segments
        l1_bigrams, l2_bigrams, l1_bigram_segs, l2_bigram_segs = get_bigrams(complex_ngrams)

        # Step 2: Process gaps and identify merge ranges and candidates
        gaps = self.gaps()
        merge_ranges, merge_candidates = self.process_gaps(
            gaps, l1_bigrams, l2_bigrams, l1_bigram_segs, l2_bigram_segs, complex_ngrams
        )

        # Step 3: Select best merge group(s)
        merge_ranges = self.select_best_merge_group(list(merge_ranges), complex_ngrams, simple_ngrams, merge_candidates)

        # Step 4: Apply merge operations
        for merge_range in merge_ranges:
            if merge_range is not None:
                self.merge_align_positions(indices=list(merge_range))

        # Step 5: Update sequence map and length for compacted alignment
        self.update()

    def process_gaps(self, gaps, l1_bigrams, l2_bigrams, l1_bigram_segs, l2_bigram_segs, complex_ngrams):
        """Process gaps and collect merge ranges and candidates."""
        merge_ranges = set()
        merge_candidates = {}

        for gap in reversed(gaps):  # Reverse the gaps to process in reverse order
            slices = gap.bigram_slices()
            # If the gap is in the first position, that means lang1 has a single seg where lang2 has two segs
            # gap.segment is the segment this gap is aligned to
            # gap.segment should be part of l2_bigram_segs
            if gap.gap_i == 0 and gap.segment in l2_bigram_segs:
                merge_ranges, merge_candidates = self.handle_l2_bigram_gaps(
                    gap, l2_bigrams, slices, complex_ngrams, merge_ranges, merge_candidates
                )
            # If gap is not in first position, this means that lang1 has two segs where lang2 has one seg
            # gap.segment should be part of l1_bigram_segs
            elif gap.gap_i != 0 and gap.segment in l1_bigram_segs:
                merge_ranges, merge_candidates = self.handle_l1_bigram_gaps(
                    gap, l1_bigrams, slices, complex_ngrams, merge_ranges, merge_candidates
                )

        return merge_ranges, merge_candidates

    def handle_l2_bigram_gaps(self, gap, l2_bigrams, slices, complex_ngrams, merge_ranges, merge_candidates):
        """Handle gaps where lang2 has two segments and lang1 has one."""
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
                            merge_ranges.add(tuple(range(start_i, end_i)))
                            merge_candidates[tuple(range(start_i, end_i))] = (unigram_corr, bigram)
                        #     break
                        # break
        return merge_ranges, merge_candidates

    def handle_l1_bigram_gaps(self, gap, l1_bigrams, slices, complex_ngrams, merge_ranges, merge_candidates):
        """Handle gaps where lang1 has two segments and lang2 has one."""
        for bigram in l1_bigrams:
            if gap.segment in bigram.ngram:
                gap_aligned_corr, seg_aligned_corr = gap.bigram_aligned_segs(bigram.ngram)
                for unigram_corr in complex_ngrams[bigram]:
                    for start_i, end_i in slices:
                        if self.alignment[start_i:end_i] in (
                            [(gap_aligned_corr, gap.gap_ch), (seg_aligned_corr, unigram_corr.string)],
                            [(seg_aligned_corr, unigram_corr.string), (gap_aligned_corr, gap.gap_ch)],
                        ):
                            merge_ranges.add(tuple(range(start_i, end_i)))
                            merge_candidates[tuple(range(start_i, end_i))] = (bigram, unigram_corr)
                        #     break
                        # break
        return merge_ranges, merge_candidates

    def get_merge_range_pmi(self, merge_range, complex_ngrams, simple_ngrams, merge_candidates):
        """Calculate the PMI for an alignment if a range of alignment positions are to be merged."""
        global_pmi = 0
        for i, position in enumerate(self.alignment):
            if i not in merge_range:
                seg1, seg2 = position
                global_pmi += simple_ngrams[Ngram(seg1).ngram].get(Ngram(seg2).ngram, 0)
            else:
                complex_pmi = complex_ngrams[merge_candidates[tuple(merge_range)][0]][merge_candidates[tuple(merge_range)][-1]]
                global_pmi += complex_pmi

        return global_pmi

    def select_best_merge_group(self, merge_ranges, complex_ngrams, simple_ngrams, merge_candidates):
        """Select the best merge groups based on PMI and conflicts."""
        if len(merge_ranges) > 1:
            merge_ranges = sorted(merge_ranges, key=lambda x: (x[0], x[-1]), reverse=True)
            i = 0
            while i < len(merge_ranges) - 1:
                merge_range = merge_ranges[i]
                next_merge_range = merge_ranges[i + 1]

                # Check if ranges conflict
                if merge_range and next_merge_range and min(merge_range) <= max(next_merge_range):
                    # Resolve conflict and remove the weaker merge range
                    to_remove = self.resolve_merge_range_conflict(
                        merge_range, next_merge_range, complex_ngrams, simple_ngrams, merge_candidates
                    )
                    if to_remove is not None:
                        merge_ranges.remove(to_remove)
                    else:
                        i += 1
                else:
                    i += 1

        return [mr for mr in merge_ranges if mr is not None]

    def resolve_merge_range_conflict(self, merge_range, next_merge_range, complex_ngrams, simple_ngrams, merge_candidates):
        """Resolve conflict between two merge ranges based on PMI. Return the conflicting range to be removed."""
        merge_range_pmi = self.get_merge_range_pmi(merge_range, complex_ngrams, simple_ngrams, merge_candidates)
        next_merge_range_pmi = self.get_merge_range_pmi(next_merge_range, complex_ngrams, simple_ngrams, merge_candidates)

        if merge_range_pmi > next_merge_range_pmi:
            return next_merge_range  # Remove the next merge range
        elif next_merge_range_pmi > merge_range_pmi:
            return merge_range  # Remove the current merge range
        else:
            #logger.warning("PMI is equal, conflict resolution not implemented.")
            return None

    def start_boundary(self, size=2):
        # ('<#', '<#')
        return tuple([self.start_boundary_token]*size)

    def end_boundary(self, size=2):
        # ('#>', '#>')
        return tuple([self.end_boundary_token]*size)
    
    def start_boundary_gaps(self):
        # ('<#', '-') and ('-', '<#')
        return [(self.start_boundary_token, self.gap_ch), (self.gap_ch, self.start_boundary_token)]

    def end_boundary_gaps(self):
        # ('#>', '-') and ('-', '#>')
        return [(self.end_boundary_token, self.gap_ch), (self.gap_ch, self.end_boundary_token)]

    def pad(self, ngram_size, alignment=None, pad_ch=PAD_CH_DEFAULT, pad_n=None):
        self.pad_ch = pad_ch
        if alignment is None:
            alignment = self.alignment
        if pad_n is None:
            pad_n = max(0, ngram_size - 1)
        self.alignment = [self.start_boundary()] * pad_n + alignment + [self.end_boundary()] * pad_n
        self.update()
        return self.alignment

    def remove_padding(self):
        """Removes non-complex pad ngrams from beginning and end of alignment.
        Removes both fully boundary alignments ('<#', '<#') and ('#>', '#>')
        as well as boundary gap alignments, e.f. ('<#', '-') and ('-', '#>')
        """
        # ('<#', '<#')
        start_boundary = self.start_boundary()
        # ('<#', '-') and ('-', '<#')
        start_boundary_gap_right, start_boundary_gap_left = self.start_boundary_gaps()
        start_pad_i_left, start_pad_i_right = 0, 0
        while self.alignment[start_pad_i_left] in (start_boundary, start_boundary_gap_right):
            start_pad_i_left += 1
        while self.alignment[start_pad_i_right] in (start_boundary, start_boundary_gap_left):
            start_pad_i_right += 1
        start_pad_i = max(start_pad_i_left, start_pad_i_right)
        end_pad_i_left, end_pad_i_right = -1, -1
        end_boundary = self.end_boundary()
        end_boundary_gap_right, end_boundary_gap_left = self.end_boundary_gaps()
        while self.alignment[end_pad_i_left] in (end_boundary, end_boundary_gap_right):
            end_pad_i_left -= 1
        while self.alignment[end_pad_i_right] in (end_boundary, end_boundary_gap_left):
            end_pad_i_right -= 1
        end_pad_i = min(end_pad_i_left, end_pad_i_right)
        align_length = len(self.alignment)
        if end_pad_i > -2:
            end_pad_i = align_length
        self.alignment = self.alignment[start_pad_i:end_pad_i + 1]
        self.padded = False
        # If the input sequence was padded, also modify self.seq1, and self.seq2
        if self.input_seq_is_padded():
            if end_pad_i_left < -1:
                self.seq1 = self.seq1[start_pad_i_left:end_pad_i_left + 1]
            else:
                self.seq1 = self.seq1[start_pad_i_left:]
            
            if end_pad_i_right < -1:
                self.seq2 = self.seq2[start_pad_i_right:end_pad_i_right + 1]
            else:
                self.seq2 = self.seq2[start_pad_i_right:]

    def input_seq_is_padded(self):
        """Returns True if either input sequence is padded with boundary tokens on either side."""
        return (
            Ngram(self.seq1[0]).is_boundary(self.pad_ch)
            or 
            Ngram(self.seq2[0]).is_boundary(self.pad_ch)
            or
            Ngram(self.seq1[-1]).is_boundary(self.pad_ch)
            or
            Ngram(self.seq2[-1]).is_boundary(self.pad_ch)
        )

    def map_to_seqs(self):
        """Maps aligned pair indices to their respective sequence indices
        e.g.
        self.alignment = [('ʃ', 's'), ('ˈa', 'ˈɛ'), ('p', '-'), ('t', 't'), ('e', '-')]
        map1 = {0:0, 1:1, 2:2, 3:3, 4:4}
        map2 = {0:0, 1:1, 2:None, 3:3, 4:None}
        """
        # Update current alignment length, in case it was padded or otherwise modified
        self.length = len(self.alignment)

        map1, map2 = {}, {}
        adjust_gap1, adjust_gap2 = 0, 0
        adjust_complex1, adjust_complex2 = 0, 0
        adjust_complex_start = 0
        n_complex = sum([1 for left, right in self.alignment if Ngram(left).size > 1 or Ngram(right).size > 1])
        for i in range(self.length):
            # Skip alignment positions containing only boundary padding, e.g. ('<#', '<#')
            if i == 0 and self.alignment[i] == self.start_boundary():
                adjust_gap1 += 1
                adjust_gap2 += 1
                map1[i] = None
                map2[i] = None
                continue
            # ('#>', '#>')
            elif i == self.length - 1 and self.alignment[i] == self.end_boundary():
                continue

            seg1_i = i - adjust_gap1 + adjust_complex1
            seg2_i = i - adjust_gap2 + adjust_complex2

            if i >= self.length:
                if n_complex > 1 and ((i + 1 + adjust_complex_start) - self.length) == (n_complex - adjust_complex_start):
                    continue
                last_index = self.length - 1
                left, right = self.alignment[last_index]
                left_ngram, right_ngram = Ngram(left), Ngram(right)

                if left == self.gap_ch or self.pad_ch in left:
                    pass
                elif left_ngram.size > 1 and len(map1[last_index]) < left_ngram.size:
                    if self.pad_ch not in left_ngram.ngram[len(map1[last_index])]:
                        map1[last_index].append(min(seg1_i, map1[last_index][-1] + 1))
                        adjust_complex1 += 1

                if right == self.gap_ch or self.pad_ch in right:
                    pass
                elif right_ngram.size > 1 and len(map2[last_index]) < right_ngram.size:
                    if self.pad_ch not in right_ngram.ngram[len(map2[last_index])]:
                        map2[last_index].append(min(seg2_i, map2[last_index][-1] + 1))
                        adjust_complex1 += 1

            else:
                seg1, seg2 = self.alignment[i]
                if seg1 == self.gap_ch:
                    map1[i] = None
                    adjust_gap1 += 1
                elif self.pad_ch in seg1:
                    map1[i] = None
                    adjust_gap1 += 1
                else:
                    map1[i] = []
                    ngram = Ngram(seg1)
                    if ngram.size > 1: #and i < self.length - 1:
                        adjust_complex1 += ngram.size - 1
                        adjust_ngram = 0
                        for n in range(seg1_i, min(seg1_i + ngram.size, len(self.seq1) + 1)):
                            if self.pad_ch not in ngram.ngram[n - seg1_i]:
                                map1[i].append(n - adjust_ngram)
                            else:
                                adjust_gap1 += 1
                                adjust_ngram += 1
                                if i == 0:
                                    adjust_complex_start += 1
                    else:
                        map1[i].append(seg1_i)

                if seg2 == self.gap_ch:
                    map2[i] = None
                    adjust_gap2 += 1
                elif self.pad_ch in seg2:
                    map2[i] = None
                    adjust_gap2 += 1
                else:
                    map2[i] = []
                    ngram = Ngram(seg2)
                    if ngram.size > 1: #and i < self.length - 1:
                        adjust_complex2 += ngram.size - 1
                        adjust_ngram = 0
                        for n in range(seg2_i, min(seg2_i + ngram.size, len(self.seq2) + 1)):
                            if self.pad_ch not in ngram.ngram[n - seg2_i]:
                                map2[i].append(n - adjust_ngram)
                            else:
                                adjust_gap2 += 1
                                adjust_ngram += 1
                                if i == 0:
                                    adjust_complex_start += 1

                    else:
                        map2[i].append(seg2_i)

        # Check that all sequence units were mapped to alignment positions
        try:
            assert sum(len(value) for value in map1.values() if value is not None) == len(self.seq1)
            assert sum(len(value) for value in map2.values() if value is not None) == len(self.seq2)
        except AssertionError as exc:
            raise AssertionError(f"Error re-mapping aligned sequences: {self.alignment}") from exc

        return map1, map2

    def add_phon_env(self, env_func=get_phon_env):
        """Adds the phonological environment value of segments to an alignment
        e.g.
        [('m', 'm'), ('j', '-'), ('ɔu̯', 'ɪ'), ('l̥', 'l'), ('k', 'ç')])
        becomes
        [(('m', '#S<'), 'm'), (('j', '<S<'), '-'), (('ɔu̯', '<S>'), 'ɪ'), (('l̥', '>S>'), 'l'), (('k', '>S#'), 'ç')]
        """
        self.phon_env = True
        word1_aligned, word2_aligned = tuple(zip(*self.alignment))
        word1_aligned = list(word1_aligned)  # TODO use as tuple if possible, but this might disrupt some behavior elsewhere if lists are expected
        seq_map = self.seq_map[0]

        for align_i in seq_map:
            if seq_map[align_i] is not None:
                # for complex ngrams, consider only the preceding context of the first component segment and the following context of the last component segment
                # therefore skip computing phon envs for any segs in between first and last within an alignment position
                for j, seg_j, in enumerate(list(set(seq_map[align_i][:1] + seq_map[align_i][-1:]))):
                    phon_env = env_func(self.seq1, seg_j)
                    target = word1_aligned[align_i]
                    if isinstance(target, str):
                        word1_aligned[align_i] = word1_aligned[align_i], phon_env
                    else:
                        word1_aligned[align_i] = list(word1_aligned[align_i])  # needed because tuples don't support item assignment
                        word1_aligned[align_i][j] = word1_aligned[align_i][j], phon_env
                        word1_aligned[align_i] = PhonEnvNgram(word1_aligned[align_i]).ngram_w_context
                if len(seq_map[align_i]) > 1:
                    # Extract only preceding and following contexts from complex ngrams
                    # e.g. (('s', '#|S|>'), ('k', '>|S|<')) -> (('s', 'k'), '#|S|<')
                    word1_aligned[align_i] = PhonEnvNgram(word1_aligned[align_i]).ngram_w_context
                    # segs = tuple(pair[0] for pair in word1_aligned[align_i])
                    # pre_env = word1_aligned[align_i][0][-1].split('|')[0]
                    # post_env = word1_aligned[align_i][-1][-1].split('|')[-1]
                    # phon_env = f'{pre_env}|S|{post_env}'
                    # word1_aligned[align_i] = segs, phon_env

        # TODO use as tuple if possible, but this might disrupt some behavior elsewhere if lists are expected
        return list(zip(word1_aligned, word2_aligned))

    def reverse(self):
        return ReversedAlignment(self)

    def update(self):
        self.seq_map = self.map_to_seqs()
        self.length = len(self.alignment)

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
        self.pad_ch = alignment.pad_ch
        self.added_penalty_dict = alignment.added_penalty_dict
        self.kwargs = alignment.kwargs
        self.n_best = [(reverse_alignment(alignment_n), cost) for alignment_n, cost in alignment.n_best]
        self.alignment = reverse_alignment(alignment.alignment)

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
    def __init__(self, alignment, index, gap_ch=GAP_CH_DEFAULT):
        self.alignment = alignment
        self.gap_ch = gap_ch
        self.index = index
        self.pair = self.alignment.alignment[self.index] if isinstance(self.alignment, Alignment) else self.alignment[self.index]
        self.ngrams = [Ngram(pos) for pos in self.pair]
        self.shape = self.get_shape()
        self.is_complex = self._is_complex()
        self.is_gappy = self.contains_gap()

    def prev_pair(self):
        if self.index > 0:
            return AlignedPair(self.alignment, self.index - 1, self.gap_ch)
        else:
            return None

    def next_pair(self):
        try:
            return AlignedPair(self.alignment, self.index + 1, self.gap_ch)
        except IndexError:
            return None

    def context(self):
        return self.prev_pair(), self.next_pair()

    def contains_gap(self):
        return any(ngram.is_gappy(self.gap_ch) for ngram in self.ngrams)

    def get_shape(self):
        ngram1, ngram2 = self.ngrams
        ngram_size1 = ngram1.size
        ngram_size2 = ngram2.size
        return ngram_size1, ngram_size2

    def _is_complex(self):
        ngram_size1, ngram_size2 = self.shape
        if ngram_size1 > 1 or ngram_size2 > 1:
            return True
        return False


class Gap(AlignedPair):
    def __init__(self, alignment, index, **kwargs):
        super().__init__(alignment, index, **kwargs)
        self.gap_i = self.pair.index(self.gap_ch)
        self.seg_i = abs(self.gap_i - 1)
        self.segment = self.pair[self.seg_i]

    def bigram_slices(self):
        slices = [
            # e.g. kt ~ ʧ
            # ('-', 'k'), ('ʧ', 't') or ('k', '-'), ('t', 'ʧ')
            (self.index, self.index + 2),
            # ('ʧ', 'k'), ('-', 't') or ('k', 'ʧ'), ('t', '-')
            (self.index - 1, self.index + 1)
        ]
        return slices

    def bigram_aligned_segs(self, bigram):
        """Given a bigram tuple that the gap forms a larger alignment unit with,
        this function identifies the bigram segment aligned to the gap
        and the bigram segment aligned to another segment"""
        gap_aligned_corr_i = bigram.index(self.segment)
        seg_aligned_corr_i = abs(gap_aligned_corr_i - 1)
        gap_aligned_corr = bigram[gap_aligned_corr_i]
        seg_aligned_corr = bigram[seg_aligned_corr_i]
        return gap_aligned_corr, seg_aligned_corr


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


def get_bigrams(complex_ngrams):
    """Extract bigrams and their aligned segments from complex_ngrams."""
    l1_bigrams = [ngram for ngram in complex_ngrams if ngram.size > 1]
    l2_bigrams = [nested_ngram for ngram in complex_ngrams
                  for nested_ngram in complex_ngrams[ngram]
                  if nested_ngram.size > 1]
    l1_bigram_segs = set(seg for bigram in l1_bigrams for seg in bigram.ngram)
    l2_bigram_segs = set(seg for bigram in l2_bigrams for seg in bigram.ngram)

    return l1_bigrams, l2_bigrams, l1_bigram_segs, l2_bigram_segs


def visual_align(alignment, gap_ch=GAP_CH_DEFAULT, null=NULL_CH_DEFAULT, phon_env=False):
    """Renders list of aligned segment pairs as an easily interpretable
    alignment string, with <{NULL_CH_DEFAULT}> representing null segments,
    e.g.:
    visual_align([('z̪', 'ɡ'),('vʲ', 'v'),('ɪ', 'j'),('-', 'ˈa'),('z̪', 'z̪'),('d̪', 'd'),('ˈa', 'a')])
    = 'z̪-ɡ / vʲ-v / ɪ-j / {NULL_CH_DEFAULT}-ˈa / z̪-z̪ / d̪-d / ˈa-a' """

    if isinstance(alignment, Alignment) and gap_ch != alignment.gap_ch:
        raise ValueError(f'Gap character "{gap_ch}" does not match gap character of alignment "{alignment.gap_ch}"')

    alignment = get_alignment_iter(alignment, phon_env)
    if phon_env:
        raise NotImplementedError('TODO needs to be updated for phon_env')  # TODO

    a = []
    for pair in alignment:
        pair = list(pair)
        for i, seg in enumerate(pair):
            if isinstance(seg, tuple):
                flattened_seg = flatten_tuple(seg)
                pair[i] = SEG_JOIN_CH.join(flattened_seg)

        seg1, seg2 = pair
        if gap_ch not in pair:
            a.append(f'{seg1}-{seg2}')
        else:
            if seg1 == gap_ch:
                a.append(f'{null}-{seg2}')
            else:
                a.append(f'{seg1}-{null}')

    return ' / '.join(a)


def undo_visual_align(visual_alignment, gap_ch=GAP_CH_DEFAULT):
    """Reverts a visual alignment to a list of tuple segment pairs"""
    seg_pairs = visual_alignment.split(' / ')
    seg_pairs = [tuple(pair.split(gap_ch)) for pair in seg_pairs]
    return seg_pairs


def flatten_tuple(nested_tuple):
    flattened = []
    for item in nested_tuple:
        if isinstance(item, tuple):
            flattened.extend(flatten_tuple(item))
        else:
            flattened.append(item)
    return flattened
