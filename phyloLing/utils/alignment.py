from collections import defaultdict
from itertools import zip_longest
from math import inf
import numpy as np
from constants import GAP_CH_DEFAULT
from utils.sequence import Ngram

def needleman_wunsch_extended(seq1: list,
                              seq2: list,
                              align_cost: dict,
                              gap_cost: dict,
                              default_gop: int | float,
                              gap_ch: str=GAP_CH_DEFAULT,
                              allow_complex: bool=True,
                              maximize_score: bool=False,
                              ):
    f"""Aligns two sequences with an extended version of the Needleman-Wunsch algorithm, optionally allowing for complex alignments.

    Args:
        seq1 (list): List of units in sequence 1.
        seq2 (list): List of units in sequence 2.
        align_cost (dict): Dictionary of alignment costs between units.
        gap_cost (dict): Dictionary of costs to align a unit with a gap.
        default_gop (int | float): Default gap opening penalty in case pair is not in gap_cost dictionary.
        gap_ch (str, optional): Gap character. Defaults to "{GAP_CH_DEFAULT}".
        allow_complex (bool, optional): Allow complex alignments (one-to-many, many-to-one, many-to-many, etc.). Defaults to True.
        maximize_score (bool, optional): If True, optimal alignments are computed by maximizing the alignment score; else by minimizing the alignment cost. Defaults to False.

    Returns:
        (cost, alignment, (seq_map1, seq_map2)) (tuple): Tuple containing the alignment cost/score, alignment (list of tuples), and a tuple aligned sequence map dicts.
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
                ngram_unit1 = seq2ngram(seq1[i-k:i])
                for l in range(1, j + 1):
                    max_size = max(j-(j-l), i-(i-k), 1)
                    if max_size > 1 and allow_complex is False:
                        continue
                    ngram_unit2 = seq2ngram(seq2[j-l:j])
                    cost = align_cost.get(ngram_unit1, {}).get(ngram_unit2, default_gop * max_size)
                    score = dp[i-k][j-l] + cost
                    if score_is_better(score, best_score):
                        best_score = score
                        best_move = (k, l)

            # Align seq1 to a gap
            for k in range(1, i + 1):
                ngram_unit1 = seq2ngram(seq1[i-k:i])
                size = max(1, (i-(i-k)))
                if size > 1 and allow_complex is False:
                    continue
                cost = gap_cost.get(ngram_unit1, {}).get(gap_ch, default_gop * size)
                score = dp[i-k][j] + cost
                if score_is_better(score, best_score):
                    best_score = score
                    best_move = (k, 0)

            # Align seq2 to a gap
            for l in range(1, j + 1):
                ngram_unit2 = seq2ngram(seq2[j-l:j])
                size = max(1, (j-(j-l)))
                if size > 1 and allow_complex is False:
                    continue
                cost = gap_cost.get(gap_ch, {}).get(ngram_unit2, default_gop * size)
                score = dp[i][j-l] + cost
                if score_is_better(score, best_score):
                    best_score = score
                    best_move = (0, l)

            dp[i][j] = best_score
            traceback[i][j] = best_move  # Store the move in traceback

    # Backtrack to find the alignment
    aligned_seq1, aligned_seq2 = [], []
    seq_map1, seq_map2 = defaultdict(lambda:[]), defaultdict(lambda:[])
    i, j = n, m
    while i > 0 or j > 0:
        move = traceback[i][j]
        if move[0] > 0 and move[1] > 0:
            # Align subsequences of both seq1 and seq2
            aligned_seq1.append(seq2ngram(seq1[i-move[0]:i]))
            aligned_seq2.append(seq2ngram(seq2[j-move[1]:j]))
            seq_map1[len(seq_map1)].extend([x for x in range(i-move[0], i)])
            seq_map2[len(seq_map2)].extend([x for x in range(j-move[1], j)])
            i -= move[0]
            j -= move[1]
        elif move[0] > 0:
            # Align subsequence of seq1 to a gap
            aligned_seq1.append(seq2ngram(seq1[i-move[0]:i]))
            aligned_seq2.append(seq2ngram([gap_ch]))
            seq_map1[len(seq_map1)].extend([x for x in range(i-move[0], i)])
            seq_map2[len(seq_map2)] = None
            i -= move[0]
        else:
            # Align subsequence of seq2 to a gap
            aligned_seq1.append(seq2ngram([gap_ch]))
            aligned_seq2.append(seq2ngram(seq2[j-move[1]:j]))
            seq_map1[len(seq_map1)] = None
            seq_map2[len(seq_map2)].extend([x for x in range(j-move[1], j)])
            j -= move[1]

    # Reverse the aligned sequences to be in correct order
    aligned_seq1.reverse()
    aligned_seq2.reverse()
    alignment = list(zip(aligned_seq1, aligned_seq2))

    # Reverse the sequence maps
    seq_map1, seq_map2 = dict(seq_map1), dict(seq_map2)
    adj_seqmap1, adj_seqmap2 = {}, {}
    for i in range(len(alignment)-1, -1, -1):
        adj_seqmap1[abs(i-(len(alignment)-1))] = seq_map1[i]
        adj_seqmap2[abs(i-(len(alignment)-1))] = seq_map2[i]

    return dp[n][m], alignment, (adj_seqmap1, adj_seqmap2)


def to_unigram_alignment(bigram, fillvalue=GAP_CH_DEFAULT):
    unigrams = [[] for _ in range(len(bigram))]
    for i, pos in enumerate(bigram):
        if isinstance(pos, str):
            unigrams[i].append(pos)
        elif isinstance(pos, tuple):
            unigrams[i].extend(pos)

    return list(zip_longest(*unigrams, fillvalue=fillvalue))
