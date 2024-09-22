"""MIT License

Copyright (c) 2018 mmtechslv

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Modified by Philip Georgis from:
    https://github.com/mmtechslv/nwunch
"""
from utils.utils import dict_tuplelist


def return_alignment(ALIGNMENTS):
    final_alignments = []
    for final_alignment in ALIGNMENTS:
        final_alignments.append(list(zip(final_alignment[1], final_alignment[0])))
    return final_alignments


def find_each_path(c_i, c_j, ALN_PATHWAYS, MATRIX, path=''):  # Nested function to discover new aln pathways
    # global ALN_PATHWAYS
    i = c_i
    j = c_j
    if i == 0 and j == 0:
        ALN_PATHWAYS.append(path)
        return 2
    dir_t = len(MATRIX[i][j][1])
    while dir_t <= 1:
        n_dir = MATRIX[i][j][1][0] if (i != 0 and j != 0) else (1 if i == 0 else (3 if j == 0 else 0))
        path = path + str(n_dir)
        if n_dir == 1:
            j = j - 1
        elif n_dir == 2:
            i = i - 1
            j = j - 1
        elif n_dir == 3:
            i = i - 1
        dir_t = len(MATRIX[i][j][1])
        if i == 0 and j == 0:
            ALN_PATHWAYS.append(path)
            return 3
    if dir_t > 1:
        for dir_c in range(dir_t):
            n_dir = MATRIX[i][j][1][dir_c] if (i != 0 and j != 0) else (1 if i == 0 else (3 if j == 0 else 0))
            tmp_path = path + str(n_dir)
            if n_dir == 1:
                n_i = i
                n_j = j - 1
            elif n_dir == 2:
                n_i = i - 1
                n_j = j - 1
            elif n_dir == 3:
                n_i = i - 1
                n_j = j
            find_each_path(n_i, n_j, ALN_PATHWAYS, MATRIX, tmp_path)
    return len(ALN_PATHWAYS)


def best_alignment(SEQUENCE_1, SEQUENCE_2, SCORES_DICT, DEFAULT_GAP_SCORE=-1, GAP_CHARACTER='-', GAP_SCORE_DICT={}, N_BEST=1):
    MATRIX_ROW_N = len(SEQUENCE_1) + 1  # Initiation Matrix Size (Rows)
    MATRIX_COLUMN_N = len(SEQUENCE_2) + 1  # Initiation Matrix Size (Columns)
    ALN_PATHWAYS = []  # Initiating List of Discovered aln Pathways
    MATRIX = [[[[None] for i in range(2)] for i in range(MATRIX_COLUMN_N)] for i in range(MATRIX_ROW_N)]  # Initiating Score Matrix

    for i in range(MATRIX_ROW_N):
        MATRIX[i][0] = [GAP_SCORE_DICT.get((i - 1, GAP_CHARACTER), DEFAULT_GAP_SCORE) * i, []]
    for j in range(MATRIX_COLUMN_N):
        MATRIX[0][j] = [GAP_SCORE_DICT.get((GAP_CHARACTER, j - 1), DEFAULT_GAP_SCORE) * j, []]

    # Main matrix filling loop
    for i in range(1, MATRIX_ROW_N):
        for j in range(1, MATRIX_COLUMN_N):
            score = SCORES_DICT[(i - 1, j - 1)]

            # Get dynamic gap scores for horizontal, vertical, and diagonal moves
            gap_score_h = GAP_SCORE_DICT.get((GAP_CHARACTER, j - 1), DEFAULT_GAP_SCORE)
            gap_score_v = GAP_SCORE_DICT.get((i - 1, GAP_CHARACTER), DEFAULT_GAP_SCORE)

            h_val = MATRIX[i][j - 1][0] + gap_score_h  # Horizontal gap (sequence 1 aligns with gap)
            d_val = MATRIX[i - 1][j - 1][0] + score    # Diagonal move (alignment of characters)
            v_val = MATRIX[i - 1][j][0] + gap_score_v  # Vertical gap (sequence 2 aligns with gap)

            o_val = [h_val, d_val, v_val]
            MATRIX[i][j] = [max(o_val), [i + 1 for i, v in enumerate(o_val) if v == max(o_val)]]  # h = 1, d = 2, v = 3

    # Matrix Evaulation [end]
    OVERALL_SCORE = MATRIX[i][j][0]
    score = OVERALL_SCORE
    l_i = i
    l_j = j
    ALIGNMENTS = []
    _ = find_each_path(i, j, ALN_PATHWAYS, MATRIX)
    aln_count = 0
    # Compiling alignments based on discovered matrix pathways
    for elem in ALN_PATHWAYS:
        i = l_i - 1
        j = l_j - 1
        side_aln = []
        top_aln = []
        step = 0
        aln_info = []
        for n_dir_c in range(len(elem)):
            n_dir = elem[n_dir_c]
            score = MATRIX[i + 1][j + 1][0]
            step = step + 1
            aln_info.append([step, score, n_dir])
            if n_dir == '2':
                side_aln.append(SEQUENCE_1[i])
                top_aln.append(SEQUENCE_2[j])
                i = i - 1
                j = j - 1
            elif n_dir == '1':
                side_aln.append(GAP_CHARACTER)
                top_aln.append(SEQUENCE_2[j])
                j = j - 1
            elif n_dir == '3':
                side_aln.append(SEQUENCE_1[i])
                top_aln.append(GAP_CHARACTER)
                i = i - 1
        aln_count = aln_count + 1
        side_aln.reverse()
        top_aln.reverse()
        ALIGNMENTS.append([top_aln, side_aln, elem, aln_info, aln_count])

    # Return N best alignments
    alignment_scores = {}
    for i, item in enumerate(ALIGNMENTS):
        a = item[3]
        alignment_pmi_scores = sum(item[1] for item in a)
        alignment_scores[i] = alignment_pmi_scores
    alignment_scores = dict_tuplelist(alignment_scores)
    best_N_alignments = [(return_alignment(ALIGNMENTS)[i], score) for i, score in alignment_scores[:N_BEST]]
    return best_N_alignments
