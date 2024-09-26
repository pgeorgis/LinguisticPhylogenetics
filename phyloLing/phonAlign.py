import importlib
import logging
from collections.abc import Iterable
from collections import defaultdict
from math import inf, log
from statistics import mean

from constants import (END_PAD_CH, GAP_CH_DEFAULT, NULL_CH_DEFAULT,
                       PAD_CH_DEFAULT, SEG_JOIN_CH, START_PAD_CH)
from nwunschAlign import best_alignment
from phonUtils.phonEnv import get_phon_env
from phonUtils.phonSim import phone_sim
from phonUtils.segment import _toSegment
from utils.distance import Distance
from utils.sequence import Ngram, PhonEnvNgram
from utils.utils import validate_class


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


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
                 alignment=None,
                 lang1=None,
                 lang2=None,
                 cost_func=AlignmentCost,
                 added_penalty_dict=None,
                 gap_ch=GAP_CH_DEFAULT,
                 gop=-0.2,  # TODO possibly need to recalibrate ** changed from -0.3 to -0.2 when changing to PMI log base 2, given that those PMI values will be 69% of value of PMI with log base math.e
                 pad_ch=PAD_CH_DEFAULT,
                 n_best=1,
                 phon_env=False,
                 **kwargs
                 ):
        f"""Produces a pairwise alignment of two phone sequences.

        Args:
            seq1 (phyloLing.Word or str): first phone sequence
            seq2 (phyloLing.Word or str): second phone sequence
            alignment (list, optional): list of tuples indicating predetermined alignment
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
        self.seq1 = self.prepare_seq(seq1, lang1)
        self.seq2 = self.prepare_seq(seq2, lang2)

        # Designate alignment parameters
        self.gap_ch = gap_ch
        self.gop = gop
        self.pad_ch = pad_ch
        self.cost_func = cost_func
        self.added_penalty_dict = added_penalty_dict
        self.kwargs = kwargs

        # Perform alignment
        if not alignment:
            _, self.n_best = self.align(n_best)
            self.alignment = self.n_best[0][0][:]
            self.cost = self.n_best[0][-1]
        else:
            self.imported_alignment = alignment
            self.alignment = self.import_alignment(alignment)
            self.cost = None # TODO evaluate later if needed

        # Save length and cost of single best alignment
        self.length = len(self.alignment)
        self.original_length = self.length

        # Map aligned pairs to respective sequence indices
        self.padded = self.is_padded()
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
        validate_class((seq1,), ((phyloLing.Word, str, list),))
        validate_class((seq2,), ((phyloLing.Word, str, list),))
        for lang in (lang1, lang2):
            if lang:  # skip if None
                validate_class((lang,), (phyloLing.Language,))

    def prepare_seq(self, seq, lang):
        phyloLing = importlib.import_module('phyloLing')
        if isinstance(seq, list):
            return seq
        if isinstance(seq, phyloLing.Word):
            word = seq
        elif isinstance(seq, str):
            word = phyloLing.Word(seq, language=lang)
        return word.segments
    
    def import_alignment(self, alignment, remove_non_sequential_complex_alignments=True):
        """Import and pre-specified alignment of seq1 and seq2."""
        # Initialize alignment dicts
        aligned_seq1 = defaultdict(lambda:[])
        aligned_seq2 = defaultdict(lambda:[])
        
        # Fill in aligned dicts
        for idx1, idx2 in alignment:
            if idx1 is not None:
                aligned_seq1[idx1].append(idx2)
            if idx2 is not None:
                aligned_seq2[idx2].append(idx1)
        
        # Add gaps
        for idx1, _ in enumerate(self.seq1):
            if idx1 not in aligned_seq1:
                aligned_seq1[idx1].append(None)
        for idx2, _ in enumerate(self.seq2):
            if idx2 not in aligned_seq2:
                aligned_seq2[idx2].append(None)
        
        if remove_non_sequential_complex_alignments:
            """IBM alignment can produce alignments with units aligned in multiple places, e.g.
            [(0, 0), (1, 1), (2, 9), (3, 3), (4, 4), (5, 7), (6, 4), (7, 9), (8, 10)]
            where idx 9 in seq2 is aligned to both idx 2 and idx 7 in seq1
            
            Disallow such complex/double alignments if they are not consecutive
            """
            for j, i_s in aligned_seq2.items():
                i_s.sort()
                if len(i_s) > 1:
                    i_i = 0
                    while i_i < len(i_s)-1:
                        i = i_s[i_i]
                        i_next = i_s[i_i + 1]
                        if abs(i - i_next) > 1:
                            anchor = mean(i_s)
                            
                            # Find which of the two is more distant from anchor
                            more_distant_i = max(i_next, i, key=lambda x: abs(x - anchor))
                            aligned_seq2[j].remove(more_distant_i)
                            if len(aligned_seq1[more_distant_i]) > 1:
                                aligned_seq1[more_distant_i].remove(j)
                            else:
                                aligned_seq1[more_distant_i] = [None]
                        i_i += 1

        # Alignment
        aligned_units = []
        aligned_units1 = set()
        aligned_units2 = set()
        last_j = None
        for i in range(max(len(self.seq1), len(self.seq2))):
            if i in aligned_seq1:

                unit1 = self.seq1[i]
                if i == 0 or (last_j is not None and last_j not in aligned_seq1[i]):
                    aligned_units.append([[[unit1]], [i]])
                elif i > 0 and (last_j is not None and last_j in aligned_seq1[i]):
                    aligned_units[-1][0][0].append(unit1)
                elif i > 0 and (last_j is None and last_j in aligned_seq1[i]):
                    aligned_units.append([[[unit1]], [i]])
                aligned_units1.add(i)
            for j in aligned_seq1[i]:
                if len(aligned_units[-1][0]) == 1:
                    aligned_units[-1][0].append([])
                if j is not None:
                    if last_j != j:
                        aligned_unit2 = self.seq2[j]
                        aligned_units[-1][0][-1].append(aligned_unit2)
                        aligned_units[-1][-1].append(j)
                        aligned_units2.add(j)
                        last_j = j
                else:
                    aligned_units[-1][0][-1].append(self.gap_ch)
                    #aligned_units.append((unit1, self.gap_ch, (i, )))
            if i not in aligned_units1 and i < len(self.seq1):
                breakpoint()
        for j in range(len(self.seq2)):
            if j not in aligned_units2:
                unit2 = self.seq2[j]
                aligned_units.append([[[self.gap_ch], [unit2]], [j]])
        aligned_units.sort(key=lambda x: (mean([min(x[-1]), max(x[-1])]), x[0][0][0], x[0][0][-1]))
        aligned_units = [(Ngram(unit[0][0]).undo(), Ngram(unit[0][-1]).undo()) for unit in aligned_units]
        
        # Move boundary alignments to edges
        start_boundary = self.start_boundary()
        end_boundary = self.end_boundary()
        if start_boundary in aligned_units and aligned_units.index(start_boundary) != 0:
            aligned_units.remove(start_boundary)
            aligned_units.insert(0, start_boundary)
        if end_boundary in aligned_units and aligned_units.index(end_boundary) != len(aligned_units)-1:
            aligned_units.remove(end_boundary)
            aligned_units.append(end_boundary)
        
        return aligned_units

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
        if self.added_penalty_dict:  # TODO this could be a separate class method
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

    def start_boundary(self):
        # ('<#', '<#')
        return (f'{START_PAD_CH}{self.pad_ch}', f'{START_PAD_CH}{self.pad_ch}')

    def end_boundary(self):
        # ('#>', '#>')
        return (f'{self.pad_ch}{END_PAD_CH}', f'{self.pad_ch}{END_PAD_CH}')

    def pad(self, ngram_size, alignment=None, pad_ch=PAD_CH_DEFAULT, pad_n=None):
        self.pad_ch = pad_ch
        if alignment is None:
            alignment = self.alignment
        if pad_n is None:
            pad_n = max(0, ngram_size - 1)
        self.alignment = [self.start_boundary()] * pad_n + alignment + [self.end_boundary()] * pad_n
        self.update()
        self.padded = True
        return self.alignment

    def remove_padding(self, no_update=False):
        start_pad_i = 0
        start_pad = self.start_boundary()
        while self.alignment[start_pad_i] == start_pad:
            start_pad_i += 1
        align_length = len(self.alignment)
        end_pad_i = align_length - 1
        end_pad = self.end_boundary()
        while self.alignment[end_pad_i] == end_pad:
            end_pad_i -= 1
        self.alignment = self.alignment[start_pad_i:end_pad_i + 1]
        self.padded = False
        # If the input sequence was padded, also modify self.seq1, and self.seq2
        if self.is_padded():
            n_end_pad = align_length - 1 - end_pad_i
            if n_end_pad > 0:
                self.seq1 = self.seq1[start_pad_i:-n_end_pad]
                self.seq2 = self.seq2[start_pad_i:-n_end_pad]
            else:
                self.seq1 = self.seq1[start_pad_i:]
                self.seq2 = self.seq2[start_pad_i:]
        if not no_update:
            self.update()
    
    def is_padded(self):
        return Ngram(self.seq1[0]).is_boundary(self.pad_ch)

    def map_to_seqs(self):
        """Maps aligned pair indices to their respective sequence indices
        e.g.
        self.alignment = [('ʃ', 's'), ('ˈa', 'ˈɛ'), ('p', '-'), ('t', 't'), ('e', '-')]
        map1 = {0:0, 1:1, 2:2, 3:3, 4:4}
        map2 = {0:0, 1:1, 2:None, 3:3, 4:None}
        """
        # Remove extra padding
        if self.padded:
            # Set no_update=True to avoid a recursion error, because self.update() also calls this function
            self.remove_padding(no_update=True)
        
        # Update current alignment length, in case it was padded or otherwise modified
        self.length = len(self.alignment)

        map1, map2 = {}, {}
        adjust_gap1, adjust_gap2 = 0, 0
        adjust_complex1, adjust_complex2 = 0, 0
        adjust_complex_start = 0
        n_complex = sum([1 for left, right in self.alignment if Ngram(left).size > 1 or Ngram(right).size > 1])
        for i in range(max(self.length, self.original_length)):
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
                        for n in range(seg1_i, min(seg1_i + ngram.size, len(self.seq1))):
                            if self.pad_ch not in ngram.ngram[n - seg1_i]:
                                map1[i].append(n - adjust_ngram)
                            elif i == 0 and len(self.seq1) == 1:
                                map1[i].append(n - adjust_ngram)
                            elif i == 1 and len(self.seq1) == 1 and self.alignment[0] == self.start_boundary():
                                # e.g. [('<#', '<#'), (('<#', 'ɑ'), '<#'), ...] # TODO ensure that this is actually a valid alignment and not a bug
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
                        for n in range(seg2_i, min(seg2_i + ngram.size, len(self.seq2))):
                            if self.pad_ch not in ngram.ngram[n - seg2_i]:
                                map2[i].append(n - adjust_ngram)
                            elif i == 0 and len(self.seq2) == 1:
                                map2[i].append(n - adjust_ngram)
                            elif i == 1 and len(self.seq2) == 1 and self.alignment[0] == self.start_boundary():
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
    def __init__(self, alignment, index):
        self.alignment = alignment
        self.gap_ch = self.alignment.gap_ch
        self.index = index
        self.pair = self.alignment.alignment[self.index]

    def prev_pair(self):
        if self.index > 0:
            return AlignedPair(self.alignment, self.index - 1)
        else:
            return None

    def next_pair(self):
        try:
            return AlignedPair(self.alignment, self.index + 1)
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
