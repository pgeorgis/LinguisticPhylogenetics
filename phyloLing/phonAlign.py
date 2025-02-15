import logging
import re
from collections.abc import Iterable

from constants import (ALIGNED_PAIR_DELIMITER, ALIGNMENT_KEY_REGEX,
                       ALIGNMENT_POSITION_DELIMITER, END_PAD_CH,
                       GAP_CH_DEFAULT, NULL_CH_DEFAULT, PAD_CH_DEFAULT,
                       SEG_JOIN_CH, START_PAD_CH)
from phonUtils.phonEnv import get_phon_env
from utils import PhonemeMap
from utils.alignment import needleman_wunsch_extended, to_unigram_alignment
from utils.sequence import (Ngram, PhonEnvNgram, end_token, flatten_ngram,
                            pad_sequence, start_token)
from utils.utils import validate_class
from utils.word import Word

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_align_key(word1, word2):
    """Generate an alignment key string representing the IPA strings of the words to be aligned."""
    def _get_word_key(word):
        if isinstance(word, str):
            return word
        if isinstance(word, Word):
            return word.ipa
        else:
            raise TypeError
    word1_key = _get_word_key(word1)
    word2_key = _get_word_key(word2)
    key = f'/{word1_key}/ - /{word2_key}/'
    return key


class Alignment:
    def __init__(self,
                 seq1, seq2,
                 align_costs: PhonemeMap,
                 gap_ch=GAP_CH_DEFAULT,
                 gop=-5,
                 pad_ch=PAD_CH_DEFAULT,
                 pad_n=1,
                 allow_complex=True,
                 phon_env=False,
                 override_alignment=None,
                 override_seq_map=None,
                 override_cost=None,
                 **kwargs
                 ):
        """Produces a pairwise alignment of two phone sequences.

        Args:
            seq1 (Word | str): First phone sequence.
            seq2 (Word | str): Second phone sequence.
            align_costs (PhonemeMap): Dictionary of alignment costs or scores.
            gap_ch (str, optional): Gap character.
            gop (int, optional): Default gap opening penalty.
            pad_ch (str, optional): Pad character.
            pad_n (int, optional): Number of padding units to add to beginning and end of sequences.
            allow_complex (bool, optional): Allow complex alignments instead of only simple one-to-one alignments.
            phon_env (bool, optional): Adds phonological environment to alignment.
            override_alignment (list): Precomputed alignment to use instead of dynamically computing a new alignment from costs.
            override_seq_map (tuple): Precomputed sequence map associated with precomputed alignment.
            override_cost (float): Precomputed alignment cost associated with precomputed alignment.
        """

        # Verify that input arguments are of the correct types
        self.validate_args(seq1, seq2)

        # Set alignment key
        self.key = get_align_key(seq1, seq2)

        # Prepare the input sequences for alignment
        self.seq1 = self.prepare_seq(seq1)
        self.seq2 = self.prepare_seq(seq2)

        # Designate alignment parameters
        self.gap_ch = gap_ch
        self.gop = gop
        self.pad_ch = pad_ch
        self.start_boundary_token = self.get_start_boundary_token()
        self.end_boundary_token = self.get_end_boundary_token()
        self.align_costs: PhonemeMap = align_costs
        self.kwargs = kwargs

        if override_alignment is not None:
            # Load precomputed alignment; at least seq map must also be specified
            assert override_seq_map is not None
            self.alignment = override_alignment
            self.seq_map = override_seq_map
            self.cost = override_cost
            self.length = len(self.alignment)
        else:
            # Standard: Perform alignment and parse results
            self.alignment, self.seq_map, self.cost = self.align(
                allow_complex=allow_complex,
                pad_n=pad_n,
            )
            self.length = len(self.alignment)
            self.seq_map = self.validate_seq_map(*self.seq_map)

            # Compact boundary aligned gaps
            self.alignment = self.compact_boundary_gaps(self.alignment)
            self.update()

        # Phonological environment alignment
        self.phon_env = phon_env
        if self.phon_env:
            self.phon_env_alignment = self.add_phon_env()
        else:
            self.phon_env_alignment = None

    def validate_args(self, seq1, seq2):
        """Verifies that all input arguments are of the correct types"""
        validate_class((seq1,), ((Word, str),))
        validate_class((seq2,), ((Word, str),))

    def prepare_seq(self, seq):
        if isinstance(seq, Word):
            return seq.segments
        else:
            raise TypeError("Expected seq to be a Word object")  # TODO need to adjust validate_args

    def align(self, allow_complex=True, pad_n=1):
        """Align segments of two words word1 with segments of word2 according to an extended Needleman-Wunsch algorithm.

        Returns:
            tuple: alignment, sequence_maps, alignment_score
        """

        # Pad sequences
        if pad_n > 0:
            seq1 = pad_sequence(self.seq1, pad_ch=self.pad_ch, pad_n=1)
            seq2 = pad_sequence(self.seq2, pad_ch=self.pad_ch, pad_n=1)
        else:
            seq1, seq2 = self.seq1, self.seq2

        # Compute alignment using extended Needleman-Wunsch algorithm
        alignment_score, alignment, seq_maps = needleman_wunsch_extended(
            seq1=seq1,
            seq2=seq2,
            align_cost=self.align_costs,
            gap_cost=self.align_costs,
            gap_ch=self.gap_ch,
            default_gop=self.gop,
            allow_complex=allow_complex,
            maximize_score=True,
        )

        return alignment, seq_maps, alignment_score

    def postprocess_boundary_gaps(self, alignment):
        """Post-process certain boundary gap alignments."""
        # ('-', '#>'), ('#>', '-') -> ('#>', '#>')
        if alignment[-2:] in [
            [(self.gap_ch, self.end_boundary_token), (self.end_boundary_token, self.gap_ch)],
            [(self.end_boundary_token, self.gap_ch), (self.gap_ch, self.end_boundary_token)],
        ]:
            alignment = alignment[:-2]
            alignment.append((self.end_boundary_token, self.end_boundary_token))

        # ('-', '<#'), ('<#', '-') -> ('<#', '<#')
        if alignment[:2] in [
            [(self.gap_ch, self.start_boundary_token), (self.start_boundary_token, self.gap_ch)],
            [(self.start_boundary_token, self.gap_ch), (self.gap_ch, self.start_boundary_token)],
        ]:
            alignment = alignment[2:]
            alignment.insert(0, (self.start_boundary_token, self.start_boundary_token))

        # Move non-final ('#>', '-') or ('-', '#>') to end of alignment
        if (self.end_boundary_token, self.gap_ch) in alignment:
            if alignment.index((self.end_boundary_token, self.gap_ch)) != len(alignment) - 1:
                alignment.remove((self.end_boundary_token, self.gap_ch))
                alignment.append((self.end_boundary_token, self.gap_ch))
        if (self.gap_ch, self.end_boundary_token) in alignment:
            if alignment.index((self.gap_ch, self.end_boundary_token)) != len(alignment) - 1:
                alignment.remove((self.gap_ch, self.end_boundary_token))
                alignment.append((self.gap_ch, self.end_boundary_token))

        # Move non-initial ('<#', '-') or ('-', '<#') to start of alignment
        if (self.start_boundary_token, self.gap_ch) in alignment:
            if alignment.index((self.start_boundary_token, self.gap_ch)) != 0:
                alignment.remove((self.start_boundary_token, self.gap_ch))
                alignment.insert(0, (self.start_boundary_token, self.gap_ch))
        if (self.gap_ch, self.start_boundary_token) in alignment:
            if alignment.index((self.gap_ch, self.start_boundary_token)) != 0:
                alignment.remove((self.gap_ch, self.start_boundary_token))
                alignment.insert(0, (self.gap_ch, self.start_boundary_token))

        # Convert (('-', '#>'), '#>') or ('#>', ('#>', '-')) to ('#>', '#>')
        if alignment[-1] in [
            ((self.gap_ch, self.end_boundary_token), self.end_boundary_token),
            (self.end_boundary_token, (self.end_boundary_token, self.gap_ch)),
        ]:
            alignment[-1] = (self.end_boundary_token, self.end_boundary_token)

        # Convert  (('-', '<#'), '<#') or ('<#', ('<#', '-')) to ('<#', '<#')
        if alignment[0] in [
            ((self.gap_ch, self.start_boundary_token), self.start_boundary_token),
            (self.start_boundary_token, (self.start_boundary_token, self.gap_ch)),
        ]:
            alignment[0] = (self.start_boundary_token, self.start_boundary_token)

        return alignment

    def compact_boundary_gaps(self, complex_alignment):
        # Add compacting of boundary gap alignment in situations like:
        # (('ˈa', '#>'), ('ˈɐ̃', 'w̃')), ('-', '#>') -> (('ˈa', '#>'), ('ˈɐ̃', 'w̃', '#>')) (Catalan/Portuguese)

        # Do nothing if the alignment consists of a single unit
        # or if there are unmatched boundary tokens (which could occur if realigning a subsequence)
        if len(complex_alignment) == 1:
            return complex_alignment
        else:
            flat = flatten_ngram(complex_alignment)

        complex_alignment = self.postprocess_boundary_gaps(complex_alignment)

        if flat.count(end_token()) == 2:
            last_ngram = Ngram(complex_alignment[-1])
            if last_ngram.is_gappy(self.gap_ch) and last_ngram.is_boundary(self.pad_ch):
                penult_ngram = Ngram(complex_alignment[-2])
                end_boundary_gap = complex_alignment.pop()
                end_boundary_gap = Gap([end_boundary_gap], 0, gap_ch=self.gap_ch)
                if penult_ngram.is_boundary(self.pad_ch):
                    final_complex = [[], []]
                    final_complex[end_boundary_gap.gap_i].extend([x for x in Ngram(complex_alignment[-1][end_boundary_gap.gap_i]).ngram if x not in (self.gap_ch, self.end_boundary_token)])
                    final_complex[end_boundary_gap.seg_i].extend([x for x in Ngram(complex_alignment[-1][end_boundary_gap.seg_i]).ngram if x not in (self.gap_ch, self.end_boundary_token)])
                    final_complex[end_boundary_gap.seg_i].extend([x for x in Ngram(end_boundary_gap.pair).ngram if x != self.gap_ch])
                    final_complex[end_boundary_gap.gap_i].append(self.end_boundary_token)
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
                    complex_alignment.append((Ngram(final_complex[0]).undo(), Ngram(final_complex[-1]).undo()))

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
                        initial_complex[start_boundary_gap.seg_i].extend([x for x in Ngram(start_boundary_gap.pair).ngram if x != self.gap_ch])
                        initial_complex[start_boundary_gap.seg_i].extend([x for x in Ngram(complex_alignment[1][start_boundary_gap.seg_i]).ngram if x not in (self.gap_ch, self.start_boundary_token)])
                        initial_complex[start_boundary_gap.gap_i].insert(0, self.start_boundary_token)
                        complex_alignment = [(Ngram(initial_complex[0]).undo(), Ngram(initial_complex[-1]).undo())] + complex_alignment[2:]
                    else:
                        breakpoint()
                        bp = 4
                        initial_complex[start_boundary_gap.gap_i].extend([x for x in complex_alignment[1][start_boundary_gap.gap_i] if x != self.gap_ch])
                        initial_complex[start_boundary_gap.seg_i].extend([x for x in complex_alignment[0][start_boundary_gap.seg_i] if x != self.gap_ch])
                        initial_complex[start_boundary_gap.seg_i].insert(0, self.start_boundary_token)
                        complex_alignment[0] = (Ngram(initial_complex[0]).undo(), tuple(initial_complex[-1]))

        complex_alignment = self.postprocess_boundary_gaps(complex_alignment)
        return complex_alignment

    def get_unigram_alignment(self, complex_alignment):
        """Simplifies complex ngram alignments to the best unigram alignment."""
        unigram_alignment = []
        for pos in complex_alignment:
            unigrams = to_unigram_alignment(pos)
            if unigrams != [pos]:
                unigrams_seq1 = [seg1 for seg1, seg2 in unigrams if seg1 != self.gap_ch]
                unigrams_seq2 = [seg2 for seg1, seg2 in unigrams if seg2 != self.gap_ch]
                _, unigram_pos_alignment, _ = needleman_wunsch_extended(
                    seq1=unigrams_seq1,
                    seq2=unigrams_seq2,
                    align_cost=self.align_costs,
                    gap_cost=self.align_costs,
                    gap_ch=self.gap_ch,
                    default_gop=self.gop,
                    allow_complex=False,
                    maximize_score=True,
                )
                unigram_alignment.extend(unigram_pos_alignment)
            else:
                unigram_alignment.append(pos)

        return unigram_alignment

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

    def get_start_boundary_token(self):
        return f'{START_PAD_CH}{self.pad_ch}'
    
    def get_end_boundary_token(self):
        return f'{self.pad_ch}{END_PAD_CH}'

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
        self.update()

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
        self.validate_seq_map(map1, map2)

        return map1, map2

    def validate_seq_map(self, map1, map2):
        padded_len1 = len(self.seq1) + 2
        padded_len2 = len(self.seq2) + 2
        map1_seg_n = sum(len(value) for value in map1.values() if value is not None)
        map2_seg_n = sum(len(value) for value in map2.values() if value is not None)

        # Adjust the map if it includes indices of padding
        def adjust_padded_map(padded_map, padded_len):
            map_new = {}
            for k, values in padded_map.items():
                if values is None:
                    map_new[k] = None
                else:
                    filtered = [v-1 for v in values if v not in {0, padded_len-1}]
                    if len(filtered) > 0:
                        map_new[k] = filtered
                    else:
                        map_new[k] = None
            return map_new

        if map1_seg_n == padded_len1:
            map1 = adjust_padded_map(map1, padded_len1)
        if map2_seg_n == padded_len2:
            map2 = adjust_padded_map(map2, padded_len2)

        try:
            assert sum(len(value) for value in map1.values() if value is not None) == len(self.seq1)
            assert sum(len(value) for value in map2.values() if value is not None) == len(self.seq2)
        except AssertionError as exc:
            raise AssertionError(f"Error mapping aligned sequences: {self.alignment}") from exc

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
        self.key = self.reverse_align_key(alignment.key)
        self.gap_ch = alignment.gap_ch
        self.gop = alignment.gop
        self.pad_ch = alignment.pad_ch
        self.start_boundary_token = self.get_start_boundary_token()
        self.end_boundary_token = self.get_end_boundary_token()
        self.align_costs: PhonemeMap = alignment.align_costs
        self.kwargs = alignment.kwargs
        self.alignment = reverse_alignment(alignment.alignment)
        self.cost = alignment.cost
        self.length = len(self.alignment)
        self.seq_map = tuple(reversed(alignment.seq_map))

        # Phonological environment alignment
        self.phon_env = alignment.phon_env
        if self.phon_env:
            self.phon_env_alignment = super().add_phon_env()
        else:
            self.phon_env_alignment = None

    @staticmethod
    def reverse_align_key(align_key):
        return re.sub(r"/(.+)/ - /(.+)/", r"/\2/ - /\1/", align_key)


class AlignedPair:
    def __init__(self, alignment, index, gap_ch=GAP_CH_DEFAULT, pad_ch=PAD_CH_DEFAULT):
        self.alignment = alignment
        self.gap_ch = gap_ch
        self.pad_ch = pad_ch
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

    def contains_boundary(self):
        return any(ngram.is_boundary(self.pad_ch) for ngram in self.ngrams)

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

    return ALIGNMENT_POSITION_DELIMITER.join(a)


def undo_visual_align(visual_alignment, gap_ch=GAP_CH_DEFAULT, undo_ngrams=True):
    """Reverts a visual alignment to a list of tuple segment pairs"""
    seg_pairs = visual_alignment.split(ALIGNMENT_POSITION_DELIMITER)
    seg_pairs = [tuple(pair.split(ALIGNED_PAIR_DELIMITER)) for pair in seg_pairs]
    # Replace null character with gap character
    seg_pairs = [
        (left.replace(NULL_CH_DEFAULT, gap_ch), right.replace(NULL_CH_DEFAULT, gap_ch))
        for left, right in seg_pairs
    ]
    seg_pairs = [(Ngram(left), Ngram(right)) for left, right in seg_pairs]
    if undo_ngrams:
        seg_pairs = [(left.undo(), right.undo()) for left, right in seg_pairs]
    return seg_pairs


def flatten_tuple(nested_tuple):
    flattened = []
    for item in nested_tuple:
        if isinstance(item, tuple):
            flattened.extend(flatten_tuple(item))
        else:
            flattened.append(item)
    return flattened


def init_precomputed_alignment(alignment,
                               align_key,
                               seq_map,
                               cost,
                               lang1,
                               lang2,
                               gap_ch=GAP_CH_DEFAULT,
                               pad_ch=PAD_CH_DEFAULT,
                               **kwargs):
    """Creates an Alignment object from a precomputed alignment."""
    # Convert alignment string to list of aligned Ngrams
    if isinstance(alignment, str):
        alignment = undo_visual_align(alignment, gap_ch=gap_ch, undo_ngrams=False)
    else:
        alignment = [(Ngram(left), Ngram(right)) for left, right in alignment]
  
    # Create dummy align cost map
    align_costs = PhonemeMap()

    # Extract IPA strings from alignment key
    word1 = ALIGNMENT_KEY_REGEX.sub(r"\1", align_key)
    word2 = ALIGNMENT_KEY_REGEX.sub(r"\2", align_key)
    word1 = Word(word1, transcription_parameters=lang1.transcription_params)
    word2 = Word(word2, transcription_parameters=lang2.transcription_params)
    
    # Extract segments from aligned Ngrams
    # (needs to occur after sequence extraction in order to enable filtering by boundary/gap token)
    alignment = [(left.undo(), right.undo()) for left, right in alignment]
    
    alignment = Alignment(
        seq1=word1,
        seq2=word2,
        override_alignment=alignment,
        override_seq_map=seq_map,
        override_cost=cost,
        align_costs=align_costs,
        gap_ch=gap_ch,
        pad_ch=pad_ch,
        **kwargs
    )
    
    return alignment