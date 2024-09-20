import re
from functools import lru_cache
from math import factorial
from statistics import mean

from constants import END_PAD_CH, GAP_CH_DEFAULT, PAD_CH_DEFAULT, SEG_JOIN_CH, START_PAD_CH
from phonUtils.phonEnv import PHON_ENV_REGEX


class Ngram:
    def __init__(self, ngram, lang=None, seg_sep=SEG_JOIN_CH):
        self.raw = ngram
        self.ngram = self.get_ngram(ngram, seg_sep)
        self.string = seg_sep.join(self.ngram)
        self.size = len(self.ngram)
        self.lang = lang

    @staticmethod
    def get_ngram(ngram, seg_sep=SEG_JOIN_CH):
        if isinstance(ngram, str):
            return tuple(re.split(rf'(?<!^){seg_sep}(?!$)', ngram))
        elif isinstance(ngram, Ngram):
            return ngram.ngram
        elif isinstance(ngram, tuple):
            return flatten_ngram(ngram)
        else:
            return flatten_ngram(tuple(ngram))

    def unigrams(self):
        return (Ngram(seg) for seg in self.ngram)

    def undo(self):
        if self.size == 1:
            return self.string
        else:
            return self.ngram

    def is_boundary(self, pad_ch=PAD_CH_DEFAULT):
        """Returns Boolean indicating whether the ngram contains a boundary/pad token."""
        for seg in self.ngram:
            if pad_ch in seg:
                return True
        return False

    def is_gappy(self, gap_ch=GAP_CH_DEFAULT):
        """Returns Boolean indicating whether the ngram contains a gap."""
        for seg in self.ngram:
            if seg == gap_ch:
                return True
        return False

    def remove_gaps(self, gap_ch=GAP_CH_DEFAULT):
        return Ngram(
            [
                unigram.ngram for unigram in self.unigrams()
                if not unigram.is_gappy(gap_ch)
            ]
        )

    def __str__(self):
        return self.string


class PhonEnvNgram(Ngram):
    def __init__(self, ngram, **kwargs):
        super().__init__(ngram, **kwargs)
        self.ngram, self.phon_env = self.separate_phon_env_from_ngram()
        self.ngram_w_context = (Ngram(self.ngram).undo(), self.phon_env)
        self.size = len(self.ngram)

    def separate_phon_env_from_ngram(self):
        ngram, phon_env = [], []
        for part in self.ngram:
            if isinstance(part, str) and PHON_ENV_REGEX.search(part):
                phon_env.append(part)
            else:
                part_ngram = self.get_ngram(part)
                for part_i in part_ngram:
                    if PHON_ENV_REGEX.search(part_i):
                        phon_env.append(part_i)
                    else:
                        ngram.append(part_i)

        ngram = self.get_ngram(ngram)
        phon_env = self.get_ngram(phon_env)
        # Assert that there is exactly one phon env
        # Otherwise combine them based on first component's preceding context
        # and last component's following context
        # e.g. ('<|S|>_N', 'F_>|S|>') -> '<|S|>'
        phon_env = self.combine_phon_envs(phon_env)

        return ngram, phon_env

    def combine_phon_envs(self, phon_envs):
        if len(phon_envs) > 1:
            if '|T|' in phon_envs:
                phon_envs = list(phon_envs)
                phon_envs.remove('|T|')
                return self.combine_phon_envs(phon_envs)
            else:
                pre_env = phon_envs[0].split('|')[0]
                post_env = phon_envs[-1].split('|')[-1]
                return f'{pre_env}|S|{post_env}'
        else:
            assert len(phon_envs) == 1
            return phon_envs[0]


@lru_cache(maxsize=None)
def count_subsequences(seq_len, subseq_len):
    if subseq_len > seq_len:
        return 0

    # Calculate factorials
    L_factorial = factorial(seq_len)
    S_factorial = factorial(subseq_len)
    LS_factorial = factorial(seq_len - subseq_len)

    # Calculate the binomial coefficient
    num_subsequences = L_factorial // (S_factorial * LS_factorial)

    return num_subsequences


def flatten_ngram(nested_ngram):
    flat = []
    for item in nested_ngram:
        if isinstance(item, tuple):
            flat.extend(flatten_ngram(item))
        else:
            flat.append(item)
    return tuple(flat)


def pad_sequence(seq, pad_ch=PAD_CH_DEFAULT, pad_n=1):
    return [f'{START_PAD_CH}{pad_ch}'] * pad_n + seq + [f'{pad_ch}{END_PAD_CH}'] * pad_n

def generate_ngrams(seq, ngram_size, pad_ch=PAD_CH_DEFAULT, as_ngram=True):
    # Ensure ngram_size is less than or equal to the length of the sequence, else pad
    if ngram_size > len(seq):
        return generate_ngrams(
            pad_sequence(
                seq,
                pad_ch=pad_ch,
                pad_n=ngram_size-1,
            )
        )

    ngrams = []
    for i in range(len(seq) - ngram_size + 1):
        ngram = Ngram(seq[i:i+ngram_size])
        if as_ngram:
            ngrams.append(ngram)
        else:
            ngrams.append(ngram.undo())
    return ngrams

def remove_overlapping_bigrams(bigrams, lang, pad_ch=PAD_CH_DEFAULT):
    """Remove overlapping bigrams from a bigram alignment."""
    filtered = []
    # instead of trying to remove overlapping bigrams once alignment has been done, 
    # which is proving to be very complex
    # instead remove overlapping bigrams from input sequences, preferring the bigram with the higher
    # probability in the language / higher PMI of the component segments to each other in the language
    # this will also in a way pre-select truer n>1gram units

    def bigrams_dont_overlap(prev, bigram_i):
        prev_ngram = Ngram(prev)
        bigram_i_ngram = Ngram(bigram_i)
        if prev_ngram.size == 2 and bigram_i_ngram.size == 2:
            if prev[-1] != bigram_i[0]:
                return True
            return False
        elif prev_ngram.size == 1 and bigram_i_ngram.size == 2:
            if prev_ngram.ngram[0] != bigram_i_ngram.ngram[0]:
                return True
            return False
        else:
            raise ValueError # TODO handle

    for i, bigram_i in enumerate(bigrams):
        if i == 0:
            filtered.append(bigram_i)
            continue

        prev = filtered[-1]
        if bigrams_dont_overlap(prev, bigram_i):
            raise ValueError # we expect them always to overlap in this schema now
        else:
            bigram_i_ngram = Ngram(bigram_i)
            prev_ngram = Ngram(prev)
            bigram_i_info = lang.self_surprisal(list(bigram_i_ngram.ngram), as_seq=True, ngram_size=bigram_i_ngram.size)
            prev_info = lang.self_surprisal(list(prev_ngram.ngram), as_seq=True, ngram_size=prev_ngram.size)
            bigram_i_score = mean([bigram_i_info[j][-1] for j in bigram_i_info])
            prev_score = mean([prev_info[j][-1] for j in prev_info])

            if bigram_i_score < prev_score:
                # Favor bigram_i, revert prev to a unigram including its first half
                if prev_ngram.size == 1:
                    # if prev is already a unigram, replace it with the bigram (as they overlap)
                    filtered[-1] = bigram_i
                else:
                    filtered[-1] = Ngram(filtered[-1]).ngram[0]
                    filtered.append(bigram_i)
            else:
                # Favor previous bigram, revert current bigram to a unigram including its second half
                filtered.append(bigram_i[-1])

    return filtered


# SMOOTHING
def lidstone_smoothing(x, N, d, alpha=0.3):
    """Given x (unsmoothed counts), N (total observations),
    and d (number of possible outcomes), returns smoothed Lidstone probability"""
    return (x + alpha) / (N + (alpha * d))
