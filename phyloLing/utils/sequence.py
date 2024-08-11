import re
from functools import lru_cache
from math import factorial

from constants import END_PAD_CH, PAD_CH_DEFAULT, SEG_JOIN_CH, START_PAD_CH
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

    def __str__(self):
        return self.string


class PhonEnvNgram(Ngram):
    def __init__(self, ngram, **kwargs):
        super().__init__(ngram, **kwargs)
        self.ngram, self.phon_env = self.separate_phon_env_from_ngram()
        self.ngram_w_context = (Ngram(self.ngram).undo(), self.phon_env)

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
    return [f'{pad_ch}{START_PAD_CH}'] * pad_n + seq + [f'{END_PAD_CH}{pad_ch}'] * pad_n


# SMOOTHING
def lidstone_smoothing(x, N, d, alpha=0.3):
    """Given x (unsmoothed counts), N (total observations),
    and d (number of possible outcomes), returns smoothed Lidstone probability"""
    return (x + alpha) / (N + (alpha * d))
