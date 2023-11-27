from functools import lru_cache
from math import factorial
import re
from constants import SEG_JOIN_CH, PAD_CH_DEFAULT, START_PAD_CH, END_PAD_CH

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
    return [f'{pad_ch}{START_PAD_CH}']*pad_n + seq + [f'{END_PAD_CH}{pad_ch}']*pad_n

# SMOOTHING
def lidstone_smoothing(x, N, d, alpha=0.3):
    """Given x (unsmoothed counts), N (total observations), 
    and d (number of possible outcomes), returns smoothed Lidstone probability"""
    return (x + alpha) / (N + (alpha*d))