from functools import lru_cache
from typing import Self

from constants import PAD_CH_DEFAULT
from utils.doculect import Doculect
from utils.sequence import (Ngram, PhonEnvNgram, count_subsequences,
                            pad_sequence)


def ngram_count_word(ngram, word):
    count = 0
    for i in range(len(word) - len(ngram) + 1):
        if word[i:i + len(ngram)] == list(ngram):
            count += 1
    return count


def ngram_count_wordlist(ngram, seq_list):
    """Retrieve the count of an ngram of segments from a list of segment sequences"""
    count = 0
    for seq in seq_list:
        count += ngram_count_word(ngram, seq)
    return count


def sort_wordlist(wordlist):
    # Get all unique combinations of L1 and L2 word
    # Sort the wordlists in order to ensure that random samples of same/different meaning pairs are reproducible
    # Sort pairs symmetrically (using min/max) to ensure consistent ordering no matter which language is first
    sorted_wordlist = sorted(
        wordlist,
        key=lambda pair: (
            min(pair[0].ipa, pair[1].ipa), max(pair[0].ipa, pair[1].ipa),
            min(pair[0].concept, pair[1].concept), max(pair[0].concept, pair[1].concept),
            min(pair[0].orthography, pair[1].orthography), max(pair[0].orthography, pair[1].orthography),
            pair[0].ipa, pair[1].ipa,
            pair[0].orthography, pair[1].orthography,
            pair[0].concept, pair[1].concept,
        )
    )
    return sorted_wordlist


class Wordlist:
    def __init__(self, word_pairs: list[tuple], pad_n=1):
        self.pad_n = pad_n
        self.word_pairs = list(word_pairs) if not isinstance(word_pairs, list) else word_pairs
        self.wordlist_lang1, self.wordlist_lang2 = map(lambda x: list(x), zip(*word_pairs))
        self.seqs1, self.seqs2 = self.extract_seqs()
        self.seq_lens1, self.seq_lens2 = self.seq_lens()
        self.total_seq_len1, self.total_seq_len2 = self.total_lens()
        self.ngram_probs1, self.ngram_probs2 = {}, {}

    def extract_seqs(self) -> tuple:
        seqs1 = [word.segments for word in self.wordlist_lang1]
        seqs2 = [word.segments for word in self.wordlist_lang2]
        if self.pad_n > 0:
            seqs1 = [pad_sequence(seq, pad_ch=PAD_CH_DEFAULT, pad_n=self.pad_n) for seq in seqs1]
            seqs2 = [pad_sequence(seq, pad_ch=PAD_CH_DEFAULT, pad_n=self.pad_n) for seq in seqs2]
        return seqs1, seqs2

    def seq_lens(self) -> tuple:
        seq_lens1 = [len(seq) for seq in self.seqs1]
        seq_lens2 = [len(seq) for seq in self.seqs2]
        return seq_lens1, seq_lens2

    def total_lens(self) -> tuple:
        total_seq_len1 = sum(self.seq_lens1)
        total_seq_len2 = sum(self.seq_lens2)
        return total_seq_len1, total_seq_len2

    def ngram_probability(self, ngram: Ngram | PhonEnvNgram, lang=1, normalize=True) -> float | int:
        """Calculate or lookup the probability of an ngram in the wordlist."""
        assert isinstance(ngram, (Ngram, PhonEnvNgram))
        if lang == 1:
            seqs = self.seqs1
            seq_lens = self.seq_lens1
            total_seq_len = self.total_seq_len1
            saved = self.ngram_probs1
        elif lang == 2:
            seqs = self.seqs2
            seq_lens = self.seq_lens2
            total_seq_len = self.total_seq_len2
            saved = self.ngram_probs2
        else:
            raise ValueError

        key = (ngram.ngram, normalize)
        if key not in saved:
            count = ngram_count_wordlist(ngram.ngram, seqs)
            if normalize:
                if ngram.size > 1:
                    prob = count / sum([count_subsequences(length, ngram.size) for length in seq_lens])
                else:
                    prob = count / total_seq_len
                result = prob
            else:
                result = count
            saved[key] = result
        return saved[key]

    @lru_cache(maxsize=None)
    def phones_below_min_corr(self, min_corr: int, lang1: Doculect, lang2: Doculect) -> tuple:
        """Return sets of phones in each language with fewer occurrences than required by min_corr value."""
        low_coverage_l1 = set(
            phone for phone in lang1.phoneme_counts
            if self.ngram_probability(Ngram(phone), lang=1, normalize=False) < min_corr
        )
        low_coverage_l2 = set(
            phone for phone in lang2.phoneme_counts
            if self.ngram_probability(Ngram(phone), lang=2, normalize=False) < min_corr
        )

        return low_coverage_l1, low_coverage_l2

    def reset_ngram_probabilities(self):
        self.ngram_probs1, self.ngram_probs2 = {}, {}

    def remove_by_index(self, idx: int):
        """Remove an entry from the wordlist by index and update sequence lists and ngram probabilities."""
        seq_len1 = self.seq_lens1[idx]
        seq_len2 = self.seq_lens2[idx]
        for lst in [
            self.word_pairs,
            self.wordlist_lang1,
            self.wordlist_lang2,
            self.seqs1,
            self.seqs2,
            self.seq_lens1,
            self.seq_lens2,
        ]:
            del lst[idx]
        self.total_seq_len1 -= seq_len1
        self.total_seq_len2 -= seq_len2
        self.reset_ngram_probabilities()

    def reverse(self) -> Self:
        """Return a Wordlist object with reversed sequences, i.e. with language1 and language2 reversed."""
        return Wordlist([(word2, word1) for word1, word2 in self.word_pairs], pad_n=self.pad_n)
