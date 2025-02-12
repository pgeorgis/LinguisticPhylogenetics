from constants import PAD_CH_DEFAULT
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
    # Sort pairs symmetrically to ensure consistent ordering no matter which language is first
    sorted_wordlist = sorted(
        wordlist,
        key=lambda pair: (
            min(pair[0].ipa, pair[1].ipa), max(pair[0].ipa, pair[1].ipa),
            pair[0].ipa, pair[1].ipa,
            min(pair[0].concept, pair[1].concept), max(pair[0].concept, pair[1].concept),
            min(pair[0].orthography, pair[1].orthography), max(pair[0].orthography, pair[1].orthography),
            #min(pair[0].getInfoContent(total=True, doculect=self.lang1), pair[1].getInfoContent(total=True, doculect=self.lang2)),
            #max(pair[0].getInfoContent(total=True, doculect=self.lang1), pair[1].getInfoContent(total=True, doculect=self.lang2))
        )
    )
    return sorted_wordlist


class Wordlist:
    def __init__(self, word_pairs, pad_n=1):
        self.pad_n = pad_n
        self.wordlist_lang1, self.wordlist_lang2 = zip(*word_pairs)
        self.seqs1, self.seqs2 = self.extract_seqs()
        self.seq_lens1, self.seq_lens2 = self.seq_lens()
        self.total_seq_len1, self.total_seq_len2 = self.total_lens()
        self.ngram_probs1, self.ngram_probs2 = {}, {}

    def extract_seqs(self):
        seqs1 = [word.segments for word in self.wordlist_lang1]
        seqs2 = [word.segments for word in self.wordlist_lang2]
        if self.pad_n > 0:
            seqs1 = [pad_sequence(seq, pad_ch=PAD_CH_DEFAULT, pad_n=self.pad_n) for seq in seqs1]
            seqs2 = [pad_sequence(seq, pad_ch=PAD_CH_DEFAULT, pad_n=self.pad_n) for seq in seqs2]
        return seqs1, seqs2

    def seq_lens(self):
        seq_lens1 = [len(seq) for seq in self.seqs1]
        seq_lens2 = [len(seq) for seq in self.seqs2]
        return seq_lens1, seq_lens2

    def total_lens(self):
        total_seq_len1 = sum(self.seq_lens1)
        total_seq_len2 = sum(self.seq_lens2)
        return total_seq_len1, total_seq_len2

    def ngram_probability(self, ngram, lang=1, normalize=True):
        # if not isinstance(ngram, [Ngram, PhonEnvNgram]):
        #     if PHON_ENV_REGEX.search(ngram):
        #         ngram = PhonEnvNgram(ngram)
        #     else:
        #         ngram = Ngram(ngram)
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

        if ngram.ngram in saved:
            return saved[ngram.ngram]

        else:
            count = ngram_count_wordlist(ngram.ngram, seqs)
            if normalize:
                if ngram.size > 1:
                    prob = count / sum([count_subsequences(length, ngram.size) for length in seq_lens])
                else:
                    prob = count / total_seq_len
            else:
                prob = count

        saved[ngram.ngram] = prob
        return prob
