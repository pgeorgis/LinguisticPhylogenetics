import re
from statistics import mean

from constants import PAD_CH_DEFAULT, TRANSCRIPTION_PARAM_DEFAULTS
from phonUtils.ipaTools import normalize_ipa_ch
from phonUtils.phonEnv import get_phon_env
from phonUtils.phonTransforms import normalize_geminates
from phonUtils.segment import segment_ipa
from phonUtils.syllables import syllabify
from utils.sequence import (Ngram, generate_ngrams, pad_sequence,
                            remove_overlapping_ngrams)
from utils.string import asjp_in_ipa


class Word:
    def __init__(self,
                 ipa_string,
                 concept=None,
                 orthography=None,
                 doculect_key=None,
                 cognate_class=None,
                 loanword=False,
                 transcription_parameters=TRANSCRIPTION_PARAM_DEFAULTS,
                 ):
        """Initialize Word object."""
        self.doculect_key = doculect_key
        self.parameters = transcription_parameters
        self.raw_ipa = ipa_string
        self.ipa = self.preprocess(ipa_string)
        self.concept = concept
        self.cognate_class = cognate_class
        self.loanword = loanword
        self.orthography = orthography
        self.segments = self.segment()
        self.ngrams = {}
        self.complex_segments = None
        self.syllables = None
        self.phon_env = self.getPhonEnv()
        self.info_content = {}
        self.total_info_content = {}

    def get_doculect(self, doculect_index):
        if self.doculect_key not in doculect_index:
            raise ValueError(f"No doculect with name '{self.doculect_key}' found!")
        return doculect_index[self.doculect_key]

    def get_parameter(self, label):
        return self.parameters.get(label, TRANSCRIPTION_PARAM_DEFAULTS[label])

    def preprocess(self, ipa_string):

        # Normalize common IPA character mistakes
        # Normalize affricates to special ligature characters, where available
        ipa_string = normalize_ipa_ch(ipa_string)

        # Normalize geminate consonants to /Cː/
        if self.get_parameter('normalize_geminates'):
            ipa_string = normalize_geminates(ipa_string)

        # Convert to ASJP transcriptions
        if self.get_parameter('asjp'):
            # Convert some non-IPA ASJP characters to IPA equivalents
            # Preserves set of ASJP characters/mapping, but keeps IPA compatibility
            ipa_string = asjp_in_ipa(ipa_string)

            return ipa_string

        return ipa_string

    def segment(self):
        return segment_ipa(
            self.ipa,
            # Remove stress and tone diacritics from segmented words; syllabic diacritics (above and below); spaces and <‿> linking tie
            remove_ch=''.join(self.get_parameter('ch_to_remove')),
            combine_diphthongs=self.get_parameter('combine_diphthongs'),
            preaspiration=self.get_parameter('preaspiration'),
            suprasegmentals=self.get_parameter('suprasegmentals')
        )
    
    def get_ngrams(self, size, pad_ch=PAD_CH_DEFAULT, phon_env=False):
        """Get word's segments as ngram sequences of specified size."""
        if (size, phon_env) in self.ngrams:
            return self.ngrams[(size, phon_env)]
        if phon_env:
            seq = list(zip(self.segments, self.phon_env))
        else:
            seq = self.segments
        padded = pad_sequence(seq, pad_ch=pad_ch, pad_n=max(1, size-1))
        ngram_seq = generate_ngrams(padded, ngram_size=size, pad_ch=pad_ch, as_ngram=False)
        self.ngrams[(size, phon_env)] = ngram_seq
        return ngram_seq
    
    def complex_segmentation(self, doculect_index, pad_ch=PAD_CH_DEFAULT):
        """Create non-overlapping complex ngram segmentation with ngrams of variable sizes based on self-surprisal."""
        if self.complex_segments is not None:
            return self.complex_segmentation

        assert self.doculect_key is not None
        # Generate bigrams
        bigrams_seq = self.get_ngrams(size=2, pad_ch=pad_ch)
        
        # Remove overlapping bigrams and decompose into unigrams if appropriate
        def get_ngram_self_surprisal(ngram):
            ngram = Ngram(ngram)
            doculect = self.get_doculect(doculect_index=doculect_index)
            ngram_info = doculect.sequence_information_content(
                ngram.ngram, ngram_size=ngram.size
            )
            return mean([ngram_info[j][-1] for j in ngram_info])
        
        complex_ngram_seq = remove_overlapping_ngrams(
            bigrams_seq,
            ngram_score_func=get_ngram_self_surprisal,
            maximize_score=False,
            pad_ch=pad_ch,
        )
        self.complex_segments = complex_ngram_seq
        return complex_ngram_seq

    def get_syllables(self, **kwargs):
        self.syllables = syllabify(
            word=self.ipa,
            segments=self.segments,
            **kwargs
        )
        return self.syllables

    def getPhonEnv(self):
        phon_env = []
        for i in range(len(self.segments)):
            phon_env.append(get_phon_env(self.segments, i))
        return phon_env

    def get_information_content(self, doculect=None, total=False, doculect_index=None, ngram_size=3):
        if ngram_size in self.info_content:
            if total:
                return self.total_info_content[ngram_size]
            return self.info_content[ngram_size]

        if self.doculect_key is None and doculect is None:
            raise AssertionError('Doculect must be specified in order to calculate information content.')
        elif doculect is not None:
            if self.doculect_key is not None:
                assert doculect.name == self.doculect_key
        else:
            doculect = self.get_doculect(doculect_index)

        padded = pad_sequence(self.segments, pad_ch=PAD_CH_DEFAULT, pad_n=ngram_size - 1)
        self.info_content[ngram_size] = doculect.sequence_information_content(seq=padded, ngram_size=ngram_size)
        self.total_info_content[ngram_size] = sum([val for _, val in self.info_content[ngram_size].values()])
        if total:
            return self.total_info_content[ngram_size]
        return self.info_content[ngram_size]

    def __str__(self):
        syllables = self.get_syllables()
        syl_tr = ".".join(syllable.syl for _, syllable in syllables.items())
        form_tr = "/" + syl_tr + "/"
        if self.orthography and self.orthography != "":
            form_tr = f"<{self.orthography}> {syl_tr}"
        if self.concept and self.concept != "":
            form_tr = f"{form_tr}\n'{self.concept}'"
        if self.doculect_key:
            form_tr = f"{form_tr} ({self.doculect_key})"
        return form_tr
