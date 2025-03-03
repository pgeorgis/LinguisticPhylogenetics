import logging
import os
import random
import re
from functools import lru_cache
from math import log
from typing import Iterable, Self

from constants import ALIGNMENT_PARAM_DEFAULTS, STRESS_DIACRITICS, TRANSCRIPTION_PARAM_DEFAULTS
from phonUtils.phonSim import phone_sim
from phonUtils.segment import _toSegment
from utils.cluster import draw_dendrogram
from utils.information import entropy
from utils.sequence import Ngram, PhonEnvNgram, flatten_ngram, pad_sequence
from utils.string import format_as_variable, strip_ch
from utils.utils import (create_default_dict, create_default_dict_of_dicts,
                         dict_of_sets, dict_tuplelist, normalize_dict,
                         validate_class)
from utils.word import Word

logger = logging.getLogger(__name__)

class Doculect:
    def __init__(self,
                 name: str,
                 data,
                 columns,
                 transcription_params=TRANSCRIPTION_PARAM_DEFAULTS,
                 alignment_params=ALIGNMENT_PARAM_DEFAULTS,
                 lang_id=None,
                 glottocode=None,
                 iso_code=None,
                 doculect_dir="",
                 ):

        # Language data
        self.name: str = name
        self.path_name: str = format_as_variable(name)
        self.doculect_dir = doculect_dir
        self.lang_id = lang_id
        self.glottocode = glottocode
        self.iso_code = iso_code
        
        # Logging / outfile directory
        self.doculect_dir = doculect_dir
        os.makedirs(self.doculect_dir, exist_ok=True)

        # Attributes for parsing data dictionary (TODO could this be inherited via a subclass?)
        self.data = data
        self.columns = columns

        # Phonemic inventory
        self.phonemes = create_default_dict(0)
        self.phoneme_counts = create_default_dict(0)
        self.vowels = create_default_dict(0)
        self.consonants = create_default_dict(0)
        self.tonemes = create_default_dict(0)
        self.tonal = False

        # Phonological contexts
        self.unigrams = create_default_dict(0)
        self.bigrams = create_default_dict(0)
        self.trigrams = create_default_dict(0)
        self.ngrams = create_default_dict(0, 2)
        self.gappy_bigrams = create_default_dict(0)
        self.gappy_trigrams = create_default_dict(0)
        self.phon_environments = create_default_dict(0, 2)
        self.phon_env_ngrams = create_default_dict(0, 2)

        # Lexical inventory
        self.vocabulary = dict_of_sets()
        self.loanwords = dict_of_sets()

        # Transcription, segmentation, and alignment parameters
        self.transcription_params = transcription_params
        if self.transcription_params['ignore_stress']:
            self.transcription_params['ch_to_remove'].update(STRESS_DIACRITICS)
        if self.transcription_params['suprasegmentals']:
            self.transcription_params['suprasegmentals'] = set(self.transcription_params['suprasegmentals'])
        self.alignment_params = alignment_params

        # Initialize vocabulary and phoneme inventory
        self.create_vocabulary()
        self.create_phoneme_inventory()
        self.write_phoneme_inventory()
        self.phoneme_entropy: float = entropy(self.phonemes)

        # Comparison with other languages
        self.lexical_comparison: dict[str, dict] = create_default_dict_of_dicts(2)
        self.lexical_comparison_measures = set()

    def create_vocabulary(self) -> None:
        for i in self.data:
            entry = self.data[i]
            concept = entry[self.columns['concept']]
            loan = True if re.match(r'((TRUE)|1)$', entry[self.columns['loan']], re.IGNORECASE) else False
            cognate_class = entry[self.columns['cognate_class']]
            cognate_class = cognate_class if cognate_class.strip() != '' else concept
            word = Word(
                ipa_string=entry[self.columns['ipa']],
                concept=concept,
                orthography=entry[self.columns['orthography']],
                transcription_parameters=self.transcription_params,
                doculect_key=self.name,
                cognate_class=cognate_class,
                loanword=loan,
            )
            if len(word.segments) > 0:
                self.vocabulary[concept].add(word)

                # Mark known loanwords
                if loan:
                    self.loanwords[concept].add(word)

    def create_phoneme_inventory(self):
        pad_ch = self.alignment_params['pad_ch']
        for concept in self.vocabulary:
            for word in self.vocabulary[concept]:
                segments = word.segments

                # Count phones and unigrams
                for segment in segments:
                    self.phoneme_counts[segment] += 1
                padded_segments = pad_sequence(segments, pad_ch=pad_ch, pad_n=1)
                for segment in padded_segments:
                    self.unigrams[Ngram(segment).ngram] += 1

                # Count phonological environments
                for seg, env in zip(segments, word.phon_env):
                    self.phon_environments[seg][env] += 1
                # for seg in self.phon_environments:
                #     self.phon_environments[seg] = normalize_dict(self.phon_environments[seg], default=True, lmbda=0)

                # Count trigrams and gappy trigrams
                padded_segments = pad_sequence(segments, pad_ch=pad_ch, pad_n=2)
                for j in range(1, len(padded_segments) - 1):
                    trigram = (padded_segments[j - 1], padded_segments[j], padded_segments[j + 1])
                    self.trigrams[Ngram(trigram).ngram] += 1
                    self.gappy_trigrams[('X', padded_segments[j], padded_segments[j + 1])] += 1
                    self.gappy_trigrams[(padded_segments[j - 1], 'X', padded_segments[j + 1])] += 1
                    self.gappy_trigrams[(padded_segments[j - 1], padded_segments[j], 'X')] += 1

                # Count bigrams
                padded_segments = padded_segments[1:-1]
                for j in range(1, len(padded_segments)):
                    bigram = (padded_segments[j - 1], padded_segments[j])
                    bigram_ngram = Ngram(bigram).ngram
                    self.bigrams[bigram_ngram] += 1
                    self.gappy_bigrams[('X', bigram_ngram[-1])] += 1
                    self.gappy_bigrams[(bigram_ngram[0], 'X')] += 1
        self.ngrams[1] = self.unigrams
        self.ngrams[2] = self.bigrams
        self.ngrams[3] = self.trigrams

        # Normalize counts
        total_tokens: int = sum(self.phoneme_counts.values())
        for phoneme in self.phoneme_counts:
            count = self.phoneme_counts[phoneme]
            if count < self.transcription_params["min_phone_instances"]:
                logger.warning(f'Only {count} instance(s) of /{phoneme}/ in {self.name}.')
            self.phonemes[phoneme] = count / total_tokens

        # Phone classes
        phone_classes = {p: _toSegment(p).phone_class for p in self.phonemes}

        # Get dictionaries of vowels and consonants
        self.vowels = normalize_dict({v: self.phonemes[v]
                                      for v in self.phonemes
                                      if phone_classes[v] in ('VOWEL', 'DIPHTHONG')},
                                     default=True,
                                     default_value=0)

        self.consonants = normalize_dict({c: self.phonemes[c]
                                         for c in self.phonemes
                                         if phone_classes[c] in ('CONSONANT', 'GLIDE')},
                                         default=True,
                                         default_value=0)

        # TODO: rename as self.suprasegmentals, possibly distinguish tonemes from other suprasegmentals
        self.tonemes = normalize_dict({t: self.phonemes[t]
                                       for t in self.phonemes
                                       if phone_classes[t] in ('TONEME', 'SUPRASEGMENTAL')},
                                      default=True,
                                      default_value=0)

        # Designate language as tonal if it has tonemes
        if len(self.tonemes) > 0:
            self.tonal = True
        if set(self.tonemes.keys()) in ({"ˈ"}, {"ˈ", "ˌ"}):
            self.prosodic_typology = 'STRESS'
        elif len(self.tonemes) > 1:
            self.prosodic_typology = "TONE/PITCH ACCENT"
        else:
            self.prosodic_typology = 'OTHER'

    def write_phoneme_inventory(self, n_examples=3, seed=1):
        random.seed(seed)
        with open(os.path.join(self.doculect_dir, 'phones.lst'), 'w') as f:
            for group, label in zip([
                self.vowels,
                self.consonants,
                self.tonemes], [
                    'VOWELS',
                    'CONSONANTS',
                    'SUPRASEGMENTALS'
            ]):
                if len(group) > 0:
                    f.write(f'{label}\n')
                    # Sort in descending order by probability, then also by the phone IPA string in case the probabilities are equal
                    sorted_phones = sorted(dict_tuplelist(group), key=lambda x: (x[-1], x[0]), reverse=True)
                    for phone, prob in sorted_phones:
                        prob = round(self.phonemes[phone], 3)
                        f.write(f'/{phone}/ ({prob})\n')
                        examples = self.lookup(phone, return_list=True)
                        examples = random.sample(examples, min(n_examples, len(examples)))
                        for concept, orth, ipa in examples:
                            f.write(f'\t<{orth}> /{ipa}/ "{concept}"\n')
                        f.write('\n')
                    f.write('\n\n')
        for file, phone_list in zip(['vowels.lst', 'consonants.lst', 'tonemes.lst'],
                                    [self.vowels, self.consonants, self.tonemes]):
            if len(phone_list) > 0:
                phone_list = '\n'.join(sorted(list(phone_list.keys())))
                with open(os.path.join(self.doculect_dir, file), 'w') as f:
                    f.write(phone_list)

    def list_ngrams(self, ngram_size, phon_env=False):
        """Returns a dictionary of ngrams of a particular size, with their counts"""

        # Retrieve pre-calculated ngrams
        if not phon_env and sum(self.ngrams[ngram_size].values()) > 0:
            return self.ngrams[ngram_size]

        elif phon_env and sum(self.phon_env_ngrams[ngram_size].values()) > 0:
            return self.phon_env_ngrams[ngram_size]

        else:
            pad_ch = self.alignment_params['pad_ch']
            for concept in self.vocabulary:
                for word in self.vocabulary[concept]:
                    segments = word.segments
                    if phon_env:
                        phon_env_segments = list(zip(segments, word.phon_env))
                    pad_n = ngram_size - 1
                    padded = pad_sequence(segments, pad_ch=pad_ch, pad_n=pad_n)
                    if phon_env:
                        padded_phon_env = pad_sequence(phon_env_segments, pad_ch=pad_ch, pad_n=pad_n)
                    for i in range(len(padded) - pad_n):
                        ngram = Ngram(padded[i:i + ngram_size])
                        self.ngrams[ngram.size][ngram.ngram] += 1
                        if phon_env and (i > 0 or pad_n == 0):
                            phon_env_ngram = PhonEnvNgram(padded_phon_env[i:i + ngram_size])
                            for subcontext in phon_env_ngram.list_subcontexts():
                                if ngram_size == 1:
                                    key = (*phon_env_ngram.ngram, subcontext)
                                else:
                                    key = (phon_env_ngram.ngram, subcontext)
                                self.phon_env_ngrams[phon_env_ngram.size][key] += 1

            if phon_env:
                return self.phon_env_ngrams[ngram_size]

            else:
                return self.ngrams[ngram_size]

    def lookup(self, segment, field='segments', return_list=False):
        """Prints or returns a list of all word entries containing a given
        segment/character or regular expression"""
        if field not in ('transcription', 'segments', 'orthography'):
            raise ValueError('Error: search field must be either "transcription", "segments", "orthography"!')

        matches = []
        for concept in self.vocabulary:
            for word in self.vocabulary[concept]:
                orthography = word.orthography
                transcription = word.ipa
                segments = word.segments
                if field == 'transcription' and re.search(segment, transcription):
                    matches.append((concept, orthography, transcription))
                elif field == 'segments' and segment in segments:
                    matches.append((concept, orthography, transcription))
                elif field == 'orthography' and re.search(segment, orthography):
                    matches.append((concept, orthography, transcription))

        if return_list:
            return sorted(matches)

        else:
            for match in sorted(matches):
                concept, orthography, transcription = match
                print(f"<{orthography}> /{transcription}/ '{concept}'")

    def _get_Word(self, ipa, concept=None, orthography=None, transcription_params=None, cognate_class=None, loan=None):
        # Check if the word already exists in the vocabulary
        if concept in self.vocabulary:
            for word in self.vocabulary[concept]:
                if ipa in (word.ipa, word.raw_ipa):
                    return word

        # If not, create a new Word object
        if not transcription_params:
            transcription_params = self.transcription_params
        word = Word(
            ipa_string=ipa,
            concept=concept,
            orthography=orthography,
            transcription_parameters=transcription_params,
            doculect_key=self.name,
            cognate_class=cognate_class,
            loanword=loan,
        )

        return word

    def sequence_information_content(self, seq: Iterable, ngram_size: int=3):
        if len(seq) < ngram_size:
            pad_ch = self.alignment_params['pad_ch']
            add_pad_n = ngram_size-len(seq)
            seq = pad_sequence(seq, pad_n=add_pad_n, pad_ch=pad_ch)
        pad_n = ngram_size - 1
        info_content = {}
        for i in range(pad_n, len(seq) - pad_n):
            if ngram_size == 1:
                unigram_count = self.unigrams.get(Ngram(seq[i]).ngram, 0)
                gappy_count = sum(self.unigrams.values())
                info_content_value = (seq[i], -log(unigram_count / gappy_count, 2))
                info_content[i] = info_content_value
            elif ngram_size == 2:
                bigram_counts = 0
                if i > 0:
                    bigram_counts += self.bigrams.get((seq[i - 1], seq[i]), 0)
                if i < len(seq) - 1:
                    bigram_counts += self.bigrams.get((seq[i], seq[i + 1]), 0)
                gappy_counts = 0
                if i > 0:
                    gappy_counts += self.gappy_bigrams.get((seq[i - 1], 'X'), 0)
                if i < len(seq) - 1:
                    gappy_counts += self.gappy_bigrams.get(('X', seq[i + 1]), 0)
                info_content_value = (seq[i], -log(bigram_counts / gappy_counts, 2))
                info_content[i] = info_content_value
            elif ngram_size == 3:
                trigram_counts = 0
                trigram_counts += self.trigrams.get((seq[i - 2], seq[i - 1], seq[i]), 0)
                trigram_counts += self.trigrams.get((seq[i - 1], seq[i], seq[i + 1]), 0)
                trigram_counts += self.trigrams.get((seq[i], seq[i + 1], seq[i + 2]), 0)
                gappy_counts = 0
                gappy_counts += self.gappy_trigrams.get((seq[i - 2], seq[i - 1], 'X'), 0)
                gappy_counts += self.gappy_trigrams.get((seq[i - 1], 'X', seq[i + 1]), 0)
                gappy_counts += self.gappy_trigrams.get(('X', seq[i + 1], seq[i + 2]), 0)
                # TODO : needs smoothing
                info_content_value = (seq[i], -log(trigram_counts / gappy_counts, 2))
                info_content[i - 2] = info_content_value
            else:
                raise ValueError(f"Unsupported ngram_size {ngram_size}. Supported values are {{1, 2, 3}}")
            
        return info_content

    def ngram_count(self, ngram):
        ngram = Ngram(ngram)
        if ngram.size not in self.ngrams:
            self.list_ngrams(ngram.size)
        count = self.ngrams[ngram.size][ngram.ngram]
        return count

    def ngram_probability(self, ngram):
        count = self.ngram_count(ngram)
        prob = count / sum(self.ngrams[ngram.size].values())
        return prob

    @lru_cache(maxsize=None)
    def KN_bigram_probability(self, bigram, delta=0.7):
        """Returns Kneser-Ney smoothed conditional probability P(p2|p1)"""
        bigram = flatten_ngram(bigram)
        if len(bigram) > 2:
            breakpoint()
            raise NotImplementedError
        p1, p2 = bigram

        # Total number of distinct bigrams
        n_bigrams = len(self.bigrams)

        # List of bigrams starting with p1
        start_p1 = [b for b in self.bigrams if b[0] == p1]

        # Number of bigrams starting with p1
        n_start_p1 = len(start_p1)

        # Number of bigrams ending in p2
        n_end_p2 = len([b for b in self.bigrams if b[1] == p2])

        # Unigram probability estimation
        pKN_p1 = n_end_p2 / n_bigrams

        # Normalizing constant lambda
        total_start_p1_counts = sum([self.bigrams[b] for b in start_p1])
        l_KN = (delta / total_start_p1_counts) * n_start_p1

        # Bigram probability estimation
        numerator = max((self.bigrams.get(bigram, 0) - delta), 0)

        return (numerator / total_start_p1_counts) + (l_KN * pKN_p1)

    def phone_dendrogram(self,
                         similarity='weighted_dice',
                         method='ward',
                         exclude_length=True,
                         exclude_tones=True,
                         title=None,
                         save_directory=None,
                         **kwargs):
        if title is None:
            title = f'Phonetic Inventory of {self.name}'

        if save_directory is None:
            save_directory = self.plots_dir

        phonemes = list(self.phonemes.keys())

        if exclude_length:
            phonemes = list(set(strip_ch(p, ['ː']) for p in phonemes))

        if exclude_tones:
            phonemes = [p for p in phonemes if p not in self.tonemes]

        return draw_dendrogram(group=phonemes,
                               labels=phonemes,
                               dist_func=phone_sim,
                               sim=True,  # phone_sim
                               similarity=similarity,
                               method=method,
                               title=title,
                               save_directory=save_directory,
                               **kwargs)

    def write_lexical_comparison(self, lang2: Self, outfile):  # TODO this would be better in utils/logging.py
        measures = sorted(list(self.lexical_comparison_measures))
        with open(outfile, "w") as f:
            header = "\t".join([self.name, lang2.name, "alignment"] + measures)
            f.write(f"{header}\n")
            for word1, word2 in self.lexical_comparison[lang2.name]:
                entry = self.lexical_comparison[lang2.name][(word1, word2)]
                values = [entry.get(measure, "") for measure in measures]
                alignment = entry.get("alignment", "")
                values = [str(v) for v in values]
                line = "\t".join([word1.ipa, word2.ipa, alignment] + values)
                f.write(f"{line}\n")

    def __str__(self):
        """Print a summary of the language object"""
        # TODO improve this
        s = f'{self.name.upper()} [{self.glottocode}][{self.iso_code}]'
        s += f'\nConsonants: {len(self.consonants)}'
        consonant_inventory = ', '.join([pair[0] for pair in dict_tuplelist(self.consonants)])
        s += f'\n/{consonant_inventory}/'
        s += f'\nVowels: {len(self.vowels)}'
        vowel_inventory = ', '.join([pair[0] for pair in dict_tuplelist(self.vowels)])
        s += f'\n/{vowel_inventory}/'
        if self.tonal:
            toneme_inventory = ', '.join([pair[0] for pair in dict_tuplelist(self.tonemes)])
            s += f', Tones: {len(self.tonemes)}'
            s += f'\n/{toneme_inventory}/'
        percent_loanwords = len([1 for concept in self.loanwords for entry in self.loanwords[concept]]) / len([1 for concept in self.vocabulary for entry in self.vocabulary[concept]])
        percent_loanwords *= 100
        if percent_loanwords > 0:
            s += f'\nLoanwords: {round(percent_loanwords, 1)}%'

        s += '\nExample Words:'
        for i in range(5):
            concept = random.choice(list(self.vocabulary.keys()))
            word = random.choice(self.vocabulary[concept])
            s += f'\n\t"{concept.upper()}": /{word.ipa}/ <{word.orthography}>'

        return s
