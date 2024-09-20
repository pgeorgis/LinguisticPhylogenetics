import copy
import glob
import logging
import os
import random
import re
from collections import defaultdict
from collections.abc import Iterable
from functools import lru_cache
from itertools import combinations, product
from math import log, sqrt
from statistics import mean

import bcubed
import numpy as np
import pandas as pd
import seaborn as sns
from constants import (ALIGNMENT_PARAM_DEFAULTS, SEG_JOIN_CH,
                       TRANSCRIPTION_PARAM_DEFAULTS, 
                       END_PAD_CH, START_PAD_CH, PAD_CH_DEFAULT)
from matplotlib import pyplot as plt
from phonCorr import PhonCorrelator
from phonUtils.initPhoneData import suprasegmental_diacritics
from phonUtils.ipaTools import invalid_ch, normalize_ipa_ch, strip_diacritics
from phonUtils.phonEnv import get_phon_env
from phonUtils.phonSim import phone_sim
from phonUtils.phonTransforms import normalize_geminates
from phonUtils.segment import _toSegment, segment_ipa
from phonUtils.syllables import syllabify
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from skbio import DistanceMatrix
from skbio.tree import nj
from unidecode import unidecode
from utils.cluster import cluster_items, draw_dendrogram, linkage2newick
from utils.distance import Distance, distance_matrix
from utils.information import calculate_infocontent_of_word, entropy
from utils.network import dm2coords, newer_network_plot
from utils.sequence import Ngram, flatten_ngram, pad_sequence
from utils.string import asjp_in_ipa, format_as_variable, strip_ch
from utils.utils import (create_timestamp, csv2dict, default_dict,
                         dict_tuplelist, normalize_dict)


class LexicalDataset:
    def __init__(self, filepath, name,
                 outdir=None,
                 id_c='ID',
                 language_name_c='Language_ID',
                 concept_c='Parameter_ID',
                 orthography_c='Value',
                 ipa_c='Form',
                 segments_c='Segments',
                 cognate_class_c='Cognate_ID',
                 loan_c='Loan',
                 glottocode_c='Glottocode',
                 iso_code_c='ISO 639-3',
                 included_doculects=None,
                 excluded_doculects=None,
                 transcription_params={'global': TRANSCRIPTION_PARAM_DEFAULTS},
                 alignment_params=ALIGNMENT_PARAM_DEFAULTS,
                 logger=None
                 ):

        # Dataset name and logger
        self.name = name
        self.path_name = format_as_variable(name)
        if logger:
            self.logger = logger
        else:
            # Configure the logger from scratch, default loglevel=INFO
            logging.basicConfig(level=logging.INFO, format='%(asctime)s phyloLing %(levelname)s: %(message)s')
            logger = logging.getLogger(__name__)
        self.logger.info(f'Loading {self.name}...')

        # Directory to dataset
        self.filepath = filepath
        self.directory = outdir if outdir else os.path.dirname(os.path.abspath(filepath))

        # Create a folder for plots and detected cognate sets within the dataset's directory
        self.plots_dir = os.path.join(self.directory, 'plots')
        self.cognates_dir = os.path.join(self.directory, 'cognates')
        self.phone_corr_dir = os.path.join(self.directory, 'phone_corr')
        self.doculects_dir = os.path.join(self.directory, 'doculects')
        self.dist_matrix_dir = os.path.join(self.directory, 'dist_matrices')
        self.tree_dir = os.path.join(self.directory, 'trees')
        for dir in (
            self.plots_dir,
            self.cognates_dir,
            self.phone_corr_dir,
            self.doculects_dir,
            self.dist_matrix_dir,
            self.tree_dir
        ):
            os.makedirs(dir, exist_ok=True)

        # Columns of dataset TSV # TODO make exactly match official CLDF columns
        self.columns = {
            'id': id_c,
            'language_name': language_name_c,
            'concept': concept_c,
            'orthography': orthography_c,
            'ipa': ipa_c,
            'segments': segments_c,
            'cognate_class': cognate_class_c,
            'loan': loan_c,
            'glottocode': glottocode_c,
            'iso_code': iso_code_c,
        }

        # Information about languages included
        self.languages = {}
        self.lang_ids = {}
        self.glottocodes = {}
        self.iso_codes = {}
        self.distance_matrices = {}

        # Transcription and alignment parameters
        self.transcription_params = transcription_params
        if not self.transcription_params['global']['ignore_stress']:
            self.transcription_params['global']['ch_to_remove'] = set(self.transcription_params['global']['ch_to_remove']) - {'ˈ', 'ˌ'}
        self.alignment_params = alignment_params

        # Concepts in dataset
        self.concepts = defaultdict(lambda: defaultdict(lambda: []))
        self.cognate_sets = defaultdict(lambda: defaultdict(lambda: set()))
        self.clustered_cognates = defaultdict(lambda: {})
        self.load_data(self.filepath, included_doculects=included_doculects, excluded_doculects=excluded_doculects)
        self.load_gold_cognate_sets()
        self.mutual_coverage = self.calculate_mutual_coverage()

    def load_data(self, filepath, included_doculects=[], excluded_doculects=None, sep='\t'):

        # Load data file
        data = csv2dict(filepath, sep=sep)
        self.data = data

        # Initialize languages
        language_vocab_data = defaultdict(lambda: defaultdict(lambda: {}))
        for i in data:
            lang = data[i][self.columns['language_name']]
            if ((included_doculects == []) or (lang in included_doculects)) and ((excluded_doculects == []) or (lang not in excluded_doculects)):
                features = list(data[i].keys())
                for feature in features:
                    value = data[i][feature]
                    language_vocab_data[lang][i][feature] = value
                self.glottocodes[lang] = data[i][self.columns['glottocode']]
                self.iso_codes[lang] = data[i][self.columns['iso_code']]
                self.lang_ids[lang] = data[i][self.columns['id']].split('_')[0]

        language_list = sorted(list(language_vocab_data.keys()))
        for lang in language_list:
            os.makedirs(os.path.join(self.doculects_dir, format_as_variable(lang)), exist_ok=True)
            self.languages[lang] = Language(name=lang,
                                            lang_id=self.lang_ids[lang],
                                            glottocode=self.glottocodes[lang],
                                            iso_code=self.iso_codes[lang],
                                            family=self,
                                            data=language_vocab_data[lang],
                                            columns=self.columns,
                                            transcription_params=self.transcription_params.get('doculects', {}).get(lang, self.transcription_params['global']),
                                            alignment_params=self.alignment_params,
                                            )
            self.logger.info(f'Loaded doculect {lang}.')
            for concept in self.languages[lang].vocabulary:
                self.concepts[concept][lang].extend(self.languages[lang].vocabulary[concept])
        for lang in language_list:
            self.languages[lang].write_missing_concepts()

    def load_gold_cognate_sets(self):
        """Creates dictionary sorted by cognate sets"""
        for lang in self.languages:
            lang = self.languages[lang]
            for concept in lang.vocabulary:
                for word in lang.vocabulary[concept]:
                    cognate_class = word.cognate_class
                    self.cognate_sets[cognate_class][lang].add(word)

    def write_lexical_index(self,
                            output_file=None,
                            concept_list=None,
                            sep='\t',
                            variants_sep='~'
                            ):
        """Write cognate set index to TSV file."""
        assert sep != variants_sep
        if output_file is None:
            output_file = os.path.join(self.cognates_dir, f'{self.path_name.lower()}_cognate_index.tsv')

        if concept_list is None:
            concept_list = sorted(list(self.cognate_sets.keys()))
        else:
            concept_list = sorted([c for c in concept_list if c in self.concepts])
            concept_list = sorted([c for c in self.cognate_sets.keys()
                                   if c.split('_')[0] in concept_list])

        with open(output_file, 'w') as f:
            language_names = sorted([self.languages[lang].name for lang in self.languages])
            header = sep.join(['Gloss'] + language_names)
            f.write(f'{header}\n')

            for cognate_set_id in concept_list:
                forms = [cognate_set_id]
                for lang in language_names:
                    lang = self.languages[lang]
                    lang_forms = self.cognate_sets[cognate_set_id].get(lang, [])
                    lang_forms = [form.ipa if not form.loanword else f"({form.ipa})" for form in lang_forms]
                    forms.append(variants_sep.join(lang_forms))
                f.write(sep.join(forms))
                f.write('\n')

    def calculate_mutual_coverage(self, concept_list=None):
        """Calculate the mutual coverage and average mutual coverage (AMC)
        of the dataset on a particular wordlist"""

        # By default use the entire vocabulary if no specific concept list is given
        if concept_list is None:
            concept_list = self.concepts

        # Calculate mutual coverage
        concept_counts = {concept: len([lang for lang in self.languages
                                       if concept in self.languages[lang].vocabulary])
                          for concept in concept_list}
        shared_concepts = [concept for concept in concept_counts
                           if concept_counts[concept] == len(self.languages)]
        mutual_coverage = len(shared_concepts)

        # Calculate average mutual coverage
        mutual_coverages = {}
        for lang_pair in product(self.languages.values(), self.languages.values()):
            lang1, lang2 = lang_pair
            if lang1 != lang2:
                pair_mutual_coverage = len([concept for concept in concept_list
                                            if concept in lang1.vocabulary
                                            if concept in lang2.vocabulary])
                mutual_coverages[lang_pair] = pair_mutual_coverage
        avg_mutual_coverage = mean(mutual_coverages.values()) / len([c for c in concept_list
                                                                     if c in self.concepts])

        return mutual_coverage, avg_mutual_coverage

    def prune_languages(self, min_amc=0.8, concept_list=None):
        """Prunes the language with the smallest number of transcribed words
        until the dataset's AMC score reaches the minimum value"""

        # By default use the entire vocabulary if no specific concept list is given
        if concept_list is None:
            concept_list = self.concepts

        pruned = []
        start_n_langs = len(self.languages)
        original_amc = self.calculate_mutual_coverage(concept_list)[1]
        while self.calculate_mutual_coverage(concept_list)[1] < min_amc:
            smallest_lang = min(self.languages.keys(),
                                key=lambda x: len(self.languages[x].vocabulary))
            pruned.append((smallest_lang, len(self.languages[smallest_lang].vocabulary)))
            self.remove_languages([smallest_lang])

        self.mutual_coverage = self.calculate_mutual_coverage(concept_list)

        if len(pruned) > 0:
            prune_log = f'Pruned {len(pruned)} of {start_n_langs} {self.name} languages:'
            for item in pruned:
                lang, vocab_size = item
                if vocab_size == 1:
                    prune_log += f'\n\t\t{lang} ({vocab_size} concept)'
                else:
                    prune_log += f'\n\t\t{lang} ({vocab_size} concepts)'
            prune_log += f'\tAMC increased from {round(original_amc, 2)} to {round(self.mutual_coverage[1], 2)}.'
            if self.logger:
                self.logger.info(prune_log)

    def calculate_phoneme_pmi(self, **kwargs):
        """Calculates phoneme PMI for all language pairs in the dataset and saves
        the results to file"""
        lang_pairs = self.get_doculect_pairs(bidirectional=False)
        # Check whether phoneme PMI has been calculated already for this pair
        # If not, calculate it now
        for lang1, lang2 in lang_pairs:
            if len(lang1.phoneme_pmi[lang2]) == 0:
                correlator = lang1.get_phoneme_correlator(lang2)
                correlator.calc_phoneme_pmi(**kwargs)

    def load_phoneme_pmi(self, pmi_dir=None, excepted=[], sep='\t', **kwargs):
        """Loads pre-calculated phoneme PMI values from file"""

        # Designate the default directory to search for if no alternative is provided
        if pmi_dir is None:
            pmi_dir = os.path.join(self.phone_corr_dir, 'pmi')

        def str2ngram(str, join_ch='_'):
            return Ngram(str, lang=self, seg_sep=join_ch)

        for lang1, lang2 in self.get_doculect_pairs(bidirectional=False):
            if (lang1.name not in excepted) and (lang2.name not in excepted):
                pmi_file = os.path.join(pmi_dir, lang1.path_name, lang2.path_name, 'phonPMI.tsv')

                # Try to load the file of saved PMI values, otherwise calculate PMI first
                if not os.path.exists(pmi_file):
                    correlator = lang1.get_phoneme_correlator(lang2)
                    correlator.calc_phoneme_pmi(**kwargs)
                pmi_data = pd.read_csv(pmi_file, sep=sep)

                # Iterate through the dataframe and save the PMI values to the Language
                # class objects' phoneme_pmi attribute
                for index, row in pmi_data.iterrows():
                    phone1, phone2 = row['Phone1'], row['Phone2']
                    pmi_value = row['PMI']
                    ngram1, ngram2 = map(str2ngram, [phone1, phone2])
                    lang1.phoneme_pmi[lang2][ngram1.undo()][ngram2.undo()] = pmi_value
                    lang2.phoneme_pmi[lang1][ngram2.undo()][ngram1.undo()] = pmi_value
                    if ngram1.size > 1 or ngram2.size > 1:
                        lang1.complex_ngrams[lang2][ngram1][ngram2] = pmi_value
                        lang2.complex_ngrams[lang1][ngram2][ngram1] = pmi_value

    def write_phoneme_pmi(self, **kwargs):
        self.logger.info(f'Saving {self.name} phoneme PMI...')
        for lang1, lang2 in self.get_doculect_pairs(bidirectional=False):
            # Retrieve the precalculated values
            if len(lang1.phoneme_pmi[lang2]) == 0:
                self.logger.warning(f'Phoneme PMI has not been calculated for pair: {lang1.name} - {lang2.name}.')
                continue
            correlator = lang1.get_phoneme_correlator(lang2)

            # Skip rewriting PMI for doculect pairs for which PMI was not calculated in this run
            # This happens when recalculating for specific doculects only and loading others from file
            if len(correlator.pmi_dict) == 0:
                continue

            correlator.log_phoneme_pmi(**kwargs)

    def write_phoneme_surprisal(self, phon_env=True, ngram_size=1, **kwargs):
        self.logger.info(f'Saving {self.name} phoneme surprisal...')
        for lang1, lang2 in self.get_doculect_pairs(bidirectional=True):

            # Retrieve the precalculated values
            if len(lang1.phoneme_surprisal[(lang2.name, ngram_size)]) == 0:
                self.logger.warning(f'{ngram_size}-gram phoneme surprisal has not been calculated for pair: {lang1.name} - {lang2.name}')
                continue
            correlator = lang1.get_phoneme_correlator(lang2)

            # Skip rewriting surprisal for doculect pairs for which surprisal was not calculated in this run
            # This happens when recalculating for specific doculects only and loading others from file
            if len(correlator.surprisal_dict) == 0:
                continue

            correlator.log_phoneme_surprisal(phon_env=False, ngram_size=ngram_size, **kwargs)
            if phon_env:
                correlator.log_phoneme_surprisal(phon_env=True, **kwargs)

    def calculate_phoneme_surprisal(self, ngram_size=1, **kwargs):
        """Calculates phoneme surprisal for all language pairs in the dataset and saves
        the results to file"""

        # First ensure that phoneme PMI has been calculated and loaded
        self.load_phoneme_pmi()

        # Check whether phoneme surprisal has been calculated already for this pair
        lang_pairs = self.get_doculect_pairs(bidirectional=True)
        for lang1, lang2 in lang_pairs:
            # If not, calculate it now
            if len(lang1.phoneme_surprisal[(lang2.name, ngram_size)]) == 0:
                correlator = lang1.get_phoneme_correlator(lang2)
                correlator.calc_phoneme_surprisal(ngram_size=ngram_size, **kwargs)

    def load_phoneme_surprisal(self, ngram_size=1, surprisal_dir=None, excepted=[], sep='\t', **kwargs):
        """Loads pre-calculated phoneme surprisal values from file"""
        # Designate the default file name to search for if no alternative is provided
        if surprisal_dir is None:
            surprisal_dir = os.path.join(self.phone_corr_dir, 'surprisal')

        def str2ngram(str, join_ch=SEG_JOIN_CH):
            return Ngram(str, lang=self, seg_sep=join_ch)

        def extract_surprisal_from_df(surprisal_data, lang2, phon_env=False):
            surprisal_dict = defaultdict(lambda: {})
            oov_vals = {}
            for index, row in surprisal_data.iterrows():
                phone1, phone2 = row['Phone1'], row['Phone2']
                surprisal_value = row['Surprisal']
                if ngram_size > 1:
                    breakpoint()  # TODO need to decide format for how to save/load larger ngrams from logs; previously they were separated by whitespace
                ngram1, ngram2 = map(str2ngram, [phone1, phone2])
                ngram2_dict_form = ngram2.undo()
                if phon_env:
                    env = row['PhonEnv']
                    ngram1_dict_form = (Ngram(phone1).undo(), env)
                else:
                    ngram1_dict_form = ngram1.undo()
                surprisal_dict[ngram1_dict_form][ngram2_dict_form] = surprisal_value
                if ngram1_dict_form not in oov_vals:
                    oov_smoothed = row['OOV_Smoothed']
                    oov_vals[ngram1_dict_form] = oov_smoothed

            # Iterate back through language pairs and phone1 combinations and set OOV values
            for phone1 in oov_vals:
                oov_val = oov_vals[phone1]
                surprisal_dict[phone1] = default_dict(surprisal_dict[phone1], lmbda=oov_val)
            surprisal_dict = default_dict(surprisal_dict, lmbda=defaultdict(lambda: oov_val))

            return surprisal_dict

        for lang1, lang2 in self.get_doculect_pairs(bidirectional=True):
            if (lang1.name not in excepted) and (lang2.name not in excepted):
                surprisal_file = os.path.join(surprisal_dir, lang1.path_name, lang2.path_name, f'{ngram_size}-gram', 'phonSurprisal.tsv')
                surprisal_file_phon_env = os.path.join(surprisal_dir, lang1.path_name, lang2.path_name, 'phonEnv', 'phonEnvSurprisal.tsv')

                # Try to load the file of saved surprisal values, otherwise calculate surprisal first
                if not os.path.exists(surprisal_file):
                    correlator = lang1.get_phoneme_correlator(lang2)
                    correlator.calc_phoneme_surprisal(ngram_size=ngram_size, **kwargs)
                surprisal_data = pd.read_csv(surprisal_file, sep=sep)

                # Extract and save the surprisal values to phoneme_surprisal attribute of language object
                loaded_surprisal = extract_surprisal_from_df(surprisal_data, lang2, phon_env=False)
                lang1.phoneme_surprisal[(lang2.name, ngram_size)] = loaded_surprisal

                # Do the same for phonological environment surprisal
                if os.path.exists(surprisal_file_phon_env):
                    phon_env_surprisal_data = pd.read_csv(surprisal_file_phon_env, sep=sep)
                    loaded_phon_env_surprisal = extract_surprisal_from_df(phon_env_surprisal_data, lang2, phon_env=True)
                    lang1.phon_env_surprisal[lang2.name] = loaded_phon_env_surprisal

                else:
                    self.logger.warning(f'No saved phonological environment surprisal file found for {lang1.name}-{lang2.name}')

    def get_doculect_pairs(self, bidirectional=False):
        if bidirectional:
            doculect_pairs = product(self.languages.values(), self.languages.values())
        else:
            doculect_pairs = combinations(self.languages.values(), 2)
        return sorted([(lang1, lang2) for lang1, lang2 in doculect_pairs if lang1 != lang2],
                      key=lambda x: (x[0].name, x[1].name))

    def cognate_set_dendrogram(self,  # TODO UPDATE THIS FUNCTION
                               cognate_id,
                               dist_func,
                               combine_cognate_sets=True,
                               method='average',
                               title=None,
                               save_directory=None,
                               **kwargs):
        if combine_cognate_sets:
            cognate_ids = [c for c in self.cognate_sets if c.split('_')[0] == cognate_id]
        else:
            cognate_ids = [cognate_id]

        words = [strip_ch(item[i], ['(', ')'])
                 for cognate_id in cognate_ids
                 for item in self.cognate_sets[cognate_id].values()
                 for i in range(len(item))]

        lang_labels = [key for cognate_id in cognate_ids
                       for key in self.cognate_sets[cognate_id].keys()
                       for i in range(len(self.cognate_sets[cognate_id][key]))]
        labels = [f'{lang_labels[i]} /{words[i]}/' for i in range(len(words))]

        # Create tuple input of (word, lang)
        langs = [self.languages[lang] for lang in lang_labels]
        words = list(zip(words, langs))

        if title is None:
            title = f'{self.name} "{cognate_id}"'

        if save_directory is None:
            save_directory = self.plots_dir

        draw_dendrogram(group=words,
                        labels=labels,
                        dist_func=dist_func,
                        sim=dist_func.sim,
                        method=method,
                        title=title,
                        save_directory=save_directory,
                        **kwargs
                        )

    def cluster_cognates(self,
                         concept_list,
                         dist_func,
                         method='average',
                         **kwargs):

        # TODO make option for instead using k-means clustering given a known/desired number of clusters, as a mutually exclusive parameter with cutoff

        self.logger.info('Clustering cognates...')
        self.logger.debug(f'Cluster function: {dist_func.name}')
        self.logger.debug(f'Cluster threshold: {dist_func.cluster_threshold}')

        concept_list = [concept for concept in concept_list if len(self.concepts[concept]) > 1]
        self.logger.debug(f'Total concepts: {len(concept_list)}')
        clustered_cognates = {}
        for concept in sorted(concept_list):
            self.logger.debug(f'Clustering words for "{concept}"...')
            words = [word for lang in self.concepts[concept] for word in self.concepts[concept][lang]]
            clusters = cluster_items(group=words,
                                     dist_func=dist_func,
                                     sim=dist_func.sim,
                                     cutoff=dist_func.cluster_threshold,
                                     **kwargs)
            clustered_cognates[concept] = clusters

        # Create code and store the result
        code = self.generate_test_code(dist_func, cognates='auto')
        self.clustered_cognates[code] = clustered_cognates
        self.write_cognate_index(clustered_cognates, os.path.join(self.cognates_dir, f'{code}.cog'))

        return clustered_cognates

    def write_cognate_index(self,
                            clustered_cognates,
                            output_file,
                            sep='\t',
                            variants_sep='~'):
        assert sep != variants_sep

        cognate_index = defaultdict(lambda: defaultdict(lambda: []))
        languages = []
        for concept in clustered_cognates:
            for i in clustered_cognates[concept]:
                cognate_id = f'{concept}_{i}'
                for word in clustered_cognates[concept][i]:
                    lang = word.language.name
                    cognate_index[cognate_id][lang].append(word.ipa)
                    languages.append(lang)
        languages = sorted(list(set(languages)))

        with open(output_file, 'w') as f:
            header = '\t'.join([''] + languages)
            f.write(header)
            f.write('\n')
            for cognate_id in cognate_index:
                line = [cognate_id]
                for lang in languages:
                    entry = '~'.join(cognate_index[cognate_id][lang])
                    line.append(entry)
                line = sep.join(line)
                f.write(f'{line}\n')

        self.logger.info(f'Wrote clustered cognate index to {output_file}')

    def load_cognate_index(self, index_file, sep='\t', variants_sep='~'):
        assert sep != variants_sep
        index = defaultdict(lambda: defaultdict(lambda: []))
        with open(index_file, 'r') as f:
            f = f.readlines()
            doculects = [name.strip() for name in f[0].split(sep)[1:]]
            for line in f[1:]:
                line = line.split(sep)
                cognate_id = line[0].rsplit('_', maxsplit=1)
                try:
                    concept, cognate_class = cognate_id
                except ValueError:
                    concept, cognate_class = cognate_id, ''  # TODO confirm that this is correct
                    breakpoint()
                for lang, form in zip(doculects, line[1:]):
                    forms = form.split(variants_sep)
                    for form_i in forms:
                        form_i = form_i.strip()

                        # Verify that all characters used in transcriptions are recognized
                        form_i = normalize_ipa_ch(form_i)
                        unk_ch = invalid_ch(form_i)
                        if len(unk_ch) > 0:
                            unk_ch_s = '< ' + ' '.join(unk_ch) + ' >'
                            raise ValueError(f'Error: Unable to parse characters {unk_ch_s} in {lang} /{form_i}/ "{concept}"!')
                        if len(form_i.strip()) > 0:
                            if isinstance(lang, str):
                                lang = self.languages[lang]
                            word = lang._get_Word(form_i, concept=concept, cognate_class='_'.join(cognate_id))
                            index[concept][cognate_class].append(word)

        return index

    def load_clustered_cognates(self, **kwargs):
        cognate_files = glob.glob(f'{self.cognates_dir}/*.cog')
        for cognate_file in cognate_files:
            code = os.path.splitext(os.path.basename(cognate_file))[0]
            self.clustered_cognates[code] = self.load_cognate_index(cognate_file, **kwargs)
        n = len(cognate_files)
        s = f'Loaded {n} cognate'
        if n > 1 or n < 1:
            s += ' indices.'
        else:
            s += ' index.'
        self.logger.info(s)

    def evaluate_clusters(self, clustered_cognates, method='bcubed'):
        """Evaluates B-cubed precision, recall, and F1 of results of automatic
        cognate clustering against dataset's gold cognate classes"""

        precision_scores, recall_scores, f1_scores, mcc_scores = {}, {}, {}, {}
        ch_to_remove = self.transcription_params['global']['ch_to_remove'].union({'(', ')'})
        for concept in clustered_cognates:
            clusters = {'/'.join([strip_diacritics(unidecode.unidecode(item.split('/')[0])),
                                  strip_ch(item.split('/')[1], ch_to_remove)]) + '/': set([i]) for i in clustered_cognates[concept]
                        for item in clustered_cognates[concept][i]}

            gold_clusters = {f'{strip_diacritics(unidecode.unidecode(lang))} /{strip_ch(tr, ch_to_remove)}/': set([c])
                             for c in self.cognate_sets
                             if re.split('[-|_]', c)[0] == concept
                             for lang in self.cognate_sets[c]
                             for tr in self.cognate_sets[c][lang]}

            # Skip concepts without any gold cognate class information
            if len(gold_clusters) == 0:
                continue

            if method == 'bcubed':

                precision = bcubed.precision(clusters, gold_clusters)
                recall = bcubed.recall(clusters, gold_clusters)
                fscore = bcubed.fscore(precision, recall)
                precision_scores[concept] = precision
                recall_scores[concept] = recall
                f1_scores[concept] = fscore

            elif method == 'mcc':
                pairs = [(item1, item2) for item1 in gold_clusters for item2 in gold_clusters if item1 != item2]
                results = []
                for pair in pairs:
                    w1, w2 = pair
                    gold_value = gold_clusters[w1] == gold_clusters[w2]
                    test_value = clusters[w1] == clusters[w2]
                    results.append((gold_value, test_value))

                TP = results.count((True, True))
                FP = results.count((False, True))
                TN = results.count((False, False))
                FN = results.count((True, False))
                num = (TP * TN) - (FP * FN)
                dem = sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
                try:
                    mcc = num / dem
                except ZeroDivisionError:
                    mcc = 0
                mcc_scores[concept] = mcc

            else:
                raise ValueError(f'Error: Method "{method}" not recognized for cluster evaluation!')
        if method == 'bcubed':
            return mean(precision_scores.values()), mean(recall_scores.values()), mean(f1_scores.values())
        elif method == 'mcc':
            return mean(mcc_scores.values())

    def generate_test_code(self, dist_func, cognates=None, exclude=['logger'], **kwargs):  # TODO would it make more sense to create a separate class rather than the LexicalDataset for this?
        """Creates a unique identifier for the current experiment"""

        if not isinstance(dist_func, Distance):
            self.logger.error(f'dist_func must be a Distance class object, found {type(dist_func)} instead.')
            raise TypeError

        name = dist_func.name if dist_func.name else dist_func.func.__name__
        code = f'cognates-{cognates}_distfunc-{name}_'
        if cognates == 'auto':
            code += f'cutoff-{dist_func.cluster_threshold}'

        def val_to_str(value):
            if isinstance(value, str):
                return value
            elif isinstance(value, Iterable):
                return '-'.join([val_to_str(val) for val in value])
            elif isinstance(value, (bool, int, float)):
                return str(value)
            elif isinstance(value, Distance):
                return value.name if value.name else value.func.__name__
            elif value is None:
                return None
            else:
                raise TypeError

        def kwargs_to_str(kwargs, exclude):
            str_vals = []
            for key, value in kwargs:
                if key in exclude:
                    continue
                val = val_to_str(value)
                if val:
                    str_vals.append(f'{key}_{val}')
            return '_'.join(str_vals)

        code += kwargs_to_str(dist_func.hashable_kwargs, exclude)
        code += kwargs_to_str(kwargs.items(), exclude)

        # Replace spaces with underscores
        code = re.sub(r'\s', '_', code)
        # Remove trailing underscore
        code = re.sub(r'_$', '', code)

        return code

    def distance_matrix(self,
                        dist_func,
                        concept_list=None,
                        cluster_func=None,
                        cognates='auto',
                        outfile=None,
                        **kwargs):

        # Try to skip re-calculation of distance matrix by retrieving
        # a previously computed distance matrix by its code
        code = self.generate_test_code(dist_func, cognates=cognates, **kwargs)

        if code in self.distance_matrices:
            return self.distance_matrices[code]

        # Use all available concepts by default
        if concept_list is None:
            concept_list = sorted([concept for concept in self.concepts.keys()
                                   if len(self.concepts[concept]) > 1])
        else:
            concept_list = sorted([concept for concept in concept_list
                                   if len(self.concepts[concept]) > 1])

        # Automatic cognate clustering
        if cognates == 'auto':
            assert cluster_func is not None
            cluster_code = self.generate_test_code(cluster_func, cognates='auto')

            if cluster_code in self.clustered_cognates:
                clustered_concepts = self.clustered_cognates[cluster_code]
            else:
                clustered_concepts = self.cluster_cognates(concept_list, dist_func=cluster_func)

        # Use gold cognate classes
        elif cognates == 'gold':
            clustered_concepts = defaultdict(lambda: defaultdict(lambda: []))
            for concept in concept_list:
                # TODO there may be a better way to isolate these cognate IDs
                cognate_ids = [cognate_id for cognate_id in self.cognate_sets if cognate_id.rsplit('_', maxsplit=1)[0] == concept]
                for cognate_id in cognate_ids:
                    for lang in self.cognate_sets[cognate_id]:
                        for word in self.cognate_sets[cognate_id][lang]:
                            clustered_concepts[concept][cognate_id].append(word)

        # No separation of cognates/non-cognates:
        # all synonymous words are evaluated irrespective of cognacy
        # The concept itself is used as a dummy cognate class ID
        # NB: this logic will not work if the base concept ID already encodes cognate class
        elif cognates == 'none':
            clustered_concepts = {concept: {concept: [
                word for lang in self.concepts[concept]
                for word in self.concepts[concept][lang]]} for concept in concept_list}

        # Raise error for unrecognized cognate clustering methods
        else:
            self.logger.error(f'Cognate clustering method "{cognates}" not recognized!')
            raise ValueError

        # Compute distance matrix over Language objects
        dm = distance_matrix(group=list(self.languages.values()),
                             dist_func=dist_func,
                             clustered_cognates=clustered_concepts,
                             **kwargs)

        # Store computed distance matrix
        self.distance_matrices[code] = dm

        # Write distance matrix file
        if outfile is None:
            outfile = os.path.join(self.dist_matrix_dir, f'{code}.tsv')
        self.write_distance_matrix(dm, outfile)

        return dm

    def linkage_matrix(self,
                       dist_func,
                       cluster_func=None,
                       concept_list=None,
                       cognates='auto',
                       linkage_method='ward',
                       metric='euclidean',
                       **kwargs):

        # Ensure the linkage method is valid
        valid_linkage = {'nj', 'average', 'complete', 'single', 'weighted', 'ward'}
        if linkage_method not in valid_linkage:
            raise ValueError(f'Error: Unrecognized linkage type "{linkage_method}". Accepted values are: {valid_linkage}')

        # Create distance matrix
        dm = self.distance_matrix(dist_func=dist_func,
                                  cluster_func=cluster_func,
                                  concept_list=concept_list,
                                  cognates=cognates,
                                  **kwargs
                                  )
        dists = squareform(dm)

        # Neighbor Joining linkage
        if linkage_method == 'nj':
            languages = self.languages.values()
            names = [lang.name for lang in languages]
            lang_names = [re.sub(r'\(', '{', lang) for lang in names]
            lang_names = [re.sub(r'\)', '}', lang) for lang in lang_names]
            nj_dm = DistanceMatrix(dists, ids=lang_names)
            return nj_dm

        # Other linkage methods
        else:
            lm = linkage(dists, linkage_method, metric)
            return lm

    def write_distance_matrix(self, dist_matrix, outfile, ordered_labels=None, float_format="%.5f"):
        """Writes numpy distance matrix object to a TSV with decimals rounded to 5 places by default"""

        languages = [self.languages[lang] for lang in self.languages]
        names = [lang.name for lang in languages]

        # np.savetxt(outfile, dist_matrix, delimiter='\t', fmt=fmt)

        # Create a DataFrame using the distance matrix and labels
        df = pd.DataFrame(dist_matrix, index=names, columns=names)

        # Reorder the columns and rows based on the ordered list of labels
        if ordered_labels:
            ordered_labels = [label for label in ordered_labels if label in names]
            if len(ordered_labels) == df.shape[0]:  # only if the dimensions match
                df = df.reindex(index=ordered_labels, columns=ordered_labels)
                names = ordered_labels

        # Add an empty column and row for the labels
        df.insert(0, "Labels", names)
        df.insert(0, " ", [" "] * len(names))
        df.to_csv(outfile, sep='\t', index=False, float_format=float_format)

    def draw_tree(self,
                  dist_func,
                  concept_list=None,
                  cluster_func=None,
                  cognates='auto',
                  linkage_method='ward',
                  metric='euclidean',
                  outtree=None,
                  title=None,
                  save_directory=None,
                  return_newick=False,
                  orientation='left',
                  p=30,
                  **kwargs):

        group = [self.languages[lang] for lang in self.languages]
        labels = [lang.name for lang in group]
        code = self.generate_test_code(dist_func, cognates, **kwargs)

        if title is None:
            title = f'{self.name}'
        if save_directory is None:
            save_directory = self.plots_dir
        if outtree is None:
            _, timestamp = create_timestamp()
            outtree = os.path.join(self.tree_dir, f'{timestamp}.tre')

        lm = self.linkage_matrix(dist_func,
                                 concept_list=concept_list,
                                 cluster_func=cluster_func,
                                 cognates=cognates,
                                 linkage_method=linkage_method,
                                 metric=metric,
                                 **kwargs)

        # Not possible to plot NJ trees in Python (yet? TBD) # TODO
        if linkage_method != 'nj':
            sns.set(font_scale=1.0)
            if len(group) >= 100:
                plt.figure(figsize=(20, 20))
            elif len(group) >= 60:
                plt.figure(figsize=(10, 10))
            else:
                plt.figure(figsize=(10, 8))

            dendrogram(lm, p=p, orientation=orientation, labels=labels)
            if title:
                plt.title(title, fontsize=30)
            plt.savefig(f'{save_directory}{title}.png', bbox_inches='tight', dpi=300)
            plt.show()

        if return_newick or outtree:
            if linkage_method == 'nj':
                newick_tree = nj(lm, disallow_negative_branch_length=True, result_constructor=str)
            else:
                newick_tree = linkage2newick(lm, labels)

            # Fix formatting of Newick string
            newick_tree = re.sub(r'\s', '_', newick_tree)
            newick_tree = re.sub(r',_', ',', newick_tree)

            # Write tree to file
            if outtree:
                with open(outtree, 'w') as f:
                    f.write(newick_tree)

            return newick_tree

    def plot_languages(self,
                       dist_func,
                       concept_list=None,
                       cluster_func=None,
                       cognates='auto',
                       dimensions=2,
                       top_connections=0.3,
                       max_dist=1,
                       alpha_func=None,
                       plotsize=None,
                       invert_xaxis=False,
                       invert_yaxis=False,
                       title=None,
                       save_directory=None,
                       **kwargs):

        # Get lists of language objects and their labels
        group = [self.languages[lang] for lang in self.languages]
        labels = [lang.name for lang in group]

        # Compute a distance matrix
        dm = self.distance_matrix(dist_func=dist_func,
                                  concept_list=concept_list,
                                  cluster_func=cluster_func,
                                  cognates=cognates,
                                  **kwargs)

        # Use MDS to compute coordinate embeddings from distance matrix
        coords = dm2coords(dm, dimensions)

        # Set the plot dimensions
        sns.set(font_scale=1.0)
        if plotsize is None:
            x_coords = [coords[i][0] for i in range(len(coords))]
            y_coords = [coords[i][1] for i in range(len(coords))]
            x_range = max(x_coords) - min(x_coords)
            y_range = max(y_coords) - min(y_coords)
            y_ratio = y_range / x_range
            n = max(10, round((len(group) / 10) * 2))
            plotsize = (n, n * y_ratio)
        plt.figure(figsize=plotsize)

        # Draw scatterplot points
        plt.scatter(coords[:, 0], coords[:, 1], marker='o')

        # Add labels to points
        for label, x, y in zip(labels, coords[:, 0], coords[:, 1]):
            plt.annotate(
                label,
                xy=(x, y), xytext=(5, -5),
                textcoords='offset points', ha='left', va='bottom',
            )

        # Add lines connecting points with a distance under a certain threshold
        connected = []
        for i in range(len(coords)):
            for j in range(len(coords)):
                if (j, i) not in connected:
                    dist = dm[i][j]
                    if dist <= max_dist:
                        # if dist <= np.mean(dm[i]): # if the distance is lower than average
                        if dist in np.sort(dm[i])[1:round(top_connections * (len(dm) - 1))]:
                            coords1, coords2 = coords[i], coords[j]
                            x1, y1 = coords1
                            x2, y2 = coords2
                            if alpha_func is None:
                                plt.plot([x1, x2], [y1, y2], alpha=1 - dist)
                            else:
                                plt.plot([x1, x2], [y1, y2], alpha=alpha_func(dist))
                            connected.append((i, j))

        # Optionally invert axes
        if invert_yaxis:
            plt.gca().invert_yaxis()
        if invert_xaxis:
            plt.gca().invert_xaxis()

        # Save the plot
        if title is None:
            title = f'{self.name} plot'
            if save_directory is None:
                save_directory = self.plots_dir
            plt.savefig(f'{os.path.join(save_directory, title)}.png', bbox_inches='tight', dpi=300)

        # Show the figure
        plt.show()
        plt.close()

    def draw_network(self,
                     dist_func,
                     concept_list=None,
                     cluster_func=None,
                     cognates='auto',
                     method='spring',
                     title=None,
                     save_directory=None,
                     network_function=newer_network_plot,
                     **kwargs):

        # Use all available concepts by default
        if concept_list is None:
            concept_list = sorted([concept for concept in self.concepts.keys()
                                   if len(self.concepts[concept]) > 1])
        else:
            concept_list = sorted([concept for concept in concept_list
                                   if len(self.concepts[concept]) > 1])

        # Automatic cognate clustering
        if cognates == 'auto':
            assert cluster_func is not None
            code = self.generate_test_code(cluster_func, cognates, **kwargs)
            if code in self.clustered_cognates:
                clustered_concepts = self.clustered_cognates[code]
            else:
                clustered_concepts = self.cluster_cognates(concept_list, dist_func=cluster_func)

        # Use gold cognate classes
        elif cognates == 'gold':
            clustered_concepts = defaultdict(lambda: defaultdict(lambda: []))
            for concept in concept_list:
                cognate_ids = [cognate_id for cognate_id in self.cognate_sets
                               if cognate_id.split('_')[0] == concept]
                for cognate_id in cognate_ids:
                    for lang in self.cognate_sets[cognate_id]:
                        for form in self.cognate_sets[cognate_id][lang]:
                            form = strip_ch(form, ['(', ')'])
                            clustered_concepts[concept][cognate_id].append(f'{lang} /{form}/')

        # No separation of cognates/non-cognates:
        # all synonymous words are evaluated irrespective of cognacy
        elif cognates == 'none':
            clustered_concepts = {concept: {concept: [f'{lang} /{self.concepts[concept][lang][i][1]}/'
                                  for lang in self.concepts[concept]
                                  for i in range(len(self.concepts[concept][lang]))]}
                                  for concept in concept_list}

        # Raise error for unrecognized cognate clustering methods
        else:
            raise ValueError(f'Error: cognate clustering method "{cognates}" not recognized!')

        # Dendrogram characteristics
        languages = [self.languages[lang] for lang in self.languages]
        names = [lang.name for lang in languages]
        if title is None:
            title = f'{self.name} network'
        if save_directory is None:
            save_directory = self.plots_dir

        return network_function(group=languages,
                                labels=names,
                                dist_func=dist_func,
                                sim=dist_func.sim,
                                method=method,
                                title=title,
                                save_directory=save_directory,
                                clustered_cognates=clustered_concepts,
                                **kwargs)

    def examine_cognates(self, language_list=None, concepts=None, cognate_sets=None,
                         min_langs=2):
        if language_list is None:
            language_list = self.languages.values()
        else:
            language_list = [self.languages[lang] for lang in language_list]

        if (concepts is None) and (cognate_sets is None):
            cognate_sets = sorted(list(self.cognate_sets.keys()))

        elif concepts:
            cognate_sets = []
            for concept in concepts:
                cognate_sets.extend([c for c in self.cognate_sets if '_'.join(c.split('_')[:-1]) == concept])

        for cognate_set in cognate_sets:
            lang_count = [lang for lang in language_list if lang.name in self.cognate_sets[cognate_set]]
            if len(lang_count) >= min_langs:
                print(cognate_set)
                for lang in lang_count:
                    print(f'{lang.name}: {" ~ ".join(self.cognate_sets[cognate_set][lang.name])}')
                print('\n')

    def remove_languages(self, langs_to_delete):
        """Removes a list of languages from a dataset"""

        for lang in langs_to_delete:
            try:
                del self.languages[lang]
                del self.lang_ids[lang]
                del self.glottocodes[lang]
                del self.iso_codes[lang]
            except KeyError:
                pass

            for concept in self.concepts:
                try:
                    del self.concepts[concept][lang]
                except KeyError:
                    pass

            for cognate_set in self.cognate_sets:
                try:
                    del self.cognate_sets[cognate_set][lang]
                except KeyError:
                    pass

        # Remove empty concepts and cognate sets
        self.concepts = default_dict({concept: self.concepts[concept]
                                      for concept in self.concepts
                                      if len(self.concepts[concept]) > 0},
                                     lmbda=defaultdict(lambda: []))

        self.cognate_sets = default_dict({cognate_set: self.cognate_sets[cognate_set]
                                          for cognate_set in self.cognate_sets
                                          if len(self.cognate_sets[cognate_set]) > 0},
                                         lmbda=defaultdict(lambda: []))

    def subset(self, new_name, include=None, exclude=None, **kwargs):
        """Creates a subset of the existing dataset, including only select languages"""

        new_dataset = copy.deepcopy(self)

        # Remove languages not part of new subset
        if include:
            to_delete = [lang for lang in self.languages if lang not in include]
        else:
            assert exclude is not None
            to_delete = exclude
        new_dataset.remove_languages(to_delete)

        # Assign the new name and any other specified attributes
        new_dataset.name = new_name
        for key, value in kwargs.items():
            setattr(new_dataset, key, value)

        # Recalculate mutual coverage among remaining langauges
        new_dataset.mutual_coverage = new_dataset.calculate_mutual_coverage()

        return new_dataset

    def add_language(self, name, data_path, **kwargs):
        self.load_data(data_path, included_doculects=[name], **kwargs)

    def __str__(self):
        """Print a summary of the Family object"""
        s = f'{self.name.upper()}'
        s += f'\nLanguages: {len(self.languages)}'
        s += f'\nConcepts: {len(self.concepts)}\nCognate Classes: {len(self.cognate_sets)}'

        return s


class Language:
    def __init__(self,
                 name,
                 data,
                 columns,
                 transcription_params=TRANSCRIPTION_PARAM_DEFAULTS,
                 alignment_params=ALIGNMENT_PARAM_DEFAULTS,
                 lang_id=None,
                 glottocode=None,
                 iso_code=None,
                 family=None,
                 ):

        # Language data
        self.name = name
        self.path_name = format_as_variable(name)
        self.lang_id = lang_id
        self.glottocode = glottocode
        self.iso_code = iso_code
        self.family = family

        # Attributes for parsing data dictionary (TODO could this be inherited via a subclass?)
        self.data = data
        self.columns = columns

        # Phonemic inventory
        self.phonemes = defaultdict(lambda: 0)
        self.vowels = defaultdict(lambda: 0)
        self.consonants = defaultdict(lambda: 0)
        self.tonemes = defaultdict(lambda: 0)
        self.tonal = False

        # Phonological contexts
        self.unigrams = defaultdict(lambda: 0)
        self.bigrams = defaultdict(lambda: 0)
        self.trigrams = defaultdict(lambda: 0)
        self.ngrams = defaultdict(lambda: defaultdict(lambda: 0))
        self.gappy_bigrams = defaultdict(lambda: 0)
        self.gappy_trigrams = defaultdict(lambda: 0)
        self.phon_environments = defaultdict(lambda: defaultdict(lambda: 0))
        self.phon_env_ngrams = defaultdict(lambda: defaultdict(lambda: 0))

        # Lexical inventory
        self.vocabulary = defaultdict(lambda: set())
        self.loanwords = defaultdict(lambda: set())

        # Transcription, segmentation, and alignment parameters
        self.transcription_params = transcription_params
        if self.transcription_params['ignore_stress']:
            self.transcription_params['ch_to_remove'].update({'ˈ', 'ˌ'})
        if self.transcription_params['suprasegmentals']:
            self.transcription_params['suprasegmentals'] = suprasegmental_diacritics.union(self.transcription_params['suprasegmentals'])
        self.alignment_params = alignment_params

        # Initialize vocabulary and phoneme inventory
        self.create_vocabulary()
        self.create_phoneme_inventory()
        self.write_phoneme_inventory()
        self.phoneme_entropy = entropy(self.phonemes)

        # Comparison with other languages
        self.phoneme_correlators = {}
        self.phoneme_correspondences = defaultdict(lambda: defaultdict(lambda: 0))
        self.phoneme_pmi = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.complex_ngrams = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.phoneme_surprisal = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: -self.phoneme_entropy)))
        self.phon_env_surprisal = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: -self.phoneme_entropy)))
        self.noncognate_thresholds = defaultdict(lambda: [])
        self.lexical_comparison = defaultdict(lambda: defaultdict(lambda: {}))
        self.lexical_comparison['measures'] = set()

    def create_vocabulary(self):
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
                language=self,
                cognate_class=cognate_class,
                loanword=loan,
            )
            if len(word.segments) > 0:
                self.vocabulary[concept].add(word)

                # Mark known loanwords
                if loan:
                    self.loanwords[concept].add(word)

    def write_missing_concepts(self):
        missing_lst = os.path.join(self.family.doculects_dir, self.path_name, 'missing_concepts.lst')
        missing_concepts = '\n'.join(sorted(list(self.family.concepts.keys() - self.vocabulary.keys())))
        with open(missing_lst, 'w') as f:
            f.write(missing_concepts)

    def create_phoneme_inventory(self, warn_n=1):
        pad_ch = self.alignment_params['pad_ch']
        for concept in self.vocabulary:
            for word in self.vocabulary[concept]:
                segments = word.segments

                # Count phones and unigrams
                for segment in segments:
                    self.phonemes[segment] += 1
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
        total_tokens = sum(self.phonemes.values())
        for phoneme in self.phonemes:
            count = self.phonemes[phoneme]
            if count <= warn_n:
                self.family.logger.warning(f'Only {count} instance(s) of /{phoneme}/ in {self.name}.')
            self.phonemes[phoneme] = count / total_tokens

        # Phone classes
        phone_classes = {p: _toSegment(p).phone_class for p in self.phonemes}

        # Get dictionaries of vowels and consonants
        self.vowels = normalize_dict({v: self.phonemes[v]
                                      for v in self.phonemes
                                      if phone_classes[v] in ('VOWEL', 'DIPHTHONG')},
                                     default=True, lmbda=0)

        self.consonants = normalize_dict({c: self.phonemes[c]
                                         for c in self.phonemes
                                         if phone_classes[c] in ('CONSONANT', 'GLIDE')},
                                         default=True, lmbda=0)

        # TODO: rename as self.suprasegmentals, possibly distinguish tonemes from other suprasegmentals
        self.tonemes = normalize_dict({t: self.phonemes[t]
                                       for t in self.phonemes
                                       if phone_classes[t] in ('TONEME', 'SUPRASEGMENTAL')},
                                      default=True, lmbda=0)

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
        doculect_dir = os.path.join(self.family.doculects_dir, self.path_name)
        os.makedirs(doculect_dir, exist_ok=True)
        random.seed(seed)
        vowels, consonants, tonemes = map(dict_tuplelist, [self.vowels, self.consonants, self.tonemes])
        with open(os.path.join(doculect_dir, 'phones.lst'), 'w') as f:
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
                with open(os.path.join(doculect_dir, file), 'w') as f:
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
                        self.ngrams[ngram_size][ngram.ngram] += 1
                        if phon_env:
                            phon_env_ngram = Ngram(padded_phon_env[i:i + ngram_size])
                            self.phon_env_ngrams[ngram_size][phon_env_ngram.ngram] += 1

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
            language=self,
            cognate_class=cognate_class,
            loanword=loan,
        )

        return word

    def calculate_infocontent(self, word, as_seq=False, ngram_size=3, **kwargs):
        # Disambiguation by type of input
        if isinstance(word, Word):  # Word object, no further modification needed
            pass
        elif as_seq and isinstance(word, (list, tuple)):
            pass
        elif isinstance(word, str) or isinstance(word, list):
            if isinstance(word, str):
                ipa_string = word
            else:
                ipa_string = ''.join(word)
            word = Word(
                ipa_string=ipa_string,
                transcription_parameters=self.transcription_params,
                language=self)
        else:
            raise TypeError

        # Return the pre-calculated information content of the word, if possible
        if isinstance(word, Word) and word.info_content:
            return word.info_content

        # Otherwise calculate it from scratch
        # Pad the segmented word
        pad_ch = self.alignment_params['pad_ch']
        sequence = word.segments if not as_seq else word
        pad_n = ngram_size - 1
        padded = pad_sequence(sequence, pad_ch=pad_ch, pad_n=pad_n)
        info_content = calculate_infocontent_of_word(seq=padded, lang=self, ngram_size=ngram_size, **kwargs)
        return info_content

    def self_surprisal(self, word, normalize=False, **kwargs):
        info_content = self.calculate_infocontent(word, **kwargs)
        if normalize:
            return mean(info_content[j][1] for j in info_content)
        else:
            # return sum(info_content[j][1] for j in info_content)
            return info_content

    def ngram_probability(self, ngram):
        ngram = Ngram(ngram)
        if ngram.size not in self.ngrams:
            self.list_ngrams(ngram.size)
        prob = self.ngrams[ngram.size][ngram.ngram] / sum(self.ngrams[ngram.size].values())
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

    def get_phoneme_correlator(self, lang2, wordlist=None, seed=1):
        key = (lang2, wordlist, seed)
        if key not in self.phoneme_correlators:
            self.phoneme_correlators[key] = PhonCorrelator(lang1=self,
                                                           lang2=lang2,
                                                           wordlist=wordlist,
                                                           gap_ch=self.alignment_params.get('gap_ch', ALIGNMENT_PARAM_DEFAULTS['gap_ch']),
                                                           pad_ch=self.alignment_params.get('pad_ch', ALIGNMENT_PARAM_DEFAULTS['pad_ch']),
                                                           seed=seed,
                                                           logger=self.family.logger)
        correlator = self.phoneme_correlators[key]
        return correlator

    def write_lexical_comparison(self, lang2, outfile):
        measures = sorted(list(self.lexical_comparison['measures']))
        with open(outfile, 'w') as f:
            header = '\t'.join([self.name, lang2.name] + measures)
            f.write(f'{header}\n')
            for word1, word2 in self.lexical_comparison[lang2]:
                values = [self.lexical_comparison[lang2][(word1, word2)].get(measure, 'n/a') for measure in measures]
                values = [str(v) for v in values]
                line = '\t'.join([word1.ipa, word2.ipa] + values)
                f.write(f'{line}\n')

    def __str__(self):
        """Print a summary of the language object"""
        # TODO improve this
        s = f'{self.name.upper()} [{self.glottocode}][{self.iso_code}]'
        s += f'\nFamily: {self.family.name}'
        s += f'\nRelatives: {len(self.family.languages)}'
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


class Word:
    def __init__(self,
                 ipa_string,
                 concept=None,
                 orthography=None,
                 language=None,
                 cognate_class=None,
                 loanword=False,
                 transcription_parameters=TRANSCRIPTION_PARAM_DEFAULTS,
                 ):
        self.language = language
        self.parameters = transcription_parameters
        self.raw_ipa = ipa_string
        self.ipa = self.preprocess(ipa_string)
        self.concept = concept
        self.cognate_class = cognate_class
        self.loanword = loanword
        self.orthography = orthography
        self.segments = self.segment()
        self.syllables = None
        self.phon_env = self.getPhonEnv()
        self.info_content = None

    def get_parameter(self, label):
        return self.parameters.get(label, TRANSCRIPTION_PARAM_DEFAULTS[label])

    def preprocess(self, ipa_string):

        # Normalize common IPA character mistakes
        # Normalize affricates to special ligature characters, where available
        ipa_string = normalize_ipa_ch(ipa_string)

        # Normalize geminate consonants to /Cː/
        if self.get_parameter('normalize_geminates'):
            ipa_string = normalize_geminates(ipa_string)

        # Level any suprasegmentals to stress annotation
        supraseg_target = self.get_parameter('level_suprasegmentals')
        if supraseg_target:
            suprasegs = self.get_parameter('suprasegmentals')
            ipa_string = re.sub(fr'[{suprasegs}]', supraseg_target, ipa_string)

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

    def get_syllables(self, **kwargs):
        self.syllables = syllabify(
            word=self.ipa,
            segments=self.segments,
            **kwargs
        )
        return self.syllables

    def getPhonEnv(self):
        phon_env = []
        for i, seg in enumerate(self.segments):
            phon_env.append(get_phon_env(self.segments, i))
        return phon_env

    def getInfoContent(self):
        if self.language is None:
            raise AssertionError('Language must be specified in order to calculate information content.')

        self.info_content = self.language.calculate_infocontent(self)
        return self.info_content

    def __str__(self):
        syllables = self.get_syllables()
        syl_tr = ".".join(syllable.syl for i, syllable in syllables.items())
        form_tr = "/" + syl_tr + "/"
        if self.orthography and self.orthography != "":
            form_tr = f"<{self.orthography}> {syl_tr}"
        if self.concept and self.concept != "":
            form_tr = f"{form_tr}\n'{self.concept}'"
        if self.language:
            form_tr = f"{form_tr} ({self.language.name})"
        return form_tr


# COMBINING DATASETS
def combine_datasets(dataset_list):
    # TODO
    pass


def load_family(family,
                data_file,
                min_amc=None,
                concept_list=None,
                excluded_doculects=None,
                included_doculects=None,
                logger=None,
                **kwargs):
    family = LexicalDataset(data_file,
                            family,
                            excluded_doculects=excluded_doculects,
                            included_doculects=included_doculects,
                            logger=logger,
                            **kwargs)
    if min_amc:
        family.prune_languages(min_amc=float(min_amc), concept_list=concept_list)
    family.write_lexical_index()
    language_variables = {format_as_variable(lang): family.languages[lang]
                          for lang in family.languages}
    globals().update(language_variables)
    return family
