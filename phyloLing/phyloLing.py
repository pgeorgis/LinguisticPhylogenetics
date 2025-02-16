import ast
import copy
import logging
import os
import re
from collections import defaultdict
from collections.abc import Iterable
from itertools import combinations, product
from math import sqrt
from statistics import mean, stdev
from tqdm import tqdm

import bcubed
import pandas as pd
from constants import (ALIGNMENT_DELIMITER, ALIGNMENT_KEY_REGEX,
                       ALIGNMENT_PARAM_DEFAULTS, COGNATE_CLASS_LABEL,
                       CONCEPT_LABEL, DOCULECT_INDEX_KEY, GLOTTOCODE_LABEL,
                       ID_COLUMN_LABEL, ISO_CODE_LABEL, LANGUAGE_NAME_LABEL,
                       LOAN_LABEL, ORTHOGRAPHY_LABEL,
                       PHONE_CORRELATORS_INDEX_KEY, PHONETIC_FORM_LABEL,
                       SEG_JOIN_CH, SEGMENTS_LABEL,
                       TRANSCRIPTION_PARAM_DEFAULTS)
from lingDist import get_noncognate_scores
from phonAlign import init_precomputed_alignment
from phonCorr import get_phone_correlator
from phonUtils.ipaTools import invalid_ch, normalize_ipa_ch, strip_diacritics
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from skbio import DistanceMatrix
from skbio.tree import nj
from unidecode import unidecode
from utils.cluster import cluster_items, linkage2newick
from utils.distance import Distance, distance_matrix
from utils.doculect import Doculect
from utils.phoneme_map import PhonemeMap
from utils.sequence import Ngram
from utils.string import format_as_variable, strip_ch
from utils.tree import postprocess_newick, reroot_tree
from utils.utils import create_default_dict_of_dicts, csv2dict, default_dict

logger = logging.getLogger(__name__)
FAMILY_INDEX: dict[str, dict] = {}

class LexicalDataset:
    def __init__(self, filepath, name,
                 outdir=None,
                 id_c=ID_COLUMN_LABEL,
                 language_name_c=LANGUAGE_NAME_LABEL,
                 concept_c=CONCEPT_LABEL,
                 orthography_c=ORTHOGRAPHY_LABEL,
                 ipa_c=PHONETIC_FORM_LABEL,
                 segments_c=SEGMENTS_LABEL,
                 cognate_class_c=COGNATE_CLASS_LABEL,
                 loan_c=LOAN_LABEL,
                 glottocode_c=GLOTTOCODE_LABEL,
                 iso_code_c=ISO_CODE_LABEL,
                 included_doculects=None,
                 excluded_doculects=None,
                 transcription_params={'global': TRANSCRIPTION_PARAM_DEFAULTS},
                 alignment_params=ALIGNMENT_PARAM_DEFAULTS,
                 ):

        # Dataset name and logger
        self.data: dict[int, dict[str, str]] = {}
        self.name = name
        self.path_name = format_as_variable(name)
        logger.info(f'Loading {self.name}...')

        # Directory to dataset
        self.filepath = filepath
        self.directory = outdir if outdir else os.path.dirname(os.path.abspath(filepath))

        # Create a folder for plots and detected cognate sets within the dataset's directory
        self.plots_dir = os.path.join(self.directory, 'plots')
        self.cognates_dir = os.path.join(self.directory, 'cognates')
        self.phone_corr_dir = os.path.join(self.directory, 'phone_corr')
        self.doculects_dir = os.path.join(self.directory, 'doculects')
        self.tree_dir = os.path.join(self.directory, 'trees')
        for dir in (
            self.plots_dir,
            self.cognates_dir,
            self.phone_corr_dir,
            self.doculects_dir,
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
        self.languages: dict[str, Doculect] = {}
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
        language_vocab_data = create_default_dict_of_dicts(2)
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
            path_name = format_as_variable(lang)
            self.languages[lang] = Doculect(
                name=lang,
                lang_id=self.lang_ids[lang],
                glottocode=self.glottocodes[lang],
                iso_code=self.iso_codes[lang],
                doculect_dir=os.path.join(self.doculects_dir, path_name),
                data=language_vocab_data[lang],
                columns=self.columns,
                transcription_params=self.transcription_params.get('doculects', {}).get(lang, self.transcription_params['global']),
                alignment_params=self.alignment_params,
            )
            logger.info(f'Loaded doculect {lang}.')
            for concept in self.languages[lang].vocabulary:
                self.concepts[concept][lang].extend(self.languages[lang].vocabulary[concept])
        for lang in language_list:
            self.write_missing_concepts(lang)
    
    def write_missing_concepts(self, doculect):
        """Writes a missing_concepts.lst file indicating which concepts present
        in the lexical dataset are missing from a particular doculect."""
        doculect = self.languages[doculect]
        missing_lst = os.path.join(doculect.doculect_dir, "missing_concepts.lst")
        missing_concepts = self.concepts.keys() - doculect.vocabulary.keys()
        missing_concepts = '\n'.join(sorted(missing_concepts))
        with open(missing_lst, 'w') as f:
            f.write(missing_concepts)

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
            logger.info(prune_log)

    def calculate_phone_corrs(self, **kwargs):
        """Calculates phone correspondence values for all
        language pairs in the dataset and saves the results to file"""
        lang_pairs = self.get_doculect_pairs(bidirectional=False)
        with tqdm(total=len(lang_pairs), unit="pair") as pbar:
            for lang1, lang2 in lang_pairs:
                pbar.set_description(f"Computing phone correspondences... [{lang1.name}-{lang2.name}]")
                # Retrieve or initialize PhoneCorrelator for this pair
                correlator, FAMILY_INDEX[self.name][PHONE_CORRELATORS_INDEX_KEY] = get_phone_correlator(
                    lang1,
                    lang2,
                    phone_correlators_index=FAMILY_INDEX[self.name][PHONE_CORRELATORS_INDEX_KEY],
                    log_outdir=self.phone_corr_dir,
                )
                # Check whether phone correspondences habe been calculated already for this pair
                # If not, calculate it now
                if len(correlator.pmi_results) == 0:
                    correlator.compute_phone_corrs(
                        family_index=FAMILY_INDEX[self.name],
                        **kwargs
                    )
                pbar.update(1)

    def compute_noncognate_thresholds(self, eval_func, doculect_pairs=None, **kwargs):
        """Computes the mean and standard deviation score between a sample of non-synonymous word pairs from a set of doculects according to a specified evaluation function."""
        combined_noncognate_scores = []
        if doculect_pairs is None:
            doculect_pairs = self.get_doculect_pairs()
        for lang1, lang2 in doculect_pairs:
            logger.info(f"Computing non-cognate thresholds: {lang1.name}-{lang2.name}")
            noncognate_scores, FAMILY_INDEX[self.name][PHONE_CORRELATORS_INDEX_KEY] = get_noncognate_scores(
                lang1,
                lang2,
                eval_func=eval_func,
                phone_correlators_index=FAMILY_INDEX[self.name][PHONE_CORRELATORS_INDEX_KEY],
                log_outdir=self.phone_corr_dir,
                **kwargs
            )
            combined_noncognate_scores.extend(noncognate_scores)
        mean_nc_score = mean(noncognate_scores)
        nc_score_stdev = stdev(noncognate_scores)
        return mean_nc_score, nc_score_stdev

    def load_alignments(self, excepted=[], **kwargs):
        """Loads pre-computed phonetic sequence alignments from file."""
        
        def parse_alignment_log(alignment_log, lang1, lang2, **kwargs):
            align_dict = {}
            with open(alignment_log, "r") as f:
                f = f.read()
                alignment_sections = [section.strip() for section in f.split(ALIGNMENT_DELIMITER)]
                for alignment_section in alignment_sections:
                    align_key, alignment, seq_map1, seq_map2, cost = alignment_section.split("\n")
                    seq_map1 = ast.literal_eval(seq_map1)
                    seq_map2 = ast.literal_eval(seq_map2)
                    cost = float(re.search(r"[\d\.]+", cost).group())
                    alignment = init_precomputed_alignment(
                        alignment.strip(),
                        align_key,
                        seq_map=(seq_map1, seq_map2),
                        cost=cost,
                        lang1=lang1,
                        lang2=lang2,
                        **kwargs
                    )
                    align_dict[align_key.strip()] = alignment
            return align_dict
        
        for lang1, lang2 in self.get_doculect_pairs(bidirectional=False):
            if (lang1.name not in excepted) and (lang2.name not in excepted):
                alignment_file = os.path.join(
                    self.phone_corr_dir,
                    lang1.path_name,
                    lang2.path_name,
                    "alignments.log",
                )
                align_dict = parse_alignment_log(alignment_file, lang1=lang1, lang2=lang2, **kwargs)
                # Get reverse alignments dict
                reverse_align_dict = {}
                for key, alignment in align_dict.items():
                    reverse_key = ALIGNMENT_KEY_REGEX.sub(r"/\2/ - /\1/", key)
                    reverse_alignment = alignment.reverse()
                    reverse_align_dict[reverse_key] = reverse_alignment
                
                # Fetch correlators
                correlator, FAMILY_INDEX[self.name][PHONE_CORRELATORS_INDEX_KEY] = get_phone_correlator(
                    lang1,
                    lang2,
                    phone_correlators_index=FAMILY_INDEX[self.name][PHONE_CORRELATORS_INDEX_KEY],
                    log_outdir=self.phone_corr_dir,
                )
                twin, FAMILY_INDEX[self.name][PHONE_CORRELATORS_INDEX_KEY] = correlator.get_twin(
                    phone_correlators_index=FAMILY_INDEX[self.name][PHONE_CORRELATORS_INDEX_KEY],
                )
                
                # Update correlators with alignments
                correlator.align_log.update(align_dict)
                twin.align_log.update(reverse_align_dict)

    def load_phoneme_pmi(self, excepted=[], sep='\t', **kwargs):
        """Loads pre-calculated phoneme PMI values from file"""

        def str2ngram(str, join_ch='_'):
            return Ngram(str, lang=self, seg_sep=join_ch)
        doculect_pairs = self.get_doculect_pairs(bidirectional=False)
        with tqdm(total=len(doculect_pairs), unit="pair") as pbar:
            for lang1, lang2 in doculect_pairs:
                pbar.set_description(f"Loading phoneme PMI... [{lang1.name}-{lang2.name}]")
                if (lang1.name not in excepted) and (lang2.name not in excepted):
                    pmi_file = os.path.join(
                        self.phone_corr_dir,
                        lang1.path_name,
                        lang2.path_name,
                        "phonPMI.tsv"
                    )

                    # Try to load the file of saved PMI values, otherwise calculate PMI first
                    correlator, FAMILY_INDEX[self.name][PHONE_CORRELATORS_INDEX_KEY] = get_phone_correlator(
                        lang1,
                        lang2,
                        phone_correlators_index=FAMILY_INDEX[self.name][PHONE_CORRELATORS_INDEX_KEY],
                        log_outdir=self.phone_corr_dir,
                    )
                    twin, FAMILY_INDEX[self.name][PHONE_CORRELATORS_INDEX_KEY] = correlator.get_twin(
                        phone_correlators_index=FAMILY_INDEX[self.name][PHONE_CORRELATORS_INDEX_KEY],
                    )
                    if not os.path.exists(pmi_file):
                        correlator.compute_phone_corrs(
                            phone_correlators_index=FAMILY_INDEX[self.name],
                            **kwargs
                        )
                    pmi_data = pd.read_csv(pmi_file, sep=sep)

                    # Iterate through the dataframe and save the PMI values to the Language
                    # class objects' phoneme_pmi attribute
                    for _, row in pmi_data.iterrows():
                        phone1, phone2 = row['Phone1'], row['Phone2']
                        pmi_value = row['PMI']
                        ngram1, ngram2 = map(str2ngram, [phone1, phone2])
                        correlator.pmi_results.set_value(ngram1.undo(), ngram2.undo(), pmi_value)
                        twin.pmi_results.set_value(ngram2.undo(), ngram1.undo(), pmi_value)
                pbar.update(1)

    def write_phoneme_pmi(self, **kwargs):
        logger.info(f'Saving {self.name} phoneme PMI...')
        for lang1, lang2 in self.get_doculect_pairs(bidirectional=False):
            correlator, FAMILY_INDEX[self.name][PHONE_CORRELATORS_INDEX_KEY] = get_phone_correlator(
                lang1,
                lang2,
                phone_correlators_index=FAMILY_INDEX[self.name][PHONE_CORRELATORS_INDEX_KEY],
                log_outdir=self.phone_corr_dir,
            )

            # Retrieve the precalculated values
            # Skip rewriting PMI for doculect pairs for which PMI was not calculated in this run
            # This happens when recalculating for specific doculects only and loading others from file
            if len(correlator.pmi_results) == 0:
                logger.warning(f'Phoneme PMI has not been calculated for pair: {lang1.name} - {lang2.name}.')
                continue

            correlator.log_phoneme_pmi(**kwargs)

    def write_phoneme_surprisal(self, phon_env=True, ngram_size=1, **kwargs):
        logger.info(f'Saving {self.name} phoneme surprisal...')
        for lang1, lang2 in self.get_doculect_pairs(bidirectional=True):
            correlator, FAMILY_INDEX[self.name][PHONE_CORRELATORS_INDEX_KEY] = get_phone_correlator(
                lang1,
                lang2,
                phone_correlators_index=FAMILY_INDEX[self.name][PHONE_CORRELATORS_INDEX_KEY],
                log_outdir=self.phone_corr_dir,
            )

            # Retrieve the precalculated values
            if len(correlator.surprisal_results[ngram_size]) == 0:
                logger.warning(f'{ngram_size}-gram phoneme surprisal has not been calculated for pair: {lang1.name} - {lang2.name}')
                continue

            correlator.log_phoneme_surprisal(phon_env=False, ngram_size=ngram_size, **kwargs)
            if phon_env:
                correlator.log_phoneme_surprisal(phon_env=True, **kwargs)

    def load_phoneme_surprisal(self, ngram_size=1, phon_env=False, excepted=[], sep='\t', **kwargs):
        """Loads pre-calculated phoneme surprisal values from file"""

        def str2ngram(str, join_ch=SEG_JOIN_CH):
            return Ngram(str, lang=self, seg_sep=join_ch)

        def extract_surprisal_from_df(surprisal_data, lang2, phon_env=False):
            surprisal_dict = PhonemeMap(lang2.phoneme_entropy * ngram_size)
            for _, row in surprisal_data.iterrows():
                phone1, phone2 = row['Phone1'], row['Phone2']
                surprisal_value = row['Surprisal']
                if ngram_size > 1:
                    raise NotImplementedError  # TODO need to decide format for how to save/load larger ngrams from logs; previously they were separated by whitespace
                ngram1, ngram2 = map(str2ngram, [phone1, phone2])
                ngram2_dict_form = ngram2.undo()
                if phon_env:
                    env = row['PhonEnv']
                    ngram1_dict_form = (Ngram(phone1).undo(), env)
                else:
                    ngram1_dict_form = ngram1.undo()
                surprisal_dict.set_value(ngram1_dict_form, ngram2_dict_form, surprisal_value)

            return surprisal_dict

        for lang1, lang2 in self.get_doculect_pairs(bidirectional=True):
            if (lang1.name not in excepted) and (lang2.name not in excepted):
                phon_corr_dir = os.path.join(
                    self.phone_corr_dir,
                    lang1.path_name,
                    lang2.path_name,
                )
                surprisal_file = os.path.join(phon_corr_dir, 'phonSurprisal.tsv')
                if phon_env:
                    surprisal_file_phon_env = os.path.join(phon_corr_dir, 'phonEnvSurprisal.tsv')

                correlator, FAMILY_INDEX[self.name][PHONE_CORRELATORS_INDEX_KEY] = get_phone_correlator(
                    lang1,
                    lang2,
                    phone_correlators_index=FAMILY_INDEX[self.name][PHONE_CORRELATORS_INDEX_KEY],
                    log_outdir=self.phone_corr_dir,
                )
                # Try to load the file of saved surprisal values, otherwise calculate surprisal first
                if not os.path.exists(surprisal_file):
                    correlator.compute_phone_corrs(
                        phone_correlators_index=FAMILY_INDEX[self.name],
                        ngram_size=ngram_size,
                        phon_env=phon_env,
                        **kwargs
                    )
                surprisal_data = pd.read_csv(surprisal_file, sep=sep)

                # Extract and save the surprisal values to phoneme_surprisal attribute of language object
                loaded_surprisal = extract_surprisal_from_df(surprisal_data, lang2, phon_env=False)
                correlator.surprisal_results[ngram_size] = loaded_surprisal

                # Do the same for phonological environment surprisal
                if phon_env and os.path.exists(surprisal_file_phon_env):
                    phon_env_surprisal_data = pd.read_csv(surprisal_file_phon_env, sep=sep)
                    loaded_phon_env_surprisal = extract_surprisal_from_df(phon_env_surprisal_data, lang2, phon_env=True)
                    correlator.phon_env_surprisal_results = loaded_phon_env_surprisal

                elif phon_env:
                    logger.warning(f'No saved phonological environment surprisal file found for {lang1.name}-{lang2.name}')

    def get_doculect_pairs(self, bidirectional=False, include_self_pairs=True):
        if bidirectional:
            doculect_pairs = product(self.languages.values(), self.languages.values())
        else:
            doculect_pairs = combinations(self.languages.values(), 2)
        if include_self_pairs:
            doculect_pairs = list(doculect_pairs)
            doculect_pairs.extend([(lang, lang) for lang in self.languages.values()])
        return sorted([(lang1, lang2) for lang1, lang2 in doculect_pairs],
                      key=lambda x: (x[0].name, x[1].name))

    def cluster_cognates(self,
                         concept_list,
                         dist_func,
                         cluster_threshold=None,
                         code=None,
                         **kwargs):

        # TODO make option for instead using k-means clustering given a known/desired number of clusters, as a mutually exclusive parameter with cutoff
        concept_list = [concept for concept in concept_list if len(self.concepts[concept]) > 1]
        clustered_cognates = {}
        
        # Unless otherwise specified, compute cluster threshold based on mean and standard deviation
        # of non-synonymous word pair scores across all doculect pairs
        if cluster_threshold is None and dist_func.cluster_threshold is not None:
            cluster_threshold = dist_func.cluster_threshold
        elif cluster_threshold is None:
            mean_nc, stdev_nc = self.compute_noncognate_thresholds(dist_func)
            cluster_threshold = mean_nc - stdev_nc
            logger.info(f"Auto-computed cluster threshold: {round(cluster_threshold, 3)}")
        logger.info(f'Clustering cognates with threshold={round(cluster_threshold, 3)}...')

        for concept in sorted(concept_list):
            logger.info(f"Clustering cognates for concept '{concept}'...")
            words = [word for lang in self.concepts[concept] for word in self.concepts[concept][lang]]
            clusters = cluster_items(group=words,
                                     dist_func=dist_func,
                                     sim=dist_func.sim,
                                     cutoff=cluster_threshold,
                                     **kwargs)
            clustered_cognates[concept] = clusters

        # Create code and store the result
        if code is None:
            code = self.generate_test_code(dist_func, cognates='auto', cluster_threshold=cluster_threshold)
        self.clustered_cognates[code] = clustered_cognates

        return clustered_cognates, code

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
                    lang = word.doculect_key
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

        logger.info(f'Wrote clustered cognate index to {output_file}')

    def load_cognate_index(self, index_file, code=None, sep='\t', variants_sep='~'):
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

        if code:
            self.clustered_cognates[code] = index

        return index

    def evaluate_clusters(self, clustered_cognates, method='bcubed'):
        """Evaluates B-cubed precision, recall, and F1 of results of automatic
        cognate clustering against dataset's gold cognate classes"""

        precision_scores, recall_scores, f1_scores, mcc_scores = {}, {}, {}, {}
        ch_to_remove = self.transcription_params['global']['ch_to_remove'].union({'(', ')'})
        for concept in clustered_cognates:
            clusters = {'/'.join([strip_diacritics(unidecode.unidecode(item.split('/')[0])),
                                  strip_ch(item.split('/')[1], ch_to_remove)]) + '/': {i} for i in clustered_cognates[concept]
                        for item in clustered_cognates[concept][i]}

            gold_clusters = {f'{strip_diacritics(unidecode.unidecode(lang))} /{strip_ch(tr, ch_to_remove)}/': {c}
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
            logger.error(f'dist_func must be a Distance class object, found {type(dist_func)} instead.')
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
                        dm_outfile=None,
                        code=None,
                        **kwargs):

        # Try to skip re-calculation of distance matrix by retrieving
        # a previously computed distance matrix by its code
        if code and code in self.distance_matrices:
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

            if code and code in self.clustered_cognates:
                clustered_concepts = self.clustered_cognates[code]
            else:
                clustered_concepts, _ = self.cluster_cognates(concept_list, dist_func=cluster_func, code=code)

        # Use gold cognate classes
        elif cognates == 'gold':
            if 'gold' in self.clustered_cognates:
                clustered_concepts = self.clustered_cognates['gold']
            else:
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
            clustered_concepts = {
                concept: {
                    1: [word for lang in self.concepts[concept] for word in self.concepts[concept][lang]]
                }
                for concept in concept_list
            }

        # Raise error for unrecognized cognate clustering methods
        else:
            logger.error(f'Cognate clustering method "{cognates}" not recognized!')
            raise ValueError

        # Compute distance matrix over Doculect objects
        dm = distance_matrix(group=list(self.languages.values()),
                             dist_func=dist_func,
                             clustered_cognates=clustered_concepts,
                             **kwargs)

        # Store computed distance matrix
        self.distance_matrices[code] = dm
        
        # Write distance matrix to outfile
        if dm_outfile:
            outfile_dir = os.path.dirname(dm_outfile)
            os.makedirs(outfile_dir, exist_ok=True)
            self.write_distance_matrix(dm, outfile=dm_outfile)
            logger.info(f"Wrote distance matrix to {dm_outfile}")

        return dm

    def linkage_matrix(self,
                       dist_func,
                       cluster_func=None,
                       concept_list=None,
                       cognates='auto',
                       linkage_method='nj',
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

    def generate_tree(self,
                      dist_func,
                      concept_list=None,
                      cluster_func=None,
                      cognates='auto',
                      linkage_method='nj',
                      outtree=None,
                      root=None,
                      **kwargs):

        group = [self.languages[lang] for lang in self.languages]
        labels = [lang.name for lang in group]
        lm = self.linkage_matrix(dist_func,
                                 concept_list=concept_list,
                                 cluster_func=cluster_func,
                                 cognates=cognates,
                                 linkage_method=linkage_method,
                                 **kwargs)

        if linkage_method == 'nj':
            newick_tree = str(nj(lm, neg_as_zero=True))
        else:
            newick_tree = linkage2newick(lm, labels)

        # Fix formatting of Newick string
        newick_tree = postprocess_newick(newick_tree)
        
        # Optionally root the tree at a specified tip or clade
        if root:
            newick_tree = reroot_tree(newick_tree, root)

        # Write tree to file
        if outtree:
            with open(outtree, 'w') as f:
                f.write(newick_tree)

        return newick_tree

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

    def __str__(self):
        """Print a summary of the Family object"""
        s = f'{self.name.upper()}'
        s += f'\nDoculects: {len(self.languages)}'
        s += f'\nConcepts: {len(self.concepts)}\nCognate Classes: {len(self.cognate_sets)}'

        return s


def load_family(family,
                data_file,
                min_amc=None,
                concept_list=None,
                excluded_doculects=None,
                included_doculects=None,
                **kwargs):
    family = LexicalDataset(data_file,
                            family,
                            excluded_doculects=excluded_doculects,
                            included_doculects=included_doculects,
                            **kwargs)
    if min_amc:
        family.prune_languages(min_amc=float(min_amc), concept_list=concept_list)
    family.write_lexical_index()
    family_index = {
        DOCULECT_INDEX_KEY: family.languages,
        PHONE_CORRELATORS_INDEX_KEY: create_default_dict_of_dicts(2)
    }
    FAMILY_INDEX[family.name] = family_index
    return family, FAMILY_INDEX
