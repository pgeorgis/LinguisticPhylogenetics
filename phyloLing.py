import os, re, copy, glob
from collections import defaultdict
from itertools import product
from math import log, sqrt
from statistics import mean
import bcubed, random
from matplotlib import pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from skbio import DistanceMatrix
from skbio.tree import nj
import seaborn as sns
from unidecode import unidecode
import numpy as np
from auxFuncs import default_dict, normalize_dict, strip_ch, format_as_variable, csv2dict, dict_tuplelist
from auxFuncs import Distance, surprisal, entropy, distance_matrix, draw_dendrogram, linkage2newick, cluster_items, dm2coords, newer_network_plot
from PhoneticSimilarity.phonSim import vowels, consonants, tonemes, suprasegmental_diacritics
from PhoneticSimilarity.phonSim import normalize_ipa_ch, invalid_ch, strip_diacritics, segment_ipa, phone_sim, get_phon_env
from phonCorr import PhonemeCorrDetector
from lingDist import Z_score_dist
import logging

class LexicalDataset: 
    def __init__(self, filepath, name, 
                 id_c = 'ID',
                 language_name_c='Language_ID',
                 concept_c = 'Parameter_ID',
                 orthography_c = 'Value',
                 ipa_c = 'Form',
                 segments_c = 'Segments',
                 cognate_class_c = 'Cognate_ID',
                 loan_c = 'Loan',
                 glottocode_c='Glottocode',
                 iso_code_c='ISO 639-3',
                 ignore_stress=False,
                 combine_diphthongs=True,
                 logger=None):
        
        # Dataset name and logger
        self.name = name
        if logger:
            self.logger = logger
        else:
            # Configure the logger from scratch, default loglevel=INFO
            logging.basicConfig(level=logging.INFO, format='%(asctime)s phyloLing %(levelname)s: %(message)s')
            logger = logging.getLogger(__name__)
        self.logger.info(f'Loading {self.name}...')

        # Directory to dataset 
        self.filepath = filepath
        self.directory = self.filepath.rsplit('/', maxsplit=1)[0] + '/' # TODO this is bad form
        
        # Create a folder for plots and detected cognate sets within the dataset's directory
        self.plots_dir = os.path.join(self.directory, 'plots')
        self.cognates_dir = os.path.join(self.directory, 'cognates')
        self.phone_corr_dir = os.path.join(self.directory, 'phone_corr')
        self.dist_matrix_dir = os.path.join(self.directory, 'dist_matrices')
        self.tree_dir = os.path.join(self.directory, 'trees')
        for dir in (
            self.plots_dir, 
            self.cognates_dir, 
            self.phone_corr_dir, 
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
        
        # Transcription parameters
        self.ch_to_remove = suprasegmental_diacritics.union({' '})
        if not ignore_stress:
            self.ch_to_remove = self.ch_to_remove - {'ˈ', 'ˌ'}
        self.combine_diphthongs = combine_diphthongs
            
        # Concepts in dataset
        self.concepts = defaultdict(lambda:defaultdict(lambda:[]))
        self.cognate_sets = defaultdict(lambda:defaultdict(lambda:set()))
        self.clustered_cognates = defaultdict(lambda:{})
        self.load_data(self.filepath)
        self.load_gold_cognate_sets()
        self.mutual_coverage = self.calculate_mutual_coverage()

        
    def load_data(self, filepath, doculects=None, sep='\t'):
        
        # Load data file
        data = csv2dict(filepath, sep=sep)
        self.data = data
        
        # Initialize languages
        language_vocab_data = defaultdict(lambda:defaultdict(lambda:{}))
        for i in data:
            lang = data[i][self.columns['language_name']]
            if ((doculects is None) or (lang in doculects)):
                features = list(data[i].keys())
                for feature in features:
                    value = data[i][feature]
                    language_vocab_data[lang][i][feature] = value
                self.glottocodes[lang] = data[i][self.columns['glottocode']]
                self.iso_codes[lang] = data[i][self.columns['iso_code']]
                self.lang_ids[lang] = data[i][self.columns['id']].split('_')[0]
        
        language_list = sorted(list(language_vocab_data.keys()))
        for lang in language_list:
            self.languages[lang] = Language(name = lang, 
                                            lang_id=self.lang_ids[lang],
                                            glottocode=self.glottocodes[lang],
                                            iso_code=self.iso_codes[lang],
                                            family=self,
                                            data = language_vocab_data[lang],
                                            columns = self.columns,
                                            combine_diphthongs=self.combine_diphthongs
                                            )
            for concept in self.languages[lang].vocabulary:
                self.concepts[concept][lang].extend(self.languages[lang].vocabulary[concept])
        
    
    def load_gold_cognate_sets(self):
        """Creates dictionary sorted by cognate sets"""
        for lang in self.languages:
            lang = self.languages[lang]
            for concept in lang.vocabulary:
                for word in lang.vocabulary[concept]:
                    cognate_class = word.cognate_class
                    self.cognate_sets[cognate_class][lang].add(word)
    

    def write_vocab_index(self, output_file=None,
                          concept_list=None,
                          sep='\t', variants_sep='~'):
        """Write cognate set index to .csv file"""
        assert sep != variants_sep
        if output_file is None:
            output_file = f'{self.directory}{self.name} Vocabulary Index.csv'
        
        if concept_list is None:
            concept_list = sorted(list(self.cognate_sets.keys()))
        else:
            concept_list = sorted([c for c in concept_list if c in self.concepts])
            concept_list = sorted([c for c in self.cognate_sets.keys() 
                                   if c.split('_')[0] in concept_list])
        
        with open(output_file, 'w') as f:
            language_names = sorted([self.languages[lang].name for lang in self.languages])
            header = '\t'.join(['Gloss'] + language_names)
            f.write(f'{header}\n')
            
            for cognate_set_id in concept_list:
                forms = [cognate_set_id]
                for lang in language_names:
                    breakpoint()
                    raise NotImplementedError('Needs to be updated using new Word class and loan attribute (loans marked in parentheses)') # TODO
                    lang_forms = self.cognate_sets[cognate_set_id].get(lang, [''])
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
        concept_counts = {concept:len([lang for lang in self.languages 
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
    
    
    def calculate_phoneme_pmi(self, save=True, **kwargs):
        """Calculates phoneme PMI for all language pairs in the dataset and saves
        the results to file"""
        
        l = list(self.languages.values())
        
        # Check whether phoneme PMI has been calculated already for this pair
        # If not, calculate it now
        checked = []
        printed = []
        for pair in product(l, l):
            lang1, lang2 = pair
            if lang1.name not in printed:
                self.logger.info(f'Calculating phoneme PMI for {lang1.name}...') # TODO move this logging message into PhonemeCorrDetector
                printed.append(lang1.name)
            if (lang2, lang1) not in checked:
                    
                if len(lang1.phoneme_pmi[lang2]) == 0:
                    correlator = lang1.get_phoneme_correlator(lang2)
                    correlator.calc_phoneme_pmi(**kwargs)
        
        if save:
            self.write_phoneme_pmi()


    def write_phoneme_pmi(self, output_file=None):
        # Specify output file name if none is specified
        if output_file is None:
            output_file = os.path.join(self.phone_corr_dir, f'{self.name}_phoneme_PMI.csv')

        l = list(self.languages.values())
        # Save calculated PMI values to file
        with open(output_file, 'w') as f:
            f.write('Language1,Phone1,Language2,Phone2,PMI\n')
            checked = []
            for pair in product(l, l):
                lang1, lang2 = pair
                if (lang2, lang1) not in checked:
                
                    # Retrieve the precalculated values
                    pmi = lang1.phoneme_pmi[lang2]
                    if len(pmi) == 0:
                        self.logger.warning(f'Phoneme PMI has not been calculated for pair: {lang1.name} - {lang2.name}.')
                        continue

                    # Save all segment pairs with non-zero PMI values to file
                    # Also skip extremely small decimals that are close to zero
                    for seg1 in pmi:
                        for seg2 in pmi[seg1]:
                            if abs(pmi[seg1][seg2]) > lang1.phonemes[seg1] * lang2.phonemes[seg2]:
                                f.write(f'{lang1.name},{seg1},{lang2.name},{seg2},{pmi[seg1][seg2]}\n')
                    checked.append((lang1, lang2))


    def load_phoneme_pmi(self, pmi_file=None, excepted=[], **kwargs):
        """Loads pre-calculated phoneme PMI values from file"""
        
        # Designate the default file name to search for if no alternative is provided
        if pmi_file is None:
            pmi_file = os.path.join(self.phone_corr_dir, f'{self.name}_phoneme_PMI.csv')
        
        # Try to load the file of saved PMI values
        # If the file is not found, recalculate the PMI values and save to 
        # a file with the specified name
        if os.path.exists(pmi_file):
            pmi_data = pd.read_csv(pmi_file)
            
        else:
            self.calculate_phoneme_pmi(**kwargs)
            self.write_phoneme_pmi(output_file=pmi_file)
            pmi_data = pd.read_csv(pmi_file)
        
        # Iterate through the dataframe and save the PMI values to the Language
        # class objects' phoneme_pmi attribute
        for index, row in pmi_data.iterrows():
            try:
                lang1 = self.languages[row['Language1']]
                lang2 = self.languages[row['Language2']]
                if (lang1.name not in excepted) and (lang2.name not in excepted):
                    phone1, phone2 = row['Phone1'], row['Phone2']
                    pmi_value = row['PMI']
                    lang1.phoneme_pmi[lang2][phone1][phone2] = pmi_value
                    lang2.phoneme_pmi[lang1][phone2][phone1] = pmi_value
            
            # Skip loaded PMI values for languages which are not in dataset
            except KeyError:
                pass
    

    def calculate_phoneme_surprisal(self, ngram_size=1, save=True, **kwargs):
        """Calculates phoneme surprisal for all language pairs in the dataset and saves
        the results to file"""
        
        # First ensure that phoneme PMI has been calculated and loaded
        self.load_phoneme_pmi()
                
        # Check whether phoneme surprisal has been calculated already for this pair
        for lang1 in self.languages.values():
            self.logger.info(f'Calculating phoneme surprisal for {lang1.name}...')
            for lang2 in self.languages.values():
                    
                # If not, calculate it now
                if len(lang1.phoneme_surprisal[(lang2, ngram_size)]) == 0:
                    correlator = lang1.get_phoneme_correlator(lang2)
                    correlator.calc_phoneme_surprisal(ngram_size=ngram_size, **kwargs)
                
        if save:
            self.write_phoneme_surprisal(ngram_size=ngram_size)


    def write_phoneme_surprisal(self, ngram_size=1):
        # Specify output file names
        outfile = os.path.join(self.phone_corr_dir, f'phoneme_surprisal_{ngram_size}gram.csv')
        outfile_phon_env = os.path.join(self.phone_corr_dir, 'phoneme_env_surprisal.csv')
        
        # Save calculated surprisal values to file
        with open(outfile, 'w') as f:
            f.write('Language1,Phone1,Language2,Phone2,Surprisal,OOV_Smoothed\n')
            for lang1 in self.languages.values():
                for lang2 in self.languages.values():
                        
                    phoneme_surprisal = lang1.phoneme_surprisal[(lang2, ngram_size)]
                    if len(phoneme_surprisal) == 0:
                        self.logger.warning(f'Phoneme surprisal has not been calculated for pair: {lang1.name} - {lang2.name}.')
                        continue
                        
                    # Save values
                    for seg1 in phoneme_surprisal:
                        
                        # Determine the smoothed value for unseen ("out of vocabulary") correspondences
                        # Check using a non-IPA character
                        non_IPA = '?'
                        oov_smoothed = phoneme_surprisal[seg1][non_IPA]
                        
                        # Then remove this character from the surprisal dictionary
                        del phoneme_surprisal[seg1][non_IPA]
                        
                        # Save values which are not equal to (less than) the OOV smoothed value
                        for seg2 in phoneme_surprisal[seg1]:
                            if phoneme_surprisal[seg1][seg2] < oov_smoothed:
                                f.write(f'{lang1.name},{" ".join(seg1)},{lang2.name},{seg2},{phoneme_surprisal[seg1][seg2]},{oov_smoothed}\n')

        # Save calculated phonological environment surprisal values to file
        with open(outfile_phon_env, 'w') as f:
            f.write('Language1,Phone1,Language2,Phone2,Surprisal,OOV_Smoothed\n')
            for lang1 in self.languages.values():
                for lang2 in self.languages.values():
                        
                    phoneme_surprisal = lang1.phon_env_surprisal[(lang2, ngram_size)]
                    if len(phoneme_surprisal) == 0:
                        self.logger.warning(f'Phonological environment surprisal has not been calculated for pair: {lang1.name} - {lang2.name}.')
                        continue
                        
                    # Save values
                    for seg1 in phoneme_surprisal:
                        
                        # Determine the smoothed value for unseen ("out of vocabulary") correspondences
                        # Check using a non-IPA character
                        non_IPA = '?'
                        oov_smoothed = phoneme_surprisal[seg1][non_IPA]
                        
                        # Then remove this character from the surprisal dictionary
                        del phoneme_surprisal[seg1][non_IPA]
                        
                        # Save values which are not equal to the OOV smoothed value
                        for seg2 in phoneme_surprisal[seg1]:
                            if phoneme_surprisal[seg1][seg2] < oov_smoothed:
                                f.write(f'{lang1.name},{" ".join(seg1[0])},{lang2.name},{seg2},{phoneme_surprisal[seg1][seg2]},{oov_smoothed}\n')

        # Write a report on the most likely phoneme correspondences per language pair (TODO : create a cross-linguistic chart automatically)
        self.write_phoneme_corr_report(ngram_size=ngram_size, n=2)
        # TODO doesn't yet include phon_env 


    def write_phoneme_corr_report(self, langs=None, ngram_size=1, n=2):
        if langs is None:
            langs = self.languages.values()

        with open(os.path.join(self.phone_corr_dir, f'phoneme_correspondences_{ngram_size}gram.tsv'), 'w') as f:
            header = '\t'.join(['l1', 'phone_l1', 'l2', 'phone_l2', 'surprisal'])
            f.write(f'{header}\n')
            for lang1, lang2 in product(langs, langs):
                if lang1 != lang2:
                    threshold = surprisal(1/len(lang2.phonemes))
                    l1_phons = sorted([p for p in lang1.phoneme_surprisal[(lang2, ngram_size)].keys() if p != '-'])
                    for p1 in l1_phons:
                        p2_candidates = lang1.phoneme_surprisal[(lang2, ngram_size)][p1]
                        if len(p2_candidates) > 0:
                            p2_candidates = dict_tuplelist(p2_candidates)[-n:]
                            p2_candidates.reverse()
                            for p2, sur in p2_candidates:
                                if sur >= threshold:
                                    break
                                line = '\t'.join([lang1.name, str(p1), lang2.name, str(p2), str(round(sur, 3))])
                                f.write(f'{line}\n')
                        
    
    def load_phoneme_surprisal(self, ngram_size=1, surprisal_file=None, excepted=[], **kwargs):
        """Loads pre-calculated phoneme surprisal values from file"""
        
        # Designate the default file name to search for if no alternative is provided
        if surprisal_file is None:
            surprisal_file = os.path.join(self.phone_corr_dir, f'phoneme_surprisal_{ngram_size}gram.csv')
            surprisal_file_phon_env = os.path.join(self.phone_corr_dir, 'phoneme_env_surprisal.csv')
        
        # Try to load the file of saved surprisal values
        # If the file is not found, recalculate the surprisal values and save to 
        # a file with the specified name
        if os.path.exists(surprisal_file):
            surprisal_data = pd.read_csv(surprisal_file)
            
        else:
            self.calculate_phoneme_surprisal(ngram_size=ngram_size, **kwargs)
            surprisal_data = pd.read_csv(surprisal_file)
        
        def extract_surprisal_from_df(surprisal_data, phon_env=False):
            # Iterate through the dataframe and save the surprisal values to the Language
            # class objects' phoneme_surprisal attribute
            surprisal_dict = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0))))
            oov_vals = defaultdict(lambda:{})
            for index, row in surprisal_data.iterrows():
                try:
                    lang1 = self.languages[row['Language1']]
                    lang2 = self.languages[row['Language2']]
                    if (lang1.name not in excepted) and (lang2.name not in excepted):
                        phone1, phone2 = row['Phone1'], row['Phone2']
                        phone1 = tuple(phone1.split())
                        if phon_env:
                            phone1 = (phone1,) # phon_env needs to be saved as ((phone, env),)
                        surprisal_value = row['Surprisal']
                        surprisal_dict[lang1][(lang2, ngram_size)][phone1][phone2] = surprisal_value
                        if phone1 not in oov_vals[(lang1, lang2)]:
                            oov_smoothed = row['OOV_Smoothed']
                            oov_vals[(lang1, lang2)][phone1] = oov_smoothed
                
                # Skip loaded surprisal values for languages which are not in dataset
                except KeyError:
                    pass
            
            # Iterate back through language pairs and phone1 combinations and set OOV values
            for lang1, lang2 in oov_vals:
                for phone1 in oov_vals[(lang1, lang2)]:
                    oov_val = oov_vals[(lang1, lang2)][phone1]
                    surprisal_dict[lang1][(lang2, ngram_size)][phone1] = default_dict(surprisal_dict[lang1][(lang2, ngram_size)][phone1],
                                                                                        l=oov_val)
            
            return surprisal_dict
        
        # Extract and save the surprisal values to phoneme_surprisal attribute of language object
        loaded_surprisal = extract_surprisal_from_df(surprisal_data, phon_env=False)
        for lang in self.languages.values():
            lang.phoneme_surprisal = loaded_surprisal[lang]
        
        # Do the same for phonological environment surprisal
        if os.path.exists(surprisal_file):
            phon_env_surprisal_data = pd.read_csv(surprisal_file_phon_env)
            loaded_phon_env_surprisal = extract_surprisal_from_df(phon_env_surprisal_data, phon_env=True)
            for lang in self.languages.values():
                lang.phon_env_surprisal = loaded_phon_env_surprisal[lang]

        else:
            self.logger.error(f'No saved phonological environment surprisal file found for {self.name}!')
            raise FileNotFoundError

    
    def phonetic_diversity(self, ch_to_remove=[]): # TODO
        # diversity_scores = {}
        diversity_scores = defaultdict(lambda:[])
        for cognate_set in self.cognate_sets:
            concept = cognate_set.split('_')[0]
            forms = []
            for lang1 in self.cognate_sets[cognate_set]:
                forms.extend([strip_ch(w, ch_to_remove.union({'(', ')'})) for w in self.cognate_sets[cognate_set][lang1]])
            lf = len(forms)
            if lf > 1:
                # diversity_scores[cognate_set] = len(set(forms)) / lf
                diversity_scores[concept].append(len(set(forms)) / lf)
        
        for concept in diversity_scores:
            diversity_scores[concept] = mean(diversity_scores[concept])
        
        return mean(diversity_scores.values())
                    
        
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
    
        breakpoint()
        raise NotImplementedError # TODO needs to be updated to handle Word objects and output a dictionary in the same format as for gold and none methods in self.distance_matrix()
        # TODO make option for instead using k-means clustering given a known/desired number of clusters, as a mutually exclusive parameter with cutoff

        self.logger.info('Clustering cognates...')
        self.logger.info(f'Cluster function: {dist_func.name}')
        self.logger.info(f'Cluster threshold: {dist_func.cluster_threshold}')

        concept_list = [concept for concept in concept_list 
                        if len(self.concepts[concept]) > 1]
        self.logger.info(f'Total concepts: {len(concept_list)}')
        
        clustered_cognates = {}
        for concept in sorted(concept_list):
            # self.logger.info(f'Clustering words for "{concept}"...')
            words = [word.ipa for lang in self.concepts[concept] 
                     for word in self.concepts[concept][lang]]
            lang_labels = [lang for lang in self.concepts[concept] 
                           for entry in self.concepts[concept][lang]]
            labels = [f'{lang_labels[i]} /{words[i]}/' for i in range(len(words))]
            
            # Create tuple input of (word, lang)
            langs = [self.languages[lang] for lang in lang_labels]
            words = list(zip(words, langs))
    
            
            clusters = cluster_items(group=words,
                                     labels=labels,
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


    def write_cognate_index(self, clustered_cognates, output_file,
                        sep='\t', variants_sep='~'):
        assert sep != variants_sep
        
        cognate_index = defaultdict(lambda:defaultdict(lambda:[]))
        languages = []
        for concept in clustered_cognates:
            for i in clustered_cognates[concept]:
                cognate_id = f'{concept}_{i}'
                for entry in clustered_cognates[concept][i]:
                    lang, word = entry[:-1].split(' /')
                    cognate_index[cognate_id][lang].append(word)
                    languages.append(lang)
        languages = sorted(list(set(languages)))
        
        with open(output_file, 'w') as f:
            header = '\t'.join(['']+languages)
            f.write(header)
            f.write('\n')
            for cognate_id in cognate_index:
                line = [cognate_id]
                for lang in languages:
                    entry = '~'.join(cognate_index[cognate_id][lang])
                    line.append(entry)
                line = sep.join(line)
                f.write(f'{line}\n')


    def load_cognate_index(self, index_file, sep='\t', variants_sep='~'):
        assert sep != variants_sep
        index = defaultdict(lambda:defaultdict(lambda:[]))
        with open(index_file, 'r') as f:
            f = f.readlines()
            doculects = [name.strip() for name in f[0].split(sep)[1:]]
            for line in f[1:]:
                line = line.split(sep)
                cognate_id = line[0].rsplit('_', maxsplit=1)
                # print(cognate_id, line[0].rsplit('_', maxsplit=1))
                try:
                    gloss, cognate_class = cognate_id
                except ValueError:
                    gloss, cognate_class = cognate_id, '' # confirm that this is correct
                for lang, form in zip(doculects, line[1:]):
                    forms = form.split(variants_sep)
                    for form_i in forms:
                        form_i = form_i.strip()
                        
                        # Verify that all characters used in transcriptions are recognized
                        form_i = normalize_ipa_ch(form_i)
                        unk_ch = invalid_ch(form_i)
                        if len(unk_ch) > 0:
                            unk_ch_s = '< ' + ' '.join(unk_ch) + ' >'
                            raise ValueError(f'Error: Unable to parse characters {unk_ch_s} in {lang} /{form_i}/ "{gloss}"!')
                        if len(form_i.strip()) > 0:
                            index[gloss][cognate_class].append(f'{lang} /{form_i}/')   
        
        return index


    def load_clustered_cognates(self, **kwargs):
        cognate_files = glob.glob(f'{self.cognates_dir}/*.cog')
        for cognate_file in cognate_files:
            code = cognate_file.rsplit('.', maxsplit=1)[0]
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
        ch_to_remove = self.ch_to_remove.union({'(', ')'})
        for concept in clustered_cognates:
            clusters = {'/'.join([strip_diacritics(unidecode.unidecode(item.split('/')[0])), 
                                  strip_ch(item.split('/')[1], ch_to_remove)])+'/':set([i]) for i in clustered_cognates[concept] 
                        for item in clustered_cognates[concept][i]}
            
            gold_clusters = {f'{strip_diacritics(unidecode.unidecode(lang))} /{strip_ch(tr, ch_to_remove)}/':set([c]) 
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
                dem = sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
                try:
                    mcc = num/dem
                except ZeroDivisionError:
                    mcc = 0
                mcc_scores[concept] = mcc
                
            else:
                raise ValueError(f'Error: Method "{method}" not recognized for cluster evaluation!')
        if method == 'bcubed':
            return mean(precision_scores.values()), mean(recall_scores.values()), mean(f1_scores.values())
        elif method == 'mcc':
            return mean(mcc_scores.values())


    def generate_test_code(self, dist_func, cognates, exclude=['logger'], **kwargs): # TODO would it make more sense to create a separate class rather than the LexicalDataset for this?
        if type(dist_func) != Distance:
            self.logger.error(f'dist_func must be a Distance class object, found {type(dist_func)} instead.')
            raise TypeError
        name = dist_func.name if dist_func.name else dist_func.func.__name__
        code = f'cognates-{cognates}_distfunc-{name}'
        if cognates == 'auto':
            code += f'_cutoff-{dist_func.cluster_threshold}'
        for key, value in dist_func.hashable_kwargs:
            if key not in exclude:
                code += f'_{key}-{value}'
        for key, value in kwargs.items():
            if key not in kwargs:
                code += f'_{key}-{value}'
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
        
        if dist_func == Z_score_dist:
            cognates = 'none'
            raise NotImplementedError # TODO

        # Automatic cognate clustering        
        if cognates == 'auto':
            assert cluster_func is not None
            
            if code in self.clustered_cognates:
                clustered_concepts = self.clustered_cognates[code]
            else:
                clustered_concepts = self.cluster_cognates(concept_list, dist_func=cluster_func)
            breakpoint()
            raise NotImplementedError # TODO needs to be updated to handle Word objects and output a dictionary in the same format as for gold and none methods below

        # Use gold cognate classes
        elif cognates == 'gold':
            clustered_concepts = defaultdict(lambda:defaultdict(lambda:[]))
            for concept in concept_list:
                # TODO there may be a better way to isolate these cognate IDs
                cognate_ids = [cognate_id for cognate_id in self.cognate_sets if cognate_id.rsplit('_', maxsplit=1)[0] == concept]
                for cognate_id in cognate_ids:
                    for lang in self.cognate_sets[cognate_id]:
                        for word in self.cognate_sets[cognate_id][lang]:
                            clustered_concepts[concept][cognate_id].append(word)
        
        # No separation of cognates/non-cognates: 
        # all synonymous words are evaluated irrespective of cognacy
        # The concept itself is used as a dummy cognate class ID # TODO what happens when the dataset already includes cognate class anotation in the concept names?
        elif cognates == 'none':
            clustered_concepts = {concept: {concept: [word
                                            for lang in self.concepts[concept]
                                            for word in self.concepts[concept][lang]]}
                                            for concept in concept_list}
        
        # Raise error for unrecognized cognate clustering methods
        else:
            self.logger.error(f'Cognate clustering method "{cognates}" not recognized!')
            raise ValueError
        
        
        # Compute distance matrix over Language objects
        languages = [self.languages[lang] for lang in self.languages]
        dm = distance_matrix(group=languages,
                             dist_func=dist_func, 
                             sim=dist_func.sim,
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
                       method='ward', 
                       metric='euclidean',
                       **kwargs):
        
        # Ensure the linkage method is valid
        if method not in ('nj', 'average', 'complete', 'single', 'weighted', 'ward'):
            raise ValueError(f'Error: Unrecognized linkage type "{method}". Accepted values are: "average", "complete", "single", "weighted", "ward", "nj"')

        # Create distance matrix
        dm = self.distance_matrix(dist_func=dist_func,
                                  cluster_func=cluster_func,
                                  concept_list=concept_list,
                                  cognates=cognates,
                                  **kwargs
                                  )
        dists = squareform(dm)

        # Neighbor Joining linkage
        if method == 'nj':
            languages = self.languages.values()
            names = [lang.name for lang in languages]
            lang_names = [re.sub('\(', '{', l) for l in names]
            lang_names = [re.sub('\)', '}', l) for l in lang_names]
            nj_dm = DistanceMatrix(dists, ids=lang_names)
            return nj_dm
        
        # Other linkage methods
        else:
            lm = linkage(dists, method, metric)
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
            if len(ordered_labels) == df.shape[0]: # only if the dimensions match
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
                  method='ward', 
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
            outtree = os.path.join(self.tree_dir, f'{code}.tre')

        lm = self.linkage_matrix(dist_func,
                                 concept_list, 
                                 cluster_func,
                                 cognates, 
                                 method, 
                                 metric, 
                                 **kwargs)
        
        # Not possible to plot NJ trees in Python (yet? TBD) # TODO
        if method != 'nj':
            sns.set(font_scale=1.0)
            if len(group) >= 100:
                plt.figure(figsize=(20,20))
            elif len(group) >= 60:
                plt.figure(figsize=(10,10))
            else:
                plt.figure(figsize=(10,8))
            
            dendrogram(lm, p=p, orientation=orientation, labels=labels)
            if title:
                plt.title(title, fontsize=30)
            plt.savefig(f'{save_directory}{title}.png', bbox_inches='tight', dpi=300)
            plt.show()

        if return_newick or outtree:
            if method == 'nj':
                newick_tree = nj(lm, disallow_negative_branch_length=True, result_constructor=str)
            else:
                newick_tree = linkage2newick(lm, labels)

            # Fix formatting of Newick string
            newick_tree = re.sub('\s', '_', newick_tree)
            newick_tree = re.sub(',_', ',', newick_tree)

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
            n = max(10, round((len(group)/10)*2))
            plotsize = (n, n*y_ratio)
        plt.figure(figsize=plotsize)
        
        # Draw scatterplot points
        plt.scatter(
            coords[:, 0], coords[:, 1], marker = 'o'
            )
        
        # Add labels to points
        for label, x, y in zip(labels, coords[:, 0], coords[:, 1]):
            plt.annotate(
                label,
                xy = (x, y), xytext = (5, -5),
                textcoords = 'offset points', ha = 'left', va = 'bottom',
                )
        
        # Add lines connecting points with a distance under a certain threshold
        connected = []
        for i in range(len(coords)):
            for j in range(len(coords)):
                if (j, i) not in connected:
                    dist = dm[i][j]
                    if dist <= max_dist:
                        # if dist <= np.mean(dm[i]): # if the distance is lower than average
                        if dist in np.sort(dm[i])[1:round(top_connections*(len(dm)-1))]:
                            coords1, coords2 = coords[i], coords[j]
                            x1, y1 = coords1
                            x2, y2 = coords2
                            if alpha_func is None:
                                plt.plot([x1, x2],[y1, y2], alpha=1-dist)
                            else:
                                plt.plot([x1, x2],[y1, y2], alpha=alpha_func(dist))
                            connected.append((i,j))
        
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
        
        if dist_func == Z_score_dist:
            cognates = 'none'
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
            clustered_concepts = defaultdict(lambda:defaultdict(lambda:[]))
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
            clustered_concepts = {concept:{concept:[f'{lang} /{self.concepts[concept][lang][i][1]}/'
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
            language_list = [self.languages[l] for l in language_list]
        
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
        self.concepts = default_dict({concept:self.concepts[concept] 
                         for concept in self.concepts 
                         if len(self.concepts[concept]) > 0}, 
                                     l=defaultdict(lambda:[]))
        
        self.cognate_sets = default_dict({cognate_set:self.cognate_sets[cognate_set] 
                                          for cognate_set in self.cognate_sets 
                                          if len(self.cognate_sets[cognate_set]) > 0}, 
                                         l=defaultdict(lambda:[]))

    
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
            setattr(new_dataset,key,value)
        
        # Recalculate mutual coverage among remaining langauges
        new_dataset.mutual_coverage = new_dataset.calculate_mutual_coverage()
        
        return new_dataset


    def add_language(self, name, data_path, **kwargs):
        self.load_data(data_path, doculects=[name], **kwargs)


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
                 lang_id=None, 
                 glottocode=None, 
                 iso_code=None, 
                 family=None,
                 combine_diphthongs=True,
                 normalize_geminates=False,
                 preaspiration=True,
                 ignore_stress=False):
        
        # Language data
        self.name = name
        self.lang_id = lang_id
        self.glottocode = glottocode
        self.iso_code = iso_code
        self.family = family

        # Attributes for parsing data dictionary (TODO could this be inherited via a subclass?)
        self.data = data
        self.columns = columns
        
        # Phonemic inventory
        self.phonemes = defaultdict(lambda:0)
        self.vowels = defaultdict(lambda:0)
        self.consonants = defaultdict(lambda:0)
        self.tonemes = defaultdict(lambda:0)
        self.tonal = False
        
        # Phonological contexts
        self.unigrams = defaultdict(lambda:0)
        self.bigrams = defaultdict(lambda:0)
        self.trigrams = defaultdict(lambda:0)
        self.ngrams = defaultdict(lambda:defaultdict(lambda:0))
        self.gappy_trigrams = defaultdict(lambda:0)
        self.phon_environments = defaultdict(lambda:defaultdict(lambda:0))
        self.phon_env_ngrams = defaultdict(lambda:defaultdict(lambda:0))
        
        # Lexical inventory
        self.vocabulary = defaultdict(lambda:set())
        self.loanwords = defaultdict(lambda:set())
        
        # Transcription and segmentation parameters
        if self.family:
            self.ch_to_remove = self.family.ch_to_remove
        else:
            self.ch_to_remove = suprasegmental_diacritics.union({' ', '‿'})
            if ignore_stress:
                self.ch_to_remove = self.ch_to_remove - {'ˈ', 'ˌ'}
        self.segmentation_params = {
            'ch_to_remove':self.ch_to_remove,
            'normalize_geminates':normalize_geminates,
            'combine_diphthongs':combine_diphthongs,
            'preaspiration':preaspiration
        }
        
        self.create_vocabulary()
        self.create_phoneme_inventory()
        
        self.phoneme_entropy = entropy(self.phonemes)
        
        # Comparison with other languages
        self.phoneme_correlators = {}
        self.phoneme_correspondences = defaultdict(lambda:defaultdict(lambda:0))
        self.phoneme_pmi = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0)))
        self.phoneme_surprisal = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:-self.phoneme_entropy)))
        self.phon_env_surprisal = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:-self.phoneme_entropy)))
        self.detected_cognates = defaultdict(lambda:[]) # TODO is this used?
        self.detected_noncognates = defaultdict(lambda:[]) # TODO is this used?
        self.noncognate_thresholds = defaultdict(lambda:[]) # TODO is this used?


    def create_vocabulary(self):
        for i in self.data:
            entry = self.data[i]
            concept = entry[self.columns['concept']]
            loan = True if re.match(r'((TRUE)|1)$', entry[self.columns['loan']], re.IGNORECASE) else False
            cognate_class = entry[self.columns['cognate_class']]
            cognate_class = cognate_class if cognate_class.strip() != '' else concept
            word = Word(
                ipa_string = entry[self.columns['ipa']], 
                concept = concept, 
                orthography = entry[self.columns['orthography']], 
                ch_to_remove = self.ch_to_remove,
                normalize_geminates = self.segmentation_params['normalize_geminates'],
                combine_diphthongs = self.segmentation_params['combine_diphthongs'],
                preaspiration = self.segmentation_params['preaspiration'],
                language = self,
                cognate_class = cognate_class,
                loanword = loan,
                )            
            if len(word.segments) > 0:
                self.vocabulary[concept].add(word)
                
                # Mark known loanwords
                if loan:
                    self.loanwords[concept].add(word)
                    

    def create_phoneme_inventory(self):
        for concept in self.vocabulary:
            for word in self.vocabulary[concept]:
                segments = word.segments
                
                # Count phones
                for segment in segments:
                    self.phonemes[segment] += 1
                    self.unigrams[segment] += 1
                
                # Count phonological environments
                for seg, env in zip(segments, word.phon_env):
                    self.phon_environments[seg][env] += 1
                #for seg in self.phon_environments:
                #    self.phon_environments[seg] = normalize_dict(self.phon_environments[seg], default=True, lmbda=0) 
            
                # Count trigrams and gappy trigrams
                padded_segments = ['# ', '# '] + segments + ['# ', '# ']
                for j in range(1, len(padded_segments)-1):
                    trigram = (padded_segments[j-1], padded_segments[j], padded_segments[j+1])
                    self.trigrams[trigram] += 1
                    self.gappy_trigrams[('X', padded_segments[j], padded_segments[j+1])] += 1
                    self.gappy_trigrams[(padded_segments[j-1], 'X', padded_segments[j+1])] += 1
                    self.gappy_trigrams[(padded_segments[j-1], padded_segments[j], 'X')] += 1
                
                # Count bigrams
                padded_segments = padded_segments[1:-1]
                for j in range(1, len(padded_segments)):
                    bigram = (padded_segments[j-1], padded_segments[j])
                    self.bigrams[bigram] += 1
        self.ngrams[1] = self.unigrams
        self.ngrams[2] = self.bigrams
        self.ngrams[3] = self.trigrams
        
        # Normalize counts
        total_tokens = sum(self.phonemes.values())
        for phoneme in self.phonemes:
            self.phonemes[phoneme] = self.phonemes[phoneme] / total_tokens
        
        # Get dictionaries of vowels and consonants
        self.vowels = normalize_dict({v:self.phonemes[v] 
                                      for v in self.phonemes 
                                      if strip_diacritics(v)[0] in vowels}, 
                                     default=True, lmbda=0)
        
        self.consonants = normalize_dict({c:self.phonemes[c] 
                                         for c in self.phonemes 
                                         if strip_diacritics(c)[0] in consonants}, 
                                         default=True, lmbda=0)
        
        self.tonemes = normalize_dict({t:self.phonemes[t] 
                                       for t in self.phonemes 
                                       if strip_diacritics(t)[0] in tonemes}, 
                                      default=True, lmbda=0)
        
        # Designate language as tonal if it has tonemes
        if len(self.tonemes) > 0:
            self.tonal = True


    def list_ngrams(self, ngram_size, phon_env=False):
        """Returns a dictionary of ngrams of a particular size, with their counts"""

        # Retrieve pre-calculated ngrams
        if not phon_env and len(self.ngrams[ngram_size]) > 0:
            return self.ngrams[ngram_size]
        
        elif phon_env and len(self.phon_env_ngrams[ngram_size]) > 0:
            return self.phon_env_ngrams[ngram_size]
        
        else:
            for concept in self.vocabulary:
                for word in self.vocabulary[concept]:
                    segments = word.segments
                    if phon_env:
                        phon_env_segments = list(zip(segments, word.phon_env))
                    pad_n = ngram_size - 1
                    padded = ['# ']*pad_n + segments + ['# ']*pad_n
                    if phon_env:
                        padded_phon_env = ['# ']*pad_n + phon_env_segments + ['# ']*pad_n
                    for i in range(len(padded)-pad_n):
                        ngram = tuple(padded[i:i+ngram_size])
                        self.ngrams[ngram_size][ngram] += 1
                        if phon_env:
                            phon_env_ngram = tuple(padded_phon_env[i:i+ngram_size])
                            self.phon_env_ngrams[ngram_size][phon_env_ngram] += 1

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
            return matches
        
        else:
            for match in matches:
                concept, orthography, transcription = match
                print(f"<{orthography}> /{transcription}/ '{concept}'")
    
    
    def calculate_infocontent(self, word):
        # Disambiguation by type of input
        if isinstance(word, Word): # Word object, no further modification needed
            pass
        elif isinstance(word, str) or isinstance(word, list):
            if isinstance(word, str):
                ipa_string = word
            else:
                ipa_string = ''.join(word)
            word = Word(
                ipa_string=ipa_string,
                ch_to_remove=self.ch_to_remove,
                normalize_geminates = self.segmentation_params['normalize_geminates'],
                combine_diphthongs = self.segmentation_params['combine_diphthongs'],
                preaspiration = self.segmentation_params['preaspiration'],
                language=self)
        else:
            raise TypeError

        # Return the pre-calculated information content of the word, if possible
        if word.info_content:
            return word.info_content
        
        # Otherwise calculate it from scratch
        # Pad the segmented word
        padded = ['# ', '# '] + word.segments + ['# ', '# ']
        info_content = {}
        for i in range(2, len(padded)-2):
            trigram_counts = 0
            trigram_counts += self.trigrams[(padded[i-2], padded[i-1], padded[i])]
            trigram_counts += self.trigrams[(padded[i-1], padded[i], padded[i+1])]
            trigram_counts += self.trigrams[(padded[i], padded[i+1], padded[i+2])]
            gappy_counts = 0
            gappy_counts += self.gappy_trigrams[(padded[i-2], padded[i-1], 'X')]
            gappy_counts += self.gappy_trigrams[(padded[i-1], 'X', padded[i+1])]
            gappy_counts += self.gappy_trigrams[('X', padded[i+1], padded[i+2])]
            # TODO : needs smoothing
            info_content[i-2] = (padded[i], -log(trigram_counts/gappy_counts, 2))
        
        return info_content
    
    
    def self_surprisal(self, word, normalize=False):
        info_content = self.calculate_infocontent(word)
        if normalize:
            return mean(info_content[j][1] for j in info_content)
        else:
            #return sum(info_content[j][1] for j in info_content)
            return info_content
    
    
    def bigram_probability(self, bigram, delta=0.7): # TODO is this used for anything?
        """Returns Kneser-Ney smoothed conditional probability P(p2|p1)"""
        
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
        numerator = max((self.bigrams.get(bigram, 0)-delta), 0)
        
        return (numerator/total_start_p1_counts) + (l_KN*pKN_p1)
                
    
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
                               sim=True, # phone_sim
                               similarity=similarity, 
                               method=method, 
                               title=title, 
                               save_directory=save_directory, 
                               **kwargs)
    
    
    def get_phoneme_correlator(self, lang2, wordlist=None, seed=1):
        key = (lang2, wordlist, seed)
        if key not in self.phoneme_correlators:
            self.phoneme_correlators[key] = PhonemeCorrDetector(lang1=self,
                                                                lang2=lang2,
                                                                wordlist=wordlist, 
                                                                seed=seed)

        return self.phoneme_correlators[key]
    
    
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
            s+= f'\n\t"{concept.upper()}": /{word.ipa}/ <{word.orthography}>'

        return s


class Word:
    def __init__(self, 
                 ipa_string, 
                 concept=None, 
                 orthography=None, 
                 language=None,
                 cognate_class=None,
                 loanword=False,
                 # Parameters for preprocessing and segmentation
                 ch_to_remove=[], 
                 normalize_geminates=False,
                 combine_diphthongs=True,
                 preaspiration=True
                 ):
        self.language = language
        self.parameters = {
            'ch_to_remove':ch_to_remove,
            'normalize_geminates':normalize_geminates,
            'combine_diphthongs':combine_diphthongs,
            'preaspiration':preaspiration
        }
        self.ipa = self.preprocess(ipa_string, normalize_geminates=normalize_geminates)
        self.concept = concept
        self.cognate_class = cognate_class
        self.loanword = loanword
        self.orthography = orthography
        self.segments = self.segment(ch_to_remove, 
                                     combine_diphthongs=combine_diphthongs, 
                                     preaspiration=preaspiration)
        self.phon_env = self.getPhonEnv()
        self.info_content = None


    def preprocess(self, ipa_string, normalize_geminates=False):
        # Normalize common IPA character mistakes
        # Normalize affricates to special ligature characters, where available
        ipa_string = normalize_ipa_ch(ipa_string)

        # Normalize geminate consonants to /Cː/
        if normalize_geminates:
            ipa_string = re.sub(fr'([{consonants}])\1', r'\1ː', ipa_string)

        return ipa_string


    def segment(self, ch_to_remove, combine_diphthongs, preaspiration):
        return segment_ipa(
            self.ipa, 
            # Remove stress and tone diacritics from segmented words; syllabic diacritics (above and below); spaces and <‿> linking tie
            remove_ch=''.join(ch_to_remove), 
            combine_diphthongs=combine_diphthongs,
            preaspiration=preaspiration
        )
    

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
        # TODO add this
        pass


# COMBINING DATASETS
def combine_datasets(dataset_list):
    # TODO
    pass


def load_family(family, data_file, min_amc=None, concept_list=None, exclude=None, logger=None, **kwargs):
    family = LexicalDataset(data_file, family, logger=logger, **kwargs)
    if exclude:
        family.remove_languages(exclude)
    if min_amc:
        family.prune_languages(min_amc=float(min_amc), concept_list=concept_list)
    # families[family].write_vocab_index() # TODO
    language_variables = {format_as_variable(lang):family.languages[lang] 
                        for lang in family.languages}
    globals().update(language_variables)
    return family
