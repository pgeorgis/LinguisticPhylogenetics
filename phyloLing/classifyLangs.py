import argparse
import json
import logging
import os
import shutil
from collections import defaultdict

import yaml
from constants import SPECIAL_JOIN_CHS, TRANSCRIPTION_PARAM_DEFAULTS
from lingDist import binary_cognate_sim, gradient_cognate_dist
from utils.tree import (calculate_tree_distance, gqd, load_newick_tree,
                        plot_tree)
from utils.utils import (calculate_time_interval, convert_sets_to_lists,
                         create_datestamp, create_timestamp, csv2dict,
                         get_git_commit_hash)
from wordDist import (HYBRID_DIST_KEY, LEVENSHTEIN_DIST_KEY,
                      PHONOLOGICAL_DIST_KEY, PMI_DIST_KEY, SURPRISAL_DIST_KEY,
                      LevenshteinDist, PhonDist, PMIDist, SurprisalDist,
                      WordDistance, hybrid_dist)

from phyloLing import load_family

# Loglevel mapping
log_levels = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
}

# Valid parameter values for certain parameters
valid_params = {
    'cognates': {
        'cluster': {'auto', 'gold', 'none'},
        'cluster_method': {'phon', 'pmi', 'surprisal', 'levenshtein', 'hybrid'},
    },
    'evaluation': {
        'similarity': {'gradient', 'binary'},
        'method': {'phon', 'pmi', 'surprisal', 'levenshtein', 'hybrid'},
    },
    'tree': {
        'linkage': {'nj', 'average', 'complete', 'ward', 'weighted', 'single'},
    }
}

# Mapping of distance method labels to WordDistance objects
function_map = {
    PMI_DIST_KEY: PMIDist,
    SURPRISAL_DIST_KEY: SurprisalDist,
    PHONOLOGICAL_DIST_KEY: PhonDist,
    LEVENSHTEIN_DIST_KEY: LevenshteinDist
}
aux_func_map = {
    'pmi': PMI_DIST_KEY,
    'surprisal': SURPRISAL_DIST_KEY,
    'levenshtein': LEVENSHTEIN_DIST_KEY,
    'hybrid': HYBRID_DIST_KEY,
} # TODO unify these with function_map


def load_config(config_path):
    """Returns a dictionary containing parameters from a specified config.yml file

    Args:
        config_path (str): Path to config.yml file

    Returns:
        config: nested dictionary of parameter names and values
    """

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def validate_params(params, valid_params, logger):
    for section_name in valid_params:
        for param_name in valid_params[section_name]:
            param_value = params[section_name][param_name]
            if param_value not in valid_params[section_name][param_name]:
                logger.error(f'Invalid parameter value "{param_value}" for parameter "{param_name}". Valid options: {", ".join(valid_params[section_name][param_name])}')
                raise ValueError

    # Ensure that minimally the input file path is specified
    if 'file' not in params['family']:
        logger.error('Input file (`file`) argument must be specified!')
        raise ValueError
    params['family']['file'] = os.path.abspath(params['family']['file'])
    logger.debug(f"Data source: {params['family']['file']}")

    # Designate outdir as the input file directory if unspecified
    if params['family']['outdir'] is None:
        outdir = os.path.dirname(os.path.abspath(params['family']['file']))
        params['family']['outdir'] = outdir
    else:
        outdir = os.path.abspath(params['family']['outdir'])
    logger.debug(f'Experiment outdir: {outdir}')

    # Designate global transcription parameter defaults
    for transcription_param in TRANSCRIPTION_PARAM_DEFAULTS:
        if transcription_param not in params['transcription']['global']:
            params['transcription']['global'][transcription_param] = TRANSCRIPTION_PARAM_DEFAULTS[transcription_param]
    params['transcription']['global']['ch_to_remove'] = set(params['transcription']['global']['ch_to_remove'])

    # If transcription parameters are specified for individual doculects, ensure all are included
    # (copy from global defaults if unspecified)
    if 'doculects' in params['transcription']:

        for doculect in params['transcription']['doculects']:
            params['transcription']['doculects'][doculect] = {
                transcription_param: params['transcription']['doculects'][doculect].get(transcription_param, params['transcription']['global'][transcription_param])
                for transcription_param in TRANSCRIPTION_PARAM_DEFAULTS
            }

    # Raise error if binary cognate similarity is used with "none" cognate clustering
    if params['cluster']['cognates'] == 'none' and params['evaluation']['similarity'] == 'binary':
        logger.error('Binary cognate similarity cannot use "none" cognate clustering. Valid options are: [`auto`, `gold`]')
        raise ValueError

    # Ensure gap character and pad character are different
    if params['alignment']['gap_ch'] == params['alignment']['pad_ch']:
        raise ValueError(f"Gap character and pad character must be different! Both are set to '{params['alignment']['gap_ch']}'")
    # Ensure pad character and gap character are not any of the special joinining/delimiting characters
    invalid_special_ch_str = ','.join([f'"{ch}"' for ch in SPECIAL_JOIN_CHS])
    if params['alignment']['pad_ch'] in SPECIAL_JOIN_CHS:
        raise ValueError(f"Invalid pad character. Pad character may not be {invalid_special_ch_str}.")

    if params['alignment']['gap_ch'] in SPECIAL_JOIN_CHS:
        raise ValueError(f"Invalid gap character. Gap character may not be {invalid_special_ch_str}.")

    # Add "run_info" and "output" sections
    params["run_info"] = {}
    params["output"] = {}


def init_hybrid(function_map, eval_params):
    HybridDist = WordDistance(
        func=hybrid_dist,
        name='HybridDist',
        funcs=[
            function_map[PMI_DIST_KEY],
            function_map[SURPRISAL_DIST_KEY],
            function_map[PHONOLOGICAL_DIST_KEY],
        ],
        weights=(
            eval_params['pmi_weight'],
            eval_params['surprisal_weight'],
            eval_params['phon_weight'],
        ),
        normalize_weights=eval_params['normalize_weights']
    )

    return HybridDist


def load_precalculated_word_scores(distance_dir, family, dist_keys, excluded_doculects):
    doculect_pairs = family.get_doculect_pairs(bidirectional=True)
    precalculated_word_scores = defaultdict(lambda:{})
    n_files_found = 0
    for lang1, lang2 in doculect_pairs:
        if lang1.name in excluded_doculects or lang2.name in excluded_doculects:
            continue
        scored_words_file = os.path.join(
            distance_dir,
            lang1.path_name,
            lang2.path_name,
            "lexical_comparison.tsv"
        )
        if os.path.exists(scored_words_file):
            n_files_found += 1
            scored_words_data = csv2dict(scored_words_file, sep="\t")
            for _, entry in scored_words_data.items():
                lang1_ipa = entry[lang1.name]
                lang2_ipa = entry[lang2.name]
                for dist_key in dist_keys:
                    if dist_key in entry:
                        score = float(entry[dist_key])
                        score_key = ((lang1.name, lang1_ipa), (lang2.name, lang2_ipa), function_map[dist_key].hashable_kwargs)
                        function_map[dist_key].measured[score_key] = score
                        precalculated_word_scores[dist_key][(lang1.name, lang1_ipa, lang2.name, lang2_ipa)] = score
    logger.info(f"Loaded pre-calculated word scores for {n_files_found} doculect pairs from {distance_dir}")
    return precalculated_word_scores


def write_lang_dists_to_tsv(dist, outfile):
    # TODO add description
    with open(outfile, 'w') as f:
        header = '\t'.join(['Language1', 'Language2', 'Measurement'])
        f.write(f'{header}\n')
        for key, value in dist.measured.items():
            lang1, lang2, kwargs = key
            line = '\t'.join([lang1.name, lang2.name, str(value)])
            f.write(f'{line}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Loads a lexical dataset in CLDF format and produces a phylogenetic tree according to user specifications')
    parser.add_argument('config', help='Path to config.yml file')
    parser.add_argument('--loglevel', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Log level for printed log messages')
    args = parser.parse_args()
    start_time, start_timestamp = create_timestamp()

    # Configure the logger
    logging.basicConfig(level=log_levels[args.loglevel], format='%(asctime)s classifyLangs %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    # Load parameters from config file
    params = load_config(args.config)
    # Load default parameters from default config file
    default_params = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/default_config.yml'))
    # Set default values in case unspecified in config file
    for section_name, section_params in default_params.items():
        if section_name not in params:
            params[section_name] = default_params[section_name]
            continue
        for param_name, param_value in section_params.items():
            if param_name not in params[section_name]:
                params[section_name][param_name] = default_params[section_name][param_name]
    # Validate parameters
    validate_params(params, valid_params, logger)

    # Add git commmit hash to run config
    params["run_info"]["version"] = get_git_commit_hash()

    # Log config param settings
    logger.info(json.dumps(convert_sets_to_lists(params), indent=4))

    # Get shorthand for each parameter section
    family_params = params['family']
    transcription_params = params['transcription']
    alignment_params = params['alignment']
    phon_corr_params = params['phon_corr']
    cognate_params = params['cognates']
    eval_params = params['evaluation']
    tree_params = params['tree']
    experiment_params = params['experiment']

    # Generate experiment ID and outdir
    exp_name = experiment_params['name']
    if exp_name is None:
        exp_id = start_timestamp
    else:
        exp_id = os.path.join(exp_name, start_timestamp)
    logger.info(f'Experiment ID: {exp_id}')
    exp_outdir = os.path.join(family_params["outdir"], "experiments", create_datestamp(), exp_id)
    os.makedirs(exp_outdir, exist_ok=True)
    params["run_info"]["experimentID"] = exp_id

    # Set ngram size used for surprisal
    surprisal_funcs = ('surprisal', 'hybrid')
    if eval_params['method'] in surprisal_funcs or cognate_params['cluster_method'] in phon_corr_params:
        function_map[SURPRISAL_DIST_KEY].set('ngram_size', phon_corr_params['ngram'])
        function_map[SURPRISAL_DIST_KEY].set('phon_env', phon_corr_params['phon_env'])
        SurprisalDist = function_map[SURPRISAL_DIST_KEY]

        # Initialize hybrid distance object
        if eval_params['method'] == 'hybrid' or cognate_params['cluster_method'] == 'hybrid':
            function_map[HYBRID_DIST_KEY] = init_hybrid(function_map, eval_params)

    # Designate cluster function if performing auto cognate clustering
    if cognate_params['cluster'] == 'auto':
        clusterDist = function_map[aux_func_map[cognate_params['cluster_method']]]
        # Set cluster threshold if specified
        if cognate_params['cluster_threshold']:
            clusterDist.cluster_threshold = cognate_params['cluster_threshold']
    else:
        clusterDist = None

    # Designate evaluation function
    evalDist = function_map[aux_func_map[eval_params['method']]]

    # Load CLDF dataset
    if family_params['min_amc']:
        family_params['min_amc'] = float(family_params['min_amc'])
    # Set the warn threshold for instances of phone in a doculect to the maximum of the
    # threshold set in transcription parameters and the minimum correlation instance value
    # Ensures that if minimum correlation is set to a higher value,
    # warnings will be issued about any phones with fewer instances than this
    transcription_params["global"]["min_phone_instances"] = max(
        transcription_params["global"]["min_phone_instances"], phon_corr_params['min_corr']
    )
    family = load_family(family_params['name'],
                         family_params['file'],
                         outdir=family_params['outdir'],
                         excluded_doculects=family_params['exclude'],
                         included_doculects=family_params['include'],
                         min_amc=family_params['min_amc'],
                         transcription_params=transcription_params,
                         alignment_params=alignment_params,
                         logger=logger
                         )

    # Print some summary info about the loaded dataset
    logger.info(f'Loaded {len(family.languages)} doculects.')
    abs_mc, avg_mc = family.calculate_mutual_coverage()
    if avg_mc <= 0.7:
        logger.warning(f'Average mutual coverage = {round(avg_mc, 2)}. Recommend minimum is 0.7.')
    else:
        logger.info(f'Average mutual coverage is {round(avg_mc, 2)} ({abs_mc}/{len(family.concepts)} concepts in all {len(family.languages)} doculects).')

    # Load or calculate phone correspondences
    if not phon_corr_params['refresh_all']:
        logger.info(f'Loading {family.name} phoneme PMI...')
        family.load_phoneme_pmi(excepted=phon_corr_params['refresh'])

        if eval_params['method'] in ('surprisal', 'hybrid'):
            logger.info(f'Loading {family.name} phoneme surprisal...')
            if cognate_params['cluster'] == 'gold':
                family.load_phoneme_surprisal(
                    ngram_size=phon_corr_params['ngram'],
                    phon_env=phon_corr_params['phon_env'],
                    gold=True,
                    excepted=phon_corr_params['refresh'],
                )
            else:
                family.load_phoneme_surprisal(
                    ngram_size=phon_corr_params['ngram'],
                    phon_env=phon_corr_params['phon_env'],
                    gold=False,
                    excepted=phon_corr_params['refresh'],
                )

    # If phoneme PMI/surprisal was refreshed for one or more languages, rewrite the saved files
    # Needs to occur after PMI/surprisal was recalculated for the language(s) in question
    if phon_corr_params['refresh_all'] or len(phon_corr_params['refresh']) > 0:
        family.calculate_phone_corrs(
            sample_size=phon_corr_params['sample_size'],
            n_samples=phon_corr_params['n_samples'],
            min_corr=phon_corr_params['min_corr'],
            ngram_size=phon_corr_params['ngram'],
            phon_env=phon_corr_params['phon_env'],
        )
        family.write_phoneme_pmi()
        if eval_params['method'] in ('surprisal', 'hybrid'):
            family.write_phoneme_surprisal(
                ngram_size=phon_corr_params['ngram'],
                phon_env=phon_corr_params['phon_env'],
            )

    # Auto cognate clustering only
    if cognate_params['cluster'] in ('auto', 'gold'):
        # Load pre-clustered cognate sets, if specified
        cognate_index_file = cognate_params['cognate_index']
        if cognate_index_file:
            cognate_index = family.load_cognate_index(cognate_index_file, code=exp_id)
            logger.info(f"Loaded cognate index from {cognate_index_file}")

    # Load precalculated word scores from specified directory
    precalculated_word_scores = None
    if eval_params['precalculated_word_scores']:
        precalculated_word_scores = load_precalculated_word_scores(
            distance_dir=eval_params['precalculated_word_scores'],
            family=family,
            dist_keys=[PMI_DIST_KEY, SURPRISAL_DIST_KEY, PHONOLOGICAL_DIST_KEY],  # TODO maybe needs to be more customizable
            excluded_doculects=phon_corr_params['refresh'],
        )

    # Create cognate similarity (WordDistance object) measure according to settings
    if eval_params['similarity'] == 'gradient':
        dist_func = gradient_cognate_dist
        distFunc = WordDistance(
            func=dist_func,
            name='GradientCognateDist',
            eval_func=evalDist,
            n_samples=eval_params['n_samples'],
            sample_size=eval_params['sample_size'],
            exclude_synonyms=eval_params['exclude_synonyms'],
            calibrate=eval_params['calibrate'],
            min_similarity=eval_params['min_similarity'],
            logger=logger,
        )
    elif eval_params['similarity'] == 'binary':
        dist_func = binary_cognate_sim
        distFunc = WordDistance(
            func=dist_func,
            name='BinaryCognateSim',
            sim=True,
            # n_samples=eval_params['n_samples'],
            # sample_size=eval_params['sample_size'],
        )

    # Generate Newick tree string
    logger.info('Generating phylogenetic tree...')
    outtree = os.path.join(exp_outdir, "newick.tre")
    tree = family.generate_tree(
        cluster_func=clusterDist,
        dist_func=distFunc,
        cognates=cognate_params['cluster'],
        linkage_method=tree_params['linkage'],
        outtree=outtree,
        root=tree_params['root'],
        code=exp_id,
    )
    params["tree"]["newick"] = tree
    with open(outtree, 'w') as f:
        f.write(tree)
    logger.info(f'Wrote Newick tree to {os.path.abspath(outtree)}')
    
    # Write clustered cognate class index
    if cognate_params['cluster'] == 'auto':
        clustered_cognates = family.clustered_cognates[exp_id]
        family.write_cognate_index(clustered_cognates, os.path.join(exp_outdir, f'cognate_classes.tsv'))

    # Plot the phylogenetic tree
    out_png = os.path.abspath(os.path.join(exp_outdir, "tree.png"))
    plot_tree(os.path.abspath(outtree), out_png)
    logger.info(f'Plotted phylogenetic tree to {out_png}')

    # Optionally evaluate tree wrt to reference tree(s)
    if tree_params["reference"]:
        tree_scores = defaultdict(dict)
        for ref_tree_file in tree_params["reference"]:
            ref_tree = load_newick_tree(ref_tree_file)
            tree_scores[ref_tree_file]["newick"] = ref_tree.as_string("newick").strip()
            gqd_score = gqd(
                tree,
                ref_tree,
                is_rooted=tree_params['root'] is not None
            )
            tree_scores[ref_tree_file]["GQD"] = gqd_score
            logger.info(f"GQD wrt reference tree {ref_tree_file}: {round(gqd_score, 3)}")
            tree_mutual_info = calculate_tree_distance(tree, ref_tree)
            tree_scores[ref_tree_file]["TreeDist"] = tree_mutual_info
            logger.info(f"TreeDist wrt reference tree {ref_tree_file}: {round(tree_mutual_info, 3)}")
        params["tree"]["eval"] = tree_scores

    # Write distance matrix TSV
    out_distmatrix = os.path.join(exp_outdir, f'distance-matrix.tsv')
    write_lang_dists_to_tsv(distFunc, outfile=out_distmatrix)
    # Write lexical comparison files
    for lang1, lang2 in family.get_doculect_pairs(bidirectional=True):
        dist_outdir = os.path.join(exp_outdir, 'distances')
        lex_comp_log_dir = os.path.join(dist_outdir, lang1.path_name, lang2.path_name)
        os.makedirs(lex_comp_log_dir, exist_ok=True)
        lex_comp_log = os.path.join(lex_comp_log_dir, 'lexical_comparison.tsv')
        lang1.write_lexical_comparison(lang2, lex_comp_log)

    # Copy phone corr files to experiment outdir
    if phon_corr_params["copy_to_outdir"]:
        shutil.copytree(
            family.phone_corr_dir,
            os.path.join(exp_outdir, "phone_corr"),
            dirs_exist_ok=True
        )

    # Add outfiles and final run info to config and dump
    params["output"]["tree"] = outtree
    params["output"]["dist_matrix"] = out_distmatrix
    params["output"]["lexical_comparisons"] = dist_outdir
    end_time, end_timestamp = create_timestamp()
    params["run_info"]["start_time"] = start_timestamp
    params["run_info"]["end_time"] = end_timestamp
    params["run_info"]["duration"] = calculate_time_interval(start_time, end_time)
    config_copy = os.path.join(exp_outdir, "config.yml")
    with open(config_copy, 'w') as f:
        yaml.dump(convert_sets_to_lists(params), f)
    logger.info(f"Wrote experiment run config to {os.path.abspath(config_copy)}")

    logger.info('Completed successfully.')
