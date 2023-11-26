import argparse
import os
import logging
import yaml
from auxFuncs import Distance, create_timestamp
from phyloLing import load_family, TRANSCRIPTION_PARAM_DEFAULTS
from wordDist import PMIDist, SurprisalDist, PhonologicalDist, LevenshteinDist, hybrid_dist, cascade_sim
from lingDist import gradient_cognate_sim, binary_cognate_sim

# Loglevel mapping
log_levels = {
    'DEBUG':logging.DEBUG,
    'INFO':logging.INFO,
    'WARNING':logging.WARNING,
    'ERROR':logging.ERROR,
}

# Valid parameter values for certain parameters
valid_params = {
    'cluster':{
        'cognates':{'auto', 'gold', 'none'},
        'method':{'phon', 'pmi', 'surprisal', 'levenshtein', 'hybrid', 'cascade'},
    },
    'evaluation':{
        'similarity':{'gradient', 'binary'},
        'method':{'phon', 'pmi', 'surprisal', 'levenshtein', 'hybrid', 'cascade'},
    },
    'tree':{
        'linkage':{'nj', 'average', 'complete', 'ward', 'weighted', 'single'},
    }
}

# Mapping of distance method labels to Distance objects
function_map = {
    'pmi':PMIDist,
    'surprisal':SurprisalDist,
    'phon':PhonologicalDist, # TODO name doesn't match (I think phon is fine because it is neutral between phonological and phonetic, could rename Distance as PhonDist)
    'levenshtein':LevenshteinDist
    }


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
        # If transcription parameters are specified for individual doculects, ensure all are included 
        # (copy from global defaults if unspecified)
    if 'doculects' in params['transcription']: 
        for doculect in params['transcription']['doculects']:
            params['transcription']['doculects'][doculect] = {
                transcription_param:params['transcription']['doculects'][doculect].get(transcription_param, params['transcription']['global'][transcription_param])
                for transcription_param in TRANSCRIPTION_PARAM_DEFAULTS
            }
            
    # Raise error if binary cognate similarity is used with "none" cognate clustering
    if params['cluster']['cognates'] == 'none' and params['evaluation']['similarity'] == 'binary':
        logger.error('Binary cognate similarity cannot use "none" cognate clustering. Valid options are: [`auto`, `gold`]')
        raise ValueError
    
    # Ensure gap character and pad character are different
    if params['alignment']['gap_ch'] == params['alignment']['pad_ch']:
        raise ValueError(f"Gap character and pad character must be different! Both are set to '{params['alignment']['gap_ch']}'")
    # Ensure pad character is not < or >, which are used in combination with the pad character for indicating side of padding
    # Neither pad nor gap character can be _, which is used for joining ngrams
    if params['alignment']['pad_ch'] in ('<', '>', '_'):
        raise ValueError('Invalid pad character. Pad character may not be "<", ">", or "_".') 
    if params['alignment']['gap_ch'] in ('_'):
        raise ValueError('Invalid gap character. Gap character may not be "_".')

def init_hybrid(function_map, eval_params):
    HybridDist = Distance(
        func=hybrid_dist,
        name='HybridDist',
        funcs=[function_map['pmi'], function_map['surprisal'], function_map['phon']],
        weights=(
            eval_params['pmi_weight'],
            eval_params['surprisal_weight'],
            eval_params['phon_weight'],
        )
    )
    HybridSim = HybridDist.to_similarity(name='HybridSim') 
    
    return HybridSim


def init_cascade(params):
    eval_params = params['evaluation']
    surprisal_params = params['surprisal']
    CascadeSim = Distance(
        func=cascade_sim,
        name='CascadeSim',
        sim=True,
        pmi_weight=eval_params['pmi_weight'],
        surprisal_weight=eval_params['surprisal_weight'],
        ngram_size=surprisal_params['ngram'],
        phon_env=surprisal_params['phon_env'],
    )
    CascadeDist = CascadeSim.to_distance('CascadeDist', alpha=0.8)
    
    return CascadeDist


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
    
    # Get shorthand for each parameter section
    family_params = params['family']
    transcription_params = params['transcription']
    alignment_params = params['alignment']
    pmi_params = params['pmi']
    surprisal_params = params['surprisal']
    cluster_params = params['cluster']
    eval_params = params['evaluation']
    tree_params = params['tree']

    # Set ngram size used for surprisal
    if eval_params['method'] in ('surprisal', 'hybrid', 'cascade'):
        function_map['surprisal'].set('ngram_size', surprisal_params['ngram'])
        SurprisalDist = function_map['surprisal']

        # Initialize hybrid or cascade distance/similarity objects 
        if eval_params['method'] == 'hybrid':
            function_map['hybrid'] = init_hybrid(function_map, eval_params)
        elif eval_params['method'] == 'cascade':
            function_map['cascade'] = init_cascade(params)

    # Designate cluster function if performing auto cognate clustering 
    if cluster_params['cognates'] == 'auto':
        clusterDist = function_map[cluster_params['method']]
        # Set specified cluster threshold # TODO cluster_threshold needs to be recalibrated
        clusterDist.cluster_threshold = cluster_params['cluster_threshold']
    else:
        clusterDist = None
    
    # Designate evaluation function
    evalDist = function_map[eval_params['method']]
    
    # Load CLDF dataset
    if family_params['min_amc']:
        family_params['min_amc'] = float(family_params['min_amc'])
    family = load_family(family_params['name'], 
                         family_params['file'], 
                         outdir=family_params['outdir'],
                         exclude=family_params['exclude'], 
                         min_amc=family_params['min_amc'],
                         transcription_params=transcription_params,
                         alignment_params=alignment_params,
                         logger=logger
                         )

    # Print some summary info about the loaded dataset
    logger.info(f'Loaded {len(family.languages)} doculects.')
    abs_mc, avg_mc = family.calculate_mutual_coverage()
    if avg_mc <= 0.7: # TODO should this be hard-coded?
        logger.warning(f'Average mutual coverage = {round(avg_mc, 2)}. Recommend minimum is 0.7.')
    else:
        logger.info(f'Average mutual coverage is {round(avg_mc, 2)} ({abs_mc}/{len(family.concepts)} concepts in all {len(family.languages)} doculects).')

    # Load or calculate phoneme PMI
    if not pmi_params['refresh_all_pmi']:
        logger.info(f'Loading {family.name} phoneme PMI...')
        family.load_phoneme_pmi(excepted=pmi_params['refresh'])

    # Load or calculate phoneme surprisal
    if not surprisal_params['refresh_all_surprisal']:
        if eval_params['method'] in ('surprisal', 'hybrid', 'cascade'):
            logger.info(f'Loading {family.name} phoneme surprisal...')
            if cluster_params['cognates'] == 'gold':
                family.load_phoneme_surprisal(
                    ngram_size=surprisal_params['ngram'], 
                    gold=True, 
                    excepted=surprisal_params['refresh'])
            else:
                family.load_phoneme_surprisal(
                    ngram_size=surprisal_params['ngram'], 
                    gold=False, 
                    excepted=surprisal_params['refresh'])

    # If phoneme PMI/surprisal was refreshed for one or more languages, rewrite the saved files
    # Needs to occur after PMI/surprisal was recalculated for the language(s) in question
    if pmi_params['refresh_all_pmi'] or surprisal_params['refresh_all_surprisal'] or len(pmi_params['refresh']) or len(surprisal_params['refresh']) > 0:
        family.calculate_phoneme_pmi()
        family.write_phoneme_pmi()
        if eval_params['method'] in ('surprisal', 'hybrid', 'cascade'):
            family.calculate_phoneme_surprisal(ngram_size=surprisal_params['ngram'])
            family.write_phoneme_surprisal(ngram_size=surprisal_params['ngram'])

    # Auto cognate clustering only
    if cluster_params['cognates'] == 'auto':
        # Load pre-clustered cognate sets, if available
        family.load_clustered_cognates()

        # Set cognate cluster ID according to settings
        cog_id = f"{family.name}_distfunc-{cluster_params['method']}_cutoff-{cluster_params['cluster_threshold']}"
        # TODO cog_id should include weights and any other params for hybrid

    # Create cognate similarity (Distance object) measure according to settings
    if eval_params['similarity'] == 'gradient':
        dist_func = gradient_cognate_sim
        distFunc = Distance(
            func=dist_func, 
            name='GradientCognateSim',
            sim=True,
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
        distFunc = Distance(
            func=dist_func, 
            name='BinaryCognateSim',
            sim=True,
            #n_samples=eval_params['n_samples'], 
            #sample_size=eval_params['sample_size'],
            )
    
    # Generate test code 
    code = family.generate_test_code(distFunc, cognates=cluster_params['cognates'], cutoff=cluster_params['cluster_threshold'])
    if eval_params['similarity'] == 'gradient':
        code += family.generate_test_code(evalDist)
    logger.debug(f'Experiment ID: {code}')

    # Generate Newick tree string
    logger.info(f'Generating phylogenetic tree...')
    timestamp = create_timestamp()
    outtree = os.path.join(family_params['outdir'], 'trees', timestamp+'.tre') # TODO this could still be improved
    tree = family.draw_tree(
        cluster_func=clusterDist,
        dist_func=distFunc,
        cognates=cluster_params['cognates'], 
        method=tree_params['linkage'], # TODO this should be changed within draw_tree() to linkage rather than method
        title=family.name, 
        outtree=outtree,
        return_newick=tree_params['newick'])
    with open(outtree, 'w') as f:
        f.write(tree)
    logger.info(f'Wrote Newick tree to {outtree}')
    print(tree)
    
    # if log_scores:
    write_lang_dists_to_tsv(distFunc, outfile=os.path.join(family.dist_matrix_dir, f'{timestamp}_scored.tsv'))
    for lang1, lang2 in family.get_doculect_pairs(bidirectional=True):
        lex_comp_log_dir = os.path.join(family_params['outdir'], 'distances', lang1.name, lang2.name)
        os.makedirs(lex_comp_log_dir, exist_ok=True)
        lex_comp_log = os.path.join(lex_comp_log_dir, 'lexical_comparison.tsv')
        lang1.write_lexical_comparison(lang2, lex_comp_log)
    
    logger.info('Completed successfully.')