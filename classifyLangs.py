import argparse
import os
import logging
import yaml
from auxFuncs import Distance
from phyloLing import load_family
from wordDist import PMIDist, SurprisalDist, PhonologicalDist, LevenshteinDist, hybrid_dist
from lingDist import cognate_sim

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
        'method':{'phon', 'pmi', 'surprisal', 'hybrid', 'levenshtein'},
    },
    'evaluation':{
        'method':{'phon', 'pmi', 'surprisal', 'hybrid', 'levenshtein'},
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
    'levenshtein':LevenshteinDist,
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
            try:
                param_value = params[section_name][param_name]
            except KeyError:
                breakpoint()
            if param_value not in valid_params[section_name][param_name]:
                logger.error(f'Invalid parameter value "{param_value}" for parameter "{param_name}". Valid options: {", ".join(valid_params[section_name][param_name])}')
                raise ValueError
            
    # Ensure that minimally the input file path is specified
    if 'file' not in params['family']:
        logger.error('Input file ("file") argument must be specified!')
        raise ValueError


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
    
    
    # Set ngram size used for surprisal
    if params['evaluation']['method'] == 'surprisal' or params['evaluation']['method'] == 'hybrid':
        function_map['surprisal'].set('ngram_size', params['surprisal']['ngram'])
        SurprisalDist = function_map['surprisal']

        # Initialize HybridDist object 
        if params['evaluation']['method'] == 'hybrid':
            HybridDist = Distance(
                func=hybrid_dist,
                name='HybridDist',
                cluster_threshold=0.57, # TODO cluster_threshold needs to be recalibrated
                funcs=[PMIDist, SurprisalDist, PhonologicalDist],
                # this weighting scheme works well seemingly (PMI, surprisal, phonological): 0.5, 0.25, 0.25 OR 0.25, 0.5, 0.25
                weights=(
                    params['evaluation']['pmi_weight'],
                    params['evaluation']['surprisal_weight'],
                    params['evaluation']['phon_weight'],
                )
            )
            HybridSim = HybridDist.to_similarity(name='HybridSim') 

            # Add HybridSim to function map
            function_map['hybrid'] = HybridSim

    # Designate cluster function if performing auto cognate clustering 
    if params['cluster']['cognates'] == 'auto':
        clusterDist = function_map[params['cognates']['method']]
        # Set specified cluster threshold
        clusterDist.cluster_threshold = params['cluster']['cluster_threshold']
    else:
        clusterDist = None
    
    # Designate evaluation function
    evalDist = function_map[params['evaluation']['method']]
    
    # Load CLDF dataset
    if params['family']['min_amc']:
        params['family']['min_amc'] = float(params['family']['min_amc'])
    family = load_family(params['family']['name'], 
                         params['family']['file'], 
                         exclude=params['family']['exclude'], 
                         min_amc=params['family']['min_amc'],
                         ignore_stress=params['transcription']['ignore_stress'],
                         combine_diphthongs=params['transcription']['combine_diphthongs'],
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
    if not params['pmi']['refresh_all_pmi']:
        logger.info(f'Loading {family.name} phoneme PMI...')
        family.load_phoneme_pmi(excepted=params['pmi']['refresh'])

    # Load or calculate phoneme surprisal
    if not params['surprisal']['refresh_all_surprisal']:
        if params['evaluation']['method'] == 'surprisal' or params['evaluation']['method'] == 'hybrid':
            logger.info(f'Loading {family.name} phoneme surprisal...')
            if params['cluster']['cognates'] == 'gold':
                family.load_phoneme_surprisal(ngram_size=params['surprisal']['ngram'], gold=True, excepted=params['surprisal']['refresh'])
            else:
                family.load_phoneme_surprisal(ngram_size=params['surprisal']['ngram'], gold=False, excepted=params['surprisal']['refresh'])

    # If phoneme PMI/surprisal was refreshed for one or more languages, rewrite the saved files
    # Needs to occur after PMI/surprisal was recalculated for the language(s) in question
    if params['pmi']['refresh_all_pmi'] or params['surprisal']['refresh_all_surprisal'] or len(params['pmi']['refresh']) or len(params['surprisal']['refresh']) > 0:
        family.calculate_phoneme_pmi()
        family.write_phoneme_pmi()
        if params['evaluation']['method'] == 'surprisal' or params['evaluation']['method'] == 'hybrid':
            family.calculate_phoneme_surprisal(ngram_size=params['surprisal']['ngram'])
            family.write_phoneme_surprisal(ngram_size=params['surprisal']['ngram'])

    # Auto cognate clustering only
    if params['cluster']['cognates'] == 'auto':
        # Load pre-clustered cognate sets, if available
        family.load_clustered_cognates()

        # Set cognate cluster ID according to settings
        # TODO this ID won't work because of function_map[params['cognates']['method']][1]
        cog_id = f"{family.name}_distfunc-{params['cognates']['method']}-{function_map[params['cognates']['method']][1]}_cutoff-{params['cluster']['cluster_threshold']}"

    # Create Distance measure according to settings
    dist_func = cognate_sim # TODO other options?
    distFunc = Distance(
        func=dist_func, 
        name='CognateSim',
        sim=True, 
        # cognate_sim kwargs
        eval_func=evalDist,
        n_samples=params['evaluation']['n_samples'], 
        sample_size=params['evaluation']['sample_size'], 
        calibrate=params['evaluation']['calibrate'],
        min_similarity=params['evaluation']['min_similarity'],
        logger=logger,
        )
    
    # Generate test code 
    code = family.generate_test_code(distFunc, cognates=params['cluster']['cognates'], cutoff=params['cluster']['cluster_threshold'])
    code += family.generate_test_code(evalDist)
    logger.debug(f'Experiment ID: {code}')

    # Generate Newick tree string
    logger.info(f'Generating phylogenetic tree...')
    outtree = os.path.join(params['family']['outdir'], 'trees', code+'.tre') # TODO this could still be improved
    tree = family.draw_tree(
        cluster_func=clusterDist,
        dist_func=distFunc,
        cognates=params['cluster']['cognates'], 
        method=params['tree']['linkage'], # TODO this should be changed within draw_tree() to linkage rather than method
        title=family.name, 
        outtree=outtree,
        return_newick=params['tree']['newick'])
    with open(outtree, 'w') as f:
        f.write(tree)
    logger.info(f'Wrote Newick tree to {outtree}')
    print(tree)
    
    write_lang_dists_to_tsv(distFunc, outfile=os.path.join(family.dist_matrix_dir, f'{code}_scored.tsv'))
    
    logger.info('Completed successfully.')