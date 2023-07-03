import argparse, os
from phyloLing import load_family
from lingDist import cognate_sim
from wordSim import pmi_dist, mutual_surprisal, phon_word_dist, hybrid_sim, LevenshteinDist
import logging

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Loads a lexical dataset in CLDF format and classifies the doculects according to user specifications')
    parser.add_argument('--family', required=True, help='Name of language group to classify')
    parser.add_argument('--file', required=True, help='Input CLDF data file path')
    parser.add_argument('--linkage', default='nj', choices=['nj', 'average', 'complete', 'ward', 'weighted', 'single'], help='Linkage method')
    parser.add_argument('--cognates', default='auto', choices=['auto', 'gold', 'none'], help='Cognate cluster type used for evaluation: "gold" cognates uses labels from dataset assuming that data are sorted into cognate classe; "auto" cognates auto-detects and clusters same-meaning words into cognate classes; "none" performs no separation of cognates from non-cognates')
    parser.add_argument('--cluster', default='hybrid', choices=['phonetic', 'pmi', 'surprisal', 'hybrid', 'levenshtein'], help='Cognate clustering method')
    parser.add_argument('--cutoff', default=None, type=float, help='Cutoff threshold in range [0,1] for clustering cognate sets')
    parser.add_argument('--eval', default='hybrid', choices=['phonetic', 'pmi', 'surprisal', 'hybrid', 'levenshtein'], help='Word form evaluation method')
    parser.add_argument('--min_similarity', default=0, type=float, help='Minimum similarity threshold for word form evaluation')
    parser.add_argument('--ngram', default=1, type=int, help='Phoneme ngram size used for phoneme surprisal calculation')
    parser.add_argument('--n_samples', default=10, type=int, help='Number of random samples for distance evaluation')
    parser.add_argument('--sample_size', default=0.8, type=float, help='Percent of shared concepts to evaluate per sample (default 70%)')
    parser.add_argument('--no_calibration', dest='calibrate', action='store_false', help='Does not use cumulative density function calibration')
    parser.add_argument('--ignore_stress', dest='ignore_stress', action='store_true', help='Ignores stress annotation when loading CLDF dataset and computing phone correspondences')
    parser.add_argument('--no_diphthongs', dest='no_diphthongs', action='store_true', help='Performs IPA string segmentation without diphthongs as single segmental units')
    parser.add_argument('--newick', dest='newick', action='store_true', help='Returns a Newick tree instead of a dendrogram')
    parser.add_argument('--exclude', default=None, nargs='+', help='Languages from CLDF data file to exclude')
    parser.add_argument('--refresh', default=[], nargs='+', help='Languages whose phoneme PMI and/or surprisal should be recalculated')
    parser.add_argument('--min_amc', default=0.65, help='Minimum average mutual coverage among doculects: doculect with lowest coverage is dropped until minimum value is reached')
    parser.add_argument('--outtree', default=None, help='Output file to which Newick tree string should be written')
    parser.add_argument('--loglevel', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Log level for printed log messages')
    parser.set_defaults(
        ignore_stress=False,
        no_diphthongs=False,
        calibrate=True,
        newick=False,
    )
    args = parser.parse_args()

    # Configure the logger
    log_levels = {
        'DEBUG':logging.DEBUG,
        'INFO':logging.INFO,
        'WARNING':logging.WARNING,
        'ERROR':logging.ERROR,
    }
    logging.basicConfig(level=log_levels[args.loglevel], format='%(asctime)s classifyLangs %(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    # Mapping of function labels and default cutoff values
    def hybridSim(x, y):

        return hybrid_sim(
            x, y, 
            funcs={
                pmi_dist:{}, 
                mutual_surprisal:{'ngram_size':args.ngram},
                phon_word_dist:{}
            },
            func_sims=[
                False, 
                False, 
                False
            ],
            # TODO: seems to be best when all funcs weighted equally; or else with ~0.4, 0.2, 0.4 (PMI, surprisal, phonetic) scheme
            # basically, PMI alone produces the best results, at least for Romance
            # but it can be improved by adding in surprisal and phonetic to a lesser extent
            # weights=[0.7, 0.2, 0.1] # best so far
            #weights=[0.7, 0.25, 0.05]
            )
    function_map = {
        # 'label':(function, sim, cutoff)
        'pmi':(pmi_dist, False, {}, 0.36),
        'surprisal':(mutual_surprisal, False, {'ngram_size':args.ngram}, 0.74), # TODO cutoff needs to be recalibrated
        'phonetic':(phon_word_dist, False, {}, 0.16), # TODO cutoff needs to be recalibrated
        'levenshtein':(LevenshteinDist, False, {}, 0.73),
        'hybrid':(hybridSim, True, {}, 0.57), # TODO cutoff needs to be recalibrated
        }
    
    # Set cutoff to default for specified function, if not otherwise specified
    if args.cutoff is None:
        args.cutoff = function_map[args.cluster][-1]

    # Load CLDF dataset
    if args.min_amc:
        args.min_amc = float(args.min_amc)
    family = load_family(args.family, 
                         args.file, 
                         exclude=args.exclude, 
                         min_amc=args.min_amc,
                         ignore_stress=args.ignore_stress,
                         combine_diphthongs=not args.no_diphthongs,
                         logger=logger
                         )

    # Print some summary info about the loaded dataset
    logger.info(f'Loaded {len(family.languages)} doculects.')
    abs_mc, avg_mc = family.calculate_mutual_coverage()
    if avg_mc <= 0.7:
        logger.warning(f'Average mutual coverage = {round(avg_mc, 2)}. Recommend minimum is 0.7.')
    else:
        logger.info(f'Average mutual coverage is {round(avg_mc, 2)} ({abs_mc}/{len(family.concepts)} concepts in all {len(family.languages)} doculects).')

    # Load or calculate phoneme PMI
    logger.info(f'Loading {family.name} phoneme PMI...')
    family.load_phoneme_pmi(excepted=args.refresh)

    # Load or calculate phoneme surprisal
    if args.eval == 'surprisal' or args.eval == 'hybrid':
        logger.info(f'Loading {family.name} phoneme surprisal...')
        if args.cognates == 'gold':
            family.load_phoneme_surprisal(ngram_size=args.ngram, gold=True, excepted=args.refresh)
        else:
            family.load_phoneme_surprisal(ngram_size=args.ngram, gold=False, excepted=args.refresh)

    # If phoneme PMI/surprisal was refreshed for one or more languages, rewrite the saved files
    if len(args.refresh) > 0:
        family.write_phoneme_pmi()
        if args.eval == 'surprisal' or args.eval == 'hybrid':
            family.write_phoneme_surprisal(ngram_size=args.ngram)

    # Load pre-clustered cognate sets, if available
    family.load_clustered_cognates()

    # Set cognate cluster ID according to settings
    if args.cognates == 'auto':
        cog_id = f'{family.name}_distfunc-{args.cluster}-{function_map[args.cluster][1]}_cutoff-{args.cutoff}'

    # Generate Newick tree string
    logger.info(f'Generating phylogenetic tree...')
    dist_func = cognate_sim # TODO other options?
    code = family.generate_test_code(dist_func, sim=True, cognates=args.cognates, cutoff=args.cutoff)
    tree = family.draw_tree(
        dist_func=dist_func,
        sim=True, # cognate_sim
        cluster_func=function_map[args.cluster][0],
        cluster_sim=function_map[args.cluster][1],
        cutoff=args.cutoff,
        eval_func=(function_map[args.eval][0], function_map[args.eval][-2]), #function, kwargs
        eval_sim=function_map[args.eval][1],
        cognates=args.cognates, 
        method=args.linkage, # this should be changed to linkage rather than method
        calibrate=args.calibrate, # argument for cognate_sim
        n_samples=args.n_samples, # argument for cognate_sim
        sample_size=args.sample_size, # argument for cognate_sim
        min_similarity=args.min_similarity, # argument for cognate_sim
        logger=logger, # argument for cognate_sim
        title=family.name, 
        outtree=args.outtree,
        return_newick=args.newick)
    if args.outtree:
        logger.info(f'Wrote Newick tree to {args.outtree}')
    else:
        logger.info(f'Wrote Newick tree to {os.path.join(family.tree_dir, f"{code}.tre")}')
    
    # family.plot_languages(
    #     dist_func=cognate_sim, # other options?
    #     sim=True, # cognate_sim
    #     cluster_func=function_map[args.cluster][0],
    #     cluster_sim=function_map[args.cluster][1],
    #     cutoff=args.cutoff,
    #     concept_list=None,
    #     eval_func=(function_map[args.eval][0], function_map[args.eval][-2]), #function, kwargs
    #     eval_sim=function_map[args.eval][1],
    #     cognates=args.cognates)            
    #     # dimensions=2, top_connections=0.3, max_dist=1, alpha_func=None,
    #     # plotsize=None, invert_xaxis=False, invert_yaxis=False,
    #     # title=None, save_directory=None
    #     # **kwargs)
    
    if tree:
        if args.outtree:
            with open(args.outtree, 'w') as f:
                f.write(tree)
        else:
            print(tree)