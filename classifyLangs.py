import argparse, os
from phyloLing import load_family
from lingDist import cognate_sim
from wordDist import PMIDist, SurprisalDist, PhonologicalDist, HybridSim, LevenshteinDist
from auxFuncs import Distance
import logging

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Loads a lexical dataset in CLDF format and classifies the doculects according to user specifications')
    parser.add_argument('--family', required=True, help='Name of language group to classify')
    parser.add_argument('--file', required=True, help='Input CLDF data file path')
    parser.add_argument('--linkage', default='nj', choices=['nj', 'average', 'complete', 'ward', 'weighted', 'single'], help='Linkage method')
    parser.add_argument('--cognates', default='auto', choices=['auto', 'gold', 'none'], help='Cognate cluster type used for evaluation: "gold" cognates uses labels from dataset assuming that data are sorted into cognate classe; "auto" cognates auto-detects and clusters same-meaning words into cognate classes; "none" performs no separation of cognates from non-cognates')
    # TODO possibly change default setting to "none"; possibly rename "none" to "no clustering" or make clearer that there is no explicit clustering, but the calibration argument would still effectively apply some kind of weighting towards true cognates
    parser.add_argument('--cluster', default='hybrid', choices=['phonetic', 'pmi', 'surprisal', 'hybrid', 'levenshtein'], help='Cognate clustering method')
    parser.add_argument('--cluster_threshold', default=None, type=float, help='Cutoff threshold in range [0,1] for clustering cognate sets')
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

    function_map = {
        'pmi':PMIDist,
        'surprisal':SurprisalDist,
        'phonetic':PhonologicalDist, # TODO name doesn't match
        'levenshtein':LevenshteinDist,
        'hybrid':HybridSim,
        }
    # this weighting scheme works well seemingly (PMI, surprisal, phonological): 0.5, 0.25, 0.25 OR 0.25, 0.5, 0.25
    #function_map['hybrid'].set('weights', (0.25, 0.5, 0.25)) # PMI, surprisal, phonological
    function_map['hybrid'].set('weights', (0.5, 0.25, 0.25)) # PMI, surprisal, phonological
    if args.cognates == 'auto':
        clusterDist = function_map[args.cluster]
    else:
        clusterDist = None
    evalDist = function_map[args.eval]
    
    # Set specified cluster threshold, if different from default
    if args.cluster_threshold:
        clusterDist.cluster_threshold = args.cluster_threshold
    
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
    # Needs to occur after PMI/surprisal was recalculated for the language(s) in question
    if len(args.refresh) > 0:
        family.calculate_phoneme_pmi()
        family.write_phoneme_pmi()
        if args.eval == 'surprisal' or args.eval == 'hybrid':
            family.calculate_phoneme_surprisal(ngram_size=args.ngram)
            family.write_phoneme_surprisal(ngram_size=args.ngram)

    # Load pre-clustered cognate sets, if available
    family.load_clustered_cognates()

    # Set cognate cluster ID according to settings
    if args.cognates == 'auto':
        cog_id = f'{family.name}_distfunc-{args.cluster}-{function_map[args.cluster][1]}_cutoff-{args.cluster_threshold}'

    # Create Distance measure according to settings
    dist_func = cognate_sim # TODO other options?
    distFunc = Distance(
        func=dist_func, 
        name='CognateSim',
        sim=True, 
        # cognate_sim kwargs
        eval_func=evalDist,
        n_samples=args.n_samples, 
        sample_size=args.sample_size, 
        calibrate=args.calibrate,
        min_similarity=args.min_similarity,
        logger=logger,
        )
    
    # Generate test code 
    code = family.generate_test_code(distFunc, sim=True, cognates=args.cognates, cutoff=args.cluster_threshold)
    logger.debug(f'Experiment ID: {code}')

    # Generate Newick tree string
    logger.info(f'Generating phylogenetic tree...')
    tree = family.draw_tree(
        dist_func=distFunc,
        cluster_func=clusterDist,
        cognates=args.cognates, 
        method=args.linkage, # this should be changed to linkage rather than method
        title=family.name, 
        outtree=args.outtree,
        return_newick=args.newick)
    if args.outtree:
        logger.info(f'Wrote Newick tree to {args.outtree}')
    else:
        logger.info(f'Wrote Newick tree to {os.path.join(family.tree_dir, f"{code}.tre")}')

    if tree:
        if args.outtree:
            with open(args.outtree, 'w') as f:
                f.write(tree)
        else:
            print(tree)
    
    def write_lang_dists_to_tsv(dist, outfile):
        with open(outfile, 'w') as f:
            header = '\t'.join(['Language1', 'Language2', 'Measurement'])
            f.write(f'{header}\n')
            for key, value in dist.measured.items():
                lang1, lang2, kwargs = key
                line = '\t'.join([lang1.name, lang2.name, str(value)])
                f.write(f'{line}\n')
    
    write_lang_dists_to_tsv(distFunc, outfile=os.path.join(family.dist_matrix_dir, f'{code}_scored.tsv'))