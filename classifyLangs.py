import argparse, os
from loadLangs import load_family
from lingDist import cognate_sim
from wordSim import pmi_dist, surprisal_sim, word_sim, hybrid_sim, LevenshteinDist

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Loads a lexical dataset in CLDF format and classifies the doculects according to user specifications')
    parser.add_argument('--family', required=True, help='Name of language group to classify')
    parser.add_argument('--file', required=True, help='Input CLDF data file path')
    parser.add_argument('--linkage', default='nj', choices=['nj', 'average', 'complete', 'ward', 'weighted', 'single'], help='Linkage method')
    parser.add_argument('--cognates', default='auto', choices=['auto', 'gold', 'none'], help='Cognate clusters to use') # needs better description
    parser.add_argument('--cluster', default='hybrid', choices=['phonetic', 'pmi', 'surprisal', 'hybrid', 'levenshtein'], help='Cognate clustering method')
    parser.add_argument('--cutoff', default=None, type=float, help='Cutoff threshold in range [0,1] for clustering cognate sets')
    parser.add_argument('--eval', default='hybrid', choices=['phonetic', 'pmi', 'surprisal', 'hybrid', 'levenshtein'], help='Word form evaluation method')
    parser.add_argument('--min_similarity', default=0, type=float, help='Minimum similarity threshold for word form evaluation')
    parser.add_argument('--ngram', default=1, type=int, help='Phoneme ngram size used for phoneme surprisal calculation')
    parser.add_argument('--no_calibration', dest='calibrate', action='store_false', help='Does not use cumulative density function calibration')
    parser.add_argument('--newick', dest='newick', action='store_true', help='Returns a Newick tree instead of a dendrogram')
    parser.set_defaults(
        calibrate=True,
        newick=False,
    )
    args = parser.parse_args()

    # Mapping of function labels and default cutoff values
    function_map = {
        # 'label':(function, sim, cutoff)
        'pmi':(pmi_dist, False, {}, 0.36),
        #'surprisal':(lambda x, y: surprisal_sim(x, y, ngram_size=args.ngram), True, 0.74),
        'surprisal':(surprisal_sim, True, {'ngram_size':args.ngram}, 0.74),
        'phonetic':(word_sim, True, 0.16),
        'levenshtein':(LevenshteinDist, False, 0.73)
        }
    function_map['hybrid'] = (lambda x, y: hybrid_sim(
        x, y, 
        funcs={
            pmi_dist:{}, 
            surprisal_sim:{'ngram_size':args.ngram}, 
            word_sim:{}
            },
        func_sims=[False, True, True]),
        True,
        {},
        0.57) # this cutoff value pertains to ngram_size=1 only, would need to be recalculated for other ngram sizes
    
    # Set cutoff to default for specified function, if not otherwise specified
    if args.cutoff is None:
        args.cutoff = function_map[args.cluster][-1]

    # Load CLDF dataset
    family = load_family(args.family, args.file)

    # Load or calculate phoneme PMI
    print(f'Loading {family.name} phoneme PMI...')
    family.load_phoneme_pmi()

    # Load or calculate phoneme surprisal
    print(f'Loading {family.name} phoneme surprisal...')
    family.load_phoneme_surprisal(ngram_size=args.ngram)

    # Load pre-clustered cognate sets, if available
    family.load_clustered_cognates()

    # Set cognate cluster ID according to settings
    if args.cognates == 'auto':
        cog_id = f'{family.name}_distfunc-{args.cluster}-{function_map[args.cluster][1]}_cutoff-{args.cutoff}'

    # Generate Newick tree string
    tree = family.draw_tree(
        dist_func=cognate_sim, # other options?
        sim=True, # cognate_sim
        cluster_func=function_map[args.cluster][0],
        cluster_sim=function_map[args.cluster][1],
        cutoff=args.cutoff,
        eval_func=(function_map[args.eval][0], function_map[args.eval][-2]), #function, kwargs
        eval_sim=function_map[args.eval][1],
        cognates=args.cognates, 
        method=args.linkage, # this should be changed to linkage rather than method
        calibrate=args.calibrate,
        min_similarity=args.min_similarity,
        title=family.name, 
        save_directory=os.path.join(family.directory, 'Plots'),
        return_newick=args.newick)
    
    if tree:
        print(tree)