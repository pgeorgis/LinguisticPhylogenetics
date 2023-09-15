# LinguisticPhylogenetics

## Table of Contents

* [Setup](#setup)
* [Running a Phylogenetic Experiment](#running-a-phylogenetic-experiment)
* [Parameters](#parameters)

## Setup 
When setting up for the first time, run the following command to create the virtual environment and install required packages. After the initial set up, only activation of the virtual environment (venv) is required. 

`make init`

To activate the virtual environment after initial setup:

`source venv/bin/activate`

## Running a phylogenetic experiment
A phylogenetic experiment can be run using classifyLangs.py along with a config.yml file that specifies parameters for various aspects of the phylogenetic analysis:

### `make classify`
`make classify CONFIG=<PATH TO CONFIG FILE> [LOGLEVEL=<DESIRED LOG LEVEL>]`

Replace <PATH TO CONFIG FILE> with the path to your configuration file. You can also include an optional LOGLEVEL argument to specify the desired log level (e.g., DEBUG, INFO, etc.). 

You will find the results in your specified `outdir` (see below) under `logs/classify.log`.

### Using Python directly
Alternatively, you can run the classification script directly using Python:

Activate the virtual environment (if not already activated):

`source venv/bin/activate`

Run the classification script:

`python3 classifyLangs.py <PATH TO CONFIG FILE> [--loglevel <DESIRED LOG LEVEL>]`

### Config
A sample config file with default parameters is saved at `config/default_config.yml`. The user only needs to specify the path to the input CLDF file in the `file` parameter under the `family` section, e.g.:

```

family:

    file: path/to/your/config.yml

```

Any parameters left unspecified in the config file will default to those listed in `config/default_config.yml`.

### Results
If no `outdir` parameter is specified (also under the `family` section), the results will be saved to the same directory where the input file is saved.

## Parameters
The following is a brief description of the configurable parameters.

### `family`:

  `name`: Name of language group to classify

  `file`: File path to input CLDF TSV file

  `min_amc`: Minimum average mutual coverage among doculects. The doculect with lowest coverage is pruned until the specified minimum value is reached. Recommended minimum is 0.7 and below 0.65 results may be unreliable. # TODO add minimum concepts

  `exclude`: List of doculects from the input file to exclude from the experiment.

  `outdir`: Directory where results of the experiment are saved. 


### `transcription`:

#### `global`: Default transcription parameters to apply to all doculects unless overridden below for individual doculects.
    
  `ignore_stress`: If `true`, removes stress annotation from transcriptions in order to treat stressed and unstressed segments identically. If `false`, stressed and unstressed segments are treated separately. Default is `false`.
  
  `combine_diphthongs`: If `true`, sequences of a non-syllabic vowel plus a syllabic vowel (/VV̯/ or /V̯V/) are treated as a single diphthongal segment during IPA string segmentation. If `false`, they are segmented into separate units. Default is `true`.

  `normalize_geminates`: If `true`, geminates transcribed as double consonants, e.g. /bb/, are normalized to, e.g. /bː/. In this case /b/ and /bː/ are treated as separate phonemes. Default is `false`, which does not alter the input transcriptions.

  `preaspiration`: If `true`, accounts for prespirated consonants during IPA string segmentation, e.g. Icelandic <þakka> /θˈaʰka/would be segmented as /θ/, /ˈa/, /ʰk/, /a/. If `false`, aspiration diacritics (<ʰ>, <ʱ>) are assumed to modify the preceding segment. Default is `true`.

  `ch_to_remove`: Set of characters which are preprocessed out of transcriptions. By default this includes all suprasegmental diacritics. # TODO should this be changed? Maybe make a `keep_suprasegmentals` parameter that is set to `false` by default?

#### `doculects`: Override transcription parameters for individual doculects, if different from the globally specified transcription parameters. Unspecified doculects or parameters will default to the globally specified parameters. See `config/default_config.yml` for an example.


### `pmi`:

  `refresh`: List of individual doculects for which phoneme PMI should be recalculated from scratch. Otherwise pre-calculated PMI values will be reloaded, if available.

  `refresh_all_pmi`: Recalculates phoneme PMI for all doculect pairs. Default is `false`.


### `surprisal`:

  `ngram`: Ngram size used for phoneme surprisal calculation. Default is 1.

  `phon_env`: Incorporates the phonological environment of a segment into the surprisal calculation. Default is `true`.

  `refresh`: List of individual doculects for which phoneme surprisal should be recalculated from scratch. Otherwise pre-calculated surprisal values will be reloaded, if available.

  `refresh_all_surprisal`: Recalculates phoneme surprisal for all doculect pairs. Default is `false`.


### `cluster`:

  `cognates`: Method for handling cognate clustering: [`auto`, `gold`, `none`]. `auto` performs cognate clustering using the `method` and `cluster_threshold` specified below to sort same-meaning words into cognate classes. `gold` uses cognate class labels from the input dataset, assuming that data are already annotated for the relevant cognate classes. `none` performs no cognate clustering and treats all same-meaning words as if they belong to the same cognate class. Default is `auto`. # TODO change to `none`?

  `method`: Distance measure used for cognate clustering: [`pmi`, `surprisal`, `phon`, `hybrid`, `levenshtein`]. The default is `hybrid`, which combines `pmi`, `surprisal`, and `phon`.

  `cluster_threshold`: Cutoff threshold in range [0,1] for clustering cognate sets. Default is 0.5.


### `evaluation`:

  `similarity`: Type of cognate similarity measure: [`gradient`, `binary`]. `gradient` calculates linguistic similarity using a gradient measure applied to cognate word forms. `binary` calculates linguistic similarity based solely on the proportion of shared cognates two doculects share. Default is `gradient`.

  `method`: Distance measure used for `gradient` cognate similarity: [`pmi`, `surprisal`, `phon`, `hybrid`, `levenshtein`]. The default is `hybrid`, which combines `pmi`, `surprisal`, and `phon`.

  `pmi_weight`: Weight for `pmi` contribution to `hybrid` distance.

  `surprisal_weight`: Weight for `surprisal` contribution to `hybrid` distance.

  `phon_weight`: Weight for `phon` contribution to `hybrid` distance. 

  `n_samples`: Number of random forest-like samples of cognates to draw for phylogenetic distance evaluation. Default is 10.

  `sample_size`: Proportion of shared concepts to evaluate per sample. Default is 0.8.

  `calibrate`: Uses the cumulative distribution function (CDF) of a normal (Gaussian) distribution to calibrate and weight the contribution of each shared word pair by its likelihood of being cognate, compared against a sample of non-synonymous word pairs from the two doculects. Default is `true`. Only available for `similarity`=`gradient`. # TODO possibly change the name of this parameter

  `min_similarity`: Minimum similarity threshold for word pairs, adding non-linearity to the evaluation. If the evaluated similarity is below the threshold, the similarity is evaluated as 0 instead. Default is 0. Only available for `similarity`=`gradient`. # TODO experiment with this parameter


### `tree`:

  `linkage`: Linkage method for producing a phylogenetic tree: [`nj`, `average`, `centroid`, `median`, `single`, `complete`, `ward`, `weighted`]. Default is `nj`, using the Neighbor-Joining algorithm (Saitou & Nei, 1987). Other methods use hierarchical clustering.

  `newick`: Returns a Newick string instead of a dendrogram. Default is `true`.