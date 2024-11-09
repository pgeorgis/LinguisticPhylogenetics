import random
from functools import lru_cache
from statistics import StatisticsError, mean, stdev
import numpy as np

from scipy.stats import norm
from utils.distance import dist_to_sim, Distance
from utils.utils import balanced_resample, create_default_dict_of_dicts


# HELPER FUNCTIONS
def get_shared_concepts(lang1, lang2, clustered_cognates):
    """Returns a sorted list of concepts from a clustered cognate class dictionary that both doculects share.

    Args:
        lang1 (phyloLing.Language): First Language object
        lang2 (phyloLing.Language): Second Language object
        clustered_cognates (dict): Nested dictionary of Word objects organized by concepts and cognate classes

    Raises:
        StatisticsError: if no concepts are shared between the two doculects

    Returns:
        list: Sorted list of concepts from the cognate class dictionary shared by both doculects.
    """
    # Needs to be sorted in order to guarantee that random samples of these will be reproducible
    shared_concepts = sorted(list(clustered_cognates.keys() & lang1.vocabulary.keys() & lang2.vocabulary.keys()))
    if len(shared_concepts) == 0:
        raise StatisticsError(f'Error: no shared concepts found between {lang1.name} and {lang2.name}!')
    return shared_concepts


def filter_cognates_by_lang(lang, cluster):
    """Filters an iterable of Word objects and returns only those that belong to a particular language.

    Args:
        lang (phyloLing.Language): Language object of interest
        cluster (iterable): Iterable of Word objects

    Returns:
        list: Word objects belonging to the specified Language
    """
    # Filter by language and sort
    filtered_cognates = list(filter(lambda word: word.language == lang, cluster))
    filtered_cognates.sort(key=lambda word: (
        word.ipa,
        word.orthography,
        word.concept,
        word.getInfoContent(total=True)
        )
    )
    
    return filtered_cognates


def get_noncognate_scores(lang1, lang2, eval_func, seed=1, sample_size=None, as_similarity=False):
    # Get the non-synonymous word pair scores against which to calibrate the synonymous word scores
    key: tuple[str, Distance, int, int] = (lang2.name, eval_func, sample_size, seed)
    if len(lang1.noncognate_thresholds[key]) > 0:
        noncognate_scores = lang1.noncognate_thresholds[key]
    else:
        correlator = lang1.get_phoneme_correlator(lang2)
        noncognate_scores = correlator.noncognate_thresholds(eval_func, seed=seed, sample_size=sample_size)

    # Transform distance scores into similarity scores
    if as_similarity and not eval_func.sim:
        noncognate_scores = [dist_to_sim(score) for score in noncognate_scores]
    
    return noncognate_scores


@lru_cache(maxsize=None)
def get_calibration_params(lang1, lang2, eval_func, seed=1, sample_size=None, as_similarity=False):
    """Gets the mean and standard deviation of similarity of a random sample of non-cognates (word pairs with different concepts) from two doculects to use for CDF weighting.

    Args:
        lang1 (phyloLing.Language): First doculect to compare
        lang2 (phyloLing.Language): Second doculect to compare
        eval_func (Distance): Distance to apply to word pairs
        seed (int): Random seed
        sample_size (int): Size of random sample

    Returns:
        tuple: Mean and standard deviation of similarity of random sampling of non-cognate word pairs
    """
    noncognate_scores = get_noncognate_scores(lang1, lang2, eval_func, seed, sample_size, as_similarity=as_similarity)

    # Calculate mean and standard deviation from this sample distribution
    mean_nc_score = mean(noncognate_scores)  # TODO why is this being recalculated each time?
    nc_score_stdev = stdev(noncognate_scores)

    # Save calibration parameters
    return mean_nc_score, nc_score_stdev


# LINGUISTIC SIMILARITY MEASURES
def binary_cognate_sim(lang1,
                       lang2,
                       clustered_cognates,
                       exclude_synonyms=True):
    """Calculates linguistic similarity based on the proportion of shared cognates between two doculects.

    Args:
        lang1 (phyloLing.Language): First doculect to compare
        lang2 (phyloLing.Language): Second doculect to compare
        clustered_cognates (dict): Nested dictionary of Word objects organized by concepts and cognate classes
        exclude_synonyms (bool, optional): If more than one cognate class is present for the same concept, takes only the most similar pair. Defaults to True.

    Returns:
        float: Similarity value
    """
    # TODO add random forest sampling
    sims = {}
    total_cognate_ids = 0
    shared_concepts = get_shared_concepts(lang1, lang2, clustered_cognates)
    for concept in shared_concepts:
        shared, not_shared = 0, 0
        n_l1_words, n_l2_words = 0, 0
        for cognate_id in clustered_cognates[concept]:
            l1_words = filter_cognates_by_lang(lang1, clustered_cognates[concept][cognate_id])
            l2_words = filter_cognates_by_lang(lang2, clustered_cognates[concept][cognate_id])
            if len(l1_words) > 0 and len(l2_words) > 0:
                n_l1_words += 1
                n_l2_words += 1
                shared += 1
            elif len(l1_words) > 0:
                n_l1_words += 1
                not_shared += 1
            elif len(l2_words) > 0:
                n_l2_words += 1
                not_shared += 1

        if exclude_synonyms:
            sims[concept] = min(shared, 1) if shared > 0 else 0
            total_cognate_ids += 1
        else:
            sims[concept] = shared
            total_cognate_ids += shared + not_shared

    similarity = sum(sims.values()) / total_cognate_ids

    return similarity


def gradient_cognate_dist(lang1,
                          lang2,
                          clustered_cognates,
                          eval_func,
                          exclude_synonyms=True,
                          calibrate=True,
                          min_similarity=0,
                          p_threshold=0.01,
                          seed=1,
                          n_samples=50,
                          sample_size=0.8,
                          logger=None):

    # Set random seed and initialize random number generator
    random.seed(seed)
    rng = np.random.default_rng(seed)

    # Get list of shared concepts between the two languages
    shared_concepts = get_shared_concepts(lang1, lang2, clustered_cognates)

    # Random forest-like sampling
    # Take N samples of the available concepts of size K
    # Calculate the cognate sim for each sample, then average together
    group_scores = {}
    group_counts = np.zeros(len(shared_concepts))
    if n_samples > 1:
        # Set default sample size to 80% of shared concepts
        sample_n = round(sample_size * len(shared_concepts))
        # Create N samples of size K
        concept_groups = {}
        for n in range(n_samples):
            # Take balanced resampling of same-meaning words
            concept_group, group_counts = balanced_resample(
                shared_concepts, sample_n, group_counts, rng
            )
            concept_groups[n] = set(concept_group)

        # Ensure that every shared concept is in at least one of the groups
        # If not, add to smallest (if equal sizes then add to one at random)
        sampled_concepts = set(c for n in concept_groups for c in concept_groups[n])
        for concept in shared_concepts:
            if concept not in sampled_concepts:
                smallest_group = min(concept_groups.keys(), key=lambda x: len(concept_groups[x]))
                concept_groups[smallest_group].add(concept)
    else:
        concept_groups = {0: shared_concepts}

    # Score all shared concepts in advance, then calculate similarity based on the N samples of concepts
    scored_pairs = create_default_dict_of_dicts()
    for concept in shared_concepts:
        for cognate_id in clustered_cognates[concept]:
            concept_cognate_id = f"{concept}_{cognate_id}"
            l1_words = filter_cognates_by_lang(lang1, clustered_cognates[concept][cognate_id])
            l2_words = filter_cognates_by_lang(lang2, clustered_cognates[concept][cognate_id])
            for l1_word in l1_words:
                for l2_word in l2_words:
                    score = eval_func.eval(l1_word, l2_word)

                    # Transform distance into similarity
                    if not eval_func.sim:
                        score = dist_to_sim(score)

                    # Save in scored_pairs
                    scored_pairs[concept_cognate_id][(l1_word, l2_word)] = score

    for n, group in concept_groups.items():
        sims = {}
        group_size: int = len(group)
        for concept in group:
            concept_sims = {}
            l1_wordcount, l2_wordcount = 0, 0

            for cognate_id in clustered_cognates[concept]:
                concept_cognate_id = f"{concept}_{cognate_id}"
                l1_wordcount += len(set(pair[0] for pair in scored_pairs[concept_cognate_id]))
                l2_wordcount += len(set(pair[1] for pair in scored_pairs[concept_cognate_id]))
                concept_sims.update(scored_pairs[concept_cognate_id])

            if len(concept_sims) > 0:
                if exclude_synonyms:
                    sims[concept] = max(concept_sims.values())
                else:
                    sims[concept] = mean(concept_sims.values())

            else:
                if (l1_wordcount > 0) and (l2_wordcount > 0):
                    sims[concept] = 0

        if len(sims) == 0:
            continue

        # Get the non-synonymous word pair scores against which to calibrate the synonymous word scores
        if calibrate:
            # TODO theoretically this would probably be better to recalculate per group as seed+n rather than seed, but doesn't seem to make a significant difference in tree topology but does significantly impact computational time
            # should investigate explicitly whether this makes a significant difference
            mean_nc_score, nc_score_stdev = get_calibration_params(lang1, lang2, eval_func, seed, group_size, as_similarity=True)

        # Apply minimum similarity and calibration
        for concept, score in sims.items():
            sims[concept] = score if score >= min_similarity else 0

            # Calibrate score against scores of non-synonymous word pairs
            # pnorm: proportion of values from a normal distribution with
            # mean and standard deviation defined by those of the sample
            # of non-synonymous word pair scores, which are lower than
            # a particular value (score)
            # e.g. pnorm = 0.99 = 99% of values from the distribution
            # of non-synonymous word pair scores are lower than the given score
            # The higher this value, the more confident we can be that
            # the given score does not come from that distribution,
            # i.e. that it is truly a cognate
            # If lower than p-threshold, reset similarity to 0
            if calibrate and score > 0:
                pnorm = norm.cdf(score, loc=mean_nc_score, scale=nc_score_stdev)
                if 1 - pnorm > p_threshold:
                    sims[concept] = 0

        group_scores[n] = mean(sims.values())

    similarity_score = mean(group_scores.values())
    # if logger:
    #     logger.info(f'Similarity of {lang1.name} and {lang2.name}: {round(similarity_score, 3)}')
        
    distance_score = 1 - similarity_score
    return distance_score
