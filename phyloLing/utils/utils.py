"""General auxiliary functions."""

import datetime
import operator
from collections import defaultdict
import subprocess

from networkx import Graph
from networkx.algorithms.components.connected import connected_components
import numpy as np
from numpy import array, array_split
from numpy.random import permutation


def csv2dict(csvfile, header=True, sep=',', start=0, encoding='utf_8') -> dict[int, dict[str, str]]:
    """Reads a CSV file into a dictionary"""
    csv_dict: dict[int, dict[str, str]] = create_default_dict(2, '')
    with open(csvfile, 'r', encoding=encoding) as csv_file:
        csv_file = csv_file.readlines()
        columns: list[str] = [item.strip() for item in csv_file[start].split(sep)]
        if header:
            start += 1
        for i in range(start, len(csv_file)):
            line: list[str] = [item.strip() for item in csv_file[i].split(sep)]
            for j in range(len(columns)):
                key = ''
                if header:
                    key += columns[j]
                else:
                    key += str(j)
                try:
                    csv_dict[i][key] = line[j]
                except IndexError:
                    pass
    return csv_dict


def validate_class(objs, classes):
    """Validates that a set of of objects are of the expected classes.

    Args:
        objs (iterable): iterable of objects
        classes (iterable): iterable of possible classes

    Raises:
        TypeError: if the object is of an unexpected class
    """
    for obj in objs:
        if not any(isinstance(obj, cls) for cls in classes):
            err = f"Object {obj} (type = {type(obj)}) is not of the expected classes: {classes}"
            raise TypeError(err)


def create_timestamp():
    """Create time stamp with current date and time."""
    # Get the current date and time
    current_datetime = datetime.datetime.now()
    # Format the date and time as a string
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    return current_datetime, formatted_datetime


def create_datestamp():
    """Create a timestamp with the current date in YYYY-MM-DD format."""
    # Get the current date
    current_date = datetime.datetime.now().date()
    # Format the date as a string
    formatted_date = current_date.strftime("%Y-%m-%d")
    return formatted_date


def calculate_time_interval(datetime1, datetime2):
    """
    Calculate the time interval elapsed between two datetime objects.

    Args:
        datetime1 (datetime): The first datetime object.
        datetime2 (datetime): The second datetime object.

    Returns:
        str: The formatted string representing the time interval elapsed between the two datetimes.
    """
    time_interval = datetime2 - datetime1
    total_seconds = int(time_interval.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def dict_tuplelist(dic, sort=True, n=1, reverse=True):
    """Returns a list of (key, value) tuples from the dictionary
    if sort == True, sorts the list by the nth tuple item, by default in decending order"""
    d = [(key, dic[key]) for key in dic]
    if sort:
        d.sort(key=operator.itemgetter(n), reverse=reverse)
    return d


def default_dict(dic, lmbda):
    """Turns an existing dictionary into a default dictionary with default value l"""
    return defaultdict(lambda: lmbda, dic)


def combine_dicts(*dicts):
    """Combines a series of dictionaries into a single dictionary."""
    combined = {}
    for dic in dicts:
        combined.update(dic)
    return combined


def normalize_dict(dict_, default=False, default_value=None, return_=True):
    """Normalizes the values of a dictionary"""
    """If default==True, returns a default dictionary with default value lmbda"""
    """If return_==False, modifies the input dictionary without returning anything"""
    if default is True:
        normalized = create_default_dict(1, default_value)
    else:
        normalized = {}
    total = sum(list(dict_.values()))
    for key in dict_:
        if return_:
            normalized[key] = dict_[key] / total
        else:
            dict_[key] = dict_[key] / total
    if return_:
        return normalized


def chunk_list(lis, n):
    """Splits a list into sublists of length n; if not evenly divisible by n,
    the final sublist contains the remainder"""
    return [lis[i * n:(i + 1) * n] for i in range((len(lis) + n - 1) // n)]


def split_list_randomly(lst, n):
    if n <= 0:
        raise ValueError("Number of groups (n) should be greater than zero.")
    if n > len(lst):
        raise ValueError("Number of groups (n) should not exceed the length of the list.")

    random_indices = permutation(len(lst))
    lst_array = array(lst)
    groups = array_split(lst_array[random_indices], n)

    return groups


def combine_overlapping_lists(list_of_lists):
    # https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements

    def to_graph(lis):
        G = Graph()
        for part in lis:
            # each sublist is a bunch of nodes
            G.add_nodes_from(part)
            # it also imlies a number of edges:
            G.add_edges_from(to_edges(part))
        return G

    def to_edges(graph):
        """Returns edges tuples of graph:
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]"""
        it = iter(graph)
        last = next(it)
        for current in it:
            yield last, current
            last = current

    G = to_graph(list_of_lists)
    return list(connected_components(G))

def convert_sets_to_lists(obj):
    """Recursively convert list objects to sets."""
    if isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(i) for i in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj

def rescale(val, lis, new_min=0, new_max=1):
    """Rescales a value between new_min and new_max according to the values of lis"""
    numerator = new_max - new_min
    old_max, old_min = max(lis), min(lis)
    denominator = old_max - old_min
    part1 = numerator / denominator
    part2 = val - old_max
    part3 = new_max
    return part1 * part2 + part3


def rescale_dict_values(d, comparison_list=None):
    # If no comparison list is provided, use the dictionary's values
    if comparison_list is None:
        comparison_list = list(d.values())

    min_val = min(comparison_list)
    max_val = max(comparison_list)

    # Handle the edge case where all values are the same (avoid division by zero)
    if max_val == min_val:
        return {k: 0.5 for k in d}

    # Rescale the values between 0 and 1
    return {k: (v - min_val) / (max_val - min_val) for k, v in d.items()}


def keywithminval(d):
    """Returns the dictionary key with the lowest value"""
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(min(v))]


def keywithmaxval(d):
    """Returns the dictionary key with the maximum value"""
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]


def balanced_resample(population, sample_size, sampled_counts, rng):
    """Resample wordlist taking into account how many times each item has already been sampled."""
    # Inverse weighting to prefer less-sampled data
    prob_weights = 1 / (sampled_counts + 1)
    # Normalize to sum to 1
    prob_weights /= prob_weights.sum()
    # Sample with weighted probabilities without replacement
    pop_size = len(population)
    sample_indices = rng.choice(np.arange(pop_size), size=sample_size, p=prob_weights, replace=False)
    # Update count based on selected indices
    sampled_counts[sample_indices] += 1
    # Extract sampled words from selected indices
    sample = [population[i] for i in sample_indices]
    return sample, sampled_counts


def get_git_commit_hash():
    """Get latest git commit hash of current repository."""
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', "--short", 'HEAD']).strip().decode('utf-8')
        return commit_hash
    except subprocess.CalledProcessError as e:
        return None

def top_n_keys(d, n=None):
    if n is None:
        n = len(d)
    top_keys = sorted(d, key=d.get, reverse=True)[:n]
    return top_keys


def segment_ranges(numbers):
    if not numbers:
        return []

    segmented = []
    start = numbers[0]
    end = numbers[0]

    for i in range(1, len(numbers)):
        # Check if current number is consecutive
        if numbers[i] == end + 1:
            end = numbers[i]
        else:
            # Append the range or standalone number
            if start == end:
                segmented.append(start)
            else:
                segmented.append((start, end))
            # Start a new range
            start = numbers[i]
            end = numbers[i]

    # Append the last range or standalone number
    if start == end:
        segmented.append(start)
    else:
        segmented.append((start, end))

    return segmented

class _NestedDefaultDictFactoryForPrimitiveValues:
    def __init__(self, depth, value):
        self.depth = depth
        if not isinstance(value, (int, float, str)):
            raise TypeError("value must be a primitive type")
        self.value = value

    def __call__(self):
        if self.depth == 0:
            return self.value
        else:
            return defaultdict(_NestedDefaultDictFactoryForPrimitiveValues(self.depth - 1, self.value))


def create_default_dict(depth: int, value):
    if not isinstance(depth, int):
        raise TypeError("depth must be an integer")
    if depth < 1:
        raise ValueError("depth must be greater than 0")
    if not isinstance(value, (int, float, str)):
        raise TypeError("value must be a primitive type")

    return defaultdict(_NestedDefaultDictFactoryForPrimitiveValues(depth - 1, value))


class _NestedDefaultDictFactoryForValuesWithFactory:
    def __init__(self, depth, leaf_factory):
        self.depth = depth
        if not callable(leaf_factory):
            raise TypeError("leaf_factory must be a callable")
        self.leaf_factory = leaf_factory

    def __call__(self):
        if self.depth == 0:
            return self.leaf_factory()
        else:
            return defaultdict(_NestedDefaultDictFactoryForValuesWithFactory(self.depth - 1, self.leaf_factory))


def create_default_dict_of_dicts(depth: int = 1):
    if not isinstance(depth, int):
        raise TypeError("depth must be an integer")
    if depth < 1:
        raise ValueError("depth must be greater than 0")

    return defaultdict(_NestedDefaultDictFactoryForValuesWithFactory(depth - 1, dict))


def dict_of_sets():
    return defaultdict(set)


def dict_to_defaultdict_with_value(d: dict, value, depth: int) -> defaultdict:
    if not isinstance(depth, int):
        raise TypeError("depth must be an integer")
    if depth < 1:
        raise ValueError("depth must be greater than 0")
    if depth == 1:
        dd = defaultdict(lambda: value)
        dd.update(d)
        return dd
    def default_factory():
        return dict_to_defaultdict_with_value({}, value, depth - 1)
    dd = defaultdict(default_factory)
    for key, subdict in d.items():
        dd[key] = dict_to_defaultdict_with_value(subdict, value, depth - 1)
    return dd


def update_default_dict(target: dict, source: dict, depth: int) -> None:
    if not isinstance(depth, int):
        raise TypeError("depth must be an integer")
    if depth < 1:
        raise ValueError("depth must be greater than 0")
    if depth == 1:
        target.update(source)
    else:
        for key, subdict in source.items():
            update_default_dict(target[key], subdict, depth - 1)


def ddict2dict(d) -> dict:
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)
