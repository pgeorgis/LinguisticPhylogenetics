from collections import defaultdict
import datetime
from networkx import Graph
from networkx.algorithms.components.connected import connected_components
from numpy import array, array_split
from numpy.random import permutation
import operator

# GENERAL AUXILIARY FUNCTIONS

def csv2dict(csvfile, header=True, sep=',', start=0, encoding='utf_8'):
    """Reads a CSV file into a dictionary"""
    csv_dict = defaultdict(lambda:defaultdict(lambda:''))
    with open(csvfile, 'r', encoding=encoding) as csv_file:
        csv_file = csv_file.readlines()
        columns = [item.strip() for item in csv_file[start].split(sep)]
        if header:
            start += 1
        for i in range(start, len(csv_file)):
            line = [item.strip() for item in csv_file[i].split(sep)]
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
            raise TypeError(f"Object {obj} (type = {type(obj)}) is not of the expected classes: {classes}")

def create_timestamp():
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Format the date and time as a string
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    
    return formatted_datetime

def dict_tuplelist(dic, sort=True, n=1, reverse=True):
    """Returns a list of (key, value) tuples from the dictionary
    if sort == True, sorts the list by the nth tuple item, by default in decending order"""
    d = [(key, dic[key]) for key in dic]
    if sort:
        d.sort(key=operator.itemgetter(n), reverse=reverse)
    return d

def default_dict(dic, l):
    """Turns an existing dictionary into a default dictionary with default value l"""
    return defaultdict(lambda: l, dic)

def normalize_dict(dict_, default=False, lmbda=None, return_=True):
    """Normalizes the values of a dictionary"""
    """If default==True, returns a default dictionary with default value lmbda"""
    """If return_==False, modifies the input dictionary without returning anything"""
    if default==True:
        normalized = defaultdict(lambda:lmbda)
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
    
    def to_graph(l):
        G = Graph()
        for part in l:
            # each sublist is a bunch of nodes
            G.add_nodes_from(part)
            # it also imlies a number of edges:
            G.add_edges_from(to_edges(part))
        return G

    def to_edges(l):
        """ 
            treat `l` as a Graph and returns its edges 
            to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
        """
        it = iter(l)
        last = next(it)
    
        for current in it:
            yield last, current
            last = current
    
    G = to_graph(list_of_lists)
    return list(connected_components(G))

def rescale(val, lis, new_min = 0.0, new_max = 1.0):
    """Rescales a value between new_min and new_max according to the values of lis"""
    numerator = new_max - new_min
    old_max, old_min = max(lis), min(lis)
    denominator = old_max - old_min
    part1 = numerator / denominator
    part2 = val - old_max
    part3 = new_max
    return part1 * part2 + part3

def keywithminval(d):
    """Returns the dictionary key with the lowest value"""
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(min(v))]

