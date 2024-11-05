import operator
from math import exp, log, sqrt

from numpy import zeros


# PAIRWISE SIMILARITY / DISTANCE
class Distance:
    def __init__(self, func, cluster_threshold=0.5, sim=False, name=None, alpha=0.5, **kwargs):
        self.func = func
        self.kwargs = kwargs
        self.sim = sim
        self.alpha = alpha
        self.cluster_threshold = cluster_threshold
        self.name = name if name else self.func.__name__
        self.measured = {}
        self.hashable_kwargs = self.get_hashable_kwargs(self.kwargs)

    def set(self, param, value):
        self.kwargs[param] = value
        self.hashable_kwargs = self.get_hashable_kwargs(self.kwargs)

    # TODO possibly use lru_cache instead
    def eval(self, x, y, **kwargs):
        if (x, y, self.hashable_kwargs) in self.measured:
            return self.measured[(x, y, self.hashable_kwargs)]
        else:
            for arg, val in kwargs.items():
                self.set(arg, val)
            result = self.func(x, y, **self.kwargs)
            self.measured[(x, y, self.hashable_kwargs)] = result
            return result

    def to_similarity(self, name=None):
        if self.sim is False:
            def sim_func(x, y, **kwargs):
                # func=lambda x, y: 1/(1+self.func(x, y, **self.kwargs)),
                # TODO make this conversion option possible via method argument
                return dist_to_sim(self.func(x, y, **kwargs))

            if name is None:
                name = self.name + '_asSimilarity'

            return Distance(
                func=sim_func,
                cluster_threshold=self.cluster_threshold,
                sim=True,
                name=name,
                **self.kwargs)

        else:
            return self

    def to_distance(self, name=None, alpha=None):
        if alpha is None:
            alpha = self.alpha
        if self.sim is False:
            return self
        else:
            def dist_func(x, y, **kwargs):
                return sim_to_dist(self.func(x, y, **kwargs), alpha=alpha)

            if name is None:
                name = self.name + '_asDistance'

            return Distance(
                func=dist_func,
                cluster_threshold=self.cluster_threshold,
                sim=False,
                name=name,
                **self.kwargs
            )

    def get_hashable_kwargs(self, kwargs):
        hashable = []
        for key, value in kwargs.items():
            # Recursively convert nested dictionaries
            if isinstance(value, dict):
                value = self.get_hashable_kwargs(value)

            # Convert lists to tuples
            elif isinstance(value, list):
                value = tuple(value)

            # Get just name of other Distance functions as kwargs
            elif isinstance(value, Distance):
                value = value.name

            hashable.append((key, value))

        return tuple(sorted(hashable))


def dist_to_sim(distance):
    return exp(-distance)


def sim_to_dist(similarity, alpha):
    return exp(-max(similarity, 0)**alpha)


def euclidean_dist(dists):
    return sqrt(sum([dist**2 for dist in dists]))


def list_mostsimilar(item1,
                     comp_group,
                     dist_func,
                     n=5,
                     sim=True,
                     return_=False,
                     **kwargs):
    n = min(len(comp_group), n)
    sim_list = [(item2, dist_func(item1, item2, **kwargs))
                for item2 in comp_group if item1 != item2]
    sim_list.sort(key=operator.itemgetter(1), reverse=sim)
    if return_:
        return sim_list[:n]
    else:
        for item in sim_list[:n]:
            print(f'{item[0].name}: {round(item[1], 2)}')


def distance_matrix(group, dist_func, scalar=1, **kwargs):
    if not isinstance(dist_func, Distance):
        raise TypeError(f'dist_func expected to be Distance class object, found {type(dist_func)}')

    # Initialize nxn distance matrix filled with zeros
    mat = zeros((len(group), len(group)))

    # Calculate pairwise distances between items and add to matrix
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            dist = dist_func.eval(group[i], group[j], **kwargs)

            # Convert similarities to distances
            if dist_func.sim:
                dist = sim_to_dist(dist, dist_func.alpha)

            mat[i][j] = dist
            mat[j][i] = dist

    # Scale matrix to accentuate differences
    if scalar > 1:
        mat = mat ** scalar

    return mat
