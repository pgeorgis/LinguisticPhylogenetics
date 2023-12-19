from collections import defaultdict
from matplotlib import pyplot as plt
import re
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, to_tree
from scipy.spatial.distance import squareform
from utils.distance import distance_matrix

def linkage_matrix(group, dist_func, sim=False, 
                   method='average', metric='euclidean',
                   **kwargs):
    """Methods: average, centroid, median, single, complete, ward, weighted
        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html"""
    mat = distance_matrix(group, dist_func, sim, **kwargs)
    dists = squareform(mat)
    lm = linkage(dists, method, metric)
    return lm

def cluster_items(group, 
                  dist_func, 
                  sim, 
                  cutoff,
                  labels=None,
                  method='average', 
                  metric='euclidean',
                  return_labels=False,
                  **kwargs):
    lm = linkage_matrix(group, dist_func, sim, method, metric, **kwargs)
    cluster_labels = fcluster(lm, cutoff, 'distance')
    targets = labels if labels else group
    clusters = defaultdict(lambda:[])
    for item, cluster in sorted(zip(targets, cluster_labels), key=lambda x: x[1]):
        # sorting just makes it so that the cluster dictionary will be returned
        # with the clusters in numerical order
        clusters[cluster].append(item)
    
    return clusters

def draw_dendrogram(group, dist_func, title=None, sim=False, labels=None, 
                    p=30, method='average', metric='euclidean',
                    orientation='left', 
                    save_directory='',
                    return_newick=False,
                    **kwargs):
    sns.set(font_scale=1.0)
    if len(group) >= 100:
        plt.figure(figsize=(20,20))
    elif len(group) >= 60:
        plt.figure(figsize=(10,10))
    else:
        plt.figure(figsize=(10,8))
    lm = linkage_matrix(group, dist_func, sim, method, metric, **kwargs)
    dendrogram(lm, p=p, orientation=orientation, labels=labels)
    if title:
        plt.title(title, fontsize=30)
    plt.savefig(f'{save_directory}{title}.png', bbox_inches='tight', dpi=300)
    plt.show()
    if return_newick:
        return linkage2newick(lm, labels)

def getNewick(node, newick, parentdist, leaf_names):
    # source: https://stackoverflow.com/questions/28222179/save-dendrogram-to-newick-format
    if node.is_leaf():
        return "%s:%.2f%s" % (leaf_names[node.id], parentdist - node.dist, newick)
    else:
        if len(newick) > 0:
            newick = "):%.2f%s" % (parentdist - node.dist, newick)
        else:
            newick = ");"
        newick = getNewick(node.get_left(), newick, node.dist, leaf_names)
        newick = getNewick(node.get_right(), ",%s" % (newick), node.dist, leaf_names)
        newick = "(%s" % (newick)
        return newick

def linkage2newick(linkage_matrix, leaf_labels):
    # Convert parentheses in labels to brackets, as parentheses are part of Newick format
    for i in range(len(leaf_labels)):
        leaf_labels[i] = re.sub("\(", "{", leaf_labels[i])
        leaf_labels[i] = re.sub("\)", "}", leaf_labels[i])
    
    tree = to_tree(linkage_matrix, False)
    return getNewick(tree, "", tree.dist, leaf_labels)