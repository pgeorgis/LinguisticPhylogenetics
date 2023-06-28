from collections import defaultdict
import pandas as pd
from math import log, sqrt, e
import re, operator, os
from unidecode import unidecode
from numpy import array, amax, zeros
from statistics import mean, stdev
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, to_tree
from scipy.spatial.distance import squareform
from sklearn import manifold
import seaborn as sns
import networkx as nx

# GENERAL AUXILIARY FUNCTIONS
def dict_tuplelist(dic, sort=True, n=1, reverse=True):
    """Returns a list of (key, value) tuples from the dictionary
    if sort == True, sorts the list by the nth tuple item, by default in decending order"""
    d = [(key, dic[key]) for key in dic]
    if sort:
        d.sort(key=operator.itemgetter(n), reverse=reverse)
    return d

def default_dict(dic, l):
    """Turns an existing dictionary into a default dictionary with default value l"""
    dd = defaultdict(lambda:l)
    for key in dic:
        dd[key] = dic[key]
    return dd

def keywithmaxval(d):
    """Returns the dictionary key with the highest value"""
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(max(v))]

def keywithminval(d):
    """Returns the dictionary key with the lowest value"""
    v = list(d.values())
    k = list(d.keys())
    return k[v.index(min(v))]
    
def chunk_list(lis, n):
    """Splits a list into sublists of length n; if not evenly divisible by n,
    the final sublist contains the remainder"""
    return [lis[i * n:(i + 1) * n] for i in range((len(lis) + n - 1) // n)]

def rescale(val, lis, new_min = 0.0, new_max = 1.0):
    """Rescales a value between new_min and new_max according to the values of lis"""
    numerator = new_max - new_min
    old_max, old_min = max(lis), min(lis)
    denominator = old_max - old_min
    part1 = numerator / denominator
    part2 = val - old_max
    part3 = new_max
    return part1 * part2 + part3


# STRING MANIPULATION
def strip_ch(string, to_remove):
    """Removes a set of characters from strings"""
    return ''.join([ch for ch in string if ch not in to_remove])

def format_as_variable(string):
    variable = unidecode(string)
    variable = re.sub(' ', '', variable)
    variable = re.sub("'", '', variable)
    variable = re.sub('-', '_', variable)
    variable = re.sub('\(', '', variable)
    variable = re.sub('\)', '', variable)
    return variable



# CSV/EXCEL FILE TOOLS
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

def xlsx_to_csv(excel_path, csv_path=None, sheet=None, 
                sep=',', index=None, header=True):
    """Converts an Excel file to a CSV file"""
    if sheet:
        excel_file = pd.read_excel(excel_path, sheet_name=sheet)
    else:
        excel_file = pd.read_excel(excel_path)
    
    # Automatically name the output .csv file the same as the Excel file if 
    # no other name is specified
    if csv_path is None:
        csv_path = excel_path.split('.')[0] + '.csv'
    
    # Write to .csv file
    excel_file.to_csv(csv_path, index=index, header=header, sep=sep)
    print(f'Wrote file to {csv_path}.')
    

# NORMALIZATION
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
   

# INFORMATION CONTENT
def surprisal(p):
    try:
        return -log(p, 2)
    except ValueError:
        raise ValueError(f'Math Domain Error: cannot take the log of {p}')
        
# def adaptation_surprisal(alignment, surprisal_dict, normalize=True):
#     """Calculates the surprisal of an aligned sequence, given a dictionary of 
#         surprisal values for the sequence corresponcences"""
#     values = [surprisal_dict[pair[0]][pair[1]] for pair in alignment]

#     if normalize:
#         return mean(values)
#     else:
#         return sum(values)

def adaptation_surprisal(alignment, surprisal_dict, ngram_size=1, normalize=True):
    """Calculates the surprisal of an aligned sequence, given a dictionary of 
    surprisal values for the sequence corresponcences"""
    
    pad_n = ngram_size - 1
    if ngram_size > 1:
        alignment = [('# ', '# ')]*pad_n + alignment + [('# ', '# ')]*pad_n
    
    values = []
    for i in range(pad_n, len(alignment)-pad_n):
        ngram = alignment[i:i+ngram_size]
        segs = list(zip(*ngram))
        seg1, seg2 = segs
        # forward
        # seg2 = seg2[-1]
        
        # backward
        seg2 = seg2[0]
        values.append(surprisal_dict[seg1][seg2])

    if normalize:
        return mean(values)
    else:
        return values

def entropy(X):
    """X should be a dictionary with absolute counts"""
    total = sum(X.values())
    E = 0
    for i in X:
        p = X[i]/total
        if p > 0:
            E += p * surprisal(p)
    return E


# SMOOTHING
def lidstone_smoothing(x, N, d, alpha=0.3):
    """Given x (unsmoothed counts), N (total observations), 
    and d (number of possible outcomes), returns smoothed Lidstone probability"""
    return (x + alpha) / (N + (alpha*d))


# PLOTTING PAIRWISE SIMILARITY / DISTANCE
def euclidean_dist(dists):
    return sqrt(sum([dist**2 for dist in dists]))

def list_mostsimilar(item1, comp_group, dist_func, n=5, sim=True, return_=False,
                     **kwargs):
    n = min(len(comp_group), n)
    sim_list = [(item2, dist_func(item1, item2, **kwargs)) for item2 in comp_group if item1 != item2]
    sim_list.sort(key=operator.itemgetter(1), reverse=sim)
    if return_:
        return sim_list[:n]
    else:
        for item in sim_list[:n]:
            print(f'{item[0].name}: {round(item[1], 2)}')

def distance_matrix(group, dist_func, sim=False, **kwargs):
    # Initialize nxn distance matrix filled with zeros
    mat = zeros((len(group),len(group)))
    
    # Calculate pairwise distances between items and add to matrix
    for i in range(len(group)):
        for j in range(i+1, len(group)):
            dist = dist_func(group[i], group[j], **kwargs)
            
            # Convert similarities to distances
            if sim:
                dist = 1 - min(1, dist)
                
            mat[i][j] = dist
            mat[j][i] = dist
            
    return mat


def linkage_matrix(group, dist_func, sim=False, 
                   method='average', metric='euclidean',
                   **kwargs):
    """Methods: average, centroid, median, single, complete, ward, weighted
        See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html"""
    mat = distance_matrix(group, dist_func, sim, **kwargs)
    dists = squareform(mat)
    lm = linkage(dists, method, metric)
    return lm


def cluster_items(group, labels,
                  dist_func, sim, cutoff,
                  method = 'average', metric='euclidean',
                  **kwargs):
    lm = linkage_matrix(group, dist_func, sim, method, metric, **kwargs)
    cluster_labels = fcluster(lm, cutoff, 'distance')
    clusters = defaultdict(lambda:[])
    for item, cluster in sorted(zip(labels, cluster_labels), key=lambda x: x[1]):
        # sorting just makes it so that the cluster dictionary will be returned
        # with the clusters in numerical order
        clusters[cluster].append(item)
    
    return clusters 

# def lm2dendrogram(lm, p=30, orientation='left', labels=None):
#     dendrogram(lm, p=p, orientation=orientation, labels=labels)
#     if title:
#         plt.title(title, fontsize=30)
#     plt.savefig(f'{save_directory}{title}.png', bbox_inches='tight', dpi=300)
#     plt.show()
#     if return_newick:
#         return linkage2newick(lm, labels)

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

def dm2coords(dm, dimensions=2):
    """Returns coordinate embeddings of an array of items from their distance matrix"""
    adist = array(dm)
    a_max = amax(adist)
    adist /= a_max
    mds = manifold.MDS(n_components=dimensions, 
                       dissimilarity="precomputed", 
                       random_state=42)
    results = mds.fit(adist)
    coords = results.embedding_
    return coords
    

def plot_distances(group, dist_func=None, sim=False, dimensions=2, labels=None, 
                   title=None, plotsize=None, invert_yaxis=False, invert_xaxis=False,
                   directory='',
                   **kwargs):   
    dm = distance_matrix(group, dist_func, sim, **kwargs)
    coords = dm2coords(dm, dimensions)
    sns.set(font_scale=1.0)
    if plotsize is None:
        x_coords = [coords[i][0] for i in range(len(coords))]
        y_coords = [coords[i][1] for i in range(len(coords))]
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        y_ratio = y_range / x_range
        n = max(10, round((len(group)/10)*2))
        plotsize = (n, n*y_ratio)
    plt.figure(figsize=plotsize)
    plt.scatter(
        coords[:, 0], coords[:, 1], marker = 'o'
        )
    for label, x, y in zip(labels, coords[:, 0], coords[:, 1]):
        plt.annotate(
            label,
            xy = (x, y), xytext = (5, -5),
            textcoords = 'offset points', ha = 'left', va = 'bottom',
            )
    if invert_yaxis:    
        plt.gca().invert_yaxis()
    if invert_xaxis:
        plt.gca().invert_xaxis()
    plt.savefig(f'{directory}{title}.png', bbox_inches='tight', dpi=300)
    plt.show()

"""
from kneed import KneeLocator
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def k_means_cluster(dm, k='elbow', n_init=10, max_iter=300, random_state=42, scaler=True, item_labels=None):
    
    kmeans_kwargs = {"init":"random",
                     "n_init":n_init,
                     "max_iter":max_iter,
                     "random_state":random_state}
    
    # Scale features to have mean=0 and stdev=1
    if scaler:
        scaler = StandardScaler()
        dm = scaler.fit_transform(dm)
    
    # Use elbow method to automatically set value for k if not specified
    if k == 'elbow':
        sse = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(dm)
            sse.append(kmeans.inertia_)
        kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
        k = kl.elbow
    
    # Use silhouette method to automatically set value for k if not specified
    elif k == 'silhouette':
        silhouette_coef = []
        
        # Start at 2 clusters for silhouette coefficient
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(dm)
            score = silhouette_score(dm, kmeans.labels_)
            silhouette_coef.append((k, score))
        
        # Select k with maximum silhouette coefficient
        k = max(silhouette_coef, key=lambda x:x[1])[0]
    
    else:
        assert type(k) == int
    
    # Initialize k-means clustering class
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    
    # Fit kmeans clustering to dm
    kmeans.fit(dm)
    
    # Access cluster assignments
    labels = kmeans.labels_
    
    # If item labels are provided, return dictionary of item labels within each cluster
    if item_labels:
        assert len(item_labels) == len(labels)
        clusters = defaultdict(lambda:[])
        for item, cluster in zip(item_labels, labels):
            clusters[cluster].append(item)
        return clusters
    
    # Otherwise just return the list of cluster labels
    else:
        return labels

def dbscan_cluster(dm, scaler=True, item_labels=None):
    
    if scaler:
        scaler = StandardScaler()
        dm = scaler.fit_transform(dm)
    
    dbscan = DBSCAN(eps=0.3)
    dbscan.fit(dm)
    
    labels = dbscan.labels_
    
    # If item labels are provided, return dictionary of item labels within each cluster
    if item_labels:
        assert len(item_labels) == len(labels)
        clusters = defaultdict(lambda:[])
        for item, cluster in zip(item_labels, labels):
            clusters[cluster].append(item)
        return clusters
    
    # Otherwise just return the list of cluster labels
    else:
        return labels
"""
        


def network_plot(group, labels, 
                 dist_func=None, sim=True,
                 min_edges=None, max_edges=None, threshold=None,
                 method='spring', coordpos=True, dimensions=2, seed=1,
                 edgelabels=False, edge_label_dist=True, 
                 scale_dist=100, edge_decimals=1,
                 scale_nodes=False, node_sizes=None, node_colors=None,
                 invert_yaxis=False, invert_xaxis=False,
                 title=None, save_directory='',
                 **kwargs):

    # warnings.filterwarnings("ignore", category=UserWarning)
    
    # Determine the minimum number of edges per node to display
    # By default, take the square root of total number of network nodes
    # for spring networks, and the total number of network nodes for coordinate networks
    if min_edges is None:
        if method == 'coords':
            min_edges = len(group)
        else:
            min_edges = round(sqrt(len(group)))
            
    # Determine the maximum number of edges to display per node
    # By default, set the maximum to the total number of nodes
    if max_edges is None:
        max_edges = len(group)
    
    # Create dictionary of node indices and their labels
    item_labels = {n:labels[n] for n in range(len(labels))}
    
    # Calculate initial coordinates for nodes from a distance matrix using MDS
    dm = distance_matrix(group, dist_func, sim, **kwargs)
    coords = dm2coords(dm, dimensions)
    coordinates = {}
    for n, x, y in zip(item_labels, coords[:, 0], coords[:, 1]):
        coordinates[n] = (x, y)
        
    # Calculate the threshold for plotting edges: the mean distance among items in the network
    if threshold is None:
        dists = [dm[i][j] for i in range(len(dm)) for j in range(len(dm[i])) if i != j]
        threshold = mean(dists) + stdev(dists)
    
    # Create a figure for the network; larger plot if >30 nodes
    if len(group) >= 30:
        plt.figure(figsize=(15,12))
    else:
        plt.figure(figsize=(10, 8))
    
    # Initialize the network graph
    gr = nx.Graph()
    
    # Iterate through pairs of nodes and adding edges between them
    # For every node, add edges connecting it to the n least distance node (min_edges)
    # And then add more edges up until the maximum number of edges if their distance is lower than the threshold
    # Label edges with distances/similarities
    edge_labels = {}
    item_edges = defaultdict(lambda:0)
    edges = {}
    
    def add_edge(node_i, node_j, dist):
        gr.add_edge(node_i, node_j, distance=dist, weight=(1-dist)**2)
        
        # Label edges with distances; scale and round according to parameter specifications
        if edge_label_dist:
            edge_labels[(node_i, node_j)] = str(round(dist*scale_dist, edge_decimals))
        
        # Label edges with similarities; scale and round according to parameter specifications
        else: 
            edge_labels[(node_i, node_j)] = str(round(dist*scale_dist, edge_decimals))
        

    for i in range(len(dm)):
        i_dists = list(enumerate(dm[i]))
        i_dists.sort(key=lambda x: x[1])
        i_dists = [item for item in i_dists if item[0] != i]
        min_i_dists = i_dists[:min_edges]
        extra_i_dists = i_dists[min_edges:max_edges]
        for j, dist in min_i_dists:
            add_edge(i, j, dist)
        for j, dist in extra_i_dists:
            if dist <= threshold:
                add_edge(i, j, dist)
                
    # Add any nodes which were skipped in the preceding iteration due to not 
    # meeting the similarity threshold with any other node
    for i in range(len(group)):
        if i not in gr.nodes():
            gr.add_node(i)

    # Generate node positions according to method
    # coords: use coordinate positions from MDS
    if method == 'coords':
        pos = coordinates
    
    # spring: use spring layout positions
    elif method == 'spring':
        
        # Initialize either using MDS coordinates or random initialization
        if coordpos==True:
            pos = nx.spring_layout(gr, seed=seed, pos=coordinates)
        else:
            pos = nx.spring_layout(gr, seed=seed)
    
    # Raise error for unrecognized methods
    else:
        raise ValueError(f'Error: Method {method} is not recognized!')
        
    # Get lists of edges and their distances for color-coding 
    edgeList, colorsList = zip(*nx.get_edge_attributes(gr,'distance').items())
    
    # Scale nodes according to specified sizes, if given
    if scale_nodes:
        if node_sizes is None:
            raise ValueError('Error: Provide a list of node sizes in order to scale nodes!')
        node_sizes = [node_sizes[node] for node in gr.nodes()]
        nz_node_sizes = [rescale(i, node_sizes, 200, 2000) for i in node_sizes]
    
    # Otherwise plot all nodes with equal size, either specified through 
    # node_sizes parameter or 300 by default
    else:
        if node_sizes is None:
            nz_node_sizes = [300 for i in range(len(group))]
        else:
            nz_node_sizes = [node_sizes for i in range(len(group))]
    
    # Color all nodes light blue by default if no other color is specified 
    if node_colors is None:
        node_colors = ['# 70B0F0' for i in range(len(group))]
    
    # Draw the network
    nx.draw(gr, pos=pos, edgelist = edgeList, edge_color=colorsList, font_size=8, 
            edge_cmap = plt.cm.hot, vmin = 0, vmax = 1, labels=item_labels,
            node_color=node_colors, font_weight='bold', node_size=nz_node_sizes)
    
    # Add edge labels
    if edgelabels:
        nx.draw_networkx_edge_labels(gr, pos=pos, edge_labels=edge_labels, font_size=8)
    
    # Invert axes
    if invert_yaxis:    
        plt.gca().invert_yaxis()
    if invert_xaxis:
        plt.gca().invert_xaxis()
    
    # Add title to the plot and save 
    if title:
        plt.title(f'{title}: (method = {method}, {dimensions}-D, {min_edges} min edges, {max_edges} max edges, threshold = {round(threshold, 3)})', fontsize=20)
        plt.savefig(f'{save_directory}{title}.png', bbox_inches='tight', dpi=600)
    
    # Show the network plot and then close
    plt.show()
    plt.close()
    

def new_network_plot(group, labels, 
                 dist_func=None, sim=True,
                 cluster_threshold=0.3, 
                 nearest_clusters=2, nearest_items=0.4,
                 method='spring', coordpos=True, dimensions=3, seed=1,
                 edgelabels=False, edge_label_dist=True, 
                 scale_dist=100, edge_decimals=1,
                 scale_nodes=False, node_sizes=None, node_colors=None,
                 invert_yaxis=False, invert_xaxis=False,
                 title=None, save_directory='',
                 **kwargs):

    # warnings.filterwarnings("ignore", category=UserWarning)
    
    # Create dictionary of node indices and their labels
    item_labels = {n:labels[n] for n in range(len(labels))}
    
    # Calculate initial coordinates for nodes from a distance matrix using MDS
    dm = distance_matrix(group, dist_func, sim, **kwargs)
    
    # Distance dictionary
    dist_dict = {}
    for i in range(len(dm)):
        for j in range(len(dm[i])):
            dist_dict[(item_labels[i], item_labels[j])] = dm[i][j]
    
    # Get linkage matrix and hierarchical clustering of items
    dists = squareform(dm)
    lm = linkage(dists, method='average', metric='euclidean')
    cluster_labels = fcluster(lm, cluster_threshold, 'distance')
    clusters = defaultdict(lambda:[])
    for item, cluster in sorted(zip(labels, cluster_labels), key=lambda x: x[1]):
        # sorting just makes it so that the cluster dictionary will be returned
        # with the clusters in numerical order
        clusters[cluster].append(item)
    
    coords = dm2coords(dm, dimensions)
    coordinates = {}
    for n, x, y in zip(item_labels, coords[:, 0], coords[:, 1]):
        coordinates[n] = (x, y)
    
    # Create a figure for the network; larger plot if >30 nodes
    if len(group) >= 30:
        plt.figure(figsize=(15,12))
    else:
        plt.figure(figsize=(10, 8))
    
    # Iterate through pairs of nodes and adding edges between them
    # For every node, add edges connecting it to the n least distance node (min_edges)
    # And then add more edges up until the maximum number of edges if their distance is lower than the threshold
    # Label edges with distances/similarities
    edge_labels = {}
    item_edges = defaultdict(lambda:0)
    edges = {}
    
    # Initialize the network graph
    gr = nx.Graph()
    
    def add_edge(node_i, node_j, dist):
        gr.add_edge(node_i, node_j, distance=dist)
        
        # Label edges with distances; scale and round according to parameter specifications
        if edge_label_dist:
            edge_labels[(node_i, node_j)] = str(round(dist*scale_dist, edge_decimals))
        
        # Label edges with similarities; scale and round according to parameter specifications
        else: 
            edge_labels[(node_i, node_j)] = str(round(dist*scale_dist, edge_decimals))
    
    edges_to_add = []
    cluster_indices = list(clusters.keys())
    for cluster in cluster_indices:
        
        # Add edges between every pair of items within each cluster
        cluster_members = list(clusters[cluster])
        for i in range(len(cluster_members)):
            item_i = cluster_members[i]
            for j in range(i+1, len(cluster_members)):
                item_j = cluster_members[j]
                edges_to_add.append((labels.index(item_i), labels.index(item_j), dist_dict[(item_i, item_j)]))
    
    cluster_iterations = defaultdict(lambda:[])
    iteration = 0
    cluster_iterations[iteration] = clusters
    original_nearest_clusters = nearest_clusters
    nearest_clusters /= 1.5
    while len(cluster_iterations[iteration]) > 1 and (iteration < 10):
        iteration += 1
        # Iterate through pairs of clusters
        cluster_indices = list(cluster_iterations[iteration-1].keys())
        nearest_clusters = min(round(nearest_clusters*1.5), 3)
        for cluster in cluster_indices:
            cluster_members = list(cluster_iterations[iteration-1][cluster])
            cluster_dists = {}
            
            # Measure the distance between every pair of items between the two clusters
            for cluster2 in cluster_indices:
                if cluster2 != cluster:
                    cluster2_members = list(cluster_iterations[iteration-1][cluster2])
                    cluster2dists = {}
                    for i in range(len(cluster_members)):
                        item_i = cluster_members[i]
                        for j in range(len(cluster2_members)):
                            item_j = cluster2_members[j]
                            cluster2dists[(labels.index(item_i), labels.index(item_j))] = dist_dict[(item_i, item_j)]
                    cluster_dists[cluster2] = mean(list(cluster2dists.values())), cluster2dists
            
            # Take the mean distance between all item pairs between the clusters
            mean_cluster_dists = dict_tuplelist({cluster2:cluster_dists[cluster2][0] for cluster2 in cluster_dists})
            
            
            # Find the n closest clusters
            closest_clusters = [item[0] for item in mean_cluster_dists[-nearest_clusters:]]
                
            
            # Retrieve pairwise distances between items in nearest clusters  
            combined = cluster_members[:]
            for cluster2 in closest_clusters:
                distances = dict_tuplelist(cluster_dists[cluster2][1])
                
                # Find the n nearest individual pairs between the clusters (min 1, max 5)
                n_nearest_items = round(min(max(1, nearest_items * len(distances)), 5))
                
                closest_pairs = distances[-n_nearest_items:]
                
                for pair, dist in closest_pairs:
                    i, j = pair
                    edges_to_add.append((i, j, dist)) # sqrt(dist)))# *log((iteration+1), 2))) # sqrt(iteration)))# 
            
                # Update cluster lists
                combined.extend(list(cluster_iterations[iteration-1][cluster2]))
                
            cluster_iterations[iteration].append(combined)
    
        cluster_iterations[iteration] = {index:items for index, items in enumerate(combine_overlapping_lists(cluster_iterations[iteration]))}
        
    gr.add_weighted_edges_from(edges_to_add, weight='distance')
    # for node_i, node_j, dist in edges_to_add:
    #    add_edge(node_i, node_j, dist)
        
    # Add any nodes which were skipped in the preceding iteration due to not 
    # meeting the similarity threshold with any other node
    for i in range(len(group)):
        if i not in gr.nodes():
            gr.add_node(i)

    # Generate node positions according to method
    # coords: use coordinate positions from MDS
    if method == 'coords':
        pos = coordinates
    
    # spring: use spring layout positions
    elif method == 'spring':
        
        # Initialize either using MDS coordinates or random initialization
        if coordpos==True:
            pos = nx.spring_layout(gr, seed=seed, pos=coordinates)
        else:
            pos = nx.spring_layout(gr, seed=seed)
    
    # Raise error for unrecognized methods
    else:
        raise ValueError(f'Error: Method {method} is not recognized!')
        
    # Get lists of edges and their distances for color-coding 
    edgeList, colorsList = zip(*nx.get_edge_attributes(gr,'distance').items())
    
    # Scale nodes according to specified sizes, if given
    if scale_nodes:
        if node_sizes is None:
            raise ValueError('Error: Provide a list of node sizes in order to scale nodes!')
        node_sizes = [node_sizes[node] for node in gr.nodes()]
        nz_node_sizes = [rescale(i, node_sizes, 200, 2000) for i in node_sizes]
    
    # Otherwise plot all nodes with equal size, either specified through 
    # node_sizes parameter or 300 by default
    else:
        if node_sizes is None:
            nz_node_sizes = [300 for i in range(len(group))]
        else:
            nz_node_sizes = [node_sizes for i in range(len(group))]
    
    # Color all nodes light blue by default if no other color is specified 
    if node_colors is None:
        node_colors = ['# 70B0F0' for i in range(len(group))]
    
    # Draw the network
    nx.draw(gr, pos=pos, edgelist = edgeList, edge_color=colorsList, font_size=8, 
            edge_cmap = plt.cm.hot, vmin = 0, vmax = 1, labels=item_labels,
            node_color=node_colors, font_weight='bold', node_size=nz_node_sizes)
    
    # Add edge labels
    if edgelabels:
        nx.draw_networkx_edge_labels(gr, pos=pos, edge_labels=edge_labels, font_size=8)
    
    # Invert axes
    if invert_yaxis:    
        plt.gca().invert_yaxis()
    if invert_xaxis:
        plt.gca().invert_xaxis()
    
    # Add title to the plot and save 
    if title:
        plt.title(f'{title}: (method = {method}, {dimensions}-D, cluster_threshold={cluster_threshold}, items={nearest_items}, clusters={original_nearest_clusters}', fontsize=20)
        plt.savefig(f'{save_directory}{title}.png', bbox_inches='tight', dpi=600)
    
    # Show the network plot and then close
    plt.show()
    plt.close()


def newer_network_plot(group, labels, 
                 dist_func=None, sim=True,
                 coordpos=True, dimensions=3, seed=1,
                 step_size=0.05, connection_decay=0.5,
                 edgelabels=False, edge_label_dist=True,
                 scale_dist_func=sqrt,
                 scale_dist=100, edge_decimals=1,
                 scale_nodes=False, node_sizes=None, node_colors=None,
                 invert_yaxis=False, invert_xaxis=False,
                 title=None, save_directory='',
                 **kwargs):

    # warnings.filterwarnings("ignore", category=UserWarning)
    
    # Create dictionary of node indices and their labels
    item_labels = {n:labels[n] for n in range(len(labels))}
    
    # Calculate initial coordinates for nodes from a distance matrix using MDS
    dm = distance_matrix(group, dist_func, sim, **kwargs)
    
    # Distance dictionary
    dist_dict = {}
    for i in range(len(dm)):
        for j in range(len(dm[i])):
            dist_dict[(item_labels[i], item_labels[j])] = dm[i][j]
    
    # Get linkage matrix and hierarchical clustering of items
    dists = squareform(dm)
    lm = linkage(dists, method='average', metric='euclidean')
    
    def get_clusters(lm, cutoff):
        cluster_labels = fcluster(lm, cutoff, 'distance')
        clusters = defaultdict(lambda:[])
        for item, cluster in sorted(zip(labels, cluster_labels), key=lambda x: x[1]):
            # sorting just makes it so that the cluster dictionary will be returned
            # with the clusters in numerical order
            clusters[cluster].append(item)
        return clusters
    
    coords = dm2coords(dm, dimensions)
    coordinates = {}
    for n, x, y in zip(item_labels, coords[:, 0], coords[:, 1]):
        coordinates[n] = (x, y)
    
    # Create a figure for the network; larger plot if >30 nodes
    if len(group) >= 30:
        plt.figure(figsize=(15,12))
    else:
        plt.figure(figsize=(10, 8))
    
    # Iterate through pairs of nodes and adding edges between them
    # For every node, add edges connecting it to the n least distant nodes (min_edges)
    # And then add more edges up until the maximum number of edges if their distance is lower than the threshold
    # Label edges with distances/similarities
    edge_labels = {}
    item_edges = defaultdict(lambda:0)
    edges = {}
    
    # Initialize the network graph
    gr = nx.Graph()
    
    def add_edge(node_i, node_j, dist):
        gr.add_edge(node_i, node_j, distance=dist, weight=(1-dist)**2)# ((e**-dist))**2)
        
        # Label edges with distances; scale and round according to parameter specifications
        if edge_label_dist:
            edge_labels[(node_i, node_j)] = str(round(dist*scale_dist, edge_decimals))
        
        # Label edges with similarities; scale and round according to parameter specifications
        else: 
            edge_labels[(node_i, node_j)] = str(round(dist*scale_dist, edge_decimals))
    
    edges_to_add = {}
    already_clustered = []
    
    def connect_clusters(cluster_dict, connect_proportion=1, scale_distance=1):
        
        # Iterate through clusters in cluster dictionary
        for cluster in cluster_dict:
            # Get list of cluster members
            cluster_members = list(cluster_dict[cluster])
            
            # Connect nodes if there are at least 2 items in the cluster
            if len(cluster_members) > 1:
                
                # Get dictionary with pairwise distances between each pair of nodes within the cluster
                cluster_dists = {(cluster_members[i], cluster_members[j]):dist_dict[(cluster_members[i], cluster_members[j])] 
                                 for i in range(len(cluster_members)) 
                                 for j in range(i+1, len(cluster_members))}
                
                # Filter out node pairs which are already connected to the cluster
                filtered_cluster_dists = {}
                for node_pair in cluster_dists:
                    node_i, node_j = node_pair
                    index_i, index_j = labels.index(node_i), labels.index(node_j)
                    if (index_i, index_j) not in already_clustered:
                        filtered_cluster_dists[(index_i, index_j)] = cluster_dists[node_pair]
                
                # Rank each new pair by distance in ascending order
                ranked_cluster_dists = dict_tuplelist(filtered_cluster_dists, reverse=False)
                
                # Select only the nearest n% of nodes to connect
                n = max(round(len(ranked_cluster_dists) * connect_proportion), 1)
                to_connect = ranked_cluster_dists[:n]
                
                # Scale distances and and add edges for this selection
                for item in to_connect:
                    node_pair, dist = item
                    node_i, node_j = node_pair
                    edges_to_add[node_pair] = dist * scale_distance
                
                # Ensure that each member of the cluster is minimally connected 
                # to its nearest neighbor within the cluster
                for i in range(len(cluster_members)):
                    member = cluster_members[i]
                    neighbor_dists = {}
                    for j in range(len(cluster_members)):
                        neighbor = cluster_members[j]
                        if member != neighbor:
                            try:
                                dist = cluster_dists[(member, neighbor)]
                            except KeyError:
                                dist = cluster_dists[(neighbor, member)]
                            neighbor_dists[neighbor] = dist
                    nearest_neighbor = keywithminval(neighbor_dists)
                    node_i, node_j = labels.index(member), labels.index(nearest_neighbor)
                    if (node_i, node_j) not in edges_to_add:
                        if (node_j, node_i) not in edges_to_add:
                            edges_to_add[(node_i, node_j)] = neighbor_dists[nearest_neighbor] * scale_distance
                    
                                
                # Add all clustered nodes to index
                already_clustered.extend([(labels.index(node_i), labels.index(node_j)) for node_i, node_j in cluster_dists.keys() 
                                          if (node_i, node_j) not in already_clustered])
        
    # Get initial clusters
    cluster_threshold = 0
    while len(get_clusters(lm, cutoff=cluster_threshold)) == len(labels):
        cluster_threshold += 0.01
    initial_clusters = get_clusters(lm, cutoff=cluster_threshold)
    
    # Fully connect initial clusters without any distance scaling
    connect_proportion = 1
    connect_clusters(initial_clusters, 
                     connect_proportion=connect_proportion, scale_distance=1)
    
    cluster_iterations = defaultdict(lambda:[])
    iteration = 0
    cluster_iterations[iteration] = initial_clusters
    while len(cluster_iterations[max(cluster_iterations.keys())]) > 1:
        iteration += 1
        cluster_threshold += step_size
        connect_proportion *= connection_decay
        iteration_clusters = get_clusters(lm, cutoff=cluster_threshold)
        if len(iteration_clusters) < len(cluster_iterations[max(cluster_iterations.keys())]):
            # connection_proportion = 1/(iteration**2)
            # scale_distance = sqrt(iteration) # log((iteration+1), 2) # iteration
            scale_distance = scale_dist_func(iteration)
            connect_clusters(iteration_clusters,
                             connect_proportion=connect_proportion, 
                             scale_distance=scale_distance)
            cluster_iterations[iteration] = iteration_clusters
            
    for node_pair in edges_to_add:
        node_i, node_j = node_pair
        dist = edges_to_add[node_pair]
        add_edge(node_i, node_j, dist)
        
    # Add any nodes which were skipped in the preceding iteration due to not 
    # meeting the similarity threshold with any other node
    for i in range(len(group)):
        if i not in gr.nodes():
            gr.add_node(i)

    
    # spring: use spring layout positions
    # Initialize either using MDS coordinates or random initialization
    if coordpos==True:
        pos = nx.spring_layout(gr, seed=seed, pos=coordinates)
    else:
        pos = nx.spring_layout(gr, seed=seed)
        
    # Get lists of edges and their distances for color-coding 
    edgeList, colorsList = zip(*nx.get_edge_attributes(gr,'distance').items())
    
    # Scale nodes according to specified sizes, if given
    if scale_nodes:
        if node_sizes is None:
            raise ValueError('Error: Provide a list of node sizes in order to scale nodes!')
        node_sizes = [node_sizes[node] for node in gr.nodes()]
        nz_node_sizes = [rescale(i, node_sizes, 200, 2000) for i in node_sizes]
    
    # Otherwise plot all nodes with equal size, either specified through 
    # node_sizes parameter or 300 by default
    else:
        if node_sizes is None:
            nz_node_sizes = [300 for i in range(len(group))]
        else:
            nz_node_sizes = [node_sizes for i in range(len(group))]
    
    # Color all nodes light blue by default if no other color is specified 
    if node_colors is None:
        node_colors = ['# 70B0F0' for i in range(len(group))]
    
    # Draw the network
    nx.draw(gr, pos=pos, edgelist = edgeList, edge_color=colorsList, font_size=8, 
            edge_cmap = plt.cm.hot, vmin = 0, vmax = 1, labels=item_labels,
            node_color=node_colors, font_weight='bold', node_size=nz_node_sizes)
    
    # Add edge labels
    if edgelabels:
        nx.draw_networkx_edge_labels(gr, pos=pos, edge_labels=edge_labels, font_size=8)
    
    # Invert axes
    if invert_yaxis:    
        plt.gca().invert_yaxis()
    if invert_xaxis:
        plt.gca().invert_xaxis()
    
    # Add title to the plot and save 
    if title:
        plt.title(f'{title}: step_size={step_size}, connection_decay={connection_decay}', fontsize=20)
        plt.savefig(f'{save_directory}{title}.png', bbox_inches='tight', dpi=600)
    
    # Show the network plot and then close
    plt.show()
    plt.close()


def k_means_network(dm, labels, 
                 dist_func=None, sim=True,
                 edgelabels=False, edge_label_dist=True,
                 edge_decimals=1,
                 scale_nodes=False, node_sizes=None, node_colors=None,
                 invert_yaxis=False, invert_xaxis=False,
                 title=None, save_directory='',
                 scale_func=lambda x:x+0.5,
                 **kwargs):
    
    # Create dictionary of node indices and their labels
    item_labels = {n:labels[n] for n in range(len(labels))}
    assert len(item_labels) == len(dm)
    
    # Calculate initial distance matrix
    # dm = distance_matrix(group, dist_func, sim, **kwargs)

    # Given n items in the group, perform k means clustering n-2 times
    assert len(labels) > 2
    k_clusters = {}
    for k in range(2,len(group)):
        k_clusters[k] = k_means_cluster(dm, k=k, item_labels=item_labels.keys())
    
    # Initialize the network graph
    gr = nx.Graph()
    edge_labels = {}

    def add_edge(node_i, node_j, dist, weight=1):
        gr.add_edge(node_i, node_j, distance=dist, weight=1)
        edge_labels[(node_i, node_j)] = str(round(dist, edge_decimals))

    # Iterate from the largest number of clusters to the smallest
    # On each iteration, connect all previously unconnected nodes within the new non-singleton clusters
    # Scale/weight the distance down on each iteration
    # Logic: weight clusters more heavily that emerge even when k approaches n
    weight = 1
    for k in range(len(group)-1,1,-1):
        non_singletons = [cluster for cluster in k_clusters[k] if len(k_clusters[k][cluster]) > 1]
        for cluster in non_singletons:
            cluster = k_clusters[k][cluster]
            for i in range(len(cluster)):
                item_i = cluster[i]
                for j in range(i+1,len(cluster)):
                    item_j = cluster[j]
                    if (item_i, item_j) not in edge_labels and (item_j, item_i) not in edge_labels:
                        dist = dm[item_i][item_j]
                        # dist *= scale
                        add_edge(item_i, item_j, dist, weight=weight)
        # scale = scale_func(scale)
        weight = scale_func(weight)
    
    # Add any nodes which were skipped in the preceding iteration sequence due to 
    # never clustering with any other node
    for i in range(len(group)):
        if i not in gr.nodes():
            gr.add_node(i)
            
    # Get lists of edges and their distances for color-coding 
    edgeList, colorsList = zip(*nx.get_edge_attributes(gr,'distance').items())


    # Draw the network
    nx.draw(gr, 
            # pos=pos, 
            edgelist = edgeList, 
            # edge_color=colorsList, 
            font_size=8, 
            # edge_cmap = plt.cm.hot, vmin = 0, vmax = 1, 
            labels=item_labels,
            # node_color=node_colors, 
            font_weight='bold', # node_size=nz_node_sizes
            )
    
    



    


def combine_overlapping_lists(list_of_lists):
    # https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements
    import networkx
    from networkx.algorithms.components.connected import connected_components
    
    def to_graph(l):
        G = networkx.Graph()
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
