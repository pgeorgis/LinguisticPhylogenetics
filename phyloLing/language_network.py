import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from collections import defaultdict

def load_language_similarities(tsv_file_path):
    # Initialize the nested dictionary to store similarities
    sim_dict = {}

    # Open and read the TSV file
    with open(tsv_file_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        # Skip the header
        next(reader)

        for row in reader:
            # Extract the two languages and their similarity measurement
            lang1, lang2, measurement = row[0], row[1], float(row[2])

            # If lang1 is not already in the dictionary, add it
            if lang1 not in sim_dict:
                sim_dict[lang1] = {}
            # If lang2 is not already in the dictionary, add it
            if lang2 not in sim_dict:
                sim_dict[lang2] = {}

            # Set the similarity measurement for both directions
            sim_dict[lang1][lang2] = measurement
            sim_dict[lang2][lang1] = measurement

    return sim_dict


def compute_average_distances(sim_dict, n_iterations=50, n_components=3):
    # Number of languages
    languages = list(sim_dict.keys())
    num_languages = len(languages)

    # Initialize an empty array to accumulate distances
    distance_sums = np.zeros((num_languages, num_languages))

    for _ in range(n_iterations):
        # Initialize MDS with 3 components (3D)
        mds = MDS(n_components=n_components, dissimilarity="precomputed", random_state=42+n_iterations, normalized_stress=False)

        # Convert similarity dict to a distance matrix with accentuated distances
        distance_matrix = np.zeros((num_languages, num_languages))
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                if i == j:
                    distance_matrix[i][j] = 0
                else:
                    distance_matrix[i][j] = (1 - sim_dict[lang1][lang2]) ** n_components

        # Apply MDS to reduce dimensions to 3D
        pos_temp = mds.fit_transform(distance_matrix)

        # Calculate and accumulate distances from the current embedding
        for i in range(num_languages):
            for j in range(num_languages):
                if i != j:
                    distance_sums[i][j] += np.linalg.norm(pos_temp[i] - pos_temp[j])

    # Average the distances
    distance_averages = distance_sums / n_iterations

    # Initialize MDS with 3 components (3D) using the averaged distances
    mds_final = MDS(n_components=n_components, dissimilarity="precomputed", random_state=42, normalized_stress=False)
    pos_final = mds_final.fit_transform(distance_averages)

    return pos_final

def plot_language_similarity(sim_dict,
                             min_connections=1,
                             max_connections=None,
                             min_similarity=0,
                             normalize_color=True,
                             is_3d=False,
                             apply_pythagorean=False,
                             n_iterations=50,
                             save_path=None,
                             ):
    # Number of languages
    languages = list(sim_dict.keys())
    num_languages = len(languages)

    # Compute the final positions by averaging distances over multiple MDS runs
    if is_3d:
        n_components = 3
        pos = compute_average_distances(sim_dict, n_iterations=n_iterations, n_components=3)
    else:
        n_components = 2
        # Handle 2D case as before
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress=False)
        distance_matrix = np.zeros((num_languages, num_languages))
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                if i == j:
                    distance_matrix[i][j] = 0
                else:
                    distance_matrix[i][j] = (1 - sim_dict[lang1][lang2]) ** 2
        pos = mds.fit_transform(distance_matrix)

    # Create a NetworkX graph
    G = nx.Graph()

    # Add nodes to the graph
    for i, lang in enumerate(languages):
        G.add_node(lang, pos=pos[i])

    # Create edges with similarity as weights
    edges = []
    for i, lang1 in enumerate(languages):
        for j, lang2 in enumerate(languages):
            if i != j and sim_dict[lang1][lang2] >= min_similarity:
                edges.append((lang1, lang2, sim_dict[lang1][lang2]))

    # Sort edges by similarity (descending)
    edges = sorted(edges, key=lambda x: x[2], reverse=True)

    # Add edges while respecting max_connections and min_connections
    connections = defaultdict(int)
    for edge in edges:
        lang1, lang2, similarity = edge
        if connections[lang1] < max_connections and connections[lang2] < max_connections:
            G.add_edge(lang1, lang2, weight=similarity)
            connections[lang1] += 1
            connections[lang2] += 1

    # Ensure that all nodes have at least min_connections
    for lang in languages:
        if connections[lang] < min_connections:
            potential_edges = [e for e in edges if (e[0] == lang or e[1] == lang) and G.has_edge(*e[:2]) == False]
            potential_edges = sorted(potential_edges, key=lambda x: x[2], reverse=True)
            for edge in potential_edges:
                lang1, lang2, similarity = edge
                if connections[lang1] < max_connections and connections[lang2] < max_connections:
                    G.add_edge(lang1, lang2, weight=similarity)
                    connections[lang1] += 1
                    connections[lang2] += 1
                if connections[lang] >= min_connections:
                    break

    # Apply Pythagorean adjustment
    if apply_pythagorean:
        for lang1 in languages:
            neighbors = list(G.neighbors(lang1))
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    lang2 = neighbors[i]
                    lang3 = neighbors[j]
                    if G.has_edge(lang2, lang3):
                        d1 = G[lang1][lang2]['weight']
                        d2 = G[lang1][lang3]['weight']
                        d3 = G[lang2][lang3]['weight']

                        # Sort distances to identify the two shortest and one longest
                        distances = sorted([(d1, lang1, lang2), (d2, lang1, lang3), (d3, lang2, lang3)], key=lambda x: x[0])

                        # Calculate the new distance using the Pythagorean theorem
                        new_distance = np.sqrt(distances[0][0]**2 + distances[1][0]**2)

                        # Check if the new distance is different from the original
                        if new_distance != distances[2][0]:
                            # Print statement to indicate a change
                            print(f"Changing distance between {distances[2][1]} and {distances[2][2]} from {distances[2][0]} to {new_distance:.6f}")

                        # Update the longest edge with the new distance
                        G[distances[2][1]][distances[2][2]]['weight'] = new_distance


    # Extract positions of nodes for plotting
    pos = nx.get_node_attributes(G, 'pos')

    # Get edge weights for coloring
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())

    # Normalize weights if required
    if normalize_color:
        weights = np.array(weights)
        weights = (weights - weights.min()) / (weights.max() - weights.min())

    # Draw the graph in 3D
    if is_3d:
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection='3d')

        # Extract 3D positions
        pos_array = np.array(list(pos.values()))

        # Draw the nodes
        ax.scatter(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], c='lightgreen', s=500)

        # Draw the labels
        for lang, (x, y, z) in pos.items():
            ax.text(x, y, z, lang, fontsize=10, fontweight='bold')

        # Draw edges
        for (lang1, lang2), weight in zip(edges, weights):
            x = [pos[lang1][0], pos[lang2][0]]
            y = [pos[lang1][1], pos[lang2][1]]
            z = [pos[lang1][2], pos[lang2][2]]
            ax.plot(x, y, z, color=plt.cm.Blues(weight), lw=2)
    else:
        # Draw the graph in 2D
        plt.figure(figsize=(20, 20))
        nx.draw_networkx_nodes(G, pos, node_color='lightgreen', node_size=500)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

        # Draw edges separately to control edge color and weights
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=weights, width=2, edge_cmap=plt.cm.Blues)

    # Save the plot to a file if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)

    # Show the plot
    plt.show()

# Example usage:
tsv_file_path = "/Users/philip.georgis/Documents/Projects/LinguisticPhylogenetics.nosync/datasets/Romance/dist_matrices/2024-08-11_11-15-58_scored.tsv"
sim_dict = load_language_similarities(tsv_file_path)
plot_language_similarity(sim_dict, 
                         apply_pythagorean=False,
                         min_connections=5,
                         max_connections=10,
                         is_3d=True,
                         save_path='romance_test.png'
                         )
