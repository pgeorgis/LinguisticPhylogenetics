import dendropy
from ete3 import Tree
import itertools
import os
import re
import subprocess
from utils.utils import csv2dict


def load_newick_tree(file_path, as_string=True):
    """
    Loads a tree from a Newick file.

    Parameters:
    file_path (str): Path to the Newick file.

    Returns:
    dendropy.Tree | str: A dendropy Tree object or Newick string representing the tree.
    """
    tree = dendropy.Tree.get(path=file_path, schema="newick")
    if as_string:
        tree.as_string("newick").strip()
    return tree


def postprocess_newick(newick_tree):
    # Fix formatting of Newick string
    newick_tree = newick_tree.strip()
    newick_tree = re.sub(r'\s+', '_', newick_tree)
    newick_tree = re.sub(r',_', ',', newick_tree)
    return newick_tree


def list_tips(tree):
    """
    Lists all the tip names in a dendropy Tree object.

    Parameters:
    tree (dendropy.Tree): The tree from which to list tip names.

    Returns:
    list of str: A list containing the names of all tips (leaf nodes) in the tree.
    """
    if isinstance(tree, str):
        tree = dendropy.Tree.get(data=tree, schema="newick")

    # Extract tip names from the tree's leaf nodes
    tip_names = [leaf.taxon.label for leaf in tree.leaf_nodes() if leaf.taxon]
    return tip_names


def reroot_tree(newick_str: str, outgroup: str | tuple) -> str:
    """
    Reroots the tree at the specified tip or clade outgroup.

    Parameters:
    newick_str (str): The input Newick tree string.
    outgroup (str or tuple or list): A single tip name or a tuple/list of tip names defining the outgroup clade.

    Returns:
    str: The Newick string of the rerooted tree.
    """
    # Parse the Newick string into a tree
    tree = Tree(newick_str, format=1)

    # Determine whether we're rerooting at a single tip or a clade
    if isinstance(outgroup, (tuple, list)):
        # Find the most recent common ancestor of the clade
        nodes = [tree.search_nodes(name=postprocess_newick(tip))[0] for tip in outgroup]
        mrca = tree.get_common_ancestor(nodes)
        tree.set_outgroup(mrca)
    else:
        # Reroot at a single tip
        target_node = tree.search_nodes(name=postprocess_newick(outgroup))
        if target_node:
            tree.set_outgroup(target_node[0])
        else:
            raise ValueError(f"Tip '{outgroup}' not found in the tree")

    return tree.write(format=1)


def flip_clades(newick_str, clade1_tips, clade2_tips, reroot_at=None):
    """
    Flips the order of two clades in a Newick string phylogenetic tree.

    Parameters:
    newick_str (str): The input Newick tree string.
    clade1_tips (str or tuple): A single tip name or a tuple of tip names defining the first clade or tip.
    clade2_tips (str or tuple): A single tip name or a tuple of tip names defining the second clade or tip.
    reroot_at (str, optional): The name of a tip to reroot the tree at before flipping clades. Defaults to None.

    Returns:
    str: The Newick string with the two clades flipped.
    """
    # Parse the Newick string into a tree
    tree = Tree(newick_str, format=1)

    # Optionally reroot the tree at the specified tip
    if reroot_at:
        target_node = tree.search_nodes(name=reroot_at)
        if target_node:
            tree.set_outgroup(target_node[0])
        else:
            raise ValueError(f"Tip '{reroot_at}' not found in the tree")

    # Determine if each input is a single tip or a clade, and get the corresponding node
    # Find the common ancestor of each clade
    def get_node(tips):
        if isinstance(tips, str):
            node = tree.search_nodes(name=tips)
            if not node:
                raise ValueError(f"Tip '{tips}' not found in the tree")
            return node[0]
        elif isinstance(tips, tuple):
            return tree.get_common_ancestor(tips)
        else:
            raise TypeError("Tips should be a string or a tuple of strings")

    clade1_node = get_node(clade1_tips)
    clade2_node = get_node(clade2_tips)

    # Swap the positions of the two clades
    clade1_parent = clade1_node.up
    clade2_parent = clade2_node.up

    # If clades are from the same parent, just swap positions within the parent's children
    if clade1_parent == clade2_parent:
        children = clade1_parent.children
        index1 = children.index(clade1_node)
        index2 = children.index(clade2_node)
        children[index1], children[index2] = children[index2], children[index1]
    else:
        # If clades have different parents, we move them by removing and reattaching
        clade1_node.detach()
        clade2_node.detach()
        clade1_parent.add_child(clade2_node)
        clade2_parent.add_child(clade1_node)

    # Return the modified Newick string
    return tree.write(format=1)


def drop_tips(tree, tip_names):
    """
    Drops specified tips from a dendropy Tree object based on their names.

    Parameters:
    tree (dendropy.Tree | str): The tree from which to drop tips.
    tip_names (list of str): A list of tip names to be removed from the tree.

    Returns:
    dendropy.Tree: The modified tree with specified tips removed.
    """
    if isinstance(tree, str):
        tree = dendropy.Tree.get(data=tree, schema="newick")

    for tip_name in tip_names:
        taxon = tree.find_node_with_taxon_label(tip_name)
        if taxon:
            tree.prune_subtree(taxon)

    return tree


def build_tree_dict(classification_data: dict, clade_sep=" > ") -> dict:
    """Builds a nested dictionary tree from classification string associated with each doculect, e.g.
    'Indo-European > Germanic > West Germanic > Anglo-Frisian > English'
    """
    tree = {}

    for doculect, classification in classification_data.items():
        doculect = postprocess_newick(doculect)
        levels = classification.split(clade_sep)
        current = tree
        for level in levels:
            level = postprocess_newick(level)
            current = current.setdefault(level, {})
        current[doculect] = {}  # Add doculect as a leaf

    return tree


def treedict_to_newick(tree, final=True):
    """Recursively converts a dictionary tree structure into Newick format."""
    if not tree:
        return ""

    subtrees = []
    for key, subtree in tree.items():
        if subtree:
            subtrees.append(f"({treedict_to_newick(subtree, final=False)}){key}")
        else:
            subtrees.append(key)

    if final:
        return ",".join(subtrees) + ";"
    return ",".join(subtrees)


def classification_to_newick(infile: str,
                             sep: str=",",
                             doculect_column: str="Doculect",
                             classification_column: str="Classification",
                             **kwargs) -> str:
    """Loads classification data from a CSV file and generates a corresponding Newick tree string."""
    data = csv2dict(infile, sep=sep, **kwargs)
    classification_dict = {
        row[doculect_column]: row[classification_column]
        for _, row in data.items()
    }
    tree_dict = build_tree_dict(classification_dict)
    newick_tree = treedict_to_newick(tree_dict)
    return newick_tree


def prep_trees_for_comparison(tree1, tree2):
    # Initialize a shared TaxonNamespace
    taxon_namespace = dendropy.TaxonNamespace()

    # Load trees from Newick strings with the shared TaxonNamespace
    if isinstance(tree1, str):
        tree1 = dendropy.Tree.get(data=tree1, schema="newick", taxon_namespace=taxon_namespace)
    if isinstance(tree2, str):
        tree2 = dendropy.Tree.get(data=tree2, schema="newick", taxon_namespace=taxon_namespace)

    # Get list of tips missing from either tree
    missing_tips = set(list_tips(tree1)) - set(list_tips(tree2))
    missing_tips.update(set(list_tips(tree2)) - set(list_tips(tree1)))

    # Drop missing tips from each tree
    tree1 = drop_tips(tree1, missing_tips)
    tree2 = drop_tips(tree2, missing_tips)

    # Ensure both trees reference the same TaxonNamespace after modifications
    tree1.migrate_taxon_namespace(taxon_namespace)
    tree2.migrate_taxon_namespace(taxon_namespace)

    return tree1, tree2


def robinson_foulds(tree1, tree2):
    """Computes unweighted Robinson-Foulds distance between two trees (Newick strings)."""

    # Preprocess trees to share same taxa and namespace
    tree1, tree2 = prep_trees_for_comparison(tree1, tree2)

    # Compute the quartet distance between the trees
    qd = dendropy.calculate.treecompare.symmetric_difference(tree1, tree2)

    return qd


def get_tip_groups(n_tips, group_size=2):
    """
    Lists all groups of N taxa from a set of n_tips.

    Parameters:
    n_tips (int): The number of tips (leaves) in the tree.
    group_size (int): The number of tips per group.

    Returns:
    list: A list of tuples, each containing a unique group of N taxa.
    """
    return list(itertools.combinations(range(n_tips), group_size))


def get_mrcas(tips, tree):
    """
    Get the most recent common ancestors (MRCA) for each pair of tips in the given subset.

    Parameters:
    tips (tuple): A tuple of tip indices.
    tree (dendropy.Tree): The tree to retrieve MRCA information from.

    Returns:
    list: A list of MRCA nodes for each pair in the subset.
    """
    mrcas = {}
    for tip1, tip2 in itertools.combinations(tips, 2):
        mrca = tree.mrca(taxon_labels=[tree.taxon_namespace[tip1].label,
                                       tree.taxon_namespace[tip2].label])
        mrcas[(tip1, tip2)] = mrca
        mrcas[(tip2, tip1)] = mrca
    return mrcas



def taxa_subset_state(tips, tree):
    """
    Determines the state of a subset of taxa in a tree, generalized for 3 or more tips.

    Parameters:
    tips (tuple): A tuple of N taxa indices representing the subset.
    tree (dendropy.Tree): The tree in which to evaluate the subset state.

    Returns:
    int: The subset state:
         0 for unresolved,
         1-3 for resolved pairs,
         4+i for singleton branch states (where i is the index of the singleton tip).
    """
    n_tips = len(tips)
    if n_tips < 3:
        raise ValueError(f"Minimum 3 tips are required, found only {n_tips}")

    # Get MRCA nodes for each pair of tips
    mrcas = get_mrcas(tips, tree)

    if n_tips == 3:
        # Handling the triplet state (3 tips)
        mrca12 = mrcas[(tips[0], tips[1])]
        mrca13 = mrcas[(tips[0], tips[2])]
        mrca13 = mrcas[(tips[0], tips[2])]
        mrca23 = mrcas[(tips[1], tips[2])]
        if mrca12 == mrca13 and mrca12 != mrca23:
            return 1  # A-B | C
        elif mrca12 == mrca23 and mrca12 != mrca13:
            return 2  # A-C | B
        elif mrca13 == mrca23 and mrca13 != mrca12:
            return 3  # B-C | A

    elif n_tips == 4:
        # Handling the quartet state (4 tips)
        mrca12 = mrcas[(tips[0], tips[1])]
        mrca34 = mrcas[(tips[2], tips[3])]
        mrca13 = mrcas[(tips[0], tips[2])]
        mrca24 = mrcas[(tips[1], tips[3])]
        # Determine quartet state based on closeness of MRCA pairs
        if mrca12 == mrca34 and mrca12 != mrca13 and mrca12 != mrca24:
            return 1  # A-B | C-D
        elif mrca13 == mrca24 and mrca13 != mrca12:
            return 2  # A-C | B-D
        elif mrca12 == mrca24 and mrca12 != mrca13:
            return 3  # A-D | B-C

        # Singleton branch cases: A | BCD, B | ACD, etc.
        # Iterate over each tip as a potential singleton
        for i, tip_n in enumerate(tips):
            triplet_clade = [tree.taxon_namespace[t] for t in tips if t != tip_n]
            triplet_labels = [t.label for t in triplet_clade]
            mrca_triplet = tree.mrca(taxon_labels=triplet_labels)
            singleton_label = tree.taxon_namespace[tip_n].label
            mrca_triplet_leaves = [leaf.label for leaf in mrca_triplet.leaf_nodes()]

            if singleton_label not in mrca_triplet_leaves:
                return 4 + i  # Singleton branch pattern

    else:
        raise NotImplementedError("Not supported for >4 tips")

    # If no clear structure, return unresolved
    return 0


def gqd(non_binary_tree, binary_tree, is_rooted=True, group_size=4, weight_by_depth_in_tree=False):
    """
    Calculates the Generalized Quartet Distance (GQD) between two trees.

    Parameters:
    non_binary_tree (dendropy.Tree): A non-binary tree object.
    binary_tree (dendropy.Tree): A binary tree object.

    Returns:
    float: The generalized quartet distance between the two trees.
    """
    non_binary_tree, binary_tree = prep_trees_for_comparison(non_binary_tree, binary_tree)

    # List all n-tets (cf. "ngram" for "quartet") for the non-binary tree
    n_tips = len(non_binary_tree.taxon_namespace)
    all_ntets = get_tip_groups(n_tips, group_size)

    # Set trees to be rooted to avoid warnings
    if is_rooted:
        non_binary_tree.is_rooted = True
        binary_tree.is_rooted = True

    # Count resolved quartets and differing quartets
    resolved_count = 0
    differing_count = 0

    for ntet in all_ntets:
        b_state = taxa_subset_state(ntet, binary_tree)
        nb_state = taxa_subset_state(ntet, non_binary_tree)

        # Only count resolved states in non-binary tree
        if nb_state > 0:
            mrca = non_binary_tree.mrca(
                taxon_labels=[
                    non_binary_tree.taxon_namespace[tip].label
                    for tip in ntet
                ]
            )
            incr = 1
            if weight_by_depth_in_tree:
                weight = len(mrca.leaf_nodes()) / n_tips
                incr = weight
            
            resolved_count += incr
            if nb_state != b_state:
                differing_count += incr

    # Compute and return the GQD
    gqd = differing_count / resolved_count if resolved_count > 0 else 0
    return gqd



def calculate_tree_distance(tree1, tree2):
    """Calculates mutual information distance between two Newick trees."""
    if isinstance(tree1, dendropy.Tree):
        tree1 = tree1.as_string("newick").strip()
    if isinstance(tree2, dendropy.Tree):
        tree2 = tree2.as_string("newick").strip()
    current_directory = os.path.abspath(os.path.dirname(__file__))
    r_utils_directory = os.path.join(current_directory, "r")
    tree_dist_script = os.path.join(r_utils_directory, "treeDist.R")
    result = subprocess.run(
        [tree_dist_script, tree1, tree2],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"R script failed with error:\n{result.stdout}\n{result.stderr}")

    result = result.stdout.strip()
    result = re.search(r"TreeDistance:\s*(\d+(\.?\d+)?)", result)
    if result:
        return float(result.group(1))
    else:
        raise ValueError("Error parsing TreeDistance result")


def plot_tree(newick_path, png_path, classifications_file=None):
    """Plots a phylogenetic tree (from file saved as Newick string) to a PNG image file."""
    current_directory = os.path.abspath(os.path.dirname(__file__))
    r_utils_directory = os.path.join(current_directory, "r")
    plot_tree_script = os.path.join(
        r_utils_directory,
        "plotTreeWithClassifications.R" if classifications_file else "plotTree.R"
    )
    r_script_args = [plot_tree_script, newick_path, png_path]
    if classifications_file:
        r_script_args.append(classifications_file)
    result = subprocess.run(
        r_script_args,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"R script failed with error:\n{result.stdout}\n{result.stderr}")
