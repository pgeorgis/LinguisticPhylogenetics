from ete3 import Tree

def reroot_tree(newick_str: str, outgroup: str | tuple) -> str:
    """
    Reroots the tree at the specified tip or clade outgroup.

    Parameters:
    newick_str (str): The input Newick tree string.
    outgroup (str or tuple): A single tip name or a tuple of tip names defining the outgroup clade.

    Returns:
    str: The Newick string of the rerooted tree.
    """
    # Parse the Newick string into a tree
    tree = Tree(newick_str, format=1)

    # Determine whether we're rerooting at a single tip or a clade
    if isinstance(outgroup, tuple):
        # Find the most recent common ancestor of the clade
        nodes = [tree.search_nodes(name=tip)[0] for tip in outgroup]
        mrca = tree.get_common_ancestor(nodes)
        tree.set_outgroup(mrca)
    else:
        # Reroot at a single tip
        target_node = tree.search_nodes(name=outgroup)
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
    clade1_tips (tuple): A tuple of tip names defining the first clade.
    clade2_tips (tuple): A tuple of tip names defining the second clade.
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

    # Find the common ancestor of each clade
    clade1_node = tree.get_common_ancestor(clade1_tips)
    clade2_node = tree.get_common_ancestor(clade2_tips)

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
