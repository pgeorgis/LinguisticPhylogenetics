
from collections import defaultdict
import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree

def progressive_alignment(pairwise_alignments):
    """Performs multiple sequence alignment (MSA) using a progressive alignment approach.
    
    Args:
        pairwise_alignments (list of tuples): List of (seq1, seq2, cost, alignment) tuples
        from extended Needleman-Wunsch alignments.
    
    Returns:
        list: Multiple sequence alignment result
    """
    sequences = set()
    costs = {}
    alignments = {}
    
    # Extract sequences and pairwise costs
    for seq1, seq2, cost, alignment in pairwise_alignments:
        sequences.update([seq1, seq2])
        costs[frozenset([seq1, seq2])] = cost
        alignments[frozenset([seq1, seq2])] = alignment
    
    sequences = list(sequences)
    num_seqs = len(sequences)
    
    # Construct distance matrix
    dist_matrix = np.zeros((num_seqs, num_seqs))
    for i in range(num_seqs):
        for j in range(i + 1, num_seqs):
            key = frozenset([sequences[i], sequences[j]])
            dist_matrix[i, j] = dist_matrix[j, i] = costs.get(key, np.inf)
    
    # Generate guide tree using hierarchical clustering
    linkage_matrix = linkage(dist_matrix, method='average')
    root, nodes = to_tree(linkage_matrix, rd=True)
    
    # Perform progressive alignment
    msa = {seq: [seq] for seq in sequences}
    
    def align_cluster(node):
        if node.is_leaf():
            return msa[sequences[node.id]]
        
        left_msa = align_cluster(node.left)
        right_msa = align_cluster(node.right)
        key = frozenset([tuple(left_msa), tuple(right_msa)])
        aligned = alignments.get(key, None)
        
        if aligned:
            return [x for pair in aligned for x in pair]
        return left_msa + right_msa
    
    return align_cluster(root)
