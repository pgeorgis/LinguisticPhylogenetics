import os
import sys

from phyloLing.test.utils.types import TreeDistance, ExecutionResult

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from phyloLing.utils.tree import (calculate_tree_distance,
                                  get_gqd_score_to_reference, load_newick_tree)


def _get_tree_distance(tree_path: str,
                       reference_tree_path: str,
                       execution_result: ExecutionResult) -> TreeDistance:
    tree = load_newick_tree(tree_path)
    gqd, reference_tree = get_gqd_score_to_reference(
        tree,
        reference_tree_path,
        len(execution_result.languages),
        execution_result.root_language,
    )
    return TreeDistance(
        gqd=gqd,
        wrt=calculate_tree_distance(tree, reference_tree),
    )


def get_tree_distances(tree_path: str,
                       execution_result: ExecutionResult) -> dict[str, TreeDistance]:
    result: dict = {}
    for reference_tree in execution_result.reference_trees:
        result[reference_tree] = _get_tree_distance(
            tree_path,
            reference_tree,
            execution_result
        )
    return result
