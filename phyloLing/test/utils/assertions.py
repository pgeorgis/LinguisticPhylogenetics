import datetime
import logging
import math
import unittest
from datetime import datetime, timedelta

from prettytable import PrettyTable

from phyloLing.test.utils.classify_langs_wrapper import execute_classify_langs
from phyloLing.test.utils.test_configuration import (
    assert_distance_matrice_equality, decimal_places, default_logging_format,
    default_logging_level, test_output_time_zone, determinism_test_iterations)
from phyloLing.test.utils.tree_utils import get_tree_distances
from phyloLing.test.utils.types import (DistanceMatrix, ExecutionResult,
                                        ExecutionResultInformation,
                                        TreeDistance, TreeDistanceMapper)

logging.basicConfig(level=default_logging_level, format=default_logging_format)
_logger = logging.getLogger(__name__)

_max_value_difference = 10 ** -decimal_places
_output_decimal_places = decimal_places + 1


def _format_float(value: float) -> str:
    return f"{value:.{_output_decimal_places}f}"


def _format_elapsed_time(time: timedelta) -> str:
    return str(time).split('.')[0]


def _format_datetime(time: datetime) -> str:
    return time.astimezone(test_output_time_zone).strftime("%Y-%m-%d %H:%M:%S")


def _is_close(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=_max_value_difference)


def _assert_distance_matrices_equal(test: unittest.TestCase,
                                    first: DistanceMatrix,
                                    second: DistanceMatrix,
                                    fail_if_distance_matrices_differ: bool) -> None:
    test.assertCountEqual(first.keys(), second.keys(), "The number of language pairs is not equal.")
    different = []
    equal_under_threshold = []
    completely_equal: bool = True
    for (lang1, lang2), expected_value in first.items():
        if (lang1, lang2) not in second and (lang2, lang1) not in second:
            test.fail(f"Pair ({lang1}, {lang2}) is missing.")
        actual_value = second.get((lang1, lang2), second.get((lang2, lang1)))
        abs_diff = abs(expected_value - actual_value)
        output_row = [f"{lang1} - {lang2}",
                      expected_value,
                      actual_value,
                      abs_diff]
        if abs_diff > 0:
            completely_equal = False
        if _is_close(expected_value, actual_value):
            equal_under_threshold.append(output_row)
        else:
            different.append(output_row)

    output_columns = ['Language pair', 'Expected', 'Actual', 'Difference']
    summary: PrettyTable = PrettyTable(output_columns)
    summary.align = 'r'
    summary.float_format = f".{_output_decimal_places}"
    different_count = len(different)
    if different:
        for i, row in enumerate(different):
            summary.add_row(row, divider=(i == different_count - 1))
    for row in equal_under_threshold:
        summary.add_row(row)

    if different:
        pairs_label = 'pair differs' if different_count == 1 else 'pairs differ'
        message = f"Distance matrices are not equal, {different_count} {pairs_label} more than {_max_value_difference}:\n{summary.get_string()}"
        if fail_if_distance_matrices_differ:
            test.fail(message)
        else:
            _logger.warning(message)
    else:
        if completely_equal:
            _logger.info(f"Distance matrices are equal.")
        else:
            _logger.info(f"Distance matrices are almost equal:\n{summary.get_string()}")


def _assert_tree_distances_equal(test: unittest.TestCase,
                                 first: TreeDistance,
                                 second: TreeDistance,
                                 tree_distance_mapper: TreeDistanceMapper,
                                 distance_label: str) -> None:
    first_value = tree_distance_mapper(first)
    second_value = tree_distance_mapper(second)
    formatted_first_value = _format_float(first_value)
    formatted_second_value = _format_float(second_value)
    delta = abs(first_value - second_value)
    formatted_delta = _format_float(delta)

    if _is_close(first_value, second_value):
        _logger.info(f"{distance_label} tree distances are equal.")
    else:
        test.fail(f"{distance_label} tree distances are not equal: "
                  f"{formatted_first_value} != {formatted_second_value}, |Δ|: {formatted_delta}")



def _assert_all_tree_distances_equal(test: unittest.TestCase,
                                     actual: TreeDistance,
                                     expected: TreeDistance) -> None:
    _assert_tree_distances_equal(test, actual, expected, lambda distance: distance.gqd,"GQD")
    _assert_tree_distances_equal(test, actual, expected, lambda distance: distance.wrt,"WRT")


def _assert_results_equal(test: unittest.TestCase,
                          current_result: ExecutionResult,
                          expected_result: ExecutionResult) -> None:
    _assert_distance_matrices_equal(test,
                                    current_result.distance_matrix,
                                    expected_result.distance_matrix,
                                    assert_distance_matrice_equality)
    for reference_tree in expected_result.reference_trees:
        current_tree_distance = current_result.tree_distances[reference_tree]
        expected_tree_distance = expected_result.tree_distances[reference_tree]
        _assert_all_tree_distances_equal(test,
                                         current_tree_distance,
                                         expected_tree_distance)


def _log_estimated_time_remaining(current_iteration: int,
                                  last_iteration_time: timedelta,
                                  language_family: str) -> None:
    formatted_iteration: int = current_iteration + 1
    _logger.info(f"Running iteration {formatted_iteration} for {language_family}...")
    remaining_iterations: int = determinism_test_iterations - current_iteration
    estimated_remaining_time_seconds: int = remaining_iterations * round(last_iteration_time.total_seconds())
    estimated_remaining_time: timedelta = timedelta(seconds=estimated_remaining_time_seconds)
    _logger.info(f"Estimated time remaining: {str(estimated_remaining_time)}")
    estimated_end_time = _format_datetime(datetime.now() + estimated_remaining_time)
    _logger.info(f"Estimated finish time: {estimated_end_time}")


def assert_determinism(test: unittest.TestCase,
                       initial_result: ExecutionResultInformation) -> None:
    initial_run_duration: timedelta = initial_result.result.run_duration
    _logger.info(f"Initial run done in {_format_elapsed_time(initial_run_duration)}")
    last_result: ExecutionResult = initial_result.result
    language_family: str = initial_result.language_family
    test_config_file: str = initial_result.test_configuration_file
    tail_execution_output: bool = False

    last_iteration_time: timedelta = initial_run_duration
    for i in range(determinism_test_iterations):
        _log_estimated_time_remaining(i, last_iteration_time, language_family)

        current_result: ExecutionResult = execute_classify_langs(
            test_config_file, tail_execution_output, test
        )
        last_iteration_time = current_result.run_duration
        _logger.info(f"Iteration {i + 1} done in {_format_elapsed_time(last_iteration_time)}")

        _logger.info(f"Comparing with last iteration values...")
        _assert_results_equal(test, current_result, last_result)

        _logger.info(f"Comparing with initial iteration values...")
        _assert_results_equal(test, current_result, initial_result.result)

        last_result = current_result

def assert_tree_distances_improved(test: unittest.TestCase,
                                   result: ExecutionResult,
                                   best_tree_path: str,
                                   tree_distance_label: str,
                                   tree_distance_mapper: TreeDistanceMapper) -> None:
    best_tree_distances = get_tree_distances(best_tree_path, result)
    distances_have_worsened: bool = False

    summary = PrettyTable(["Reference tree", "Test distance", "Best distance", "Difference"])
    summary.align = 'r'
    summary.float_format = f".{decimal_places}"
    for reference_tree in result.reference_trees:
        best_tree_distance: float = tree_distance_mapper(best_tree_distances[reference_tree])
        result_tree_distance: float = tree_distance_mapper(result.tree_distances[reference_tree])

        is_result_close_to_best: bool = _is_close(result_tree_distance, best_tree_distance)
        if result_tree_distance > best_tree_distance and not is_result_close_to_best:
            distances_have_worsened = True

        reference_tree_name: str = reference_tree.split('/')[-1]
        summary.add_row([reference_tree_name,
                         result_tree_distance,
                         best_tree_distance,
                         result_tree_distance - best_tree_distance])

    if distances_have_worsened:
        test.fail(f"{tree_distance_label} tree distances have worsened:\n{summary.get_string()}")
    else:
        _logger.info(f"{tree_distance_label} tree distances:\n{summary.get_string()}")
