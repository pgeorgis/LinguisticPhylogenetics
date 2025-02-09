import csv
import datetime
import importlib
import math
import os
import subprocess
import sys
import unittest
from dataclasses import dataclass
from enum import Enum

import xmlrunner
import yaml
from prettytable import PrettyTable

current_path: str = os.path.dirname(__file__)
root_project_path: str = os.path.join(current_path, os.pardir, os.pardir)
all_datasets_folder = os.path.join(root_project_path, 'datasets')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from phyloLing.utils.tree import (calculate_tree_distance,
                                  get_gqd_score_to_reference, load_newick_tree)


class LanguageFamily(Enum):
    BaltoSlavic = 'BaltoSlavic',
    Germanic = 'Germanic',
    Romance = 'Romance',
    Sinitic = 'Sinitic',


type DistanceMatrix = dict[tuple[str, str], float]


def assert_distance_matrices_equal(test: unittest.TestCase,
                                   first: DistanceMatrix,
                                   second: DistanceMatrix,
                                   places: int) -> None:
    test.assertCountEqual(
        first.keys(), second.keys(), "The number of language pairs is not equal.")

    max_delta=10 ** -places
    output_places = places + 2

    different = []
    identical = []
    for (lang1, lang2), expected_value in first.items():
        if (lang1, lang2) not in second and (lang2, lang1) not in second:
            test.fail(f"Pair ({lang1}, {lang2}) is missing.")
        actual_value = second.get((lang1, lang2), second.get((lang2, lang1)))
        output_row = [f"{lang1} - {lang2}",
                      expected_value,
                      actual_value,
                      abs(expected_value - actual_value)]
        if math.isclose(expected_value, actual_value, rel_tol=max_delta):
            identical.append(output_row)
        else:
            different.append(output_row)

    output_columns = ['Language pair', 'Expected', 'Actual', 'Difference']
    summary: PrettyTable = PrettyTable(output_columns)
    summary.align = 'r'
    summary.float_format = f".{output_places}"
    different_count = len(different)
    if different:
        for i, row in enumerate(different):
            summary.add_row(row, divider=(i == different_count - 1))
    for row in identical:
        summary.add_row(row)

    if different:
        pairs_label = 'pair differs' if different_count == 1 else 'pairs differ'
        test.fail(f"Distance matrices are not equal, {different_count} {pairs_label} more than {max_delta}:\n{summary.get_string()}")
    else:
        print(f"Distance matrices are equal:\n{summary.get_string()}")


def read_experiment_values(result_path: str) -> DistanceMatrix:
    actual_values: DistanceMatrix = {}
    with (open(result_path, 'r') as file):
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            lang1 = row['Labels']
            for lang2, measurement in row.items():
                if lang2 == 'Labels' or lang1 == lang2 or measurement.strip() == '' or actual_values.get((lang2, lang1)) is not None:
                    continue
                actual_values[(lang1, lang2)] = float(measurement)
    return actual_values


def get_test_report_path() -> str:
    test_result_folder = os.path.join(root_project_path, 'tests')
    if not os.path.exists(test_result_folder):
        os.makedirs(test_result_folder)

    timestamp = datetime.datetime.now().isoformat()
    test_output_path = os.path.join(test_result_folder, f'test-junit_{timestamp}.xml')
    return test_output_path

def run_test_suite() -> None:
    test_output_path = get_test_report_path()
    with open(test_output_path, 'wb') as output:
        unittest.main(
            testRunner=xmlrunner.XMLTestRunner(output=output),
            failfast=False,
            buffer=False,
            catchbreak=False
        )


@dataclass(
    frozen=True,
)
class TreeDistance:
    gqd: float
    wrt: float


@dataclass(
    frozen=True,
)
class ExecutionReference:
    reference_trees: list[str]
    languages: list[str]
    root_language: str

@dataclass(
    frozen=True,
)
class ExecutionResult:
    reference_trees: list[str]
    languages: list[str]
    root_language: str
    distance_matrix: DistanceMatrix
    tree_distance: TreeDistance


class TestConfiguration(Enum):
    MINIMAL = 'minimal',
    FULL = 'full',

class TestDataset:
    results_cache: dict[str, dict[str, ExecutionResult]] = {}
    exec_in_subprocess: bool = True

    def __init__(self,
                 language_family: LanguageFamily,
                 config_files: dict[TestConfiguration, str] = None):
        self.places = 4

        self.language_family = language_family
        dataset_path = os.path.join('datasets', language_family.name)
        config_path = os.path.join(dataset_path, 'config')

        self.best_tree_path = os.path.join(dataset_path, 'trees', 'newick.tre')

        default_config_files = {
            TestConfiguration.MINIMAL: 'test_config_minimal.yml',
            TestConfiguration.FULL: 'test_config_full.yml',
        }

        self.test_configurations = {
            TestConfiguration.MINIMAL:
                os.path.join(config_path,
                             config_files.get(TestConfiguration.MINIMAL,
                                              default_config_files[TestConfiguration.MINIMAL])),
            TestConfiguration.FULL:
                os.path.join(config_path,
                             config_files.get(TestConfiguration.FULL,
                                              default_config_files[TestConfiguration.FULL])),
        } if config_files else {
            TestConfiguration.MINIMAL:
                os.path.join(config_path,
                             default_config_files[TestConfiguration.MINIMAL]),
            TestConfiguration.FULL:
                os.path.join(config_path,
                             default_config_files[TestConfiguration.FULL]),
        }

    @staticmethod
    def get_tree_distance(tree_path,
                          reference_tree_path,
                          execution_reference: ExecutionReference) -> TreeDistance:
        tree = load_newick_tree(tree_path)
        gqd, reference_tree = get_gqd_score_to_reference(
            tree,
            reference_tree_path,
            len(execution_reference.languages),
            execution_reference.root_language,
        )
        return TreeDistance(
            gqd=gqd,
            wrt=calculate_tree_distance(tree, reference_tree),
        )

    @staticmethod
    def get_output_config(dist_matrix_path: str) -> ExecutionReference:
        output_config_path: str = os.path.abspath(os.path.join(dist_matrix_path, os.pardir, 'config.yml'))
        with open(output_config_path, 'r') as file:
            config = yaml.safe_load(file)

        languages: list[str] = config.get('family', {}).get('include', [])
        tree_config: dict = config.get('tree', {})
        tree_root_language: str | None = tree_config.get('root')
        reference_trees: list[str] = tree_config.get('reference', [])
        return ExecutionReference(
            reference_trees=reference_trees,
            languages=languages,
            root_language=tree_root_language,
        )

    def execute_classify_langs_in_subprocess(self,
            test_configuration: TestConfiguration) -> ExecutionResult:
        config_file = self.test_configurations[test_configuration]
        make_command = [
            "make",
            "classify",
            "CONFIG=" + config_file,
        ]
        result = subprocess.run(
            make_command,
            capture_output=True,
            universal_newlines=True,
            cwd=root_project_path,
        )
        if result.returncode != 0:
            raise Exception(f"Command failed with return code {result.returncode}.\n\nstdout:\n{result.stdout}\n\n{result.stderr}\n")
        stdout_lines = result.stdout.splitlines()

        dist_matrix = None
        gqd_distance = None
        wrt_distance = None
        for line in stdout_lines:
            distance_matrix_separator = "Wrote distance matrix to "
            gqd_distance_separator = "GQD wrt reference tree "
            wrt_distance_separator = "TreeDist wrt reference tree "
            if distance_matrix_separator in line:
                dist_matrix = line.split(distance_matrix_separator)[1]
            elif gqd_distance_separator in line:
                gqd_distance_line = line.split(gqd_distance_separator)[1]
                gqd_distance = float(gqd_distance_line.split(": ")[1])
            elif wrt_distance_separator in line:
                wrt_distance_line = line.split(wrt_distance_separator)[1]
                wrt_distance = float(wrt_distance_line.split(": ")[1])

        if not dist_matrix:
            raise Exception("Distance matrix not found.")
        if gqd_distance is None:
            raise Exception("GQD distance not found.")
        if wrt_distance is None:
            raise Exception("WRT distance not found.")

        output_config = self.get_output_config(dist_matrix)
        return ExecutionResult(
            reference_trees=output_config.reference_trees,
            languages=output_config.languages,
            root_language=output_config.root_language,
            distance_matrix=read_experiment_values(dist_matrix),
            tree_distance=TreeDistance(
                gqd=gqd_distance,
                wrt=wrt_distance,
            ),
        )

    def execute_classify_langs_directly(
            self,
            test_configuration: TestConfiguration) -> ExecutionResult:
        classify_langs = importlib.import_module('phyloLing.classifyLangs')
        result = classify_langs.main(self.test_configurations[test_configuration])
        dist_matrix_path = os.path.join(os.path.abspath(root_project_path), result['distance_matrix'])
        output_config = self.get_output_config(dist_matrix_path)
        result = ExecutionResult(
            reference_trees=output_config.reference_trees,
            root_language=output_config.root_language,
            languages=output_config.languages,
            distance_matrix=read_experiment_values(dist_matrix_path),
            tree_distance=TreeDistance(
                gqd=result['gqd_distance'],
                wrt=result['wrt_distance'],
            ),
        )
        importlib.reload(classify_langs)
        return result

    def execute_classify_langs(self,
                               test_configuration: TestConfiguration) -> ExecutionResult:
        return self.execute_classify_langs_in_subprocess(test_configuration) \
            if self.exec_in_subprocess \
            else self.execute_classify_langs_directly(test_configuration)

    def get_result(self,
                   config_key: TestConfiguration) -> ExecutionResult:
        family_name = self.language_family.name
        if family_name not in TestDataset.results_cache:
            TestDataset.results_cache[family_name] = {}
        language_result_cache = TestDataset.results_cache[family_name]

        config_name = config_key.name
        if config_name in language_result_cache:
            return language_result_cache[config_name]
        print(f"No cached result found for language family '{family_name}' with config '{config_name}'. Calculating...")
        result = self.execute_classify_langs(config_key)
        language_result_cache[config_name] = result
        return result

    @staticmethod
    def get_execution_reference(result: ExecutionResult) -> ExecutionReference:
        return ExecutionReference(
            reference_trees=result.reference_trees,
            languages=result.languages,
            root_language=result.root_language,
        )

    def get_best_tree_distances(self,
            execution_reference: ExecutionReference) -> dict[str, TreeDistance]:
        return self.get_tree_distances(self.best_tree_path, execution_reference)

    def get_tree_distances(self,
            tree_path,
            execution_reference: ExecutionReference) -> dict[str, TreeDistance]:

        result: dict = {}
        for reference_tree in execution_reference.reference_trees:
            result[reference_tree] = self.get_tree_distance(
                tree_path,
                reference_tree,
                execution_reference
            )
        return result

    def assert_determinism(self,
            test_configuration: TestConfiguration,
            test: unittest.TestCase) -> None:
        initial_result = self.get_result(test_configuration)
        last_values = initial_result.distance_matrix
        for i in range(5):
            print(f"Running iteration {i + 1} for {self.language_family.name}...")
            start_time = datetime.datetime.now()
            current_result = self.execute_classify_langs(test_configuration)
            end_time = datetime.datetime.now()
            time_elapsed_seconds = round((end_time - start_time).total_seconds())
            total_time_string = str(datetime.timedelta(seconds=time_elapsed_seconds))
            print(f"Iteration {i + 1} done in {total_time_string}.")
            current_matrix = current_result.distance_matrix
            assert_distance_matrices_equal(test,
                last_values, current_matrix, self.places
            )
            last_values = current_matrix

    def assert_gqd_distance(self,
            configuration: TestConfiguration,
            test: unittest.TestCase) -> None:
        result: ExecutionResult = self.get_result(configuration)
        best_tree_distances = self.get_best_tree_distances(
            self.get_execution_reference(result)
        )
        print("GQD distances:")
        for reference_tree in result.reference_trees:
            best_tree_distance = best_tree_distances[reference_tree].gqd
            result_tree_distance = result.tree_distance.gqd

            print(f"\tReference tree: {reference_tree}")
            print(f"\t\tBest tree: \t{best_tree_distance}")
            print(f"\t\tTest tree: \t{result_tree_distance}")

            test.assertLessEqual(
                result_tree_distance, best_tree_distance,
                f"Test tree distance is greater than the best tree distance for {reference_tree}"
            )

    def assert_wrt_distance(self,
                            test_configuration: TestConfiguration,
                            test: unittest.TestCase) -> None:
        result: ExecutionResult = self.get_result(test_configuration)
        best_tree_distances = self.get_best_tree_distances(
            self.get_execution_reference(result)
        )
        print("WRT distances:")
        for reference_tree in result.reference_trees:
            best_tree_distance = best_tree_distances[reference_tree].wrt
            result_tree_distance = result.tree_distance.wrt

            print(f"\tReference tree: {reference_tree}")
            print(f"\t\tBest tree: \t{best_tree_distance}")
            print(f"\t\tTest tree: \t{result_tree_distance}")

            test.assertLessEqual(
                result_tree_distance, best_tree_distance,
                f"Test tree distance is greater than the best tree distance for {reference_tree}"
            )
