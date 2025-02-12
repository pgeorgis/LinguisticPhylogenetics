import csv
import importlib
import logging
import os
import subprocess
import unittest
from datetime import timedelta, datetime

import yaml

from phyloLing.test.utils.test_configuration import (default_logging_format,
                                                     default_logging_level,
                                                     exec_in_subprocess,
                                                     root_project_path)
from phyloLing.test.utils.types import (DistanceMatrix, ExecutionReference,
                                        ExecutionResult, TreeDistance, ReferenceTreePath)

logging.basicConfig(level=default_logging_level, format=default_logging_format)
logger = logging.getLogger(__name__)


def _read_experiment_values(result_path: str) -> DistanceMatrix:
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


def _load_experiment_from_dist_matrix(dist_matrix_path: str) -> ExecutionResult:
    output_config_path = os.path.abspath(os.path.join(dist_matrix_path, os.pardir, 'config.yml'))
    return load_experiment_from_output_config(output_config_path)


def _execute_classify_langs_in_subprocess(config_file: str,
                                          tail_output: bool,
                                          test: unittest.TestCase) -> ExecutionResult:
    make_command = [
        "make",
        "classify",
        "CONFIG=" + config_file,
    ]
    stdout_lines = []
    if tail_output:
        result = subprocess.Popen(
            make_command,
            cwd=root_project_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            text=True,
        )
        logging.basicConfig(level=default_logging_level, format='%(message)s', force=True)
        while True:
            line = result.stdout.readline()
            if not line:
                break
            logger.info(line.strip('\n'))
            stdout_lines.append(line)
        result.wait()
        logging.basicConfig(level=default_logging_level, format=default_logging_format, force=True)
    else:
        result = subprocess.run(
            make_command,
            capture_output=True,
            universal_newlines=True,
            cwd=root_project_path,
            text=True,
        )
        stdout_lines = result.stdout.splitlines()

    test.assertEqual(result.returncode, 0,
                     f"Command failed with return code {result.returncode}.\n\nstdout:\n{result.stdout}\n\nstderr:{result.stderr}\n")

    dist_matrix = None
    for line in stdout_lines:
        distance_matrix_separator = "Wrote distance matrix to "
        if distance_matrix_separator in line:
            dist_matrix = line.split(distance_matrix_separator)[1].strip()

    if not dist_matrix:
        raise Exception("Distance matrix not found.")

    return _load_experiment_from_dist_matrix(dist_matrix)

def _execute_classify_langs_directly(config_file: str) -> ExecutionResult:
    classify_langs = importlib.import_module('phyloLing.classifyLangs')
    result = classify_langs.main(config_file)
    dist_matrix_path = os.path.join(os.path.abspath(root_project_path), result['distance_matrix'])
    result = _load_experiment_from_dist_matrix(dist_matrix_path)
    importlib.reload(classify_langs)
    return result


def _load_result_config(output_config_path: str) -> ExecutionReference:
    with open(output_config_path, 'r') as file:
        config = yaml.safe_load(file)

    dist_matrix_path: str = config.get('output', {}).get('dist_matrix')
    languages: list[str] = []
    with (open(dist_matrix_path, 'r') as file):
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            language: str = row['Labels'].strip()
            if language != 'Labels':
                languages.append(language)

    tree_config: dict = config.get('tree', {})
    tree_root_language: str | None = tree_config.get('root')
    reference_trees: list[ReferenceTreePath] = list(dict.fromkeys(tree_config.get('reference', [])))

    tree_distances: dict[ReferenceTreePath, TreeDistance] = {}
    tree_evaluation: dict[ReferenceTreePath, dict[str, float | str]] = tree_config.get('eval', {})
    for reference_tree in reference_trees:
        evaluation_result = tree_evaluation.get(reference_tree, {})
        tree_distances[reference_tree] = TreeDistance(
            gqd=float(evaluation_result.get('GQD', 0)),
            wrt=float(evaluation_result.get('TreeDist', 0)),
        )

    run_duration_string: str = config.get('run_info', {}).get('duration', '')
    run_duration_time = datetime.strptime(run_duration_string,"%H:%M:%S")
    run_duration = timedelta(hours=run_duration_time.hour,
                             minutes=run_duration_time.minute,
                             seconds=run_duration_time.second)
    return ExecutionReference(
        reference_trees=reference_trees,
        languages=languages,
        root_language=tree_root_language,
        tree_distances=tree_distances,
        run_duration=run_duration,
        dist_matrix_path=dist_matrix_path,
    )


def load_experiment_from_output_config(config_path: str) -> ExecutionResult:
    output_config = _load_result_config(config_path)
    return ExecutionResult(
        reference_trees=output_config.reference_trees,
        languages=output_config.languages,
        root_language=output_config.root_language,
        distance_matrix=_read_experiment_values(output_config.dist_matrix_path),
        tree_distances=output_config.tree_distances,
        run_duration=output_config.run_duration,
        dist_matrix_path=output_config.dist_matrix_path,
    )


def execute_classify_langs(test_configuration_file: str,
                           tail_output: bool,
                           test: unittest.TestCase) -> ExecutionResult:
    return _execute_classify_langs_in_subprocess(test_configuration_file, tail_output, test) \
        if exec_in_subprocess \
        else _execute_classify_langs_directly(test_configuration_file)
