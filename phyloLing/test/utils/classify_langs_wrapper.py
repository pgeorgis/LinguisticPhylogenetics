import csv
import importlib
import logging
import os
import subprocess
import unittest

import yaml

from phyloLing.test.utils.test_configuration import (default_logging_format,
                                                     default_logging_level,
                                                     exec_in_subprocess,
                                                     root_project_path)
from phyloLing.test.utils.types import (DistanceMatrix, ExecutionReference,
                                        ExecutionResult, TreeDistance)

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


def _get_output_config(dist_matrix_path: str) -> ExecutionReference:
    output_config_path: str = os.path.abspath(os.path.join(dist_matrix_path, os.pardir, 'config.yml'))
    with open(output_config_path, 'r') as file:
        config = yaml.safe_load(file)

    languages: list[str] = []
    with (open(dist_matrix_path, 'r') as file):
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            language: str = row['Labels'].strip()
            if language != 'Labels':
                languages.append(language)

    tree_config: dict = config.get('tree', {})
    tree_root_language: str | None = tree_config.get('root')
    reference_trees: list[str] = list(dict.fromkeys(tree_config.get('reference', [])))
    return ExecutionReference(
        reference_trees=reference_trees,
        languages=languages,
        root_language=tree_root_language,
    )

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

    test.assertEqual(result.returncode, 0, f"Command failed with return code {result.returncode}.\n\nstdout:\n{result.stdout}\n\nstderr:{result.stderr}\n")

    dist_matrix = None
    gqd_distance = None
    wrt_distance = None
    for line in stdout_lines:
        distance_matrix_separator = "Wrote distance matrix to "
        gqd_distance_separator = "GQD wrt reference tree "
        wrt_distance_separator = "TreeDist wrt reference tree "
        if distance_matrix_separator in line:
            dist_matrix = line.split(distance_matrix_separator)[1].strip()
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

    output_config = _get_output_config(dist_matrix)
    return ExecutionResult(
        reference_trees=output_config.reference_trees,
        languages=output_config.languages,
        root_language=output_config.root_language,
        distance_matrix=_read_experiment_values(dist_matrix),
        tree_distance=TreeDistance(
            gqd=gqd_distance,
            wrt=wrt_distance,
        ),
    )

def _execute_classify_langs_directly(config_file: str) -> ExecutionResult:
    classify_langs = importlib.import_module('phyloLing.classifyLangs')
    result = classify_langs.main(config_file)
    dist_matrix_path = os.path.join(os.path.abspath(root_project_path), result['distance_matrix'])
    output_config = _get_output_config(dist_matrix_path)
    result = ExecutionResult(
        reference_trees=output_config.reference_trees,
        root_language=output_config.root_language,
        languages=output_config.languages,
        distance_matrix=_read_experiment_values(dist_matrix_path),
        tree_distance=TreeDistance(
            gqd=result['gqd_distance'],
            wrt=result['wrt_distance'],
        ),
    )
    importlib.reload(classify_langs)
    return result

def execute_classify_langs(test_configuration_file: str,
                           tail_output: bool,
                           test: unittest.TestCase) -> ExecutionResult:
    return _execute_classify_langs_in_subprocess(test_configuration_file, tail_output, test) \
        if exec_in_subprocess \
        else _execute_classify_langs_directly(test_configuration_file)
