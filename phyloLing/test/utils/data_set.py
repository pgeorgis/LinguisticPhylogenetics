import logging
import os
import unittest
from datetime import datetime

from phyloLing.test.utils.assertions import (assert_determinism,
                                             assert_tree_distances_improved)
from phyloLing.test.utils.classify_langs_wrapper import execute_classify_langs
from phyloLing.test.utils.test_configuration import (default_logging_format,
                                                     default_logging_level,
                                                     tail_output)
from phyloLing.test.utils.types import (ExecutionResultInformation,
                                        LanguageFamily, TestConfiguration,
                                        TreeDistanceMapper)

logging.basicConfig(level=default_logging_level, format=default_logging_format)
logger = logging.getLogger(__name__)


class TestDataset:
    results_cache: dict[str, dict[str, ExecutionResultInformation]] = {}
    exec_in_subprocess: bool = True

    def __init__(self,
                 language_family: LanguageFamily,
                 config_files: dict[TestConfiguration, str] = None):
        self.language_family = language_family.name
        dataset_path = os.path.join('datasets', self.language_family)
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

    def get_result(self,
                   config_key: TestConfiguration,
                   test: unittest.TestCase) -> ExecutionResultInformation:
        if self.language_family not in TestDataset.results_cache:
            TestDataset.results_cache[self.language_family] = {}
        language_result_cache: dict[str, ExecutionResultInformation] = TestDataset.results_cache[self.language_family]
        test_configuration_file: str = self.test_configurations[config_key]
        config_name = config_key.name
        if config_name in language_result_cache:
            return language_result_cache[config_name]
        logger.info(f"No cached result found for language family '{self.language_family}' with config '{config_name}'. Running classifyLangs...")
        start_time = datetime.now()
        test_config_file = self.test_configurations[config_key]
        tail_process_output = tail_output[config_key]
        result = execute_classify_langs(test_config_file, tail_process_output, test)
        end_time = datetime.now()
        time_elapsed = end_time - start_time
        language_result_cache[config_name] = ExecutionResultInformation(
            language_family=self.language_family,
            test_configuration_file=test_configuration_file,
            result=result,
            time_elapsed=time_elapsed,
        )
        return language_result_cache[config_name]

    def assert_determinism(self,
                           test_configuration: TestConfiguration,
                           test: unittest.TestCase) -> None:
        initial_result: ExecutionResultInformation = self.get_result(test_configuration, test)
        assert_determinism(test, initial_result)

    def assert_tree_distances_improved(self,
                                       tree_distance_mapper: TreeDistanceMapper,
                                       distance_label: str,
                                       configuration: TestConfiguration,
                                       test: unittest.TestCase) -> None:
        result_information: ExecutionResultInformation = self.get_result(configuration, test)
        assert_tree_distances_improved(test,
                                       result_information.result,
                                       self.best_tree_path,
                                       distance_label,
                                       tree_distance_mapper)

    def assert_gqd_distance_improved(self,
                                     configuration: TestConfiguration,
                                     test: unittest.TestCase) -> None:
        self.assert_tree_distances_improved(
            lambda distance: distance.gqd, "GQD", configuration, test)

    def assert_wrt_distance_improved(self,
                                     configuration: TestConfiguration,
                                     test: unittest.TestCase) -> None:
        self.assert_tree_distances_improved(
            lambda distance: distance.wrt, "WRT", configuration, test)
