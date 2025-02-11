import logging
import os
import unittest

from phyloLing.test.utils.assertions import (assert_determinism,
                                             assert_tree_distances_improved)
from phyloLing.test.utils.classify_langs_wrapper import execute_classify_langs, load_experiment_from_output_config
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
        self.experiments_path = os.path.join(dataset_path, 'experiments')
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

    def load_result_from_cache(self,
            config_key: TestConfiguration) -> ExecutionResultInformation or None:
        if self.language_family not in TestDataset.results_cache:
            TestDataset.results_cache[self.language_family] = {}
        language_result_cache: dict[str, ExecutionResultInformation] = TestDataset.results_cache[self.language_family]
        config_name = config_key.name
        if config_name in language_result_cache:
            return language_result_cache[config_name]

        # get latest result from file system if available and load its data
        # into the cache
        if not os.path.exists(self.experiments_path) or not os.listdir(self.experiments_path):
            return None
        experiment_path = os.path.join(self.experiments_path,
                                       sorted(os.listdir(self.experiments_path))[-1],
                                       config_key.get_experiment_name())
        if not os.path.exists(experiment_path) or not os.listdir(experiment_path):
            return None
        latest_experiment_path = os.path.join(experiment_path, os.listdir(experiment_path)[-1])
        latest_experiment_config_path = os.path.join(latest_experiment_path, 'config.yml')
        if not os.path.exists(latest_experiment_config_path):
            return None
        test_output = load_experiment_from_output_config(latest_experiment_config_path)
        language_result_cache[config_name] = ExecutionResultInformation(
            language_family=self.language_family,
            test_configuration_file=self.test_configurations[config_key],
            result=test_output,
        )
        return language_result_cache[config_name]

    def get_new_result_and_cache(self,
            config_key: TestConfiguration,
            tail_process_output: bool,
            test: unittest.TestCase) -> ExecutionResultInformation:
        config_name = config_key.name
        test_configuration_file = self.test_configurations[config_key]
        if self.language_family not in TestDataset.results_cache:
            TestDataset.results_cache[self.language_family] = {}

        language_result_cache: dict[str, ExecutionResultInformation] = TestDataset.results_cache[self.language_family]
        result = execute_classify_langs(test_configuration_file, tail_process_output, test)
        result_information = ExecutionResultInformation(
            language_family=self.language_family,
            test_configuration_file=test_configuration_file,
            result=result,
        )
        language_result_cache[config_name] = result_information
        return result_information

    def get_result(self,
                   config_key: TestConfiguration,
                   test: unittest.TestCase) -> ExecutionResultInformation:
        cached_result = self.load_result_from_cache(config_key)
        if cached_result:
            return cached_result

        logger.info(
            f"No cached result found for language family '{self.language_family}' "
            f"with config '{config_key.name}'. "
            f"Running classifyLangs..."
        )
        tail_process_output = tail_output[config_key]
        return self.get_new_result_and_cache(config_key, tail_process_output, test)

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
