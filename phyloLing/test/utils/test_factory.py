import unittest

from phyloLing.test.utils.data_set import TestDataset
from phyloLing.test.utils.types import TestConfiguration


def create_test_classes(dataset: TestDataset) -> tuple[type, type]:
    class TestTreeDistance(unittest.TestCase):
        def test_gqd_tree_distance(self):
            dataset.assert_gqd_distance_improved(TestConfiguration.FULL, self)

        def test_wrt_tree_distance(self):
            dataset.assert_wrt_distance_improved(TestConfiguration.FULL, self)

    class TestDeterminism(unittest.TestCase):
        def test_determinism(self):
            dataset.assert_determinism(TestConfiguration.MINIMAL, self)

        def test_minimal_gqd_tree_distance(self):
            dataset.assert_gqd_distance_improved(TestConfiguration.MINIMAL, self)

        def test_minimal_wrt_tree_distance(self):
            dataset.assert_wrt_distance_improved(TestConfiguration.MINIMAL, self)

    return TestTreeDistance, TestDeterminism
