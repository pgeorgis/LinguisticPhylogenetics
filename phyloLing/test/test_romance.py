import unittest

from phyloLing.test.testUtils import (LanguageFamily, TestConfiguration,
                                      TestDataset, run_test_suite)

dataset = TestDataset(LanguageFamily.Romance)

class TestTreeDistance(unittest.TestCase):
    def test_gqd_tree_distance(self):
        dataset.assert_gqd_distance(TestConfiguration.FULL, self)

    def test_wrt_tree_distance(self):
        dataset.assert_wrt_distance(TestConfiguration.FULL, self)


class TestDeterminism(unittest.TestCase):
    def test_determinism(self):
        dataset.assert_determinism(TestConfiguration.MINIMAL, self)

    def test_minimal_gqd_tree_distance(self):
        dataset.assert_gqd_distance(TestConfiguration.MINIMAL, self)

    def test_minimal_wrt_tree_distance(self):
        dataset.assert_wrt_distance(TestConfiguration.MINIMAL, self)



if __name__ == '__main__':
    run_test_suite()
