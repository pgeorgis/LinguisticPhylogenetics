from phyloLing.test.utils.data_set import TestDataset
from phyloLing.test.utils.test_factory import create_test_classes
from phyloLing.test.utils.test_runner import run_test_suite
from phyloLing.test.utils.types import LanguageFamily

dataset = TestDataset(LanguageFamily.Germanic)
TestTreeDistanceClass, TestDeterminismClass = create_test_classes(dataset)

class TestTreeDistance(TestTreeDistanceClass):
    pass

class TestDeterminism(TestDeterminismClass):
    pass

if __name__ == '__main__':
    run_test_suite()
