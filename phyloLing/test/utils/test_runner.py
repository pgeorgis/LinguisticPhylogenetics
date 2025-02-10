import os
import unittest
from datetime import datetime

import xmlrunner

from phyloLing.test.utils.test_configuration import root_project_path


def get_test_report_path() -> str:
    test_result_folder = os.path.join(root_project_path, 'tests')
    if not os.path.exists(test_result_folder):
        os.makedirs(test_result_folder)

    timestamp = datetime.now().isoformat()
    test_output_path = os.path.join(test_result_folder, f'test-junit_{timestamp}.xml')
    return test_output_path


def run_test_suite() -> None:
    test_output_path = get_test_report_path()
    with open(test_output_path, 'wb') as output:
        unittest.main(
            testRunner=xmlrunner.XMLTestRunner(output=output),
            failfast=False,
            buffer=False,
            catchbreak=False,
        )
