import logging
import os
from datetime import tzinfo
from zoneinfo import ZoneInfo

from phyloLing.test.utils.types import TestConfiguration

root_project_path: str = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)

determinism_test_iterations: int = 5
decimal_places: int = 4
test_output_time_zone: tzinfo = ZoneInfo('Europe/Berlin')

# should not be changed until a clean way to reload the module is found
exec_in_subprocess: bool = True
assert_distance_matrice_equality: bool = True

tail_output: dict[TestConfiguration, bool] = {
    TestConfiguration.MINIMAL: False,
    TestConfiguration.FULL: True,
}

default_logging_level: int = logging.INFO
default_logging_format: str = '%(asctime)s %(name)s %(levelname)s: %(message)s'
