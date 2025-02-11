from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Callable


class LanguageFamily(Enum):
    BaltoSlavic = 'BaltoSlavic',
    Germanic = 'Germanic',
    Romance = 'Romance',
    Sinitic = 'Sinitic',


class TestConfiguration(Enum):
    MINIMAL = 'minimal',
    FULL = 'full',

    def get_experiment_name(self) -> str:
        return f"test-{self.name.lower()}"


@dataclass(frozen=True)
class TreeDistance:
    gqd: float
    wrt: float


type TreeDistanceMapper = Callable[[TreeDistance], float]


type ReferenceTreePath = str


@dataclass(frozen=True)
class ExecutionReference:
    reference_trees: list[ReferenceTreePath]
    languages: list[str]
    root_language: str
    tree_distances: dict[ReferenceTreePath, TreeDistance]
    run_duration: timedelta
    dist_matrix_path: str


type DistanceMatrix = dict[tuple[str, str], float]


@dataclass(frozen=True)
class ExecutionResult(ExecutionReference):
    distance_matrix: DistanceMatrix


@dataclass(frozen=True)
class ExecutionResultInformation:
    test_configuration_file: str
    language_family: str
    result: ExecutionResult


