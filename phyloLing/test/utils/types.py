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


type DistanceMatrix = dict[tuple[str, str], float]


@dataclass(frozen=True)
class ExecutionResult(ExecutionReference):
    distance_matrix: DistanceMatrix


@dataclass(frozen=True)
class ExecutionResultInformation:
    test_configuration_file: str
    language_family: str
    result: ExecutionResult
    time_elapsed: timedelta


