from enum import IntEnum
from dataclasses import dataclass
from typing import Tuple, TypeVar
from gym.core import ObsType, ActType
from numpy import ndarray
from collections.abc import Collection


@dataclass
class Bot:
    pos: Tuple[int, int]  # 2d position
    rot: Tuple[int, int]  # rotation in terms of axes (e.g. (0, 1) is facing towards positive axis 1)


@dataclass
class Agent:
    bots: list[Bot]


@dataclass
class AgentObs:
    bots: list[ndarray]
    grid: ndarray


FullObs = Tuple[AgentObs, ...]

AgentReward = Collection[float]
FullReward = Tuple[AgentReward, ...]


class MainActions(IntEnum):
    NOOP = 0
    FORWARD = 1
    BACK = 2
    LEFT = 3
    RIGHT = 4
    ATTACK = 5
    CHARGE = 6
    BLOCK = 7
    CLAIM = 8


class TurnActions(IntEnum):
    NOOP = 0  # keeps agent in same direction
    LEFT = 1
    RIGHT = 2
    BACK = 3


BotAction = list[IntEnum]  # first element is main action, second is turn action
AgentAction = Collection[BotAction]
FullAction = Tuple[AgentAction, ...]


class Blocks(IntEnum):
    EMPTY = 0
    WALL = 1
    UNKNOWN = 2


class Layers(IntEnum):
    GRID = 0
    BOT = 1
