from enum import Enum


class MainActions(Enum):
    NOOP = 0
    FORWARD = 1
    BACK = 2
    LEFT = 3
    RIGHT = 4
    ATTACK = 5
    CHARGE = 6
    BLOCK = 7
    CLAIM = 8


class TurnActions(Enum):
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    BACK = 3


class Blocks(Enum):
    UNKNOWN = -2
    WALL = -1
    EMPTY = 0


class Views(Enum):
    GRID = 0
    BOT = 1
