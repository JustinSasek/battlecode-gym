from enum import IntEnum


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
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    BACK = 3


class Blocks(IntEnum):
    EMPTY = 0
    WALL = 1
    UNKNOWN = 2


class Layers(IntEnum):
    GRID = 0
    BOT = 1
