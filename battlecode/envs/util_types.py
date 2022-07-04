from abc import abstractmethod
from enum import IntEnum
from dataclasses import dataclass
from typing import Tuple, overload, Iterable
from numpy.typing import NDArray
from collections.abc import MutableSequence, Sequence


@dataclass
class Bot:
    pos: Tuple[int, int]  # 2d position
    rot: Tuple[int, int]  # rotation in terms of axes (e.g. (0, 1) is facing towards positive axis 1)
    block: bool = False  # whether this bot is blocking or not
    ammo: int = 0
    agent_id: int = 0
    id: int = 0


class Bots(MutableSequence[Bot]):
    def __init__(self, *bots: Bot):
        self._bots = list(bots)
        self._agent_id = 0
        for i in range(len(self._bots)):
            self._bots[i].id = i
            self._bots[i].agent_id = self.agent_id

    def insert(self, index: int, value: Bot) -> None:
        self._bots.insert(index, value)
        for i in range(index, len(self._bots)):
            self._bots[i].id = i
            self._bots[i].agent_id = self.agent_id

    @property
    def agent_id(self) -> int:
        return self._agent_id

    @agent_id.setter
    def agent_id(self, o: int) -> None:
        self._agent_id = o
        for bot in self._bots:
            bot.agent_id = o

    @overload
    @abstractmethod
    def __getitem__(self, s: slice) -> MutableSequence[Bot]:
        return self._bots[s]

    def __getitem__(self, i: int) -> Bot:
        return self._bots[i]

    @overload
    @abstractmethod
    def __setitem__(self, s: slice, o: Iterable[Bot]) -> None:
        for i, bot in zip(range(s.start, s.stop, s.step), o):
            bot.id = i
            bot.agent_id = self.agent_id
            self._bots[i] = bot

    def __setitem__(self, i: int, o: Bot) -> None:
        o.id = i
        o.agent_id = self.agent_id
        self._bots[i] = o

    @overload
    @abstractmethod
    def __delitem__(self, s: slice) -> None:
        del self._bots[s]
        for i in range(s.start, len(self._bots)):
            self._bots[i].id = i

    def __delitem__(self, i: int) -> None:
        del self._bots[i]
        for j in range(i, len(self._bots)):
            self._bots[j].id = j

    def __len__(self) -> int:
        return len(self._bots)


class Agent:
    def __init__(self, bots: Bots, agent_id: int = 0):
        self.bots: Bots = bots
        self.bots.agent_id = agent_id
        self.id: int = agent_id

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, o: int) -> None:
        self._id = o
        self.bots.agent_id = o


@dataclass
class AgentObs:
    bots: list[NDArray]
    grid: NDArray


FullObs = Tuple[AgentObs, ...]

AgentReward = Sequence[float]
FullReward = Tuple[AgentReward, ...]


class MainActions(IntEnum):
    NOOP = 0
    FORWARD = 1  # forward, right, back, left should be grouped in this order
    RIGHT = 2
    BACK = 3
    LEFT = 4
    ATTACK = 5
    CHARGE = 6
    BLOCK = 7
    CLAIM = 8


class TurnActions(IntEnum):
    NOOP = 0  # keeps agent in same direction
    RIGHT = 1
    BACK = 2
    LEFT = 3


BotAction = Tuple[MainActions, TurnActions] | NDArray[int]  # first element is main action, second is turn action
AgentAction = Sequence[BotAction]
FullAction = Tuple[AgentAction, ...]


@dataclass
class ActionRepr:  # for internal use
    agent_id: int
    bot_id: int
    main_action: MainActions


class Cells(IntEnum):
    EMPTY = 0
    WALL = 1
    UNKNOWN = 2


class Layers(IntEnum):
    GRID = 0
    BOT = 1
