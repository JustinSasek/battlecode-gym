import gym
from gym import spaces
from spaces import List
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from collections.abc import Collection
from enums import *


@dataclass
class Bot:
    pos: Tuple[int, int]  # 2d position
    rot: Tuple[int, int]  # rotation in terms of axes (e.g. (0, 1) is facing towards positive axis 1)


class TerritoryBattleMultiEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 4
    }

    def __init__(self,
                 shape: int | Tuple[int, int] = (5, 7),
                 agents: Collection[Bot] = (Bot((2, 0), (0, 1)), Bot((2, 6), (0, -1))),
                 window_height: int = 560,
                 ) -> None:
        """
        Create 2d grid world based on shape and spawn agents at given start points.

        :param shape: The 2d shape of the environment.
        :type shape: int or Tuple[int, int], optional
        :param agents: An n-tuple of 2-tuples, creates n agents at given 2-tuple spawn point.
        :type agents: Iterable[Tuple[int, int]], optional
        :param window_height: Height of the pygame window in human-mode.
        :type window_height: int, optional
        """
        super(TerritoryBattleMultiEnv, self).__init__()
        if isinstance(shape, int):
            shape = (shape, shape)

        self.env_shape = shape  # shape of the environment
        self.window_height = window_height

        self.n_agents = len(agents)

        # each agent controls an array of bots
        self.agents = tuple([Bot(agent_pos)] for agent_pos in agents)

        # tuple of action spaces for each agent, where each agent's action space is a list of bot action spaces
        # which is initialized with a single bot action space (one bot to start)
        self.action_space = spaces.Tuple(tuple(List([self.bot_action_space()]) for _ in range(self.n_agents)))

        # tuple of observation spaces for each agent, where each agent's observation space is a dict that includes a
        # limited view of the entire grid and a list of bot observation spaces which is initialized with a single bot
        # observation space (one bot to start)
        self.observation_space = spaces.Tuple(tuple(spaces.Dict({
            'bots': List([self.bot_observation_space()]),
            'grid': spaces.MultiDiscrete(  # grid array + extra axis (of size 2) to hold the grid view and bot view
                np.full(shape + (2,), self.n_agents + len(Blocks))
            ),  # type: ignore
        }) for _ in range(self.n_agents)))

        self.grid = np.zeros(shape, dtype=np.uint32)

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    @staticmethod
    def bot_action_space():  # each bot can do a main action and a turn action right after the main
        return spaces.MultiDiscrete([len(MainActions), len(TurnActions)])

    # each bot can see a 3x3 in front and a bit to left and right. last axis is for grid/bot view
    def bot_observation_space(self):
        return spaces.MultiDiscrete(np.full((3, 3, 2), self.n_agents + len(Blocks)))  # type: ignore

    def _get_obs(self):  # TODO: observation !
        agent_observations = []
        for agent in self.agents:
            agent_observation = {
                'bots': [],
                'grid': 0,
            }
            for bot in agent:
                agent_observation['bots'].append(bot)
            agent_observations.append(agent_observation)


a = TerritoryBattleMultiEnv()
print(a._get_obs())
