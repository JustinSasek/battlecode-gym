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
    n_layers = len(Layers)
    n_default_blocks = len(Blocks)

    def __init__(self,
                 shape: int | Tuple[int, int] = (7, 5),
                 agents: Collection[Bot] = (Bot((0, 2), (1, 0)), Bot((6, 2), (-1, 0))),
                 bot_vision: int | Tuple[int, int] = (3, 3),
                 window_height: int = 560,
                 ) -> None:
        """
        Create 2d grid world based on shape and spawn agents at given start points.

        :param shape: The 2d shape of the environment.
        :type shape: int or Tuple[int, int], optional
        :param agents: An n-tuple of 2-tuples, creates n agents at given 2-tuple spawn point.
        :type agents: Iterable[Tuple[int, int]], optional
        :param bot_vision: Shape that represents the area the bot can see ahead of itself.
        :type bot_vision: int or Tuple[int, int], optional
        :param window_height: Height of the pygame window in human-mode.
        :type window_height: int, optional
        """
        super(TerritoryBattleMultiEnv, self).__init__()
        if isinstance(shape, int):
            shape = (shape, shape)
        if isinstance(bot_vision, int):
            bot_vision = (bot_vision, bot_vision)

        self.window_height = window_height
        self.bot_vision = bot_vision + (self.n_layers,)

        self.n_agents = len(agents)

        # tuple of action spaces for each agent, where each agent's action space is a list of bot action spaces
        # which is initialized with a single bot action space (one bot to start)
        self.action_space = spaces.Tuple(tuple(List([self.bot_action_space()]) for _ in range(self.n_agents)))

        # tuple of observation spaces for each agent, where each agent's observation space is a dict that includes a
        # limited view of the entire grid and a list of bot observation spaces which is initialized with a single bot
        # observation space (one bot to start)
        self.observation_space = spaces.Tuple(tuple(spaces.Dict({
            'bots': List([self.bot_observation_space()]),
            'grid': spaces.MultiDiscrete(  # grid array + extra axis (of size 2) to hold the grid view and bot view
                np.full(shape + (self.n_layers,), self.n_agents + self.n_default_blocks),
            ),  # type: ignore
        }) for _ in range(self.n_agents)))


        # each agent controls an array of bots
        self.agents = []
        self.grid = np.zeros(shape + (self.n_layers,), dtype=np.int32)

        for i, bot in enumerate(agents):
            self.agents.append([bot])
            self.grid[bot.pos + (Layers.BOT,)] = self.n_default_blocks + i

        self.agents = tuple(self.agents)

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct frame rate in
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
        return spaces.MultiDiscrete(np.full(self.bot_vision, self.n_agents + self.n_default_blocks))  # type: ignore

    def _get_obs(self):  # TODO: observation !
        agent_observations = []
        for agent in self.agents:
            agent_observation = {
                'bots': [],
                'grid': np.full_like(self.grid, Blocks.UNKNOWN),  # all unknown by default
            }
            for bot in agent:
                view_size = tuple(self.bot_vision[:2][bot.rot[1-i]] for i in range(2))
                view_offset = tuple((view_size_axis - 1) // 2 for view_size_axis in view_size)
                view_pos = tuple(bot.pos[i] + view_offset[i] * (bot.rot[i] - 1) for i in range(2))

                bot_view_global = np.empty(view_size + (self.n_layers,))  # bot view from global perspective

                for i in range(view_size[0]):
                    for j in range(view_size[1]):
                        global_pos = (view_pos[0] + i, view_pos[1] + j)

                        for axis in range(2):
                            if not 0 <= global_pos[axis] < self.grid.shape[axis]:  # if outside of bounds in either axis
                                bot_view_global[i, j] = Blocks.WALL
                                break
                        else:  # only runs if the loop exits normally (global pos is within bounds)
                            bot_view_global[i, j, :] = self.grid[global_pos]

                bot_view = np.rot90(bot_view_global, abs(bot.rot[0] + 2 * bot.rot[1] - 1))  # relative bot view

                agent_observation['bots'].append(bot_view)
            agent_observations.append(agent_observation)

        return tuple(agent_observations)


a = TerritoryBattleMultiEnv()
# print(a.grid[:, :, 1])
print(a._get_obs()[0]['bots'][0][:, :, 1])
