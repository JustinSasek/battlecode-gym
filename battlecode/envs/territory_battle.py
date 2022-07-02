import gym
from gym import spaces
from gym.core import ObsType, ActType
from spaces import List
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from collections.abc import Collection
from enums import *
from copy import deepcopy


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
        self.shape = shape + (self.n_layers,)

        self.n_agents = len(agents)

        self.grid = np.empty(self.shape, dtype=np.int32)
        self.agents = (None,) * self.n_agents

        # tuple of action spaces for each agent, where each agent's action space is a list of bot action spaces
        # which is initialized with a single bot action space (one bot to start)
        self.action_space = spaces.Tuple(tuple(List([self.bot_action_space()]) for _ in range(self.n_agents)))

        # tuple of observation spaces for each agent, where each agent's observation space is a dict that includes a
        # limited view of the entire grid and a list of bot observation spaces which is initialized with a single bot
        # observation space (one bot to start)
        self.observation_space = spaces.Tuple(tuple(spaces.Dict({
            'bots': List([self.bot_observation_space()]),
            'grid': spaces.MultiDiscrete(  # grid array + extra axis (of size 2) to hold the grid view and bot view
                np.full(self.shape, self.n_agents + self.n_default_blocks),
            ),  # type: ignore
        }) for _ in range(self.n_agents)))

        self.agents_init = deepcopy(agents)

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

    def _get_obs(self) -> ObsType:
        agent_observations = []
        for agent in self.agents:
            agent_view = np.full_like(self.grid, Blocks.UNKNOWN)  # all unknown by default
            agent_observation = {
                'bots': [],
                'grid': agent_view,
            }
            for bot in agent:
                view_size = tuple(self.bot_vision[:2][bot.rot[1 - i]] for i in range(2))
                view_offset = tuple((view_size_axis - 1) // 2 for view_size_axis in view_size)
                view_pos = tuple(bot.pos[i] + view_offset[i] * (bot.rot[i] - 1) for i in range(2))

                bot_view_global = np.full(view_size + (self.n_layers,), Blocks.UNKNOWN)  # bot view in global coords

                grid_intersect = (  # area of intersection between global grid and bot view
                    max(0, view_pos[0]),
                    min(self.grid.shape[0], view_pos[0] + view_size[0]),
                    max(0, view_pos[1]),
                    min(self.grid.shape[1], view_pos[1] + view_size[1])
                )
                view_intersect = (
                    max(0, -view_pos[0]),
                    view_size[0] + min(0, self.grid.shape[0] - (view_pos[0] + view_size[0])),
                    max(0, -view_pos[1]),
                    view_size[1] + min(0, self.grid.shape[1] - (view_pos[1] + view_size[1])),
                )

                bot_view_global[view_intersect[0]:view_intersect[1], view_intersect[2]:view_intersect[3]] = \
                    self.grid[grid_intersect[0]:grid_intersect[1], grid_intersect[2]:grid_intersect[3]]

                n_rotations = abs(bot.rot[0] + 2 * bot.rot[1] - 1)  # how to rotate from world to local coordinates
                bot_view = np.rot90(bot_view_global, n_rotations)  # relative bot view

                # very crude shadow casting time, just cast shadows vertically from perspective of bot
                # if you want to take the time to rly make this accurate, have at it:
                # https://ir.lib.uwo.ca/cgi/viewcontent.cgi?article=8883&context=etd
                for axis_0 in bot_view:
                    for j in range(bot_view.shape[1] - 1):  # walls in last layer cannot cast shadows
                        if axis_0[j, 0] == Blocks.WALL:
                            axis_0[j + 1:] = Blocks.UNKNOWN  # every block after this one in 1-axis is unknown

                bot_view_global = np.rot90(bot_view, -n_rotations)  # sending shadow casting result back to world
                agent_view[grid_intersect[0]:grid_intersect[1], grid_intersect[2]:grid_intersect[3]] = \
                    bot_view_global[view_intersect[0]:view_intersect[1], view_intersect[2]:view_intersect[3]]

                agent_observation['bots'].append(bot_view)
            agent_observations.append(agent_observation)

        return tuple(agent_observations)

    def _get_info(self) -> dict:
        return {
            'grid': self.grid
        }

    def reset(self,
              *,
              seed: int | None = None,
              return_info: bool = False,
              options: dict | None = None,
              ) -> ObsType | Tuple[ObsType, dict]:
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # each agent controls an array of bots
        self.agents = []
        self.grid = np.zeros(self.shape, dtype=np.int32)

        for i, bot in enumerate(self.agents_init):
            self.agents.append([bot])
            self.grid[bot.pos + (Layers.BOT,)] = self.n_default_blocks + i

        self.agents = tuple(self.agents)

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action: ActType) -> Tuple[ObsType, Tuple[Collection[float, ...], ...], bool, dict]:
        done = False  # TODO: step function !!
        reward = ([1], [1])
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info


a = TerritoryBattleMultiEnv()
# print(a.grid[:, :, 1])
print(a._get_obs()[0]['grid'][:, :, 1])
