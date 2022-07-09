from gym import spaces
from .mutable_spaces import List
import numpy as np
from .util_types import *
from .territory_battle import TerritoryBattleMultiEnv


class TerritoryBattleSingleEnv(TerritoryBattleMultiEnv):
    action_space: List
    observation_space: spaces.Dict

    def __init__(self,
                 shape: int | Tuple[int, int] = (7, 5),
                 agents_init: Tuple[Bot, ...] = (Bot((0, 2), (1, 0)), Bot((6, 2), (-1, 0))),
                 bot_vision: int | Tuple[int, int] = (3, 3),
                 max_ammo: int = 3,
                 spawn_chance: float = 0.01,
                 window_height: int = 560,
                 agent_id: int = 0,
                 ) -> None:
        """
        Create 2d grid world based on shape and spawn agents at given start points.

        :param shape: The 2d shape of the environment.
        :type shape: int or Tuple[int, int], optional
        :param agents_init: An n-tuple of 2-tuples, creates n agents with bots at given 2-tuple spawn point.
        :type agents_init: Tuple[Bot, ...], optional
        :param bot_vision: Shape that represents the area the bot can see ahead of itself.
        :type bot_vision: int or Tuple[int, int], optional
        :param max_ammo: Max ammo that a bot can have to attack with.
        :type max_ammo: int, optional
        :param spawn_chance: Probability of a given claimed territory to spawn a bot on a given tick. For some reason
        this breaks the reproducibility of np.random_seed when it's nonzero.
        :type spawn_chance: float, optional
        :param window_height: Height of the pygame window in human-mode.
        :type window_height: int, optional
        :param agent_id: id of the agent we are getting the perspective of.
        :type agent_id: int, optional
        """
        super(TerritoryBattleMultiEnv, self).__init__(shape, agents_init, bot_vision,
                                                      max_ammo, spawn_chance, window_height)
        # a list of bot action spaces
        self.action_space = List([])

        # a limited view of the entire grid and a list of bot observation spaces which is initialized with a single bot
        # observation space (one bot to start)
        self.observation_space = spaces.Dict({
            'bots': List([self.bot_observation_space()]),
            'grid': spaces.MultiDiscrete(  # grid array + extra axis (of size 2) to hold the grid view and bot view
                np.full(self.shape, self.n_agents + self.n_default_cells),
            ),
        })

        self.agent_id = agent_id

    def _get_obs(self) -> AgentObs:
        return super()._get_obs()[self.agent_id]

    def reset(self,
              *,
              seed: int | None = None,
              return_info: bool = False,
              options: dict | None = None,
              ) -> FullObs | Tuple[FullObs, dict]:
        return_info = super().reset(seed=seed, return_info=return_info, options=options)

        # reset action space
        self.action_space: List[spaces.MultiDiscrete] = self.agent_action_space(0, seed)

        return return_info

    def step(self, action: AgentAction) -> Tuple[AgentObs, AgentReward, bool, dict]:
        observation, reward, done, info = super().step((action,))

        return observation[self.agent_id], reward[self.agent_id], done, info


a = TerritoryBattleSingleEnv()
a.reset()
# print(a.grid[:, :, 1])
# print(a._get_obs()[0].grid[:, :, 0])
