from gym import spaces
from ..util import Bot, BotReward, BotAction, Cells, BotObs
from ..policies import AgentPolicy, BotPolicy
from typing import Tuple
from numpy.typing import NDArray
import numpy as np
from . import TerritoryBattleSingleEnv


class TerritoryBattleBotEnv(TerritoryBattleSingleEnv):
    action_space: spaces.MultiDiscrete
    observation_space: spaces.MultiDiscrete

    def __init__(self,
                 bot_policies: Tuple[BotPolicy, ...] = (BotPolicy(),),
                 shape: int | Tuple[int, int] = (7, 5),
                 agents_init: Tuple[Bot, ...] = (Bot((0, 2), (1, 0)), Bot((6, 2), (-1, 0))),
                 bot_vision: int | Tuple[int, int] = (3, 3),
                 max_ammo: int = 3,
                 window_height: int = 560,
                 agent_id: int = 0,
                 ) -> None:
        """
        Create 2d grid world based on shape and spawn agents at given start points.

        :param bot_policies: The policies that dictate the actions of the other bots. Should be a tuple of length
        1 - bots_init, as it handles every agent except for agent agent_id
        :type bot_policies: Tuple[AgentPolicy, ...]
        :param shape: The 2d shape of the environment.
        :type shape: int or Tuple[int, int], optional
        :param agents_init: An n-tuple of 2-tuples, creates n agents with bots at given 2-tuple spawn point.
        :type agents_init: Tuple[Bot, ...], optional
        :param bot_vision: Shape that represents the area the bot can see ahead of itself.
        :type bot_vision: int or Tuple[int, int], optional
        :param max_ammo: Max ammo that a bot can have to attack with.
        :type max_ammo: int, optional
        :param window_height: Height of the pygame window in human-mode.
        :type window_height: int, optional
        :param agent_id: id of the agent we are getting the perspective of.
        :type agent_id: int, optional
        """
        agent_policies = tuple(AgentPolicy(bot_policy) for bot_policy in bot_policies)
        super().__init__(agent_policies, shape, agents_init, bot_vision, max_ammo, 0.0, window_height, agent_id)

        self.observation_space = self.bot_observation_space()

    @property
    def action_space(self):
        return self._action_space[self.agent_id][0]

    def reset(self,
              *,
              seed: int | None = None,
              return_info: bool = False,
              options: dict | None = None,
              ) -> Tuple[BotObs, dict] | BotObs:
        reset_info = super().reset(seed=seed, return_info=return_info, options=options)

        if return_info:
            return reset_info[0].bots[0], reset_info[1]
        else:
            return reset_info.bots[0]

    def step(self, action: BotAction) -> Tuple[BotObs, BotReward, bool, dict]:
        observation, reward, done, info = super().step([action])

        if len(observation.bots) == 0:  # if we are dead
            done = True
            bot_observation = BotObs(np.full(self.bot_vision, Cells.UNKNOWN), 0)
            bot_reward = 0
        else:
            bot_observation = observation.bots[0]
            bot_reward = reward[0]

        return bot_observation, bot_reward, done, info
