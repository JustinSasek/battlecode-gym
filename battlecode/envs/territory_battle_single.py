from gym import spaces
import numpy as np
from ..util import Bot, FullObs, AgentObs, AgentReward, AgentAction, Cells
from ..policies import AgentPolicy
from typing import Tuple
from battlecode.mutable_spaces import List
from . import TerritoryBattleMultiEnv


class TerritoryBattleSingleEnv(TerritoryBattleMultiEnv):
    action_space: List
    observation_space: spaces.Dict

    def __init__(self,
                 agent_policies: Tuple[AgentPolicy, ...] = (AgentPolicy(),),
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

        :param agent_policies: The policies that dictate the actions of the other agents. Should be a tuple of length
        1 - agents_init, as it handles every agent except for agent agent_id
        :type agent_policies: Tuple[AgentPolicy, ...]
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
        super().__init__(shape, agents_init, bot_vision, max_ammo, spawn_chance, window_height)

        assert len(agent_policies) == len(agents_init) - 1, 'agent_policies and agents_init must have the same ' \
                                                            'number of agents!'

        # a limited view of the entire grid and a list of bot observation spaces which is initialized with a single bot
        # observation space (one bot to start)
        self.observation_space = spaces.Dict({
            'bots': List([self.bot_observation_space()]),
            'grid': spaces.MultiDiscrete(  # grid array + extra axis (of size 2) to hold the grid view and bot view
                np.full(self.shape, self.n_agents + Cells.AGENT),
            ),
        })

        self.agent_id = agent_id
        self.agent_policies: Tuple[AgentPolicy, ...] = agent_policies
        self.last_obs = None

    @property
    def action_space(self):
        return self._action_space[self.agent_id]

    def reset(self,
              *,
              seed: int | None = None,
              return_info: bool = False,
              options: dict | None = None,
              ) -> Tuple[AgentObs, dict] | AgentObs:
        reset_info = super().reset(seed=seed, return_info=return_info, options=options)

        if return_info:
            self.last_obs: FullObs = reset_info[0]
            return self.last_obs[self.agent_id], reset_info[1]
        else:
            self.last_obs: FullObs = reset_info
            return self.last_obs[self.agent_id]

    def step(self, action: AgentAction) -> Tuple[AgentObs, AgentReward, bool, dict]:
        actions = []  # actions of manual and policy-controlled agents
        for i, policy in enumerate(self.agents):
            agent_policy_id = i
            if i == self.agent_id:  # if we are at the manually-controlled agent
                actions.append(action)
                continue
            elif i > self.agent_id:
                agent_policy_id -= 1

            actions.append(self.agent_policies[agent_policy_id].produce_action(self.last_obs[i], self._action_space[i]))

        observation, reward, done, info = super().step(tuple(actions))

        for i, agent_observation in enumerate(observation):
            agent_policy_id = i
            if i == self.agent_id:  # if we are at the manually-controlled agent
                continue
            elif i > self.agent_id:
                agent_policy_id -= 1

            self.agent_policies[agent_policy_id].process_transition(self.last_obs[i], actions[i], reward[i],
                                                                    observation[i])

        self.last_obs = observation

        return observation[self.agent_id], reward[self.agent_id], done, info
