from . import AgentObs, AgentAction, AgentReward, RewardType, BotAction, BotReward
from ..mutable_spaces import List
from gym.core import ObsType, ActType
from gym.spaces import MultiDiscrete
from numpy.typing import NDArray


class Policy:
    def produce_action(self, obs: ObsType, action_space: ActType) -> ActType:
        return action_space.sample()

    def process_transition(self, old_obs: ObsType, action: ActType, reward: RewardType, obs: ObsType) -> None:
        pass


class AgentPolicy(Policy):  # TODO: create constructor which takes a bot policy
    # action_space should be consistent with observation
    def produce_action(self, obs: AgentObs, action_space: List) -> AgentAction:
        assert len(obs.bots) == len(action_space), 'observation and action space must have the same number of bots!'

        return super().produce_action(obs, action_space)

    def process_transition(self, old_obs: AgentObs, action: AgentAction, reward: AgentReward, obs: AgentObs) -> None:
        pass


class BotPolicy(Policy):
    def produce_action(self, obs: NDArray, action_space: MultiDiscrete) -> BotAction:
        return action_space.sample()

    def process_transition(self, old_obs: NDArray, action: BotAction, reward: BotReward, obs: NDArray) -> None:
        pass
