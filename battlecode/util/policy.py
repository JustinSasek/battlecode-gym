from . import AgentObs, AgentAction, AgentReward
from ..mutable_spaces import List


class AgentPolicy:
    # action_space should be consistent with observation
    def produce_action(self, obs: AgentObs, action_space: List) -> AgentAction:
        assert len(obs.bots) == len(action_space), 'observation and action space must have the same number of bots!'

        return action_space.sample()

    def process_transition(self, old_obs: AgentObs, action: AgentAction, reward: AgentReward, obs: AgentObs) -> None:
        pass
