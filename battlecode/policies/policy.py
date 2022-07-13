from battlecode.util import AgentObs, AgentAction, AgentReward, RewardType, BotAction, BotReward
from battlecode.mutable_spaces import List
from gym.core import ObsType, ActType
from gym.spaces import MultiDiscrete
from numpy.typing import NDArray


class Policy:
    def produce_action(self, obs: ObsType, action_space: ActType) -> ActType:
        """
        Method for a policy to produce an action. By default, it returns a random action.

        @param obs: The observation to base the action off of
        @param action_space: The space of valid actions to take
        @return: Valid action according to action_space
        """
        return action_space.sample()

    def process_transition(self, old_obs: ObsType, action: ActType, reward: RewardType, obs: ObsType) -> None:
        """
        Method for a policy to process a transition (S, A, R, S') after it takes an action.

        @param old_obs: S, starting state
        @param action: A, action taken at starting state
        @param reward: R, reward given for performing A at S
        @param obs: S', new state after performing A at S
        @return: None
        """
        pass


class BotPolicy(Policy):
    def produce_action(self, obs: NDArray, action_space: MultiDiscrete) -> BotAction:
        return action_space.sample()

    def process_transition(self, old_obs: NDArray, action: BotAction, reward: BotReward, obs: NDArray) -> None:
        pass


class AgentPolicy(Policy):
    def __init__(self, bot_policy: BotPolicy | None = None) -> None:
        """
        Constructor that optionally takes bot policy.

        @param bot_policy: If supplied, the agent policy will apply this bot policy to each bot independently.
        """
        self.bot_policy: BotPolicy | None = bot_policy

    # action_space should be consistent with observation
    def produce_action(self, obs: AgentObs, action_space: List) -> AgentAction:
        assert len(obs.bots) == len(action_space), 'observation and action space must have the same number of bots!'

        if self.bot_policy is None:  # if we have no bot policy, produce random action
            return super().produce_action(obs, action_space)

        action = []  # if we have bot policy, control each bot according to it
        for i, bot_action_space in enumerate(action_space):
            action.append(self.bot_policy.produce_action(obs.bots[i], action_space[i]))

        return action

    def process_transition(self, old_obs: AgentObs, action: AgentAction, reward: AgentReward, obs: AgentObs) -> None:
        if self.bot_policy is None:
            return

        for i, bot_reward in enumerate(reward):  # if we have bot policy, process the transition for every bot
            if bot_reward < 0:  # if this bot died
                del old_obs.bots[i]

        # loops through each bot that existed before and after this transition and processes it
        for old_bot_obs, bot_action, bot_reward, bot_obs in zip(old_obs.bots, action, reward, obs.bots):
            self.bot_policy.process_transition(old_bot_obs, bot_action, bot_reward, bot_obs)
