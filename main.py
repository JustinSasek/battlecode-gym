import gym
from battlecode.envs import *
from battlecode.util import *
# env = gym.make('CartPole-v1')
other_agent_policy = AgentPolicy()
env = TerritoryBattleSingleEnv((other_agent_policy,))
env.reset(seed=1)
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    # action = ([(MainActions.FORWARD, TurnActions.NOOP)], [(MainActions.FORWARD, TurnActions.NOOP)])
    env.step(action)  # take a random action
env.close()
