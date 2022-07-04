import gym
from battlecode.envs import TerritoryBattleMultiEnv, MainActions, TurnActions
# env = gym.make('CartPole-v1')
env = TerritoryBattleMultiEnv()
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())  # take a random action
env.close()
