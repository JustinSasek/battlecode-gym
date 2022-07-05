import gym
from battlecode.envs import TerritoryBattleMultiEnv, MainActions, TurnActions
# env = gym.make('CartPole-v1')
env = TerritoryBattleMultiEnv()
env.reset(seed=1)
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    # action = ([(MainActions.FORWARD, TurnActions.NOOP)], [(MainActions.FORWARD, TurnActions.NOOP)])
    env.step(action)  # take a random action
env.close()
