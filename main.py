import gym
from battlecode.envs import TerritoryBattleMultiEnv, MainActions, TurnActions
# env = gym.make('CartPole-v1')
env = TerritoryBattleMultiEnv()
env.reset()
print(env.grid[:, :, 1])
env.step(([(MainActions.FORWARD, TurnActions.NOOP)], [(MainActions.CHARGE, TurnActions.NOOP)]))
print(env.grid[:, :, 1])
env.step(([(MainActions.FORWARD, TurnActions.NOOP)], [(MainActions.RIGHT, TurnActions.NOOP)]))
print(env.grid[:, :, 1])
env.step(([(MainActions.FORWARD, TurnActions.NOOP)], [(MainActions.FORWARD, TurnActions.NOOP)]))
print(env.grid[:, :, 1])
env.step(([(MainActions.FORWARD, TurnActions.RIGHT)], [(MainActions.CHARGE, TurnActions.NOOP)]))
print(env.grid[:, :, 1])
env.step(([(MainActions.CLAIM, TurnActions.NOOP)], [(MainActions.CHARGE, TurnActions.NOOP)]))
print(env.grid[:, :, 1])
env.step(([(MainActions.FORWARD, TurnActions.NOOP)], [(MainActions.ATTACK, TurnActions.NOOP)]))
print(env.grid[:, :, 1])
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())  # take a random action
env.close()
