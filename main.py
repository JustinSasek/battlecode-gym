import gym
from battlecode.envs import *
from battlecode.policies import *
from battlecode.util import *
# env = gym.make('CartPole-v1')
# other_bot_policy = BotPolicy()
env = TerritoryBattleBotEnv((ClaimSimplePolicy(),))
policy = ClaimSimplePolicy()
obs = env.reset()
for _ in range(1000):
    env.render()
    # action = env.action_space.sample()
    action = policy.produce_action(obs, env.action_space)
    # action = ([(MainActions.FORWARD, TurnActions.NOOP)], [(MainActions.FORWARD, TurnActions.NOOP)])
    # action = (MainActions.RIGHT, TurnActions.RIGHT)
    obs, reward, done, info = env.step(action)  # take a random action
env.close()
