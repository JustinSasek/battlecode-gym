import gym
from battlecode.envs import *
from battlecode.policies import *
from battlecode.util import *
# env = gym.make('CartPole-v1')
# other_bot_policy = BotPolicy()
env = TerritoryBattleBotEnv((DefensivePolicy(),))  #environment
policy = OffensivePolicy()
obs = env.reset()
for _ in range(1000):
    env.render()  #render environment
    print(obs) # prints input
    bot_in_front = obs.view[1, 1, Layers.BOT]
    if bot_in_front > Cells.AGENT:
        action = (MainActions.ATTACK, TurnActions.NOOP)
    else:
        action = (MainActions.CHARGE, TurnActions.NOOP)
    #3d grid
    #x(reversed) and y and bot/color layer
    #main - noop - nothing
    #forware, right, back, left
    #attack, charge, clock, claim
    #goal is to claim as much land as possible

    #turn - left, right, 
    env.action_space.sample() # samples random action space from possible action
    # action = env.action_space.sample()
    # action = policy.produce_action(obs, env.action_space)
    # action = ([(MainActions.FORWARD, TurnActions.NOOP)], [(MainActions.FORWARD, TurnActions.NOOP)])
    # action = (MainActions.RIGHT, TurnActions.RIGHT)
    obs, reward, done, info = env.step(action)  # take a random action

    # records observation, reward, state
env.close()
