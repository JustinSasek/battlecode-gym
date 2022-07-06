from gym.envs.registration import register

register(
    id='battlecode/TerritoryBattle-v0',
    entry_point='battlecode.envs:TerritoryBattleMultiEnv',
    nondeterministic=True,
)
