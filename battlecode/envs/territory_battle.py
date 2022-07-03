import gym
from gym import spaces
from spaces import List
import numpy as np
from util_types import *
from copy import deepcopy


class TerritoryBattleMultiEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 4
    }
    n_layers = len(Layers)
    n_default_cells = len(Cells)

    def __init__(self,
                 shape: int | Tuple[int, int] = (7, 5),
                 agents_init: Tuple[Bot] = (Bot((0, 2), (1, 0)), Bot((6, 2), (-1, 0))),
                 bot_vision: int | Tuple[int, int] = (3, 3),
                 max_ammo: int = 3,
                 window_height: int = 560,
                 ) -> None:
        """
        Create 2d grid world based on shape and spawn agents at given start points.

        :param shape: The 2d shape of the environment.
        :type shape: int or Tuple[int, int], optional
        :param agents_init: An n-tuple of 2-tuples, creates n agents with bots at given 2-tuple spawn point.
        :type agents_init: Tuple[Bot], optional
        :param bot_vision: Shape that represents the area the bot can see ahead of itself.
        :type bot_vision: int or Tuple[int, int], optional
        :param window_height: Height of the pygame window in human-mode.
        :type window_height: int, optional
        """
        super(TerritoryBattleMultiEnv, self).__init__()
        if isinstance(shape, int):
            shape = (shape, shape)
        if isinstance(bot_vision, int):
            bot_vision = (bot_vision, bot_vision)

        self.window_height = window_height
        self.bot_vision = bot_vision + (self.n_layers,)
        self.max_ammo = max_ammo
        self.shape = shape + (self.n_layers,)

        self.n_agents = len(agents_init)

        self.grid = np.empty(self.shape, dtype=np.int32)
        self.agents_init = deepcopy(agents_init)
        self.agents = []
        for i, bot in enumerate(agents_init):
            bot_container = Bots()
            self.agents.append(Agent(bot_container, i))
        self.agents = tuple(self.agents)
        self.position_bots = {}  # mapping from Tuple(int, int) positions to bots

        # tuple of action spaces for each agent, where each agent's action space is a list of bot action spaces
        # which is initialized with a single bot action space (one bot to start)
        self.action_space = spaces.Tuple(tuple(List([self.bot_action_space()]) for _ in range(self.n_agents)))

        # tuple of observation spaces for each agent, where each agent's observation space is a dict that includes a
        # limited view of the entire grid and a list of bot observation spaces which is initialized with a single bot
        # observation space (one bot to start)
        self.observation_space = spaces.Tuple(tuple(spaces.Dict({
            'bots': List([self.bot_observation_space()]),
            'grid': spaces.MultiDiscrete(  # grid array + extra axis (of size 2) to hold the grid view and bot view
                np.full(self.shape, self.n_agents + self.n_default_cells),
            ),
        }) for _ in range(self.n_agents)))

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct frame rate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    @staticmethod
    def bot_action_space():  # each bot can do a main action and a turn action right after the main
        return spaces.MultiDiscrete([len(MainActions), len(TurnActions)])

    # each bot can see a 3x3 in front and a bit to left and right. last axis is for grid/bot view
    def bot_observation_space(self):
        return spaces.MultiDiscrete(np.full(self.bot_vision, self.n_agents + self.n_default_cells))

    def _get_obs(self) -> FullObs:
        agent_observations = []
        for agent in self.agents:
            agent_view = np.full_like(self.grid, Cells.UNKNOWN)  # all unknown by default
            agent_observation = AgentObs([], agent_view)

            for bot in agent.bots:
                view_size = tuple(self.bot_vision[:2][bot.rot[1 - i]] for i in range(2))
                view_offset = tuple((view_size_axis - 1) // 2 for view_size_axis in view_size)
                view_pos = tuple(bot.pos[i] + view_offset[i] * (bot.rot[i] - 1) for i in range(2))

                bot_view_global = np.full(view_size + (self.n_layers,), Cells.UNKNOWN)  # bot view in global coords

                grid_intersect = (  # area of intersection between global grid and bot view, global perspective
                    slice(max(0, view_pos[0]), min(self.grid.shape[0], view_pos[0] + view_size[0])),
                    slice(max(0, view_pos[1]), min(self.grid.shape[1], view_pos[1] + view_size[1]))
                )
                view_intersect = (  # area of intersection between global grid and bot view, local perspective
                    slice(max(0, -view_pos[0]),
                          view_size[0] + min(0, self.grid.shape[0] - (view_pos[0] + view_size[0]))),
                    slice(max(0, -view_pos[1]),
                          view_size[1] + min(0, self.grid.shape[1] - (view_pos[1] + view_size[1]))),
                )

                bot_view_global[view_intersect[0], view_intersect[1]] = \
                    self.grid[grid_intersect[0], grid_intersect[1]]

                n_rotations = abs(bot.rot[0] + 2 * bot.rot[1] - 1)  # how to rotate from world to local coordinates
                bot_view = np.rot90(bot_view_global, n_rotations)  # relative bot view

                # very crude shadow casting time, just cast shadows vertically from perspective of bot
                # if you want to take the time to rly make this accurate, have at it:
                # https://ir.lib.uwo.ca/cgi/viewcontent.cgi?article=8883&context=etd
                for axis_0 in bot_view:
                    for j in range(bot_view.shape[1] - 1):  # walls in last layer cannot cast shadows
                        if axis_0[j, 0] == Cells.WALL:
                            axis_0[j + 1:] = Cells.UNKNOWN  # every block after this one in 1-axis is unknown

                bot_view_global = np.rot90(bot_view, -n_rotations)  # sending shadow casting result back to world
                agent_view[grid_intersect[0], grid_intersect[1]] = \
                    bot_view_global[view_intersect[0], view_intersect[1]]

                agent_observation.bots.append(bot_view)
            agent_observations.append(agent_observation)

        return tuple(agent_observations)

    def _get_info(self) -> dict:
        return {
            'grid': self.grid
        }

    def reset(self,
              *,
              seed: int | None = None,
              return_info: bool = False,
              options: dict | None = None,
              ) -> FullObs | Tuple[FullObs, dict]:
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.grid.fill(Cells.EMPTY)  # reset grid
        self.position_bots.clear()  # reset position-bot mapping

        for i, bot_init in enumerate(self.agents_init):
            self.agents[i].bots.clear()  # remove all bots and add in initial bot
            starting_bot = deepcopy(bot_init)
            self.agents[i].bots.append(starting_bot)
            self.grid[bot_init.pos] = self.n_default_cells + i  # fill in all layers with the bot id at its spawn
            self.position_bots[starting_bot.pos] = starting_bot  # update

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action: FullAction) -> Tuple[FullObs, FullReward, bool, dict]:
        assert isinstance(spaces, FullAction), \
            'action must be a tuple of agent actions, where each agent action is a collection of bot actions'

        # assert there is an action for each agent
        assert len(action) == len(self.agents), 'there must be a set of bot actions for every agent'

        # assert there is an action for each bot
        for agent_id, agent_action, agent in enumerate(zip(action, self.agents)):
            assert len(agent_action) == len(agent.bots), \
                f'there must be an action for every bot, agent {agent_id} does not have the correct amount of actions'

        # steps are processed in the following order:
        # block/charge
        # attack
        # unblock/claim/noop
        # movement
        # turning
        # bot creation
        # observation

        # block/charge
        for agent_id, agent_action in enumerate(action):
            for bot_id, bot_action in enumerate(agent_action):
                match bot_action[0]:
                    case MainActions.BLOCK:
                        self.agents[agent_id].bots[bot_id].block = True  # bot is now blocking
                    case MainActions.CHARGE:
                        bot = self.agents[agent_id].bots[bot_id]
                        if bot.ammo < self.max_ammo:
                            bot.ammo += 1

        # attack
        attack_targets = []  # bot positions that get attacked
        for agent_id, agent_action in enumerate(action):
            for bot_id, bot_action in enumerate(agent_action):
                match bot_action[0]:
                    case MainActions.ATTACK:
                        bot = self.agents[agent_id].bots[bot_id]
                        attack_pos = (bot.pos[0] + bot.rot[0], bot.pos[1] + bot.rot[1])  # cell to attack

                        for axis in range(2):
                            if not 0 <= attack_pos[axis] < self.grid.shape[axis]:
                                break
                        else:  # only runs if loop ends normally, meaning attack_pos is within bounds
                            if attack_pos in self.bots:
                                attack_targets.append(attack_pos)
        for attack_pos in attack_targets:
            bot = self.position_bots[attack_pos]
            del self.position_bots[attack_pos]
            del self.agents[bot.agent_id].bots[bot.id]
        # make sure to update self.position_bots

        done = False
        reward = ([1], [1])
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info


a = TerritoryBattleMultiEnv()
a.reset()
# print(a.grid[:, :, 1])
print(a._get_obs()[0].grid[:, :, 0])
