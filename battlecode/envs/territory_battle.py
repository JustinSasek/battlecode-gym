import gym
from gym import spaces
from .mutable_spaces import List
import numpy as np
from .util_types import *
from copy import deepcopy
from itertools import chain
from typeguard import check_argument_types


class TerritoryBattleMultiEnv(gym.Env):
    DEFAULT_MAX_TIMESTEP = 100
    # rewards per bot:
    # -5 for death
    # 0 for blocking, unsuccessfully attacking, noop, and movement
    # +1 for charging an attack
    # +3 for claiming
    # +10 for kill
    REWARDS = {
        'death': -5,
        'block': 0,
        'miss': 0,
        'noop': 0,
        'miss_movement': 0,
        'movement': 0,
        'charge': 1,
        'claim': 3,
        'kill': 10
    }

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
        self.max_timestep = self.DEFAULT_MAX_TIMESTEP
        self.timestep = 0

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

    def _valid_cell(self, pos: Sequence[int, int] | NDArray[int]) -> bool:  # if 2d point is in bounds of the grid
        for axis in range(2):
            if not 0 <= pos[axis] < self.grid.shape[axis]:  # if out of bounds in some axis
                return False
        return self.grid[pos][Layers.GRID] != Cells.WALL

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

        self.max_timestep = options['max_timestep']if isinstance(options, dict) and 'max_timestep' in options else \
            self.DEFAULT_MAX_TIMESTEP

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def step(self, action: FullAction) -> Tuple[FullObs, FullReward, bool, dict]:
        assert check_argument_types(), \
            'action must be a tuple of agent actions, where each agent action is a sequence of bot actions'

        # assert there is an action for each agent
        assert len(action) == len(self.agents), 'there must be a set of bot actions for every agent'

        # assert there is an action for each bot
        for agent_id, (agent_action, agent) in enumerate(zip(action, self.agents)):
            assert len(agent_action) == len(agent.bots), \
                f'there must be an action for every bot, agent {agent_id} does not have the correct amount of actions'

        # rewards per bot:
        # -5 for death
        # 0 for blocking, unsuccessfully attacking, noop, and movement
        # +1 for charging an attack
        # +3 for claiming
        # +10 for kill

        # steps are processed in the following order:
        # block/charge
        # attack
        # unblock/claim/noop
        # movement
        # turning
        # bot creation
        # observation

        reward = tuple([0] * len(agent.bots) for agent in self.agents)

        # group actions by main action types
        agent_actions = {action_type: [] for action_type in MainActions}
        for agent_id, agent_action in enumerate(action):
            for bot_id, (main_action, turn_action) in enumerate(agent_action):
                agent_actions[main_action].append(ActionRepr(agent_id, bot_id, main_action))

        # block/charge - just apply block/charge actions
        for block in agent_actions[MainActions.BLOCK]:
            bot = self.agents[block.agent_id].bots[block.bot_id]
            bot.block = True  # bot is now blocking
            reward[bot.agent_id][bot.id] += self.REWARDS['block']
        for charge in agent_actions[MainActions.CHARGE]:
            bot = self.agents[charge.agent_id].bots[charge.bot_id]
            if bot.ammo < self.max_ammo:
                bot.ammo += 1
                reward[bot.agent_id][bot.id] += self.REWARDS['charge']

        # attack - go through each action and store bots that get successfully attacked, then delete them
        attack_targets = []  # positions that get attacked along with their attackers
        for attack in agent_actions[MainActions.ATTACK]:
            bot = self.agents[attack.agent_id].bots[attack.bot_id]
            attack_pos = tuple(np.array(bot.pos) + np.array(bot.rot))  # cell to attack
            if self._valid_cell(attack_pos) and bot.ammo > 0:
                attack_targets.append((attack_pos, bot,))  # bot at attack position is target
            else:  # if not valid attack
                reward[bot.agent_id][bot.id] += self.REWARDS['miss']
        for attack_pos, attacker in attack_targets:  # delete attacked bots if they are not blocking
            if attack_pos in self.position_bots and not self.position_bots[attack_pos].block:
                bot = self.position_bots[attack_pos]
                del self.position_bots[attack_pos]  # delete position-bot mapping
                del self.agents[bot.agent_id].bots[bot.id]  # delete bot from agent
                self.grid[attack_pos][Layers.BOT] = Cells.EMPTY  # delete killed bot from grid
                for i, bot_action in enumerate(agent_actions[action[bot.agent_id][bot.id][0]]):  # search for killed bot
                    if bot_action.agent_id == bot.agent_id and bot_action.bot_id == bot.id:  # when killed bot found
                        del agent_actions[action[bot.agent_id][bot.id][0]][i]  # delete killed bot's action (1)
                del action[bot.agent_id][bot.id]  # delete killed bot's action (2)
                reward[attacker.agent_id][attacker.id] += self.REWARDS['kill']
            else:  # if no bot there or they were blocking
                reward[attacker.agent_id][attacker.id] += self.REWARDS['miss']

        # unblock/claim/noop - unblock if bot is blocking and claim territory/noop
        for block in agent_actions[MainActions.BLOCK]:
            self.agents[block.agent_id].bots[block.bot_id].block = False  # bot is no longer blocking
        for claim in agent_actions[MainActions.CLAIM]:
            bot = self.agents[claim.agent_id].bots[claim.bot_id]
            self.grid[bot.pos][Layers.GRID] = claim.agent_id + self.n_default_cells  # grid cell now belongs to agent
            reward[bot.agent_id][bot.id] += self.REWARDS['claim']
        for noop in agent_actions[MainActions.NOOP]:
            bot = self.agents[noop.agent_id].bots[noop.bot_id]
            reward[bot.agent_id][bot.id] += self.REWARDS['noop']

        # movement - move if able to
        pending_movements = []
        for movement in chain(*[agent_actions[movement] for movement in
                                [MainActions.FORWARD, MainActions.LEFT, MainActions.RIGHT, MainActions.BACK]]):
            bot = self.agents[movement.agent_id].bots[movement.bot_id]
            movement_rot = movement.main_action - MainActions.FORWARD
            new_relative_pos = np.fromiter((
                (-1 if bot.rot[axis] < 0 else 1) * ((bot.rot[axis] + movement_rot) % 2) * (1 - 2 * (movement_rot // 2))
                for axis in range(2)
            ), dtype=int)
            new_pos = tuple(np.array(bot.pos) + new_relative_pos)  # cell to move to
            if self._valid_cell(new_pos) and self.grid[new_pos][Layers.BOT] == Cells.EMPTY:
                pending_movements.append((  # move the bot if the cell it wants to move to is valid (empty, in bounds)
                    bot,
                    new_pos,
                ))
            else:  # if the bot cannot move here
                reward[bot.agent_id][bot.id] += self.REWARDS['miss_movement']
        for bot, new_pos in pending_movements:
            self.grid[bot.pos][Layers.BOT] = Cells.EMPTY
            self.grid[new_pos][Layers.BOT] = self.n_default_cells + bot.agent_id  # update world grid
            del self.position_bots[bot.pos]
            self.position_bots[new_pos] = bot  # update position-bot mapping
            bot.pos = new_pos  # update bot pos property
            reward[bot.agent_id][bot.id] += self.REWARDS['movement']

        # turning
        for agent_id, agent_action in enumerate(action):
            for bot_id, (main_action, turn_rot) in enumerate(agent_action):
                bot = self.agents[agent_id].bots[bot_id]
                bot.rot = tuple(np.fromiter((
                    bot.rot[axis] * ((bot.rot[axis] + turn_rot) % 2) * (1 - 2 * (turn_rot // 2))
                    for axis in range(2)
                ), dtype=int))

        # bot creation

        # observation
        self.timestep += 1
        done = self.timestep >= self.max_timestep  # end if time has run out
        if not done:
            cell_types = list(np.unique(self.grid))  # get unique cell types (excluding default types)
            for default_cell in Cells:
                if default_cell in cell_types:
                    cell_types.remove(default_cell)
            done = len(cell_types) <= 1   # end if there is only one kind of agent territory left

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info


a = TerritoryBattleMultiEnv()
a.reset()
# print(a.grid[:, :, 1])
# print(a._get_obs()[0].grid[:, :, 0])
