from gym import spaces
from .mutable_spaces import List
import numpy as np
from .util_types import *
from copy import deepcopy
from itertools import chain
from typeguard import check_argument_types
from .territory_battle import TerritoryBattleMultiEnv
import pygame


class TerritoryBattleSingleEnv(TerritoryBattleMultiEnv):
    action_space: List
    observation_space: spaces.Dict

    def __init__(self,
                 shape: int | Tuple[int, int] = (7, 5),
                 agents_init: Tuple[Bot, ...] = (Bot((0, 2), (1, 0)), Bot((6, 2), (-1, 0))),
                 bot_vision: int | Tuple[int, int] = (3, 3),
                 max_ammo: int = 3,
                 spawn_chance: float = 0.01,
                 window_height: int = 560,
                 ) -> None:
        """
        Create 2d grid world based on shape and spawn agents at given start points.

        :param shape: The 2d shape of the environment.
        :type shape: int or Tuple[int, int], optional
        :param agents_init: An n-tuple of 2-tuples, creates n agents with bots at given 2-tuple spawn point.
        :type agents_init: Tuple[Bot, ...], optional
        :param bot_vision: Shape that represents the area the bot can see ahead of itself.
        :type bot_vision: int or Tuple[int, int], optional
        :param max_ammo: Max ammo that a bot can have to attack with.
        :type max_ammo: int, optional
        :param spawn_chance: Probability of a given claimed territory to spawn a bot on a given tick. For some reason
        this breaks the reproducibility of np.random_seed when it's nonzero.
        :type spawn_chance: float, optional
        :param window_height: Height of the pygame window in human-mode.
        :type window_height: int, optional
        """
        super(TerritoryBattleMultiEnv, self).__init__(shape, agents_init, bot_vision,
                                                      max_ammo, spawn_chance, window_height)
        # a list of bot action spaces
        self.action_space = List([])

        # a limited view of the entire grid and a list of bot observation spaces which is initialized with a single bot
        # observation space (one bot to start)
        self.observation_space = spaces.Dict({
            'bots': List([self.bot_observation_space()]),
            'grid': spaces.MultiDiscrete(  # grid array + extra axis (of size 2) to hold the grid view and bot view
                np.full(self.shape, self.n_agents + self.n_default_cells),
            ),
        })

    def _get_obs(self) -> AgentObs:
        return super()._get_obs()[0]

    def reset(self,
              *,
              seed: int | None = None,
              return_info: bool = False,
              options: dict | None = None,
              ) -> FullObs | Tuple[FullObs, dict]:
        return_info = super().reset(seed=seed, return_info=return_info, options=options)

        # reset action space
        self.action_space: List[spaces.MultiDiscrete] = self.agent_action_space(0, seed)

        return return_info

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
                agent_actions[main_action].append(ActionRepr(self.agents[agent_id].bots[bot_id], main_action))

        # block/charge - just apply block/charge actions
        for block in agent_actions[MainActions.BLOCK]:
            bot = block.bot
            bot.block = True  # bot is now blocking
            reward[bot.agent_id][bot.id] += self.REWARDS['block']
        for charge in agent_actions[MainActions.CHARGE]:
            bot = charge.bot
            if bot.ammo < self.max_ammo:
                bot.ammo += 1
                reward[bot.agent_id][bot.id] += self.REWARDS['charge']

        # attack - go through each action and store bots that get successfully attacked, then delete them
        attack_targets = []  # positions that get attacked along with their attackers
        for attack in agent_actions[MainActions.ATTACK]:
            bot = attack.bot
            attack_pos = tuple(np.array(bot.pos) + np.array(bot.rot))  # cell to attack
            if self._valid_cell(attack_pos) and bot.ammo > 0:
                attack_targets.append((attack_pos, bot,))  # bot at attack position is target
            else:  # if not valid attack
                reward[bot.agent_id][bot.id] += self.REWARDS['fail_attack']
        for attack_pos, attacker in attack_targets:  # delete attacked bots if they are not blocking
            if attack_pos in self.position_bots and not self.position_bots[attack_pos].block:
                bot = self.position_bots[attack_pos]
                del self.action_space[bot.agent_id][bot.id]  # delete bot from action space
                del self.position_bots[attack_pos]  # delete position-bot mapping
                del self.agents[bot.agent_id].bots[bot.id]  # delete bot from agent
                self.grid[attack_pos][Layers.BOT] = Cells.EMPTY  # delete killed bot from grid
                for i, bot_action in enumerate(agent_actions[action[bot.agent_id][bot.id][0]]):  # search for killed bot
                    if bot_action.bot == bot:  # when killed bot found
                        del agent_actions[action[bot.agent_id][bot.id][0]][i]  # delete killed bot's action (1)
                del action[bot.agent_id][bot.id]  # delete killed bot's action (2)
                reward[attacker.agent_id][attacker.id] += self.REWARDS['kill']
            else:  # if no bot there or they were blocking
                reward[attacker.agent_id][attacker.id] += self.REWARDS['fail_attack']

        # unblock/claim/noop - unblock if bot is blocking and claim territory/noop
        for block in agent_actions[MainActions.BLOCK]:
            self.agents[block.bot.agent_id].bots[block.bot.id].block = False  # bot is no longer blocking
        for claim in agent_actions[MainActions.CLAIM]:
            bot = claim.bot
            if self.grid[bot.pos][Layers.GRID] == bot.agent_id + self.n_default_cells:  # if already claimed
                reward[bot.agent_id][bot.id] += self.REWARDS['fail_claim']
            else:  # if not claimed yet, we can claim
                self.grid[bot.pos][Layers.GRID] = bot.agent_id + self.n_default_cells  # claim grid cell
                reward[bot.agent_id][bot.id] += self.REWARDS['claim']
        for noop in agent_actions[MainActions.NOOP]:
            bot = noop.bot
            reward[bot.agent_id][bot.id] += self.REWARDS['noop']

        # movement - move if able to
        pending_movements = {}  # tracks which bots want to go to which positions
        for movement in chain(*[agent_actions[movement] for movement in
                                [MainActions.FORWARD, MainActions.LEFT, MainActions.RIGHT, MainActions.BACK]]):
            bot = movement.bot
            movement_rot = movement.main_action - MainActions.FORWARD
            new_relative_pos = np.array(self._new_relative_rot(bot, movement_rot), dtype=int)
            new_pos = tuple(np.array(bot.pos) + new_relative_pos)  # cell to move to
            if self._valid_cell(new_pos) and self.grid[new_pos][Layers.BOT] == Cells.EMPTY:
                if new_pos in pending_movements:  # move the bot if the cell it wants to move to is valid
                    pending_movements[new_pos].append(bot)
                else:
                    pending_movements[new_pos] = [bot]
            else:  # if the bot cannot move here
                reward[bot.agent_id][bot.id] += self.REWARDS['fail_movement']
        for new_pos, bots in pending_movements.items():
            bot = bots[self.np_random.integers(len(bots))]  # pick a random bot to proceed to the spot
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
                bot.rot = tuple(np.array(self._new_relative_rot(bot, turn_rot), dtype=int))

        # bot creation
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                if cell[Layers.GRID] >= self.n_default_cells and cell[Layers.BOT] == Cells.EMPTY:
                    agent_id = cell[Layers.GRID] - self.n_default_cells
                    rand = self.np_random.uniform(low=0, high=1, size=1)  # if cell is claimed and no bot is on it
                    if rand < self.spawn_chance:  # if we get lucky and get to spawn a bot here
                        bot = Bot(pos=(i, j), rot=self.agents_init[agent_id].rot)
                        self.action_space[agent_id].append(self.bot_action_space())  # add to action space
                        self.position_bots[(i, j)] = bot  # add position-bot mapping
                        self.agents[agent_id].bots.append(bot)  # add bot to agent
                        self.grid[i, j, Layers.BOT] = self.n_default_cells + agent_id  # add bot to grid

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

    def render(self, mode: str = 'human') -> NDArray | None:
        if self.window is None and mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
        if self.clock is None and mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))

        # first we draw in the cells
        for i, row in enumerate(self.grid[:, :, Layers.GRID]):
            for _j, cell in enumerate(row):
                j = self.grid.shape[1] - _j - 1  # flip y axis
                if cell != Cells.EMPTY:
                    pygame.draw.rect(
                        canvas,
                        self.COLORS[cell],
                        pygame.Rect(
                            (i * self.cell_width, j * self.cell_width),
                            self.cell_size,
                        ),
                    )
        # Now we draw the agent
        for agent in self.agents:
            for bot in agent.bots:
                # draw directional line
                pygame.draw.line(
                    canvas,
                    self.BOT_OUTLINE['color'],
                    ((bot.pos[0] + 0.5) * self.cell_width, (self.grid.shape[1] - bot.pos[1] - 0.5) * self.cell_width),
                    ((bot.pos[0] + 0.5 + bot.rot[0]/2) * self.cell_width,
                     (self.grid.shape[1] - bot.pos[1] - 0.5 - bot.rot[1]/2) * self.cell_width),
                    width=int(self.cell_width * self.BOT_OUTLINE['width'])
                )
                # draw circle
                pygame.draw.circle(
                    canvas,
                    self.COLORS[bot.agent_id + self.n_default_cells],
                    ((bot.pos[0] + 0.5) * self.cell_width, (self.grid.shape[1] - bot.pos[1] - 0.5) * self.cell_width),
                    self.cell_width // 3
                )
                # draw outline
                pygame.draw.circle(
                    canvas,
                    self.BOT_OUTLINE['color'],
                    ((bot.pos[0] + 0.5) * self.cell_width, (self.grid.shape[1] - bot.pos[1] - 0.5) * self.cell_width),
                    self.cell_width // 3,
                    width=int(self.cell_width * self.BOT_OUTLINE['width'])
                )

        if mode == 'human':
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined frame rate.
            # The following line will automatically add a delay to keep the frame rate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


a = TerritoryBattleMultiEnv()
a.reset()
# print(a.grid[:, :, 1])
# print(a._get_obs()[0].grid[:, :, 0])
