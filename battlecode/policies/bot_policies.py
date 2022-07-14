from . import BotPolicy
from ..util import Cells, MainActions, TurnActions, Layers

RandomPolicy = BotPolicy


def forward_avoid_obstacle(obs):  # assumes 3x3xn observation space
    # if the cell in front is available to move to
    if is_free(obs[1, 1]):
        return MainActions.FORWARD, TurnActions.NOOP
    elif is_free(obs[0, 2]):
        return MainActions.LEFT, TurnActions.LEFT
    elif is_free(obs[0, 0]):
        return MainActions.RIGHT, TurnActions.RIGHT
    else:
        return MainActions.BACK, TurnActions.BACK


def is_free(cell):  # check if cell (1-dimensional array) is free to move on to
    return cell[Layers.GRID] not in (Cells.WALL, Cells.UNKNOWN, Cells.AGENT) and cell[Layers.BOT] == Cells.EMPTY


class ClaimSimplePolicy(BotPolicy):
    def produce_action(self, obs, action_space):
        if obs.view[0, 1, Layers.GRID] != Cells.AGENT:  # if the grid cell we are over is not claimed by us
            return MainActions.CLAIM, TurnActions.NOOP

        return forward_avoid_obstacle(obs.view)
