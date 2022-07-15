from . import BotPolicy
from ..util import Cells, MainActions, TurnActions, Layers


class ClaimSimplePolicy(BotPolicy):

    @staticmethod
    def is_free_unclaimed(cell):  # check if cell (1-dimensional array) is free to move on to and not claimed by us
        return cell[Layers.GRID] not in (Cells.WALL, Cells.UNKNOWN, Cells.AGENT) and cell[Layers.BOT] == Cells.EMPTY

    @staticmethod
    def forward_avoid_obstacle(obs):  # assumes 3x3xn observation space
        # if the cell in front is available to move to and claim
        if ClaimSimplePolicy.is_free_unclaimed(obs[1, 1]):
            return MainActions.FORWARD, TurnActions.NOOP
        elif ClaimSimplePolicy.is_free_unclaimed(obs[0, 2]):
            return MainActions.LEFT, TurnActions.LEFT
        elif ClaimSimplePolicy.is_free_unclaimed(obs[0, 0]):
            return MainActions.RIGHT, TurnActions.RIGHT
        else:
            return MainActions.BACK, TurnActions.BACK

    def produce_action(self, obs, action_space):
        if obs.view[0, 1, Layers.GRID] != Cells.AGENT:  # if the grid cell we are over is not claimed by us
            return MainActions.CLAIM, TurnActions.NOOP

        return ClaimSimplePolicy.forward_avoid_obstacle(obs.view)
