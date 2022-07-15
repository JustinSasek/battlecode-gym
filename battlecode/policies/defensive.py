from . import BotPolicy
from ..util import Cells, MainActions, TurnActions, Layers


class DefensivePolicy(BotPolicy):
    @staticmethod
    def is_free_unclaimed(cell):  # check if cell (1-dimensional array) is free to move on to
        return cell[Layers.GRID] not in (Cells.WALL, Cells.UNKNOWN) and cell[Layers.BOT] == Cells.EMPTY

    @staticmethod
    def immediate_danger(obs):
        for cell in ((0, 0), (0, 2), (1, 0)):  # check adjacent cells
            if obs.view[cell][Layers.BOT] > Cells.AGENT:  # if there is an enemy on this cell
                return True

        return False

    def produce_action(self, obs, action_space):
        if DefensivePolicy.immediate_danger(obs):  # if there is an enemy next to us
            return MainActions.BLOCK, TurnActions.LEFT

        if obs.ammo < 3:
            return MainActions.CHARGE, TurnActions.LEFT

        if obs.view[0, 1, Layers.GRID] == Cells.AGENT:  # if this spot is unclaimed
            return MainActions.CLAIM, TurnActions.LEFT

        return MainActions.FORWARD, TurnActions.LEFT
