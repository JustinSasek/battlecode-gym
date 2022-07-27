from . import BotPolicy
from ..util import Cells, MainActions, TurnActions, Layers


class OffensivePolicy(BotPolicy):
    def produce_action(self, obs, action_space):
        if obs.ammo > 0:  # if we can attack
            if obs.view[1, 1, Layers.BOT] > Cells.AGENT:
                return MainActions.ATTACK, TurnActions.NOOP

            if obs.view[0, 0, Layers.BOT] > Cells.AGENT:
                return MainActions.BLOCK, TurnActions.RIGHT

            if obs.view[0, 2, Layers.BOT] > Cells.AGENT:
                return MainActions.BLOCK, TurnActions.LEFT

        if obs.ammo < 3:
            return MainActions.CHARGE, TurnActions.NOOP

        if obs.view[0, 1, Layers.GRID] != Cells.AGENT:  # if this spot is unclaimed
            return MainActions.CLAIM, TurnActions.NOOP

        if obs.view[1, 1, Layers.GRID] == Cells.WALL:
            return MainActions.LEFT, TurnActions.LEFT

        return MainActions.FORWARD, TurnActions.NOOP
