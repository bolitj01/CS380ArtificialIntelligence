"""Minimax search implementation for adversarial games."""

import math


def minmax_decision(state, game):
    """Given a state in a game, calculate the best move by searching
    forward all the way to the terminal states. [Figure 5.3]"""

    player = game.to_move(state)

    def max_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -math.inf
        for action in game.actions(state):
            v = max(v, min_value(game.result(state, action)))
        return v

    def min_value(state):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = math.inf
        for action in game.actions(state):
            v = min(v, max_value(game.result(state, action)))
        return v

    # Body of minmax_decision:
    return max(game.actions(state), key=lambda action: min_value(game.result(state, action)))
