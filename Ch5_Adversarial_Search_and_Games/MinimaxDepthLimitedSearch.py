"""Minimax search with depth limiting for adversarial games."""

import math


def minmax_decision_depth_limited(state, game, depth_limit=6, use_heuristic=True):
    """Given a state in a game, calculate the best move by searching
    forward to a limited depth using minimax.
    
    Args:
        state: The current game state
        game: The game object
        depth_limit: Maximum depth to search (default 6)
        use_heuristic: Whether to use heuristic evaluation at cutoff (default True)
    """

    player = game.to_move(state)

    def max_value(state, depth):
        if game.terminal_test(state):
            return game.utility(state, player)
        if depth == 0:
            if use_heuristic:
                evaluate = getattr(game, "evaluate", None)
                return evaluate(state, player) if evaluate else game.utility(state, player)
            else:
                return game.utility(state, player)
        v = -math.inf
        for action in game.actions(state):
            v = max(v, min_value(game.result(state, action), depth - 1))
        return v

    def min_value(state, depth):
        if game.terminal_test(state):
            return game.utility(state, player)
        if depth == 0:
            if use_heuristic:
                evaluate = getattr(game, "evaluate", None)
                return evaluate(state, player) if evaluate else game.utility(state, player)
            else:
                return game.utility(state, player)
        v = math.inf
        for action in game.actions(state):
            v = min(v, max_value(game.result(state, action), depth - 1))
        return v

    # Body of minmax_decision_depth_limited:
    return max(
        game.actions(state),
        key=lambda action: min_value(game.result(state, action), depth_limit - 1),
    )
