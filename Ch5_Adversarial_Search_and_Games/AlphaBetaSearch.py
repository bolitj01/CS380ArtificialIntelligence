"""Alpha-beta pruning search for adversarial games."""

import math


def alpha_beta_search(state, game, depth_limit=6, use_heuristic=True):
    """Search game to determine best action; use alpha-beta pruning.
    
    Args:
        state: The current game state
        game: The game object
        depth_limit: Maximum depth to search (default 6)
        use_heuristic: Whether to use heuristic evaluation at cutoff (default True)
    """

    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta, depth):
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
            v = max(v, min_value(game.result(state, action), alpha, beta, depth - 1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta, depth):
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
            v = min(v, max_value(game.result(state, action), alpha, beta, depth - 1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_search:
    best_score = -math.inf
    beta = math.inf
    best_action = None
    for action in game.actions(state):
        v = min_value(game.result(state, action), best_score, beta, depth_limit - 1)
        if v > best_score:
            best_score = v
            best_action = action
    return best_action
