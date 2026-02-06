"""Expectiminimax search implementation for stochastic games with chance nodes."""

import math


def expect_minimax_decision(state, game):
    """Given a state in a stochastic game with chance nodes, calculate the best move
    using Expectiminimax (minimax with averaging for chance nodes).
    
    This algorithm handles games with uncertain events (like dice rolls) by:
    - Using max_value for maximizing player's turns
    - Using min_value for minimizing player's turns  
    - Using chance_value to average over possible chance outcomes
    """
    
    player = game.to_move(state)
    
    def max_value(state):
        """Maximizing player chooses the move with highest expected value."""
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -math.inf
        for action in game.actions(state):
            v = max(v, chance_value(game.result(state, action)))
        return v
    
    def min_value(state):
        """Minimizing player chooses the move with lowest expected value."""
        if game.terminal_test(state):
            return game.utility(state, player)
        v = math.inf
        for action in game.actions(state):
            v = min(v, chance_value(game.result(state, action)))
        return v
    
    def chance_value(state):
        """Average over all possible chance outcomes weighted by probability."""
        if game.terminal_test(state):
            return game.utility(state, player)
        
        # If there's a specific chance outcome (dice roll already determined)
        if state.chance is not None:
            # Continue with the max/min player's turn
            if game.to_move(state) == player:
                return max_value(state)
            else:
                return min_value(state)
        
        # Need to average over all possible chance outcomes
        value_sum = 0
        chance_outcomes = game.chances(state)
        
        if not chance_outcomes:
            return game.utility(state, player)
        
        for chance in chance_outcomes:
            outcome_state = game.outcome(state, chance)
            prob = game.probability(chance)
            if game.to_move(outcome_state) == player:
                value_sum += prob * max_value(outcome_state)
            else:
                value_sum += prob * min_value(outcome_state)
        
        return value_sum
    
    # Body of expect_minimax_decision:
    # Find the action that maximizes the expected utility
    best_action = None
    best_value = -math.inf
    
    for action in game.actions(state):
        result_state = game.result(state, action)
        value = chance_value(result_state)
        if value > best_value:
            best_value = value
            best_action = action
    
    return best_action


def expect_minimax_decision_depth_limited(state, game, depth=4, eval_fn=None):
    """Expectiminimax search with depth limit and evaluation function.
    
    Useful for games with large branching factors where full lookahead isn't feasible.
    """
    
    player = game.to_move(state)
    eval_fn = eval_fn or (lambda state: game.utility(state, player) if game.terminal_test(state) else 0)
    
    def max_value(state, depth):
        """Maximizing player with depth limit."""
        if depth == 0 or game.terminal_test(state):
            return eval_fn(state)
        actions = game.actions(state)
        if not actions:  # No legal moves - treat as terminal for this branch
            return eval_fn(state)
        v = -math.inf
        for action in actions:
            v = max(v, chance_value(game.result(state, action), depth))
        return v
    
    def min_value(state, depth):
        """Minimizing player with depth limit."""
        if depth == 0 or game.terminal_test(state):
            return eval_fn(state)
        actions = game.actions(state)
        if not actions:  # No legal moves - treat as terminal for this branch
            return eval_fn(state)
        v = math.inf
        for action in actions:
            v = min(v, chance_value(game.result(state, action), depth))
        return v
    
    def chance_value(state, depth):
        """Average over chance outcomes with depth limit."""
        if depth == 0 or game.terminal_test(state):
            return eval_fn(state)
        
        # If there's a specific chance outcome (dice roll already determined)
        if state.chance is not None:
            # Continue with the max/min player's turn
            if game.to_move(state) == player:
                return max_value(state, depth)
            else:
                return min_value(state, depth)
        
        # Need to average over all possible chance outcomes
        value_sum = 0
        chance_outcomes = game.chances(state)
        
        if not chance_outcomes:
            return eval_fn(state)
        
        for chance in chance_outcomes:
            outcome_state = game.outcome(state, chance)
            prob = game.probability(chance)
            # Check if this dice roll results in any legal moves
            if not game.actions(outcome_state):
                # No legal moves with this dice roll - evaluate state as-is
                value_sum += prob * eval_fn(outcome_state)
            elif game.to_move(outcome_state) == player:
                value_sum += prob * max_value(outcome_state, depth - 1)
            else:
                value_sum += prob * min_value(outcome_state, depth - 1)
        
        return value_sum
    
    # Find the action with best expected value
    best_action = None
    best_value = -math.inf
    
    for action in game.actions(state):
        result_state = game.result(state, action)
        value = chance_value(result_state, depth)
        if value > best_value:
            best_value = value
            best_action = action
    
    return best_action
