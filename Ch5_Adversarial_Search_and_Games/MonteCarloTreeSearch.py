"""Monte Carlo Tree Search with UCB selection policy and persistent transposition table."""

from __future__ import annotations

import math
import pickle
import random
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Transposition table for caching evaluated positions
TRANSPOSITION_TABLE_FILE = Path(__file__).parent / "mcts_transposition_table.pkl"
transposition_table: Dict[str, Tuple[float, int]] = {}


def load_transposition_table() -> Dict[str, Tuple[float, int]]:
    """Load transposition table from disk if it exists.
    
    Returns:
        Dictionary mapping state hashes to (avg_value, visit_count) tuples
    """
    global transposition_table
    if TRANSPOSITION_TABLE_FILE.exists():
        try:
            with open(TRANSPOSITION_TABLE_FILE, 'rb') as f:
                transposition_table = pickle.load(f)
                print(f"Loaded transposition table with {len(transposition_table)} entries")
            return transposition_table
        except Exception as e:
            print(f"Warning: Could not load transposition table: {e}")
            transposition_table = {}
            return transposition_table
    else:
        transposition_table = {}
        return transposition_table


def save_transposition_table() -> None:
    """Save transposition table to disk."""
    try:
        with open(TRANSPOSITION_TABLE_FILE, 'wb') as f:
            pickle.dump(transposition_table, f)
            print(f"Saved transposition table with {len(transposition_table)} entries")
    except Exception as e:
        print(f"Warning: Could not save transposition table: {e}")


def state_to_hash(state) -> str:
    """Convert game state to a hashable string representation.
    
    Uses the board configuration and current player.
    """
    board_str = ";".join(
        f"{x},{y}:{piece}" for (x, y), piece in sorted(state.board.items())
    )
    return f"{state.to_move}|{board_str}"


def get_transposition_entry(state) -> Optional[Tuple[float, int]]:
    """Look up a state in the transposition table.
    
    Returns:
        Tuple of (avg_value, visit_count) or None if not found
    """
    state_hash = state_to_hash(state)
    return transposition_table.get(state_hash)


def store_transposition_entry(state, avg_value: float, visits: int) -> None:
    """Store a state evaluation in the transposition table.
    
    Args:
        state: The game state
        avg_value: Average value from simulations (0-1)
        visits: Number of visits to this state
    """
    state_hash = state_to_hash(state)
    transposition_table[state_hash] = (avg_value, visits)


class MCTSNode:
    """Node in the Monte Carlo Tree Search."""

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.wins = 0.0
        self.untried_actions: Optional[List] = None

    def ucb_score(self, exploration_weight: float = 1.41) -> float:
        """Calculate Upper Confidence Bound for Trees (UCT) score."""
        if self.visits == 0:
            return math.inf
        exploitation = self.wins / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def select_child(self) -> MCTSNode:
        """Select child with highest UCB score."""
        return max(self.children, key=lambda child: child.ucb_score())

    def expand(self, game) -> MCTSNode:
        """Expand tree by adding one child node."""
        if self.untried_actions is None:
            self.untried_actions = game.actions(self.state)
        
        action = self.untried_actions.pop()
        next_state = game.result(self.state, action)
        child_node = MCTSNode(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def update(self, result: float) -> None:
        """Update node statistics after simulation."""
        self.visits += 1
        self.wins += result

    def is_fully_expanded(self, game) -> bool:
        """Check if all possible actions have been expanded."""
        if self.untried_actions is None:
            self.untried_actions = game.actions(self.state)
        return len(self.untried_actions) == 0

    def is_terminal(self, game) -> bool:
        """Check if this node represents a terminal state."""
        return game.terminal_test(self.state)


def monte_carlo_tree_search(state, game, num_simulations: int = 1000, debug: bool = False) -> any:
    """Perform Monte Carlo Tree Search to find the best action.
    
    Args:
        state: Current game state
        game: Game object
        num_simulations: Number of MCTS simulations to run (default 1000)
        debug: If True, print diagnostic information
        
    Returns:
        Best action found by MCTS
    """
    # Check if we've already evaluated this position
    cached_entry = get_transposition_entry(state)
    if cached_entry:
        cached_value, cached_visits = cached_entry
        if debug:
            print(f"  Found transposition table entry: {cached_visits} visits, value {cached_value:.3f}")
    
    root = MCTSNode(state)
    terminal_sims = 0
    cache_hits = 0
    
    for _ in range(num_simulations):
        node = root
        
        # Selection: traverse tree using UCB
        while not node.is_terminal(game) and node.is_fully_expanded(game):
            node = node.select_child()
        
        # Check cache before expansion (now useful since playouts are deterministic)
        cached_entry = get_transposition_entry(node.state)
        if cached_entry and not node.is_terminal(game):
            # Use cached result for remainder of playout
            cached_value, _ = cached_entry
            result = cached_value
            cache_hits += 1
        else:
            # Expansion: add new child node
            if not node.is_terminal(game):
                node = node.expand(game)
            
            # Simulation: play out with greedy/heavy policy (deterministic)
            result, reached_terminal = simulate(node.state, game)
            if reached_terminal:
                terminal_sims += 1
            
            # Cache the result for future simulations
            store_transposition_entry(node.state, result, 1)
        
        # Backpropagation: update statistics
        temp_node = node
        while temp_node is not None:
            temp_node.update(result)
            # Flip result for parent (alternating players)
            result = 1 - result
            temp_node = temp_node.parent
    
    if debug:
        print(f"  MCTS: {terminal_sims}/{num_simulations} terminal states ({100*terminal_sims/num_simulations:.1f}%), {cache_hits} cache hits ({100*cache_hits/num_simulations:.1f}%)")
        if root.children:
            visits = [child.visits for child in root.children]
            print(f"  Child visits: min={min(visits)}, max={max(visits)}, avg={sum(visits)/len(visits):.1f}")
    
    # Return action of most visited child
    if not root.children:
        # If no children, return random action
        actions = game.actions(state)
        return random.choice(actions) if actions else None
    
    best_child = max(root.children, key=lambda child: child.visits)
    return best_child.action


def score_action(action, state, game) -> float:
    """Score an action based on tactical priorities.
    
    Scoring:
    +100: Winning move
    +50: Block opponent win
    +10: Extend own 2-in-a-row
    -10: Allow opponent 3-in-a-row
    +5: Center column preference
    """
    current_player = state.to_move
    opponent = "O" if current_player == "X" else "X"
    next_state = game.result(state, action)
    score = 0.0
    
    # Winning move: +100
    if game.terminal_test(next_state) and game.utility(next_state, current_player) > 0:
        return 100.0
    
    # Center column preference: +5
    board_width = getattr(game, "h", 7)
    center_x = (board_width + 1) // 2
    if action[0] == center_x:
        score += 5
    elif abs(action[0] - center_x) <= 1:  # Near center
        score += 2
    
    # Check for blocking opponent win
    opponent_state = game.result(state, action)
    # Simulate opponent's next moves to see if we blocked a win
    for opp_action in game.actions(state):
        opp_next = game.result(state, opp_action)
        if game.terminal_test(opp_next) and game.utility(opp_next, opponent) > 0:
            # Opponent could win - did we block it?
            if action == opp_action:
                score += 50
                break
    
    # Count threats and opportunities using board analysis
    board = next_state.board
    k = getattr(game, "k", 4)
    
    # Helper function to count sequences
    def count_sequences(player_mark, length):
        count = 0
        # Check all windows
        if hasattr(game, "iter_windows"):
            for window in game.iter_windows(board):
                player_count = window.count(player_mark)
                empty_count = window.count(".")
                opponent_count = window.count("O" if player_mark == "X" else "X")
                
                if player_count == length and opponent_count == 0:
                    count += 1
        return count
    
    # Extend own 2-in-a-row: +10
    own_two_count = count_sequences(current_player, 2)
    score += own_two_count * 10
    
    # Penalize allowing opponent 3-in-a-row: -10
    # Check if opponent now has 3-in-a-row opportunities
    opponent_three_count = count_sequences(opponent, 3)
    score -= opponent_three_count * 10
    
    return score


def simulate(state, game, max_depth: int = 10000) -> tuple:
    """Simulate a playout using greedy/heavy policy (no randomness).
    
    Always selects the highest-scoring action based on tactical evaluation.
    This makes MCTS simulate strong play and benefit from cached positions.
    
    Returns tuple of (result, reached_terminal) where:
    - result: 1.0 for win, 0.0 for loss, 0.5 for draw (from current player's perspective)
    - reached_terminal: True if playout reached a terminal state
    """
    current_state = state
    original_player = state.to_move
    depth = 0
    
    # Play out game until terminal or max depth - GREEDY (no randomness)
    while not game.terminal_test(current_state) and depth < max_depth:
        actions = game.actions(current_state)
        if not actions:
            break
        
        # Score each action using tactical policy
        action_scores = []
        for action in actions:
            score = score_action(action, current_state, game)
            action_scores.append((action, score))
        
        # Always take the best action (greedy/heavy playout - deterministic)
        action_scores.sort(key=lambda x: x[1], reverse=True)
        best_action = action_scores[0][0]
        
        current_state = game.result(current_state, best_action)
        depth += 1
    
    # Determine result from original player's perspective
    reached_terminal = game.terminal_test(current_state)
    if reached_terminal:
        utility = game.utility(current_state, original_player)
        if utility > 0:
            return (1.0, True)  # Win
        elif utility < 0:
            return (0.0, True)  # Loss
        else:
            return (0.5, True)  # Draw
    else:
        # If max depth reached without terminal, use heuristic evaluation
        evaluate = getattr(game, "evaluate", None)
        if evaluate:
            heuristic_value = evaluate(current_state, original_player)
            # Normalize to [0, 1] range
            normalized = 0.5 + heuristic_value / 200.0
            normalized = max(0.0, min(1.0, normalized))
            return (normalized, False)
        else:
            return (0.5, False)  # Default to draw
