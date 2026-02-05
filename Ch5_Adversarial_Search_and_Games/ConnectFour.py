"""Connect Four game with two AI players using different search algorithms."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from TicTacToe import GameState, Game

from AlphaBetaSearch import alpha_beta_search
from MinimaxDepthLimitedSearch import minmax_decision_depth_limited
from MonteCarloTreeSearch import (
    monte_carlo_tree_search,
    load_transposition_table,
    save_transposition_table,
)

Position = Tuple[int, int]
Board = Dict[Position, str]


class ConnectFour(Game):
    """A Connect Four game where you can only make a move on the bottom
    row, or in a square directly above an occupied square.  Traditionally
    played on a 7x6 board and requiring 4 in a row."""

    def __init__(self, h: int = 7, v: int = 6, k: int = 4, first_player: str = "X") -> None:
        self.h = h
        self.v = v
        self.k = k
        moves = [(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]
        self.initial = GameState(to_move=first_player, utility=0, board={}, moves=moves)

    def actions(self, state: GameState) -> List[Position]:
        """Legal moves are positions at the bottom or directly above occupied squares."""
        return [(x, y) for (x, y) in state.moves if y == 1 or (x, y - 1) in state.board]

    def result(self, state: GameState, move: Position) -> GameState:
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        return GameState(
            to_move=("O" if state.to_move == "X" else "X"),
            utility=self.compute_utility(board, move, state.to_move),
            board=board,
            moves=moves,
        )

    def utility(self, state: GameState, player: str) -> int:
        """Return the value to player; 1 for win, -1 for loss, 0 otherwise."""
        return state.utility if player == "X" else -state.utility

    def terminal_test(self, state: GameState) -> bool:
        """A state is terminal if it is won or there are no empty squares."""
        return state.utility != 0 or len(state.moves) == 0

    def evaluate(self, state: GameState, player: str) -> int:
        """Heuristic evaluation for non-terminal states.
        Prioritizes center column and counts potential connect-fours."""
        board = state.board
        opponent = "O" if player == "X" else "X"
        score = 0

        # Center column preference
        center_x = (self.h + 1) // 2
        center_count_player = 0
        center_count_opponent = 0
        for y in range(1, self.v + 1):
            cell = board.get((center_x, y))
            if cell == player:
                center_count_player += 1
            elif cell == opponent:
                center_count_opponent += 1
        score += center_count_player * 3
        score -= center_count_opponent * 3

        # Score all windows of length k
        for window in self.iter_windows(board):
            score += self.score_window(window, player, opponent)

        return score

    def display(self, state: GameState) -> None:
        board = state.board
        print()
        for y in range(self.v, 0, -1):
            for x in range(1, self.h + 1):
                print(board.get((x, y), "."), end=" ")
            print()
        print("-" * (self.h * 2 - 1))
        print(" ".join(str(x) for x in range(1, self.h + 1)))
        print()

    def compute_utility(self, board: Board, move: Position, player: str) -> int:
        """If 'X' wins with this move, return 1; if 'O' wins return -1; else return 0."""
        if (
            self.k_in_row(board, move, player, (0, 1))
            or self.k_in_row(board, move, player, (1, 0))
            or self.k_in_row(board, move, player, (1, -1))
            or self.k_in_row(board, move, player, (1, 1))
        ):
            return +1 if player == "X" else -1
        return 0

    def k_in_row(self, board: Board, move: Position, player: str, delta_x_y: Position) -> bool:
        """Return true if there is a line through move on board for player."""
        (delta_x, delta_y) = delta_x_y
        x, y = move
        n = 0  # n is number of moves in row
        while board.get((x, y)) == player:
            n += 1
            x, y = x + delta_x, y + delta_y
        x, y = move
        while board.get((x, y)) == player:
            n += 1
            x, y = x - delta_x, y - delta_y
        n -= 1  # Because we counted move itself twice
        return n >= self.k

    def iter_windows(self, board: Board) -> List[List[str]]:
        """Generate all possible windows of length k on the board."""
        windows: List[List[str]] = []

        # Horizontal windows
        for y in range(1, self.v + 1):
            for x in range(1, self.h - self.k + 2):
                window = [board.get((x + i, y), ".") for i in range(self.k)]
                windows.append(window)

        # Vertical windows
        for x in range(1, self.h + 1):
            for y in range(1, self.v - self.k + 2):
                window = [board.get((x, y + i), ".") for i in range(self.k)]
                windows.append(window)

        # Diagonal (positive slope)
        for x in range(1, self.h - self.k + 2):
            for y in range(1, self.v - self.k + 2):
                window = [board.get((x + i, y + i), ".") for i in range(self.k)]
                windows.append(window)

        # Diagonal (negative slope)
        for x in range(1, self.h - self.k + 2):
            for y in range(self.k, self.v + 1):
                window = [board.get((x + i, y - i), ".") for i in range(self.k)]
                windows.append(window)

        return windows

    def score_window(self, window: List[str], player: str, opponent: str) -> int:
        """Score a window of length k for the given player and opponent."""
        score = 0
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(".")

        if player_count == self.k:
            score += 100000
        elif player_count == self.k - 1 and empty_count == 1:
            score += 50
        elif player_count == self.k - 2 and empty_count == 2:
            score += 10

        if opponent_count == self.k:
            score -= 100000
        elif opponent_count == self.k - 1 and empty_count == 1:
            score -= 80
        elif opponent_count == self.k - 2 and empty_count == 2:
            score -= 5

        return score


def play_game(
    delay: float,
    player_x_algo: str = "minimax",
    player_o_algo: str = "alphabeta",
    use_heuristic: bool = True,
    minimax_depth: int = 6,
    alphabeta_depth: int = 7,
    mcts_simulations: int = 1000,
    show_output: bool = True,
    mcts_debug: bool = False,
) -> str:
    """Play Connect Four with two chosen algorithms.
    
    Args:
        delay: Display delay between moves in seconds
        player_x_algo: Algorithm for player X ('minimax', 'alphabeta', or 'mcts')
        player_o_algo: Algorithm for player O ('minimax', 'alphabeta', or 'mcts')
        use_heuristic: If True, use heuristic evaluation at depth cutoff
        minimax_depth: Search depth for Minimax algorithm
        alphabeta_depth: Search depth for Alpha-Beta algorithm
        mcts_simulations: Number of simulations for MCTS
        show_output: If True, display game progress; if False, play silently
        mcts_debug: If True, show MCTS diagnostic info
        
    Returns:
        Winner: 'X', 'O', or 'Draw'
    """
    # Load transposition table if MCTS is used
    if "mcts" in (player_x_algo, player_o_algo):
        load_transposition_table()
    
    game = ConnectFour(first_player="X")
    state = game.initial
    move_number = 1
    
    # Algorithm names for display
    algo_names = {
        "minimax": "Minimax",
        "alphabeta": "Alpha-Beta",
        "mcts": "MCTS"
    }
    
    # Track timing for each player
    player_x_time = 0.0
    player_o_time = 0.0

    if show_output:
        print("=" * 50)
        print(f"Connect Four: {algo_names[player_x_algo]} (X) vs {algo_names[player_o_algo]} (O)")
        print("=" * 50)
        game.display(state)
        time.sleep(delay)

    while not game.terminal_test(state):
        current_player = state.to_move
        current_algo = player_x_algo if current_player == "X" else player_o_algo

        if show_output:
            print(f"Move {move_number}: Player {current_player} ({algo_names[current_algo]}) thinking...")
        
        start_time = time.time()
        
        # Select algorithm and get move
        if current_algo == "minimax":
            move = minmax_decision_depth_limited(state, game, depth_limit=minimax_depth, use_heuristic=use_heuristic)
        elif current_algo == "alphabeta":
            move = alpha_beta_search(state, game, depth_limit=alphabeta_depth, use_heuristic=use_heuristic)
        elif current_algo == "mcts":
            move = monte_carlo_tree_search(state, game, num_simulations=mcts_simulations, debug=mcts_debug)
        else:
            raise ValueError(f"Unknown algorithm: {current_algo}")
        
        elapsed = time.time() - start_time
        
        if current_player == "X":
            player_x_time += elapsed
        else:
            player_o_time += elapsed
            
        if show_output:
            print(f"  {algo_names[current_algo]} took {elapsed:.3f}s")

        state = game.result(state, move)

        if show_output:
            print(f"Move {move_number}: Player {current_player} -> Column {move[0]}")
            game.display(state)
            time.sleep(delay)
        move_number += 1

    # Determine winner
    if state.utility == 1:
        winner = "X"
        if show_output:
            print(f"Result: X ({algo_names[player_x_algo]}) wins!")
    elif state.utility == -1:
        winner = "O"
        if show_output:
            print(f"Result: O ({algo_names[player_o_algo]}) wins!")
    else:
        winner = "Draw"
        if show_output:
            print("Result: Draw!")

    if show_output:
        print("\n" + "=" * 50)
        print("Timing Summary")
        print("=" * 50)
        print(f"Player X ({algo_names[player_x_algo]}): {player_x_time:.3f}s")
        print(f"Player O ({algo_names[player_o_algo]}): {player_o_time:.3f}s")
        print(f"Total time: {player_x_time + player_o_time:.3f}s")
        print("=" * 50)
    
    # Save transposition table if MCTS was used
    if "mcts" in (player_x_algo, player_o_algo):
        save_transposition_table()

    return winner


def play_tournament(
    num_games: int = 100,
    player_x_algo: str = "minimax",
    player_o_algo: str = "alphabeta",
    minimax_depth: int = 6,
    alphabeta_depth: int = 7,
    mcts_simulations: int = 1000,
    use_heuristic: bool = True,
) -> None:
    """Play multiple games and track win statistics.
    
    Args:
        num_games: Number of games to play
        player_x_algo: Algorithm for player X ('minimax', 'alphabeta', or 'mcts')
        player_o_algo: Algorithm for player O ('minimax', 'alphabeta', or 'mcts')
        minimax_depth: Search depth for Minimax algorithm
        alphabeta_depth: Search depth for Alpha-Beta algorithm
        mcts_simulations: Number of simulations for MCTS
        use_heuristic: Whether to use heuristic evaluation
    """
    # Load transposition table if MCTS is used
    if "mcts" in (player_x_algo, player_o_algo):
        load_transposition_table()
    
    algo_names = {
        "minimax": "Minimax",
        "alphabeta": "Alpha-Beta",
        "mcts": "MCTS"
    }
    
    player_x_wins = 0
    player_o_wins = 0
    draws = 0
    
    print("=" * 60)
    print(f"Playing {num_games} games tournament")
    print(f"Player X: {algo_names[player_x_algo]}, Player O: {algo_names[player_o_algo]}")
    print(f"Heuristic: {'Enabled' if use_heuristic else 'Disabled'}")
    print("=" * 60)
    
    for game_num in range(1, num_games + 1):
        # Alternate who goes first by swapping X and O assignments
        if game_num % 2 == 1:
            current_x_algo = player_x_algo
            current_o_algo = player_o_algo
        else:
            current_x_algo = player_o_algo
            current_o_algo = player_x_algo
        
        print("=" * 60)
        print(f"Tournament Progress: Game {game_num}/{num_games}")
        print("=" * 60)
        print(f"{algo_names[player_x_algo]} wins: {player_x_wins}")
        print(f"{algo_names[player_o_algo]} wins: {player_o_wins}")
        print(f"Draws: {draws}")
        print("=" * 60)
        print(f"Playing game {game_num}...")
        print(f"  X: {algo_names[current_x_algo]}, O: {algo_names[current_o_algo]}")
        
        winner = play_game(
            delay=0,
            player_x_algo=current_x_algo,
            player_o_algo=current_o_algo,
            use_heuristic=use_heuristic,
            minimax_depth=minimax_depth,
            alphabeta_depth=alphabeta_depth,
            mcts_simulations=mcts_simulations,
            show_output=False,
        )
        
        # Track wins for the original player assignments
        if game_num % 2 == 1:
            # Normal assignment
            if winner == "X":
                player_x_wins += 1
            elif winner == "O":
                player_o_wins += 1
            else:
                draws += 1
        else:
            # Swapped assignment
            if winner == "X":
                player_o_wins += 1
            elif winner == "O":
                player_x_wins += 1
            else:
                draws += 1
    
    # Final results
    print("\n" + "=" * 60)
    print(f"Tournament Complete: {num_games} games played")
    print("=" * 60)
    print(f"{algo_names[player_x_algo]} wins: {player_x_wins} ({player_x_wins/num_games*100:.1f}%)")
    print(f"{algo_names[player_o_algo]} wins: {player_o_wins} ({player_o_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print("=" * 60)
    
    # Save transposition table if MCTS was used
    if "mcts" in (player_x_algo, player_o_algo):
        save_transposition_table()


if __name__ == "__main__":
    # Configuration
    delay = 0.3  # seconds
    use_heuristic = True  # Set to False to disable heuristic evaluation
    
    # Algorithm settings
    minimax_depth = 6  # Search depth for Minimax
    alphabeta_depth = 5  # Search depth for Alpha-Beta
    mcts_simulations = 1000  # Number of simulations for MCTS
    
    # Player algorithm selection - choose from: 'minimax', 'alphabeta', 'mcts'
    player_x_algo = "alphabeta"  # Algorithm for Player X
    player_o_algo = "mcts"  # Algorithm for Player O
    
    # Mode selection
    play_single_game = True  # Set to True for single game, False for tournament
    num_tournament_games = 100  # Number of games in tournament mode
    
    if play_single_game:
        play_game(
            delay=delay,
            player_x_algo=player_x_algo,
            player_o_algo=player_o_algo,
            use_heuristic=use_heuristic,
            minimax_depth=minimax_depth,
            alphabeta_depth=alphabeta_depth,
            mcts_simulations=mcts_simulations,
            mcts_debug=True,  # Show MCTS diagnostics in single game mode
        )
    else:
        play_tournament(
            num_games=num_tournament_games,
            player_x_algo=player_x_algo,
            player_o_algo=player_o_algo,
            minimax_depth=minimax_depth,
            alphabeta_depth=alphabeta_depth,
            mcts_simulations=mcts_simulations,
            use_heuristic=use_heuristic,
        )
