"""Connect Four game with two AI players using different search algorithms."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from TicTacToe import GameState, Game

from AlphaBetaSearch import alpha_beta_search
from MinimaxDepthLimitedSearch import minmax_decision_depth_limited

Position = Tuple[int, int]
Board = Dict[Position, str]


class ConnectFour(Game):
    """A Connect Four game where you can only make a move on the bottom
    row, or in a square directly above an occupied square.  Traditionally
    played on a 7x6 board and requiring 4 in a row."""

    def __init__(self, h: int = 7, v: int = 6, k: int = 4, minimax_first: bool = True) -> None:
        self.h = h
        self.v = v
        self.k = k
        moves = [(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]
        first_player = "X" if minimax_first else "O"
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


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def play_connect_four(
    delay: float,
    minimax_first: bool = True,
    use_heuristic: bool = True,
    minimax_depth: int = 6,
    alphabeta_depth: int = 7,
    show_output: bool = True,
) -> str:
    """Play Connect Four with Minimax (X) vs Alpha-Beta (O).
    
    Args:
        delay: Display delay between moves in seconds
        minimax_first: If True, Minimax goes first (X); if False, Alpha-Beta goes first (O)
        use_heuristic: If True, use heuristic evaluation at depth cutoff; if False, just use terminal utility
        minimax_depth: Search depth for Minimax algorithm
        alphabeta_depth: Search depth for Alpha-Beta algorithm
        show_output: If True, display game progress; if False, play silently
        
    Returns:
        Winner: 'X' (Minimax), 'O' (Alpha-Beta), or 'Draw'
    """
    game = ConnectFour(minimax_first=minimax_first)
    state = game.initial
    move_number = 1
    
    # Track timing for each algorithm
    minimax_time = 0.0
    alpha_beta_time = 0.0

    if show_output:
        clear_screen()
        print("=" * 40)
        print("Connect Four: Minimax (X) vs Alpha-Beta (O)")
        print("=" * 40)
        game.display(state)
        time.sleep(delay)

    while not game.terminal_test(state):
        current_player = state.to_move

        if current_player == "O":
            if show_output:
                print(f"Move {move_number}: Player O (Alpha-Beta) thinking...")
            start_time = time.time()
            move = alpha_beta_search(state, game, depth_limit=alphabeta_depth, use_heuristic=use_heuristic)
            elapsed = time.time() - start_time
            alpha_beta_time += elapsed
            if show_output:
                print(f"  Alpha-Beta took {elapsed:.3f}s")
        else:
            if show_output:
                print(f"Move {move_number}: Player X (Minimax) thinking...")
            start_time = time.time()
            move = minmax_decision_depth_limited(state, game, depth_limit=minimax_depth, use_heuristic=use_heuristic)
            elapsed = time.time() - start_time
            minimax_time += elapsed
            if show_output:
                print(f"  Minimax took {elapsed:.3f}s")
        

        state = game.result(state, move)

        if show_output:
            clear_screen()
            print(f"Move {move_number}: Player {current_player} -> Column {move[0]}")
            game.display(state)
            time.sleep(delay)
        move_number += 1

    # Determine winner
    if state.utility == 1:
        winner = "X"
        if show_output:
            print("Result: X (Minimax) wins!")
    elif state.utility == -1:
        winner = "O"
        if show_output:
            print("Result: O (Alpha-Beta) wins!")
    else:
        winner = "Draw"
        if show_output:
            print("Result: Draw!")

    if show_output:
        print("\n" + "=" * 40)
        print("Search Algorithm Timing Summary")
        print("=" * 40)
        print(f"Minimax (Depth-Limited):  {minimax_time:.3f}s")
        print(f"Alpha-Beta (Depth-Limited): {alpha_beta_time:.3f}s")
        print(f"Total time:               {minimax_time + alpha_beta_time:.3f}s")
        print("=" * 40)

    return winner


def play_tournament(
    num_games: int = 100,
    minimax_depth: int = 6,
    alphabeta_depth: int = 7,
    use_heuristic: bool = True,
) -> None:
    """Play multiple games and track win statistics.
    
    Args:
        num_games: Number of games to play
        minimax_depth: Search depth for Minimax algorithm
        alphabeta_depth: Search depth for Alpha-Beta algorithm
        use_heuristic: Whether to use heuristic evaluation
    """
    minimax_wins = 0
    alphabeta_wins = 0
    draws = 0
    
    print("=" * 50)
    print(f"Playing {num_games} games tournament")
    print(f"Minimax depth: {minimax_depth}, Alpha-Beta depth: {alphabeta_depth}")
    print(f"Heuristic: {'Enabled' if use_heuristic else 'Disabled'}")
    print("=" * 50)
    
    for game_num in range(1, num_games + 1):
        # Alternate who goes first
        minimax_first = (game_num % 2 == 1)
        
        clear_screen()
        print("=" * 50)
        print(f"Tournament Progress: Game {game_num}/{num_games}")
        print("=" * 50)
        print(f"Minimax (X) wins: {minimax_wins}")
        print(f"Alpha-Beta (O) wins: {alphabeta_wins}")
        print(f"Draws: {draws}")
        print("=" * 50)
        print(f"Playing game {game_num}...")
        print(f"  {'Minimax' if minimax_first else 'Alpha-Beta'} goes first")
        
        winner = play_connect_four(
            delay=0,
            minimax_first=minimax_first,
            use_heuristic=use_heuristic,
            minimax_depth=minimax_depth,
            alphabeta_depth=alphabeta_depth,
            show_output=False,
        )
        
        if winner == "X":
            minimax_wins += 1
        elif winner == "O":
            alphabeta_wins += 1
        else:
            draws += 1
    
    # Final results
    clear_screen()
    print("\n" + "=" * 50)
    print(f"Tournament Complete: {num_games} games played")
    print("=" * 50)
    print(f"Minimax (X) wins: {minimax_wins} ({minimax_wins/num_games*100:.1f}%)")
    print(f"Alpha-Beta (O) wins: {alphabeta_wins} ({alphabeta_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")
    print("=" * 50)


if __name__ == "__main__":
    # Configuration
    delay = 0.3  # seconds
    minimax_first = True  # Set to False to have Alpha-Beta go first
    use_heuristic = True  # Set to False to disable heuristic evaluation at depth cutoff
    minimax_depth = 5  # Search depth for Minimax
    alphabeta_depth = 7  # Search depth for Alpha-Beta
    
    # Mode selection
    play_single_game = False  # Set to True for single game, False for tournament
    num_tournament_games = 100  # Number of games in tournament mode
    
    if play_single_game:
        play_connect_four(
            delay=delay,
            minimax_first=minimax_first,
            use_heuristic=use_heuristic,
            minimax_depth=minimax_depth,
            alphabeta_depth=alphabeta_depth,
        )
    else:
        play_tournament(
            num_games=num_tournament_games,
            minimax_depth=minimax_depth,
            alphabeta_depth=alphabeta_depth,
            use_heuristic=use_heuristic,
        )
