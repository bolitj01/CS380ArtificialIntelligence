"""Tic Tac Toe with two AI players using minimax search."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from MinimaxSearch import minmax_decision

Position = Tuple[int, int]
Board = Dict[Position, str]


@dataclass(frozen=True)
class GameState:
    to_move: str
    utility: int
    board: Board
    moves: List[Position]


class Game:
    """Minimal game interface used by minimax."""

    def actions(self, state: GameState) -> List[Position]:
        raise NotImplementedError

    def result(self, state: GameState, move: Position) -> GameState:
        raise NotImplementedError

    def utility(self, state: GameState, player: str) -> int:
        raise NotImplementedError

    def terminal_test(self, state: GameState) -> bool:
        raise NotImplementedError

    def to_move(self, state: GameState) -> str:
        return state.to_move


class TicTacToe(Game):
    """Play TicTacToe on an h x v board, with Max (first player) playing 'X'.
    A state has the player to move, a cached utility, a list of moves in
    the form of a list of (x, y) positions, and a board, in the form of
    a dict of {(x, y): Player} entries, where Player is 'X' or 'O'."""

    def __init__(self, h: int = 3, v: int = 3, k: int = 3) -> None:
        self.h = h
        self.v = v
        self.k = k
        moves = [(x, y) for x in range(1, h + 1) for y in range(1, v + 1)]
        self.initial = GameState(to_move="X", utility=0, board={}, moves=moves)

    def actions(self, state: GameState) -> List[Position]:
        """Legal moves are any square not yet taken."""
        return state.moves

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

    def display(self, state: GameState) -> None:
        board = state.board
        for x in range(1, self.h + 1):
            for y in range(1, self.v + 1):
                print(board.get((x, y), "."), end=" ")
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


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def play_ai_vs_ai(delay_seconds: float = 0.8) -> None:
    game = TicTacToe()
    state = game.initial
    move_number = 1

    clear_screen()
    print("Tic Tac Toe: Minimax vs Minimax")
    game.display(state)
    time.sleep(delay_seconds)

    while not game.terminal_test(state):
        current_player = state.to_move
        move = minmax_decision(state, game)
        state = game.result(state, move)

        clear_screen()
        print(f"Move {move_number}: Player {current_player} -> {move}")
        game.display(state)
        time.sleep(delay_seconds)
        move_number += 1

    if state.utility == 1:
        print("Result: X wins")
    elif state.utility == -1:
        print("Result: O wins")
    else:
        print("Result: Draw")


def prompt_human_move(game: TicTacToe, state: GameState) -> Position:
    while True:
        raw = input("Enter your move as 'row col' like 1-3 1-3: ").strip()
        parts = raw.split()
        if len(parts) != 2:
            print("Invalid format. Use two numbers like: 2 3")
            continue
        try:
            x, y = int(parts[0]), int(parts[1])
        except ValueError:
            print("Invalid numbers. Try again.")
            continue
        move = (x, y)
        if move not in state.moves:
            print("That square is not available. Try again.")
            continue
        return move


def play_human_vs_ai(delay_seconds: float = 0.6) -> None:
    game = TicTacToe()
    state = game.initial
    move_number = 1

    clear_screen()
    print("Tic Tac Toe: Human (X) vs Minimax (O)")
    game.display(state)

    while not game.terminal_test(state):
        current_player = state.to_move
        if current_player == "X":
            move = prompt_human_move(game, state)
        else:
            time.sleep(delay_seconds)
            move = minmax_decision(state, game)

        state = game.result(state, move)
        clear_screen()
        print(f"Move {move_number}: Player {current_player} -> {move}")
        game.display(state)
        move_number += 1

    if state.utility == 1:
        print("Result: X wins")
    elif state.utility == -1:
        print("Result: O wins")
    else:
        print("Result: Draw")


if __name__ == "__main__":
    play_human = False # Set to True to play against AI, False for AI vs AI

    if play_human:
        play_human_vs_ai()
    else:
        play_ai_vs_ai()
