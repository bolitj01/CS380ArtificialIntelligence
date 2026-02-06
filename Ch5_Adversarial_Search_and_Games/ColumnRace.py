"""Simplified Racing Game with Expectiminimax AI using tkinter."""

import copy
import random
import tkinter as tk
from tkinter import messagebox
from collections import namedtuple
from ExpectiminiMaxSearch import expect_minimax_decision_depth_limited

# Game state representation: pieces are at positions 0-9
# White: starts at 0, moves towards 9
# Black: starts at 9, moves towards 0
GameState = namedtuple('GameState', 'to_move, utility, pieces, moves, chance')


class RacingGame:
    """Simplified racing game on a number line (0-9)."""
    
    def __init__(self):
        """Initialize game with 2 pieces per player on opposite ends."""
        # pieces: {'W': [pos_row1, pos_row2], 'B': [pos_row1, pos_row2]}
        # White starts at 0, Black starts at 9 (both rows)
        pieces = {
            'W': [0, 0],  # 2 white pieces at position 0
            'B': [9, 9]  # 2 black pieces at position 9
        }
        
        self.initial = GameState(
            to_move='W',
            utility=0,
            pieces=pieces,
            moves=self.get_all_moves(pieces, 'W'),
            chance=None
        )
    
    def actions(self, state):
        """Return legal moves for current state."""
        # If dice are set, filter for legal moves with that die value
        if state.chance is not None:
            legal_moves = []
            for move in state.moves:
                # move is an index of which piece to move
                if self.is_legal_move(state.pieces, move, state.chance, state.to_move):
                    legal_moves.append(move)
            return legal_moves
        
        # If no dice yet, return all possible moves (all pieces)
        return state.moves
    
    def result(self, state, move):
        """Return new state after making a move."""
        pieces = copy.deepcopy(state.pieces)
        player = state.to_move
        opponent = self._opponent(player)
        
        # move is piece index
        # Move the piece by the die value
        if state.chance is not None:
            direction = 1 if player == 'W' else -1
            new_pos = pieces[player][move] + direction * state.chance
            pieces[player][move] = new_pos
            
            # Check for capture: if landing on opponent's position in any row, reset those pieces to start
            for idx, opp_pos in enumerate(pieces[opponent]):
                if opp_pos == new_pos:
                    pieces[opponent][idx] = 0 if opponent == 'W' else 9
        
        to_move = 'B' if player == 'W' else 'W'
        
        return GameState(
            to_move=to_move,
            utility=self.compute_utility(pieces),
            pieces=pieces,
            moves=self.get_all_moves(pieces, to_move),
            chance=None
        )
    
    def utility(self, state, player):
        """Return utility value for terminal state."""
        if state.utility != 0:
            return state.utility if player == 'W' else -state.utility
        return 0
    
    def terminal_test(self, state):
        """Check if game is over (one player has won)."""
        return state.utility != 0
    
    def to_move(self, state):
        """Return whose turn it is."""
        return state.to_move
    
    def chances(self, state):
        """Return all possible dice roll outcomes (single die: 1-6)."""
        return list(range(1, 7))
    
    def outcome(self, state, chance):
        """Return state with a specific chance outcome (die roll)."""
        return GameState(
            to_move=state.to_move,
            utility=state.utility,
            pieces=state.pieces,
            moves=state.moves,
            chance=chance
        )
    
    def probability(self, chance):
        """Return probability of a dice outcome (uniform 1/6)."""
        return 1/6
    
    def get_all_moves(self, pieces, player):
        """Get all possible moves (piece indices)."""
        moves = []
        if player == 'W':
            for idx, pos in enumerate(pieces[player]):
                if pos < 9:  # Goal is at position 9
                    moves.append(idx)
        else:  # Black
            for idx, pos in enumerate(pieces[player]):
                if pos > 0:  # Goal is at position 0
                    moves.append(idx)
        return moves
    
    def is_legal_move(self, pieces, move_index, die_value, player, _=None):
        """Check if a move is legal given a die value."""
        direction = 1 if player == 'W' else -1
        new_pos = pieces[player][move_index] + direction * die_value
        
        # Legal if piece can move and doesn't overshoot goal
        if player == 'W':
            # Can move if at goal position is <= 9
            return pieces[player][move_index] < 9 and new_pos <= 9
        else:  # Black
            # Can move if at goal position is >= 0
            return pieces[player][move_index] > 0 and new_pos >= 0
    
    def compute_utility(self, pieces):
        """Calculate game utility."""
        # Check if White has won (all pieces at position >= 9)
        if all(pos >= 9 for pos in pieces['W']):
            return 1
        # Check if Black has won (all pieces at position <= 0)
        elif all(pos <= 0 for pos in pieces['B']):
            return -1
        else:
            return 0
    
    @staticmethod
    def _opponent(player):
        """Get opponent of a player."""
        return 'B' if player == 'W' else 'W'


class RacingGameGUI:
    """GUI for racing game using tkinter."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Racing Game - Expectiminimax AI")
        self.root.geometry("1200x800")
        self.game = RacingGame()
        self.state = self.game.initial
        self.move_count = 0
        self.legal_moves = []
        self.current_move_choice = None
        self.selected_piece = None
        
        # Create main frame
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create canvas for game board
        self.canvas = tk.Canvas(self.main_frame, bg='#E8D7C3', width=800, height=600)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Create right panel for info
        self.info_frame = tk.Frame(self.main_frame, width=300, bg='#F0F0F0')
        self.info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10)
        
        # row
        title_label = tk.Label(self.info_frame, text="RACING GAME", font=("Arial", 16, "bold"), bg='#F0F0F0')
        title_label.pack(pady=10)
        
        # Game info
        self.info_text = tk.Text(self.info_frame, height=13, width=35, font=("Courier", 12), wrap=tk.WORD)
        self.info_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # AI Decision display
        ai_label = tk.Label(self.info_frame, text="AI Decision:", font=("Arial", 20, "bold"), bg='#F0F0F0')
        ai_label.pack()
        
        self.ai_text = tk.Text(self.info_frame, height=8, width=35, font=("Courier", 12), wrap=tk.WORD)
        self.ai_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # Buttons
        button_frame = tk.Frame(self.info_frame, bg='#F0F0F0')
        button_frame.pack(pady=10)
        
        self.next_button = tk.Button(button_frame, text="Confirm Move", command=self.next_turn, 
                                     font=("Arial", 10), bg='#4CAF50', fg='white', padx=10, pady=5)
        self.next_button.pack(pady=5)
        
        self.reset_button = tk.Button(button_frame, text="New Game", command=self.reset_game,
                                      font=("Arial", 10), bg='#2196F3', fg='white', padx=10, pady=5)
        self.reset_button.pack(pady=5)
        
        # Start game
        self.roll_dice()
        # Ensure board renders after window layout is ready
        self.root.after(50, self.update_display)
    
    def roll_dice(self):
        """Roll a single die and set up new turn."""
        if self.state.chance is None:
            die_value = random.randint(1, 6)
            self.state = self.game.outcome(self.state, die_value)
            self.move_count += 1
        
        self.legal_moves = self.game.actions(self.state)
        self.current_move_choice = None
        self.selected_piece = None
        self.next_button.config(state=tk.DISABLED)
        
        self.update_display()
        
        # Check if no legal moves
        if not self.legal_moves:
            player_name = "White" if self.state.to_move == 'W' else "Black (AI)"
            self.update_info(f"No legal moves for {player_name}. Skipping turn...")
            self.root.after(1000, self.skip_turn)
            return
        
        # If AI's turn, make AI move
        if self.state.to_move == 'B':
            self.ai_text.config(state=tk.NORMAL)
            self.ai_text.delete(1.0, tk.END)
            self.ai_text.insert(tk.END, "AI is thinking...")
            self.ai_text.config(state=tk.DISABLED)
            self.root.after(1000, self.ai_move)
    
    def ai_move(self):
        """Execute AI move using Expectiminimax."""
        if not self.legal_moves:
            self.update_info("AI passes (no legal moves)")
            self.state = GameState(to_move='W', utility=self.state.utility,
            pieces=self.state.pieces, moves=self.state.moves, chance=None)
            self.root.after(2000, self.roll_dice)
            return
        
        # Get AI decision
        best_move = expect_minimax_decision_depth_limited(self.state, self.game, depth=6)
        
        if best_move is None:
            self.update_info("AI passes (no legal moves)")
            self.state = GameState(to_move='W', utility=self.state.utility,
            pieces=self.state.pieces, moves=self.state.moves, chance=None)
            self.root.after(2000, self.roll_dice)
            return
        
        # Display AI decision
        piece_idx = best_move
        old_pos = self.state.pieces['B'][piece_idx]
        new_pos = old_pos - self.state.chance  # Black moves backwards (towards 0)
        move_str = f"Piece {piece_idx} from position {old_pos} to {new_pos}"
        ai_info = f"AI Move: {move_str}\nDie Roll: {self.state.chance}\n\nExpectiminimax evaluates:\n• All possible future states\n• Both players' moves\n• Chance outcomes (die rolls)\n• Uses depth-limited search (depth=2)"
        
        self.ai_text.config(state=tk.NORMAL)
        self.ai_text.delete(1.0, tk.END)
        self.ai_text.insert(tk.END, ai_info)
        self.ai_text.config(state=tk.DISABLED)
        
        # Make the move
        self.state = self.game.result(self.state, best_move)
        self.update_display()
        
        # Switch to human player
        self.root.after(3000, self.human_turn)
    
    def human_turn(self):
        """Set up human player turn."""
        self.state = GameState(to_move='W', utility=self.state.utility,
                             pieces=self.state.pieces, moves=self.state.moves, chance=None)
        self.roll_dice()
    
    def skip_turn(self):
        """Skip current player's turn and switch to opponent."""
        to_move = 'B' if self.state.to_move == 'W' else 'W'
        self.state = GameState(to_move=to_move, utility=self.state.utility,
                             pieces=self.state.pieces, moves=self.state.moves, chance=None)
        self.roll_dice()
    
    def on_canvas_click(self, event):
        """Handle canvas clicks for piece selection via grid."""
        if self.state.to_move != 'W' or not self.legal_moves:
            return
        
        x, y = event.x, event.y
        
        # Determine which grid cell was clicked
        result = self.get_piece_from_coords(x, y)
        
        if result is None:
            return
        
        row, col = result
        
        # Row indicates which white piece to move (row 0 -> piece 0, row 1 -> piece 1)
        if row not in (0, 1):
            return
        
        piece_idx = row
        
        if piece_idx not in self.legal_moves:
            self.update_info("Piece cannot move (already at goal).")
            return
        
        # Selected piece and target column
        self.selected_piece = piece_idx
        old_pos = self.state.pieces['W'][piece_idx]
        die_value = self.state.chance
        new_pos = old_pos + die_value
        
        # Validate move: must be exact position match and legal
        if new_pos == col:
            # Check if this is a legal move (within board bounds)
            if self.game.is_legal_move(self.state.pieces, piece_idx, die_value, 'W'):
                self.current_move_choice = piece_idx
                self.next_button.config(state=tk.NORMAL)
                # Check if landing on opponent in any row
                capture_msg = " (CAPTURE!)" if any(pos == new_pos for pos in self.state.pieces['B']) else ""
                self.update_info(f"Selected: {old_pos} → {new_pos} (roll: +{die_value}){capture_msg}. Click 'Confirm Move'.")
            else:
                self.update_info(f"Invalid: Move {old_pos} + {die_value} = {new_pos} goes beyond board. Turn skipped.")
                self.selected_piece = None
                self.skip_turn()
        else:
            self.update_info(f"Invalid: {old_pos} + {die_value} = {new_pos}. Click on column {new_pos}.")
            self.selected_piece = None
        
        self.update_display()
    
    def get_piece_from_coords(self, x, y):
        """Convert canvas coordinates to grid cell (row, col)."""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return None
        
        # Grid parameters
        margin = 40
        grid_width = canvas_width - 2 * margin
        grid_height = canvas_height - 2 * margin
        num_rows = 2
        num_cols = 10
        cell_width = grid_width / num_cols
        cell_height = grid_height / num_rows
        
        # Check if click is in grid area
        if x < margin or x > margin + grid_width or y < margin or y > margin + grid_height:
            return None
        
        # Calculate row and column
        col = int((x - margin) / cell_width)
        row = int((y - margin) / cell_height)
        
        # Clamp to valid range
        col = max(0, min(col, num_cols - 1))
        row = max(0, min(row, num_rows - 1))
        
        # Return (row, col) tuple
        return (row, col)
    
    def next_turn(self):
        """Execute the selected move and proceed to next turn."""
        if self.current_move_choice is not None:
            self.state = self.game.result(self.state, self.current_move_choice)
            self.current_move_choice = None
            self.selected_piece = None
            self.ai_text.config(state=tk.NORMAL)
            self.ai_text.delete(1.0, tk.END)
            self.ai_text.config(state=tk.DISABLED)
            self.roll_dice()
    
    def reset_game(self):
        """Reset game to initial state."""
        self.game = RacingGame()
        self.state = self.game.initial
        self.move_count = 0
        self.legal_moves = []
        self.current_move_choice = None
        self.selected_piece = None
        self.ai_text.config(state=tk.NORMAL)
        self.ai_text.delete(1.0, tk.END)
        self.ai_text.config(state=tk.DISABLED)
        self.roll_dice()
    
    def update_display(self):
        """Update the board display and info."""
        # Draw board first
        self.draw_board()
        
        # Update info text
        info = f"Move {self.move_count}\n"
        info += f"Current Player: {'You (White)' if self.state.to_move == 'W' else 'AI (Black)'}\n"
        if self.state.chance:
            info += f"Die Roll: {self.state.chance}\n"
        info += f"Legal Moves: {len(self.legal_moves)} pieces\n\n"
        
        # Only show current player's pieces
        current_player = self.state.to_move
        if current_player == 'W':
            info += "Your Pieces (White - move right):\n"
            for row in range(2):
                pos = self.state.pieces['W'][row]
                is_movable = row in self.legal_moves
                status = "GOAL!" if pos >= 9 else f"Pos {pos}"
                move_indicator = " CAN MOVE" if is_movable else " (blocked)"
                info += f"  Row {row + 1}: {status}{move_indicator}\n"
        else:  # Black's turn
            info += "AI Pieces (Black - move left):\n"
            for row in range(2):
                pos = self.state.pieces['B'][row]
                is_movable = row in self.legal_moves
                status = "GOAL!" if pos <= 0 else f"Pos {pos}"
                move_indicator = " CAN MOVE" if is_movable else " (blocked)"
                info += f"  Row {row + 1}: {status}{move_indicator}\n"
        
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info)
        self.info_text.config(state=tk.DISABLED)
        
        # Check for terminal condition after drawing
        if self.game.terminal_test(self.state):
            winner = "You Win! (White)" if self.state.utility == 1 else "AI Wins! (Black)"
            messagebox.showinfo("Game Over", winner)
            self.reset_game()
            return
    
    def draw_board(self):
        """Draw the racing game as a 2x10 grid with pieces on both rows."""
        self.canvas.delete("all")
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
        
        # Grid parameters
        margin = 40
        grid_width = canvas_width - 2 * margin
        grid_height = canvas_height - 2 * margin
        
        num_rows = 2
        num_cols = 10
        
        cell_width = grid_width / num_cols
        cell_height = grid_height / num_rows
        
        # Draw grid lines
        for row in range(num_rows + 1):
            y = margin + row * cell_height
            self.canvas.create_line(margin, y, margin + grid_width, y, width=2, fill="black")
        
        for col in range(num_cols + 1):
            x = margin + col * cell_width
            self.canvas.create_line(x, margin, x, margin + grid_height, width=2, fill="black")
        
        # Label columns (positions 0-9)
        for col in range(num_cols):
            x = margin + (col + 0.5) * cell_width
            self.canvas.create_text(x, margin - 15, text=str(col), font=("Arial", 10, "bold"))
        
        # Row labels
        self.canvas.create_text(margin - 20, margin + 0.5 * cell_height,
                              text="R1", font=("Arial", 13, "bold"))
        self.canvas.create_text(margin - 20, margin + 1.5 * cell_height,
                              text="R2", font=("Arial", 13, "bold"))
        
        # Draw pieces on both rows (row 1 and row 2)
        for row in range(num_rows):
            # Draw White piece
            white_pos = self.state.pieces['W'][row]
            if 0 <= white_pos <= 9:
                col = int(white_pos)
                x1 = margin + col * cell_width
                y1 = margin + row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                is_selected = (self.selected_piece == row)
                outline_color = "blue" if is_selected else "black"
                outline_width = 4 if is_selected else 2
                
                # Draw white piece in bottom half of cell
                self.canvas.create_rectangle(x1, y1 + cell_height // 2, x2, y2, 
                                            fill="white", outline=outline_color, width=outline_width)
                self.canvas.create_text((x1 + x2) // 2, y1 + 3 * cell_height // 4, text="W", 
                                      font=("Arial", 14, "bold"), fill="black")
            elif white_pos > 9:
                self.canvas.create_text(margin + grid_width + 30, margin + (row + 0.5) * cell_height,
                                      text="GOAL", font=("Arial", 10, "bold"), fill="green")
            
            # Draw Black piece
            black_pos = self.state.pieces['B'][row]
            if 0 <= black_pos <= 9:
                col = int(black_pos)
                x1 = margin + col * cell_width
                y1 = margin + row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                # Draw black piece in top half of cell
                self.canvas.create_rectangle(x1, y1, x2, y1 + cell_height // 2, 
                                            fill="black", outline="black", width=2)
                self.canvas.create_text((x1 + x2) // 2, y1 + cell_height // 4, text="B", 
                                      font=("Arial", 14, "bold"), fill="white")
            elif black_pos < 0:
                self.canvas.create_text(margin + grid_width + 30, margin + (row + 0.5) * cell_height,
                                      text="GOAL", font=("Arial", 10, "bold"), fill="green")
    
    def update_info(self, message):
        """Update the info display with a message."""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.insert(tk.END, "\n" + message)
        self.info_text.config(state=tk.DISABLED)
        self.info_text.see(tk.END)


def main():
    """Start the GUI game."""
    root = tk.Tk()
    gui = RacingGameGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
