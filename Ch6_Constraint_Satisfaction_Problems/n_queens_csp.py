"""
N-Queens Constraint Satisfaction Problem

This module defines the N-Queens problem as a CSP with backtracking search
and various constraint propagation and heuristic techniques.

Constraints:
1. One queen per column (enforced by representation)
2. No two queens can attack each other (no same row, no same diagonal)
"""

import os
import sys
from typing import Dict, List, Optional, Set, Tuple
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Add Ch4 to path to import existing utilities
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CH4_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "Ch4_Searching_in_Complex_Environments")
sys.path.insert(0, CH4_DIR)

from n_queens import draw_board, QUEEN_IMG_PATH, QUEEN_IMG


class NQueensCSP:
    """
    N-Queens problem formulated as a Constraint Satisfaction Problem.
    
    Variables: Columns (0 to n-1)
    Domain: Rows (0 to n-1) for each column
    Constraints: No two queens attack each other
    """
    
    def __init__(self, n: int):
        """
        Initialize N-Queens CSP.
        
        Args:
            n: Board size (n x n)
        """
        self.n = n
        # Variables are columns (we assign one queen per column)
        self.variables = list(range(n))
        # Initial domain: each column can have queen in any row
        self.domains = {col: set(range(n)) for col in self.variables}
        # Assignment: maps column -> row (None if unassigned)
        self.assignment: Dict[int, Optional[int]] = {col: None for col in self.variables}
        

    
    def is_consistent(self, var: int, value: int) -> bool:
        """
        Check if assigning value to var is consistent with current assignment.
        
        Args:
            var: Column to assign
            value: Row to place queen
            
        Returns:
            True if assignment is consistent with constraints
        """
        col = var
        row = value
        
        # Check against all previously assigned queens
        for assigned_col, assigned_row in self.assignment.items():
            if assigned_row is None:
                continue
            
            # Can't be in same row
            if assigned_row == row:
                return False
            
            # Can't be on same diagonal
            if abs(assigned_row - row) == abs(assigned_col - col):
                return False
        
        return True
    

    
    def get_state_tuple(self) -> Tuple[int, ...]:
        """
        Convert current assignment to state tuple for visualization.
        
        Returns:
            Tuple of row positions (one per column)
        """
        return tuple(self.assignment[col] if self.assignment[col] is not None else -1 
                    for col in self.variables)
    
    def visualize_solution(self, solution: Dict[int, int]):
        """
        Visualize the solution on a chessboard.
        
        Args:
            solution: Complete assignment (column -> row mapping)
        """
        state = tuple(solution[col] for col in sorted(solution.keys()))
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        draw_board(ax, state, f"{self.n}-Queens CSP Solution")
        plt.tight_layout()
        plt.show()
    
    def print_statistics(self):
        """Print search statistics."""
        print(f"\n{'='*50}")
        print(f"CSP Backtracking Search Statistics")
        print(f"{'='*50}")
        print(f"Nodes explored: {self.nodes_explored}")
        print(f"Backtracks: {self.backtracks}")
        print(f"{'='*50}\n")



