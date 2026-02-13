"""
N-Queens Constraint Satisfaction Problem

This module defines the N-Queens problem as a CSP with backtracking search
and various constraint propagation and heuristic techniques.

Constraints:
1. One queen per column (enforced by representation)
2. No two queens can attack each other (no same row, no same diagonal)

USAGE:
------
Edit the configuration toggles in the main() function to change settings:
  - N: Board size (8-Queens, 20-Queens, etc.)
  - USE_MRV: Enable Minimum Remaining Values heuristic
  - USE_LCV: Enable Least Constraining Value heuristic
  - USE_FORWARD_CHECKING: Enable Forward Checking constraint propagation

The program will automatically visualize solutions for boards up to 50x50.

Example configurations:
  N=8, USE_MRV=True, USE_LCV=False       # Basic MRV
  N=8, USE_MRV=True, USE_LCV=True        # MRV + LCV
  N=20, USE_MRV=True, USE_LCV=False      # Larger 20-Queens
"""

import os
import sys
from typing import Dict, List, Optional, Set, Tuple

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from Ch4_Searching_in_Complex_Environments.n_queens import draw_board, QUEEN_IMG_PATH, QUEEN_IMG
from BacktrackingSearch import GeneralizedBacktrackingSearcher


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
        # Build neighbors for each column (all other columns)
        self.neighbors = {col: set(self.variables) - {col} for col in self.variables}
        

    
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
    
    def propagate_constraints(self) -> bool:
        """
        After an assignment, remove conflicting values from unassigned neighbors' domains.
        This dramatically reduces the search space.
        
        Returns:
            True if propagation succeeded, False if a domain became empty
        """
        # Find all assigned queens
        for assigned_col, assigned_row in self.assignment.items():
            if assigned_row is None:
                continue
            
            # For each unassigned column, remove conflicting rows
            for col in self.variables:
                if self.assignment[col] is not None:
                    continue
                
                # Remove rows that would conflict with this assigned queen
                rows_to_remove = set()
                for row in self.domains[col]:
                    # Same row conflict
                    if row == assigned_row:
                        rows_to_remove.add(row)
                    # Diagonal conflict
                    elif abs(row - assigned_row) == abs(col - assigned_col):
                        rows_to_remove.add(row)
                
                self.domains[col] -= rows_to_remove
                
                # If domain becomes empty, this path is invalid
                if not self.domains[col]:
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


def main():
    """
    Main function to solve a single N-Queens problem using backtracking search.
    
    Configure the following toggles before running:
    - N: Board size
    - USE_MRV: Use Minimum Remaining Values heuristic
    - USE_LCV: Use Least Constraining Value heuristic
    - VISUALIZE: Show GUI visualization of solution
    """
    
    # ===== CONFIGURATION TOGGLES =====
    N = 100                            # Board size (change this to solve different N-Queens)
    USE_MRV = True                      # Enable Minimum Remaining Values heuristic
    USE_LCV = False                     # Enable Least Constraining Value heuristic
    VISUALIZE = False                   # Show GUI visualization of solution (default: False)
    # ================================
    
    print(f"\n{'='*70}")
    print(f"N-Queens CSP Backtracking Search")
    print(f"{'='*70}\n")
    
    print(f"Board Size: {N}-Queens")
    print(f"Heuristics:")
    print(f"  MRV (Minimum Remaining Values): {USE_MRV}")
    print(f"  LCV (Least Constraining Value): {USE_LCV}")
    print(f"  Visualization: {VISUALIZE}\n")
    
    # Create CSP
    csp = NQueensCSP(N)
    searcher = GeneralizedBacktrackingSearcher(csp)
    
    # Solve the problem
    import time
    start_time = time.time()
    
    if searcher.search(use_mrv=USE_MRV, use_lcv=USE_LCV):
        end_time = time.time()
        solution = {col: csp.assignment[col] for col in csp.variables}
        
        print(f"✓ Solution found!")
        print(f"\nSolution: {[solution[col] for col in range(N)]}")
        print(f"\nSearch Statistics:")
        print(f"  Time: {end_time - start_time:.3f} seconds")
        print(f"  Nodes explored: {searcher.nodes_explored}")
        print(f"  Backtracks: {searcher.backtracks}")
        print(f"  Total assignments: {searcher.assignments}")
        
        # Visualize if enabled and board is not too large
        if VISUALIZE:
            if N <= 50:
                print(f"\nGenerating visualization...")
                csp.visualize_solution(solution)
            else:
                print(f"\n(Board size {N}x{N} is too large for GUI visualization)")
    else:
        end_time = time.time()
        print(f"✗ No solution found.")
        print(f"\nSearch Statistics:")
        print(f"  Time: {end_time - start_time:.3f} seconds")
        print(f"  Nodes explored: {searcher.nodes_explored}")
        print(f"  Backtracks: {searcher.backtracks}")
        print(f"  Total assignments: {searcher.assignments}")


if __name__ == "__main__":
    main()



