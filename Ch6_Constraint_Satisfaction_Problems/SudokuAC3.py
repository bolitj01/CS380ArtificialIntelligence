"""
Sudoku Solver using AC-3 Constraint Propagation

This program solves Sudoku puzzles purely through the AC-3 (Arc Consistency 3)
algorithm without backtracking. AC-3 achieves arc consistency by iteratively
removing values from variable domains that have no support from other variables.

The Sudoku puzzle is represented as a binary Constraint Satisfaction Problem:
- Variables: 81 cells in a 9x9 grid (indexed 0-80 where cell = row*9 + col)
- Domain: Each cell can have values 1-9 (0 represents blank in input)
- Constraints: All-different constraints for rows, columns, and 3x3 boxes

The AC-3 algorithm propagates constraints without any backtracking search,
relying entirely on domain reduction to achieve a unique solution.
"""

from typing import Dict, List, Optional, Set, Tuple
from collections import deque
import time
import random
from GeneralBacktracking import GeneralizedBacktrackingSearcher
from GeneralAC3 import GeneralAC3


class SudokuCSP:
    """
    Sudoku problem formulated as a binary Constraint Satisfaction Problem.
    
    Variables: 81 cells in 9x9 grid
    Domain: Values 1-9 for each cell
    Constraints: All-different constraints for rows, columns, and 3x3 boxes
    """
    
    def __init__(self, puzzle_str: str):
        """
        Initialize Sudoku CSP from puzzle string.
        
        Args:
            puzzle_str: 81-character string where '0' represents blank cells,
                       other digits 1-9 represent given values
        """
        self.puzzle_str = puzzle_str
        self.n = 9
        self.box_size = 3
        
        # Variables: cell indices 0-80 (row*9 + col)
        self.variables = list(range(81))
        
        # Domains: initial domain is 1-9 for empty cells, single value for given cells
        self.domains: Dict[int, Set[int]] = {}
        for i, char in enumerate(puzzle_str):
            if char == '0':
                self.domains[i] = set(range(1, 10))
            else:
                self.domains[i] = {int(char)}
        
        # Assignment: maps cell -> assigned value (None if not yet assigned)
        self.assignment: Dict[int, Optional[int]] = {i: None for i in self.variables}
        
        # Set initial assignments for given cells
        for i, char in enumerate(puzzle_str):
            if char != '0':
                self.assignment[i] = int(char)
        
        # Build constraint graph (neighbors for each cell)
        self._build_constraints()
        
        self.decisions_made = []  # Track decisions for output
        
    def _build_constraints(self):
        """Build the constraint graph - for each cell, find all its neighbors."""
        self.neighbors: Dict[int, Set[int]] = {cell: set() for cell in self.variables}
        
        for cell in self.variables:
            row, col = divmod(cell, 9)
            
            # Add row neighbors
            for c in range(9):
                neighbor = row * 9 + c
                if neighbor != cell:
                    self.neighbors[cell].add(neighbor)
            
            # Add column neighbors
            for r in range(9):
                neighbor = r * 9 + col
                if neighbor != cell:
                    self.neighbors[cell].add(neighbor)
            
            # Add 3x3 box neighbors
            box_row = (row // 3) * 3
            box_col = (col // 3) * 3
            for r in range(box_row, box_row + 3):
                for c in range(box_col, box_col + 3):
                    neighbor = r * 9 + c
                    if neighbor != cell:
                        self.neighbors[cell].add(neighbor)
    
    def is_complete(self) -> bool:
        """Check if all variables are assigned."""
        return all(self.assignment[var] is not None for var in self.variables)
    
    def is_consistent(self) -> bool:
        """Check if current assignment is consistent with constraints."""
        # Check rows
        for row in range(9):
            values = [self.assignment[row * 9 + col] for col in range(9)]
            assigned_values = [v for v in values if v is not None]
            if len(assigned_values) != len(set(assigned_values)):
                return False
        
        # Check columns
        for col in range(9):
            values = [self.assignment[row * 9 + col] for row in range(9)]
            assigned_values = [v for v in values if v is not None]
            if len(assigned_values) != len(set(assigned_values)):
                return False
        
        # Check boxes
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                values = []
                for r in range(box_row, box_row + 3):
                    for c in range(box_col, box_col + 3):
                        values.append(self.assignment[r * 9 + c])
                assigned_values = [v for v in values if v is not None]
                if len(assigned_values) != len(set(assigned_values)):
                    return False
        
        return True
    
    def propagate_constraints(self) -> bool:
        """
        Run AC-3 constraint propagation for backtracking support.
        
        Returns:
            True if arc consistency achieved, False if inconsistency detected
        """
        # Create an AC-3 solver and run it
        ac3_solver = GeneralAC3(self)
        return ac3_solver.ac3()


class SudokuAC3Solver:
    """
    Sudoku-specific AC-3 solver that uses the general AC-3 algorithm.
    """
    
    def __init__(self, csp: SudokuCSP):
        """Initialize solver with a Sudoku CSP."""
        self.csp = csp
        self.ac3_solver = GeneralAC3(csp)
        
    @property
    def constraint_checks(self) -> int:
        """Get constraint checks count from underlying AC-3 solver."""
        return self.ac3_solver.constraint_checks
    
    def solve(self) -> bool:
        """
        Solve the Sudoku puzzle using only AC-3 constraint propagation.
        
        Returns:
            True if solution found, False if inconsistency detected
        """
        print("Starting AC-3 constraint propagation...\n")
        
        # Run AC-3 to achieve arc consistency
        if not self.ac3_solver.ac3():
            print("AC-3 detected inconsistency!")
            return False
        
        # After AC-3, update assignment from domains
        # Cells with domain size 1 get their value
        # Cells with empty domain get marked as 'X'
        for cell in self.csp.variables:
            if len(self.csp.domains[cell]) == 1:
                self.csp.assignment[cell] = list(self.csp.domains[cell])[0]
            elif len(self.csp.domains[cell]) == 0:
                # Mark cells with empty domain as 'X'
                self.csp.assignment[cell] = 'X'
        
        # Check if complete
        if self.csp.is_complete():
            return True
        
        print(f"\nWarning: AC-3 did not find complete solution.")
        print(f"Cells with multiple values in domain:")
        for cell in self.csp.variables:
            if len(self.csp.domains[cell]) > 1:
                row, col = divmod(cell, 9)
                print(f"  Cell ({row}, {col}): {sorted(self.csp.domains[cell])}")
        
        return False


def print_sudoku(assignment: Dict[int, Optional[int]], title: str = "Sudoku"):
    """
    Print a formatted Sudoku grid.
    
    Args:
        assignment: Dictionary mapping cell index to value (or 'X' for no solution)
        title: Title to display
    """
    print(f"\n{title}:")
    print("=" * 31)
    
    for row in range(9):
        if row % 3 == 0 and row != 0:
            print("-" * 31)
        
        row_str = ""
        for col in range(9):
            if col % 3 == 0 and col != 0:
                row_str += "| "
            
            cell = row * 9 + col
            value = assignment[cell]
            if value is None:
                row_str += ". "
            elif value == 'X':
                row_str += "X "
            else:
                row_str += str(value) + " "
        
        print(row_str)
    
    print("=" * 31)


def puzzle_to_assignment(puzzle_str: str) -> Dict[int, Optional[int]]:
    """Convert puzzle string to assignment dictionary."""
    assignment = {}
    for i, char in enumerate(puzzle_str):
        if char == '0':
            assignment[i] = None
        else:
            assignment[i] = int(char)
    return assignment


def main():
    """Main function to solve a Sudoku puzzle."""
    
    # Change this variable to resort to backtracking when AC-3 fails to find complete solution
    use_backtracking = True
    
    # Read puzzles from file
    try:
        with open('Ch6_Constraint_Satisfaction_Problems/sudoku_puzzles.txt', 'r') as f:
            puzzles = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Error: sudoku_puzzles.txt not found")
        return
    
    if not puzzles:
        print("Error: No puzzles found in sudoku_puzzles.txt")
        return
    
    # Select random puzzle
    puzzle_index = random.randint(0, len(puzzles) - 1)
    puzzle_str = puzzles[puzzle_index]
    print(f"\nSolving Sudoku puzzle {puzzle_index + 1} of {len(puzzles)}")
    print(f"Backtracking enabled: {use_backtracking}")
    print(f"Puzzle: {puzzle_str}\n")
    
    # Create CSP and print initial state
    csp = SudokuCSP(puzzle_str)
    initial_assignment = puzzle_to_assignment(puzzle_str)
    print_sudoku(initial_assignment, "Initial Puzzle")
    
    # Solve with AC-3
    start_time = time.time()
    ac3_solver = SudokuAC3Solver(csp)
    ac3_solved = ac3_solver.solve()
    
    if ac3_solved:
        end_time = time.time()
        print_sudoku(ac3_solver.csp.assignment, "Solution Found (AC-3)")
        print(f"\nSolution Statistics:")
        print(f"  Time: {end_time - start_time:.3f} seconds")
        print(f"  Constraint checks: {ac3_solver.constraint_checks}")
    else:
        # Try backtracking if enabled
        if use_backtracking:
            print("\nAC-3 alone did not find complete solution.")
            print("Attempting backtracking search with AC-3 propagation...\n")
            
            # Create fresh CSP for backtracking
            csp = SudokuCSP(puzzle_str)
            backtrack_solver = GeneralizedBacktrackingSearcher(csp)
            
            if backtrack_solver.search():
                end_time = time.time()
                
                # Extract final values from domains
                for cell in backtrack_solver.csp.variables:
                    if backtrack_solver.csp.assignment[cell] is None:
                        if len(backtrack_solver.csp.domains[cell]) == 1:
                            backtrack_solver.csp.assignment[cell] = list(backtrack_solver.csp.domains[cell])[0]
                
                print_sudoku(backtrack_solver.csp.assignment, "Solution Found (AC-3 + Backtracking)")
                print(f"\nSolution Statistics:")
                print(f"  Time: {end_time - start_time:.3f} seconds")
                print(f"  Nodes explored: {backtrack_solver.nodes_explored}")
                print(f"  Backtracks: {backtrack_solver.backtracks}")
                print(f"  Assignments: {backtrack_solver.assignments}")
            else:
                end_time = time.time()
                print_sudoku(backtrack_solver.csp.assignment, "Partial Solution (Failed)")
                print(f"\nFailed to solve in {end_time - start_time:.3f} seconds")
                print(f"Nodes explored: {backtrack_solver.nodes_explored}")
                print(f"Backtracks: {backtrack_solver.backtracks}")
                print(f"\nNote: 'X' marks cells with no possible value")
                print(f"      '.' marks cells with multiple possible values")
        else:
            end_time = time.time()
            print_sudoku(ac3_solver.csp.assignment, "Partial Solution (Failed - Backtracking Disabled)")
            print(f"\nFailed to solve in {end_time - start_time:.3f} seconds")
            print(f"Constraint checks: {ac3_solver.constraint_checks}")
            print(f"\nNote: 'X' marks cells with no possible value (unsolvable puzzle)")
            print(f"      '.' marks cells with multiple possible values")
            print(f"      Use backtracking option to attempt to solve harder puzzles")


if __name__ == "__main__":
    main()
