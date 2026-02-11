"""
Backtracking Search for N-Queens CSP

This program demonstrates backtracking search for solving the N-Queens problem.
By default, it uses simple backtracking without additional efficiencies.

To enable heuristics and see improved performance:
- Set use_mrv=True to use Minimum Remaining Values heuristic
- Set use_lcv=True to use Least Constraining Value heuristic
- Set use_forward_checking=True to use forward checking constraint propagation

Example with all heuristics enabled:
    searcher = BacktrackingSearcher(csp)
    solution = searcher.search(
        use_mrv=True,
        use_lcv=True,
        use_forward_checking=True
    )
"""

from typing import Dict, List, Optional, Set
from n_queens_csp import NQueensCSP


class BacktrackingSearcher:
    """
    Performs backtracking search on an N-Queens CSP with optional heuristics.
    """
    
    def __init__(self, csp: NQueensCSP):
        """
        Initialize the searcher with a CSP.
        
        Args:
            csp: The N-Queens CSP to solve
        """
        self.csp = csp
        self.nodes_explored = 0
        self.backtracks = 0
    
    def select_unassigned_variable(self, use_mrv: bool = True) -> Optional[int]:
        """
        Select next variable to assign using heuristics.
        
        Args:
            use_mrv: If True, use Minimum Remaining Values heuristic
            
        Returns:
            Next column to assign, or None if all assigned
        """
        unassigned = [col for col in self.csp.variables if self.csp.assignment[col] is None]
        
        if not unassigned:
            return None
        
        if use_mrv:
            # Choose variable with fewest legal values (MRV heuristic)
            return min(unassigned, key=lambda col: len(self.csp.domains[col]))
        else:
            # Just pick first unassigned
            return unassigned[0]
    
    def order_domain_values(self, var: int, use_lcv: bool = True) -> List[int]:
        """
        Order domain values for variable using heuristics.
        
        Args:
            var: Column to order values for
            use_lcv: If True, use Least Constraining Value heuristic
            
        Returns:
            Ordered list of row values to try
        """
        if not use_lcv:
            return list(self.csp.domains[var])
        
        # LCV: Prefer values that rule out fewest choices for neighbors
        def count_conflicts(value: int) -> int:
            """Count how many values this would eliminate from unassigned neighbors."""
            conflicts = 0
            col = var
            row = value
            
            for other_col in self.csp.variables:
                if self.csp.assignment[other_col] is not None:
                    continue
                if other_col == col:
                    continue
                
                # Count how many values in other_col's domain would conflict
                for other_row in self.csp.domains[other_col]:
                    # Same row conflict
                    if other_row == row:
                        conflicts += 1
                    # Diagonal conflict
                    elif abs(other_row - row) == abs(other_col - col):
                        conflicts += 1
            
            return conflicts
        
        values = list(self.csp.domains[var])
        # Sort by fewest conflicts (least constraining first)
        values.sort(key=count_conflicts)
        return values
    
    def forward_check(self, var: int, value: int) -> Optional[Dict[int, Set[int]]]:
        """
        Perform forward checking: remove inconsistent values from unassigned neighbors.
        
        Args:
            var: Column just assigned
            value: Row value assigned to var
            
        Returns:
            Dictionary of removed values for each variable (for backtracking), or None if inconsistent
        """
        removed = {col: set() for col in self.csp.variables}
        col = var
        row = value
        
        for other_col in self.csp.variables:
            if self.csp.assignment[other_col] is not None:
                continue
            if other_col == col:
                continue
            
            # Remove values from other_col's domain that conflict with (col, row)
            to_remove = set()
            for other_row in self.csp.domains[other_col]:
                # Same row conflict
                if other_row == row:
                    to_remove.add(other_row)
                # Diagonal conflict
                elif abs(other_row - row) == abs(other_col - col):
                    to_remove.add(other_row)
            
            # Remove the conflicting values
            self.csp.domains[other_col] -= to_remove
            removed[other_col] = to_remove
            
            # If domain becomes empty, this assignment is inconsistent
            if not self.csp.domains[other_col]:
                # Restore all previously removed values before returning
                self.restore_domains(removed)
                return None
        
        return removed
    
    def restore_domains(self, removed: Dict[int, Set[int]]):
        """
        Restore domain values that were removed during forward checking.
        
        Args:
            removed: Dictionary mapping variables to sets of removed values
        """
        for col, values in removed.items():
            self.csp.domains[col] |= values
    
    def search(self, use_mrv: bool = True, use_lcv: bool = True, 
               use_forward_checking: bool = True) -> Optional[Dict[int, int]]:
        """
        Solve N-Queens using backtracking search with optional heuristics.
        
        Args:
            use_mrv: Use Minimum Remaining Values heuristic
            use_lcv: Use Least Constraining Value heuristic
            use_forward_checking: Use forward checking for constraint propagation
            
        Returns:
            Complete assignment (column -> row mapping) if solution found, None otherwise
        """
        self.nodes_explored = 0
        self.backtracks = 0
        
        result = self._backtrack(use_mrv, use_lcv, use_forward_checking)
        return result
    
    def _backtrack(self, use_mrv: bool, use_lcv: bool, use_forward_checking: bool) -> Optional[Dict[int, int]]:
        """
        Recursive backtracking algorithm.
        
        Returns:
            Complete assignment if solution found, None otherwise
        """
        self.nodes_explored += 1
        
        # Check if assignment is complete
        if all(val is not None for val in self.csp.assignment.values()):
            return {k: v for k, v in self.csp.assignment.items()}  # Return complete assignment
        
        # Select unassigned variable
        var = self.select_unassigned_variable(use_mrv)
        
        # Try each value in the domain
        for value in self.order_domain_values(var, use_lcv):
            # Check consistency
            if self.csp.is_consistent(var, value):
                # Make assignment
                self.csp.assignment[var] = value
                
                # Forward checking
                if use_forward_checking:
                    removed = self.forward_check(var, value)
                    if removed is None:
                        # Forward checking detected inconsistency
                        self.csp.assignment[var] = None
                        self.backtracks += 1
                        continue
                else:
                    removed = None
                
                # Recurse
                result = self._backtrack(use_mrv, use_lcv, use_forward_checking)
                
                if result is not None:
                    return result
                
                # Backtrack
                self.backtracks += 1
                self.csp.assignment[var] = None
                
                # Restore domains if we used forward checking
                if removed is not None:
                    self.restore_domains(removed)
        
        return None


def main():
    """
    Main function to solve N-Queens using backtracking search.
    
    Heuristics are disabled by default. To enable them, see the module docstring.
    """
    # Set the number of queens
    N = 20
    
    print(f"\n{'='*60}")
    print(f"Backtracking Search for {N}-Queens Problem")
    print(f"{'='*60}")
    print("Default: No heuristics (pure backtracking)")
    print("To enable heuristics, see module docstring")
    print(f"{'='*60}\n")
    
    # Create CSP instance
    csp = NQueensCSP(N)
    
    # Create searcher
    searcher = BacktrackingSearcher(csp)
    
    # Solve with simple backtracking (heuristics disabled by default)
    # To enable heuristics, change the parameters to True
    solution = searcher.search(
        use_mrv=False,      # Set to True to enable MRV heuristic
        use_lcv=False,      # Set to True to enable LCV heuristic
        use_forward_checking=False  # Set to True to enable forward checking
    )
    
    # Display results
    if solution:
        print(f"\n✓ Solution found!")
        print(f"Solution: {[solution[col] for col in range(N)]}")
        print(f"\nInterpretation: Queen in column i is placed at row {solution}")
        print(f"\n{'='*50}")
        print(f"Search Statistics")
        print(f"{'='*50}")
        print(f"Nodes explored: {searcher.nodes_explored}")
        print(f"Backtracks: {searcher.backtracks}")
        print(f"{'='*50}\n")
        
        # Visualize the solution (skip for large boards)
        if N <= 50:
            csp.visualize_solution(solution)
        else:
            print(f"(Board size {N}x{N} is too large for GUI visualization)")
    else:
        print(f"\n✗ No solution found.")
        print(f"\n{'='*50}")
        print(f"Search Statistics")
        print(f"{'='*50}")
        print(f"Nodes explored: {searcher.nodes_explored}")
        print(f"Backtracks: {searcher.backtracks}")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
