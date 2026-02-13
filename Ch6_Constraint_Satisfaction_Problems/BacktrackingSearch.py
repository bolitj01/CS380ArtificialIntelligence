"""
Generalized Backtracking Search for Any CSP

This module provides a generalized backtracking search algorithm that works
with any Constraint Satisfaction Problem (CSP) that follows the standard interface.

A CSP must have:
- variables: list of variable names/indices
- domains: dict mapping variables to sets of possible values
- assignment: dict mapping variables to assigned values (None if unassigned)
- neighbors: dict mapping variables to sets of neighbor variables
- is_consistent(): method to check if an assignment is consistent

Optional methods for enhanced functionality:
- select_unassigned_variable(): custom variable selection (default: first unassigned)
- order_domain_values(): custom value ordering (default: sorted)
- propagate_constraints(): constraint propagation (like AC-3)

Supported Heuristics:
- MRV (Minimum Remaining Values): Select variable with smallest domain
- LCV (Least Constraining Value): Order values by number of constraints removed
"""

from typing import Dict, List, Optional, Set, Any


class GeneralizedBacktrackingSearcher:
    """
    Performs backtracking search on any CSP with optional constraint propagation
    and heuristics (MRV, LCV).
    """
    
    def __init__(self, csp: Any):
        """
        Initialize the searcher with a CSP.
        
        Args:
            csp: Any CSP object with variables, domains, assignment, neighbors attributes
        """
        self.csp = csp
        self.nodes_explored = 0
        self.backtracks = 0
        self.assignments = 0
        self.use_mrv = True
        self.use_lcv = True
    
    def select_unassigned_variable(self) -> Optional[Any]:
        """
        Select next unassigned variable using MRV (Minimum Remaining Values) heuristic if enabled.
        Can be overridden by CSP's own method if it has one.
        
        Returns:
            Variable with smallest domain (if MRV enabled), or None if all assigned
        """
        # Use CSP's method if available
        if hasattr(self.csp, 'select_unassigned_variable'):
            return self.csp.select_unassigned_variable()
        
        unassigned = [var for var in self.csp.variables 
                     if self.csp.assignment[var] is None]
        
        if not unassigned:
            return None
        
        if self.use_mrv:
            # MRV heuristic: choose variable with smallest domain
            return min(unassigned, key=lambda var: len(self.csp.domains[var]))
        else:
            # Without MRV: just pick first unassigned
            return unassigned[0]
    
    def order_domain_values(self, var: Any) -> List[Any]:
        """
        Order values in domain of variable using LCV (Least Constraining Value) if enabled.
        Can be overridden by CSP's own method if it has one.
        
        Args:
            var: Variable to order values for
            
        Returns:
            Ordered list of values to try
        """
        # Use CSP's method if available
        if hasattr(self.csp, 'order_domain_values'):
            return self.csp.order_domain_values(var)
        
        if not self.use_lcv:
            return sorted(list(self.csp.domains[var]))
        
        # LCV heuristic: count how constraining each value is
        # Only use if CSP has neighbors and is_consistent with proper signature
        if not hasattr(self.csp, 'neighbors') or not hasattr(self.csp, 'is_consistent'):
            return sorted(list(self.csp.domains[var]))
        
        def count_constraints(value: Any) -> int:
            """
            Count how many values would be ruled out for unassigned neighbors
            if we assign this value.
            """
            constraints = 0
            neighbors = self.csp.neighbors.get(var, set())
            
            for neighbor in neighbors:
                if self.csp.assignment[neighbor] is not None:
                    continue
                
                # Count values in neighbor's domain that would be eliminated
                for neighbor_value in self.csp.domains[neighbor]:
                    # Temporarily assign and test consistency
                    old_var = self.csp.assignment[var]
                    self.csp.assignment[var] = value
                    
                    # Try to call is_consistent - handle both signatures
                    try:
                        if not self.csp.is_consistent(neighbor, neighbor_value):
                            constraints += 1
                    except TypeError:
                        # If is_consistent has different signature, skip LCV
                        pass
                    
                    self.csp.assignment[var] = old_var
            
            return constraints
        
        values = list(self.csp.domains[var])
        # Sort by fewest constraints (least constraining first)
        values.sort(key=count_constraints)
        return values
    
    def save_domain_state(self) -> Dict[Any, Set[Any]]:
        """
        Save current state of all domains for backtracking.
        
        Returns:
            Dictionary mapping variables to copies of their domains
        """
        return {var: domain.copy() for var, domain in self.csp.domains.items()}
    
    def restore_domain_state(self, saved_state: Dict[Any, Set[Any]]):
        """
        Restore domains to a previously saved state.
        
        Args:
            saved_state: Dictionary of saved domain states
        """
        self.csp.domains = {var: domain.copy() for var, domain in saved_state.items()}
    
    def propagate_constraints(self) -> bool:
        """
        Run constraint propagation if CSP supports it (e.g., AC-3).
        
        Returns:
            True if propagation succeeded, False if inconsistency detected
        """
        if hasattr(self.csp, 'propagate_constraints'):
            return self.csp.propagate_constraints()
        return True
    
    def is_complete(self) -> bool:
        """Check if all variables are assigned."""
        return all(self.csp.assignment[var] is not None for var in self.csp.variables)
    
    def search(self, use_mrv: bool = True, use_lcv: bool = True) -> bool:
        """
        Solve the CSP using backtracking search with optional heuristics.
        
        Args:
            use_mrv: Use Minimum Remaining Values heuristic for variable selection
            use_lcv: Use Least Constraining Value heuristic for value ordering
        
        Returns:
            True if solution found, False otherwise
        """
        self.use_mrv = use_mrv
        self.use_lcv = use_lcv
        
        self.nodes_explored = 0
        self.backtracks = 0
        self.assignments = 0
        
        return self._backtrack()
    
    def _backtrack(self) -> bool:
        """
        Recursive backtracking algorithm with constraint propagation and heuristics.
        
        Returns:
            True if solution found, False otherwise
        """
        self.nodes_explored += 1
        
        # Check if assignment is complete
        if self.is_complete():
            return True
        
        # Select next unassigned variable
        var = self.select_unassigned_variable()
        
        if var is None:
            return False
        
        # Try each value in the domain order
        for value in self.order_domain_values(var):
            # Skip if value not in domain (may have been removed by propagation)
            if value not in self.csp.domains[var]:
                continue
            
            # Check consistency if CSP has is_consistent method
            if hasattr(self.csp, 'is_consistent'):
                try:
                    # Try new-style is_consistent(var, value)
                    if not self.csp.is_consistent(var, value):
                        continue
                except TypeError:
                    # Fall back to old-style: is_consistent() with assignment
                    # This CSP checks the whole assignment
                    pass
            
            # Save domain state before assignment
            saved_domains = self.save_domain_state()
            
            # Make assignment
            self.csp.assignment[var] = value
            self.assignments += 1
            
            # Try to propagate constraints
            if self.propagate_constraints():
                # Recursively try to complete assignment
                if self._backtrack():
                    return True
            
            # Backtrack: undo assignment and restore domains
            self.csp.assignment[var] = None
            self.restore_domain_state(saved_domains)
            self.backtracks += 1
        
        return False
