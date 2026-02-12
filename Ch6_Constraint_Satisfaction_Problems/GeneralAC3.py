"""
General AC-3 Constraint Propagation Algorithm

This module implements a generalized AC-3 (Arc Consistency 3) solver that works
with any binary Constraint Satisfaction Problem (CSP).

The AC-3 algorithm achieves arc consistency by iteratively removing values from
variable domains that have no support from other variables' domains.

Algorithm Overview:
1. Initialize a queue with all arcs (Xi, Xj) where i != j
2. While queue is not empty:
   - Remove arc (Xi, Xj) from queue
   - If REVISE(Xi, Xj) removed values from Xi's domain:
     - If Xi's domain is empty, return failure
     - Add all arcs (Xk, Xi) back to queue (where Xk != Xj)
3. Return success

Time Complexity: O(e * d³) where e = number of arcs, d = domain size
Space Complexity: O(e) for the queue
"""

from typing import Protocol, Dict, Set, Optional
from collections import deque


class CSPInterface(Protocol):
    """
    Protocol that any CSP must implement to use GeneralAC3.
    
    A CSP needs:
    - variables: List of variable identifiers
    - domains: Dict mapping variables to their possible values (as sets)
    - neighbors: Dict mapping each variable to its neighbors in constraint graph
    """
    variables: list
    domains: Dict
    neighbors: Dict


class GeneralAC3:
    """
    General AC-3 solver for any binary Constraint Satisfaction Problem.
    
    This solver works with any CSP that implements the CSPInterface protocol,
    providing domain reduction through arc consistency.
    """
    
    def __init__(self, csp: CSPInterface):
        """
        Initialize the AC-3 solver.
        
        Args:
            csp: A CSP object with variables, domains, and neighbors attributes
        """
        self.csp = csp
        self.constraint_checks = 0
    
    def ac3(self) -> bool:
        """
        Execute AC-3 algorithm to achieve arc consistency.
        
        Returns:
            True if arc consistency is achieved (problem may be solvable),
            False if an inconsistency is detected (empty domain found)
        """
        # Queue of all arcs (Xi, Xj) where Xi != Xj
        queue = deque()
        for xi in self.csp.variables:
            for xj in self.csp.neighbors.get(xi, set()):
                queue.append((xi, xj))
        
        while queue:
            xi, xj = queue.popleft()
            
            if self._revise(xi, xj):
                # Domain of Xi was revised
                if not self.csp.domains[xi]:
                    # Domain is empty - inconsistency detected
                    return False
                
                # Add all neighbors of Xi back to queue (except Xj)
                for xk in self.csp.neighbors.get(xi, set()):
                    if xk != xj:
                        queue.append((xk, xi))
        
        return True
    
    def _revise(self, xi, xj) -> bool:
        """
        Revise domain of Xi based on constraint with Xj.
        Remove values from Xi's domain that have no support in Xj.
        
        For binary constraints with "all different" semantics:
        - A value x in Xi has support if there exists y in Xj where x != y
        
        Args:
            xi: First variable in the arc
            xj: Second variable in the arc
        
        Returns:
            True if domain was revised (values removed), False otherwise
        """
        revised = False
        
        # Check each value in Xi's domain
        for x in list(self.csp.domains[xi]):
            # Check if there's a value in Xj's domain that supports this value
            has_support = False
            for y in self.csp.domains[xj]:
                # Binary "all different" constraint: values must be different
                if x != y:
                    has_support = True
                    break
            
            # If no support found, remove value from Xi's domain
            if not has_support:
                self.csp.domains[xi].remove(x)
                revised = True
            
            self.constraint_checks += 1
        
        return revised
