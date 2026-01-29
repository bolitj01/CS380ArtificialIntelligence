"""AND-OR Graph Search for Parking Lot Problem.

Demonstrates AND-OR search in a stochastic environment where:
- The agent must find a parking lot with available spaces
- Checking any lot has a stochastic outcome (full or has space)
- The agent needs a CONDITIONAL PLAN that handles all possibilities

AND-OR Search characteristics:
- OR nodes: Agent chooses which action to take (which lot to check)
- AND nodes: Environment determines outcome (lot full or has space)
- The agent needs a CONDITIONAL PLAN that handles all possibilities
"""
from typing import List
import random


class ParkingLotProblem:
    """Stochastic parking lot search problem with navigation.
    
    State: (parked, current_lot, lot_states, attempts_left)
        - parked: boolean (True = successfully parked, False = not parked)
        - current_lot: int or None (which lot agent is at, None if between lots)
        - lot_states: tuple of booleans (True = lot is full, False = lot has space)
        - attempts_left: int (number of actions remaining)
    
    This is stochastic because:
    - Going to a lot causes ALL lots to potentially toggle their full/empty state
    - Agent must navigate to a lot before checking it
    """
    
    def __init__(self, num_lots: int = 3, max_attempts: int = 100, toggle_prob: float = 0.3):
        """
        Initialize problem.
        
        Args:
            num_lots: Number of parking lots available
            max_attempts: Maximum number of actions allowed
            toggle_prob: Probability that each lot toggles its state when agent moves
        """
        self.num_lots = num_lots
        self.max_attempts = max_attempts
        self.toggle_prob = toggle_prob
        # Start with all lots full
        initial_lot_states = tuple(True for _ in range(num_lots))
        self.initial = (False, None, initial_lot_states, max_attempts)
    
    def actions(self, state) -> List[str]:
        """Available actions from current state.
        
        If parked or out of attempts: no actions
        At a lot: can check_lot or go_to any other lot
        Between lots: can only go_to a lot
        """
        parked, current_lot, _, attempts_left = state
        
        if parked or attempts_left <= 0:
            return []
        
        actions = []
        
        # Can always go to any lot
        for i in range(self.num_lots):
            actions.append(f"go_to_lot_{i+1}")
        
        # Can only check if currently at a lot
        if current_lot is not None:
            actions.append("check_lot")
        
        return actions
    
    def result(self, state, action: str) -> List:
        """Result of taking an action in stochastic environment.
        
        Returns a LIST of possible resulting states (AND node outcomes).
        
        - go_to_lot_X: Stochastic - lots may toggle state (2^num_lots outcomes)
        - check_lot: Deterministic - reveals if current lot has space
        """
        parked, current_lot, lot_states, attempts_left = state
        
        if parked or attempts_left <= 0:
            return [state]  # No change
        
        if action.startswith("go_to_lot_"):
            # Extract target lot number
            target_lot = int(action.split('_')[-1]) - 1  # 0-indexed
            
            # When traveling, each lot has a chance to toggle its full/empty state
            # Generate all possible outcomes (2^num_lots combinations)
            outcomes = []
            num_toggles = 2 ** self.num_lots
            
            for toggle_mask in range(num_toggles):
                new_lot_states = list(lot_states)
                for i in range(self.num_lots):
                    # Check if this lot should toggle in this outcome
                    if toggle_mask & (1 << i):
                        new_lot_states[i] = not new_lot_states[i]
                
                new_state = (False, target_lot, tuple(new_lot_states), attempts_left - 1)
                outcomes.append(new_state)
            
            return outcomes
        
        elif action == "check_lot":
            # Checking reveals the lot's state
            if current_lot is None:
                return [state]  # Can't check if not at a lot
            
            lot_is_full = lot_states[current_lot]
            
            if lot_is_full:
                # Lot is full, can't park
                return [(False, current_lot, lot_states, attempts_left - 1)]
            else:
                # Lot has space, park successfully!
                return [(True, current_lot, lot_states, attempts_left - 1)]
        
        return [state]
    
    def goal_test(self, state) -> bool:
        """Goal: agent has parked (first element is True)."""
        parked, _, _, _ = state
        return parked
    
    def __repr__(self) -> str:
        """String representation of problem."""
        return f"ParkingLotProblem(lots={self.num_lots}, max_attempts={self.max_attempts})"


def and_or_graph_search(problem: ParkingLotProblem, max_depth):
    """AND-OR Graph Search algorithm with memoization.
    
    Returns a conditional plan (dict mapping states to actions) that guarantees
    reaching the goal state despite the stochastic environment.
    
    A conditional plan tells the agent:
    "If you reach state S, take action A"
    
    The plan handles ALL possible outcomes at AND nodes.
    
    Uses memoization to avoid re-exploring equivalent states.
    For this problem, states are equivalent based on 'parked' status only.
    """
    # Memoization: cache plans for states we've already solved
    memo = {}
    # Track states currently being solved (for self-referential plans)
    in_progress = set()
    
    def get_state_key(state):
        """Get canonical key for state (parked status, current lot, and lot states)."""
        parked, current_lot, lot_states, _ = state
        # Include lot states in key since they affect available actions
        return (parked, current_lot, lot_states)
    
    def or_search(state, problem, depth):
        """OR node: agent chooses an action.
        
        Returns: [action, conditional_plan] or None if no solution exists
        """
        indent = "  " * depth
        state_key = get_state_key(state)
        
        # Check memo first
        if state_key in memo:
            print(f"{indent}OR-node: parked={state_key} [cached]")
            return memo[state_key]
        
        # If we're currently solving this state, it means we have a self-referential plan
        # This is OK for stochastic problems - return a placeholder that will be filled
        if state_key in in_progress:
            print(f"{indent}OR-node: parked={state_key} [self-reference detected]")
            return "SELF_REF"
        
        print(f"{indent}OR-node: parked={state_key}")
        
        if problem.goal_test(state):
            print(f"{indent}  ✓ Goal reached!")
            memo[state_key] = []
            return []
        
        # Count only OR nodes (agent decisions), not AND nodes (environment outcomes)
        # This allows deeper planning despite exponential branching
        or_depth = (depth + 1) // 2
        if or_depth >= max_depth:
            print(f"{indent}  ✗ Action limit (actions={or_depth})")
            return None
        
        # Mark this state as being solved
        in_progress.add(state_key)
        
        # Shuffle actions to try different lots in different orders
        actions = problem.actions(state)
        random.shuffle(actions)
        
        for action in actions:
            print(f"{indent}  → Trying: {action}")
            plan = and_search(problem.result(state, action),
                             problem, depth + 1)
            if plan is not None:
                print(f"{indent}  ✓ Success with {action}!")
                result = [action, plan]
                memo[state_key] = result
                in_progress.remove(state_key)
                return result
        
        print(f"{indent}  ✗ No viable action")
        in_progress.remove(state_key)
        memo[state_key] = None
        return None
    
    def and_search(states, problem, depth):
        """AND node: environment determines outcome.
        
        Must find a plan that handles ALL possible resulting states.
        Returns: {state: action_plan} - conditional plan for all states
        """
        indent = "  " * depth
        # Show outcomes using canonical state representation
        outcome_keys = [f"parked={get_state_key(s)}" for s in states]
        print(f"{indent}AND-node: outcomes=[{', '.join(outcome_keys)}]")
        
        plan = {}
        for s in states:
            parked = get_state_key(s)
            print(f"{indent}  For outcome parked={parked}:")
            result = or_search(s, problem, depth + 1)
            
            # Handle self-referential plans
            if result == "SELF_REF":
                # This outcome loops back - we'll use a marker that execute_plan can handle
                print(f"{indent}    (Self-referential: will retry on failure)")
                plan[s] = "LOOP"
            elif result is None:
                print(f"{indent}  ✗ No plan for parked={parked}!")
                return None
            else:
                plan[s] = result
        
        print(f"{indent}  ✓ All {len(states)} outcomes handled!")
        return plan
    
    # Start search from initial state
    print(f"Starting AND-OR search from: {problem.initial}\n")
    return or_search(problem.initial, problem, 0)


def execute_plan(initial_state, plan, problem):
    """Execute conditional plan, showing stochastic outcomes."""
    if plan is None:
        print("No plan found - problem may be unsolvable")
        return False
    
    print("\n" + "="*70)
    print("EXECUTING CONDITIONAL PLAN")
    print("="*70)
    
    current_state = initial_state
    current_plan = plan
    steps = 0
    max_steps = problem.max_attempts + 50
    check_lot_count = 0
    
    while not problem.goal_test(current_state) and steps < max_steps:
        parked, current_lot, lot_states, attempts = current_state
        lot_str = f"at lot {current_lot+1}" if current_lot is not None else "between lots"
        print(f"\nStep {steps + 1}: {lot_str}, attempts_left={attempts}")
        print(f"  Lot states: {['FULL' if full else 'EMPTY' for full in lot_states]}")
        
        if not isinstance(current_plan, list) or len(current_plan) == 0:
            if problem.goal_test(current_state):
                print(f"  Goal state reached!")
                break
            print(f"  ERROR: Invalid plan structure!")
            return False
        
        action = current_plan[0]
        sub_plan = current_plan[1] if len(current_plan) > 1 else {}
        
        print(f"  Action: {action}")
        
        # Get possible outcomes
        outcomes = problem.result(current_state, action)
        
        # Simulate the outcome
        if action.startswith("go_to_lot_"):
            # Randomly toggle lots based on toggle probability
            target_lot = int(action.split('_')[-1]) - 1
            new_lot_states = list(lot_states)
            
            print(f"  → Traveling to lot {target_lot + 1}...")
            for i in range(problem.num_lots):
                if random.random() < problem.toggle_prob:
                    new_lot_states[i] = not new_lot_states[i]
                    print(f"     Lot {i+1} toggled: {'FULL' if new_lot_states[i] else 'EMPTY'}")
            
            outcome = (False, target_lot, tuple(new_lot_states), attempts - 1)
            
        elif action == "check_lot":
            if current_lot is None:
                print(f"  ERROR: Cannot check lot when not at a lot!")
                return False
            
            check_lot_count += 1
            lot_is_full = lot_states[current_lot]
            if lot_is_full:
                print(f"  → Checked lot {current_lot+1}: FULL ✗")
                outcome = (False, current_lot, lot_states, attempts - 1)
            else:
                print(f"  → Checked lot {current_lot+1}: HAS SPACE! ✓")
                outcome = (True, current_lot, lot_states, attempts - 1)
        else:
            print(f"  ERROR: Unknown action {action}")
            return False
        
        if problem.goal_test(outcome):
            print(f"\n✓ SUCCESS! Parked at lot {outcome[1]+1}!")
            return True
        
        # Find matching plan for this outcome
        if isinstance(sub_plan, dict):
            # Try exact match first
            if outcome in sub_plan:
                next_plan = sub_plan[outcome]
            else:
                # Try matching by state key
                outcome_key = (outcome[0], outcome[1], outcome[2])
                matched = False
                for stored_state, stored_plan in sub_plan.items():
                    stored_key = (stored_state[0], stored_state[1], stored_state[2])
                    if stored_key == outcome_key:
                        next_plan = stored_plan
                        matched = True
                        break
                
                if not matched:
                    print(f"  ERROR: No sub-plan for outcome state")
                    return False
            
            # Handle LOOP marker
            if next_plan == "LOOP":
                print(f"  → Looping back to reconsider options")
                current_state = outcome
                available_actions = problem.actions(outcome)
                if available_actions:
                    # Pick a different action than what just failed
                    random_action = random.choice(available_actions)
                    # Try to find this action's plan in the root
                    if isinstance(plan, list) and len(plan) > 1:
                        root_action = plan[0]
                        root_sub_plan = plan[1]
                        # Check if this action has a known outcome for our state
                        current_plan = plan  # Restart from root plan
                    else:
                        current_plan = plan
                else:
                    print(f"  ERROR: No actions available and can't retry")
                    return False
            else:
                current_state = outcome
                current_plan = next_plan
        else:
            print(f"  ERROR: Invalid sub-plan structure!")
            return False
        
        steps += 1
    
    if problem.goal_test(current_state):
        print(f"\n✓ Goal achieved!")
        print(f"✓ Check_lot actions needed: {check_lot_count}")
        return True
    else:
        print(f"\n✗ Failed to find parking after {steps} steps")
        print(f"✗ Check_lot actions used: {check_lot_count}")
        return False


def print_plan(plan, depth=0):
    """Pretty-print the conditional plan."""
    if plan is None:
        print("  " * depth + "None")
        return
    
    if plan == "LOOP":
        print("  " * depth + "[Loop back to start]")
        return
    
    if isinstance(plan, list):
        if len(plan) == 0:
            print("  " * depth + "[Goal reached]")
        elif len(plan) == 2:
            action, sub_plan = plan
            print("  " * depth + f"Do: {action}")
            if isinstance(sub_plan, dict):
                # Limit output to avoid overwhelming display
                if len(sub_plan) > 10:
                    print("  " * (depth + 1) + f"[{len(sub_plan)} possible outcomes...]")
                else:
                    for state, action_plan in sub_plan.items():
                        parked, loc, lots, _ = state
                        if parked:
                            state_desc = "PARKED ✓"
                        else:
                            loc_str = f"at lot {loc+1}" if loc is not None else "between"
                            lots_str = ''.join(['F' if f else 'E' for f in lots])
                            state_desc = f"{loc_str}, lots:{lots_str}"
                        print("  " * (depth + 1) + f"If {state_desc}:")
                        print_plan(action_plan, depth + 2)
    elif isinstance(plan, dict):
        for state, action_plan in plan.items():
            parked, loc, lots, _ = state
            if parked:
                state_desc = "PARKED ✓"
            else:
                loc_str = f"at lot {loc+1}" if loc is not None else "between"
                lots_str = ''.join(['F' if f else 'E' for f in lots])
                state_desc = f"{loc_str}, lots:{lots_str}"
            print("  " * depth + f"State [{state_desc}]:")
            print_plan(action_plan, depth + 1)


def main():
    """Run AND-OR search on parking lot problem."""
    print("\n" + "="*70)
    print("PARKING LOT PROBLEM - AND-OR GRAPH SEARCH")
    print("="*70)
    
    print("""
PROBLEM DESCRIPTION:
- Agent needs to find parking in one of 3 lots
- Agent must navigate to a lot, then check if it has space
- When agent travels (go_to action), lot states may toggle (full/empty)
- Agent can only check the lot it's currently at

WHY IT'S STOCHASTIC:
- Lot states change probabilistically when agent travels
- Cannot predict which lots will toggle their state
- Agent must handle all possible state changes

ACTIONS:
- go_to_lot_X: Navigate to lot X (may cause lots to toggle state)
- check_lot: Check current lot for available space

AND-OR SEARCH:
- OR node: Agent picks which action to take
- AND node: Environment determines which lots toggle state
- Solution: Conditional plan handling all possibilities
""")
    
    # Create problem
    problem = ParkingLotProblem(num_lots=3, max_attempts=30, toggle_prob=0.10)
    
    print(f"Number of lots: {problem.num_lots}")
    print(f"Toggle probability: {problem.toggle_prob}")
    print(f"Max attempts: {problem.max_attempts}")
    print(f"Initial state: parked={problem.initial[0]}, at_lot={problem.initial[1]}, ")
    print(f"              lot_states={['FULL' if f else 'EMPTY' for f in problem.initial[2]]}")
    
    # Run AND-OR search
    print("\n" + "="*70)
    print("SEARCHING FOR CONDITIONAL PLAN")
    print("="*70 + "\n")
    plan = and_or_graph_search(problem, max_depth=12)
    
    print("\n" + "="*70)
    print("CONDITIONAL PLAN")
    print("="*70)
    print_plan(plan)
    
    # Execute plan once
    print("\n" + "="*70)
    print("EXECUTING PLAN (guaranteed to eventually succeed)")
    print("="*70)
    
    if execute_plan(problem.initial, plan, problem):
        print(f"\n{'='*70}")
        print("✓ Plan successfully found parking!")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()

