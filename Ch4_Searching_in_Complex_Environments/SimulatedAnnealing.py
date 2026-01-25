import time
import random
import math
from typing import Tuple, Dict, Callable

import matplotlib.pyplot as plt
import numpy as np

from n_queens import random_state, conflicts, value, draw_board, draw_fitness_landscape

# Configuration
N = 12  # size of the board and number of queens
K = 22  # scaling factor for temperature schedule
LAM = 0.005  # cooling rate for exponential decay
LIMIT = 2500  # iteration limit for schedule
# Explanation of parameters:
# K: Higher values lead to higher initial temperatures, allowing more exploration.
# LAM: Higher values lead to faster cooling, reducing exploration sooner.
# LIMIT: After this many iterations, temperature drops to zero.


def exp_schedule(k=K, lam=LAM, limit=LIMIT) -> Callable:
    """Exponential temperature schedule for simulated annealing.
    
    Args:
        k: Scaling factor (higher = higher initial temperature)
        lam: Cooling rate (higher = faster cooling)
        limit: Iteration limit (after this, temperature is 0)
        
    Returns:
        A function that takes iteration t and returns temperature T
    """
    # A lambda function is an anonymous function in Python that can take any number of arguments but has only one expression.
    return lambda t: (k * np.exp(-lam * t) if t < limit else 0)


def probability(p):
    """Return True with probability p to decide on accepting a worse state.
    
    Args:
        p: Probability value in [0, 1]
        
    Returns:
        True with probability p, False otherwise
    """
    return random.random() < p


def simulated_annealing(n: int, value_fn: Callable, schedule: Callable) -> Tuple[Tuple[int, ...], Tuple[int, ...], int, list, int]:
    """Simulated annealing: probabilistically accept worse moves to escape local optima.
    
    Returns the initial state, final state, number of iterations, trajectory, and evaluation count.
    
    Args:
        n: Problem size (board size)
        value_fn: Function to evaluate state quality (lower is better - minimization)
        schedule: Temperature schedule function that takes iteration t
        
    Returns:
        Tuple of (initial_state, final_state, iterations, trajectory, evaluations)
    """
    initial = random_state(n)
    current = initial
    iterations = 0  # Count of accepted state transitions (moves actually taken)
    evaluations = 1  # Count initial state evaluation
    trajectory = [current]  # Track all states visited
    current_value = value_fn(current)
    
    time_step = 0  # Total loop iterations for temperature schedule (includes rejected moves)
    while True:
        temperature = schedule(time_step)
        if temperature == 0:
            # Temperature cooled to 0 - done
            return initial, current, iterations, trajectory, evaluations
        
        # Generate random neighbor by moving one queen
        col = random.randint(0, n - 1)  # Random column
        row = random.randint(0, n - 1)  # Random row
        
        # Skip if no move needed (queen already at that row)
        if current[col] == row:
            time_step += 1
            continue
        
        # Create neighbor state
        neighbor = list(current)
        neighbor[col] = row
        neighbor_tuple = tuple(neighbor)
        neighbor_value = value_fn(neighbor_tuple)
        evaluations += 1
        
        # Decision rule: accept if better, or probabilistically if worse
        delta_e = neighbor_value - current_value  # For minimization, we want delta_e < 0
        
        # For minimization: accept if delta_e < 0 (better) or probabilistically if delta_e >= 0
        if delta_e < 0 or probability(math.exp(-delta_e / temperature)):
            current = neighbor_tuple
            current_value = neighbor_value
            trajectory.append(current)
            iterations += 1
            
            # Terminate early if solution found
            if neighbor_value == 0:
                return initial, current, iterations, trajectory, evaluations
        
        time_step += 1
        
        # Safety limit to prevent infinite loops
        if time_step > LIMIT * 2:
            return initial, current, iterations, trajectory, evaluations



def main():
    """Solve N-Queens using simulated annealing with tuned temperature schedule."""
    schedule = exp_schedule()  # Create schedule with current parameters
    start, end, iterations, trajectory, evaluations = simulated_annealing(N, value, schedule)
    
    start_conflicts = conflicts(start)
    end_conflicts = conflicts(end)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"N-Queens Simulated Annealing Results (N={N})")
    print(f"{'='*60}")
    print(f"Iterations: {iterations}")
    print(f"Total state evaluations: {evaluations}")
    print(f"Temperature Schedule: exp(-{LAM}*t) * {K} (limit: {LIMIT})")
    print(f"\nStart state: {start}")
    print(f"  → {start_conflicts} attacking pairs (queens threatening each other)")
    print(f"\nEnd state:   {end}")
    print(f"  → {end_conflicts} attacking pairs (goal: 0)")
    
    if end_conflicts == 0:
        print(f"\n🎉 SUCCESS! Found a valid {N}-Queens solution!")
    else:
        print(f"\n⚠ Did not find a perfect solution with {end_conflicts} conflicts remaining.")
        print(f"Try adjusting K, LAM, or LIMIT to explore longer or with different initial temperature.")
    print(f"{'='*60}")
    
    # Create visualization with 3 subplots: start board, end board, fitness landscape
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    draw_board(axes[0], start, f"Start State\n{start_conflicts} attacking pairs")
    draw_board(axes[1], end, f"Final State\n{end_conflicts} attacking pairs")
    draw_fitness_landscape(axes[2], trajectory, value)
    
    plt.suptitle(f"Simulated Annealing for {N}-Queens", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
