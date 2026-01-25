import time
from typing import Tuple

import matplotlib.pyplot as plt

from n_queens import random_state, conflicts, value, draw_board, draw_fitness_landscape

# Configuration
N = 12  # size of the board and number of queens
MAX_RESTARTS = 100


def hill_climb(initial: Tuple[int, ...], value_fn) -> Tuple[Tuple[int, ...], int, list, int]:
    """Simple hill climbing: move to the best neighbor with lowest value.
    Returns the final state, number of steps, trajectory, and evaluation count.
    
    Args:
        initial: Initial state
        value_fn: Function to evaluate state quality (lower is better - minimization)
        
    Returns:
        Tuple of (final_state, steps, trajectory, evaluations)
    """
    current = initial
    steps = 0
    evaluations = 1  # Count initial state evaluation
    max_steps = 100  # Prevent infinite loops
    trajectory = [current]  # Track all states visited
    
    while steps < max_steps:
        current_value = value_fn(current)
        
        # Find best neighbor (lowest value for minimization)
        best_neighbor = current
        best_value = current_value
        
        n = len(current)
        for col in range(n):
            current_row = current[col]
            for row in range(n):
                if row != current_row:
                    # Create neighbor by moving queen in column col to row
                    neighbor = list(current)
                    neighbor[col] = row
                    neighbor_tuple = tuple(neighbor)
                    neighbor_value = value_fn(neighbor_tuple)
                    evaluations += 1  # Count each neighbor evaluation
                    
                    if neighbor_value < best_value:  # MINIMIZATION: seek lower values
                        best_value = neighbor_value
                        best_neighbor = neighbor_tuple
        
        if best_value >= current_value:
            break  # local minimum - no improvement
        
        current = best_neighbor
        trajectory.append(current)
        steps += 1
    
    return current, steps, trajectory, evaluations


def hill_climb_with_restarts(n: int, max_restarts: int, random_fn, value_fn, conflicts_fn):
    """Run hill climbing with random restarts until a solution is found or max restarts reached.
    
    Args:
        n: Problem size
        max_restarts: Maximum number of random restarts
        random_fn: Function to generate random initial state
        value_fn: Function to evaluate state quality (conflicts - lower is better)
        conflicts_fn: Function to count conflicts/violations
        
    Returns:
        Tuple of (initial_state, final_state, steps, restart_number, trajectory, performance_metrics)
    """
    best_state = None
    best_value = float('inf')  # For minimization, start with infinity
    total_evaluations = 0
    start_time = time.time()
    
    for restart in range(max_restarts):
        start = random_fn(n)
        end, steps, trajectory, evaluations = hill_climb(start, value_fn)
        total_evaluations += evaluations
        end_value = value_fn(end)
        end_conflicts = conflicts_fn(end)
        
        print(f"Restart {restart}: {steps} steps, {evaluations} evaluations → {end_conflicts} attacking pairs (goal: 0)")
        
        if end_value < best_value:  # MINIMIZATION: seek lower values
            best_value = end_value
            best_state = (start, end, steps, restart, trajectory)
            
        if conflicts_fn(end) == 0:
            elapsed_time = time.time() - start_time
            print(f"✓ Solution found! No attacking queen pairs.")
            
            # Compile performance metrics
            metrics = {
                'algorithm': 'Hill Climbing with Restarts',
                'success': True,
                'restarts': restart,
                'total_evaluations': total_evaluations,
                'final_steps': steps,
                'final_conflicts': end_conflicts,
                'time_seconds': elapsed_time,
                'problem_size': n
            }
            return start, end, steps, restart, trajectory, metrics
    
    # If no solution, return the best we saw
    elapsed_time = time.time() - start_time
    print(f"\nNo perfect solution found in {max_restarts} restarts.")
    print(f"Returning best state with {conflicts_fn(best_state[1])} attacking pairs (goal was 0).")
    
    # Compile performance metrics for failed attempt
    metrics = {
        'algorithm': 'Hill Climbing with Restarts',
        'success': False,
        'restarts': max_restarts - 1,
        'total_evaluations': total_evaluations,
        'final_steps': best_state[2],
        'final_conflicts': conflicts_fn(best_state[1]),
        'time_seconds': elapsed_time,
        'problem_size': n
    }
    return best_state[0], best_state[1], best_state[2], best_state[3], best_state[4], metrics


def main():
    """Solve N-Queens using hill climbing with random restarts."""
    start, end, steps, restart, trajectory, metrics = hill_climb_with_restarts(
        N, MAX_RESTARTS, random_state, value, conflicts
    )
    start_conflicts = conflicts(start)
    end_conflicts = conflicts(end)

    print(f"\n{'='*60}")
    print(f"N-Queens Hill Climbing Results (N={N})")
    print(f"{'='*60}")
    print(f"Restarts used: {restart}")
    print(f"Steps in final climb: {steps}")
    print(f"Total state evaluations: {metrics['total_evaluations']}")
    print(f"Time elapsed: {metrics['time_seconds']:.4f} seconds")
    print(f"\nStart state: {start}")
    print(f"  → {start_conflicts} attacking pairs (queens threatening each other)")
    print(f"\nEnd state:   {end}")
    print(f"  → {end_conflicts} attacking pairs (goal: 0)")
    
    if end_conflicts == 0:
        print(f"\n🎉 SUCCESS! Found a valid {N}-Queens solution!")
    else:
        print(f"\n⚠ Stuck at local minimum with {end_conflicts} conflicts remaining.")
    print(f"{'='*60}")

    # Create visualization with 3 subplots: start board, end board, fitness landscape
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    draw_board(axes[0], start, f"Start State\n{start_conflicts} attacking pairs")
    draw_board(axes[1], end, f"Final State\n{end_conflicts} attacking pairs")
    draw_fitness_landscape(axes[2], trajectory, value)
    
    plt.suptitle(f"Hill Climbing for {N}-Queens", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
