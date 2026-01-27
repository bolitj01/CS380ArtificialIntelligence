import random
import math
from typing import Tuple, Callable

import matplotlib.pyplot as plt
import numpy as np

from tsp import random_cities, random_tour, value, draw_tour_comparison, tour_distance, count_intersections

# Configuration
NUM_CITIES = 15  # Number of cities in the TSP
K = 40  # Scaling factor for temperature schedule
LAM = 0.03  # Cooling rate for exponential decay
LIMIT = 1000  # Iteration limit for schedule
UPDATE_INTERVAL_MS = 300  # Milliseconds between visualization updates

# Explanation of parameters:
# K: Higher values lead to higher initial temperatures, allowing more exploration.
# LAM: Higher values lead to faster cooling, reducing exploration sooner.
# LIMIT: After this many iterations, temperature drops to zero.
# UPDATE_INTERVAL_MS: How often to refresh the visualization (200ms = 5 updates/second)


def exp_schedule(k=K, lam=LAM, limit=LIMIT) -> Callable:
    """Exponential temperature schedule for simulated annealing. Based on the standard formula Temperature(time_step) = k * e^(-lam * time_step).
    
    Args:
        k: Scaling factor (higher = higher initial temperature)
        lam: Cooling rate (higher = faster cooling)
        time_step: current iteration/time step
        limit: Iteration limit (after this, temperature is 0)
        
    Returns:
        A function that takes iteration time_step and returns temperature
    """
    return lambda time_step: (k * np.exp(-lam * time_step) if time_step < limit else 0)


def probability(prob):
    """Return True with probability prob to decide on accepting a worse state.
    
    Args:
        prob: Probability value in [0, 1]
        
    Returns:
        True with probability prob, False otherwise
    """
    return random.random() < prob


def swap_neighbor(tour: Tuple[int, ...]) -> Tuple[int, ...]:
    """Generate a neighbor by swapping two random cities in the tour.
    
    Args:
        tour: Current tour
        
    Returns:
        New tour with two cities swapped
    """
    num_cities = len(tour)
    tour_list = list(tour)
    
    # Pick two random positions
    idx1, idx2 = random.sample(range(num_cities), 2)
    
    # Swap them
    tour_list[idx1], tour_list[idx2] = tour_list[idx2], tour_list[idx1]
    
    return tuple(tour_list)


def reverse_neighbor(tour: Tuple[int, ...]) -> Tuple[int, ...]:
    """Generate a neighbor by reversing a random subsequence of the tour.
    
    This is known as a 2-opt move and often produces better neighbors for TSP.
    
    Args:
        tour: Current tour
        
    Returns:
        New tour with a subsequence reversed
    """
    num_cities = len(tour)
    tour_list = list(tour)
    
    # Pick two random positions
    idx1, idx2 = sorted(random.sample(range(num_cities), 2))
    
    # Reverse the subsequence between them
    tour_list[idx1:idx2+1] = reversed(tour_list[idx1:idx2+1])
    
    return tuple(tour_list)


def simulated_annealing_tsp_realtime(num_cities: int, cities: list, value_fn: Callable, 
                                     schedule: Callable, update_interval_ms: int = 200):
    """Simulated annealing for TSP with real-time visualization updates.
    
    Args:
        num_cities: Number of cities
        cities: List of city coordinates
        value_fn: Function to evaluate tour quality (lower is better)
        schedule: Temperature schedule function
        update_interval_ms: Not used - updates happen on every accepted move
        
    Returns:
        Tuple of (initial_tour, final_tour, iterations, evaluations)
    """
    # Initialize tour and tracking
    initial = random_tour(num_cities)
    current = initial
    iterations = 0  # Count of accepted state transitions
    evaluations = 1  # Count of state evaluations
    current_value = value_fn(current, cities)
    
    # Setup real-time visualization
    plt.ion()  # Turn on interactive mode
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Track best tour found
    best_tour = current
    best_value = current_value
    
    time_step = 0  # Total loop iterations for temperature schedule
    while True:
        temperature = schedule(time_step)
        if temperature == 0:
            print(f"\nTemperature reached 0. Search complete.")
            break
        
        # Generate random neighbor (use 2-opt reversal for better TSP performance)
        if random.random() < 0.8:
            neighbor = reverse_neighbor(current)
        else:
            neighbor = swap_neighbor(current)
        
        neighbor_value = value_fn(neighbor, cities)
        evaluations += 1
        
        # Decision rule: accept if better, or probabilistically if worse
        delta_e = neighbor_value - current_value
        
        # For minimization: accept if delta_e < 0 (better) or probabilistically if delta_e >= 0
        if delta_e < 0 or probability(math.exp(-delta_e / temperature)):
            current = neighbor
            current_value = neighbor_value
            iterations += 1
            
            # Track best solution
            if current_value < best_value:
                best_tour = current
                best_value = current_value
            
            # Update visualization on EVERY accepted move
            current_intersections = count_intersections(current, cities)
            current_distance = tour_distance(current, cities)
            best_intersections = count_intersections(best_tour, cities)
            best_distance = tour_distance(best_tour, cities)
            
            draw_tour_comparison(axes[0], current, cities, "Current Tour", 
                               current_intersections, current_distance)
            draw_tour_comparison(axes[1], best_tour, cities, "Best Tour Found", 
                               best_intersections, best_distance)
            
            fig.suptitle(f"TSP Simulated Annealing - Move {iterations} | Temp: {temperature:.2f} | Evaluations: {evaluations}", 
                        fontsize=14, weight='bold')
            
            plt.pause(0.0001)  # Minimal pause to allow plot to render
            
            # Terminate early if perfect solution found (0 intersections)
            if neighbor_value == 0:
                print(f"\n✓ Perfect solution found! No edge intersections.")
                break
        
        time_step += 1
        
        # Safety limit to prevent infinite loops
        if time_step > LIMIT * 2:
            print(f"\nReached safety limit ({LIMIT * 2} steps).")
            break
    
    # Final visualization update
    plt.ioff()  # Turn off interactive mode
    
    final_intersections = count_intersections(best_tour, cities)
    final_distance = tour_distance(best_tour, cities)
    
    draw_tour_comparison(axes[0], initial, cities, "Initial Tour", 
                       count_intersections(initial, cities), tour_distance(initial, cities))
    draw_tour_comparison(axes[1], best_tour, cities, "Best Tour Found", 
                       final_intersections, final_distance)
    
    fig.suptitle(f"TSP Simulated Annealing - Final Results | {iterations} accepted moves", 
                fontsize=14, weight='bold')
    
    return initial, best_tour, iterations, evaluations


def main():
    """Solve TSP using simulated annealing with real-time visualization."""
    # Generate random cities
    cities = random_cities(NUM_CITIES)
    
    print(f"\n{'='*60}")
    print(f"TSP Simulated Annealing with Real-Time Visualization")
    print(f"{'='*60}")
    print(f"Number of cities: {NUM_CITIES}")
    print(f"Temperature Schedule: exp(-{LAM}*t) * {K} (limit: {LIMIT})")
    print(f"Update interval: {UPDATE_INTERVAL_MS}ms")
    print(f"\nStarting search... (watch the visualization update in real-time)")
    
    schedule = exp_schedule()
    initial, final, iterations, evaluations = simulated_annealing_tsp_realtime(
        NUM_CITIES, cities, value, schedule, UPDATE_INTERVAL_MS
    )
    
    initial_intersections = count_intersections(initial, cities)
    final_intersections = count_intersections(final, cities)
    initial_distance = tour_distance(initial, cities)
    final_distance = tour_distance(final, cities)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Accepted state transitions: {iterations}")
    print(f"Total state evaluations: {evaluations}")
    print(f"\nInitial tour:")
    print(f"  → {initial_intersections} edge intersections")
    print(f"  → Total distance: {initial_distance:.2f}")
    print(f"\nBest tour found:")
    print(f"  → {final_intersections} edge intersections (goal: 0)")
    print(f"  → Total distance: {final_distance:.2f}")
    print(f"  → Improvement: {initial_distance - final_distance:.2f} ({((initial_distance - final_distance)/initial_distance * 100):.1f}%)")
    
    if final_intersections == 0:
        print(f"\n🎉 SUCCESS! Found a tour with no edge intersections!")
    else:
        print(f"\n⚠ {final_intersections} intersections remaining.")
        print(f"Try adjusting K, LAM, or LIMIT for better results.")
    print(f"{'='*60}")
    
    plt.show()


if __name__ == "__main__":
    main()
