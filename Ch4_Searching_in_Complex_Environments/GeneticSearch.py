"""Genetic algorithm example for the Traveling Salesman Problem.

This adapts the textbook-style genetic_algorithm to work with TSP tours (permutations
of city indices). Tours are scored by a fitness function that rewards short, 
non-intersecting paths. Crossover uses ordered crossover (OX) to keep tours valid,
while mutation performs a simple swap of two cities.
"""
import random
from typing import List, Tuple

import matplotlib.pyplot as plt

from tsp import (
    random_cities,
    random_tour,
    draw_tour_comparison,
    tour_distance,
    count_intersections,
)

# Hyperparameters
NUM_CITIES = 25
POPULATION_SIZE = 100 # number of individuals in each generation
GENERATIONS = 200 # repetitions of selection, crossover, mutation
MUTATION_RATE = 0.2 # probability of mutation per child
ELITE_COUNT = 2  # number of top individuals carried forward each generation
STOP_ON_GOAL = False  # whether to stop when 0 intersections found (True) or run all generations (False)

# Fitness weights
INTERSECTION_PENALTY = 5.0  # how much each edge crossing hurts fitness

Tour = Tuple[int, ...]


def fitness(tour: Tour, cities: List[Tuple[float, float]]) -> float:
    """Higher is better. Reward shorter distance and penalize intersections."""
    dist = tour_distance(tour, cities)
    crossings = count_intersections(tour, cities)
    penalty = 1.0 + INTERSECTION_PENALTY * crossings
    return 1.0 / (dist * penalty)


def ordered_crossover(parent1: Tour, parent2: Tour) -> Tour:
    """Ordered crossover (OX) preserves permutations for TSP tours."""
    a, b = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[a:b] = parent1[a:b]

    fill = [city for city in parent2 if city not in child]
    fill_iter = iter(fill)
    for i in range(len(child)):
        if child[i] is None:
            child[i] = next(fill_iter)

    return tuple(child)


def mutate(tour: Tour, pmut: float) -> Tour:
    """Swap two cities with probability pmut."""
    if random.random() > pmut:
        return tour

    a, b = random.sample(range(len(tour)), 2)
    tour_list = list(tour)
    tour_list[a], tour_list[b] = tour_list[b], tour_list[a]
    return tuple(tour_list)


def select(scored_population: List[Tuple[Tour, float]]) -> Tour:
    """Roulette-wheel selection based on fitness weights."""
    total_fitness = sum(score for _, score in scored_population)
    pick = random.uniform(0, total_fitness)
    current = 0.0
    for tour, score in scored_population:
        current += score
        if current >= pick:
            return tour
    return scored_population[-1][0]


def genetic_algorithm_tsp(cities: List[Tuple[float, float]],
                          pop_size: int = POPULATION_SIZE,
                          ngen: int = GENERATIONS,
                          pmut: float = MUTATION_RATE,
                          elite: int = ELITE_COUNT) -> Tuple[Tour, Tour]:
    """Run the genetic algorithm and return (initial_best, final_best)."""
    population: List[Tour] = [random_tour(len(cities)) for _ in range(pop_size)]

    best_initial = max(population, key=lambda t: fitness(t, cities))
    best_global = best_initial

    for gen in range(ngen):
        scored = [(tour, fitness(tour, cities)) for tour in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Track best found so far
        current_best = scored[0][0]
        if fitness(current_best, cities) > fitness(best_global, cities):
            best_global = current_best

        # Print progress every 10 generations
        if (gen + 1) % 10 == 0:
            best_fitness = scored[0][1]
            avg_fitness = sum(score for _, score in scored) / len(scored)
            best_dist = tour_distance(best_global, cities)
            best_inter = count_intersections(best_global, cities)
            print(f"Gen {gen + 1:3d}: Best fitness={best_fitness:.6f}, Avg fitness={avg_fitness:.6f} | Best tour: dist={best_dist:.2f}, intersections={best_inter}")
        
        # Early stopping: if we found a tour with no intersections, we're done
        if STOP_ON_GOAL and count_intersections(best_global, cities) == 0:
            print(f"\n✓ Optimal solution found at generation {gen + 1}! No edge intersections.")
            break

        # Elitism: carry best tours forward unchanged
        next_population: List[Tour] = [tour for tour, _ in scored[:elite]]

        # Create children until population restored
        while len(next_population) < pop_size:
            parent1 = select(scored)
            parent2 = select(scored)
            child = ordered_crossover(parent1, parent2)
            child = mutate(child, pmut)
            next_population.append(child)

        population = next_population

    best_final = best_global
    return best_initial, best_final


def main():
    cities = random_cities(NUM_CITIES)
    print("\n=== Genetic Algorithm TSP ===")
    print(f"Cities: {NUM_CITIES}, Population: {POPULATION_SIZE}, Generations: {GENERATIONS}, Mutation: {MUTATION_RATE}")

    initial, best = genetic_algorithm_tsp(cities)

    initial_intersections = count_intersections(initial, cities)
    best_intersections = count_intersections(best, cities)
    initial_distance = tour_distance(initial, cities)
    best_distance = tour_distance(best, cities)

    print("\nInitial tour:")
    print(f"  Intersections: {initial_intersections}")
    print(f"  Distance: {initial_distance:.2f}")

    print("\nBest tour:")
    print(f"  Intersections: {best_intersections}")
    print(f"  Distance: {best_distance:.2f}")
    improvement = (initial_distance - best_distance) / initial_distance * 100
    print(f"  Improvement: {improvement:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    draw_tour_comparison(
        axes[0], initial, cities, "Initial Tour",
        initial_intersections, initial_distance,
    )
    draw_tour_comparison(
        axes[1], best, cities, "Genetic Algorithm Best",
        best_intersections, best_distance,
    )
    fig.suptitle("TSP Genetic Algorithm", fontsize=14, weight="bold")
    plt.show()


if __name__ == "__main__":
    main()
