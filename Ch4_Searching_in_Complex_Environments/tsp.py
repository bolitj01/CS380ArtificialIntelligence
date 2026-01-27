import random
from typing import Tuple, List


def random_cities(num_cities: int, width: float = 10.0, height: float = 10.0) -> List[Tuple[float, float]]:
    """Generate random city coordinates in a 2D plane.
    
    Args:
        num_cities: Number of cities to generate
        width: Width of the coordinate space
        height: Height of the coordinate space
        
    Returns:
        List of (x, y) coordinate tuples
    """
    cities = []
    for _ in range(num_cities):
        x = random.uniform(0, width)
        y = random.uniform(0, height)
        cities.append((x, y))
    return cities


def random_tour(num_cities: int) -> Tuple[int, ...]:
    """Generate a random tour (permutation of city indices).
    
    Args:
        num_cities: Number of cities in the tour
        
    Returns:
        Tuple of city indices representing the tour order
    """
    tour = list(range(num_cities))
    random.shuffle(tour)
    return tuple(tour)


def edges_intersect(p1: Tuple[float, float], p2: Tuple[float, float], 
                   p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
    """Check if two line segments (p1-p2 and p3-p4) intersect.
    
    Uses the cross product method to determine intersection.
    
    Args:
        p1, p2: Endpoints of first line segment
        p3, p4: Endpoints of second line segment
        
    Returns:
        True if segments intersect, False otherwise
    """
    def ccw(a, b, c):
        """Check if three points are in counter-clockwise order."""
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    
    # Two segments intersect if the endpoints of each segment are on opposite sides of the other segment
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


def count_intersections(tour: Tuple[int, ...], cities: List[Tuple[float, float]]) -> int:
    """Count the number of edge intersections in a tour.
    
    Args:
        tour: Tour represented as a permutation of city indices
        cities: List of city coordinates
        
    Returns:
        Number of intersecting edge pairs
    """
    num_cities = len(tour)
    intersections = 0
    
    # Check all pairs of edges
    for i in range(num_cities):
        for j in range(i + 2, num_cities):
            # Get endpoints of edge i
            city1_idx = tour[i]
            city2_idx = tour[(i + 1) % num_cities]
            
            # Get endpoints of edge j
            city3_idx = tour[j]
            city4_idx = tour[(j + 1) % num_cities]
            
            # Skip if edges share a vertex
            if city1_idx == city3_idx or city1_idx == city4_idx or \
               city2_idx == city3_idx or city2_idx == city4_idx:
                continue
            
            # Check intersection
            if edges_intersect(cities[city1_idx], cities[city2_idx],
                             cities[city3_idx], cities[city4_idx]):
                intersections += 1
    
    return intersections


def tour_distance(tour: Tuple[int, ...], cities: List[Tuple[float, float]]) -> float:
    """Calculate the total distance of a tour.
    
    Args:
        tour: Tour represented as a permutation of city indices
        cities: List of city coordinates
        
    Returns:
        Total distance traveled in the tour
    """
    total_distance = 0.0
    num_cities = len(tour)
    
    for i in range(num_cities):
        city1 = cities[tour[i]]
        city2 = cities[tour[(i + 1) % num_cities]]
        
        # Euclidean distance
        distance = ((city2[0] - city1[0])**2 + (city2[1] - city1[1])**2)**0.5
        total_distance += distance
    
    return total_distance


def value(tour: Tuple[int, ...], cities: List[Tuple[float, float]]) -> int:
    """Evaluate tour quality based on number of intersections (lower is better).
    
    Args:
        tour: Tour represented as a permutation of city indices
        cities: List of city coordinates
        
    Returns:
        Number of edge intersections (0 = no intersections, optimal)
    """
    return count_intersections(tour, cities)


def draw_tour(ax, tour: Tuple[int, ...], cities: List[Tuple[float, float]], title: str):
    """Draw the TSP tour on a matplotlib axis.
    
    Args:
        ax: Matplotlib axis to draw on
        tour: Tour represented as a permutation of city indices
        cities: List of city coordinates
        title: Title for the plot
    """
    ax.clear()
    
    num_cities = len(tour)
    
    # Draw edges
    for i in range(num_cities):
        city1 = cities[tour[i]]
        city2 = cities[tour[(i + 1) % num_cities]]
        ax.plot([city1[0], city2[0]], [city1[1], city2[1]], 'b-', linewidth=1.5, alpha=0.6)
    
    # Draw cities as points
    for idx, city in enumerate(cities):
        ax.plot(city[0], city[1], 'ro', markersize=8)
        ax.text(city[0], city[1], f' {idx}', fontsize=8, verticalalignment='bottom')
    
    # Highlight start city
    start_city = cities[tour[0]]
    ax.plot(start_city[0], start_city[1], 'go', markersize=12, label='Start')
    
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()


def draw_tour_comparison(ax, tour: Tuple[int, ...], cities: List[Tuple[float, float]], 
                        title: str, intersections: int, distance: float):
    """Draw the TSP tour with statistics.
    
    Args:
        ax: Matplotlib axis to draw on
        tour: Tour represented as a permutation of city indices
        cities: List of city coordinates
        title: Base title for the plot
        intersections: Number of intersections in the tour
        distance: Total distance of the tour
    """
    draw_tour(ax, tour, cities, f"{title}\n{intersections} intersections, dist={distance:.2f}")
