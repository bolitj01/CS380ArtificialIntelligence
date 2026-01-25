import os
import random
from typing import Tuple, List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QUEEN_IMG_PATH = os.path.join(SCRIPT_DIR, "queen.png")
QUEEN_IMG = mpimg.imread(QUEEN_IMG_PATH)


def random_state(n: int) -> Tuple[int, ...]:
    """Generate a random assignment: one queen per column at random rows."""
    return tuple(random.randint(0, n - 1) for _ in range(n))


def conflicts(state: Tuple[int, ...]) -> int:
    """Count attacking pairs of queens in the given state."""
    count = 0
    n = len(state)
    for c1 in range(n):
        r1 = state[c1]
        for c2 in range(c1 + 1, n):
            r2 = state[c2]
            same_row = r1 == r2
            same_diag = abs(r1 - r2) == abs(c1 - c2)
            if same_row or same_diag:
                count += 1
    return count


def value(state: Tuple[int, ...]) -> int:
    """Return number of conflicts (lower is better).
    
    Goal is to minimize this value to 0 (no attacking pairs).
    """
    return conflicts(state)


def draw_board(ax, state: Tuple[int, ...], title: str):
    """Draw an N-Queens board with queen images."""
    n = len(state)
    # Draw squares
    for row in range(n):
        for col in range(n):
            color = "#f0d9b5" if (row + col) % 2 == 0 else "#b58863"
            ax.add_patch(plt.Rectangle((col, row), 1, 1, facecolor=color, edgecolor="black"))
    
    # Place queens - each column has one queen at the row specified in state
    for col in range(n):
        row = state[col]
        # Image coordinates: col to col+1, (n-row-1) to (n-row) for proper board orientation
        ax.imshow(QUEEN_IMG, extent=(col + 0.1, col + 0.9, n - row - 0.9, n - row - 0.1), 
                  aspect="auto", zorder=10)
    
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, linewidth=0.5, color='black', alpha=0.3)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_aspect("equal")

def draw_fitness_landscape(ax, trajectory: List[Tuple[int, ...]], value_fn):
    """Visualize the fitness landscape showing the hill climbing trajectory.
    
    For minimization: seeks valleys (lower values) rather than peaks.
    
    Args:
        ax: Matplotlib axis to draw on
        trajectory: List of states visited during search
        value_fn: Function to compute objective value of a state (conflicts)
    """
    if not trajectory:
        ax.text(0.5, 0.5, 'No trajectory data', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Compute objective values for each state in trajectory
    values_list = [value_fn(state) for state in trajectory]
    
    # Plot the trajectory
    steps = range(len(trajectory))
    ax.plot(steps, values_list, 'b-o', linewidth=2, markersize=4, label='Search Trajectory')
    
    # Mark start and end
    ax.plot(0, values_list[0], 'ro', markersize=10, label=f'Start ({values_list[0]} conflicts)')
    ax.plot(len(trajectory)-1, values_list[-1], 'go', markersize=10, label=f'End ({values_list[-1]} conflicts)')
    
    # Draw goal line at 0 conflicts
    ax.axhline(0, color='gold', linestyle='--', linewidth=2, label='Goal (0 conflicts)')
    
    ax.set_xlabel('Step Number', fontsize=10, weight='bold')
    ax.set_ylabel('Number of Conflicts (Attacking Queen Pairs)', fontsize=10, weight='bold')
    ax.set_title('Fitness Landscape\n(Seeking Valleys - Lower is Better)', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    
    # Set y-axis limits with some padding
    max_conflicts = max(values_list) if values_list else 1
    ax.set_ylim(bottom=-0.5, top=max_conflicts + 1)