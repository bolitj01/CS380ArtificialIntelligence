"""Newton's method for optimization on a simple trigonometric function.

Newton's method for optimization finds critical points (local minima/maxima)
where the derivative equals zero. Starting from a random initial guess x0, it iteratively computes:
    x_{n+1} = x_n - f'(x_n) / f''(x_n)

This function is f(x) = sin(x), which has simple derivatives and multiple peaks.
"""
from typing import Tuple, List
import random

import matplotlib.pyplot as plt
import numpy as np


def f(x):
    """Simple trigonometric function: f(x) = sin(x)"""
    return np.sin(x)


def f_prime(x):
    """First derivative: f'(x) = cos(x)"""
    return np.cos(x)


def f_double_prime(x):
    """Second derivative: f''(x) = -sin(x)"""
    return -np.sin(x)


def newton_raphson(x0: float, tolerance: float = 1e-6, max_iterations: int = 100) -> Tuple[float, List[float], int]:
    """
    Find a critical point (maximum) of f(x) using Newton's optimization method.
    
    Args:
        x0: Initial guess
        tolerance: Convergence tolerance (stop when |f'(x)| < tolerance)
        max_iterations: Maximum number of iterations
        
    Returns:
        Tuple of (critical_point, iterations_path, num_iterations)
    """
    iterations_path = [x0]
    x_current = x0
    
    for i in range(max_iterations):
        f_prime_val = f_prime(x_current)
        f_double_prime_val = f_double_prime(x_current)
        
        # Avoid division by zero
        if abs(f_double_prime_val) < 1e-10:
            print(f"Warning: Second derivative near zero at iteration {i+1}")
            break
        
        # Newton optimization update: x = x - f'(x) / f''(x)
        x_next = x_current - f_prime_val / f_double_prime_val
        iterations_path.append(x_next)
        
        # Check convergence (derivative close to zero)
        if abs(f_prime(x_next)) < tolerance:
            f_double = f_double_prime(x_next)
            is_maximum = f_double < 0
            peak_type = "MAXIMUM" if is_maximum else "MINIMUM"
            print(f"Converged after {i+1} iterations")
            print(f"Critical point: x = {x_next:.10f}")
            print(f"Function value: f(x) = {f(x_next):.10f}")
            print(f"Type: {peak_type} (f''(x) = {f_double:.6f})")
            print(f"Derivative: f'(x) = {f_prime(x_next):.2e}")
            return x_next, iterations_path, i + 1
        
        x_current = x_next
    
    f_double = f_double_prime(x_current)
    is_maximum = f_double < 0
    peak_type = "MAXIMUM" if is_maximum else "MINIMUM"
    print(f"Max iterations reached ({max_iterations})")
    print(f"Critical point: x = {x_current:.10f}")
    print(f"Function value: f(x) = {f(x_current):.10f}")
    print(f"Type: {peak_type} (f''(x) = {f_double:.6f})")
    print(f"Derivative: f'(x) = {f_prime(x_current):.2e}")
    return x_current, iterations_path, max_iterations


def main():
    """Run Newton's optimization method and visualize results."""
    print("\n" + "="*60)
    print("Newton's Method for Finding Critical Points")
    print("Function: f(x) = sin(x)")
    print("="*60)
    
    # Random initial guess within reasonable viewing window
    x0 = random.uniform(-np.pi, np.pi)
    print(f"\nStarting from x0 = {x0:.6f}")
    
    # Run Newton's optimization method
    critical_point, path, iterations = newton_raphson(x0)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot the function
    x_range = np.linspace(-2*np.pi, 2*np.pi, 500)
    y_range = f(x_range)
    
    ax.plot(x_range, y_range, 'b-', linewidth=2.5, label='f(x) = sin(x)')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
    ax.grid(True, alpha=0.3)
    
    # Plot iteration steps
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, max(len(path)-1, 1)))
    for i in range(len(path) - 1):
        x_curr = path[i]
        x_next = path[i + 1]
        y_curr = f(x_curr)
        
        # Draw vertical line from curve to x-axis
        ax.plot([x_curr, x_curr], [y_curr, 0], 'r--', alpha=0.4, linewidth=1)
        
        # Draw connection point on curve
        label = 'Iteration steps' if i == 0 else ""
        ax.plot(x_curr, y_curr, 'o', color=colors[i], markersize=6, alpha=0.7, label=label)
    
    # Mark start point
    ax.plot(x0, f(x0), 'go', markersize=12, label=f'Start: x₀ = {x0:.4f}', zorder=5)
    
    # Mark final critical point
    f_double = f_double_prime(critical_point)
    is_max = f_double < 0
    marker = '*' if is_max else 'v'
    marker_label = 'Maximum' if is_max else 'Minimum'
    ax.plot(critical_point, f(critical_point), marker, color='r', markersize=20, 
            label=f'{marker_label}: x = {critical_point:.4f}, f(x) = {f(critical_point):.4f}', zorder=5)
    
    # Mark peak locations for reference
    peak_x = np.linspace(-2*np.pi, 2*np.pi, 20)
    peak_y = f(peak_x)
    ax.plot(peak_x[::4], peak_y[::4], 'ko', markersize=5, alpha=0.2)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title(f"Newton's Method: {iterations} iterations to convergence", fontsize=14, weight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Initial guess:        x₀ = {x0:.10f}")
    print(f"Critical point found: x  = {critical_point:.10f}")
    print(f"Function value:       f(x) = {f(critical_point):.10f}")
    print(f"Iterations taken:     {iterations}")
    print(f"Path: {' → '.join([f'{x:.4f}' for x in path[:min(6, len(path))]])}{'...' if len(path) > 6 else ''}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
