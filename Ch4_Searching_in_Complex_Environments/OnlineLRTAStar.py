"""
CS380 AI - LRTA* (Learning Real-Time A*) Search
Based on textbook Figure 4.24

The agent learns from experience by updating heuristic estimates as it explores.
Discovers terrain incrementally while improving path quality over time.
"""
import tkinter as tk
from tkinter import messagebox
from typing import List, Tuple, Optional, Set, Dict

# ============================================================================
# TERRAIN COSTS AND COLORS
# ============================================================================

TERRAIN_COSTS = {
    'P': 1,  # Paved
    'M': 2,  # Mud
    'F': 4,  # Flood
    'Q': 6   # Quicksand
}

TERRAIN_COLORS = {
    'P': '#808080',  # Gray - Paved
    'M': '#8B4513',  # Brown - Mud
    'F': '#4169E1',  # Blue - Flood
    'Q': '#DEB887'   # Tan - Quicksand
}

# ============================================================================
# GRID CLASS
# ============================================================================

class Grid:
    """Represents a 2D grid with terrain costs."""
    
    def __init__(self, width: int, height: int):
        """Initialize grid with dimensions."""
        self.width = width
        self.height = height
        self.terrain = [['P' for _ in range(width)] for _ in range(height)]
    
    def set_terrain(self, row: int, col: int, terrain_type: str):
        """Set terrain type at position."""
        if 0 <= row < self.height and 0 <= col < self.width:
            self.terrain[row][col] = terrain_type
    
    def get_terrain(self, row: int, col: int) -> str:
        """Get terrain type at position."""
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.terrain[row][col]
        return None
    
    def get_cost(self, row: int, col: int) -> int:
        """Get movement cost for terrain at position."""
        terrain = self.get_terrain(row, col)
        return TERRAIN_COSTS.get(terrain, float('inf'))
    
    def get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid neighboring cells (up, down, left, right)."""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.height and 0 <= new_col < self.width:
                neighbors.append((new_row, new_col))
        
        return neighbors


def load_map(filename: str) -> Grid:
    """Load map from file."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse dimensions
    width, height = map(int, lines[0].strip().split())
    grid = Grid(width, height)
    
    # Parse terrain
    for row in range(height):
        terrain_line = lines[row + 1].strip().split()
        for col in range(width):
            if col < len(terrain_line):
                grid.set_terrain(row, col, terrain_line[col])
    
    return grid


# ============================================================================
# ONLINE SEARCH PROBLEM (Based on textbook Figure 4.23)
# ============================================================================

class OnlineSearchProblem:
    """
    A problem which is solved by an agent executing actions,
    rather than by just computation. Adapted for grid-based pathfinding.
    """
    
    def __init__(self, initial: Tuple[int, int], goal: Tuple[int, int], grid: Grid):
        self.initial = initial
        self.goal = goal
        self.grid = grid
    
    def actions(self, state: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Returns available actions (neighbor cells) from current state."""
        return self.grid.get_neighbors(*state)
    
    def result(self, state: Tuple[int, int], action: Tuple[int, int]) -> Tuple[int, int]:
        """Returns the state that results from executing action in state."""
        # In our grid, the action IS the next state (neighbor cell)
        return action
    
    def c(self, s: Tuple[int, int], a: Tuple[int, int], s1: Tuple[int, int]) -> int:
        """Returns cost estimate to move from state s to state s1."""
        return self.grid.get_cost(*s1)
    
    def goal_test(self, state: Tuple[int, int]) -> bool:
        """Returns True if state is a goal state."""
        return state == self.goal
    
    def h(self, state: Tuple[int, int]) -> float:
        """Returns heuristic estimate from state to goal (Manhattan distance * min cost)."""
        manhattan = abs(state[0] - self.goal[0]) + abs(state[1] - self.goal[1])
        return manhattan * 1  # Multiply by minimum terrain cost (Paved = 1)


# ============================================================================
# LRTA* AGENT (Based on textbook Figure 4.24)
# ============================================================================

class LRTAStarAgent:
    """
    LRTA* (Learning Real-Time A*) agent that learns heuristic values
    from experience while exploring an unknown environment.
    """
    
    def __init__(self, problem: OnlineSearchProblem):
        self.problem = problem
        self.H = {}  # Learned heuristic values
        self.s = None  # Previous state
        self.a = None  # Previous action
        # Start at user-selected initial state
        initial_state = problem.initial
        self.path = [initial_state]  # Path taken
        self.visited = {initial_state}  # All visited states
        self.total_cost = 0
        self.discovered = {initial_state}  # Cells whose terrain we know
        
    def __call__(self, s1: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Execute one step of LRTA*.
        
        Args:
            s1: Current state (position) after last action
            
        Returns:
            Next action to take, or None if goal reached
        """
        # Goal test
        if self.problem.goal_test(s1):
            self.a = None
            self.path.append(s1) if s1 not in self.path else None
            return self.a
        
        # Initialize heuristic for new state
        if s1 not in self.H:
            self.H[s1] = self.problem.h(s1)
        
        # Update heuristic of previous state based on experience
        if self.s is not None:
            # H[s] = min over all actions b: cost(s, b) + H[result(s, b)]
            discovered_neighbors = [n for n in self.problem.actions(self.s) if n in self.discovered]
            if discovered_neighbors:
                self.H[self.s] = min(
                    self.LRTA_cost(self.s, b, self.problem.result(self.s, b))
                    for b in discovered_neighbors
                )
        
        # Discover neighbors of current state
        neighbors = self.problem.actions(s1)
        for neighbor in neighbors:
            if neighbor not in self.discovered:
                self.discovered.add(neighbor)
                # Initialize heuristic for discovered neighbor
                if neighbor not in self.H:
                    self.H[neighbor] = self.problem.h(neighbor)
        
        # Choose action that minimizes LRTA cost among discovered neighbors
        discovered_neighbors = [n for n in neighbors if n in self.discovered]
        if discovered_neighbors:
            self.a = min(
                discovered_neighbors,
                key=lambda b: self.LRTA_cost(s1, b, self.problem.result(s1, b))
            )
        else:
            # No discovered neighbors (shouldn't happen)
            self.a = None
            return self.a
        
        # Update state and tracking info
        self.s = s1
        next_state = self.problem.result(s1, self.a)
        self.path.append(next_state)
        self.visited.add(next_state)
        self.total_cost += self.problem.c(s1, self.a, next_state)
        
        return self.a
    
    def LRTA_cost(self, s: Tuple[int, int], a: Tuple[int, int], s1: Tuple[int, int]) -> float:
        """
        Returns cost to move from state s to state s1 plus
        estimated cost to get to goal from s1.
        """
        if s1 is None:
            return self.problem.h(s)
        else:
            # Use learned H value if available, otherwise use initial heuristic
            h_value = self.H.get(s1, self.problem.h(s1))
            return self.problem.c(s, a, s1) + h_value


# ============================================================================
# LRTA* GUI
# ============================================================================

class LRTAStarGUI:
    """GUI for visualizing LRTA* exploration."""
    
    def __init__(self, grid: Grid, map_name: str = "large_map.txt"):
        self.grid = grid
        self.current_map = map_name
        self.cell_size = 50
        self.start = None
        self.goal = None
        self.agent = None
        self.running = False
        self.step_delay = 100  # milliseconds
        
        # Create window
        self.root = tk.Tk()
        self.root.title("LRTA* - Learning Online Search")
        
        # Create canvas
        self.canvas = tk.Canvas(
            self.root,
            width=grid.width * self.cell_size,
            height=grid.height * self.cell_size,
            bg='white'
        )
        self.canvas.pack()
        
        # Only one map: large_map.txt
        
        # Create control buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack()
        
        tk.Button(button_frame, text="Start", command=self.start_search, font=('Arial', 11)).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Reset", command=self.reset, font=('Arial', 11)).pack(side=tk.LEFT, padx=5)
        
        # Speed control
        speed_frame = tk.Frame(self.root)
        speed_frame.pack()
        
        tk.Label(speed_frame, text="Speed (ms):").pack(side=tk.LEFT, padx=5)
        self.speed_var = tk.IntVar(value=100)
        speed_slider = tk.Scale(speed_frame, from_=10, to=500, orient=tk.HORIZONTAL,
                               variable=self.speed_var, command=self.update_speed)
        speed_slider.pack(side=tk.LEFT, padx=5)
        
        # Info label
        self.info_label = tk.Label(self.root, text="Click to set START, click again to set GOAL", 
                                   font=('Arial', 12))
        self.info_label.pack()
        
        # Statistics label
        self.stats_label = tk.Label(self.root, text="", font=('Arial', 11), justify=tk.LEFT)
        self.stats_label.pack()
        
        # Bind click event
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Draw initial grid
        self.draw_grid()
    
    def draw_grid(self):
        """Draw the grid with hidden/discovered cells and learned heuristics."""
        self.canvas.delete("all")
        
        discovered = set()
        if self.agent:
            discovered = self.agent.discovered
        
        for row in range(self.grid.height):
            for col in range(self.grid.width):
                x1 = col * self.cell_size
                y1 = row * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                
                # Show discovered cells, start, and goal
                if (row, col) in discovered or (row, col) == self.start or (row, col) == self.goal:
                    terrain = self.grid.get_terrain(row, col)
                    color = TERRAIN_COLORS.get(terrain, 'white')
                    
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='black')
                    
                    # Draw terrain type and cost
                    cost = self.grid.get_cost(row, col)
                    self.canvas.create_text(x1 + self.cell_size // 2, y1 + 15,
                                          text=f"{terrain}({cost})", font=('Arial', 8))
                    
                    # Draw learned heuristic value if available
                    if self.agent and (row, col) in self.agent.H:
                        h_val = self.agent.H[(row, col)]
                        self.canvas.create_text(x1 + self.cell_size // 2, y1 + 35,
                                              text=f"h={h_val:.1f}", font=('Arial', 7), fill='blue')
                else:
                    # Hidden cell
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill='#2F2F2F', outline='black')
                    self.canvas.create_text(x1 + self.cell_size // 2, y1 + self.cell_size // 2,
                                          text="?", font=('Arial', 14), fill='white')
        
        # Draw path
        if self.agent and len(self.agent.path) > 1:
            for i in range(len(self.agent.path) - 1):
                r1, c1 = self.agent.path[i]
                r2, c2 = self.agent.path[i + 1]
                x1 = c1 * self.cell_size + self.cell_size // 2
                y1 = r1 * self.cell_size + self.cell_size // 2
                x2 = c2 * self.cell_size + self.cell_size // 2
                y2 = r2 * self.cell_size + self.cell_size // 2
                self.canvas.create_line(x1, y1, x2, y2, fill='yellow', width=3)
        
        # Draw start marker
        if self.start:
            row, col = self.start
            x = col * self.cell_size + self.cell_size // 2
            y = row * self.cell_size + self.cell_size // 2
            self.canvas.create_oval(x - 15, y - 15, x + 15, y + 15, 
                                   fill='green', outline='black', width=2)
            self.canvas.create_text(x, y, text='S', font=('Arial', 14, 'bold'), fill='white')
        
        # Draw goal marker
        if self.goal:
            row, col = self.goal
            x = col * self.cell_size + self.cell_size // 2
            y = row * self.cell_size + self.cell_size // 2
            self.canvas.create_oval(x - 15, y - 15, x + 15, y + 15, 
                                   fill='red', outline='black', width=2)
            self.canvas.create_text(x, y, text='G', font=('Arial', 14, 'bold'), fill='white')
        
        # Draw current agent position
        if self.agent and self.agent.s:
            row, col = self.agent.s
            x = col * self.cell_size + self.cell_size // 2
            y = row * self.cell_size + self.cell_size // 2
            self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, 
                                   fill='orange', outline='black', width=2)
            self.canvas.create_text(x, y, text='A', font=('Arial', 12, 'bold'), fill='white')
    
    def update_stats(self):
        """Update statistics display."""
        if self.agent:
            stats = (f"Cells Discovered: {len(self.agent.discovered)}\n"
                    f"Cells Visited: {len(self.agent.visited)}\n"
                    f"Path Length: {len(self.agent.path) - 1}\n"
                    f"Total Cost: {self.agent.total_cost}\n"
                    f"Learned H-values: {len(self.agent.H)}")
            self.stats_label.config(text=stats)
    
    def on_click(self, event):
        """Handle mouse click to set start and goal."""
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        
        if 0 <= row < self.grid.height and 0 <= col < self.grid.width:
            if self.start is None:
                self.start = (row, col)
                self.info_label.config(text="START set. Click to set GOAL")
                self.draw_grid()
            elif self.goal is None:
                self.goal = (row, col)
                self.info_label.config(text="GOAL set. Click Start to begin search")
                self.draw_grid()
    
    def start_search(self):
        """Start or resume automatic search."""
        if not self.start or not self.goal:
            messagebox.showwarning("Warning", "Please set both START and GOAL")
            return
        
        if not self.agent:
            problem = OnlineSearchProblem(self.start, self.goal, self.grid)
            self.agent = LRTAStarAgent(problem)
        
        self.running = True
        self.run_step()
    
    def run_step(self):
        """Execute one step and schedule next if running."""
        if not self.running or not self.agent:
            return
        
        # Get current state (last in path or start)
        current_state = self.agent.path[-1] if self.agent.path else self.start
        action = self.agent(current_state)
        
        self.draw_grid()
        self.update_stats()
        
        if action is None:
            self.running = False
            self.info_label.config(text="Goal reached!")
            return
        
        # Schedule next step
        self.root.after(self.step_delay, self.run_step)
    
    def step_search(self):
        """Execute single search step."""
        if not self.start or not self.goal:
            messagebox.showwarning("Warning", "Please set both START and GOAL")
            return
        
        if not self.agent:
            problem = OnlineSearchProblem(self.start, self.goal, self.grid)
            self.agent = LRTAStarAgent(problem)
        
        self.running = False  # Stop automatic execution
        
        # Get current state
        current_state = self.agent.path[-1] if self.agent.path else self.start
        action = self.agent(current_state)
        
        self.draw_grid()
        self.update_stats()
        
        if action is None:
            self.info_label.config(text="Goal reached!")
    
    def reset(self):
        """Reset the search."""
        self.start = None
        self.goal = None
        self.agent = None
        self.running = False
        self.info_label.config(text="Click to set START, click again to set GOAL")
        self.stats_label.config(text="")
        self.draw_grid()
    
    def update_speed(self, value):
        """Update animation speed."""
        self.step_delay = int(value)
    
    def run(self):
        """Start the GUI event loop."""
        self.root.mainloop()


# ============================================================================
# MAIN PROGRAM
# ============================================================================

if __name__ == "__main__":
    map_file = "Ch4_Searching_in_Complex_Environments/large_map.txt"
    
    try:
        grid = load_map(map_file)
        gui = LRTAStarGUI(grid, map_file)
        print(f"Loaded map: {map_file} ({grid.width}x{grid.height})")
        print("LRTA* - Agent learns heuristic values from experience")
        print("Blue numbers show learned h-values")
        print("Click to set START and GOAL, then click Start to begin")
        gui.run()
        
    except FileNotFoundError:
        print(f"Error: Could not find map file '{map_file}'")
    except Exception as e:
        print(f"Error: {e}")
