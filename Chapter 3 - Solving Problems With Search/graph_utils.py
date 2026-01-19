import json
import matplotlib.pyplot as plt
import networkx as nx
import os
from matplotlib.lines import Line2D

# Default path to the Indiana locations JSON
# Use the directory of this file as the base
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INDIANA_JSON_PATH = os.path.join(_SCRIPT_DIR, "indiana_locations.json")


class Node:
    """A node in the search tree."""
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def expand(self, graph):
        """Return a list of child nodes from this node given the graph."""
        return [
            Node(state=child, parent=self, action=None, path_cost=self.path_cost + graph[self.state][child].get('weight', 1))
            for child in sorted(graph.neighbors(self.state))
        ]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)


def load_indiana_graph(json_path: str = INDIANA_JSON_PATH):
    """Load the Indiana graph and coordinate layout from JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)

    graph = nx.Graph()
    for u, v, w in data["edges"]:
        graph.add_edge(u, v, weight=w)

    positions = {city: tuple(coords) for city, coords in data["positions"].items()}
    return graph, positions


def reconstruct_path(parents: dict, goal: str):
    """Reconstruct a path from start to goal using a parent map."""
    if goal not in parents:
        return []
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parents.get(node)
    return list(reversed(path))


def manhattan_distance(city_a: str, city_b: str, positions: dict) -> float:
    """Heuristic: Manhattan distance between two cities using their coordinates."""
    if city_a not in positions or city_b not in positions:
        # Fallback to 0 when coordinates are missing to avoid breaking the search
        return 0.0
    x1, y1 = positions[city_a]
    x2, y2 = positions[city_b]
    return abs(x1 - x2) + abs(y1 - y2)


def euclidean_distance(city_a: str, city_b: str, positions: dict) -> float:
    """Heuristic: Euclidean distance between two cities using their coordinates."""
    if city_a not in positions or city_b not in positions:
        # Fallback to 0 when coordinates are missing to avoid breaking the search
        return 0.0
    x1, y1 = positions[city_a]
    x2, y2 = positions[city_b]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def draw_search_result(graph: nx.Graph, pos: dict, visited_order: list, path: list,
                       start: str, goal: str, algorithm_name: str = "Search", 
                       show_edge_weights: bool = True):
    """Render the graph, highlighting visited nodes/edges and the final path.
    
    Args:
        show_edge_weights: If True, display edge weight labels. Set to False for uniform-cost searches.
    """
    visited_set = set(visited_order)
    path_set = set(path)

    node_colors = []
    for node in graph.nodes():
        if node in path_set:
            node_colors.append("#ff9999")  # path nodes
        elif node in visited_set:
            node_colors.append("#c5e5f8")  # visited
        else:
            node_colors.append("#e0e0e0")  # unvisited

    path_edges = set()
    if len(path) > 1:
        path_edges = set(zip(path, path[1:])) | set(zip(path[1:], path))

    edge_colors = []
    edge_widths = []
    for u, v in graph.edges():
        if (u, v) in path_edges or (v, u) in path_edges:
            edge_colors.append("red")
            edge_widths.append(3.5)
        elif u in visited_set and v in visited_set:
            edge_colors.append("#4c72b0")
            edge_widths.append(2.0)
        else:
            edge_colors.append("#b0b0b0")
            edge_widths.append(1.0)

    fig, ax = plt.subplots(figsize=(14, 10))

    nx.draw_networkx_nodes(
        graph, pos,
        node_size=3200,
        node_color=node_colors,
        edgecolors="black",
        linewidths=1.5,
        ax=ax
    )

    nx.draw_networkx_edges(
        graph, pos,
        width=edge_widths,
        edge_color=edge_colors,
        connectionstyle="arc3,rad=0.08",
        ax=ax
    )

    labels = {node: node.replace(" ", "\n") for node in graph.nodes()}
    nx.draw_networkx_labels(
        graph, pos,
        labels=labels,
        font_size=11,
        font_color="black",
        font_weight="bold",
        ax=ax
    )

    if show_edge_weights:
        edge_labels = nx.get_edge_attributes(graph, "weight")
        nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels=edge_labels,
            font_size=14,
            font_color="black",
            ax=ax
        )

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Path node', markerfacecolor='#ff9999', markeredgecolor='black', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Visited', markerfacecolor='#c5e5f8', markeredgecolor='black', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Unvisited', markerfacecolor='#e0e0e0', markeredgecolor='black', markersize=12),
        Line2D([0], [0], color='red', linewidth=3.5, label='Final path'),
        Line2D([0], [0], color='#4c72b0', linewidth=2.0, label='Visited edges')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=20)

    ax.set_title(
        f"{algorithm_name} from {start} to {goal}\nVisited: {len(visited_order)} states",
        fontsize=16,
        weight="bold"
    )

    ax.axis("off")
    plt.tight_layout()
    plt.show()

    return fig, ax
