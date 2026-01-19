import json
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------
# Generated graph of locations in Northeastern Indiana
# for use in search problem examples.
# -----------------------------

# Load location data from JSON file
with open(".\\Chapter 3 - Solving Problems With Search\\indiana_locations.json", "r") as f:
    location_data = json.load(f)

G = nx.Graph()

# Road segments with costs - loaded from JSON
edges = [tuple(edge) for edge in location_data["edges"]]

for u, v, w in edges:
    G.add_edge(u, v, weight=w)

# -----------------------------
# Geographic layout based on actual Indiana map positions
# (Approximate relative coordinates)
# -----------------------------
# Convert position lists to tuples
pos = {city: tuple(coords) for city, coords in location_data["positions"].items()}

# -----------------------------
# Draw the graph
# -----------------------------
plt.figure(figsize=(14, 10))

# Nodes
nx.draw_networkx_nodes(
    G, pos,
    node_size=4000,
    node_color="#c5e5f8",   # lighter blue
    edgecolors="black",
    linewidths=1.5
)

# Edges
nx.draw_networkx_edges(G, pos, width=1.5, connectionstyle="arc3,rad=0.1")

# Node labels with wrapped text for two-word cities
labels = {node: node.replace(" ", "\n") for node in G.nodes()}
nx.draw_networkx_labels(
    G, pos,
    labels=labels,
    font_size=12,
    font_color="black",
    font_weight="bold"
)

# Edge labels (costs)
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_size=14,
    font_color="black"
)

# Title and cleanup
plt.title(
    "Northeastern Indiana Search Graph",
    fontsize=18,
    weight="bold"
)

plt.axis("off")
plt.show()

# Create a method to highlight frontier edges (between a node and its children)
def highlight_frontier(ax, graph, pos, node, color, depth_label):
    """Highlight edges between a node and its neighbors with colored lines"""
    neighbors = list(graph.neighbors(node))
    for neighbor in neighbors:
        x_values = [pos[node][0], pos[neighbor][0]]
        y_values = [pos[node][1], pos[neighbor][1]]
        ax.plot(x_values, y_values, color=color, linewidth=4, zorder=5)

# Draw the graph with highlighted frontier for Huntington
fig, ax = plt.subplots(figsize=(14, 10))

# Nodes
nx.draw_networkx_nodes(
    G, pos,
    node_size=4000,
    node_color="#c5e5f8",   # lighter blue
    edgecolors="black",
    linewidths=1.5,
    ax=ax
)

# Edges
nx.draw_networkx_edges(G, pos, width=1.5, connectionstyle="arc3,rad=0.1", ax=ax)

# Highlight first depth children (Huntington's neighbors) in green
huntington_neighbors = list(G.neighbors("Huntington"))
for neighbor in huntington_neighbors:
    x_values = [pos["Huntington"][0], pos[neighbor][0]]
    y_values = [pos["Huntington"][1], pos[neighbor][1]]
    ax.plot(x_values, y_values, color='green', linewidth=4, zorder=5)

# Highlight second depth children (neighbors of Huntington's neighbors) in blue
second_depth = set()
for neighbor in huntington_neighbors:
    for second_neighbor in G.neighbors(neighbor):
        if second_neighbor != "Huntington":  # Don't include Huntington itself
            second_depth.add(second_neighbor)
            x_values = [pos[neighbor][0], pos[second_neighbor][0]]
            y_values = [pos[neighbor][1], pos[second_neighbor][1]]
            ax.plot(x_values, y_values, color='blue', linewidth=4, zorder=5)

# Node labels with wrapped text for two-word cities
labels = {node: node.replace(" ", "\n") for node in G.nodes()}
nx.draw_networkx_labels(
    G, pos,
    labels=labels,
    font_size=12,
    font_color="black",
    font_weight="bold",
    ax=ax
)

# Edge labels (costs)
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_size=14,
    font_color="black",
    ax=ax
)

# Add legend for the highlighted depths
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='green', linewidth=4, label='Depth 1: Huntington\'s Children'),
    Line2D([0], [0], color='blue', linewidth=4, label='Depth 2: Grandchildren')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=18)

# Title and cleanup
ax.set_title(
    "Northeastern Indiana Search Graph (Huntington Frontier by Depth)",
    fontsize=18,
    weight="bold"
)

ax.axis("off")
plt.tight_layout()
plt.show()
