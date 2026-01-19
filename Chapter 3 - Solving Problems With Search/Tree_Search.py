import json
import networkx as nx
import matplotlib.pyplot as plt

# Load location data from JSON file
with open(".\\Chapter 3 - Solving Problems With Search\\indiana_locations.json", "r") as f:
    location_data = json.load(f)

# Build the graph
G = nx.Graph()
edges = [tuple(edge) for edge in location_data["edges"]]
for u, v, w in edges:
    G.add_edge(u, v, weight=w)

# Starting node
start_node = "Huntington"

# Build a tree structure with 2 depth levels
def build_tree(graph, start, max_depth=2):
    """Build a tree from the graph starting at start node with specified depth"""
    tree = nx.DiGraph()
    visited = set()
    
    def explore(node, depth, parent_id=None):
        """Recursively explore nodes to build tree"""
        # Create unique node ID for tree (node_name + depth + parent to handle duplicates)
        node_id = f"{node}_d{depth}_p{parent_id}" if parent_id else node
        tree.add_node(node_id, label=node, depth=depth)
        
        if depth < max_depth:
            # Get neighbors from original graph
            neighbors = list(graph.neighbors(node))
            for neighbor in neighbors:
                neighbor_id = f"{neighbor}_d{depth+1}_p{node_id}"
                tree.add_node(neighbor_id, label=neighbor, depth=depth+1)
                
                # Get edge weight
                weight = graph[node][neighbor]['weight']
                tree.add_edge(node_id, neighbor_id, weight=weight)
                
                # Recursively explore neighbor
                explore(neighbor, depth + 1, node_id)
    
    explore(start, 0)
    return tree

# Build the tree starting from Fort Wayne
tree = build_tree(G, start_node, max_depth=2)

# Create hierarchical layout for visualization
def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """Create a hierarchical tree layout with staggered heights for alternating children"""
    pos = {root: (xcenter, vert_loc)}
    
    def _hierarchy_pos(G, node, left, right, vert_loc, pos, parent=None, depth=0):
        children = [n for n in G.neighbors(node) if n != parent]
        if len(children) != 0:
            dx = (right - left) / len(children)
            nextx = left + dx/2
            for i, child in enumerate(children):
                # Depth 1 (first generation): 2-tier stagger
                if depth == 0:
                    stagger = 0.02 if i % 2 == 0 else -0.02
                # Depth 2 (second generation): 3-tier stagger
                elif depth == 1:
                    tier = i % 3
                    if tier == 0:
                        stagger = 0.04
                    elif tier == 1:
                        stagger = 0.0
                    else:  # tier == 2
                        stagger = -0.04
                else:
                    # For other depths: 2-tier stagger
                    stagger = 0.02 if i % 2 == 0 else -0.02
                pos[child] = (nextx, vert_loc - vert_gap + stagger)
                pos = _hierarchy_pos(G, child, nextx - dx/2, nextx + dx/2,
                                    vert_loc - vert_gap, pos, node, depth + 1)
                nextx += dx
        return pos
    
    return _hierarchy_pos(G, root, xcenter - width/2, xcenter + width/2,
                         vert_loc, pos)

# Get hierarchical positions
pos = hierarchy_pos(tree, start_node, width=20, vert_gap=0.1)

# Draw the tree
plt.figure(figsize=(16, 10))

# Draw nodes
node_labels = nx.get_node_attributes(tree, 'label')
nx.draw_networkx_nodes(
    tree, pos,
    node_size=3000,
    node_color="#c5e5f8",
    edgecolors="black",
    linewidths=1.5
)

# Draw edges with arrows
nx.draw_networkx_edges(
    tree, pos,
    arrows=True,
    arrowsize=20,
    arrowstyle='->',
    width=1.5,
    edge_color='black'
)

# Draw node labels (city names)
nx.draw_networkx_labels(
    tree, pos,
    labels=node_labels,
    font_size=10,
    font_weight="bold"
)

plt.title(
    f"Tree Search from {start_node} (2 Depth Levels)",
    fontsize=16,
    weight="bold"
)
plt.axis("off")
plt.tight_layout()
plt.show()

# Print tree statistics
print(f"Tree Search starting from: {start_node}")
print(f"Maximum depth: 2")
print(f"Total nodes in tree: {tree.number_of_nodes()}")
print(f"Total edges in tree: {tree.number_of_edges()}")
print(f"\nLevel 0 (Root): {start_node}")
print(f"Level 1 (Children): {[node_labels[n] for n in tree.nodes() if tree.nodes[n]['depth'] == 1]}")
print(f"Level 2 (Grandchildren): {len([n for n in tree.nodes() if tree.nodes[n]['depth'] == 2])} nodes")
