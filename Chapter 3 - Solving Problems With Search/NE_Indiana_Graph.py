import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------
# Create the graph
# -----------------------------
G = nx.Graph()

# Road segments with costs
edges = [
    ("Fort Wayne","Huntington",25),
    ("Fort Wayne","Columbia City",20),
    ("Fort Wayne","Auburn",20),
    ("Fort Wayne","Decatur",22),
    ("Fort Wayne","Bluffton",35),

    ("Auburn","Angola",35),
    ("Angola","Coldwater",40),

    ("Huntington","Wabash",28),
    ("Wabash","Peru",32),
    ("Peru","Kokomo",38),

    ("Bluffton","Decatur",25),
    ("Decatur","Portland",28),
    ("Portland","Muncie",45),

    ("Columbia City","Warsaw",30),
    ("Warsaw","Plymouth",45),
    ("Plymouth","South Bend",30),

    # Additional cross-connections
    ("Auburn","Columbia City",18),
    ("Auburn","Huntington",30),
    ("Decatur","Huntington",35),
    ("Bluffton","Huntington",32),
    ("Decatur","Columbia City",34),
    ("Warsaw","Fort Wayne",42),
    ("Warsaw","Auburn",40),
    ("Wabash","Bluffton",45),
    ("Peru","Decatur",50),
    ("Portland","Bluffton",33),
    ("Warsaw","Decatur",48),
    ("Plymouth","Fort Wayne",65),
    ("South Bend","Warsaw",55),
]

for u, v, w in edges:
    G.add_edge(u, v, weight=w)

# -----------------------------
# Fixed layout (Romania-map-like)
# -----------------------------
pos = {
    "Fort Wayne": (0, 0),

    "Auburn": (-1, 1),
    "Angola": (-2, 2),
    "Coldwater": (-3, 3),

    "Huntington": (-1, -1),
    "Wabash": (-2, -2),
    "Peru": (-3, -3),
    "Kokomo": (-4, -4),

    "Bluffton": (1, -1),
    "Decatur": (1, 0),
    "Portland": (2, -1),
    "Muncie": (3, -2),

    "Columbia City": (1, 1),
    "Warsaw": (2, 2),
    "Plymouth": (3, 3),
    "South Bend": (4, 4),
}

# -----------------------------
# Draw the graph
# -----------------------------
plt.figure(figsize=(15, 13))

# Nodes
nx.draw_networkx_nodes(
    G, pos,
    node_size=3500,
    node_color="#9ad0f5",   # light blue
    edgecolors="black",
    linewidths=1.5
)

# Edges
nx.draw_networkx_edges(G, pos, width=1.5)

# Node labels
nx.draw_networkx_labels(
    G, pos,
    font_size=12,
    font_color="black",
    font_weight="bold"
)

# Edge labels (costs)
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_size=9,
    font_color="black"
)

# Title and cleanup
plt.title(
    "Northeastern Indiana Search Graph (High-Contrast Classroom Version)",
    fontsize=18,
    weight="bold"
)
plt.axis("off")
plt.show()
