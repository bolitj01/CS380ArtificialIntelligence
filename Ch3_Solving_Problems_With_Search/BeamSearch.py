import networkx as nx

from graph_utils import (
    draw_search_result,
    load_indiana_graph,
    reconstruct_path,
    euclidean_distance,
    label_heuristic,
    Node,
)

START = "Kokomo"
GOAL = "Coldwater"
BEAM_WIDTH = 3 
# For "Kokomo" to "Coldwater", a beam width of 1 fails!
# If only the heuristic is considered withou the path cost, beam search also fails at beam width 2!

def beam_search(graph: nx.Graph, pos: dict, start: str, goal: str, beam_width: int = 2):
    """
    Beam Search using Euclidean distance heuristic.
    Keeps only the best k nodes at each level (where k = beam_width).
    Returns the visited order and parent map.
    """
    beam = [Node(start)]
    parents = {start: None}
    visited = [start]
    explored = set()

    while beam:
        # Check if goal is in current beam
        for node in beam:
            if node.state == goal:
                return visited, parents

        # Expand all nodes in current beam
        all_children = []
        for node in beam:
            if node.state in explored:
                continue
            explored.add(node.state)
            
            for child in node.expand(graph):
                if child.state not in explored and child.state not in parents:
                    parents[child.state] = node.state
                    child_h = euclidean_distance(child.state, goal, pos)
                    child_f = child.path_cost + child_h  # g(n) + h(n)
                    all_children.append((child_f, child))

        # If no children, search fails
        if not all_children:
            break

        # Sort children by heuristic and keep only top beam_width
        all_children.sort(key=lambda x: x[0])
        beam = [child for _, child in all_children[:beam_width]]
        
        # Add new beam nodes to visited
        for node in beam:
            if node.state not in visited:
                visited.append(node.state)

    return visited, parents


def main():
    graph, pos = load_indiana_graph()
    visited_order, parents = beam_search(graph, pos, START, GOAL, beam_width=BEAM_WIDTH)
    path = reconstruct_path(parents, GOAL)

    print(f"Beam Search start: {START}")
    print(f"Beam Search goal : {GOAL}")
    print(f"Beam width: {BEAM_WIDTH}")
    print(f"Visited {len(visited_order)} states: {visited_order}")
    print(f"Path length: {len(path)}")
    print(f"Path: {path if path else 'No path found'}")

    # Create node labels with heuristic values
    node_labels = label_heuristic(graph, pos, GOAL, euclidean_distance)

    draw_search_result(
        graph,
        pos,
        visited_order,
        path,
        START,
        GOAL,
        algorithm_name=f"Beam Search (width={BEAM_WIDTH})",
        show_edge_weights=False,
        node_labels=node_labels,
    )


if __name__ == "__main__":
    main()
