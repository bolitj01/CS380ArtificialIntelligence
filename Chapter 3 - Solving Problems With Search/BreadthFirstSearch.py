import networkx as nx
from collections import deque

from graph_utils import draw_search_result, load_indiana_graph, reconstruct_path, Node

START = "Kokomo"
GOAL = "Coldwater"


def breadth_first_search(graph: nx.Graph, start: str, goal: str):
    """
    Search the shallowest nodes in the search tree first.
    Returns visited order and parent map for path reconstruction.
    """
    node = Node(start)
    if node.state == goal:
        return [start], {start: None}
    
    frontier = deque([node]) # States to be explored (queue)
    explored = set() # States already explored
    visited = [] # All the visited states in order
    parents = {start: None} # Used for path reconstruction

    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        visited.append(node.state)
        print(f"Exploring: {node.state} (frontier size: {len(frontier)})")
        
        for child in node.expand(graph):
            if child.state not in explored and child not in frontier:
                parents[child.state] = node.state
                if child.state == goal:
                    visited.append(child.state)
                    print(f"✓ Found goal: {child.state}")
                    return visited, parents
                frontier.append(child)
                print(f"  Adding to frontier: {child.state}")
    
    return visited, parents


def main():
    graph, pos = load_indiana_graph()
    visited_order, parents = breadth_first_search(graph, START, GOAL)
    path = reconstruct_path(parents, GOAL)

    print(f"BFS start: {START}")
    print(f"BFS goal : {GOAL}")
    print(f"Visited {len(visited_order)} states: {visited_order}")
    print(f"Path length: {len(path)}")
    print(f"Path: {path if path else 'No path found'}")

    draw_search_result(
        graph,
        pos,
        visited_order,
        path,
        START,
        GOAL,
        algorithm_name="Breadth-First Search",
        show_edge_weights=False,
    )


if __name__ == "__main__":
    main()
