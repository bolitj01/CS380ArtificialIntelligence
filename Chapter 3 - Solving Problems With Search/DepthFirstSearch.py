import networkx as nx

from graph_utils import draw_search_result, load_indiana_graph, reconstruct_path, Node

START = "Kokomo"
GOAL = "Coldwater"


def depth_first_graph_search(graph, start, goal):
    """
    Search the deepest nodes in the search tree first.
    Returns the visited order and parent map for path reconstruction.
    """
    frontier = [Node(start)]  # States to be explored
    explored = set() # States already explored
    visited = [] # All the visited states in order
    parents = {start: None} # Used for path reconstruction

    while frontier:
        node = frontier.pop()
        
        if node.state in explored:
            continue
        
        visited.append(node.state)
        explored.add(node.state)
        
        if node.state == goal:
            break
        
        # Extend frontier with children not in explored and not already in frontier
        frontier_states = {n.state for n in frontier}
        for child in node.expand(graph):
            if child.state not in explored and child.state not in frontier_states:
                parents[child.state] = node.state
                frontier.append(child)
    
    return visited, parents


def main():
    graph, pos = load_indiana_graph()
    visited_order, parents = depth_first_graph_search(graph, START, GOAL)
    path = reconstruct_path(parents, GOAL)

    print(f"DFS start: {START}")
    print(f"DFS goal : {GOAL}")
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
        algorithm_name="Depth-First Search",
    )


if __name__ == "__main__":
    main()
