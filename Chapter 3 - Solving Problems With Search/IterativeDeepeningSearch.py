import networkx as nx

from graph_utils import draw_search_result, load_indiana_graph, reconstruct_path, Node

START = "Kokomo"
GOAL = "Coldwater"


def depth_limited_search(graph: nx.Graph, start: str, goal: str, limit: int):
    """
    Depth-limited search using recursion (textbook style).
    Returns: a Node if goal found, 'cutoff' if limit reached, None if no solution in this limit.
    """
    visited_nodes = []
    
    def recursive_dls(node: Node, current_depth: int, path: tuple):
        visited_nodes.append(node.state)
        
        if node.state == goal:
            return node
        elif current_depth == 0:
            return 'cutoff'
        else:
            cutoff_occurred = False
            children = node.expand(graph)
            for child in children:
                # Avoid cycles: don't revisit states on current path
                if child.state in path:
                    continue
                result = recursive_dls(child, current_depth - 1, path + (child.state,))
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
            return 'cutoff' if cutoff_occurred else None

    node = Node(start)
    result = recursive_dls(node, limit, (start,))
    return result, visited_nodes


def iterative_deepening_search(graph: nx.Graph, start: str, goal: str, max_depth: int = 20):
    """
    Iterative Deepening Depth-First Search (IDDFS) using textbook approach.
    Progressively increases depth limit until goal is found.
    Returns the goal node and visited order.
    """
    for depth_limit in range(max_depth):
        print(f"\n=== Trying depth limit: {depth_limit} ===")
        result, visited = depth_limited_search(graph, start, goal, depth_limit)
        
        unique_visited = len(set(visited))
        print(f"  Explored {len(visited)} nodes (unique: {unique_visited}) at this depth limit")
        
        if result != 'cutoff' and result is not None:
            print(f"\n✓ Goal found at depth {depth_limit}!")
            return result, visited
    
    print(f"\nGoal not found within depth limit {max_depth}")
    return None, []


def main():
    graph, pos = load_indiana_graph()
    print("Starting Iterative Deepening Depth-First Search...")
    goal_node, visited_order = iterative_deepening_search(graph, START, GOAL, max_depth=20)
    
    if goal_node:
        path = reconstruct_path_from_node(goal_node, START)
        unique_visited = list(dict.fromkeys(visited_order))  # Preserve order, remove duplicates
        print(f"\nIDDFS start: {START}")
        print(f"IDDFS goal : {GOAL}")
        print(f"Final iteration explored {len(visited_order)} nodes (unique: {len(unique_visited)})")
        print(f"Path length: {len(path)}")
        print(f"Path: {path}")

        draw_search_result(
            graph,
            pos,
            unique_visited,
            path,
            START,
            GOAL,
            algorithm_name="Iterative Deepening Depth-First Search",
            show_edge_weights=False
        )
    else:
        print(f"\nNo path found from {START} to {GOAL}")


def reconstruct_path_from_node(node: Node, start: str) -> list:
    """Reconstruct path by following parent pointers from goal node back to start."""
    path = []
    current = node
    while current is not None:
        path.append(current.state)
        current = current.parent
    return list(reversed(path))


if __name__ == "__main__":
    main()
