import heapq
import networkx as nx

from graph_utils import (
    draw_search_result,
    load_indiana_graph,
    reconstruct_path,
    euclidean_distance, # alternative heuristic
    manhattan_distance, # alternative heuristic
    label_heuristic,
    Node,
)

START = "Kokomo"
GOAL = "Coldwater"


def greedy_best_first_search(graph: nx.Graph, pos: dict, start: str, goal: str):
    """
    Greedy Best-First Search using Manhattan distance to the goal as the heuristic.
    Returns the visited order and parent map.
    """
    frontier = []  # priority queue of (heuristic, tie_breaker, Node)
    counter = 0

    start_node = Node(start)
    # start_h = euclidean_distance(start, goal, pos) # Alternative heuristic
    start_h = manhattan_distance(start, goal, pos) # Alternative heuristic
    heapq.heappush(frontier, (start_h, counter, start_node))

    parents = {start: None}
    visited = []
    explored = set()

    while frontier:
        h, _, node = heapq.heappop(frontier)

        if node.state in explored:
            continue

        explored.add(node.state)
        visited.append(node.state)

        if node.state == goal:
            return visited, parents

        for child in node.expand(graph):
            if child.state in explored:
                continue
            counter += 1
            child_h = manhattan_distance(child.state, goal, pos) # Alternative heuristic
            # child_h = euclidean_distance(child.state, goal, pos) # Alternative heuristic
            print(f" h(n) between {child.state} and {goal}: {child_h}")
            parents[child.state] = node.state
            heapq.heappush(frontier, (child_h, counter, child))

    return visited, parents


def main():
    graph, pos = load_indiana_graph()
    visited_order, parents = greedy_best_first_search(graph, pos, START, GOAL)
    path = reconstruct_path(parents, GOAL)

    print(f"Greedy start: {START}")
    print(f"Greedy goal : {GOAL}")
    print(f"Visited {len(visited_order)} states: {visited_order}")
    print(f"Path length: {len(path)}")
    print(f"Path: {path if path else 'No path found'}")

    # Create node labels with heuristic values (using manhattan_distance)
    node_labels = label_heuristic(graph, pos, GOAL, manhattan_distance)

    draw_search_result(
        graph,
        pos,
        visited_order,
        path,
        START,
        GOAL,
        algorithm_name="Greedy Best-First Search",
        show_edge_weights=False,
        node_labels=node_labels,
    )


if __name__ == "__main__":
    main()
