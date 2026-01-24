import heapq
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


def a_star_search(graph: nx.Graph, pos: dict, start: str, goal: str):
    """
    A* Search using Euclidean distance heuristic.
    f(n) = g(n) + h(n), where g(n) is path cost and h(n) is heuristic estimate.
    Returns the visited order and parent map.
    """
    frontier = []  # priority queue of (f_score, tie_breaker, Node)
    counter = 0

    start_node = Node(start, path_cost=0)
    start_h = euclidean_distance(start, goal, pos)
    start_f = 0 + start_h
    heapq.heappush(frontier, (start_f, counter, start_node))

    g_score = {start: 0}  # cheapest known cost from start to each state
    parents = {start: None}
    visited = []
    explored = set()

    while frontier:
        f, _, node = heapq.heappop(frontier)

        if node.state in explored:
            continue

        explored.add(node.state)
        visited.append(node.state)

        if node.state == goal:
            return visited, parents

        for child in node.expand(graph):
            if child.state in explored:
                continue

            new_g = g_score[node.state] + child.path_cost - node.path_cost
            
            if child.state not in g_score or new_g < g_score[child.state]:
                g_score[child.state] = new_g
                child_h = euclidean_distance(child.state, goal, pos)
                child_f = new_g + child_h
                parents[child.state] = node.state
                counter += 1
                heapq.heappush(frontier, (child_f, counter, child))

    return visited, parents


def main():
    graph, pos = load_indiana_graph()
    visited_order, parents = a_star_search(graph, pos, START, GOAL)
    path = reconstruct_path(parents, GOAL)

    print(f"A* start: {START}")
    print(f"A* goal : {GOAL}")
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
        algorithm_name="A* Search",
        show_edge_weights=True,
        node_labels=node_labels,
    )


if __name__ == "__main__":
    main()
