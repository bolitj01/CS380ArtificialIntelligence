import heapq
import networkx as nx

from graph_utils import draw_search_result, load_indiana_graph, reconstruct_path, Node

START = "Kokomo"
GOAL = "Coldwater"


def uniform_cost_search(graph: nx.Graph, start: str, goal: str):
    """
    Uniform Cost Search (Dijkstra-style) using path costs as priorities.
    Returns the visit order, parent map, and goal cost (if found).
    """
    frontier = []  # priority queue of (path_cost, tie_breaker, Node)
    counter = 0    # tie-breaker to keep heapq deterministic

    start_node = Node(start)
    heapq.heappush(frontier, (0, counter, start_node))

    best_cost = {start: 0}  # cheapest known cost to each state
    parents = {start: None}  # used for path reconstruction
    visited = []  # states as they are expanded (popped from frontier)

    while frontier:
        cost, _, node = heapq.heappop(frontier)

        # Skip if we have already found a cheaper path to this state
        if cost > best_cost[node.state]:
            continue

        visited.append(node.state)

        if node.state == goal:
            return visited, parents, cost

        for child in node.expand(graph):
            new_cost = child.path_cost
            if child.state not in best_cost or new_cost < best_cost[child.state]:
                best_cost[child.state] = new_cost
                parents[child.state] = node.state
                counter += 1
                heapq.heappush(frontier, (new_cost, counter, child))

    return visited, parents, None


def main():
    graph, pos = load_indiana_graph()
    visited_order, parents, goal_cost = uniform_cost_search(graph, START, GOAL)
    path = reconstruct_path(parents, GOAL)

    print(f"UCS start: {START}")
    print(f"UCS goal : {GOAL}")
    print(f"Visited {len(visited_order)} states: {visited_order}")
    if goal_cost is not None:
        print(f"Path cost: {goal_cost}")
    else:
        print("Path cost: not found")
    print(f"Path length: {len(path)}")
    print(f"Path: {path if path else 'No path found'}")

    draw_search_result(
        graph,
        pos,
        visited_order,
        path,
        START,
        GOAL,
        algorithm_name="Uniform-Cost Search",
        show_edge_weights=True,
    )


if __name__ == "__main__":
    main()
