"""GridWorld example using Value Iteration.

Based on AIMA's MDP/GridMDP setup, adapted to be self-contained for this repo.
"""


def vector_add(a, b):
    """Add two 2D vectors."""
    return a[0] + b[0], a[1] + b[1]


def turn_right(action):
    """Rotate action vector 90 degrees clockwise."""
    x, y = action
    return y, -x


def turn_left(action):
    """Rotate action vector 90 degrees counter-clockwise."""
    x, y = action
    return -y, x


orientations = [(1, 0), (0, 1), (-1, 0), (0, -1)]


class MDP:
    """A Markov Decision Process with transition and reward models."""

    def __init__(self, init, actlist, terminals, transitions=None, reward=None, states=None, gamma=0.9):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        self.states = states or self.get_states_from_transitions(transitions)
        self.init = init
        self.actlist = actlist
        self.terminals = terminals
        self.transitions = transitions or {}
        self.gamma = gamma
        self.reward = reward or {s: 0 for s in self.states}

    def R(self, state):
        return self.reward[state]

    def T(self, state, action):
        if not self.transitions:
            raise ValueError("Transition model is missing")
        return self.transitions[state][action]

    def actions(self, state):
        if state in self.terminals:
            return [None]
        return self.actlist

    @staticmethod
    def get_states_from_transitions(transitions):
        if isinstance(transitions, dict):
            s1 = set(transitions.keys())
            s2 = set(
                tr[1]
                for actions in transitions.values()
                for effects in actions.values()
                for tr in effects
            )
            return s1.union(s2)
        return None


class GridMDP(MDP):
    """A 2D grid MDP where None cells are walls/obstacles."""

    def __init__(self, grid, terminals, init=(0, 0), gamma=0.9):
        grid = list(reversed(grid))  # row 0 is bottom, matching AIMA coordinate convention
        reward = {}
        states = set()
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid

        for x in range(self.cols):
            for y in range(self.rows):
                if grid[y][x] is not None:
                    states.add((x, y))
                    reward[(x, y)] = grid[y][x]

        self.states = states

        transitions = {}
        for s in states:
            transitions[s] = {}
            for a in orientations:
                transitions[s][a] = self.calculate_T(s, a)

        super().__init__(
            init=init,
            actlist=orientations,
            terminals=terminals,
            transitions=transitions,
            reward=reward,
            states=states,
            gamma=gamma,
        )

    def calculate_T(self, state, action):
        if action is None:
            return [(0.0, state)]
        return [
            (0.8, self.go(state, action)),
            (0.1, self.go(state, turn_right(action))),
            (0.1, self.go(state, turn_left(action))),
        ]

    def T(self, state, action):
        if action is None:
            return [(0.0, state)]
        return self.transitions[state][action]

    def go(self, state, direction):
        next_state = vector_add(state, direction)
        return next_state if next_state in self.states else state

    def to_grid(self, mapping):
        return list(
            reversed(
                [
                    [mapping.get((x, y), None) for x in range(self.cols)]
                    for y in range(self.rows)
                ]
            )
        )

    def to_arrows(self, policy):
        chars = {(1, 0): ">", (0, 1): "^", (-1, 0): "<", (0, -1): "v", None: "."}
        return self.to_grid({s: chars[a] for s, a in policy.items()})


def q_value(mdp, s, a, U):
    if a is None:
        return mdp.R(s)
    return sum(p * (mdp.R(s) + mdp.gamma * U[s_prime]) for p, s_prime in mdp.T(s, a))


def value_iteration(mdp, epsilon=0.001, show_iterations=False):
    """Solve an MDP by value iteration.

    If show_iterations is True, print utilities after each
    value-update iteration.
    """
    U1 = {s: 0.0 for s in mdp.states}
    gamma = mdp.gamma
    iteration = 0

    if show_iterations:
        print("Initial utilities (iteration 0):")
        print_grid(mdp.to_grid({s: round(v, 3) for s, v in U1.items()}))
        print()

    while True:
        iteration += 1
        U = U1.copy()
        delta = 0.0
        for s in mdp.states:
            U1[s] = max(q_value(mdp, s, a, U) for a in mdp.actions(s))
            delta = max(delta, abs(U1[s] - U[s]))

        if show_iterations:
            print(f"Utilities after value iteration {iteration}:")
            print_grid(mdp.to_grid({s: round(v, 3) for s, v in U1.items()}))
            print()

        if delta <= epsilon * (1 - gamma) / gamma:
            return U


def best_policy(mdp, U):
    """Extract the best policy from utilities U."""
    return {s: max(mdp.actions(s), key=lambda a: q_value(mdp, s, a, U)) for s in mdp.states}


def print_grid(grid):
    for row in grid:
        print("\t".join(str(cell) for cell in row))


def run_example():
    sequential_decision_environment = GridMDP(
        [
            [-0.04, -0.04, -0.04, +1],
            [-0.04, None, -0.04, -1],
            [-0.04, -0.04, -0.04, -0.04],
        ],
        terminals=[(3, 2), (3, 1)],
        gamma=0.9,
    )

    utilities = value_iteration(
        sequential_decision_environment,
        epsilon=0.001,
        show_iterations=True,
    )
    policy = best_policy(sequential_decision_environment, utilities)

    print("Utilities from Value Iteration:")
    utility_grid = sequential_decision_environment.to_grid(
        {s: round(v, 3) for s, v in utilities.items()}
    )
    print_grid(utility_grid)

    print("\nDerived Policy Arrows:")
    arrow_grid = sequential_decision_environment.to_arrows(policy)
    print_grid(arrow_grid)


if __name__ == "__main__":
    run_example()
