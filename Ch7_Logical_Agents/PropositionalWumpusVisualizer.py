"""Interactive propositional Wumpus World (4x4) visualizer.

This script demonstrates a step-wise, knowledge-based agent that reasons only with propositional logic. 
The GUI shows the discovered map on the left and a
logic trace of derived facts and decisions on the right.

It makes inferences using a DPLL-based SAT solver to check entailment of safety and hazard conditions for unvisited tiles. The agent prioritizes moving to tiles that are provably safe based on its KB, and takes a risk to explore unknown tiles when no safe moves are provable.

DPLL is a truth-table checking algorithm for propositional satisfiability that incorporates unit clause propagation and pure literal elimination to efficiently prune the search space. This allows the agent to derive new knowledge from its percepts and make informed decisions in the Wumpus world.
"""

from __future__ import annotations

import copy
import random
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

GridPos = Tuple[int, int]
Literal = Tuple[str, bool]  # (symbol, is_positive)
Clause = frozenset[Literal]


class PropositionalKB:
    """CNF knowledge base with SAT-based entailment checks."""

    def __init__(self) -> None:
        self.clauses: List[Clause] = []

    def tell_clause(self, literals: Iterable[Literal]) -> None:
        clause = frozenset(literals)
        if clause and clause not in self.clauses:
            self.clauses.append(clause)

    def tell_unit(self, symbol: str, value: bool) -> None:
        self.tell_clause([(symbol, value)])

    def entails(self, symbol: str, value: bool) -> bool:
        # KB entails q iff KB and not(q) is unsatisfiable.
        clauses = self.clauses + [frozenset([(symbol, not value)])]
        return not is_satisfiable(clauses)


def is_satisfiable(clauses: Sequence[Clause]) -> bool:
    symbols = sorted({sym for c in clauses for sym, _ in c})
    return _dpll(list(clauses), {}, symbols)


def _dpll(clauses: List[Clause], assignment: Dict[str, bool], symbols: List[str]) -> bool:
    # Evaluate current clause status.
    all_true = True
    for clause in clauses:
        clause_val = _eval_clause(clause, assignment)
        if clause_val is False:
            return False
        if clause_val is None:
            all_true = False
    if all_true:
        return True

    # Unit clause propagation.
    unit = _find_unit_clause(clauses, assignment)
    if unit is not None:
        sym, val = unit
        new_assignment = assignment.copy()
        new_assignment[sym] = val
        rem = [s for s in symbols if s != sym]
        return _dpll(clauses, new_assignment, rem)

    # Pure literal elimination.
    pure = _find_pure_literal(clauses, assignment)
    if pure is not None:
        sym, val = pure
        new_assignment = assignment.copy()
        new_assignment[sym] = val
        rem = [s for s in symbols if s != sym]
        return _dpll(clauses, new_assignment, rem)

    # Branch on first unassigned symbol.
    for sym in symbols:
        if sym not in assignment:
            for val in (True, False):
                new_assignment = assignment.copy()
                new_assignment[sym] = val
                rem = [s for s in symbols if s != sym]
                if _dpll(clauses, new_assignment, rem):
                    return True
            return False
    return False


def _eval_clause(clause: Clause, assignment: Dict[str, bool]) -> Optional[bool]:
    has_unbound = False
    for sym, is_pos in clause:
        if sym in assignment:
            if assignment[sym] == is_pos:
                return True
        else:
            has_unbound = True
    if has_unbound:
        return None
    return False


def _find_unit_clause(clauses: Sequence[Clause], assignment: Dict[str, bool]) -> Optional[Tuple[str, bool]]:
    for clause in clauses:
        undecided: List[Literal] = []
        satisfied = False
        for sym, is_pos in clause:
            if sym in assignment:
                if assignment[sym] == is_pos:
                    satisfied = True
                    break
            else:
                undecided.append((sym, is_pos))
        if not satisfied and len(undecided) == 1:
            return undecided[0]
    return None


def _find_pure_literal(clauses: Sequence[Clause], assignment: Dict[str, bool]) -> Optional[Tuple[str, bool]]:
    signs: Dict[str, Set[bool]] = {}
    for clause in clauses:
        if _eval_clause(clause, assignment) is True:
            continue
        for sym, is_pos in clause:
            if sym in assignment:
                continue
            signs.setdefault(sym, set()).add(is_pos)
    for sym, sign_set in signs.items():
        if len(sign_set) == 1:
            return sym, next(iter(sign_set))
    return None


@dataclass
class World:
    size: int
    pits: Set[GridPos]
    wumpus: GridPos
    gold: GridPos
    start: GridPos


def neighbors_for(size: int, p: GridPos) -> List[GridPos]:
    x, y = p
    cands = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    return [q for q in cands if 1 <= q[0] <= size and 1 <= q[1] <= size]


def find_path(size: int, start: GridPos, goal: GridPos, blocked: Set[GridPos]) -> Optional[List[GridPos]]:
    frontier: List[GridPos] = [start]
    parent: Dict[GridPos, Optional[GridPos]] = {start: None}

    while frontier:
        current = frontier.pop(0)
        if current == goal:
            path: List[GridPos] = []
            node: Optional[GridPos] = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path

        for nxt in neighbors_for(size, current):
            if nxt in blocked or nxt in parent:
                continue
            parent[nxt] = current
            frontier.append(nxt)

    return None


def build_random_world(size: int = 4) -> World:
    corners = [(1, 1), (1, size), (size, 1), (size, size)]
    all_cells = [(x, y) for y in range(1, size + 1) for x in range(1, size + 1)]

    while True:
        start = random.choice(corners)
        gold = random.choice([cell for cell in all_cells if cell != start])

        # Reserve a safe route so the random board is always solvable.
        safe_path = set(find_path(size, start, gold, set()) or [])
        if not safe_path:
            continue

        blocked_for_hazards = set(safe_path)
        remaining = [cell for cell in all_cells if cell not in blocked_for_hazards]
        if not remaining:
            continue

        wumpus = random.choice(remaining)
        pit_candidates = [cell for cell in remaining if cell != wumpus]
        pits = {cell for cell in pit_candidates if random.random() < 0.25}

        if find_path(size, start, gold, pits | {wumpus}) is not None:
            return World(size=size, pits=pits, wumpus=wumpus, gold=gold, start=start)


def build_default_world() -> World:
    return World(
        size=4,
        pits={(3, 1), (3, 3), (4, 4)},
        wumpus=(4, 2),
        gold=(2, 4),
        start=(1, 1),
    )


class WumpusPropositionalAgent:
    """Small propositional-only agent for the 4x4 Wumpus world."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.kb = PropositionalKB()
        self.pos: GridPos = world.start
        self.visited: Set[GridPos] = {world.start}
        self.discovered: Set[GridPos] = {world.start}
        self.safe: Set[GridPos] = {world.start}
        self.dead = False
        self.found_gold = False
        self.step_count = 0

        self._add_static_axioms()
        # Start square is always safe in this map.
        self.kb.tell_unit(self._pit_symbol(world.start), False)
        self.kb.tell_unit(self._wumpus_symbol(world.start), False)

    def _in_bounds(self, p: GridPos) -> bool:
        x, y = p
        return 1 <= x <= self.world.size and 1 <= y <= self.world.size

    def _neighbors(self, p: GridPos) -> List[GridPos]:
        x, y = p
        cands = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [q for q in cands if self._in_bounds(q)]

    def _pit_symbol(self, p: GridPos) -> str:
        return f"P{p[0]}{p[1]}"

    def _wumpus_symbol(self, p: GridPos) -> str:
        return f"W{p[0]}{p[1]}"

    def _breeze_symbol(self, p: GridPos) -> str:
        return f"B{p[0]}{p[1]}"

    def _stench_symbol(self, p: GridPos) -> str:
        return f"S{p[0]}{p[1]}"

    def _add_static_axioms(self) -> None:
        # Breeze and stench biconditionals encoded in CNF.
        all_cells = [(x, y) for y in range(1, self.world.size + 1) for x in range(1, self.world.size + 1)]
        for cell in all_cells:
            neigh = self._neighbors(cell)
            b = self._breeze_symbol(cell)
            s = self._stench_symbol(cell)

            # Bxy -> (P neighbors)
            self.kb.tell_clause([(b, False)] + [(self._pit_symbol(n), True) for n in neigh])
            # Each Pn -> Bxy
            for n in neigh:
                self.kb.tell_clause([(self._pit_symbol(n), False), (b, True)])

            # Sxy -> (W neighbors)
            self.kb.tell_clause([(s, False)] + [(self._wumpus_symbol(n), True) for n in neigh])
            # Each Wn -> Sxy
            for n in neigh:
                self.kb.tell_clause([(self._wumpus_symbol(n), False), (s, True)])

        # Exactly one Wumpus: at least one and at most one.
        w_symbols = [self._wumpus_symbol(cell) for cell in all_cells]
        self.kb.tell_clause([(w, True) for w in w_symbols])
        for i in range(len(w_symbols)):
            for j in range(i + 1, len(w_symbols)):
                self.kb.tell_clause([(w_symbols[i], False), (w_symbols[j], False)])

    def current_percepts(self) -> Dict[str, bool]:
        breeze = any(n in self.world.pits for n in self._neighbors(self.pos))
        stench = self.world.wumpus in self._neighbors(self.pos)
        glitter = self.pos == self.world.gold
        return {"breeze": breeze, "stench": stench, "glitter": glitter}

    def _update_kb_with_percepts(self, log: List[str]) -> None:
        percepts = self.current_percepts()
        b_sym = self._breeze_symbol(self.pos)
        s_sym = self._stench_symbol(self.pos)

        self.kb.tell_unit(b_sym, percepts["breeze"])
        self.kb.tell_unit(s_sym, percepts["stench"])

        log.append(
            f"Percept at {self.pos}: breeze={percepts['breeze']}, stench={percepts['stench']}, glitter={percepts['glitter']}"
        )
        log.append(f"Tell KB: {b_sym} = {percepts['breeze']} and {s_sym} = {percepts['stench']}")

        if percepts["glitter"]:
            self.found_gold = True
            log.append("Gold glitter detected. Agent grabs treasure and stops.")

    def _infer_cells(self, log: List[str]) -> Tuple[List[GridPos], List[GridPos], List[GridPos]]:
        safe_candidates: List[GridPos] = []
        risky_known: List[GridPos] = []
        unknown: List[GridPos] = []

        for y in range(1, self.world.size + 1):
            for x in range(1, self.world.size + 1):
                cell = (x, y)
                if cell in self.visited:
                    continue

                no_pit = self.kb.entails(self._pit_symbol(cell), False)
                no_wumpus = self.kb.entails(self._wumpus_symbol(cell), False)
                has_pit = self.kb.entails(self._pit_symbol(cell), True)
                has_wumpus = self.kb.entails(self._wumpus_symbol(cell), True)

                if no_pit and no_wumpus:
                    safe_candidates.append(cell)
                    self.safe.add(cell)
                    log.append(f"Derived safe({cell}) because KB entails ~P{cell[0]}{cell[1]} and ~W{cell[0]}{cell[1]}")
                elif has_pit or has_wumpus:
                    risky_known.append(cell)
                    hazard = "pit" if has_pit else "wumpus"
                    log.append(f"Derived hazard({cell}) because KB entails {hazard} is present")
                else:
                    unknown.append(cell)

        return safe_candidates, risky_known, unknown

    def _choose_next(self, safe_candidates: List[GridPos], unknown: List[GridPos], log: List[str]) -> Optional[GridPos]:
        if safe_candidates:
            safe_candidates.sort(key=lambda c: abs(c[0] - self.pos[0]) + abs(c[1] - self.pos[1]))
            choice = safe_candidates[0]
            log.append(f"Decision: move to nearest proved-safe unvisited tile {choice}")
            return choice

        if unknown:
            # Fallback exploration policy if no safe proof is available.
            unknown.sort(key=lambda c: abs(c[0] - self.pos[0]) + abs(c[1] - self.pos[1]))
            choice = unknown[0]
            log.append(f"Decision: no tile proved safe, exploring nearest unknown tile {choice}")
            return choice

        log.append("Decision: no available moves")
        return None

    def _apply_move(self, target: GridPos, log: List[str]) -> None:
        self.pos = target
        self.visited.add(target)
        self.discovered.add(target)
        log.append(f"Action: move to {target}")

        if target in self.world.pits:
            self.dead = True
            log.append("Outcome: fell into a pit. Simulation ends.")
            return
        if target == self.world.wumpus:
            self.dead = True
            log.append("Outcome: encountered the Wumpus. Simulation ends.")

    def step(self) -> List[str]:
        self.step_count += 1
        log: List[str] = [f"--- Step {self.step_count} ---"]

        if self.dead:
            log.append("Agent is dead. No further reasoning.")
            return log
        if self.found_gold:
            log.append("Gold already secured. No further reasoning.")
            return log

        self._update_kb_with_percepts(log)
        if self.found_gold:
            return log

        safe_candidates, _, unknown = self._infer_cells(log)
        choice = self._choose_next(safe_candidates, unknown, log)
        if choice is not None:
            self._apply_move(choice, log)
        else:
            log.append("No-op: agent remains in place.")

        return log


class WumpusVisualizer(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Propositional Wumpus World (4x4)")
        self.geometry("1325x700")
        self.minsize(980, 620)

        # Increase default UI font sizing for better readability.
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="Segoe UI", size=12)
        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(family="Segoe UI", size=12)
        menu_font = tkfont.nametofont("TkMenuFont")
        menu_font.configure(family="Segoe UI", size=12)
        heading_font = tkfont.nametofont("TkHeadingFont")
        heading_font.configure(family="Segoe UI", size=13)
        self.button_font = tkfont.Font(family="Segoe UI", size=16)
        self.style = ttk.Style(self)
        self.style.configure("Control.TButton", font=self.button_font)

        self.world = build_default_world()
        self.agent = WumpusPropositionalAgent(copy.deepcopy(self.world))

        self._build_ui()
        self._render_grid()
        self._append_logic(
            f"Simulation initialized. Start={self.world.start}, gold={self.world.gold}. Click 'Next Step' to run propositional inference."
        )

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")

        self.canvas = tk.Canvas(left, bg="#f6f6f2", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(left)
        controls.pack(fill=tk.X, pady=(8, 0))

        button_row = ttk.Frame(controls)
        button_row.pack(fill=tk.X)

        self.next_button = ttk.Button(button_row, text="Next Step", command=self.next_step)
        self.next_button.configure(style="Control.TButton")
        self.next_button.pack(side=tk.LEFT)

        self.static_button = ttk.Button(button_row, text="Static", command=self.static)
        self.static_button.configure(style="Control.TButton")
        self.static_button.pack(side=tk.LEFT, padx=(8, 0))

        self.random_button = ttk.Button(button_row, text="Randomize", command=self.randomize_world)
        self.random_button.configure(style="Control.TButton")
        self.random_button.pack(side=tk.LEFT, padx=(8, 0))

        self.status_var = tk.StringVar(value="Status: running")
        ttk.Label(controls, textvariable=self.status_var).pack(anchor="w", pady=(8, 10))

        ttk.Label(right, text="Logic Progression", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        self.logic_text = tk.Text(
            right,
            font=("Consolas", 12),
            wrap=tk.WORD,
            state=tk.DISABLED,
            padx=8,
            pady=8,
        )
        scroll = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self.logic_text.yview)
        self.logic_text.config(yscrollcommand=scroll.set)

        self.logic_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=(6, 0))
        scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=(6, 0))

        self.canvas.bind("<Configure>", lambda _e: self._render_grid())

    def _append_logic(self, message: str) -> None:
        self.logic_text.config(state=tk.NORMAL)
        self.logic_text.insert(tk.END, message + "\n")
        self.logic_text.see(tk.END)
        self.logic_text.config(state=tk.DISABLED)

    def _render_grid(self) -> None:
        self.canvas.delete("all")

        w = max(self.canvas.winfo_width(), 10)
        h = max(self.canvas.winfo_height(), 10)
        size = min(w, h) - 20
        cell = size / self.world.size
        x0 = (w - size) / 2
        y0 = (h - size) / 2

        for row in range(self.world.size):
            for col in range(self.world.size):
                x = col + 1
                y = self.world.size - row
                p = (x, y)

                left = x0 + col * cell
                top = y0 + row * cell
                right = left + cell
                bottom = top + cell

                if p in self.agent.discovered:
                    fill = "#efe9d5"
                else:
                    fill = "#c9d1d9"

                self.canvas.create_rectangle(left, top, right, bottom, fill=fill, outline="#222", width=2)

                label_lines: List[str] = [f"({x},{y})"]

                if p in self.agent.visited:
                    breeze = any(n in self.world.pits for n in self._neighbors(p))
                    stench = self.world.wumpus in self._neighbors(p)
                    if breeze:
                        label_lines.append("B")
                    if stench:
                        label_lines.append("S")
                    if p == self.world.gold and self.agent.found_gold:
                        label_lines.append("G")

                if p == self.agent.pos and not self.agent.dead:
                    label_lines.append("A")

                if p in self.agent.safe and p not in self.agent.visited:
                    label_lines.append("safe")

                text = "\n".join(label_lines)
                self.canvas.create_text(
                    (left + right) / 2,
                    (top + bottom) / 2,
                    text=text,
                    fill="#111",
                    font=("Segoe UI", 12),
                )

    def _neighbors(self, p: GridPos) -> List[GridPos]:
        x, y = p
        cands = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        return [q for q in cands if 1 <= q[0] <= self.world.size and 1 <= q[1] <= self.world.size]

    def next_step(self) -> None:
        for line in self.agent.step():
            self._append_logic(line)

        if self.agent.dead:
            self.status_var.set("Status: failed")
            self.next_button.config(state=tk.DISABLED)
        elif self.agent.found_gold:
            self.status_var.set("Status: treasure found")
            self.next_button.config(state=tk.DISABLED)
        else:
            self.status_var.set("Status: running")

        self._render_grid()

    def static(self) -> None:
        self.world = build_default_world()
        self.agent = WumpusPropositionalAgent(copy.deepcopy(self.world))
        self.logic_text.config(state=tk.NORMAL)
        self.logic_text.delete("1.0", tk.END)
        self.logic_text.config(state=tk.DISABLED)
        self._append_logic(
            f"Simulation reset to default board. Start={self.world.start}, gold={self.world.gold}. Click 'Next Step' to continue."
        )
        self.status_var.set("Status: running")
        self.next_button.config(state=tk.NORMAL)
        self._render_grid()

    def randomize_world(self) -> None:
        self.world = build_random_world()
        self.agent = WumpusPropositionalAgent(copy.deepcopy(self.world))
        self.logic_text.config(state=tk.NORMAL)
        self.logic_text.delete("1.0", tk.END)
        self.logic_text.config(state=tk.DISABLED)
        self._append_logic(
            f"Random board generated. Start={self.world.start}, gold={self.world.gold}. A safe path exists from start to goal."
        )
        self.status_var.set("Status: running")
        self.next_button.config(state=tk.NORMAL)
        self._render_grid()


if __name__ == "__main__":
    app = WumpusVisualizer()
    app.mainloop()
