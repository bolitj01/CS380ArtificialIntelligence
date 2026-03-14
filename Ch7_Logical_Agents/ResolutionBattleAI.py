"""Turn-based battle demo with propositional resolution inference.

The human controls one party of three heroes. The AI controls the opposing party,
and uses propositional-resolution entailment queries to decide when to prioritize
special attacks.

Character class special features used in this demo:
- Wizard: special attack hits all living enemies (area-of-effect damage).
- Warrior: builds rage with regular attacks; special attack consumes rage for a strong hit.
- Ranger: deals bonus damage to enemies at or below 50% HP.
- Templar: special attack damages one target and heals all living allies.
"""

from __future__ import annotations

import random
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from tkinter import ttk
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Literal = str
Clause = frozenset[Literal]


# ---------------------------------------------------------------------------
# Propositional-resolution helpers


def negate(literal: Literal) -> Literal:
    return literal[1:] if literal.startswith("~") else f"~{literal}"


def has_complementary_pair(clause: Clause) -> bool:
    return any(negate(lit) in clause for lit in clause)


def pl_resolve(ci: Clause, cj: Clause) -> Set[Clause]:
    resolvents: Set[Clause] = set()
    # Try each literal in ci against its negation in cj.
    # If lit and ~lit both appear, they can cancel and produce a resolvent.
    for lit in ci:
        comp = negate(lit)
        if comp in cj:
            # Resolution rule: (lit v A) and (~lit v B) derive (A v B).
            merged = (set(ci) - {lit}) | (set(cj) - {comp})
            resolvent = frozenset(merged)
            # Skip tautological clauses like (X v ~X), which add no useful constraint.
            if not has_complementary_pair(resolvent):
                resolvents.add(resolvent)
    return resolvents


def pl_resolution_entails(kb_clauses: Sequence[Clause], query: Literal) -> bool:
    # Refutation method:
    # KB |= query iff KB U {~query} is unsatisfiable.
    # In resolution, unsatisfiable means we can derive the empty clause {}.
    clauses: List[Clause] = list(kb_clauses) + [frozenset({negate(query)})]
    known: Set[Clause] = set(clauses)

    while True:
        new: Set[Clause] = set()
        n = len(clauses)
        # Resolve every pair of known clauses and collect new consequences.
        for i in range(n):
            for j in range(i + 1, n):
                resolvents = pl_resolve(clauses[i], clauses[j])
                # Empty clause means contradiction was found, so query is entailed.
                if frozenset() in resolvents:
                    return True
                for r in resolvents:
                    if r not in known:
                        new.add(r)

        # No new clauses means saturation without contradiction; entailment failed.
        if not new:
            return False

        clauses.extend(new)
        known.update(new)


# ---------------------------------------------------------------------------
# Battle model


@dataclass
class CharacterState:
    role: str
    team_name: str
    max_hp: int
    hp: int
    attack_damage: int
    special_damage: int
    special_name: str
    mana: int = 0
    rage: int = 0
    cooldown: int = 0

    @property
    def uid(self) -> str:
        return f"{self.team_name}_{self.role}"

    @property
    def short_name(self) -> str:
        return f"{self.team_name} {self.role}"

    def alive(self) -> bool:
        return self.hp > 0

    def can_use_special(self) -> bool:
        if not self.alive():
            return False
        if self.role == "Wizard":
            return self.mana >= 3
        if self.role == "Warrior":
            return self.rage >= 2
        if self.role in {"Ranger", "Templar"}:
            return self.cooldown == 0
        return False

    def spend_special_cost(self) -> None:
        if self.role == "Wizard":
            self.mana -= 3
        elif self.role == "Warrior":
            self.rage -= 2
        elif self.role == "Ranger":
            self.cooldown = 2
        elif self.role == "Templar":
            self.cooldown = 3

    def end_turn_updates(self) -> None:
        if self.cooldown > 0:
            self.cooldown -= 1

    def gain_for_regular_attack(self) -> None:
        if self.role == "Wizard":
            self.mana = min(6, self.mana + 1)
        elif self.role == "Warrior":
            self.rage = min(4, self.rage + 1)


def make_character(role: str, team_name: str) -> CharacterState:
    if role == "Wizard":
        return CharacterState(role, team_name, max_hp=75, hp=75, attack_damage=13, special_damage=24, special_name="Arcane Burst", mana=2)
    if role == "Warrior":
        return CharacterState(role, team_name, max_hp=120, hp=120, attack_damage=11, special_damage=22, special_name="Crushing Blow")
    if role == "Ranger":
        return CharacterState(role, team_name, max_hp=88, hp=88, attack_damage=12, special_damage=19, special_name="Piercing Volley")
    if role == "Templar":
        return CharacterState(role, team_name, max_hp=132, hp=132, attack_damage=9, special_damage=15, special_name="Judgment Smash")
    raise ValueError(f"Unknown role: {role}")


class BattleState:
    ROLES = ["Wizard", "Warrior", "Ranger", "Templar"]

    def __init__(self) -> None:
        self.player_team: List[CharacterState] = []
        self.ai_team: List[CharacterState] = []
        self.round_number = 1
        self.new_battle()

    def new_battle(self) -> None:
        player_roles = random.sample(self.ROLES, 3)
        ai_roles = random.sample(self.ROLES, 3)
        self.player_team = [make_character(role, "Player") for role in player_roles]
        self.ai_team = [make_character(role, "AI") for role in ai_roles]
        self.round_number = 1

    def alive_player(self) -> List[CharacterState]:
        return [c for c in self.player_team if c.alive()]

    def alive_ai(self) -> List[CharacterState]:
        return [c for c in self.ai_team if c.alive()]

    def is_over(self) -> bool:
        return not self.alive_player() or not self.alive_ai()

    def winner(self) -> Optional[str]:
        if self.alive_player() and not self.alive_ai():
            return "Player"
        if self.alive_ai() and not self.alive_player():
            return "AI"
        return None


# ---------------------------------------------------------------------------
# GUI + controller


class ResolutionBattleApp(tk.Tk):
    RANGER_LOW_HP_MULTIPLIER = 1.5

    def __init__(self) -> None:
        super().__init__()
        self.title("Resolution Inference Battle AI")
        self.geometry("1518x760")
        self.minsize(1334, 660)

        self._increase_gui_fonts(4)
        self._configure_styles()

        self.state = BattleState()
        self.selected_target_uid: Optional[str] = None
        self.player_used_this_cycle: Set[str] = set()
        self.ai_used_this_cycle: Set[str] = set()

        self.status_var = tk.StringVar(value="Your turn")

        self._build_ui()
        self._refresh_everything()
        self._log_event("New battle started.")

    def _increase_gui_fonts(self, delta: int) -> None:
        for name in (
            "TkDefaultFont",
            "TkTextFont",
            "TkFixedFont",
            "TkMenuFont",
            "TkHeadingFont",
            "TkCaptionFont",
            "TkSmallCaptionFont",
            "TkTooltipFont",
        ):
            try:
                named = tkfont.nametofont(name)
            except tk.TclError:
                continue
            current_size = named.cget("size")
            named.configure(size=current_size + delta)

    def _configure_styles(self) -> None:
        default_font = tkfont.nametofont("TkDefaultFont")
        self.targeted_button_font = tkfont.Font(
            family=default_font.cget("family"),
            size=default_font.cget("size"),
            weight="bold",
        )
        style = ttk.Style(self)
        style.configure("Targeted.TButton", font=self.targeted_button_font)

    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=4)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.rowconfigure(3, weight=1)
        right.columnconfigure(0, weight=1)

        self.ai_status = ttk.LabelFrame(left, text="AI Team Status", padding=8)
        self.ai_status.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

        self.player_status = ttk.LabelFrame(left, text="Player Team Controls", padding=8)
        self.player_status.grid(row=1, column=0, sticky="nsew", pady=(0, 8))

        controls = ttk.LabelFrame(left, text="Player Controls", padding=8)
        controls.grid(row=2, column=0, sticky="ew")

        self.new_battle_btn = ttk.Button(controls, text="New Battle", command=self.new_battle)
        self.new_battle_btn.grid(row=0, column=0, pady=(2, 0), sticky="ew")

        ttk.Label(controls, textvariable=self.status_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        controls.columnconfigure(0, weight=0)
        controls.columnconfigure(1, weight=1)

        ttk.Label(right, text="Turn Events", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")
        self.event_text = tk.Text(right, wrap=tk.WORD, height=14, font=("Consolas", 15), state=tk.DISABLED)
        self.event_text.grid(row=1, column=0, sticky="nsew", pady=(6, 10))

        ttk.Label(right, text="AI Resolution Reasoning", font=("Segoe UI", 16, "bold")).grid(row=2, column=0, sticky="w")
        self.ai_text = tk.Text(right, wrap=tk.WORD, height=14, font=("Consolas", 15), state=tk.DISABLED)
        self.ai_text.grid(row=3, column=0, sticky="nsew", pady=(6, 0))

    # ------------------------- Rendering helpers -------------------------

    def _set_text(self, widget: tk.Text, message: str, append: bool = True) -> None:
        widget.config(state=tk.NORMAL)
        if append:
            widget.insert(tk.END, message + "\n")
            widget.see(tk.END)
        else:
            widget.delete("1.0", tk.END)
            widget.insert(tk.END, message)
        widget.config(state=tk.DISABLED)

    def _log_event(self, message: str) -> None:
        self._set_text(self.event_text, message)

    def _log_ai(self, message: str) -> None:
        self._set_text(self.ai_text, message)

    def _format_char_line(self, c: CharacterState) -> str:
        if not c.alive():
            return f"{c.role:8} HP: 0/{c.max_hp} [DEFEATED]"

        info = ""
        if c.role == "Wizard":
            info = f" mana={c.mana}"
        elif c.role == "Warrior":
            info = f" rage={c.rage}"
        elif c.role in {"Ranger", "Templar"}:
            info = f" cd={c.cooldown}"

        return f"{c.role:8} HP: {c.hp}/{c.max_hp}{info}"

    def _sync_player_cycle(self) -> None:
        alive_uids = {c.uid for c in self.state.alive_player()}
        self.player_used_this_cycle.intersection_update(alive_uids)

    def _eligible_player_uids(self) -> Set[str]:
        self._sync_player_cycle()
        alive_uids = {c.uid for c in self.state.alive_player()}
        if len(alive_uids) <= 1:
            return alive_uids

        unused = alive_uids - self.player_used_this_cycle
        if not unused:
            self.player_used_this_cycle.clear()
            unused = set(alive_uids)
        return unused

    def _sync_ai_cycle(self) -> None:
        alive_uids = {c.uid for c in self.state.alive_ai()}
        self.ai_used_this_cycle.intersection_update(alive_uids)

    def _eligible_ai_uids(self) -> Set[str]:
        self._sync_ai_cycle()
        alive_uids = {c.uid for c in self.state.alive_ai()}
        if len(alive_uids) <= 1:
            return alive_uids

        unused = alive_uids - self.ai_used_this_cycle
        if not unused:
            self.ai_used_this_cycle.clear()
            unused = set(alive_uids)
        return unused

    def _selected_target(self) -> Optional[CharacterState]:
        alive_targets = self.state.alive_ai()
        if not alive_targets:
            return None

        target = next((c for c in alive_targets if c.uid == self.selected_target_uid), None)
        if target is None:
            target = alive_targets[0]
            self.selected_target_uid = target.uid
        return target

    def _select_target(self, uid: str) -> None:
        self.selected_target_uid = uid
        target = self._selected_target()
        if target is not None:
            self.status_var.set(f"Target selected: {target.role}")
        self._refresh_everything()

    def _render_ai_panel(self) -> None:
        for child in self.ai_status.winfo_children():
            child.destroy()

        target = self._selected_target()
        for idx, char in enumerate(self.state.ai_team):
            ttk.Label(self.ai_status, text=self._format_char_line(char), font=("Consolas", 15)).grid(
                row=idx, column=0, sticky="w", padx=(0, 8), pady=2
            )

            btn_text = "Target"
            if target is not None and char.uid == target.uid:
                btn_text = "Targeted"

            btn = ttk.Button(
                self.ai_status,
                text=btn_text,
                command=lambda uid=char.uid: self._select_target(uid),
                width=10,
            )
            if target is not None and char.uid == target.uid:
                btn.config(style="Targeted.TButton")
            if not char.alive() or self.state.is_over():
                btn.config(state=tk.DISABLED)
            btn.grid(row=idx, column=1, sticky="e", pady=2)

    def _render_player_panel(self) -> None:
        for child in self.player_status.winfo_children():
            child.destroy()

        eligible = self._eligible_player_uids()
        battle_over = self.state.is_over()

        for idx, char in enumerate(self.state.player_team):
            ttk.Label(self.player_status, text=self._format_char_line(char), font=("Consolas", 15)).grid(
                row=idx, column=0, sticky="w", padx=(0, 8), pady=2
            )

            attack_btn = ttk.Button(
                self.player_status,
                text="Attack",
                width=9,
                command=lambda role=char.role: self.player_take_action(role, "Attack"),
            )
            special_btn = ttk.Button(
                self.player_status,
                text="Special",
                width=9,
                command=lambda role=char.role: self.player_take_action(role, "Special"),
            )

            can_take_turn = char.alive() and char.uid in eligible and not battle_over
            if not can_take_turn:
                attack_btn.config(state=tk.DISABLED)
                special_btn.config(state=tk.DISABLED)
            elif not char.can_use_special():
                special_btn.config(state=tk.DISABLED)

            attack_btn.grid(row=idx, column=1, padx=(0, 6), pady=2)
            special_btn.grid(row=idx, column=2, pady=2)

    def _update_status_hint(self) -> None:
        if self.state.is_over():
            return

        target = self._selected_target()
        eligible_roles = [c.role for c in self.state.player_team if c.uid in self._eligible_player_uids() and c.alive()]
        if target is None:
            self.status_var.set("No target available")
            return

        if len(eligible_roles) == 1:
            self.status_var.set(f"Target: {target.role}. Next actor: {eligible_roles[0]}")
        else:
            self.status_var.set(f"Target: {target.role}. Choose one of: {', '.join(eligible_roles)}")

    def _refresh_everything(self) -> None:
        self._render_ai_panel()
        self._render_player_panel()
        self._update_status_hint()

    # ------------------------- Battle helpers -------------------------

    def _find_by_role(self, team: Sequence[CharacterState], role: str) -> Optional[CharacterState]:
        for c in team:
            if c.role == role and c.alive():
                return c
        return None

    def _deal_damage(self, source: CharacterState, target: CharacterState, amount: int, action_name: str) -> None:
        target.hp = max(0, target.hp - amount)
        self._log_event(f"{source.short_name} uses {action_name} on {target.short_name} for {amount} damage.")
        if not target.alive():
            self._log_event(f"{target.short_name} is defeated.")

    def _alive_allies(self, actor: CharacterState) -> List[CharacterState]:
        if actor.team_name == "Player":
            team = self.state.player_team
        else:
            team = self.state.ai_team
        return [c for c in team if c.alive()]

    def _alive_enemies(self, actor: CharacterState) -> List[CharacterState]:
        if actor.team_name == "Player":
            team = self.state.ai_team
        else:
            team = self.state.player_team
        return [c for c in team if c.alive()]

    def _any_ally_needs_heal(self, actor: CharacterState) -> bool:
        return any(c.hp < c.max_hp for c in self._alive_allies(actor))

    def _is_low_hp(self, target: CharacterState) -> bool:
        return target.hp <= int(0.5 * target.max_hp)

    def _effective_damage(self, actor: CharacterState, base_damage: int, target: CharacterState) -> int:
        damage = float(base_damage)
        if actor.role == "Ranger" and self._is_low_hp(target):
            damage *= self.RANGER_LOW_HP_MULTIPLIER
        return int(round(damage))

    def _estimate_team_damage(self, actor: CharacterState, action: str, enemies: Sequence[CharacterState]) -> int:
        if not enemies:
            return 0

        if action == "Special" and not actor.can_use_special():
            return 0

        weakest = self._weakest_enemy(enemies)
        if weakest is None:
            return 0

        if action == "Special" and actor.role == "Wizard":
            return sum(self._effective_damage(actor, actor.special_damage, enemy) for enemy in enemies)

        if action == "Special":
            return self._effective_damage(actor, actor.special_damage, weakest)

        return self._effective_damage(actor, actor.attack_damage, weakest)

    def _weakest_enemy(self, enemies: Sequence[CharacterState]) -> Optional[CharacterState]:
        if not enemies:
            return None
        return min(enemies, key=lambda e: (e.hp, e.hp / max(1, e.max_hp)))

    def _apply_special(self, actor: CharacterState, target: CharacterState) -> None:
        """Apply the special action of the character on the target, with any special effect that class may have."""
        if actor.role == "Templar":
            # Templar's special damages the target and heals all allies for 10 HP.
            target_damage = self._effective_damage(actor, actor.special_damage, target)
            self._deal_damage(actor, target, target_damage, actor.special_name)
            heal_amount = 10
            healed_anyone = False
            for ally in self._alive_allies(actor):
                before = ally.hp
                ally.hp = min(ally.max_hp, ally.hp + heal_amount)
                if ally.hp > before:
                    healed_anyone = True
                    self._log_event(f"{actor.short_name} heals {ally.short_name} for {ally.hp - before} HP.")
            if not healed_anyone:
                self._log_event(f"{actor.short_name}'s healing wave had no effect (all allies already full HP).")
        elif actor.role == "Wizard":
            # Wizard special splashes the full enemy team.
            for enemy in self._alive_enemies(actor):
                dmg = self._effective_damage(actor, actor.special_damage, enemy)
                self._deal_damage(actor, enemy, dmg, actor.special_name)
        else:
            target_damage = self._effective_damage(actor, actor.special_damage, target)
            self._deal_damage(actor, target, target_damage, actor.special_name)

        actor.spend_special_cost()

    def _apply_regular(self, actor: CharacterState, target: CharacterState) -> None:
        target_damage = self._effective_damage(actor, actor.attack_damage, target)
        self._deal_damage(actor, target, target_damage, "Attack")
        actor.gain_for_regular_attack()

    def _end_actor_turn(self, actor: CharacterState) -> None:
        actor.end_turn_updates()

    def _run_action(self, actor: CharacterState, action: str, target: CharacterState) -> None:
        if action == "Special":
            if not actor.can_use_special():
                self._log_event(f"{actor.short_name} cannot use special now, defaulting to Attack.")
                self._apply_regular(actor, target)
            else:
                self._apply_special(actor, target)
        else:
            self._apply_regular(actor, target)

        self._end_actor_turn(actor)

    # ------------------------- Player turn -------------------------

    def player_take_action(self, actor_role: str, action: str) -> None:
        if self.state.is_over():
            return

        actor = self._find_by_role(self.state.player_team, actor_role)
        target = self._selected_target()

        if actor is None or target is None or action not in {"Attack", "Special"}:
            self.status_var.set("Pick a valid actor and live target")
            self._refresh_everything()
            return

        if actor.uid not in self._eligible_player_uids():
            self.status_var.set(f"{actor.role} already acted this cycle. Use another teammate first.")
            self._refresh_everything()
            return

        self.status_var.set("Player acted; AI thinking...")
        self._run_action(actor, action, target)
        self.player_used_this_cycle.add(actor.uid)

        if self._check_battle_end():
            return

        self.ai_take_turn()
        self._check_battle_end()
        self._refresh_everything()

    # ------------------------- AI resolution policy -------------------------

    def _build_ai_strategy_kb(
        self, ai_actors: Sequence[CharacterState], enemies: Sequence[CharacterState]
    ) -> Tuple[List[Clause], Dict[str, bool], Tuple[CharacterState, str]]:
        # This function builds a small propositional KB for AI action selection.
        # We encode strategic facts as unit clauses and add implication clauses
        # that map those facts to choice symbols (e.g., Choose_Templar_Special).
        clauses: List[Clause] = []
        facts: Dict[str, bool] = {}

        templar = next((a for a in ai_actors if a.role == "Templar"), None)
        can_templar_heal = bool(templar and templar.can_use_special() and self._any_ally_needs_heal(templar))
        facts["CanTemplarHealAnyone"] = can_templar_heal

        # Priority rule:
        # CanTemplarHealAnyone -> Choose_Templar_Special
        # This is the highest-priority decision rule.
        clauses.append(frozenset({"~CanTemplarHealAnyone", "Choose_Templar_Special"}))
        clauses.append(frozenset({"CanTemplarHealAnyone" if can_templar_heal else "~CanTemplarHealAnyone"}))

        # Damage planner:
        # Estimate each eligible actor/action pair's immediate team damage output.
        # The best-scoring pair is converted to a proposition symbol such as
        # BestDamageChoice_Wizard_Special, which we can then reason over.
        damage_options: List[Tuple[int, CharacterState, str]] = []
        for actor in ai_actors:
            attack_damage = self._estimate_team_damage(actor, "Attack", enemies)
            damage_options.append((attack_damage, actor, "Attack"))
            if actor.can_use_special():
                special_damage = self._estimate_team_damage(actor, "Special", enemies)
                damage_options.append((special_damage, actor, "Special"))

        best_damage, best_actor, best_action = max(damage_options, key=lambda t: t[0])
        best_symbol = f"BestDamageChoice_{best_actor.role}_{best_action}"
        facts[best_symbol] = True
        facts["BestDamageAmount"] = best_damage > 0

        # For each candidate symbol, add:
        # BestDamageChoice_X -> Choose_X
        # Then assert exactly one candidate as true (the best one) and the rest false
        # so entailment checks identify the selected action deterministically.
        for _, actor, action in damage_options:
            symbol = f"BestDamageChoice_{actor.role}_{action}"
            clauses.append(frozenset({f"~{symbol}", f"Choose_{actor.role}_{action}"}))
            if symbol == best_symbol:
                clauses.append(frozenset({symbol}))
            else:
                clauses.append(frozenset({f"~{symbol}"}))

        return clauses, facts, (best_actor, best_action)

    def ai_take_turn(self) -> None:
        if self.state.is_over():
            return

        eligible_uids = self._eligible_ai_uids()
        ai_actors = [c for c in self.state.alive_ai() if c.uid in eligible_uids]
        enemies = self.state.alive_player()
        if not ai_actors or not enemies:
            return

        self._log_ai("--- AI reasoning start ---")

        kb, facts, (best_actor, best_action) = self._build_ai_strategy_kb(ai_actors, enemies)

        templar_priority_query = "Choose_Templar_Special"
        templar_priority = pl_resolution_entails(kb, templar_priority_query)

        self._log_ai(
            "Entailment query: Can the templar heal anyone now?"
        )
        self._log_ai(
            f"Resolution check: KB |= {templar_priority_query}  (facts include CanTemplarHealAnyone={facts['CanTemplarHealAnyone']})"
        )
        self._log_ai(f"Result: {templar_priority}")

        if templar_priority:
            actor = next(a for a in ai_actors if a.role == "Templar")
            action = "Special"
        else:
            best_query = f"Choose_{best_actor.role}_{best_action}"
            best_entails = pl_resolution_entails(kb, best_query)
            self._log_ai("Entailment query: Who will do the most damage with their next attack?")
            self._log_ai(
                f"Resolution check: KB |= {best_query}  (computed best option symbol is BestDamageChoice_{best_actor.role}_{best_action})"
            )
            self._log_ai(f"Result: {best_entails}")

            actor = best_actor
            action = best_action

        target = self._weakest_enemy(enemies)
        if target is None:
            return

        self._log_ai(f"Target rule: attack weakest opponent -> {target.role} ({target.hp} HP)")
        self._log_ai(f"Decision: {actor.role} uses {action} on {target.role}")
        self._log_ai("--- AI reasoning end ---")

        self._run_action(actor, action, target)
        self.ai_used_this_cycle.add(actor.uid)
        self.status_var.set("Your turn")

    # ------------------------- Battle lifecycle -------------------------

    def _check_battle_end(self) -> bool:
        self._refresh_everything()
        if not self.state.is_over():
            return False

        winner = self.state.winner()
        if winner is None:
            self.status_var.set("Battle ended")
            self._log_event("Battle ended.")
        elif winner == "Player":
            self.status_var.set("Victory")
            self._log_event("Player team wins!")
        else:
            self.status_var.set("Defeat")
            self._log_event("AI team wins!")

        return True

    def new_battle(self) -> None:
        self.state.new_battle()
        self.player_used_this_cycle.clear()
        self.ai_used_this_cycle.clear()
        self.selected_target_uid = None
        self.status_var.set("Your turn")

        self._set_text(self.event_text, "", append=False)
        self._set_text(self.ai_text, "", append=False)

        self._log_event("New battle started.")
        player_roles = ", ".join(c.role for c in self.state.player_team)
        ai_roles = ", ".join(c.role for c in self.state.ai_team)
        self._log_event(f"Player party: {player_roles}")
        self._log_event(f"AI party: {ai_roles}")

        self._refresh_everything()


if __name__ == "__main__":
    app = ResolutionBattleApp()
    app.mainloop()
