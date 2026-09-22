"""Is ``material_persistent`` flat exactly where it should be?  (7.3.14)

Diagnostic, not production. Nothing in the engine imports this, and
nothing here modifies a production source file.

Phase 7.3.13 found that ``material_persistent`` (M1) returns an
unchanged value across roughly 40.8% of reachable legal shedding
transitions, and that ``trump_quality``'s apparent pathology lives in
that flat region. That is a hypothesis about M1, not a finding about it.

The question this module asks is **not** "why is M1 flat?" but "is M1
flat exactly where its own stated semantics say it should be?"

M1's stated semantics, quoted from the class docstring
------------------------------------------------------
    At bout resolution, while the talon still holds cards, ``_refill``
    tops each hand back up to ``rules.hand_size`` but never removes a
    card. A deficit below that size is therefore *transient* -- the
    refill fills it back in -- while a surplus above it is permanent
    until the cards are played.

So M1 makes a falsifiable empirical claim: **a hand below ``refill_to``
gets topped back up.** The flatness is correct exactly to the extent
that claim is true on reachable play.

The frozen rule, quoted from ``game/rules.py::_refill``
-------------------------------------------------------
    order = (state.attacker, state.defender)
    for player in order:
        needed = state.rules.hand_size - len(hands[player])
        drawn = talon[cursor : cursor + needed]

Two conditions M1's formula does not consult: the draw is **limited by
what the talon holds**, and the **attacker draws first**. M1 applies the
flat rule whenever ``talon_size > 0``, including when the talon holds
one card and the deficit is four.

Whether that gap matters is an empirical question about reachable play,
and this module answers it by following real games forward rather than
by arithmetic on hypothetical states.

Run: ``python benchmarks/material_audit.py <stage>`` with stage in
``map``, ``flatness``, ``decompose``, ``transience``, ``falsify``,
``refill``, ``interaction``.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from durakfish.ai.evaluation import MechanisticEvaluator, StructuralEvaluator
from durakfish.game import apply_move, get_legal_moves, new_game

RESULTS = Path("benchmarks/data")


def walk_states(seed: int):
    """Reachable states from one seeded random legal playout."""
    import random as _random

    rng = _random.Random(seed)
    state = new_game(seed=seed)
    while not state.is_over:
        yield state
        state = apply_move(state, rng.choice(get_legal_moves(state)))


def shedding_transitions(seeds):
    """Every reachable legal transition in which the mover sheds a card."""
    from benchmarks.incentive_repair import node_for

    material = MechanisticEvaluator()
    for seed in seeds:
        for state in walk_states(seed):
            player = state.current_player
            if player is None:
                continue
            before = node_for(state, player)
            trump = state.trump_suit
            trumps_before = {c for c in state.hands[player] if c.suit is trump}
            for move in get_legal_moves(state):
                after_state = apply_move(state, move)
                if after_state.current_player is None:
                    continue
                if len(after_state.hands[player]) >= len(state.hands[player]):
                    continue
                after = node_for(after_state, player)
                trumps_after = {
                    c for c in after_state.hands[player] if c.suit is trump
                }
                yield {
                    "seed": seed,
                    "player": player,
                    "state": state,
                    "before": before,
                    "after": after,
                    "delta": material.material(after) - material.material(before),
                    "played_trump": bool(trumps_before - trumps_after),
                    "attacker": before.searcher_is_attacker,
                    "own_before": before.own_cards,
                    "own_after": after.own_cards,
                    "opp": before.opponent_cards,
                    "talon": before.talon_size,
                    "refill_to": before.refill_to,
                    "attack_size": len(before.table),
                    "attack_limit": before.attack_limit,
                }


# ======================================================================
def source_map(_args: list[str]) -> None:
    """Section 1 -- the source-of-truth table."""
    import inspect

    print("component            exact formula / value")
    print("-" * 78)
    print("M1 material          " + " ".join(
        line.strip() for line in
        inspect.getsource(MechanisticEvaluator.material).splitlines()[2:]
    ))
    weights = StructuralEvaluator().structural
    print(f"\nweight               material = {weights.material}")
    print("gate                 talon_size == 0 selects the raw branch")
    print("normalisation        none; card units throughout")
    print("reference quantity   node.refill_to")

    from durakfish.ai.searchnode import SearchNode

    doc = inspect.getdoc(SearchNode) or ""
    for line in doc.splitlines():
        if "refill_to" in line:
            print(f"refill_to (declared) {line.strip()}")

    from durakfish.game.ruleset import STANDARD_RULES

    print(f"refill_to (value)    rules.hand_size = {STANDARD_RULES.hand_size}"
          "  -- a rules constant, not an evaluator heuristic")

    print("\nconsumers            MechanisticEvaluator.evaluate, and every")
    print("                     subclass: Structural, Flexible, R1-R4")
    print("intended meaning     material that survives the refill; a deficit")
    print("                     below refill_to is asserted to be transient")


def flatness(args: list[str]) -> None:
    """Section 3 -- reproduce the 40.8%, from independent seeds."""
    lo = int(args[0]) if args else 0
    hi = int(args[1]) if len(args) > 1 else 25
    rows = list(shedding_transitions(range(lo, hi)))
    flat = [r for r in rows if abs(r["delta"]) < 1e-12]
    deltas = [r["delta"] for r in rows]
    print(f"seeds {lo}..{hi - 1}: {len(rows)} reachable legal shedding "
          f"transitions\n")
    print(f"  zero M1 delta        {len(flat):>6} "
          f"({100 * len(flat) / len(rows):.1f}%)")
    print(f"  mean delta           {statistics.mean(deltas):>+8.4f}")
    print(f"  median delta         {statistics.median(deltas):>+8.4f}")

    def split(title: str, key) -> None:
        print(f"\n  by {title}")
        groups: dict = {}
        for row in rows:
            groups.setdefault(key(row), []).append(row)
        for name in sorted(groups, key=str):
            group = groups[name]
            zero = sum(1 for r in group if abs(r["delta"]) < 1e-12)
            print(f"    {name!s:<16} n={len(group):<6} flat "
                  f"{100 * zero / len(group):>5.1f}%   mean delta "
                  f"{statistics.mean(r['delta'] for r in group):>+7.4f}")

    split("seat", lambda r: "attacker" if r["attacker"] else "defender")
    split("card played", lambda r: "trump" if r["played_trump"] else "non-trump")
    split("own hand size before", lambda r: r["own_before"])
    split("attack size", lambda r: r["attack_size"])
    split("attack limit", lambda r: r["attack_limit"])
    split("refill_to", lambda r: r["refill_to"])
    split("talon", lambda r: "empty" if r["talon"] == 0
          else "1-5" if r["talon"] < 6 else "6+")


def decompose(_args: list[str]) -> None:
    """Section 4 -- is the flatness a coherent strategic concept?"""
    rows = list(shedding_transitions(range(25)))
    flat = [r for r in rows if abs(r["delta"]) < 1e-12]
    print(f"{len(flat)} flat transitions of {len(rows)}\n")

    at_or_below = sum(
        1 for r in flat
        if r["own_before"] <= r["refill_to"] and r["talon"] > 0
    )
    above = sum(
        1 for r in flat
        if r["own_before"] > r["refill_to"] and r["talon"] > 0
    )
    empty = sum(1 for r in flat if r["talon"] == 0)
    print("  flat because:")
    print(f"    hand at or below refill_to, talon alive   {at_or_below:>6}")
    print(f"    hand above refill_to, talon alive         {above:>6}")
    print(f"    talon empty (raw branch, genuinely flat?) {empty:>6}")

    print("\n  and the converse -- transitions where M1 DID change:")
    moved = [r for r in rows if abs(r["delta"]) >= 1e-12]
    crossing = sum(
        1 for r in moved
        if r["own_before"] > r["refill_to"] >= r["own_after"]
    )
    print(f"    total                                     {len(moved):>6}")
    print(f"    of those, the hand crossed refill_to      {crossing:>6}")
    print(f"    of those, the talon was empty             "
          f"{sum(1 for r in moved if r['talon'] == 0):>6}")


def transience(args: list[str]) -> None:
    """Sections 5-6 -- THE decisive test.

    M1 claims a sub-``refill_to`` deficit is transient because the refill
    restores it. That is an empirical claim about reachable play, so it
    is tested by following the real game forward to the next bout
    resolution and asking whether the hand actually came back up.

    A deficit that is *not* restored was never transient, and M1 scored
    it as free.
    """
    hi = int(args[0]) if args else 25
    restored = unrestored = 0
    shortfalls: list[int] = []
    by_talon: dict = {}
    for seed in range(hi):
        states = list(walk_states(seed))
        for index, state in enumerate(states):
            player = state.current_player
            if player is None or not state.talon:
                continue
            refill_to = state.rules.hand_size
            deficit = refill_to - len(state.hands[player])
            if deficit <= 0:
                continue
            # Follow the real continuation to the next moment this player's
            # hand is topped up or the game ends.
            outcome = None
            for later in states[index + 1:]:
                if len(later.hands[player]) >= refill_to:
                    outcome = "restored"
                    break
                if not later.talon:
                    outcome = "unrestored"
                    break
            if outcome is None:
                outcome = "unrestored"
            bucket = by_talon.setdefault(
                "talon 1-5" if len(state.talon) < 6 else "talon 6+",
                [0, 0],
            )
            bucket[1] += 1
            if outcome == "restored":
                restored += 1
                bucket[0] += 1
            else:
                unrestored += 1
                shortfalls.append(deficit)

    total = restored + unrestored
    print(f"M1's transience claim, tested on {total} reachable positions "
          f"where\na player was below refill_to with a non-empty talon "
          f"(seeds 0..{hi - 1})\n")
    print(f"  deficit later restored to refill_to   {restored:>6} "
          f"({100 * restored / total:.1f}%)")
    print(f"  deficit NEVER restored                {unrestored:>6} "
          f"({100 * unrestored / total:.1f}%)")
    if shortfalls:
        print(f"  mean unrestored deficit               "
              f"{statistics.mean(shortfalls):>6.2f} cards")
    print("\n  by talon size at the time of the deficit:")
    for name in sorted(by_talon):
        ok, seen = by_talon[name]
        print(f"    {name:<12} n={seen:<6} restored {100 * ok / seen:>5.1f}%")
    print("\n  M1 treats every one of these as transient. Where the deficit")
    print("  was never restored, the flatness priced a real, permanent")
    print("  difference at zero.")


def refill(_args: list[str]) -> None:
    """Section 6 -- what ``refill_to`` is, exactly."""
    from durakfish.game.ruleset import STANDARD_RULES

    print("refill_to provenance\n")
    print(f"  value                {STANDARD_RULES.hand_size}")
    print("  source               SearchNode.refill_to <- "
          "state.rules.hand_size")
    print("  set at               ai/determinized.py, evaluation_node()")
    print("  state-dependent?     no -- a RuleSet constant, fixed for the "
          "whole game")
    print("  changes after moves? no")
    print("  rule or heuristic?   a game-rule quantity")

    rows = list(shedding_transitions(range(20)))
    values = {r["refill_to"] for r in rows}
    print(f"\n  distinct refill_to values observed in play: {sorted(values)}")
    print("  so the flat region is induced by a constant, and its width is")
    print("  fixed rather than adapting to how much talon is actually left.")

    print("\n  what the frozen rule actually does, by talon size:")
    for talon_size in (0, 1, 2, 3, 6, 12):
        need = 4
        drawn = min(need, talon_size)
        print(f"    talon {talon_size:>2}, attacker needs {need}: draws "
              f"{drawn}, still short by {need - drawn}")


def interaction(_args: list[str]) -> None:
    """Section 7 -- the controlled feature matrix.

    Four separate questions per feature, kept separate: what the feature
    means, what it does to the local move incentive, how it interacts,
    and what the total evaluator does.
    """
    rows = list(shedding_transitions(range(20)))
    evaluator = StructuralEvaluator()
    weights = evaluator.structural
    features = {
        "material (M1)": (
            lambda n: evaluator.material(n), weights.material),
        "trump_quality": (
            lambda n: evaluator._trump_quality(n), weights.trump_quality),
        "obligation": (
            lambda n: evaluator._obligation_pressure(n),
            -weights.obligation_pressure),
        "terminal_proximity": (
            lambda n: evaluator._terminal_proximity(n),
            weights.terminal_proximity),
    }
    print(f"feature matrix over {len(rows)} reachable shedding transitions\n")
    print(f"  {'feature':<20} {'mean raw d':>11} {'mean weighted':>14} "
          f"{'d<0':>7} {'d==0':>7}")
    for name, (compute, weight) in features.items():
        deltas = [compute(r["after"]) - compute(r["before"]) for r in rows]
        neg = 100 * sum(1 for d in deltas if d < -1e-12) / len(deltas)
        zero = 100 * sum(1 for d in deltas if abs(d) < 1e-12) / len(deltas)
        print(f"  {name:<20} {statistics.mean(deltas):>+11.4f} "
              f"{weight * statistics.mean(deltas):>+14.4f} "
              f"{neg:>6.1f}% {zero:>6.1f}%")
    totals = [
        evaluator.evaluate(r["after"], 0) - evaluator.evaluate(r["before"], 0)
        for r in rows
    ]
    print(f"  {'TOTAL evaluator':<20} {statistics.mean(totals):>+11.4f} "
          f"{'':>14} "
          f"{100 * sum(1 for t in totals if t < -1e-9) / len(totals):>6.1f}%")

    print("\n  conditional on M1 being flat vs moving:")
    for label, keep in (("M1 flat", lambda r: abs(r["delta"]) < 1e-12),
                        ("M1 moves", lambda r: abs(r["delta"]) >= 1e-12)):
        group = [r for r in rows if keep(r)]
        if not group:
            continue
        sub = [
            evaluator.evaluate(r["after"], 0) - evaluator.evaluate(r["before"], 0)
            for r in group
        ]
        fell = 100 * sum(1 for t in sub if t < -1e-9) / len(sub)
        print(f"    {label:<10} n={len(group):<6} total evaluation fell in "
              f"{fell:>5.1f}%   mean {statistics.mean(sub):>+7.4f}")


STAGES = {
    "map": source_map,
    "flatness": flatness,
    "decompose": decompose,
    "transience": transience,
    "refill": refill,
    "interaction": interaction,
}


def main() -> None:
    args = sys.argv[1:]
    stage = args[0] if args else "map"
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; pick from {sorted(STAGES)}")
    STAGES[stage](args[1:])


if __name__ == "__main__":
    main()
