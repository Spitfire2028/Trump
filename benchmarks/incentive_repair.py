"""Evaluator incentive defects: reproduction and repair.  (Phase 7.3.12)

Diagnostic, not production. Nothing in the engine imports this.

Phase 7.3.11 found, in play rather than in analysis, that two features
introduced in Phase 7.3.10 reward behaviour opposed to the objective of
Durak. This module reproduces both defects as arithmetic before anything
is repaired, then tests whether the repairs change behaviour in the
predicted direction.

The two defects, quoted from source
-----------------------------------
``FlexibleEvaluator._throw_in_options``::

    sum(1 for card in node.hand if card.rank in node.known_ranks)

applied with weight +0.48 when the searcher is the attacker.

``StructuralEvaluator._obligation_pressure``::

    open_attacks / own_cards            (0 when own_cards == 0)

applied with weight -1.98 at every node.

Run: ``python benchmarks/incentive_repair.py <stage>`` with stage in
``reproduce``, ``counterfactual``, ``shed``, ``distribution``.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from durakfish.ai.determinized import build_position
from durakfish.ai.evaluation import EVALUATORS, get_evaluator
from durakfish.ai.search import SearchConfig
from durakfish.ai.search_bot import SearchBot
from durakfish.game import apply_move, get_legal_moves, new_game
from durakfish.information import (
    DeterminizedWorld,
    SearchInformation,
    observe,
)
from durakfish.simulation import play_game

RESULTS = Path("benchmarks/data")

#: R0 is the unrepaired Phase 7.3.10 candidate; R1-R4 are this phase's.
REPAIRS = ("flexible", "flex_r1", "flex_r2", "flex_r3", "flex_r4")


def node_for(state, player: int):
    """The evaluation node, through the frozen materialisation boundary."""
    information = SearchInformation.from_view(observe(state, player))
    world = DeterminizedWorld(
        player=player,
        opponent_hand=frozenset(state.hands[1 - player]),
        talon=tuple(state.talon),
    )
    return build_position(information, world).evaluation_node(player)


def walk(seed: int):
    import random as _random

    rng = _random.Random(seed)
    state = new_game(seed=seed)
    while not state.is_over:
        yield state
        state = apply_move(state, rng.choice(get_legal_moves(state)))


# ======================================================================
# Section 4 -- reproduce both defects before repairing anything
# ======================================================================
def reproduce(_args: list[str]) -> None:
    """Find real positions where shedding a card lowers the score.

    Both defects are reported as exact before/after feature values on
    positions taken from legal play, not from hand-built artificial
    states, so that neither can be dismissed as a constructed corner.
    """
    from durakfish.ai.evaluation import FlexibleEvaluator, StructuralEvaluator

    flexible = FlexibleEvaluator()
    structural = StructuralEvaluator()

    print("DEFECT A -- throw_in_options penalises shedding\n")
    shown = 0
    found_a = 0
    for seed in range(60):
        for state in walk(seed):
            player = state.current_player
            if player is None:
                continue
            node = node_for(state, player)
            if not node.searcher_is_attacker:
                continue
            legal = get_legal_moves(state)
            if len(legal) < 2:
                continue
            for move in legal:
                after = apply_move(state, move)
                if after.current_player is None:
                    continue
                if len(after.hands[player]) >= len(state.hands[player]):
                    continue  # not a shedding move
                child = node_for(after, player)
                before_t = flexible._throw_in_options(node)
                after_t = flexible._throw_in_options(child)
                if after_t >= before_t:
                    continue
                found_a += 1
                if shown < 2:
                    shown += 1
                    weight = flexible.structural.throw_in_options
                    print(f"  example {shown}: seed {seed}, attacker P{player}")
                    print(f"    own hand        {len(state.hands[player])} "
                          f"-> {len(after.hands[player])}   (a card was played)")
                    print(f"    throw_in_options {before_t:.1f} -> {after_t:.1f}"
                          f"   (weight +{weight})")
                    print(f"    contribution    "
                          f"{weight * before_t:+.3f} -> {weight * after_t:+.3f}"
                          f"   = {weight * (after_t - before_t):+.3f}")
                    print(f"    full evaluation {flexible.evaluate(node, 0):+.3f}"
                          f" -> {flexible.evaluate(child, 0):+.3f}")
                    print()
                break
    print(f"  positions where shedding lowered throw_in_options: {found_a}\n")

    print("DEFECT B -- obligation_pressure rises when the hand shrinks\n")
    shown = 0
    found_b = 0
    for seed in range(60):
        for state in walk(seed):
            player = state.current_player
            if player is None:
                continue
            node = node_for(state, player)
            if not any(not slot.defended for slot in node.table):
                continue
            legal = get_legal_moves(state)
            for move in legal:
                after = apply_move(state, move)
                if after.current_player is None:
                    continue
                if len(after.hands[player]) >= len(state.hands[player]):
                    continue
                child = node_for(after, player)
                before_p = structural._obligation_pressure(node)
                after_p = structural._obligation_pressure(child)
                if after_p <= before_p:
                    continue
                # Isolate the defect. If the number of open attacks also
                # rose, the penalty would have risen for a legitimate
                # reason and the example proves nothing about the
                # denominator. Only same-numerator cases are shown.
                if sum(1 for s in child.table if not s.defended) != sum(
                    1 for s in node.table if not s.defended
                ):
                    continue
                found_b += 1
                if shown < 2:
                    shown += 1
                    weight = structural.structural.obligation_pressure
                    open_before = sum(
                        1 for slot in node.table if not slot.defended
                    )
                    open_after = sum(
                        1 for slot in child.table if not slot.defended
                    )
                    print(f"  example {shown}: seed {seed}, P{player}")
                    print(f"    open attacks    {open_before} -> {open_after}")
                    print(f"    own cards       {node.own_cards} -> "
                          f"{child.own_cards}   (a card was played)")
                    print(f"    pressure        {before_p:.4f} -> {after_p:.4f}"
                          f"   (penalty weight -{weight})")
                    print(f"    contribution    "
                          f"{-weight * before_p:+.3f} -> {-weight * after_p:+.3f}"
                          f"   = {-weight * (after_p - before_p):+.3f}")
                    print()
                break
    print(f"  positions where shedding raised obligation_pressure with the\n"
          f"  open-attack count UNCHANGED: {found_b}")

    print("\n  The arithmetic, isolated from any position:")
    for open_attacks, before_cards in ((1, 3), (1, 6), (2, 4)):
        after_cards = before_cards - 1
        before_v = open_attacks / before_cards
        after_v = open_attacks / after_cards
        weight = structural.structural.obligation_pressure
        print(f"    {open_attacks} open attack(s), hand {before_cards} -> "
              f"{after_cards}: pressure {before_v:.4f} -> {after_v:.4f}, "
              f"score {-weight * (after_v - before_v):+.4f}")
    print("    The numerator is untouched. Playing a card is punished")
    print("    purely for shrinking the denominator.")

    print("\n  Neither defect needs a contrived position: both occur in")
    print("  ordinary play. The incentive is small per move and relentless,")
    print("  which is why static inspection missed it and play did not.")


# ======================================================================
# Sections 10-11 -- counterfactual incentive tests
# ======================================================================
def counterfactual(_args: list[str]) -> None:
    """Paired positions differing in exactly one property.

    For each evaluator, over real positions, measure how often playing a
    card **lowers** its evaluation with everything else held as the rules
    dictate. A well-formed evaluator should essentially never prefer the
    position it was in before shedding.
    """
    print("counterfactual: does playing a card lower the evaluation?\n")
    print(f"  {'evaluator':<14} {'shedding moves':>15} {'score fell':>12} "
          f"{'rate':>8} {'mean delta':>12}")
    names = ("baseline", "mechanistic", "structural", *REPAIRS)
    for name in names:
        if name not in EVALUATORS:
            continue
        evaluator = get_evaluator(name)
        fell = seen = 0
        deltas = []
        for seed in range(25):
            for state in walk(seed):
                player = state.current_player
                if player is None:
                    continue
                node = node_for(state, player)
                for move in get_legal_moves(state):
                    after = apply_move(state, move)
                    if after.current_player is None:
                        continue
                    if len(after.hands[player]) >= len(state.hands[player]):
                        continue
                    child = node_for(after, player)
                    delta = evaluator.evaluate(child, 0) - evaluator.evaluate(
                        node, 0
                    )
                    deltas.append(delta)
                    seen += 1
                    fell += delta < -1e-9
        rate = 100 * fell / seen if seen else 0.0
        print(f"  {name:<14} {seen:>15} {fell:>12} {rate:>7.1f}% "
              f"{statistics.mean(deltas) if deltas else 0:>12.4f}")
    print("\n  'score fell' counts moves after which the evaluator prefers")
    print("  the position before the card was played. A high rate is the")
    print("  hoarding incentive Phase 7.3.11 measured in play.")


def shed(args: list[str]) -> None:
    """Section 15 -- the behavioural test that matters.

    On identical positions from real games, how often does each
    evaluator's depth-N policy choose a move that sheds a card? This is
    the measurement that exposed the defect in Phase 7.3.11, repeated
    here for every repair.
    """
    depth = int(args[0]) if args else 1
    games = int(args[1]) if len(args) > 1 else 25
    names = ("baseline", "mechanistic", "structural", *REPAIRS)
    bots = {
        name: SearchBot(SearchConfig(depth=depth, evaluator=name))
        for name in names
        if name in EVALUATORS
    }
    tally = {name: [0, 0] for name in bots}
    for seed in range(games):
        record = play_game(
            [SearchBot(SearchConfig(depth=depth, evaluator="baseline")),
             SearchBot(SearchConfig(depth=depth, evaluator="baseline"))],
            seed=seed, observe_history=False,
        )
        state = record.initial_state
        for move in record.moves:
            legal = get_legal_moves(state)
            player = state.current_player
            if player is not None and len(legal) >= 2:
                before = len(state.hands[player])
                view = observe(state, player)
                for name, bot in bots.items():
                    pick = bot.choose(view, legal)
                    after = len(apply_move(state, pick).hands[player])
                    tally[name][0] += after < before
                    tally[name][1] += 1
            state = apply_move(state, move.move)
    total = next(iter(tally.values()))[1]
    print(f"shed rate on identical positions: depth {depth}, "
          f"{games} games, n={total} choices\n")
    print(f"  {'evaluator':<14} {'sheds a card':>14} {'rate':>9}")
    for name, (shed_count, seen) in tally.items():
        print(f"  {name:<14} {shed_count:>14} "
              f"{100 * shed_count / seen if seen else 0:>8.1f}%")
    print("\n  Durak is won by running out of cards, so a policy that")
    print("  declines to shed is not expressing a subtlety.")


def distribution(_args: list[str]) -> None:
    """Section 12 -- static distribution of the repaired features."""
    from durakfish.ai.evaluation import FlexibleEvaluator

    nodes = []
    for seed in range(20):
        for state in walk(seed):
            player = state.current_player
            if player is not None:
                nodes.append(node_for(state, player))
    original = FlexibleEvaluator()
    repaired = get_evaluator("flex_r3") if "flex_r3" in EVALUATORS else None
    series = {
        "throw_in_options (R0)": [
            original._throw_in_options(n) for n in nodes
        ],
        "obligation_pressure (R0)": [
            original._obligation_pressure(n) for n in nodes
        ],
    }
    if repaired is not None:
        series["attack_continuation (R1/R3)"] = [
            repaired._throw_in_options(n) for n in nodes
        ]
        series["obligation_load (R2/R3)"] = [
            repaired._obligation_pressure(n) for n in nodes
        ]
    print(f"feature distributions over {len(nodes)} positions "
          f"from 20 seeded games\n")
    print(f"  {'feature':<30} {'min':>7} {'max':>7} {'mean':>8} "
          f"{'stdev':>8} {'zeros':>8}")
    for name, values in series.items():
        zeros = sum(1 for v in values if abs(v) < 1e-12)
        print(f"  {name:<30} {min(values):>7.2f} {max(values):>7.2f} "
              f"{statistics.mean(values):>8.3f} "
              f"{statistics.pstdev(values):>8.3f} "
              f"{100 * zeros / len(values):>7.1f}%")


def oracle(args: list[str]) -> None:
    """Section 13 -- exact-solver sanity check, coverage reported.

    Phase 7.3.10 established that the solver's reachable domain is almost
    entirely empty-talon positions. The repairs change the obligation
    term, which is live at every talon size, so unlike the Phase 7.3.10
    material candidate they are not identical to the baseline here. The
    coverage is still far too small to rank anything.
    """
    import time

    depth = int(args[0]) if args else 3
    seconds = float(args[1]) if len(args) > 1 else 200.0
    sys.setrecursionlimit(60_000)
    from talon_reference import solve

    names = ("baseline", "flexible", *REPAIRS[1:])
    bots = {
        name: SearchBot(SearchConfig(depth=depth, evaluator=name))
        for name in names if name in EVALUATORS
    }
    solved = 0
    hits = {name: 0 for name in bots}
    losses = {name: [] for name in bots}
    start = time.time()
    for seed in range(60):
        if time.time() - start > seconds:
            break
        for state in walk(seed):
            if time.time() - start > seconds:
                break
            player = state.current_player
            legal = get_legal_moves(state)
            if player is None or len(legal) < 2:
                continue
            truth = {}
            ok = True
            for move in legal:
                result = solve(apply_move(state, move), player, budget=120_000)
                if not result.exact:
                    ok = False
                    break
                truth[move] = result.value
            if not ok:
                continue
            solved += 1
            best = max(truth.values())
            view = observe(state, player)
            for name, bot in bots.items():
                chosen = bot.choose(view, legal)
                hits[name] += truth[chosen] == best
                losses[name].append(best - truth[chosen])
    print(f"exact-reference check: {solved} solvable positions, depth {depth}\n")
    if not solved:
        print("  nothing solvable within budget")
        return
    print(f"  {'evaluator':<14} {'oracle-optimal':>16} {'mean regret':>13}")
    for name in bots:
        print(f"  {name:<14} {100 * hits[name] / solved:>15.1f}% "
              f"{statistics.mean(losses[name]):>13.3f}")
    print("\n  Coverage is a small, structurally atypical slice. This cannot")
    print("  rank the candidates and is not used to.")


def performance(args: list[str]) -> None:
    """Section 20 -- evaluator cost of each repair."""
    import time

    limit = int(args[0]) if args else 200
    nodes = []
    for seed in range(12):
        for state in walk(seed):
            player = state.current_player
            if player is not None:
                nodes.append(node_for(state, player))
            if len(nodes) >= limit:
                break
        if len(nodes) >= limit:
            break
    names = ("baseline", "flexible", *REPAIRS[1:])
    print(f"evaluator latency: {len(nodes)} nodes x 5 repetitions\n")
    print(f"  {'evaluator':<14} {'calls/sec':>12} {'mean us':>9} "
          f"{'p95 us':>9} {'vs baseline':>12}")
    reference = None
    for name in names:
        if name not in EVALUATORS:
            continue
        evaluator = get_evaluator(name)
        samples = []
        start = time.perf_counter()
        for _ in range(5):
            for node in nodes:
                tick = time.perf_counter()
                evaluator.evaluate(node, 0)
                samples.append((time.perf_counter() - tick) * 1e6)
        elapsed = time.perf_counter() - start
        rate = (5 * len(nodes)) / elapsed
        reference = reference or rate
        ordered = sorted(samples)
        print(f"  {name:<14} {rate:>12,.0f} {statistics.mean(samples):>9.2f} "
              f"{ordered[max(int(0.95 * len(ordered)) - 1, 0)]:>9.2f} "
              f"{reference / rate:>11.2f}x")


STAGES = {
    "reproduce": reproduce,
    "oracle": oracle,
    "performance": performance,
    "counterfactual": counterfactual,
    "shed": shed,
    "distribution": distribution,
}


def main() -> None:
    args = sys.argv[1:]
    stage = args[0] if args else "reproduce"
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; pick from {sorted(STAGES)}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    STAGES[stage](args[1:])


if __name__ == "__main__":
    main()
