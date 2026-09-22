"""Evaluator sensitivity & causal impact study.  (Phase 7.3.9)

Diagnostic, not production. Nothing in the engine imports this.

The question is narrow and causal: **when the leaf evaluator is swapped
and everything else is held identical, how often does the search actually
decide differently, and is that a real ranking change or a coin-flip
between equals?**

Controls
--------
Every variant sees the same position, the same determinized world, the
same root moves, the same depth, the same move ordering, the same
alpha-beta rules, the same terminal values and the same RNG. Only the
non-terminal leaf score differs. That is what makes a difference
attributable to the evaluator rather than to search noise.

Injection point
---------------
No production change is needed for the deterministic path: variants are
registered in the existing ``EVALUATORS`` registry and selected by name
through ``SearchConfig.evaluator``, which ``world_search`` already honours.

Tie-break rule, fixed BEFORE any results were seen
--------------------------------------------------
A changed decision is only evidence of evaluator influence if the
evaluators genuinely rank the moves differently. With ``m0`` the move E0
picks and ``mk`` the move Ek picks::

    Type T (tie-break artifact)   |V0(m0) - V0(mk)| <= TOL
                             and  |Vk(m0) - Vk(mk)| <= TOL

    Type R (genuine ranking)      otherwise

Type T means *both* evaluators consider the two moves equally good, so
the switch is an ordering artifact and carries no information. Type R
means at least one evaluator strictly prefers one move.

``TOL = 1e-9``: evaluator terms are sums of coefficients of order 0.3–1.0
and terminal scores of order 10^4, so anything closer than 1e-9 is float
noise, never a real preference. The tolerance and the classification were
written down before the first run.

Information safety
------------------
Every variant scores a ``SearchNode``. That type represents the opponent
as a **count**, never as cards, so an evaluator built on it is
structurally incapable of reading a hidden hand or hidden talon contents.
Feature classification is in ``docs/evaluator-sensitivity.md``.

Run: ``python benchmarks/eval_sensitivity.py <stage>`` where stage is
``deterministic``, ``ismcts`` or ``strata``.

ISMCTS metric, also fixed before the first run
----------------------------------------------
Visit distributions are compared by total variation distance::

    TVD(p, q) = 0.5 * sum_a |p(a) - q(a)|

over root visit counts normalised to sum to one. TVD is 0 when the
search spends its budget identically and 1 when the two searches share
no visited action. Common random numbers are used throughout — same
seed, same iteration budget, same determinization stream — so a non-zero
TVD is attributable to the evaluator and not to a different random walk.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from durakfish.ai.evaluation import (  # noqa: E402
    EVALUATORS,
    BaselineEvaluator,
    EvaluationWeights,
    Score,
    terminal_score,
)
from durakfish.ai.search import SearchConfig  # noqa: E402
from durakfish.ai.searchnode import SearchNode  # noqa: E402
from durakfish.ai.world_search import search_world  # noqa: E402
from durakfish.game import get_legal_moves  # noqa: E402
from durakfish.game.state import GameState  # noqa: E402
from durakfish.information import (  # noqa: E402
    DeterminizedWorld,
    SearchInformation,
    observe,
)
from durakfish.ai.determinized import build_position  # noqa: E402

CORPUS_PATH = Path("benchmarks/data/eval_sensitivity_corpus.json")

#: Declared before the first experiment run. See the module docstring.
TOL = 1e-9

DEPTHS = (1, 3, 5)


# ----------------------------------------------------------------------
# Evaluator variants
# ----------------------------------------------------------------------
class TalonAwareEvaluator:
    """E3: the baseline terms, with card advantage discounted while the
    talon still refills hands.

    Motivated by the Phase 7.3.7/7.3.8 finding that ``card_advantage``
    dominates the baseline yet is close to meaningless while both hands
    refill to six after every bout. The discount is a single a-priori
    structural choice (halve it while cards still come back), **not** a
    tuned coefficient: no weight in this file was fitted to any result.

    Reads only ``SearchNode`` fields — own hand, opponent *count*, talon
    *size*, trump, table, phase — so it cannot see hidden cards.
    """

    name = "talon_aware"

    def __init__(self) -> None:
        self.weights = EvaluationWeights()

    def evaluate(self, node: SearchNode, ply: int = 0) -> Score:
        outcome = node.outcome()
        if outcome is not None:
            return terminal_score(outcome, ply)
        weights = self.weights
        refilling = node.talon_size > 0
        discount = 0.5 if refilling else 1.0
        score = (
            weights.card_advantage
            * discount
            * (node.opponent_cards - node.own_cards)
        )
        score += weights.own_trump * sum(
            1 for card in node.hand if card.suit == node.trump
        )
        if not node.searcher_is_attacker and node.table:
            open_attacks = sum(1 for slot in node.table if not slot.defended)
            score -= weights.open_obligation * open_attacks
        return score


def _weighted(label: str, weights: EvaluationWeights):
    class _Variant(BaselineEvaluator):
        name = label

        def __init__(self) -> None:
            super().__init__(weights)

    _Variant.__name__ = f"Evaluator_{label}"
    return _Variant


#: E0 is the untouched production baseline; E1/E2 are single-feature
#: ablations of it; E3 is the structural alternative above; V3/V4 are
#: amplitude perturbations added in the Phase 7.3.9 completion pass.
#:
#: **V3 and V4 are deliberately implausible.** Tripling and sextupling a
#: coefficient is not a proposal; it is a probe asking whether the search
#: can be moved by that term *at all*, at an amplitude far beyond
#: anything defensible. A high change rate from V3 says the trump term
#: can reach the decision, not that more trump weight is better. Reading
#: either of them as a candidate evaluator inverts their purpose.
VARIANTS = {
    "E0_baseline": BaselineEvaluator,
    "E1_card_only": _weighted(
        "E1_card_only", EvaluationWeights(own_trump=0.0, open_obligation=0.0)
    ),
    "E2_no_card": _weighted("E2_no_card", EvaluationWeights(card_advantage=0.0)),
    "E3_talon_aware": TalonAwareEvaluator,
    # Direction preserved, magnitude amplified: 0.30 -> 3.0 (x10).
    "V3_strong_trump": _weighted(
        "V3_strong_trump", EvaluationWeights(own_trump=3.0)
    ),
    # Direction preserved, magnitude amplified: 0.50 -> 3.0 (x6).
    "V4_strong_obligation": _weighted(
        "V4_strong_obligation", EvaluationWeights(open_obligation=3.0)
    ),
}

for _label, _factory in VARIANTS.items():
    EVALUATORS.setdefault(_label, _factory)


# ----------------------------------------------------------------------
@dataclass
class RootScan:
    """Every root move's value under one evaluator, plus the pick."""

    values: dict
    chosen: object
    nodes: int


def load_corpus() -> list[dict]:
    if not CORPUS_PATH.exists():
        raise SystemExit(
            "corpus missing; run benchmarks/eval_sensitivity_corpus.py first"
        )
    return json.loads(CORPUS_PATH.read_text())


def position_for(row: dict):
    """Rebuild the position and its true-world determinization.

    The true hidden allocation is used so that every variant faces one
    fixed, identical world: the study isolates the evaluator, and
    resampling worlds per variant would confound it with world noise.
    """
    state = GameState.from_dict(row["state"])
    player = row["player"]
    information = SearchInformation.from_view(observe(state, player))
    world = DeterminizedWorld(
        player=player,
        opponent_hand=frozenset(state.hands[1 - player]),
        talon=tuple(state.talon),
    )
    return state, player, build_position(information, world)


def scan_root(position, player: int, depth: int, evaluator: str) -> RootScan:
    """Value every root move under one evaluator, all else held equal."""
    moves = position.legal_moves()
    values = {}
    nodes = 0
    for move in moves:
        config = SearchConfig(depth=depth, evaluator=evaluator)
        result = search_world(
            position, config, root_player=player, root_moves=(move,)
        )
        values[move] = result.value
        nodes += result.stats.nodes
    best = max(values.values())
    chosen = next(m for m in moves if values[m] == best)
    return RootScan(values=values, chosen=chosen, nodes=nodes)


def classify(base: RootScan, other: RootScan) -> str:
    """Fixed-in-advance Type T / Type R rule. See the module docstring."""
    if base.chosen == other.chosen:
        return "same"
    base_gap = abs(base.values[base.chosen] - base.values[other.chosen])
    other_gap = abs(other.values[base.chosen] - other.values[other.chosen])
    if base_gap <= TOL and other_gap <= TOL:
        return "tie_break"
    return "ranking"


def margin(scan: RootScan) -> float:
    """Best-minus-second-best gap: how robustly the pick is separated."""
    ordered = sorted(scan.values.values(), reverse=True)
    if len(ordered) < 2:
        return float("inf")
    return ordered[0] - ordered[1]


# ----------------------------------------------------------------------
def deterministic(rows: list[dict], limit: int = 220) -> dict:
    usable = [r for r in rows if r["legal_move_count"] >= 2][:limit]
    print(f"deterministic world search: {len(usable)} positions, "
          f"depths {DEPTHS}, TOL={TOL:g}\n")
    records = []
    print(f"  {'depth':>5} {'variant':<15} {'changed':>8} {'ranking':>8} "
          f"{'tie':>6} {'rank%':>7} {'|dV| mean':>10}")
    for depth in DEPTHS:
        for label in VARIANTS:
            if label == "E0_baseline":
                continue
            changed = ranking = tie = 0
            deltas: list[float] = []
            for row in usable:
                _, player, position = position_for(row)
                base = scan_root(position, player, depth, "E0_baseline")
                other = scan_root(position, player, depth, label)
                verdict = classify(base, other)
                if verdict != "same":
                    changed += 1
                    if verdict == "ranking":
                        ranking += 1
                        deltas.append(
                            abs(base.values[base.chosen]
                                - base.values[other.chosen])
                        )
                    else:
                        tie += 1
                records.append(
                    {"depth": depth, "variant": label, "verdict": verdict,
                     "game_phase": row["game_phase"],
                     "talon_size": row["talon_size"],
                     "legal_move_count": row["legal_move_count"],
                     "is_attacker": row["is_attacker"],
                     "margin": margin(base)}
                )
            share = 100 * ranking / len(usable)
            mean_delta = statistics.mean(deltas) if deltas else 0.0
            print(f"  {depth:>5} {label:<15} {changed:>8} {ranking:>8} "
                  f"{tie:>6} {share:>6.1f}% {mean_delta:>10.3f}")
    Path("benchmarks/data/eval_sensitivity_records.json").write_text(
        json.dumps(records)
    )
    print(f"\n  records saved: {len(records)}")
    return {"records": records, "positions": len(usable)}


def strata() -> None:
    path = Path("benchmarks/data/eval_sensitivity_records.json")
    if not path.exists():
        raise SystemExit("run the deterministic stage first")
    records = json.loads(path.read_text())

    def report(title: str, key) -> None:
        print(f"\n  by {title}")
        buckets: dict = {}
        for rec in records:
            if rec["depth"] != 3:
                continue
            bucket = key(rec)
            slot = buckets.setdefault(bucket, [0, 0])
            slot[0] += 1
            if rec["verdict"] == "ranking":
                slot[1] += 1
        for bucket in sorted(buckets, key=str):
            total, ranking = buckets[bucket]
            flag = "" if total >= 30 else "   (insufficient sample)"
            print(f"    {str(bucket):<14} n={total:<5} ranking-change "
                  f"{100 * ranking / total:5.1f}%{flag}")

    print("stratified genuine-ranking-change rate (depth 3, all variants)")
    report("game phase", lambda r: r["game_phase"])
    report("talon size", lambda r: "empty" if r["talon_size"] == 0
           else "1-6" if r["talon_size"] <= 6 else "7+")
    report("legal moves", lambda r: "2-3" if r["legal_move_count"] <= 3
           else "4-6" if r["legal_move_count"] <= 6 else "7+")
    report("seat", lambda r: "attacker" if r["is_attacker"] else "defender")

    margins = [r["margin"] for r in records
               if r["depth"] == 3 and r["margin"] != float("inf")]
    near_ties = sum(1 for m in margins if m <= TOL)
    print(f"\n  root best-vs-second margin (depth 3): "
          f"median {statistics.median(margins):.3f}, "
          f"near-ties {100 * near_ties / len(margins):.1f}%")


def total_variation(left: dict, right: dict) -> float:
    """TVD between two root visit distributions. Defined in advance."""
    actions = set(left) | set(right)
    total_left = sum(left.values()) or 1
    total_right = sum(right.values()) or 1
    return 0.5 * sum(
        abs(left.get(a, 0) / total_left - right.get(a, 0) / total_right)
        for a in actions
    )


def ismcts(rows: list[dict], limit: int = 40) -> None:
    """Same experiment through ISMCTS, under common random numbers."""
    from durakfish.ai.ismcts import ISMCTS, ISMCTSConfig

    usable = [r for r in rows if r["legal_move_count"] >= 2][:limit]
    budgets = (50, 200, 800)
    print(f"ISMCTS: {len(usable)} positions, budgets {budgets}, "
          f"common random numbers (seed fixed per position)\n")
    print(f"  {'iters':>6} {'variant':<15} {'changed':>8} {'rate':>7} "
          f"{'TVD mean':>9} {'TVD p95':>8} {'topQ diff':>10}")
    for budget in budgets:
        for label in VARIANTS:
            if label == "E0_baseline":
                continue
            changed = 0
            tvds: list[float] = []
            qdiffs: list[float] = []
            for row in usable:
                state = GameState.from_dict(row["state"])
                player = row["player"]
                information = SearchInformation.from_view(observe(state, player))
                seed = row["seed"] * 1000 + row["ply_index"]
                base = ISMCTS(
                    ISMCTSConfig(iterations=budget, evaluator="E0_baseline")
                ).search(information, random.Random(seed))
                other = ISMCTS(
                    ISMCTSConfig(iterations=budget, evaluator=label)
                ).search(information, random.Random(seed))
                if base.move != other.move:
                    changed += 1
                tvds.append(total_variation(base.visit_counts(),
                                            other.visit_counts()))
                if base.root is not None and other.root is not None:
                    try:
                        qdiffs.append(abs(
                            base.root.edge_for(base.move).mean_value
                            - other.root.edge_for(base.move).mean_value
                        ))
                    except Exception:  # move absent from the other tree
                        pass
            ordered = sorted(tvds)
            p95 = ordered[max(int(0.95 * len(ordered)) - 1, 0)]
            print(f"  {budget:>6} {label:<15} {changed:>8} "
                  f"{100 * changed / len(usable):>6.1f}% "
                  f"{statistics.mean(tvds):>9.3f} {p95:>8.3f} "
                  f"{statistics.mean(qdiffs) if qdiffs else 0.0:>10.3f}")


def reference(rows: list[dict], seconds: float = 170.0) -> None:
    """Optional: compare picks against the independent Phase 7.3.8 solver.

    The solver is the alpha-beta reference from ``talon_reference.py``,
    which uses only the frozen Phase 2 transitions — no BaselineEvaluator,
    no world_search, no ISMCTS. It is therefore a genuine oracle, not a
    pseudo-oracle built from the thing under test.

    Coverage is the limitation, not the method: Phase 7.3.8 established
    that exactly solvable positions are ~1% of talon-bearing play, so this
    stage reports how few positions it could reach and refuses to
    generalise from them.
    """
    import time

    sys.setrecursionlimit(60_000)
    from talon_reference import solve

    from durakfish.game import apply_move

    start = time.time()
    solved = 0
    agree = {label: 0 for label in VARIANTS}
    loss = {label: [] for label in VARIANTS}
    for row in sorted(rows, key=lambda r: r["card_count"]):
        if time.time() - start > seconds:
            break
        if row["legal_move_count"] < 2:
            continue
        state, player, position = position_for(row)
        truth = {}
        ok = True
        for move in get_legal_moves(state):
            result = solve(apply_move(state, move), player, budget=120_000)
            if not result.exact:
                ok = False
                break
            truth[move] = result.value
        if not ok:
            continue
        best = max(truth.values())
        solved += 1
        for label in VARIANTS:
            pick = scan_root(position, player, 3, label).chosen
            agree[label] += truth[pick] == best
            loss[label].append(best - truth[pick])

    print(f"exact-reference comparison: {solved} positions solvable "
          f"(of {len(rows)} in corpus)")
    if solved == 0:
        print("  no position was exactly solvable within the budget")
        return
    print(f"  {'variant':<15} {'oracle-optimal':>15} {'mean move-value loss':>22}")
    for label in VARIANTS:
        print(f"  {label:<15} {100 * agree[label] / solved:>14.1f}% "
              f"{statistics.mean(loss[label]):>22.3f}")
    print("\n  Coverage is far too small to rank the variants; reported only")
    print("  to show the reference was applied where it legitimately could be.")


# ======================================================================
# Phase 7.3.9 completion pass: the four measurements the original run
# did not make. Nothing below changes production code; the counters and
# the injection point it uses already existed.
# ======================================================================
def leaves(rows: list[dict], limit: int = 160) -> None:
    """Section 10 -- terminal leaves versus depth-cutoff leaves.

    This is the measurement that tells you whether the evaluator can
    matter at a given depth at all. ``SearchStats`` already separates the
    two: ``world_search`` calls ``stats.leaf(terminal=True)`` when the
    frozen engine says the game ended and ``stats.leaf(terminal=False)``
    when the depth budget ran out and the evaluator was consulted. So no
    instrumentation was added -- the counters were there and unread.

    A leaf scored by the evaluator is the only place an evaluator can
    influence anything. If a depth produces almost no cutoff leaves, low
    sensitivity at that depth is arithmetic, not evidence about the
    evaluator.
    """
    usable = [r for r in rows if r["legal_move_count"] >= 2][:limit]
    print(f"terminal vs depth-cutoff leaves: {len(usable)} positions, "
          f"baseline evaluator, deterministic\n")
    print(f"  {'depth':>5} {'leaves':>10} {'terminal':>10} {'cutoff':>10} "
          f"{'cutoff share':>13} {'nodes':>10}")
    for depth in (1, 2, 3, 4, 5):
        total = terminal = nodes = 0
        for row in usable:
            _, player, position = position_for(row)
            for move in position.legal_moves():
                result = search_world(
                    position,
                    SearchConfig(depth=depth, evaluator="E0_baseline"),
                    root_player=player,
                    root_moves=(move,),
                )
                total += result.stats.leaves
                terminal += result.stats.terminals
                nodes += result.stats.nodes
        cutoff = total - terminal
        share = 100 * cutoff / total if total else 0.0
        print(f"  {depth:>5} {total:>10} {terminal:>10} {cutoff:>10} "
              f"{share:>12.1f}% {nodes:>10}")
    print("\n  A cutoff leaf is one the evaluator scored. Where that share")
    print("  is near zero, evaluator sensitivity cannot be high whatever")
    print("  the evaluator does.")


class _Recorder:
    """Baseline evaluator that also keeps the leaves it was asked about.

    Used only to collect the *exact* set of positions a baseline search
    scores, so that amplitude can be measured on the same leaves rather
    than on a corpus of root positions the search never reaches.
    """

    name = "rec_baseline"

    def __init__(self) -> None:
        self.weights = EvaluationWeights()
        self._inner = BaselineEvaluator()

    def evaluate(self, node: SearchNode, ply: int = 0) -> Score:
        _RECORDED.append(node)
        return self._inner.evaluate(node, ply)


_RECORDED: list = []
EVALUATORS.setdefault("rec_baseline", _Recorder)


def amplitude(rows: list[dict], limit: int = 120, depth: int = 3) -> None:
    """Section 11 -- how far each perturbation actually moves a leaf score.

    For every leaf a baseline depth-``depth`` search scored, compute
    ``delta = E_variant(leaf) - E_baseline(leaf)``. Terminal leaves are
    excluded: every variant scores those identically by construction, so
    including them would dilute the distribution with structural zeros.

    The comparison that matters is the last column: a perturbation whose
    typical leaf delta is far below the typical root margin cannot be
    expected to change decisions, and one far above it should change many.
    """
    usable = [r for r in rows if r["legal_move_count"] >= 2][:limit]
    _RECORDED.clear()
    margins = []
    for row in usable:
        _, player, position = position_for(row)
        scan = scan_root(position, player, depth, "rec_baseline")
        if margin(scan) != float("inf"):
            margins.append(margin(scan))
    pool = [n for n in _RECORDED if n.outcome() is None]
    print(f"perturbation amplitude at shared leaves: {len(pool)} non-terminal "
          f"leaves\n  from {len(usable)} positions at depth {depth}, "
          f"deterministic\n")
    base = BaselineEvaluator()
    reference_values = [base.evaluate(node, 0) for node in pool]
    median_margin = statistics.median(margins) if margins else float("nan")
    print(f"  {'variant':<22} {'mean |d|':>9} {'median |d|':>11} "
          f"{'p95 |d|':>9} {'max |d|':>9} {'mean |d| / median margin':>26}")
    for label in VARIANTS:
        if label == "E0_baseline":
            continue
        variant = EVALUATORS[label]()
        deltas = [
            abs(variant.evaluate(node, 0) - value)
            for node, value in zip(pool, reference_values, strict=True)
        ]
        ordered = sorted(deltas)
        p95 = ordered[max(int(0.95 * len(ordered)) - 1, 0)]
        mean_delta = statistics.mean(deltas)
        ratio = mean_delta / median_margin if median_margin else float("nan")
        print(f"  {label:<22} {mean_delta:>9.3f} "
              f"{statistics.median(deltas):>11.3f} {p95:>9.3f} "
              f"{max(deltas):>9.3f} {ratio:>26.2f}")
    print(f"\n  median root best-vs-second margin at depth {depth}: "
          f"{median_margin:.3f}  (n={len(margins)})")


def margins(rows: list[dict], limit: int = 160, depth: int = 3) -> None:
    """Section 12 -- are decisions overturned only when they were close?

    Distinguishes "the evaluator decides positions that were genuinely
    balanced" from "the perturbation is strong enough to overturn clearly
    separated decisions". The margin is always measured under the
    **baseline**, so the bucket a position falls into does not depend on
    the variant being tested.
    """
    usable = [r for r in rows if r["legal_move_count"] >= 2][:limit]
    buckets = {"tie (<=1e-9)": [0, 0], "0-0.5": [0, 0], "0.5-1.5": [0, 0],
               "1.5+": [0, 0]}

    def bucket_of(value: float) -> str:
        if value <= TOL:
            return "tie (<=1e-9)"
        if value < 0.5:
            return "0-0.5"
        if value < 1.5:
            return "0.5-1.5"
        return "1.5+"

    values = []
    for row in usable:
        _, player, position = position_for(row)
        base = scan_root(position, player, depth, "E0_baseline")
        gap = margin(base)
        if gap == float("inf"):
            continue
        values.append(gap)
        slot = buckets[bucket_of(gap)]
        for label in VARIANTS:
            if label == "E0_baseline":
                continue
            other = scan_root(position, player, depth, label)
            slot[0] += 1
            slot[1] += classify(base, other) == "ranking"

    print(f"root-margin analysis: {len(values)} positions, depth {depth}, "
          f"all variants pooled, deterministic\n")
    print(f"  margin under baseline: median {statistics.median(values):.3f}, "
          f"mean {statistics.mean(values):.3f}, max {max(values):.3f}")
    print(f"\n  {'margin bucket':<16} {'n (pos x variant)':>18} "
          f"{'ranking change':>16}")
    for name, (total, changed) in buckets.items():
        if not total:
            continue
        flag = "" if total >= 30 else "   (n<30)"
        print(f"  {name:<16} {total:>18} "
              f"{100 * changed / total:>15.1f}%{flag}")


def performance(rows: list[dict], limit: int = 40) -> None:
    """Section 18 -- diagnostic cost, reported separately from production.

    Nothing here is a production performance claim. The point is to show
    what the diagnostic costs and that the recording wrapper used for
    amplitude does not distort the searches it observes.
    """
    import time

    usable = [r for r in rows if r["legal_move_count"] >= 2][:limit]
    positions = [position_for(row) for row in usable]

    def timed(label: str, depth: int) -> tuple[float, int]:
        start = time.perf_counter()
        nodes = 0
        for _, player, position in positions:
            nodes += scan_root(position, player, depth, label).nodes
        return time.perf_counter() - start, nodes

    print(f"diagnostic performance: {len(usable)} positions, "
          f"one process, wall clock\n")
    print(f"  {'depth':>5} {'variant':<22} {'seconds':>9} {'pos/s':>9} "
          f"{'nodes':>10} {'vs baseline':>12}")
    for depth in (1, 3, 5):
        base_seconds, base_nodes = timed("E0_baseline", depth)
        for label in [*VARIANTS, "rec_baseline"]:
            seconds, nodes = (
                (base_seconds, base_nodes) if label == "E0_baseline"
                else timed(label, depth)
            )
            ratio = seconds / base_seconds if base_seconds else float("nan")
            print(f"  {depth:>5} {label:<22} {seconds:>9.2f} "
                  f"{len(usable) / seconds:>9.1f} {nodes:>10} "
                  f"{ratio:>11.2f}x")
        print()
    print("  rec_baseline is the amplitude recorder. Its node count must")
    print("  equal the baseline's: it records leaves, it does not change")
    print("  which leaves are reached.")


def inert(rows: list[dict], limit: int = 120, depth: int = 3) -> None:
    """Section 8 -- why a perturbation that changes leaf values can still
    change no decision.

    ``V4_strong_obligation`` changed 0 of 220 root decisions at every
    depth tested. That is either a probe that never fires or a term the
    search genuinely cannot be moved by, and the difference matters. This
    stage separates them by counting, in order: leaves where the variant
    differs from the baseline at all, leaves where the obligation term is
    even applicable under the baseline's own guard, and positions where a
    root *value* moved without the *pick* moving.
    """
    usable = [r for r in rows if r["legal_move_count"] >= 2][:limit]
    _RECORDED.clear()
    for row in usable:
        _, player, position = position_for(row)
        scan_root(position, player, depth, "rec_baseline")
    pool = [n for n in _RECORDED if n.outcome() is None]
    base = BaselineEvaluator()

    print(f"inertness diagnosis: {len(pool)} non-terminal leaves from "
          f"{len(usable)} positions at depth {depth}\n")
    for label in ("V3_strong_trump", "V4_strong_obligation"):
        variant = EVALUATORS[label]()
        differing = sum(
            1 for node in pool
            if abs(variant.evaluate(node, 0) - base.evaluate(node, 0)) > TOL
        )
        print(f"  {label:<22} leaves changed: {differing:>5} "
              f"({100 * differing / len(pool):>5.1f}%)")
    applicable = sum(
        1 for node in pool
        if BaselineEvaluator._facing_attacks(node)
        and any(not slot.defended for slot in node.table)
    )
    print(f"  {'obligation applicable':<22} leaves         : {applicable:>5} "
          f"({100 * applicable / len(pool):>5.1f}%)")

    print(f"\n  root values moved but the pick did not (depth {depth}):")
    for label in ("V3_strong_trump", "V4_strong_obligation"):
        moved = quiet = 0
        for row in usable:
            _, player, position = position_for(row)
            left = scan_root(position, player, depth, "E0_baseline")
            right = scan_root(position, player, depth, label)
            if any(abs(left.values[m] - right.values[m]) > TOL
                   for m in left.values):
                moved += 1
                quiet += left.chosen == right.chosen
        print(f"    {label:<22} values moved in {moved:>4} of {len(usable)}"
              f" positions; pick unchanged in {quiet:>4}")

    # A perturbation that shifts every sibling by the same amount cannot
    # change an argmax, however large the shift. Test that directly:
    # the spread of the per-move delta within a position is zero exactly
    # when the perturbation acts as a common offset.
    print("\n  within-position spread of the per-move value delta "
          "(0 => common offset, cannot reorder):")
    for label in ("V3_strong_trump", "V4_strong_obligation"):
        spreads = []
        for row in usable:
            _, player, position = position_for(row)
            left = scan_root(position, player, depth, "E0_baseline")
            right = scan_root(position, player, depth, label)
            deltas = [right.values[m] - left.values[m] for m in left.values]
            if max(abs(d) for d in deltas) <= TOL:
                continue
            spreads.append(max(deltas) - min(deltas))
        flat = sum(1 for s in spreads if s <= TOL)
        print(f"    {label:<22} n={len(spreads):>4}  "
              f"mean spread {statistics.mean(spreads) if spreads else 0:>7.3f}  "
              f"pure offset in {flat:>4} of {len(spreads)}")

    # The spread is not a common offset, so the perturbation really does
    # reorder the value landscape. Then why does V4 never change a pick?
    # Test the remaining explanation: amplifying a term the baseline
    # already contains, pointing the same way, *reinforces* the existing
    # ranking. That shows up as the baseline's own pick receiving the
    # largest delta of any root move.
    print("\n  does the perturbation favour the move the baseline already "
          "chose?")
    for label in ("V3_strong_trump", "V4_strong_obligation"):
        reinforced = counted = 0
        for row in usable:
            _, player, position = position_for(row)
            left = scan_root(position, player, depth, "E0_baseline")
            right = scan_root(position, player, depth, label)
            deltas = {m: right.values[m] - left.values[m] for m in left.values}
            if max(abs(d) for d in deltas.values()) <= TOL:
                continue
            counted += 1
            reinforced += deltas[left.chosen] >= max(deltas.values()) - TOL
        if counted:
            print(f"    {label:<22} baseline's pick got the largest delta in "
                  f"{reinforced:>4} of {counted:>4} "
                  f"({100 * reinforced / counted:.1f}%)")


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "deterministic"
    if stage == "strata":
        strata()
        return
    rows = load_corpus()
    stages = {
        "deterministic": deterministic,
        "ismcts": ismcts,
        "reference": reference,
        "leaves": leaves,
        "amplitude": amplitude,
        "margins": margins,
        "performance": performance,
        "inert": inert,
    }
    if stage not in stages:
        raise SystemExit(f"unknown stage {stage!r}")
    stages[stage](rows)


if __name__ == "__main__":
    main()
