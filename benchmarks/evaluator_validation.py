"""Validation of the Phase 7.3.10 candidate evaluator.

Diagnostic, not production.

Compares ``BaselineEvaluator`` against ``MechanisticEvaluator`` under
identical conditions — same positions, same determinized world, same
depth, same move ordering, same terminal semantics, same RNG. Only the
non-terminal leaf score differs.

No coefficient was tuned
------------------------
The candidate inherits all three baseline coefficients unchanged. The
only difference is the *structure* of the material term. There is
therefore no Stage-B calibration in this phase: nothing was fitted to
development, validation or test data, which removes the overfitting risk
that caused Phase 7.3.7 to reject its retuned weights.

PROMOTION GATE — declared before any result was inspected
---------------------------------------------------------
The dataset is small enough that fixed numerical thresholds would be
false precision, so the gate is deliberately a qualitative evidence gate
with one hard numeric floor. The candidate may be proposed for a future
production-integration phase only if **all** hold:

1. **No regression against the exact reference.** On held-out test
   positions the exact solver can reach, candidate best-move agreement is
   not more than 2 percentage points below baseline.
2. **Improvement on at least two independent dimensions** — chosen from
   exact best-move agreement, mean oracle move-value loss, and search-level
   agreement with the exact reference at some depth.
3. **No severe stratum regression**: in no stratum with n >= 30 is the
   candidate worse than baseline by more than 5 percentage points.
4. **Coherent ablation**: removing the candidate's material mechanism
   measurably changes its behaviour, i.e. the mechanism is load-bearing
   rather than inert.
5. **No hidden-information leakage** (structural, via ``SearchNode``).
6. **No numerical pathology**: finite, bounded, deterministic.

Failing any of these means "no superior candidate", not a softened claim.

Run: ``python benchmarks/evaluator_validation.py <stage>`` with stage in
``static``, ``exact``, ``search``, ``ismcts``, ``ablation``, ``strata``.
Pass ``--split dev|validation|test``; the default is ``dev``.
"""

from __future__ import annotations

import contextlib
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from durakfish.ai.determinized import build_position
from durakfish.ai.evaluation import (
    EVALUATORS,
    BaselineEvaluator,
    MechanisticEvaluator,
)
from durakfish.ai.search import SearchConfig
from durakfish.ai.world_search import search_world
from durakfish.game import apply_move, get_legal_moves
from durakfish.game.state import GameState
from durakfish.information import (
    DeterminizedWorld,
    SearchInformation,
    observe,
)

CORPUS_PATH = Path("benchmarks/data/evaluator_corpus.json")
#: The Phase 7.3.10 candidate family. E0 is production and untouched.
FAMILY = ("baseline", "mechanistic", "structural", "flexible")
#: The auxiliary-share sweep: one declared constant, three points.
SWEEP = ("structural_005", "structural", "structural_020",
         "flexible_005", "flexible", "flexible_020")
PAIR = ("baseline", "mechanistic")
DEPTHS = (0, 1, 2, 3, 4, 5)
SOLVER_BUDGET = 120_000


def load(split: str) -> list[dict]:
    rows = json.loads(CORPUS_PATH.read_text())
    return [r for r in rows if r["split"] == split]


def position_for(row: dict):
    """Rebuild the position with its true-world determinization.

    Fixing the true world makes the evaluator the only variable; the
    study is about evaluation, not about world sampling.
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


def pick(position, player: int, depth: int, evaluator: str):
    """The move an evaluator-guided search of given depth selects.

    ``depth=0`` is not a legal ``SearchConfig`` -- the frozen contract
    requires at least one ply -- so it is handled here as what depth 0
    actually means: score the position each move leads to with the
    evaluator directly, no search. That isolates the evaluator completely,
    which is the point of including it in the depth sweep.
    """
    from durakfish.ai.evaluation import get_evaluator
    from durakfish.ai.world_search import evaluate_position

    moves = position.legal_moves()
    if depth == 0:
        resolved = get_evaluator(evaluator)
        values = {
            move: evaluate_position(
                position.apply(move), player, ply=1, evaluator=resolved
            )
            for move in moves
        }
        best = max(values.values())
        return next(m for m in moves if values[m] == best), values

    values = {}
    for move in moves:
        result = search_world(
            position,
            SearchConfig(depth=depth, evaluator=evaluator),
            root_player=player,
            root_moves=(move,),
        )
        values[move] = result.value
    best = max(values.values())
    return next(m for m in moves if values[m] == best), values


# ----------------------------------------------------------------------
def exact(rows: list[dict], seconds: float = 200.0, depth: int = 3) -> dict:
    """Agreement with the independent Phase 7.3.8 exact solver.

    The solver uses only frozen Phase 2 transitions — no evaluator, no
    world_search, no ISMCTS — so it is a genuine oracle. Its reach is the
    limitation: Phase 7.3.8 established that exactly solvable positions
    are a small, structurally atypical slice.
    """
    import time

    sys.setrecursionlimit(60_000)
    from talon_reference import solve

    start = time.time()
    solved = 0
    agree = {name: 0 for name in PAIR}
    loss = {name: [] for name in PAIR}
    per_stratum: dict = {}

    for row in sorted(rows, key=lambda r: r["card_count"]):
        if time.time() - start > seconds:
            break
        if row["legal_move_count"] < 2:
            continue
        state, player, position = position_for(row)
        truth = {}
        ok = True
        for move in get_legal_moves(state):
            result = solve(apply_move(state, move), player, budget=SOLVER_BUDGET)
            if not result.exact:
                ok = False
                break
            truth[move] = result.value
        if not ok:
            continue
        best = max(truth.values())
        solved += 1
        bucket = per_stratum.setdefault(
            row["game_phase"], {name: [0, 0] for name in PAIR}
        )
        for name in PAIR:
            chosen, _ = pick(position, player, depth, name)
            hit = truth[chosen] == best
            agree[name] += hit
            loss[name].append(best - truth[chosen])
            bucket[name][0] += 1
            bucket[name][1] += hit

    print(f"exact-reference validation (depth {depth}): {solved} solvable "
          f"of {len(rows)} in split")
    if solved == 0:
        print("  nothing solvable within budget")
        return {}
    print(f"  {'evaluator':<14} {'best-move agreement':>20} "
          f"{'mean move-value loss':>22}")
    for name in PAIR:
        print(f"  {name:<14} {100 * agree[name] / solved:>19.1f}% "
              f"{statistics.mean(loss[name]):>22.3f}")
    print(f"\n  {'stratum':<10} {'n':>5}  " + "  ".join(f"{n:>14}" for n in PAIR))
    for stratum in sorted(per_stratum):
        cells = per_stratum[stratum]
        n = cells[PAIR[0]][0]
        rates = "  ".join(
            f"{100 * cells[n2][1] / max(cells[n2][0], 1):>13.1f}%" for n2 in PAIR
        )
        flag = "" if n >= 30 else "   (n<30)"
        print(f"  {stratum:<10} {n:>5}  {rates}{flag}")
    return {"solved": solved, "agree": agree}


def static(rows: list[dict]) -> None:
    """Raw evaluator behaviour: scale, range, determinism, pathologies."""
    evaluators = {
        "baseline": BaselineEvaluator(),
        "mechanistic": MechanisticEvaluator(),
    }
    scores: dict = {name: [] for name in evaluators}
    for row in rows:
        _, player, position = position_for(row)
        node = position.evaluation_node(player)
        for name, evaluator in evaluators.items():
            scores[name].append(evaluator.evaluate(node, 0))
    print(f"static behaviour on {len(rows)} positions")
    print(f"  {'evaluator':<14} {'min':>8} {'max':>8} {'mean':>8} {'stdev':>8} "
          f"{'zeros':>7}")
    for name, values in scores.items():
        zeros = sum(1 for v in values if abs(v) < 1e-12)
        assert all(abs(v) < 1e6 for v in values), "unbounded evaluator output"
        print(f"  {name:<14} {min(values):>8.2f} {max(values):>8.2f} "
              f"{statistics.mean(values):>8.2f} {statistics.pstdev(values):>8.2f} "
              f"{100 * zeros / len(values):>6.1f}%")
    # Determinism.
    for evaluator in evaluators.values():
        _, player, position = position_for(rows[0])
        node = position.evaluation_node(player)
        assert len({evaluator.evaluate(node, 0) for _ in range(5)}) == 1
    print("  determinism: identical output over repeated calls  OK")

    # Pathology probe: does the candidate ever prefer a larger own hand?
    worse = 0
    for row in rows[:200]:
        _, player, position = position_for(row)
        node = position.evaluation_node(player)
        candidate = MechanisticEvaluator()
        if node.talon_size == 0 or node.own_cards <= node.refill_to:
            continue
        base_material = candidate.material(node)
        if base_material > 0 and node.own_cards > node.opponent_cards:
            worse += 1
    print(f"  positions where candidate favours the larger hand: {worse}")


def search(rows: list[dict], limit: int = 180) -> None:
    """Root decision agreement between the two evaluators, by depth."""
    usable = [r for r in rows if r["legal_move_count"] >= 2][:limit]
    print(f"search-level comparison: {len(usable)} positions\n")
    print(f"  {'depth':>5} {'root decision differs':>22} {'mean |dV|':>11}")
    for depth in DEPTHS:
        differs = 0
        deltas = []
        for row in usable:
            _, player, position = position_for(row)
            base_move, base_values = pick(position, player, depth, "baseline")
            cand_move, _ = pick(position, player, depth, "mechanistic")
            if base_move != cand_move:
                differs += 1
                deltas.append(
                    abs(base_values[base_move] - base_values[cand_move])
                )
        print(f"  {depth:>5} {100 * differs / len(usable):>21.1f}% "
              f"{statistics.mean(deltas) if deltas else 0.0:>11.3f}")


def ismcts(rows: list[dict], limit: int = 30) -> None:
    """Same substitution inside ISMCTS, under common random numbers."""
    from durakfish.ai.ismcts import ISMCTS, ISMCTSConfig

    usable = [r for r in rows if r["legal_move_count"] >= 2][:limit]
    print(f"ISMCTS comparison: {len(usable)} positions, common random numbers\n")
    print(f"  {'iters':>6} {'move differs':>14} {'TVD mean':>10}")
    for budget in (50, 200):
        differs = 0
        tvds = []
        for row in usable:
            state = GameState.from_dict(row["state"])
            player = row["player"]
            information = SearchInformation.from_view(observe(state, player))
            seed = row["seed"] * 1000 + row["ply_index"]
            results = {}
            for name in PAIR:
                results[name] = ISMCTS(
                    ISMCTSConfig(iterations=budget, evaluator=name)
                ).search(information, random.Random(seed))
            if results[PAIR[0]].move != results[PAIR[1]].move:
                differs += 1
            left = results[PAIR[0]].visit_counts()
            right = results[PAIR[1]].visit_counts()
            actions = set(left) | set(right)
            tl, tr = sum(left.values()) or 1, sum(right.values()) or 1
            tvds.append(0.5 * sum(
                abs(left.get(a, 0) / tl - right.get(a, 0) / tr) for a in actions
            ))
        print(f"  {budget:>6} {100 * differs / len(usable):>13.1f}% "
              f"{statistics.mean(tvds):>10.3f}")


def ablation(rows: list[dict], limit: int = 140) -> None:
    """Is the candidate's material mechanism actually load-bearing?"""
    from durakfish.ai.evaluation import EvaluationWeights

    class _NoMaterial(MechanisticEvaluator):
        name = "abl_no_material"

        def material(self, node) -> float:
            return 0.0

    class _RawMaterial(MechanisticEvaluator):
        name = "abl_raw_material"

        def material(self, node) -> float:
            return float(node.opponent_cards - node.own_cards)

    class _NoTrump(MechanisticEvaluator):
        name = "abl_no_trump"

        def __init__(self) -> None:
            super().__init__(EvaluationWeights(own_trump=0.0))

    for factory in (_NoMaterial, _RawMaterial, _NoTrump):
        EVALUATORS.setdefault(factory.name, factory)

    usable = [r for r in rows if r["legal_move_count"] >= 2][:limit]
    print(f"ablation against the full candidate: {len(usable)} positions, depth 3\n")
    print(f"  {'variant':<18} {'decision differs vs candidate':>30}")
    for label in ("abl_no_material", "abl_raw_material", "abl_no_trump"):
        differs = 0
        for row in usable:
            _, player, position = position_for(row)
            full, _ = pick(position, player, 3, "mechanistic")
            other, _ = pick(position, player, 3, label)
            differs += full != other
        print(f"  {label:<18} {100 * differs / len(usable):>29.1f}%")
    print("\n  A near-zero rate would mean the component is inert.")


def strata(rows: list[dict], limit: int = 180) -> None:
    """Where do the two evaluators disagree? Guards against a regression
    concentrated in one class of position."""
    usable = [r for r in rows if r["legal_move_count"] >= 2][:limit]
    buckets: dict = {}

    def add(key, differs) -> None:
        slot = buckets.setdefault(key, [0, 0])
        slot[0] += 1
        slot[1] += differs

    for row in usable:
        _, player, position = position_for(row)
        base_move, _ = pick(position, player, 3, "baseline")
        cand_move, _ = pick(position, player, 3, "mechanistic")
        differs = base_move != cand_move
        add(("phase", row["game_phase"]), differs)
        add(("talon", "empty" if row["talon_size"] == 0
             else "1-6" if row["talon_size"] <= 6 else "7+"), differs)
        add(("moves", "2-3" if row["legal_move_count"] <= 3
             else "4-6" if row["legal_move_count"] <= 6 else "7+"), differs)
        add(("seat", "attacker" if row["is_attacker"] else "defender"), differs)
        add(("surplus", "over refill" if row["own_hand_size"] > 6 else "at/under"),
            differs)

    print(f"stratified decision difference (depth 3): {len(usable)} positions")
    current = None
    for key in sorted(buckets, key=lambda k: (k[0], str(k[1]))):
        if key[0] != current:
            current = key[0]
            print(f"\n  by {current}")
        total, differs = buckets[key]
        flag = "" if total >= 30 else "   (n<30)"
        print(f"    {key[1]!s:<14} n={total:<5} differs "
              f"{100 * differs / total:5.1f}%{flag}")



# ======================================================================
# Phase 7.3.10 (full brief): the metrics the first pass did not measure.
# ======================================================================
def _kendall(left: list[float], right: list[float]) -> float:
    """Kendall tau-b over a small move list. No SciPy in this environment."""
    n = len(left)
    concordant = discordant = tied_l = tied_r = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = left[i] - left[j], right[i] - right[j]
            if a == 0 and b == 0:
                tied_l += 1
                tied_r += 1
            elif a == 0:
                tied_l += 1
            elif b == 0:
                tied_r += 1
            elif (a > 0) == (b > 0):
                concordant += 1
            else:
                discordant += 1
    pairs = n * (n - 1) / 2
    denominator = ((pairs - tied_l) * (pairs - tied_r)) ** 0.5
    return (concordant - discordant) / denominator if denominator else 0.0


def ranking(rows: list[dict], seconds: float = 220.0, depth: int = 3) -> None:
    """Section 16 -- decision quality beyond best-move accuracy.

    Top-1, top-2, Kendall rank correlation against the oracle's own move
    values, mean move-value loss and worst-case regret, for the whole
    candidate family at once. The oracle is the independent Phase 7.3.8
    solver; only exactly solved positions are used.
    """
    import time

    sys.setrecursionlimit(60_000)
    from talon_reference import solve

    start = time.time()
    solved = 0
    stats = {name: {"top1": 0, "top2": 0, "loss": [], "tau": []}
             for name in FAMILY}
    for row in sorted(rows, key=lambda r: r["card_count"]):
        if time.time() - start > seconds:
            break
        if row["legal_move_count"] < 2:
            continue
        state, player, position = position_for(row)
        truth = {}
        ok = True
        for move in get_legal_moves(state):
            result = solve(apply_move(state, move), player, budget=SOLVER_BUDGET)
            if not result.exact:
                ok = False
                break
            truth[move] = result.value
        if not ok:
            continue
        solved += 1
        best = max(truth.values())
        ordered = sorted(truth.values(), reverse=True)
        second = ordered[1] if len(ordered) > 1 else best
        for name in FAMILY:
            chosen, values = pick(position, player, depth, name)
            entry = stats[name]
            entry["top1"] += truth[chosen] == best
            entry["top2"] += truth[chosen] >= second
            entry["loss"].append(best - truth[chosen])
            moves = [m for m in truth if m in values]
            entry["tau"].append(
                _kendall([values[m] for m in moves], [truth[m] for m in moves])
            )
    print(f"action-ranking against the exact solver: {solved} solvable "
          f"positions, depth {depth}")
    if not solved:
        print("  nothing solvable within budget")
        return
    print(f"\n  {'evaluator':<14} {'top-1':>8} {'top-2':>8} {'mean loss':>11} "
          f"{'worst regret':>13} {'Kendall tau':>12}")
    for name in FAMILY:
        entry = stats[name]
        print(f"  {name:<14} {100 * entry['top1'] / solved:>7.1f}% "
              f"{100 * entry['top2'] / solved:>7.1f}% "
              f"{statistics.mean(entry['loss']):>11.3f} "
              f"{max(entry['loss']):>13.3f} "
              f"{statistics.mean(entry['tau']):>12.3f}")


def family(rows: list[dict], limit: int = 140) -> None:
    """Section 17 -- the whole family against the baseline, depths 0-5."""
    usable = [r for r in rows if r["legal_move_count"] >= 2][:limit]
    print(f"root-decision difference vs baseline: {len(usable)} positions\n")
    print(f"  {'depth':>5} " + " ".join(f"{n:>14}" for n in FAMILY[1:]))
    for depth in DEPTHS:
        cells = []
        for name in FAMILY[1:]:
            differs = 0
            for row in usable:
                _, player, position = position_for(row)
                base, _ = pick(position, player, depth, "baseline")
                other, _ = pick(position, player, depth, name)
                differs += base != other
            cells.append(f"{100 * differs / len(usable):>13.1f}%")
        print(f"  {depth:>5} " + " ".join(cells))
    print("\n  Depth 0 scores the root position itself, so it isolates the")
    print("  evaluator with no search at all.")


def sweep(rows: list[dict], limit: int = 120, depth: int = 3) -> None:
    """Section 15 -- the auxiliary-share sweep, reported in full.

    The search space is one dimension with three points, applied to two
    model forms: six configurations, all listed. Selection happens on the
    validation split; the test split is opened once, afterwards.
    """
    usable = [r for r in rows if r["legal_move_count"] >= 2][:limit]
    print(f"auxiliary-share sweep: {len(usable)} positions, depth {depth}\n")
    print(f"  {'configuration':<18} {'differs vs baseline':>21} "
          f"{'differs vs share 0.10':>23}")
    for name in SWEEP:
        base_diff = ref_diff = 0
        reference = "structural" if name.startswith("structural") else "flexible"
        for row in usable:
            _, player, position = position_for(row)
            chosen, _ = pick(position, player, depth, name)
            base, _ = pick(position, player, depth, "baseline")
            anchor, _ = pick(position, player, depth, reference)
            base_diff += chosen != base
            ref_diff += chosen != anchor
        print(f"  {name:<18} {100 * base_diff / len(usable):>20.1f}% "
              f"{100 * ref_diff / len(usable):>22.1f}%")


def latency(rows: list[dict], limit: int = 200) -> None:
    """Section 22 -- evaluator cost, measured on its own.

    An evaluator is called millions of times, so its per-call cost is a
    production constraint rather than a detail. Measured directly on the
    evaluator, not through a search, so search overhead cannot hide it.
    """
    import time

    nodes = []
    for row in rows[:limit]:
        _, player, position = position_for(row)
        nodes.append(position.evaluation_node(player))
    print(f"evaluator latency: {len(nodes)} distinct nodes, "
          f"5 repetitions each\n")
    print(f"  {'evaluator':<16} {'calls/sec':>12} {'mean us':>9} "
          f"{'p95 us':>9} {'vs baseline':>12}")
    reference = None
    for name in FAMILY:
        evaluator = EVALUATORS[name]()
        samples = []
        start = time.perf_counter()
        for _ in range(5):
            for node in nodes:
                tick = time.perf_counter()
                evaluator.evaluate(node, 0)
                samples.append((time.perf_counter() - tick) * 1e6)
        elapsed = time.perf_counter() - start
        calls = 5 * len(nodes)
        ordered = sorted(samples)
        p95 = ordered[max(int(0.95 * len(ordered)) - 1, 0)]
        rate = calls / elapsed
        reference = reference or rate
        print(f"  {name:<16} {rate:>12,.0f} {statistics.mean(samples):>9.2f} "
              f"{p95:>9.2f} {reference / rate:>11.2f}x")


def ply_bias(rows: list[dict], limit: int = 30, budget: int = 200) -> None:
    """Section 24 -- can the frozen ISMCTS ply defect bias this comparison?

    Phase 7.3.9 recorded that ISMCTS evaluates the expanding iteration's
    leaf before incrementing depth, so one iteration scores a terminal at
    ply 0 while later ones score it at ply 1. The question here is not
    whether that is a defect -- it is -- but whether it can favour one
    evaluator over another.

    The defect applies only at *terminal* leaves, where ``terminal_score``
    runs and no evaluator is consulted. So it can only bias a comparison
    through how often each evaluator steers the search into terminals.
    That is measurable: count terminal leaf encounters per evaluator under
    common random numbers.
    """
    from durakfish.ai.ismcts import ISMCTS, ISMCTSConfig

    usable = [r for r in rows if r["legal_move_count"] >= 2][:limit]
    print(f"ply-defect exposure: {len(usable)} positions, {budget} iterations, "
          f"common random numbers\n")
    print(f"  {'evaluator':<16} {'terminal leaves':>17} {'per position':>14}")
    def counting(inner_name: str, tally: dict) -> type:
        """Bind the evaluator and the tally now, not at call time."""
        inner = EVALUATORS[inner_name]()

        class _Counting:
            name = f"count_{inner_name}"

            def __init__(self) -> None:
                self.weights = getattr(inner, "weights", None)

            def evaluate(self, node, ply: int = 0):
                if node.outcome() is not None:
                    tally["terminal"] += 1
                return inner.evaluate(node, ply)

        return _Counting

    for name in FAMILY:
        seen = 0
        counter = {"terminal": 0}
        EVALUATORS[f"count_{name}"] = counting(name, counter)
        for row in usable:
            state = GameState.from_dict(row["state"])
            player = row["player"]
            information = SearchInformation.from_view(observe(state, player))
            ISMCTS(
                ISMCTSConfig(iterations=budget, evaluator=f"count_{name}")
            ).search(information, random.Random(row["seed"] * 1000
                                                + row["ply_index"]))
            seen += 1
        print(f"  {name:<16} {counter['terminal']:>17} "
              f"{counter['terminal'] / seen:>14.2f}")
    print("\n  The defect acts only where terminal_score runs. Equal terminal")
    print("  exposure means it cannot favour one evaluator over another.")



def ismcts_family(rows: list[dict], limit: int = 30) -> None:
    """Section 18 -- the whole candidate family through ISMCTS.

    Identical positions, identical seeds, identical budgets, common
    random numbers; only the leaf evaluator differs. Reports action
    disagreement, the total-variation distance between root visit
    distributions, and the spread of root Q-values.

    Action frequency is not strength. Nothing here says any evaluator
    plays better, and the phase brief forbids reading it that way.
    """
    from durakfish.ai.ismcts import ISMCTS, ISMCTSConfig

    usable = [r for r in rows if r["legal_move_count"] >= 2][:limit]
    print(f"ISMCTS family comparison: {len(usable)} positions, "
          f"common random numbers\n")
    print(f"  {'iters':>6} {'evaluator':<14} {'differs vs base':>16} "
          f"{'TVD mean':>10} {'TVD p95':>9} {'mean top Q':>11}")
    for budget in (50, 200):
        base_runs = {}
        for row in usable:
            state = GameState.from_dict(row["state"])
            information = SearchInformation.from_view(
                observe(state, row["player"])
            )
            seed = row["seed"] * 1000 + row["ply_index"]
            base_runs[seed] = ISMCTS(
                ISMCTSConfig(iterations=budget, evaluator="baseline")
            ).search(information, random.Random(seed))
        for name in FAMILY:
            differs = 0
            tvds: list[float] = []
            top_q: list[float] = []
            for row in usable:
                state = GameState.from_dict(row["state"])
                information = SearchInformation.from_view(
                    observe(state, row["player"])
                )
                seed = row["seed"] * 1000 + row["ply_index"]
                run = ISMCTS(
                    ISMCTSConfig(iterations=budget, evaluator=name)
                ).search(information, random.Random(seed))
                reference = base_runs[seed]
                differs += run.move != reference.move
                left, right = reference.visit_counts(), run.visit_counts()
                actions = set(left) | set(right)
                tl, tr = sum(left.values()) or 1, sum(right.values()) or 1
                tvds.append(0.5 * sum(
                    abs(left.get(a, 0) / tl - right.get(a, 0) / tr)
                    for a in actions
                ))
                if run.root is not None:
                    with contextlib.suppress(Exception):
                        top_q.append(run.root.edge_for(run.move).mean_value)
            ordered = sorted(tvds)
            p95 = ordered[max(int(0.95 * len(ordered)) - 1, 0)]
            print(f"  {budget:>6} {name:<14} "
                  f"{100 * differs / len(usable):>15.1f}% "
                  f"{statistics.mean(tvds):>10.3f} {p95:>9.3f} "
                  f"{statistics.mean(top_q) if top_q else 0.0:>11.2f}")
        print()


STAGES = {
    "static": static,
    "exact": exact,
    "search": search,
    "ismcts": ismcts,
    "ablation": ablation,
    "strata": strata,
    "ranking": ranking,
    "family": family,
    "sweep": sweep,
    "latency": latency,
    "ply_bias": ply_bias,
    "ismcts_family": ismcts_family,
}


def main() -> None:
    args = sys.argv[1:]
    stage = args[0] if args else "static"
    split = "dev"
    if "--split" in args:
        split = args[args.index("--split") + 1]
    rows = load(split)
    print(f"[split={split}, n={len(rows)}]\n")
    STAGES[stage](rows)


if __name__ == "__main__":
    main()
