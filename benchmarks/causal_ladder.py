"""Controlled causal ladder for the evaluator repairs.  (Phase 7.3.13)

Diagnostic, not production. Nothing in the engine imports this, and
nothing here modifies a production source file: the missing ladder cell
is registered into the existing ``EVALUATORS`` registry at runtime, the
pattern Phase 7.3.9 established.

Why this module exists
----------------------
Phase 7.3.12 compared R2 against R3 and noted the comparison was
confounded. It is worse than confounded -- it is not a comparison of
repairs at all:

===========  ======================================  ==================
variant      base class                              flexibility terms
===========  ======================================  ==================
R2 flex_r2   StructuralEvaluator (E2)                **none**
R3 flex_r3   RepairedThrowInEvaluator -> E3          throw-in + defence
===========  ======================================  ==================

So R2-vs-R3 varies the obligation repair, the throw-in repair, *and*
whether the evaluator has flexibility terms at all, three at once. No
conclusion about "which repairs belong together" can survive that.

The ladder below fixes it by completing a 2x2 factorial on one base.
The cell Phase 7.3.12 never built is **E3 with the obligation repair and
the original per-card throw-in** -- without it the obligation repair
cannot be isolated on the E3 line.

Exact formulas, read from source
--------------------------------
``trump_quality``      sum((rank - 6) / 8) over own trumps          w +0.69
``obligation`` (E2)    open_attacks / own_cards, **ungated**        w -1.98
``obligation`` (rep.)  0 if attacker else open_attacks              w -1.98
``obligation`` (R4)    0 if attacker else open_attacks/attack_limit w -1.98
``throw_in`` (E3)      count of hand cards whose rank is in play    w +0.48
``throw_in`` (rep.)    count of *distinct ranks* in play covered    w +0.48
``defense_flex``       hand cards beating every open attack         w +0.29

The ladder
----------
=============  ====================================================
lad_e3_none    E3 unchanged                       (= flexible / R0)
lad_e3_ti      E3 + throw-in repair               (= flex_r1 / R1)
lad_e3_ob      E3 + obligation repair             (NEW -- the gap)
lad_e3_both    E3 + both repairs                  (= flex_r3 / R3)
lad_e2_none    E2 unchanged                       (= structural)
lad_e2_ob      E2 + obligation repair             (= flex_r2 / R2)
=============  ====================================================

The throw-in factor does not exist on the E2 line: E2 has no throw-in
term, so "E2 + throw-in repair" is E2. That asymmetry is why the design
is a 2x2 on E3 plus a 1x2 on E2, and not a 2x3.

Run: ``python benchmarks/causal_ladder.py <stage>`` with stage in
``map``, ``pilot``, ``play``, ``analyse``, ``ismcts``, ``reproduce``.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from durakfish.ai.evaluation import (
    EVALUATORS,
    FlexibleEvaluator,
    RepairedEvaluator,
    RepairedObligationEvaluator,
    RepairedThrowInEvaluator,
    SearchNode,
    StructuralEvaluator,
    get_evaluator,
)

RESULTS = Path("benchmarks/data")


class LadderE3Obligation(FlexibleEvaluator):
    """The cell Phase 7.3.12 never built: E3 + obligation repair only.

    Identical to ``FlexibleEvaluator`` in every term except the
    obligation, which takes the seat gate verbatim from
    ``RepairedObligationEvaluator``. The per-card throw-in is left
    **unrepaired** on purpose -- that is what makes this the control that
    isolates the obligation repair on the E3 line.
    """

    name = "lad_e3_ob"

    _obligation_pressure = RepairedObligationEvaluator._obligation_pressure


#: Every ladder cell, mapped to the production class it is or extends.
#: Names are deliberately verbose: no cell can be mistaken for another.
LADDER: dict[str, type] = {
    "lad_e3_none": FlexibleEvaluator,
    "lad_e3_ti": RepairedThrowInEvaluator,
    "lad_e3_ob": LadderE3Obligation,
    "lad_e3_both": RepairedEvaluator,
    "lad_e2_none": StructuralEvaluator,
    "lad_e2_ob": RepairedObligationEvaluator,
}

for _label, _factory in LADDER.items():
    if _label not in EVALUATORS:
        # Registering a diagnostic alias, not altering a production class.
        EVALUATORS[_label] = type(
            f"Ladder_{_label}", (_factory,), {"name": _label}
        )


def describe(node: SearchNode) -> str:  # pragma: no cover - reporting aid
    return (
        f"hand={node.own_cards} opp={node.opponent_cards} "
        f"talon={node.talon_size} attacker={node.searcher_is_attacker}"
    )


# ======================================================================
def source_map(_args: list[str]) -> None:
    """Section 1 -- which formula each cell actually differs by."""
    import inspect

    print("ladder cells, and the single method each overrides\n")
    print(f"  {'cell':<14} {'base chain':<46} {'differs by'}")
    for label, factory in LADDER.items():
        chain = " <- ".join(
            c.__name__ for c in factory.__mro__[:4] if c.__name__ != "object"
        )
        overrides = [
            m for m in ("_throw_in_options", "_obligation_pressure")
            if m in factory.__dict__
        ]
        print(f"  {label:<14} {chain:<46} {overrides or 'nothing (base cell)'}")

    print("\n  resolved obligation formula per cell:")
    for label in LADDER:
        evaluator = get_evaluator(label)
        method = type(evaluator)._obligation_pressure
        owner = next(
            c.__name__ for c in type(evaluator).__mro__
            if "_obligation_pressure" in c.__dict__
        )
        src = inspect.getsource(method).splitlines()[1:]
        body = " ".join(
            line.strip() for line in src
            if line.strip() and not line.strip().startswith('"')
        )
        print(f"    {label:<14} from {owner:<28} {body[:70]}")

    print("\n  resolved throw-in formula per cell:")
    for label in LADDER:
        evaluator = get_evaluator(label)
        if not hasattr(evaluator, "_throw_in_options"):
            print(f"    {label:<14} (no throw-in term -- E2 line)")
            continue
        owner = next(
            c.__name__ for c in type(evaluator).__mro__
            if "_throw_in_options" in c.__dict__
        )
        print(f"    {label:<14} from {owner}")


def reproduce(_args: list[str]) -> None:
    """Confirm each cell differs from its neighbours on real positions.

    A factorial is worthless if two cells are secretly the same function.
    This checks every pair over reachable positions before any match is
    played.
    """
    from benchmarks.incentive_repair import node_for, walk

    nodes = []
    for seed in range(15):
        for state in walk(seed):
            player = state.current_player
            if player is not None:
                nodes.append(node_for(state, player))
    labels = list(LADDER)
    values = {
        label: [get_evaluator(label).evaluate(n, 0) for n in nodes]
        for label in labels
    }
    print(f"pairwise distinctness over {len(nodes)} reachable positions\n")
    print(f"  {'pair':<30} {'positions differing':>20}")
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            differ = sum(
                1 for a, b in zip(values[left], values[right], strict=True)
                if abs(a - b) > 1e-12
            )
            flag = "  <-- IDENTICAL" if differ == 0 else ""
            print(f"  {left + ' vs ' + right:<30} "
                  f"{differ:>13} / {len(nodes)}{flag}")


def pilot(args: list[str]) -> None:
    """Paired-outcome variance for the ladder, before sizing the run."""
    from benchmarks.crossplay import paired_match, required_pairs

    pairs_wanted = int(args[0]) if args else 40
    print("ladder pilot: paired variance, deterministic depth 1\n")
    print(f"  {'cell':<14} {'n':>4} {'baseline score':>15} {'sd':>7} "
          f"{'pairs for 5pp':>14}")
    for label in LADDER:
        rows = paired_match(
            "deterministic", 1, "baseline", label, range(pairs_wanted)
        )
        scores = [r["left_score"] for r in rows]
        sd = statistics.pstdev(scores)
        print(f"  {label:<14} {len(rows):>4} "
              f"{statistics.mean(scores):>15.3f} {sd:>7.3f} "
              f"{required_pairs(sd):>14}")


def play(args: list[str]) -> None:
    """Run the ladder against the baseline at depths 1, 3 and 5."""
    from benchmarks.crossplay import paired_match

    pairs_wanted = int(args[0]) if args else 400
    rows: list[dict] = []
    started = time.perf_counter()
    for budget in (1, 3, 5):
        for label in LADDER:
            rows.extend(
                paired_match(
                    "deterministic", budget, "baseline", label,
                    range(pairs_wanted),
                )
            )
            print(f"  done d{budget} baseline vs {label} "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)
    path = RESULTS / "ladder_deterministic.json"
    path.write_text(json.dumps(rows))
    print(f"\n{len(rows)} pairs -> {path}")


def ismcts(args: list[str]) -> None:
    """Section 4 -- the mandatory ISMCTS revalidation.

    The Phase 7.3.11 ISMCTS pilot measured sd 0.419 at ~9.5 s per seed
    pair, so a 5pp-powered cell needs ~551 pairs, about 1.5 hours each.
    This runs the four cells that answer the factorial on the E3 line at
    the largest sample the budget allows, and reports the effect size it
    can actually resolve rather than implying more.
    """
    from benchmarks.crossplay import paired_match

    pairs_wanted = int(args[0]) if args else 40
    budget = int(args[1]) if len(args) > 1 else 50
    cells = ("lad_e3_none", "lad_e3_ti", "lad_e3_ob", "lad_e3_both")
    rows: list[dict] = []
    started = time.perf_counter()
    for label in cells:
        rows.extend(
            paired_match("ismcts", budget, "baseline", label,
                         range(pairs_wanted))
        )
        print(f"  done {budget}it baseline vs {label} "
              f"({time.perf_counter() - started:.0f}s)", flush=True)
    path = RESULTS / f"ladder_ismcts_{budget}.json"
    path.write_text(json.dumps(rows))
    print(f"\n{len(rows)} pairs -> {path}")


def analyse(args: list[str]) -> None:
    """Report each cell, then the isolated effect of each factor."""
    from benchmarks.crossplay import (
        MEANINGFUL_WIN_RATE_DIFFERENCE,
        paired_bootstrap,
    )

    pattern = args[0] if args else "ladder_*.json"
    rows: list[dict] = []
    for path in sorted(RESULTS.glob(pattern)):
        rows.extend(json.loads(path.read_text()))
    if not rows:
        raise SystemExit(f"no results matching {pattern}")

    for mode, budget in sorted({(r["mode"], r["budget"]) for r in rows}):
        subset = [
            r for r in rows if r["mode"] == mode and r["budget"] == budget
        ]
        print(f"== {mode} budget={budget}   ({len(subset)} pairs)")
        print(f"  {'cell':<14} {'n':>4} {'baseline score':>15} "
              f"{'95% CI':>17} {'p':>7} {'candidate score':>16}")
        means = {}
        for label in LADDER:
            cell = [r["left_score"] for r in subset if r["right"] == label]
            if not cell:
                continue
            mean, lo, hi, p = paired_bootstrap(cell)
            means[label] = (mean, len(cell))
            print(f"  {label:<14} {len(cell):>4} {mean:>15.3f} "
                  f"[{lo:>6.3f},{hi:>6.3f}] {p:>7.3f} {1 - mean:>16.3f}")

        print("\n  isolated factor effects (change in CANDIDATE score):")
        contrasts = [
            ("throw-in repair, no obligation repair",
             "lad_e3_none", "lad_e3_ti"),
            ("throw-in repair, with obligation repair",
             "lad_e3_ob", "lad_e3_both"),
            ("obligation repair, no throw-in repair",
             "lad_e3_none", "lad_e3_ob"),
            ("obligation repair, with throw-in repair",
             "lad_e3_ti", "lad_e3_both"),
            ("base E2 -> E3, both unrepaired",
             "lad_e2_none", "lad_e3_none"),
            ("base E2 -> E3, obligation repaired",
             "lad_e2_ob", "lad_e3_ob"),
        ]
        for title, before, after in contrasts:
            if before not in means or after not in means:
                continue
            delta = means[before][0] - means[after][0]
            flag = ("MEANINGFUL" if abs(delta) >= MEANINGFUL_WIN_RATE_DIFFERENCE
                    else "below threshold")
            print(f"    {title:<42} {delta:+.3f}   {flag}")
        print()


STAGES = {
    "map": source_map,
    "reproduce": reproduce,
    "pilot": pilot,
    "play": play,
    "ismcts": ismcts,
    "analyse": analyse,
}


def main() -> None:
    args = sys.argv[1:]
    stage = args[0] if args else "map"
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; pick from {sorted(STAGES)}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    STAGES[stage](args[1:])


if __name__ == "__main__":
    main()
