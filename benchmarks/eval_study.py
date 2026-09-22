"""Evaluator quality study against an exact endgame oracle.  (Phase 7.3.7)

Diagnostic, not production. Nothing in the engine imports this.

Reconstructed after a container filesystem reset destroyed the original
working tree, from the Phase 7.3.7 record. Every number it reports is
recomputed on each run, so a stale or mistranscribed constant cannot
masquerade as a result.

The oracle is ``perfect_information_value`` from
``tests/test_abstraction_audit.py``: exhaustive minimax over the frozen
Phase 2 transitions, memoised on the engine's transposition key. It uses
**no** evaluator, no production search, no ISMCTS and no aggregation, so
it is independent of everything under test here. Reusing the frozen
transitions is legitimate — they *define* the rules.

Domain chosen by measurement, not guesswork. Oracle cost per position
with an empty talon, measured in Phase 7.3.7:

====== ============ ============
cards  mean         p95
====== ============ ============
3        1.2 ms        5.6 ms
5        3.4 ms        9.8 ms
6       44.5 ms      204.2 ms
7      264.8 ms        1.24 s
8       11.4 s       (max 219 s)
====== ============ ============

So the study uses **empty talon, at most 6 total cards**, with the
6-card tail sampled rather than exhausted. That keeps a full regeneration
to well under a minute and lets every stage rebuild the corpus
deterministically rather than depend on a cache.

What this study established
---------------------------
* ``BaselineEvaluator`` sign accuracy sits at the majority-class base
  rate — 64.4% dev / 58.8% validation against a 61.1% base rate.
* Two of its three features are inert: **negating** ``own_trump``
  changed correlation from +0.444 to +0.439 and left sign accuracy
  identical, because 0.30 cannot outweigh 1.0 per card.
* A dev-only weight retune improved dev and validation and then got
  *worse* on held-out test (70.4% vs 72.8% baseline, against a 66.7%
  base rate). It was rejected.

Run: ``python benchmarks/eval_study.py <stage>`` where stage is
``accuracy``, ``moves``, ``ablation`` or ``tune``.
"""

from __future__ import annotations

import random
import statistics
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from durakfish.ai import (  # noqa: E402
    BaselineEvaluator,
    SearchConfig,
    build_position,
    search_world,
)
from durakfish.ai.evaluation import EvaluationWeights, Score  # noqa: E402
from durakfish.ai.searchnode import SearchNode  # noqa: E402
from durakfish.game import apply_move, get_legal_moves, new_game  # noqa: E402
from durakfish.information import (  # noqa: E402
    DeterminizedWorld,
    SearchInformation,
    observe,
)
from tests.test_abstraction_audit import perfect_information_value  # noqa: E402

MAX_CARDS = 6
#: Capped by measured oracle cost: the 6-card tail reaches 0.4 s per
#: position, so it is sampled rather than exhausted.
CAPS = {1: 4, 2: 10, 3: 99, 4: 96, 5: 200, 6: 60}


def walk(seed: int):
    rng = random.Random(seed)
    state = new_game(seed=seed)
    while not state.is_over:
        yield state
        state = apply_move(state, rng.choice(get_legal_moves(state)))


def dataset():
    """Exact endgame positions with their game-theoretic values.

    Deterministic: the same seeds always give the same corpus, so every
    stage sees the same dev/validation/test split without a cache.
    """
    sys.setrecursionlimit(50_000)
    pool: dict[int, list] = {}
    for seed in range(90):
        for state in walk(seed):
            if state.talon or state.is_over or state.current_player is None:
                continue
            total = len(state.hands[0]) + len(state.hands[1])
            if total <= MAX_CARDS:
                pool.setdefault(total, []).append(state)

    rows = []
    for total in sorted(pool):
        seen = set()
        for state in pool[total]:
            player = state.current_player
            identity = (state.transposition_key(), player)
            if identity in seen:
                continue
            seen.add(identity)
            rows.append((state, player, perfect_information_value(state, player, {})))
            if len(seen) >= CAPS.get(total, 0):
                break
    random.Random(0).shuffle(rows)
    return rows


def split(rows):
    """Development / validation / held-out test, in that order."""
    half, quarter = len(rows) // 2, len(rows) // 4
    return rows[:half], rows[half : half + quarter], rows[half + quarter :]


def node_for(state, player) -> SearchNode:
    """The evaluation node for a position, via the frozen adapter."""
    information = SearchInformation.from_view(observe(state, player))
    world = DeterminizedWorld(
        player=player,
        opponent_hand=frozenset(state.hands[1 - player]),
        talon=tuple(state.talon),
    )
    return build_position(information, world).evaluation_node(player)


def correlation(xs, ys) -> float:
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else 0.0


def score_with(weights: EvaluationWeights, state, player) -> Score:
    return BaselineEvaluator(weights).evaluate(node_for(state, player), 0)


def stats_for(rows, weights: EvaluationWeights) -> tuple[float, float, float]:
    """Correlation, sign accuracy over decisive positions, flat-score share."""
    scores = [score_with(weights, s, p) for s, p, _ in rows]
    truth = [v for _, _, v in rows]
    decisive = [(x, y) for x, y in zip(scores, truth, strict=True) if y != 0]
    sign = sum(1 for x, y in decisive if (x > 0) == (y > 0)) / max(len(decisive), 1)
    flat = sum(1 for x in scores if abs(x) < 1e-9) / len(scores)
    return correlation(scores, truth), sign, flat


def report(label: str, rows, weights: EvaluationWeights) -> None:
    corr, sign, flat = stats_for(rows, weights)
    print(f"  {label:<26} corr {corr:+.3f}  sign {100 * sign:5.1f}%  "
          f"flat {100 * flat:5.1f}%  (n={len(rows)})")


def move_quality(rows, weights: EvaluationWeights, depth: int | None) -> dict:
    """Best-move agreement with the oracle, and the loss when it disagrees.

    ``depth=None`` ranks moves by the evaluator alone. Ties are handled
    properly: any move achieving the oracle optimum counts as correct,
    since several moves are often equally best.
    """
    correct = considered = 0
    losses: list[int] = []
    for state, player, _ in rows:
        moves = get_legal_moves(state)
        if len(moves) < 2:
            continue
        oracle = {
            m: perfect_information_value(apply_move(state, m), player, {})
            for m in moves
        }
        best = max(oracle.values())
        information = SearchInformation.from_view(observe(state, player))
        world = DeterminizedWorld(
            player=player,
            opponent_hand=frozenset(state.hands[1 - player]),
            talon=tuple(state.talon),
        )
        position = build_position(information, world)
        if depth is None:
            ranked = {
                m: BaselineEvaluator(weights).evaluate(
                    position.apply(m).evaluation_node(player), 1
                )
                for m in moves
            }
        else:
            config = SearchConfig(depth=depth, evaluator="baseline")
            ranked = {
                m: search_world(
                    position, config, root_player=player, root_moves=(m,)
                ).value
                for m in moves
            }
        choice = max(ranked, key=lambda m: (ranked[m], -hash(m)))
        considered += 1
        correct += oracle[choice] == best
        losses.append(best - oracle[choice])
    return {
        "positions": considered,
        "accuracy": correct / max(considered, 1),
        "mean_loss": statistics.mean(losses) if losses else 0.0,
        "max_loss": max(losses) if losses else 0,
    }


BASELINE = EvaluationWeights()


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "accuracy"
    rows = dataset()
    dev, validation, test = split(rows)
    counts: dict[int, int] = {}
    outcomes: dict[int, int] = {}
    for state, _, value in rows:
        total = len(state.hands[0]) + len(state.hands[1])
        counts[total] = counts.get(total, 0) + 1
        outcomes[value] = outcomes.get(value, 0) + 1
    print(f"dataset {len(rows)} positions  by cards {dict(sorted(counts.items()))}")
    print(f"oracle outcomes (+1/0/-1) {dict(sorted(outcomes.items()))}")
    print(f"split dev {len(dev)} / validation {len(validation)} / test {len(test)}\n")

    if stage == "accuracy":
        print("=== BaselineEvaluator vs exact oracle ===")
        report("dev", dev, BASELINE)
        report("validation", validation, BASELINE)

    elif stage == "moves":
        print("=== best-move agreement with the oracle (dev) ===")
        for depth in (None, 1, 2, 3, 4, 5):
            result = move_quality(dev[:70], BASELINE, depth)
            name = "depth 0 (eval only)" if depth is None else f"depth {depth}"
            print(f"  {name:<20} accuracy {100 * result['accuracy']:5.1f}%  "
                  f"mean loss {result['mean_loss']:.3f}  "
                  f"max {result['max_loss']}  (n={result['positions']})")

    elif stage == "ablation":
        print("=== feature ablation (dev) ===")
        variants = {
            "baseline (1.0/0.30/0.50)": BASELINE,
            "no card advantage": EvaluationWeights(card_advantage=0.0),
            "no trump term": EvaluationWeights(own_trump=0.0),
            "no open obligation": EvaluationWeights(open_obligation=0.0),
            "card advantage only": EvaluationWeights(
                own_trump=0.0, open_obligation=0.0
            ),
            "trump term doubled": EvaluationWeights(own_trump=0.60),
            "trump term negated": EvaluationWeights(own_trump=-0.30),
        }
        for label, weights in variants.items():
            report(label, dev, weights)

    elif stage == "tune":
        print("=== grid search on DEV ONLY, then held-out test ===")
        best = None
        for card in (0.0, 0.25, 0.5, 1.0, 2.0):
            for trump in (0.0, 0.3, 1.0, 2.0, 3.0):
                for obligation in (0.0, 0.5, 1.5):
                    weights = EvaluationWeights(
                        card_advantage=card,
                        own_trump=trump,
                        open_obligation=obligation,
                    )
                    corr, sign, _ = stats_for(dev, weights)
                    if best is None or (sign, corr) > (best[0], best[1]):
                        best = (sign, corr, weights)
        assert best is not None
        print(f"  best dev weights: card={best[2].card_advantage} "
              f"trump={best[2].own_trump} obligation={best[2].open_obligation}")
        for name, part in (("validation", validation), ("HELD-OUT TEST", test)):
            _, base_sign, _ = stats_for(part, BASELINE)
            _, tuned_sign, _ = stats_for(part, best[2])
            majority = max(
                sum(1 for _, _, v in part if v > 0),
                sum(1 for _, _, v in part if v < 0),
            ) / len(part)
            print(f"  {name:<14} baseline {100 * base_sign:5.1f}%  "
                  f"tuned {100 * tuned_sign:5.1f}%  "
                  f"(base rate {100 * majority:5.1f}%)")
        print("\n  Phase 7.3.7 rejected the tuned weights: they improved dev and")
        print("  validation and got worse on held-out test.")


if __name__ == "__main__":
    main()
