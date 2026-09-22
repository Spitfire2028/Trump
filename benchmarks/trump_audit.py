"""Is ``trump_quality`` a defect, or a trade-off?  (Phase 7.3.13)

Diagnostic, not production. Nothing in the engine imports this, and
nothing here modifies a production source file.

Phase 7.3.12 observed that ``trump_quality`` is negative on 20.5% of
shedding moves and deliberately left it unrepaired. An observation is
not a diagnosis. Phase 7.3.12's own lesson -- *a diagnosis that has not
been reproduced on reachable legal transitions is a hypothesis* -- is
what this module applies to the observation itself.

The formula, read from source::

    StructuralEvaluator._trump_quality(node) =
        sum((card.rank - 6) / 8.0 for card in node.hand
            if card.suit == node.trump)

weighted +0.69, applied at every node with no seat or phase gate.

The seven candidate explanations the phase brief lists are each
answerable from reachable positions, and this module works through them
rather than asserting one.

Run: ``python benchmarks/trump_audit.py <stage>`` with stage in
``reproduce``, ``decompose``, ``gates``, ``legitimacy``.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from durakfish.ai.evaluation import StructuralEvaluator
from durakfish.game import apply_move, get_legal_moves

RESULTS = Path("benchmarks/data")


def transitions(seeds=range(25)):
    """Every reachable legal shedding transition, with context.

    A *shedding* move is one after which the mover holds strictly fewer
    cards. Nothing here is constructed: each row comes from applying a
    legal move to a state reached by legal play.
    """
    from benchmarks.incentive_repair import node_for, walk

    evaluator = StructuralEvaluator()
    for seed in seeds:
        for state in walk(seed):
            player = state.current_player
            if player is None:
                continue
            before = node_for(state, player)
            trumps_before = {
                card for card in state.hands[player]
                if card.suit is state.trump_suit
            }
            for move in get_legal_moves(state):
                after_state = apply_move(state, move)
                if after_state.current_player is None:
                    continue
                if len(after_state.hands[player]) >= len(state.hands[player]):
                    continue
                after = node_for(after_state, player)
                trumps_after = {
                    card for card in after_state.hands[player]
                    if card.suit is after_state.trump_suit
                }
                played = trumps_before - trumps_after
                ranks_before = {c.rank for c in trumps_before}
                ranks_after = {c.rank for c in trumps_after}
                yield {
                    "before": before,
                    "after": after,
                    "raw_delta": (
                        evaluator._trump_quality(after)
                        - evaluator._trump_quality(before)
                    ),
                    "played_trump": bool(played),
                    "played_rank": max(
                        ((c.rank - 6) / 8.0 for c in played), default=0.0
                    ),
                    "last_of_trump_rank": bool(
                        played and (ranks_before - ranks_after)
                    ),
                    "attacker": before.searcher_is_attacker,
                    "phase": state.phase.name,
                    "hand_size": len(state.hands[player]),
                    "talon": len(state.talon),
                    "open_attacks": sum(
                        1 for slot in before.table if not slot.defended
                    ),
                    "attack_limit": before.attack_limit,
                    "trumps_held": len(trumps_before),
                }


def reproduce(_args: list[str]) -> None:
    """Does the 20.5% observation hold, and what is it made of?"""
    rows = list(transitions())
    weight = StructuralEvaluator().structural.trump_quality
    negative = [r for r in rows if r["raw_delta"] < -1e-12]
    print(f"reachable legal shedding transitions: {len(rows)}\n")
    print(f"  trump_quality delta negative in {len(negative)} "
          f"({100 * len(negative) / len(rows):.1f}%)")
    mean_raw = statistics.mean(r["raw_delta"] for r in rows)
    print(f"  mean raw delta        {mean_raw:+.4f}")
    print(f"  mean weighted delta   "
          f"{weight * statistics.mean(r['raw_delta'] for r in rows):+.4f}"
          f"   (weight +{weight})")

    print("\n  every negative case, classified:")
    played = sum(1 for r in negative if r["played_trump"])
    print(f"    the move actually played a trump      {played:>6} "
          f"({100 * played / max(len(negative), 1):.1f}%)")
    print(f"    the move played no trump at all       "
          f"{len(negative) - played:>6}")

    print("\n  and the converse -- every move that played a trump:")
    trump_moves = [r for r in rows if r["played_trump"]]
    fell = sum(1 for r in trump_moves if r["raw_delta"] < -1e-12)
    print(f"    moves playing a trump                 {len(trump_moves):>6}")
    print(f"    of those, trump_quality fell          {fell:>6} "
          f"({100 * fell / max(len(trump_moves), 1):.1f}%)")

    print("\n  Where the two counts agree exactly, the feature is not")
    print("  misfiring: it is reporting that a trump left the hand.")


def decompose(_args: list[str]) -> None:
    """Section 5 -- the full decomposition the brief asks for."""
    rows = list(transitions())
    negative = [r for r in rows if r["raw_delta"] < -1e-12]
    weight = StructuralEvaluator().structural.trump_quality

    def bucket(title: str, key) -> None:
        print(f"\n  by {title}")
        groups: dict = {}
        for row in rows:
            groups.setdefault(key(row), []).append(row)
        for name in sorted(groups, key=str):
            group = groups[name]
            fell = sum(1 for r in group if r["raw_delta"] < -1e-12)
            print(f"    {name!s:<18} n={len(group):<6} "
                  f"negative {100 * fell / len(group):>5.1f}%   "
                  f"mean weighted delta "
                  f"{weight * statistics.mean(r['raw_delta'] for r in group):+.4f}")

    print(f"decomposition over {len(rows)} reachable shedding transitions")
    print(f"  ({len(negative)} with a negative trump_quality delta)")
    bucket("seat", lambda r: "attacker" if r["attacker"] else "defender")
    bucket("phase", lambda r: r["phase"])
    bucket("trump played", lambda r: "trump" if r["played_trump"] else "non-trump")
    bucket("last card of that trump rank",
           lambda r: "last of rank" if r["last_of_trump_rank"] else "other")
    bucket("hand size", lambda r: "<=3" if r["hand_size"] <= 3
           else "4-6" if r["hand_size"] <= 6 else "7+")
    bucket("open attacks", lambda r: str(min(r["open_attacks"], 3)))
    bucket("talon", lambda r: "empty" if r["talon"] == 0 else "non-empty")

    print("\n  rank of the trump played, where one was:")
    played = [r for r in rows if r["played_trump"]]
    if played:
        ranks = [r["played_rank"] for r in played]
        print(f"    n={len(played)}  mean normalised rank "
              f"{statistics.mean(ranks):.3f}  "
              f"min {min(ranks):.3f}  max {max(ranks):.3f}")
        exact = sum(
            1 for r in played
            if abs(r["raw_delta"] + r["played_rank"]) < 1e-12
        )
        print(f"    the delta is exactly minus that rank in {exact} "
              f"of {len(played)} cases")


def gates(_args: list[str]) -> None:
    """Section 6 -- would a seat or phase gate change anything?

    Defect B turned out to be a missing gate, so the same question is
    asked here before any formula change is contemplated.
    """
    rows = list(transitions())
    weight = StructuralEvaluator().structural.trump_quality
    print(f"gate analysis over {len(rows)} reachable shedding transitions\n")
    for label, keep in (
        ("no gate (current)", lambda r: True),
        ("defender only", lambda r: not r["attacker"]),
        ("attacker only", lambda r: r["attacker"]),
        ("DEFENSE phase only", lambda r: r["phase"] == "DEFENSE"),
        ("talon non-empty only", lambda r: r["talon"] > 0),
        ("talon empty only", lambda r: r["talon"] == 0),
    ):
        kept = [r for r in rows if keep(r)]
        if not kept:
            print(f"  {label:<24} no reachable positions")
            continue
        fell = sum(1 for r in kept if r["raw_delta"] < -1e-12)
        print(f"  {label:<24} applies to {len(kept):>6} "
              f"({100 * len(kept) / len(rows):>5.1f}%)   "
              f"negative {100 * fell / len(kept):>5.1f}%   "
              f"mean {weight * statistics.mean(r['raw_delta'] for r in kept):+.4f}")
    print("\n  A gate helps only if the pathology concentrates in the")
    print("  positions it excludes. If every subset looks the same, the")
    print("  feature is not mis-gated and a gate would only hide it.")


def legitimacy(_args: list[str]) -> None:
    """Section 7 -- is a falling trump_quality pathological or correct?

    The decisive question, and it is not about the feature in isolation:
    when a trump leaves the hand, does the *whole* evaluator still prefer
    the position after the move? If it does, the term is a priced
    trade-off rather than an incentive against shedding.
    """
    rows = list(transitions())
    evaluator = StructuralEvaluator()
    trump_moves = [r for r in rows if r["played_trump"]]
    other = [r for r in rows if not r["played_trump"]]

    def summarise(label: str, group: list[dict]) -> None:
        if not group:
            print(f"  {label:<26} none")
            return
        totals = [
            evaluator.evaluate(r["after"], 0) - evaluator.evaluate(r["before"], 0)
            for r in group
        ]
        fell = sum(1 for t in totals if t < -1e-9)
        print(f"  {label:<26} n={len(group):<6} "
              f"full evaluation fell in {100 * fell / len(group):>5.1f}%   "
              f"mean total delta {statistics.mean(totals):+.4f}")

    print(f"does the whole evaluator still prefer shedding? "
          f"({len(rows)} transitions)\n")
    summarise("moves playing a trump", trump_moves)
    summarise("moves playing a non-trump", other)

    print("\n  same split, restricted to E2's own obligation defect being")
    print("  absent (defender moves only, where the obligation term is")
    print("  correctly signed even before repair):")
    summarise(
        "trump, defender",
        [r for r in trump_moves if not r["attacker"]],
    )
    summarise(
        "non-trump, defender",
        [r for r in other if not r["attacker"]],
    )


STAGES = {
    "reproduce": reproduce,
    "decompose": decompose,
    "gates": gates,
    "legitimacy": legitimacy,
}


def main() -> None:
    args = sys.argv[1:]
    stage = args[0] if args else "reproduce"
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; pick from {sorted(STAGES)}")
    STAGES[stage](args[1:])


if __name__ == "__main__":
    main()
