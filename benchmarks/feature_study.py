"""Feature archaeology for evaluator design.  (Phase 7.3.10)

Diagnostic, not production. Nothing in the engine imports this.

The phase brief forbids starting from coefficients, so this module starts
from *quantities*: every feature the frozen rules make computable from a
``SearchNode``, formalised, classified, measured on its own, and checked
for redundancy against the others -- all before any candidate evaluator
combines them.

What a SearchNode can supply
----------------------------
A ``SearchNode`` carries the searcher's own hand card-by-card, the table
with defended flags, the opponent as a **count**, the talon as a **size**,
the trump suit, the refill size, the attack limit, the seat and the phase.
There is no field holding hidden cards, so no feature defined here can
read one: HIDDEN_WORLD_ONLY features are not rejected, they are
unconstructible.

One structural warning, measured rather than assumed
----------------------------------------------------
``SearchNode.self_moves()`` returns ``()`` whenever the node is not the
searcher's turn to move. Phase 7.3.9 measured that ISMCTS evaluates on
the *opponent's* turn in 7,295 of 7,591 leaves, so any feature built on
``self_moves()`` would read zero at essentially every ISMCTS leaf. Every
flexibility feature here is therefore defined directly over the hand and
the public table, independent of whose turn it is. ``turn_gated_moves``
is kept as a control so that the size of that trap is visible rather than
argued about.

Run: ``python benchmarks/feature_study.py <stage>`` with stage in
``formalize``, ``distribution``, ``redundancy``, ``oracle``, ``antisymmetry``.
"""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from durakfish.ai.determinized import build_position
from durakfish.game import apply_move, get_legal_moves
from durakfish.game.rules import beats
from durakfish.game.state import GameState
from durakfish.information import (
    DeterminizedWorld,
    SearchInformation,
    observe,
)

CORPUS_PATH = Path("benchmarks/data/evaluator_corpus.json")


# ======================================================================
# Section 8 -- feature formalization
# ======================================================================
@dataclass(frozen=True)
class Feature:
    """One computable quantity, fully specified before it is measured."""

    name: str
    definition: str
    inputs: str
    information: str
    value_range: str
    normalization: str
    expected_sign: str
    symmetry: str
    cost: str
    pathologies: str
    compute: object


def _trumps(node) -> list:
    return [card for card in node.hand if card.suit == node.trump]


def f_material_raw(node) -> float:
    """A. Raw card advantage -- the baseline's material term."""
    return float(node.opponent_cards - node.own_cards)


def f_material_persistent(node) -> float:
    """A. Material that survives refill (the Phase 7.3.10 mechanism)."""
    if node.talon_size == 0:
        return float(node.opponent_cards - node.own_cards)
    refill = node.refill_to
    return float(
        max(0, node.opponent_cards - refill) - max(0, node.own_cards - refill)
    )


def f_own_hand_size(node) -> float:
    """A. Own hand size alone, to test redundancy against material."""
    return float(node.own_cards)


def f_trump_count(node) -> float:
    """B. Own trump count -- the baseline's trump term."""
    return float(len(_trumps(node)))


def f_trump_quality(node) -> float:
    """B. Trump strength, not merely trump quantity.

    Each own trump contributes its rank normalised to [0, 1] over the
    36-card deck's rank span (6..Ace). A hand with the ace of trumps and
    a hand with the six of trumps both score 1 under ``trump_count`` and
    differently here, which is the whole question this feature asks.
    """
    return float(sum((card.rank - 6) / 8.0 for card in _trumps(node)))


def f_trump_control(node) -> float:
    """B. Share of the searcher's hand that is trump.

    Normalised by hand size, so it measures concentration rather than
    quantity and does not simply restate ``trump_count``.
    """
    return float(len(_trumps(node)) / node.own_cards) if node.own_cards else 0.0


def f_distinct_ranks(node) -> float:
    """C. Distinct ranks in hand: how many different attacks can open."""
    return float(len({card.rank for card in node.hand}))


def f_throw_in_options(node) -> float:
    """C. Own cards matching a rank already on the table.

    The rules allow an addition only at a rank already in play, so this
    counts the attack continuations available from the current table.
    Independent of whose turn it is, unlike ``self_moves``.
    """
    return float(sum(1 for card in node.hand if card.rank in node.known_ranks))


def f_attack_room(node) -> float:
    """C. Slots left in the bout before the attack limit binds."""
    return float(max(0, node.attack_limit - len(node.table)))


def f_defense_flexibility(node) -> float:
    """D. Own cards that beat the currently open attack.

    Zero when nothing is open, which is the honest reading: there is no
    defensive obligation to be flexible about.
    """
    open_cards = [
        slot.attacking_card
        for slot in node.table
        if not slot.defended and slot.attacking_card is not None
    ]
    if not open_cards:
        return 0.0
    return float(
        sum(
            1
            for card in node.hand
            if all(beats(card, attack, node.trump) for attack in open_cards)
        )
    )


def f_defense_breadth(node) -> float:
    """D. Cards that beat *some* rank currently on the table.

    A softer reading than ``defense_flexibility``: preserved defensive
    capability rather than the answer to this exact attack.
    """
    ranks = node.known_ranks
    if not ranks:
        return 0.0
    return float(
        sum(
            1
            for card in node.hand
            if card.suit == node.trump or card.rank > min(ranks)
        )
    )


def f_open_obligation(node) -> float:
    """E. Undefended slots -- unguarded by seat, unlike the baseline.

    The baseline counts these only when ``_facing_attacks`` holds. Phase
    7.3.9 measured that guard as never satisfied at an ISMCTS leaf, so
    this ungated version exists to separate "the quantity is useless"
    from "the guard makes it unreachable".
    """
    return float(sum(1 for slot in node.table if not slot.defended))


def f_obligation_pressure(node) -> float:
    """E. Obligation normalised by the hand that has to answer it.

    Two undefended slots facing a two-card hand is a different position
    from two facing an eight-card hand; the raw count cannot say so.
    """
    return (
        float(f_open_obligation(node) / node.own_cards) if node.own_cards else 0.0
    )


def f_seat(node) -> float:
    """F. Tempo: +1 attacking, -1 defending. A constant-per-node indicator."""
    return 1.0 if node.searcher_is_attacker else -1.0


def f_talon_size(node) -> float:
    """G. Talon size. Public; contents are not."""
    return float(node.talon_size)


def f_refill_deficit(node) -> float:
    """G. Cards the searcher will draw at the next resolution.

    ``min(refill_to - own_cards, talon_size)`` clipped at zero: what the
    refill will actually hand back, given both the rule and the supply.
    """
    if node.talon_size == 0:
        return 0.0
    return float(min(max(0, node.refill_to - node.own_cards), node.talon_size))


def f_talon_exhaustion(node) -> float:
    """G. Proximity to the endgame, in [0, 1]: 1 means the talon is gone."""
    return float(max(0.0, 1.0 - node.talon_size / 24.0))


def f_terminal_proximity(node) -> float:
    """H. How close either side is to running out, once that can matter.

    Zero while the talon refills hands, because a small hand is not near
    exhaustion while it is about to be topped up. This is deliberately
    *not* a terminal score: it never approaches ``MATE_MARGIN`` and it
    does not attempt to detect a decided position.
    """
    if node.talon_size > 0:
        return 0.0
    return float(
        (1.0 / (1 + node.own_cards)) - (1.0 / (1 + node.opponent_cards))
    )


def f_control_constant(node) -> float:
    """CONTROL. A constant, so the argmax always falls to the first legal
    move under the harness's deterministic tie-break.

    Every single-feature number must be read against this. A feature that
    is constant across the sibling moves of a position -- ``talon_size``,
    ``seat``, anything shared or turn-invariant -- cannot do better or
    worse than this control, and its score says nothing about the feature.
    """
    return 0.0


def f_turn_gated_moves(node) -> float:
    """Control feature: the Phase 5 legal-move count, turn-gated.

    Not a candidate. It is here to measure how often ``self_moves()``
    reads zero at evaluated leaves, which is the trap described in the
    module docstring.
    """
    return float(len(node.self_moves()))


FEATURES: list[Feature] = [
    Feature("material_raw", "opponent_cards - own_cards",
            "SearchNode.opponent_cards, .hand, .unknown_own",
            "PUBLIC + OWN_PRIVATE", "[-36, 36]",
            "none; natural card units", "positive is better for searcher",
            "antisymmetric under seat swap in a fully known position",
            "O(1)", "ignores refill: a deficit may be transient",
            f_material_raw),
    Feature("material_persistent",
            "talon>0: max(0,opp-refill) - max(0,own-refill); else raw",
            "SearchNode.opponent_cards, .own_cards, .talon_size, .refill_to",
            "PUBLIC + OWN_PRIVATE", "[-36, 36]", "none; card units",
            "positive is better", "antisymmetric", "O(1)",
            "discontinuous at talon exhaustion -- by rule, not by choice",
            f_material_persistent),
    Feature("own_hand_size", "len(hand) + unknown_own",
            "SearchNode.hand, .unknown_own", "OWN_PRIVATE", "[0, 36]",
            "none", "negative is better (shedding wins)",
            "not antisymmetric alone", "O(1)",
            "redundant with material_raw by construction",
            f_own_hand_size),
    Feature("trump_count", "|{c in hand : c.suit == trump}|",
            "SearchNode.hand, .trump", "OWN_PRIVATE + PUBLIC", "[0, 9]",
            "none", "positive", "not antisymmetric (own-only)", "O(hand)",
            "a six of trumps counts as much as the ace", f_trump_count),
    Feature("trump_quality", "sum((rank-6)/8) over own trumps",
            "SearchNode.hand, .trump", "OWN_PRIVATE + PUBLIC", "[0, 9]",
            "rank normalised to [0,1] per card", "positive",
            "not antisymmetric (own-only)", "O(hand)",
            "still blind to which trumps the opponent holds",
            f_trump_quality),
    Feature("trump_control", "trump_count / own_cards",
            "SearchNode.hand, .trump, .own_cards",
            "OWN_PRIVATE + PUBLIC", "[0, 1]", "divided by hand size",
            "positive", "not antisymmetric", "O(hand)",
            "undefined at an empty hand; defined to 0", f_trump_control),
    Feature("distinct_ranks", "|{c.rank : c in hand}|",
            "SearchNode.hand", "OWN_PRIVATE", "[0, 9]", "none", "positive",
            "not antisymmetric", "O(hand)",
            "counts breadth, not strength", f_distinct_ranks),
    Feature("throw_in_options", "|{c in hand : c.rank in known_ranks}|",
            "SearchNode.hand, .known_ranks", "OWN_PRIVATE + PUBLIC",
            "[0, 36]", "none", "positive when attacking",
            "not antisymmetric", "O(hand)",
            "reads 0 at an empty table, where it is meaningless",
            f_throw_in_options),
    Feature("attack_room", "max(0, attack_limit - |table|)",
            "SearchNode.attack_limit, .table", "PUBLIC", "[0, 6]", "none",
            "sign depends on seat", "not antisymmetric", "O(1)",
            "a bout-structure quantity, not a hand quantity",
            f_attack_room),
    Feature("defense_flexibility",
            "|{c in hand : c beats every open attack}|",
            "SearchNode.hand, .table, .trump", "OWN_PRIVATE + PUBLIC",
            "[0, 36]", "none", "positive when defending",
            "not antisymmetric", "O(hand x table)",
            "0 when nothing is open, which is not the same as 'cannot "
            "defend'", f_defense_flexibility),
    Feature("defense_breadth",
            "|{c in hand : trump or rank > min(known_ranks)}|",
            "SearchNode.hand, .known_ranks, .trump",
            "OWN_PRIVATE + PUBLIC", "[0, 36]", "none", "positive",
            "not antisymmetric", "O(hand)",
            "approximate: uses ranks on the table, not suits",
            f_defense_breadth),
    Feature("open_obligation", "|{slot in table : not defended}|",
            "SearchNode.table", "PUBLIC", "[0, 6]", "none",
            "negative when the searcher must answer", "not antisymmetric",
            "O(table)", "ungated: counts slots that are not the "
            "searcher's problem", f_open_obligation),
    Feature("obligation_pressure", "open_obligation / own_cards",
            "SearchNode.table, .own_cards", "PUBLIC + OWN_PRIVATE",
            "[0, 6]", "divided by hand size", "negative",
            "not antisymmetric", "O(table)",
            "explodes as the hand empties; 0 at an empty hand",
            f_obligation_pressure),
    Feature("seat", "+1 attacker, -1 defender",
            "SearchNode.searcher_is_attacker", "PUBLIC", "{-1, +1}",
            "none", "unknown a priori -- to be measured",
            "antisymmetric by construction", "O(1)",
            "constant across a bout; cannot separate moves within one",
            f_seat),
    Feature("talon_size", "len(talon)", "SearchNode.talon_size", "PUBLIC",
            "[0, 24]", "none", "no independent sign", "symmetric (shared)",
            "O(1)", "identical for both players, so it can only act as a "
            "gate on other terms", f_talon_size),
    Feature("refill_deficit", "min(max(0, refill_to - own), talon_size)",
            "SearchNode.own_cards, .refill_to, .talon_size",
            "PUBLIC + OWN_PRIVATE", "[0, 6]", "none",
            "positive (cards are coming)", "not antisymmetric", "O(1)",
            "assumes the searcher draws, which the rules only grant at "
            "bout resolution", f_refill_deficit),
    Feature("talon_exhaustion", "max(0, 1 - talon_size/24)",
            "SearchNode.talon_size", "PUBLIC", "[0, 1]",
            "divided by the initial talon", "no independent sign",
            "symmetric (shared)", "O(1)",
            "the 24 is the deal-time talon, not a tuned constant",
            f_talon_exhaustion),
    Feature("terminal_proximity",
            "talon==0: 1/(1+opp) - 1/(1+own); else 0",
            "SearchNode.opponent_cards, .own_cards, .talon_size",
            "PUBLIC + OWN_PRIVATE", "[-1, 1]", "reciprocal",
            "positive is better", "antisymmetric", "O(1)",
            "must not shadow terminal scoring; bounded well inside "
            "MATE_MARGIN", f_terminal_proximity),
    Feature("control_constant",
            "0 -- CONTROL: picks the first legal move via the tie-break",
            "none", "n/a", "{0}", "n/a", "n/a", "n/a", "O(1)",
            "the floor every other single-feature score must be read "
            "against", f_control_constant),
    Feature("turn_gated_moves", "len(self_moves()) -- CONTROL, not a candidate",
            "SearchNode.self_moves", "OWN_PRIVATE + PUBLIC", "[0, ~20]",
            "none", "n/a", "n/a", "O(hand x table)",
            "reads 0 whenever it is not the searcher's turn -- the trap "
            "this control exists to expose", f_turn_gated_moves),
]

BY_NAME = {feature.name: feature for feature in FEATURES}


# ======================================================================
def load(split: str) -> list[dict]:
    rows = json.loads(CORPUS_PATH.read_text())
    return [row for row in rows if row["split"] == split]


def node_of(row: dict):
    state = GameState.from_dict(row["state"])
    player = row["player"]
    information = SearchInformation.from_view(observe(state, player))
    world = DeterminizedWorld(
        player=player,
        opponent_hand=frozenset(state.hands[1 - player]),
        talon=tuple(state.talon),
    )
    position = build_position(information, world)
    return state, player, position, position.evaluation_node(player)


def pearson(left: list[float], right: list[float]) -> float:
    n = len(left)
    if n < 2:
        return float("nan")
    mean_l, mean_r = statistics.mean(left), statistics.mean(right)
    num = sum((a - mean_l) * (b - mean_r) for a, b in zip(left, right,
                                                          strict=True))
    den_l = sum((a - mean_l) ** 2 for a in left) ** 0.5
    den_r = sum((b - mean_r) ** 2 for b in right) ** 0.5
    return num / (den_l * den_r) if den_l and den_r else 0.0


def spearman(left: list[float], right: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    return pearson(ranks(left), ranks(right))


# ======================================================================
def formalize(_rows: list[dict]) -> None:
    """Section 8 -- print the specification of every feature."""
    print(f"{len(FEATURES)} features formalized "
          f"({sum(1 for f in FEATURES if 'CONTROL' in f.definition)} control)\n")
    for feature in FEATURES:
        print(f"  {feature.name}")
        print(f"    definition   {feature.definition}")
        print(f"    inputs       {feature.inputs}")
        print(f"    information  {feature.information}")
        print(f"    range        {feature.value_range}")
        print(f"    normalized   {feature.normalization}")
        print(f"    sign         {feature.expected_sign}")
        print(f"    symmetry     {feature.symmetry}")
        print(f"    cost         {feature.cost}")
        print(f"    pathology    {feature.pathologies}\n")
    classes: dict = {}
    for feature in FEATURES:
        classes[feature.information] = classes.get(feature.information, 0) + 1
    print("  information classes present:")
    for name, count in sorted(classes.items()):
        print(f"    {name:<24} {count}")
    print("    HIDDEN_WORLD_ONLY        0   (unconstructible from SearchNode)")


def distribution(rows: list[dict]) -> None:
    """Section 9 -- what each feature actually looks like on real play."""
    nodes = [node_of(row)[3] for row in rows]
    print(f"feature distributions: n={len(nodes)} positions "
          f"(split as loaded), deterministic\n")
    print(f"  {'feature':<22} {'min':>7} {'max':>7} {'mean':>8} {'stdev':>8} "
          f"{'zeros':>7} {'distinct':>9}")
    for feature in FEATURES:
        values = [feature.compute(node) for node in nodes]
        zeros = sum(1 for v in values if abs(v) < 1e-12)
        print(f"  {feature.name:<22} {min(values):>7.2f} {max(values):>7.2f} "
              f"{statistics.mean(values):>8.3f} "
              f"{statistics.pstdev(values):>8.3f} "
              f"{100 * zeros / len(values):>6.1f}% "
              f"{len(set(values)):>9}")
    print("\n  A feature that is constant, or zero almost everywhere, cannot")
    print("  separate positions however sound its motivation.")


def redundancy(rows: list[dict]) -> None:
    """Section 10 -- which features are restatements of each other."""
    nodes = [node_of(row)[3] for row in rows]
    values = {
        feature.name: [feature.compute(node) for node in nodes]
        for feature in FEATURES
    }
    live = [
        name for name, series in values.items()
        if statistics.pstdev(series) > 1e-9
    ]
    print(f"pairwise |Pearson r| over {len(nodes)} positions, "
          f"{len(live)} non-constant features\n")
    pairs = []
    for i, left in enumerate(live):
        for right in live[i + 1:]:
            r = pearson(values[left], values[right])
            pairs.append((abs(r), r, left, right))
    pairs.sort(reverse=True)
    print("  strongest relationships:")
    for magnitude, r, left, right in pairs[:12]:
        verdict = ("REDUNDANT" if magnitude > 0.90
                   else "overlapping" if magnitude > 0.70 else "related")
        print(f"    {left:<22} {right:<22} r={r:>6.3f}  {verdict}")
    print("\n  the specific pairs the phase brief asks about:")
    for left, right in (
        ("material_raw", "own_hand_size"),
        ("trump_count", "trump_quality"),
        ("turn_gated_moves", "defense_flexibility"),
        ("talon_size", "refill_deficit"),
        ("open_obligation", "defense_flexibility"),
        ("material_raw", "material_persistent"),
    ):
        if left in values and right in values:
            r = pearson(values[left], values[right])
            print(f"    {left:<22} vs {right:<22} r={r:>6.3f}")


def oracle(rows: list[dict], seconds: float = 200.0) -> None:
    """Section 9 -- does a feature, alone, rank moves like the oracle?

    Each feature is used as a one-term evaluator at the root: score every
    legal move by the feature value of the position it leads to, take the
    argmax, and compare with the Phase 7.3.8 exact solver. This is a
    structural sanity check on a small, atypical slice, never a ranking.
    """
    import time

    sys.setrecursionlimit(60_000)
    from talon_reference import solve

    start = time.time()
    solved = 0
    hits = {feature.name: 0 for feature in FEATURES}
    for row in sorted(rows, key=lambda r: r["card_count"]):
        if time.time() - start > seconds:
            break
        if row["legal_move_count"] < 2:
            continue
        state, player, _, _ = node_of(row)
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
        for feature in FEATURES:
            scored = {}
            for move in truth:
                child = apply_move(state, move)
                information = SearchInformation.from_view(observe(child, player))
                world = DeterminizedWorld(
                    player=player,
                    opponent_hand=frozenset(child.hands[1 - player]),
                    talon=tuple(child.talon),
                )
                node = build_position(information, world).evaluation_node(player)
                scored[move] = feature.compute(node)
            top = max(scored.values())
            pick = next(m for m in truth if scored[m] == top)
            hits[feature.name] += truth[pick] == best
    print(f"single-feature agreement with the exact solver: "
          f"{solved} solvable positions\n")
    if not solved:
        print("  nothing solvable within budget")
        return
    ordered = sorted(hits.items(), key=lambda kv: -kv[1])
    for name, hit in ordered:
        print(f"  {name:<22} {100 * hit / solved:>6.1f}%")
    print("\n  Phase 7.3.8/7.3.10 established that this domain is almost")
    print("  entirely empty-talon positions, so these numbers are a sanity")
    print("  check on sign and sanity, not a feature ranking.")


def antisymmetry(rows: list[dict]) -> None:
    """Section 12 -- is E(pos, P0) = -E(pos, P1)?

    Measured on fully known positions: both hands and the talon are
    materialised, then the same position is evaluated from each seat. A
    feature built only from the searcher's own hand has no reason to be
    antisymmetric, and the measurement says which ones are.
    """
    from durakfish.ai.evaluation import BaselineEvaluator, MechanisticEvaluator

    errors: dict = {feature.name: [] for feature in FEATURES}
    errors["EVAL_baseline"] = []
    errors["EVAL_mechanistic"] = []
    baseline, mechanistic = BaselineEvaluator(), MechanisticEvaluator()

    for row in rows:
        state = GameState.from_dict(row["state"])
        nodes = {}
        for player in (0, 1):
            information = SearchInformation.from_view(observe(state, player))
            world = DeterminizedWorld(
                player=player,
                opponent_hand=frozenset(state.hands[1 - player]),
                talon=tuple(state.talon),
            )
            nodes[player] = build_position(
                information, world
            ).evaluation_node(player)
        for feature in FEATURES:
            errors[feature.name].append(
                abs(feature.compute(nodes[0]) + feature.compute(nodes[1]))
            )
        errors["EVAL_baseline"].append(
            abs(baseline.evaluate(nodes[0], 0) + baseline.evaluate(nodes[1], 0))
        )
        errors["EVAL_mechanistic"].append(
            abs(mechanistic.evaluate(nodes[0], 0)
                + mechanistic.evaluate(nodes[1], 0))
        )

    print(f"antisymmetry error |E(pos,P0) + E(pos,P1)| over {len(rows)} "
          f"fully known positions\n")
    print(f"  {'quantity':<22} {'mean':>8} {'median':>8} {'p95':>8} "
          f"{'max':>8} {'exact zero':>11}")
    for name, values in errors.items():
        ordered = sorted(values)
        p95 = ordered[max(int(0.95 * len(ordered)) - 1, 0)]
        zero = sum(1 for v in values if v < 1e-9)
        print(f"  {name:<22} {statistics.mean(values):>8.3f} "
              f"{statistics.median(values):>8.3f} {p95:>8.3f} "
              f"{max(values):>8.3f} {100 * zero / len(values):>10.1f}%")


STAGES = {
    "formalize": formalize,
    "distribution": distribution,
    "redundancy": redundancy,
    "oracle": oracle,
    "antisymmetry": antisymmetry,
}


def main() -> None:
    args = sys.argv[1:]
    stage = args[0] if args else "formalize"
    split = "dev"
    if "--split" in args:
        split = args[args.index("--split") + 1]
    rows = load(split)
    print(f"[split={split}, n={len(rows)}]\n")
    STAGES[stage](rows)


if __name__ == "__main__":
    main()
