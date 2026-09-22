"""Talon-aware persistence and draw order.  (Phase 7.3.15)

Diagnostic, not production. Nothing in the engine imports this, and
nothing here modifies a production source file.

Phase 7.3.14 left one hypothesis standing: M1 treats ``refill_to = 6``
as a constant reference and models neither the talon's actual capacity
nor the attacker-first draw order. The question is **not** whether that
omission exists -- it does -- but whether the constant abstraction is
semantically wrong under the information the evaluator may use.

The information boundary, quoted from source
--------------------------------------------
``information/information_set.py`` line 37:

    The talon's *contents and order* are hidden; only its size is public.

So the three quantities the brief insists on separating are:

============================  ===========================================
talon **size**                PUBLIC -- permitted
talon **composition**         HIDDEN -- forbidden, and absent from
                              ``SearchNode`` by construction
future **draw order**         the card *sequence* is hidden; but *which
                              player draws first* is a rule constant
                              (attacker, then defender) derivable from
                              ``searcher_is_attacker``, which is public
============================  ===========================================

A counts-only talon-aware model is therefore architecturally
constructible. This phase cannot reject the candidate on information
grounds, which means it has to be decided on semantics.

How the models are adjudicated
------------------------------
Not by shed rate and not by self-play. M1 claims to estimate the
material that survives replenishment, so each model is scored on how
accurately it predicts **the hand sizes the real game actually produced
at the next bout resolution**. That is the quantity the model claims to
estimate, measured on reachable play.

Run: ``python benchmarks/talon_aware_audit.py <stage>`` with stage in
``boundary``, ``models``, ``exceptions``, ``draworder``, ``scope``,
``accuracy``, ``incentive``, ``interaction``.
"""

from __future__ import annotations

import statistics
import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from durakfish.ai.evaluation import MechanisticEvaluator, StructuralWeights
from durakfish.ai.searchnode import SearchNode
from durakfish.game import apply_move, get_legal_moves, new_game


# ======================================================================
# Section 2 / 6 -- the competing semantic models, stated as formulas
# ======================================================================
def needed(cards: int, refill_to: int) -> int:
    """Cards this player would draw if the talon were unlimited."""
    return max(0, refill_to - cards)


def model_a(node: SearchNode) -> float:
    """Model A -- the current production M1, restated for comparison.

    ``talon==0 -> opp-own; else max(0,opp-refill) - max(0,own-refill)``
    """
    if node.talon_size == 0:
        return float(node.opponent_cards - node.own_cards)
    refill = node.refill_to
    return float(
        max(0, node.opponent_cards - refill) - max(0, node.own_cards - refill)
    )


def shares(node: SearchNode, attacker_first: bool) -> tuple[int, int]:
    """How many cards each side can actually draw, from counts alone.

    Uses only ``own_cards``, ``opponent_cards``, ``talon_size``,
    ``refill_to`` and ``searcher_is_attacker`` -- every one public or
    OWN_PRIVATE. No card identity and no talon order is consulted, so
    this is constructible inside the Phase 7 boundary.
    """
    own_need = needed(node.own_cards, node.refill_to)
    opp_need = needed(node.opponent_cards, node.refill_to)
    talon = node.talon_size
    if not attacker_first:
        # Model B: no priority. Split the shortfall proportionally, which
        # is the neutral reading when order is ignored.
        total = own_need + opp_need
        if total == 0 or talon >= total:
            return own_need, opp_need
        own_share = min(own_need, talon * own_need // total)
        return own_share, min(opp_need, talon - own_share)
    if node.searcher_is_attacker:
        own_share = min(own_need, talon)
        return own_share, min(opp_need, talon - own_share)
    opp_share = min(opp_need, talon)
    return min(own_need, talon - opp_share), opp_share


def projected(node: SearchNode, attacker_first: bool) -> tuple[int, int]:
    """Hand sizes after the next refill, predicted from counts."""
    own_share, opp_share = shares(node, attacker_first)
    return node.own_cards + own_share, node.opponent_cards + opp_share


def model_b(node: SearchNode) -> float:
    """Model B -- talon-aware, no draw priority (proportional split)."""
    own, opp = projected(node, attacker_first=False)
    return float(opp - own)


def model_c(node: SearchNode) -> float:
    """Model C -- talon-aware **and** attacker-first.

    ``post_x = x_cards + min(needed(x), share_x)`` with the attacker's
    share taken first, then ``post_opp - post_own``.

    This is not an ad-hoc third formula: it **reduces to the production
    M1 in both of that formula's branches.** With an ample talon every
    hand reaches ``refill_to`` and the difference becomes
    ``max(0,opp-refill) - max(0,own-refill)``; with an empty talon no one
    draws and it becomes ``opp - own``. Model C is the single expression
    that interpolates between the two branches the current M1 switches
    between.
    """
    own, opp = projected(node, attacker_first=True)
    return float(opp - own)


MODELS = {"A_current": model_a, "B_talon": model_b, "C_attacker_first": model_c}


# ======================================================================
def walk_states(seed: int):
    import random as _random

    rng = _random.Random(seed)
    state = new_game(seed=seed)
    while not state.is_over:
        yield state
        state = apply_move(state, rng.choice(get_legal_moves(state)))


def positions(seeds):
    """Every reachable decision position, with its evaluation node."""
    from benchmarks.incentive_repair import node_for

    for seed in seeds:
        states = list(walk_states(seed))
        for index, state in enumerate(states):
            player = state.current_player
            if player is None:
                continue
            yield {
                "seed": seed,
                "index": index,
                "states": states,
                "state": state,
                "player": player,
                "node": node_for(state, player),
            }


def shedding(seeds):
    """Reachable legal shedding transitions."""
    from benchmarks.incentive_repair import node_for

    for seed in seeds:
        for state in walk_states(seed):
            player = state.current_player
            if player is None:
                continue
            before = node_for(state, player)
            for move in get_legal_moves(state):
                after_state = apply_move(state, move)
                if after_state.current_player is None:
                    continue
                if len(after_state.hands[player]) >= len(state.hands[player]):
                    continue
                yield before, node_for(after_state, player), state, player


# ======================================================================
def boundary(_args: list[str]) -> None:
    """Section 1/3 -- what M1 may legitimately know."""
    import dataclasses

    print("SearchNode surface available to an evaluator:\n")
    for field in dataclasses.fields(SearchNode):
        print(f"  {field.name}")
    print("\nclassification for the talon:\n")
    print("  talon SIZE          PUBLIC      -- node.talon_size, an int")
    print("  talon COMPOSITION   HIDDEN      -- no field holds cards;")
    print("                                     unconstructible, not refused")
    print("  future DRAW ORDER   split:")
    print("     which player draws first  -- rule constant + public seat")
    print("     which CARDS they draw     -- HIDDEN, and never consulted")
    print("\n  source, information_set.py line 37:")
    print("    'The talon's contents and order are hidden; only its size")
    print("     is public.'")
    print("\n  Model C therefore uses only: own_cards, opponent_cards,")
    print("  talon_size, refill_to, searcher_is_attacker. All permitted.")


def models(_args: list[str]) -> None:
    """Section 6/7 -- show Model C reduces to the current M1."""
    current = MechanisticEvaluator()
    print("Model C reduces to production M1 in both of its branches.\n")
    checked = agree_empty = agree_ample = 0
    disagree = []
    for row in positions(range(20)):
        node = row["node"]
        a, c = model_a(node), model_c(node)
        assert a == current.material(node)
        checked += 1
        ample = node.talon_size >= (
            needed(node.own_cards, node.refill_to)
            + needed(node.opponent_cards, node.refill_to)
        )
        if node.talon_size == 0:
            agree_empty += a == c
        elif ample:
            agree_ample += a == c
        elif a != c:
            disagree.append((node, a, c))
    print(f"  positions checked                     {checked}")
    print(f"  empty talon, A == C                   {agree_empty}")
    print(f"  ample talon, A == C                   {agree_ample}")
    print(f"  short talon, A != C                   {len(disagree)}")
    for node, a, c in disagree[:4]:
        own_share, opp_share = shares(node, True)
        print(f"\n    own={node.own_cards} opp={node.opponent_cards} "
              f"talon={node.talon_size} attacker={node.searcher_is_attacker}")
        print(f"      A (constant refill_to) = {a:+.0f}")
        print(f"      C (draws {own_share} / {opp_share}) = {c:+.0f}")


def exceptions(args: list[str]) -> None:
    """Section 4 -- the complete table of talon-short exceptions."""
    lo = int(args[0]) if args else 200
    hi = int(args[1]) if len(args) > 1 else 225
    rows = []
    for seed in range(lo, hi):
        states = list(walk_states(seed))
        for index, state in enumerate(states):
            player = state.current_player
            if player is None or not state.talon:
                continue
            refill_to = state.rules.hand_size
            deficit = refill_to - len(state.hands[player])
            if deficit <= 0:
                continue
            outcome = None
            for later in states[index + 1:]:
                if len(later.hands[player]) >= refill_to:
                    outcome = "restored"
                    break
                if not later.talon:
                    outcome = "unrestored"
                    break
            if (outcome or "unrestored") == "unrestored":
                rows.append({
                    "seed": seed,
                    "player": player,
                    "attacker": state.attacker == player,
                    "hand": len(state.hands[player]),
                    "opp": len(state.hands[1 - player]),
                    "talon": len(state.talon),
                    "needed": deficit,
                })
    print(f"talon-short exceptions, fresh seeds {lo}..{hi - 1}: "
          f"{len(rows)} cases\n")
    if not rows:
        print("  none found")
        return
    print(f"  {'seed':>5} {'seat':<9} {'hand':>5} {'opp':>5} {'talon':>6} "
          f"{'needs':>6} {'shortfall':>10}")
    for row in rows:
        short = max(0, row["needed"] - row["talon"])
        print(f"  {row['seed']:>5} "
              f"{'attacker' if row['attacker'] else 'defender':<9} "
              f"{row['hand']:>5} {row['opp']:>5} {row['talon']:>6} "
              f"{row['needed']:>6} {short:>10}")
    print(f"\n  mean unrecovered deficit  "
          f"{statistics.mean(r['needed'] for r in rows):.2f} cards")
    print(f"  attacker / defender       "
          f"{sum(1 for r in rows if r['attacker'])} / "
          f"{sum(1 for r in rows if not r['attacker'])}")


def draworder(args: list[str]) -> None:
    """Section 5 -- does attacker-first change the semantic quantity?"""
    hi = int(args[0]) if args else 25
    contested = both_short = 0
    changed_by_order = 0
    rows = []
    for row in positions(range(hi)):
        node = row["node"]
        if node.talon_size == 0:
            continue
        own_need = needed(node.own_cards, node.refill_to)
        opp_need = needed(node.opponent_cards, node.refill_to)
        if own_need + opp_need == 0:
            continue
        if node.talon_size >= own_need + opp_need:
            continue
        contested += 1
        if own_need and opp_need:
            both_short += 1
        first = shares(node, attacker_first=True)
        other = shares(node, attacker_first=False)
        if first != other:
            changed_by_order += 1
            rows.append((node, first, other))
    print(f"draw-order audit over reachable positions, seeds 0..{hi - 1}\n")
    print(f"  positions where the talon cannot satisfy both       "
          f"{contested}")
    print(f"  of those, both players actually need cards          "
          f"{both_short}")
    print(f"  where priority changes the allocation               "
          f"{changed_by_order}")
    for node, first, other in rows[:5]:
        print(f"\n    own={node.own_cards} opp={node.opponent_cards} "
              f"talon={node.talon_size} "
              f"searcher_is_attacker={node.searcher_is_attacker}")
        print(f"      attacker-first: own draws {first[0]}, opp draws {first[1]}")
        print(f"      proportional  : own draws {other[0]}, opp draws {other[1]}")


def scope(args: list[str]) -> None:
    """Section 8 -- the denominators, not just the exceptional rate."""
    hi = int(args[0]) if args else 25
    total = below = short = differs = 0
    diffs = []
    by_talon: dict = {}
    by_hand: dict = {}
    by_seat: dict = {}
    for row in positions(range(hi)):
        node = row["node"]
        total += 1
        if node.own_cards < node.refill_to:
            below += 1
        own_need = needed(node.own_cards, node.refill_to)
        opp_need = needed(node.opponent_cards, node.refill_to)
        if node.talon_size and node.talon_size < own_need + opp_need:
            short += 1
        a, c = model_a(node), model_c(node)
        if a != c:
            differs += 1
            diffs.append(abs(a - c))
            key = ("empty" if node.talon_size == 0
                   else f"talon {node.talon_size}")
            by_talon[key] = by_talon.get(key, 0) + 1
            by_hand[node.own_cards] = by_hand.get(node.own_cards, 0) + 1
            seat = "attacker" if node.searcher_is_attacker else "defender"
            by_seat[seat] = by_seat.get(seat, 0) + 1
    print(f"scope over {total} reachable decision positions, "
          f"seeds 0..{hi - 1}\n")
    print(f"  hand below refill_to                     {below:>6} "
          f"({100 * below / total:.1f}%)")
    print(f"  talon too short to satisfy both          {short:>6} "
          f"({100 * short / total:.1f}%)")
    print(f"  Model A differs from Model C             {differs:>6} "
          f"({100 * differs / total:.1f}%)")
    if diffs:
        print(f"  mean |A - C| where they differ           "
              f"{statistics.mean(diffs):>6.2f}")
        print(f"  max  |A - C|                             {max(diffs):>6.0f}")
        print(f"\n  by talon size: {dict(sorted(by_talon.items()))}")
        print(f"  by own hand:   {dict(sorted(by_hand.items()))}")
        print(f"  by seat:       {by_seat}")

    shed_total = shed_differ = 0
    for before, after, _state, _player in shedding(range(min(hi, 20))):
        shed_total += 1
        if (model_a(after) - model_a(before)) != (
            model_c(after) - model_c(before)
        ):
            shed_differ += 1
    print(f"\n  shedding transitions                     {shed_total:>6}")
    print(f"  where the A and C deltas differ          {shed_differ:>6} "
          f"({100 * shed_differ / max(shed_total, 1):.1f}%)")


def accuracy(args: list[str]) -> None:
    """Section 9 (semantic) -- which model predicts reality better?

    M1 claims to estimate material surviving replenishment. So each
    model's prediction of the post-refill hand-size difference is scored
    against **what the real game actually produced** at the next bout
    resolution. Reachable play, no self-play, no shed rate.
    """
    hi = int(args[0]) if args else 25
    errors: dict = {name: [] for name in MODELS}
    scored = 0
    for row in positions(range(hi)):
        node, states, index = row["node"], row["states"], row["index"]
        player = row["player"]
        refill_to = row["state"].rules.hand_size
        # The real outcome: hand sizes at the next moment a refill has
        # happened, i.e. the next state whose talon is smaller.
        truth = None
        for later in states[index + 1:]:
            if len(later.talon) < len(row["state"].talon):
                truth = (
                    len(later.hands[1 - player]) - len(later.hands[player])
                )
                break
        if truth is None:
            continue
        scored += 1
        for name, model in MODELS.items():
            errors[name].append(abs(model(node) - truth))
        _ = refill_to
    print(f"predictive accuracy against the real post-refill difference\n"
          f"  {scored} reachable positions followed to an actual refill, "
          f"seeds 0..{hi - 1}\n")
    print(f"  {'model':<18} {'mean |error|':>13} {'median':>9} "
          f"{'exact hits':>12}")
    for name in MODELS:
        values = errors[name]
        exact = 100 * sum(1 for v in values if v == 0) / len(values)
        print(f"  {name:<18} {statistics.mean(values):>13.4f} "
              f"{statistics.median(values):>9.2f} {exact:>11.1f}%")
    print("\n  Lower mean error means the model better represents the")
    print("  quantity M1 claims to estimate. This is the semantic test.")


def incentive(args: list[str]) -> None:
    """Section 11 -- local incentive, partitioned, interpreted later."""
    hi = int(args[0]) if args else 20
    buckets: dict = {}
    for before, after, state, player in shedding(range(hi)):
        trump = state.trump_suit
        played_trump = any(
            c.suit is trump for c in state.hands[player]
        ) and len([c for c in state.hands[player] if c.suit is trump]) > 0
        keys = [
            "empty talon" if before.talon_size == 0
            else "short talon" if before.talon_size < 6 else "normal talon",
            "attacker" if before.searcher_is_attacker else "defender",
            "hand below six" if before.own_cards < 6
            else "hand at six" if before.own_cards == 6 else "hand above six",
        ]
        _ = played_trump
        for key in keys:
            slot = buckets.setdefault(key, {name: [0, 0, 0] for name in MODELS})
            for name, model in MODELS.items():
                delta = model(after) - model(before)
                cell = slot[name]
                if delta > 1e-12:
                    cell[0] += 1
                elif delta < -1e-12:
                    cell[1] += 1
                else:
                    cell[2] += 1
    print(f"local incentive on reachable shedding transitions, "
          f"seeds 0..{hi - 1}\n")
    print(f"  {'partition':<16} {'model':<18} {'rewards':>9} "
          f"{'discourages':>13} {'neutral':>9}")
    for key in sorted(buckets):
        for name in MODELS:
            up, down, flat = buckets[key][name]
            n = up + down + flat
            print(f"  {key:<16} {name:<18} {100 * up / n:>8.1f}% "
                  f"{100 * down / n:>12.1f}% {100 * flat / n:>8.1f}%")
        print()
    print("  Direction is not labelled good or bad here; it is read")
    print("  against each model's own semantic definition.")


def interaction(args: list[str]) -> None:
    """Section 12 -- under the REPAIRED evaluator, never the old one."""
    hi = int(args[0]) if args else 20
    import benchmarks.causal_ladder  # noqa: F401  registers ladder cells
    from durakfish.ai.evaluation import get_evaluator

    repaired = get_evaluator("lad_e3_both")
    weights = StructuralWeights()
    rows = list(shedding(range(hi)))
    print(f"interaction under the repaired evaluator (seat-gated "
          f"obligation)\n  {len(rows)} reachable shedding transitions\n")
    print(f"  {'M1 model':<18} {'mean weighted dM1':>18} "
          f"{'total fell':>12} {'mean total':>12}")
    for name, model in MODELS.items():
        deltas = [
            weights.material * (model(after) - model(before))
            for before, after, _s, _p in rows
        ]
        # Substitute the model's material into the repaired evaluator by
        # adding the difference from the production term.
        totals = []
        for (before, after, _s, _p), swapped in zip(rows, deltas, strict=True):
            base = repaired.evaluate(after, 0) - repaired.evaluate(before, 0)
            production = weights.material * (
                model_a(after) - model_a(before)
            )
            totals.append(base - production + swapped)
        fell = 100 * sum(1 for t in totals if t < -1e-9) / len(totals)
        print(f"  {name:<18} {statistics.mean(deltas):>+18.4f} "
              f"{fell:>11.1f}% {statistics.mean(totals):>+12.4f}")


STAGES = {
    "boundary": boundary,
    "models": models,
    "exceptions": exceptions,
    "draworder": draworder,
    "scope": scope,
    "accuracy": accuracy,
    "incentive": incentive,
    "interaction": interaction,
}


def main() -> None:
    args = sys.argv[1:]
    stage = args[0] if args else "boundary"
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; pick from {sorted(STAGES)}")
    STAGES[stage](args[1:])


if __name__ == "__main__":
    main()
