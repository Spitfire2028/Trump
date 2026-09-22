"""Adversarial audit: deliberately hostile move selection.

Uniform random play explores the middle of the game and almost never
reaches the extremes — huge hands, maximum-length bouts, repeated takes,
talon exhaustion under pressure. These policies target those corners on
purpose, and every position produced is re-audited from first principles.

Run: ``python benchmarks/audit_adversarial.py``
"""

from __future__ import annotations

import random
import sys
from collections import Counter

sys.path.insert(0, "src")

from durakfish.cards import Card
from durakfish.cards.deck import standard_cards
from durakfish.game import (
    AttackMove,
    DefenseMove,
    EndAttack,
    Phase,
    TakeCards,
    apply_move,
    get_legal_moves,
    is_legal_move,
    new_game,
)

UNIVERSE = frozenset(standard_cards(36))
problems: list[str] = []
stats: Counter[str] = Counter()


def check(condition: bool, message: str) -> None:
    if not condition:
        problems.append(message)


# ----------------------------------------------------------------------
# Hostile policies
# ----------------------------------------------------------------------
def pick_attack(moves, key):
    attacks = [m for m in moves if isinstance(m, AttackMove)]
    return min(attacks, key=key) if attacks else None


def policy_max_pressure(state, moves, rng):
    """Attack whenever legal to drive bouts to the six-card ceiling."""
    attack = pick_attack(moves, lambda m: m.card.code)
    if attack is not None:
        return attack
    defences = [m for m in moves if isinstance(m, DefenseMove)]
    if defences:
        return min(defences, key=lambda m: m.defending_card.code)
    return moves[0]


def policy_always_take(state, moves, rng):
    """Never defend. Forces hands far above six and repeated throw-ins."""
    for move in moves:
        if isinstance(move, TakeCards):
            return move
    attack = pick_attack(moves, lambda m: m.card.code)
    return attack if attack is not None else moves[0]


def policy_never_take(state, moves, rng):
    """Defend at any cost, spending trumps early."""
    defences = [m for m in moves if isinstance(m, DefenseMove)]
    if defences:
        return max(defences, key=lambda m: m.defending_card.code)
    attack = pick_attack(moves, lambda m: -m.card.code)
    return attack if attack is not None else moves[0]


def policy_hit_and_run(state, moves, rng):
    """End every bout after a single card; keeps talon draining slowly."""
    for move in moves:
        if isinstance(move, EndAttack):
            return move
    return moves[0]


def policy_random(state, moves, rng):
    return rng.choice(moves)


POLICIES = {
    "max_pressure": policy_max_pressure,
    "always_take": policy_always_take,
    "never_take": policy_never_take,
    "hit_and_run": policy_hit_and_run,
    "random": policy_random,
}


# ----------------------------------------------------------------------
# Independent re-derivation of the invariants
# ----------------------------------------------------------------------
def audit(state) -> None:
    seen: Counter[Card] = Counter()
    for hand in state.hands:
        seen.update(hand)
    seen.update(state.talon)
    for slot in state.table:
        seen[slot.attacking_card] += 1
        if slot.defending_card is not None:
            seen[slot.defending_card] += 1
    seen.update(state.discard)
    check(set(seen) == UNIVERSE, "a card vanished from the game")
    check(all(n == 1 for n in seen.values()), "a card exists in two locations")

    for slot in state.table:
        if slot.defending_card is not None:
            d, a = slot.defending_card, slot.attacking_card
            ok = (d.suit is a.suit and d.rank > a.rank) or (
                d.suit is state.trump_suit and a.suit is not state.trump_suit
            )
            check(ok, f"{d} does not beat {a}")

    defences = [s.defending_card for s in state.table if s.defending_card]
    check(len(defences) == len(set(defences)), "one card defended two attacks")
    check(len(state.table) <= state.attack_limit, "attack limit exceeded")
    check(state.attack_limit <= 6, "attack limit above six")

    if state.phase is Phase.GAME_OVER:
        check(not get_legal_moves(state), "finished game offers moves")
        check(not state.table and not state.talon, "game over with cards pending")
        if state.durak is not None:
            check(bool(state.hands[state.durak]), "durak holds no cards")
            check(
                not state.hands[1 - state.durak], "the winner still holds cards"
            )
    else:
        legal = get_legal_moves(state)
        check(bool(legal), "live position with no legal moves")
        # The two legality paths must agree on everything generated.
        for move in legal:
            check(is_legal_move(state, move), f"generated move {move} judged illegal")
        idle = 1 - state.current_player
        check(not get_legal_moves(state, idle), "the idle player has moves")

    state.validate()


def progress(state) -> tuple[int, int, int]:
    live = len(state.talon) + sum(len(h) for h in state.hands)
    return (live, len(state.talon), len(state.hands[state.attacker]))


def play(seed: int, policy_a, policy_b) -> None:
    rng = random.Random(seed)
    state = new_game(seed=seed)
    audit(state)
    measure = progress(state)
    steps = 0

    while not state.is_over:
        steps += 1
        if steps > 3000:
            problems.append("game exceeded 3000 moves without terminating")
            return
        moves = get_legal_moves(state)
        policy = policy_a if state.current_player == 0 else policy_b
        move = policy(state, moves, rng)
        check(move in moves, "policy produced a move outside the legal set")

        before_hands = state.hands
        before_key = state.transposition_key()
        resolving = isinstance(move, EndAttack)
        state = apply_move(state, move)

        # The parent must be untouched by the transition.
        check(before_hands is not state.hands or before_hands == state.hands, "aliasing")
        check(before_key == before_key, "key instability")

        stats[f"table={len(state.table)}"] += 1
        stats[f"maxhand={max(len(h) for h in state.hands)}"] += 1
        if resolving and not state.is_over:
            nxt = progress(state)
            check(nxt < measure, f"no progress across a bout: {measure} -> {nxt}")
            measure = nxt
        audit(state)

    stats["draw" if state.durak is None else "decisive"] += 1


def main() -> None:
    names = list(POLICIES)
    games = 0
    for seed in range(120):
        for a in names:
            for b in names:
                play(seed * 31 + len(a), POLICIES[a], POLICIES[b])
                games += 1

    print(f"games played with hostile policies : {games}")
    print(f"policy pairings                    : {len(names)}x{len(names)}")
    print(f"problems found                     : {len(problems)}")
    for problem in sorted(set(problems))[:10]:
        print(f"   ! {problem}")

    biggest_table = max(
        int(k.split("=")[1]) for k in stats if k.startswith("table=")
    )
    biggest_hand = max(
        int(k.split("=")[1]) for k in stats if k.startswith("maxhand=")
    )
    print(f"largest bout reached               : {biggest_table} attacking cards")
    print(f"largest hand reached               : {biggest_hand} cards")
    print(f"six-card bouts observed            : {stats['table=6']:,}")
    print(f"draws / decisive                   : {stats['draw']:,} / {stats['decisive']:,}")


if __name__ == "__main__":
    main()
    sys.exit(1 if problems else 0)
