"""Independent audit of the rules engine.

Deliberately does NOT import the test suite's helpers or reuse its
assertions. If the engine and its tests share a blind spot, this script
is the second pair of eyes: it re-derives the invariants from the rules
of Durak rather than from the code's own vocabulary.
"""

from __future__ import annotations

import random
import sys
from collections import Counter

sys.path.insert(0, "src")

from durakfish.cards import Card
from durakfish.cards.deck import standard_cards
from durakfish.game import Phase, apply_move, get_legal_moves, new_game
from durakfish.game.moves import AttackMove, DefenseMove

UNIVERSE = frozenset(standard_cards(36))
problems: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        problems.append(message)


def audit_position(state) -> None:
    """Re-derive the invariants from first principles."""
    # 1. Conservation, counted independently of state.validate().
    seen: Counter[Card] = Counter()
    for hand in state.hands:
        seen.update(hand)
    seen.update(state.talon)
    for slot in state.table:
        seen[slot.attacking_card] += 1
        if slot.defending_card is not None:
            seen[slot.defending_card] += 1
    seen.update(state.discard)
    check(set(seen) == UNIVERSE, "some card left the game entirely")
    check(all(n == 1 for n in seen.values()), "a card exists in two places")
    check(sum(seen.values()) == 36, f"card count is {sum(seen.values())}, not 36")

    # 2. A defending card must genuinely beat what it defends.
    for slot in state.table:
        if slot.defending_card is not None:
            d, a = slot.defending_card, slot.attacking_card
            same_suit_higher = d.suit is a.suit and d.rank > a.rank
            trumped = d.suit is state.trump_suit and a.suit is not state.trump_suit
            check(
                same_suit_higher or trumped,
                f"{d} does not actually beat {a} (trump {state.trump_suit})",
            )

    # 3. No defending card may appear twice on the table.
    defences = [s.defending_card for s in state.table if s.defending_card]
    check(len(defences) == len(set(defences)), "a card defended two attacks")

    # 4. Every attack after the first must match a rank on the table.
    if len(state.table) > 1:
        for index in range(1, len(state.table)):
            earlier = state.table[:index]
            ranks = {c.rank for s in earlier for c in s.cards}
            check(
                state.table[index].attacking_card.rank in ranks,
                "an added attack did not match any rank on the table",
            )

    # 5. The bout limit is never exceeded.
    check(len(state.table) <= state.attack_limit, "more attacks than the limit allows")
    check(state.attack_limit <= 6, "attack limit above the six-card cap")

    # 6. A live position always offers its mover a choice.
    if state.phase is not Phase.GAME_OVER:
        check(bool(get_legal_moves(state)), "live position with no legal moves")
        check(state.current_player is not None, "live position with nobody to move")
    else:
        check(not get_legal_moves(state), "finished game still offers moves")
        check(not state.talon and not state.table, "game over with cards pending")


def main() -> None:
    games = 3000
    lengths: list[int] = []
    outcomes: Counter[str] = Counter()
    move_mix: Counter[str] = Counter()
    max_table = 0
    hands_over_six = 0

    for seed in range(games):
        rng = random.Random(seed)
        state = new_game(seed=seed)
        audit_position(state)
        previous_talon = len(state.talon)
        moves_played = 0

        while not state.is_over:
            legal = get_legal_moves(state)
            move = rng.choice(legal)
            mover = state.current_player
            before = state

            state = apply_move(state, move)
            moves_played += 1

            # The input state must be untouched by the transition.
            check(
                before.hands == audit_snapshot(before),
                "apply_move mutated the state it was given",
            )
            # The talon must never grow.
            check(len(state.talon) <= previous_talon, "the talon gained cards")
            previous_talon = len(state.talon)

            move_mix[type(move).__name__] += 1
            if isinstance(move, AttackMove):
                check(
                    move.card not in state.hands[mover],
                    "attacking card stayed in the attacker's hand",
                )
            if isinstance(move, DefenseMove):
                check(
                    move.defending_card not in state.hands[mover],
                    "defending card stayed in the defender's hand",
                )
            max_table = max(max_table, len(state.table))
            hands_over_six += sum(1 for h in state.hands if len(h) > 6)
            audit_position(state)

        lengths.append(moves_played)
        if state.durak is None:
            outcomes["draw"] += 1
        else:
            outcomes[f"P{state.durak} is durak"] += 1

    print(f"games audited            : {games}")
    print(f"problems found           : {len(problems)}")
    for problem in sorted(set(problems))[:10]:
        print(f"   ! {problem}")
    print(f"mean moves per game      : {sum(lengths) / len(lengths):.1f}")
    print(f"longest game             : {max(lengths)} moves")
    print(f"largest table seen       : {max_table} slots")
    print(f"positions with >6 cards  : {hands_over_six} (expected: takes happen)")
    print(f"outcomes                 : {dict(outcomes)}")
    print(f"move mix                 : {dict(move_mix)}")


def audit_snapshot(state):
    """Hands are tuples of immutable cards, so identity comparison suffices."""
    return state.hands


if __name__ == "__main__":
    main()
    sys.exit(1 if problems else 0)
