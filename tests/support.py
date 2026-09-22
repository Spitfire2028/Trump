"""Helpers for building exact positions in tests.

``make_state`` lets a test state only the cards it cares about. Every
card not mentioned is placed in the discard pile, so the 36-card
conservation invariant holds and ``validate()`` can run on every
constructed position.
"""

from __future__ import annotations

from durakfish.cards import Card, Suit, parse_cards
from durakfish.cards.deck import standard_cards
from durakfish.game import (
    STANDARD_RULES,
    GameState,
    Phase,
    RuleSet,
    TableSlot,
)
from durakfish.game.state import add_cards


def cards(text: str) -> tuple[Card, ...]:
    """``cards("6S 7H")`` -> tuple of cards, sorted canonically."""
    return add_cards((), parse_cards(text))


def make_state(
    *,
    hand0: str = "",
    hand1: str = "",
    trump: Suit,
    talon: str = "",
    table: tuple[tuple[str, str | None], ...] = (),
    attacker: int = 0,
    phase: Phase = Phase.ATTACK,
    attack_limit: int | None = None,
    durak: int | None = None,
    rules: RuleSet = STANDARD_RULES,
    validate: bool = True,
) -> GameState:
    """Build a valid position from compact card notation.

    Args:
        table: ``(("7S", "9S"), ("QH", None))`` — attack card, then the
            card that beat it or ``None``.
        talon: draw order, top first. The last entry becomes the face-up
            trump card; if the talon is empty a trump-suited card
            elsewhere in play is used instead.
    """
    hands = (cards(hand0), cards(hand1))
    talon_cards = parse_cards(talon)
    slots = tuple(
        TableSlot(
            Card.parse(attack), None if defense is None else Card.parse(defense)
        )
        for attack, defense in table
    )
    table_cards = tuple(card for slot in slots for card in slot.cards)

    accounted = set(hands[0]) | set(hands[1]) | set(talon_cards) | set(table_cards)
    universe = standard_cards(rules.deck_size)
    discard = frozenset(card for card in universe if card not in accounted)

    if talon_cards:
        trump_card = talon_cards[-1]
        assert trump_card.suit is trump, "last talon card must be the trump card"
    else:
        trump_card = next(card for card in universe if card.suit is trump)

    if attack_limit is None:
        defended = sum(1 for slot in slots if slot.is_defended)
        defender_hand = hands[(attacker + 1) % 2]
        attack_limit = min(
            rules.max_attacks_per_bout, len(defender_hand) + defended
        )

    state = GameState(
        hands=hands,
        talon=tuple(talon_cards),
        discard=discard,
        table=slots,
        trump_card=trump_card,
        trump_suit=trump,
        attacker=attacker,
        phase=phase,
        attack_limit=attack_limit,
        durak=durak,
        rules=rules,
    )
    if validate:
        state.validate()
    return state
