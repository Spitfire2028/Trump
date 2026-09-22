"""Property-based tests (spec §48).

Hypothesis is an optional dev dependency; these tests skip cleanly when it
is absent so that the core engine keeps a zero-dependency test path.
"""

from __future__ import annotations

import random

import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from durakfish.cards import ALL_CARDS, Card, Deck  # noqa: E402

cards = st.sampled_from(ALL_CARDS)
seeds = st.integers(min_value=0, max_value=2**32 - 1)


@given(cards)
def test_notation_round_trips(card: Card) -> None:
    assert Card.parse(card.short) is card
    assert Card.parse(card.unicode) is card
    assert Card.from_code(card.code) is card


@given(st.sets(cards, min_size=0, max_size=52))
def test_a_card_set_survives_a_deck_round_trip(hand: set[Card]) -> None:
    deck = Deck(sorted(hand, key=lambda c: c.code))
    assert set(Deck.from_list(deck.to_list())) == hand


@settings(max_examples=50)
@given(seeds, st.integers(min_value=0, max_value=40))
def test_shuffling_and_drawing_conserves_every_card(seed: int, taken: int) -> None:
    deck = Deck.shuffled(seed=seed)
    drawn = deck.draw_up_to(taken)
    remaining = list(deck)
    everything = drawn + remaining
    assert len(everything) == 36
    assert len(set(everything)) == 36  # no card in two places at once


@settings(max_examples=50)
@given(seeds)
def test_shuffle_preserves_the_multiset(seed: int) -> None:
    deck = Deck.standard()
    before = set(deck)
    deck.shuffle(random.Random(seed))
    assert set(deck) == before
