"""Tests for Deck: composition, determinism, draw order, fairness."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from durakfish.cards import Card, Deck, Rank, Suit
from durakfish.cards.deck import SUPPORTED_DECK_SIZES, standard_cards
from durakfish.exceptions import (
    CardNotInDeckError,
    DeckExhaustedError,
    DuplicateCardError,
    InvalidDeckSizeError,
)

# Chi-square critical value, 35 degrees of freedom, p = 0.001.
# Tests are seeded, so this bound is a bug detector, not a flaky guard.
CHI2_CRITICAL_DF35_P001 = 66.62


# ----------------------------------------------------------------------
# Composition
# ----------------------------------------------------------------------
def test_standard_deck_has_36_unique_cards() -> None:
    deck = Deck.standard()
    assert len(deck) == 36
    assert len(set(deck)) == 36


def test_standard_deck_has_nine_ranks_and_four_suits() -> None:
    deck = Deck.standard()
    assert {card.rank for card in deck} == {r for r in Rank if r >= Rank.SIX}
    assert {card.suit for card in deck} == set(Suit)
    counts = Counter(card.suit for card in deck)
    assert set(counts.values()) == {9}


def test_no_card_below_six_in_a_36_card_deck() -> None:
    assert all(card.rank >= Rank.SIX for card in Deck.standard())


@pytest.mark.parametrize("size", SUPPORTED_DECK_SIZES)
def test_supported_deck_sizes(size: int) -> None:
    deck = Deck.standard(size)
    assert len(deck) == size
    assert len(set(deck)) == size
    assert len({card.suit for card in deck}) == 4


@pytest.mark.parametrize("size", [0, 1, 35, 37, 54])
def test_unsupported_deck_size_rejected(size: int) -> None:
    with pytest.raises(InvalidDeckSizeError):
        Deck.standard(size)


def test_duplicate_cards_rejected_at_construction() -> None:
    with pytest.raises(DuplicateCardError):
        Deck([Card.parse("AS"), Card.parse("AS")])


def test_non_cards_rejected_at_construction() -> None:
    with pytest.raises(TypeError):
        Deck(["AS"])  # type: ignore[list-item]


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------
def test_same_seed_gives_identical_deck() -> None:
    assert Deck.shuffled(seed=42).to_list() == Deck.shuffled(seed=42).to_list()


def test_different_seeds_give_different_decks() -> None:
    assert Deck.shuffled(seed=1).to_list() != Deck.shuffled(seed=2).to_list()


def test_shuffle_is_a_permutation_never_a_replacement() -> None:
    deck = Deck.standard()
    before = set(deck)
    deck.shuffle(random.Random(7))
    assert set(deck) == before
    assert len(deck) == 36


def test_shuffle_requires_an_explicit_generator() -> None:
    # No implicit global randomness anywhere in the engine.
    with pytest.raises(TypeError):
        Deck.standard().shuffle()  # type: ignore[call-arg]


def test_explicit_rng_is_equivalent_to_seed() -> None:
    assert Deck.shuffled(rng=random.Random(99)) == Deck.shuffled(seed=99)


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def test_draw_takes_from_the_top_in_order() -> None:
    deck = Deck.shuffled(seed=5)
    expected = deck.remaining()
    drawn = [deck.draw() for _ in range(36)]
    assert tuple(drawn) == expected
    assert len(deck) == 0


def test_bottom_card_is_drawn_last_and_is_stable() -> None:
    deck = Deck.shuffled(seed=11)
    trump_card = deck.bottom_card
    assert trump_card is not None
    for _ in range(35):
        assert deck.bottom_card is trump_card
        deck.draw()
    assert deck.draw() is trump_card
    assert deck.bottom_card is None
    assert deck.top_card is None


def test_draw_from_empty_deck_raises() -> None:
    deck = Deck([])
    with pytest.raises(DeckExhaustedError):
        deck.draw()


def test_draw_up_to_returns_fewer_when_deck_runs_short() -> None:
    deck = Deck.shuffled(seed=3)
    deck.draw_up_to(33)
    assert len(deck.draw_up_to(6)) == 3
    assert deck.draw_up_to(6) == []


def test_draw_up_to_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        Deck.standard().draw_up_to(-1)


def test_dealing_two_hands_never_duplicates_a_card() -> None:
    deck = Deck.shuffled(seed=123)
    hand_a = deck.draw_up_to(6)
    hand_b = deck.draw_up_to(6)
    all_seen = hand_a + hand_b + list(deck)
    assert len(all_seen) == 36
    assert len(set(all_seen)) == 36
    assert not set(hand_a) & set(hand_b)


# ----------------------------------------------------------------------
# Removal, reset, copying
# ----------------------------------------------------------------------
def test_remove_deletes_a_specific_card() -> None:
    deck = Deck.standard()
    card = Card.parse("AS")
    deck.remove(card)
    assert card not in deck
    assert len(deck) == 35


def test_remove_missing_card_raises() -> None:
    deck = Deck.standard()
    deck.remove(Card.parse("AS"))
    with pytest.raises(CardNotInDeckError):
        deck.remove(Card.parse("AS"))


def test_remove_all_is_atomic() -> None:
    deck = Deck.standard()
    deck.remove(Card.parse("AS"))
    with pytest.raises(CardNotInDeckError):
        deck.remove_all([Card.parse("KS"), Card.parse("AS")])
    assert Card.parse("KS") in deck  # nothing was removed
    assert len(deck) == 35


def test_remove_all_rejects_duplicate_requests() -> None:
    deck = Deck.standard()
    with pytest.raises(DuplicateCardError):
        deck.remove_all([Card.parse("KS"), Card.parse("KS")])


def test_reset_restores_original_cards_and_order() -> None:
    deck = Deck.shuffled(seed=17)
    original = deck.to_list()
    deck.draw_up_to(12)
    deck.remove(deck.remaining()[0])
    deck.reset()
    assert deck.to_list() == original
    assert len(deck) == 36


def test_reset_preserves_the_shuffle_not_new_deck_order() -> None:
    deck = Deck.shuffled(seed=17)
    shuffled_order = deck.to_list()
    deck.draw_up_to(12)
    deck.reset()
    assert deck.to_list() == shuffled_order
    assert deck.to_list() != Deck.standard().to_list()


def test_shuffling_a_partial_deck_snapshots_only_what_remains() -> None:
    deck = Deck.shuffled(seed=4)
    deck.draw_up_to(6)
    deck.shuffle(random.Random(4))
    deck.draw_up_to(5)
    deck.reset()
    assert len(deck) == 30


def test_copy_is_independent() -> None:
    deck = Deck.shuffled(seed=21)
    clone = deck.copy()
    clone.draw()
    assert len(deck) == 36
    assert len(clone) == 35
    clone.reset()
    assert clone == deck


def test_serialization_round_trip() -> None:
    deck = Deck.shuffled(seed=8)
    deck.draw_up_to(6)
    restored = Deck.from_list(deck.to_list())
    assert restored == deck
    assert restored.remaining() == deck.remaining()


def test_container_protocol() -> None:
    deck = Deck.standard()
    assert bool(deck) is True
    assert Card.parse("6C") in deck
    assert Card.parse("2C") not in deck
    assert bool(Deck([])) is False
    assert repr(Deck([])) == "Deck(empty)"
    assert repr(deck).startswith("Deck(36 cards")


# ----------------------------------------------------------------------
# Fairness (spec §76)
# ----------------------------------------------------------------------
def test_dealt_hands_are_approximately_uniform() -> None:
    """Deal many seeded games; every card should reach player 0 equally often.

    Expected count per card = deals * 6 / 36. A chi-square statistic well
    below the p=0.001 critical value means the shuffle is not biased.
    """
    deals = 4000
    hand_size = 6
    rng = random.Random(2024)
    counts: Counter[Card] = Counter()
    deck = Deck.standard()
    for _ in range(deals):
        deck.reset()
        deck.shuffle(rng)
        counts.update(deck.draw_up_to(hand_size))

    assert sum(counts.values()) == deals * hand_size
    assert len(counts) == 36  # every card was dealt at least once

    expected = deals * hand_size / 36
    chi2 = sum((counts[card] - expected) ** 2 / expected for card in standard_cards())
    assert chi2 < CHI2_CRITICAL_DF35_P001, f"chi2={chi2:.2f} suggests a biased shuffle"


def test_trump_suit_is_approximately_uniform() -> None:
    """The face-up bottom card should land on each suit about equally often."""
    trials = 4000
    rng = random.Random(31337)
    counts: Counter[Suit] = Counter()
    deck = Deck.standard()
    for _ in range(trials):
        deck.reset()
        deck.shuffle(rng)
        bottom = deck.bottom_card
        assert bottom is not None
        counts[bottom.suit] += 1

    expected = trials / 4
    chi2 = sum((counts[suit] - expected) ** 2 / expected for suit in Suit)
    assert chi2 < 16.27  # df=3, p=0.001
