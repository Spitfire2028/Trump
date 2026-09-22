"""Tests for the card vocabulary: Suit, Rank, Card."""

from __future__ import annotations

import dataclasses
import pickle

import pytest

from durakfish.cards import ALL_CARDS, CARD_SPACE_SIZE, Card, Rank, Suit
from durakfish.cards.card import format_cards, parse_cards, sort_cards
from durakfish.exceptions import CardParseError, InvalidCardError

DURAK_RANKS = [r for r in Rank if r >= Rank.SIX]


# ----------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------
def test_four_distinct_suits() -> None:
    assert len(Suit) == 4
    assert len({s.letter for s in Suit}) == 4
    assert len({s.symbol for s in Suit}) == 4


def test_thirteen_ranks_nine_of_them_used_by_durak() -> None:
    assert len(Rank) == 13
    assert len(DURAK_RANKS) == 9
    assert DURAK_RANKS[0] is Rank.SIX and DURAK_RANKS[-1] is Rank.ACE


def test_rank_values_express_natural_strength() -> None:
    assert Rank.SIX < Rank.TEN < Rank.JACK < Rank.QUEEN < Rank.KING < Rank.ACE
    assert int(Rank.TEN) == 10


def test_suit_order_is_canonical_not_strength() -> None:
    # Suits are ordered for deterministic display/serialization only.
    assert [s.letter for s in Suit] == ["C", "D", "H", "S"]


def test_suit_parse_accepts_letters_symbols_and_case() -> None:
    for suit in Suit:
        assert Suit.parse(suit.letter) is suit
        assert Suit.parse(suit.letter.lower()) is suit
        assert Suit.parse(suit.symbol) is suit
        assert Suit.parse(f"  {suit.letter}  ") is suit


def test_rank_parse_accepts_labels_and_ten_shorthand() -> None:
    for rank in Rank:
        assert Rank.parse(rank.label) is rank
        assert Rank.parse(rank.label.lower()) is rank
    assert Rank.parse("T") is Rank.TEN
    assert Rank.parse("t") is Rank.TEN


@pytest.mark.parametrize("token", ["", " ", "X", "1", "11", "0", "?", "10H"])
def test_rank_parse_rejects_garbage(token: str) -> None:
    with pytest.raises(CardParseError):
        Rank.parse(token)


@pytest.mark.parametrize("token", ["", " ", "X", "SS", "1", "♤"])
def test_suit_parse_rejects_garbage(token: str) -> None:
    with pytest.raises(CardParseError):
        Suit.parse(token)


# ----------------------------------------------------------------------
# Card space
# ----------------------------------------------------------------------
def test_card_space_is_complete_and_unique() -> None:
    assert CARD_SPACE_SIZE == 52
    assert len(ALL_CARDS) == 52
    assert len(set(ALL_CARDS)) == 52
    assert {(c.rank, c.suit) for c in ALL_CARDS} == {
        (r, s) for r in Rank for s in Suit
    }


def test_codes_are_dense_and_round_trip() -> None:
    codes = [card.code for card in ALL_CARDS]
    assert sorted(codes) == list(range(CARD_SPACE_SIZE))
    for card in ALL_CARDS:
        assert Card.from_code(card.code) is card


@pytest.mark.parametrize("code", [-1, 52, 1000])
def test_from_code_rejects_out_of_range(code: int) -> None:
    with pytest.raises(InvalidCardError):
        Card.from_code(code)


def test_from_code_rejects_non_int() -> None:
    with pytest.raises(InvalidCardError):
        Card.from_code("5")  # type: ignore[arg-type]


def test_of_returns_interned_instances() -> None:
    assert Card.of(Rank.ACE, Suit.SPADES) is Card.of(Rank.ACE, Suit.SPADES)


def test_constructor_rejects_raw_ints() -> None:
    with pytest.raises(InvalidCardError):
        Card(6, Suit.SPADES)  # type: ignore[arg-type]
    with pytest.raises(InvalidCardError):
        Card(Rank.SIX, 0)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Notation
# ----------------------------------------------------------------------
def test_short_and_unicode_notation() -> None:
    assert Card.of(Rank.ACE, Suit.SPADES).short == "AS"
    assert str(Card.of(Rank.ACE, Suit.SPADES)) == "A\u2660"
    assert Card.of(Rank.TEN, Suit.HEARTS).short == "10H"
    assert str(Card.of(Rank.TEN, Suit.HEARTS)) == "10\u2665"
    assert str(Card.of(Rank.QUEEN, Suit.DIAMONDS)) == "Q\u2666"
    assert str(Card.of(Rank.SIX, Suit.CLUBS)) == "6\u2663"


def test_repr_is_short_and_parseable_content() -> None:
    assert repr(Card.parse("10H")) == "Card('10H')"


def test_parse_round_trips_every_card_in_both_notations() -> None:
    for card in ALL_CARDS:
        assert Card.parse(card.short) is card
        assert Card.parse(card.unicode) is card
        assert Card.parse(card.short.lower()) is card
        assert Card.parse(f"  {card.short} ") is card


def test_parse_documented_examples() -> None:
    assert Card.parse("AS") == Card.of(Rank.ACE, Suit.SPADES)
    assert Card.parse("10H") == Card.of(Rank.TEN, Suit.HEARTS)
    assert Card.parse("TH") == Card.parse("10H")


@pytest.mark.parametrize(
    "token", ["", "A", "S", "10", "AA", "11H", "1S", "ZZ", "10X", "  ", "A S S"]
)
def test_parse_rejects_invalid_notation(token: str) -> None:
    with pytest.raises(CardParseError):
        Card.parse(token)


def test_parse_rejects_non_string() -> None:
    with pytest.raises(CardParseError):
        Card.parse(7)  # type: ignore[arg-type]


def test_parse_cards_handles_separators_and_iterables() -> None:
    expected = [Card.parse("6S"), Card.parse("7H"), Card.parse("10D")]
    assert parse_cards("6S 7H 10D") == expected
    assert parse_cards("6S, 7H; 10D") == expected
    assert parse_cards(["6S", "7H", "10D"]) == expected
    assert parse_cards("") == []


def test_format_cards() -> None:
    hand = parse_cards("6S 10H")
    assert format_cards(hand) == "6\u2660 10\u2665"
    assert format_cards(hand, unicode=False, separator=",") == "6S,10H"


# ----------------------------------------------------------------------
# Value semantics
# ----------------------------------------------------------------------
def test_equality_is_by_value_not_identity() -> None:
    assert Card(Rank.KING, Suit.CLUBS) == Card.of(Rank.KING, Suit.CLUBS)
    assert Card.parse("KC") != Card.parse("KD")
    assert Card.parse("KC") != "KC"
    assert (Card.parse("KC") == 17) is False


def test_hashable_in_sets_and_dicts() -> None:
    assert len(set(ALL_CARDS)) == 52
    probabilities: dict[Card, float] = {card: 0.0 for card in ALL_CARDS}
    probabilities[Card(Rank.ACE, Suit.SPADES)] = 1.0
    assert probabilities[Card.of(Rank.ACE, Suit.SPADES)] == 1.0
    assert len(probabilities) == 52


def test_hash_equals_code_so_cards_can_index_tables() -> None:
    for card in ALL_CARDS:
        assert hash(card) == card.code


def test_cards_are_immutable() -> None:
    card = Card.parse("AS")
    with pytest.raises(dataclasses.FrozenInstanceError):
        card.rank = Rank.KING  # type: ignore[misc]


def test_cards_survive_pickling() -> None:
    card = Card.parse("10H")
    assert pickle.loads(pickle.dumps(card)) == card


def test_card_has_no_strength_comparison() -> None:
    # "Higher" is meaningless without a trump suit; comparison lives in
    # the rules layer (Phase 2), not on Card.
    with pytest.raises(TypeError):
        _ = Card.parse("6S") < Card.parse("7S")  # type: ignore[operator]


def test_sort_cards_is_display_order_with_trump_last() -> None:
    hand = parse_cards("AS 6H 7S 10H")
    assert format_cards(sort_cards(hand), unicode=False) == "6H 10H 7S AS"
    assert format_cards(sort_cards(hand, trump=Suit.HEARTS), unicode=False) == (
        "7S AS 6H 10H"
    )
