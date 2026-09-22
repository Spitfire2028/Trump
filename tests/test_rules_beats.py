"""Exhaustive tests for trump-aware card comparison.

``beats()`` is three lines, which is exactly why it deserves the most
thorough test in the project: every probability estimate, every search
node and every self-play game inherits its correctness.

The exhaustive test compares it against a *deliberately different*
reference implementation (explicit branching on trump membership) over
all 36 x 36 x 4 = 5,184 combinations. Two implementations written from
different angles agreeing everywhere is real evidence; one implementation
agreeing with itself is not.
"""

from __future__ import annotations

import itertools

import pytest

from durakfish.cards import Card, Suit
from durakfish.cards.deck import standard_cards
from durakfish.game import beats

DECK = standard_cards(36)


def reference_beats(defending: Card, attacking: Card, trump: Suit) -> bool:
    """Independent implementation, written by cases rather than by suit equality."""
    defending_is_trump = defending.suit is trump
    attacking_is_trump = attacking.suit is trump

    if defending_is_trump and attacking_is_trump:
        return defending.rank > attacking.rank
    if defending_is_trump and not attacking_is_trump:
        return True
    if not defending_is_trump and attacking_is_trump:
        return False
    # Neither is trump.
    if defending.suit is not attacking.suit:
        return False
    return defending.rank > attacking.rank


def test_exhaustive_36x36x4_against_reference() -> None:
    checked = 0
    for trump in Suit:
        for attacking, defending in itertools.product(DECK, DECK):
            assert beats(defending, attacking, trump) == reference_beats(
                defending, attacking, trump
            ), f"{defending} vs {attacking}, trump {trump}"
            checked += 1
    assert checked == 36 * 36 * 4


def test_spec_examples_with_hearts_as_trump() -> None:
    trump = Suit.HEARTS
    seven_s, eight_s = Card.parse("7S"), Card.parse("8S")
    seven_h, eight_h = Card.parse("7H"), Card.parse("8H")

    assert beats(eight_s, seven_s, trump)  # higher card, same suit
    assert beats(seven_h, seven_s, trump)  # any trump beats a non-trump
    assert beats(eight_h, seven_s, trump)
    assert beats(eight_h, seven_h, trump)  # higher trump beats lower trump
    assert not beats(seven_h, eight_h, trump)  # lower trump does not
    assert not beats(Card.parse("AS"), seven_h, trump)  # non-trump never beats trump


def test_no_card_beats_itself() -> None:
    for trump in Suit:
        for card in DECK:
            assert not beats(card, card, trump)


def test_beating_is_antisymmetric() -> None:
    for trump in Suit:
        for a, b in itertools.combinations(DECK, 2):
            assert not (beats(a, b, trump) and beats(b, a, trump))


def test_rank_alone_never_wins_across_incompatible_suits() -> None:
    """A high card of an unrelated, non-trump suit is powerless."""
    trump = Suit.HEARTS
    ace_of_clubs = Card.parse("AC")
    six_of_spades = Card.parse("6S")
    assert not beats(ace_of_clubs, six_of_spades, trump)
    assert not beats(six_of_spades, ace_of_clubs, trump)


def test_every_trump_beats_every_non_trump() -> None:
    for trump in Suit:
        trumps = [c for c in DECK if c.suit is trump]
        others = [c for c in DECK if c.suit is not trump]
        assert all(
            beats(t, o, trump) for t in trumps for o in others
        )
        assert not any(beats(o, t, trump) for t in trumps for o in others)


@pytest.mark.parametrize("trump", list(Suit))
def test_lowest_trump_beats_everything_except_higher_trumps(trump: Suit) -> None:
    lowest_trump = min(
        (c for c in DECK if c.suit is trump), key=lambda c: int(c.rank)
    )
    beaten = [c for c in DECK if beats(lowest_trump, c, trump)]
    # 27 non-trumps, and no trump at all (it is the lowest one).
    assert len(beaten) == 27
    assert all(c.suit is not trump for c in beaten)


def test_beat_counts_for_a_mid_ranking_card() -> None:
    """9♠ with hearts trump is beaten by 5 higher spades and all 9 hearts."""
    trump = Suit.HEARTS
    attacking = Card.parse("9S")
    beaters = [c for c in DECK if beats(c, attacking, trump)]
    higher_spades = {c.short for c in beaters if c.suit is Suit.SPADES}
    hearts = {c.short for c in beaters if c.suit is Suit.HEARTS}
    assert higher_spades == {"10S", "JS", "QS", "KS", "AS"}
    assert len(hearts) == 9
    assert len(beaters) == 14
