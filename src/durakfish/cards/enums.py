"""Card vocabulary: :class:`Suit` and :class:`Rank`.

This module sits at the very bottom of the dependency graph. It imports
nothing from DurakFish except :mod:`durakfish.exceptions`, and it knows
nothing about Durak, trumps, or games.

Two deliberate decisions:

1. ``Rank`` covers the full 2..A range even though 36-card Durak only uses
   6..A. The *deck specification* decides which ranks are in play (see
   :func:`durakfish.cards.deck.standard_cards`), not the rank vocabulary.
   Fixing the vocabulary once avoids re-numbering every serialized card
   code the day a 52-card variant is added.

2. ``Rank`` is an ``IntEnum`` whose value equals the card's natural rank
   strength, so ``Rank.KING > Rank.TEN`` is true and cheap. ``Suit`` is an
   ``IntEnum`` too, but its order is **canonical only** (alphabetical,
   for deterministic sorting and hashing). Suit order carries no strength:
   which suit is strong depends on the trump, which is a *game* concept
   and therefore lives in the rules layer, not here.
"""

from __future__ import annotations

from enum import IntEnum, unique

from durakfish.exceptions import CardParseError

__all__ = ["Rank", "Suit"]


@unique
class Suit(IntEnum):
    """The four suits.

    The integer values define a canonical ordering used for display and
    for deterministic serialization. They do **not** express strength.
    """

    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    @property
    def symbol(self) -> str:
        """Unicode symbol, e.g. ``'♠'``."""
        return _SUIT_SYMBOLS[self]

    @property
    def letter(self) -> str:
        """ASCII letter used in compact notation, e.g. ``'S'``."""
        return _SUIT_LETTERS[self]

    @property
    def is_red(self) -> bool:
        """True for diamonds and hearts. Presentation helper only."""
        return self is Suit.DIAMONDS or self is Suit.HEARTS

    @classmethod
    def parse(cls, token: str) -> Suit:
        """Parse a suit from a letter (``'S'``) or symbol (``'♠'``).

        Case-insensitive and whitespace-tolerant.

        Raises:
            CardParseError: if the token is not a recognised suit.
        """
        key = token.strip().upper()
        try:
            return _SUIT_TOKENS[key]
        except KeyError:
            raise CardParseError(f"unknown suit token: {token!r}") from None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.symbol


@unique
class Rank(IntEnum):
    """Card ranks, valued by natural strength (2 low, Ace high)."""

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def label(self) -> str:
        """Short human label, e.g. ``'10'``, ``'Q'``."""
        return _RANK_LABELS[self]

    @classmethod
    def parse(cls, token: str) -> Rank:
        """Parse a rank from its label. ``'T'`` is accepted for ten.

        Raises:
            CardParseError: if the token is not a recognised rank.
        """
        key = token.strip().upper()
        try:
            return _RANK_TOKENS[key]
        except KeyError:
            raise CardParseError(f"unknown rank token: {token!r}") from None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.label


_SUIT_SYMBOLS: dict[Suit, str] = {
    Suit.CLUBS: "\u2663",
    Suit.DIAMONDS: "\u2666",
    Suit.HEARTS: "\u2665",
    Suit.SPADES: "\u2660",
}

_SUIT_LETTERS: dict[Suit, str] = {
    Suit.CLUBS: "C",
    Suit.DIAMONDS: "D",
    Suit.HEARTS: "H",
    Suit.SPADES: "S",
}

# Accepted input tokens for a suit: ASCII letter and Unicode symbol.
# Cyrillic letters are deliberately NOT accepted: Cyrillic 'К' and Latin
# 'K' are visually identical homoglyphs, which would make parse errors
# invisible in a terminal. Localisation belongs in the interface layer.
_SUIT_TOKENS: dict[str, Suit] = {}
for _suit in Suit:
    _SUIT_TOKENS[_SUIT_LETTERS[_suit]] = _suit
    _SUIT_TOKENS[_SUIT_SYMBOLS[_suit]] = _suit

_RANK_LABELS: dict[Rank, str] = {
    Rank.TWO: "2",
    Rank.THREE: "3",
    Rank.FOUR: "4",
    Rank.FIVE: "5",
    Rank.SIX: "6",
    Rank.SEVEN: "7",
    Rank.EIGHT: "8",
    Rank.NINE: "9",
    Rank.TEN: "10",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
}

_RANK_TOKENS: dict[str, Rank] = {label: rank for rank, label in _RANK_LABELS.items()}
_RANK_TOKENS["T"] = Rank.TEN  # common shorthand in card-game notation
