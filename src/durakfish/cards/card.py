"""The :class:`Card` value type.

A card is an immutable, hashable value object. Three representations are
supported and round-trip losslessly:

===============  ==================  ==========================
Representation   Accessor            Example
===============  ==================  ==========================
Structured       ``card.rank``       ``Rank.TEN``, ``Suit.HEARTS``
Compact ASCII    ``card.short``      ``"10H"``
Human readable   ``str(card)``       ``"10♥"``
Integer code     ``card.code``       ``36``
===============  ==================  ==========================

Deliberate omissions
--------------------
``Card`` has **no** ``beats()`` method and **no** ``is_trump()`` helper.
Beating another card is a *rule*, and rules depend on the trump suit,
which is game state. Putting it here would invert the dependency
direction and make the card layer un-reusable for other Durak variants.
It lands in ``durakfish.game.rules`` in Phase 2.

``Card`` also defines no ``<`` operator. In Durak "higher" is ambiguous
without a trump, and a silently-wrong ``card_a < card_b`` would be a
nasty bug. Use :func:`sort_cards` for display ordering, or sort by
``card.code`` for a canonical order.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from durakfish.cards.enums import Rank, Suit
from durakfish.exceptions import CardParseError, InvalidCardError

__all__ = [
    "ALL_CARDS",
    "CARD_SPACE_SIZE",
    "Card",
    "format_cards",
    "parse_cards",
    "sort_cards",
]

SUITS_PER_RANK: Final[int] = len(Suit)
_MIN_RANK: Final[int] = int(min(Rank))
CARD_SPACE_SIZE: Final[int] = len(Rank) * len(Suit)
_SEPARATORS: Final[str] = ",;"


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class Card:
    """A single playing card.

    Instances are immutable and hashable, so they can be used directly as
    set members, dict keys, probability-table keys and transposition-table
    components.

    Equality is by value, so a card built with the constructor compares
    equal to the interned instance returned by :meth:`of`. Prefer
    :meth:`of` / :meth:`from_code` / :meth:`parse` in hot paths: they
    return shared instances from a precomputed table instead of
    allocating.
    """

    rank: Rank
    suit: Suit

    def __post_init__(self) -> None:
        # Cards are constructed at most 52 times per process in normal use
        # (everything else reuses interned instances), so validation here
        # is free and catches Card(6, 0) style mistakes immediately.
        if not isinstance(self.rank, Rank):
            raise InvalidCardError(f"rank must be a Rank, got {self.rank!r}")
        if not isinstance(self.suit, Suit):
            raise InvalidCardError(f"suit must be a Suit, got {self.suit!r}")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def of(cls, rank: Rank, suit: Suit) -> Card:
        """Return the interned card for ``rank`` and ``suit``."""
        return ALL_CARDS[(int(rank) - _MIN_RANK) * SUITS_PER_RANK + int(suit)]

    @classmethod
    def from_code(cls, code: int) -> Card:
        """Return the card for a dense integer code in ``0..51``.

        Raises:
            InvalidCardError: if ``code`` is out of range.
        """
        if not isinstance(code, int) or isinstance(code, bool):
            raise InvalidCardError(f"card code must be an int, got {code!r}")
        if not 0 <= code < CARD_SPACE_SIZE:
            raise InvalidCardError(
                f"card code {code} outside 0..{CARD_SPACE_SIZE - 1}"
            )
        return ALL_CARDS[code]

    @classmethod
    def parse(cls, text: str) -> Card:
        """Parse compact notation such as ``"AS"``, ``"10h"`` or ``"Q♦"``.

        Rank comes first, suit last. Case-insensitive; surrounding
        whitespace is ignored; ``"T"`` is accepted for ten.

        Raises:
            CardParseError: if the text is not a single valid card.
        """
        if not isinstance(text, str):
            raise CardParseError(f"expected a string, got {text!r}")
        token = text.strip()
        if len(token) < 2:
            raise CardParseError(f"card notation too short: {text!r}")
        suit = Suit.parse(token[-1])
        rank = Rank.parse(token[:-1])
        return cls.of(rank, suit)

    # ------------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------------
    @property
    def code(self) -> int:
        """Dense integer code in ``0..51``, stable across runs."""
        return (int(self.rank) - _MIN_RANK) * SUITS_PER_RANK + int(self.suit)

    @property
    def short(self) -> str:
        """Compact ASCII notation, e.g. ``"10H"``. Round-trips via :meth:`parse`."""
        return f"{self.rank.label}{self.suit.letter}"

    @property
    def unicode(self) -> str:
        """Human-readable notation, e.g. ``"10♥"``."""
        return f"{self.rank.label}{self.suit.symbol}"

    def __str__(self) -> str:
        return self.unicode

    def __repr__(self) -> str:
        # Debug-friendly and short: engine dumps print thousands of cards.
        return f"Card({self.short!r})"

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Card):
            return self.rank is other.rank and self.suit is other.suit
        return NotImplemented

    def __hash__(self) -> int:
        # Equal to `code`: unique per card, collision-free, and doubles as
        # an array index for future bitboard / lookup-table optimisations.
        return (int(self.rank) - _MIN_RANK) * SUITS_PER_RANK + int(self.suit)


#: Every card in the 52-card space, ordered by :attr:`Card.code`.
ALL_CARDS: Final[tuple[Card, ...]] = tuple(
    Card(rank, suit) for rank in Rank for suit in Suit
)


def parse_cards(text: str | Iterable[str]) -> list[Card]:
    """Parse many cards from ``"6S 7H 10D"`` or from an iterable of tokens.

    Commas and semicolons are treated as whitespace.

    Raises:
        CardParseError: if any token is invalid.
    """
    if isinstance(text, str):
        cleaned = text
        for separator in _SEPARATORS:
            cleaned = cleaned.replace(separator, " ")
        tokens: list[str] = cleaned.split()
    else:
        tokens = [str(token) for token in text]
    return [Card.parse(token) for token in tokens]


def format_cards(
    cards: Iterable[Card], *, unicode: bool = True, separator: str = " "
) -> str:
    """Render a collection of cards as a single string."""
    return separator.join(card.unicode if unicode else card.short for card in cards)


def sort_cards(cards: Iterable[Card], *, trump: Suit | None = None) -> list[Card]:
    """Sort cards for **display**.

    Ascending by rank within a suit; suits in canonical order; the trump
    suit, if given, is placed last. This is a presentation concern only —
    never use the resulting order as a measure of card strength.
    """
    return sorted(
        cards,
        key=lambda card: (
            trump is not None and card.suit is trump,
            int(card.suit),
            int(card.rank),
        ),
    )
