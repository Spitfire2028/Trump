"""The :class:`Deck` — an ordered pile of unique cards (the *talon*).

Orientation
-----------
A deck has a **top** (the next card to be drawn) and a **bottom**. In
Durak the bottom card is turned face up and its suit is the trump; it is
the last card anyone draws. ``Deck`` exposes :attr:`bottom_card` as a
neutral fact about the pile — it does not know that this means "trump".
Interpreting it is the rules layer's job (Phase 2).

Modelling note: real Durak turns the trump card up *after* dealing. Here
we shuffle once and deal from the top, so the bottom card of the
remaining talon is the trump. The two procedures are distributionally
identical and this one has no special cases.

Determinism
-----------
:meth:`shuffle` requires an explicit :class:`random.Random`. There is no
implicit use of the global ``random`` module anywhere in the engine, so a
seed always reproduces a game exactly (spec §52).
"""

from __future__ import annotations

import random
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from typing import Final

from durakfish.cards.card import ALL_CARDS, Card, parse_cards
from durakfish.cards.enums import Rank
from durakfish.exceptions import (
    CardNotInDeckError,
    DeckExhaustedError,
    DuplicateCardError,
    InvalidDeckSizeError,
)

__all__ = ["STANDARD_DECK_SIZE", "SUPPORTED_DECK_SIZES", "Deck", "standard_cards"]

STANDARD_DECK_SIZE: Final[int] = 36

# Durak is played with several deck sizes; each is the top-N ranks of the
# full 52-card space. Supporting them costs one dict, which is why it is
# here in Phase 1 rather than retrofitted later.
_LOWEST_RANK_BY_SIZE: Final[dict[int, Rank]] = {
    20: Rank.TEN,
    24: Rank.NINE,
    36: Rank.SIX,
    52: Rank.TWO,
}
SUPPORTED_DECK_SIZES: Final[tuple[int, ...]] = tuple(sorted(_LOWEST_RANK_BY_SIZE))


def standard_cards(size: int = STANDARD_DECK_SIZE) -> tuple[Card, ...]:
    """Return the cards of a standard deck of ``size`` cards, in new-deck order.

    Raises:
        InvalidDeckSizeError: for an unsupported size.
    """
    try:
        lowest = _LOWEST_RANK_BY_SIZE[size]
    except KeyError:
        raise InvalidDeckSizeError(
            f"unsupported deck size {size!r}; supported: {SUPPORTED_DECK_SIZES}"
        ) from None
    return tuple(card for card in ALL_CARDS if card.rank >= lowest)


def _ensure_unique(cards: Sequence[Card]) -> None:
    """Guard the core invariant: a physical card exists in exactly one place."""
    seen: set[Card] = set()
    for card in cards:
        if not isinstance(card, Card):
            raise TypeError(f"deck may only contain Card objects, got {card!r}")
        if card in seen:
            raise DuplicateCardError(f"duplicate card in deck: {card}")
        seen.add(card)


class Deck:
    """An ordered pile of distinct cards.

    Iteration and :meth:`remaining` yield cards from top to bottom, i.e.
    in the order they will be drawn.
    """

    __slots__ = ("_cards", "_initial")

    def __init__(self, cards: Iterable[Card] = (), *, validate: bool = True) -> None:
        """Build a deck from ``cards``, first element = top of the pile.

        Args:
            cards: the cards, in draw order.
            validate: check for duplicates. Only disable this on input
                already known to be unique (internal fast path).
        """
        ordered = tuple(cards)
        if validate:
            _ensure_unique(ordered)
        self._cards: deque[Card] = deque(ordered)
        self._initial: tuple[Card, ...] = ordered

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def standard(cls, size: int = STANDARD_DECK_SIZE) -> Deck:
        """A fresh, unshuffled deck in new-deck order."""
        return cls(standard_cards(size), validate=False)

    @classmethod
    def shuffled(
        cls,
        seed: int | None = None,
        *,
        size: int = STANDARD_DECK_SIZE,
        rng: random.Random | None = None,
    ) -> Deck:
        """A shuffled deck.

        Pass ``seed`` or ``rng`` for reproducibility. With neither, the
        shuffle is seeded from OS entropy and the game is *not*
        reproducible — fine for casual play, never for debugging.
        """
        deck = cls.standard(size)
        deck.shuffle(rng if rng is not None else random.Random(seed))
        return deck

    @classmethod
    def from_list(cls, codes: Iterable[str]) -> Deck:
        """Rebuild a deck from :meth:`to_list` output."""
        return cls(parse_cards(codes))

    def copy(self) -> Deck:
        """An independent copy; mutating the copy never touches the original."""
        clone = Deck.__new__(Deck)
        clone._cards = deque(self._cards)
        clone._initial = self._initial
        return clone

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def shuffle(self, rng: random.Random) -> None:
        """Shuffle in place using an explicitly supplied generator.

        Shuffling re-establishes the snapshot used by :meth:`reset`, so a
        shuffled deck that has been dealt from resets to *the shuffled
        order*, not to new-deck order. Shuffling a partially drawn deck
        therefore snapshots only the cards still in it.
        """
        cards = list(self._cards)
        rng.shuffle(cards)
        self._cards = deque(cards)
        self._initial = tuple(cards)

    def draw(self) -> Card:
        """Remove and return the top card.

        Raises:
            DeckExhaustedError: if the deck is empty.
        """
        try:
            return self._cards.popleft()
        except IndexError:
            raise DeckExhaustedError("cannot draw from an empty deck") from None

    def draw_up_to(self, count: int) -> list[Card]:
        """Draw at most ``count`` cards; returns fewer if the deck runs out.

        This is the operation Durak actually needs: at the end of a bout
        players refill toward six cards, and the talon may not have enough.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        taken = min(count, len(self._cards))
        return [self._cards.popleft() for _ in range(taken)]

    def remove(self, card: Card) -> None:
        """Remove a specific card from anywhere in the deck.

        Used to construct analysis positions, not during normal play.

        Raises:
            CardNotInDeckError: if the card is not present.
        """
        try:
            self._cards.remove(card)
        except ValueError:
            raise CardNotInDeckError(f"card not in deck: {card}") from None

    def remove_all(self, cards: Iterable[Card]) -> None:
        """Remove several cards atomically: either all go, or none do."""
        wanted = list(cards)
        present = set(self._cards)
        missing = [card for card in wanted if card not in present]
        if missing:
            raise CardNotInDeckError(
                "cards not in deck: " + ", ".join(str(card) for card in missing)
            )
        _ensure_unique(wanted)
        for card in wanted:
            self._cards.remove(card)

    def reset(self) -> None:
        """Undo all draws and removals, restoring the last snapshot.

        The snapshot is taken at construction and refreshed by
        :meth:`shuffle`, so ``reset()`` returns a dealt deck to the full
        pile in the order the shuffle produced. Call :meth:`shuffle`
        afterwards for a genuinely new order.
        """
        self._cards = deque(self._initial)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    @property
    def top_card(self) -> Card | None:
        """The next card to be drawn, or ``None`` if empty."""
        return self._cards[0] if self._cards else None

    @property
    def bottom_card(self) -> Card | None:
        """The last card that will be drawn — in Durak, the face-up trump."""
        return self._cards[-1] if self._cards else None

    def remaining(self) -> tuple[Card, ...]:
        """All remaining cards, top to bottom."""
        return tuple(self._cards)

    def to_list(self) -> list[str]:
        """Serialize to compact notation, top to bottom (JSON-friendly)."""
        return [card.short for card in self._cards]

    def __len__(self) -> int:
        return len(self._cards)

    def __bool__(self) -> bool:
        return bool(self._cards)

    def __iter__(self) -> Iterator[Card]:
        return iter(tuple(self._cards))

    def __contains__(self, card: object) -> bool:
        return card in self._cards

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Deck):
            return tuple(self._cards) == tuple(other._cards)
        return NotImplemented

    def __repr__(self) -> str:
        if not self._cards:
            return "Deck(empty)"
        return (
            f"Deck({len(self._cards)} cards, "
            f"top={self.top_card}, bottom={self.bottom_card})"
        )
