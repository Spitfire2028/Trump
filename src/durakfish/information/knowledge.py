"""Representing what a player knows.

Three kinds of information, never conflated
-------------------------------------------
The single most important rule of this layer is that these stay apart:

* **Observed** — the player saw it. Their own hand, cards played to the
  table, the discard pile, the face-up trump card, their own draws.
* **Deduced** — it follows logically from public facts plus the rules.
  A card the opponent was seen taking and has not played since *must*
  still be in their hand, because cards leave a hand only by being played.
* **Probable** — possible but not proven. A 0.95 marginal is not a fact,
  and this module refuses to express it as one.

Nothing here returns a probability. Beliefs live in
:mod:`durakfish.information.belief`, are computed *from* a
:class:`KnowledgeState`, and can never write back into one. That
one-directional flow is what stops a likelihood becoming a certainty.

Certainty levels
----------------
Two related questions, deliberately answered by two different methods:

``certainty(card)`` — how do we know where this card is?

===============  ==========================================================
``KNOWN``        directly observed to be where it is now
``DEDUCED``      proven by public information and the rules
``UNKNOWN``      more than one location remains possible
===============  ==========================================================

``location_certainty(card, location)`` — what do we know about this
specific placement?

===============  ==========================================================
``KNOWN``        observed to be there
``DEDUCED``      proven to be there
``POSSIBLE``     consistent with everything known; not proven
``IMPOSSIBLE``   ruled out
===============  ==========================================================

``UNKNOWN`` is a statement about a *card*; ``POSSIBLE`` and ``IMPOSSIBLE``
are statements about a *placement*. A probability of zero and "not
currently estimated" are therefore distinguishable: the first is
``IMPOSSIBLE``, the second cannot arise, because every card has an
explicit placement status.

The compact representation
--------------------------
Every card is either **settled** — its location is established, with a
record of whether by observation or deduction — or a **candidate**, in
which case it is in the opponent's hand or the talon and the state says
which counts remain. This is exact rather than lossy: the observer has no
card-specific information distinguishing one unobserved card from another,
so "possible in both hidden locations" is the complete truth about every
candidate. The invariant

    len(candidates) == opponent_slots + talon_slots

is checked on construction and follows from card conservation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any

from durakfish.cards import Card
from durakfish.exceptions import DurakFishError

__all__ = [
    "HIDDEN_LOCATIONS",
    "PUBLIC_LOCATIONS",
    "CardLocation",
    "Certainty",
    "KnowledgeError",
    "KnowledgeState",
    "Placement",
]


class KnowledgeError(DurakFishError):
    """A knowledge state violates an invariant, or was asked something absurd."""


class CardLocation(Enum):
    """Where a card can be.

    ``TABLE`` and ``DISCARD`` are public; ``SELF_HAND`` is the observer's
    own. Only ``OPPONENT_HAND`` and ``TALON`` are hidden, and keeping them
    distinct matters: an unobserved card is *not* simply "an opponent
    card", and treating it as one is the mistake this whole layer exists
    to avoid.
    """

    SELF_HAND = "self_hand"
    OPPONENT_HAND = "opponent_hand"
    TALON = "talon"
    TABLE = "table"
    DISCARD = "discard"


#: The two places a card can hide. Everything else is visible.
HIDDEN_LOCATIONS: frozenset[CardLocation] = frozenset(
    {CardLocation.OPPONENT_HAND, CardLocation.TALON}
)
PUBLIC_LOCATIONS: frozenset[CardLocation] = frozenset(
    {CardLocation.SELF_HAND, CardLocation.TABLE, CardLocation.DISCARD}
)


class Certainty(IntEnum):
    """How firmly something is established. Ordered from least to most."""

    IMPOSSIBLE = 0
    UNKNOWN = 1
    POSSIBLE = 2
    DEDUCED = 3
    KNOWN = 4


@dataclass(frozen=True, slots=True)
class Placement:
    """A settled card: where it is, and how we came to know.

    ``rule`` names the deduction that established it, or ``"observed"``.
    Keeping the justification attached is what makes every DEDUCED fact
    explainable rather than merely asserted.
    """

    location: CardLocation
    certainty: Certainty
    rule: str

    def __post_init__(self) -> None:
        if self.certainty not in (Certainty.KNOWN, Certainty.DEDUCED):
            raise KnowledgeError(
                f"a settled placement is KNOWN or DEDUCED, not {self.certainty.name}"
            )


@dataclass(frozen=True, slots=True)
class KnowledgeState:
    """What one player knows about where every card is. Immutable.

    Attributes:
        player: whose knowledge this is. Knowledge is observer-specific;
            there is deliberately no omniscient tracker.
        settled: cards whose location is established, with justification.
        candidates: cards that could be in either hidden location. Every
            candidate is ``POSSIBLE`` in both and ``IMPOSSIBLE`` nowhere
            else.
        opponent_slots: how many candidates are in the opponent's hand.
        talon_slots: how many are in the talon.
        deck: every card in play, so conservation can be checked.
    """

    player: int
    settled: Mapping[Card, Placement]
    candidates: frozenset[Card]
    opponent_slots: int
    talon_slots: int
    deck: frozenset[Card]

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Check every invariant this layer promises.

        Raises:
            KnowledgeError: on any violation. Reaching this is a bug in
                the deduction engine, never bad input from a player.
        """
        if self.opponent_slots < 0 or self.talon_slots < 0:
            raise KnowledgeError(
                f"negative hidden slots: opponent={self.opponent_slots}, "
                f"talon={self.talon_slots}"
            )
        overlap = set(self.settled) & self.candidates
        if overlap:
            raise KnowledgeError(
                "cards both settled and undetermined: "
                + str(sorted(c.short for c in overlap))
            )
        covered = set(self.settled) | self.candidates
        if covered != set(self.deck):
            missing = set(self.deck) - covered
            extra = covered - set(self.deck)
            raise KnowledgeError(
                "card conservation broken: missing "
                + f"{sorted(c.short for c in missing)}, "
                + f"unexpected {sorted(c.short for c in extra)}"
            )
        if len(self.candidates) != self.opponent_slots + self.talon_slots:
            raise KnowledgeError(
                f"{len(self.candidates)} undetermined cards cannot fill "
                f"{self.opponent_slots} opponent and {self.talon_slots} talon slots"
            )

    # ------------------------------------------------------------------
    # Querying a card
    # ------------------------------------------------------------------
    def location_of(self, card: Card) -> CardLocation | None:
        """Where the card is, or ``None`` if that is not established."""
        placement = self.settled.get(card)
        return placement.location if placement else None

    def certainty(self, card: Card) -> Certainty:
        """How the card's location is known: KNOWN, DEDUCED or UNKNOWN."""
        placement = self.settled.get(card)
        if placement is not None:
            return placement.certainty
        if card in self.candidates:
            return Certainty.UNKNOWN
        raise KnowledgeError(f"{card} is not part of this deck")

    def justification(self, card: Card) -> str | None:
        """Which rule established this card's location, if any."""
        placement = self.settled.get(card)
        return placement.rule if placement else None

    def possible_locations(self, card: Card) -> frozenset[CardLocation]:
        """Every location the card could be in, given what is known."""
        placement = self.settled.get(card)
        if placement is not None:
            return frozenset({placement.location})
        if card not in self.candidates:
            raise KnowledgeError(f"{card} is not part of this deck")
        possible = set()
        if self.opponent_slots:
            possible.add(CardLocation.OPPONENT_HAND)
        if self.talon_slots:
            possible.add(CardLocation.TALON)
        return frozenset(possible)

    def location_certainty(self, card: Card, location: CardLocation) -> Certainty:
        """What is known about this specific card being in this location."""
        placement = self.settled.get(card)
        if placement is not None:
            return (
                placement.certainty
                if placement.location is location
                else Certainty.IMPOSSIBLE
            )
        if location in self.possible_locations(card):
            return Certainty.POSSIBLE
        return Certainty.IMPOSSIBLE

    def is_possible(self, card: Card, location: CardLocation) -> bool:
        return self.location_certainty(card, location) >= Certainty.POSSIBLE

    def is_impossible(self, card: Card, location: CardLocation) -> bool:
        return self.location_certainty(card, location) is Certainty.IMPOSSIBLE

    # ------------------------------------------------------------------
    # Querying a location
    # ------------------------------------------------------------------
    def cards_in(self, location: CardLocation) -> frozenset[Card]:
        """Cards established to be in ``location``, observed or deduced.

        Never speculative: a card that is merely likely to be there is not
        included, however likely.
        """
        return frozenset(
            card
            for card, placement in self.settled.items()
            if placement.location is location
        )

    def known_cards(self) -> frozenset[Card]:
        """Cards whose location was directly observed."""
        return frozenset(
            c for c, p in self.settled.items() if p.certainty is Certainty.KNOWN
        )

    def deduced_cards(self) -> frozenset[Card]:
        """Cards whose location was proven rather than seen."""
        return frozenset(
            c for c, p in self.settled.items() if p.certainty is Certainty.DEDUCED
        )

    @property
    def hidden_card_count(self) -> int:
        """How many cards are in a hidden location, settled or not.

        A **count**, not a collection. Named to match
        :attr:`~durakfish.information.InformationSet.hidden_card_count`
        and, deliberately, not to look like a way of getting at hidden
        cards — ``hidden_cards`` is on the forbidden-attribute list that
        the Phase 7.1 contract audit enforces, and a legitimate accessor
        should not collide with it.
        """
        return (
            len(self.cards_in(CardLocation.OPPONENT_HAND))
            + len(self.cards_in(CardLocation.TALON))
            + len(self.candidates)
        )

    @property
    def is_complete(self) -> bool:
        """True when nothing is left undetermined."""
        return not self.candidates

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    def key(self) -> tuple[Any, ...]:
        """Canonical, process-stable key.

        Built from sorted card codes and enum *values*, never from set or
        dict iteration order, so two semantically identical states hash
        identically in any process.
        """
        return (
            self.player,
            tuple(
                sorted(
                    (card.code, placement.location.value, int(placement.certainty))
                    for card, placement in self.settled.items()
                )
            ),
            tuple(sorted(card.code for card in self.candidates)),
            self.opponent_slots,
            self.talon_slots,
        )

    def to_dict(self) -> dict[str, Any]:
        """Deterministic JSON-friendly form.

        Contains only what the observer legitimately knows, so a
        checkpoint can never smuggle hidden state.
        """
        return {
            "player": self.player,
            "settled": [
                {
                    "card": card.short,
                    "location": placement.location.value,
                    "certainty": placement.certainty.name,
                    "rule": placement.rule,
                }
                for card, placement in sorted(
                    self.settled.items(), key=lambda item: item[0].code
                )
            ],
            "candidates": sorted(c.short for c in self.candidates),
            "opponent_slots": self.opponent_slots,
            "talon_slots": self.talon_slots,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], deck: frozenset[Card]) -> KnowledgeState:
        """Rebuild from :meth:`to_dict`."""
        settled = {
            Card.parse(entry["card"]): Placement(
                CardLocation(entry["location"]),
                Certainty[entry["certainty"]],
                entry["rule"],
            )
            for entry in data["settled"]
        }
        return cls(
            player=int(data["player"]),
            settled=settled,
            candidates=frozenset(Card.parse(c) for c in data["candidates"]),
            opponent_slots=int(data["opponent_slots"]),
            talon_slots=int(data["talon_slots"]),
            deck=deck,
        )

    def describe(self) -> str:
        """Human-readable summary for debugging."""
        lines = [
            f"knowledge of P{self.player}: "
            f"{len(self.known_cards())} observed, "
            f"{len(self.deduced_cards())} deduced, "
            f"{len(self.candidates)} undetermined"
        ]
        for location in CardLocation:
            cards = self.cards_in(location)
            if cards:
                shorts = " ".join(
                    sorted((c.short for c in cards), key=len)
                )
                lines.append(f"  {location.value:<14} {len(cards):>2}  {shorts}")
        lines.append(
            f"  undetermined   {len(self.candidates):>2}  "
            f"({self.opponent_slots} in hand, {self.talon_slots} in talon)"
        )
        return "\n".join(lines)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, KnowledgeState):
            return self.key() == other.key()
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.key())

    def __repr__(self) -> str:
        return (
            f"KnowledgeState(P{self.player}, settled={len(self.settled)}, "
            f"undetermined={len(self.candidates)})"
        )
