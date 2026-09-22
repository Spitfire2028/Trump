"""Determinized hidden worlds.

Determinization means committing to one concrete possibility for
everything an observer cannot see, so that machinery which needs a fully
specified position has one to work with. In DurakFish that means fixing:

* which cards are in the opponent's hand;
* which cards are in the talon, and in what order.

A world is **not** a random assignment of unknown cards. It is a member
of ``W(I)`` — the set of hidden states compatible with the observer's
legal information ``I``. Every constraint Phase 6 established has to
hold, or the world describes a game that could not have happened.

What makes a world legal
------------------------
Seven conditions, all enforced by :func:`validate_world`, which is
written independently of the sampler so that it can catch the sampler
being wrong:

1. **Conservation** — every card in exactly one place, none duplicated,
   none missing, none invented.
2. **Sizes** — the opponent's hand and the talon are exactly as large as
   public information says.
3. **Certain ownership** — cards Phase 6 established as the opponent's or
   the talon's stay there.
4. **Impossible placements** — a card Phase 6 rules out of a location is
   never put there.
5. **Public preservation** — the observer's hand, the table and the
   discard are untouched.
6. **Trump position** — the face-up trump card is the *bottom* card of a
   non-empty talon. This is a real structural rule of Durak that
   ``GameState.validate()`` does not check, so a sampler that ignored it
   would produce worlds the engine treats as valid and that could never
   arise. Verified over 22,646 real non-empty talons without exception.
7. **Talon integrity** — no card appears twice in the ordered talon.

Two of these are **defensive rather than load-bearing**, and it is worth
saying so rather than implying every check is doing independent work.
Mutation testing showed each could be deleted without any test failing,
and investigation showed why:

* the **talon size** check is implied by conservation plus the hand-size
  check — with the hand right and every card accounted for, counting
  fixes the talon;
* the **impossible placement** check is implied by the others — every
  card impossible in the opponent's hand is settled somewhere else, so
  public-trespass, the overlap check or certain-ownership rejects it
  first.

They stay because the implications hold only while the surrounding checks
do. If a later phase relaxes one — a variant where the observer cannot
read the discard, say — these would become the checks that catch it, and
discovering that through a malformed world would be far worse than
carrying two redundant conditions.

Why marginal sampling would be wrong
------------------------------------
Given an undetermined card, Phase 6 can say it is in the opponent's hand
with probability ``k/n``. Sampling each card independently from that
marginal is a mistake, and a tempting one, because the cards are
**dependent**: with 4 undetermined cards and 2 opponent slots, each is
theirs with probability 1/2, but two specific cards are both theirs with
probability 1/6, not 1/4. Independent sampling would also routinely
produce hands of the wrong size, since nothing would hold the total to
``k``.

This generator instead samples a **joint allocation**: it draws a
``k``-subset of the undetermined cards uniformly, which is a member of
the same combinatorial space Phase 6's probabilities are defined over.
Every compatible allocation is equally likely by construction.

Why uniform, and nothing cleverer
---------------------------------
Where no information distinguishes two compatible allocations, they get
equal probability. No card-strength weighting, no behavioural inference,
no opponent model. An opponent declining to beat a card is evidence they
lack a beater, but taking is always legal and sometimes correct, so
acting on it means choosing a model of the opponent — and this phase
makes none. Weighted sampling is a later phase's decision, and it needs a
correct uniform sampler underneath it either way.

Reproducibility
---------------
Sampling takes an explicit :class:`random.Random`. Nothing here touches
the global ``random`` module, so a seed reproduces a world sequence
exactly, which is what makes a future search debuggable.

Not yet a game state
--------------------
A :class:`DeterminizedWorld` is pure information-level data: card sets
and an order. It deliberately holds no ``GameState`` and offers no way to
build one. Turning a world into a position that search can step through
is Phase 7.3's job, and doing it here would put a ``GameState``
constructor inside the information layer for no present benefit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from durakfish.cards import Card
from durakfish.exceptions import DurakFishError
from durakfish.information.knowledge import CardLocation
from durakfish.information.search_contract import SearchInformation

__all__ = [
    "DeterminizedWorld",
    "WorldError",
    "WorldGenerator",
    "is_valid_world",
    "validate_world",
]


class WorldError(DurakFishError):
    """A world is not compatible with the information it claims to fit."""


@dataclass(frozen=True, slots=True)
class DeterminizedWorld:
    """One concrete possibility for everything the observer cannot see.

    Attributes:
        player: the observer this world was generated for. A world is
            only meaningful relative to somebody's information.
        opponent_hand: the cards the opponent holds in this world.
        talon: the talon, top to bottom. Order matters because draws take
            from the top, and the face-up trump card is always last.

    Visible information is deliberately absent. The observer's own hand,
    the table and the discard are already known exactly, so copying them
    into every sampled world would waste space and create a second copy
    that could drift from the first.
    """

    player: int
    opponent_hand: frozenset[Card]
    talon: tuple[Card, ...]

    @property
    def talon_cards(self) -> frozenset[Card]:
        return frozenset(self.talon)

    @property
    def hidden_cards(self) -> frozenset[Card]:
        """Every card this world places, across both hidden locations."""
        return self.opponent_hand | self.talon_cards

    def location_of(self, card: Card) -> CardLocation | None:
        """Where this world puts a card, or ``None`` if it is not hidden."""
        if card in self.opponent_hand:
            return CardLocation.OPPONENT_HAND
        if card in self.talon_cards:
            return CardLocation.TALON
        return None

    def key(self) -> tuple[Any, ...]:
        """Canonical, process-stable identity.

        The hand is a set, so it is sorted; the talon is a sequence, so
        its order is preserved. Two worlds that differ only in talon order
        are genuinely different hidden states and hash differently.
        """
        return (
            self.player,
            tuple(sorted(c.code for c in self.opponent_hand)),
            tuple(c.code for c in self.talon),
        )

    def allocation_key(self) -> tuple[int, ...]:
        """Identity of the *allocation* alone, ignoring talon order.

        The unit Phase 6's probability model is defined over, and what
        the uniformity tests count.
        """
        return tuple(sorted(c.code for c in self.opponent_hand))

    def to_dict(self) -> dict[str, Any]:
        return {
            "player": self.player,
            "opponent_hand": sorted(c.short for c in self.opponent_hand),
            "talon": [c.short for c in self.talon],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeterminizedWorld:
        return cls(
            player=int(data["player"]),
            opponent_hand=frozenset(Card.parse(c) for c in data["opponent_hand"]),
            talon=tuple(Card.parse(c) for c in data["talon"]),
        )

    def __repr__(self) -> str:
        return (
            f"DeterminizedWorld(P{self.player}, "
            f"opponent={len(self.opponent_hand)}, talon={len(self.talon)})"
        )


# ----------------------------------------------------------------------
# Validation — written independently of the sampler
# ----------------------------------------------------------------------
def validate_world(world: DeterminizedWorld, information: SearchInformation) -> None:
    """Check a world against every constraint the information implies.

    Deliberately re-derives each condition from the contract rather than
    trusting how the world was produced, so it is capable of failing the
    generator. Used on every sampled world in the test suite.

    Raises:
        WorldError: naming the first violated constraint.
    """
    view = information.view
    knowledge = information.knowledge

    if world.player != information.player:
        raise WorldError(
            f"world generated for P{world.player} checked against "
            f"P{information.player}'s information"
        )

    # 7. Talon integrity, checked first so later set logic is meaningful.
    if len(set(world.talon)) != len(world.talon):
        raise WorldError("the talon contains a duplicated card")

    opponent = world.opponent_hand
    talon = world.talon_cards
    overlap = opponent & talon
    if overlap:
        raise WorldError(
            f"cards in both hidden locations: {sorted(c.short for c in overlap)}"
        )

    # 5. Public information must be untouched.
    public = frozenset(view.hand) | frozenset(view.table_cards) | view.discard
    trespass = (opponent | talon) & public
    if trespass:
        raise WorldError(
            "world places publicly located cards in hiding: "
            f"{sorted(c.short for c in trespass)}"
        )

    # 1. Conservation.
    placed = public | opponent | talon
    deck = knowledge.deck
    if placed != deck:
        missing = deck - placed
        phantom = placed - deck
        raise WorldError(
            f"card conservation broken: missing {sorted(c.short for c in missing)}, "
            f"phantom {sorted(c.short for c in phantom)}"
        )

    # 2. Sizes.
    if len(opponent) != view.opponent_hand_size:
        raise WorldError(
            f"opponent holds {len(opponent)} cards but public information "
            f"says {view.opponent_hand_size}"
        )
    if len(world.talon) != view.talon_size:
        raise WorldError(
            f"talon has {len(world.talon)} cards but public information "
            f"says {view.talon_size}"
        )

    # 3. Certain ownership.
    certain_opponent = knowledge.cards_in(CardLocation.OPPONENT_HAND)
    if not certain_opponent <= opponent:
        lost = certain_opponent - opponent
        raise WorldError(
            "world moved cards known to be the opponent's: "
            f"{sorted(c.short for c in lost)}"
        )
    certain_talon = knowledge.cards_in(CardLocation.TALON)
    if not certain_talon <= talon:
        lost = certain_talon - talon
        raise WorldError(
            f"world moved cards known to be in the talon: "
            f"{sorted(c.short for c in lost)}"
        )

    # 4. Impossible placements.
    for card in opponent:
        if knowledge.is_impossible(card, CardLocation.OPPONENT_HAND):
            raise WorldError(
                f"{card} was placed in the opponent's hand although the "
                "information rules that out"
            )
    for card in talon:
        if knowledge.is_impossible(card, CardLocation.TALON):
            raise WorldError(
                f"{card} was placed in the talon although the information "
                "rules that out"
            )

    # 6. Trump position.
    if world.talon and world.talon[-1] != view.trump_card:
        raise WorldError(
            f"the talon's bottom card is {world.talon[-1]} but the face-up "
            f"trump is {view.trump_card}; the trump card is drawn last"
        )


def is_valid_world(
    world: DeterminizedWorld, information: SearchInformation
) -> bool:
    """Boolean form of :func:`validate_world`, for filtering."""
    try:
        validate_world(world, information)
    except WorldError:
        return False
    return True


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------
class WorldGenerator:
    """Samples worlds uniformly from those compatible with the information.

    Constructed once per position and reused: the expensive part of the
    information pipeline has already happened by the time a generator
    exists, so sampling is cheap.

    The generator consumes a :class:`SearchInformation` and nothing else.
    It has no access to a ``GameState`` and no way to acquire one, so the
    real hidden state is irrelevant to what it produces — a property the
    noninterference test in ``tests/test_worlds.py`` proves by generating
    from two different real worlds and comparing.
    """

    __slots__ = (
        "_candidates",
        "_certain_opponent",
        "_fixed_talon",
        "_information",
        "_opponent_slots",
        "_talon_size",
        "_trump_card",
    )

    def __init__(self, information: SearchInformation) -> None:
        knowledge = information.knowledge
        self._information = information
        # Sorted for determinism: sampling must not depend on set order.
        self._candidates = tuple(
            sorted(knowledge.candidates, key=lambda c: c.code)
        )
        self._opponent_slots = knowledge.opponent_slots
        self._certain_opponent = knowledge.cards_in(CardLocation.OPPONENT_HAND)
        self._fixed_talon = knowledge.cards_in(CardLocation.TALON)
        self._trump_card = information.view.trump_card
        self._talon_size = information.view.talon_size

    @property
    def information(self) -> SearchInformation:
        return self._information

    @property
    def allocations(self) -> int:
        """How many distinct allocations are compatible. Ignores talon order."""
        return self._information.allocations

    @property
    def is_determined(self) -> bool:
        """True when only one world is possible and sampling is a formality."""
        return not self._candidates

    def sample(self, rng: random.Random) -> DeterminizedWorld:
        """Draw one world uniformly from the compatible allocations.

        Args:
            rng: the generator to draw from. Required, and never
                defaulted to the global module, so a seed reproduces a
                sequence exactly.

        Returns:
            A world satisfying every constraint in :func:`validate_world`.
        """
        # A uniformly random k-subset: the joint allocation, never k
        # independent draws from the marginals.
        chosen = rng.sample(self._candidates, self._opponent_slots)
        opponent = self._certain_opponent | frozenset(chosen)

        remaining = [c for c in self._candidates if c not in opponent]
        body = sorted(self._fixed_talon | frozenset(remaining), key=lambda c: c.code)
        if self._talon_size:
            # The trump card is the bottom card and is drawn last; only the
            # cards above it may be permuted.
            body = [c for c in body if c != self._trump_card]
            rng.shuffle(body)
            talon = tuple(body) + (self._trump_card,)
        else:
            talon = ()

        return DeterminizedWorld(
            player=self._information.player,
            opponent_hand=opponent,
            talon=talon,
        )

    def sample_many(
        self, count: int, rng: random.Random
    ) -> tuple[DeterminizedWorld, ...]:
        """Draw ``count`` worlds independently, with replacement.

        Repeats are expected and correct: the sample space may be smaller
        than ``count``, and de-duplicating would distort the distribution.

        Raises:
            ValueError: for a negative count.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        return tuple(self.sample(rng) for _ in range(count))

    def __repr__(self) -> str:
        return (
            f"WorldGenerator(P{self._information.player}, "
            f"undetermined={len(self._candidates)}, "
            f"allocations={self.allocations})"
        )
