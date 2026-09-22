"""The information-aware search contract.

What a future search algorithm is allowed to consume, and nothing more.

Why this exists
---------------
Phase 5's search took an :class:`~durakfish.information.InformationSet`
and reasoned from raw observation alone. It was correct and weak: it
credited the opponent with replies they often could not make, because it
had no way to know better. Phase 6 built the knowledge that would let it
know better. This module is the connector — the single object a
belief-aware search will be handed.

Bundling matters more than it looks. Without it, each future search would
assemble its own inputs, and the day one of them reached for a
``GameState`` "just to get the talon size", nothing would catch it. With
it, there is exactly one door into the search layer and one place to
audit.

Why it is not a new information model
-------------------------------------
It deliberately adds no new representation. Everything here already
exists and was audited in Phase 6:

======================  ==================================================
``view``                what was *observed* (Phase 3 redaction)
``knowledge``           what is *certain* or *deduced*, with the rule that
                        established each placement
``belief``              what is *probable*, and only that
======================  ==================================================

There is no separate ``DeductionState``, because there is nothing for it
to hold: Phase 6 records the justifying rule on every placement inside
:class:`~durakfish.information.KnowledgeState`. A parallel type would
duplicate that and invite the two copies to disagree.

The four levels stay distinct
-----------------------------
A search must be able to tell "the opponent *has* this card" from "the
opponent has this card with probability 0.8", so certainty and
probability are never collapsed into one number:

===============  ==========================================================
``CERTAIN``      ``knowledge.certainty(card) is Certainty.KNOWN`` — seen
``DEDUCED``      proven from public facts and the rules
``PROBABILISTIC``  ``certainty`` is ``UNKNOWN``; ask :meth:`probability`
``UNKNOWN``      more than one location remains possible
===============  ==========================================================

:meth:`classify` returns exactly this, and a probability is never
reported through it.

What cannot be in here
----------------------
No ``GameState``, no opponent hand, no talon contents, no hidden history,
and no callback that could fetch any of them.
:data:`FORBIDDEN_ATTRIBUTES` names the escape hatches explicitly so the
rule is machine-checkable rather than a matter of good intentions;
``tests/test_search_contract.py`` walks the whole reachable object graph
and asserts none of them exist.

The invariant that matters
--------------------------
If two hidden worlds look identical to an observer, the contract built
from them must be identical:

    observe(G1, p) == observe(G2, p)  =>  SearchInformation(G1, p) ==
                                          SearchInformation(G2, p)

Not merely equal objects — identical under a full fingerprint covering
serialisation, keys, possible sets, certainties, probabilities and
entropy. That is what makes it safe to hand this to a search that will
later sample hidden worlds.

Deferred on purpose
-------------------
Determinization, world sampling, MCTS, PIMC and ISMCTS are **not** here.
Phase 7.1 defines what may cross the boundary; Phase 7.2 decides what to
do with it. Building both at once would mean debugging a sampler and its
information source together, and any disagreement between them would be
ambiguous.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from durakfish.cards import Card
from durakfish.exceptions import DurakFishError
from durakfish.information.belief import BeliefState
from durakfish.information.information_set import InformationSet
from durakfish.information.knowledge import (
    CardLocation,
    Certainty,
    KnowledgeState,
)
from durakfish.information.tracker import CardTracker

__all__ = [
    "FORBIDDEN_ATTRIBUTES",
    "InformationClass",
    "SearchContractError",
    "SearchInformation",
]

#: Attribute names that would let a caller reach hidden state. None of
#: these may exist anywhere in the reachable object graph of a contract.
#: Kept as data so the prohibition is checkable by a test rather than
#: merely described in prose.
FORBIDDEN_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "game_state",
        "real_state",
        "actual_state",
        "actual_opponent_hand",
        "opponent_hand",
        "hidden_state",
        "hidden_hand",
        "hidden_cards",
        "true_state",
        "ground_truth",
        "hands",
        "_state",
        "_game",
    }
)


class SearchContractError(DurakFishError):
    """A contract was constructed from mismatched or unusable parts."""


class InformationClass(Enum):
    """How well a card's location is established, for a search to branch on.

    Coarser than :class:`~durakfish.information.knowledge.Certainty` and
    aimed at the consumer: a search cares whether it may treat a placement
    as fact, must reason about it probabilistically, or knows nothing.
    """

    #: Directly observed. Treat as fact.
    CERTAIN = "certain"
    #: Proven from public information and the rules. Also fact.
    DEDUCED = "deduced"
    #: Not established; a probability is available instead.
    PROBABILISTIC = "probabilistic"
    #: Not established and no meaningful distribution applies.
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SearchInformation:
    """Everything a search may legitimately consume, in one immutable object.

    Attributes:
        view: the observed position, straight from Phase 3 redaction.
        knowledge: certain and deduced placements, each with its rule.
        belief: probabilities over what remains undetermined.

    All three components are themselves immutable, so a contract cannot be
    changed after construction and cannot be changed *through* the objects
    it was built from. A :class:`CardTracker` that keeps running does not
    drag its old contracts along with it.
    """

    view: InformationSet
    knowledge: KnowledgeState
    belief: BeliefState

    def __post_init__(self) -> None:
        if self.view.player != self.knowledge.player:
            raise SearchContractError(
                f"view belongs to P{self.view.player} but knowledge to "
                f"P{self.knowledge.player}"
            )
        if self.belief.knowledge is not self.knowledge:
            raise SearchContractError(
                "belief must be derived from this contract's knowledge, "
                "not from a different state"
            )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_view(cls, view: InformationSet) -> SearchInformation:
        """Build the contract from an information set alone.

        Runs the Phase 6 pipeline from scratch: it walks the view's public
        history to recover what the opponent was seen taking, deduces, and
        derives beliefs. Convenient and self-contained; prefer
        :meth:`from_tracker` when a tracker is already being maintained,
        which is the cheaper path.
        """
        return cls.from_tracker(CardTracker.rebuild(view), view)

    @classmethod
    def from_tracker(
        cls, tracker: CardTracker, view: InformationSet
    ) -> SearchInformation:
        """Snapshot a running tracker into a contract.

        Args:
            tracker: a tracker already updated with ``view``.
            view: the observation the tracker was last updated with.

        Raises:
            SearchContractError: if the tracker and view belong to
                different players, or if the tracker has not yet seen this
                view's history — either means the caller is pairing
                mismatched inputs, and a contract built from them would
                describe a position that never existed.
        """
        if tracker.player != view.player:
            raise SearchContractError(
                f"tracker for P{tracker.player} paired with P{view.player}'s view"
            )
        if tracker.events_seen != len(view.public_history):
            raise SearchContractError(
                f"tracker has seen {tracker.events_seen} events but the view "
                f"carries {len(view.public_history)}; update the tracker first"
            )
        knowledge = tracker.knowledge
        return cls(
            view=view,
            knowledge=knowledge,
            belief=BeliefState.from_knowledge(knowledge),
        )

    # ------------------------------------------------------------------
    # Identity of the observer
    # ------------------------------------------------------------------
    @property
    def player(self) -> int:
        return self.view.player

    @property
    def opponent(self) -> int:
        return self.view.opponent

    # ------------------------------------------------------------------
    # The four levels, kept apart
    # ------------------------------------------------------------------
    def classify(self, card: Card) -> InformationClass:
        """How this card's location is established.

        Never returns a probability. A card the opponent holds with
        probability 0.99 classifies as ``PROBABILISTIC``, not ``CERTAIN``
        — which is the whole point of keeping the two apart.
        """
        certainty = self.knowledge.certainty(card)
        if certainty is Certainty.KNOWN:
            return InformationClass.CERTAIN
        if certainty is Certainty.DEDUCED:
            return InformationClass.DEDUCED
        if self.knowledge.possible_locations(card):
            return InformationClass.PROBABILISTIC
        return InformationClass.UNKNOWN  # pragma: no cover - defensive

    def location_of(self, card: Card) -> CardLocation | None:
        """Where the card certainly is, or ``None`` if not established."""
        return self.knowledge.location_of(card)

    def justification(self, card: Card) -> str | None:
        """Which rule established the card's location, if any."""
        return self.knowledge.justification(card)

    def possible_locations(self, card: Card) -> frozenset[CardLocation]:
        return self.knowledge.possible_locations(card)

    def is_impossible(self, card: Card, location: CardLocation) -> bool:
        return self.knowledge.is_impossible(card, location)

    def certain_cards_in(self, location: CardLocation) -> frozenset[Card]:
        """Cards established to be in ``location``. Never speculative."""
        return self.knowledge.cards_in(location)

    # ------------------------------------------------------------------
    # Probability
    # ------------------------------------------------------------------
    def probability(self, card: Card, location: CardLocation) -> float:
        """Marginal probability the card is in the location."""
        return self.belief.probability(card, location)

    def probability_all_in(
        self, cards: Iterable[Card], location: CardLocation
    ) -> float:
        """Joint probability for a set of cards. Accounts for dependence."""
        return self.belief.probability_all_in(frozenset(cards), location)

    @property
    def undetermined(self) -> frozenset[Card]:
        """Cards whose location is not established."""
        return self.knowledge.candidates

    @property
    def allocations(self) -> int:
        """How many hidden worlds remain consistent with what is known.

        The quantity a Phase 7.2 sampler will need. Exposing the count
        here does not sample anything.
        """
        return self.belief.allocations

    @property
    def uncertainty_bits(self) -> float:
        return self.belief.uncertainty_bits

    @property
    def is_open_information(self) -> bool:
        """True when nothing remains hidden and exact search is possible."""
        return self.knowledge.is_complete

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    def fingerprint(self) -> tuple[Any, ...]:
        """A deterministic value covering every observable aspect.

        Deliberately broad. Comparing one field could pass while another
        quietly depended on hidden state, so this covers the view, the
        knowledge, the beliefs and the derived summaries together.
        Built from sorted card codes and enum values, so it is stable
        across processes and hash seeds.
        """
        cards = sorted(self.knowledge.deck, key=lambda c: c.code)
        return (
            self.view.key(),
            self.knowledge.key(),
            tuple(
                (card.code, self.classify(card).value) for card in cards
            ),
            tuple(
                (
                    card.code,
                    tuple(sorted(loc.value for loc in self.possible_locations(card))),
                )
                for card in cards
            ),
            tuple(
                (
                    card.code,
                    round(self.probability(card, CardLocation.OPPONENT_HAND), 12),
                    round(self.probability(card, CardLocation.TALON), 12),
                )
                for card in cards
            ),
            tuple(sorted(c.code for c in self.undetermined)),
            self.allocations,
            round(self.uncertainty_bits, 12),
            self.is_open_information,
        )

    def to_dict(self) -> dict[str, Any]:
        """Deterministic, JSON-friendly form.

        Contains only what the observer legitimately knows, so a serialised
        contract cannot smuggle hidden state to another process.
        """
        return {
            "player": self.player,
            "view": self.view.to_dict(),
            "knowledge": self.knowledge.to_dict(),
            "belief": self.belief.to_dict(),
            "classification": {
                card.short: self.classify(card).value
                for card in sorted(self.knowledge.deck, key=lambda c: c.code)
            },
        }

    def describe(self) -> str:
        """Human-readable summary for debugging."""
        return "\n".join(
            [
                self.view.describe(),
                self.knowledge.describe(),
                f"  beliefs: {len(self.undetermined)} undetermined, "
                f"{self.allocations} allocations, "
                f"{self.uncertainty_bits:.2f} bits",
            ]
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SearchInformation):
            return self.fingerprint() == other.fingerprint()
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.fingerprint())

    def __repr__(self) -> str:
        return (
            f"SearchInformation(P{self.player}, "
            f"certain={len(self.knowledge.known_cards())}, "
            f"deduced={len(self.knowledge.deduced_cards())}, "
            f"undetermined={len(self.undetermined)})"
        )
