"""The deduction engine.

Turns observations into knowledge by applying rules that are *logically
valid*, never merely plausible. If something is only likely, it belongs in
:mod:`durakfish.information.belief` and must not appear here.

The pipeline is deliberately one-directional and separable:

    observation  ->  deduction  ->  knowledge  ->  belief

Each stage is a pure function of the one before it. Beliefs cannot write
back into knowledge, so a probability can never harden into a fact.

The rules
---------
Five, each named, each justified from the rules of Durak alone. Every
settled card records which rule placed it, so any claim the tracker makes
can be traced back to a stated reason.

``observed``
    The observer's own hand, the cards on the table, the discard pile.
    Seen, not inferred.

``trump-at-talon-bottom``
    While the talon holds cards, the face-up trump card is the bottom one
    and will be drawn last, so it is in the talon. Straight from
    ``docs/rules.md`` §1.

``opponent-retains-taken``
    A card the observer watched the opponent pick up, and has not seen
    played since, is still in their hand. Sound because a card leaves a
    hand only by being played to the table, which is public. Verified
    against ground truth over 44,076 observations without a single
    counterexample.

``talon-exhausted``
    With no unaccounted talon slots left, every undetermined card must be
    in the opponent's hand. This is the counting rule that makes the
    endgame fully known.

``opponent-hand-saturated``
    The mirror image: when every card in the opponent's hand is already
    accounted for, the remaining undetermined cards must all be in the
    talon.

The last two are the same principle — cards must go somewhere, and the
slots are counted — applied in the two directions. Neither guesses; both
follow from conservation and public hand sizes.

What is deliberately *not* deduced
----------------------------------
Nothing is inferred from an opponent's *choice*. That they declined to
beat a card suggests they lack a beater; it does not prove it, because
taking is always legal and sometimes correct. Behavioural inference of
that sort is a belief, not a deduction, and this engine does not make it.
Discarding a card likewise licenses no inference beyond the card's own
location.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from durakfish.cards import Card
from durakfish.cards.deck import standard_cards
from durakfish.information.information_set import InformationSet
from durakfish.information.knowledge import (
    CardLocation,
    Certainty,
    KnowledgeError,
    KnowledgeState,
    Placement,
)

__all__ = [
    "RULES",
    "RULE_OBSERVED",
    "RULE_OPPONENT_RETAINS",
    "RULE_OPPONENT_SATURATED",
    "RULE_TALON_EXHAUSTED",
    "RULE_TRUMP_BOTTOM",
    "Observation",
    "deduce",
]

RULE_OBSERVED = "observed"
RULE_TRUMP_BOTTOM = "trump-at-talon-bottom"
RULE_OPPONENT_RETAINS = "opponent-retains-taken"
RULE_TALON_EXHAUSTED = "talon-exhausted"
RULE_OPPONENT_SATURATED = "opponent-hand-saturated"

#: Every rule, with the justification it rests on.
RULES: dict[str, str] = {
    RULE_OBSERVED: "the observer saw the card in this location",
    RULE_TRUMP_BOTTOM: (
        "the face-up trump card lies at the bottom of the talon and is "
        "drawn last, so while the talon is non-empty it is in the talon"
    ),
    RULE_OPPONENT_RETAINS: (
        "the observer saw the opponent take this card and has not seen it "
        "played since; a card leaves a hand only by being played publicly"
    ),
    RULE_TALON_EXHAUSTED: (
        "no unaccounted talon slots remain, so every undetermined card "
        "must be in the opponent's hand"
    ),
    RULE_OPPONENT_SATURATED: (
        "every card in the opponent's hand is already accounted for, so "
        "every undetermined card must be in the talon"
    ),
}


@dataclass(frozen=True, slots=True)
class Observation:
    """The raw facts a deduction is computed from.

    Everything here comes from an :class:`InformationSet` plus the
    tracker's record of what it has watched happen. No ``GameState``, and
    nothing that is not the observer's to know.

    Attributes:
        retained_by_opponent: cards the observer watched the opponent take
            and has not seen played since. Maintained incrementally by the
            tracker, because it is the one fact that cannot be read off
            the current position.
    """

    player: int
    own_hand: frozenset[Card]
    table: frozenset[Card]
    discard: frozenset[Card]
    trump_card: Card
    talon_size: int
    opponent_hand_size: int
    retained_by_opponent: frozenset[Card]
    deck: frozenset[Card]

    @classmethod
    def from_view(
        cls, view: InformationSet, retained_by_opponent: Iterable[Card] = ()
    ) -> Observation:
        """Read the observable facts out of an information set."""
        return cls(
            player=view.player,
            own_hand=frozenset(view.hand),
            table=frozenset(view.table_cards),
            discard=frozenset(view.discard),
            trump_card=view.trump_card,
            talon_size=view.talon_size,
            opponent_hand_size=view.opponent_hand_size,
            retained_by_opponent=frozenset(retained_by_opponent),
            deck=frozenset(standard_cards(view.rules.deck_size)),
        )


def deduce(observation: Observation) -> KnowledgeState:
    """Apply every rule and return the resulting knowledge.

    Pure: the same observation always yields the same knowledge, with no
    randomness and no dependence on iteration order.

    Raises:
        KnowledgeError: if the observation is internally inconsistent —
            more cards accounted for in a location than it can hold, say.
            That indicates a bug upstream, not an unusual position.
    """
    settled: dict[Card, Placement] = {}

    def place(
        card: Card, location: CardLocation, certainty: Certainty, rule: str
    ) -> None:
        existing = settled.get(card)
        if existing is not None:
            if existing.location is not location:
                raise KnowledgeError(
                    f"{card} placed in {existing.location.value} by "
                    f"{existing.rule!r} and in {location.value} by {rule!r}"
                )
            return
        settled[card] = Placement(location, certainty, rule)

    # --- observed ----------------------------------------------------
    for card in observation.own_hand:
        place(card, CardLocation.SELF_HAND, Certainty.KNOWN, RULE_OBSERVED)
    for card in observation.table:
        place(card, CardLocation.TABLE, Certainty.KNOWN, RULE_OBSERVED)
    for card in observation.discard:
        place(card, CardLocation.DISCARD, Certainty.KNOWN, RULE_OBSERVED)

    # --- trump card at the bottom of the talon -----------------------
    if observation.talon_size > 0 and observation.trump_card not in settled:
        place(
            observation.trump_card,
            CardLocation.TALON,
            Certainty.DEDUCED,
            RULE_TRUMP_BOTTOM,
        )

    # --- cards the opponent took and has not played ------------------
    for card in observation.retained_by_opponent:
        if card not in settled:
            place(
                card,
                CardLocation.OPPONENT_HAND,
                Certainty.DEDUCED,
                RULE_OPPONENT_RETAINS,
            )

    # --- counting ----------------------------------------------------
    candidates = set(observation.deck) - set(settled)
    opponent_slots = observation.opponent_hand_size - sum(
        1 for p in settled.values() if p.location is CardLocation.OPPONENT_HAND
    )
    talon_slots = observation.talon_size - sum(
        1 for p in settled.values() if p.location is CardLocation.TALON
    )

    if opponent_slots < 0 or talon_slots < 0:
        raise KnowledgeError(
            f"observation is inconsistent: {opponent_slots} opponent slots and "
            f"{talon_slots} talon slots remain after placing known cards"
        )

    if talon_slots == 0 and candidates:
        for card in candidates:
            place(
                card,
                CardLocation.OPPONENT_HAND,
                Certainty.DEDUCED,
                RULE_TALON_EXHAUSTED,
            )
        candidates, opponent_slots = set(), 0
    elif opponent_slots == 0 and candidates:
        for card in candidates:
            place(card, CardLocation.TALON, Certainty.DEDUCED, RULE_OPPONENT_SATURATED)
        candidates, talon_slots = set(), 0

    return KnowledgeState(
        player=observation.player,
        settled=settled,
        candidates=frozenset(candidates),
        opponent_slots=opponent_slots,
        talon_slots=talon_slots,
        deck=observation.deck,
    )
