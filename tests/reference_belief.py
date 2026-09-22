"""A brute-force reference model, for validation only.

The production belief model in :mod:`durakfish.information.belief` uses a
closed form. This module computes the same quantities the slow, obvious
way: it enumerates every allocation of the undetermined cards to the two
hidden locations and counts. If the two disagree, the closed form is
wrong.

Deliberately kept in the test tree rather than shipped. It is exponential
in the number of undetermined cards, so it is only usable on small
positions, and having it importable from the library would invite someone
to reach for it on a hot path. It also must never be handed real hidden
state: it takes a :class:`KnowledgeState`, which is already redacted, so
it cannot leak anything the observer does not know.
"""

from __future__ import annotations

from itertools import combinations

from durakfish.cards import Card
from durakfish.information.knowledge import CardLocation, KnowledgeState


def enumerate_allocations(
    knowledge: KnowledgeState,
) -> list[frozenset[Card]]:
    """Every set of undetermined cards that could be the opponent's.

    The talon gets whatever is left, so one subset determines the whole
    allocation.
    """
    candidates = sorted(knowledge.candidates, key=lambda c: c.code)
    return [
        frozenset(subset)
        for subset in combinations(candidates, knowledge.opponent_slots)
    ]


def reference_marginals(
    knowledge: KnowledgeState, location: CardLocation
) -> dict[Card, float]:
    """Marginal probability per undetermined card, counted by enumeration."""
    allocations = enumerate_allocations(knowledge)
    if not allocations:
        return {}
    counts: dict[Card, int] = {card: 0 for card in knowledge.candidates}
    for opponent_set in allocations:
        chosen = (
            opponent_set
            if location is CardLocation.OPPONENT_HAND
            else knowledge.candidates - opponent_set
        )
        for card in chosen:
            counts[card] += 1
    total = len(allocations)
    return {card: hits / total for card, hits in counts.items()}


def reference_joint(
    knowledge: KnowledgeState, cards: frozenset[Card], location: CardLocation
) -> float:
    """Probability that every card in ``cards`` sits in ``location``."""
    allocations = enumerate_allocations(knowledge)
    if not allocations:
        return 1.0 if not cards else 0.0

    settled_elsewhere = any(
        knowledge.location_of(card) is not None
        and knowledge.location_of(card) is not location
        for card in cards
    )
    if settled_elsewhere:
        return 0.0

    undetermined = cards & knowledge.candidates
    hits = 0
    for opponent_set in allocations:
        chosen = (
            opponent_set
            if location is CardLocation.OPPONENT_HAND
            else knowledge.candidates - opponent_set
        )
        if undetermined <= chosen:
            hits += 1
    return hits / len(allocations)
