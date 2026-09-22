"""An enumerating reference for the set of compatible worlds.

The production sampler in :mod:`durakfish.information.worlds` draws a
uniformly random subset. This module instead **enumerates every
allocation** and filters the incompatible ones out with a validator.
Deliberately a different algorithm: if the sampler and this agree, that
is evidence, not a tautology.

Exponential in the number of undetermined cards, so it is only usable on
small positions — which is why it lives in the test tree rather than the
library, where somebody would eventually call it on a hot path.

It is handed a :class:`SearchInformation`, which is already redacted, so
it cannot see or leak real hidden state either.
"""

from __future__ import annotations

from itertools import combinations

from durakfish.information.knowledge import CardLocation
from durakfish.information.search_contract import SearchInformation
from durakfish.information.worlds import DeterminizedWorld, is_valid_world


def enumerate_allocations(
    information: SearchInformation,
) -> list[frozenset]:
    """Every opponent-hand assignment compatible with the information.

    Built the obvious way: take all subsets of the undetermined cards of
    the right size, add the cards already known to be the opponent's, and
    keep the ones a validator accepts. No sampling, no shortcuts.
    """
    knowledge = information.knowledge
    candidates = sorted(knowledge.candidates, key=lambda c: c.code)
    certain = knowledge.cards_in(CardLocation.OPPONENT_HAND)
    fixed_talon = knowledge.cards_in(CardLocation.TALON)
    trump = information.view.trump_card

    allocations: list[frozenset] = []
    for subset in combinations(candidates, knowledge.opponent_slots):
        opponent = certain | frozenset(subset)
        remaining = [c for c in candidates if c not in opponent]
        body = sorted(fixed_talon | frozenset(remaining), key=lambda c: c.code)
        if information.view.talon_size:
            body = [c for c in body if c != trump]
            talon = tuple(body) + (trump,)
        else:
            talon = ()
        world = DeterminizedWorld(
            player=information.player, opponent_hand=opponent, talon=talon
        )
        if is_valid_world(world, information):
            allocations.append(opponent)
    return allocations


def uniform_probability(information: SearchInformation) -> float:
    """The probability each compatible allocation should receive."""
    total = len(enumerate_allocations(information))
    return 1.0 / total if total else 0.0
