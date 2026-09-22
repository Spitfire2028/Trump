"""GreedyBot — a simple, fully deterministic baseline.

This is **not** a strong Durak player and is not trying to be. Its job is
to be a fixed, explainable, non-random opponent that later agents can be
measured against, and to prove that a real policy can be written against
nothing but an :class:`~durakfish.information.InformationSet` and a list
of legal moves.

The heuristic
-------------
One idea, applied everywhere: **spend the cheapest card you can, and
never spend a trump when you had a choice about spending anything.**

Card cost is the pair ``(is_trump, code)``:

* non-trump cards are cheaper than every trump — trumps are the scarce
  resource in Durak and a baseline that squanders them is not a useful
  yardstick;
* within that, lower cards are cheaper, because a card's dense integer
  code orders by rank and then by suit.

From that, three rules:

============  ======================================================
Situation     Rule
============  ======================================================
Opening       Attacking is compulsory, so play the cheapest card.
a bout
Adding to     Add the cheapest card, **unless it is a trump**, in
a bout        which case end the bout instead. Junk is worth
(or throwing  throwing in; trumps are not.
in)
Defending     Beat the attack with the cheapest card that does the
              job. Never take voluntarily.
============  ======================================================

Deliberately absent, and left to later phases: any notion of what the
opponent might hold, any lookahead, any weighing of taking against
defending, any awareness of the endgame. "Never take voluntarily" is
plainly wrong in real play — beating a six with the ace of trumps is
usually a blunder — and it stays wrong on purpose, because a baseline
whose weaknesses are obvious is more useful than one whose weaknesses are
subtle.

Determinism and tie-breaking
----------------------------
The policy is a pure function of ``(view, legal)``. It ignores its RNG
entirely, so the master seed changes the cards dealt but never
GreedyBot's reaction to them.

Every comparison ends with
:func:`~durakfish.ai.tiebreak.canonical_move_key`, so the ordering never
depends on object identity, hashing, or the order the rules engine
emitted its moves in — permuting ``legal`` cannot change the answer.

That final tie-breaker is, as things stand, **defensive rather than
load-bearing**, and it is worth saying so rather than implying it is
exercised. Two card-carrying moves cannot tie, because ``card_cost`` is
injective over the deck; and the card-less branch cannot tie either,
because ``TakeCards`` and ``EndAttack`` are never legal in the same
position (0 occurrences in 22,038 positions sampled from 200 games). A
mutation replacing the tie-breaker with ``id()`` therefore passes the
whole suite — it is unreachable, not untested. It stays because a future
scoring function with coarser buckets would make ties reachable
immediately, and discovering that through a non-reproducible bot would be
far worse than carrying three unused lines.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import TypeVar

from durakfish.ai.base import AgentBase
from durakfish.ai.tiebreak import canonical_move_key
from durakfish.cards import Card, Suit
from durakfish.exceptions import AgentError
from durakfish.game.moves import AttackMove, DefenseMove, EndAttack, Move
from durakfish.information import InformationSet

__all__ = ["GreedyBot", "card_cost"]

#: Any move carrying a card, so ``_cheapest`` returns the concrete type in.
_CardMove = TypeVar("_CardMove", bound=Move)


def card_cost(card: Card, trump: Suit) -> tuple[int, int]:
    """How reluctant GreedyBot is to part with ``card``. Lower is cheaper.

    ``(is_trump, code)`` — every non-trump is cheaper than every trump,
    and within a group the lower rank is cheaper. Total over the deck, so
    two distinct cards never tie.
    """
    return (1 if card.suit == trump else 0, card.code)


class GreedyBot(AgentBase):
    """A deterministic, view-only baseline policy."""

    name = "greedy"

    def reset(self, rng: random.Random) -> None:
        """No per-game state to establish: the policy is stateless.

        The generator is accepted and ignored. Two GreedyBots reset with
        different seeds play identically, which is expected rather than a
        defect — see the module docstring.
        """
        return None

    # ------------------------------------------------------------------
    def choose(self, view: InformationSet, legal: Sequence[Move]) -> Move:
        """Apply the documented heuristic.

        Raises:
            AgentError: if ``legal`` is empty, or if it contains no move
                this policy knows how to rank. Both mean the caller is
                broken, and guessing would hide it.
        """
        if not legal:
            raise AgentError(
                f"{self.name} was asked to choose from an empty legal-move list"
            )

        defences = [m for m in legal if isinstance(m, DefenseMove)]
        if defences:
            return self._cheapest(defences, view.trump_suit, lambda m: m.defending_card)

        attacks = [m for m in legal if isinstance(m, AttackMove)]
        if attacks:
            cheapest = self._cheapest(attacks, view.trump_suit, lambda m: m.card)
            opening = not view.table
            spending_a_trump = cheapest.card.suit == view.trump_suit
            if opening or not spending_a_trump:
                return cheapest
            # Only trumps left to add, and adding is optional: stop instead.
            end = [m for m in legal if isinstance(m, EndAttack)]
            if end:
                return min(end, key=canonical_move_key)
            return cheapest

        # No card play available: take, or end the bout, as the rules allow.
        return min(legal, key=canonical_move_key)

    @staticmethod
    def _cheapest(
        moves: Sequence[_CardMove],
        trump: Suit,
        card_of: Callable[[_CardMove], Card],
    ) -> _CardMove:
        """Pick the lowest-cost move, breaking any remaining tie canonically."""
        if not moves:  # pragma: no cover - callers check first
            raise AgentError("no candidate moves to rank")
        return min(
            moves,
            key=lambda m: (card_cost(card_of(m), trump), canonical_move_key(m)),
        )
