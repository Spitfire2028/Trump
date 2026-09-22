"""RandomBot — the null policy.

Chooses uniformly at random from the legal moves it is given. It exists
as the floor every later agent must beat: an engine that cannot outscore
uniform random play is not doing anything.

Reproducibility
---------------
The bot draws exclusively from the :class:`random.Random` instance handed
to it by :meth:`reset`. It never touches the global ``random`` module and
never constructs a generator of its own, so a game is fully determined by
the master seed the simulation layer derived that instance from.

Order independence
------------------
Sampling by index into the supplied list would make the choice depend on
the order the rules engine happened to emit moves in. The bot sorts into
:func:`~durakfish.ai.tiebreak.canonical_order` first, so "uniform over
the legal moves" is a statement about the *set*. Two callers passing the
same moves in different orders get the same answer from the same RNG
state — which is what makes the metamorphic test in
``tests/test_phase4_audit.py`` meaningful rather than vacuous.

Hidden information
------------------
The view is not consulted at all. That is not an oversight: it makes
RandomBot trivially immune to information leaks, and any future claim
that a stronger bot beats it is then a claim about using information
well, not about having more of it.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from durakfish.ai.base import AgentBase
from durakfish.ai.tiebreak import canonical_order
from durakfish.exceptions import AgentError
from durakfish.game.moves import Move
from durakfish.information import InformationSet

__all__ = ["RandomBot"]


class RandomBot(AgentBase):
    """Picks uniformly from the supplied legal moves."""

    name = "random"

    def __init__(self, name: str | None = None) -> None:
        super().__init__(name)
        # Replaced wholesale by reset(); never used to make a decision in a
        # driven game, and deliberately not seeded from entropy so that a
        # forgotten reset() is reproducible rather than mysteriously random.
        self._rng = random.Random(0)

    def reset(self, rng: random.Random) -> None:
        """Adopt ``rng`` as the sole source of randomness for a new game.

        This is the whole of the bot's per-game state, so a reset agent is
        indistinguishable from a fresh one given the same generator.
        """
        self._rng = rng

    def choose(self, view: InformationSet, legal: Sequence[Move]) -> Move:
        """Return one of ``legal``, uniformly at random.

        Raises:
            AgentError: if ``legal`` is empty. The rules engine never
                offers an empty choice in a live position, so this means
                the caller is broken; inventing a move or returning
                ``None`` would hide that.
        """
        if not legal:
            raise AgentError(
                f"{self.name} was asked to choose from an empty legal-move list"
            )
        ordered = canonical_order(legal)
        return ordered[self._rng.randrange(len(ordered))]
