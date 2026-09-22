"""SearchBot — the first agent that looks ahead.

Implements the Phase 4 :class:`~durakfish.ai.base.Agent` protocol
unchanged: it receives an :class:`~durakfish.information.InformationSet`
and the legal moves, and returns one of them. No ``GameState``, no extra
arguments, no callbacks.

What it does
------------
1. Builds a :class:`~durakfish.ai.searchnode.SearchNode` from the view.
2. Searches, restricted at the root to exactly the moves it was handed.
3. Returns the best of those moves.

Restricting the root to the supplied list is not a formality. It makes the
contract structural: a move that was not offered cannot be returned, no
matter what the search concludes about it.

What it does not do
-------------------
No card tracking, no hidden-hand enumeration, no determinization, no
probabilities. The search is bounded by what the view actually contains,
which is why it reasons about opponent *action classes* rather than
opponent cards. See :mod:`durakfish.ai.searchnode` for the abstraction and
its limits.

Determinism
-----------
Fully deterministic. The RNG required by the agent protocol is accepted
and ignored, so the master seed changes the cards dealt but never
SearchBot's reaction to them — the same property GreedyBot has, for the
same reason.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from durakfish.ai.base import AgentBase
from durakfish.ai.search import SearchConfig, SearchResult, search
from durakfish.ai.searchnode import SearchNode
from durakfish.exceptions import AgentError
from durakfish.game.moves import Move
from durakfish.information import InformationSet

__all__ = ["SearchBot"]


class SearchBot(AgentBase):
    """Chooses by searching the bout-local tree."""

    name = "search"

    def __init__(
        self, config: SearchConfig | None = None, name: str | None = None
    ) -> None:
        super().__init__(name)
        self.config = config or SearchConfig()
        self.last_result: SearchResult | None = None

    def reset(self, rng: random.Random) -> None:
        """Nothing per-game to establish; the policy is deterministic.

        The generator is accepted to satisfy the protocol and ignored. The
        only state kept between calls is :attr:`last_result`, which is
        diagnostic and is cleared here so a reused bot cannot report a
        stale search.
        """
        self.last_result = None

    def choose(self, view: InformationSet, legal: Sequence[Move]) -> Move:
        """Search, then return the best of the supplied moves.

        Raises:
            AgentError: if ``legal`` is empty, or if the search somehow
                returns nothing to play. Both mean something upstream is
                broken, and guessing would hide it.
        """
        if not legal:
            raise AgentError(
                f"{self.name} was asked to choose from an empty legal-move list"
            )
        node = SearchNode.from_view(view)
        result = search(node, self.config, root_moves=tuple(legal))
        self.last_result = result
        if result.move is None:  # pragma: no cover - guarded above
            raise AgentError(f"{self.name} produced no move for {view.describe()}")
        return result.move

    def __repr__(self) -> str:
        return f"SearchBot({self.name!r}, depth={self.config.depth})"
