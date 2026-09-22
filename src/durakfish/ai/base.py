"""The agent boundary.

An agent is anything that can choose a move given an
:class:`~durakfish.information.InformationSet` and the legal moves
available. That signature is the whole point of the layer: an agent is
handed a redacted view, never a :class:`~durakfish.game.GameState`, so it
cannot consult hidden information even by accident.

Why ``legal`` is passed in
--------------------------
The legal moves available to the player to move are derivable from their
information set alone — they depend only on that player's own hand and the
public table — so passing them leaks nothing. It also keeps a single
source of truth for legality: the rules engine. A second implementation
living in the information or AI layer would eventually drift from the
first, and the drift would show up as an engine that believes in moves the
rules reject. ``tests/test_information_set.py`` re-derives legal moves
from the view independently and asserts the two agree, which is the
cross-check without the duplicate production code.

What is deliberately *not* in the protocol
------------------------------------------
No time budget or search-configuration parameter. An agent owns its own
budget; ``SearchConfig`` arrives with the search engine in a later phase,
and baking a deadline into every agent now would be speculative.

No event-notification hook. A stateful agent — an opponent model, say —
does not need one, because the information set already carries the full
redacted history. Adding a push API before something needs it would be
interface surface with no caller.

This module contains **no strategy**. ``ScriptedAgent`` and
``CallableAgent`` are infrastructure: a test double and an adapter.
RandomBot, GreedyBot and the rest are Phase 4 onward.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from typing import Protocol, runtime_checkable

from durakfish.exceptions import AgentError
from durakfish.game.moves import Move
from durakfish.information import InformationSet

__all__ = ["Agent", "AgentBase", "CallableAgent", "ScriptedAgent"]


@runtime_checkable
class Agent(Protocol):
    """Anything that can pick a move from a redacted view of the game."""

    name: str

    def choose(self, view: InformationSet, legal: Sequence[Move]) -> Move:
        """Return one of ``legal``.

        Args:
            view: what this player is allowed to know. Immutable.
            legal: every legal move, in the engine's deterministic order.
                Never empty when it is this player's turn.

        Returns:
            A move drawn from ``legal``. Returning anything else is an
            error the driver reports rather than silently repairs.
        """
        ...

    def reset(self, rng: random.Random) -> None:
        """Prepare for a new game with a dedicated random stream.

        The driver derives ``rng`` from the master seed, so an agent that
        uses only this generator keeps whole games reproducible. Agents
        that reach for the global ``random`` module break that guarantee.
        """
        ...


class AgentBase:
    """Convenience base supplying a name and a no-op :meth:`reset`."""

    name: str = "agent"

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name

    def reset(self, rng: random.Random) -> None:
        return None

    def choose(self, view: InformationSet, legal: Sequence[Move]) -> Move:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class ScriptedAgent(AgentBase):
    """Plays a fixed sequence of moves. A test double, not a strategy.

    Useful for pinning an exact line of play in a test, and for driving a
    recorded game back through the engine.

    Raises:
        AgentError: if the script runs out, or if the next scripted move
            is not legal in the position reached. Both mean the script and
            the engine disagree, which is worth failing loudly over rather
            than substituting some other move.
    """

    name = "scripted"

    def __init__(self, moves: Iterable[Move], name: str | None = None) -> None:
        super().__init__(name)
        self._script: tuple[Move, ...] = tuple(moves)
        self._index = 0

    def reset(self, rng: random.Random) -> None:
        self._index = 0

    def choose(self, view: InformationSet, legal: Sequence[Move]) -> Move:
        if self._index >= len(self._script):
            raise AgentError(
                f"{self.name} ran out of scripted moves after {self._index}"
            )
        move = self._script[self._index]
        self._index += 1
        if move not in legal:
            raise AgentError(
                f"scripted move {move} is not legal here; "
                f"legal moves are: {', '.join(str(m) for m in legal)}"
            )
        return move

    @property
    def moves_played(self) -> int:
        return self._index


class CallableAgent(AgentBase):
    """Adapts a plain function into an agent.

    The function receives the same three things an agent has: the view,
    the legal moves, and its own seeded generator. This is the adapter the
    later bot phases build on; it embodies no policy itself.

        >>> lowest = CallableAgent(lambda view, legal, rng: legal[0], "first-legal")
    """

    name = "callable"

    def __init__(
        self,
        choose: Callable[[InformationSet, Sequence[Move], random.Random], Move],
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self._choose = choose
        self._rng = random.Random(0)

    def reset(self, rng: random.Random) -> None:
        self._rng = rng

    def choose(self, view: InformationSet, legal: Sequence[Move]) -> Move:
        return self._choose(view, legal, self._rng)
