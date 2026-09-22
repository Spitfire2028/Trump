"""Exception hierarchy for DurakFish.

Design rule (see docs/architecture.md): the engine **fails loudly**. An
impossible internal state is a bug, not something to paper over with a
default value. Every exception below derives from :class:`DurakFishError`
so that callers can catch everything from this project with one clause,
while also deriving from the closest standard-library exception so that
generic code (``except ValueError``) keeps working.

Only Phase 1 (cards + deck) exceptions live here so far. Later phases add
``IllegalMoveError``, ``InvalidStateError``, etc. to this same module.
"""

from __future__ import annotations

__all__ = [
    "AgentError",
    "CardNotInDeckError",
    "CardParseError",
    "DeckError",
    "DeckExhaustedError",
    "DuplicateCardError",
    "DurakFishError",
    "GameError",
    "IllegalAgentMoveError",
    "IllegalMoveError",
    "InvalidCardError",
    "InvalidDeckSizeError",
    "InvalidStateError",
    "RecordError",
    "SimulationError",
    "UnsupportedRuleError",
]


class DurakFishError(Exception):
    """Base class for every error raised by DurakFish."""


class InvalidCardError(DurakFishError, ValueError):
    """A card could not be constructed from the supplied data."""


class CardParseError(InvalidCardError):
    """A string could not be parsed into a card, rank or suit."""


class DeckError(DurakFishError):
    """Base class for deck-related errors."""


class DeckExhaustedError(DeckError):
    """A card was requested from an empty deck."""


class DuplicateCardError(DeckError, ValueError):
    """The same physical card was found twice in one pile.

    This is the single most important invariant in the whole engine: a card
    exists in exactly one place at any time. Violating it silently would
    corrupt every probability estimate downstream.
    """


class CardNotInDeckError(DeckError, LookupError):
    """A card was expected in the deck but is not present."""


class InvalidDeckSizeError(DeckError, ValueError):
    """An unsupported deck size was requested."""


class GameError(DurakFishError):
    """Base class for rule- and state-related errors (Phase 2)."""


class IllegalMoveError(GameError, ValueError):
    """A move was applied that the rules do not permit in this state.

    Raised by ``apply_move``. Legal-move generation must never produce a
    move that triggers this: if it does, the two implementations of the
    rules disagree and one of them is wrong.
    """


class InvalidStateError(GameError):
    """A game state violates an invariant — always an engine bug.

    Raised by ``GameState.validate()``. The canonical example is a card
    existing in two locations at once.
    """


class UnsupportedRuleError(GameError, NotImplementedError):
    """A rule configuration that this engine version does not implement."""


class SimulationError(DurakFishError):
    """Base class for errors raised while driving or replaying a game (Phase 3)."""


class AgentError(SimulationError):
    """An agent misbehaved."""


class IllegalAgentMoveError(AgentError, IllegalMoveError):
    """An agent returned a move that is not in the legal set it was given.

    Deriving from :class:`IllegalMoveError` as well means a caller that
    already guards against illegal moves catches this too, while the
    driver can still distinguish "the agent is broken" from "the rules
    rejected a move".
    """


class RecordError(SimulationError):
    """A game record is malformed, or a replay diverged from it."""
