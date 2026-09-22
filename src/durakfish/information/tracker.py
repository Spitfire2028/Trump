"""Incremental card tracking.

:class:`CardTracker` maintains one player's knowledge across a game,
consuming the redacted public event stream from their
:class:`~durakfish.information.InformationSet`. It is the only stateful
object in this layer; :class:`~durakfish.information.knowledge.KnowledgeState`
and :class:`~durakfish.information.belief.BeliefState` are immutable
snapshots taken from it.

What is actually incremental
----------------------------
Almost everything a deduction needs can be read off the current position:
the observer's hand, the table, the discard, the counts. Exactly one fact
cannot, and it is the one the tracker carries forward:

    which cards the observer watched the opponent take, and has not seen
    played since.

That set is updated event by event: a take adds the table's cards to it, a
card the opponent later plays is removed, and anything that becomes public
by another route drops out when the knowledge is rebuilt. Because that set
is the whole of the carried state, "incremental" and "rebuilt from the
full history" are provably the same thing, and
``tests/test_tracker.py`` checks that after every event of thousands of
games rather than taking it on trust.

Observer-specific by construction
---------------------------------
A tracker belongs to a player. There is no omniscient variant, and the
constructor takes a player id precisely so that the question "whose
knowledge is this?" always has an answer. Two trackers watching the same
game hold genuinely different knowledge.

No hidden information
---------------------
The tracker never sees a ``GameState``. Its inputs are an information set
and its own accumulated record, both of which are already redacted.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from durakfish.cards import Card
from durakfish.exceptions import DurakFishError
from durakfish.information.belief import BeliefState
from durakfish.information.deduction import Observation, deduce
from durakfish.information.information_set import InformationSet, PublicEvent
from durakfish.information.knowledge import KnowledgeState

__all__ = ["CardTracker", "TrackerError"]


class TrackerError(DurakFishError):
    """The tracker was fed something it cannot reconcile."""


class CardTracker:
    """One player's accumulated knowledge, updated event by event."""

    __slots__ = ("_events_seen", "_knowledge", "_player", "_retained")

    def __init__(self, player: int) -> None:
        self._player = player
        self._retained: set[Card] = set()
        self._events_seen = 0
        self._knowledge: KnowledgeState | None = None

    # ------------------------------------------------------------------
    @property
    def player(self) -> int:
        return self._player

    @property
    def events_seen(self) -> int:
        """How many public events have been folded in so far."""
        return self._events_seen

    @property
    def retained_by_opponent(self) -> frozenset[Card]:
        """Cards seen taken by the opponent and not seen played since.

        Exposed as an immutable snapshot: the tracker's internal set is
        never handed out, so a caller cannot corrupt its state.
        """
        return frozenset(self._retained)

    @property
    def knowledge(self) -> KnowledgeState:
        """The current knowledge state.

        Raises:
            TrackerError: before the first :meth:`update`, because a
                tracker that has observed nothing has nothing to report
                and returning an empty state would invite mistaking it
                for genuine knowledge.
        """
        if self._knowledge is None:
            raise TrackerError(
                f"tracker for P{self._player} has not observed anything yet"
            )
        return self._knowledge

    @property
    def belief(self) -> BeliefState:
        """Probabilities derived from the current knowledge."""
        return BeliefState.from_knowledge(self.knowledge)

    # ------------------------------------------------------------------
    def update(self, view: InformationSet) -> KnowledgeState:
        """Fold in any events not yet seen, then recompute knowledge.

        Args:
            view: the observer's current information set. Must belong to
                this tracker's player.

        Returns:
            The updated knowledge state.

        Raises:
            TrackerError: if the view belongs to another player, or if the
                event stream has shrunk, which would mean the tracker is
                being fed a different game.
        """
        if view.player != self._player:
            raise TrackerError(
                f"tracker for P{self._player} was given P{view.player}'s view"
            )
        if len(view.public_history) < self._events_seen:
            raise TrackerError(
                f"event stream shrank from {self._events_seen} to "
                f"{len(view.public_history)}; this is a different game"
            )

        for event in view.public_history[self._events_seen :]:
            self._absorb(event)
        self._events_seen = len(view.public_history)

        self._knowledge = deduce(
            Observation.from_view(view, retained_by_opponent=self._retained)
        )
        return self._knowledge

    def _absorb(self, event: PublicEvent) -> None:
        """Update the one fact that cannot be re-read from the position."""
        opponent = 1 - self._player
        # A card the opponent plays becomes public again, wherever it came from.
        if event.player == opponent:
            self._retained.difference_update(event.revealed)
        # Cards the opponent picks up enter their hand.
        if event.taken_by == opponent:
            self._retained.update(event.taken)
        elif event.taken:
            # The observer took them; they are in their own hand now.
            self._retained.difference_update(event.taken)
        # Retired cards are public and can never be in a hand.
        self._retained.difference_update(event.discarded)

    # ------------------------------------------------------------------
    @classmethod
    def replay(
        cls, player: int, views: Sequence[InformationSet]
    ) -> CardTracker:
        """Rebuild a tracker by replaying a sequence of views in order."""
        tracker = cls(player)
        for view in views:
            tracker.update(view)
        return tracker

    @classmethod
    def rebuild(cls, view: InformationSet) -> CardTracker:
        """Build knowledge from a single view's whole history at once.

        The reference implementation: it re-reads the entire event stream
        rather than carrying anything forward. Used by the tests as an
        independent check on the incremental path, and never on a hot
        path itself.
        """
        tracker = cls(view.player)
        tracker.update(view)
        return tracker

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Deterministic checkpoint.

        Stores only the observer's legitimate knowledge: the retained-card
        record and the event cursor. No hidden state can travel in a
        checkpoint because none is held.
        """
        return {
            "player": self._player,
            "events_seen": self._events_seen,
            "retained": sorted(c.short for c in self._retained),
            "knowledge": None if self._knowledge is None else self._knowledge.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], deck: Iterable[Card] | None = None
    ) -> CardTracker:
        """Restore a checkpoint written by :meth:`to_dict`."""
        tracker = cls(int(data["player"]))
        tracker._events_seen = int(data["events_seen"])
        tracker._retained = {Card.parse(c) for c in data["retained"]}
        stored = data.get("knowledge")
        if stored is not None:
            if deck is None:
                raise TrackerError(
                    "restoring a knowledge state needs the deck it was built over"
                )
            tracker._knowledge = KnowledgeState.from_dict(stored, frozenset(deck))
        return tracker

    def key(self) -> tuple[Any, ...]:
        """Canonical key covering both the carried state and the knowledge."""
        return (
            self._player,
            self._events_seen,
            tuple(sorted(c.code for c in self._retained)),
            () if self._knowledge is None else self._knowledge.key(),
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CardTracker):
            return self.key() == other.key()
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.key())

    def __repr__(self) -> str:
        return (
            f"CardTracker(P{self._player}, events={self._events_seen}, "
            f"retained={len(self._retained)})"
        )
