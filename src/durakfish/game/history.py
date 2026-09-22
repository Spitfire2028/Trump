"""Game history.

Two design decisions here differ from the original specification, both
for reasons that only become visible once search is running. Both are
explained in ``docs/architecture.md``.

**1. History is a persistent linked list, not a tuple.**
Appending to a tuple copies it, so carrying history in every state would
cost O(history) per ``apply_move`` — paid millions of times during
search. A cons list appends in O(1) and *shares structure* between
sibling search branches, which is exactly the access pattern a game tree
has. :meth:`GameState.events` materialises it in order when a human or a
replay actually needs it.

**2. Events do not embed the resulting state.**
The spec asked for a ``resulting_state`` field. Storing it would make
every state retain every earlier state, so a single search branch would
pin the whole game's memory, and a deep search would exhaust RAM. Events
instead record the *observable delta*; any state is reconstructible by
replaying events from the deal. Nothing is lost and the memory profile
becomes flat.

Events are **ground truth**: they record which cards were actually drawn.
Redacting them into what a given player may know is the job of the
information layer in Phase 6, never of the recorder here. Keeping the log
honest and filtering it later is what makes "the AI cannot cheat"
enforceable rather than aspirational.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from durakfish.cards import Card
from durakfish.game.moves import Move, move_from_dict

__all__ = ["DrawRecord", "EventKind", "GameEvent", "HistoryNode", "push_event"]


class EventKind(str, Enum):
    """What kind of thing happened."""

    DEAL = "deal"
    MOVE = "move"
    BOUT_END = "bout_end"
    GAME_OVER = "game_over"


@dataclass(frozen=True, slots=True)
class DrawRecord:
    """Cards a player drew from the talon. Ground truth, not public."""

    player: int
    cards: tuple[Card, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"player": self.player, "cards": [c.short for c in self.cards]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DrawRecord:
        return cls(
            int(data["player"]), tuple(Card.parse(c) for c in data["cards"])
        )


@dataclass(frozen=True, slots=True)
class GameEvent:
    """One observable step of a game.

    Attributes:
        kind: what happened.
        player: the actor, where one applies.
        move: the move played, for ``MOVE`` events.
        revealed: cards that became publicly visible at this step.
        discarded: cards permanently removed from play (*bito*).
        taken: cards picked up by the defender.
        taken_by: who picked them up.
        draws: talon draws, in the order they happened. Ground truth.
        durak: the loser, on ``GAME_OVER``; ``None`` means a draw.
    """

    kind: EventKind
    player: int | None = None
    move: Move | None = None
    revealed: tuple[Card, ...] = ()
    discarded: tuple[Card, ...] = ()
    taken: tuple[Card, ...] = ()
    taken_by: int | None = None
    draws: tuple[DrawRecord, ...] = ()
    durak: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind.value}
        if self.player is not None:
            data["player"] = self.player
        if self.move is not None:
            data["move"] = self.move.to_dict()
        if self.revealed:
            data["revealed"] = [c.short for c in self.revealed]
        if self.discarded:
            data["discarded"] = [c.short for c in self.discarded]
        if self.taken:
            data["taken"] = [c.short for c in self.taken]
        if self.taken_by is not None:
            data["taken_by"] = self.taken_by
        if self.draws:
            data["draws"] = [d.to_dict() for d in self.draws]
        if self.durak is not None:
            data["durak"] = self.durak
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameEvent:
        return cls(
            kind=EventKind(data["kind"]),
            player=data.get("player"),
            move=move_from_dict(data["move"]) if "move" in data else None,
            revealed=tuple(Card.parse(c) for c in data.get("revealed", ())),
            discarded=tuple(Card.parse(c) for c in data.get("discarded", ())),
            taken=tuple(Card.parse(c) for c in data.get("taken", ())),
            taken_by=data.get("taken_by"),
            draws=tuple(DrawRecord.from_dict(d) for d in data.get("draws", ())),
            durak=data.get("durak"),
        )


@dataclass(frozen=True, slots=True)
class HistoryNode:
    """One link of the persistent history list, newest first."""

    event: GameEvent
    parent: HistoryNode | None = None
    length: int = field(default=1)


def push_event(history: HistoryNode | None, event: GameEvent) -> HistoryNode:
    """Return a new history with ``event`` appended. O(1), shares the tail."""
    return HistoryNode(
        event=event,
        parent=history,
        length=1 if history is None else history.length + 1,
    )
