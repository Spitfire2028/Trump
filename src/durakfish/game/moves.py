"""Typed move objects.

Moves are immutable, hashable value objects. There is deliberately no
string-based move format: the search engine should never parse text, and
a typed move carries everything ``apply_move`` needs to act without
guessing.

The four move types are exactly the decisions a Podkidnoy player makes:

======================  ====================================================
:class:`AttackMove`     attacker puts a card on the table
:class:`DefenseMove`    defender beats one specific attacking card
:class:`TakeCards`      defender announces they will pick the table up
:class:`EndAttack`      attacker stops adding cards; the bout resolves
======================  ====================================================

``TakeCards`` and ``EndAttack`` carry no data, so the singletons
:data:`TAKE` and :data:`END_ATTACK` exist to avoid pointless allocation
in search. Equality is by value, so freshly constructed instances compare
equal to the singletons.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from durakfish.cards import Card

__all__ = [
    "END_ATTACK",
    "TAKE",
    "AttackMove",
    "DefenseMove",
    "EndAttack",
    "Move",
    "MoveType",
    "TakeCards",
    "move_from_dict",
]


class MoveType(IntEnum):
    """Discriminator for move objects, for fast switching and encoding."""

    ATTACK = 0
    DEFEND = 1
    TAKE = 2
    END_ATTACK = 3


class Move:
    """Abstract base for all moves."""

    __slots__ = ()

    #: Filled in by each concrete subclass.
    type: MoveType

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly encoding."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class AttackMove(Move):
    """Put ``card`` on the table as an attack."""

    card: Card

    type = MoveType.ATTACK

    def to_dict(self) -> dict[str, Any]:
        return {"type": "attack", "card": self.card.short}

    def __str__(self) -> str:
        return f"attack {self.card}"


@dataclass(frozen=True, slots=True)
class DefenseMove(Move):
    """Beat ``attacking_card`` (already on the table) with ``defending_card``.

    Both cards are named explicitly so the move is unambiguous even when
    several attacks are unresolved.
    """

    attacking_card: Card
    defending_card: Card

    type = MoveType.DEFEND

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "defend",
            "attacking_card": self.attacking_card.short,
            "defending_card": self.defending_card.short,
        }

    def __str__(self) -> str:
        return f"defend {self.attacking_card} with {self.defending_card}"


@dataclass(frozen=True, slots=True)
class TakeCards(Move):
    """Defender announces they will take the table.

    This is an action, not a card play. In standard Podkidnoy the bout
    does not end here: the attacker may still throw in matching ranks.
    """

    type = MoveType.TAKE

    def to_dict(self) -> dict[str, Any]:
        return {"type": "take"}

    def __str__(self) -> str:
        return "take"


@dataclass(frozen=True, slots=True)
class EndAttack(Move):
    """Attacker stops adding cards, resolving the bout.

    After a complete defence this is *bito* (the table is discarded);
    during a take it hands the table to the defender.
    """

    type = MoveType.END_ATTACK

    def to_dict(self) -> dict[str, Any]:
        return {"type": "end_attack"}

    def __str__(self) -> str:
        return "end attack"


#: Shared instances for the two data-free moves.
TAKE = TakeCards()
END_ATTACK = EndAttack()


def move_from_dict(data: dict[str, Any]) -> Move:
    """Rebuild a move from :meth:`Move.to_dict` output."""
    kind = data.get("type")
    if kind == "attack":
        return AttackMove(Card.parse(data["card"]))
    if kind == "defend":
        return DefenseMove(
            Card.parse(data["attacking_card"]), Card.parse(data["defending_card"])
        )
    if kind == "take":
        return TAKE
    if kind == "end_attack":
        return END_ATTACK
    raise ValueError(f"unknown move encoding: {data!r}")
