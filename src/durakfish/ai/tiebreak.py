"""A canonical total order on moves.

Two agent requirements need a stable ordering that does not depend on how
a legal-move list happened to be built:

* **Deterministic tie-breaking.** When a policy scores two moves equally,
  something has to break the tie. Object identity, memory address, hash
  order and container iteration order are all forbidden — they vary
  between runs and processes. A total order derived from the moves' own
  contents does not.
* **Order-independent randomness.** A bot that samples by index would
  otherwise give different answers for the same move *set* presented in a
  different order. Sorting first makes "uniform over the legal moves" a
  statement about the set, not about the list.

This is **not** search move ordering. Nothing here ranks moves by how
good they are; the order is arbitrary but fixed. Ordering moves by
promise, to make alpha-beta cut off sooner, is a different job for a
later phase and belongs in its own module.

The order is: move type first (attack, defend, take, end attack — the
declaration order of :class:`~durakfish.game.moves.MoveType`), then the
cards involved by their dense integer code. Card codes are stable across
processes and platforms, so the resulting order is too.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from durakfish.game.moves import AttackMove, DefenseMove, Move

__all__ = ["canonical_move_key", "canonical_order"]

#: Sorts after any real card code, so card-less moves order consistently.
_NO_CARD = -1


def canonical_move_key(move: Move) -> tuple[int, int, int]:
    """A stable sort key derived only from the move's own contents.

    Total on the four move types: no two distinct moves share a key.
    Contains no object identity, no hashing and nothing process-dependent.
    """
    if isinstance(move, AttackMove):
        return (int(move.type), move.card.code, _NO_CARD)
    if isinstance(move, DefenseMove):
        return (
            int(move.type),
            move.attacking_card.code,
            move.defending_card.code,
        )
    # TakeCards and EndAttack carry no cards and are singletons by type.
    return (int(move.type), _NO_CARD, _NO_CARD)


def canonical_order(moves: Iterable[Move]) -> tuple[Move, ...]:
    """Return ``moves`` sorted into the canonical order.

    Sorting is stable and depends only on move contents, so the same set
    of moves always produces the same sequence regardless of the order it
    arrived in.
    """
    return tuple(sorted(moves, key=canonical_move_key))


def is_canonically_ordered(moves: Sequence[Move]) -> bool:
    """True when ``moves`` is already in canonical order. Used by tests."""
    keys = [canonical_move_key(m) for m in moves]
    return keys == sorted(keys)
