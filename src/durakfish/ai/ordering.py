"""Deterministic move ordering for search.

Different from ``ai/tiebreak.py``, and the difference is worth stating.
:mod:`durakfish.ai.tiebreak` provides an *arbitrary but fixed* total order,
used so that ties and sampling cannot depend on list order. This module
orders moves by how **promising** they look, so that alpha-beta meets good
moves first and cuts off sooner.

Ordering can change how many nodes are visited. It must never change the
result, and ``tests/test_search.py`` checks exactly that by searching the
same positions under several orderings and comparing scores and chosen
moves.

Determinism
-----------
Every ordering ends with :func:`~durakfish.ai.tiebreak.canonical_move_key`,
so no comparison can fall through to object identity, hashing or the order
the rules engine happened to emit. Permuting the input cannot change the
output.

Scope
-----
A single static ordering, computed from the node in front of it. No
history heuristic, no killer moves, no transposition-table hints, nothing
learned — those need either a transposition table or search state carried
between nodes, and Phase 5 is about getting the tree right first.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from durakfish.ai.searchnode import OpponentAction, SearchNode
from durakfish.ai.tiebreak import canonical_move_key
from durakfish.game.moves import AttackMove, DefenseMove, EndAttack, Move, TakeCards

__all__ = ["ORDERINGS", "order_opponent_actions", "order_self_moves"]

#: Opponent branches, most-constraining first. Taking hands the searcher a
#: free choice of throw-ins, so it tends to be the branch that resolves
#: fastest; beating keeps the bout alive.
_OPPONENT_RANK = {
    OpponentAction.BEAT: 0,
    OpponentAction.TAKE: 1,
    OpponentAction.ADD: 2,
    OpponentAction.END: 3,
}


def _promise(node: SearchNode, move: Move) -> tuple[int, int]:
    """Static promise of a move. Lower sorts earlier.

    The ranking, in words:

    * spend cheap cards before expensive ones, and non-trumps before
      trumps, reusing the cost notion the Phase 4 baseline established;
    * try ending a bout before committing further cards to it;
    * try taking last, since it is usually the concession.
    """
    if isinstance(move, AttackMove):
        card = move.card
        return (0, (1 if card.suit == node.trump else 0) * 100 + card.code)
    if isinstance(move, DefenseMove):
        card = move.defending_card
        return (0, (1 if card.suit == node.trump else 0) * 100 + card.code)
    if isinstance(move, EndAttack):
        return (1, 0)
    if isinstance(move, TakeCards):
        return (2, 0)
    return (3, 0)  # pragma: no cover - no other move types exist


def order_self_moves(
    node: SearchNode, moves: Iterable[Move], *, ordering: str = "static"
) -> tuple[Move, ...]:
    """Order the searcher's moves for search.

    Args:
        node: the position, used for the trump suit.
        moves: the legal moves.
        ordering: ``"static"`` for the promise heuristic, or
            ``"canonical"`` for the fixed arbitrary order. The second
            exists so tests can prove that ordering changes node counts
            and nothing else.

    Raises:
        ValueError: for an unknown ordering name.
    """
    if ordering == "canonical":
        return tuple(sorted(moves, key=canonical_move_key))
    if ordering != "static":
        raise ValueError(
            f"unknown ordering {ordering!r}; available: {', '.join(sorted(ORDERINGS))}"
        )
    return tuple(
        sorted(moves, key=lambda m: (_promise(node, m), canonical_move_key(m)))
    )


def order_opponent_actions(
    node: SearchNode, actions: Sequence[OpponentAction], *, ordering: str = "static"
) -> tuple[OpponentAction, ...]:
    """Order opponent branches deterministically.

    ``canonical`` falls back to the declaration order of the action enum,
    which is fixed in source and therefore stable across processes.
    """
    if ordering == "canonical":
        return tuple(sorted(actions, key=lambda a: a.value))
    return tuple(sorted(actions, key=lambda a: (_OPPONENT_RANK[a], a.value)))


#: Ordering strategies a search configuration may name.
ORDERINGS: frozenset[str] = frozenset({"static", "canonical"})
