"""Information-set nodes: one node per decision point, statistics shared.

This is where the Phase 7.3.3 defect is actually fixed. PIMC builds a
separate tree per sampled world, so two observationally identical worlds
get independently optimal policies — the observer ends up planning
``attack 8♠`` in one and ``end attack`` in the other, a choice it could
only make by knowing which world it is in.

Here a node is keyed by :class:`~durakfish.ai.ismcts.key.InformationSetKey`
alone. Every world consistent with that key reaches the *same* node
object and updates the *same* counters:

    world A ─┐
    world B ─┼──► InformationSetNode K ──► shared N, W, Q ──► one policy
    world C ─┘

Only the observer gets nodes
----------------------------
Single-observer ISMCTS: nodes exist for the observer's decision points
only. The opponent is simulated inside the determinized world and keeps
no statistics.

That is not merely a simplification, it removes a problem. At an
observer's decision point the legal moves depend on the observer's own
hand and the public table, so they are the same in every compatible
world — measured across 2,719 positions, all sampled worlds gave
identical move sets. At an opponent's decision point they differ, because
the opponent's options depend on cards the observer cannot see. Keeping
nodes only where the action set is world-independent means an edge means
the same thing to every world that reaches it, and the "action legal in
one world but not another" question never arises at a node.

Statistics
----------
``N(s)`` visits to the node; ``N(s,a)`` visits to an edge; ``W(s,a)``
summed value; ``Q(s,a) = W/N``. Every value is from the **root player's**
point of view, matching the frozen convention used by the evaluator and
the terminal scores, so no sign transformation happens anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from durakfish.ai.evaluation import Score
from durakfish.ai.ismcts.key import InformationSetKey
from durakfish.ai.tiebreak import canonical_move_key
from durakfish.exceptions import DurakFishError
from durakfish.game.moves import Move

__all__ = ["Edge", "ISMCTSError", "InformationSetNode", "NodeRegistry"]


class ISMCTSError(DurakFishError):
    """The search was asked for something inconsistent."""


@dataclass(slots=True)
class Edge:
    """Statistics for one action out of an information-set node.

    Attributes:
        move: the action. Edges are keyed by
            :func:`~durakfish.ai.tiebreak.canonical_move_key`, never by
            object identity, so equal moves arriving from different
            worlds share one edge.
        visits: ``N(s,a)``.
        total_value: ``W(s,a)``, summed from the root player's viewpoint.
    """

    move: Move
    visits: int = 0
    total_value: Score = 0.0

    @property
    def mean_value(self) -> Score:
        """``Q(s,a)``. Zero for an unvisited edge, which selection treats
        as "must try first" rather than as an estimate of zero."""
        return self.total_value / self.visits if self.visits else 0.0

    def record(self, value: Score) -> None:
        self.visits += 1
        self.total_value += value

    def __repr__(self) -> str:
        return f"Edge({self.move}, N={self.visits}, Q={self.mean_value:.3f})"


@dataclass(slots=True)
class InformationSetNode:
    """One decision point of one observer, with statistics shared by worlds."""

    key: InformationSetKey
    visits: int = 0
    edges: dict[tuple[int, int, int], Edge] = field(default_factory=dict)

    @property
    def owner(self) -> int:
        return self.key.owner

    def ensure_edges(self, moves: tuple[Move, ...]) -> None:
        """Create any edge that does not exist yet.

        Idempotent, and safe to call from every world that reaches the
        node: an edge already present keeps its accumulated statistics.
        """
        for move in moves:
            identity = canonical_move_key(move)
            if identity not in self.edges:
                self.edges[identity] = Edge(move=move)

    def edge_for(self, move: Move) -> Edge:
        """The edge for ``move``.

        Raises:
            ISMCTSError: if the edge was never created. Reaching this
                means an action was played that the node never offered,
                which is a bug worth surfacing rather than papering over
                by creating the edge late.
        """
        identity = canonical_move_key(move)
        edge = self.edges.get(identity)
        if edge is None:
            raise ISMCTSError(f"{move} is not an edge of {self.key}")
        return edge

    def record_visit(self) -> None:
        self.visits += 1

    def uct_select(self, moves: tuple[Move, ...], exploration: float) -> Move:
        """Pick a move by UCT over the node's shared statistics.

        Unvisited edges come first, in canonical order, so the choice is
        deterministic and does not depend on which world happened to
        arrive first. Ties on the UCT score break the same way.

        Args:
            moves: actions legal in the current determinized world. At an
                observer node this is world-independent, but it is passed
                in rather than assumed so the assumption stays visible.
            exploration: the UCT constant.

        Raises:
            ISMCTSError: if no moves were offered.
        """
        if not moves:
            raise ISMCTSError(f"no moves offered at {self.key}")
        ordered = sorted(moves, key=canonical_move_key)

        unvisited = [m for m in ordered if self.edge_for(m).visits == 0]
        if unvisited:
            return unvisited[0]

        parent = max(self.visits, 1)
        best, best_score = ordered[0], -math.inf
        for move in ordered:
            edge = self.edge_for(move)
            score = edge.mean_value + exploration * math.sqrt(
                math.log(parent) / edge.visits
            )
            if score > best_score:
                best, best_score = move, score
        return best

    def best_move(self) -> Move | None:
        """The most-visited action, the standard MCTS recommendation.

        Visit count rather than value: it is the more stable estimator,
        and it is what the search actually committed effort to. Ties break
        canonically.
        """
        visited = [edge for edge in self.edges.values() if edge.visits]
        if not visited:
            return None
        return max(
            visited, key=lambda e: (e.visits, [-k for k in canonical_move_key(e.move)])
        ).move

    def __repr__(self) -> str:
        return f"InformationSetNode({self.key}, N={self.visits}, edges={len(self.edges)})"


class NodeRegistry:
    """Canonical store: one node per information-set key.

    The registry *is* the mechanism that makes statistics shared. Two
    worlds reaching the same key are handed the same object, so there is
    no code path in which they could accumulate separately.
    """

    __slots__ = ("_nodes",)

    def __init__(self) -> None:
        self._nodes: dict[InformationSetKey, InformationSetNode] = {}

    def get_or_create(self, key: InformationSetKey) -> InformationSetNode:
        node = self._nodes.get(key)
        if node is None:
            node = InformationSetNode(key=key)
            self._nodes[key] = node
        return node

    def get(self, key: InformationSetKey) -> InformationSetNode | None:
        return self._nodes.get(key)

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, key: object) -> bool:
        return key in self._nodes

    def __repr__(self) -> str:
        return f"NodeRegistry({len(self._nodes)} information sets)"
