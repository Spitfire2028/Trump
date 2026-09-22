"""Minimax and alpha-beta over the search-node abstraction.

Depth semantics
---------------
``depth`` counts **plies**: individual decisions, by either side.

============  ==========================================================
``depth``     Meaning
============  ==========================================================
0             Evaluate the root immediately. No move is chosen; the
              search reports the position's score. Rejected at the root
              by :class:`SearchConfig`, which requires depth >= 1.
1             Expand the root's moves and evaluate each child. The
              searcher looks one decision ahead.
2             Expand the root's moves and the reply to each. Because
              opponent replies are action classes, this is the shallowest
              depth at which a minimising node appears.
n             ``n`` decisions from the root.
============  ==========================================================

A branch stops early, whatever the remaining depth, when it reaches a
leaf: a resolved bout or an unknown-card frontier. Those are properties of
the position, not of the budget.

Maximising and minimising
-------------------------
Every score is from the searcher's point of view. Nodes where the searcher
moves maximise; nodes where the opponent moves minimise. Leaves are
evaluated, and proven terminals are scored exactly and never overridden by
the heuristic.

Two implementations, on purpose
-------------------------------
:func:`minimax` is written for obviousness — full width, no pruning.
:func:`alpha_beta` is written for speed. They must agree exactly on both
score and chosen move for every position, at every depth, and the test
suite compares them over thousands of positions. The plain version earns
its keep as the oracle; deleting it would leave the fast one unchecked.

Purity
------
Nodes are immutable and transitions return new nodes, so nothing the
search touches can be modified. The searcher's information set, the legal
moves it was handed, and the root node all come back unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from durakfish.ai.evaluation import (
    MATE_MARGIN,
    Evaluator,
    Score,
    get_evaluator,
    terminal_score,
)
from durakfish.ai.ordering import ORDERINGS, order_opponent_actions, order_self_moves
from durakfish.ai.searchnode import Actor, OpponentAction, SearchNode
from durakfish.ai.tiebreak import canonical_move_key
from durakfish.exceptions import DurakFishError
from durakfish.game.moves import Move

__all__ = [
    "SearchConfig",
    "SearchResult",
    "SearchStats",
    "alpha_beta",
    "minimax",
    "search",
]

_INFINITY = float("inf")


class SearchConfigurationError(DurakFishError, ValueError):
    """A search configuration describes a search that cannot be run."""


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """How to search. Invalid configurations fail loudly at construction.

    Attributes:
        depth: plies to look ahead. Must be at least 1.
        algorithm: ``"alpha_beta"`` or ``"minimax"``. The latter exists as
            a reference implementation and is much slower.
        ordering: move-ordering strategy, see :mod:`durakfish.ai.ordering`.
        evaluator: name of the evaluator to score leaves with.
        collect_statistics: gather node counts. Statistics never influence
            the search; a test compares results with it on and off.
    """

    depth: int = 3
    algorithm: str = "alpha_beta"
    ordering: str = "static"
    evaluator: str = "baseline"
    collect_statistics: bool = True

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise SearchConfigurationError(
                f"depth must be at least 1, got {self.depth}"
            )
        if self.algorithm not in ("alpha_beta", "minimax"):
            raise SearchConfigurationError(
                f"unknown algorithm {self.algorithm!r}; "
                "available: alpha_beta, minimax"
            )
        if self.ordering not in ORDERINGS:
            raise SearchConfigurationError(
                f"unknown ordering {self.ordering!r}; "
                f"available: {', '.join(sorted(ORDERINGS))}"
            )
        # Raises for an unknown evaluator name.
        get_evaluator(self.evaluator)


@dataclass(slots=True)
class SearchStats:
    """Counters describing one search. Purely observational.

    Attributes:
        nodes: nodes visited, including the root.
        leaves: nodes scored by the evaluator.
        terminals: leaves that were proven wins, losses or draws.
        cutoffs: alpha-beta pruning events. Always 0 for plain minimax.
        max_depth: deepest ply reached.
        root_moves: how many moves were considered at the root.
    """

    nodes: int = 0
    leaves: int = 0
    terminals: int = 0
    cutoffs: int = 0
    max_depth: int = 0
    root_moves: int = 0
    enabled: bool = field(default=True)

    def visit(self, ply: int) -> None:
        if self.enabled:
            self.nodes += 1
            if ply > self.max_depth:
                self.max_depth = ply

    def leaf(self, *, terminal: bool) -> None:
        if self.enabled:
            self.leaves += 1
            if terminal:
                self.terminals += 1

    def cutoff(self) -> None:
        if self.enabled:
            self.cutoffs += 1

    @property
    def pruning_ratio(self) -> float:
        """Share of visited nodes that ended in a cutoff. Reporting only."""
        return self.cutoffs / self.nodes if self.nodes else 0.0


@dataclass(frozen=True, slots=True)
class SearchResult:
    """What a search found.

    Attributes:
        move: the chosen move, always drawn from the root moves supplied.
            ``None`` only when there were no moves to choose from.
        score: the value of the position, from the searcher's point of view.
        stats: counters for the search that produced this.
        depth: the depth actually requested.
    """

    move: Move | None
    score: Score
    stats: SearchStats
    depth: int

    @property
    def is_proven(self) -> bool:
        """The score is an exact terminal value, not a heuristic guess.

        Exact *over the abstract tree*. Whether it is also true of the
        real game depends on the sign — see the two properties below,
        which is why this one should rarely be used on its own.
        """
        return abs(self.score) >= MATE_MARGIN

    @property
    def is_proven_win(self) -> bool:
        """A genuine forced win in the real game.

        Sound. The abstraction gives the opponent a superset of their real
        options and the searcher a subset of its own, so an abstract value
        is a lower bound on the true value whenever no heuristic leaf
        contributes to it. A win that survives that pessimism is real.
        Verified against a perfect-information oracle in
        ``tests/test_abstraction_audit.py``: 358 of 358 claims correct.
        """
        return self.score >= MATE_MARGIN

    @property
    def is_lower_bound_loss(self) -> bool:
        """The abstraction sees no escape. **This is not a proven loss.**

        The same pessimism that makes wins sound makes losses unsound: the
        opponent is credited with replies they may not hold the cards for.
        Measured against the oracle, 60 of 221 such claims were positions
        the searcher could actually have won or drawn. Named to make
        mistaking it for a proof difficult.
        """
        return self.score <= -MATE_MARGIN

    def __repr__(self) -> str:
        return (
            f"SearchResult({self.move}, score={self.score:.2f}, "
            f"depth={self.depth}, nodes={self.stats.nodes})"
        )


# ----------------------------------------------------------------------
# Leaf scoring, shared by both implementations
# ----------------------------------------------------------------------
def _score_leaf(
    node: SearchNode, ply: int, evaluator: Evaluator, stats: SearchStats
) -> Score:
    outcome = node.outcome()
    if outcome is not None:
        stats.leaf(terminal=True)
        return terminal_score(outcome, ply)
    stats.leaf(terminal=False)
    return evaluator.evaluate(node, ply)


def _children(
    node: SearchNode, ordering: str
) -> Iterator[tuple[Move | OpponentAction, SearchNode]]:
    """Yield ``(edge, child)`` pairs in a deterministic order.

    The edge type differs by node: a real :class:`Move` where the searcher
    decides, an abstract :class:`OpponentAction` where the opponent does.
    Keeping them distinct means an abstract action can never be mistaken
    for something the rules engine would accept.
    """
    if node.to_move is Actor.SELF:
        for move in order_self_moves(node, node.self_moves(), ordering=ordering):
            yield move, node.after_self(move)
    else:
        for action in order_opponent_actions(
            node, node.opponent_actions(), ordering=ordering
        ):
            yield action, node.after_opponent(action)


# ----------------------------------------------------------------------
# Plain minimax — the reference implementation
# ----------------------------------------------------------------------
def _minimax_value(
    node: SearchNode,
    depth: int,
    ply: int,
    evaluator: Evaluator,
    ordering: str,
    stats: SearchStats,
) -> Score:
    stats.visit(ply)
    if node.is_leaf or depth == 0:
        return _score_leaf(node, ply, evaluator, stats)

    children = list(_children(node, ordering))
    if not children:
        # Nobody can move and the position is not flagged as a leaf: score
        # it where it stands rather than inventing a result.
        return _score_leaf(node, ply, evaluator, stats)

    maximising = node.to_move is Actor.SELF
    best = -_INFINITY if maximising else _INFINITY
    for _edge, child in children:
        value = _minimax_value(child, depth - 1, ply + 1, evaluator, ordering, stats)
        best = max(best, value) if maximising else min(best, value)
    return best


# ----------------------------------------------------------------------
# Alpha-beta — the same tree, pruned
# ----------------------------------------------------------------------
def _alpha_beta_value(
    node: SearchNode,
    depth: int,
    ply: int,
    alpha: Score,
    beta: Score,
    evaluator: Evaluator,
    ordering: str,
    stats: SearchStats,
) -> Score:
    stats.visit(ply)
    if node.is_leaf or depth == 0:
        return _score_leaf(node, ply, evaluator, stats)

    children = list(_children(node, ordering))
    if not children:
        return _score_leaf(node, ply, evaluator, stats)

    if node.to_move is Actor.SELF:
        best = -_INFINITY
        for _edge, child in children:
            best = max(
                best,
                _alpha_beta_value(
                    child, depth - 1, ply + 1, alpha, beta, evaluator, ordering, stats
                ),
            )
            alpha = max(alpha, best)
            if beta <= alpha:
                stats.cutoff()
                break
        return best

    best = _INFINITY
    for _edge, child in children:
        best = min(
            best,
            _alpha_beta_value(
                child, depth - 1, ply + 1, alpha, beta, evaluator, ordering, stats
            ),
        )
        beta = min(beta, best)
        if beta <= alpha:
            stats.cutoff()
            break
    return best


# ----------------------------------------------------------------------
# Root
# ----------------------------------------------------------------------
def _root(
    node: SearchNode,
    config: SearchConfig,
    root_moves: Sequence[Move] | None,
    pruning: bool,
) -> SearchResult:
    """Search the root, choosing among ``root_moves``.

    Restricting the root to a supplied list is what lets an agent honour
    the contract that it returns one of the moves it was given: a move
    that was not offered can never be selected, whatever the search thinks
    of it.
    """
    evaluator = get_evaluator(config.evaluator)
    stats = SearchStats(enabled=config.collect_statistics)
    stats.visit(0)

    candidates = (
        tuple(root_moves) if root_moves is not None else node.self_moves()
    )
    if not candidates:
        return SearchResult(None, _score_leaf(node, 0, evaluator, stats), stats,
                            config.depth)

    ordered = order_self_moves(node, candidates, ordering=config.ordering)
    if config.collect_statistics:
        stats.root_moves = len(ordered)

    best_move: Move | None = None
    best_score = -_INFINITY
    alpha = -_INFINITY

    for move in ordered:
        child = node.after_self(move)
        if pruning:
            value = _alpha_beta_value(
                child, config.depth - 1, 1, alpha, _INFINITY,
                evaluator, config.ordering, stats,
            )
        else:
            value = _minimax_value(
                child, config.depth - 1, 1, evaluator, config.ordering, stats
            )
        # Strictly greater, so an earlier move wins a tie. Combined with a
        # deterministic ordering this makes the choice a pure function of
        # the position, never of list order or object identity.
        if best_move is None or value > best_score:
            best_score, best_move = value, move
        if pruning:
            alpha = max(alpha, best_score)

    return SearchResult(best_move, best_score, stats, config.depth)


def minimax(
    node: SearchNode,
    config: SearchConfig | None = None,
    *,
    root_moves: Sequence[Move] | None = None,
) -> SearchResult:
    """Full-width minimax. The reference implementation; no pruning."""
    return _root(node, config or SearchConfig(), root_moves, pruning=False)


def alpha_beta(
    node: SearchNode,
    config: SearchConfig | None = None,
    *,
    root_moves: Sequence[Move] | None = None,
) -> SearchResult:
    """Alpha-beta over the same tree. Must agree with :func:`minimax`."""
    return _root(node, config or SearchConfig(), root_moves, pruning=True)


def search(
    node: SearchNode,
    config: SearchConfig | None = None,
    *,
    root_moves: Sequence[Move] | None = None,
) -> SearchResult:
    """Run whichever algorithm the configuration names."""
    settings = config or SearchConfig()
    if settings.algorithm == "minimax":
        return minimax(node, settings, root_moves=root_moves)
    return alpha_beta(node, settings, root_moves=root_moves)


# Referenced so the canonical key's role in root tie-breaking is discoverable
# from this module.
_ = canonical_move_key
