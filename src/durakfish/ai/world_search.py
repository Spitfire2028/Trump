"""Perfect-information search over one determinized world.

Why a second search exists
--------------------------
Phase 5's search is an *information-set* search. Its ``SearchNode`` holds
the opponent as a **count** plus abstract action classes, because the
observer cannot see their cards. That was the right design for what it
had, and it is deliberately incapable of holding a hidden hand.

The consequence matters here. Routing a determinized position through
``observe()`` to build a ``SearchNode`` re-redacts it: the sampled world
is discarded and two different worlds become observationally identical
again. Measured directly, two distinct sampled worlds produced a
byte-identical Phase 5 result — same move, same score. Aggregating across
worlds that way would be averaging the same number.

So this module searches a different space:

============  ============================================================
Phase 5       ``InformationSet -> SearchNode -> SearchBot``
              abstract opponent, bout-local, conservative
Phase 7.3.2a  ``DeterminizedPosition -> card-level moves -> frozen rules``
              real opponent cards *within one hypothesis*, full game
============  ============================================================

Both are correct for their own space. Neither replaces the other, and
``SearchBot`` is untouched.

What it may see
---------------
Everything in the hypothetical position, opponent hand included. That is
legitimate precisely because the position is a *hypothesis* sampled from
the observer's own information (Phase 7.2), not the real game. The real
hidden state is never an input and never reachable.

Value convention
----------------
Every value is from the **root player's** point of view, at every node,
whoever is to move. Positive is good for the root player. Nodes where the
root player moves maximise; nodes where the opponent moves minimise.
Evaluations are therefore not side-to-move-relative, which removes a
whole category of sign errors — the same convention Phase 5 uses.

Depth
-----
``depth`` counts **plies**: individual decisions by either side. Matching
Phase 5, ``depth=0`` scores the position where it stands, ``depth=1``
expands the root's moves and evaluates each child, and ``depth=n`` looks
``n`` decisions ahead. :class:`~durakfish.ai.search.SearchConfig` requires
at least 1 at the root; :func:`evaluate_position` is the depth-0 case,
exposed separately so it can be used and tested on its own.

Terminals are detected **before** the depth cutoff and come from the
frozen Phase 2 engine, never re-derived here.

Proven results
--------------
Unlike Phase 5, a proven result here is exact **within the hypothesis**:
the search sees every card, so a win it proves is a real forced win *in
that world*. It says nothing about the real game, where that world may
not be the true one. The distinction is in the property names.

One world only
--------------
This module evaluates exactly one world. **No world aggregation exists
yet** — no sampling of several, no averaging, no voting, no expected
values, no move selection across worlds. Combining per-world results is
Phase 7.3.2b, and it needs this primitive to be trustworthy first.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from durakfish.ai.determinized import DeterminizedPosition
from durakfish.ai.evaluation import (
    MATE_MARGIN,
    Evaluator,
    Score,
    get_evaluator,
    terminal_score,
)
from durakfish.ai.ordering import order_self_moves
from durakfish.ai.search import SearchConfig, SearchStats
from durakfish.ai.searchnode import Outcome
from durakfish.game.moves import Move

__all__ = [
    "WorldSearchResult",
    "alpha_beta_world",
    "evaluate_position",
    "minimax_world",
    "search_world",
]

_INFINITY = float("inf")


@dataclass(frozen=True, slots=True)
class WorldSearchResult:
    """What a search of one hypothetical world found.

    Attributes:
        move: the chosen move, always drawn from the root moves supplied.
            ``None`` only when the root had no moves — a terminal position.
        value: the position's value from the root player's point of view.
        stats: counters for the search that produced this.
        depth: the depth requested.
        root_player: whose point of view ``value`` takes.
    """

    move: Move | None
    value: Score
    stats: SearchStats
    depth: int
    root_player: int

    @property
    def is_proven_win(self) -> bool:
        """A forced win **in this hypothetical world**.

        Exact, because the search sees every card in the world. It says
        nothing about the real game unless this world happens to be the
        true one — which is why the naming differs from Phase 5's
        deliberately cautious ``is_lower_bound_loss``.
        """
        return self.value >= MATE_MARGIN

    @property
    def is_proven_loss(self) -> bool:
        """A forced loss in this hypothetical world. Also exact here."""
        return self.value <= -MATE_MARGIN

    def __repr__(self) -> str:
        return (
            f"WorldSearchResult({self.move}, value={self.value:.2f}, "
            f"depth={self.depth}, P{self.root_player}, "
            f"nodes={self.stats.nodes})"
        )


# ----------------------------------------------------------------------
# Terminal and leaf scoring
# ----------------------------------------------------------------------
def _terminal_value(
    position: DeterminizedPosition, root_player: int, ply: int
) -> Score:
    """Score a finished position, using the frozen engine's own verdict."""
    loser = position.durak
    if loser is None:
        outcome = Outcome.DRAW
    elif loser == root_player:
        outcome = Outcome.LOSS
    else:
        outcome = Outcome.WIN
    return terminal_score(outcome, ply)


def evaluate_position(
    position: DeterminizedPosition,
    root_player: int,
    *,
    ply: int = 0,
    evaluator: Evaluator | None = None,
) -> Score:
    """Score a position without searching: the ``depth = 0`` case.

    Terminal positions get their exact value; anything else is handed to
    the evaluator. Reuses Phase 5's ``BaselineEvaluator`` by default
    rather than introducing a second heuristic that could drift from it.
    """
    if position.is_terminal:
        return _terminal_value(position, root_player, ply)
    scorer = evaluator if evaluator is not None else get_evaluator("baseline")
    return scorer.evaluate(position.evaluation_node(root_player), ply)


# ----------------------------------------------------------------------
# Plain minimax — the reference implementation
# ----------------------------------------------------------------------
def _minimax_value(
    position: DeterminizedPosition,
    depth: int,
    ply: int,
    root_player: int,
    evaluator: Evaluator,
    ordering: str,
    stats: SearchStats,
) -> Score:
    stats.visit(ply)
    if position.is_terminal:
        stats.leaf(terminal=True)
        return _terminal_value(position, root_player, ply)
    if depth == 0:
        stats.leaf(terminal=False)
        return evaluate_position(position, root_player, ply=ply, evaluator=evaluator)

    moves = _ordered_moves(position, root_player, ordering)
    if not moves:  # pragma: no cover - the engine always offers a move
        stats.leaf(terminal=False)
        return evaluate_position(position, root_player, ply=ply, evaluator=evaluator)

    maximising = position.current_player == root_player
    best = -_INFINITY if maximising else _INFINITY
    for move in moves:
        value = _minimax_value(
            position.apply(move), depth - 1, ply + 1, root_player,
            evaluator, ordering, stats,
        )
        best = max(best, value) if maximising else min(best, value)
    return best


# ----------------------------------------------------------------------
# Alpha-beta — the same tree, pruned
# ----------------------------------------------------------------------
def _alpha_beta_value(
    position: DeterminizedPosition,
    depth: int,
    ply: int,
    alpha: Score,
    beta: Score,
    root_player: int,
    evaluator: Evaluator,
    ordering: str,
    stats: SearchStats,
) -> Score:
    stats.visit(ply)
    if position.is_terminal:
        stats.leaf(terminal=True)
        return _terminal_value(position, root_player, ply)
    if depth == 0:
        stats.leaf(terminal=False)
        return evaluate_position(position, root_player, ply=ply, evaluator=evaluator)

    moves = _ordered_moves(position, root_player, ordering)
    if not moves:  # pragma: no cover - the engine always offers a move
        stats.leaf(terminal=False)
        return evaluate_position(position, root_player, ply=ply, evaluator=evaluator)

    if position.current_player == root_player:
        best = -_INFINITY
        for move in moves:
            best = max(
                best,
                _alpha_beta_value(
                    position.apply(move), depth - 1, ply + 1, alpha, beta,
                    root_player, evaluator, ordering, stats,
                ),
            )
            alpha = max(alpha, best)
            if beta <= alpha:
                stats.cutoff()
                break
        return best

    best = _INFINITY
    for move in moves:
        best = min(
            best,
            _alpha_beta_value(
                position.apply(move), depth - 1, ply + 1, alpha, beta,
                root_player, evaluator, ordering, stats,
            ),
        )
        beta = min(beta, best)
        if beta <= alpha:
            stats.cutoff()
            break
    return best


def _ordered_moves(
    position: DeterminizedPosition, root_player: int, ordering: str
) -> tuple[Move, ...]:
    """Legal moves from the frozen engine, in a deterministic order.

    Reuses Phase 5's ordering, which needs only the trump suit from the
    node it is given. Ordering changes node counts and never results.
    """
    moves = position.legal_moves()
    if not moves:
        return ()
    return order_self_moves(
        position.evaluation_node(root_player), moves, ordering=ordering
    )


# ----------------------------------------------------------------------
# Root
# ----------------------------------------------------------------------
def _search(
    position: DeterminizedPosition,
    config: SearchConfig,
    root_player: int | None,
    root_moves: Sequence[Move] | None,
    pruning: bool,
) -> WorldSearchResult:
    player = position.player if root_player is None else root_player
    evaluator = get_evaluator(config.evaluator)
    stats = SearchStats(enabled=config.collect_statistics)
    stats.visit(0)

    if position.is_terminal:
        stats.leaf(terminal=True)
        return WorldSearchResult(
            None, _terminal_value(position, player, 0), stats, config.depth, player
        )

    candidates = (
        tuple(root_moves) if root_moves is not None else position.legal_moves()
    )
    if not candidates:
        stats.leaf(terminal=False)
        return WorldSearchResult(
            None,
            evaluate_position(position, player, evaluator=evaluator),
            stats,
            config.depth,
            player,
        )

    ordered = order_self_moves(
        position.evaluation_node(player), candidates, ordering=config.ordering
    )
    if config.collect_statistics:
        stats.root_moves = len(ordered)

    # The root player may not be to move — a determinized search can be
    # asked for the value of a position where the opponent acts first.
    maximising = position.current_player == player
    best_move: Move | None = None
    best_value = -_INFINITY if maximising else _INFINITY
    alpha, beta = -_INFINITY, _INFINITY

    for move in ordered:
        child = position.apply(move)
        if pruning:
            value = _alpha_beta_value(
                child, config.depth - 1, 1, alpha, beta, player,
                evaluator, config.ordering, stats,
            )
        else:
            value = _minimax_value(
                child, config.depth - 1, 1, player, evaluator, config.ordering, stats
            )
        # Strictly better, so an earlier move wins a tie. With a
        # deterministic ordering that makes the choice a pure function of
        # the position, never of list order or object identity.
        improved = value > best_value if maximising else value < best_value
        if best_move is None or improved:
            best_value, best_move = value, move
        if pruning:
            if maximising:
                alpha = max(alpha, best_value)
            else:
                beta = min(beta, best_value)

    return WorldSearchResult(best_move, best_value, stats, config.depth, player)


def minimax_world(
    position: DeterminizedPosition,
    config: SearchConfig | None = None,
    *,
    root_player: int | None = None,
    root_moves: Sequence[Move] | None = None,
) -> WorldSearchResult:
    """Full-width minimax over one world. The reference; no pruning."""
    return _search(position, config or SearchConfig(), root_player, root_moves, False)


def alpha_beta_world(
    position: DeterminizedPosition,
    config: SearchConfig | None = None,
    *,
    root_player: int | None = None,
    root_moves: Sequence[Move] | None = None,
) -> WorldSearchResult:
    """Alpha-beta over the same tree. Must agree with :func:`minimax_world`."""
    return _search(position, config or SearchConfig(), root_player, root_moves, True)


def search_world(
    position: DeterminizedPosition,
    config: SearchConfig | None = None,
    *,
    root_player: int | None = None,
    root_moves: Sequence[Move] | None = None,
) -> WorldSearchResult:
    """Search one world with whichever algorithm the configuration names."""
    settings = config or SearchConfig()
    if settings.algorithm == "minimax":
        return minimax_world(
            position, settings, root_player=root_player, root_moves=root_moves
        )
    return alpha_beta_world(
        position, settings, root_player=root_player, root_moves=root_moves
    )
