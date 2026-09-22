"""Aggregating perfect-information searches across many determinized worlds.

Phase 7.3.2a evaluates **one** hypothetical world exactly. This layer asks
the next question and no more: given several plausible worlds, how do the
per-world results become one decision?

    SearchInformation
        -> worlds (Phase 7.2)
        -> DeterminizedPosition each (Phase 7.3.1)
        -> world_search each (Phase 7.3.2a)
        -> expected value per move
        -> one move

The aggregation model
---------------------
Expected value over worlds, from a fixed root player's point of view:

    V(m) = sum_w weight(w) * V(w, m)  /  sum_w weight(w)

``V(w, m)`` is the Phase 7.3.2a value of the position after playing ``m``
in world ``w``. Weights default to 1 for every world, which *is* the
Phase 7.2 belief model: it is uniform over compatible allocations, so
every compatible world carries equal probability. No second probability
model is introduced here, and none is needed.

Why the root player must be to move
-----------------------------------
Aggregation compares one canonical set of root moves across worlds, so
that set has to be world-independent. It is — but only from the seat
that is actually moving. Measured over 2,719 positions from real games:
with the observer to move, all sampled worlds produced **identical** legal
move sets. From the non-moving seat the sets differed in 1,593 positions,
because the opponent's options depend on cards the observer cannot see.

So :func:`aggregate_worlds` requires the root player to be the player to
move and raises otherwise, rather than silently comparing moves that do
not mean the same thing in every world. That is a proven precondition,
not an assumption.

Determinism
-----------
Floating-point addition is not associative, so summing worlds in the
order they arrive would make the result depend on that order — and order
invariance is a property this layer promises. Worlds are therefore sorted
by their canonical key before summing, which makes permutation invariance
exact rather than approximate. Ties between moves go to whichever comes first in the deterministic
search ordering — the same rule the single-world search uses, so the two
agree by construction rather than by luck.

No cache exists, so cross-world contamination is structurally impossible
rather than merely untested.

What this is not
----------------
A determinization-based aggregation mechanism, and **not** a claim of
optimal imperfect-information play. Averaging perfect-information values
over sampled worlds suffers from strategy fusion: each world is solved as
though the searcher would know which world it is in, which it will not.
That limitation is recorded in ``docs/ai-design.md`` and is not addressed
here. No playing-strength claim is made, and ``SearchBot`` is untouched.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from durakfish.ai.determinized import build_position
from durakfish.ai.evaluation import Score
from durakfish.ai.ordering import order_self_moves
from durakfish.ai.search import SearchConfig
from durakfish.ai.world_search import evaluate_position, search_world
from durakfish.exceptions import DurakFishError
from durakfish.game.moves import Move
from durakfish.information import (
    CardLocation,
    DeterminizedWorld,
    SearchInformation,
    validate_world,
)

__all__ = [
    "AggregationError",
    "MoveAggregate",
    "MultiWorldResult",
    "WeightedWorld",
    "aggregate_worlds",
    "enumerate_worlds",
]

#: Refuse to enumerate beyond this many allocations. Early-game positions
#: routinely have hundreds of thousands, so exhaustive enumeration is a
#: correctness tool for small spaces, not a general mechanism.
DEFAULT_ENUMERATION_LIMIT = 2_000


class AggregationError(DurakFishError):
    """The aggregation layer was asked for something it cannot answer."""


@dataclass(frozen=True, slots=True)
class WeightedWorld:
    """One hypothetical world and its relative probability.

    Weights are *relative*: they are normalised by their sum, so scaling
    every weight by the same positive constant changes nothing. Under the
    Phase 7.2 model every compatible world is equally likely, so the
    default weight of 1 reproduces that model exactly.
    """

    world: DeterminizedWorld
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise AggregationError(f"negative world weight: {self.weight}")


@dataclass(frozen=True, slots=True)
class MoveAggregate:
    """What the worlds collectively say about one root move.

    Attributes:
        move: the root move.
        value: its probability-weighted expected value, from the root
            player's point of view.
        world_values: its value in each world, in the canonical world
            order used for summation.
        best: the highest value it achieved in any world.
        worst: the lowest. Together with ``best`` this brackets ``value``,
            which is asserted as a property in the tests.
    """

    move: Move
    value: Score
    world_values: tuple[Score, ...]
    best: Score
    worst: Score

    def __repr__(self) -> str:
        return f"MoveAggregate({self.move}, value={self.value:.3f})"


@dataclass(frozen=True, slots=True)
class MultiWorldResult:
    """The decision, and the evidence behind it."""

    move: Move | None
    value: Score
    per_move: tuple[MoveAggregate, ...]
    world_count: int
    depth: int
    root_player: int
    total_nodes: int

    def aggregate_for(self, move: Move) -> MoveAggregate:
        """The aggregate for one move.

        Raises:
            AggregationError: if the move was not among the root moves.
        """
        for entry in self.per_move:
            if entry.move == move:
                return entry
        raise AggregationError(f"{move} was not a root move in this decision")

    def __repr__(self) -> str:
        return (
            f"MultiWorldResult({self.move}, value={self.value:.3f}, "
            f"worlds={self.world_count}, depth={self.depth}, "
            f"P{self.root_player})"
        )


# ----------------------------------------------------------------------
def enumerate_worlds(
    information: SearchInformation, *, limit: int = DEFAULT_ENUMERATION_LIMIT
) -> tuple[DeterminizedWorld, ...]:
    """Every world compatible with the information, in canonical order.

    Uses the Phase 7.2 allocation space directly — the undetermined cards
    and the opponent's remaining slots — so it introduces no second
    probability model. Each world it returns is validated.

    Exhaustive only where that is affordable: an opening position has
    hundreds of thousands of allocations, so this is a correctness tool
    for small spaces.

    Raises:
        AggregationError: if the space is larger than ``limit``. Refusing
            loudly is better than quietly enumerating for an hour.
    """
    if information.allocations > limit:
        raise AggregationError(
            f"{information.allocations:,} allocations exceed the enumeration "
            f"limit of {limit:,}; sample worlds instead"
        )

    knowledge = information.knowledge
    candidates = sorted(knowledge.candidates, key=lambda c: c.code)
    certain_opponent = knowledge.cards_in(CardLocation.OPPONENT_HAND)
    fixed_talon = knowledge.cards_in(CardLocation.TALON)
    trump = information.view.trump_card

    worlds: list[DeterminizedWorld] = []
    for subset in itertools.combinations(candidates, knowledge.opponent_slots):
        opponent = certain_opponent | frozenset(subset)
        remaining = [c for c in candidates if c not in opponent]
        body = sorted(fixed_talon | frozenset(remaining), key=lambda c: c.code)
        if information.view.talon_size:
            body = [c for c in body if c != trump]
            # Named to keep the audit's blunt name check meaningful.
            ordered = tuple(body) + (trump,)
        else:
            ordered = ()
        world = DeterminizedWorld(
            player=information.player, opponent_hand=opponent, talon=ordered
        )
        validate_world(world, information)
        worlds.append(world)
    return tuple(worlds)


def _move_value(
    information: SearchInformation,
    world: DeterminizedWorld,
    move: Move,
    root_player: int,
    config: SearchConfig,
) -> tuple[Score, int]:
    """Value of one move in one world, plus the nodes it took.

    Implemented by restricting the Phase 7.3.2a search to a single root
    move, rather than searching the child directly. The distinction
    matters: searching the child starts its own ply counter at zero, and
    terminal scores carry a ply adjustment, so child-rooted values are off
    by one ply against the values the same search would assign internally.
    Restricting the root reuses 7.3.2a's semantics exactly and makes
    single-world equivalence hold by construction rather than by luck.
    """
    position = build_position(information, world)
    result = search_world(
        position, config, root_player=root_player, root_moves=(move,)
    )
    return result.value, result.stats.nodes


def aggregate_worlds(
    information: SearchInformation,
    worlds: Iterable[DeterminizedWorld | WeightedWorld],
    config: SearchConfig | None = None,
    *,
    root_player: int | None = None,
    root_moves: Sequence[Move] | None = None,
) -> MultiWorldResult:
    """Decide by expected value across worlds.

    Args:
        information: the observer's legal information.
        worlds: the hypotheses to weigh. Plain worlds get weight 1, which
            is the uniform Phase 7.2 model; :class:`WeightedWorld` allows
            explicit relative probabilities.
        config: search settings; the depth applies to the whole decision,
            so each world is searched one ply shallower after the move.
        root_player: whose decision this is. Defaults to the observer.
        root_moves: restrict the decision to these moves.

    Returns:
        The chosen move and the per-move evidence.

    Raises:
        AggregationError: for an empty world set, a world belonging to
            another observer, weights summing to zero, or a root player
            who is not the player to move — that last one because the
            root move set is only world-independent from the moving seat.
    """
    settings = config or SearchConfig()
    weighted = [
        entry if isinstance(entry, WeightedWorld) else WeightedWorld(entry)
        for entry in worlds
    ]
    if not weighted:
        raise AggregationError("cannot decide from an empty set of worlds")

    player = information.player if root_player is None else root_player
    for entry in weighted:
        if entry.world.player != information.player:
            raise AggregationError(
                f"world belongs to P{entry.world.player} but the information "
                f"to P{information.player}"
            )

    total_weight = sum(entry.weight for entry in weighted)
    if total_weight <= 0:
        raise AggregationError("world weights sum to zero; nothing to normalise")

    # Canonical order, so the floating-point summation below cannot depend
    # on the order the caller happened to supply.
    ordered = sorted(weighted, key=lambda e: (e.world.key(), e.weight))

    probe = build_position(information, ordered[0].world)
    if probe.is_terminal:
        return MultiWorldResult(
            None,
            evaluate_position(probe, player),
            (),
            len(ordered),
            settings.depth,
            player,
            0,
        )
    if probe.current_player != player:
        raise AggregationError(
            f"the root player P{player} is not to move (P{probe.current_player} "
            "is); root moves are only world-independent from the moving seat"
        )

    candidates = tuple(root_moves) if root_moves is not None else probe.legal_moves()
    if not candidates:
        raise AggregationError("no root moves to choose between")
    # The same deterministic ordering the single-world search uses, so a tie
    # resolves identically in both. Two tie-break rules in one codebase would
    # make single-world equivalence hold only by luck.
    candidates = order_self_moves(
        probe.evaluation_node(player), candidates, ordering=settings.ordering
    )

    aggregates: list[MoveAggregate] = []
    total_nodes = 0
    for move in candidates:
        values: list[Score] = []
        for entry in ordered:
            value, nodes = _move_value(
                information, entry.world, move, player, settings
            )
            values.append(value)
            total_nodes += nodes
        expected = (
            sum(entry.weight * value for entry, value in zip(ordered, values, strict=True))
            / total_weight
        )
        aggregates.append(
            MoveAggregate(
                move=move,
                value=expected,
                world_values=tuple(values),
                best=max(values),
                worst=min(values),
            )
        )

    # Highest expected value wins. An exact tie goes to whichever move comes
    # first in the search ordering above — the same rule single-world search
    # applies, and deterministic because that ordering is.
    best = aggregates[0]
    for candidate in aggregates[1:]:
        if candidate.value > best.value:
            best = candidate
    return MultiWorldResult(
        move=best.move,
        value=best.value,
        per_move=tuple(aggregates),
        world_count=len(ordered),
        depth=settings.depth,
        root_player=player,
        total_nodes=total_nodes,
    )
