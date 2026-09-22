"""Single-observer ISMCTS with root re-determinization.

The iteration
-------------
Each iteration samples one hidden world, then walks it while keeping
statistics on information sets rather than on worlds:

1. sample a world consistent with the root information (Determinizer);
2. build the determinized position once;
3. from the root node, repeatedly:
   - at an **observer** decision: find the information-set node, select
     an action by UCT over its *shared* statistics, play it;
   - at an **opponent** decision: let the world-aware opponent policy
     play, keeping no statistics;
   - derive the observation, advance the carried epistemic state;
4. stop at a terminal position, at the depth limit, or on expanding a new
   node;
5. evaluate;
6. backpropagate along the observer's edges.

Where the hidden world may be used
----------------------------------
The determinized world decides **physical legality, transitions and the
simulated outcome**. It never touches node identity or action selection.
That separation is the phase's entire purpose, and it is enforced by
construction: :func:`~durakfish.ai.ismcts.key.information_key` is
computed from the observer's view plus carried knowledge, and receives no
channel through which a hidden card could reach it.

Re-determinization
------------------
**Once per iteration, at the root.** Not per node. Phase 7.3.4 chose this
on Durak's information structure — information is revealed quickly and
the endgame becomes open — and on a hard constraint: per-node
re-determinization would need to rebuild belief from a hypothesis, which
Phase 7.3.3 proved impossible because a hypothesis carries no history.

Opponent model, stated plainly
------------------------------
The modelled opponent acts **inside the determinized world, with full
knowledge of it**. This is deliberate SO-ISMCTS behaviour and it is a
real limitation:

    the observer's policy respects information sets;
    the opponent model does not.

So opponent-side strategy fusion remains: the simulated opponent plays
too knowledgeably, because it sees cards a real opponent could not infer.
Fixing that is MO-ISMCTS, which is future work and is not implemented
here. **No playing-strength claim is made anywhere in this phase.**
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from durakfish.ai.determinized import DeterminizedPosition, build_position
from durakfish.ai.evaluation import Score, get_evaluator
from durakfish.ai.ismcts.key import (
    EpistemicState,
    Observation,
    information_key,
)
from durakfish.ai.ismcts.node import InformationSetNode, ISMCTSError, NodeRegistry
from durakfish.ai.search import SearchConfig
from durakfish.ai.tiebreak import canonical_move_key
from durakfish.ai.world_search import evaluate_position, search_world
from durakfish.cards import Card
from durakfish.game.moves import AttackMove, DefenseMove, Move
from durakfish.information import (
    DeterminizedWorld,
    SearchInformation,
    WorldGenerator,
)

__all__ = [
    "ISMCTS",
    "Determinizer",
    "ISMCTSConfig",
    "ISMCTSResult",
    "observation_of",
]


class Determinizer:
    """Samples worlds consistent with the observer's root information.

    A thin, deliberate wrapper over the verified Phase 7.2
    :class:`~durakfish.information.WorldGenerator`. It introduces **no
    second probability model**: sampling remains uniform over compatible
    allocations, hard deductions are respected because the generator
    respects them, and every world it returns has already satisfied the
    Phase 7.2 validator.

    It exists as a named seam for two reasons. It is the single place the
    search obtains hidden state, which makes the leakage audit a matter
    of checking one call site; and it is where a future belief-aware or
    action-conditioned sampler would go without disturbing the search
    loop.
    """

    __slots__ = ("_generator", "_information")

    def __init__(self, information: SearchInformation) -> None:
        self._information = information
        self._generator = WorldGenerator(information)

    @property
    def allocations(self) -> int:
        """How many worlds remain consistent with what the observer knows."""
        return self._generator.allocations

    def sample(self, rng: random.Random) -> DeterminizedWorld:
        """One consistent world, drawn from the injected generator."""
        return self._generator.sample(rng)

    def position(self, world: DeterminizedWorld) -> DeterminizedPosition:
        """Materialise a world into a complete hypothetical position."""
        return build_position(self._information, world)

    def __repr__(self) -> str:
        return f"Determinizer(P{self._information.player}, {self.allocations} worlds)"


def observation_of(
    position: DeterminizedPosition, move: Move, actor: int, observer: int
) -> Observation:
    """What ``observer`` learns from ``actor`` playing ``move``.

    Every Durak move is public, and a move that carries cards places them
    face up, so the cards in the move itself are revealed. Nothing else is
    added here: cards that merely change location between hidden places —
    a draw, for instance — reveal nothing, and the epistemic state picks
    up whatever became publicly visible when it unions the new position's
    visible cards.
    """
    revealed: set[Card] = set()
    if isinstance(move, AttackMove):
        revealed.add(move.card)
    elif isinstance(move, DefenseMove):
        revealed.add(move.defending_card)
    return Observation(
        move=move,
        actor=actor,
        revealed=frozenset(revealed),
        terminal=position.is_terminal,
    )


@dataclass(frozen=True, slots=True)
class ISMCTSConfig:
    """How to search.

    Attributes:
        iterations: how many worlds to sample and walk.
        exploration: the UCT constant.
        max_depth: plies to walk before evaluating, a guard against very
            long simulations rather than a strength parameter.
        opponent: search settings for the world-aware opponent policy.
            Depth 1 means it minimises the observer's value over its own
            legal moves in the sampled world.
        evaluator: name of the leaf evaluator, reusing the frozen
            Phase 5 semantics.
    """

    iterations: int = 100
    exploration: float = 1.4
    max_depth: int = 24
    opponent: SearchConfig = field(default_factory=lambda: SearchConfig(depth=1))
    evaluator: str = "baseline"

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ISMCTSError(f"iterations must be at least 1, got {self.iterations}")
        if self.max_depth < 1:
            raise ISMCTSError(f"max_depth must be at least 1, got {self.max_depth}")
        if self.exploration < 0:
            raise ISMCTSError(f"exploration must not be negative, got {self.exploration}")


@dataclass(frozen=True, slots=True)
class ISMCTSResult:
    """What a search found, and the evidence behind it."""

    move: Move | None
    root: InformationSetNode | None
    registry: NodeRegistry
    iterations: int
    observer: int
    worlds_sampled: int

    @property
    def information_sets(self) -> int:
        """Distinct information sets visited. Far below the world count if
        statistics are being shared, which is the point."""
        return len(self.registry)

    def visit_counts(self) -> dict[Move, int]:
        if self.root is None:
            return {}
        return {edge.move: edge.visits for edge in self.root.edges.values()}

    def __repr__(self) -> str:
        return (
            f"ISMCTSResult({self.move}, iterations={self.iterations}, "
            f"information_sets={self.information_sets}, P{self.observer})"
        )


class ISMCTS:
    """Single-observer ISMCTS over an observer's information sets."""

    __slots__ = ("_config", "_evaluator")

    def __init__(self, config: ISMCTSConfig | None = None) -> None:
        self._config = config or ISMCTSConfig()
        # Resolved once, eagerly: a misspelled evaluator name should fail
        # when the search is built, not thousands of rollouts later.
        self._evaluator = get_evaluator(self._config.evaluator)

    @property
    def config(self) -> ISMCTSConfig:
        return self._config

    # ------------------------------------------------------------------
    def search(
        self,
        information: SearchInformation,
        rng: random.Random,
        *,
        root_moves: Sequence[Move] | None = None,
    ) -> ISMCTSResult:
        """Run the search and recommend one of the root's legal moves.

        Args:
            information: the observer's legal information. The only
                source of hidden state anywhere in the search, and only
                by way of the Determinizer.
            rng: injected generator. Required — the search never touches
                the global ``random`` module, so a seed reproduces a run.
            root_moves: restrict the recommendation to these moves.

        Returns:
            The recommendation and the search tree that produced it.

        Raises:
            ISMCTSError: if the observer is not the player to move. Root
                move sets are only world-independent from the moving
                seat, which is the same precondition Phase 7.3.2b
                established and for the same reason.
        """
        observer = information.player
        determinizer = Determinizer(information)
        registry = NodeRegistry()
        seed_epistemic = EpistemicState.from_view(information.view)

        probe = determinizer.position(determinizer.sample(random.Random(0)))
        if probe.is_terminal:
            return ISMCTSResult(None, None, registry, 0, observer, 0)
        if probe.current_player != observer:
            raise ISMCTSError(
                f"observer P{observer} is not to move (P{probe.current_player} is)"
            )

        root_key = information_key(probe, seed_epistemic)
        sampled = 0
        for _ in range(self._config.iterations):
            world = determinizer.sample(rng)
            position = determinizer.position(world)
            sampled += 1
            self._iterate(position, seed_epistemic, registry, observer, root_moves)

        root = registry.get(root_key)
        move = root.best_move() if root is not None else None
        return ISMCTSResult(
            move=move,
            root=root,
            registry=registry,
            iterations=self._config.iterations,
            observer=observer,
            worlds_sampled=sampled,
        )

    # ------------------------------------------------------------------
    def _iterate(
        self,
        position: DeterminizedPosition,
        epistemic: EpistemicState,
        registry: NodeRegistry,
        observer: int,
        root_moves: Sequence[Move] | None,
    ) -> None:
        """One walk of one sampled world, keeping information-set statistics."""
        path: list[tuple[InformationSetNode, Move]] = []
        depth = 0
        expanded = False
        first = True

        while not position.is_terminal and depth < self._config.max_depth:
            legal = position.legal_moves()
            if not legal:  # pragma: no cover - the engine always offers a move
                break
            actor = position.current_player
            assert actor is not None

            if actor == observer:
                if first and root_moves is not None:
                    legal = tuple(m for m in legal if m in set(root_moves))
                    if not legal:
                        raise ISMCTSError("no supplied root move is legal here")
                key = information_key(position, epistemic)
                node = registry.get_or_create(key)
                fresh = node.visits == 0
                node.ensure_edges(legal)
                move = node.uct_select(legal, self._config.exploration)
                path.append((node, move))
                if fresh and not expanded:
                    # Expand one new information set per iteration, then
                    # evaluate — the standard MCTS stopping rule.
                    expanded = True
                    position = self._play(position, move)
                    epistemic = epistemic.advance(
                        position, observation_of(position, move, actor, observer)
                    )
                    break
            else:
                move = self._opponent_move(position, observer)

            position = self._play(position, move)
            epistemic = epistemic.advance(
                position, observation_of(position, move, actor, observer)
            )
            depth += 1
            first = False

        value = self._value(position, observer, depth)
        for node, move in path:
            node.record_visit()
            node.edge_for(move).record(value)

    # ------------------------------------------------------------------
    @staticmethod
    def _play(position: DeterminizedPosition, move: Move) -> DeterminizedPosition:
        """Advance the hypothesis through the frozen rules engine."""
        return position.apply(move)

    def _opponent_move(self, position: DeterminizedPosition, observer: int) -> Move:
        """The world-aware opponent's choice.

        Reuses the verified Phase 7.3.2a search with the *observer* as
        root player, so it minimises the observer's value — the opponent
        playing as well as it can given full sight of the sampled world.
        Deterministic, so it adds no randomness to the iteration.
        """
        result = search_world(position, self._config.opponent, root_player=observer)
        if result.move is not None:
            return result.move
        return min(position.legal_moves(), key=canonical_move_key)  # pragma: no cover

    def _value(
        self, position: DeterminizedPosition, observer: int, ply: int
    ) -> Score:
        """Score a leaf from the observer's point of view.

        Reuses the frozen evaluation semantics wholesale: terminal
        positions get their exact ply-adjusted score, everything else is
        handed to the evaluator named by the configuration, which
        defaults to the existing ``BaselineEvaluator``. The evaluator
        sees the determinized world, which is legitimate — it is scoring
        a simulation, not identifying a node.

        Before Phase 7.3.9 this ignored ``ISMCTSConfig.evaluator`` and
        always used the baseline, so the documented field had no effect.
        """
        return evaluate_position(
            position, observer, ply=ply, evaluator=self._evaluator
        )
