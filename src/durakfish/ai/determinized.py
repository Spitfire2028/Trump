"""Evaluating one determinized world.

Phase 7.2 produces legal hidden worlds. This module answers the next
question and no more: given **one** such world, how does the existing
search machinery evaluate a candidate action, without touching the real
hidden state and without a second copy of the rules?

    SearchInformation + DeterminizedWorld  ->  DeterminizedPosition
                                                      |
                                        the frozen rules engine

Real state versus hypothetical state
------------------------------------
Two things in this codebase are now both "a complete position", and
confusing them would undo every guarantee since Phase 3:

**The real ``GameState``** is ground truth. It is what actually happened.
Nothing downstream of ``observe()`` may read it, and nothing here does.

**A ``DeterminizedPosition``** is a complete position too, and it does
contain cards the observer cannot see — but every one of those cards came
from a :class:`~durakfish.information.DeterminizedWorld` that was sampled
from the observer's own information. It is a *hypothesis*: one of the
worlds consistent with what the observer knows. Knowing all the cards
inside a hypothesis is legitimate and is the entire point; knowing them
in reality is cheating.

The distinction is enforced structurally. This module's inputs are a
:class:`~durakfish.information.SearchInformation` and a
:class:`~durakfish.information.DeterminizedWorld`, both of which Phases
7.1 and 7.2 proved carry no route to the real state. A hypothetical
position is therefore a pure function of those two things, which is what
the substitution test in ``tests/test_determinized.py`` checks: identical
information plus an identical world gives an identical position, whatever
reality happened to be underneath.

Why this module may name ``GameState``
--------------------------------------
It is the **materialisation boundary**, and the only module in ``ai/``
permitted to name a game state — the mirror image of
``information/information_set.py``, which is the only module in
``information/`` permitted to name one because it is the *redaction*
boundary. One place converts ground truth into information; one place
converts information back into a hypothesis. Everything between them is
banned from both, and ``tests/test_phase3_audit.py`` enforces exactly
that scoping.

Reusing the rules rather than restating them
--------------------------------------------
The position wraps a genuine ``GameState``, so legality, transitions and
terminal detection are the frozen Phase 2 engine, unchanged. No second
game engine, no reimplemented rules, no divergence to keep in sync. The
equivalence tests show that a position built from the *true* world has
the same ``transposition_key`` as the real state it came from, which
means everything the rules engine computes downstream is identical by
construction.

The one legitimate difference is history: an observer's view has draws
redacted, so a hypothetical position cannot reconstruct the real event
log and carries none. History is excluded from ``transposition_key`` and
from legality, so nothing search-relevant depends on it.

Scope
-----
One world at a time. No aggregation across worlds, no averaging, no
expected values, no move selection. Combining per-world results without
assuming the searcher will know which world it is in is a genuinely hard
problem — strategy fusion, recorded in ``docs/ai-design.md`` since Phase
1 — and it is deferred to Phase 7.3.2 precisely so that it can be
attacked on top of a contract already known to be correct.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from durakfish.ai.searchnode import Actor, NodeKind, SearchNode, SearchSlot
from durakfish.exceptions import DurakFishError
from durakfish.game import apply_move, get_legal_moves
from durakfish.game.moves import Move
from durakfish.game.state import GameState, Phase, add_cards
from durakfish.information import (
    DeterminizedWorld,
    SearchInformation,
    validate_world,
)

__all__ = [
    "DeterminizationError",
    "DeterminizedPosition",
    "build_position",
]


class DeterminizationError(DurakFishError):
    """A world could not be turned into a position."""


@dataclass(frozen=True, slots=True)
class DeterminizedPosition:
    """A complete hypothetical position, ready for the rules engine.

    Attributes:
        player: the observer this hypothesis belongs to. A world is only
            meaningful relative to somebody's information.
        state: the hypothetical position. A real ``GameState`` object, but
            **not** the real game's state: every card in it came from the
            observer's information or from a sampled world.
        root_world: the world this position was built from, kept for
            provenance. Cleared by :meth:`apply`, because after a move the
            allocation describes where cards *were*, not where they are,
            and a stale allocation is worse than none.
    """

    player: int
    state: GameState
    root_world: DeterminizedWorld | None = None

    # ------------------------------------------------------------------
    # The rules engine, unchanged
    # ------------------------------------------------------------------
    def legal_moves(self) -> tuple[Move, ...]:
        """Legal moves in this hypothesis, from the frozen rules engine."""
        return get_legal_moves(self.state)

    def apply(self, move: Move) -> DeterminizedPosition:
        """Play a move, returning a new position.

        Delegates to the frozen engine, so an illegal move raises exactly
        what it always raises and nothing is repaired silently.
        """
        return DeterminizedPosition(
            player=self.player, state=apply_move(self.state, move), root_world=None
        )

    @property
    def current_player(self) -> int | None:
        return self.state.current_player

    @property
    def is_terminal(self) -> bool:
        return self.state.is_over

    @property
    def durak(self) -> int | None:
        """The loser, or ``None`` for a draw or an unfinished game."""
        return self.state.durak

    def outcome_for(self, player: int) -> int | None:
        """``+1`` won, ``-1`` lost, ``0`` drawn, or ``None`` if unfinished."""
        if not self.is_terminal:
            return None
        loser = self.state.durak
        if loser is None:
            return 0
        return -1 if loser == player else 1

    # ------------------------------------------------------------------
    # Leaf evaluation
    # ------------------------------------------------------------------
    def evaluation_node(self, root_player: int) -> SearchNode:
        """A :class:`SearchNode` describing this position for the evaluator.

        Phase 5's ``BaselineEvaluator`` scores a ``SearchNode``, so reusing
        it — rather than writing a second heuristic that could drift from
        the first — needs a node-shaped view of a determinized position.
        This builds one.

        The node is used for **leaf scoring only**. Its
        ``self_moves``/``opponent_actions`` machinery, which is the Phase 5
        abstraction, is never consulted: move generation in a determinized
        search comes from the frozen rules engine, which sees real cards.
        Two consequences of that are deliberate:

        * ``kind`` is ``DECISION``, so ``SearchNode.outcome()`` always
          returns ``None`` and cannot pre-empt terminal detection. The
          searcher decides terminals from the frozen engine's own
          ``is_over``/``durak``.
        * ``unknown_own`` is zero. In a determinized world every card is
          specified, so nothing is unidentified.

        Building this lives here, in the materialisation boundary, because
        it is the only module in ``ai/`` permitted to read a game state's
        fields.

        Args:
            root_player: whose point of view the evaluation takes.
        """
        state = self.state
        opponent = 1 - root_player
        slots = tuple(
            SearchSlot(
                attacking_card=slot.attacking_card,
                defending_card=slot.defending_card,
                defended=slot.is_defended,
            )
            for slot in state.table
        )
        return SearchNode(
            hand=state.hands[root_player],
            table=slots,
            known_ranks=frozenset(
                card.rank for slot in slots for card in slot.known_cards
            ),
            opponent_cards=len(state.hands[opponent]),
            talon_size=len(state.talon),
            trump=state.trump_suit,
            refill_to=state.rules.hand_size,
            attack_limit=state.attack_limit,
            searcher_is_attacker=state.attacker == root_player,
            phase=state.phase if state.phase is not Phase.GAME_OVER else Phase.ATTACK,
            to_move=(
                Actor.SELF if state.current_player == root_player else Actor.OPPONENT
            ),
            kind=NodeKind.DECISION,
            unknown_own=0,
        )

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    def fingerprint(self) -> tuple[Any, ...]:
        """Everything search-relevant about this position.

        Built on the engine's own ``transposition_key``, which covers
        hands, talon, discard, table, trump, attacker, phase and attack
        limit — deliberately not just hand sizes or a score.
        """
        return (self.player, self.state.transposition_key())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DeterminizedPosition):
            return self.fingerprint() == other.fingerprint()
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.fingerprint())

    def __repr__(self) -> str:
        return (
            f"DeterminizedPosition(P{self.player}, "
            f"phase={self.state.phase.name}, "
            f"terminal={self.is_terminal})"
        )


def build_position(
    information: SearchInformation,
    world: DeterminizedWorld,
    *,
    validate: bool = True,
) -> DeterminizedPosition:
    """Combine an observer's information with one sampled world.

    Every field comes from exactly one of the two inputs: the visible
    position from ``information``, the hidden allocation from ``world``.
    Nothing is read from the real game.

    Args:
        information: what the observer legitimately knows.
        world: a hidden allocation compatible with that information.
        validate: run the Phase 7.2 validator first. On by default,
            because an invalid world produces a position describing a game
            that could not have happened, and searching it would be worse
            than failing.

    Returns:
        A complete hypothetical position.

    Raises:
        DeterminizationError: if the world does not belong to this
            observer, or is not compatible with the information. Invalid
            worlds are refused, never repaired.
    """
    if world.player != information.player:
        raise DeterminizationError(
            f"world belongs to P{world.player} but the information to "
            f"P{information.player}"
        )
    if validate:
        try:
            validate_world(world, information)
        except DurakFishError as failure:
            raise DeterminizationError(
                f"refusing to build a position from an incompatible world: "
                f"{failure}"
            ) from failure

    view = information.view
    observer = information.player
    hands = tuple(
        add_cards((), view.hand) if seat == observer
        else add_cards((), world.opponent_hand)
        for seat in range(len(view.hand_sizes))
    )

    state = GameState(
        hands=hands,
        talon=world.talon,
        discard=view.discard,
        table=view.table,
        trump_card=view.trump_card,
        trump_suit=view.trump_suit,
        attacker=view.attacker,
        phase=view.phase,
        attack_limit=view.attack_limit,
        durak=view.durak,
        # An observer's history has draws redacted, so a hypothesis cannot
        # reconstruct the real event log. Nothing search-relevant depends
        # on it: history is excluded from transposition_key and from
        # legality alike.
        history=None,
        rules=view.rules,
    )
    # Cheap and worth it: a malformed hypothesis would otherwise surface
    # much later as a confusing rules error deep inside a search.
    state.validate()

    return DeterminizedPosition(player=observer, state=state, root_world=world)


def legal_moves_for(
    information: SearchInformation, world: DeterminizedWorld
) -> Sequence[Move]:
    """Convenience: the legal moves in the position that world implies."""
    return build_position(information, world).legal_moves()
