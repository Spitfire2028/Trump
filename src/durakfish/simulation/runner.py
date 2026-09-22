"""Driving and replaying games.

:func:`play_game` runs one complete game from a master seed and returns a
:class:`~durakfish.simulation.record.GameRecord`. :func:`replay` walks that
record back through the rules engine, and :func:`verify_record` checks the
walk reproduces the recorded outcome.

Why this lives above the AI layer
---------------------------------
The driver knows about agents, so it sits above them:
``cards → game → information → ai → simulation``. Putting it in ``game/``
— as the original project tree proposed — would force the rules package to
import agent vocabulary and invert the dependency arrow. The rules engine
stays ignorant of everything above it, and ``tests/test_architecture.py``
enforces that by parsing the import graph.

Determinism
-----------
One master seed fans out into independent streams:

    seed ──► deal_seed
         └─► agent_seeds[0], agent_seeds[1], ...

Separate streams mean an agent's randomness cannot perturb the deal, and
swapping in a different agent does not change the cards it is dealt —
which is what makes two bots comparable on identical deals later. Every
derived seed is written into the record, so a replay reconstructs the
agents and not merely the moves.

The driver itself draws no random numbers. Given the same seed, ruleset
and agents, it produces a byte-identical record.
"""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence

from durakfish.ai.base import Agent
from durakfish.cards import Deck
from durakfish.exceptions import IllegalAgentMoveError, RecordError, SimulationError
from durakfish.game import apply_move, get_legal_moves, new_game
from durakfish.game.ruleset import STANDARD_RULES, RuleSet
from durakfish.game.state import GameState
from durakfish.information import observe
from durakfish.simulation.record import GameRecord, MoveRecord

__all__ = [
    "DEFAULT_MOVE_LIMIT",
    "derive_seeds",
    "play_game",
    "replay",
    "verify_record",
]

#: Random play averages ~110 plies; a game far past this indicates a bug,
#: not a long game. The Phase 2 audit proved termination, so this guard
#: exists to catch a regression in that argument, not to paper over it.
DEFAULT_MOVE_LIMIT = 2000

_SEED_BITS = 64


def derive_seeds(seed: int, count: int) -> tuple[int, tuple[int, ...]]:
    """Fan a master seed out into a deal seed and one seed per agent.

    Uses ``random.Random``'s Mersenne Twister, which is stable across
    Python versions and platforms — unlike :func:`hash`, whose string
    salting differs between processes and would make records
    irreproducible.

    Returns:
        ``(deal_seed, agent_seeds)``.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    stream = random.Random(seed)
    deal_seed = stream.getrandbits(_SEED_BITS)
    agent_seeds = tuple(stream.getrandbits(_SEED_BITS) for _ in range(count))
    return deal_seed, agent_seeds


def play_game(
    agents: Sequence[Agent],
    *,
    seed: int = 0,
    rules: RuleSet = STANDARD_RULES,
    deck: Deck | None = None,
    first_attacker: int | None = None,
    move_limit: int = DEFAULT_MOVE_LIMIT,
    validate_states: bool = False,
    observe_history: bool = True,
) -> GameRecord:
    """Play one complete game and return its record.

    Each agent sees only :func:`~durakfish.information.observe` output for
    its own seat. The ground-truth state never leaves this function.

    Args:
        agents: one per seat, indexed by player id.
        seed: master seed; fans out into deal and per-agent streams.
        rules: the variant.
        deck: an exact deck to deal from, for reproducing a known
            position. When given, the deal seed does not determine the
            cards, and the record says so.
        first_attacker: override the opening attacker. Useful for fair
            comparison, since the lowest-trump rule is not seat-symmetric.
        move_limit: safety guard against a non-terminating rules bug.
        validate_states: run ``GameState.validate()`` on every position.
            Off by default because it roughly doubles the cost of a game;
            the test suite turns it on.
        observe_history: give agents the redacted event log. On by
            default, because an opponent model needs it. Redaction is
            linear in the log, so it dominates the cost of a long game;
            agents that ignore history can switch it off for roughly a
            5x speedup. Measured in ``benchmarks/bench_phase3.py``.

    Returns:
        A :class:`GameRecord` sufficient to reproduce the game exactly.

    Raises:
        SimulationError: wrong number of agents, or the move limit was hit.
        IllegalAgentMoveError: an agent returned a move outside the legal
            set it was given. The driver reports this rather than
            substituting a legal move — silently repairing an agent's bug
            would corrupt every statistic gathered from it.
    """
    if len(agents) != rules.num_players:
        raise SimulationError(
            f"{len(agents)} agents supplied for a {rules.num_players}-player game"
        )

    deal_seed, agent_seeds = derive_seeds(seed, len(agents))
    for agent, agent_seed in zip(agents, agent_seeds, strict=True):
        agent.reset(random.Random(agent_seed))

    state = new_game(
        seed=deal_seed, deck=deck, rules=rules, first_attacker=first_attacker
    )
    initial_state = state
    moves: list[MoveRecord] = []

    while not state.is_over:
        if len(moves) >= move_limit:
            raise SimulationError(
                f"game exceeded {move_limit} plies; suspecting a rules bug:\n"
                f"{state.describe()}"
            )
        player = state.current_player
        assert player is not None  # guaranteed by `not state.is_over`
        legal = get_legal_moves(state)
        view = observe(state, player, include_history=observe_history)

        move = agents[player].choose(view, legal)
        if move not in legal:
            raise IllegalAgentMoveError(
                f"agent {agents[player].name!r} (P{player}) returned {move}, "
                f"which is not among the {len(legal)} legal moves offered"
            )

        moves.append(MoveRecord(ply=len(moves), player=player, move=move))
        state = apply_move(state, move)
        if validate_states:
            state.validate()

    return GameRecord(
        seed=seed,
        deal_seed=deal_seed,
        agent_seeds=agent_seeds,
        agent_names=tuple(agent.name for agent in agents),
        rules=rules,
        initial_state=initial_state,
        moves=tuple(moves),
        durak=state.durak,
        deck_supplied=deck is not None,
    )


def replay(record: GameRecord, *, validate_states: bool = True) -> Iterator[GameState]:
    """Re-walk a recorded game, yielding every position in order.

    Yields the opening position first, then the position after each move,
    so a record of *n* plies yields *n + 1* states.

    Raises:
        RecordError: if a recorded move is illegal in the position it was
            recorded from, or was attributed to the wrong player. Either
            means the record and the current rules engine disagree.
    """
    state = record.initial_state
    if validate_states:
        state.validate()
    yield state

    for entry in record.moves:
        if state.is_over:
            raise RecordError(
                f"record has a move at ply {entry.ply} but the game had ended"
            )
        if state.current_player != entry.player:
            raise RecordError(
                f"ply {entry.ply} is recorded for P{entry.player} but "
                f"P{state.current_player} was to move"
            )
        if entry.move not in get_legal_moves(state):
            raise RecordError(
                f"ply {entry.ply}: recorded move {entry.move} is not legal here"
            )
        state = apply_move(state, entry.move)
        if validate_states:
            state.validate()
        yield state


def verify_record(record: GameRecord) -> GameState:
    """Replay a record fully and confirm it reproduces what it claims.

    Checks that the game ends exactly when the moves run out, that the
    outcome matches, and — unless an explicit deck was used — that the
    recorded opening position is still what the recorded deal seed
    produces. That last check is what turns a future change to dealing or
    to seed derivation into a test failure rather than a silent
    divergence.

    Returns:
        The final state.

    Raises:
        RecordError: on any mismatch.
    """
    states = replay(record)
    final: GameState | None = None
    for state in states:
        final = state
    assert final is not None  # replay always yields the opening position

    if not final.is_over:
        raise RecordError("replaying the record did not finish the game")
    if final.durak != record.durak:
        raise RecordError(
            f"record says durak={record.durak}, replay produced {final.durak}"
        )

    if not record.deck_supplied:
        expected = new_game(
            seed=record.deal_seed,
            rules=record.rules,
            first_attacker=record.initial_state.attacker,
        )
        if expected.transposition_key() != record.initial_state.transposition_key():
            raise RecordError(
                "the recorded deal seed no longer reproduces the recorded "
                "opening position; dealing or seed derivation has changed"
            )
    return final
