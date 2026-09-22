"""Property-based and fuzz testing for the rules engine.

The numbered properties come from the Phase 2 specification. The
randomised game test is the broadest safety net in the project: it plays
thousands of complete games and checks that the rules never crash, never
lose or duplicate a card, and always terminate.

Hypothesis is optional; the sections that need it skip cleanly without it.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from durakfish.cards import Card
from durakfish.cards.deck import standard_cards
from durakfish.exceptions import IllegalMoveError
from durakfish.game import (
    END_ATTACK,
    TAKE,
    AttackMove,
    DefenseMove,
    GameState,
    apply_move,
    get_legal_moves,
    is_legal_move,
    new_game,
    random_playout,
)

DECK = standard_cards(36)
UNIVERSE = set(DECK)


def card_locations(state: GameState) -> Counter[Card]:
    """Count every card across every location."""
    counts: Counter[Card] = Counter()
    for hand in state.hands:
        counts.update(hand)
    counts.update(state.talon)
    counts.update(state.table_cards)
    counts.update(state.discard)
    return counts


# ----------------------------------------------------------------------
# Properties 1-8, checked along real game trajectories
# ----------------------------------------------------------------------
def test_properties_hold_along_random_games() -> None:
    rng = random.Random(2024)
    games = 0
    for seed in range(120):
        state = new_game(seed=seed)
        talon_was_empty = False
        for _ in range(2000):
            if state.is_over:
                break
            legal = get_legal_moves(state)

            # P1: every generated move is legal.
            assert legal
            assert all(is_legal_move(state, m) for m in legal)

            # P6: a defence card never answers two attacks.
            defenders = [
                slot.defending_card for slot in state.table if slot.is_defended
            ]
            assert len(defenders) == len(set(defenders))

            # P4 + P7: exactly 36 cards, each in exactly one place.
            counts = card_locations(state)
            assert sum(counts.values()) == 36
            assert all(n == 1 for n in counts.values())
            assert set(counts) == UNIVERSE

            # P8: once the talon is empty it never yields another card.
            if talon_was_empty:
                assert state.talon_size == 0
            talon_was_empty = talon_was_empty or state.talon_size == 0

            move = rng.choice(legal)
            successor = apply_move(state, move)

            # P2: a legal move always produces a valid state.
            successor.validate()
            state = successor
        else:  # pragma: no cover - would mean a non-terminating game
            pytest.fail(f"game {seed} did not terminate")
        games += 1
    assert games == 120


def test_illegal_moves_are_always_rejected() -> None:
    """P3: applying an illegal move raises rather than corrupting state."""
    rng = random.Random(31)
    checked = 0
    for seed in range(20):
        state = new_game(seed=seed)
        for _ in range(60):
            if state.is_over:
                break
            legal = set(get_legal_moves(state))
            candidates = (
                [AttackMove(c) for c in rng.sample(DECK, 6)]
                + [
                    DefenseMove(a, d)
                    for a, d in zip(rng.sample(DECK, 4), rng.sample(DECK, 4))
                ]
                + [TAKE, END_ATTACK]
            )
            for move in candidates:
                if move in legal:
                    continue
                with pytest.raises(IllegalMoveError):
                    apply_move(state, move)
                checked += 1
            state = apply_move(state, rng.choice(tuple(legal)))
    assert checked > 1000


def test_cards_leave_play_only_through_the_discard() -> None:
    """P5: a card vanishes from hands/talon/table only by being discarded."""
    rng = random.Random(88)
    for seed in range(40):
        state = new_game(seed=seed)
        discarded_so_far: set[Card] = set()
        while not state.is_over:
            state = apply_move(state, rng.choice(get_legal_moves(state)))
            # The discard pile only ever grows.
            assert discarded_so_far <= set(state.discard)
            discarded_so_far = set(state.discard)
        assert set(state.discard) | set(state.hands[0]) | set(
            state.hands[1]
        ) == UNIVERSE


# ----------------------------------------------------------------------
# The big fuzz run
# ----------------------------------------------------------------------
def test_thousands_of_random_games_terminate_cleanly() -> None:
    lengths = []
    duraks = Counter()
    for seed in range(3000):
        state = new_game(seed=seed)
        final = random_playout(state, random.Random(seed), max_moves=4000)
        final.validate()
        assert final.is_over
        assert final.table == ()
        assert not final.talon
        assert sum(len(h) for h in final.hands) + len(final.discard) == 36
        lengths.append(final.event_count)
        duraks[final.durak] += 1

    assert len(lengths) == 3000
    # Sanity on the distribution rather than an exact figure: random play
    # should produce a decisive result the overwhelming majority of the time.
    assert duraks[None] < 3000 * 0.1
    assert min(lengths) > 3
    assert max(lengths) < 4000


def test_random_games_do_not_favour_a_seat_absurdly() -> None:
    """With the opening seat fixed, random play should be roughly balanced."""
    wins = Counter()
    for seed in range(1500):
        final = random_playout(
            new_game(seed=seed, first_attacker=0), random.Random(seed)
        )
        wins[final.durak] += 1
    decisive = wins[0] + wins[1]
    share = wins[0] / decisive
    assert 0.35 < share < 0.65, f"seat bias looks wrong: {share:.3f}"


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------
def test_same_seed_reproduces_the_same_game_exactly() -> None:
    for seed in (1, 2, 3, 99):
        a = random_playout(new_game(seed=seed), random.Random(seed))
        b = random_playout(new_game(seed=seed), random.Random(seed))
        assert a.transposition_key() == b.transposition_key()
        assert [e.to_dict() for e in a.events()] == [e.to_dict() for e in b.events()]


def test_replaying_recorded_moves_reproduces_the_state() -> None:
    """Determinism from the decision sequence, not just from the seed."""
    from durakfish.game.history import EventKind

    state = new_game(seed=123)
    final = random_playout(state, random.Random(123))
    moves = [e.move for e in final.events() if e.kind is EventKind.MOVE]

    replay = new_game(seed=123)
    for move in moves:
        replay = apply_move(replay, move)
    assert replay.transposition_key() == final.transposition_key()
    assert len(moves) > 20


# ----------------------------------------------------------------------
# Hypothesis
# ----------------------------------------------------------------------
hypothesis = pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

seeds = st.integers(min_value=0, max_value=2**31 - 1)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seeds, st.integers(min_value=0, max_value=60))
def test_any_reachable_position_is_valid(seed: int, depth: int) -> None:
    rng = random.Random(seed)
    state = new_game(seed=seed)
    for _ in range(depth):
        if state.is_over:
            break
        state = apply_move(state, rng.choice(get_legal_moves(state)))
    state.validate()
    assert sum(card_locations(state).values()) == 36


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seeds)
def test_generated_moves_are_exactly_the_legal_ones(seed: int) -> None:
    rng = random.Random(seed)
    state = new_game(seed=seed)
    for _ in range(40):
        if state.is_over:
            break
        legal = set(get_legal_moves(state))
        for move in legal:
            assert is_legal_move(state, move)
        # Every card in play, tried as an attack.
        for card in DECK:
            move = AttackMove(card)
            assert is_legal_move(state, move) == (move in legal)
        state = apply_move(state, rng.choice(tuple(legal)))


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seeds)
def test_state_serialization_round_trips_anywhere_in_a_game(seed: int) -> None:
    rng = random.Random(seed)
    state = new_game(seed=seed)
    for _ in range(rng.randrange(0, 40)):
        if state.is_over:
            break
        state = apply_move(state, rng.choice(get_legal_moves(state)))
    restored = GameState.from_dict(state.to_dict())
    assert restored.transposition_key() == state.transposition_key()
    restored.validate()
