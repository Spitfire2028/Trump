"""GameState: invariants, canonical hashing, serialization, history."""

from __future__ import annotations

import dataclasses
import json
import random

import pytest

from durakfish.cards import Card, Suit
from durakfish.exceptions import InvalidStateError
from durakfish.game import (
    END_ATTACK,
    TAKE,
    AttackMove,
    DefenseMove,
    EventKind,
    GameState,
    Phase,
    apply_move,
    get_legal_moves,
    new_game,
    random_playout,
)
from tests.support import make_state


# ----------------------------------------------------------------------
# Immutability
# ----------------------------------------------------------------------
def test_state_is_frozen() -> None:
    state = new_game(seed=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.attacker = 1  # type: ignore[misc]


def test_apply_move_does_not_touch_the_input_state() -> None:
    state = new_game(seed=4)
    snapshot = state.transposition_key()
    for move in get_legal_moves(state):
        apply_move(state, move)
    assert state.transposition_key() == snapshot


def test_deep_search_style_branching_leaves_the_root_intact() -> None:
    root = new_game(seed=6)
    snapshot = root.transposition_key()
    rng = random.Random(6)
    for _ in range(50):
        state = root
        for _ in range(20):
            if state.is_over:
                break
            state = apply_move(state, rng.choice(get_legal_moves(state)))
    assert root.transposition_key() == snapshot


# ----------------------------------------------------------------------
# Roles and phases
# ----------------------------------------------------------------------
def test_current_player_is_derived_from_the_phase() -> None:
    state = make_state(hand0="6S", hand1="7S", trump=Suit.HEARTS, attack_limit=1)
    assert state.current_player == state.attacker == 0
    defending = apply_move(state, AttackMove(Card.parse("6S")))
    assert defending.phase is Phase.DEFENSE
    assert defending.current_player == defending.defender == 1
    taking = apply_move(defending, TAKE)
    assert taking.phase is Phase.TAKING
    assert taking.current_player == taking.attacker == 0


def test_finished_game_has_no_current_player() -> None:
    final = random_playout(new_game(seed=9), random.Random(9))
    assert final.current_player is None


# ----------------------------------------------------------------------
# Invariant checking
# ----------------------------------------------------------------------
def test_validate_accepts_a_freshly_dealt_game() -> None:
    new_game(seed=12).validate()


def test_validate_catches_a_duplicated_card() -> None:
    state = new_game(seed=13)
    broken = state.replace(hands=(state.hands[0] + (state.hands[1][0],), state.hands[1]))
    with pytest.raises(InvalidStateError, match="two places"):
        broken.validate()


def test_validate_catches_a_missing_card() -> None:
    state = new_game(seed=14)
    broken = state.replace(hands=(state.hands[0][1:], state.hands[1]))
    with pytest.raises(InvalidStateError, match="cards accounted for"):
        broken.validate()


def test_validate_catches_an_uncanonical_hand() -> None:
    state = new_game(seed=15)
    broken = state.replace(hands=(tuple(reversed(state.hands[0])), state.hands[1]))
    with pytest.raises(InvalidStateError, match="canonical order"):
        broken.validate()


def test_validate_catches_a_trump_mismatch() -> None:
    state = new_game(seed=16)
    other = next(s for s in Suit if s is not state.trump_suit)
    with pytest.raises(InvalidStateError, match="trump suit"):
        state.replace(trump_suit=other).validate()


def test_validate_catches_a_defended_slot_after_an_open_one() -> None:
    broken = make_state(
        hand0="6S",
        hand1="7S",
        trump=Suit.HEARTS,
        table=(("8C", None), ("9C", "10C")),
        phase=Phase.TAKING,
        attack_limit=3,
        validate=False,
    )
    with pytest.raises(InvalidStateError, match="follows an undefended"):
        broken.validate()


def test_validate_catches_exceeding_the_attack_limit() -> None:
    state = make_state(
        hand0="6D",
        hand1="KC",
        trump=Suit.HEARTS,
        table=(("6S", "7S"), ("6H", "7H")),
        attack_limit=1,
        validate=False,
    )
    with pytest.raises(InvalidStateError, match="attacks made"):
        state.validate()


def test_validate_catches_phase_and_table_disagreement() -> None:
    state = make_state(
        hand0="6D",
        hand1="KC",
        trump=Suit.HEARTS,
        table=(("6S", None),),
        phase=Phase.ATTACK,
        attack_limit=2,
        validate=False,
    )
    with pytest.raises(InvalidStateError, match="ATTACK phase"):
        state.validate()


def test_validate_catches_a_durak_with_no_cards() -> None:
    state = make_state(
        hand0="",
        hand1="",
        trump=Suit.HEARTS,
        phase=Phase.GAME_OVER,
        durak=1,
        attack_limit=0,
        validate=False,
    )
    with pytest.raises(InvalidStateError, match="durak must be"):
        state.validate()


# ----------------------------------------------------------------------
# Hashing
# ----------------------------------------------------------------------
def test_transposition_key_is_stable_and_hashable() -> None:
    a = new_game(seed=20)
    b = new_game(seed=20)
    assert a.transposition_key() == b.transposition_key()
    assert hash(a.transposition_key()) == hash(b.transposition_key())


def test_transposition_key_ignores_history() -> None:
    """Two identical positions reached differently must share a key."""
    state = new_game(seed=22)
    bare = state.replace(history=None)
    assert state.transposition_key() == bare.transposition_key()
    assert state.event_count != bare.event_count


def test_transposition_key_distinguishes_talon_order() -> None:
    """Different talon orders are different positions: the next draw differs."""
    a = make_state(hand0="6S", hand1="7S", trump=Suit.HEARTS, talon="8C 9C AH")
    b = make_state(hand0="6S", hand1="7S", trump=Suit.HEARTS, talon="9C 8C AH")
    assert a.transposition_key() != b.transposition_key()


def test_transposition_key_is_insensitive_to_pickup_order() -> None:
    """Hands are canonicalised, so the order cards arrived cannot leak in."""
    from durakfish.game.state import add_cards

    hand_a = add_cards((), [Card.parse("6S"), Card.parse("AH")])
    hand_b = add_cards((), [Card.parse("AH"), Card.parse("6S")])
    assert hand_a == hand_b


# ----------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------
def test_state_round_trips_through_json() -> None:
    state = new_game(seed=30)
    for _ in range(12):
        if state.is_over:
            break
        state = apply_move(state, get_legal_moves(state)[0])
    encoded = json.dumps(state.to_dict())
    restored = GameState.from_dict(json.loads(encoded))
    assert restored.transposition_key() == state.transposition_key()
    assert [e.to_dict() for e in restored.events()] == [
        e.to_dict() for e in state.events()
    ]
    restored.validate()


def test_serialization_can_omit_history() -> None:
    state = new_game(seed=31)
    data = state.to_dict(include_history=False)
    assert "history" not in data
    assert GameState.from_dict(data).transposition_key() == state.transposition_key()


def test_serialized_form_is_human_readable() -> None:
    data = new_game(seed=32).to_dict(include_history=False)
    assert all(isinstance(c, str) for c in data["hands"][0])
    assert data["phase"] == "ATTACK"


# ----------------------------------------------------------------------
# History
# ----------------------------------------------------------------------
def test_history_records_the_deal_then_every_move() -> None:
    state = new_game(seed=40)
    events = state.events()
    assert len(events) == 1 and events[0].kind is EventKind.DEAL
    assert events[0].revealed == (state.trump_card,)

    mover = state.attacker
    state = apply_move(state, get_legal_moves(state)[0])
    events = state.events()
    assert events[-1].kind is EventKind.MOVE
    assert events[-1].player == mover


def test_history_is_shared_between_branches_not_copied() -> None:
    """Structural sharing: branches extend the same tail object."""
    state = new_game(seed=41)
    moves = get_legal_moves(state)
    a = apply_move(state, moves[0])
    b = apply_move(state, moves[1])
    assert a.history is not b.history
    assert a.history.parent is b.history.parent is state.history


def test_bout_end_event_records_where_the_cards_went() -> None:
    state = make_state(hand0="6S", hand1="7S KC", trump=Suit.HEARTS, attack_limit=2)
    state = apply_move(state, AttackMove(Card.parse("6S")))
    state = apply_move(state, DefenseMove(Card.parse("6S"), Card.parse("7S")))
    state = apply_move(state, END_ATTACK)
    end = state.events()[-2] if state.is_over else state.events()[-1]
    assert end.kind is EventKind.BOUT_END
    assert set(end.discarded) == {Card.parse("6S"), Card.parse("7S")}
    assert end.taken == ()


def test_take_event_records_the_recipient() -> None:
    state = make_state(hand0="6S", hand1="KC KD", trump=Suit.HEARTS, attack_limit=2)
    state = apply_move(state, AttackMove(Card.parse("6S")))
    state = apply_move(state, TAKE)
    state = apply_move(state, END_ATTACK)
    bout_end = next(e for e in state.events() if e.kind is EventKind.BOUT_END)
    assert bout_end.taken == (Card.parse("6S"),)
    assert bout_end.taken_by == 1
    assert bout_end.discarded == ()


# ----------------------------------------------------------------------
# Endgame observability
# ----------------------------------------------------------------------
def test_open_information_begins_exactly_when_the_talon_empties() -> None:
    state = new_game(seed=50)
    assert not state.is_open_information()
    rng = random.Random(50)
    while not state.is_over:
        assert state.is_open_information() == (state.talon_size == 0)
        state = apply_move(state, rng.choice(get_legal_moves(state)))


def test_open_information_means_the_opponent_hand_is_deducible() -> None:
    """The claim behind the endgame solver, checked directly."""
    from durakfish.cards.deck import standard_cards

    state = new_game(seed=51)
    rng = random.Random(51)
    while not state.is_over and state.talon_size:
        state = apply_move(state, rng.choice(get_legal_moves(state)))
    if state.is_over:
        pytest.skip("game finished before the talon emptied")

    assert state.is_open_information()
    known_to_player_0 = (
        set(state.hands[0]) | set(state.discard) | set(state.table_cards)
    )
    deduced = set(standard_cards(36)) - known_to_player_0
    assert deduced == set(state.hands[1])


# ----------------------------------------------------------------------
# Debug output
# ----------------------------------------------------------------------
def test_describe_mentions_the_essentials() -> None:
    text = new_game(seed=60).describe()
    assert "trump" in text and "P0" in text and "P1" in text
    assert "table" in text
