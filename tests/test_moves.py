"""Typed move objects."""

from __future__ import annotations

import dataclasses

import pytest

from durakfish.cards import Card
from durakfish.game import (
    END_ATTACK,
    TAKE,
    AttackMove,
    DefenseMove,
    EndAttack,
    MoveType,
    TakeCards,
    move_from_dict,
)

MOVES = [
    AttackMove(Card.parse("7S")),
    DefenseMove(Card.parse("7S"), Card.parse("9S")),
    TAKE,
    END_ATTACK,
]


def test_each_move_carries_its_type() -> None:
    assert AttackMove(Card.parse("7S")).type is MoveType.ATTACK
    assert DefenseMove(Card.parse("7S"), Card.parse("9S")).type is MoveType.DEFEND
    assert TAKE.type is MoveType.TAKE
    assert END_ATTACK.type is MoveType.END_ATTACK


def test_moves_are_immutable_and_hashable() -> None:
    assert len(set(MOVES)) == 4
    with pytest.raises(dataclasses.FrozenInstanceError):
        MOVES[0].card = Card.parse("8S")  # type: ignore[misc]


def test_singletons_equal_freshly_built_instances() -> None:
    assert TakeCards() == TAKE
    assert EndAttack() == END_ATTACK
    assert hash(TakeCards()) == hash(TAKE)


def test_different_move_types_are_never_equal() -> None:
    assert TAKE != END_ATTACK
    assert AttackMove(Card.parse("7S")) != DefenseMove(
        Card.parse("7S"), Card.parse("9S")
    )


def test_defence_move_distinguishes_attacker_and_defender_cards() -> None:
    a = DefenseMove(Card.parse("7S"), Card.parse("9S"))
    b = DefenseMove(Card.parse("9S"), Card.parse("7S"))
    assert a != b
    assert a.attacking_card == b.defending_card


@pytest.mark.parametrize("move", MOVES)
def test_moves_round_trip_through_dicts(move: object) -> None:
    assert move_from_dict(move.to_dict()) == move  # type: ignore[attr-defined]


def test_move_encoding_is_json_friendly() -> None:
    import json

    for move in MOVES:
        json.dumps(move.to_dict())


def test_unknown_encoding_is_rejected() -> None:
    with pytest.raises(ValueError):
        move_from_dict({"type": "teleport"})


def test_moves_have_readable_descriptions() -> None:
    assert str(AttackMove(Card.parse("7S"))) == "attack 7\u2660"
    assert (
        str(DefenseMove(Card.parse("7S"), Card.parse("9S")))
        == "defend 7\u2660 with 9\u2660"
    )
    assert str(TAKE) == "take"
    assert str(END_ATTACK) == "end attack"
