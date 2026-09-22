"""Legal move generation, and the agreement between the two rule paths.

The most valuable test here is
:func:`test_generation_and_predicate_agree_exhaustively`: for many random
positions it enumerates *every conceivable* move and asserts that
``is_legal_move`` accepts exactly those that ``get_legal_moves``
produced. The two functions are written independently, so agreement is
evidence rather than tautology.
"""

from __future__ import annotations

import random

import pytest

from durakfish.cards import Card, Rank, Suit
from durakfish.cards.deck import standard_cards
from durakfish.exceptions import IllegalMoveError
from durakfish.game import (
    END_ATTACK,
    TAKE,
    AttackMove,
    DefenseMove,
    Phase,
    apply_move,
    beats,
    get_legal_moves,
    is_legal_move,
    legal_attack_cards,
    new_game,
)
from tests.support import make_state

DECK = standard_cards(36)


# ----------------------------------------------------------------------
# Opening the bout
# ----------------------------------------------------------------------
def test_opening_attack_allows_any_card_and_forbids_passing() -> None:
    state = make_state(hand0="6S 10H QD AC", hand1="7S 8H 9D JC", trump=Suit.HEARTS)
    moves = get_legal_moves(state)
    assert {m.card for m in moves if isinstance(m, AttackMove)} == set(state.hands[0])
    assert END_ATTACK not in moves  # a bout cannot begin with a pass
    assert TAKE not in moves


def test_defender_has_no_moves_during_the_attack_phase() -> None:
    state = make_state(hand0="6S", hand1="7S", trump=Suit.HEARTS)
    assert get_legal_moves(state, player=1) == ()
    assert not is_legal_move(state, AttackMove(Card.parse("7S")), player=1)


def test_finished_game_offers_no_moves() -> None:
    state = make_state(
        hand0="",
        hand1="6S",
        trump=Suit.HEARTS,
        phase=Phase.GAME_OVER,
        durak=1,
        attack_limit=0,
    )
    assert get_legal_moves(state) == ()
    assert not is_legal_move(state, TAKE)
    with pytest.raises(IllegalMoveError):
        apply_move(state, TAKE)


# ----------------------------------------------------------------------
# The rank-addition rule
# ----------------------------------------------------------------------
def test_additions_must_match_a_rank_on_the_table() -> None:
    """Spec example: table holds 7♠ and Q♥, so only 7s and Qs may be added."""
    state = make_state(
        hand0="7D 7C QS 9C 6D AC",
        hand1="10S 10D",
        trump=Suit.SPADES,
        table=(("7S", "8S"), ("QH", "AH")),
        attack_limit=6,
    )
    # Table shown to the test as attack/defence pairs; ranks present are
    # 7, 8, Q, A. The hand's 9♣ and 6♦ must be rejected.
    legal = {m.card.short for m in get_legal_moves(state) if isinstance(m, AttackMove)}
    assert legal == {"7D", "7C", "QS", "AC"}
    assert "9C" not in legal
    assert "6D" not in legal


def test_defence_card_ranks_count_as_addable_ranks() -> None:
    """Standard Podkidnoy: the rank of a defending card may also be added."""
    state = make_state(
        hand0="9H 9D",
        hand1="KC KD",
        trump=Suit.CLUBS,
        table=(("7S", "9S"),),
        attack_limit=3,
    )
    legal = {m.card.short for m in get_legal_moves(state) if isinstance(m, AttackMove)}
    assert legal == {"9H", "9D"}  # 9 came from the *defending* card


def test_every_off_table_rank_is_rejected_exhaustively() -> None:
    """For a fixed table, check all 9 ranks: exactly the table ranks pass."""
    table_ranks = {Rank.SEVEN, Rank.QUEEN}
    for rank in (r for r in Rank if r >= Rank.SIX):
        candidate = Card.of(rank, Suit.DIAMONDS)
        filler = "6C" if candidate.short == "7D" else "7D"
        hand = f"{candidate.short} {filler}"
        state = make_state(
            hand0=hand,
            hand1="10C 10H",
            trump=Suit.CLUBS,
            table=(("7S", "7H"), ("QS", "QH")),
            attack_limit=6,
        )
        move = AttackMove(candidate)
        expected = rank in table_ranks
        assert is_legal_move(state, move) is expected, candidate
        assert (move in get_legal_moves(state)) is expected, candidate


def test_opening_card_is_unrestricted_but_additions_are_not() -> None:
    opening = make_state(hand0="6S 9C", hand1="7S 7D", trump=Suit.HEARTS)
    assert len(legal_attack_cards(opening)) == 2

    after = apply_move(opening, AttackMove(Card.parse("6S")))
    after = apply_move(after, DefenseMove(Card.parse("6S"), Card.parse("7S")))
    # Table now shows 6 and 7; the 9♣ is no longer addable.
    assert {c.short for c in legal_attack_cards(after)} == set()


# ----------------------------------------------------------------------
# The attack limit
# ----------------------------------------------------------------------
def test_attack_limit_equals_defender_hand_size_at_bout_start() -> None:
    state = make_state(hand0="6S 6H 6D 6C", hand1="7S 7H", trump=Suit.SPADES)
    assert state.attack_limit == 2  # defender holds two cards


def test_attack_limit_is_capped_by_the_rules_maximum() -> None:
    state = new_game(seed=3)
    assert state.attack_limit == 6  # both hands hold six


def test_cards_already_on_the_table_count_toward_the_limit() -> None:
    state = make_state(
        hand0="6D 6C",
        hand1="8H",
        trump=Suit.HEARTS,
        table=(("6S", "7S"), ("6H", "9H")),
        attack_limit=2,
    )
    assert state.attacks_made == 2
    assert legal_attack_cards(state) == ()
    assert get_legal_moves(state) == (END_ATTACK,)
    assert not is_legal_move(state, AttackMove(Card.parse("6D")))


def test_attacker_may_fill_the_limit_but_not_exceed_it() -> None:
    state = make_state(
        hand0="6D 6C",
        hand1="8H 9H 10H",
        trump=Suit.HEARTS,
        table=(("6S", "7S"),),
        attack_limit=3,
    )
    assert {m.card.short for m in get_legal_moves(state) if isinstance(m, AttackMove)} == {
        "6D",
        "6C",
    }
    after = apply_move(state, AttackMove(Card.parse("6D")))
    after = apply_move(after, DefenseMove(Card.parse("6D"), Card.parse("8H")))
    assert after.attacks_made == 2
    # One slot left in the limit.
    assert any(isinstance(m, AttackMove) for m in get_legal_moves(after))
    after = apply_move(after, AttackMove(Card.parse("6C")))
    after = apply_move(after, DefenseMove(Card.parse("6C"), Card.parse("9H")))
    assert after.attacks_made == 3 == after.attack_limit
    assert get_legal_moves(after) == (END_ATTACK,)


def test_limit_guarantees_the_defender_always_has_a_card_to_play() -> None:
    """The limit rule makes 'more attacks than the defender can answer' impossible."""
    rng = random.Random(99)
    for seed in range(60):
        state = new_game(seed=seed)
        for _ in range(200):
            if state.is_over:
                break
            if state.phase is Phase.DEFENSE:
                assert state.hands[state.defender], "defender was left with no cards"
            state = apply_move(state, rng.choice(get_legal_moves(state)))


# ----------------------------------------------------------------------
# Defence
# ----------------------------------------------------------------------
def test_defence_generation_matches_beats_exactly() -> None:
    trump = Suit.HEARTS
    for attacking in DECK[:12]:
        hand = " ".join(c.short for c in DECK[12:20])
        state = make_state(
            hand0="6C" if attacking.short != "6C" else "7C",
            hand1=hand,
            trump=trump,
            table=((attacking.short, None),),
            phase=Phase.DEFENSE,
            attack_limit=6,
        )
        generated = {
            m.defending_card
            for m in get_legal_moves(state)
            if isinstance(m, DefenseMove)
        }
        expected = {c for c in state.hands[1] if beats(c, attacking, trump)}
        assert generated == expected, attacking


def test_defence_move_names_both_cards() -> None:
    state = make_state(
        hand0="6C",
        hand1="9S KH",
        trump=Suit.HEARTS,
        table=(("7S", None),),
        phase=Phase.DEFENSE,
        attack_limit=2,
    )
    moves = [m for m in get_legal_moves(state) if isinstance(m, DefenseMove)]
    assert {(m.attacking_card.short, m.defending_card.short) for m in moves} == {
        ("7S", "9S"),
        ("7S", "KH"),
    }


def test_a_card_that_does_not_beat_is_rejected() -> None:
    state = make_state(
        hand0="6C",
        hand1="6S KC",
        trump=Suit.HEARTS,
        table=(("7S", None),),
        phase=Phase.DEFENSE,
        attack_limit=2,
    )
    assert not is_legal_move(
        state, DefenseMove(Card.parse("7S"), Card.parse("6S"))
    )
    assert not is_legal_move(
        state, DefenseMove(Card.parse("7S"), Card.parse("KC"))
    )


def test_defender_cannot_use_a_card_from_the_attackers_hand() -> None:
    state = make_state(
        hand0="AH",
        hand1="6S",
        trump=Suit.HEARTS,
        table=(("7S", None),),
        phase=Phase.DEFENSE,
        attack_limit=1,
    )
    # A♥ would beat 7♠, but it belongs to the attacker.
    assert not is_legal_move(
        state, DefenseMove(Card.parse("7S"), Card.parse("AH"))
    )


def test_defender_cannot_defend_a_card_that_is_not_open() -> None:
    state = make_state(
        hand0="6C",
        hand1="AH",
        trump=Suit.HEARTS,
        table=(("7S", "8S"), ("9S", None)),
        phase=Phase.DEFENSE,
        attack_limit=3,
    )
    assert is_legal_move(state, DefenseMove(Card.parse("9S"), Card.parse("AH")))
    assert not is_legal_move(state, DefenseMove(Card.parse("7S"), Card.parse("AH")))


def test_take_is_always_available_to_the_defender() -> None:
    state = make_state(
        hand0="6C",
        hand1="AH",
        trump=Suit.HEARTS,
        table=(("7S", None),),
        phase=Phase.DEFENSE,
        attack_limit=1,
    )
    assert TAKE in get_legal_moves(state)
    assert is_legal_move(state, TAKE)


def test_take_is_not_a_card_move_and_is_illegal_when_attacking() -> None:
    state = make_state(hand0="6S", hand1="7S", trump=Suit.HEARTS)
    assert not is_legal_move(state, TAKE)


# ----------------------------------------------------------------------
# Ending the attack
# ----------------------------------------------------------------------
def test_end_attack_needs_something_on_the_table() -> None:
    empty = make_state(hand0="6S", hand1="7S", trump=Suit.HEARTS)
    assert not is_legal_move(empty, END_ATTACK)
    with pytest.raises(IllegalMoveError):
        apply_move(empty, END_ATTACK)


def test_end_attack_is_the_only_move_when_nothing_can_be_added() -> None:
    state = make_state(
        hand0="9C",
        hand1="KH",
        trump=Suit.HEARTS,
        table=(("7S", "8S"),),
        attack_limit=3,
    )
    assert get_legal_moves(state) == (END_ATTACK,)


# ----------------------------------------------------------------------
# The cross-check
# ----------------------------------------------------------------------
def _all_conceivable_moves() -> list[object]:
    moves: list[object] = [TAKE, END_ATTACK]
    moves.extend(AttackMove(c) for c in DECK)
    return moves


def test_generation_and_predicate_agree_exhaustively() -> None:
    """Over many real positions, the two legality paths must agree on every move."""
    rng = random.Random(4242)
    candidates = _all_conceivable_moves()
    positions = 0
    for seed in range(40):
        state = new_game(seed=seed)
        for _ in range(300):
            if state.is_over:
                break
            legal = set(get_legal_moves(state))

            for move in candidates:
                assert is_legal_move(state, move) == (move in legal), (
                    f"disagreement on {move}\n{state.describe()}"
                )
            # Defence moves depend on the open attack, so build those pairs too.
            for slot in state.table:
                for card in DECK:
                    move = DefenseMove(slot.attacking_card, card)
                    assert is_legal_move(state, move) == (move in legal), (
                        f"disagreement on {move}\n{state.describe()}"
                    )
            positions += 1
            state = apply_move(state, rng.choice(tuple(legal)))
    assert positions > 500


def test_every_generated_move_applies_without_error() -> None:
    rng = random.Random(7)
    for seed in range(30):
        state = new_game(seed=seed)
        for _ in range(300):
            if state.is_over:
                break
            for move in get_legal_moves(state):
                successor = apply_move(state, move)
                successor.validate()
            state = apply_move(state, rng.choice(get_legal_moves(state)))
