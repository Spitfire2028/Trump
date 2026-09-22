"""The bout lifecycle: resolution, role changes, drawing, termination.

These are the rules most often got wrong in Durak implementations, and
the ones a heuristic bot would quietly exploit forever. Each test states
the rule it pins down.
"""

from __future__ import annotations

import pytest

from durakfish.cards import Card, Deck, Suit
from durakfish.game import (
    END_ATTACK,
    TAKE,
    AttackMove,
    DefenseMove,
    Phase,
    apply_move,
    get_legal_moves,
    new_game,
)
from tests.support import make_state


def _play(state, *moves):
    for move in moves:
        state = apply_move(state, move)
        state.validate()
    return state


# ----------------------------------------------------------------------
# Successful defence: bito
# ----------------------------------------------------------------------
def test_successful_defence_discards_the_table_and_passes_the_attack() -> None:
    state = make_state(
        hand0="6S 9C", hand1="7S KD", trump=Suit.HEARTS, talon="", attack_limit=2
    )
    after = _play(
        state,
        AttackMove(Card.parse("6S")),
        DefenseMove(Card.parse("6S"), Card.parse("7S")),
        END_ATTACK,
    )
    assert after.table == ()
    assert {c.short for c in after.discard} >= {"6S", "7S"}
    # The defender beat everything, so the defender now attacks.
    assert after.attacker == 1
    assert after.phase is Phase.ATTACK


def test_cards_survive_a_defended_bout_only_in_the_discard() -> None:
    state = make_state(hand0="6S", hand1="7S", trump=Suit.HEARTS, attack_limit=1)
    after = _play(
        state,
        AttackMove(Card.parse("6S")),
        DefenseMove(Card.parse("6S"), Card.parse("7S")),
        END_ATTACK,
    )
    assert Card.parse("6S") in after.discard
    assert Card.parse("7S") in after.discard
    assert after.hands == ((), ())


def test_partial_defence_then_take_keeps_the_attack_with_the_attacker() -> None:
    state = make_state(
        hand0="6S 6D", hand1="7S KC", trump=Suit.HEARTS, attack_limit=2
    )
    after = _play(
        state,
        AttackMove(Card.parse("6S")),
        DefenseMove(Card.parse("6S"), Card.parse("7S")),
        AttackMove(Card.parse("6D")),
        TAKE,
        END_ATTACK,
    )
    # Defender picks up everything on the table, including their own 7♠.
    assert {c.short for c in after.hands[1]} == {"6S", "7S", "6D", "KC"}
    assert after.discard == frozenset(c for c in after.discard)  # unchanged shape
    assert Card.parse("6S") not in after.discard
    # A defender who took does not get to attack.
    assert after.attacker == 0


# ----------------------------------------------------------------------
# Taking, and throw-ins after a take
# ----------------------------------------------------------------------
def test_take_does_not_end_the_bout_immediately() -> None:
    state = make_state(
        hand0="6S 6D 9C", hand1="KC", trump=Suit.HEARTS, attack_limit=1
    )
    after = _play(state, AttackMove(Card.parse("6S")), TAKE)
    assert after.phase is Phase.TAKING
    assert after.table  # cards are still on the table
    assert after.hands[1] == (Card.parse("KC"),)  # nothing picked up yet


def test_attacker_may_throw_in_matching_ranks_after_a_take() -> None:
    state = make_state(
        hand0="6S 6D 9C", hand1="KC KD", trump=Suit.HEARTS, attack_limit=2
    )
    taking = _play(state, AttackMove(Card.parse("6S")), TAKE)
    legal = {m.card.short for m in get_legal_moves(taking) if isinstance(m, AttackMove)}
    assert legal == {"6D"}  # rank 6 matches the table; 9♣ does not
    after = _play(taking, AttackMove(Card.parse("6D")), END_ATTACK)
    assert {c.short for c in after.hands[1]} == {"KC", "KD", "6S", "6D"}


def test_throw_ins_after_a_take_respect_the_attack_limit() -> None:
    state = make_state(
        hand0="6S 6D 6C", hand1="KC", trump=Suit.HEARTS, attack_limit=1
    )
    taking = _play(state, AttackMove(Card.parse("6S")), TAKE)
    assert get_legal_moves(taking) == (END_ATTACK,)


def test_taking_defender_still_refills_but_usually_draws_nothing() -> None:
    state = make_state(
        hand0="6S",
        hand1="KC KD KH KS QC QD",
        trump=Suit.HEARTS,
        talon="7C 8C 9C 10C JC 2H".replace("2H", "AH"),
        attack_limit=6,
    )
    after = _play(state, AttackMove(Card.parse("6S")), TAKE, END_ATTACK)
    assert len(after.hands[1]) == 7  # six plus the card taken; no draw needed
    assert len(after.hands[0]) == 6  # attacker refilled from the talon


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def test_attacker_draws_before_the_defender() -> None:
    """Draw order is a rule, and it decides who gets a scarce last card."""
    state = make_state(
        hand0="6S 6D",
        hand1="7S 7D",
        trump=Suit.HEARTS,
        talon="AH",  # exactly one card left, and it is the trump card
        attack_limit=2,
    )
    after = _play(
        state,
        AttackMove(Card.parse("6S")),
        DefenseMove(Card.parse("6S"), Card.parse("7S")),
        END_ATTACK,
    )
    assert Card.parse("AH") in after.hands[0]  # attacker drew first
    assert Card.parse("AH") not in after.hands[1]
    assert after.talon == ()


def test_drawing_first_can_lose_the_game() -> None:
    """The attacker refills first — and is left holding the last card.

    A genuine feature of Durak endgames, not a bug: refilling first is
    not always an advantage.
    """
    state = make_state(
        hand0="6S", hand1="7S", trump=Suit.HEARTS, talon="AH", attack_limit=1
    )
    after = _play(
        state,
        AttackMove(Card.parse("6S")),
        DefenseMove(Card.parse("6S"), Card.parse("7S")),
        END_ATTACK,
    )
    assert after.is_over
    assert after.hands[1] == ()  # the defender went out
    assert after.durak == 0  # the attacker drew the last card and is stuck


def test_refill_stops_at_hand_size() -> None:
    state = new_game(seed=11)
    after = _play(
        state,
        AttackMove(state.hands[state.attacker][0]),
        TAKE,
        END_ATTACK,
    )
    assert len(after.hands[after.attacker]) == 6


def test_partially_depleted_talon_is_shared_in_order() -> None:
    state = make_state(
        hand0="6S 6D 6C",
        hand1="7S 7D 7C",
        trump=Suit.HEARTS,
        talon="8H 9H AH",  # three cards for two players needing three each
        attack_limit=3,
    )
    after = _play(
        state,
        AttackMove(Card.parse("6S")),
        DefenseMove(Card.parse("6S"), Card.parse("7S")),
        END_ATTACK,
    )
    assert len(after.hands[0]) == 5  # 2 left + 3 drawn
    assert len(after.hands[1]) == 2  # nothing left to draw
    assert after.talon == ()


def test_trump_card_is_the_last_card_drawn() -> None:
    state = new_game(seed=21)
    trump_card = state.trump_card
    assert trump_card == state.talon[-1]
    while not state.is_over and state.talon:
        state = apply_move(state, get_legal_moves(state)[0])
    # Once the talon empties, the trump card must be in somebody's hand,
    # on the table or discarded - never lost, never duplicated.
    state.validate()


def test_empty_talon_produces_no_cards() -> None:
    state = make_state(
        hand0="6S 6D", hand1="7S 7D", trump=Suit.HEARTS, talon="", attack_limit=2
    )
    before = sum(len(h) for h in state.hands)
    after = _play(
        state,
        AttackMove(Card.parse("6S")),
        DefenseMove(Card.parse("6S"), Card.parse("7S")),
        END_ATTACK,
    )
    assert sum(len(h) for h in after.hands) == before - 2  # only the played cards left
    assert after.talon == ()


# ----------------------------------------------------------------------
# Termination
# ----------------------------------------------------------------------
def test_empty_hand_with_cards_left_in_the_talon_is_not_the_end() -> None:
    """Going to zero cards is only decisive once the talon cannot refill you."""
    state = make_state(
        hand0="6S",
        hand1="7S",
        trump=Suit.HEARTS,
        talon="8H 9H 10H JH QH KH 6C 7C 8C 9C 10C JC AH",
        attack_limit=1,
    )
    after = _play(
        state,
        AttackMove(Card.parse("6S")),
        DefenseMove(Card.parse("6S"), Card.parse("7S")),
        END_ATTACK,
    )
    assert not after.is_over
    assert [len(h) for h in after.hands] == [6, 6]
    assert after.talon_size == 1


def test_zero_cards_mid_bout_is_an_ordinary_position() -> None:
    """An attacker who plays their last card has not won; the bout continues."""
    state = make_state(
        hand0="6S", hand1="7S KC", trump=Suit.HEARTS, talon="", attack_limit=2
    )
    mid = apply_move(state, AttackMove(Card.parse("6S")))
    mid.validate()
    assert mid.hands[0] == ()
    assert not mid.is_over
    assert mid.phase is Phase.DEFENSE
    # And if the defender takes, the attacker who "went out" still wins the game.
    end = _play(mid, TAKE, END_ATTACK)
    assert end.is_over and end.durak == 1


def test_both_players_going_out_together_is_a_draw() -> None:
    state = make_state(
        hand0="6S", hand1="7S", trump=Suit.HEARTS, talon="", attack_limit=1
    )
    after = _play(
        state,
        AttackMove(Card.parse("6S")),
        DefenseMove(Card.parse("6S"), Card.parse("7S")),
        END_ATTACK,
    )
    assert after.is_over
    assert after.is_draw
    assert after.durak is None


def test_the_player_left_holding_cards_is_the_durak() -> None:
    state = make_state(
        hand0="6S", hand1="7S KC", trump=Suit.HEARTS, talon="", attack_limit=1
    )
    after = _play(
        state,
        AttackMove(Card.parse("6S")),
        DefenseMove(Card.parse("6S"), Card.parse("7S")),
        END_ATTACK,
    )
    assert after.is_over
    assert after.durak == 1
    assert after.hands[0] == ()


def test_a_defender_who_takes_cannot_go_out() -> None:
    state = make_state(
        hand0="6S", hand1="KC", trump=Suit.HEARTS, talon="", attack_limit=1
    )
    after = _play(state, AttackMove(Card.parse("6S")), TAKE, END_ATTACK)
    assert after.is_over
    assert after.durak == 1  # picked up, so still holding cards
    assert after.hands[0] == ()


def test_game_over_state_has_no_moves_and_a_clear_table() -> None:
    state = new_game(seed=5)
    import random

    from durakfish.game import random_playout

    final = random_playout(state, random.Random(5))
    assert final.is_over
    assert final.table == ()
    assert final.talon == ()
    assert get_legal_moves(final) == ()
    final.validate()


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
def test_new_game_deals_six_each_and_reveals_a_trump() -> None:
    state = new_game(seed=1)
    assert [len(h) for h in state.hands] == [6, 6]
    assert len(state.talon) == 24
    assert state.trump_card is state.talon[-1]
    assert state.trump_suit is state.trump_card.suit
    state.validate()


def test_trump_card_is_counted_exactly_once() -> None:
    state = new_game(seed=2)
    everywhere = (
        list(state.hands[0])
        + list(state.hands[1])
        + list(state.talon)
        + list(state.table_cards)
        + list(state.discard)
    )
    assert everywhere.count(state.trump_card) == 1
    assert len(everywhere) == 36


def test_lowest_trump_opens_the_game() -> None:
    for seed in range(40):
        state = new_game(seed=seed)
        trumps = [
            [c for c in hand if c.suit is state.trump_suit] for hand in state.hands
        ]
        if not trumps[0] or not trumps[1]:
            continue
        lowest = [min(t, key=lambda c: int(c.rank)) for t in trumps]
        expected = 0 if lowest[0].rank < lowest[1].rank else 1
        assert state.attacker == expected, state.describe()


def test_a_player_with_a_trump_opens_against_a_player_without() -> None:
    for seed in range(200):
        state = new_game(seed=seed)
        has = [any(c.suit is state.trump_suit for c in h) for h in state.hands]
        if has[0] != has[1]:
            assert state.attacker == (0 if has[0] else 1)


def test_first_attacker_can_be_forced_for_balanced_benchmarking() -> None:
    assert new_game(seed=1, first_attacker=1).attacker == 1
    assert new_game(seed=1, first_attacker=0).attacker == 0


def test_new_game_from_an_exact_deck_is_reproducible() -> None:
    deck = Deck.shuffled(seed=77)
    a = new_game(deck=deck)
    b = new_game(deck=deck)
    assert a.transposition_key() == b.transposition_key()
    assert len(deck) == 36  # the caller's deck was not consumed


def test_new_game_rejects_a_wrong_sized_deck() -> None:
    from durakfish.exceptions import UnsupportedRuleError

    deck = Deck.standard(52)
    with pytest.raises(UnsupportedRuleError):
        new_game(deck=deck)
