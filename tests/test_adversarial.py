"""Adversarial audit of the Phase 2 rules engine.

Methodology: wherever practical these tests check the engine against
*independently derived* expectations rather than against another engine
function. ``beats`` is checked against a set-membership construction (the
set of cards that beat X is built directly, not by asking a predicate).
Card conservation is recounted from scratch. Legality is cross-checked by
enumerating a synthetic universe of moves — including malformed ones that
``get_legal_moves`` could never emit — and confirming the two independent
legality paths partition it identically.

These are regression tests, not one-off scripts: an audit that is not
re-run on every commit stops being true the moment someone edits.
"""

from __future__ import annotations

import itertools
import random

import pytest

from durakfish.cards import Card, Rank, Suit, parse_cards
from durakfish.cards.deck import standard_cards
from durakfish.exceptions import IllegalMoveError, InvalidStateError
from durakfish.game import (
    AttackMove,
    DefenseMove,
    EndAttack,
    GameState,
    Move,
    Phase,
    TakeCards,
    apply_move,
    beats,
    get_legal_moves,
    is_legal_move,
    new_game,
)
from durakfish.game.moves import END_ATTACK, TAKE
from tests.support import make_state

DECK = standard_cards(36)
UNIVERSE = frozenset(DECK)


# ======================================================================
# 1-3. Card comparison, audited by construction rather than by predicate
# ======================================================================
def beaters_of(attacking: Card, trump: Suit) -> frozenset[Card]:
    """Build the set of cards that beat ``attacking``, from the rules text.

    Deliberately constructive: it *enumerates* the beaters instead of
    testing a condition, so it cannot share a logic error with ``beats``.
    """
    if attacking.suit is trump:
        # Only a higher trump.
        return frozenset(
            c for c in DECK if c.suit is trump and c.rank > attacking.rank
        )
    higher_same_suit = {
        c for c in DECK if c.suit is attacking.suit and c.rank > attacking.rank
    }
    all_trumps = {c for c in DECK if c.suit is trump}
    return frozenset(higher_same_suit | all_trumps)


def test_beats_matches_a_constructive_beater_set_exhaustively() -> None:
    """36 x 36 x 4 = 5,184 combinations against set construction."""
    checked = 0
    for trump in Suit:
        for attacking in DECK:
            expected = beaters_of(attacking, trump)
            for defending in DECK:
                assert beats(defending, attacking, trump) == (
                    defending in expected
                ), f"{defending} vs {attacking}, trump {trump.symbol}"
                checked += 1
    assert checked == 5184


def test_same_rank_different_suits_across_every_trump() -> None:
    """Equal ranks never beat each other unless the defender is trump."""
    for rank in (r for r in Rank if r >= Rank.SIX):
        for trump in Suit:
            for suit_a, suit_b in itertools.permutations(Suit, 2):
                a = Card.of(rank, suit_a)
                d = Card.of(rank, suit_b)
                expected = d.suit is trump and a.suit is not trump
                assert beats(d, a, trump) is expected


def test_highest_and_lowest_card_of_every_suit() -> None:
    for trump in Suit:
        for suit in Suit:
            highest = Card.of(Rank.ACE, suit)
            lowest = Card.of(Rank.SIX, suit)
            # Nothing in the deck beats the ace of trumps.
            if suit is trump:
                assert not any(beats(c, highest, trump) for c in DECK)
                # The six of trumps is beaten by the other eight trumps only.
                beaten_by = [c for c in DECK if beats(c, lowest, trump)]
                assert len(beaten_by) == 8
                assert all(c.suit is trump for c in beaten_by)
            else:
                # A non-trump ace is beaten by the nine trumps and nothing else.
                beaten_by = [c for c in DECK if beats(c, highest, trump)]
                assert len(beaten_by) == 9
                assert all(c.suit is trump for c in beaten_by)
                # The lowest non-trump is beaten by 8 of its suit + 9 trumps.
                assert len([c for c in DECK if beats(c, lowest, trump)]) == 17


def test_no_card_beats_itself_and_beating_is_antisymmetric() -> None:
    for trump in Suit:
        for a, b in itertools.combinations_with_replacement(DECK, 2):
            if a == b:
                assert not beats(a, b, trump)
            else:
                assert not (beats(a, b, trump) and beats(b, a, trump))


# ======================================================================
# 4. Several attack/defence pairs inside one bout
# ======================================================================
def test_four_sequential_pairs_keep_their_pairing() -> None:
    state = make_state(
        hand0="7S 7C 7D 7H 8C",
        hand1="8S 9C 10D JH KC",
        trump=Suit.HEARTS,
        talon="6C 6D 6S 6H",
        attacker=0,
    )
    plan = [("7S", "8S"), ("7C", "9C"), ("7D", "10D"), ("7H", "JH")]
    for attack, defence in plan:
        state = apply_move(state, AttackMove(Card.parse(attack)))
        assert state.phase is Phase.DEFENSE
        state = apply_move(state, DefenseMove(Card.parse(attack), Card.parse(defence)))
        assert state.phase is Phase.ATTACK
        state.validate()

    # Pairing survived intact and in play order.
    assert [
        (s.attacking_card.short, s.defending_card.short) for s in state.table
    ] == plan
    assert len(state.table) == 4


def test_defence_card_can_never_defend_twice() -> None:
    """A card used in defence has left the hand, so re-use is impossible."""
    state = make_state(
        hand0="7S 7C",
        hand1="AH 9C",
        trump=Suit.HEARTS,
        talon="6C 6H",
        attacker=0,
    )
    state = apply_move(state, AttackMove(Card.parse("7S")))
    state = apply_move(state, DefenseMove(Card.parse("7S"), Card.parse("AH")))
    state = apply_move(state, AttackMove(Card.parse("7C")))
    assert Card.parse("AH") not in state.hands[1]
    assert not is_legal_move(state, DefenseMove(Card.parse("7C"), Card.parse("AH")))
    with pytest.raises(IllegalMoveError):
        apply_move(state, DefenseMove(Card.parse("7C"), Card.parse("AH")))


# ======================================================================
# 5. The attack limit, and its fixing at the start of the bout
# ======================================================================
def test_limit_is_the_defenders_hand_size_when_below_six() -> None:
    state = make_state(
        hand0="6S 7S 8S 9S 10S JS",
        hand1="6C 6D 6H",
        trump=Suit.HEARTS,
        talon="7C 7D 7H",
        attacker=0,
    )
    assert state.attack_limit == 3


def test_limit_stays_fixed_as_the_defenders_hand_shrinks() -> None:
    """The defender ends the bout with one card, but the limit is still 3."""
    state = make_state(
        hand0="6S 6C 6D 6H",
        hand1="7S 7C 7D 8H",
        trump=Suit.HEARTS,
        talon="9C 9H",
        attacker=0,
    )
    assert state.attack_limit == 4
    for attack, defence in (("6S", "7S"), ("6C", "7C"), ("6D", "7D")):
        state = apply_move(state, AttackMove(Card.parse(attack)))
        state = apply_move(state, DefenseMove(Card.parse(attack), Card.parse(defence)))
    assert len(state.hands[1]) == 1
    assert state.attack_limit == 4  # unchanged by the defender's shrinking hand
    # A fourth attack is still allowed, a fifth would not be.
    assert is_legal_move(state, AttackMove(Card.parse("6H")))
    state = apply_move(state, AttackMove(Card.parse("6H")))
    state = apply_move(state, DefenseMove(Card.parse("6H"), Card.parse("8H")))
    assert len(state.table) == 4
    assert [m for m in get_legal_moves(state)] == [END_ATTACK]


def test_limit_is_capped_at_six_when_the_defender_holds_more() -> None:
    state = make_state(
        hand0="6S 6C 6D 6H 7S 7C 7D",
        hand1="8S 8C 8D 8H 9S 9C 9D 9H",
        trump=Suit.HEARTS,
        talon="10C 10H",
        attacker=0,
    )
    assert len(state.hands[1]) == 8
    assert state.attack_limit == 6


def test_limit_counts_cards_already_on_the_table() -> None:
    state = make_state(
        hand0="6C 6D",
        hand1="7C 7D",
        trump=Suit.HEARTS,
        talon="8C 8H",
        table=(("6S", "7S"), ("6H", "8S")),
        attacker=0,
        attack_limit=3,
    )
    assert len(state.table) == 2
    assert is_legal_move(state, AttackMove(Card.parse("6C")))
    state = apply_move(state, AttackMove(Card.parse("6C")))
    state = apply_move(state, DefenseMove(Card.parse("6C"), Card.parse("7C")))
    assert len(state.table) == 3
    assert not is_legal_move(state, AttackMove(Card.parse("6D")))


# ======================================================================
# 6-7. Taking the table, and continued throw-ins
# ======================================================================
def test_defender_takes_the_entire_table_including_their_own_defences() -> None:
    state = make_state(
        hand0="6S 6C",
        hand1="7S 9H",
        trump=Suit.HEARTS,
        talon="8C 8H",
        attacker=0,
    )
    state = apply_move(state, AttackMove(Card.parse("6S")))
    state = apply_move(state, DefenseMove(Card.parse("6S"), Card.parse("7S")))
    state = apply_move(state, AttackMove(Card.parse("6C")))
    state = apply_move(state, TAKE)
    assert state.phase is Phase.TAKING
    discard_before = state.discard
    state = apply_move(state, END_ATTACK)
    # Every table card went to the defender, and nothing was retired.
    assert set(parse_cards("6S 7S 6C")) <= set(state.hands[1])
    assert state.discard == discard_before


def test_throw_ins_after_a_take_obey_the_rank_and_limit_rules() -> None:
    state = make_state(
        hand0="6S 6C 6D 9S",
        hand1="7C 7D 7H",
        trump=Suit.HEARTS,
        talon="8C 8H",
        attacker=0,
    )
    assert state.attack_limit == 3
    state = apply_move(state, AttackMove(Card.parse("6S")))
    state = apply_move(state, TAKE)
    # 9S does not match rank 6 and is refused even while taking.
    assert not is_legal_move(state, AttackMove(Card.parse("9S")))
    state = apply_move(state, AttackMove(Card.parse("6C")))
    state = apply_move(state, AttackMove(Card.parse("6D")))
    assert len(state.table) == 3
    # Limit reached: only ending remains.
    assert get_legal_moves(state) == (END_ATTACK,)


def test_defender_cannot_resume_defending_after_announcing_a_take() -> None:
    state = make_state(
        hand0="6S 6C",
        hand1="AS AH",
        trump=Suit.HEARTS,
        talon="8C 8H",
        attacker=0,
    )
    state = apply_move(state, AttackMove(Card.parse("6S")))
    state = apply_move(state, TAKE)
    assert not is_legal_move(state, DefenseMove(Card.parse("6S"), Card.parse("AS")))
    with pytest.raises(IllegalMoveError):
        apply_move(state, DefenseMove(Card.parse("6S"), Card.parse("AS")))
    assert state.current_player == 0  # the attacker acts during a take


# ======================================================================
# 8. Successful defence, resolution, role switching
# ======================================================================
def test_bito_discards_the_table_and_hands_the_attack_to_the_defender() -> None:
    state = make_state(
        hand0="6S 9C",
        hand1="7S 9H",
        trump=Suit.HEARTS,
        talon="8C 8D 8S 8H",
        attacker=0,
    )
    discard_before = state.discard
    state = apply_move(state, AttackMove(Card.parse("6S")))
    state = apply_move(state, DefenseMove(Card.parse("6S"), Card.parse("7S")))
    state = apply_move(state, END_ATTACK)
    assert state.attacker == 1  # the defender now attacks
    assert state.discard - discard_before == frozenset(parse_cards("6S 7S"))
    assert state.table == ()
    assert state.phase is Phase.ATTACK
    state.validate()


def test_taking_leaves_the_attack_with_the_same_player() -> None:
    state = make_state(
        hand0="6S 9C",
        hand1="7C 9H",
        trump=Suit.HEARTS,
        talon="8C 8D 8S 8H",
        attacker=0,
    )
    state = apply_move(state, AttackMove(Card.parse("6S")))
    state = apply_move(state, TAKE)
    state = apply_move(state, END_ATTACK)
    assert state.attacker == 0  # punished: the defender loses their turn


# ======================================================================
# 9-11. Drawing, the talon, and the trump card
# ======================================================================
def _bito_once(state: GameState, attack: str, defence: str) -> GameState:
    state = apply_move(state, AttackMove(Card.parse(attack)))
    state = apply_move(state, DefenseMove(Card.parse(attack), Card.parse(defence)))
    return apply_move(state, END_ATTACK)


def test_draw_with_plenty_of_cards_refills_both_to_six() -> None:
    state = make_state(
        hand0="6S 6C 6D 6H 7S 7C",
        hand1="8S 8C 8D 8H 9S 9C",
        trump=Suit.HEARTS,
        talon="10S 10C 10D JS JC JD QS QH",
        attacker=0,
    )
    state = _bito_once(state, "6S", "8S")
    assert [len(h) for h in state.hands] == [6, 6]
    assert len(state.talon) == 6


def test_draw_with_exactly_enough_cards_empties_the_talon_cleanly() -> None:
    """Both players spend a card, then the talon covers both refills exactly."""
    state = make_state(
        hand0="6S 6C 6D 6H 7S",
        hand1="8S 8C 8D 8H 9S",
        trump=Suit.HEARTS,
        talon="10S 10C 10D 10H",
        attacker=0,
    )
    state = _bito_once(state, "6S", "8S")
    # Each side dropped to four and needed two; the talon held exactly four.
    assert [len(h) for h in state.hands] == [6, 6]
    assert state.talon == ()
    assert not state.is_over
    state.validate()


def test_draw_with_too_few_cards_favours_the_attacker() -> None:
    state = make_state(
        hand0="6S 6C",
        hand1="8S 8C",
        trump=Suit.HEARTS,
        talon="10S 10H",
        attacker=0,
    )
    state = _bito_once(state, "6S", "8S")
    # Attacker drew first and took both remaining cards.
    assert len(state.hands[0]) == 3
    assert len(state.hands[1]) == 1
    assert state.talon == ()


def test_draw_from_an_empty_talon_produces_nothing() -> None:
    state = make_state(
        hand0="6S 6C",
        hand1="8S 8H",
        trump=Suit.HEARTS,
        talon="",
        attacker=0,
    )
    before = [len(h) for h in state.hands]
    state = _bito_once(state, "6S", "8S")
    assert [len(h) for h in state.hands] == [before[0] - 1, before[1] - 1]
    assert state.talon == ()


def test_the_trump_card_is_the_very_last_card_drawn() -> None:
    """Audited on real deals, not a constructed position."""
    for seed in range(50):
        state = new_game(seed=seed)
        trump_card = state.trump_card
        assert state.talon[-1] is trump_card
        rng = random.Random(seed)
        holder_seen = False
        while not state.is_over:
            if state.talon:
                # While any talon remains, the trump card is still its last card.
                assert state.talon[-1] is trump_card
            else:
                holder_seen = True
            state = apply_move(state, rng.choice(get_legal_moves(state)))
        assert holder_seen
        # It never vanished: it is in a hand, the discard, or was played.
        located = (
            any(trump_card in hand for hand in state.hands)
            or trump_card in state.discard
        )
        assert located


def test_talon_never_grows_and_transitions_once_to_empty() -> None:
    for seed in range(40):
        rng = random.Random(seed)
        state = new_game(seed=seed)
        previous = len(state.talon)
        emptied_at = None
        step = 0
        while not state.is_over:
            state = apply_move(state, rng.choice(get_legal_moves(state)))
            step += 1
            assert len(state.talon) <= previous
            if previous > 0 and len(state.talon) == 0:
                assert emptied_at is None, "talon emptied twice"
                emptied_at = step
            previous = len(state.talon)
        assert emptied_at is not None


# ======================================================================
# 12-14. Going out, and the difference between empty and finished
# ======================================================================
def test_empty_hand_with_talon_remaining_is_not_the_end() -> None:
    state = make_state(
        hand0="6S",
        hand1="8S 9H",
        trump=Suit.HEARTS,
        talon="10S 10C 10H",
        attacker=0,
    )
    state = _bito_once(state, "6S", "8S")
    assert not state.is_over
    assert len(state.hands[0]) == 3  # refilled from the talon


def test_zero_cards_mid_bout_is_an_ordinary_position() -> None:
    state = make_state(
        hand0="6S",
        hand1="8S 9H",
        trump=Suit.HEARTS,
        talon="10S 10H",
        attacker=0,
    )
    state = apply_move(state, AttackMove(Card.parse("6S")))
    assert state.hands[0] == ()
    assert not state.is_over  # the bout has not resolved yet
    assert get_legal_moves(state)


def test_single_player_out_makes_the_other_the_durak() -> None:
    state = make_state(
        hand0="6S",
        hand1="8S 9H",
        trump=Suit.HEARTS,
        talon="",
        attacker=0,
    )
    state = _bito_once(state, "6S", "8S")
    assert state.is_over
    assert state.durak == 1
    assert state.hands[0] == ()
    assert state.hands[1] != ()
    state.validate()


def test_simultaneous_double_out_is_a_draw() -> None:
    state = make_state(
        hand0="6S",
        hand1="8S",
        trump=Suit.HEARTS,
        talon="",
        attacker=0,
    )
    state = _bito_once(state, "6S", "8S")
    assert state.is_over
    assert state.is_draw
    assert state.durak is None
    assert all(hand == () for hand in state.hands)
    state.validate()


def test_a_defender_who_takes_can_never_go_out() -> None:
    state = make_state(
        hand0="6S",
        hand1="8C",
        trump=Suit.HEARTS,
        talon="",
        attacker=0,
    )
    state = apply_move(state, AttackMove(Card.parse("6S")))
    state = apply_move(state, TAKE)
    state = apply_move(state, END_ATTACK)
    assert state.is_over
    assert state.durak == 1  # took the card, so still holding
    assert len(state.hands[1]) == 2


# ======================================================================
# 16-18. Copying, aliasing, canonical order, hashing
# ======================================================================
def test_apply_move_cannot_reach_back_into_the_input_state() -> None:
    """Every container in a state is immutable, so no aliasing is possible."""
    state = new_game(seed=99)
    snapshot = state.to_dict()
    key_before = state.transposition_key()
    for move in get_legal_moves(state):
        child = apply_move(state, move)
        grandchild_moves = get_legal_moves(child)
        if grandchild_moves:
            apply_move(child, grandchild_moves[0])
    assert state.to_dict() == snapshot
    assert state.transposition_key() == key_before
    # And the containers really are immutable types, not defensive copies.
    assert isinstance(state.hands, tuple)
    assert all(isinstance(h, tuple) for h in state.hands)
    assert isinstance(state.talon, tuple)
    assert isinstance(state.discard, frozenset)
    assert isinstance(state.table, tuple)


def test_hand_order_never_affects_the_transposition_key() -> None:
    """Positions differing only in pick-up order must be indistinguishable."""
    base = make_state(
        hand0="6S 9C AH",
        hand1="7D 10S",
        trump=Suit.HEARTS,
        talon="8C 8H",
        attacker=0,
    )
    shuffled_hand = tuple(reversed(base.hands[0]))
    # Same cards, reversed order — built directly to bypass add_cards().
    twin = GameState(
        hands=(shuffled_hand, base.hands[1]),
        talon=base.talon,
        discard=base.discard,
        table=base.table,
        trump_card=base.trump_card,
        trump_suit=base.trump_suit,
        attacker=base.attacker,
        phase=base.phase,
        attack_limit=base.attack_limit,
        rules=base.rules,
    )
    assert set(twin.hands[0]) == set(base.hands[0])
    # The engine rejects the non-canonical form outright...
    with pytest.raises(InvalidStateError, match="canonical order"):
        twin.validate()
    # ...and canonicalising it reproduces the original key exactly.
    from durakfish.game.state import add_cards

    fixed = twin.replace(hands=(add_cards((), shuffled_hand), base.hands[1]))
    fixed.validate()
    assert fixed.transposition_key() == base.transposition_key()
    assert hash(fixed.transposition_key()) == hash(base.transposition_key())


def test_transposition_key_is_hashable_and_ignores_history() -> None:
    a = new_game(seed=5)
    b = a.replace(history=None)
    assert a.transposition_key() == b.transposition_key()
    assert len({a.transposition_key(), b.transposition_key()}) == 1


# ======================================================================
# 22-26. Legality: the two paths must partition every move identically
# ======================================================================
class BogusMove(Move):
    """A move type the engine has never heard of."""

    __slots__ = ()


def move_universe(state: GameState) -> list[Move]:
    """Every move shape worth testing, including impossible ones."""
    moves: list[Move] = [TAKE, END_ATTACK, TakeCards(), EndAttack()]
    moves += [AttackMove(card) for card in DECK]
    table_attacks = [slot.attacking_card for slot in state.table]
    # Real attacking cards, plus cards that are NOT on the table at all.
    candidates = table_attacks + [Card.parse("6S"), Card.parse("AH")]
    for attacking in candidates:
        for defending in DECK:
            moves.append(DefenseMove(attacking, defending))
    return moves


def sampled_positions(count: int = 60) -> list[GameState]:
    positions: list[GameState] = []
    rng = random.Random(4242)
    seed = 0
    while len(positions) < count:
        state = new_game(seed=seed)
        seed += 1
        while not state.is_over and len(positions) < count:
            positions.append(state)
            state = apply_move(state, rng.choice(get_legal_moves(state)))
    return positions


def test_generation_and_prediction_agree_over_a_synthetic_move_universe() -> None:
    for state in sampled_positions():
        generated = set(get_legal_moves(state))
        predicted = {m for m in move_universe(state) if is_legal_move(state, m)}
        assert generated == predicted, state.describe()


def test_every_illegal_move_is_rejected_by_apply_move() -> None:
    for state in sampled_positions(25):
        legal = set(get_legal_moves(state))
        for move in move_universe(state):
            if move in legal:
                child = apply_move(state, move)
                child.validate()
            else:
                with pytest.raises(IllegalMoveError):
                    apply_move(state, move)


def test_unknown_move_types_are_rejected_not_ignored() -> None:
    state = new_game(seed=1)
    assert not is_legal_move(state, BogusMove())
    with pytest.raises(IllegalMoveError):
        apply_move(state, BogusMove())


def test_the_idle_player_has_no_moves_anywhere() -> None:
    for state in sampled_positions(40):
        idle = 1 - state.current_player
        assert get_legal_moves(state, idle) == ()
        for move in get_legal_moves(state):
            assert not is_legal_move(state, move, idle)


def test_live_positions_always_offer_a_move_and_finished_ones_never_do() -> None:
    for seed in range(60):
        rng = random.Random(seed)
        state = new_game(seed=seed)
        while not state.is_over:
            assert get_legal_moves(state), state.describe()
            state = apply_move(state, rng.choice(get_legal_moves(state)))
        assert get_legal_moves(state) == ()
        assert get_legal_moves(state, 0) == ()
        assert get_legal_moves(state, 1) == ()
        assert state.current_player is None


# ======================================================================
# 27. Phase-transition boundaries
# ======================================================================
def test_every_observed_phase_transition_is_one_the_rules_permit() -> None:
    allowed = {
        (Phase.ATTACK, Phase.DEFENSE),      # a card was played
        (Phase.ATTACK, Phase.ATTACK),       # bout resolved, next bout opened
        (Phase.ATTACK, Phase.GAME_OVER),    # bout resolved, someone went out
        (Phase.DEFENSE, Phase.ATTACK),      # the card was beaten
        (Phase.DEFENSE, Phase.TAKING),      # the defender gave up
        (Phase.TAKING, Phase.TAKING),       # a throw-in
        (Phase.TAKING, Phase.ATTACK),       # take resolved
        (Phase.TAKING, Phase.GAME_OVER),    # take resolved, someone went out
    }
    seen: set[tuple[Phase, Phase]] = set()
    for seed in range(200):
        rng = random.Random(seed)
        state = new_game(seed=seed)
        while not state.is_over:
            nxt = apply_move(state, rng.choice(get_legal_moves(state)))
            seen.add((state.phase, nxt.phase))
            state = nxt
    assert seen <= allowed, f"unexpected transitions: {seen - allowed}"
    # And every permitted transition is actually reachable, so the set is
    # not passing merely by being over-generous.
    assert seen == allowed


# ======================================================================
# 29. The progress argument, checked adversarially
# ======================================================================
def progress_measure(state: GameState) -> tuple[int, int, int]:
    """Strictly decreases (lexicographically) at every bout resolution.

    1. cards not yet retired to the discard;
    2. talon size;
    3. the attacker's hand size.

    A complete defence retires at least two cards, so (1) falls. A take
    retires nothing, but the attacker played a card and either refilled
    from the talon — lowering (2) — or did not, lowering (3).
    """
    live = len(state.talon) + sum(len(h) for h in state.hands)
    return (live, len(state.talon), len(state.hands[state.attacker]))


def test_every_bout_strictly_reduces_the_progress_measure() -> None:
    for seed in range(150):
        rng = random.Random(seed)
        state = new_game(seed=seed)
        measure = progress_measure(state)
        while not state.is_over:
            move = rng.choice(get_legal_moves(state))
            resolving = isinstance(move, EndAttack)
            state = apply_move(state, move)
            if resolving and not state.is_over:
                nxt = progress_measure(state)
                assert nxt < measure, f"bout did not make progress: {measure} -> {nxt}"
                measure = nxt


# ======================================================================
# 30. Rules that must NOT be present
# ======================================================================
def test_the_defender_can_never_pass_the_attack_along() -> None:
    """Perevodnoy: beating is not possible by matching rank, only by rank order."""
    state = make_state(
        hand0="7S 9C",
        hand1="7H 7D 9H",
        trump=Suit.HEARTS,
        talon="8C 8H",
        attacker=0,
    )
    state = apply_move(state, AttackMove(Card.parse("7S")))
    moves = get_legal_moves(state)
    # 7D matches the rank but does not beat 7S: it must not appear at all.
    assert not any(
        isinstance(m, DefenseMove) and m.defending_card == Card.parse("7D")
        for m in moves
    )
    # No move type exists that would transfer the attack.
    assert all(isinstance(m, (DefenseMove, TakeCards)) for m in moves)
    assert not is_legal_move(state, AttackMove(Card.parse("7D")))


def test_no_trump_six_exchange_move_exists() -> None:
    """The six of trumps cannot be swapped for the face-up trump card."""
    state = make_state(
        hand0="6H 9C",
        hand1="7S 9H",
        trump=Suit.HEARTS,
        talon="8C AH",
        attacker=0,
    )
    assert state.trump_card == Card.parse("AH")
    moves = get_legal_moves(state)
    assert all(isinstance(m, AttackMove) for m in moves)
    # Playing 6H is an ordinary attack, and the talon is untouched by it.
    after = apply_move(state, AttackMove(Card.parse("6H")))
    assert after.talon == state.talon
    assert after.trump_card == Card.parse("AH")


def test_the_first_bout_is_not_limited_to_five_cards() -> None:
    for seed in range(30):
        state = new_game(seed=seed)
        assert state.attack_limit == 6


# ======================================================================
# 31. Consistency with the Phase 1 card/deck contracts
# ======================================================================
def test_states_only_ever_contain_cards_from_the_36_card_deck() -> None:
    for state in sampled_positions(80):
        everywhere = (
            set(state.hands[0])
            | set(state.hands[1])
            | set(state.talon)
            | set(state.table_cards)
            | set(state.discard)
        )
        assert everywhere == UNIVERSE
        assert all(c.rank >= Rank.SIX for c in everywhere)


def test_hands_stay_canonically_ordered_by_card_code_throughout() -> None:
    for seed in range(40):
        rng = random.Random(seed)
        state = new_game(seed=seed)
        while not state.is_over:
            for hand in state.hands:
                codes = [c.code for c in hand]
                assert codes == sorted(codes)
                assert len(set(codes)) == len(codes)
            state = apply_move(state, rng.choice(get_legal_moves(state)))
