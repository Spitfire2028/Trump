"""The information boundary under adversarial examination.

Three claims are worth real evidence here:

1. **Nothing leaks.** An information set contains no hidden card.
2. **Nothing distinguishes.** Two states differing only in cards hidden
   from P produce byte-identical observations for P. This is the
   precondition for sound determinization in Phase 11, so it is proven
   now rather than assumed then.
3. **Nothing is missing.** The view plus the legal moves is enough to
   play correctly — demonstrated by re-deriving legality from the view
   with independent logic, and by agents that decide correctly in
   constructed positions while touching nothing but the view.
"""

from __future__ import annotations

import random

import pytest

from durakfish.cards import Card, Suit
from durakfish.cards.deck import standard_cards
from durakfish.game import (
    AttackMove,
    DefenseMove,
    Phase,
    apply_move,
    beats,
    get_legal_moves,
    new_game,
)
from durakfish.game.history import EventKind
from durakfish.game.moves import END_ATTACK, TAKE, EndAttack, Move, TakeCards
from durakfish.game.state import add_cards
from durakfish.information import InformationSet, observe
from tests.support import make_state

UNIVERSE = frozenset(standard_cards(36))


def walk(seed: int, limit: int | None = None):
    """Yield every position of a random game."""
    rng = random.Random(seed)
    state = new_game(seed=seed)
    steps = 0
    while not state.is_over:
        yield state
        state = apply_move(state, rng.choice(get_legal_moves(state)))
        steps += 1
        if limit is not None and steps >= limit:
            return
    yield state


def positions(games: int = 25) -> list:
    return [s for seed in range(games) for s in walk(seed)]


# ======================================================================
# 1. Leakage
# ======================================================================
def test_no_hidden_card_ever_appears_in_a_view() -> None:
    """Intersect everything the view exposes with everything P never saw.

    "Hidden" here means *never observed*, not merely "currently in a
    hidden location": a card that sat face up on the table and was then
    picked up by the opponent is in their hand yet was legitimately seen.
    Treating that as a leak would be demanding amnesia, not privacy.
    """
    for state in positions():
        for player in range(2):
            view = observe(state, player)
            hidden = set(state.hands[1 - player]) | set(state.talon)
            hidden &= view.unobserved_cards()

            visible: set[Card] = set(view.hand) | set(view.table_cards)
            visible |= set(view.discard)
            for event in view.public_history:
                visible |= set(event.revealed) | set(event.discarded)
                visible |= set(event.taken)
                for draw in event.draws:
                    if draw.cards is not None:
                        visible |= set(draw.cards)

            assert not (visible & hidden), (
                f"leaked {visible & hidden} to P{player}"
            )


def test_the_view_exposes_no_attribute_reaching_ground_truth() -> None:
    view = observe(new_game(seed=3), 0)
    for forbidden in ("hands", "talon", "history", "discard_order"):
        assert not hasattr(view, forbidden)
    # The talon is a number, not a container.
    assert isinstance(view.talon_size, int)
    with pytest.raises(TypeError):
        iter(view.talon_size)  # type: ignore[call-overload]


def test_opponent_draws_are_counted_but_never_named() -> None:
    for state in positions(10):
        for player in range(2):
            view = observe(state, player)
            for event in view.public_history:
                for draw in event.draws:
                    if draw.player == player:
                        assert draw.cards is not None
                        assert len(draw.cards) == draw.count
                    else:
                        assert draw.cards is None
                        assert draw.count >= 0


def test_the_deal_reveals_the_trump_card_and_own_cards_only() -> None:
    state = new_game(seed=11)
    for player in range(2):
        view = observe(state, player)
        deal = view.public_history[0]
        assert deal.kind is EventKind.DEAL
        assert deal.revealed == (state.trump_card,)
        own = next(d for d in deal.draws if d.player == player)
        other = next(d for d in deal.draws if d.player != player)
        assert own.cards is not None and set(own.cards) == set(state.hands[player])
        assert other.cards is None and other.count == 6


# ======================================================================
# 2. Indistinguishability — the Phase 11 precondition
# ======================================================================
def permute_hidden(state, player: int, rng: random.Random):
    """Redeal the never-observed cards, preserving every public fact.

    Only cards P has never seen may move. Cards P watched the opponent
    take cannot be shuffled back into the talon — that would contradict
    the observation record, and such a position is unreachable. Opponent
    hand size, talon size and the face-up trump card all stay put; only
    *which* unseen card sits where changes.

    This is exactly the redistribution a determinizing sampler will
    perform in Phase 11, which is why the property is proven here.
    """
    view = observe(state, player)
    movable = view.unobserved_cards()

    opponent_fixed = [c for c in state.hands[1 - player] if c not in movable]
    talon_fixed = [c for c in state.talon[:-1] if c not in movable]
    pool = [c for c in state.hands[1 - player] if c in movable]
    pool += [c for c in state.talon[:-1] if c in movable]
    rng.shuffle(pool)

    opponent_slots = len(state.hands[1 - player]) - len(opponent_fixed)
    new_opponent = add_cards((), opponent_fixed + pool[:opponent_slots])
    new_talon = tuple(talon_fixed + pool[opponent_slots:]) + state.talon[-1:]

    hands = list(state.hands)
    hands[1 - player] = new_opponent
    return state.replace(hands=tuple(hands), talon=new_talon)


def test_permuting_hidden_cards_leaves_the_view_identical() -> None:
    rng = random.Random(2024)
    checked = 0
    for state in positions(20):
        if not state.talon:
            continue
        for player in range(2):
            baseline = observe(state, player)
            for _ in range(3):
                twin = permute_hidden(state, player, rng)
                twin.validate()
                assert observe(twin, player) == baseline
                assert observe(twin, player).key() == baseline.key()
                assert (
                    observe(twin, player).to_dict() == baseline.to_dict()
                ), "views serialised differently"
                checked += 1
    assert checked > 200


def test_reordering_the_talon_is_invisible() -> None:
    """Talon *order* is hidden, so shuffling it must not change any view."""
    rng = random.Random(5)
    for state in positions(15):
        if len(state.talon) < 3:
            continue
        body = list(state.talon[:-1])
        rng.shuffle(body)
        twin = state.replace(talon=tuple(body) + state.talon[-1:])
        twin.validate()
        for player in range(2):
            assert observe(twin, player) == observe(state, player)


def test_a_visible_difference_does_change_the_view() -> None:
    """The converse: indistinguishability must not be vacuous."""
    state = new_game(seed=8)
    baseline = observe(state, 0)
    # Swap two cards between the observer's own hand and the talon.
    own = state.hands[0]
    hands = list(state.hands)
    hands[0] = add_cards(own[1:], (state.talon[0],))
    twin = state.replace(hands=tuple(hands), talon=(own[0],) + state.talon[1:])
    twin.validate()
    assert observe(twin, 0) != baseline


# ======================================================================
# 3a. Legal-move sufficiency, re-derived independently from the view
# ======================================================================
def legal_moves_from_view(view: InformationSet) -> tuple[Move, ...]:
    """Independent re-derivation of legality using only the view.

    Written from ``docs/rules.md`` rather than by calling the rules
    engine, so agreement between this and ``get_legal_moves`` is evidence
    that the view carries enough information, not a tautology.

    This lives in the test suite on purpose. A second legality
    implementation in production would become a second source of truth
    and would eventually drift from the engine.
    """
    if view.phase is Phase.GAME_OVER or not view.is_to_move:
        return ()

    if view.phase is Phase.DEFENSE:
        open_slot = view.table[-1]
        moves: list[Move] = [
            DefenseMove(open_slot.attacking_card, card)
            for card in view.hand
            if beats(card, open_slot.attacking_card, view.trump_suit)
        ]
        moves.append(TAKE)
        return tuple(moves)

    # ATTACK or TAKING: add a card, or end the bout.
    room = view.attack_limit - len(view.table)
    if not view.table:
        playable = view.hand if room > 0 else ()
    elif room > 0:
        on_table = view.table_ranks
        playable = tuple(c for c in view.hand if c.rank in on_table)
    else:
        playable = ()
    out: list[Move] = [AttackMove(c) for c in playable]
    if view.table:
        out.append(END_ATTACK)
    return tuple(out)


def test_legality_derived_from_the_view_matches_the_engine() -> None:
    checked = 0
    for state in positions(30):
        player = state.current_player
        if player is None:
            continue
        view = observe(state, player)
        assert legal_moves_from_view(view) == get_legal_moves(state, player)
        checked += 1
    assert checked > 2000


def test_the_idle_players_view_yields_no_moves_either() -> None:
    for state in positions(10):
        if state.current_player is None:
            continue
        idle = 1 - state.current_player
        assert legal_moves_from_view(observe(state, idle)) == ()
        assert get_legal_moves(state, idle) == ()


def test_legality_from_the_view_holds_in_constructed_corner_positions() -> None:
    """Taking, throw-ins, attack limits and trump all decided from the view."""
    scenarios = [
        # Defender with nothing that beats the open trump attack.
        make_state(
            hand0="AH 6C", hand1="6S 7S", trump=Suit.HEARTS, talon="8C 8H",
            table=(("9H", None),), attacker=0, phase=Phase.DEFENSE,
            attack_limit=2,
        ),
        # Attacker mid-bout with the limit already reached.
        make_state(
            hand0="6C 6D", hand1="7C", trump=Suit.HEARTS, talon="8C 8H",
            table=(("6S", "7S"),), attacker=0, attack_limit=1,
        ),
        # Throwing in after a take.
        make_state(
            hand0="6C 6D 9S", hand1="7C 7D", trump=Suit.HEARTS, talon="8C 8H",
            table=(("6S", None),), attacker=0, phase=Phase.TAKING,
            attack_limit=2,
        ),
        # Empty talon endgame.
        make_state(
            hand0="6S 7H", hand1="8S 9H", trump=Suit.HEARTS, talon="",
            attacker=0,
        ),
    ]
    for state in scenarios:
        player = state.current_player
        assert player is not None
        view = observe(state, player)
        assert legal_moves_from_view(view) == get_legal_moves(state, player), (
            view.describe()
        )


# ======================================================================
# 3b. Sufficiency in practice: agents deciding correctly from the view
# ======================================================================
def test_a_defender_can_tell_from_the_view_alone_that_it_must_take() -> None:
    state = make_state(
        hand0="AH 6C",
        hand1="6S 7S",
        trump=Suit.HEARTS,
        talon="8C 8H",
        table=(("9H", None),),
        attacker=0,
        phase=Phase.DEFENSE,
        attack_limit=2,
    )
    view = observe(state, 1)
    open_card = view.table[-1].attacking_card
    # Decided purely from the view: no card in hand beats the trump nine.
    assert not any(beats(c, open_card, view.trump_suit) for c in view.hand)
    assert get_legal_moves(state) == (TAKE,)


def test_an_attacker_can_tell_from_the_view_which_ranks_may_be_thrown_in() -> None:
    state = make_state(
        hand0="6C 6D 9S",
        hand1="7C 7D",
        trump=Suit.HEARTS,
        talon="8C 8H",
        table=(("6S", None),),
        attacker=0,
        phase=Phase.TAKING,
        attack_limit=2,
    )
    view = observe(state, 0)
    addable = {c for c in view.hand if c.rank in view.table_ranks}
    assert addable == {Card.parse("6C"), Card.parse("6D")}
    assert view.additions_remaining == 1
    engine_attacks = {
        m.card for m in get_legal_moves(state) if isinstance(m, AttackMove)
    }
    assert engine_attacks == addable


def test_the_attack_limit_and_bout_progress_are_visible() -> None:
    for state in positions(15):
        for player in range(2):
            view = observe(state, player)
            assert view.attack_limit == state.attack_limit
            assert view.attacks_made == len(state.table)
            assert view.additions_remaining == max(
                0, state.attack_limit - len(state.table)
            )


def test_an_agent_playing_only_from_the_view_completes_games() -> None:
    """A policy that reads nothing but the view must be able to finish."""
    from durakfish.ai import CallableAgent
    from durakfish.simulation import play_game

    def cautious(view: InformationSet, legal, rng):
        # Uses trump, own hand, table state and hand sizes — view only.
        non_trump = [
            m
            for m in legal
            if isinstance(m, AttackMove) and m.card.suit is not view.trump_suit
        ]
        if non_trump:
            return min(non_trump, key=lambda m: int(m.card.rank))
        defences = [m for m in legal if isinstance(m, DefenseMove)]
        if defences:
            return min(defences, key=lambda m: (m.defending_card.suit is view.trump_suit,
                                                int(m.defending_card.rank)))
        for move in legal:
            if isinstance(move, EndAttack):
                return move
        return legal[0]

    for seed in range(20):
        record = play_game(
            [CallableAgent(cautious, "view-only-a"), CallableAgent(cautious, "view-only-b")],
            seed=seed,
            validate_states=True,
        )
        assert record.length > 0


# ======================================================================
# Observation accounting (no deduction)
# ======================================================================
def test_observed_and_unobserved_partition_the_deck() -> None:
    for state in positions(20):
        for player in range(2):
            view = observe(state, player)
            observed = view.observed_cards()
            unobserved = view.unobserved_cards()
            assert observed | unobserved == UNIVERSE
            assert not (observed & unobserved)
            assert len(observed) + len(unobserved) == 36


def test_unobserved_cards_are_genuinely_unseen() -> None:
    """Soundness: nothing called unobserved is actually visible to P."""
    for state in positions(20):
        for player in range(2):
            view = observe(state, player)
            unobserved = view.unobserved_cards()
            assert not (unobserved & set(state.hands[player]))
            assert not (unobserved & set(state.table_cards))
            assert not (unobserved & state.discard)
            assert state.trump_card not in unobserved
            # Everything unobserved really is somewhere hidden.
            hidden = set(state.hands[1 - player]) | set(state.talon)
            assert unobserved <= hidden


def test_unobserved_cards_is_not_the_opponents_possible_hand() -> None:
    """The Phase 3/6 boundary, asserted directly.

    A card the opponent took was *seen*, so it is observed — but Phase 3
    still makes no claim that they hold it. The opponent's holdings are
    therefore strictly larger than the unobserved set, and anyone who
    mistook one for the other would get a wrong answer. Closing that gap
    needs observation plus counting, which is Phase 6.
    """
    state = make_state(
        hand0="6S 6C",
        hand1="9H 10H",
        trump=Suit.HEARTS,
        talon="8C 8D 8S 8H",
        attacker=0,
    )
    state = apply_move(state, AttackMove(Card.parse("6S")))
    state = apply_move(state, TAKE)
    state = apply_move(state, END_ATTACK)

    six_of_spades = Card.parse("6S")
    assert six_of_spades in state.hands[1], "the defender picked it up"
    view = observe(state, 0)

    # Seen on the table, so observed — remembering is not deducing.
    assert six_of_spades in view.observed_cards()
    assert six_of_spades not in view.unobserved_cards()
    # The observation is on record for Phase 6 to reason from.
    assert any(six_of_spades in event.taken for event in view.public_history)

    # Yet the unobserved set is NOT the opponent's possible hand: they
    # demonstrably hold a card that is not in it.
    opponent_holdings = set(state.hands[1])
    assert opponent_holdings - view.unobserved_cards(), (
        "unobserved_cards() must not be mistaken for the opponent's holdings"
    )
    # And Phase 3 offers no API that resolves where that card actually is.
    assert not hasattr(view, "opponent_possible_cards")


def test_is_open_information_reports_a_public_fact() -> None:
    for state in positions(20):
        for player in range(2):
            view = observe(state, player)
            assert view.is_open_information() == (view.talon_size == 0)
            assert view.is_open_information() == state.is_open_information()


def test_hidden_card_count_matches_reality() -> None:
    for state in positions(15):
        for player in range(2):
            view = observe(state, player)
            assert view.hidden_card_count == len(state.talon) + len(
                state.hands[1 - player]
            )
            # Physically out of sight is not the same as never seen: the
            # opponent may hold cards P watched them take.
            assert len(view.unobserved_cards()) <= view.hidden_card_count


# ======================================================================
# Purity, identity, serialization
# ======================================================================
def test_observe_is_pure_and_leaves_the_state_untouched() -> None:
    state = new_game(seed=17)
    before = state.to_dict()
    first = observe(state, 0)
    second = observe(state, 0)
    assert first == second
    assert first.key() == second.key()
    assert state.to_dict() == before


def test_views_of_the_two_seats_differ() -> None:
    state = new_game(seed=21)
    assert observe(state, 0) != observe(state, 1)
    assert observe(state, 0).key() != observe(state, 1).key()


def test_history_can_be_omitted_without_changing_anything_else() -> None:
    state = list(walk(4, limit=30))[-1]
    full = observe(state, 0)
    lean = observe(state, 0, include_history=False)
    assert lean.public_history == ()
    assert lean.key() == full.key()
    assert lean.to_dict(include_history=False) == full.to_dict(
        include_history=False
    )


def test_view_serialization_round_trips_as_json() -> None:
    import json

    for state in positions(5):
        for player in range(2):
            payload = json.dumps(observe(state, player).to_dict())
            assert json.loads(payload)["player"] == player


def test_observe_rejects_a_seat_that_does_not_exist() -> None:
    state = new_game(seed=1)
    for bad in (-1, 2, 99):
        with pytest.raises(IndexError):
            observe(state, bad)


def test_view_keys_are_hashable_and_usable_as_dict_keys() -> None:
    seen: dict[tuple, int] = {}
    for state in positions(10):
        for player in range(2):
            seen[observe(state, player).key()] = player
    assert len(seen) > 100


def test_take_and_end_attack_moves_are_public_in_the_history() -> None:
    state = make_state(
        hand0="6S 6C", hand1="7C 9H", trump=Suit.HEARTS, talon="8C 8H", attacker=0
    )
    state = apply_move(state, AttackMove(Card.parse("6S")))
    state = apply_move(state, TAKE)
    view = observe(state, 0)
    kinds = [type(e.move) for e in view.public_history if e.move is not None]
    assert TakeCards in kinds, "the defender's take must be visible to the attacker"
