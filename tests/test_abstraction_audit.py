"""Audit of the conservative search abstraction.

Phase 5 could not build the real game tree — opponent cards and talon
order are hidden, and determinization is out of scope — so it searches an
abstraction instead. That is a departure from the original specification
and is audited here explicitly rather than assumed sound.

The three claims under test:

1. **Soundness.** Every move that is really legal for the opponent falls
   into an action class the abstraction offers. It never omits.
2. **Strictness.** The abstraction offers classes that are sometimes
   impossible. It is a strict superset, not an exact model, and that is
   measured rather than asserted.
3. **Direction.** Because the opponent gets extra options and the
   searcher gets fewer, an abstract value is a *lower bound* on the true
   value. So a proven win is real; a "proven loss" is not. Both halves
   are checked against a perfect-information oracle that reads the real
   hidden state — something only a test may do.
"""

from __future__ import annotations

import random
import sys

import pytest

from durakfish.ai import SearchBot, SearchConfig, alpha_beta, minimax
from durakfish.ai.evaluation import MATE_MARGIN, BaselineEvaluator
from durakfish.ai.ordering import order_opponent_actions, order_self_moves
from durakfish.ai.searchnode import (
    Actor,
    NodeKind,
    OpponentAction,
    Outcome,
    SearchNode,
    SearchSlot,
)
from durakfish.cards import Card, Suit
from durakfish.game import Phase, apply_move, get_legal_moves, new_game
from durakfish.game.moves import (
    END_ATTACK,
    TAKE,
    AttackMove,
    DefenseMove,
    EndAttack,
    Move,
    TakeCards,
)
from durakfish.game.state import add_cards
from durakfish.information import observe
from tests.support import make_state

DEEP = SearchConfig(depth=8)


def action_class_of(move: Move) -> OpponentAction:
    """Which abstract class a real legal move belongs to."""
    if isinstance(move, DefenseMove):
        return OpponentAction.BEAT
    if isinstance(move, TakeCards):
        return OpponentAction.TAKE
    if isinstance(move, AttackMove):
        return OpponentAction.ADD
    if isinstance(move, EndAttack):
        return OpponentAction.END
    raise TypeError(move)  # pragma: no cover


def positions(games: int = 20):
    for seed in range(games):
        rng = random.Random(seed)
        state = new_game(seed=seed)
        while not state.is_over:
            yield state
            state = apply_move(state, rng.choice(get_legal_moves(state)))


# ======================================================================
# 1. Soundness: the abstraction never omits a real option
# ======================================================================
def test_every_real_opponent_move_falls_into_an_offered_class() -> None:
    """Uses ground truth, which only a test may do.

    If the abstraction ever omitted a class, the search would believe the
    opponent had no such reply and could claim wins that are not there.
    """
    omitted = checked = 0
    for state in positions(40):
        mover = state.current_player
        assert mover is not None
        observer = 1 - mover
        node = SearchNode.from_view(observe(state, observer))
        if node.to_move is not Actor.OPPONENT:
            continue
        offered = set(node.opponent_actions())
        real = {action_class_of(m) for m in get_legal_moves(state)}
        omitted += len(real - offered)
        checked += 1
    assert checked > 3000
    assert omitted == 0, f"{omitted} real opponent options were invisible to search"


def test_the_abstraction_is_a_strict_superset_and_that_is_measured() -> None:
    """It offers replies the opponent may not actually be able to make."""
    spurious = checked = 0
    for state in positions(40):
        mover = state.current_player
        assert mover is not None
        node = SearchNode.from_view(observe(state, 1 - mover))
        if node.to_move is not Actor.OPPONENT:
            continue
        offered = set(node.opponent_actions())
        real = {action_class_of(m) for m in get_legal_moves(state)}
        spurious += len(offered - real)
        checked += 1
    assert spurious > 0, (
        "no impossible class was ever offered; the abstraction would then be "
        "exact, which contradicts its documented semantics"
    )


def test_taking_and_ending_are_always_genuinely_available() -> None:
    """The two classes that are never spurious, unlike BEAT and ADD."""
    for state in positions(15):
        mover = state.current_player
        assert mover is not None
        node = SearchNode.from_view(observe(state, 1 - mover))
        if node.to_move is not Actor.OPPONENT:
            continue
        real = {action_class_of(m) for m in get_legal_moves(state)}
        offered = set(node.opponent_actions())
        if OpponentAction.TAKE in offered:
            assert OpponentAction.TAKE in real
        if OpponentAction.END in offered:
            assert OpponentAction.END in real


# ======================================================================
# 2. The critical adversarial test
# ======================================================================
def paired_states(defender_can_beat_card: str, defender_cannot: str):
    """Two states with identical public information and different hidden hands.

    The cards are swapped between the opponent's hand and the talon, so
    the discard pile — which is public — is untouched and P0's view is
    byte-identical in both.
    """
    base = make_state(
        hand0="9S 6C",
        hand1=f"{defender_can_beat_card}",
        trump=Suit.HEARTS,
        talon=f"{defender_cannot} 8H",
        attacker=0,
    )
    can_beat = base
    swapped_hand = add_cards((), [Card.parse(defender_cannot)])
    swapped_talon = (Card.parse(defender_can_beat_card), base.talon[-1])
    cannot_beat = base.replace(
        hands=(base.hands[0], swapped_hand), talon=swapped_talon
    )
    can_beat.validate()
    cannot_beat.validate()
    return can_beat, cannot_beat


def test_the_search_cannot_tell_a_beatable_defender_from_a_helpless_one() -> None:
    """The heart of the audit.

    In one state the defender holds a card that beats the attack; in the
    other it does not. Public information is identical. Every artefact of
    the search must therefore be identical too.
    """
    can, cannot = paired_states("10S", "6D")

    # The premise: really different hidden hands, really identical views.
    assert can.hands[1] != cannot.hands[1]
    assert observe(can, 0) == observe(cannot, 0)

    after_can = apply_move(can, AttackMove(Card.parse("9S")))
    after_cannot = apply_move(cannot, AttackMove(Card.parse("9S")))
    # Ground truth: one defender can beat, the other cannot.
    assert any(isinstance(m, DefenseMove) for m in get_legal_moves(after_can))
    assert get_legal_moves(after_cannot) == (TAKE,)

    # Yet the abstraction is identical.
    node_can = SearchNode.from_view(observe(after_can, 0))
    node_cannot = SearchNode.from_view(observe(after_cannot, 0))
    assert node_can == node_cannot
    assert node_can.key() == node_cannot.key()
    assert node_can.opponent_actions() == node_cannot.opponent_actions()
    assert set(node_can.opponent_actions()) == {
        OpponentAction.BEAT,
        OpponentAction.TAKE,
    }, "both classes must be kept: assuming either would be inference"


def test_the_whole_search_is_identical_for_the_adversarial_pair() -> None:
    can, cannot = paired_states("10S", "6D")
    results = []
    for state in (can, cannot):
        view, legal = observe(state, 0), get_legal_moves(state)
        node = SearchNode.from_view(view)
        evaluator = BaselineEvaluator()
        bot = SearchBot(SearchConfig(depth=4), "s")
        bot.reset(random.Random(0))
        move = bot.choose(view, legal)
        assert bot.last_result is not None
        results.append(
            (
                node.key(),
                node.self_moves(),
                node.opponent_actions(),
                order_self_moves(node, legal),
                order_opponent_actions(node, node.opponent_actions()),
                evaluator.evaluate(node, 0),
                minimax(node, SearchConfig(depth=4, algorithm="minimax"),
                        root_moves=legal).score,
                alpha_beta(node, SearchConfig(depth=4), root_moves=legal).score,
                move,
                bot.last_result.stats.nodes,
                bot.last_result.stats.cutoffs,
                bot.last_result.stats.leaves,
                bot.last_result.stats.terminals,
                bot.last_result.stats.max_depth,
            )
        )
    assert results[0] == results[1]


# ======================================================================
# 3. Direction: the lower-bound theorem, against a real oracle
# ======================================================================
def perfect_information_value(state, me: int, memo: dict) -> int:
    """Ground truth: +1 the searcher wins, -1 loses, 0 draws.

    A full-width search of the *real* game with both hands visible. Only a
    test may do this; it exists to check what the abstraction claims.
    """
    if state.is_over:
        if state.durak is None:
            return 0
        return 1 if state.durak != me else -1
    key = (state.transposition_key(), me)
    cached = memo.get(key)
    if cached is not None:
        return cached
    memo[key] = 0  # cycle guard; the rules forbid true cycles
    values = [
        perfect_information_value(apply_move(state, m), me, memo)
        for m in get_legal_moves(state)
    ]
    value = max(values) if state.current_player == me else min(values)
    memo[key] = value
    return value


def small_endgames(games: int = 300, budget: int = 6):
    """Empty-talon positions small enough for the oracle to solve exactly."""
    for seed in range(games):
        rng = random.Random(seed)
        state = new_game(seed=seed)
        while not state.is_over:
            if not state.talon and (
                len(state.hands[0]) + len(state.hands[1]) <= budget
            ):
                yield state
            state = apply_move(state, rng.choice(get_legal_moves(state)))


def test_every_proven_win_is_a_real_forced_win() -> None:
    """Soundness in the direction that matters. No false claims allowed."""
    sys.setrecursionlimit(20000)
    claims = confirmed = 0
    for state in small_endgames(250):
        player = state.current_player
        assert player is not None
        legal = get_legal_moves(state)
        result = alpha_beta(
            SearchNode.from_view(observe(state, player)), DEEP, root_moves=legal
        )
        if result.is_proven_win:
            claims += 1
            confirmed += perfect_information_value(state, player, {}) == 1
    assert claims > 100, "not enough proven wins to make this meaningful"
    assert confirmed == claims, f"{claims - confirmed} false wins claimed"


def test_a_lower_bound_loss_is_explicitly_not_a_proven_loss() -> None:
    """The documented unsoundness, demonstrated rather than glossed over.

    The abstraction credits the opponent with replies they may not hold
    the cards for, so it sees defeats that are not there. Finding
    counterexamples is the point of this test: if none existed, the naming
    in ``SearchResult`` would be needlessly alarmist.
    """
    sys.setrecursionlimit(20000)
    claims = actually_lost = 0
    for state in small_endgames(250):
        player = state.current_player
        assert player is not None
        legal = get_legal_moves(state)
        result = alpha_beta(
            SearchNode.from_view(observe(state, player)), DEEP, root_moves=legal
        )
        if result.is_lower_bound_loss:
            claims += 1
            actually_lost += perfect_information_value(state, player, {}) == -1
    assert claims > 50
    assert actually_lost < claims, (
        "every lower-bound loss turned out to be a real loss; the API's "
        "warning would then be misleading in the other direction"
    )


# ======================================================================
# 4. Terminal derivability
# ======================================================================
def test_terminal_detection_never_consults_a_card_identity() -> None:
    """Outcomes come from counts, which are public plus own hand."""
    node = SearchNode(
        hand=(),
        table=(),
        known_ranks=frozenset(),
        opponent_cards=2,
        talon_size=0,
        trump=Suit.HEARTS,
        refill_to=6,
        attack_limit=6,
        searcher_is_attacker=True,
        phase=Phase.ATTACK,
        to_move=Actor.SELF,
        kind=NodeKind.BOUT_RESOLVED,
    )
    from dataclasses import replace

    assert node.outcome() is Outcome.WIN
    assert replace(node, opponent_cards=0).outcome() is Outcome.DRAW
    assert replace(node, hand=(Card.parse("6S"),), opponent_cards=0).outcome() is (
        Outcome.LOSS
    )
    assert replace(node, hand=(Card.parse("6S"),)).outcome() is None
    # Unknown cards count as cards: holding two unidentified cards is not
    # being out, and no identity is guessed to establish that.
    assert replace(node, unknown_own=2).outcome() is None


def test_no_outcome_is_ever_claimed_while_the_talon_holds_cards() -> None:
    """Draws are hidden, so nothing past a refill can be proven."""
    from dataclasses import replace

    node = SearchNode(
        hand=(), table=(), known_ranks=frozenset(), opponent_cards=2,
        talon_size=0, trump=Suit.HEARTS, refill_to=6, attack_limit=6,
        searcher_is_attacker=True, phase=Phase.ATTACK, to_move=Actor.SELF,
        kind=NodeKind.BOUT_RESOLVED,
    )
    for size in range(1, 25):
        assert replace(node, talon_size=size).outcome() is None


def test_only_resolved_bouts_can_be_terminal() -> None:
    from dataclasses import replace

    node = SearchNode(
        hand=(), table=(), known_ranks=frozenset(), opponent_cards=2,
        talon_size=0, trump=Suit.HEARTS, refill_to=6, attack_limit=6,
        searcher_is_attacker=True, phase=Phase.ATTACK, to_move=Actor.SELF,
        kind=NodeKind.BOUT_RESOLVED,
    )
    assert node.outcome() is Outcome.WIN
    assert replace(node, kind=NodeKind.DECISION).outcome() is None
    assert replace(node, kind=NodeKind.UNKNOWN_FRONTIER).outcome() is None


# ======================================================================
# 5. Abstract action vs actual legal Move
# ======================================================================
def test_abstract_actions_and_real_moves_are_different_types() -> None:
    """They can never be confused, or handed to the rules engine by mistake."""
    node = SearchNode.from_view(observe(new_game(seed=1), 0))
    for action in OpponentAction:
        assert not isinstance(action, Move)
    for move in node.self_moves():
        assert isinstance(move, Move)
        assert not isinstance(move, OpponentAction)


def test_the_bot_always_returns_an_actual_supplied_legal_move() -> None:
    """Abstraction affects search, never the contract at the root."""
    bot = SearchBot(SearchConfig(depth=4), "s")
    bot.reset(random.Random(0))
    checked = 0
    for state in positions(12):
        player = state.current_player
        assert player is not None
        legal = get_legal_moves(state)
        chosen = bot.choose(observe(state, player), legal)
        assert isinstance(chosen, Move)
        assert chosen in legal
        apply_move(state, chosen).validate()
        checked += 1
    assert checked > 1000


def test_the_searchers_own_moves_are_never_abstracted() -> None:
    """Only the opponent is coarsened; our own options stay card-exact."""
    for state in positions(15):
        player = state.current_player
        assert player is not None
        node = SearchNode.from_view(observe(state, player))
        assert node.self_moves() == get_legal_moves(state, player)


# ======================================================================
# 6. Durak-specific handling of hidden opponent cards
# ======================================================================
def test_defence_by_an_unknown_card_spends_a_card_and_reveals_no_rank() -> None:
    state = make_state(
        hand0="6S 6C", hand1="10S AH", trump=Suit.HEARTS, talon="8C 8H",
        attacker=0,
    )
    node = SearchNode.from_view(observe(state, 0))
    after = node.after_self(AttackMove(Card.parse("6S")))
    beaten = after.after_opponent(OpponentAction.BEAT)
    assert beaten.opponent_cards == after.opponent_cards - 1
    assert beaten.known_ranks == after.known_ranks
    # A real trump defence would have added rank A to the table; the
    # abstraction does not assume it, so no extra throw-in appears.
    throwable = {m.card for m in beaten.self_moves() if isinstance(m, AttackMove)}
    assert throwable == {Card.parse("6C")}


def test_a_throw_in_after_a_take_is_searched_with_full_precision() -> None:
    """No opponent decision intervenes, so nothing is abstracted here."""
    state = make_state(
        hand0="6C 6D 9S", hand1="7C 7D", trump=Suit.HEARTS, talon="8C 8H",
        table=(("6S", None),), attacker=0, phase=Phase.TAKING, attack_limit=3,
    )
    node = SearchNode.from_view(observe(state, 0))
    assert node.to_move is Actor.SELF
    assert node.self_moves() == get_legal_moves(state)
    child = node.after_self(AttackMove(Card.parse("6C")))
    assert child.to_move is Actor.SELF, "throw-ins are consecutive own decisions"


def test_the_attack_limit_constrains_the_abstract_opponent_too() -> None:
    node = SearchNode(
        hand=(Card.parse("6S"),),
        table=(SearchSlot(Card.parse("7S"), Card.parse("8S"), defended=True),),
        known_ranks=frozenset({Card.parse("7S").rank}),
        opponent_cards=4,
        talon_size=3,
        trump=Suit.HEARTS,
        refill_to=6,
        attack_limit=1,
        searcher_is_attacker=False,
        phase=Phase.ATTACK,
        to_move=Actor.OPPONENT,
    )
    assert OpponentAction.ADD not in node.opponent_actions()
    assert node.opponent_actions() == (OpponentAction.END,)


def test_talon_exhaustion_is_tracked_by_count_alone() -> None:
    state = make_state(
        hand0="6S 6C", hand1="7S 9H", trump=Suit.HEARTS, talon="10C 10H",
        attacker=0,
    )
    node = SearchNode.from_view(observe(state, 0))
    resolved = (
        node.after_self(AttackMove(Card.parse("6S")))
        .after_opponent(OpponentAction.BEAT)
        .after_self(END_ATTACK)
    )
    assert resolved.talon_size == 0
    assert resolved.unknown_own == 2, "cards drawn, faces unknown"
    assert resolved.outcome() is None, "neither side is out"


def test_an_unknown_attack_ends_the_branch_rather_than_guessing() -> None:
    state = make_state(
        hand0="6C", hand1="6S 9H", trump=Suit.HEARTS, talon="8C 8H",
        table=(("7S", "8S"),), attacker=1, attack_limit=3,
    )
    node = SearchNode.from_view(observe(state, 0))
    assert node.to_move is Actor.OPPONENT
    frontier = node.after_opponent(OpponentAction.ADD)
    assert frontier.kind is NodeKind.UNKNOWN_FRONTIER
    assert frontier.self_moves() == ()
    assert frontier.outcome() is None


# ======================================================================
# 7. Full behavioural invariance across hidden permutations
# ======================================================================
def behaviour(view, legal) -> tuple:
    node = SearchNode.from_view(view)
    evaluator = BaselineEvaluator()
    plain = minimax(node, SearchConfig(depth=3, algorithm="minimax"),
                    root_moves=legal)
    fast = alpha_beta(node, SearchConfig(depth=3), root_moves=legal)
    return (
        node.key(),
        node.self_moves(),
        node.opponent_actions(),
        order_self_moves(node, legal),
        evaluator.evaluate(node, 0),
        plain.score,
        plain.move,
        fast.score,
        fast.move,
        fast.stats.nodes,
        fast.stats.leaves,
        fast.stats.terminals,
        fast.stats.cutoffs,
        fast.stats.max_depth,
        fast.stats.root_moves,
    )


@pytest.mark.parametrize("mode", ["opponent", "talon", "both"])
def test_search_behaviour_is_invariant_under_hidden_permutations(mode) -> None:
    """Opponent-hand, talon, and combined permutations, tested separately."""
    rng = random.Random(1234)
    compared = 0
    for state in positions(8):
        player = state.current_player
        if player is None or len(state.talon) < 3:
            continue
        legal = get_legal_moves(state)
        baseline = behaviour(observe(state, player), legal)

        view = observe(state, player)
        movable = view.unobserved_cards()
        opponent = [c for c in state.hands[1 - player] if c in movable]
        body = [c for c in state.talon[:-1] if c in movable]
        fixed_opp = [c for c in state.hands[1 - player] if c not in movable]
        fixed_body = [c for c in state.talon[:-1] if c not in movable]
        if not opponent or not body:
            continue

        for _ in range(2):
            if mode == "opponent":
                new_opp, new_body = list(opponent), list(body)
                rng.shuffle(new_opp)
            elif mode == "talon":
                new_opp, new_body = list(opponent), list(body)
                rng.shuffle(new_body)
            else:
                pool = opponent + body
                rng.shuffle(pool)
                new_opp, new_body = pool[: len(opponent)], pool[len(opponent) :]

            twin = state.replace(
                hands=tuple(
                    add_cards((), fixed_opp + new_opp) if p != player else state.hands[p]
                    for p in range(2)
                ),
                talon=tuple(fixed_body + new_body) + state.talon[-1:],
            )
            twin.validate()
            assert observe(twin, player) == observe(state, player)
            assert get_legal_moves(twin, player) == legal
            assert behaviour(observe(twin, player), legal) == baseline
            compared += 1
    assert compared > 60
