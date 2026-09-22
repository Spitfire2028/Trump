"""The search-node abstraction and the evaluator.

The node is the part of Phase 5 most likely to go quietly wrong: a search
that keeps the whole state around "just for now" solves the wrong problem
while looking correct. These tests pin down what a node contains, what it
refuses to contain, and that scoring depends on nothing hidden.
"""

from __future__ import annotations

import random

import pytest

from durakfish.ai.evaluation import (
    MATE_MARGIN,
    TERMINAL_DRAW,
    TERMINAL_LOSS,
    TERMINAL_WIN,
    BaselineEvaluator,
    EvaluationWeights,
    get_evaluator,
    terminal_score,
)
from durakfish.ai.searchnode import (
    Actor,
    NodeKind,
    OpponentAction,
    Outcome,
    SearchNode,
    SearchSlot,
)
from durakfish.cards import Card, Suit
from durakfish.exceptions import DurakFishError
from durakfish.game import Phase, apply_move, get_legal_moves, new_game
from durakfish.game.moves import END_ATTACK, TAKE, AttackMove
from durakfish.information import observe
from tests.support import make_state
from tests.test_phase3_audit import (
    FORBIDDEN_TYPES,
    cards_reachable_from,
    permute_hidden,
    reachable,
)


def root_of(state):
    player = state.current_player
    assert player is not None
    return SearchNode.from_view(observe(state, player)), get_legal_moves(state)


def positions(games: int = 12):
    for seed in range(games):
        rng = random.Random(seed)
        state = new_game(seed=seed)
        while not state.is_over:
            yield state
            state = apply_move(state, rng.choice(get_legal_moves(state)))


# ======================================================================
# What a node contains, and what it refuses to
# ======================================================================
def test_a_node_exposes_no_field_that_could_hold_hidden_cards() -> None:
    node, _ = root_of(new_game(seed=1))
    for forbidden in ("hands", "talon", "history", "discard_order", "state"):
        assert not hasattr(node, forbidden)
    assert isinstance(node.opponent_cards, int)
    assert isinstance(node.talon_size, int)


def test_every_card_in_a_root_node_has_been_observed() -> None:
    for state in positions(10):
        player = state.current_player
        assert player is not None
        view = observe(state, player)
        node = SearchNode.from_view(view)
        leaked = cards_reachable_from(node) - view.observed_cards()
        assert not leaked, f"node exposes unobserved {leaked}"
        for obj in reachable(node):
            assert not isinstance(obj, FORBIDDEN_TYPES)


def test_a_node_built_from_indistinguishable_views_is_identical() -> None:
    rng = random.Random(7)
    checked = 0
    for state in positions(8):
        player = state.current_player
        if player is None or not state.talon:
            continue
        baseline = SearchNode.from_view(observe(state, player))
        for _ in range(3):
            twin = permute_hidden(state, player, rng)
            assert SearchNode.from_view(observe(twin, player)) == baseline
            assert SearchNode.from_view(observe(twin, player)).key() == baseline.key()
            checked += 1
    assert checked > 100


# ======================================================================
# Node move generation agrees with the frozen rules engine
# ======================================================================
def test_node_self_moves_match_the_authoritative_generator() -> None:
    """The frozen engine stays the authority on legality."""
    checked = 0
    for state in positions(25):
        node, legal = root_of(state)
        assert node.self_moves() == legal, repr(node)
        checked += 1
    assert checked > 2000


def test_node_self_moves_match_under_adversarial_play() -> None:
    def always_take(legal):
        return next((m for m in legal if m == TAKE), legal[0])

    def max_pressure(legal):
        attacks = [m for m in legal if isinstance(m, AttackMove)]
        return attacks[0] if attacks else legal[0]

    checked = 0
    for policy in (always_take, max_pressure):
        for seed in range(10):
            state = new_game(seed=seed)
            while not state.is_over:
                node, legal = root_of(state)
                assert node.self_moves() == legal
                checked += 1
                state = apply_move(state, policy(legal))
    assert checked > 500


def test_opponent_actions_never_name_a_card() -> None:
    """The opponent is modelled by action class, never by card."""
    for state in positions(8):
        node, _ = root_of(state)
        for action in node.opponent_actions():
            assert isinstance(action, OpponentAction)
            assert not cards_reachable_from(action)


def test_an_opponent_with_no_cards_can_only_take() -> None:
    node = SearchNode(
        hand=(Card.parse("6S"),),
        table=(SearchSlot(Card.parse("7S")),),
        known_ranks=frozenset({Card.parse("7S").rank}),
        opponent_cards=0,
        talon_size=0,
        trump=Suit.HEARTS,
        refill_to=6,
        attack_limit=2,
        searcher_is_attacker=True,
        phase=Phase.DEFENSE,
        to_move=Actor.OPPONENT,
    )
    assert node.opponent_actions() == (OpponentAction.TAKE,)


def test_beating_with_an_unknown_card_adds_no_rank_to_the_table() -> None:
    """Conservative: it can understate options, never invent one."""
    node = SearchNode(
        hand=(Card.parse("6S"), Card.parse("9C")),
        table=(SearchSlot(Card.parse("7S")),),
        known_ranks=frozenset({Card.parse("7S").rank}),
        opponent_cards=3,
        talon_size=5,
        trump=Suit.HEARTS,
        refill_to=6,
        attack_limit=3,
        searcher_is_attacker=True,
        phase=Phase.DEFENSE,
        to_move=Actor.OPPONENT,
    )
    after = node.after_opponent(OpponentAction.BEAT)
    assert after.known_ranks == node.known_ranks
    assert after.table[0].defended and after.table[0].defending_card is None
    assert after.opponent_cards == 2
    assert after.to_move is Actor.SELF
    # The unknown card's rank is not available for throwing in.
    assert all(isinstance(m, AttackMove) is False or m.card.rank in after.known_ranks
               for m in after.self_moves())


def test_an_unknown_attacking_card_stops_the_branch() -> None:
    node = SearchNode(
        hand=(Card.parse("6S"),),
        table=(SearchSlot(Card.parse("7S"), Card.parse("8S"), defended=True),),
        known_ranks=frozenset(),
        opponent_cards=3,
        talon_size=5,
        trump=Suit.HEARTS,
        refill_to=6,
        attack_limit=3,
        searcher_is_attacker=False,
        phase=Phase.ATTACK,
        to_move=Actor.OPPONENT,
    )
    after = node.after_opponent(OpponentAction.ADD)
    assert after.kind is NodeKind.UNKNOWN_FRONTIER
    assert after.is_leaf
    assert after.self_moves() == ()


# ======================================================================
# Resolution and terminals
# ======================================================================
def test_a_resolved_bout_is_a_leaf_because_draws_are_hidden() -> None:
    state = make_state(
        hand0="6S 9C", hand1="7S 9H", trump=Suit.HEARTS, talon="8C 8D 8S 8H",
        attacker=0,
    )
    node, _ = root_of(state)
    after = node.after_self(AttackMove(Card.parse("6S")))
    beaten = after.after_opponent(OpponentAction.BEAT)
    resolved = beaten.after_self(END_ATTACK)
    assert resolved.kind is NodeKind.BOUT_RESOLVED
    assert resolved.is_leaf
    assert resolved.outcome() is None, "talon remains, so nothing is proven"
    assert resolved.unknown_own > 0, "cards were drawn but their faces are unknown"


def test_a_win_is_proven_when_the_talon_is_empty() -> None:
    """Attacker plays its last card, defender beats it: attacker is out."""
    state = make_state(
        hand0="6S", hand1="7S 9H", trump=Suit.HEARTS, talon="", attacker=0
    )
    node, _ = root_of(state)
    resolved = (
        node.after_self(AttackMove(Card.parse("6S")))
        .after_opponent(OpponentAction.BEAT)
        .after_self(END_ATTACK)
    )
    assert resolved.own_cards == 0
    assert resolved.opponent_cards == 1
    assert resolved.outcome() is Outcome.WIN


def test_a_loss_is_proven_when_only_the_opponent_runs_out() -> None:
    node = SearchNode(
        hand=(Card.parse("6S"), Card.parse("9C")),
        table=(SearchSlot(Card.parse("7S"), Card.parse("8S"), defended=True),),
        known_ranks=frozenset(),
        opponent_cards=0,
        talon_size=0,
        trump=Suit.HEARTS,
        refill_to=6,
        attack_limit=2,
        searcher_is_attacker=True,
        phase=Phase.ATTACK,
        to_move=Actor.SELF,
    )
    assert node.after_self(END_ATTACK).outcome() is Outcome.LOSS


def test_a_simultaneous_finish_is_a_draw() -> None:
    state = make_state(
        hand0="6S", hand1="7S", trump=Suit.HEARTS, talon="", attacker=0
    )
    node, _ = root_of(state)
    resolved = (
        node.after_self(AttackMove(Card.parse("6S")))
        .after_opponent(OpponentAction.BEAT)
        .after_self(END_ATTACK)
    )
    assert resolved.outcome() is Outcome.DRAW


def test_taking_moves_the_whole_table_to_the_taker() -> None:
    state = make_state(
        hand0="6S 6C", hand1="AH 9H", trump=Suit.HEARTS, talon="", attacker=0
    )
    node, _ = root_of(state)
    after = node.after_self(AttackMove(Card.parse("6S")))
    taken = after.after_opponent(OpponentAction.TAKE).after_self(END_ATTACK)
    assert taken.opponent_cards == 3, "two held plus the card picked up"
    assert taken.own_cards == 1


def test_the_attacker_refills_first_as_the_rules_require() -> None:
    """docs/rules.md §7: draw order decides who is left holding cards."""
    state = make_state(
        hand0="6S 6C", hand1="7S 9H", trump=Suit.HEARTS, talon="10C 10H",
        attacker=0,
    )
    node, _ = root_of(state)
    resolved = (
        node.after_self(AttackMove(Card.parse("6S")))
        .after_opponent(OpponentAction.BEAT)
        .after_self(END_ATTACK)
    )
    # The searcher attacked, so it draws first and takes both cards.
    assert resolved.talon_size == 0
    assert resolved.own_cards == 3
    assert resolved.opponent_cards == 1


def test_node_resolution_matches_the_engine_on_real_bouts() -> None:
    """Cross-check the abstraction's arithmetic against the frozen engine."""
    checked = 0
    for seed in range(40):
        rng = random.Random(seed)
        state = new_game(seed=seed)
        while not state.is_over:
            legal = get_legal_moves(state)
            move = rng.choice(legal)
            player = state.current_player
            assert player is not None
            if move == END_ATTACK:
                node = SearchNode.from_view(observe(state, player))
                predicted = node.after_self(END_ATTACK)
                actual = apply_move(state, move)
                if not actual.is_over:
                    assert predicted.own_cards == len(actual.hands[player])
                    assert predicted.opponent_cards == len(actual.hands[1 - player])
                    assert predicted.talon_size == len(actual.talon)
                    checked += 1
            state = apply_move(state, move)
    assert checked > 300


# ======================================================================
# Evaluation
# ======================================================================
def test_terminal_scores_are_ordered_and_prefer_sooner_results() -> None:
    assert terminal_score(Outcome.WIN, 0) > terminal_score(Outcome.WIN, 5)
    assert terminal_score(Outcome.LOSS, 5) > terminal_score(Outcome.LOSS, 0)
    assert terminal_score(Outcome.WIN, 50) > TERMINAL_DRAW > terminal_score(
        Outcome.LOSS, 50
    )
    assert abs(terminal_score(Outcome.WIN, 50)) > MATE_MARGIN
    assert abs(terminal_score(Outcome.LOSS, 50)) > MATE_MARGIN


def test_a_heuristic_score_can_never_be_mistaken_for_a_proven_one() -> None:
    weights = EvaluationWeights()
    assert weights.bound() < MATE_MARGIN
    evaluator = BaselineEvaluator()
    for state in positions(10):
        node, _ = root_of(state)
        if node.outcome() is None:
            assert abs(evaluator.evaluate(node, 0)) < MATE_MARGIN


def test_evaluation_is_identical_for_indistinguishable_positions() -> None:
    """The central hidden-state invariance property."""
    evaluator = BaselineEvaluator()
    rng = random.Random(31)
    compared = 0
    for state in positions(10):
        player = state.current_player
        if player is None or not state.talon:
            continue
        baseline = evaluator.evaluate(SearchNode.from_view(observe(state, player)), 0)
        for _ in range(3):
            twin = permute_hidden(state, player, rng)
            score = evaluator.evaluate(
                SearchNode.from_view(observe(twin, player)), 0
            )
            assert score == baseline
            compared += 1
    assert compared > 150


def test_evaluation_does_respond_to_public_information() -> None:
    """Non-vacuity: an invariant evaluator that returns a constant is useless."""
    evaluator = BaselineEvaluator()
    scores = {evaluator.evaluate(root_of(s)[0], 0) for s in positions(6)}
    assert len(scores) > 5, "the evaluator barely distinguishes positions"


def test_each_evaluation_term_moves_the_score_in_the_stated_direction() -> None:
    evaluator = BaselineEvaluator()
    base = SearchNode(
        hand=(Card.parse("6S"), Card.parse("9C")),
        table=(),
        known_ranks=frozenset(),
        opponent_cards=4,
        talon_size=10,
        trump=Suit.HEARTS,
        refill_to=6,
        attack_limit=6,
        searcher_is_attacker=True,
        phase=Phase.ATTACK,
        to_move=Actor.SELF,
    )
    reference = evaluator.evaluate(base, 0)

    # More cards for the opponent is better for us.
    from dataclasses import replace

    assert evaluator.evaluate(replace(base, opponent_cards=5), 0) > reference
    # More cards for us is worse.
    assert evaluator.evaluate(
        replace(base, hand=(*base.hand, Card.parse("10D"))), 0
    ) < reference
    # A trump in hand is worth more than the same card off-suit.
    with_trump = replace(base, hand=(Card.parse("6S"), Card.parse("9H")))
    assert evaluator.evaluate(with_trump, 0) > reference
    # An open attack against us is a penalty.
    facing = replace(
        base,
        searcher_is_attacker=False,
        phase=Phase.DEFENSE,
        table=(SearchSlot(Card.parse("7S")),),
    )
    assert evaluator.evaluate(facing, 0) < evaluator.evaluate(
        replace(facing, table=()), 0
    )


def test_evaluation_is_deterministic_and_repeatable() -> None:
    evaluator = BaselineEvaluator()
    for state in positions(5):
        node, _ = root_of(state)
        assert len({evaluator.evaluate(node, 0) for _ in range(20)}) == 1


def test_unknown_cards_count_toward_material_but_earn_no_trump_bonus() -> None:
    from dataclasses import replace

    evaluator = BaselineEvaluator()
    base = SearchNode(
        hand=(), table=(), known_ranks=frozenset(), opponent_cards=6,
        talon_size=0, trump=Suit.HEARTS, refill_to=6, attack_limit=6,
        searcher_is_attacker=True, phase=Phase.ATTACK, to_move=Actor.SELF,
    )
    with_unknown = replace(base, unknown_own=2)
    assert evaluator.evaluate(with_unknown, 0) == evaluator.evaluate(base, 0) - 2.0


def test_get_evaluator_rejects_an_unknown_name() -> None:
    assert get_evaluator("baseline").name == "baseline"
    with pytest.raises(DurakFishError, match="unknown evaluator"):
        get_evaluator("stockfish")


def test_terminal_constants_are_named_not_magic() -> None:
    assert TERMINAL_WIN > 0 > TERMINAL_LOSS
    assert TERMINAL_DRAW == 0.0
    assert TERMINAL_WIN == -TERMINAL_LOSS
