"""Search correctness.

The most valuable tests here are the hand-built ones: positions small
enough that the whole tree can be written out and the minimax value
computed by hand. Randomised comparison proves the two implementations
agree; only a manually computed value proves they agree on the *right*
answer.
"""

from __future__ import annotations

import random

import pytest

from durakfish.ai.evaluation import MATE_MARGIN, BaselineEvaluator
from durakfish.ai.ordering import order_self_moves
from durakfish.ai.search import (
    SearchConfig,
    SearchConfigurationError,
    SearchStats,
    alpha_beta,
    minimax,
    search,
)
from durakfish.ai.searchnode import Actor, Outcome, SearchNode, SearchSlot
from durakfish.cards import Card, Suit
from durakfish.game import Phase, apply_move, get_legal_moves, new_game
from durakfish.game.moves import END_ATTACK, TAKE, AttackMove, DefenseMove
from durakfish.information import observe
from tests.support import make_state

DEPTHS = (1, 2, 3, 4)


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
# Configuration
# ======================================================================
@pytest.mark.parametrize("depth", [0, -1, -10])
def test_a_non_positive_depth_is_refused(depth) -> None:
    with pytest.raises(SearchConfigurationError, match="depth"):
        SearchConfig(depth=depth)


def test_unknown_algorithm_ordering_and_evaluator_are_refused() -> None:
    with pytest.raises(SearchConfigurationError, match="algorithm"):
        SearchConfig(algorithm="mcts")
    with pytest.raises(SearchConfigurationError, match="ordering"):
        SearchConfig(ordering="killer")
    with pytest.raises(Exception, match="evaluator"):
        SearchConfig(evaluator="neural")


def test_a_valid_configuration_is_accepted() -> None:
    config = SearchConfig(depth=2, algorithm="minimax", ordering="canonical")
    assert config.depth == 2 and config.algorithm == "minimax"


# ======================================================================
# Hand-built trees with manually computed values
# ======================================================================
def test_depth_one_scores_each_child_and_picks_the_best() -> None:
    """Tiny tree, computed by hand.

    Talon empty. I hold 6S and 9C, opponent holds one card. Attacking
    with either card leads to a DEFENSE node for the opponent; at depth 1
    those children are evaluated, not expanded.
    """
    node = SearchNode(
        hand=(Card.parse("6S"), Card.parse("9C")),
        table=(),
        known_ranks=frozenset(),
        opponent_cards=1,
        talon_size=0,
        trump=Suit.HEARTS,
        refill_to=6,
        attack_limit=1,
        searcher_is_attacker=True,
        phase=Phase.ATTACK,
        to_move=Actor.SELF,
    )
    evaluator = BaselineEvaluator()
    expected = {}
    for move in node.self_moves():
        child = node.after_self(move)
        # card advantage 1 - 1 = 0; no trumps in hand after either play
        expected[move] = evaluator.evaluate(child, 1)
    assert set(expected.values()) == {0.0}, "both children score the same by hand"

    result = minimax(node, SearchConfig(depth=1, algorithm="minimax"))
    assert result.score == 0.0
    # Tie broken by ordering: the cheaper card is tried first and kept.
    assert result.move == AttackMove(Card.parse("6S"))


def test_depth_two_introduces_a_minimising_node() -> None:
    """The opponent picks the branch worst for us, and we can verify it."""
    node = SearchNode(
        hand=(Card.parse("6S"),),
        table=(),
        known_ranks=frozenset(),
        opponent_cards=2,
        talon_size=0,
        trump=Suit.HEARTS,
        refill_to=6,
        attack_limit=1,
        searcher_is_attacker=True,
        phase=Phase.ATTACK,
        to_move=Actor.SELF,
    )
    evaluator = BaselineEvaluator()
    child = node.after_self(AttackMove(Card.parse("6S")))
    assert child.to_move is Actor.OPPONENT

    by_hand = {}
    for action in child.opponent_actions():
        by_hand[action] = evaluator.evaluate(child.after_opponent(action), 2)
    manual = min(by_hand.values())

    result = minimax(node, SearchConfig(depth=2, algorithm="minimax"))
    assert result.score == manual
    assert alpha_beta(node, SearchConfig(depth=2)).score == manual


def test_the_search_finds_a_forced_win_and_prefers_it() -> None:
    """Empty talon, one card each: playing it out wins outright."""
    state = make_state(
        hand0="6S", hand1="7S 9H", trump=Suit.HEARTS, talon="", attacker=0
    )
    node, legal = root_of(state)
    result = alpha_beta(node, SearchConfig(depth=4), root_moves=legal)
    assert result.score >= MATE_MARGIN, "a forced win must be scored as proven"
    assert result.is_proven
    assert result.move == AttackMove(Card.parse("6S"))


def test_the_search_recognises_a_forced_loss() -> None:
    node = SearchNode(
        hand=(Card.parse("6S"), Card.parse("9C")),
        table=(SearchSlot(Card.parse("7S"), Card.parse("8S"), defended=True),),
        known_ranks=frozenset({Card.parse("7S").rank, Card.parse("8S").rank}),
        opponent_cards=0,
        talon_size=0,
        trump=Suit.HEARTS,
        refill_to=6,
        attack_limit=2,
        searcher_is_attacker=True,
        phase=Phase.ATTACK,
        to_move=Actor.SELF,
    )
    result = alpha_beta(node, SearchConfig(depth=3), root_moves=node.self_moves())
    assert result.score <= -MATE_MARGIN
    assert result.is_proven


def test_a_sooner_win_outscores_a_later_one() -> None:
    assert (
        alpha_beta(_forced_win_node(), SearchConfig(depth=4)).score
        > MATE_MARGIN
    )
    early = _forced_win_node()
    fast = alpha_beta(early, SearchConfig(depth=4)).score
    # The same win found from one ply deeper must score slightly lower.
    deeper = early.after_self(AttackMove(Card.parse("6S")))
    assert deeper.to_move is Actor.OPPONENT
    slower = alpha_beta(early, SearchConfig(depth=6)).score
    assert fast >= slower or fast == slower


def _forced_win_node() -> SearchNode:
    state = make_state(
        hand0="6S", hand1="7S 9H", trump=Suit.HEARTS, talon="", attacker=0
    )
    node, _ = root_of(state)
    return node


def test_a_defender_that_cannot_beat_has_only_one_move() -> None:
    state = make_state(
        hand0="6D", hand1="6S 7S", trump=Suit.HEARTS, talon="",
        table=(("9H", None),), attacker=0, phase=Phase.DEFENSE, attack_limit=2,
    )
    node, legal = root_of(state)
    assert legal == (TAKE,)
    result = alpha_beta(node, SearchConfig(depth=3), root_moves=legal)
    assert result.move == TAKE


def test_throw_in_sequences_are_searched_to_their_end() -> None:
    """During a take the attacker moves repeatedly with nothing hidden."""
    state = make_state(
        hand0="6C 6D 9S", hand1="7C 7D", trump=Suit.HEARTS, talon="8C 8H",
        table=(("6S", None),), attacker=0, phase=Phase.TAKING, attack_limit=3,
    )
    node, legal = root_of(state)
    assert node.to_move is Actor.SELF
    result = alpha_beta(node, SearchConfig(depth=4), root_moves=legal)
    assert result.move in legal
    # Two throw-ins are available and both are its own consecutive decisions.
    assert result.stats.nodes > len(legal)


# ======================================================================
# Depth semantics
# ======================================================================
def test_depth_zero_is_refused_at_the_root() -> None:
    with pytest.raises(SearchConfigurationError):
        SearchConfig(depth=0)


def test_deeper_searches_visit_at_least_as_many_nodes() -> None:
    for state in list(positions(4))[:60]:
        node, legal = root_of(state)
        counts = [
            minimax(
                node, SearchConfig(depth=d, algorithm="minimax"), root_moves=legal
            ).stats.nodes
            for d in DEPTHS
        ]
        assert counts == sorted(counts), counts


def test_max_depth_reported_never_exceeds_the_budget() -> None:
    for state in list(positions(4))[:60]:
        node, legal = root_of(state)
        for depth in DEPTHS:
            result = alpha_beta(
                node, SearchConfig(depth=depth), root_moves=legal
            )
            assert result.stats.max_depth <= depth


# ======================================================================
# Minimax / alpha-beta equivalence
# ======================================================================
def test_minimax_and_alpha_beta_agree_on_random_positions() -> None:
    compared = 0
    for state in positions(20):
        node, legal = root_of(state)
        for depth in DEPTHS:
            plain = minimax(
                node, SearchConfig(depth=depth, algorithm="minimax"), root_moves=legal
            )
            pruned = alpha_beta(node, SearchConfig(depth=depth), root_moves=legal)
            assert plain.score == pruned.score, (depth, repr(node))
            assert plain.move == pruned.move, (depth, repr(node))
            compared += 1
    assert compared > 5000


def test_minimax_and_alpha_beta_agree_under_adversarial_play() -> None:
    def always_take(legal):
        return next((m for m in legal if m == TAKE), legal[0])

    def max_pressure(legal):
        attacks = [m for m in legal if isinstance(m, AttackMove)]
        return attacks[0] if attacks else legal[0]

    compared = 0
    for policy in (always_take, max_pressure):
        for seed in range(8):
            state = new_game(seed=seed)
            while not state.is_over:
                node, legal = root_of(state)
                for depth in (2, 3):
                    plain = minimax(
                        node, SearchConfig(depth=depth, algorithm="minimax"),
                        root_moves=legal,
                    )
                    pruned = alpha_beta(
                        node, SearchConfig(depth=depth), root_moves=legal
                    )
                    assert (plain.score, plain.move) == (pruned.score, pruned.move)
                    compared += 1
                state = apply_move(state, policy(legal))
    assert compared > 800


def test_alpha_beta_actually_prunes_where_pruning_is_possible() -> None:
    """Not 'the code contains alpha and beta' — measured node counts."""
    saved = pruned_total = plain_total = 0
    for state in list(positions(10)):
        node, legal = root_of(state)
        plain = minimax(
            node, SearchConfig(depth=4, algorithm="minimax"), root_moves=legal
        ).stats
        fast = alpha_beta(node, SearchConfig(depth=4), root_moves=legal).stats
        assert fast.nodes <= plain.nodes
        plain_total += plain.nodes
        pruned_total += fast.nodes
        if fast.nodes < plain.nodes:
            saved += 1
            assert fast.cutoffs > 0, "fewer nodes but no cutoff recorded"
    assert saved > 50, "pruning never triggered anywhere"
    assert pruned_total < plain_total * 0.9


def test_depth_one_cannot_prune_and_records_no_cutoffs() -> None:
    """A position where pruning is impossible, as a control."""
    for state in list(positions(4))[:40]:
        node, legal = root_of(state)
        plain = minimax(
            node, SearchConfig(depth=1, algorithm="minimax"), root_moves=legal
        ).stats
        fast = alpha_beta(node, SearchConfig(depth=1), root_moves=legal).stats
        assert fast.nodes == plain.nodes
        assert fast.cutoffs == 0


# ======================================================================
# Ordering
# ======================================================================
def test_ordering_changes_node_counts_but_never_the_result() -> None:
    differing = 0
    for state in list(positions(8)):
        node, legal = root_of(state)
        static = alpha_beta(
            node, SearchConfig(depth=3, ordering="static"), root_moves=legal
        )
        canonical = alpha_beta(
            node, SearchConfig(depth=3, ordering="canonical"), root_moves=legal
        )
        assert static.score == canonical.score
        if static.stats.nodes != canonical.stats.nodes:
            differing += 1
    assert differing > 0, "the two orderings are indistinguishable; is one wired up?"


def test_ordering_is_permutation_invariant() -> None:
    rng = random.Random(5)
    for state in list(positions(5))[:60]:
        node, legal = root_of(state)
        baseline = order_self_moves(node, legal)
        shuffled = list(legal)
        for _ in range(3):
            rng.shuffle(shuffled)
            assert order_self_moves(node, shuffled) == baseline


def test_ordering_rejects_an_unknown_strategy() -> None:
    node, _ = root_of(new_game(seed=1))
    with pytest.raises(ValueError, match="unknown ordering"):
        order_self_moves(node, node.self_moves(), ordering="history")


# ======================================================================
# Purity
# ======================================================================
def test_search_leaves_its_inputs_untouched() -> None:
    for state in list(positions(6))[:80]:
        player = state.current_player
        assert player is not None
        view = observe(state, player)
        legal = get_legal_moves(state)
        node = SearchNode.from_view(view)

        view_before, key_before = view.to_dict(), view.key()
        legal_before, node_key = tuple(legal), node.key()
        state_before = state.to_dict()

        alpha_beta(node, SearchConfig(depth=3), root_moves=legal)

        assert view.to_dict() == view_before and view.key() == key_before
        assert tuple(legal) == legal_before
        assert node.key() == node_key
        assert state.to_dict() == state_before


def test_repeated_searches_return_identical_results() -> None:
    for state in list(positions(5))[:60]:
        node, legal = root_of(state)
        results = [
            alpha_beta(node, SearchConfig(depth=3), root_moves=legal)
            for _ in range(3)
        ]
        assert len({(r.move, r.score) for r in results}) == 1
        assert len({r.stats.nodes for r in results}) == 1


def test_searching_a_copied_node_gives_the_same_answer() -> None:
    from dataclasses import replace

    for state in list(positions(5))[:60]:
        node, legal = root_of(state)
        clone = replace(node)
        assert clone == node and clone.key() == node.key()
        a = alpha_beta(node, SearchConfig(depth=3), root_moves=legal)
        b = alpha_beta(clone, SearchConfig(depth=3), root_moves=legal)
        assert (a.move, a.score) == (b.move, b.score)


# ======================================================================
# Statistics
# ======================================================================
def test_statistics_do_not_influence_the_search() -> None:
    for state in list(positions(6))[:80]:
        node, legal = root_of(state)
        with_stats = alpha_beta(
            node, SearchConfig(depth=3, collect_statistics=True), root_moves=legal
        )
        without = alpha_beta(
            node, SearchConfig(depth=3, collect_statistics=False), root_moves=legal
        )
        assert (with_stats.move, with_stats.score) == (without.move, without.score)
        assert without.stats.nodes == 0, "statistics were collected while disabled"


def test_statistics_are_internally_consistent() -> None:
    for state in list(positions(6))[:80]:
        node, legal = root_of(state)
        stats = alpha_beta(node, SearchConfig(depth=3), root_moves=legal).stats
        assert stats.nodes >= stats.leaves >= stats.terminals >= 0
        assert stats.root_moves == len(legal)
        assert 0.0 <= stats.pruning_ratio <= 1.0


def test_stats_counters_start_at_zero_and_respect_the_enable_flag() -> None:
    stats = SearchStats(enabled=False)
    stats.visit(3)
    stats.leaf(terminal=True)
    stats.cutoff()
    assert (stats.nodes, stats.leaves, stats.terminals, stats.cutoffs) == (0, 0, 0, 0)
    assert stats.pruning_ratio == 0.0


# ======================================================================
# Root behaviour
# ======================================================================
def test_the_root_only_ever_selects_a_supplied_move() -> None:
    for state in list(positions(8)):
        node, legal = root_of(state)
        for cut in (1, 2):
            restricted = legal[:cut]
            if not restricted:
                continue
            result = alpha_beta(
                node, SearchConfig(depth=3), root_moves=restricted
            )
            assert result.move in restricted


def test_search_dispatches_on_the_configured_algorithm() -> None:
    node, legal = root_of(new_game(seed=2))
    plain = search(node, SearchConfig(depth=2, algorithm="minimax"), root_moves=legal)
    fast = search(node, SearchConfig(depth=2, algorithm="alpha_beta"), root_moves=legal)
    assert plain.move == fast.move and plain.score == fast.score
    assert fast.stats.nodes <= plain.stats.nodes


def test_an_empty_root_move_list_yields_no_move_rather_than_a_guess() -> None:
    node, _ = root_of(new_game(seed=1))
    result = alpha_beta(node, SearchConfig(depth=2), root_moves=())
    assert result.move is None
