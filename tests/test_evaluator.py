"""The Phase 7.3.10 candidate evaluator.

The candidate differs from the baseline in exactly one way: how it
measures material. These tests pin that mechanism, the contract it shares
with the baseline, and the structural guarantee that neither evaluator
can read hidden cards.

The sanity cases are guardrails, not proofs, and each one is justified by
the frozen Phase 2 rules rather than by card-game folklore.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from durakfish.ai.determinized import build_position
from durakfish.ai.evaluation import (
    EVALUATORS,
    MATE_MARGIN,
    TERMINAL_DRAW,
    TERMINAL_LOSS,
    TERMINAL_WIN,
    BaselineEvaluator,
    EvaluationWeights,
    FlexibleEvaluator,
    MechanisticEvaluator,
    StructuralEvaluator,
    StructuralWeights,
    get_evaluator,
    terminal_score,
)
from durakfish.ai.searchnode import (
    Actor,
    NodeKind,
    Outcome,
    SearchNode,
    SearchSlot,
)
from durakfish.cards import Card, Rank, Suit
from durakfish.game import apply_move, get_legal_moves, new_game
from durakfish.game.state import Phase
from durakfish.information import DeterminizedWorld, SearchInformation, observe
from tests.support import make_state
from tests.test_phase3_audit import FORBIDDEN_TYPES, cards_reachable_from, reachable


def walk(seed: int):
    rng = random.Random(seed)
    state = new_game(seed=seed)
    while not state.is_over:
        yield state
        state = apply_move(state, rng.choice(get_legal_moves(state)))
    yield state


def node_for(state, player: int):
    """The evaluation node for a position, via the frozen adapter."""
    information = SearchInformation.from_view(observe(state, player))
    world = DeterminizedWorld(
        player=player,
        opponent_hand=frozenset(state.hands[1 - player]),
        talon=tuple(state.talon),
    )
    return build_position(information, world).evaluation_node(player)


# ======================================================================
# Registration, without disturbing the production default
# ======================================================================
def test_the_candidate_is_registered_but_is_not_the_default() -> None:
    """Phase 7.3.10 must not silently replace the production evaluator."""
    from durakfish.ai.ismcts import ISMCTSConfig
    from durakfish.ai.search import SearchConfig

    assert "mechanistic" in EVALUATORS
    assert isinstance(get_evaluator("mechanistic"), MechanisticEvaluator)
    assert SearchConfig(depth=1).evaluator == "baseline"
    assert ISMCTSConfig().evaluator == "baseline"
    assert EVALUATORS["baseline"] is BaselineEvaluator


# ======================================================================
# The shared contract
# ======================================================================
def test_terminal_values_are_exact_and_never_approximated() -> None:
    """The frozen terminal semantics, pinned as constants.

    This covers ``terminal_score`` alone. Checking that an evaluator
    actually *routes* a decided node through it is a separate test below,
    and the two are separate because the mutation audit showed that
    conflating them leaves the routing untested.
    """
    for outcome, expected in (
        (Outcome.WIN, TERMINAL_WIN),
        (Outcome.LOSS, TERMINAL_LOSS),
        (Outcome.DRAW, TERMINAL_DRAW),
    ):
        assert terminal_score(outcome, 0) == expected
    assert terminal_score(Outcome.WIN, 4) == TERMINAL_WIN - 4
    assert terminal_score(Outcome.LOSS, 4) == TERMINAL_LOSS + 4


def terminal_node(outcome: Outcome) -> SearchNode:
    """The smallest node ``SearchNode.outcome()`` will call decided.

    A resolved bout with an empty talon and at least one empty hand, which
    is the only shape the frozen node concedes a proven result to.
    """
    mine_out = outcome in (Outcome.WIN, Outcome.DRAW)
    hand = () if mine_out else (Card(Rank.SIX, Suit.SPADES),)
    opponent = 0 if outcome in (Outcome.LOSS, Outcome.DRAW) else 1
    return SearchNode(
        hand=hand,
        table=(),
        known_ranks=frozenset(),
        opponent_cards=opponent,
        talon_size=0,
        trump=Suit.HEARTS,
        refill_to=6,
        attack_limit=6,
        searcher_is_attacker=True,
        phase=Phase.ATTACK,
        to_move=Actor.SELF,
        kind=NodeKind.BOUT_RESOLVED,
    )


@pytest.mark.parametrize("ply", [0, 1, 4, 9])
def test_a_decided_node_is_scored_exactly_at_every_ply(ply: int) -> None:
    """``terminal_score`` being correct is not enough: each evaluator must
    actually route a decided node through it, at every ply rather than
    only at the root.

    Written because the Phase 7.3.10 mutation audit found that mutations
    approximating a terminal *inside* ``evaluate`` — returning a heuristic
    once ``ply > 0``, or a flat +-500 — survived the whole suite. Both
    fail here.
    """
    for evaluator in (BaselineEvaluator(), MechanisticEvaluator()):
        for outcome in (Outcome.WIN, Outcome.LOSS, Outcome.DRAW):
            node = terminal_node(outcome)
            assert node.outcome() is outcome
            value = evaluator.evaluate(node, ply)
            assert value == terminal_score(outcome, ply)
            if outcome is not Outcome.DRAW:
                assert abs(value) > MATE_MARGIN


def test_non_terminal_values_stay_far_below_the_mate_margin() -> None:
    """A heuristic score must never be mistaken for a proven result."""
    evaluator = MechanisticEvaluator()
    for seed in range(8):
        for state in walk(seed):
            player = state.current_player
            if player is None:
                continue
            value = evaluator.evaluate(node_for(state, player), 0)
            assert abs(value) < MATE_MARGIN


def test_the_candidate_is_deterministic_and_finite() -> None:
    evaluator = MechanisticEvaluator()
    for seed in range(5):
        for state in walk(seed):
            player = state.current_player
            if player is None:
                continue
            node = node_for(state, player)
            values = {evaluator.evaluate(node, 0) for _ in range(4)}
            assert len(values) == 1
            assert abs(values.pop()) < 1e6


# ======================================================================
# The material mechanism
# ======================================================================
def test_with_an_empty_talon_the_candidate_equals_the_baseline() -> None:
    """Proven, not assumed: with no refill the two are the same function.

    This is why the independent exact solver cannot discriminate them —
    its reachable domain is almost entirely empty-talon positions.
    """
    baseline, candidate = BaselineEvaluator(), MechanisticEvaluator()
    compared = 0
    for seed in range(15):
        for state in walk(seed):
            player = state.current_player
            if player is None or state.talon:
                continue
            node = node_for(state, player)
            assert candidate.evaluate(node, 0) == baseline.evaluate(node, 0)
            assert candidate.material(node) == node.opponent_cards - node.own_cards
            compared += 1
    assert compared > 100


def test_below_the_refill_size_a_deficit_is_treated_as_transient() -> None:
    """M1: while the talon lasts, refill tops hands back up, so a shortfall
    below ``refill_to`` does not persist and must not score as material."""
    candidate = MechanisticEvaluator()
    checked = 0
    for seed in range(12):
        for state in walk(seed):
            player = state.current_player
            if player is None or not state.talon:
                continue
            node = node_for(state, player)
            if node.own_cards <= node.refill_to and (
                node.opponent_cards <= node.refill_to
            ):
                assert candidate.material(node) == 0.0
                checked += 1
    assert checked > 50


def test_above_the_refill_size_the_surplus_is_what_counts() -> None:
    """M1: cards above ``refill_to`` are never removed by refill."""
    candidate = MechanisticEvaluator()
    found = 0
    for seed in range(15):
        for state in walk(seed):
            player = state.current_player
            if player is None or not state.talon:
                continue
            node = node_for(state, player)
            if node.own_cards <= node.refill_to:
                continue
            expected = max(0, node.opponent_cards - node.refill_to) - (
                node.own_cards - node.refill_to
            )
            assert candidate.material(node) == float(expected)
            found += 1
    assert found > 50


def test_holding_more_cards_is_never_scored_as_an_improvement() -> None:
    """Direction check against the objective: the game is won by shedding
    cards, so extra cards must never raise the material term.

    ``unknown_own`` is incremented rather than the hand, because that is
    the one field which adds a card the searcher holds without inventing
    a card identity that would break conservation elsewhere.
    """
    candidate = MechanisticEvaluator()
    checked = 0
    for seed in range(10):
        for state in walk(seed):
            player = state.current_player
            if player is None:
                continue
            node = node_for(state, player)
            base = candidate.material(node)
            bigger = dataclasses.replace(node, unknown_own=node.unknown_own + 1)
            assert bigger.own_cards == node.own_cards + 1
            assert candidate.material(bigger) <= base
            checked += 1
    assert checked > 100


def test_the_talon_switch_is_bounded_not_explosive() -> None:
    """The discontinuity at talon exhaustion is real — refill stops — but
    it must stay on the heuristic scale, never near a terminal score."""
    candidate = MechanisticEvaluator()
    hands = dict(hand0="6S 6C 6D 7S 7C 7D 8S", hand1="9H 10H", attacker=0)
    with_talon = candidate.material(
        node_for(make_state(trump=Suit.HEARTS, talon="AH", **hands), 0)
    )
    without = candidate.material(
        node_for(make_state(trump=Suit.HEARTS, talon="", **hands), 0)
    )
    assert abs(without - with_talon) < 12
    assert abs(with_talon) < MATE_MARGIN and abs(without) < MATE_MARGIN


# ======================================================================
# Information boundary
# ======================================================================
def test_the_candidate_cannot_reach_hidden_cards() -> None:
    """Structural: a ``SearchNode`` carries the opponent as a count and the
    talon as a size, so no evaluator built on it can read hidden cards."""
    for seed in range(6):
        for state in walk(seed):
            player = state.current_player
            if player is None:
                continue
            node = node_for(state, player)
            view = observe(state, player)
            hidden = (
                set(state.hands[1 - player]) | set(state.talon)
            ) & view.unobserved_cards()
            assert not (cards_reachable_from(node) & hidden)
            for obj in reachable(node):
                assert not isinstance(obj, FORBIDDEN_TYPES)
            assert isinstance(node.opponent_cards, int)
            assert isinstance(node.talon_size, int)


def test_hidden_permutations_cannot_change_either_evaluator() -> None:
    """Two worlds the observer cannot tell apart must score identically."""
    from durakfish.game.state import add_cards

    shuffler = random.Random(4242)
    compared = 0
    for seed in range(12):
        for state in walk(seed):
            player = state.current_player
            if player is None or len(state.talon) < 3:
                continue
            view = observe(state, player)
            movable = view.unobserved_cards()
            opponent = [c for c in state.hands[1 - player] if c in movable]
            body = [c for c in state.talon[:-1] if c in movable]
            fixed_opp = [c for c in state.hands[1 - player] if c not in movable]
            fixed_body = [c for c in state.talon[:-1] if c not in movable]
            if not opponent or not body:
                continue
            pool = opponent + body
            shuffler.shuffle(pool)
            twin = state.replace(
                hands=tuple(
                    state.hands[p] if p == player
                    else add_cards((), fixed_opp + pool[: len(opponent)])
                    for p in range(2)
                ),
                talon=tuple(fixed_body + pool[len(opponent):]) + state.talon[-1:],
            )
            twin.validate()
            assert observe(twin, player) == view
            for evaluator in (BaselineEvaluator(), MechanisticEvaluator()):
                assert evaluator.evaluate(
                    node_for(state, player), 0
                ) == evaluator.evaluate(node_for(twin, player), 0)
            compared += 1
            break
    assert compared > 8


# ======================================================================
# Sanity guardrails, each justified by the frozen rules
# ======================================================================
def test_an_obvious_material_edge_is_scored_positively_when_the_talon_is_gone()\
        -> None:
    """With no talon the game is a race to shed cards, so holding fewer
    than the opponent must score positively."""
    state = make_state(
        hand0="6S", hand1="7S 9H 10H", trump=Suit.HEARTS, talon="", attacker=0
    )
    for evaluator in (BaselineEvaluator(), MechanisticEvaluator()):
        assert evaluator.evaluate(node_for(state, 0), 0) > 0
        assert evaluator.evaluate(node_for(state, 1), 0) < 0


def test_weights_are_inherited_from_the_baseline_unchanged() -> None:
    """No coefficient was tuned in Phase 7.3.10; only the structure moved."""
    assert MechanisticEvaluator().weights == EvaluationWeights()
    assert BaselineEvaluator().weights == EvaluationWeights()


def test_a_custom_weight_set_is_honoured() -> None:
    evaluator = MechanisticEvaluator(EvaluationWeights(own_trump=0.0))
    state = make_state(
        hand0="6H 7H", hand1="9S 10S", trump=Suit.HEARTS, talon="", attacker=0
    )
    node = node_for(state, 0)
    assert evaluator.evaluate(node, 0) != MechanisticEvaluator().evaluate(node, 0)


@pytest.mark.parametrize(
    "name",
    ["baseline", "mechanistic", "structural", "flexible",
     "structural_005", "flexible_020"],
)
def test_every_registered_evaluator_is_usable_through_the_configuration(
    name,
) -> None:
    from durakfish.ai.search import SearchConfig
    from durakfish.ai.world_search import search_world

    state = next(s for s in walk(3) if not s.is_over and s.current_player == 0)
    information = SearchInformation.from_view(observe(state, 0))
    world = DeterminizedWorld(
        player=0,
        opponent_hand=frozenset(state.hands[1]),
        talon=tuple(state.talon),
    )
    position = build_position(information, world)
    result = search_world(position, SearchConfig(depth=2, evaluator=name))
    assert result.move in position.legal_moves()


# ======================================================================
# The structural candidates (E2/E3)
# ======================================================================
CANDIDATES = ("structural", "flexible", "structural_005", "structural_020",
              "flexible_005", "flexible_020")


def test_the_whole_candidate_family_is_registered_without_moving_the_default(
) -> None:
    """Every candidate selectable by name; production default untouched."""
    from durakfish.ai.ismcts import ISMCTSConfig
    from durakfish.ai.search import SearchConfig

    for name in CANDIDATES:
        assert name in EVALUATORS
        assert get_evaluator(name) is not None
    assert SearchConfig(depth=1).evaluator == "baseline"
    assert ISMCTSConfig().evaluator == "baseline"
    assert EVALUATORS["baseline"] is BaselineEvaluator


def test_every_candidate_stays_far_inside_the_mate_margin() -> None:
    """A heuristic must never be mistakable for a proven result.

    Checked twice: analytically through the declared bound, and
    empirically over real play.
    """
    assert StructuralWeights().bound() < MATE_MARGIN
    for name in CANDIDATES:
        evaluator = get_evaluator(name)
        for seed in range(5):
            for state in walk(seed):
                player = state.current_player
                if player is None:
                    continue
                assert abs(evaluator.evaluate(node_for(state, player), 0)) < (
                    MATE_MARGIN
                )


@pytest.mark.parametrize("name", CANDIDATES)
def test_candidates_score_terminals_exactly(name) -> None:
    evaluator = get_evaluator(name)
    for ply in (0, 3):
        for outcome in (Outcome.WIN, Outcome.LOSS, Outcome.DRAW):
            node = terminal_node(outcome)
            assert evaluator.evaluate(node, ply) == terminal_score(outcome, ply)


@pytest.mark.parametrize("name", CANDIDATES)
def test_candidates_cannot_see_hidden_cards(name) -> None:
    """Two worlds the observer cannot tell apart must score identically."""
    from durakfish.game.state import add_cards

    evaluator = get_evaluator(name)
    shuffler = random.Random(99)
    compared = 0
    for seed in range(12):
        for state in walk(seed):
            player = state.current_player
            if player is None or len(state.talon) < 3:
                continue
            view = observe(state, player)
            movable = view.unobserved_cards()
            opponent = [c for c in state.hands[1 - player] if c in movable]
            body = [c for c in state.talon[:-1] if c in movable]
            fixed_opp = [c for c in state.hands[1 - player] if c not in movable]
            fixed_body = [c for c in state.talon[:-1] if c not in movable]
            if not opponent or not body:
                continue
            pool = opponent + body
            shuffler.shuffle(pool)
            twin = state.replace(
                hands=tuple(
                    state.hands[p] if p == player
                    else add_cards((), fixed_opp + pool[: len(opponent)])
                    for p in range(2)
                ),
                talon=tuple(fixed_body + pool[len(opponent):]) + state.talon[-1:],
            )
            twin.validate()
            assert evaluator.evaluate(node_for(state, player), 0) == (
                evaluator.evaluate(node_for(twin, player), 0)
            )
            compared += 1
            break
    assert compared > 8


def test_the_structural_candidate_ungates_the_obligation_term() -> None:
    """Phase 7.3.9 measured the baseline's guard as satisfied at 0 of 7,591
    ISMCTS leaves. E2 drops the guard deliberately, so an undefended slot
    must move its score at a node where the baseline's term is inert."""
    state = make_state(
        hand0="6S 7S 8S", hand1="9H 10H", trump=Suit.HEARTS, talon="", attacker=0
    )
    node = node_for(state, 0)
    assert not BaselineEvaluator._facing_attacks(node) or node.table == ()
    structural = get_evaluator("structural")
    bare = structural.evaluate(node, 0)
    pressured = structural.evaluate(
        dataclasses.replace(
            node,
            table=(SearchSlot(attacking_card=Card(Rank.SIX, Suit.CLUBS),
                              defending_card=None, defended=False),),
        ),
        0,
    )
    assert pressured < bare


def test_trump_quality_separates_a_high_trump_from_a_low_one() -> None:
    """The measured reason for E2's second term: the baseline scores the
    six and the ace of trumps identically, and these are not the same."""
    low = make_state(
        hand0="6H 7S", hand1="9S 10S", trump=Suit.HEARTS, talon="", attacker=0
    )
    high = make_state(
        hand0="AH 7S", hand1="9S 10S", trump=Suit.HEARTS, talon="", attacker=0
    )
    baseline, structural = BaselineEvaluator(), get_evaluator("structural")
    assert baseline.evaluate(node_for(low, 0), 0) == baseline.evaluate(
        node_for(high, 0), 0
    )
    assert structural.evaluate(node_for(high, 0), 0) > structural.evaluate(
        node_for(low, 0), 0
    )


def test_the_auxiliary_share_rule_scales_every_term_together() -> None:
    """One declared constant generates all coefficients, so the sweep has
    exactly one dimension. A drifting term would break that claim."""
    base = StructuralWeights()
    for share in (0.05, 0.20):
        scaled = base.scaled(share)
        factor = share / base.auxiliary_share
        assert scaled.material == base.material
        assert scaled.auxiliary_share == share
        for field in ("trump_quality", "obligation_pressure",
                      "terminal_proximity", "defense_flexibility",
                      "throw_in_options"):
            assert getattr(scaled, field) == pytest.approx(
                getattr(base, field) * factor
            )


def test_flexibility_terms_do_not_depend_on_whose_turn_it_is() -> None:
    """``self_moves()`` is empty off-turn, and Phase 7.3.9 measured that as
    the case at 7,295 of 7,591 ISMCTS leaves. E3's features are computed
    from the hand and the table instead, so they must survive a change of
    ``to_move``."""
    from durakfish.ai.searchnode import Actor

    evaluator = get_evaluator("flexible")
    checked = 0
    for seed in range(6):
        for state in walk(seed):
            player = state.current_player
            if player is None:
                continue
            node = node_for(state, player)
            flipped = dataclasses.replace(
                node,
                to_move=(Actor.OPPONENT if node.to_move is Actor.SELF
                         else Actor.SELF),
            )
            assert evaluator.evaluate(node, 0) == evaluator.evaluate(flipped, 0)
            checked += 1
    assert checked > 100


# ======================================================================
# Written because the mutation audit found these five gaps
# ======================================================================
def test_each_registered_name_resolves_to_its_own_distinct_class() -> None:
    """M26: ``"flexible"`` silently resolving to ``StructuralEvaluator``
    survived the suite, because the registration test only asserted that
    a name resolved to *something*. A silent fallback is exactly what
    this phase forbids, so the identity is pinned here."""
    assert type(get_evaluator("structural")) is StructuralEvaluator
    assert type(get_evaluator("flexible")) is FlexibleEvaluator
    assert type(get_evaluator("mechanistic")) is MechanisticEvaluator
    assert type(get_evaluator("baseline")) is BaselineEvaluator
    for name in ("structural_005", "structural_020"):
        assert isinstance(get_evaluator(name), StructuralEvaluator)
        assert not isinstance(get_evaluator(name), FlexibleEvaluator)
    for name in ("flexible_005", "flexible_020"):
        assert isinstance(get_evaluator(name), FlexibleEvaluator)


def test_the_declared_coefficients_are_the_ones_the_rule_generates() -> None:
    """M20: corrupting one coefficient from 2.50 to 7.50 survived.

    Every auxiliary weight is generated by
    ``AUXILIARY_SHARE * stdev(material) / stdev(feature)`` from the
    development-split spreads recorded in the class docstring. Recomputing
    them here means a hand-edited coefficient no longer passes silently.
    """
    weights = StructuralWeights()
    spreads = {
        "trump_quality": 0.5422,
        "obligation_pressure": 0.1891,
        "terminal_proximity": 0.1500,
        "defense_flexibility": 1.3094,
        "throw_in_options": 0.7773,
    }
    material_spread = 3.7477
    for field, spread in spreads.items():
        expected = weights.auxiliary_share * material_spread / spread
        assert getattr(weights, field) == pytest.approx(expected, abs=0.01), (
            f"{field} is not what the declared rule generates"
        )
    assert weights.material == 1.0
    assert weights.auxiliary_share == 0.10


def test_terminal_proximity_is_silent_while_the_talon_still_refills() -> None:
    """M23: running the term during refill survived.

    The feature's whole justification is that a small hand is not near
    exhaustion while the talon is about to top it up, so with cards left
    in the talon it must contribute exactly nothing.
    """
    evaluator = StructuralEvaluator()
    checked = 0
    for seed in range(8):
        for state in walk(seed):
            player = state.current_player
            if player is None or not state.talon:
                continue
            node = node_for(state, player)
            assert evaluator._terminal_proximity(node) == 0.0
            checked += 1
    assert checked > 100


def test_terminal_proximity_favours_the_shorter_hand_once_the_talon_is_gone(
) -> None:
    """M22: reversing its perspective survived. Pin the direction."""
    evaluator = StructuralEvaluator()
    ahead = make_state(
        hand0="6S", hand1="7S 9H 10H", trump=Suit.HEARTS, talon="", attacker=0
    )
    behind = make_state(
        hand0="6S 7S 9H", hand1="10H", trump=Suit.HEARTS, talon="", attacker=0
    )
    assert evaluator._terminal_proximity(node_for(ahead, 0)) > 0
    assert evaluator._terminal_proximity(node_for(behind, 0)) < 0
    assert evaluator.evaluate(node_for(ahead, 0), 0) > evaluator.evaluate(
        node_for(behind, 0), 0
    )


def test_the_flexibility_terms_actually_contribute_to_the_score() -> None:
    """M24: making E3's flexibility terms inert survived, which would have
    reduced the candidate to E2 while still calling itself E3.

    Both branches are exercised: throw-in options while attacking, and
    defensive answers while defending.
    """
    structural, flexible = StructuralEvaluator(), FlexibleEvaluator()
    attacking = make_state(
        hand0="6S 6C 6D", hand1="9H 10H", trump=Suit.HEARTS, talon="",
        attacker=0, table=(("6H", None),), phase=Phase.DEFENSE,
    )
    differing = 0
    for seed in range(10):
        for state in walk(seed):
            player = state.current_player
            if player is None:
                continue
            node = node_for(state, player)
            if flexible.evaluate(node, 0) != structural.evaluate(node, 0):
                differing += 1
    assert differing > 50, (
        "E3 never differs from E2, so its flexibility terms are inert"
    )
    # Both branches must be pinned separately. A mutation that disabled
    # only the attacker branch survived the first version of this test,
    # because the defender branch alone kept E3 different from E2.
    node = node_for(attacking, 0)
    assert node.searcher_is_attacker
    assert flexible._throw_in_options(node) > 0
    assert flexible.evaluate(node, 0) == pytest.approx(
        structural.evaluate(node, 0)
        + flexible.structural.throw_in_options * flexible._throw_in_options(node)
    )

    defending = node_for(attacking, 1)
    assert not defending.searcher_is_attacker
    assert flexible.evaluate(defending, 0) == pytest.approx(
        structural.evaluate(defending, 0)
        + flexible.structural.defense_flexibility
        * flexible._defense_flexibility(defending)
    )
    assert flexible._defense_flexibility(defending) > 0


# ======================================================================
# Phase 7.3.12 — incentive repairs
# ======================================================================
REPAIRS = ("flex_r1", "flex_r2", "flex_r3", "flex_r4")


def shedding_pairs(seeds=range(12)):
    """Real (before, after) node pairs where the mover played a card.

    Yielded from legal play rather than constructed, so an incentive test
    written against them cannot be satisfied by a contrived position.
    """
    for seed in seeds:
        for state in walk(seed):
            player = state.current_player
            if player is None:
                continue
            before = node_for(state, player)
            for move in get_legal_moves(state):
                after_state = apply_move(state, move)
                if after_state.current_player is None:
                    continue
                if len(after_state.hands[player]) >= len(state.hands[player]):
                    continue
                yield before, node_for(after_state, player)


def test_the_repairs_are_registered_and_distinct() -> None:
    """Each repair must resolve to its own class; no silent fallback."""
    from durakfish.ai.evaluation import (
        RepairedBoundedEvaluator,
        RepairedEvaluator,
        RepairedObligationEvaluator,
        RepairedThrowInEvaluator,
    )
    from durakfish.ai.ismcts import ISMCTSConfig
    from durakfish.ai.search import SearchConfig

    expected = {
        "flex_r1": RepairedThrowInEvaluator,
        "flex_r2": RepairedObligationEvaluator,
        "flex_r3": RepairedEvaluator,
        "flex_r4": RepairedBoundedEvaluator,
    }
    for name, cls in expected.items():
        assert type(get_evaluator(name)) is cls
    assert SearchConfig(depth=1).evaluator == "baseline"
    assert ISMCTSConfig().evaluator == "baseline"
    assert EVALUATORS["baseline"] is BaselineEvaluator


def test_an_attacker_is_never_charged_for_the_attack_it_makes() -> None:
    """The Phase 7.3.12 defect, stated as a property.

    An undefended card on the table is the *defender's* obligation. The
    unrepaired term charges it to whoever is searching, which penalises
    the attacker for attacking — measured at 99.5% of attacker shedding
    moves. Every repaired obligation must read exactly zero from the
    attacking seat.
    """
    for name in ("flex_r2", "flex_r3", "flex_r4"):
        evaluator = get_evaluator(name)
        checked = 0
        for seed in range(8):
            for state in walk(seed):
                player = state.current_player
                if player is None:
                    continue
                node = node_for(state, player)
                if not node.searcher_is_attacker:
                    continue
                assert evaluator._obligation_pressure(node) == 0.0, name
                checked += 1
        assert checked > 100, f"{name}: too few attacking positions checked"


def test_the_unrepaired_obligation_really_does_charge_the_attacker() -> None:
    """The control for the test above: without it, that test could pass
    against an evaluator where the term is simply dead everywhere."""
    evaluator = StructuralEvaluator()
    charged = 0
    for seed in range(8):
        for state in walk(seed):
            player = state.current_player
            if player is None:
                continue
            node = node_for(state, player)
            if node.searcher_is_attacker:
                charged += evaluator._obligation_pressure(node) > 0
    assert charged > 50, "the defect this phase repairs was not reproduced"


def test_a_defender_obligation_is_still_scored_after_repair() -> None:
    """The repair must remove the attacker charge without deleting the
    feature: a defender facing open attacks must still be penalised."""
    for name in ("flex_r2", "flex_r3", "flex_r4"):
        evaluator = get_evaluator(name)
        live = 0
        for seed in range(10):
            for state in walk(seed):
                player = state.current_player
                if player is None:
                    continue
                node = node_for(state, player)
                if node.searcher_is_attacker:
                    continue
                if any(not slot.defended for slot in node.table):
                    live += evaluator._obligation_pressure(node) > 0
        assert live > 20, f"{name}: obligation term is inert for the defender"


def test_the_repaired_throw_in_ignores_duplicate_cards_of_one_rank() -> None:
    """R1's semantics: continuation is a property of *ranks*, not of how
    many duplicate cards are hoarded against them.

    Holding three sixes is one rank of continuation, not three.
    """
    evaluator = get_evaluator("flex_r1")
    original = FlexibleEvaluator()
    node = dataclasses.replace(
        node_for(
            make_state(
                hand0="6S 6C 6D 9H", hand1="10H 10S", trump=Suit.HEARTS,
                talon="", attacker=0, table=(("6H", None),),
                phase=Phase.DEFENSE,
            ),
            0,
        ),
    )
    assert original._throw_in_options(node) == 3.0
    assert evaluator._throw_in_options(node) == 1.0


def test_playing_a_duplicate_rank_costs_the_repaired_feature_nothing() -> None:
    """The incentive property R1 exists to establish."""
    evaluator = get_evaluator("flex_r1")
    full = node_for(
        make_state(
            hand0="6S 6C 9H", hand1="10H 10S", trump=Suit.HEARTS, talon="",
            attacker=0, table=(("6H", None),), phase=Phase.DEFENSE,
        ),
        0,
    )
    played = dataclasses.replace(full, hand=tuple(full.hand[1:]))
    assert len(played.hand) == len(full.hand) - 1
    assert evaluator._throw_in_options(played) == evaluator._throw_in_options(
        full
    )


@pytest.mark.parametrize("name", REPAIRS)
def test_repairs_keep_the_frozen_contract(name) -> None:
    """Terminals exact, values bounded, deterministic, hidden-safe."""
    evaluator = get_evaluator(name)
    for ply in (0, 3):
        for outcome in (Outcome.WIN, Outcome.LOSS, Outcome.DRAW):
            node = terminal_node(outcome)
            assert evaluator.evaluate(node, ply) == terminal_score(outcome, ply)
    for seed in range(5):
        for state in walk(seed):
            player = state.current_player
            if player is None:
                continue
            node = node_for(state, player)
            values = {evaluator.evaluate(node, 0) for _ in range(3)}
            assert len(values) == 1
            assert abs(values.pop()) < MATE_MARGIN


@pytest.mark.parametrize("name", REPAIRS)
def test_repairs_cannot_see_hidden_cards(name) -> None:
    """Two worlds the observer cannot tell apart must score identically."""
    from durakfish.game.state import add_cards

    evaluator = get_evaluator(name)
    shuffler = random.Random(31)
    compared = 0
    for seed in range(12):
        for state in walk(seed):
            player = state.current_player
            if player is None or len(state.talon) < 3:
                continue
            view = observe(state, player)
            movable = view.unobserved_cards()
            opponent = [c for c in state.hands[1 - player] if c in movable]
            body = [c for c in state.talon[:-1] if c in movable]
            fixed_opp = [c for c in state.hands[1 - player] if c not in movable]
            fixed_body = [c for c in state.talon[:-1] if c not in movable]
            if not opponent or not body:
                continue
            pool = opponent + body
            shuffler.shuffle(pool)
            twin = state.replace(
                hands=tuple(
                    state.hands[p] if p == player
                    else add_cards((), fixed_opp + pool[: len(opponent)])
                    for p in range(2)
                ),
                talon=tuple(fixed_body + pool[len(opponent):]) + state.talon[-1:],
            )
            twin.validate()
            assert evaluator.evaluate(node_for(state, player), 0) == (
                evaluator.evaluate(node_for(twin, player), 0)
            )
            compared += 1
            break
    assert compared > 8


def test_the_obligation_repair_measurably_reduces_the_hoarding_incentive(
) -> None:
    """The end-to-end behavioural property, on real shedding moves.

    Not a threshold invented after the fact: the unrepaired candidate
    prefers the pre-shed position in 38.1% of shedding moves and the
    seat-gated repair in 5.3%, so requiring the repair to stay under half
    the original rate is a wide margin around a measured gap.
    """
    original = FlexibleEvaluator()
    repaired = get_evaluator("flex_r3")
    pairs = list(shedding_pairs(range(8)))
    assert len(pairs) > 500

    def fell(evaluator) -> float:
        drops = sum(
            1 for before, after in pairs
            if evaluator.evaluate(after, 0) - evaluator.evaluate(before, 0)
            < -1e-9
        )
        return drops / len(pairs)

    original_rate, repaired_rate = fell(original), fell(repaired)
    assert original_rate > 0.25, "the defect no longer reproduces"
    assert repaired_rate < original_rate / 2


def test_the_bounded_obligation_is_independent_of_hand_size() -> None:
    """R4's whole justification, pinned.

    The divisor is ``attack_limit`` precisely because it cannot be changed
    by playing a card. If it were the hand size again — the divisor R2
    rejected — the feature would move when only the hand moved, which is
    the defect this phase exists to remove.

    Written because two normalisation mutations survived the first audit:
    nothing asserted what R4 divides by.
    """
    evaluator = get_evaluator("flex_r4")
    base = node_for(
        make_state(
            hand0="6S 6C 9H", hand1="10H 10S", trump=Suit.HEARTS, talon="",
            attacker=1, table=(("10D", None),), phase=Phase.DEFENSE,
        ),
        0,
    )
    assert not base.searcher_is_attacker
    assert evaluator._obligation_pressure(base) > 0
    smaller = dataclasses.replace(base, hand=base.hand[:-1])
    assert smaller.own_cards == base.own_cards - 1
    assert evaluator._obligation_pressure(smaller) == (
        evaluator._obligation_pressure(base)
    ), "R4's obligation moved when only the hand changed"


def test_the_bounded_obligation_is_actually_bounded() -> None:
    """R4 must lie in [0, 1] and must be strictly smaller than R3's raw
    count whenever the bout allows more than one attack — otherwise the
    divisor has been dropped and R4 has silently become R3."""
    bounded, raw = get_evaluator("flex_r4"), get_evaluator("flex_r3")
    differed = 0
    for seed in range(10):
        for state in walk(seed):
            player = state.current_player
            if player is None:
                continue
            node = node_for(state, player)
            value = bounded._obligation_pressure(node)
            assert 0.0 <= value <= 1.0
            reference = raw._obligation_pressure(node)
            if reference > 0 and node.attack_limit > 1:
                assert value < reference
                differed += 1
    assert differed > 20, "R4 was never distinguishable from R3 here"


# ======================================================================
# Phase 7.3.13 — trump_quality characterisation
# ======================================================================
def trump_shedding_transitions(seeds=range(10)):
    """Reachable legal transitions in which the mover played a trump."""
    for seed in seeds:
        for state in walk(seed):
            player = state.current_player
            if player is None:
                continue
            trump = state.trump_suit
            before_trumps = {
                c for c in state.hands[player] if c.suit is trump
            }
            before = node_for(state, player)
            for move in get_legal_moves(state):
                after_state = apply_move(state, move)
                if after_state.current_player is None:
                    continue
                if len(after_state.hands[player]) >= len(state.hands[player]):
                    continue
                after_trumps = {
                    c for c in after_state.hands[player] if c.suit is trump
                }
                yield (
                    before,
                    node_for(after_state, player),
                    before_trumps - after_trumps,
                )


def test_trump_quality_changes_only_when_a_trump_leaves_the_hand() -> None:
    """Phase 7.3.13: the feature has no spurious sensitivity.

    Measured over reachable shedding transitions, ``trump_quality`` is
    unchanged on every move that plays a non-trump — 0.0% of 5,540 in
    the audit. This pins that: it is what separates ``trump_quality``
    from Defect A, where the feature moved for reasons unrelated to the
    capability it claimed to measure.
    """
    evaluator = StructuralEvaluator()
    checked = 0
    for before, after, played in trump_shedding_transitions():
        if played:
            continue
        assert evaluator._trump_quality(after) == evaluator._trump_quality(
            before
        )
        checked += 1
    assert checked > 500


def test_trump_quality_falls_by_exactly_the_rank_of_the_trump_played() -> None:
    """The exact identity, reproduced on reachable positions.

    In the audit this held in 1,640 of 1,640 cases. The feature is not
    approximating a strategic quantity — it reports precisely the value
    of the card that was spent, which is why Phase 7.3.13 classifies it
    as a priced trade-off rather than an accounting artifact.
    """
    evaluator = StructuralEvaluator()
    checked = 0
    for before, after, played in trump_shedding_transitions():
        if not played:
            continue
        expected = sum((card.rank - 6) / 8.0 for card in played)
        actual = evaluator._trump_quality(before) - evaluator._trump_quality(
            after
        )
        assert actual == pytest.approx(expected, abs=1e-12)
        checked += 1
    assert checked > 100


def test_each_trump_rank_is_unique_so_no_duplication_artifact_exists() -> None:
    """Why ``trump_quality`` cannot have Defect A's shape.

    One suit is trump, so each rank appears at most once among a hand's
    trumps. Every trump played is therefore the last of its rank, and no
    'held three, played one, lost a third of the value' artifact — the
    thing that made ``throw_in_options`` defective — is constructible.
    """
    for _before, _after, played in trump_shedding_transitions(range(6)):
        ranks = [card.rank for card in played]
        assert len(ranks) == len(set(ranks))
    for seed in range(6):
        for state in walk(seed):
            for hand in state.hands:
                trumps = [c for c in hand if c.suit is state.trump_suit]
                assert len({c.rank for c in trumps}) == len(trumps)
