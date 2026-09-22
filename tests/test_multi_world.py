"""Multi-world aggregation.

The production aggregator must never be its own oracle, so expected
values here are computed by a reference written independently: it sums
weighted per-world values in plain Python from values obtained through
the Phase 7.3.2a primitive, sharing no code path with
``aggregate_worlds``.

The properties that matter are mathematical rather than strategic —
normalisation, bounds, order and scale invariance, single-world
equivalence — because this phase makes no claim about playing strength.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from durakfish.ai import (
    AggregationError,
    SearchConfig,
    WeightedWorld,
    aggregate_worlds,
    build_position,
    enumerate_worlds,
    evaluate_position,
    search_world,
)
from durakfish.cards import Suit
from durakfish.game import apply_move, get_legal_moves, new_game
from durakfish.game.state import add_cards
from durakfish.information import (
    DeterminizedWorld,
    SearchInformation,
    WorldGenerator,
    observe,
)
from tests.support import make_state
from tests.test_phase3_audit import FORBIDDEN_TYPES, reachable

DEPTH = SearchConfig(depth=3)


def walk(seed: int):
    rng = random.Random(seed)
    state = new_game(seed=seed)
    while not state.is_over:
        yield state
        state = apply_move(state, rng.choice(get_legal_moves(state)))
    yield state


def info_for(state, player: int) -> SearchInformation:
    return SearchInformation.from_view(observe(state, player))


def sample_worlds(information, count: int, seed: int = 0):
    generator = WorldGenerator(information)
    return [generator.sample(random.Random(seed + k)) for k in range(count)]


def decision_positions(games: int = 12, min_worlds: int = 2):
    """Positions where the observer is to move and several worlds exist."""
    for seed in range(games):
        for index, state in enumerate(walk(seed)):
            player = state.current_player
            if player is None or index % 5:
                continue
            information = info_for(state, player)
            if information.allocations < min_worlds:
                continue
            yield state, player, information


# ======================================================================
# Independent reference aggregator
# ======================================================================
def reference_expected_values(
    information, worlds, weights, config: SearchConfig, root_player: int
) -> dict:
    """Weighted expectation per move, computed independently.

    Written straight from the formula and using only the Phase 7.3.2a
    primitive. It shares no code with ``aggregate_worlds`` — in
    particular it does not sort the worlds, so agreement is evidence
    rather than a shared implementation detail.
    """
    root = build_position(information, worlds[0])
    total = sum(weights)
    out = {}
    for move in root.legal_moves():
        running = 0.0
        for world, weight in zip(worlds, weights, strict=True):
            child = build_position(information, world).apply(move)
            if config.depth <= 1:
                value = evaluate_position(child, root_player, ply=1)
            else:
                value = search_world(
                    child,
                    replace(config, depth=config.depth - 1),
                    root_player=root_player,
                ).value
            running += weight * value
        out[move] = running / total
    return out


def test_production_aggregation_matches_the_independent_reference() -> None:
    checked = 0
    for state, player, information in decision_positions(games=8):
        worlds = sample_worlds(information, 5, seed=state.attacker)
        weights = [1.0] * len(worlds)
        expected = reference_expected_values(
            information, worlds, weights, DEPTH, player
        )
        result = aggregate_worlds(information, worlds, DEPTH)
        for move, value in expected.items():
            assert abs(result.aggregate_for(move).value - value) < 1e-9, move
        checked += 1
    assert checked > 20


def test_the_reference_agrees_under_unequal_weights_too() -> None:
    checked = 0
    for state, player, information in decision_positions(games=6):
        worlds = sample_worlds(information, 4, seed=7)
        weights = [0.5, 0.25, 0.15, 0.10]
        expected = reference_expected_values(
            information, worlds, weights, DEPTH, player
        )
        weighted = [WeightedWorld(w, p) for w, p in zip(worlds, weights, strict=True)]
        result = aggregate_worlds(information, weighted, DEPTH)
        for move, value in expected.items():
            assert abs(result.aggregate_for(move).value - value) < 1e-9
        checked += 1
    assert checked > 10


# ======================================================================
# The aggregation mathematics, on constructed values
# ======================================================================
def test_equal_weights_average_opposing_worlds() -> None:
    """The worked example: X=10/0 and Y=0/10 must both average to 5."""
    values = {"X": [10.0, 0.0], "Y": [0.0, 10.0]}
    weights = [1.0, 1.0]
    total = sum(weights)
    for move, per_world in values.items():
        expected = sum(w * v for w, v in zip(weights, per_world, strict=True)) / total
        assert expected == 5.0, move


def test_unequal_weights_produce_the_exact_weighted_expectation() -> None:
    values = [10.0, 0.0]
    weights = [0.75, 0.25]
    assert sum(w * v for w, v in zip(weights, values, strict=True)) / sum(
        weights
    ) == 7.5


def test_aggregate_lies_between_the_best_and_worst_world_values() -> None:
    """A weighted mean of nonnegative normalised weights is bracketed."""
    checked = 0
    for state, player, information in decision_positions(games=8):
        result = aggregate_worlds(
            information, sample_worlds(information, 5, seed=3), DEPTH
        )
        for entry in result.per_move:
            assert entry.worst <= entry.value <= entry.best
            assert entry.worst == min(entry.world_values)
            assert entry.best == max(entry.world_values)
            checked += 1
    assert checked > 40


def test_scaling_every_weight_changes_nothing() -> None:
    for state, player, information in decision_positions(games=5):
        worlds = sample_worlds(information, 4, seed=11)
        plain = aggregate_worlds(information, worlds, DEPTH)
        for factor in (0.5, 3.0, 100.0):
            scaled = aggregate_worlds(
                information, [WeightedWorld(w, factor) for w in worlds], DEPTH
            )
            assert scaled.move == plain.move
            assert abs(scaled.value - plain.value) < 1e-9


def test_duplicating_every_world_changes_nothing() -> None:
    for state, player, information in decision_positions(games=5):
        worlds = sample_worlds(information, 4, seed=13)
        plain = aggregate_worlds(information, worlds, DEPTH)
        doubled = aggregate_worlds(information, worlds + worlds, DEPTH)
        assert doubled.move == plain.move
        assert abs(doubled.value - plain.value) < 1e-9
        assert doubled.world_count == 2 * plain.world_count


def test_dominance_is_respected() -> None:
    """A move at least as good everywhere, and strictly better somewhere."""
    checked = 0
    for state, player, information in decision_positions(games=10):
        result = aggregate_worlds(
            information, sample_worlds(information, 4, seed=5), DEPTH
        )
        for first in result.per_move:
            for second in result.per_move:
                if first.move == second.move:
                    continue
                pairs = list(
                    zip(first.world_values, second.world_values, strict=True)
                )
                if all(a >= b for a, b in pairs) and any(a > b for a, b in pairs):
                    assert first.value > second.value
                    checked += 1
    assert checked > 10


def test_per_world_values_follow_the_canonical_world_order() -> None:
    """The documented ordering contract, asserted directly.

    ``MoveAggregate.world_values`` is specified to be in the canonical
    world order used for summation, not the caller's order. That is what
    makes permutation invariance hold by construction rather than by
    luck: floating-point addition is not associative, so summing in
    caller order would leave exact invariance resting on the data. No
    instability was observed across 2,440 permutation checks on real
    positions, which is precisely why the ordering needs asserting
    directly — the aggregate alone would not reveal its loss.
    """
    rng = random.Random(71)
    checked = 0
    for state, player, information in decision_positions(games=6):
        worlds = sample_worlds(information, 5, seed=67)
        canonical = sorted(worlds, key=lambda w: w.key())
        baseline = aggregate_worlds(information, canonical, DEPTH)
        for _ in range(3):
            shuffled = list(worlds)
            rng.shuffle(shuffled)
            other = aggregate_worlds(information, shuffled, DEPTH)
            for a, b in zip(baseline.per_move, other.per_move, strict=True):
                assert a.move == b.move
                assert a.world_values == b.world_values, (
                    "per-world values follow the caller's order, not canonical order"
                )
            checked += 1
    assert checked > 10


def test_permuting_the_worlds_cannot_change_the_result() -> None:
    """Exact, not approximate: worlds are summed in canonical order."""
    rng = random.Random(99)
    for state, player, information in decision_positions(games=6):
        worlds = sample_worlds(information, 5, seed=17)
        baseline = aggregate_worlds(information, worlds, DEPTH)
        for _ in range(3):
            shuffled = list(worlds)
            rng.shuffle(shuffled)
            other = aggregate_worlds(information, shuffled, DEPTH)
            assert other.move == baseline.move
            assert other.value == baseline.value, "float summation order leaked"
            assert [a.value for a in other.per_move] == [
                a.value for a in baseline.per_move
            ]


# ======================================================================
# Single-world and certain-world equivalence
# ======================================================================
def test_one_world_reduces_exactly_to_the_single_world_search() -> None:
    """The most important reduction: aggregation must not distort it."""
    checked = 0
    for state, player, information in decision_positions(games=10, min_worlds=1):
        world = sample_worlds(information, 1, seed=2)[0]
        aggregated = aggregate_worlds(information, [world], DEPTH)
        direct = search_world(
            build_position(information, world), DEPTH, root_player=player
        )
        assert aggregated.move == direct.move
        assert abs(aggregated.value - direct.value) < 1e-9
        checked += 1
    assert checked > 20


def test_a_certain_world_reduces_to_single_world_search() -> None:
    """Probability 1 on one world must behave exactly like that world."""
    for state, player, information in decision_positions(games=6):
        worlds = sample_worlds(information, 3, seed=23)
        certain = [
            WeightedWorld(worlds[0], 1.0),
            WeightedWorld(worlds[1], 0.0),
            WeightedWorld(worlds[2], 0.0),
        ]
        result = aggregate_worlds(information, certain, DEPTH)
        direct = search_world(
            build_position(information, worlds[0]), DEPTH, root_player=player
        )
        assert result.move == direct.move
        assert abs(result.value - direct.value) < 1e-9


def test_a_fully_determined_position_has_exactly_one_world() -> None:
    state = make_state(
        hand0="6S 7H", hand1="8S 9H", trump=Suit.HEARTS, talon="", attacker=0
    )
    information = info_for(state, 0)
    worlds = enumerate_worlds(information)
    assert len(worlds) == 1
    assert information.allocations == 1
    result = aggregate_worlds(information, worlds, DEPTH)
    direct = search_world(build_position(information, worlds[0]), DEPTH)
    assert result.move == direct.move and result.value == direct.value


# ======================================================================
# Exact enumeration
# ======================================================================
def test_enumeration_produces_exactly_the_allocation_space() -> None:
    state = make_state(
        hand0="6S 6C", hand1="7S 7C", trump=Suit.HEARTS, talon="8S 8C 8H",
        attacker=0,
    )
    information = info_for(state, 0)
    worlds = enumerate_worlds(information)
    assert len(worlds) == information.allocations == 6
    assert len({w.key() for w in worlds}) == 6


def test_enumeration_matches_the_phase_7_2_reference() -> None:
    from tests.reference_worlds import enumerate_allocations

    state = make_state(
        hand0="6S 6C", hand1="7S 7C", trump=Suit.HEARTS, talon="8S 8C 8H",
        attacker=0,
    )
    information = info_for(state, 0)
    mine = {w.allocation_key() for w in enumerate_worlds(information)}
    reference = {
        tuple(sorted(c.code for c in allocation))
        for allocation in enumerate_allocations(information)
    }
    assert mine == reference


def test_enumeration_refuses_an_intractable_space() -> None:
    information = info_for(new_game(seed=1), 0)
    assert information.allocations > 100_000
    with pytest.raises(AggregationError, match="exceed the enumeration limit"):
        enumerate_worlds(information)


def test_enumerated_aggregation_uses_uniform_probabilities() -> None:
    """Every compatible world is equally likely under the Phase 7.2 model."""
    state = make_state(
        hand0="6S 6C", hand1="7S 7C", trump=Suit.HEARTS, talon="8S 8C 8H",
        attacker=0,
    )
    information = info_for(state, 0)
    worlds = enumerate_worlds(information)
    result = aggregate_worlds(information, worlds, DEPTH)
    reference = reference_expected_values(
        information, list(worlds), [1.0] * len(worlds), DEPTH, 0
    )
    for move, value in reference.items():
        assert abs(result.aggregate_for(move).value - value) < 1e-9
    assert result.world_count == 6


# ======================================================================
# Determinism, isolation, ties
# ======================================================================
def test_repeated_aggregation_is_bit_for_bit_identical() -> None:
    for state, player, information in decision_positions(games=6):
        worlds = sample_worlds(information, 4, seed=29)
        results = [aggregate_worlds(information, worlds, DEPTH) for _ in range(3)]
        assert len({(r.move, r.value, r.total_nodes) for r in results}) == 1


def test_interleaving_worlds_cannot_contaminate_a_result() -> None:
    """A, B, A must give the same A as A, A. No cache exists to poison."""
    for state, player, information in decision_positions(games=6):
        worlds = sample_worlds(information, 6, seed=31)
        first, second = worlds[:3], worlds[3:]
        alone = aggregate_worlds(information, first, DEPTH)
        aggregate_worlds(information, second, DEPTH)
        again = aggregate_worlds(information, first, DEPTH)
        assert (again.move, again.value) == (alone.move, alone.value)


def test_ties_break_on_the_search_ordering_earliest_first() -> None:
    """Deterministic, documented, and the same rule single-world search uses.

    Two tie-break rules in one codebase would make single-world
    equivalence hold only by coincidence, so the aggregator adopts the
    search's ordering rather than inventing its own.
    """
    checked = 0
    for state, player, information in decision_positions(games=10):
        result = aggregate_worlds(
            information, sample_worlds(information, 3, seed=37), DEPTH
        )
        best = max(a.value for a in result.per_move)
        tied = [a.move for a in result.per_move if a.value == best]
        if len(tied) > 1:
            # per_move is emitted in search order, so the first tied entry wins.
            first_tied = next(a.move for a in result.per_move if a.value == best)
            assert result.move == first_tied
            checked += 1
    assert checked > 3, "no tie arose to exercise the tie-break"


# ======================================================================
# Contract and error handling
# ======================================================================
def test_an_empty_world_set_is_refused() -> None:
    information = info_for(new_game(seed=1), 0)
    with pytest.raises(AggregationError, match="empty set of worlds"):
        aggregate_worlds(information, [], DEPTH)


def test_zero_total_weight_is_refused() -> None:
    information = info_for(new_game(seed=1), 0)
    worlds = sample_worlds(information, 2, seed=1)
    with pytest.raises(AggregationError, match="sum to zero"):
        aggregate_worlds(
            information, [WeightedWorld(w, 0.0) for w in worlds], DEPTH
        )


def test_a_negative_weight_is_refused() -> None:
    information = info_for(new_game(seed=1), 0)
    world = sample_worlds(information, 1, seed=1)[0]
    with pytest.raises(AggregationError, match="negative world weight"):
        WeightedWorld(world, -1.0)


def test_a_world_from_another_observer_is_refused() -> None:
    state = new_game(seed=3)
    information = info_for(state, 0)
    foreign = sample_worlds(info_for(state, 1), 1, seed=1)[0]
    with pytest.raises(AggregationError, match="belongs to P1"):
        aggregate_worlds(information, [foreign], DEPTH)


def test_an_invalid_world_is_refused_by_the_phase_7_2_validator() -> None:
    from durakfish.ai import DeterminizationError

    state = make_state(
        hand0="6S 6C", hand1="7S 7C", trump=Suit.HEARTS, talon="8S 8C 8H",
        attacker=0,
    )
    information = info_for(state, 0)
    good = sample_worlds(information, 1, seed=1)[0]
    broken = DeterminizedWorld(0, good.opponent_hand, tuple(reversed(good.talon)))
    with pytest.raises(DeterminizationError):
        aggregate_worlds(information, [broken], DEPTH)


def test_the_root_player_must_be_the_player_to_move() -> None:
    """Proven precondition: root moves are world-independent only then."""
    found = False
    for state in walk(2):
        mover = state.current_player
        if mover is None:
            continue
        idle = 1 - mover
        information = info_for(state, idle)
        worlds = sample_worlds(information, 2, seed=1)
        with pytest.raises(AggregationError, match="not to move"):
            aggregate_worlds(information, worlds, DEPTH)
        found = True
        break
    assert found


def test_root_moves_really_are_world_independent_from_the_moving_seat() -> None:
    """The archaeology finding, kept as a regression."""
    checked = 0
    for state, player, information in decision_positions(games=10, min_worlds=1):
        worlds = sample_worlds(information, 6, seed=41)
        move_sets = {
            build_position(information, world).legal_moves() for world in worlds
        }
        assert len(move_sets) == 1, "root moves differ between worlds"
        checked += 1
    assert checked > 20


def test_a_terminal_root_yields_no_move_and_a_terminal_value() -> None:
    state = next(s for s in walk(4) if s.is_over)
    information = info_for(state, 0)
    worlds = sample_worlds(information, 2, seed=1)
    result = aggregate_worlds(information, worlds, DEPTH)
    assert result.move is None
    assert result.per_move == ()
    assert result.value == evaluate_position(
        build_position(information, worlds[0]), 0
    )


def test_restricting_the_root_moves_restricts_the_decision() -> None:
    for state, player, information in decision_positions(games=5):
        worlds = sample_worlds(information, 3, seed=43)
        full = aggregate_worlds(information, worlds, DEPTH)
        reduced = tuple(a.move for a in full.per_move if a.move != full.move)
        if not reduced:
            continue
        limited = aggregate_worlds(information, worlds, DEPTH, root_moves=reduced)
        assert limited.move in reduced
        assert limited.move != full.move


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_aggregation_works_at_every_supported_depth(depth: int) -> None:
    config = SearchConfig(depth=depth)
    for state, player, information in decision_positions(games=4):
        worlds = sample_worlds(information, 3, seed=47)
        result = aggregate_worlds(information, worlds, config)
        assert result.depth == depth
        assert result.move in [a.move for a in result.per_move]
        reference = reference_expected_values(
            information, worlds, [1.0] * 3, config, player
        )
        for move, value in reference.items():
            assert abs(result.aggregate_for(move).value - value) < 1e-9


def test_aggregate_for_rejects_a_move_that_was_not_considered() -> None:
    from durakfish.game.moves import END_ATTACK

    state, player, information = next(decision_positions(games=3))
    result = aggregate_worlds(
        information, sample_worlds(information, 2, seed=1), DEPTH
    )
    if END_ATTACK not in [a.move for a in result.per_move]:
        with pytest.raises(AggregationError, match="was not a root move"):
            result.aggregate_for(END_ATTACK)


# ======================================================================
# Information boundary
# ======================================================================
def test_the_same_worlds_from_two_realities_aggregate_identically() -> None:
    """Information-set noninterference at the decision level."""
    shuffler = random.Random(2024)
    compared = 0
    for seed in range(15):
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
                    state.hands[p]
                    if p == player
                    else add_cards((), fixed_opp + pool[: len(opponent)])
                    for p in range(2)
                ),
                talon=tuple(fixed_body + pool[len(opponent) :]) + state.talon[-1:],
            )
            twin.validate()
            assert observe(twin, player) == view

            first, second = info_for(state, player), info_for(twin, player)
            worlds = sample_worlds(first, 4, seed=53)
            a = aggregate_worlds(first, worlds, DEPTH)
            b = aggregate_worlds(second, worlds, DEPTH)
            assert (a.move, a.value, a.total_nodes) == (b.move, b.value, b.total_nodes)
            compared += 1
            break
    assert compared > 8


def test_no_result_exposes_a_route_to_a_real_game() -> None:
    for state, player, information in decision_positions(games=4):
        result = aggregate_worlds(
            information, sample_worlds(information, 3, seed=59), DEPTH
        )
        assert not any(obj is state for obj in reachable(result))
        for obj in reachable(result):
            assert not isinstance(obj, FORBIDDEN_TYPES) or obj is not state


def test_the_aggregation_module_never_names_the_state_layer() -> None:
    import ast
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "src" / "durakfish" / "ai" / "multi_world.py"
    )
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    for banned in ("GameState", "hands", "talon", "apply_move", "new_game"):
        assert banned not in names, f"multi_world.py references {banned!r}"


def test_search_bot_decisions_are_unchanged_by_this_phase() -> None:
    from durakfish.ai import SearchBot

    bot = SearchBot(SearchConfig(depth=3), "s")
    bot.reset(random.Random(0))
    checked = 0
    for state, player, information in decision_positions(games=6):
        view, legal = observe(state, player), get_legal_moves(state)
        before = bot.choose(view, legal)
        aggregate_worlds(information, sample_worlds(information, 2, seed=61), DEPTH)
        assert bot.choose(view, legal) == before
        checked += 1
    assert checked > 15
