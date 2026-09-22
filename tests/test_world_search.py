"""Perfect-information search over one determinized world.

The primitive this phase exists to build, and the reason it was needed:
routing a determinized position through ``observe()`` re-redacts it, so
Phase 5's search returns the *same* answer for different worlds. A search
that cannot tell worlds apart is useless to aggregate over, so that
non-degeneracy is tested first and explicitly.

Correctness rests on an oracle the repository already had:
``perfect_information_value`` in ``tests/test_abstraction_audit.py``,
which solves the real game exhaustively by a different algorithm. It is
the reference; production code never calls it.
"""

from __future__ import annotations

import random
import sys

import pytest

from durakfish.ai import (
    BaselineEvaluator,
    DeterminizedPosition,
    SearchConfig,
    alpha_beta_world,
    build_position,
    evaluate_position,
    minimax_world,
    search_world,
)
from durakfish.ai.evaluation import MATE_MARGIN, TERMINAL_DRAW
from durakfish.cards import Suit
from durakfish.game import apply_move, get_legal_moves, new_game
from durakfish.information import (
    DeterminizedWorld,
    SearchInformation,
    WorldGenerator,
    observe,
)
from tests.support import make_state
from tests.test_abstraction_audit import perfect_information_value

#: Deep enough to solve any small endgame outright.
SOLVE = SearchConfig(depth=40)
DEPTHS = (1, 2, 3, 5)


def walk(seed: int):
    rng = random.Random(seed)
    state = new_game(seed=seed)
    while not state.is_over:
        yield state
        state = apply_move(state, rng.choice(get_legal_moves(state)))
    yield state


def info_for(state, player: int) -> SearchInformation:
    return SearchInformation.from_view(observe(state, player))


def true_position(state, player: int) -> DeterminizedPosition:
    """A position built from the *real* allocation. Test-only ground truth."""
    return build_position(
        info_for(state, player),
        DeterminizedWorld(
            player=player,
            opponent_hand=frozenset(state.hands[1 - player]),
            talon=tuple(state.talon),
        ),
    )


def outcome_sign(value: float) -> int:
    """Map a solved value onto the oracle's +1 / -1 / 0."""
    if value >= MATE_MARGIN:
        return 1
    if value <= -MATE_MARGIN:
        return -1
    assert value == TERMINAL_DRAW, f"{value} is neither proven nor a draw"
    return 0


def small_endgames(games: int = 60, cards: int = 6):
    """Empty-talon positions the oracle can solve exhaustively."""
    for seed in range(games):
        for state in walk(seed):
            if state.talon:
                continue
            if len(state.hands[0]) + len(state.hands[1]) > cards:
                continue
            player = state.current_player
            if player is None:
                continue
            yield state, player


# ======================================================================
# The reason this search exists
# ======================================================================
def test_different_worlds_produce_different_search_results() -> None:
    """Non-degeneracy. Phase 5's search fails this; that is why 7.3.2a exists."""
    distinct_answers = 0
    for seed in range(15):
        for index, state in enumerate(walk(seed)):
            if index % 6 or state.current_player is None:
                continue
            information = info_for(state, state.current_player)
            if information.allocations < 4:
                continue
            generator = WorldGenerator(information)
            results = set()
            for sample_seed in range(8):
                world = generator.sample(random.Random(sample_seed))
                position = build_position(information, world)
                result = search_world(position, SearchConfig(depth=4))
                results.add((result.move, result.value))
            if len(results) > 1:
                distinct_answers += 1
    assert distinct_answers > 20, (
        "the search cannot distinguish worlds; it is re-redacting them"
    )


def test_the_search_sees_the_opponents_actual_cards() -> None:
    """The abstraction is gone: opponent moves are card-level."""
    state = make_state(
        hand0="6S", hand1="7S 9H", trump=Suit.HEARTS, talon="", attacker=0
    )
    position = true_position(state, 0)
    opponent_moves = get_legal_moves(position.apply(position.legal_moves()[0]).state)
    assert any(type(m).__name__ == "DefenseMove" for m in opponent_moves), (
        "the opponent's replies are abstract, not card-level"
    )


# ======================================================================
# Value convention and depth semantics
# ======================================================================
def test_values_are_from_the_root_players_point_of_view() -> None:
    """A won position must score positive for the winner and negative for the loser."""
    state = make_state(
        hand0="6S", hand1="7S 9H", trump=Suit.HEARTS, talon="", attacker=0
    )
    for player in (0, 1):
        position = true_position(state, player)
        result = search_world(position, SOLVE, root_player=player)
        truth = perfect_information_value(state, player, {})
        assert outcome_sign(result.value) == truth, player
    zero = search_world(true_position(state, 0), SOLVE, root_player=0)
    one = search_world(true_position(state, 0), SOLVE, root_player=1)
    assert outcome_sign(zero.value) == -outcome_sign(one.value)


def test_depth_zero_scores_the_position_where_it_stands() -> None:
    for seed in range(6):
        for state in walk(seed):
            position = true_position(state, 0)
            direct = evaluate_position(position, 0)
            if position.is_terminal:
                assert abs(direct) >= MATE_MARGIN or direct == TERMINAL_DRAW
            else:
                node = position.evaluation_node(0)
                assert direct == BaselineEvaluator().evaluate(node, 0)


def test_leaf_scoring_is_exactly_the_existing_baseline_evaluator() -> None:
    """The reuse contract, asserted directly.

    Phase 7.3.2a must not grow a second heuristic. Leaf scoring of a
    non-terminal position has to be *identical* to handing the existing
    ``BaselineEvaluator`` the position's evaluation node, for every ply.
    """
    checked = 0
    evaluator = BaselineEvaluator()
    for seed in range(5):
        for index, state in enumerate(walk(seed)):
            if index % 11 or state.is_over:
                continue
            for player in (0, 1):
                position = true_position(state, player)
                for ply in (0, 1, 4):
                    assert evaluate_position(
                        position, player, ply=ply
                    ) == evaluator.evaluate(position.evaluation_node(player), ply)
                    checked += 1
    assert checked > 100


def test_the_evaluation_node_describes_the_position_it_came_from() -> None:
    """Catches a corrupted evaluation node, which the reuse contract cannot.

    ``test_leaf_scoring_is_exactly_the_existing_baseline_evaluator``
    compares ``evaluate_position`` against the evaluator handed the *same*
    node, so both sides move together if the node itself is wrong. This
    asserts the node's content against the position independently.
    """
    from durakfish.ai.searchnode import Actor, NodeKind

    checked = 0
    for seed in range(5):
        for index, state in enumerate(walk(seed)):
            if index % 9:
                continue
            for player in (0, 1):
                position, opponent = true_position(state, player), 1 - player
                node = position.evaluation_node(player)

                assert set(node.hand) == set(state.hands[player]), "wrong seat's hand"
                assert node.opponent_cards == len(state.hands[opponent])
                assert node.talon_size == len(state.talon)
                assert node.trump == state.trump_suit
                assert node.attack_limit == state.attack_limit
                assert node.searcher_is_attacker == (state.attacker == player)
                assert node.unknown_own == 0, "a determinized world hides nothing"

                # The node must never pre-empt the frozen engine's terminal
                # verdict: that is what kind=DECISION guarantees.
                assert node.kind is NodeKind.DECISION
                assert node.outcome() is None
                if not state.is_over:
                    assert (node.to_move is Actor.SELF) == (
                        state.current_player == player
                    )
                checked += 1
    assert checked > 100


def test_depth_counts_plies_and_deeper_never_visits_fewer_nodes() -> None:
    """Kept deliberately shallow and sparse.

    Unpruned minimax at depth 5 explores the whole hypothetical game
    including draws, which costs minutes across many positions. Node-count
    monotonicity needs neither the depth nor the sample size.
    """
    checked = 0
    for seed in range(3):
        for index, state in enumerate(walk(seed)):
            if index % 25 or state.is_over:
                continue
            position = true_position(state, 0)
            counts = [
                minimax_world(position, SearchConfig(depth=d, algorithm="minimax"))
                .stats.nodes
                for d in (1, 2, 3)
            ]
            assert counts == sorted(counts), counts
            checked += 1
    assert checked > 3


def test_max_depth_reached_never_exceeds_the_budget() -> None:
    """Catches a missing depth decrement in milliseconds, and cleanly.

    A search that forgets to decrement does terminate — the game is
    finite — but only after exploring the whole subtree, so the symptom
    is a hang rather than a failure. A hang is not a detection: it looks
    like a slow test, and diagnosing it cost this phase a great deal of
    time. Asserting the reported depth turns the same fault into an
    immediate, legible assertion error.
    """
    # Deliberately tiny endgames. A search that ignores its budget still
    # terminates here, because the whole tree is a few plies deep, so the
    # assertion is reached and fails instead of the run hanging.
    positions = [
        make_state(
            hand0="6S 6C", hand1="7S 7C", trump=Suit.HEARTS, talon="", attacker=0
        ),
        make_state(hand0="6S", hand1="7S 9H", trump=Suit.HEARTS, talon="", attacker=0),
        make_state(hand0="6S 7H", hand1="8S 9H", trump=Suit.HEARTS, talon="",
                   attacker=0),
    ]
    for state in positions:
        position = true_position(state, 0)
        for depth in (1, 2):
            for algorithm in ("minimax", "alpha_beta"):
                result = search_world(
                    position, SearchConfig(depth=depth, algorithm=algorithm)
                )
                assert result.stats.max_depth <= depth, (
                    f"{algorithm} reached ply {result.stats.max_depth} "
                    f"with a budget of {depth}"
                )


def test_depth_one_expands_only_the_root_moves() -> None:
    state = next(s for s in walk(3) if not s.is_over)
    position = true_position(state, 0)
    result = minimax_world(position, SearchConfig(depth=1, algorithm="minimax"))
    # Root plus one child per legal move.
    assert result.stats.nodes == 1 + len(position.legal_moves())


def test_a_configuration_rejects_a_non_positive_depth() -> None:
    from durakfish.ai.search import SearchConfigurationError

    with pytest.raises(SearchConfigurationError):
        SearchConfig(depth=0)


# ======================================================================
# Terminals, from the frozen engine
# ======================================================================
TERMINALS = {
    "root wins after one move": (
        make_state(hand0="6S", hand1="7S 9H", trump=Suit.HEARTS, talon="", attacker=0),
        0,
    ),
    "root loses after one move": (
        make_state(hand0="6S 9C", hand1="7S", trump=Suit.HEARTS, talon="", attacker=0),
        0,
    ),
    "opponent wins": (
        make_state(hand0="6S 9C", hand1="7S", trump=Suit.HEARTS, talon="", attacker=0),
        1,
    ),
    "draw": (
        make_state(hand0="6S", hand1="7S", trump=Suit.HEARTS, talon="", attacker=0),
        0,
    ),
    "near-empty hands, empty talon": (
        make_state(
            hand0="6S 7H", hand1="8S 9H", trump=Suit.HEARTS, talon="", attacker=0
        ),
        0,
    ),
}


@pytest.mark.parametrize("label", sorted(TERMINALS))
def test_terminal_outcomes_match_the_oracle(label: str) -> None:
    state, player = TERMINALS[label]
    result = search_world(true_position(state, player), SOLVE, root_player=player)
    assert outcome_sign(result.value) == perfect_information_value(state, player, {})


def test_a_finished_position_returns_its_exact_value_and_no_move() -> None:
    state = next(s for s in walk(2) if s.is_over)
    for player in (0, 1):
        result = search_world(true_position(state, player), SOLVE, root_player=player)
        assert result.move is None
        assert outcome_sign(result.value) == perfect_information_value(
            state, player, {}
        )


def test_terminals_are_detected_before_the_depth_cutoff() -> None:
    """A terminal must be scored exactly, however much depth is left.

    The position below ends after a single move: the attacker has no
    cards, the table is fully defended and the talon is empty, so ending
    the bout puts them out. Depth 1 therefore reaches a terminal child,
    and every deeper search must return the same exact value rather than
    falling through to the heuristic.
    """
    state = make_state(
        hand0="",
        hand1="9H",
        trump=Suit.HEARTS,
        talon="",
        table=(("6S", "7S"),),
        attacker=0,
        attack_limit=3,
    )
    position = true_position(state, 0)
    assert len(position.legal_moves()) == 1  # only ending the bout

    values = set()
    for depth in (1, 2, 5, 12):
        result = search_world(position, SearchConfig(depth=depth), root_player=0)
        assert result.stats.terminals > 0, depth
        values.add(result.value)
    assert len(values) == 1, "the value changed with remaining depth"
    assert outcome_sign(values.pop()) == perfect_information_value(state, 0, {})


# ======================================================================
# Oracle agreement
# ======================================================================
def test_solved_values_match_the_independent_oracle() -> None:
    """The central correctness claim."""
    sys.setrecursionlimit(20000)
    compared = 0
    for state, player in small_endgames(games=60, cards=5):
        if compared >= 200:
            break
        result = search_world(true_position(state, player), SOLVE, root_player=player)
        assert outcome_sign(result.value) == perfect_information_value(
            state, player, {}
        ), state.describe()
        compared += 1
    assert compared >= 150


def test_the_chosen_move_matches_the_oracle_when_it_is_uniquely_best() -> None:
    """Only where the oracle says one move is strictly better than the rest."""
    sys.setrecursionlimit(20000)
    unique = 0
    for state, player in small_endgames(games=60, cards=5):
        if unique >= 60:
            break
        if state.current_player != player:
            continue
        scores = {
            move: perfect_information_value(apply_move(state, move), player, {})
            for move in get_legal_moves(state)
        }
        best = max(scores.values())
        winners = [m for m, v in scores.items() if v == best]
        if len(winners) != 1:
            continue  # genuinely equivalent moves: no unique answer to demand
        result = search_world(true_position(state, player), SOLVE, root_player=player)
        assert result.move == winners[0], state.describe()
        unique += 1
    assert unique > 20


def test_a_chosen_move_is_never_worse_than_the_oracle_optimum() -> None:
    """Where several moves tie, any of them is acceptable — but only those."""
    sys.setrecursionlimit(20000)
    checked = 0
    for state, player in small_endgames(games=40, cards=5):
        if checked >= 80:
            break
        if state.current_player != player:
            continue
        scores = {
            move: perfect_information_value(apply_move(state, move), player, {})
            for move in get_legal_moves(state)
        }
        result = search_world(true_position(state, player), SOLVE, root_player=player)
        assert scores[result.move] == max(scores.values())
        checked += 1
    assert checked > 30


# ======================================================================
# Alpha-beta equivalence
# ======================================================================
def test_alpha_beta_agrees_with_minimax_on_a_small_sample() -> None:
    """A fast counterpart to the exhaustive comparison below.

    Exists so that mutation runs, which re-execute the suite once per
    mutation, can detect a corrupted alpha-beta bound in seconds rather
    than minutes.
    """
    compared = 0
    for seed in range(6):
        for index, state in enumerate(walk(seed)):
            if index % 7 or state.is_over:
                continue
            position = true_position(state, 0)
            for depth in (2, 3, 4):
                plain = minimax_world(
                    position, SearchConfig(depth=depth, algorithm="minimax")
                )
                pruned = alpha_beta_world(position, SearchConfig(depth=depth))
                assert (plain.value, plain.move) == (pruned.value, pruned.move)
                compared += 1
    assert compared > 100


def test_alpha_beta_agrees_with_minimax_everywhere() -> None:
    compared = 0
    for seed in range(15):
        for index, state in enumerate(walk(seed)):
            if index % 4 or state.is_over:
                continue
            position = true_position(state, 0)
            for depth in (1, 2, 3):
                plain = minimax_world(
                    position, SearchConfig(depth=depth, algorithm="minimax")
                )
                pruned = alpha_beta_world(position, SearchConfig(depth=depth))
                assert plain.value == pruned.value, (depth, state.describe())
                assert plain.move == pruned.move, (depth, state.describe())
                compared += 1
    assert compared > 600


def test_alpha_beta_agrees_on_solved_endgames() -> None:
    compared = 0
    for state, player in small_endgames(games=40, cards=6):
        if compared >= 120:
            break
        position = true_position(state, player)
        plain = minimax_world(
            position, SearchConfig(depth=12, algorithm="minimax"), root_player=player
        )
        pruned = alpha_beta_world(
            position, SearchConfig(depth=12), root_player=player
        )
        assert (plain.value, plain.move) == (pruned.value, pruned.move)
        compared += 1
    assert compared > 60


def test_alpha_beta_prunes_and_records_its_cutoffs() -> None:
    saved = plain_total = pruned_total = 0
    for seed in range(12):
        for index, state in enumerate(walk(seed)):
            if index % 4 or state.is_over:
                continue
            position = true_position(state, 0)
            plain = minimax_world(
                position, SearchConfig(depth=4, algorithm="minimax")
            ).stats
            fast = alpha_beta_world(position, SearchConfig(depth=4)).stats
            assert fast.nodes <= plain.nodes
            plain_total += plain.nodes
            pruned_total += fast.nodes
            if fast.nodes < plain.nodes:
                saved += 1
                assert fast.cutoffs > 0, "fewer nodes but no cutoff recorded"
    assert saved > 30
    assert pruned_total < plain_total * 0.95


def test_depth_one_cannot_prune() -> None:
    for seed in range(5):
        for index, state in enumerate(walk(seed)):
            if index % 7 or state.is_over:
                continue
            position = true_position(state, 0)
            plain = minimax_world(
                position, SearchConfig(depth=1, algorithm="minimax")
            ).stats
            fast = alpha_beta_world(position, SearchConfig(depth=1)).stats
            assert fast.nodes == plain.nodes
            assert fast.cutoffs == 0


def test_ordering_changes_node_counts_but_never_values() -> None:
    differing = 0
    for seed in range(8):
        for index, state in enumerate(walk(seed)):
            if index % 5 or state.is_over:
                continue
            position = true_position(state, 0)
            static = alpha_beta_world(
                position, SearchConfig(depth=4, ordering="static")
            )
            canonical = alpha_beta_world(
                position, SearchConfig(depth=4, ordering="canonical")
            )
            assert static.value == canonical.value
            if static.stats.nodes != canonical.stats.nodes:
                differing += 1
    assert differing > 0


# ======================================================================
# World isolation and determinism
# ======================================================================
def test_the_same_world_always_gives_the_same_result() -> None:
    rng = random.Random(5)
    for seed in range(8):
        for index, state in enumerate(walk(seed)):
            if index % 6 or state.is_over:
                continue
            information = info_for(state, 0)
            world = WorldGenerator(information).sample(rng)
            position = build_position(information, world)
            results = [
                search_world(position, SearchConfig(depth=4)) for _ in range(3)
            ]
            assert len({(r.move, r.value, r.stats.nodes) for r in results}) == 1


def test_searching_one_world_cannot_affect_another() -> None:
    """No cache exists; this proves no hidden cross-world state either."""
    rng = random.Random(11)
    for seed in range(10):
        for index, state in enumerate(walk(seed)):
            if index % 6 or state.is_over:
                continue
            information = info_for(state, 0)
            generator = WorldGenerator(information)
            a = build_position(information, generator.sample(rng))
            b = build_position(information, generator.sample(rng))

            alone_a = search_world(a, SearchConfig(depth=4))
            alone_b = search_world(b, SearchConfig(depth=4))

            # Interleave them; results must be unchanged.
            search_world(b, SearchConfig(depth=4))
            again_a = search_world(a, SearchConfig(depth=4))
            search_world(a, SearchConfig(depth=4))
            again_b = search_world(b, SearchConfig(depth=4))

            assert (again_a.move, again_a.value) == (alone_a.move, alone_a.value)
            assert (again_b.move, again_b.value) == (alone_b.move, alone_b.value)


def test_the_same_world_from_two_realities_searches_identically() -> None:
    """Hidden-state noninterference, at the search level."""
    from durakfish.game.state import add_cards

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
            world = WorldGenerator(first).sample(random.Random(7))
            a = search_world(build_position(first, world), SearchConfig(depth=4))
            b = search_world(build_position(second, world), SearchConfig(depth=4))
            assert (a.move, a.value, a.stats.nodes) == (b.move, b.value, b.stats.nodes)
            compared += 1
            break
    assert compared > 8


# ======================================================================
# Root player may not be the mover
# ======================================================================
def test_a_value_can_be_taken_from_the_seat_that_is_not_to_move() -> None:
    for seed in range(8):
        for index, state in enumerate(walk(seed)):
            if index % 7 or state.is_over:
                continue
            mover = state.current_player
            idle = 1 - mover
            position = true_position(state, mover)
            for depth in (2, 4):
                config = SearchConfig(depth=depth)
                as_mover = search_world(position, config, root_player=mover)
                as_idle = search_world(position, config, root_player=idle)
                assert as_mover.root_player == mover
                assert as_idle.root_player == idle
                assert as_mover.move is not None and as_idle.move is not None


def test_zero_sum_holds_on_solved_endgames() -> None:
    """One seat's proven win is the other's proven loss."""
    compared = 0
    for state, player in small_endgames(games=40, cards=5):
        if compared >= 80:
            break
        position = true_position(state, player)
        mine = search_world(position, SOLVE, root_player=player)
        theirs = search_world(position, SOLVE, root_player=1 - player)
        assert outcome_sign(mine.value) == -outcome_sign(theirs.value)
        compared += 1
    assert compared > 40


# ======================================================================
# SearchBot untouched
# ======================================================================
def test_search_bot_decisions_are_unchanged_by_this_phase() -> None:
    from durakfish.ai import SearchBot

    bot = SearchBot(SearchConfig(depth=3), "s")
    bot.reset(random.Random(0))
    rng = random.Random(3)
    checked = 0
    for seed in range(10):
        for state in walk(seed):
            player = state.current_player
            if player is None:
                continue
            view, legal = observe(state, player), get_legal_moves(state)
            before = bot.choose(view, legal)

            information = info_for(state, player)
            world = WorldGenerator(information).sample(rng)
            search_world(build_position(information, world), SearchConfig(depth=3))

            assert bot.choose(view, legal) == before
            checked += 1
    assert checked > 800
