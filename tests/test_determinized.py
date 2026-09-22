"""Evaluating one determinized world.

The claim under test is narrow and checkable: a position built from an
observer's information plus a sampled world behaves *identically* to the
real position, whenever the world sampled happens to be the true one.

That gives a rare luxury — a ground-truth oracle. The test harness may
read the real ``GameState`` to construct the comparison; production code
in ``ai/determinized.py`` may not, and a static check enforces the
difference.

Equivalence is checked at four levels, because agreeing on one proves
little about the others:

* the positions themselves, by ``transposition_key``;
* the legal action sets, move for move;
* every transition, recursively;
* terminal outcomes, and full-depth perfect-information search values.
"""

from __future__ import annotations

import random
import sys

import pytest

from durakfish.ai import (
    DeterminizationError,
    DeterminizedPosition,
    SearchConfig,
    SearchNode,
    alpha_beta,
    build_position,
)
from durakfish.cards import Card, Suit
from durakfish.game import apply_move, get_legal_moves, new_game
from durakfish.game.state import add_cards
from durakfish.information import (
    DeterminizedWorld,
    SearchInformation,
    WorldGenerator,
    observe,
)
from tests.support import make_state
from tests.test_abstraction_audit import perfect_information_value
from tests.test_phase3_audit import FORBIDDEN_TYPES, reachable

DEPTHS = (1, 3, 5)


def walk(seed: int):
    rng = random.Random(seed)
    state = new_game(seed=seed)
    while not state.is_over:
        yield state
        state = apply_move(state, rng.choice(get_legal_moves(state)))
    yield state


def info_for(state, player: int) -> SearchInformation:
    return SearchInformation.from_view(observe(state, player))


def true_world(state, player: int) -> DeterminizedWorld:
    """The real hidden allocation, expressed as a world.

    Test-only: it reads ground truth so that the adapter can be compared
    against reality. Production code never does this — that is the whole
    point of the comparison.
    """
    return DeterminizedWorld(
        player=player,
        opponent_hand=frozenset(state.hands[1 - player]),
        talon=tuple(state.talon),
    )


def truthful_position(state, player: int) -> DeterminizedPosition:
    return build_position(info_for(state, player), true_world(state, player))


# ======================================================================
# Position equivalence
# ======================================================================
def test_a_position_built_from_the_true_world_matches_reality() -> None:
    """The strongest single statement of the contract."""
    checked = 0
    for seed in range(20):
        for state in walk(seed):
            for player in (0, 1):
                position = truthful_position(state, player)
                assert (
                    position.state.transposition_key()
                    == state.transposition_key()
                ), "the hypothetical position differs from the real one"
                checked += 1
    assert checked > 2000


def test_the_hypothetical_state_is_never_the_real_object() -> None:
    """Equivalent, but not the same object, and not reachable from it."""
    for seed in range(6):
        for state in walk(seed):
            position = truthful_position(state, 0)
            assert position.state is not state
            assert position.state.history is None
            assert state.event_count > 0 or state.history is None


def test_every_card_in_a_position_came_from_the_two_permitted_inputs() -> None:
    """Nothing is copied out of the real game."""
    rng = random.Random(3)
    for seed in range(8):
        for state in walk(seed):
            information = info_for(state, 0)
            world = WorldGenerator(information).sample(rng)
            position = build_position(information, world)
            view = information.view

            permitted = (
                set(view.hand)
                | set(view.table_cards)
                | set(view.discard)
                | {view.trump_card}
                | set(world.opponent_hand)
                | set(world.talon)
            )
            present = (
                set(position.state.hands[0])
                | set(position.state.hands[1])
                | set(position.state.talon)
                | set(position.state.table_cards)
                | set(position.state.discard)
            )
            assert present <= permitted


# ======================================================================
# Legal-action equivalence
# ======================================================================
def test_legal_action_sets_match_reality_exactly() -> None:
    kinds: set[str] = set()
    checked = 0
    for seed in range(20):
        for state in walk(seed):
            for player in (0, 1):
                position = truthful_position(state, player)
                actual = get_legal_moves(state)
                assert position.legal_moves() == actual, state.describe()
                kinds.update(type(m).__name__ for m in actual)
                checked += 1
    assert checked > 2000
    assert kinds == {"AttackMove", "DefenseMove", "TakeCards", "EndAttack"}, kinds


def test_legal_actions_match_for_the_player_not_to_move_as_well() -> None:
    for seed in range(8):
        for state in walk(seed):
            position = truthful_position(state, 0)
            for seat in (0, 1):
                assert get_legal_moves(position.state, seat) == get_legal_moves(
                    state, seat
                )


# ======================================================================
# Transition equivalence
# ======================================================================
def compare_subtree(position: DeterminizedPosition, state, depth: int) -> int:
    """Apply every legal move to both, recursively, comparing fingerprints."""
    assert position.state.transposition_key() == state.transposition_key()
    if depth == 0 or state.is_over:
        return 1
    moves = get_legal_moves(state)
    assert position.legal_moves() == moves
    compared = 1
    for move in moves:
        compared += compare_subtree(
            position.apply(move), apply_move(state, move), depth - 1
        )
    return compared


def test_transitions_match_reality_over_whole_subtrees() -> None:
    """Not just one move: every move, three plies deep."""
    compared = 0
    for seed in range(10):
        for index, state in enumerate(walk(seed)):
            if index % 7:  # sample positions; the subtrees are the expensive part
                continue
            compared += compare_subtree(truthful_position(state, 0), state, 3)
    assert compared > 3000


def test_distinct_positions_have_distinct_fingerprints() -> None:
    """Non-vacuity for the fingerprint.

    Every equivalence test asserts fingerprints are *equal*, so a
    fingerprint that ignored the position would satisfy all of them. This
    requires the converse: positions that genuinely differ must be
    distinguishable, or the equivalence results mean nothing.
    """
    prints: set = set()
    for seed in range(6):
        for state in walk(seed):
            for player in (0, 1):
                prints.add(truthful_position(state, player).fingerprint())
    assert len(prints) > 500, "fingerprints collapse distinct positions"


def test_a_move_changes_the_fingerprint() -> None:
    for seed in range(8):
        for state in walk(seed):
            if state.is_over:
                continue
            position = truthful_position(state, 0)
            for move in position.legal_moves():
                assert position.apply(move).fingerprint() != position.fingerprint()


def test_two_different_worlds_give_different_positions() -> None:
    """A hypothesis must depend on which world was sampled."""
    rng = random.Random(4)
    found = 0
    for seed in range(10):
        for state in walk(seed):
            information = info_for(state, 0)
            if information.allocations < 2:
                continue
            generator = WorldGenerator(information)
            worlds = {w.key(): w for w in generator.sample_many(12, rng)}
            if len(worlds) < 2:
                continue
            prints = {
                build_position(information, w).fingerprint() for w in worlds.values()
            }
            assert len(prints) == len(worlds)
            found += 1
            break
    assert found > 5


def test_applying_a_move_clears_the_stale_root_world() -> None:
    """After a transition the allocation describes where cards *were*."""
    state = next(iter(walk(4)))
    position = truthful_position(state, 0)
    assert position.root_world is not None
    child = position.apply(position.legal_moves()[0])
    assert child.root_world is None
    assert child.player == position.player


def test_an_illegal_move_is_refused_by_the_frozen_engine() -> None:
    from durakfish.exceptions import IllegalMoveError
    from durakfish.game.moves import END_ATTACK

    state = next(iter(walk(2)))
    position = truthful_position(state, 0)
    assert END_ATTACK not in position.legal_moves()
    with pytest.raises(IllegalMoveError):
        position.apply(END_ATTACK)


# ======================================================================
# Terminal evaluation equivalence
# ======================================================================
TERMINAL_CASES = {
    "observer wins after one move": make_state(
        hand0="6S", hand1="7S 9H", trump=Suit.HEARTS, talon="", attacker=0
    ),
    "observer loses after one move": make_state(
        hand0="6S 9C", hand1="7S", trump=Suit.HEARTS, talon="", attacker=0
    ),
    "simultaneous finish is a draw": make_state(
        hand0="6S", hand1="7S", trump=Suit.HEARTS, talon="", attacker=0
    ),
    "empty talon, nearly empty hands": make_state(
        hand0="6S 7H", hand1="8S 9H", trump=Suit.HEARTS, talon="", attacker=0
    ),
    "defender must take": make_state(
        hand0="6D", hand1="6S 7S", trump=Suit.HEARTS, talon="",
        table=(("9H", None),), attacker=0, phase=__import__(
            "durakfish.game", fromlist=["Phase"]
        ).Phase.DEFENSE, attack_limit=2,
    ),
}


@pytest.mark.parametrize("label", sorted(TERMINAL_CASES))
def test_terminal_evaluation_matches_reality(label: str) -> None:
    state = TERMINAL_CASES[label]
    for player in (0, 1):
        position = truthful_position(state, player)
        assert position.is_terminal == state.is_over
        assert position.durak == state.durak
        assert position.state.transposition_key() == state.transposition_key()


def test_playing_a_case_out_reaches_the_same_terminal_result() -> None:
    for label, start in TERMINAL_CASES.items():
        rng = random.Random(hash(label) % 1000)
        state = start
        position = truthful_position(state, 0)
        while not state.is_over:
            move = rng.choice(get_legal_moves(state))
            state = apply_move(state, move)
            position = position.apply(move)
        assert position.is_terminal
        assert position.durak == state.durak
        assert position.outcome_for(0) == (
            0 if state.durak is None else (-1 if state.durak == 0 else 1)
        )
        assert position.outcome_for(1) == -(position.outcome_for(0) or 0) or (
            position.outcome_for(0) == 0
        )


def test_outcome_is_none_while_a_game_is_unfinished() -> None:
    state = next(iter(walk(1)))
    position = truthful_position(state, 0)
    assert not position.is_terminal
    assert position.outcome_for(0) is None


# ======================================================================
# Search equivalence, against the existing oracles
# ======================================================================
def test_perfect_information_search_agrees_on_endgames() -> None:
    """Full-depth ground-truth oracle, run on both positions."""
    sys.setrecursionlimit(20000)
    compared = 0
    # Each call is a fresh full-depth solve, so this is capped rather than
    # run over every qualifying position; 150 endgames drawn from 40 games
    # is ample and keeps the suite usable.
    for seed in range(40):
        if compared >= 150:
            break
        for state in walk(seed):
            if state.talon or len(state.hands[0]) + len(state.hands[1]) > 5:
                continue
            player = state.current_player
            if player is None:
                continue
            position = truthful_position(state, player)
            assert perfect_information_value(
                position.state, player, {}
            ) == perfect_information_value(state, player, {})
            compared += 1
            if compared >= 150:
                break
    assert compared >= 100


@pytest.mark.parametrize("depth", DEPTHS)
def test_the_existing_phase_5_search_agrees_at_every_depth(depth: int) -> None:
    """The shipped search, run on both positions and compared.

    Histories are stripped from the real state so that the two views are
    comparable: an observer's history has draws redacted, so a hypothesis
    carries none, and ``observe`` derives its observation record from
    history. Everything search-relevant is unaffected.
    """
    config = SearchConfig(depth=depth)
    compared = 0
    for seed in range(10):
        for index, state in enumerate(walk(seed)):
            if index % 5:
                continue
            player = state.current_player
            if player is None:
                continue
            position = truthful_position(state, player)
            stripped = state.replace(history=None)

            real = alpha_beta(
                SearchNode.from_view(observe(stripped, player)),
                config,
                root_moves=get_legal_moves(stripped),
            )
            hypothetical = alpha_beta(
                SearchNode.from_view(observe(position.state, player)),
                config,
                root_moves=position.legal_moves(),
            )
            assert (real.move, real.score) == (hypothetical.move, hypothetical.score)
            assert real.stats.nodes == hypothetical.stats.nodes
            compared += 1
    assert compared > 150


# ======================================================================
# Noninterference and substitution
# ======================================================================
def paired_state(state, player: int, rng: random.Random):
    """A second real game, identical to the observer, different underneath."""
    view = observe(state, player)
    movable = view.unobserved_cards()
    opponent = [c for c in state.hands[1 - player] if c in movable]
    body = [c for c in state.talon[:-1] if c in movable]
    fixed_opp = [c for c in state.hands[1 - player] if c not in movable]
    fixed_body = [c for c in state.talon[:-1] if c not in movable]
    if not opponent or not body:
        return None
    pool = opponent + body
    rng.shuffle(pool)
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
    return twin


def test_the_same_seed_gives_the_same_position_from_different_realities() -> None:
    """Noninterference: reality must not reach the hypothesis."""
    shuffler = random.Random(31337)
    compared = 0
    for seed in range(15):
        for state in walk(seed):
            player = state.current_player
            if player is None or len(state.talon) < 3:
                continue
            twin = paired_state(state, player, shuffler)
            if twin is None:
                continue
            assert state.hands[1 - player] != twin.hands[1 - player] or (
                state.talon != twin.talon
            )
            assert observe(state, player) == observe(twin, player)

            first = info_for(state, player)
            second = info_for(twin, player)
            for _ in range(4):
                world_a = WorldGenerator(first).sample(random.Random(99))
                world_b = WorldGenerator(second).sample(random.Random(99))
                assert world_a == world_b
                position_a = build_position(first, world_a)
                position_b = build_position(second, world_b)
                assert position_a.fingerprint() == position_b.fingerprint()
                assert position_a.legal_moves() == position_b.legal_moves()
            compared += 1
            break
    assert compared > 8


def test_world_substitution_makes_evaluation_identical() -> None:
    """SI(G1) == SI(G2) and W1 == W2 implies identical evaluation.

    Stronger than testing the generator: the *same* world object is
    supplied explicitly to both, so nothing rests on the sampler.
    """
    shuffler = random.Random(4242)
    compared = 0
    for seed in range(15):
        for state in walk(seed):
            player = state.current_player
            if player is None or len(state.talon) < 3:
                continue
            twin = paired_state(state, player, shuffler)
            if twin is None:
                continue
            first, second = info_for(state, player), info_for(twin, player)
            assert first == second

            world = WorldGenerator(first).sample(random.Random(7))
            a = build_position(first, world)
            b = build_position(second, world)  # the very same world object

            assert a.fingerprint() == b.fingerprint()
            assert a.legal_moves() == b.legal_moves()
            for move in a.legal_moves():
                assert a.apply(move).fingerprint() == b.apply(move).fingerprint()
            if not a.state.talon:
                assert perfect_information_value(
                    a.state, player, {}
                ) == perfect_information_value(b.state, player, {})
            compared += 1
            break
    assert compared > 8


def test_a_position_exposes_no_route_to_a_real_game() -> None:
    rng = random.Random(5)
    for seed in range(5):
        for state in walk(seed):
            information = info_for(state, 0)
            world = WorldGenerator(information).sample(rng)
            position = build_position(information, world)
            # The hypothetical GameState is itself a GameState, so the
            # blanket type ban does not apply; what must not be reachable
            # is the *real* one.
            assert not any(obj is state for obj in reachable(position))
            assert not any(
                isinstance(obj, FORBIDDEN_TYPES) and obj is state
                for obj in reachable(position)
            )


# ======================================================================
# Invalid worlds are refused, never repaired
# ======================================================================
def test_every_kind_of_invalid_world_is_rejected() -> None:
    state = make_state(
        hand0="6S 6C", hand1="7S 7C", trump=Suit.HEARTS, talon="8S 8C 8H",
        attacker=0,
    )
    information = info_for(state, 0)
    good = WorldGenerator(information).sample(random.Random(0))
    build_position(information, good)  # the control: this one works

    held = min(good.opponent_hand, key=lambda c: c.code)
    own = min(information.view.hand, key=lambda c: c.code)
    broken = {
        "dropped card": DeterminizedWorld(
            0, good.opponent_hand - {held}, good.talon
        ),
        "duplicate card": DeterminizedWorld(
            0, good.opponent_hand, good.talon + good.talon[:1]
        ),
        "wrong hand size": DeterminizedWorld(
            0, good.opponent_hand - {held}, (held,) + good.talon
        ),
        "wrong talon size": DeterminizedWorld(
            0, good.opponent_hand, good.talon[:-1]
        ),
        "impossible ownership": DeterminizedWorld(
            0, (good.opponent_hand - {held}) | {own}, good.talon
        ),
        "trump not at bottom": DeterminizedWorld(
            0, good.opponent_hand, tuple(reversed(good.talon))
        ),
    }
    for label, world in broken.items():
        with pytest.raises(DeterminizationError, match="incompatible world"):
            build_position(information, world)


def test_a_world_belonging_to_another_observer_is_rejected() -> None:
    state = next(iter(walk(3)))
    information = info_for(state, 0)
    world = WorldGenerator(info_for(state, 1)).sample(random.Random(0))
    with pytest.raises(DeterminizationError, match="belongs to P1"):
        build_position(information, world)


def test_validation_can_be_skipped_but_is_on_by_default() -> None:
    """The escape hatch exists for hot paths and is explicitly opt-in."""
    state = next(iter(walk(3)))
    information = info_for(state, 0)
    good = WorldGenerator(information).sample(random.Random(0))
    assert build_position(information, good, validate=False) == build_position(
        information, good
    )


def test_an_invalid_world_is_never_silently_repaired() -> None:
    state = make_state(
        hand0="6S 6C", hand1="7S 7C", trump=Suit.HEARTS, talon="8S 8C 8H",
        attacker=0,
    )
    information = info_for(state, 0)
    good = WorldGenerator(information).sample(random.Random(0))
    bad = DeterminizedWorld(0, good.opponent_hand, tuple(reversed(good.talon)))
    with pytest.raises(DeterminizationError):
        build_position(information, bad)
    # And nothing was mutated on the way out.
    assert bad.talon == tuple(reversed(good.talon))


# ======================================================================
# SearchBot is untouched
# ======================================================================
def test_search_bot_decisions_are_unchanged_by_this_phase() -> None:
    from durakfish.ai import SearchBot

    bot = SearchBot(SearchConfig(depth=3), "s")
    bot.reset(random.Random(0))
    rng = random.Random(2)
    checked = 0
    for seed in range(12):
        for state in walk(seed):
            player = state.current_player
            if player is None:
                continue
            view, legal = observe(state, player), get_legal_moves(state)
            before = bot.choose(view, legal)

            information = info_for(state, player)
            build_position(information, WorldGenerator(information).sample(rng))

            assert bot.choose(view, legal) == before
            checked += 1
    assert checked > 1000
