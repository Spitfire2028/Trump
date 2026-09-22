"""Determinized worlds.

Three things need proving, and the third is the one that would be easiest
to get wrong without noticing:

* every generated world is a member of ``W(I)`` — legal under every
  constraint the information implies;
* the sampler is uniform over compatible allocations, and samples the
  **joint** allocation rather than each card's marginal;
* the real hidden state is irrelevant — two different actual worlds that
  look identical to the observer produce identical generated sequences
  from the same seed.

Expected distributions come from an enumerating reference that shares no
algorithm with the sampler.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter

import pytest

from durakfish.cards import Card, Suit
from durakfish.cards.deck import standard_cards
from durakfish.game import apply_move, get_legal_moves, new_game
from durakfish.game.moves import END_ATTACK, TAKE, AttackMove
from durakfish.game.state import add_cards
from durakfish.information import (
    CardLocation,
    CardTracker,
    DeterminizedWorld,
    SearchInformation,
    WorldError,
    WorldGenerator,
    is_valid_world,
    observe,
    validate_world,
)
from tests.reference_worlds import enumerate_allocations
from tests.support import make_state
from tests.test_phase3_audit import FORBIDDEN_TYPES, reachable

DECK36 = frozenset(standard_cards(36))

# Chi-square critical values at p = 0.001, by degrees of freedom.
CHI2 = {1: 10.83, 2: 13.82, 5: 20.52, 9: 27.88, 14: 36.12}


def walk(seed: int):
    rng = random.Random(seed)
    state = new_game(seed=seed)
    while not state.is_over:
        yield state
        state = apply_move(state, rng.choice(get_legal_moves(state)))
    yield state


def info_for(state, player: int) -> SearchInformation:
    return SearchInformation.from_view(observe(state, player))


def small_example() -> SearchInformation:
    """A position with exactly C(4,2) = 6 compatible allocations.

    Two cards in hand, two for the opponent, a three-card talon whose
    bottom card is the face-up trump. That leaves four undetermined cards
    filling two opponent slots and two talon slots.
    """
    state = make_state(
        hand0="6S 6C",
        hand1="7S 7C",
        trump=Suit.HEARTS,
        talon="8S 8C 8H",
        attacker=0,
    )
    return info_for(state, 0)


# ======================================================================
# Every generated world is legal
# ======================================================================
def test_generated_worlds_are_legal_throughout_real_games() -> None:
    rng = random.Random(11)
    checked = 0
    for seed in range(12):
        for state in walk(seed):
            for player in (0, 1):
                information = info_for(state, player)
                generator = WorldGenerator(information)
                for world in generator.sample_many(3, rng):
                    validate_world(world, information)
                    checked += 1
    assert checked > 3000


def test_generated_worlds_match_the_real_hidden_state_in_shape() -> None:
    """Ground truth is used only to check sizes, never to generate."""
    rng = random.Random(5)
    for seed in range(10):
        for state in walk(seed):
            player = state.current_player
            if player is None:
                continue
            information = info_for(state, player)
            world = WorldGenerator(information).sample(rng)
            assert len(world.opponent_hand) == len(state.hands[1 - player])
            assert len(world.talon) == len(state.talon)


def test_the_real_world_is_always_one_of_the_compatible_ones() -> None:
    """W(I) must contain the truth, or the model has excluded reality."""
    for seed in range(15):
        for state in walk(seed):
            for player in (0, 1):
                information = info_for(state, player)
                truth = DeterminizedWorld(
                    player=player,
                    opponent_hand=frozenset(state.hands[1 - player]),
                    talon=tuple(state.talon),
                )
                validate_world(truth, information)


def test_certain_and_deduced_ownership_is_preserved() -> None:
    rng = random.Random(3)
    found = 0
    for seed in range(15):
        for state in walk(seed):
            information = info_for(state, 0)
            certain = information.certain_cards_in(CardLocation.OPPONENT_HAND)
            if not certain:
                continue
            world = WorldGenerator(information).sample(rng)
            assert certain <= world.opponent_hand
            found += 1
    assert found > 100


def test_impossible_placements_never_occur() -> None:
    rng = random.Random(9)
    for seed in range(10):
        for state in walk(seed):
            information = info_for(state, 0)
            world = WorldGenerator(information).sample(rng)
            for card in world.opponent_hand:
                assert not information.is_impossible(card, CardLocation.OPPONENT_HAND)
            for card in world.talon:
                assert not information.is_impossible(card, CardLocation.TALON)


def test_the_trump_card_is_always_the_bottom_of_a_non_empty_talon() -> None:
    """A structural rule the engine relies on and does not itself check."""
    rng = random.Random(17)
    checked = 0
    for seed in range(10):
        for state in walk(seed):
            information = info_for(state, 0)
            world = WorldGenerator(information).sample(rng)
            if world.talon:
                assert world.talon[-1] == information.view.trump_card
                checked += 1
    assert checked > 200


def test_public_information_is_never_moved_into_hiding() -> None:
    rng = random.Random(21)
    for seed in range(10):
        for state in walk(seed):
            information = info_for(state, 0)
            view = information.view
            public = set(view.hand) | set(view.table_cards) | set(view.discard)
            world = WorldGenerator(information).sample(rng)
            assert not (world.hidden_cards & public)


# ======================================================================
# The validator can actually fail
# ======================================================================
def test_the_validator_rejects_each_kind_of_broken_world() -> None:
    """A validator that accepts everything would make the suite worthless."""
    information = small_example()
    rng = random.Random(1)
    good = WorldGenerator(information).sample(rng)
    validate_world(good, information)

    spare = next(iter(good.talon_cards - {information.view.trump_card}))
    stolen = next(iter(good.opponent_hand))

    broken = {
        "duplicate": DeterminizedWorld(
            good.player, good.opponent_hand, good.talon + (good.talon[0],)
        ),
        "overlap": DeterminizedWorld(
            good.player, good.opponent_hand | {spare}, good.talon
        ),
        "wrong hand size": DeterminizedWorld(
            good.player, good.opponent_hand - {stolen}, good.talon
        ),
        "wrong talon size": DeterminizedWorld(
            good.player, good.opponent_hand, good.talon[:-1]
        ),
        "trump not at the bottom": DeterminizedWorld(
            good.player, good.opponent_hand, tuple(reversed(good.talon))
        ),
        "public card hidden": DeterminizedWorld(
            good.player,
            good.opponent_hand | {Card.parse("6S")},
            good.talon,
        ),
        "wrong player": DeterminizedWorld(1, good.opponent_hand, good.talon),
    }
    for label, world in broken.items():
        with pytest.raises(WorldError):
            validate_world(world, information)
        assert not is_valid_world(world, information), label


def test_the_validator_rejects_a_world_that_moves_a_deduced_card() -> None:
    # A talon big enough that refilling both hands still leaves spare
    # non-trump cards to swap with.
    state = make_state(
        hand0="6S 6C",
        hand1="9H 10H",
        trump=Suit.HEARTS,
        talon="8C 8D 8S 9C 9D 9S 10C 10D 10S JC JD 8H",
        attacker=0,
    )
    tracker = CardTracker(0)
    tracker.update(observe(state, 0))
    for move in (AttackMove(Card.parse("6S")), TAKE, END_ATTACK):
        state = apply_move(state, move)
        tracker.update(observe(state, 0))

    information = SearchInformation.from_tracker(tracker, observe(state, 0))
    six = Card.parse("6S")
    assert information.location_of(six) is CardLocation.OPPONENT_HAND

    world = WorldGenerator(information).sample(random.Random(0))
    assert six in world.opponent_hand

    # Swap the deduced card with a talon card, so hand and talon sizes stay
    # correct and the ownership check is what has to catch it — not the
    # size check, which runs first and would mask the real question.
    swappable = next(c for c in world.talon[:-1])
    tampered = DeterminizedWorld(
        0,
        (world.opponent_hand - {six}) | {swappable},
        tuple(c for c in world.talon if c != swappable)[:-1] + (six,)
        + (world.talon[-1],),
    )
    assert len(tampered.opponent_hand) == len(world.opponent_hand)
    assert len(tampered.talon) == len(world.talon)
    with pytest.raises(WorldError, match="known to be the opponent"):
        validate_world(tampered, information)


def test_each_validator_check_is_the_first_to_reject_its_own_violation() -> None:
    """Pins down *which* check rejects, not merely that something does.

    The validator is over-determined on purpose — several checks imply one
    another — so a corruption usually trips more than one. Asserting only
    "rejected" would let any single check be deleted without a test
    failing. Asserting the specific message makes each one load-bearing.

    Two checks are deliberately absent from this table because they can
    never fire first, and are documented as defensive in ``worlds.py``:

    * **talon size** — with the hand size right and conservation intact,
      the talon size is forced by counting, so a violation always trips
      conservation or the hand-size check first.
    * **impossible placement** — every card impossible in the opponent's
      hand is settled somewhere else, so public-trespass, the overlap
      check, or certain-ownership catches it earlier.
    """
    information = small_example()
    world = WorldGenerator(information).sample(random.Random(0))
    validate_world(world, information)

    held = min(world.opponent_hand, key=lambda c: c.code)
    trump = information.view.trump_card
    own = min(information.view.hand, key=lambda c: c.code)
    phantom = Card.parse("2S")  # a real card, but not in the 36-card deck
    assert phantom not in information.knowledge.deck

    cases = {
        "duplicated": (
            "duplicated card",
            DeterminizedWorld(0, world.opponent_hand, world.talon + world.talon[:1]),
        ),
        "both hidden locations": (
            "both hidden locations",
            DeterminizedWorld(0, world.opponent_hand | {trump}, world.talon),
        ),
        "publicly located": (
            "publicly located",
            DeterminizedWorld(
                0, (world.opponent_hand - {held}) | {own}, world.talon
            ),
        ),
        "conservation": (
            "conservation broken",
            DeterminizedWorld(
                0, (world.opponent_hand - {held}) | {phantom}, world.talon
            ),
        ),
        "hand size": (
            "opponent holds",
            DeterminizedWorld(
                0, world.opponent_hand - {held}, (held,) + world.talon
            ),
        ),
        "certain talon": (
            "known to be in the talon",
            DeterminizedWorld(
                0,
                (world.opponent_hand - {held}) | {trump},
                tuple(c for c in world.talon if c != trump) + (held,),
            ),
        ),
        "trump position": (
            "drawn last",
            DeterminizedWorld(0, world.opponent_hand, tuple(reversed(world.talon))),
        ),
    }
    for label, (expected, broken) in cases.items():
        with pytest.raises(WorldError, match=expected):
            validate_world(broken, information)


def test_certain_opponent_ownership_is_the_first_check_to_reject_its_violation() -> None:
    """Needs a position with a deduced opponent card, built by play."""
    state = make_state(
        hand0="6S 6C",
        hand1="9H 10H",
        trump=Suit.HEARTS,
        talon="8C 8D 8S 9C 9D 9S 10C 10D 10S JC JD 8H",
        attacker=0,
    )
    tracker = CardTracker(0)
    tracker.update(observe(state, 0))
    for move in (AttackMove(Card.parse("6S")), TAKE, END_ATTACK):
        state = apply_move(state, move)
        tracker.update(observe(state, 0))

    information = SearchInformation.from_tracker(tracker, observe(state, 0))
    six = Card.parse("6S")
    assert information.location_of(six) is CardLocation.OPPONENT_HAND
    world = WorldGenerator(information).sample(random.Random(0))

    swappable = world.talon[0]
    broken = DeterminizedWorld(
        0,
        (world.opponent_hand - {six}) | {swappable},
        (six,) + tuple(c for c in world.talon if c != swappable),
    )
    with pytest.raises(WorldError, match="known to be the opponent"):
        validate_world(broken, information)


def test_the_generator_orders_its_candidates_explicitly() -> None:
    """Sampling must not inherit set iteration order.

    Card hashes are their dense codes, so frozenset iteration is
    deterministic — but it is *slot* order, which wraps around for small
    tables and differs from sorted order in roughly half of real
    positions. The guarantee must come from an explicit sort, not from an
    accident of how small integers land in a hash table.

    Whole games are walked rather than their opening positions: early
    candidate sets are large enough to iterate in ascending order by
    coincidence, and testing only those would miss the difference
    entirely.
    """
    unsorted_sets = 0
    for seed in range(8):
        for state in walk(seed):
            information = info_for(state, 0)
            generator = WorldGenerator(information)
            candidates = generator._candidates  # noqa: SLF001 - white-box on purpose
            codes = [c.code for c in candidates]
            assert codes == sorted(codes), "candidates are not explicitly sorted"
            assert set(candidates) == set(information.undetermined)
            raw = [c.code for c in information.undetermined]
            if raw != sorted(raw):
                unsorted_sets += 1
    assert unsorted_sets > 100, (
        "no position had a candidate set whose iteration order differed from "
        "sorted, so this test could not have detected an unsorted generator"
    )


# ======================================================================
# Uniformity, against the enumerating reference
# ======================================================================
def test_the_worked_example_has_exactly_six_allocations() -> None:
    information = small_example()
    assert len(information.undetermined) == 4
    assert information.knowledge.opponent_slots == 2
    assert information.knowledge.talon_slots == 2
    assert information.allocations == 6
    assert len(enumerate_allocations(information)) == 6


def test_sampling_the_worked_example_is_uniform() -> None:
    """6 allocations, 12,000 draws: each should land near 1/6."""
    information = small_example()
    generator = WorldGenerator(information)
    rng = random.Random(2024)
    draws = 12_000
    counts: Counter = Counter(
        generator.sample(rng).allocation_key() for _ in range(draws)
    )

    expected_allocations = {
        tuple(sorted(c.code for c in allocation))
        for allocation in enumerate_allocations(information)
    }
    assert set(counts) == expected_allocations, "sampler and reference disagree"

    expected = draws / len(expected_allocations)
    chi2 = sum((counts[a] - expected) ** 2 / expected for a in expected_allocations)
    assert chi2 < CHI2[5], f"chi2={chi2:.2f}: sampling is not uniform"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sampling_is_uniform_on_real_small_positions(seed: int) -> None:
    """Found in play rather than constructed, so it is not a special case."""
    tested = 0
    for state in walk(seed):
        for player in (0, 1):
            information = info_for(state, player)
            allocations = information.allocations
            if not 2 <= allocations <= 15:
                continue
            expected_set = {
                tuple(sorted(c.code for c in a))
                for a in enumerate_allocations(information)
            }
            assert len(expected_set) == allocations
            generator = WorldGenerator(information)
            rng = random.Random(4242)
            draws = 400 * allocations
            counts: Counter = Counter(
                generator.sample(rng).allocation_key() for _ in range(draws)
            )
            assert set(counts) <= expected_set
            assert set(counts) == expected_set, "some allocation was never drawn"
            expected = draws / allocations
            chi2 = sum((counts[a] - expected) ** 2 / expected for a in expected_set)
            assert chi2 < CHI2.get(allocations - 1, 40.0), chi2
            tested += 1
    assert tested > 0, "no small position arose in this game"


def test_sampling_is_joint_not_marginal() -> None:
    """The failure mode the specification singles out.

    Under independent marginal sampling, two specific cards would land
    together about ``p^2`` of the time. The correct joint gives a
    materially different figure, and the sampler must match the joint.
    """
    information = small_example()
    candidates = sorted(information.undetermined, key=lambda c: c.code)
    pair = frozenset(candidates[:2])

    joint = information.probability_all_in(pair, CardLocation.OPPONENT_HAND)
    marginal = information.probability(candidates[0], CardLocation.OPPONENT_HAND)
    naive = marginal**2
    assert math.isclose(joint, 1 / 6)
    assert math.isclose(naive, 0.25)

    generator = WorldGenerator(information)
    rng = random.Random(99)
    draws = 20_000
    together = sum(
        1 for _ in range(draws) if pair <= generator.sample(rng).opponent_hand
    )
    observed = together / draws
    assert abs(observed - joint) < 0.02, observed
    assert abs(observed - naive) > 0.05, (
        "the sampler matches independent marginals; it is not sampling jointly"
    )


def test_hand_sizes_are_always_exact_which_marginal_sampling_would_break() -> None:
    """Independent sampling would produce wrong-sized hands constantly."""
    rng = random.Random(6)
    for seed in range(8):
        for state in walk(seed):
            information = info_for(state, 0)
            for world in WorldGenerator(information).sample_many(2, rng):
                assert len(world.opponent_hand) == information.view.opponent_hand_size


# ======================================================================
# Noninterference — the mandatory test
# ======================================================================
def paired_worlds(state, player: int, rng: random.Random):
    """Two real states, identical to the observer, different underneath."""
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
            state.hands[p] if p == player else add_cards((), fixed_opp + pool[: len(opponent)])
            for p in range(2)
        ),
        talon=tuple(fixed_body + pool[len(opponent) :]) + state.talon[-1:],
    )
    twin.validate()
    return twin


def test_generation_is_identical_for_indistinguishable_hidden_states() -> None:
    """Mandatory: the real hidden state must not influence generation."""
    shuffler = random.Random(31337)
    compared = 0
    for seed in range(15):
        for state in walk(seed):
            player = state.current_player
            if player is None or len(state.talon) < 3:
                continue
            twin = paired_worlds(state, player, shuffler)
            if twin is None:
                continue
            assert state.hands[1 - player] != twin.hands[1 - player] or (
                state.talon != twin.talon
            ), "the premise failed: the hidden states are the same"
            assert observe(state, player) == observe(twin, player)

            first = WorldGenerator(info_for(state, player)).sample_many(
                8, random.Random(777)
            )
            second = WorldGenerator(info_for(twin, player)).sample_many(
                8, random.Random(777)
            )
            assert [w.key() for w in first] == [w.key() for w in second]
            assert [w.to_dict() for w in first] == [w.to_dict() for w in second]
            compared += 1
            break
    assert compared > 10


def test_a_generator_exposes_no_route_to_hidden_state() -> None:
    for seed in range(5):
        for state in walk(seed):
            information = info_for(state, 0)
            generator = WorldGenerator(information)
            world = generator.sample(random.Random(1))
            for payload in (generator, world):
                for obj in reachable(payload):
                    assert not isinstance(obj, FORBIDDEN_TYPES)


def test_the_worlds_module_never_names_the_state_layer() -> None:
    import ast
    import pathlib

    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "src" / "durakfish" / "information" / "worlds.py"
    )
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    for banned in ("GameState", "hands", "apply_move", "new_game", "observe"):
        assert banned not in names, f"worlds.py references {banned!r}"


# ======================================================================
# Reproducibility
# ======================================================================
def test_the_same_seed_reproduces_the_same_worlds() -> None:
    information = info_for(next(iter(walk(4))), 0)
    generator = WorldGenerator(information)
    first = [w.key() for w in generator.sample_many(20, random.Random(123))]
    second = [w.key() for w in generator.sample_many(20, random.Random(123))]
    assert first == second


def test_different_seeds_diverge() -> None:
    information = info_for(next(iter(walk(4))), 0)
    generator = WorldGenerator(information)
    sequences = {
        tuple(w.key() for w in generator.sample_many(10, random.Random(seed)))
        for seed in range(20)
    }
    assert len(sequences) == 20


def test_the_generator_never_touches_the_global_random_module(monkeypatch) -> None:
    def forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("world generation used the global random module")

    for name in ("random", "randrange", "choice", "shuffle", "sample", "randint"):
        monkeypatch.setattr(random, name, forbidden)

    information = info_for(next(iter(walk(2))), 0)
    world = WorldGenerator(information).sample(random.Random(5))
    validate_world(world, information)


def test_a_fresh_generator_gives_the_same_sequence() -> None:
    """Generators hold no sampling state of their own."""
    information = info_for(next(iter(walk(6))), 0)
    first = [
        w.key() for w in WorldGenerator(information).sample_many(10, random.Random(8))
    ]
    second = [
        w.key() for w in WorldGenerator(information).sample_many(10, random.Random(8))
    ]
    assert first == second


# ======================================================================
# Adversarial cases
# ======================================================================
def test_a_fully_determined_position_yields_exactly_one_world() -> None:
    """Empty talon: Phase 6 knows the opponent's hand outright."""
    state = make_state(
        hand0="6S 7H", hand1="8S 9H", trump=Suit.HEARTS, talon="", attacker=0
    )
    information = info_for(state, 0)
    generator = WorldGenerator(information)
    assert generator.is_determined
    assert generator.allocations == 1
    worlds = {w.key() for w in generator.sample_many(50, random.Random(1))}
    assert len(worlds) == 1
    only = generator.sample(random.Random(2))
    assert only.opponent_hand == frozenset(state.hands[1])
    assert only.talon == ()


def test_a_single_undetermined_card_has_two_or_fewer_worlds() -> None:
    state = make_state(
        hand0="6S 6C 6D",
        hand1="7S",
        trump=Suit.HEARTS,
        talon="8S 8H",
        attacker=0,
    )
    information = info_for(state, 0)
    reference = enumerate_allocations(information)
    generator = WorldGenerator(information)
    drawn = {w.allocation_key() for w in generator.sample_many(200, random.Random(3))}
    assert len(drawn) == len(reference)
    for world in generator.sample_many(20, random.Random(4)):
        validate_world(world, information)


def test_all_undetermined_cards_go_to_the_opponent_when_the_talon_is_full_of_known() -> None:
    """Talon slots zero: everything undetermined must be theirs."""
    state = make_state(
        hand0="6S 6C 6D 6H 7S 7C",
        hand1="8S 8C 8D 8H 9S 9C",
        trump=Suit.HEARTS,
        talon="AH",
        attacker=0,
    )
    information = info_for(state, 0)
    # The lone talon card is the trump and is deduced, so nothing is left
    # undetermined for the talon.
    assert information.knowledge.talon_slots == 0
    world = WorldGenerator(information).sample(random.Random(1))
    validate_world(world, information)
    assert world.talon == (Card.parse("AH"),)
    assert world.opponent_hand == frozenset(state.hands[1])


def test_endgame_and_early_game_positions_both_generate_legally() -> None:
    rng = random.Random(12)
    early = late = 0
    for seed in range(10):
        for state in walk(seed):
            information = info_for(state, 0)
            world = WorldGenerator(information).sample(rng)
            validate_world(world, information)
            if information.view.talon_size > 15:
                early += 1
            elif information.view.talon_size == 0:
                late += 1
    assert early > 20 and late > 20


def test_positions_with_many_deduced_cards_still_generate() -> None:
    """Forced ownership plus remaining freedom."""
    rng = random.Random(15)
    found = 0
    for seed in range(20):
        for state in walk(seed):
            information = info_for(state, 0)
            deduced = len(information.knowledge.deduced_cards())
            if deduced < 3 or information.allocations < 2:
                continue
            world = WorldGenerator(information).sample(rng)
            validate_world(world, information)
            found += 1
    assert found > 50


def test_sample_many_rejects_a_negative_count() -> None:
    information = small_example()
    with pytest.raises(ValueError, match="non-negative"):
        WorldGenerator(information).sample_many(-1, random.Random(0))
    assert WorldGenerator(information).sample_many(0, random.Random(0)) == ()


# ======================================================================
# Representation
# ======================================================================
def test_a_world_is_immutable_and_hashable() -> None:
    import dataclasses

    information = small_example()
    world = WorldGenerator(information).sample(random.Random(0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        world.player = 1  # type: ignore[misc]
    assert len({world, world}) == 1
    assert hash(world.key()) == hash(world.key())


def test_a_world_round_trips_through_json() -> None:
    information = small_example()
    for world in WorldGenerator(information).sample_many(10, random.Random(2)):
        restored = DeterminizedWorld.from_dict(
            json.loads(json.dumps(world.to_dict()))
        )
        assert restored == world
        assert restored.key() == world.key()
        validate_world(restored, information)


def test_a_world_carries_no_visible_information() -> None:
    """Visible cards are already known exactly; copying them would duplicate."""
    information = small_example()
    world = WorldGenerator(information).sample(random.Random(0))
    for forbidden in ("table", "discard", "own_hand", "hand", "trump_suit"):
        assert not hasattr(world, forbidden)
    assert world.hidden_cards == world.opponent_hand | world.talon_cards


def test_talon_order_varies_but_the_trump_stays_put() -> None:
    information = small_example()
    generator = WorldGenerator(information)
    orders = {w.talon for w in generator.sample_many(200, random.Random(1))}
    assert len(orders) > 1, "talon order is never permuted"
    for order in orders:
        assert order[-1] == information.view.trump_card


def test_location_of_reports_where_a_world_put_a_card() -> None:
    information = small_example()
    world = WorldGenerator(information).sample(random.Random(0))
    for card in world.opponent_hand:
        assert world.location_of(card) is CardLocation.OPPONENT_HAND
    for card in world.talon:
        assert world.location_of(card) is CardLocation.TALON
    assert world.location_of(Card.parse("6S")) is None  # observer's own card
