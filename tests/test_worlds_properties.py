"""Property-based coverage of world generation.

``test_worlds.py`` pins down specific behaviours. This module runs the
same guarantees across many positions drawn from real play, sorted by
game phase and by how much remains hidden, so that a defect showing up
only in (say) large hidden spaces or only in near-certain endgames still
gets found.

Every property is checked in both directions where that is meaningful:
generated worlds must validate, and deliberately corrupted worlds must
not. A validator that accepted everything would satisfy the first half
alone.
"""

from __future__ import annotations

import random

import pytest

from durakfish.cards import Card, Suit
from durakfish.game import apply_move, get_legal_moves, new_game
from durakfish.game.moves import END_ATTACK, TAKE, AttackMove
from durakfish.information import (
    CardLocation,
    CardTracker,
    DeterminizedWorld,
    SearchInformation,
    WorldGenerator,
    is_valid_world,
    observe,
    validate_world,
)
from tests.reference_worlds import enumerate_allocations
from tests.support import make_state


def walk(seed: int):
    rng = random.Random(seed)
    state = new_game(seed=seed)
    while not state.is_over:
        yield state
        state = apply_move(state, rng.choice(get_legal_moves(state)))
    yield state


def info_for(state, player: int) -> SearchInformation:
    return SearchInformation.from_view(observe(state, player))


def phase_of(information: SearchInformation) -> str:
    talon = information.view.talon_size
    if talon > 15:
        return "early"
    if talon > 0:
        return "mid"
    return "late"


# ======================================================================
# Coverage: every game phase and every size of hidden space
# ======================================================================
def test_generated_worlds_validate_across_every_game_phase() -> None:
    rng = random.Random(2718)
    seen: dict[str, int] = {"early": 0, "mid": 0, "late": 0}
    for seed in range(20):
        for state in walk(seed):
            for player in (0, 1):
                information = info_for(state, player)
                for world in WorldGenerator(information).sample_many(2, rng):
                    validate_world(world, information)
                seen[phase_of(information)] += 1
    assert all(count > 100 for count in seen.values()), seen


def test_generated_worlds_validate_across_every_size_of_hidden_space() -> None:
    """From fully determined to hundreds of thousands of allocations."""
    rng = random.Random(1618)
    buckets: dict[str, int] = {"determined": 0, "small": 0, "large": 0, "huge": 0}
    for seed in range(20):
        for state in walk(seed):
            information = info_for(state, 0)
            allocations = information.allocations
            world = WorldGenerator(information).sample(rng)
            validate_world(world, information)
            if allocations == 1:
                buckets["determined"] += 1
            elif allocations <= 20:
                buckets["small"] += 1
            elif allocations <= 10_000:
                buckets["large"] += 1
            else:
                buckets["huge"] += 1
    assert all(count > 5 for count in buckets.values()), buckets


def test_every_generated_world_belongs_to_the_reference_set() -> None:
    """Where enumeration is feasible, membership of W(I) is checked exactly."""
    rng = random.Random(4242)
    compared = 0
    for seed in range(25):
        for state in walk(seed):
            for player in (0, 1):
                information = info_for(state, player)
                if not 1 <= information.allocations <= 60:
                    continue
                reference = {
                    tuple(sorted(c.code for c in allocation))
                    for allocation in enumerate_allocations(information)
                }
                for world in WorldGenerator(information).sample_many(6, rng):
                    assert world.allocation_key() in reference, (
                        "the sampler produced a world outside W(I)"
                    )
                    compared += 1
    assert compared > 500


def test_the_sampler_can_reach_every_reference_world() -> None:
    """No compatible allocation is silently unreachable."""
    tested = 0
    for seed in range(30):
        for state in walk(seed):
            information = info_for(state, 0)
            allocations = information.allocations
            if not 2 <= allocations <= 12:
                continue
            reference = {
                tuple(sorted(c.code for c in a))
                for a in enumerate_allocations(information)
            }
            drawn = {
                w.allocation_key()
                for w in WorldGenerator(information).sample_many(
                    200 * allocations, random.Random(7)
                )
            }
            assert drawn == reference
            tested += 1
            break
    assert tested > 5


# ======================================================================
# Corrupted worlds must fail
# ======================================================================
def corruptions(world: DeterminizedWorld, information: SearchInformation):
    """Ways to break a world that a validator must catch."""
    deck = sorted(information.knowledge.deck, key=lambda c: c.code)
    own = next(iter(information.view.hand), None)

    if world.opponent_hand:
        victim = min(world.opponent_hand, key=lambda c: c.code)
        yield "dropped a card", DeterminizedWorld(
            world.player, world.opponent_hand - {victim}, world.talon
        )
    if world.talon:
        yield "duplicated a card", DeterminizedWorld(
            world.player, world.opponent_hand, world.talon + (world.talon[0],)
        )
    if own is not None:
        yield "phantom public card in hiding", DeterminizedWorld(
            world.player, world.opponent_hand | {own}, world.talon
        )
    if len(world.talon) > 1:
        yield "trump moved off the bottom", DeterminizedWorld(
            world.player, world.opponent_hand, tuple(reversed(world.talon))
        )
    if world.talon and world.opponent_hand:
        card = world.talon[0]
        yield "card in two hidden places", DeterminizedWorld(
            world.player, world.opponent_hand | {card}, world.talon
        )
    for card in deck:
        if information.is_impossible(card, CardLocation.OPPONENT_HAND) and (
            card not in world.opponent_hand
        ):
            yield "impossible placement", DeterminizedWorld(
                world.player, world.opponent_hand | {card}, world.talon
            )
            break


def test_every_corruption_of_a_valid_world_is_rejected() -> None:
    rng = random.Random(31)
    kinds: set[str] = set()
    checked = 0
    for seed in range(15):
        for state in walk(seed):
            information = info_for(state, 0)
            world = WorldGenerator(information).sample(rng)
            validate_world(world, information)
            for label, broken in corruptions(world, information):
                assert not is_valid_world(broken, information), label
                kinds.add(label)
                checked += 1
    assert checked > 500
    assert kinds == {
        "dropped a card",
        "duplicated a card",
        "phantom public card in hiding",
        "trump moved off the bottom",
        "card in two hidden places",
        "impossible placement",
    }, kinds


# ======================================================================
# Adversarial constructions
# ======================================================================
def test_a_trump_not_at_the_bottom_world_is_rejected_explicitly() -> None:
    """The invariant the engine relies on but does not itself check."""
    state = make_state(
        hand0="6S 6C", hand1="7S 7C", trump=Suit.HEARTS, talon="8S 8C 8H",
        attacker=0,
    )
    information = info_for(state, 0)
    good = WorldGenerator(information).sample(random.Random(0))
    assert good.talon[-1] == Card.parse("8H")
    validate_world(good, information)

    for rotation in range(1, len(good.talon)):
        moved = good.talon[rotation:] + good.talon[:rotation]
        broken = DeterminizedWorld(0, good.opponent_hand, moved)
        assert not is_valid_world(broken, information), moved


def test_a_one_card_talon_holds_only_the_trump() -> None:
    state = make_state(
        hand0="6S 6C 6D", hand1="7S 7C", trump=Suit.HEARTS, talon="AH",
        attacker=0,
    )
    information = info_for(state, 0)
    world = WorldGenerator(information).sample(random.Random(1))
    validate_world(world, information)
    assert world.talon == (Card.parse("AH"),)


def test_an_empty_talon_yields_an_empty_tuple_not_a_stray_trump() -> None:
    state = make_state(
        hand0="6S 7H", hand1="8S 9H", trump=Suit.HEARTS, talon="", attacker=0
    )
    information = info_for(state, 0)
    world = WorldGenerator(information).sample(random.Random(1))
    assert world.talon == ()
    validate_world(world, information)


def test_forced_talon_cards_stay_in_the_talon() -> None:
    """Opponent-hand-saturated: everything undetermined must be talon."""
    state = make_state(
        hand0="6S 6C",
        hand1="9H",
        trump=Suit.HEARTS,
        talon="8C 8D 8S 8H",
        attacker=0,
    )
    tracker = CardTracker(0)
    tracker.update(observe(state, 0))
    for move in (AttackMove(Card.parse("6S")), TAKE, END_ATTACK):
        state = apply_move(state, move)
        tracker.update(observe(state, 0))

    information = SearchInformation.from_tracker(tracker, observe(state, 0))
    forced = information.certain_cards_in(CardLocation.TALON)
    world = WorldGenerator(information).sample(random.Random(2))
    validate_world(world, information)
    assert forced <= world.talon_cards


def test_symmetric_allocations_are_drawn_about_equally() -> None:
    """Four interchangeable cards, two slots: no allocation is favoured."""
    from collections import Counter

    state = make_state(
        hand0="6S 6C", hand1="7S 7C", trump=Suit.HEARTS, talon="8S 8C 8H",
        attacker=0,
    )
    information = info_for(state, 0)
    assert information.allocations == 6
    counts: Counter = Counter(
        w.allocation_key()
        for w in WorldGenerator(information).sample_many(6000, random.Random(5))
    )
    assert len(counts) == 6
    lowest, highest = min(counts.values()), max(counts.values())
    assert highest / lowest < 1.25, counts


def test_highly_dependent_positions_keep_hand_sizes_exact() -> None:
    """Where only one or two slots remain, dependence is at its strongest."""
    rng = random.Random(13)
    tested = 0
    for seed in range(25):
        for state in walk(seed):
            information = info_for(state, 0)
            slots = information.knowledge.opponent_slots
            if not 1 <= slots <= 2 or len(information.undetermined) < 4:
                continue
            for world in WorldGenerator(information).sample_many(20, rng):
                validate_world(world, information)
                assert len(world.opponent_hand) == (
                    information.view.opponent_hand_size
                )
            tested += 1
    assert tested > 5


@pytest.mark.parametrize("seed", [0, 3, 7])
def test_a_whole_game_generates_only_legal_worlds(seed: int) -> None:
    """End-to-end sweep: every position, both observers, several worlds."""
    rng = random.Random(seed + 100)
    positions = 0
    for state in walk(seed):
        for player in (0, 1):
            information = info_for(state, player)
            for world in WorldGenerator(information).sample_many(4, rng):
                validate_world(world, information)
            positions += 1
    assert positions > 50
