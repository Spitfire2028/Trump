"""The information-aware search contract.

The contract is the single door into a future belief-aware search, so the
questions worth asking are about what can and cannot come through it:

* does it carry the four information levels without collapsing them?
* is it a function of the observer's legal information *only*?
* can anything reach hidden state through it, by any attribute path?
* does building it leave the existing SearchBot's play untouched?

The expected values here are computed by an independent reference that
re-derives certainty and marginals from the view's raw fields, so the
production pipeline is not used to generate its own expected results.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from durakfish.ai import SearchBot, SearchConfig
from durakfish.cards import Card
from durakfish.cards.deck import standard_cards
from durakfish.game import apply_move, get_legal_moves, new_game
from durakfish.game.state import add_cards
from durakfish.information import (
    FORBIDDEN_ATTRIBUTES,
    BeliefState,
    CardLocation,
    CardTracker,
    Certainty,
    InformationClass,
    SearchContractError,
    SearchInformation,
    observe,
)
from tests.test_phase3_audit import (
    FORBIDDEN_TYPES,
    cards_reachable_from,
    permute_hidden,
    reachable,
    rewrite_opponent_draw_identities,
)

REPO = Path(__file__).resolve().parent.parent
DECK36 = frozenset(standard_cards(36))


def walk(seed: int):
    rng = random.Random(seed)
    state = new_game(seed=seed)
    while not state.is_over:
        yield state
        state = apply_move(state, rng.choice(get_legal_moves(state)))
    yield state


def contract_for(state, player: int) -> SearchInformation:
    return SearchInformation.from_view(observe(state, player))


# ======================================================================
# Independent reference
# ======================================================================
def reference_expectations(view) -> tuple[dict[Card, str], dict[Card, float]]:
    """Re-derive classification and marginals from the view's raw fields.

    Written from ``docs/knowledge.md`` rather than by calling the tracker,
    so agreement is evidence rather than tautology.
    """
    opponent = view.opponent
    retained: set[Card] = set()
    for event in view.public_history:
        if event.player == opponent:
            retained -= set(event.revealed)
        if event.taken_by == opponent:
            retained |= set(event.taken)
        elif event.taken:
            retained -= set(event.taken)
        retained -= set(event.discarded)

    public = set(view.hand) | set(view.table_cards) | set(view.discard)
    retained -= public

    certain = {card: "certain" for card in public}
    deduced: dict[Card, str] = {}
    if view.talon_size > 0 and view.trump_card not in public:
        deduced[view.trump_card] = "deduced"
    for card in retained:
        deduced.setdefault(card, "deduced")

    settled = set(certain) | set(deduced)
    candidates = set(DECK36) - settled
    opponent_slots = view.opponent_hand_size - len(retained)
    talon_slots = view.talon_size - (1 if view.trump_card in deduced else 0)

    classes = {**certain, **deduced}
    if candidates and talon_slots == 0:
        for card in candidates:
            classes[card] = "deduced"
        candidates, opponent_slots = set(), 0
    elif candidates and opponent_slots == 0:
        for card in candidates:
            classes[card] = "deduced"
        candidates, talon_slots = set(), 0
    for card in candidates:
        classes[card] = "probabilistic"

    total = len(candidates)
    marginals = {
        card: (opponent_slots / total if total else 0.0) for card in candidates
    }
    return classes, marginals


def test_the_contract_matches_an_independent_reference() -> None:
    compared = 0
    for seed in range(15):
        for state in walk(seed):
            for player in (0, 1):
                view = observe(state, player)
                contract = SearchInformation.from_view(view)
                classes, marginals = reference_expectations(view)
                for card in DECK36:
                    assert contract.classify(card).value == classes[card], card
                for card, expected in marginals.items():
                    assert (
                        abs(
                            contract.probability(card, CardLocation.OPPONENT_HAND)
                            - expected
                        )
                        < 1e-9
                    ), card
                compared += 1
            break  # one position per game keeps the reference cheap
    assert compared > 20


# ======================================================================
# The four levels stay distinct
# ======================================================================
def test_all_four_information_classes_are_reachable() -> None:
    seen: set[InformationClass] = set()
    for seed in range(10):
        for state in walk(seed):
            contract = contract_for(state, 0)
            seen.update(contract.classify(card) for card in DECK36)
    assert InformationClass.CERTAIN in seen
    assert InformationClass.DEDUCED in seen
    assert InformationClass.PROBABILISTIC in seen


def test_a_high_probability_card_is_never_classified_as_certain() -> None:
    """The distinction the contract exists to preserve."""
    found = 0
    for seed in range(20):
        for state in walk(seed):
            contract = contract_for(state, 0)
            for card in contract.undetermined:
                probability = contract.probability(card, CardLocation.OPPONENT_HAND)
                if probability > 0.8:
                    assert contract.classify(card) is InformationClass.PROBABILISTIC
                    assert contract.location_of(card) is None
                    assert card not in contract.certain_cards_in(
                        CardLocation.OPPONENT_HAND
                    )
                    found += 1
    assert found > 20, "no high-probability card arose to test the distinction"


def test_certain_and_deduced_cards_carry_probability_one() -> None:
    for seed in range(8):
        for state in walk(seed):
            contract = contract_for(state, 0)
            for card in DECK36:
                location = contract.location_of(card)
                if location is None:
                    continue
                assert contract.probability(card, location) == 1.0
                assert contract.classify(card) in (
                    InformationClass.CERTAIN,
                    InformationClass.DEDUCED,
                )


def test_certain_and_deduced_are_not_conflated() -> None:
    state = new_game(seed=5)
    contract = contract_for(state, 0)
    own = next(iter(contract.certain_cards_in(CardLocation.SELF_HAND)))
    assert contract.classify(own) is InformationClass.CERTAIN
    assert contract.classify(state.trump_card) is InformationClass.DEDUCED
    assert contract.justification(state.trump_card) == "trump-at-talon-bottom"
    assert contract.justification(own) == "observed"


def test_impossible_placements_stay_impossible() -> None:
    for seed in range(8):
        for state in walk(seed):
            contract = contract_for(state, 0)
            for card in contract.certain_cards_in(CardLocation.SELF_HAND):
                for location in (CardLocation.OPPONENT_HAND, CardLocation.TALON):
                    assert contract.is_impossible(card, location)
                    assert contract.probability(card, location) == 0.0
            for card in contract.undetermined:
                for location in (
                    CardLocation.SELF_HAND,
                    CardLocation.TABLE,
                    CardLocation.DISCARD,
                ):
                    assert contract.is_impossible(card, location)


def test_probabilities_are_normalised() -> None:
    for seed in range(8):
        for state in walk(seed):
            contract = contract_for(state, 0)
            for card in DECK36:
                total = sum(
                    contract.probability(card, location) for location in CardLocation
                )
                assert abs(total - 1.0) < 1e-9, card


def test_joint_probability_still_respects_dependence() -> None:
    """The contract must not quietly re-introduce an independence bug."""
    import math

    differing = 0
    for seed in range(20):
        for state in walk(seed):
            contract = contract_for(state, 0)
            if len(contract.undetermined) < 2:
                continue
            pair = frozenset(sorted(contract.undetermined, key=lambda c: c.code)[:2])
            joint = contract.probability_all_in(pair, CardLocation.OPPONENT_HAND)
            naive = math.prod(
                contract.probability(c, CardLocation.OPPONENT_HAND) for c in pair
            )
            if abs(joint - naive) > 1e-9:
                differing += 1
            break
    assert differing > 5


# ======================================================================
# Hidden-state noninterference
# ======================================================================
def fingerprint_bundle(state, player: int) -> tuple:
    """Many independent channels, not one field."""
    contract = contract_for(state, player)
    cards = sorted(DECK36, key=lambda c: c.code)
    return (
        contract.fingerprint(),
        json.dumps(contract.to_dict(), sort_keys=True),
        repr(contract),
        contract.describe(),
        hash(contract),
        tuple((c.code, contract.classify(c).value) for c in cards),
        tuple(
            (c.code, tuple(sorted(x.value for x in contract.possible_locations(c))))
            for c in cards
        ),
        tuple(
            (c.code, round(contract.probability(c, CardLocation.OPPONENT_HAND), 12))
            for c in cards
        ),
        tuple((c.code, contract.justification(c)) for c in cards),
        tuple(sorted(c.code for c in contract.undetermined)),
        contract.allocations,
        round(contract.uncertainty_bits, 12),
        contract.is_open_information,
    )


@pytest.mark.parametrize("mode", ["opponent", "talon", "both", "history"])
def test_indistinguishable_worlds_produce_identical_contracts(mode) -> None:
    """The critical invariant, over four kinds of hidden difference."""
    rng = random.Random(31337)
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
            if mode != "history" and (not opponent or not body):
                continue

            baseline = fingerprint_bundle(state, player)
            for _ in range(2):
                if mode == "history":
                    twin = rewrite_opponent_draw_identities(state, player)
                else:
                    if mode == "opponent":
                        new_opp, new_body = list(opponent), list(body)
                        rng.shuffle(new_opp)
                    elif mode == "talon":
                        new_opp, new_body = list(opponent), list(body)
                        rng.shuffle(new_body)
                    else:
                        pool = opponent + body
                        rng.shuffle(pool)
                        new_opp = pool[: len(opponent)]
                        new_body = pool[len(opponent) :]
                    twin = state.replace(
                        hands=tuple(
                            state.hands[p]
                            if p == player
                            else add_cards((), fixed_opp + new_opp)
                            for p in range(2)
                        ),
                        talon=tuple(fixed_body + new_body) + state.talon[-1:],
                    )
                twin.validate()
                assert observe(twin, player) == view, "premise failed"
                assert fingerprint_bundle(twin, player) == baseline, (
                    f"{mode} permutation changed the search contract"
                )
                assert contract_for(twin, player) == contract_for(state, player)
                compared += 1
            break
    assert compared > 15


def test_a_different_information_set_can_produce_a_different_contract() -> None:
    """Non-vacuity: the fingerprint must not be constant."""
    prints = set()
    for seed in range(6):
        for state in walk(seed):
            prints.add(contract_for(state, 0).fingerprint())
    assert len(prints) > 100


# ======================================================================
# No escape hatch
# ======================================================================
def test_no_forbidden_attribute_exists_anywhere_in_the_graph() -> None:
    """Machine-checkable, over the whole reachable object graph."""
    for seed in range(6):
        for state in walk(seed):
            contract = contract_for(state, 0)
            for obj in reachable(contract):
                if isinstance(obj, (int, float, str, bytes, bool, type(None))):
                    continue
                for banned in FORBIDDEN_ATTRIBUTES:
                    assert not hasattr(obj, banned), (
                        f"{type(obj).__name__} exposes {banned!r}"
                    )


def test_no_forbidden_type_is_reachable_from_a_contract() -> None:
    audited = 0
    for seed in range(6):
        for player in (0, 1):
            for state in walk(seed):
                view = observe(state, player)
                contract = SearchInformation.from_view(view)
                for obj in reachable(contract):
                    assert not isinstance(obj, FORBIDDEN_TYPES), (
                        f"{type(obj).__name__} reachable from the contract"
                    )
                named = cards_reachable_from(contract)
                assert named <= DECK36
                for card in contract.knowledge.known_cards():
                    assert card in view.observed_cards()
                audited += 1
    assert audited > 300


def test_the_contract_module_never_names_the_state_layer() -> None:
    """`observe()` is the only permitted reference; this is downstream."""
    import ast

    path = REPO / "src" / "durakfish" / "information" / "search_contract.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    for banned in ("GameState", "hands", "talon", "apply_move", "new_game"):
        assert banned not in names, f"search_contract.py references {banned!r}"


def test_a_contract_cannot_be_built_from_mismatched_parts() -> None:
    state = new_game(seed=2)
    tracker = CardTracker(0)
    tracker.update(observe(state, 0))
    with pytest.raises(SearchContractError, match="paired with"):
        SearchInformation.from_tracker(tracker, observe(state, 1))

    later = list(walk(2))[20]
    with pytest.raises(SearchContractError, match="update the tracker"):
        SearchInformation.from_tracker(tracker, observe(later, 0))

    knowledge = tracker.knowledge
    with pytest.raises(SearchContractError, match="derived from this contract"):
        SearchInformation(
            view=observe(state, 0),
            knowledge=knowledge,
            belief=BeliefState.from_knowledge(
                CardTracker.rebuild(observe(state, 0)).knowledge
            ),
        )


# ======================================================================
# Immutability and repeatability
# ======================================================================
def test_a_contract_is_frozen() -> None:
    import dataclasses

    contract = contract_for(new_game(seed=1), 0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        contract.view = None  # type: ignore[misc]


def test_advancing_the_source_tracker_cannot_alter_an_existing_contract() -> None:
    """A running tracker must not drag its old snapshots along."""
    states = list(walk(4))
    tracker = CardTracker(0)
    tracker.update(observe(states[5], 0))
    contract = SearchInformation.from_tracker(tracker, observe(states[5], 0))
    before = contract.fingerprint()

    for state in states[6:20]:
        tracker.update(observe(state, 0))

    assert contract.fingerprint() == before
    assert tracker.events_seen > 5, "the tracker really did move on"
    later = SearchInformation.from_tracker(tracker, observe(states[19], 0))
    assert later.fingerprint() != before


def test_rebuilding_the_contract_gives_an_identical_result() -> None:
    for seed in range(10):
        for state in walk(seed):
            view = observe(state, 0)
            first = SearchInformation.from_view(view)
            second = SearchInformation.from_view(view)
            assert first == second
            assert first.fingerprint() == second.fingerprint()
            assert hash(first) == hash(second)


def test_both_construction_paths_agree() -> None:
    for seed in range(10):
        tracker = CardTracker(0)
        for state in walk(seed):
            view = observe(state, 0)
            tracker.update(view)
            assert SearchInformation.from_tracker(tracker, view) == (
                SearchInformation.from_view(view)
            )


def test_serialization_round_trips_the_observable_content() -> None:
    for seed in range(8):
        for state in walk(seed):
            contract = contract_for(state, 0)
            payload = json.loads(json.dumps(contract.to_dict()))
            assert payload["player"] == contract.player
            assert payload["knowledge"] == contract.knowledge.to_dict()
            assert payload["belief"] == contract.belief.to_dict()
            rebuilt = contract_for(state, 0)
            assert json.dumps(rebuilt.to_dict(), sort_keys=True) == json.dumps(
                payload, sort_keys=True
            )


def test_a_serialized_contract_contains_no_unobserved_card() -> None:
    for seed in range(10):
        for state in walk(seed):
            view = observe(state, 0)
            payload = json.dumps(contract_for(state, 0).to_dict())
            for card in view.unobserved_cards():
                # A card name may legitimately appear as a classification
                # key (the deck is public); what must not appear is any
                # claim about where it is.
                classification = json.loads(payload)["classification"][card.short]
                assert classification in ("probabilistic", "deduced")


def test_contract_keys_are_stable_across_processes() -> None:
    expected = [
        json.dumps(contract_for(state, 0).to_dict(), sort_keys=True)
        for state in list(walk(3))[:5]
    ]
    code = (
        "import sys, json, random; sys.path.insert(0, 'src');"
        "from durakfish.game import new_game, get_legal_moves, apply_move;"
        "from durakfish.information import SearchInformation, observe;"
        "rng=random.Random(3); s=new_game(seed=3); out=[]\n"
        "while not s.is_over and len(out) < 5:\n"
        "    out.append(json.dumps(SearchInformation.from_view(observe(s,0))"
        ".to_dict(), sort_keys=True))\n"
        "    s = apply_move(s, rng.choice(get_legal_moves(s)))\n"
        "print(json.dumps(out))"
    )
    outputs = []
    for hash_seed in ("0", "13579"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=str(REPO),
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin",
                 "PYTHONHASHSEED": hash_seed},
        )
        assert result.returncode == 0, result.stderr
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1] == expected


# ======================================================================
# SearchBot regression
# ======================================================================
def test_existing_search_bot_decisions_are_completely_unchanged() -> None:
    """Phase 7.1 must not alter playing strength or behaviour at all."""
    bot = SearchBot(SearchConfig(depth=3), "s")
    bot.reset(random.Random(0))
    checked = 0
    for seed in range(20):
        for state in walk(seed):
            player = state.current_player
            if player is None:
                continue
            view, legal = observe(state, player), get_legal_moves(state)
            before = bot.choose(view, legal)

            # Build the contract alongside, exactly as a future search would.
            contract = SearchInformation.from_view(view)
            assert contract.player == player

            after = bot.choose(view, legal)
            assert after == before, "building a contract perturbed the search"
            checked += 1
    assert checked > 1500


def test_whole_games_are_byte_identical_with_contracts_in_play() -> None:
    from durakfish.ai import GreedyBot, RandomBot
    from durakfish.simulation import play_game

    config = SearchConfig(depth=3)
    for label, agents in (
        ("search vs random", lambda: [SearchBot(config, "s"), RandomBot("r")]),
        ("search vs greedy", lambda: [SearchBot(config, "s"), GreedyBot("g")]),
    ):
        for seed in range(6):
            baseline = play_game(agents(), seed=seed).to_dict()
            # Construct a contract at every position of a replay, then
            # replay the same game again and compare.
            record = play_game(agents(), seed=seed)
            for entry_state in walk(seed):
                SearchInformation.from_view(observe(entry_state, 0))
            assert play_game(agents(), seed=seed).to_dict() == baseline
            assert record.to_dict() == baseline, label
