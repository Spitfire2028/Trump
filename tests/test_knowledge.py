"""Knowledge, deduction, incremental tracking and beliefs.

Two failure modes matter equally here, and the suite is built around
both:

* a tracker that **claims too much** — turning a likelihood into a fact,
  or deducing something the rules do not support;
* a tracker that **claims nothing** — returning "unknown" for everything,
  which trivially passes every leakage test and is useless.

So positive deductions and negative ones get equal weight, and there are
explicit non-vacuity tests proving real knowledge accumulates.
"""

from __future__ import annotations

import json
import math
import random

import pytest

from durakfish.cards import Card, Suit
from durakfish.cards.deck import standard_cards
from durakfish.game import Phase, apply_move, get_legal_moves, new_game
from durakfish.game.moves import END_ATTACK, TAKE, AttackMove
from durakfish.information import (
    RULES,
    BeliefState,
    CardLocation,
    CardTracker,
    Certainty,
    KnowledgeError,
    KnowledgeState,
    Observation,
    TrackerError,
    deduce,
    observe,
)
from durakfish.information.deduction import (
    RULE_OBSERVED,
    RULE_OPPONENT_RETAINS,
    RULE_OPPONENT_SATURATED,
    RULE_TALON_EXHAUSTED,
    RULE_TRUMP_BOTTOM,
)
from tests.reference_belief import reference_joint, reference_marginals
from tests.support import make_state

DECK36 = frozenset(standard_cards(36))


def track(state, player: int) -> CardTracker:
    tracker = CardTracker(player)
    tracker.update(observe(state, player))
    return tracker


def walk(seed: int):
    rng = random.Random(seed)
    state = new_game(seed=seed)
    while not state.is_over:
        yield state
        state = apply_move(state, rng.choice(get_legal_moves(state)))
    yield state


# ======================================================================
# The three categories stay apart
# ======================================================================
def test_observed_deduced_and_undetermined_are_distinct_categories() -> None:
    state = new_game(seed=1)
    knowledge = track(state, 0).knowledge
    observed = knowledge.known_cards()
    deduced = knowledge.deduced_cards()
    assert observed and deduced
    assert not (observed & deduced)
    assert not (observed & knowledge.candidates)
    assert not (deduced & knowledge.candidates)
    assert observed | deduced | knowledge.candidates == DECK36


def test_a_probability_is_never_reported_as_a_fact() -> None:
    """A 0.9 marginal must still read as UNKNOWN, not as knowledge."""
    state = make_state(
        hand0="6S 7S 8S 9S 10S JS",
        hand1="6C 6D 6H 7C 7D 7H",
        trump=Suit.HEARTS,
        talon="QS KH",
        attacker=0,
    )
    tracker = track(state, 0)
    knowledge, belief = tracker.knowledge, tracker.belief
    hot = [
        card
        for card in knowledge.candidates
        if belief.probability(card, CardLocation.OPPONENT_HAND) > 0.8
    ]
    assert hot, "no high-probability card to test with"
    for card in hot:
        assert knowledge.certainty(card) is Certainty.UNKNOWN
        assert knowledge.location_of(card) is None
        assert card not in knowledge.cards_in(CardLocation.OPPONENT_HAND)


def test_probability_zero_differs_from_not_estimated() -> None:
    state = new_game(seed=2)
    knowledge = track(state, 0).knowledge
    own = next(iter(knowledge.cards_in(CardLocation.SELF_HAND)))
    assert knowledge.location_certainty(own, CardLocation.TALON) is (
        Certainty.IMPOSSIBLE
    )
    assert knowledge.is_impossible(own, CardLocation.TALON)
    candidate = next(iter(knowledge.candidates))
    assert knowledge.location_certainty(candidate, CardLocation.TALON) is (
        Certainty.POSSIBLE
    )


def test_an_undetermined_card_is_impossible_in_every_public_location() -> None:
    """A candidate is hidden, so it cannot be on the table or in a pile.

    Without this, a bug that marked every placement POSSIBLE would leave
    the distributions summing to 1 and pass unnoticed.
    """
    for seed in range(10):
        tracker = CardTracker(0)
        for state in walk(seed):
            knowledge = tracker.update(observe(state, 0))
            for card in knowledge.candidates:
                for location in (
                    CardLocation.SELF_HAND,
                    CardLocation.TABLE,
                    CardLocation.DISCARD,
                ):
                    assert knowledge.is_impossible(card, location)
                    assert knowledge.location_certainty(card, location) is (
                        Certainty.IMPOSSIBLE
                    )
                assert knowledge.possible_locations(card) <= frozenset(
                    {CardLocation.OPPONENT_HAND, CardLocation.TALON}
                )


def test_the_retained_set_never_holds_a_card_that_has_become_public() -> None:
    """The tracker's carried state must mean what it says.

    A card the opponent took and has since played is public again. Leaving
    it in the retained set would be a stale claim about their hand, even
    if deduction happens to ignore it.
    """
    for seed in range(15):
        for player in (0, 1):
            tracker = CardTracker(player)
            for state in walk(seed):
                view = observe(state, player)
                tracker.update(view)
                public = set(view.hand) | set(view.table_cards) | set(view.discard)
                stale = tracker.retained_by_opponent & public
                assert not stale, f"stale retained cards: {stale}"


# ======================================================================
# Positive deductions
# ======================================================================
def test_own_hand_table_and_discard_are_observed_not_deduced() -> None:
    state = next(s for s in walk(4) if s.discard and s.table)
    knowledge = track(state, 0).knowledge
    for card in state.hands[0]:
        assert knowledge.location_of(card) is CardLocation.SELF_HAND
        assert knowledge.certainty(card) is Certainty.KNOWN
        assert knowledge.justification(card) == RULE_OBSERVED
    for card in state.discard:
        assert knowledge.location_of(card) is CardLocation.DISCARD


def test_the_trump_card_is_deduced_to_be_in_a_non_empty_talon() -> None:
    state = new_game(seed=5)
    knowledge = track(state, 0).knowledge
    trump = state.trump_card
    assert knowledge.location_of(trump) is CardLocation.TALON
    assert knowledge.certainty(trump) is Certainty.DEDUCED
    assert knowledge.justification(trump) == RULE_TRUMP_BOTTOM


def test_a_card_the_opponent_took_is_deduced_to_be_in_their_hand() -> None:
    state = make_state(
        hand0="6S 6C", hand1="9H 10H", trump=Suit.HEARTS, talon="8C 8D 8S 8H",
        attacker=0,
    )
    tracker = CardTracker(0)
    tracker.update(observe(state, 0))
    six = Card.parse("6S")
    assert tracker.knowledge.certainty(six) is Certainty.KNOWN  # still ours

    for move in (AttackMove(six), TAKE, END_ATTACK):
        state = apply_move(state, move)
        tracker.update(observe(state, 0))

    assert six in state.hands[1], "ground truth: the defender picked it up"
    knowledge = tracker.knowledge
    assert knowledge.location_of(six) is CardLocation.OPPONENT_HAND
    assert knowledge.certainty(six) is Certainty.DEDUCED
    assert knowledge.justification(six) == RULE_OPPONENT_RETAINS


def test_an_exhausted_talon_makes_the_whole_position_known() -> None:
    state = make_state(
        hand0="6S 7H", hand1="8S 9H", trump=Suit.HEARTS, talon="", attacker=0
    )
    knowledge = track(state, 0).knowledge
    assert knowledge.is_complete
    assert not knowledge.candidates
    opponent = knowledge.cards_in(CardLocation.OPPONENT_HAND)
    assert opponent == frozenset(state.hands[1]), "ground truth matches"
    for card in opponent:
        assert knowledge.justification(card) == RULE_TALON_EXHAUSTED


def test_a_fully_accounted_opponent_hand_places_the_rest_in_the_talon() -> None:
    """The mirror-image counting rule."""
    state = make_state(
        hand0="6S 6C 6D", hand1="9H", trump=Suit.HEARTS, talon="8C 8D 8S 8H",
        attacker=0,
    )
    tracker = CardTracker(0)
    tracker.update(observe(state, 0))
    for move in (AttackMove(Card.parse("6S")), TAKE, END_ATTACK):
        state = apply_move(state, move)
        tracker.update(observe(state, 0))

    knowledge = tracker.knowledge
    if knowledge.opponent_slots == 0 and knowledge.candidates:
        for card in knowledge.candidates:  # pragma: no cover - defensive
            assert knowledge.location_of(card) is CardLocation.TALON
    # Construct the saturated case directly to be sure it is exercised.
    saturated = deduce(
        Observation(
            player=0,
            own_hand=frozenset(map(Card.parse, ["6S", "6C"])),
            table=frozenset(),
            discard=frozenset(DECK36 - set(map(Card.parse, ["6S", "6C", "9H", "8H", "8C"]))),
            trump_card=Card.parse("8H"),
            talon_size=2,
            opponent_hand_size=1,
            retained_by_opponent=frozenset({Card.parse("9H")}),
            deck=DECK36,
        )
    )
    assert saturated.opponent_slots == 0
    assert saturated.is_complete
    assert saturated.location_of(Card.parse("8C")) is CardLocation.TALON
    assert saturated.justification(Card.parse("8C")) == RULE_OPPONENT_SATURATED


def test_every_deduction_names_a_documented_rule() -> None:
    for state in list(walk(6))[:40]:
        knowledge = track(state, 0).knowledge
        for card in knowledge.deduced_cards():
            rule = knowledge.justification(card)
            assert rule in RULES, f"undocumented rule {rule!r}"
            assert RULES[rule]


# ======================================================================
# Negative deductions — refusing to guess
# ======================================================================
def test_nothing_is_deduced_while_both_hidden_locations_remain_open() -> None:
    state = new_game(seed=7)
    knowledge = track(state, 0).knowledge
    assert knowledge.opponent_slots > 0 and knowledge.talon_slots > 0
    assert knowledge.candidates
    for card in knowledge.candidates:
        assert knowledge.certainty(card) is Certainty.UNKNOWN
        assert knowledge.possible_locations(card) == frozenset(
            {CardLocation.OPPONENT_HAND, CardLocation.TALON}
        )
        assert knowledge.is_possible(card, CardLocation.OPPONENT_HAND)
        assert knowledge.is_possible(card, CardLocation.TALON)


def test_no_deduced_card_is_ever_actually_elsewhere() -> None:
    """Soundness against ground truth: a DEDUCED fact must be true."""
    checked = 0
    truth = {
        CardLocation.SELF_HAND: lambda s, p: set(s.hands[p]),
        CardLocation.OPPONENT_HAND: lambda s, p: set(s.hands[1 - p]),
        CardLocation.TALON: lambda s, p: set(s.talon),
        CardLocation.TABLE: lambda s, p: set(s.table_cards),
        CardLocation.DISCARD: lambda s, p: set(s.discard),
    }
    for seed in range(25):
        for player in (0, 1):
            tracker = CardTracker(player)
            for state in walk(seed):
                knowledge = tracker.update(observe(state, player))
                for card, placement in knowledge.settled.items():
                    assert card in truth[placement.location](state, player), (
                        f"{card} claimed in {placement.location.value} by "
                        f"{placement.rule!r} but is not there"
                    )
                    checked += 1
    assert checked > 100_000


def test_declining_to_beat_licenses_no_deduction() -> None:
    """Behaviour is evidence, not proof, and this layer makes no inference."""
    state = make_state(
        hand0="6S 6C", hand1="AH 9H", trump=Suit.HEARTS, talon="8C 8D 8S 8H",
        attacker=0,
    )
    tracker = CardTracker(0)
    tracker.update(observe(state, 0))
    before = tracker.knowledge.candidates
    state = apply_move(state, AttackMove(Card.parse("6S")))
    state = apply_move(state, TAKE)  # the opponent declines to defend
    tracker.update(observe(state, 0))
    # Nothing about their remaining cards may be concluded from the choice.
    still_hidden = tracker.knowledge.candidates
    assert still_hidden == before - {Card.parse("6S")} or still_hidden <= before
    for card in still_hidden:
        assert tracker.knowledge.certainty(card) is Certainty.UNKNOWN


# ======================================================================
# Non-vacuity — knowledge really accumulates
# ======================================================================
def test_knowledge_becomes_strictly_more_precise_over_a_game() -> None:
    """A tracker that always answers 'unknown' would fail this."""
    improved = 0
    for seed in range(20):
        tracker = CardTracker(0)
        previous = None
        for state in walk(seed):
            knowledge = tracker.update(observe(state, 0))
            if previous is not None and len(knowledge.candidates) < previous:
                improved += 1
            previous = len(knowledge.candidates)
        assert tracker.knowledge.is_complete, "the endgame must be fully known"
    assert improved > 100


def test_a_public_event_can_eliminate_possibilities() -> None:
    state = make_state(
        hand0="6S 6C", hand1="9H 10H", trump=Suit.HEARTS, talon="8C 8D 8S 8H",
        attacker=0,
    )
    tracker = CardTracker(0)
    tracker.update(observe(state, 0))
    six = Card.parse("6S")
    before = tracker.knowledge

    for move in (AttackMove(six), TAKE, END_ATTACK):
        state = apply_move(state, move)
    after = tracker.update(observe(state, 0))

    assert before.location_of(six) is CardLocation.SELF_HAND
    assert after.location_of(six) is CardLocation.OPPONENT_HAND
    assert after.is_possible(six, CardLocation.OPPONENT_HAND)
    assert after.is_impossible(six, CardLocation.TALON)


def test_uncertainty_falls_to_zero_by_the_end_of_a_game() -> None:
    for seed in range(10):
        tracker = CardTracker(0)
        first = last = None
        for state in walk(seed):
            tracker.update(observe(state, 0))
            bits = tracker.belief.uncertainty_bits
            first = bits if first is None else first
            last = bits
        assert first > 0, "the opening must be uncertain"
        assert last == 0.0, "the end must be certain"


# ======================================================================
# Invariants
# ======================================================================
def test_card_conservation_and_slot_consistency_hold_throughout() -> None:
    for seed in range(15):
        for player in (0, 1):
            tracker = CardTracker(player)
            for state in walk(seed):
                knowledge = tracker.update(observe(state, player))
                knowledge.validate()
                assert set(knowledge.settled) | knowledge.candidates == DECK36
                assert len(knowledge.candidates) == (
                    knowledge.opponent_slots + knowledge.talon_slots
                )
                assert knowledge.hidden_card_count == (
                    len(state.hands[1 - player]) + len(state.talon)
                )


def test_hand_sizes_are_respected() -> None:
    for seed in range(12):
        tracker = CardTracker(0)
        for state in walk(seed):
            knowledge = tracker.update(observe(state, 0))
            assert len(knowledge.cards_in(CardLocation.SELF_HAND)) == len(
                state.hands[0]
            )
            assert (
                len(knowledge.cards_in(CardLocation.OPPONENT_HAND))
                + knowledge.opponent_slots
                == len(state.hands[1])
            )
            assert (
                len(knowledge.cards_in(CardLocation.TALON)) + knowledge.talon_slots
                == len(state.talon)
            )


def test_a_card_can_never_be_in_two_places() -> None:
    for seed in range(10):
        tracker = CardTracker(0)
        for state in walk(seed):
            knowledge = tracker.update(observe(state, 0))
            seen: dict[Card, CardLocation] = {}
            for location in CardLocation:
                for card in knowledge.cards_in(location):
                    assert card not in seen
                    seen[card] = location


def test_an_inconsistent_observation_is_rejected_loudly() -> None:
    with pytest.raises(KnowledgeError, match="inconsistent"):
        deduce(
            Observation(
                player=0,
                own_hand=frozenset(),
                table=frozenset(),
                discard=frozenset(),
                trump_card=Card.parse("6H"),
                talon_size=0,
                opponent_hand_size=0,
                retained_by_opponent=frozenset({Card.parse("6S"), Card.parse("6C")}),
                deck=DECK36,
            )
        )


def test_a_broken_knowledge_state_cannot_be_constructed() -> None:
    with pytest.raises(KnowledgeError, match="conservation"):
        KnowledgeState(
            player=0, settled={}, candidates=frozenset(), opponent_slots=0,
            talon_slots=0, deck=DECK36,
        )
    with pytest.raises(KnowledgeError, match="negative"):
        KnowledgeState(
            player=0, settled={}, candidates=DECK36, opponent_slots=-1,
            talon_slots=37, deck=DECK36,
        )


# ======================================================================
# Beliefs
# ======================================================================
def test_marginals_are_well_formed() -> None:
    for seed in range(10):
        tracker = CardTracker(0)
        for state in walk(seed):
            tracker.update(observe(state, 0))
            belief = tracker.belief
            for card in DECK36:
                total = sum(belief.distribution(card).values())
                assert math.isclose(total, 1.0, abs_tol=1e-9), card
                for value in belief.distribution(card).values():
                    assert 0.0 <= value <= 1.0


def test_impossible_locations_have_probability_zero_and_known_ones_one() -> None:
    state = next(s for s in walk(8) if s.discard)
    tracker = track(state, 0)
    knowledge, belief = tracker.knowledge, tracker.belief
    for card in knowledge.cards_in(CardLocation.SELF_HAND):
        assert belief.probability(card, CardLocation.SELF_HAND) == 1.0
        assert belief.probability(card, CardLocation.OPPONENT_HAND) == 0.0
        assert belief.probability(card, CardLocation.TALON) == 0.0
    for card in knowledge.cards_in(CardLocation.DISCARD):
        assert belief.probability(card, CardLocation.DISCARD) == 1.0


def test_marginals_match_the_brute_force_reference() -> None:
    """Closed form versus enumeration, on positions small enough to enumerate."""
    compared = 0
    for seed in range(60):
        tracker = CardTracker(0)
        for state in walk(seed):
            knowledge = tracker.update(observe(state, 0))
            if not 1 <= len(knowledge.candidates) <= 12:
                continue
            belief = tracker.belief
            for location in (CardLocation.OPPONENT_HAND, CardLocation.TALON):
                expected = reference_marginals(knowledge, location)
                for card, value in expected.items():
                    assert math.isclose(
                        belief.probability(card, location), value, abs_tol=1e-9
                    ), (card, location)
                    compared += 1
            break  # one position per game keeps the enumeration cheap
    assert compared > 200


def test_joint_probabilities_match_the_reference_and_are_not_independent() -> None:
    """The dependence is real, and the closed form captures it."""
    compared = dependent = 0
    for seed in range(40):
        tracker = CardTracker(0)
        for state in walk(seed):
            knowledge = tracker.update(observe(state, 0))
            if not 2 <= len(knowledge.candidates) <= 10:
                continue
            belief = tracker.belief
            cards = frozenset(
                sorted(knowledge.candidates, key=lambda c: c.code)[:2]
            )
            for location in (CardLocation.OPPONENT_HAND, CardLocation.TALON):
                exact = reference_joint(knowledge, cards, location)
                assert math.isclose(
                    belief.probability_all_in(cards, location), exact, abs_tol=1e-9
                )
                naive = math.prod(belief.probability(c, location) for c in cards)
                if not math.isclose(naive, exact, abs_tol=1e-9):
                    dependent += 1
                compared += 1
            break
    assert compared > 40
    assert dependent > 20, "multiplying marginals never differed; check the model"


def test_a_worked_dependence_example() -> None:
    """Four undetermined cards, two opponent slots: 1/2 each, 1/6 together."""
    cards = sorted(DECK36, key=lambda c: c.code)
    candidates = frozenset(cards[:4])
    settled = {
        card: __import__(
            "durakfish.information.knowledge", fromlist=["Placement"]
        ).Placement(CardLocation.DISCARD, Certainty.KNOWN, RULE_OBSERVED)
        for card in cards[4:]
    }
    knowledge = KnowledgeState(
        player=0, settled=settled, candidates=candidates,
        opponent_slots=2, talon_slots=2, deck=DECK36,
    )
    belief = BeliefState.from_knowledge(knowledge)
    pair = frozenset(list(candidates)[:2])
    for card in candidates:
        assert belief.probability(card, CardLocation.OPPONENT_HAND) == 0.5
    assert math.isclose(
        belief.probability_all_in(pair, CardLocation.OPPONENT_HAND), 1 / 6
    )
    assert not math.isclose(
        belief.probability_all_in(pair, CardLocation.OPPONENT_HAND), 0.25
    )
    assert belief.allocations == 6
    assert math.isclose(belief.uncertainty_bits, math.log2(6))


def test_expected_counts_are_exact_despite_dependence() -> None:
    state = new_game(seed=9)
    tracker = track(state, 0)
    knowledge, belief = tracker.knowledge, tracker.belief
    expected = belief.expected_count(knowledge.candidates, CardLocation.OPPONENT_HAND)
    assert math.isclose(expected, knowledge.opponent_slots)


def test_joint_probability_rejects_a_public_location() -> None:
    tracker = track(new_game(seed=1), 0)
    with pytest.raises(KnowledgeError, match="hidden locations"):
        tracker.belief.probability_all_in(frozenset(), CardLocation.DISCARD)


# ======================================================================
# Incrementality, replay, checkpointing
# ======================================================================
def test_incremental_tracking_equals_a_full_rebuild_after_every_event() -> None:
    for seed in range(20):
        for player in (0, 1):
            incremental = CardTracker(player)
            for state in walk(seed):
                view = observe(state, player)
                live = incremental.update(view)
                rebuilt = CardTracker.rebuild(view).knowledge
                assert live == rebuilt
                assert live.key() == rebuilt.key()


def test_replaying_the_views_reproduces_the_tracker() -> None:
    for seed in range(15):
        views = [observe(state, 0) for state in walk(seed)]
        live = CardTracker(0)
        for view in views:
            live.update(view)
        replayed = CardTracker.replay(0, views)
        assert live == replayed
        assert live.key() == replayed.key()
        assert live.knowledge == replayed.knowledge


def test_a_checkpoint_round_trips_to_an_equivalent_tracker() -> None:
    for seed in range(10):
        tracker = CardTracker(0)
        for state in walk(seed):
            tracker.update(observe(state, 0))
        payload = json.loads(json.dumps(tracker.to_dict()))
        restored = CardTracker.from_dict(payload, deck=DECK36)
        assert restored.key() == tracker.key()
        assert restored.knowledge == tracker.knowledge
        assert restored.belief.to_dict() == tracker.belief.to_dict()


def test_a_knowledge_state_round_trips_through_json() -> None:
    tracker = track(new_game(seed=3), 0)
    payload = json.loads(json.dumps(tracker.knowledge.to_dict()))
    restored = KnowledgeState.from_dict(payload, DECK36)
    assert restored == tracker.knowledge
    assert restored.key() == tracker.knowledge.key()


def test_a_checkpoint_contains_no_hidden_card() -> None:
    for seed in range(10):
        tracker = CardTracker(0)
        for state in walk(seed):
            tracker.update(observe(state, 0))
            payload = json.dumps(tracker.to_dict())
            view = observe(state, 0)
            for card in state.hands[1]:
                if card in view.unobserved_cards():
                    assert card.short not in json.loads(payload).get("retained", [])


# ======================================================================
# Observer specificity and misuse
# ======================================================================
def test_two_players_hold_genuinely_different_knowledge() -> None:
    differing = 0
    for seed in range(10):
        a, b = CardTracker(0), CardTracker(1)
        for state in walk(seed):
            first = a.update(observe(state, 0))
            second = b.update(observe(state, 1))
            if first.key() != second.key():
                differing += 1
            assert first.player == 0 and second.player == 1
    assert differing > 100


def test_a_tracker_refuses_another_players_view() -> None:
    state = new_game(seed=1)
    tracker = CardTracker(0)
    with pytest.raises(TrackerError, match="P1's view"):
        tracker.update(observe(state, 1))


def test_a_tracker_refuses_a_shrinking_event_stream() -> None:
    states = list(walk(2))
    tracker = CardTracker(0)
    tracker.update(observe(states[10], 0))
    with pytest.raises(TrackerError, match="different game"):
        tracker.update(observe(states[0], 0))


def test_an_unused_tracker_reports_nothing_rather_than_an_empty_state() -> None:
    tracker = CardTracker(0)
    with pytest.raises(TrackerError, match="has not observed"):
        _ = tracker.knowledge


def test_the_retained_set_cannot_be_mutated_from_outside() -> None:
    tracker = track(new_game(seed=1), 0)
    snapshot = tracker.retained_by_opponent
    assert isinstance(snapshot, frozenset)
    with pytest.raises(AttributeError):
        snapshot.add(Card.parse("6S"))  # type: ignore[attr-defined]
