"""Phase 6 audit.

A knowledge layer is the most dangerous place so far for a leak: unlike
the search, it is *supposed* to conclude things about hidden cards, so a
conclusion drawn from the wrong source would look like success.

The audit therefore asks the question the other way round. Rather than
checking that the tracker's conclusions are true — ``test_knowledge.py``
does that against ground truth — it checks that the tracker's conclusions
depend on **nothing but public information**. Two worlds that look
identical to a player must produce byte-identical knowledge, beliefs,
keys and serialisations, even when the hidden cards differ completely.

Reachability machinery is reused from the Phase 3 audit, so this layer is
held to literally the same standard as everything before it.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from durakfish.cards import Card
from durakfish.cards.deck import standard_cards
from durakfish.game import apply_move, get_legal_moves, new_game
from durakfish.game.state import add_cards
from durakfish.information import (
    RULES,
    BeliefState,
    CardLocation,
    CardTracker,
    Certainty,
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


def tracked(state, player: int) -> CardTracker:
    """A tracker fed the whole history of a position in one update."""
    tracker = CardTracker(player)
    tracker.update(observe(state, player))
    return tracker


def fingerprint(state, player: int) -> tuple:
    """Every observable output of the knowledge layer, in one value."""
    tracker = tracked(state, player)
    knowledge, belief = tracker.knowledge, tracker.belief
    return (
        knowledge.key(),
        json.dumps(knowledge.to_dict(), sort_keys=True),
        tuple(sorted(c.code for c in knowledge.known_cards())),
        tuple(sorted(c.code for c in knowledge.deduced_cards())),
        tuple(sorted(c.code for c in knowledge.candidates)),
        knowledge.opponent_slots,
        knowledge.talon_slots,
        tuple(
            (c.code, tuple(sorted(loc.value for loc in knowledge.possible_locations(c))))
            for c in sorted(DECK36, key=lambda c: c.code)
        ),
        tuple(
            (c.code, int(knowledge.certainty(c)))
            for c in sorted(DECK36, key=lambda c: c.code)
        ),
        tuple(
            round(belief.probability(c, CardLocation.OPPONENT_HAND), 12)
            for c in sorted(DECK36, key=lambda c: c.code)
        ),
        tuple(
            round(belief.probability(c, CardLocation.TALON), 12)
            for c in sorted(DECK36, key=lambda c: c.code)
        ),
        belief.allocations,
        round(belief.uncertainty_bits, 12),
        json.dumps(belief.to_dict(), sort_keys=True),
        json.dumps(tracker.to_dict(), sort_keys=True),
        tracker.key(),
        repr(knowledge),
        repr(belief),
        knowledge.describe(),
    )


# ======================================================================
# Indistinguishability
# ======================================================================
@pytest.mark.parametrize("mode", ["opponent", "talon", "both", "history"])
def test_knowledge_is_identical_for_indistinguishable_worlds(mode) -> None:
    """The central property, over four kinds of hidden permutation."""
    rng = random.Random(4242)
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

            baseline = fingerprint(state, player)
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
                assert fingerprint(twin, player) == baseline, (
                    f"{mode} permutation changed the knowledge layer"
                )
                compared += 1
            break  # one position per game keeps the fingerprints cheap
    assert compared > 15


def test_permuting_hidden_cards_never_changes_a_deduction() -> None:
    """Checked at every position of many games, not just a sample."""
    rng = random.Random(77)
    compared = 0
    for seed in range(8):
        for state in walk(seed):
            player = state.current_player
            if player is None or not state.talon:
                continue
            baseline = tracked(state, player).knowledge
            twin = permute_hidden(state, player, rng)
            assert tracked(twin, player).knowledge == baseline
            compared += 1
    assert compared > 300


# ======================================================================
# Reachability
# ======================================================================
def test_nothing_forbidden_is_reachable_from_the_knowledge_layer() -> None:
    audited = 0
    for seed in range(8):
        for player in (0, 1):
            tracker = CardTracker(player)
            for state in walk(seed):
                view = observe(state, player)
                tracker.update(view)
                observed = view.observed_cards()
                for payload in (tracker, tracker.knowledge, tracker.belief):
                    for obj in reachable(payload):
                        assert not isinstance(obj, FORBIDDEN_TYPES), (
                            f"{type(obj).__name__} reachable from the tracker"
                        )
                    audited += 1
                # Naming a card is not itself a leak: the deck is public,
                # so "this card is somewhere" is known to everyone. What
                # would leak is an unjustified *placement*, so that is what
                # is checked.
                knowledge = tracker.knowledge
                named = cards_reachable_from(knowledge)
                assert named <= set(knowledge.settled) | knowledge.candidates
                for card in knowledge.known_cards():
                    assert card in observed, (
                        f"{card} claimed as observed but never seen"
                    )
                for card in knowledge.deduced_cards():
                    assert knowledge.justification(card) in RULES, (
                        f"{card} placed without a documented rule"
                    )
    assert audited > 500


def test_the_knowledge_modules_never_name_the_state_layer() -> None:
    """Scoped to what sits *downstream* of redaction.

    ``information_set.py`` is exempt by design: ``observe()`` is the
    redaction boundary, so it necessarily takes ground truth as its
    input. Everything after it consumes an already-redacted view and must
    never mention a ``GameState`` at all — which is exactly the line this
    test draws.
    """
    import ast

    directory = REPO / "src" / "durakfish" / "information"
    downstream = ("knowledge.py", "deduction.py", "belief.py", "tracker.py")
    for name in downstream:
        path = directory / name
        tree = ast.parse(path.read_text(), filename=str(path))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
        }
        for banned in ("GameState", "hands", "talon", "apply_move", "new_game"):
            assert banned not in names, f"{name} references {banned!r}"


def test_a_tracker_never_names_a_card_it_could_not_place() -> None:
    """A tracker holding a hidden card it cannot justify would be a leak."""
    for seed in range(10):
        tracker = CardTracker(0)
        for state in walk(seed):
            view = observe(state, 0)
            tracker.update(view)
            for card in tracker.retained_by_opponent:
                assert card in view.observed_cards(), (
                    "the tracker retained a card it never saw"
                )


# ======================================================================
# Adversarial histories
# ======================================================================
def adversarial(seed: int, policy):
    state = new_game(seed=seed)
    while not state.is_over:
        yield state
        state = apply_move(state, policy(get_legal_moves(state)))
    yield state


ADVERSARIAL = {
    "always take": lambda legal: next(
        (m for m in legal if type(m).__name__ == "TakeCards"), legal[0]
    ),
    "maximum pressure": lambda legal: next(
        (m for m in legal if type(m).__name__ == "AttackMove"), legal[0]
    ),
    "end immediately": lambda legal: next(
        (m for m in legal if type(m).__name__ == "EndAttack"), legal[0]
    ),
}


@pytest.mark.parametrize("label", sorted(ADVERSARIAL))
def test_adversarial_histories_keep_every_invariant(label) -> None:
    """Repeated takes, floods of throw-ins, and instant bouts."""
    policy = ADVERSARIAL[label]
    truth = {
        CardLocation.SELF_HAND: lambda s, p: set(s.hands[p]),
        CardLocation.OPPONENT_HAND: lambda s, p: set(s.hands[1 - p]),
        CardLocation.TALON: lambda s, p: set(s.talon),
        CardLocation.TABLE: lambda s, p: set(s.table_cards),
        CardLocation.DISCARD: lambda s, p: set(s.discard),
    }
    for seed in range(12):
        for player in (0, 1):
            tracker = CardTracker(player)
            for state in adversarial(seed, policy):
                view = observe(state, player)
                knowledge = tracker.update(view)
                knowledge.validate()
                # Every settled card must really be where it is claimed.
                for card, placement in knowledge.settled.items():
                    assert card in truth[placement.location](state, player), (
                        f"{label}: {card} wrongly placed in "
                        f"{placement.location.value} by {placement.rule!r}"
                    )
                # And the incremental path must still match a rebuild.
                assert knowledge == CardTracker.rebuild(view).knowledge


def test_a_tracker_survives_a_game_that_reaches_a_36_card_hand() -> None:
    """The extreme found in the Phase 2 audit, replayed through the tracker."""
    def dump(legal):
        take = [m for m in legal if type(m).__name__ == "TakeCards"]
        if take:
            return take[0]
        attacks = [m for m in legal if type(m).__name__ == "AttackMove"]
        return attacks[0] if attacks else legal[0]

    biggest = 0
    for seed in range(20):
        tracker = CardTracker(0)
        for state in adversarial(seed, dump):
            tracker.update(observe(state, 0))
            tracker.knowledge.validate()
            biggest = max(biggest, max(len(h) for h in state.hands))
    assert biggest > 12, "the stress policy did not build a large hand"


# ======================================================================
# Determinism and cross-process stability
# ======================================================================
def test_knowledge_keys_are_stable_across_processes() -> None:
    expected = []
    for seed in range(3):
        tracker = CardTracker(0)
        for state in walk(seed):
            tracker.update(observe(state, 0))
        expected.append(json.dumps(tracker.knowledge.to_dict(), sort_keys=True))

    code = (
        "import sys, json, random; sys.path.insert(0, 'src');"
        "from durakfish.game import new_game, get_legal_moves, apply_move;"
        "from durakfish.information import CardTracker, observe;"
        "out=[]\n"
        "for seed in range(3):\n"
        "    rng=random.Random(seed); s=new_game(seed=seed); t=CardTracker(0)\n"
        "    while not s.is_over:\n"
        "        t.update(observe(s,0)); s=apply_move(s, rng.choice(get_legal_moves(s)))\n"
        "    t.update(observe(s,0))\n"
        "    out.append(json.dumps(t.knowledge.to_dict(), sort_keys=True))\n"
        "print(json.dumps(out))"
    )
    outputs = []
    for hash_seed in ("0", "987654"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=str(REPO),
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin",
                 "PYTHONHASHSEED": hash_seed},
        )
        assert result.returncode == 0, result.stderr
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1] == expected


def test_repeating_an_update_changes_nothing() -> None:
    """Idempotent: folding in a view twice must not double-count."""
    for seed in range(10):
        tracker = CardTracker(0)
        for state in walk(seed):
            view = observe(state, 0)
            first = tracker.update(view)
            second = tracker.update(view)
            assert first == second
            assert tracker.events_seen == len(view.public_history)


def test_equal_knowledge_states_hash_equally() -> None:
    for seed in range(8):
        for state in walk(seed):
            a = tracked(state, 0).knowledge
            b = tracked(state, 0).knowledge
            assert a == b and hash(a) == hash(b)
            assert len({a, b}) == 1


def test_different_knowledge_states_can_differ() -> None:
    """Non-vacuity for the key: it must not collapse everything to one value."""
    keys = set()
    for seed in range(5):
        tracker = CardTracker(0)
        for state in walk(seed):
            tracker.update(observe(state, 0))
            keys.add(tracker.knowledge.key())
    assert len(keys) > 50


# ======================================================================
# Minimal search integration
# ======================================================================
def test_a_belief_state_can_be_built_alongside_a_search_without_changing_it() -> None:
    """Smallest possible integration check; SearchBot is untouched."""
    from durakfish.ai import SearchBot, SearchConfig

    bot = SearchBot(SearchConfig(depth=3), "s")
    bot.reset(random.Random(0))
    for state in list(walk(5))[:40]:
        player = state.current_player
        if player is None:
            continue
        view, legal = observe(state, player), get_legal_moves(state)
        without = bot.choose(view, legal)
        tracker = CardTracker(player)
        knowledge = tracker.update(view)
        assert isinstance(BeliefState.from_knowledge(knowledge), BeliefState)
        # Building knowledge must not perturb the search in any way.
        assert bot.choose(view, legal) == without
