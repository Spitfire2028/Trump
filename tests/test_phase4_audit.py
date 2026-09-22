"""Phase 4 audit: the Phase 3 guarantees, re-run against real agents.

Phase 3 proved the boundary held for a redaction function and two test
doubles. Now actual policies read the view and act on it, so the same
questions are worth asking again with something that could actually be
tempted to cheat:

* can a bot reach hidden state through what it is handed?
* does a bot's behaviour depend on anything hidden?
* is a whole game reproducible, in this process and in a fresh one?

The reachability machinery is reused from the Phase 3 audit rather than
rewritten, so the two phases are held to literally the same standard.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from durakfish.ai import GreedyBot, RandomBot
from durakfish.cards import Card, Suit
from durakfish.exceptions import IllegalAgentMoveError
from durakfish.game import (
    AttackMove,
    DefenseMove,
    Phase,
    apply_move,
    get_legal_moves,
    new_game,
)
from durakfish.game.moves import TAKE
from durakfish.game.state import GameState, TableSlot, add_cards
from durakfish.information import InformationSet, observe
from durakfish.simulation import (
    GameRecord,
    play_game,
    replay,
    summarize,
    verify_record,
)
from tests.test_phase3_audit import (
    FORBIDDEN_TYPES,
    cards_reachable_from,
    permute_hidden,
    reachable,
    rewrite_opponent_draw_identities,
)

BOTS = [RandomBot, GreedyBot]
REPO = Path(__file__).resolve().parent.parent


def bots(seed: int = 0):
    a, b = RandomBot("a"), GreedyBot("b")
    a.reset(random.Random(seed))
    b.reset(random.Random(seed))
    return a, b


def positions(games: int = 12) -> list[GameState]:
    out: list[GameState] = []
    for seed in range(games):
        rng = random.Random(seed)
        state = new_game(seed=seed)
        while not state.is_over:
            out.append(state)
            state = apply_move(state, rng.choice(get_legal_moves(state)))
        out.append(state)
    return out


# ======================================================================
# Reachability: what a real bot can touch
# ======================================================================
@pytest.mark.parametrize("bot_class", BOTS)
def test_nothing_forbidden_is_reachable_from_a_bots_arguments(bot_class) -> None:
    captured: list[tuple[InformationSet, tuple]] = []

    class Spy(bot_class):  # type: ignore[misc, valid-type]
        def choose(self, view, legal):
            captured.append((view, tuple(legal)))
            return super().choose(view, legal)

    for seed in range(5):
        play_game([Spy("spy"), Spy("spy2")], seed=seed)

    assert len(captured) > 200
    for view, legal in captured:
        observed = view.observed_cards()
        for payload in (view, legal):
            for obj in reachable(payload):
                assert not isinstance(obj, FORBIDDEN_TYPES), (
                    f"{type(obj).__name__} reachable from {bot_class.__name__}"
                )
            leaked = cards_reachable_from(payload) - observed
            assert not leaked, f"{bot_class.__name__} can reach unobserved {leaked}"


def test_neither_bot_names_anything_from_the_game_state_layer() -> None:
    """Static: the bots' source cannot even mention the forbidden vocabulary."""
    import ast

    for module in ("random_bot.py", "greedy_bot.py", "tiebreak.py"):
        path = REPO / "src" / "durakfish" / "ai" / module
        tree = ast.parse(path.read_text(), filename=str(path))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
        }
        for banned in ("GameState", "hands", "talon", "apply_move", "new_game",
                       "events", "history"):
            assert banned not in names, f"{module} references {banned!r}"


# ======================================================================
# Hidden-state invariance for real policies
# ======================================================================
@pytest.mark.parametrize("bot_class", BOTS)
def test_permuting_hidden_cards_never_changes_a_bots_decision(bot_class) -> None:
    """The metamorphic property that matters: same view ⇒ same move."""
    rng = random.Random(4242)
    checked = 0
    for state in positions(10):
        player = state.current_player
        if player is None or not state.talon:
            continue
        legal = get_legal_moves(state)
        baseline_view = observe(state, player)

        bot = bot_class()
        bot.reset(random.Random(7))
        baseline = bot.choose(baseline_view, legal)

        for _ in range(3):
            twin = permute_hidden(state, player, rng)
            twin = rewrite_opponent_draw_identities(twin, player)
            twin_view = observe(twin, player)
            assert twin_view == baseline_view, "the premise failed, not the bot"
            assert get_legal_moves(twin, player) == legal

            other = bot_class()
            other.reset(random.Random(7))
            assert other.choose(twin_view, legal) == baseline
            checked += 1
    assert checked > 200


@pytest.mark.parametrize("bot_class", BOTS)
def test_a_bot_reset_with_the_same_seed_replays_a_whole_game(bot_class) -> None:
    """reset() must establish the complete per-game state."""
    a = bot_class("x")
    first = play_game([a, bot_class("y")], seed=3)
    play_game([a, bot_class("y")], seed=888)  # dirty it
    again = play_game([a, bot_class("y")], seed=3)
    assert first.to_dict() == again.to_dict()

    fresh_pair = [bot_class("x"), bot_class("y")]
    assert play_game(fresh_pair, seed=3).to_dict() == first.to_dict()


# ======================================================================
# Agent API isolation: no GameState anywhere
# ======================================================================
def handmade_view(phase: Phase, *, hand: str, table, attacker: int = 0) -> InformationSet:
    """An InformationSet built without constructing a GameState."""
    from durakfish.game.ruleset import STANDARD_RULES

    cards = add_cards((), [Card.parse(c) for c in hand.split()])
    slots = tuple(
        TableSlot(Card.parse(a), None if d is None else Card.parse(d))
        for a, d in table
    )
    table_cards = {c for slot in slots for c in slot.cards}
    trump_card = Card.parse("6H")
    return InformationSet(
        player=1,
        hand=cards,
        table=slots,
        trump_suit=Suit.HEARTS,
        trump_card=trump_card,
        talon_size=7,
        hand_sizes=(4, len(cards)),
        discard=frozenset({Card.parse("10D")}),
        attacker=attacker,
        phase=phase,
        attack_limit=4,
        public_history=(),
        observed=frozenset(set(cards) | table_cards | {trump_card, Card.parse("10D")}),
        rules=STANDARD_RULES,
    )


@pytest.mark.parametrize("bot_class", BOTS)
def test_both_bots_decide_from_a_view_with_no_game_state_in_existence(
    bot_class,
) -> None:
    view = handmade_view(Phase.DEFENSE, hand="6S 9H AC", table=[("7S", None)])
    legal = (
        DefenseMove(Card.parse("7S"), Card.parse("9H")),
        DefenseMove(Card.parse("7S"), Card.parse("AC")),
        TAKE,
    )
    bot = bot_class()
    bot.reset(random.Random(1))
    chosen = bot.choose(view, legal)
    assert chosen in legal
    for obj in reachable(view):
        assert not isinstance(obj, FORBIDDEN_TYPES)


def test_greedy_follows_its_documented_rule_on_a_handmade_view() -> None:
    """Cheapest defence: the non-trump ace, not the trump nine."""
    view = handmade_view(Phase.DEFENSE, hand="6S 9H AC", table=[("7S", None)])
    legal = (
        DefenseMove(Card.parse("7S"), Card.parse("9H")),
        DefenseMove(Card.parse("7S"), Card.parse("AC")),
        TAKE,
    )
    bot = GreedyBot()
    bot.reset(random.Random(0))
    assert bot.choose(view, legal) == DefenseMove(Card.parse("7S"), Card.parse("AC"))


def test_greedy_refuses_to_throw_a_trump_on_a_handmade_view() -> None:
    from durakfish.game.moves import END_ATTACK

    view = handmade_view(Phase.ATTACK, hand="9H", table=[("7S", "8S")], attacker=1)
    legal = (AttackMove(Card.parse("9H")), END_ATTACK)
    bot = GreedyBot()
    bot.reset(random.Random(0))
    assert bot.choose(view, legal) == END_ATTACK


# ======================================================================
# Invalid agent behaviour
# ======================================================================
def test_a_bot_subclass_that_returns_a_foreign_move_is_rejected() -> None:
    class Cheat(RandomBot):
        name = "cheat"

        def choose(self, view, legal):
            return AttackMove(next(iter(view.unobserved_cards())))

    with pytest.raises(IllegalAgentMoveError, match="cheat"):
        play_game([Cheat(), GreedyBot("g")], seed=1)


def test_rejection_is_deterministic() -> None:
    class Cheat(RandomBot):
        name = "cheat"

        def choose(self, view, legal):
            return AttackMove(next(iter(sorted(view.unobserved_cards(),
                                               key=lambda c: c.code))))

    messages = []
    for _ in range(3):
        with pytest.raises(IllegalAgentMoveError) as caught:
            play_game([Cheat(), GreedyBot("g")], seed=2)
        messages.append(str(caught.value))
    assert len(set(messages)) == 1


# ======================================================================
# Determinism across the three match-ups
# ======================================================================
MATCHUPS = {
    "random vs random": lambda: [RandomBot("a"), RandomBot("b")],
    "random vs greedy": lambda: [RandomBot("a"), GreedyBot("b")],
    "greedy vs greedy": lambda: [GreedyBot("a"), GreedyBot("b")],
}


@pytest.mark.parametrize("label", sorted(MATCHUPS))
def test_matchups_are_reproducible_in_process(label) -> None:
    make = MATCHUPS[label]
    for seed in range(10):
        first = play_game(make(), seed=seed, validate_states=True)
        second = play_game(make(), seed=seed, validate_states=True)
        assert first.to_dict() == second.to_dict(), label
        verify_record(first)


@pytest.mark.parametrize("label", sorted(MATCHUPS))
def test_matchups_survive_a_json_round_trip(label) -> None:
    for seed in range(6):
        record = play_game(MATCHUPS[label](), seed=seed)
        restored = GameRecord.from_dict(json.loads(json.dumps(record.to_dict())))
        assert restored.to_dict() == record.to_dict()
        assert [s.transposition_key() for s in replay(restored)] == [
            s.transposition_key() for s in replay(record)
        ]
        verify_record(restored)


@pytest.mark.parametrize("label", sorted(MATCHUPS))
def test_replay_does_not_depend_on_the_live_agents(label) -> None:
    """The record replays with the agents deleted entirely."""
    record = play_game(MATCHUPS[label](), seed=21)
    text = json.dumps(record.to_dict())
    del record
    restored = GameRecord.from_dict(json.loads(text))
    assert verify_record(restored).is_over


def test_different_seeds_move_random_bot_off_its_previous_line() -> None:
    """Not required for every pair — collisions are legitimate."""
    lines = {
        json.dumps(play_game([RandomBot("a"), RandomBot("b")], seed=s).to_dict())
        for s in range(30)
    }
    assert len(lines) >= 28


def test_greedy_is_unmoved_by_the_agent_seed_but_not_by_the_deal() -> None:
    """Documented distinction: seeds change the cards, not the policy."""
    from durakfish.simulation.runner import derive_seeds

    # Same deal seed reached from different master seeds is not available,
    # so instead: the same master seed must reproduce, and different ones
    # must be able to differ because the deal differs.
    a = play_game([GreedyBot("a"), GreedyBot("b")], seed=5)
    b = play_game([GreedyBot("a"), GreedyBot("b")], seed=5)
    assert a.to_dict() == b.to_dict()
    c = play_game([GreedyBot("a"), GreedyBot("b")], seed=6)
    assert derive_seeds(5, 2)[0] != derive_seeds(6, 2)[0]
    assert c.initial_state.transposition_key() != a.initial_state.transposition_key()


# ======================================================================
# Cross-process determinism
# ======================================================================
def test_records_are_identical_in_a_fresh_interpreter() -> None:
    """Guards against anything process-dependent: hashing, iteration order."""
    expected = {
        label: json.dumps(play_game(make(), seed=11).to_dict(), sort_keys=True)
        for label, make in MATCHUPS.items()
    }
    code = (
        "import sys, json; sys.path.insert(0, 'src');"
        "from durakfish.ai import RandomBot, GreedyBot;"
        "from durakfish.simulation import play_game;"
        "m = {'random vs random': [RandomBot('a'), RandomBot('b')],"
        " 'random vs greedy': [RandomBot('a'), GreedyBot('b')],"
        " 'greedy vs greedy': [GreedyBot('a'), GreedyBot('b')]};"
        "print(json.dumps({k: json.dumps(play_game(v, seed=11).to_dict(),"
        " sort_keys=True) for k, v in m.items()}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected


def test_cross_process_determinism_survives_hash_randomization() -> None:
    """Two child processes with different PYTHONHASHSEED must agree."""
    code = (
        "import sys, json; sys.path.insert(0, 'src');"
        "from durakfish.ai import RandomBot, GreedyBot;"
        "from durakfish.simulation import play_game;"
        "print(json.dumps(play_game([RandomBot('a'), GreedyBot('b')], seed=4)"
        ".to_dict(), sort_keys=True))"
    )
    outputs = []
    for hash_seed in ("0", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(REPO),
            env={
                "PYTHONPATH": "src",
                "PATH": "/usr/bin:/bin",
                "PYTHONHASHSEED": hash_seed,
            },
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]


# ======================================================================
# Statistics
# ======================================================================
def test_summary_counts_agree_with_the_records() -> None:
    records = [
        play_game([RandomBot("r"), GreedyBot("g")], seed=s) for s in range(25)
    ]
    summary = summarize(records)
    assert summary.games == 25
    assert summary.plies == sum(r.length for r in records)
    assert sum(summary.decisions) == summary.plies
    assert summary.wins[0] + summary.wins[1] + summary.draws == 25
    assert summary.losses[0] == summary.wins[1]
    assert summary.agent_names == ("r", "g")
    assert 0.0 <= summary.win_rate(0) <= 1.0
    assert "decisions" in summary.describe()


def test_greedy_beats_random_from_either_seat() -> None:
    """A sanity check on the baseline, not a strength claim."""
    for seat, line_up in enumerate(
        ([RandomBot("r"), GreedyBot("g")], [GreedyBot("g"), RandomBot("r")])
    ):
        records = [
            play_game(line_up, seed=s, first_attacker=s % 2) for s in range(60)
        ]
        summary = summarize(records)
        greedy_seat = 1 - seat
        assert summary.win_rate(greedy_seat) > 0.8, summary.describe()


def test_summary_refuses_to_mix_line_ups() -> None:
    from durakfish.exceptions import SimulationError

    mixed = [
        play_game([RandomBot("r"), GreedyBot("g")], seed=1),
        play_game([GreedyBot("g"), RandomBot("r")], seed=1),
    ]
    with pytest.raises(SimulationError, match="mix line-ups"):
        summarize(mixed)
    with pytest.raises(SimulationError, match="empty"):
        summarize([])
