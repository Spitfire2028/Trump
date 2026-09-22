"""Phase 5 audit.

Search is where an information boundary usually breaks. The tempting
shortcut — keep the real state around "just while we explore" — produces
an engine that looks correct, plays well, and is solving a different game
from the one it will face. So the tree itself is audited here, not only
the agent's arguments.

The reachability machinery is imported from the Phase 3 audit rather than
rewritten, so search is held to literally the same standard as everything
before it.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from durakfish.ai import GreedyBot, RandomBot, SearchBot, SearchConfig
from durakfish.ai.searchnode import Actor, SearchNode
from durakfish.exceptions import AgentError, IllegalAgentMoveError
from durakfish.game import apply_move, get_legal_moves, new_game
from durakfish.game.moves import AttackMove
from durakfish.information import observe
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

REPO = Path(__file__).resolve().parent.parent
DEPTH = SearchConfig(depth=3)


def positions(games: int = 10):
    for seed in range(games):
        rng = random.Random(seed)
        state = new_game(seed=seed)
        while not state.is_over:
            yield state
            state = apply_move(state, rng.choice(get_legal_moves(state)))


def enumerate_tree(node: SearchNode, depth: int) -> list[SearchNode]:
    """Every node the search would visit, collected for auditing."""
    found = [node]
    if depth == 0 or node.is_leaf:
        return found
    if node.to_move is Actor.SELF:
        for move in node.self_moves():
            found += enumerate_tree(node.after_self(move), depth - 1)
    else:
        for action in node.opponent_actions():
            found += enumerate_tree(node.after_opponent(action), depth - 1)
    return found


# ======================================================================
# The search tree itself
# ======================================================================
def test_no_node_anywhere_in_a_search_tree_exposes_hidden_state() -> None:
    """The audit that matters most in this phase."""
    audited = 0
    for state in positions(6):
        player = state.current_player
        assert player is not None
        view = observe(state, player)
        observed = view.observed_cards()
        for node in enumerate_tree(SearchNode.from_view(view), 4):
            for obj in reachable(node):
                assert not isinstance(obj, FORBIDDEN_TYPES), (
                    f"{type(obj).__name__} reachable from a search node"
                )
            leaked = cards_reachable_from(node) - observed
            assert not leaked, f"search node exposes unobserved {leaked}"
            audited += 1
    assert audited > 3000


def test_a_search_tree_never_invents_a_card_the_searcher_has_not_seen() -> None:
    """Unknown cards must stay unknown — counted, never materialised."""
    for state in positions(5):
        player = state.current_player
        assert player is not None
        view = observe(state, player)
        for node in enumerate_tree(SearchNode.from_view(view), 4):
            for slot in node.table:
                for card in slot.known_cards:
                    assert card in view.observed_cards()
            assert node.unknown_own >= 0
            assert node.opponent_cards >= 0


def test_the_search_layer_names_nothing_from_the_state_layer() -> None:
    import ast

    for module in ("searchnode.py", "search.py", "search_bot.py", "evaluation.py",
                   "ordering.py"):
        path = REPO / "src" / "durakfish" / "ai" / module
        tree = ast.parse(path.read_text(), filename=str(path))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
        }
        for banned in ("GameState", "hands", "talon", "apply_move", "new_game",
                       "events"):
            assert banned not in names, f"{module} references {banned!r}"


def test_search_bot_arguments_pass_the_reachability_audit() -> None:
    captured: list[tuple] = []

    class Spy(SearchBot):
        def choose(self, view, legal):
            captured.append((view, tuple(legal)))
            return super().choose(view, legal)

    for seed in range(4):
        play_game([Spy(DEPTH, "s"), RandomBot("r")], seed=seed)

    assert len(captured) > 100
    for view, legal in captured:
        observed = view.observed_cards()
        for payload in (view, legal):
            for obj in reachable(payload):
                assert not isinstance(obj, FORBIDDEN_TYPES)
            assert not (cards_reachable_from(payload) - observed)


# ======================================================================
# Hidden-state invariance of every search output
# ======================================================================
def search_outputs(view, legal) -> dict[str, object]:
    """Every channel a hidden-state dependency could escape through."""
    bot = SearchBot(DEPTH, "probe")
    bot.reset(random.Random(0))
    move = bot.choose(view, legal)
    result = bot.last_result
    assert result is not None
    node = SearchNode.from_view(view)
    return {
        "move": move,
        "score": result.score,
        "nodes": result.stats.nodes,
        "leaves": result.stats.leaves,
        "terminals": result.stats.terminals,
        "cutoffs": result.stats.cutoffs,
        "max_depth": result.stats.max_depth,
        "root_moves": result.stats.root_moves,
        "is_proven": result.is_proven,
        "node_key": node.key(),
        "node_repr": repr(node),
        "result_repr": repr(result),
        "self_moves": node.self_moves(),
        "opponent_actions": node.opponent_actions(),
    }


def test_no_search_output_depends_on_hidden_information() -> None:
    rng = random.Random(909)
    compared = 0
    for state in positions(8):
        player = state.current_player
        if player is None or not state.talon:
            continue
        legal = get_legal_moves(state)
        baseline = search_outputs(observe(state, player), legal)
        for _ in range(2):
            twin = permute_hidden(state, player, rng)
            twin = rewrite_opponent_draw_identities(twin, player)
            twin_view = observe(twin, player)
            assert twin_view == observe(state, player), "premise failed"
            candidate = search_outputs(twin_view, legal)
            for channel, value in baseline.items():
                assert candidate[channel] == value, f"{channel} leaks hidden state"
            compared += 1
    assert compared > 120


# ======================================================================
# SearchBot: determinism and the agent contract
# ======================================================================
def test_search_bot_is_deterministic_across_calls_and_instances() -> None:
    for state in list(positions(5))[:80]:
        player = state.current_player
        assert player is not None
        view, legal = observe(state, player), get_legal_moves(state)
        picks = set()
        for _ in range(3):
            bot = SearchBot(DEPTH, "s")
            bot.reset(random.Random(random.randrange(10_000)))
            picks.add(bot.choose(view, legal))
        assert len(picks) == 1


def test_the_rng_cannot_influence_search_bot() -> None:
    state = next(iter(positions(1)))
    player = state.current_player
    assert player is not None
    view, legal = observe(state, player), get_legal_moves(state)
    bot = SearchBot(DEPTH, "s")
    choices = set()
    for seed in range(30):
        bot.reset(random.Random(seed))
        choices.add(bot.choose(view, legal))
    assert len(choices) == 1


def test_search_bot_is_permutation_invariant() -> None:
    rng = random.Random(3)
    for state in list(positions(5))[:60]:
        player = state.current_player
        assert player is not None
        view, legal = observe(state, player), get_legal_moves(state)
        bot = SearchBot(DEPTH, "s")
        bot.reset(random.Random(0))
        baseline = bot.choose(view, legal)
        shuffled = list(legal)
        for _ in range(3):
            rng.shuffle(shuffled)
            assert bot.choose(view, shuffled) == baseline


def test_search_bot_stays_inside_a_restricted_legal_list() -> None:
    for state in list(positions(5))[:60]:
        player = state.current_player
        assert player is not None
        view, legal = observe(state, player), get_legal_moves(state)
        bot = SearchBot(DEPTH, "s")
        bot.reset(random.Random(0))
        preferred = bot.choose(view, legal)
        reduced = tuple(m for m in legal if m != preferred)
        if reduced:
            chosen = bot.choose(view, reduced)
            assert chosen in reduced and chosen != preferred


def test_search_bot_refuses_an_empty_choice() -> None:
    state = next(iter(positions(1)))
    player = state.current_player
    assert player is not None
    bot = SearchBot(DEPTH, "s")
    bot.reset(random.Random(0))
    with pytest.raises(AgentError, match="empty legal-move list"):
        bot.choose(observe(state, player), ())


def test_reset_clears_the_diagnostic_result() -> None:
    state = next(iter(positions(1)))
    player = state.current_player
    assert player is not None
    bot = SearchBot(DEPTH, "s")
    bot.reset(random.Random(0))
    bot.choose(observe(state, player), get_legal_moves(state))
    assert bot.last_result is not None
    bot.reset(random.Random(0))
    assert bot.last_result is None


def test_search_bot_does_not_mutate_its_inputs() -> None:
    for state in list(positions(5))[:60]:
        player = state.current_player
        assert player is not None
        view, legal = observe(state, player), get_legal_moves(state)
        before, key, moves = view.to_dict(), view.key(), tuple(legal)
        bot = SearchBot(DEPTH, "s")
        bot.reset(random.Random(0))
        bot.choose(view, legal)
        assert view.to_dict() == before and view.key() == key
        assert tuple(legal) == moves


def test_a_search_bot_subclass_returning_a_foreign_move_is_rejected() -> None:
    class Cheat(SearchBot):
        name = "cheat"

        def choose(self, view, legal):
            return AttackMove(next(iter(sorted(view.unobserved_cards(),
                                               key=lambda c: c.code))))

    with pytest.raises(IllegalAgentMoveError, match="cheat"):
        play_game([Cheat(DEPTH), RandomBot("r")], seed=1)


# ======================================================================
# Simulation integration
# ======================================================================
MATCHUPS = {
    "search vs random": lambda: [SearchBot(DEPTH, "s"), RandomBot("r")],
    "search vs greedy": lambda: [SearchBot(DEPTH, "s"), GreedyBot("g")],
    "search vs search": lambda: [SearchBot(DEPTH, "s1"), SearchBot(DEPTH, "s2")],
}


@pytest.mark.parametrize("label", sorted(MATCHUPS))
def test_matchups_are_reproducible_and_verifiable(label) -> None:
    for seed in range(5):
        first = play_game(MATCHUPS[label](), seed=seed, validate_states=True)
        second = play_game(MATCHUPS[label](), seed=seed, validate_states=True)
        assert first.to_dict() == second.to_dict(), label
        verify_record(first)


@pytest.mark.parametrize("label", sorted(MATCHUPS))
def test_matchups_survive_a_json_round_trip(label) -> None:
    for seed in range(3):
        record = play_game(MATCHUPS[label](), seed=seed)
        restored = GameRecord.from_dict(json.loads(json.dumps(record.to_dict())))
        assert restored.to_dict() == record.to_dict()
        assert [s.transposition_key() for s in replay(restored)] == [
            s.transposition_key() for s in replay(record)
        ]
        verify_record(restored)


def test_replay_does_not_need_the_search_bot() -> None:
    record = play_game(MATCHUPS["search vs greedy"](), seed=9)
    text = json.dumps(record.to_dict())
    del record
    assert verify_record(GameRecord.from_dict(json.loads(text))).is_over


def test_search_bot_coexists_with_the_phase_4_baselines() -> None:
    """A sanity check on behaviour, explicitly not a strength claim."""
    records = [
        play_game([SearchBot(DEPTH, "s"), RandomBot("r")], seed=s,
                  first_attacker=s % 2)
        for s in range(40)
    ]
    summary = summarize(records)
    assert summary.games == 40
    assert summary.win_rate(0) > 0.5, summary.describe()


# ======================================================================
# Cross-process determinism
# ======================================================================
def test_search_is_identical_in_a_fresh_interpreter() -> None:
    expected = json.dumps(
        play_game(MATCHUPS["search vs greedy"](), seed=11).to_dict(), sort_keys=True
    )
    code = (
        "import sys, json; sys.path.insert(0, 'src');"
        "from durakfish.ai import SearchBot, GreedyBot, SearchConfig;"
        "from durakfish.simulation import play_game;"
        "c = SearchConfig(depth=3);"
        "print(json.dumps(play_game([SearchBot(c, 's'), GreedyBot('g')], seed=11)"
        ".to_dict(), sort_keys=True))"
    )
    outputs = []
    for hash_seed in ("0", "424242"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=str(REPO),
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin",
                 "PYTHONHASHSEED": hash_seed},
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout.strip())
    assert outputs[0] == outputs[1] == expected
