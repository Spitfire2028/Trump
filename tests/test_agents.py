"""The agent boundary.

The central claim under test: an agent is handed a redacted view and the
legal moves, and that is all it gets. Everything else here checks that the
reference implementations fail loudly rather than improvising.
"""

from __future__ import annotations

import random

import pytest

from durakfish.ai import Agent, AgentBase, CallableAgent, ScriptedAgent
from durakfish.exceptions import AgentError
from durakfish.game import AttackMove, get_legal_moves, new_game
from durakfish.game.moves import END_ATTACK
from durakfish.information import InformationSet, observe


def first_legal(view, legal, rng):
    return legal[0]


# ----------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------
def test_reference_agents_satisfy_the_protocol() -> None:
    assert isinstance(CallableAgent(first_legal), Agent)
    assert isinstance(ScriptedAgent([]), Agent)


def test_an_object_missing_choose_is_not_an_agent() -> None:
    class NotAnAgent:
        name = "nope"

    assert not isinstance(NotAnAgent(), Agent)


def test_agents_receive_an_information_set_and_never_a_game_state() -> None:
    """Checked at runtime, not merely by type annotation."""
    from durakfish.game.state import GameState

    seen: list[object] = []

    def probe(view, legal, rng):
        seen.append(view)
        assert isinstance(view, InformationSet)
        assert not isinstance(view, GameState)
        return legal[0]

    from durakfish.simulation import play_game

    play_game([CallableAgent(probe, "a"), CallableAgent(probe, "b")], seed=3)
    assert len(seen) > 10
    assert all(isinstance(v, InformationSet) for v in seen)


def test_an_agent_only_ever_sees_its_own_seat() -> None:
    seats: list[tuple[int, int]] = []

    def probe_player(player_id):
        def probe(view, legal, rng):
            seats.append((player_id, view.player))
            return legal[0]

        return probe

    from durakfish.simulation import play_game

    play_game(
        [CallableAgent(probe_player(0), "a"), CallableAgent(probe_player(1), "b")],
        seed=9,
    )
    assert seats
    assert all(expected == actual for expected, actual in seats)


def test_the_view_is_always_of_the_player_to_move() -> None:
    def probe(view, legal, rng):
        assert view.is_to_move
        assert view.current_player == view.player
        assert legal, "an agent must never be asked to choose from nothing"
        return legal[0]

    from durakfish.simulation import play_game

    play_game([CallableAgent(probe, "a"), CallableAgent(probe, "b")], seed=12)


# ----------------------------------------------------------------------
# CallableAgent
# ----------------------------------------------------------------------
def test_callable_agent_uses_only_the_generator_it_is_given() -> None:
    """Two agents with the same seed must make identical choices."""
    picks: list[list[int]] = [[], []]

    def maker(slot):
        def choose(view, legal, rng):
            index = rng.randrange(len(legal))
            picks[slot].append(index)
            return legal[index]

        return choose

    view = observe(new_game(seed=1), 0)
    legal = get_legal_moves(new_game(seed=1))
    for slot in (0, 1):
        agent = CallableAgent(maker(slot), f"agent{slot}")
        agent.reset(random.Random(77))
        for _ in range(20):
            agent.choose(view, legal)
    assert picks[0] == picks[1]


def test_reset_rewinds_the_random_stream() -> None:
    agent = CallableAgent(lambda v, legal, r: legal[r.randrange(len(legal))], "r")
    state = new_game(seed=2)
    view, legal = observe(state, state.current_player), get_legal_moves(state)

    agent.reset(random.Random(5))
    first = [agent.choose(view, legal) for _ in range(10)]
    agent.reset(random.Random(5))
    assert [agent.choose(view, legal) for _ in range(10)] == first


def test_callable_agent_takes_a_name() -> None:
    assert CallableAgent(first_legal, "greedy-ish").name == "greedy-ish"
    assert CallableAgent(first_legal).name == "callable"
    assert "greedy-ish" in repr(CallableAgent(first_legal, "greedy-ish"))


# ----------------------------------------------------------------------
# ScriptedAgent
# ----------------------------------------------------------------------
def test_scripted_agent_plays_its_script_in_order() -> None:
    state = new_game(seed=4)
    opening = get_legal_moves(state)[0]
    agent = ScriptedAgent([opening])
    agent.reset(random.Random(0))
    view = observe(state, state.current_player)
    assert agent.choose(view, get_legal_moves(state)) is opening
    assert agent.moves_played == 1


def test_scripted_agent_refuses_to_improvise_when_the_script_runs_out() -> None:
    state = new_game(seed=4)
    agent = ScriptedAgent([])
    agent.reset(random.Random(0))
    with pytest.raises(AgentError, match="ran out"):
        agent.choose(observe(state, state.current_player), get_legal_moves(state))


def test_scripted_agent_rejects_a_move_the_position_does_not_allow() -> None:
    """A script that disagrees with the engine is a bug worth surfacing."""
    state = new_game(seed=4)
    illegal = END_ATTACK  # the table is empty; a bout cannot begin with a pass
    agent = ScriptedAgent([illegal])
    agent.reset(random.Random(0))
    with pytest.raises(AgentError, match="not legal"):
        agent.choose(observe(state, state.current_player), get_legal_moves(state))


def test_scripted_agent_can_reproduce_a_recorded_game() -> None:
    from durakfish.simulation import play_game

    random_agents = [
        CallableAgent(lambda v, legal, r: r.choice(legal), "a"),
        CallableAgent(lambda v, legal, r: r.choice(legal), "b"),
    ]
    original = play_game(random_agents, seed=31)

    scripts: list[list] = [[], []]
    for entry in original.moves:
        scripts[entry.player].append(entry.move)

    rerun = play_game(
        [ScriptedAgent(scripts[0], "s0"), ScriptedAgent(scripts[1], "s1")],
        seed=31,
    )
    assert [m.move for m in rerun.moves] == [m.move for m in original.moves]
    assert rerun.durak == original.durak


def test_reset_rewinds_a_script_so_an_agent_can_replay_a_game() -> None:
    state = new_game(seed=4)
    opening = get_legal_moves(state)[0]
    agent = ScriptedAgent([opening])
    view, legal = observe(state, state.current_player), get_legal_moves(state)
    agent.reset(random.Random(0))
    agent.choose(view, legal)
    agent.reset(random.Random(0))
    assert agent.moves_played == 0
    assert agent.choose(view, legal) is opening


# ----------------------------------------------------------------------
# AgentBase
# ----------------------------------------------------------------------
def test_agent_base_requires_a_choose_implementation() -> None:
    base = AgentBase("bare")
    base.reset(random.Random(0))  # no-op, must not raise
    with pytest.raises(NotImplementedError):
        base.choose(observe(new_game(seed=1), 0), ())


def test_a_custom_agent_needs_only_choose_and_a_name() -> None:
    class LowestCard(AgentBase):
        name = "lowest"

        def choose(self, view, legal):
            attacks = [m for m in legal if isinstance(m, AttackMove)]
            if attacks:
                return min(attacks, key=lambda m: m.card.code)
            return legal[-1]

    from durakfish.simulation import play_game

    record = play_game([LowestCard(), LowestCard()], seed=6, validate_states=True)
    assert record.length > 0
    assert record.agent_names == ("lowest", "lowest")
