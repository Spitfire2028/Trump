"""Driving, recording, replaying.

The properties that matter: a game is fully determined by its seed, a
record replays to the same outcome it claims, and a misbehaving agent is
reported rather than quietly accommodated.
"""

from __future__ import annotations

import json

import pytest

from durakfish.ai import AgentBase, CallableAgent
from durakfish.exceptions import (
    IllegalAgentMoveError,
    RecordError,
    SimulationError,
)
from durakfish.game import AttackMove, Phase
from durakfish.game.moves import END_ATTACK, TAKE
from durakfish.game.ruleset import RuleSet
from durakfish.simulation import (
    GameRecord,
    derive_seeds,
    play_game,
    replay,
    verify_record,
)


def randoms(names=("a", "b")):
    return [CallableAgent(lambda v, legal, r: r.choice(legal), n) for n in names]


# ----------------------------------------------------------------------
# Seed derivation
# ----------------------------------------------------------------------
def test_seed_derivation_is_deterministic_and_separates_streams() -> None:
    deal, agents = derive_seeds(1234, 2)
    assert (deal, agents) == derive_seeds(1234, 2)
    assert len({deal, *agents}) == 3, "streams must not collide"
    assert derive_seeds(1235, 2)[0] != deal


def test_the_deal_seed_does_not_depend_on_the_number_of_agents() -> None:
    """So the same seed deals the same cards regardless of who is playing."""
    assert derive_seeds(99, 0)[0] == derive_seeds(99, 2)[0] == derive_seeds(99, 5)[0]


def test_agent_randomness_cannot_perturb_the_deal() -> None:
    quiet = play_game(
        [CallableAgent(lambda v, legal, r: legal[0], "q")] * 2, seed=42
    )
    noisy = play_game(randoms(), seed=42)
    assert (
        quiet.initial_state.transposition_key()
        == noisy.initial_state.transposition_key()
    )


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------
def test_same_seed_and_agents_produce_an_identical_record() -> None:
    for seed in range(15):
        first = play_game(randoms(), seed=seed)
        second = play_game(randoms(), seed=seed)
        assert first.to_dict() == second.to_dict()
        assert json.dumps(first.to_dict()) == json.dumps(second.to_dict())


def test_different_seeds_produce_different_games() -> None:
    records = [play_game(randoms(), seed=s) for s in range(20)]
    assert len({json.dumps(r.to_dict()) for r in records}) == 20


def test_reusing_agent_objects_does_not_leak_state_between_games() -> None:
    """reset() must make a second game with the same seed identical."""
    agents = randoms()
    first = play_game(agents, seed=8)
    play_game(agents, seed=999)  # dirty the agents
    again = play_game(agents, seed=8)
    assert first.to_dict() == again.to_dict()


# ----------------------------------------------------------------------
# Driving
# ----------------------------------------------------------------------
def test_a_played_game_is_finished_and_internally_consistent() -> None:
    for seed in range(30):
        record = play_game(randoms(), seed=seed, validate_states=True)
        assert record.length > 0
        assert record.durak in (0, 1, None)
        assert [m.ply for m in record.moves] == list(range(record.length))
        if record.durak is not None:
            assert record.winner() == 1 - record.durak
        else:
            assert record.winner() is None


def test_every_recorded_move_was_made_by_the_player_to_move() -> None:
    record = play_game(randoms(), seed=5)
    for state, entry in zip(replay(record), record.moves, strict=False):
        assert state.current_player == entry.player


def test_the_driver_rejects_the_wrong_number_of_agents() -> None:
    with pytest.raises(SimulationError, match="agents supplied"):
        play_game(randoms(("only",)), seed=1)
    with pytest.raises(SimulationError, match="agents supplied"):
        play_game(randoms(("a", "b", "c")), seed=1)


def test_an_agent_returning_an_illegal_move_is_reported_not_repaired() -> None:
    class Cheater(AgentBase):
        name = "cheater"

        def choose(self, view, legal):
            # A card it does not hold, in a phase that forbids attacking.
            return AttackMove(view.unobserved_cards().__iter__().__next__())

    with pytest.raises(IllegalAgentMoveError, match="cheater"):
        play_game([Cheater(), *randoms(("b",))], seed=2)


def test_an_agent_returning_a_non_move_is_also_caught() -> None:
    class Nonsense(AgentBase):
        name = "nonsense"

        def choose(self, view, legal):
            return "attack the six of spades"  # type: ignore[return-value]

    with pytest.raises(IllegalAgentMoveError):
        play_game([Nonsense(), *randoms(("b",))], seed=2)


def test_the_move_limit_guards_against_a_non_terminating_bug() -> None:
    with pytest.raises(SimulationError, match="exceeded"):
        play_game(randoms(), seed=1, move_limit=5)


def test_first_attacker_can_be_forced_for_fair_comparison() -> None:
    for seed in range(10):
        for seat in (0, 1):
            record = play_game(randoms(), seed=seed, first_attacker=seat)
            assert record.initial_state.attacker == seat


def test_playing_from_an_explicit_deck_is_recorded_as_such() -> None:
    from durakfish.cards import Deck

    deck = Deck.shuffled(seed=1234)
    record = play_game(randoms(), seed=0, deck=deck)
    assert record.deck_supplied is True
    verify_record(record)  # must not try to re-derive the deal from the seed


def test_a_non_default_ruleset_is_carried_through() -> None:
    rules = RuleSet(max_attacks_per_bout=4)
    record = play_game(randoms(), seed=3, rules=rules, validate_states=True)
    assert record.rules.max_attacks_per_bout == 4
    for state in replay(record):
        assert state.attack_limit <= 4


# ----------------------------------------------------------------------
# Replay
# ----------------------------------------------------------------------
def test_replay_reproduces_every_position_of_the_original_game() -> None:
    """Compared by transposition key, not by object identity."""
    for seed in range(20):
        record = play_game(randoms(), seed=seed)
        states = list(replay(record))
        assert len(states) == record.length + 1
        assert states[0].transposition_key() == record.initial_state.transposition_key()
        assert states[-1].is_over
        assert states[-1].durak == record.durak


def test_replay_agrees_with_a_live_rerun_state_for_state() -> None:
    live_keys = []

    def spy(view, legal, rng):
        return rng.choice(legal)

    record = play_game(
        [CallableAgent(spy, "a"), CallableAgent(spy, "b")], seed=77
    )
    for state in replay(record):
        live_keys.append(state.transposition_key())
    assert len(live_keys) == record.length + 1
    assert len(set(live_keys)) == len(live_keys), "a position repeated exactly"


def test_verify_record_accepts_a_genuine_record() -> None:
    for seed in range(15):
        record = play_game(randoms(), seed=seed)
        assert verify_record(record).durak == record.durak


def test_verify_record_catches_a_tampered_outcome() -> None:
    from dataclasses import replace as dc_replace

    record = play_game(randoms(), seed=4)
    flipped = 0 if record.durak == 1 else 1
    tampered = dc_replace(record, durak=flipped)
    with pytest.raises(RecordError, match="durak"):
        verify_record(tampered)


def test_verify_record_catches_a_truncated_move_list() -> None:
    from dataclasses import replace as dc_replace

    record = play_game(randoms(), seed=6)
    short = dc_replace(record, moves=record.moves[:-3])
    with pytest.raises(RecordError, match="did not finish"):
        verify_record(short)


def test_replay_catches_a_move_attributed_to_the_wrong_player() -> None:
    from dataclasses import replace as dc_replace

    record = play_game(randoms(), seed=7)
    entry = record.moves[0]
    broken = dc_replace(
        record,
        moves=(dc_replace(entry, player=1 - entry.player), *record.moves[1:]),
    )
    with pytest.raises(RecordError, match="was to move"):
        list(replay(broken))


def test_replay_catches_an_illegal_recorded_move() -> None:
    from dataclasses import replace as dc_replace

    record = play_game(randoms(), seed=10)
    entry = record.moves[0]
    broken = dc_replace(
        record, moves=(dc_replace(entry, move=END_ATTACK), *record.moves[1:])
    )
    with pytest.raises(RecordError, match="not legal"):
        list(replay(broken))


def test_verify_record_notices_if_the_deal_seed_stops_matching() -> None:
    """Guards against a future change to dealing silently orphaning records."""
    from dataclasses import replace as dc_replace

    record = play_game(randoms(), seed=13)
    drifted = dc_replace(record, deal_seed=record.deal_seed ^ 1)
    with pytest.raises(RecordError, match="no longer reproduces"):
        verify_record(drifted)


# ----------------------------------------------------------------------
# Serialization
# ----------------------------------------------------------------------
def test_records_round_trip_through_json() -> None:
    for seed in range(10):
        record = play_game(randoms(), seed=seed)
        restored = GameRecord.from_dict(json.loads(json.dumps(record.to_dict())))
        assert restored.to_dict() == record.to_dict()
        assert restored.durak == record.durak
        assert [m.move for m in restored.moves] == [m.move for m in record.moves]
        verify_record(restored)


def test_a_restored_record_replays_identically() -> None:
    record = play_game(randoms(), seed=21)
    restored = GameRecord.from_dict(json.loads(json.dumps(record.to_dict())))
    original_keys = [s.transposition_key() for s in replay(record)]
    restored_keys = [s.transposition_key() for s in replay(restored)]
    assert original_keys == restored_keys


def test_an_unknown_record_version_is_refused() -> None:
    record = play_game(randoms(), seed=1)
    payload = record.to_dict()
    payload["version"] = 99
    with pytest.raises(RecordError, match="version"):
        GameRecord.from_dict(payload)


def test_a_record_missing_a_field_is_refused() -> None:
    record = play_game(randoms(), seed=1)
    payload = record.to_dict()
    del payload["moves"]
    with pytest.raises(RecordError, match="missing field"):
        GameRecord.from_dict(payload)


def test_annotations_are_reserved_and_empty_in_this_phase() -> None:
    record = play_game(randoms(), seed=1)
    assert all(entry.annotations == {} for entry in record.moves)
    assert "annotations" not in record.to_dict()["moves"][0]


def test_record_repr_is_informative() -> None:
    text = repr(play_game(randoms(), seed=1))
    assert "GameRecord" in text and "plies" in text


# ----------------------------------------------------------------------
# Adversarial policies through the driver
# ----------------------------------------------------------------------
def test_hostile_policies_still_produce_verifiable_records() -> None:
    """The Phase 2 audit's hostile policies, now driven through Phase 3."""

    def always_take(view, legal, rng):
        for move in legal:
            if move == TAKE:
                return move
        attacks = [m for m in legal if isinstance(m, AttackMove)]
        return attacks[0] if attacks else legal[-1]

    def max_pressure(view, legal, rng):
        attacks = [m for m in legal if isinstance(m, AttackMove)]
        return attacks[0] if attacks else legal[0]

    def hit_and_run(view, legal, rng):
        for move in legal:
            if move == END_ATTACK:
                return move
        return legal[0]

    policies = {"take": always_take, "press": max_pressure, "run": hit_and_run}
    for name_a, a in policies.items():
        for name_b, b in policies.items():
            for seed in range(6):
                record = play_game(
                    [CallableAgent(a, name_a), CallableAgent(b, name_b)],
                    seed=seed,
                    validate_states=True,
                )
                verify_record(record)


def test_a_full_game_never_shows_an_agent_an_empty_choice() -> None:
    def guard(view, legal, rng):
        assert legal, view.describe()
        assert view.phase is not Phase.GAME_OVER
        return rng.choice(legal)

    for seed in range(25):
        play_game([CallableAgent(guard, "a"), CallableAgent(guard, "b")], seed=seed)
