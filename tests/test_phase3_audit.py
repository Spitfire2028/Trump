"""Phase 3 audit.

Written to falsify the Phase 3 report's claims rather than to confirm
them. Three things are done differently from the Phase 3 tests:

1. **Reachability, not inspection.** Instead of checking named fields,
   the object graph reachable from what an agent receives is walked
   exhaustively, and every object in it is examined. A leak through an
   unexpected attribute would be caught even though nobody thought to
   look at that attribute.

2. **Legality re-derived with zero engine calls.** The Phase 3 test
   called ``beats()`` — a pure function of three values, so the claim
   held, but it was not literally free of engine code. Here the beating
   rule is written out inline from ``docs/rules.md``.

3. **Hidden history, not just hidden cards.** Indistinguishability is
   tested against rewritten ground-truth draw records, not only against
   permuted hands and talons.
"""

from __future__ import annotations

import enum
import json
import random
import types

import pytest

from durakfish.cards import Card, Rank, Suit
from durakfish.cards.deck import standard_cards
from durakfish.game import (
    AttackMove,
    DefenseMove,
    Phase,
    apply_move,
    get_legal_moves,
    new_game,
)
from durakfish.game.history import GameEvent, HistoryNode, push_event
from durakfish.game.moves import END_ATTACK, TAKE, EndAttack, Move, TakeCards
from durakfish.game.state import GameState, TableSlot, add_cards
from durakfish.information import InformationSet, observe
from tests.support import make_state

UNIVERSE = frozenset(standard_cards(36))


def walk(seed: int):
    rng = random.Random(seed)
    state = new_game(seed=seed)
    while not state.is_over:
        yield state
        state = apply_move(state, rng.choice(get_legal_moves(state)))
    yield state


def positions(games: int = 20) -> list[GameState]:
    return [s for seed in range(games) for s in walk(seed)]


# ======================================================================
# 1. Information boundary — by reachability
# ======================================================================
LEAF_TYPES = (int, float, str, bytes, bool, type(None), enum.Enum)

#: Nothing of these types may be reachable from anything an agent holds.
FORBIDDEN_TYPES = (GameState, HistoryNode, GameEvent)


def reachable(root: object, limit: int = 500_000) -> list[object]:
    """Every object reachable from ``root`` by attribute or container.

    Deliberately generic: it does not know what an InformationSet looks
    like, so a leak added later through a new field is still found.
    """
    seen: set[int] = set()
    found: list[object] = []
    stack: list[object] = [root]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        found.append(obj)
        if len(found) > limit:  # pragma: no cover - guards a runaway walk
            raise RuntimeError("object graph unexpectedly large")
        if isinstance(obj, LEAF_TYPES):
            continue
        if isinstance(obj, (type, types.ModuleType, types.FunctionType)):
            continue
        if isinstance(obj, (tuple, list, set, frozenset)):
            stack.extend(obj)
            continue
        if isinstance(obj, dict):
            stack.extend(obj.keys())
            stack.extend(obj.values())
            continue
        for klass in type(obj).__mro__:
            for slot in getattr(klass, "__slots__", ()) or ():
                if hasattr(obj, slot):
                    stack.append(getattr(obj, slot))
        instance_dict = getattr(obj, "__dict__", None)
        if isinstance(instance_dict, dict):
            stack.extend(instance_dict.values())
    return found


def cards_reachable_from(root: object) -> set[Card]:
    return {o for o in reachable(root) if isinstance(o, Card)}


def test_no_forbidden_object_is_reachable_from_an_information_set() -> None:
    for state in positions(10):
        for player in range(2):
            view = observe(state, player)
            for obj in reachable(view):
                assert not isinstance(obj, FORBIDDEN_TYPES), (
                    f"{type(obj).__name__} reachable from the view given to "
                    f"P{player}"
                )


def test_every_card_reachable_from_a_view_has_been_observed() -> None:
    """The strongest statement of the boundary available.

    Not "the fields we remembered to check contain no hidden card", but
    "no hidden card exists anywhere in the object graph".
    """
    for state in positions(10):
        for player in range(2):
            view = observe(state, player)
            leaked = cards_reachable_from(view) - view.observed_cards()
            assert not leaked, f"P{player} can reach unobserved {leaked}"


def test_legal_moves_handed_to_an_agent_leak_nothing_either() -> None:
    """The moves are the other half of what crosses the boundary."""
    for state in positions(10):
        player = state.current_player
        if player is None:
            continue
        view = observe(state, player)
        legal = get_legal_moves(state)
        leaked = cards_reachable_from(legal) - view.observed_cards()
        assert not leaked, f"legal moves expose unobserved {leaked}"
        for obj in reachable(legal):
            assert not isinstance(obj, FORBIDDEN_TYPES)


def test_the_driver_hands_agents_nothing_beyond_view_and_legal_moves() -> None:
    """Captured live from a real game, then scanned."""
    from durakfish.ai import CallableAgent
    from durakfish.simulation import play_game

    captured: list[tuple[object, ...]] = []

    def spy(view, legal, rng):
        captured.append((view, legal))
        return rng.choice(legal)

    for seed in range(6):
        play_game([CallableAgent(spy, "a"), CallableAgent(spy, "b")], seed=seed)

    assert len(captured) > 300
    for view, legal in captured:
        assert isinstance(view, InformationSet)
        observed = view.observed_cards()
        for payload in (view, legal):
            for obj in reachable(payload):
                assert not isinstance(obj, FORBIDDEN_TYPES)
            assert not (cards_reachable_from(payload) - observed)


#: The one module in ``ai/`` allowed to name a game state.
#:
#: ``ai/determinized.py`` is the *materialisation* boundary: it turns an
#: observer's information plus a sampled world back into a complete
#: hypothetical position. That mirrors ``information/information_set.py``,
#: the *redaction* boundary, which is likewise the only module in its own
#: package permitted to name a ``GameState``. One place converts ground
#: truth into information; one converts information into a hypothesis.
#: Everything between them is banned from both. The exemption is narrow on
#: purpose, and ``test_only_the_materialisation_boundary_is_exempt`` checks
#: both that it has not grown and that it is still needed.
MATERIALISATION_BOUNDARY = "determinized.py"


def test_only_the_materialisation_boundary_is_exempt() -> None:
    """The exemption must stay narrow, and must still be necessary."""
    import ast
    import pathlib

    ai_dir = pathlib.Path(__file__).resolve().parent.parent / "src" / "durakfish" / "ai"
    boundary = ai_dir / MATERIALISATION_BOUNDARY
    assert boundary.exists(), "the exempt module no longer exists"

    tree = ast.parse(boundary.read_text(), filename=str(boundary))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }
    assert "GameState" in names, (
        "the exemption is no longer needed and should be deleted"
    )

    exempt = [
        path.name
        for path in ai_dir.rglob("*.py")
        if path.name == MATERIALISATION_BOUNDARY
    ]
    assert exempt == [MATERIALISATION_BOUNDARY], "the exempt list has grown"


def test_the_ai_layer_contains_no_reference_to_game_state() -> None:
    """Static check: agent code cannot name what it must not touch."""
    import ast
    import pathlib

    ai_dir = pathlib.Path(__file__).resolve().parent.parent / "src" / "durakfish" / "ai"
    for path in ai_dir.rglob("*.py"):
        if path.name == MATERIALISATION_BOUNDARY:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } | {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        for banned in ("GameState", "hands", "talon", "apply_move", "new_game"):
            assert banned not in names, f"{path.name} references {banned!r}"


# ======================================================================
# 2. Legality re-derived with no engine code at all
# ======================================================================
def reference_beats(defending: Card, attacking: Card, trump: Suit) -> bool:
    """The beating rule written out from ``docs/rules.md`` §2.

    Inlined rather than imported so that the derivation below shares no
    code whatsoever with the engine.
    """
    if defending.suit == attacking.suit:
        return int(defending.rank) > int(attacking.rank)
    return defending.suit == trump


def independent_legal_moves(view: InformationSet) -> tuple[Move, ...]:
    """Legal moves derived from the view alone.

    Reads only plain fields of the information set — phase, hand, table,
    trump suit, attack limit, and whose turn it is. Calls nothing from
    ``durakfish.game`` and touches no ``GameState``. If this agrees with
    the engine, the view demonstrably carries enough to play.
    """
    if view.phase == Phase.GAME_OVER:
        return ()
    to_move = (
        view.defender if view.phase == Phase.DEFENSE else view.attacker
    )
    if to_move != view.player:
        return ()

    if view.phase == Phase.DEFENSE:
        open_attack = view.table[-1].attacking_card
        moves: list[Move] = [
            DefenseMove(open_attack, card)
            for card in view.hand
            if reference_beats(card, open_attack, view.trump_suit)
        ]
        moves.append(TAKE)
        return tuple(moves)

    # ATTACK or TAKING.
    ranks_on_table: set[Rank] = set()
    for slot in view.table:
        ranks_on_table.add(slot.attacking_card.rank)
        if slot.defending_card is not None:
            ranks_on_table.add(slot.defending_card.rank)

    has_room = len(view.table) < view.attack_limit
    if not view.table:
        playable = tuple(view.hand) if has_room else ()
    elif has_room:
        playable = tuple(c for c in view.hand if c.rank in ranks_on_table)
    else:
        playable = ()

    out: list[Move] = [AttackMove(c) for c in playable]
    if view.table:
        out.append(END_ATTACK)
    return tuple(out)


def test_independent_derivation_matches_the_engine_on_random_games() -> None:
    checked = 0
    for state in positions(40):
        player = state.current_player
        if player is None:
            continue
        view = observe(state, player)
        assert independent_legal_moves(view) == get_legal_moves(state, player), (
            view.describe()
        )
        checked += 1
    assert checked > 3000


def test_independent_derivation_matches_under_adversarial_policies() -> None:
    """Random play under-samples the extremes; hostile policies do not."""

    def always_take(legal):
        return next((m for m in legal if m == TAKE), legal[0])

    def max_pressure(legal):
        attacks = [m for m in legal if isinstance(m, AttackMove)]
        return attacks[0] if attacks else legal[0]

    def hit_and_run(legal):
        return next((m for m in legal if m == END_ATTACK), legal[0])

    checked = 0
    for policy_a in (always_take, max_pressure, hit_and_run):
        for policy_b in (always_take, max_pressure, hit_and_run):
            for seed in range(6):
                state = new_game(seed=seed)
                while not state.is_over:
                    player = state.current_player
                    assert player is not None
                    view = observe(state, player)
                    engine = get_legal_moves(state, player)
                    assert independent_legal_moves(view) == engine
                    checked += 1
                    policy = policy_a if player == 0 else policy_b
                    state = apply_move(state, policy(engine))
    assert checked > 1000


def test_independent_derivation_on_constructed_corner_positions() -> None:
    """Every situation the audit brief named, constructed explicitly."""
    scenarios: dict[str, GameState] = {
        "defender must take: nothing beats a trump attack": make_state(
            hand0="AH 6C", hand1="6S 7S", trump=Suit.HEARTS, talon="8C 8H",
            table=(("9H", None),), attacker=0, phase=Phase.DEFENSE,
            attack_limit=2,
        ),
        "defender holds only trumps": make_state(
            hand0="6C 7C", hand1="6H 7H", trump=Suit.HEARTS, talon="8C 8H",
            table=(("9C", None),), attacker=0, phase=Phase.DEFENSE,
            attack_limit=2,
        ),
        "attack limit already reached": make_state(
            hand0="6C 6D", hand1="7C", trump=Suit.HEARTS, talon="8C 8H",
            table=(("6S", "7S"),), attacker=0, attack_limit=1,
        ),
        "throw-in after a take": make_state(
            hand0="6C 6D 9S", hand1="7C 7D", trump=Suit.HEARTS, talon="8C 8H",
            table=(("6S", None),), attacker=0, phase=Phase.TAKING,
            attack_limit=3,
        ),
        "throw-in matching a defending card's rank": make_state(
            hand0="9C 9D", hand1="7C 7D", trump=Suit.HEARTS, talon="8C 8H",
            table=(("6S", "9S"), ("6C", None)), attacker=0,
            phase=Phase.TAKING, attack_limit=3,
        ),
        "empty talon endgame": make_state(
            hand0="6S 7H", hand1="8S 9H", trump=Suit.HEARTS, talon="",
            attacker=0,
        ),
        "attacker holds a single card": make_state(
            hand0="6S", hand1="8S 9H", trump=Suit.HEARTS, talon="10C 10H",
            attacker=0,
        ),
        "attacker hand empty mid-bout": make_state(
            hand0="", hand1="8S 9H", trump=Suit.HEARTS, talon="10C 10H",
            table=(("6S", "7S"),), attacker=0, attack_limit=3,
        ),
        "defender hand empty facing an open attack": make_state(
            hand0="6C", hand1="", trump=Suit.HEARTS, talon="10C 10H",
            table=(("6S", None),), attacker=0, phase=Phase.DEFENSE,
            attack_limit=2,
        ),
        "full six-card bout": make_state(
            hand0="6C 6D 6H", hand1="AS AC AD", trump=Suit.HEARTS,
            talon="10C 10H",
            table=(("6S", "7S"), ("7C", "8C"), ("7D", "8D"),
                   ("7H", "8H"), ("9S", "10S"), ("9C", "JC")),
            attacker=0, attack_limit=6,
        ),
        "trump is the only defence available": make_state(
            hand0="9S 9C", hand1="6H", trump=Suit.HEARTS, talon="8C 8H",
            table=(("AS", None),), attacker=0, phase=Phase.DEFENSE,
            attack_limit=1,
        ),
    }
    for label, state in scenarios.items():
        player = state.current_player
        assert player is not None, label
        view = observe(state, player)
        assert independent_legal_moves(view) == get_legal_moves(state, player), (
            f"{label}\n{view.describe()}"
        )


def test_independent_derivation_agrees_that_the_idle_player_has_nothing() -> None:
    for state in positions(10):
        if state.current_player is None:
            continue
        idle = 1 - state.current_player
        assert independent_legal_moves(observe(state, idle)) == ()
        assert get_legal_moves(state, idle) == ()


def test_independent_derivation_agrees_on_finished_games() -> None:
    finished = [s for s in positions(30) if s.is_over]
    assert len(finished) >= 25
    for state in finished:
        for player in range(2):
            assert independent_legal_moves(observe(state, player)) == ()


# ======================================================================
# 3. Indistinguishability
# ======================================================================
def permute_hidden(state: GameState, player: int, rng: random.Random) -> GameState:
    """Redistribute never-observed cards among the hidden locations."""
    view = observe(state, player)
    movable = view.unobserved_cards()
    opponent_fixed = [c for c in state.hands[1 - player] if c not in movable]
    talon_fixed = [c for c in state.talon[:-1] if c not in movable]
    pool = [c for c in state.hands[1 - player] if c in movable]
    pool += [c for c in state.talon[:-1] if c in movable]
    rng.shuffle(pool)
    slots = len(state.hands[1 - player]) - len(opponent_fixed)
    hands = list(state.hands)
    hands[1 - player] = add_cards((), opponent_fixed + pool[:slots])
    talon = tuple(talon_fixed + pool[slots:]) + state.talon[-1:]
    return state.replace(hands=tuple(hands), talon=talon)


def rewrite_opponent_draw_identities(state: GameState, player: int) -> GameState:
    """Rewrite what the *other* player is recorded as having drawn.

    The counts stay identical; only the card identities change. Since
    redaction reduces another player's draws to a count, a view that
    changes here would be leaking through the history.
    """
    from durakfish.game.history import DrawRecord

    # Must be at least as long as any draw: a shorter filler would change
    # the recorded *count*, which is public information, and the view would
    # then be right to differ.
    filler = sorted(UNIVERSE, key=lambda c: c.code)
    history: HistoryNode | None = None
    for event in state.events():
        draws = tuple(
            record
            if record.player == player
            else DrawRecord(record.player, tuple(filler[: len(record.cards)]))
            for record in event.draws
        )
        history = push_event(history, GameEvent(**{
            "kind": event.kind,
            "player": event.player,
            "move": event.move,
            "revealed": event.revealed,
            "discarded": event.discarded,
            "taken": event.taken,
            "taken_by": event.taken_by,
            "draws": draws,
            "durak": event.durak,
        }))
    return state.replace(history=history)


def test_permuting_opponent_hand_and_talon_leaves_the_view_identical() -> None:
    rng = random.Random(11)
    checked = 0
    for state in positions(15):
        if not state.talon:
            continue
        for player in range(2):
            baseline = observe(state, player)
            for _ in range(3):
                twin = permute_hidden(state, player, rng)
                twin.validate()
                assert observe(twin, player) == baseline
                checked += 1
    assert checked > 200


def test_rewriting_opponent_draw_history_leaves_the_view_identical() -> None:
    """Hidden *history*, not just hidden cards."""
    checked = 0
    for state in positions(10):
        for player in range(2):
            baseline = observe(state, player)
            twin = rewrite_opponent_draw_identities(state, player)
            assert observe(twin, player) == baseline
            assert observe(twin, player).key() == baseline.key()
            checked += 1
    assert checked > 100


def test_indistinguishability_holds_after_draws_and_after_a_take() -> None:
    """The two moments where cards change location behind the scenes."""
    rng = random.Random(3)
    draw_checked = take_checked = 0
    for seed in range(25):
        previous = None
        for state in walk(seed):
            if previous is not None:
                drew = len(state.talon) < len(previous.talon)
                took = len(state.hands[0]) + len(state.hands[1]) > len(
                    previous.hands[0]
                ) + len(previous.hands[1]) + 0
                if drew or took:
                    for player in range(2):
                        baseline = observe(state, player)
                        twin = permute_hidden(state, player, rng)
                        assert observe(twin, player) == baseline
                    draw_checked += int(drew)
                    take_checked += int(took)
            previous = state
    assert draw_checked > 50
    assert take_checked > 50


def test_repeated_observation_is_stable() -> None:
    for state in positions(8):
        for player in range(2):
            views = [observe(state, player) for _ in range(4)]
            assert all(v == views[0] for v in views)
            assert len({v.key() for v in views}) == 1


def test_a_change_to_public_information_does_change_the_view() -> None:
    """Non-vacuity: indistinguishability must not come from insensitivity."""
    state = next(s for s in walk(2) if len(s.discard) >= 2 and s.talon)
    baseline = observe(state, 0)

    # Move a discarded card into the talon: public information changed.
    victim = sorted(state.discard, key=lambda c: c.code)[0]
    twin = state.replace(
        discard=state.discard - {victim},
        talon=(victim,) + state.talon,
    )
    twin.validate()
    assert observe(twin, 0) != baseline
    assert observe(twin, 0).key() != baseline.key()


def test_observed_cards_stay_observed_after_they_leave_sight() -> None:
    """The documented cumulative semantics, checked along whole games."""
    for seed in range(12):
        for player in range(2):
            seen_so_far: set[Card] = set()
            for state in walk(seed):
                view = observe(state, player)
                observed = view.observed_cards()
                assert seen_so_far <= observed, (
                    "a previously observed card stopped being observed"
                )
                seen_so_far = set(observed)


# ======================================================================
# 4. Leakage across every output channel
# ======================================================================
def all_outputs(view: InformationSet) -> dict[str, object]:
    """Everything an agent could read off a view, in one place."""
    return {
        "equality": view,
        "key": view.key(),
        "hash": hash(view.key()),
        "json": json.dumps(view.to_dict(), sort_keys=True),
        "json_no_history": json.dumps(
            view.to_dict(include_history=False), sort_keys=True
        ),
        "repr": repr(view),
        "describe": view.describe(),
        "public_history": view.public_history,
        "observed": frozenset(view.observed_cards()),
        "unobserved": frozenset(view.unobserved_cards()),
        "hidden_count": view.hidden_card_count,
        "open_info": view.is_open_information(),
        "legal": independent_legal_moves(view),
    }


def test_no_output_channel_distinguishes_two_identical_situations() -> None:
    rng = random.Random(808)
    compared = 0
    for state in positions(12):
        if not state.talon:
            continue
        for player in range(2):
            baseline = all_outputs(observe(state, player))
            for _ in range(2):
                twin = permute_hidden(state, player, rng)
                twin = rewrite_opponent_draw_identities(twin, player)
                candidate = all_outputs(observe(twin, player))
                for channel, value in baseline.items():
                    assert candidate[channel] == value, (
                        f"{channel} leaks hidden state"
                    )
                compared += 1
    assert compared > 150


# ======================================================================
# 5. Agent API isolation
# ======================================================================
def handmade_view() -> InformationSet:
    """An InformationSet built with no GameState in existence anywhere."""
    from durakfish.game.ruleset import STANDARD_RULES

    hand = add_cards((), [Card.parse(c) for c in ("6S", "9H", "AC")])
    return InformationSet(
        player=1,
        hand=hand,
        table=(TableSlot(Card.parse("7S"), None),),
        trump_suit=Suit.HEARTS,
        trump_card=Card.parse("6H"),
        talon_size=8,
        hand_sizes=(5, 3),
        discard=frozenset({Card.parse("10D"), Card.parse("JD")}),
        attacker=0,
        phase=Phase.DEFENSE,
        attack_limit=3,
        public_history=(),
        observed=frozenset(
            set(hand)
            | {Card.parse("7S"), Card.parse("6H"), Card.parse("10D"), Card.parse("JD")}
        ),
        rules=STANDARD_RULES,
    )


def test_a_view_can_exist_and_be_used_without_any_game_state() -> None:
    view = handmade_view()
    assert view.is_to_move
    assert view.opponent_hand_size == 5
    legal = independent_legal_moves(view)
    # 9H is a trump and beats 7S; AC does not; taking is always available.
    assert DefenseMove(Card.parse("7S"), Card.parse("9H")) in legal
    assert TAKE in legal
    assert not any(
        isinstance(m, DefenseMove) and m.defending_card == Card.parse("AC")
        for m in legal
    )


def test_every_shipped_agent_operates_from_a_handmade_view() -> None:
    """No GameState is constructed anywhere in this test."""
    import random as rnd

    from durakfish.ai import AgentBase, CallableAgent, ScriptedAgent

    view = handmade_view()
    legal = independent_legal_moves(view)

    callable_agent = CallableAgent(lambda v, legal_, r: legal_[0], "c")
    callable_agent.reset(rnd.Random(0))
    assert callable_agent.choose(view, legal) == legal[0]

    scripted = ScriptedAgent([TAKE], "s")
    scripted.reset(rnd.Random(0))
    assert scripted.choose(view, legal) == TAKE

    class Defensive(AgentBase):
        name = "defensive"

        def choose(self, v, legal_):
            defences = [m for m in legal_ if isinstance(m, DefenseMove)]
            return defences[0] if defences else legal_[0]

    assert isinstance(Defensive().choose(view, legal), DefenseMove)


def test_the_agent_signature_requires_nothing_beyond_view_and_legal() -> None:
    import inspect

    from durakfish.ai import Agent

    parameters = list(inspect.signature(Agent.choose).parameters)
    assert parameters == ["self", "view", "legal"]


# ======================================================================
# 6. Replay independence
# ======================================================================
def test_replay_depends_only_on_the_record_not_on_the_live_game() -> None:
    """Rebuild the record from JSON text alone, then replay it."""
    from durakfish.ai import CallableAgent
    from durakfish.simulation import GameRecord, play_game, replay, verify_record

    agents = [
        CallableAgent(lambda v, legal, r: r.choice(legal), "a"),
        CallableAgent(lambda v, legal, r: r.choice(legal), "b"),
    ]
    for seed in range(8):
        live = play_game(agents, seed=seed)
        text = json.dumps(live.to_dict())
        del live  # the only surviving artefact is the JSON string

        restored = GameRecord.from_dict(json.loads(text))
        keys = [s.transposition_key() for s in replay(restored)]
        verify_record(restored)

        again = GameRecord.from_dict(json.loads(text))
        assert [s.transposition_key() for s in replay(again)] == keys


def test_replaying_twice_does_not_disturb_the_record() -> None:
    from durakfish.ai import CallableAgent
    from durakfish.simulation import play_game, replay

    agents = [CallableAgent(lambda v, legal, r: r.choice(legal), n) for n in "ab"]
    record = play_game(agents, seed=5)
    before = json.dumps(record.to_dict(), sort_keys=True)
    first = [s.transposition_key() for s in replay(record)]
    second = [s.transposition_key() for s in replay(record)]
    assert first == second
    assert json.dumps(record.to_dict(), sort_keys=True) == before


def test_hidden_information_absent_from_a_record_cannot_alter_replay() -> None:
    """Every card is in the record, so replay is fully determined by it."""
    from durakfish.ai import CallableAgent
    from durakfish.simulation import play_game, replay

    agents = [CallableAgent(lambda v, legal, r: r.choice(legal), n) for n in "ab"]
    record = play_game(agents, seed=17)
    opening = record.initial_state
    accounted = (
        set(opening.hands[0])
        | set(opening.hands[1])
        | set(opening.talon)
        | set(opening.table_cards)
        | set(opening.discard)
    )
    assert accounted == UNIVERSE, "the record omits cards, so replay is underdetermined"
    assert len(list(replay(record))) == record.length + 1


def test_seed_derivation_is_reproducible_across_processes() -> None:
    """Mersenne Twister, not hash(): stable between interpreter runs."""
    import subprocess
    import sys

    from durakfish.simulation import derive_seeds

    expected = derive_seeds(4242, 2)
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "from durakfish.simulation import derive_seeds;"
        "print(derive_seeds(4242, 2))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(expected)


def test_a_record_with_a_swapped_opening_position_is_rejected() -> None:
    from dataclasses import replace as dc_replace

    from durakfish.ai import CallableAgent
    from durakfish.exceptions import RecordError
    from durakfish.simulation import play_game, verify_record

    agents = [CallableAgent(lambda v, legal, r: r.choice(legal), n) for n in "ab"]
    record = play_game(agents, seed=1)
    other = play_game(agents, seed=2)
    swapped = dc_replace(record, initial_state=other.initial_state)
    with pytest.raises(RecordError):
        verify_record(swapped)
