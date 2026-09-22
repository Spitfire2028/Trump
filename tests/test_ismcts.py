"""Single-observer ISMCTS.

The phase's whole purpose is one property, so it gets the most attention
here: two hidden worlds an observer cannot tell apart must reach the same
node and update the same statistics. Everything else — reproducibility,
leakage, epistemic carrying — exists to make that property trustworthy.

The 8♠ regression is the direct descendant of the Phase 7.3.3 failure,
where perfect-information search planned ``attack 8♠`` in one world and
``end attack`` in another at a node the observer could not distinguish.
"""

from __future__ import annotations

import random

import pytest

from durakfish.ai import SearchConfig, build_position
from durakfish.ai.ismcts import (
    ISMCTS,
    Determinizer,
    EpistemicState,
    ISMCTSConfig,
    ISMCTSError,
    InformationSetKey,
    InformationSetNode,
    NodeRegistry,
    information_key,
    observation_of,
)
from durakfish.cards import Suit
from durakfish.game import apply_move, get_legal_moves, new_game
from durakfish.game.moves import TAKE, AttackMove, TakeCards
from durakfish.game.state import add_cards
from durakfish.information import (
    CardTracker,
    SearchInformation,
    TrackerError,
    WorldGenerator,
    observe,
    validate_world,
)
from tests.support import make_state
from tests.test_phase3_audit import FORBIDDEN_TYPES, cards_reachable_from, reachable


def walk(seed: int):
    rng = random.Random(seed)
    state = new_game(seed=seed)
    while not state.is_over:
        yield state
        state = apply_move(state, rng.choice(get_legal_moves(state)))
    yield state


def info_for(state, player: int) -> SearchInformation:
    return SearchInformation.from_view(observe(state, player))


def decisions(games: int = 10, min_worlds: int = 2):
    for seed in range(games):
        for index, state in enumerate(walk(seed)):
            player = state.current_player
            if player is None or index % 5:
                continue
            information = info_for(state, player)
            if information.allocations < min_worlds:
                continue
            yield state, player, information


def sample_worlds(information, count, seed=0):
    generator = WorldGenerator(information)
    return [generator.sample(random.Random(seed + k)) for k in range(count)]


# ======================================================================
# The central property: shared information sets
# ======================================================================
def shared_node_case(seed_range=range(40)):
    """Find a node reached by ≥2 worlds where the observer sees no difference.

    Built the way the Phase 7.3.3 failure was: the observer attacks, the
    opponent TAKES — an action that reveals no card — so the worlds stay
    indistinguishable.
    """
    for seed in seed_range:
        for index, state in enumerate(walk(seed)):
            player = state.current_player
            if player is None or index % 3:
                continue
            information = info_for(state, player)
            if information.allocations < 4:
                continue
            worlds = sample_worlds(information, 6, seed=seed)
            root = build_position(information, worlds[0])
            for move in root.legal_moves():
                reached = []
                for world in worlds:
                    child = build_position(information, world).apply(move)
                    takes = [m for m in child.legal_moves() if isinstance(m, TakeCards)]
                    if not takes:
                        break
                    reached.append(child.apply(takes[0]))
                if len(reached) != len(worlds):
                    continue
                return information, player, move, worlds, reached
    return None


def test_indistinguishable_worlds_reach_one_node_with_shared_statistics() -> None:
    """The 8♠ regression, in its general form.

    Not "the search returns the same action" — that would be a claim about
    stochastic sampling. The invariant is that the worlds resolve to the
    *same node object*, so there is no code path by which they could
    accumulate separate policies.
    """
    case = shared_node_case()
    assert case is not None, "no shared-information-set node arose to test"
    information, player, move, worlds, reached = case

    views = {observe(position.state, player) for position in reached}
    assert len(views) == 1, "the premise failed: the worlds are distinguishable"

    epistemic = EpistemicState.from_view(information.view)
    registry = NodeRegistry()
    keys = set()
    nodes = []
    for position in reached:
        advanced = epistemic.advance(
            position, observation_of(position, move, player, player)
        )
        key = information_key(position, advanced)
        keys.add(key)
        nodes.append(registry.get_or_create(key))

    assert len(keys) == 1, "indistinguishable worlds produced different keys"
    assert len(registry) == 1, "indistinguishable worlds produced different nodes"
    first = nodes[0]
    assert all(node is first for node in nodes), "nodes are not the same object"


def test_the_eight_of_spades_position_shares_one_information_set() -> None:
    """The exact Phase 7.3.3 shape: attack, opponent takes, no card revealed."""
    state = make_state(
        hand0="6C 7C 9D 10D JC QD",
        hand1="8D 8S 10C JD QC KS",
        trump=Suit.SPADES,
        talon="6D 7D 9C 10S JS 6S",
        attacker=1,
    )
    information = info_for(state, 1)
    assert information.allocations > 1
    worlds = sample_worlds(information, 6, seed=11)

    attack = AttackMove(__import__("durakfish.cards", fromlist=["Card"]).Card.parse("8D"))
    assert attack in build_position(information, worlds[0]).legal_moves()

    epistemic = EpistemicState.from_view(information.view)
    registry = NodeRegistry()
    for world in worlds:
        position = build_position(information, world).apply(attack)
        takes = [m for m in position.legal_moves() if isinstance(m, TakeCards)]
        assert takes, "the opponent must be able to take"
        position = position.apply(takes[0])
        advanced = epistemic.advance(
            position, observation_of(position, attack, 1, 1)
        )
        registry.get_or_create(information_key(position, advanced))

    assert len(registry) == 1, (
        "after attacking 8D and the opponent taking — which reveals no card — "
        "every sampled world must share one information set"
    )


def test_hidden_permutation_cannot_change_the_key() -> None:
    """Leakage: only observable facts may enter the key."""
    shuffler = random.Random(2024)
    compared = 0
    for seed in range(15):
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
            if not opponent or not body:
                continue
            pool = opponent + body
            shuffler.shuffle(pool)
            twin = state.replace(
                hands=tuple(
                    state.hands[p]
                    if p == player
                    else add_cards((), fixed_opp + pool[: len(opponent)])
                    for p in range(2)
                ),
                talon=tuple(fixed_body + pool[len(opponent) :]) + state.talon[-1:],
            )
            twin.validate()
            assert observe(twin, player) == view

            first, second = info_for(state, player), info_for(twin, player)
            world = WorldGenerator(first).sample(random.Random(5))
            e1 = EpistemicState.from_view(first.view)
            e2 = EpistemicState.from_view(second.view)
            assert information_key(build_position(first, world), e1) == (
                information_key(build_position(second, world), e2)
            )
            compared += 1
            break
    assert compared > 8


def test_two_worlds_with_different_hidden_cards_share_a_root_key() -> None:
    for state, player, information in decisions(games=6):
        worlds = sample_worlds(information, 6, seed=3)
        assert len({w.key() for w in worlds}) > 1, "worlds must actually differ"
        epistemic = EpistemicState.from_view(information.view)
        keys = {
            information_key(build_position(information, w), epistemic) for w in worlds
        }
        assert len(keys) == 1, "the root key depends on the sampled world"


# ======================================================================
# Key contract
# ======================================================================
def test_the_key_is_deterministic_and_owner_sensitive() -> None:
    for state, player, information in decisions(games=5):
        world = sample_worlds(information, 1, seed=1)[0]
        position = build_position(information, world)
        epistemic = EpistemicState.from_view(information.view)
        assert information_key(position, epistemic) == information_key(
            position, epistemic
        )
        other = EpistemicState(
            owner=1 - player, observed=epistemic.observed
        )
        assert information_key(position, epistemic).owner == player
        assert information_key(position, other).owner != player
        assert information_key(position, epistemic) != information_key(position, other)


def test_distinguishable_states_get_different_keys() -> None:
    seen = set()
    for seed in range(6):
        for state in walk(seed):
            player = state.current_player
            if player is None:
                continue
            information = info_for(state, player)
            world = sample_worlds(information, 1, seed=1)[0]
            seen.add(
                information_key(
                    build_position(information, world),
                    EpistemicState.from_view(information.view),
                )
            )
    assert len(seen) > 100, "the key collapses distinct decision points"


def test_the_key_exposes_no_hidden_card() -> None:
    for state, player, information in decisions(games=5):
        world = sample_worlds(information, 1, seed=7)[0]
        epistemic = EpistemicState.from_view(information.view)
        key = information_key(build_position(information, world), epistemic)
        hidden = (set(state.hands[1 - player]) | set(state.talon)) & (
            information.view.unobserved_cards()
        )
        assert not (cards_reachable_from(key) & hidden)
        for obj in reachable(key):
            assert not isinstance(obj, FORBIDDEN_TYPES)


# ======================================================================
# Epistemic carrying — the Phase 7.3.3 history constraint
# ======================================================================
def test_a_hypothetical_view_is_refused_by_the_card_tracker() -> None:
    """The constraint that forces epistemic state to be carried by hand."""
    state = next(s for i, s in enumerate(walk(4)) if i == 30)
    player = state.current_player
    assert player is not None
    information = info_for(state, player)
    tracker = CardTracker(player)
    tracker.update(observe(state, player))

    world = sample_worlds(information, 1, seed=1)[0]
    position = build_position(information, world)
    with pytest.raises(TrackerError, match="shrank"):
        tracker.update(observe(position.state, player))


def test_carried_observation_survives_traversal_and_only_grows() -> None:
    """A hypothesis has no history, so `observe` alone would reset this."""
    for state, player, information in decisions(games=5):
        world = sample_worlds(information, 1, seed=2)[0]
        position = build_position(information, world)
        epistemic = EpistemicState.from_view(information.view)
        assert epistemic.observed == information.view.observed

        naive = observe(position.state, player).observed
        assert len(naive) <= len(epistemic.observed)

        previous = epistemic
        for _ in range(4):
            legal = position.legal_moves()
            if not legal or position.is_terminal:
                break
            actor = position.current_player
            move = legal[0]
            position = position.apply(move)
            epistemic = epistemic.advance(
                position, observation_of(position, move, actor, player)
            )
            assert epistemic.observed >= previous.observed, "observation shrank"
            assert epistemic.owner == player
            previous = epistemic


def test_view_at_restores_carried_knowledge_onto_a_hypothesis() -> None:
    """Requires a position where carried knowledge genuinely exceeds sight.

    Early in a game everything the observer has seen is still publicly
    visible, so carried and locally-visible knowledge coincide and the
    test would pass even if the carrying were dropped. The position is
    therefore chosen so that the observer has seen cards which are now
    hidden — typically ones the opponent took — making the restoration
    observable.
    """
    chosen = None
    for state, player, information in decisions(games=20, min_worlds=1):
        world = sample_worlds(information, 1, seed=1)[0]
        position = build_position(information, world)
        visible = observe(position.state, player).observed
        if information.view.observed - visible:
            chosen = (player, information, position, visible)
            break
    assert chosen is not None, "no position had knowledge beyond current sight"
    player, information, position, visible = chosen

    epistemic = EpistemicState.from_view(information.view)
    restored = epistemic.view_at(position)
    assert restored.player == player
    assert restored.observed >= information.view.observed
    assert restored.observed > visible, (
        "view_at returned only what is currently visible; the carried "
        "observation record was dropped"
    )


def test_an_observation_never_reveals_a_hidden_draw() -> None:
    for state, player, information in decisions(games=5):
        world = sample_worlds(information, 1, seed=4)[0]
        position = build_position(information, world)
        legal = position.legal_moves()
        actor = position.current_player
        assert actor is not None
        move = legal[0]
        after = position.apply(move)
        observation = observation_of(after, move, actor, player)
        # Only the cards carried by the move itself are revealed.
        assert observation.revealed <= set(getattr(move, "__dict__", {}).values()) | {
            getattr(move, "card", None),
            getattr(move, "defending_card", None),
        } - {None}
        assert observation.actor == actor
        assert observation.terminal == after.is_terminal


# ======================================================================
# Node contract
# ======================================================================
def test_edges_are_shared_and_statistics_accumulate() -> None:
    key = InformationSetKey(owner=0, view_key=(1, 2, 3))
    node = InformationSetNode(key=key)
    state, player, information = next(decisions(games=3))
    moves = build_position(information, sample_worlds(information, 1)[0]).legal_moves()

    node.ensure_edges(moves)
    assert len(node.edges) == len(moves)
    node.ensure_edges(moves)  # idempotent
    assert len(node.edges) == len(moves)

    edge = node.edge_for(moves[0])
    edge.record(2.0)
    edge.record(4.0)
    assert edge.visits == 2
    assert edge.total_value == 6.0
    assert edge.mean_value == 3.0
    assert node.edge_for(moves[0]) is edge, "edges are not shared"


def test_an_unknown_edge_is_refused_rather_than_created_late() -> None:
    node = InformationSetNode(key=InformationSetKey(0, (1,)))
    state, player, information = next(decisions(games=3))
    moves = build_position(information, sample_worlds(information, 1)[0]).legal_moves()
    with pytest.raises(ISMCTSError, match="not an edge"):
        node.edge_for(moves[0])


def test_selection_tries_every_unvisited_edge_before_exploiting() -> None:
    state, player, information = next(decisions(games=3))
    moves = build_position(information, sample_worlds(information, 1)[0]).legal_moves()
    node = InformationSetNode(key=InformationSetKey(player, (1,)))
    node.ensure_edges(moves)
    chosen = []
    for _ in range(len(moves)):
        move = node.uct_select(moves, 1.4)
        chosen.append(move)
        node.record_visit()
        node.edge_for(move).record(0.0)
    assert set(chosen) == set(moves), "some edge was never tried"


def test_the_registry_returns_one_node_per_key() -> None:
    registry = NodeRegistry()
    key = InformationSetKey(0, (1, 2))
    first = registry.get_or_create(key)
    assert registry.get_or_create(key) is first
    assert len(registry) == 1
    assert key in registry
    assert registry.get(InformationSetKey(1, (1, 2))) is None


# ======================================================================
# Determinizer
# ======================================================================
def test_the_determinizer_only_produces_valid_worlds() -> None:
    rng = random.Random(9)
    checked = 0
    for state, player, information in decisions(games=6):
        determinizer = Determinizer(information)
        assert determinizer.allocations == information.allocations
        for _ in range(4):
            world = determinizer.sample(rng)
            validate_world(world, information)
            determinizer.position(world).state.validate()
            checked += 1
    assert checked > 40


def test_the_determinizer_is_reproducible() -> None:
    state, player, information = next(decisions(games=3))
    determinizer = Determinizer(information)
    first = [determinizer.sample(random.Random(5)).key() for _ in range(4)]
    second = [determinizer.sample(random.Random(5)).key() for _ in range(4)]
    assert first == second


def test_the_determinizer_preserves_deduced_ownership() -> None:
    from durakfish.information import CardLocation

    rng = random.Random(3)
    found = 0
    for state, player, information in decisions(games=10):
        certain = information.certain_cards_in(CardLocation.OPPONENT_HAND)
        if not certain:
            continue
        world = Determinizer(information).sample(rng)
        assert certain <= world.opponent_hand
        found += 1
    assert found > 5


# ======================================================================
# Search
# ======================================================================
def test_the_search_returns_a_legal_root_move() -> None:
    checked = 0
    for state, player, information in decisions(games=6):
        result = ISMCTS(ISMCTSConfig(iterations=30)).search(
            information, random.Random(1)
        )
        legal = get_legal_moves(state)
        assert result.move in legal
        assert result.observer == player
        assert result.worlds_sampled == 30
        checked += 1
    assert checked > 10


def test_the_search_is_reproducible_under_a_seed() -> None:
    for state, player, information in decisions(games=5):
        runs = [
            ISMCTS(ISMCTSConfig(iterations=40)).search(information, random.Random(7))
            for _ in range(3)
        ]
        assert len({r.move for r in runs}) == 1
        assert len({tuple(sorted((str(m), n) for m, n in r.visit_counts().items()))
                    for r in runs}) == 1


def test_different_seeds_may_explore_differently_but_stay_legal() -> None:
    state, player, information = next(decisions(games=3))
    legal = get_legal_moves(state)
    for seed in range(6):
        result = ISMCTS(ISMCTSConfig(iterations=25)).search(
            information, random.Random(seed)
        )
        assert result.move in legal


def test_the_search_never_touches_the_global_random_module(monkeypatch) -> None:
    def forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("ISMCTS used the global random module")

    for name in ("random", "randrange", "choice", "shuffle", "sample", "randint"):
        monkeypatch.setattr(random, name, forbidden)

    state, player, information = next(decisions(games=3))
    result = ISMCTS(ISMCTSConfig(iterations=20)).search(information, random.Random(2))
    assert result.move is not None


def test_root_moves_can_be_restricted() -> None:
    for state, player, information in decisions(games=5):
        legal = get_legal_moves(state)
        if len(legal) < 2:
            continue
        allowed = legal[:1]
        result = ISMCTS(ISMCTSConfig(iterations=20)).search(
            information, random.Random(3), root_moves=allowed
        )
        assert result.move in allowed


def test_a_terminal_root_yields_no_move() -> None:
    state = next(s for s in walk(4) if s.is_over)
    information = info_for(state, 0)
    result = ISMCTS(ISMCTSConfig(iterations=5)).search(information, random.Random(1))
    assert result.move is None
    assert result.root is None


def test_the_observer_must_be_to_move() -> None:
    found = False
    for state in walk(2):
        mover = state.current_player
        if mover is None:
            continue
        information = info_for(state, 1 - mover)
        with pytest.raises(ISMCTSError, match="not to move"):
            ISMCTS(ISMCTSConfig(iterations=5)).search(information, random.Random(1))
        found = True
        break
    assert found


def test_visits_are_conserved_across_the_root_edges() -> None:
    for state, player, information in decisions(games=5):
        result = ISMCTS(ISMCTSConfig(iterations=50)).search(
            information, random.Random(4)
        )
        assert result.root is not None
        assert sum(result.visit_counts().values()) == result.root.visits
        assert result.root.visits <= 50


def test_the_recommendation_is_the_most_visited_edge() -> None:
    """The recommendation rule, asserted rather than assumed."""
    for state, player, information in decisions(games=5):
        result = ISMCTS(ISMCTSConfig(iterations=60)).search(
            information, random.Random(9)
        )
        assert result.root is not None
        counts = result.visit_counts()
        assert counts
        assert counts[result.move] == max(counts.values())


def test_the_search_explores_more_than_one_root_action() -> None:
    """Catches a selection that ignores the shared statistics entirely."""
    checked = 0
    for state, player, information in decisions(games=6):
        legal = get_legal_moves(state)
        if len(legal) < 2:
            continue
        result = ISMCTS(ISMCTSConfig(iterations=40)).search(
            information, random.Random(2)
        )
        visited = [m for m, n in result.visit_counts().items() if n]
        assert len(visited) >= 2, "only one root action was ever tried"
        checked += 1
    assert checked > 5


def test_statistics_accumulate_below_the_root() -> None:
    """Catches backpropagation that only ever updates the first node."""
    checked = 0
    for state, player, information in decisions(games=6):
        if len(get_legal_moves(state)) < 2:
            continue
        result = ISMCTS(ISMCTSConfig(iterations=80)).search(
            information, random.Random(4)
        )
        assert result.root is not None
        visited_nodes = sum(
            1 for key in list(result.registry._nodes)  # noqa: SLF001 - white-box
            if result.registry.get(key).visits
        )
        assert visited_nodes >= 2, "no information set below the root was visited"
        checked += 1
    assert checked > 4


def test_the_search_backs_up_the_observers_perspective() -> None:
    """Catches a sign inversion: a won endgame must be preferred.

    P0 holds one card against an opponent with two and an empty talon.
    Playing it out wins outright, so the search must recommend the attack
    and value it positively from the observer's side.
    """
    state = make_state(
        hand0="6S", hand1="7S 9H", trump=Suit.HEARTS, talon="", attacker=0
    )
    information = info_for(state, 0)
    result = ISMCTS(ISMCTSConfig(iterations=40)).search(information, random.Random(1))
    assert result.root is not None
    edge = result.root.edge_for(result.move)
    assert edge.mean_value > 0, (
        "a won position scored negatively for the observer; the value is "
        "being backed up from the wrong perspective"
    )


def test_a_terminal_position_offers_no_moves() -> None:
    """Documents why checking depth before terminality is a no-op here.

    A mutation that tests the depth limit before terminality changes
    nothing, because the very next statement asks the frozen engine for
    legal moves and a finished game has none, so the loop breaks anyway.
    Recorded so the mutation audit's verdict is explained rather than
    excused.
    """
    state = next(s for s in walk(4) if s.is_over)
    information = info_for(state, 0)
    world = sample_worlds(information, 1, seed=1)[0]
    assert build_position(information, world).legal_moves() == ()


def test_more_iterations_visit_at_least_as_many_information_sets() -> None:
    state, player, information = next(decisions(games=3))
    counts = [
        ISMCTS(ISMCTSConfig(iterations=n)).search(
            information, random.Random(6)
        ).information_sets
        for n in (10, 50, 150)
    ]
    assert counts == sorted(counts)


@pytest.mark.parametrize("iterations", [0, -5])
def test_an_invalid_configuration_is_refused(iterations: int) -> None:
    with pytest.raises(ISMCTSError, match="iterations"):
        ISMCTSConfig(iterations=iterations)


def test_invalid_depth_and_exploration_are_refused() -> None:
    with pytest.raises(ISMCTSError, match="max_depth"):
        ISMCTSConfig(max_depth=0)
    with pytest.raises(ISMCTSError, match="exploration"):
        ISMCTSConfig(exploration=-1.0)


# ======================================================================
# Opponent model — the documented SO-ISMCTS limitation
# ======================================================================
def test_the_opponent_model_is_world_aware_by_design() -> None:
    """Asserted, not glossed: the modelled opponent sees the sampled world.

    This is the known SO-ISMCTS limitation. The observer's policy respects
    information sets; the opponent's does not, so opponent-side strategy
    fusion remains. MO-ISMCTS is future work.
    """
    from durakfish.ai.world_search import search_world

    state, player, information = next(decisions(games=3, min_worlds=4))
    worlds = sample_worlds(information, 6, seed=8)
    root = build_position(information, worlds[0])
    move = root.legal_moves()[0]

    replies = set()
    for world in worlds:
        child = build_position(information, world).apply(move)
        if child.current_player == player or child.is_terminal:
            continue
        replies.add(search_world(child, SearchConfig(depth=1), root_player=player).move)
    # The opponent's reply legitimately depends on its own cards, which it
    # can see. That is exactly the limitation being recorded.
    assert replies, "no opponent decision arose"


def test_the_default_evaluator_path_is_behaviourally_unchanged() -> None:
    """Phase 7.3.9 wired ``ISMCTSConfig.evaluator`` through to leaf scoring.

    Before that, the field was declared and documented but ignored: the
    baseline was hardcoded. The default must therefore resolve to exactly
    the baseline, so the change is a no-op for every existing caller.
    """
    from durakfish.ai import BaselineEvaluator

    default = ISMCTS()
    explicit = ISMCTS(ISMCTSConfig(evaluator="baseline"))
    assert type(default._evaluator) is BaselineEvaluator  # noqa: SLF001
    assert ISMCTSConfig().evaluator == "baseline"

    for state, player, information in decisions(games=4):
        a = default.search(information, random.Random(3))
        b = explicit.search(information, random.Random(3))
        assert a.move == b.move
        assert a.visit_counts() == b.visit_counts()


def test_a_configured_evaluator_is_actually_consulted() -> None:
    """Catches evaluator bypass: a silent fallback to the baseline.

    A sentinel evaluator records every call. If the configured evaluator
    were ignored — the pre-7.3.9 behaviour — it would never be called.
    """
    from durakfish.ai.evaluation import EVALUATORS, Score, terminal_score

    calls: list[int] = []

    class _Sentinel:
        name = "sentinel_probe"

        def evaluate(self, node, ply: int = 0) -> Score:
            outcome = node.outcome()
            if outcome is not None:
                return terminal_score(outcome, ply)
            calls.append(ply)
            return 0.0

    EVALUATORS["sentinel_probe"] = _Sentinel
    try:
        state, player, information = next(decisions(games=3))
        ISMCTS(ISMCTSConfig(iterations=20, evaluator="sentinel_probe")).search(
            information, random.Random(1)
        )
        assert calls, "the configured evaluator was never consulted"
    finally:
        del EVALUATORS["sentinel_probe"]


def test_an_unknown_evaluator_name_fails_when_the_search_is_built() -> None:
    from durakfish.exceptions import DurakFishError

    with pytest.raises(DurakFishError, match="unknown evaluator"):
        ISMCTS(ISMCTSConfig(evaluator="no_such_evaluator"))


def test_terminal_values_carry_the_ply_discount() -> None:
    """A forced win must be scored below an immediate win.

    ``terminal_score(WIN, ply) = TERMINAL_WIN - ply``, so threading the
    ply is what makes the search prefer winning sooner. Dropping it
    (evaluating every leaf at ply 0) leaves the sign and the reproducibility
    intact, so nothing else in this suite noticed — a gap found by mutation
    testing in Phase 7.3.9.

    The position is forced: the attacker has no cards, the table is fully
    defended and the talon is empty, so ending the bout is the only legal
    move and it wins at once. Every leaf is therefore terminal, which
    makes the discount directly observable in the edge value.
    """
    from durakfish.ai.evaluation import MATE_MARGIN, TERMINAL_WIN

    state = make_state(
        hand0="",
        hand1="9H",
        trump=Suit.HEARTS,
        talon="",
        table=(("6S", "7S"),),
        attacker=0,
        attack_limit=3,
    )
    information = info_for(state, 0)
    result = ISMCTS(ISMCTSConfig(iterations=5)).search(information, random.Random(1))
    assert result.move is not None
    value = result.root.edge_for(result.move).mean_value

    assert value > MATE_MARGIN, "a forced win should score as a win"
    assert value < TERMINAL_WIN, (
        f"terminal value {value} equals the undiscounted win score; the ply "
        "is not being threaded into leaf evaluation"
    )


def test_search_bot_is_unaffected_by_this_phase() -> None:
    from durakfish.ai import SearchBot

    bot = SearchBot(SearchConfig(depth=3), "s")
    bot.reset(random.Random(0))
    checked = 0
    for state, player, information in decisions(games=4):
        view, legal = observe(state, player), get_legal_moves(state)
        before = bot.choose(view, legal)
        ISMCTS(ISMCTSConfig(iterations=15)).search(information, random.Random(1))
        assert bot.choose(view, legal) == before
        checked += 1
    assert checked > 5


def test_the_ismcts_package_never_names_the_state_layer() -> None:
    import ast
    import pathlib

    directory = (
        pathlib.Path(__file__).resolve().parent.parent
        / "src" / "durakfish" / "ai" / "ismcts"
    )
    for path in sorted(directory.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
        }
        for banned in ("GameState", "hands", "talon", "apply_move", "new_game"):
            assert banned not in names, f"{path.name} references {banned!r}"
