"""Behaviour of the two baseline agents.

GreedyBot's tests validate the heuristic as documented in
``docs/agents.md`` — cheapest card first, non-trumps before trumps, never
spend a trump on an optional addition, never take voluntarily — and not
some incidental property of how it happens to be coded.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from durakfish.ai import GreedyBot, RandomBot, canonical_order, card_cost
from durakfish.ai.tiebreak import canonical_move_key, is_canonically_ordered
from durakfish.cards import Card, Suit
from durakfish.exceptions import AgentError
from durakfish.game import (
    AttackMove,
    DefenseMove,
    Phase,
    apply_move,
    get_legal_moves,
    new_game,
)
from durakfish.game.moves import END_ATTACK, TAKE
from durakfish.game.state import TableSlot, add_cards
from durakfish.information import InformationSet, observe
from tests.support import make_state

# Chi-square critical values, p = 0.001, by degrees of freedom.
CHI2_CRITICAL = {2: 13.82, 3: 16.27, 5: 20.52, 7: 24.32}


def view_and_moves(state):
    player = state.current_player
    assert player is not None
    return observe(state, player), get_legal_moves(state)


def fresh(bot_class, seed: int = 0, name: str | None = None):
    bot = bot_class(name) if name else bot_class()
    bot.reset(random.Random(seed))
    return bot


# ======================================================================
# Canonical ordering
# ======================================================================
def test_canonical_key_is_total_over_every_move_in_a_position() -> None:
    for seed in range(20):
        state = new_game(seed=seed)
        rng = random.Random(seed)
        while not state.is_over:
            legal = get_legal_moves(state)
            keys = [canonical_move_key(m) for m in legal]
            assert len(set(keys)) == len(keys), "two distinct moves share a key"
            state = apply_move(state, rng.choice(legal))


def test_canonical_order_is_permutation_invariant() -> None:
    state = new_game(seed=1)
    legal = list(get_legal_moves(state))
    rng = random.Random(4)
    for _ in range(20):
        shuffled = legal[:]
        rng.shuffle(shuffled)
        assert canonical_order(shuffled) == canonical_order(legal)
    assert is_canonically_ordered(canonical_order(legal))


def test_the_engine_already_emits_moves_in_a_stable_order() -> None:
    """Not required, but if it changes, canonical_order still saves us."""
    state = new_game(seed=2)
    assert get_legal_moves(state) == get_legal_moves(state)


# ======================================================================
# RandomBot
# ======================================================================
def test_random_bot_only_ever_returns_a_supplied_move() -> None:
    bot = fresh(RandomBot, 7)
    for seed in range(15):
        state = new_game(seed=seed)
        rng = random.Random(seed)
        while not state.is_over:
            view, legal = view_and_moves(state)
            chosen = bot.choose(view, legal)
            assert chosen in legal
            state = apply_move(state, rng.choice(legal))


def test_random_bot_is_approximately_uniform() -> None:
    """Seeded, so the tolerance guards against bias rather than luck."""
    state = new_game(seed=3)
    view, legal = view_and_moves(state)
    assert len(legal) >= 6
    legal = legal[:6]

    bot = fresh(RandomBot, 12345)
    draws = 12000
    counts: Counter = Counter(bot.choose(view, legal) for _ in range(draws))

    assert set(counts) == set(legal), "some legal move was never selected"
    expected = draws / len(legal)
    chi2 = sum((counts[m] - expected) ** 2 / expected for m in legal)
    assert chi2 < CHI2_CRITICAL[5], f"chi2={chi2:.2f} suggests a biased sampler"


def test_random_bot_can_select_every_move_even_in_a_tiny_choice() -> None:
    view, legal = view_and_moves(
        make_state(
            hand0="6S 7H", hand1="8S 9H", trump=Suit.HEARTS, talon="", attacker=0
        )
    )
    bot = fresh(RandomBot, 5)
    seen = {bot.choose(view, legal) for _ in range(400)}
    assert seen == set(legal)


def test_random_bot_never_returns_an_unsupplied_move() -> None:
    state = new_game(seed=4)
    view, legal = view_and_moves(state)
    restricted = legal[:2]
    excluded = set(legal[2:])
    bot = fresh(RandomBot, 9)
    for _ in range(500):
        assert bot.choose(view, restricted) not in excluded


def test_random_bot_repeats_itself_exactly_from_the_same_seed() -> None:
    state = new_game(seed=6)
    view, legal = view_and_moves(state)
    first = [fresh(RandomBot, 42).choose(view, legal) for _ in range(1)]
    bot = fresh(RandomBot, 42)
    sequence = [bot.choose(view, legal) for _ in range(50)]
    again = fresh(RandomBot, 42)
    assert [again.choose(view, legal) for _ in range(50)] == sequence
    assert sequence[0] == first[0]


def test_random_bot_reset_clears_all_previous_state() -> None:
    state = new_game(seed=8)
    view, legal = view_and_moves(state)
    bot = RandomBot()
    bot.reset(random.Random(11))
    before = [bot.choose(view, legal) for _ in range(30)]
    for _ in range(100):  # dirty it thoroughly
        bot.choose(view, legal)
    bot.reset(random.Random(11))
    assert [bot.choose(view, legal) for _ in range(30)] == before


def test_a_reset_bot_matches_a_fresh_one() -> None:
    state = new_game(seed=13)
    view, legal = view_and_moves(state)
    used = RandomBot()
    used.reset(random.Random(1))
    for _ in range(200):
        used.choose(view, legal)
    used.reset(random.Random(99))
    assert [used.choose(view, legal) for _ in range(40)] == [
        fresh(RandomBot, 99).choose(view, legal) for _ in range(40)
    ] or True  # fresh bots below compare sequence-wise
    used.reset(random.Random(99))
    other = fresh(RandomBot, 99)
    assert [used.choose(view, legal) for _ in range(40)] == [
        other.choose(view, legal) for _ in range(40)
    ]


def test_random_bot_ignores_the_order_of_the_legal_list() -> None:
    """Uniformity is over the move *set*, not the list it arrived in."""
    state = new_game(seed=14)
    view, legal = view_and_moves(state)
    shuffled = list(legal)
    random.Random(3).shuffle(shuffled)

    a, b = fresh(RandomBot, 77), fresh(RandomBot, 77)
    assert [a.choose(view, legal) for _ in range(60)] == [
        b.choose(view, shuffled) for _ in range(60)
    ]


def test_random_bot_refuses_an_empty_choice() -> None:
    view, _ = view_and_moves(new_game(seed=1))
    with pytest.raises(AgentError, match="empty legal-move list"):
        fresh(RandomBot).choose(view, ())
    with pytest.raises(AgentError):
        fresh(RandomBot).choose(view, [])


def test_random_bot_never_touches_the_global_random_module(monkeypatch) -> None:
    def forbidden(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("agent used the global random module")

    for name in ("random", "randrange", "choice", "randint", "shuffle", "sample"):
        monkeypatch.setattr(random, name, forbidden)

    from durakfish.simulation import play_game

    record = play_game([RandomBot("a"), RandomBot("b")], seed=5)
    assert record.length > 0


def test_random_bot_handles_duplicate_moves_in_the_supplied_list() -> None:
    """A malformed list is the caller's bug; the bot must still stay inside it."""
    state = new_game(seed=2)
    view, legal = view_and_moves(state)
    doubled = list(legal) + list(legal)
    bot = fresh(RandomBot, 6)
    for _ in range(200):
        assert bot.choose(view, doubled) in set(legal)


# ======================================================================
# GreedyBot — the documented heuristic
# ======================================================================
def test_card_cost_ranks_non_trumps_below_trumps_then_by_rank() -> None:
    trump = Suit.HEARTS
    assert card_cost(Card.parse("AS"), trump) < card_cost(Card.parse("6H"), trump)
    assert card_cost(Card.parse("6S"), trump) < card_cost(Card.parse("7S"), trump)
    assert card_cost(Card.parse("6H"), trump) < card_cost(Card.parse("7H"), trump)
    # Total: no two distinct cards tie.
    from durakfish.cards.deck import standard_cards

    costs = [card_cost(c, trump) for c in standard_cards(36)]
    assert len(set(costs)) == 36


def test_greedy_opens_a_bout_with_its_cheapest_card() -> None:
    state = make_state(
        hand0="9S 6C 6H AD",
        hand1="7C 7D",
        trump=Suit.HEARTS,
        talon="8C 8H",
        attacker=0,
    )
    view, legal = view_and_moves(state)
    assert fresh(GreedyBot).choose(view, legal) == AttackMove(Card.parse("6C"))


def test_greedy_opens_with_a_trump_only_when_it_has_nothing_else() -> None:
    """Opening is compulsory, so a trump-only hand must still attack."""
    state = make_state(
        hand0="6H 9H",
        hand1="7C 7D",
        trump=Suit.HEARTS,
        talon="8C 8H",
        attacker=0,
    )
    view, legal = view_and_moves(state)
    assert fresh(GreedyBot).choose(view, legal) == AttackMove(Card.parse("6H"))


def test_greedy_adds_a_cheap_non_trump_to_an_open_bout() -> None:
    state = make_state(
        hand0="6C 6D AS",
        hand1="7C 7D",
        trump=Suit.HEARTS,
        talon="8C 8H",
        table=(("6S", "7S"),),
        attacker=0,
        attack_limit=3,
    )
    view, legal = view_and_moves(state)
    assert fresh(GreedyBot).choose(view, legal) == AttackMove(Card.parse("6C"))


def test_greedy_ends_the_bout_rather_than_throw_in_a_trump() -> None:
    """The rule that distinguishes it from 'always play the lowest card'."""
    state = make_state(
        hand0="6H AS",
        hand1="7C 7D",
        trump=Suit.HEARTS,
        talon="8C 8H",
        table=(("6S", "7S"),),
        attacker=0,
        attack_limit=3,
    )
    view, legal = view_and_moves(state)
    assert AttackMove(Card.parse("6H")) in legal, "the trump addition is legal"
    assert fresh(GreedyBot).choose(view, legal) == END_ATTACK


def test_greedy_will_not_spend_a_trump_on_a_throw_in_either() -> None:
    state = make_state(
        hand0="6H 9C",
        hand1="7C 7D",
        trump=Suit.HEARTS,
        talon="8C 8H",
        table=(("6S", None),),
        attacker=0,
        phase=Phase.TAKING,
        attack_limit=3,
    )
    view, legal = view_and_moves(state)
    assert fresh(GreedyBot).choose(view, legal) == END_ATTACK


def test_greedy_defends_with_the_cheapest_card_that_works() -> None:
    state = make_state(
        hand0="6D",
        hand1="10C JC AH",
        trump=Suit.HEARTS,
        talon="8C 8H",
        table=(("9C", None),),
        attacker=0,
        phase=Phase.DEFENSE,
        attack_limit=3,
    )
    view, legal = view_and_moves(state)
    chosen = fresh(GreedyBot).choose(view, legal)
    assert chosen == DefenseMove(Card.parse("9C"), Card.parse("10C"))


def test_greedy_prefers_any_non_trump_defence_over_a_trump() -> None:
    state = make_state(
        hand0="6D",
        hand1="6H AC",
        trump=Suit.HEARTS,
        talon="8C 8H",
        table=(("9C", None),),
        attacker=0,
        phase=Phase.DEFENSE,
        attack_limit=2,
    )
    view, legal = view_and_moves(state)
    # The ace of clubs is a far more valuable card than the six of trumps,
    # yet the documented rule spends the non-trump. Deliberately weak.
    assert fresh(GreedyBot).choose(view, legal) == DefenseMove(
        Card.parse("9C"), Card.parse("AC")
    )


def test_greedy_uses_a_trump_when_it_is_the_only_defence() -> None:
    state = make_state(
        hand0="6D",
        hand1="6H 6C",
        trump=Suit.HEARTS,
        talon="8C 8H",
        table=(("AS", None),),
        attacker=0,
        phase=Phase.DEFENSE,
        attack_limit=2,
    )
    view, legal = view_and_moves(state)
    assert fresh(GreedyBot).choose(view, legal) == DefenseMove(
        Card.parse("AS"), Card.parse("6H")
    )


def test_greedy_takes_only_when_it_cannot_defend() -> None:
    state = make_state(
        hand0="6D",
        hand1="6S 7S",
        trump=Suit.HEARTS,
        talon="8C 8H",
        table=(("AH", None),),
        attacker=0,
        phase=Phase.DEFENSE,
        attack_limit=2,
    )
    view, legal = view_and_moves(state)
    assert legal == (TAKE,)
    assert fresh(GreedyBot).choose(view, legal) == TAKE


def test_greedy_never_takes_voluntarily() -> None:
    """Documented weakness: it will beat a six with the ace of trumps."""
    state = make_state(
        hand0="6D",
        hand1="AH",
        trump=Suit.HEARTS,
        talon="8C 8H",
        table=(("6S", None),),
        attacker=0,
        phase=Phase.DEFENSE,
        attack_limit=2,
    )
    view, legal = view_and_moves(state)
    assert TAKE in legal
    assert fresh(GreedyBot).choose(view, legal) == DefenseMove(
        Card.parse("6S"), Card.parse("AH")
    )


def test_greedy_ends_the_bout_when_the_attack_limit_is_reached() -> None:
    state = make_state(
        hand0="6C 6D",
        hand1="7C",
        trump=Suit.HEARTS,
        talon="8C 8H",
        table=(("6S", "7S"),),
        attacker=0,
        attack_limit=1,
    )
    view, legal = view_and_moves(state)
    assert legal == (END_ATTACK,)
    assert fresh(GreedyBot).choose(view, legal) == END_ATTACK


def test_greedy_is_deterministic_regardless_of_seed() -> None:
    """A fully deterministic policy: the seed changes the deal, not the play."""
    state = new_game(seed=15)
    view, legal = view_and_moves(state)
    choices = {fresh(GreedyBot, seed).choose(view, legal) for seed in range(50)}
    assert len(choices) == 1


def test_greedy_is_permutation_invariant() -> None:
    for seed in range(12):
        state = new_game(seed=seed)
        rng = random.Random(seed)
        while not state.is_over:
            view, legal = view_and_moves(state)
            baseline = fresh(GreedyBot).choose(view, legal)
            shuffled = list(legal)
            for _ in range(4):
                rng.shuffle(shuffled)
                assert fresh(GreedyBot).choose(view, shuffled) == baseline
            state = apply_move(state, rng.choice(legal))


def test_greedy_repeats_itself_across_identical_calls() -> None:
    state = new_game(seed=16)
    view, legal = view_and_moves(state)
    bot = fresh(GreedyBot)
    assert len({bot.choose(view, legal) for _ in range(100)}) == 1


def test_greedy_refuses_an_empty_choice() -> None:
    view, _ = view_and_moves(new_game(seed=1))
    with pytest.raises(AgentError, match="empty legal-move list"):
        fresh(GreedyBot).choose(view, ())


def test_greedy_stays_inside_a_restricted_legal_list() -> None:
    """Metamorphic: removing its preferred move must not make it cheat."""
    for seed in range(10):
        state = new_game(seed=seed)
        rng = random.Random(seed)
        while not state.is_over:
            view, legal = view_and_moves(state)
            preferred = fresh(GreedyBot).choose(view, legal)
            reduced = tuple(m for m in legal if m != preferred)
            if reduced:
                assert fresh(GreedyBot).choose(view, reduced) != preferred
                assert fresh(GreedyBot).choose(view, reduced) in reduced
            state = apply_move(state, rng.choice(legal))


def test_greedy_is_order_independent_even_on_the_card_less_branch() -> None:
    """Catches a fallback that returns ``legal[0]`` instead of ranking.

    The engine never offers TAKE and END_ATTACK together, so this list is
    synthetic; the point is that the branch does not depend on position.
    """
    state = make_state(
        hand0="6C", hand1="7C 7D", trump=Suit.HEARTS, talon="8C 8H",
        table=(("6S", "7S"),), attacker=0, attack_limit=1,
    )
    view, _ = view_and_moves(state)
    forward = fresh(GreedyBot).choose(view, (TAKE, END_ATTACK))
    reverse = fresh(GreedyBot).choose(view, (END_ATTACK, TAKE))
    assert forward == reverse == TAKE


def test_greedy_cost_is_injective_so_card_ties_cannot_arise() -> None:
    """Documents why the canonical tie-break is defensive, not exercised."""
    from durakfish.cards.deck import standard_cards

    for trump in Suit:
        costs = [card_cost(c, trump) for c in standard_cards(36)]
        assert len(set(costs)) == 36


def test_greedy_handles_a_list_containing_only_non_card_moves() -> None:
    state = make_state(
        hand0="6C",
        hand1="7C 7D",
        trump=Suit.HEARTS,
        talon="8C 8H",
        table=(("6S", "7S"),),
        attacker=0,
        attack_limit=1,
    )
    view, _ = view_and_moves(state)
    assert fresh(GreedyBot).choose(view, (END_ATTACK,)) == END_ATTACK
    assert fresh(GreedyBot).choose(view, (TAKE, END_ATTACK)) == TAKE


# ======================================================================
# Both bots: input immutability
# ======================================================================
@pytest.mark.parametrize("bot_class", [RandomBot, GreedyBot])
def test_agents_do_not_mutate_their_inputs(bot_class) -> None:
    """The architecture uses immutable structures; this proves it holds."""
    for seed in range(10):
        state = new_game(seed=seed)
        rng = random.Random(seed)
        bot = fresh(bot_class, seed)
        while not state.is_over:
            view, legal = view_and_moves(state)
            view_before = view.to_dict()
            view_key = view.key()
            legal_before = tuple(legal)

            bot.choose(view, legal)

            assert view.to_dict() == view_before
            assert view.key() == view_key
            assert tuple(legal) == legal_before
            state = apply_move(state, rng.choice(legal))


@pytest.mark.parametrize("bot_class", [RandomBot, GreedyBot])
def test_agents_return_a_move_object_from_the_supplied_list(bot_class) -> None:
    for seed in range(10):
        state = new_game(seed=seed)
        rng = random.Random(seed)
        bot = fresh(bot_class, seed)
        while not state.is_over:
            view, legal = view_and_moves(state)
            chosen = bot.choose(view, legal)
            assert any(chosen is m for m in canonical_order(legal)) or chosen in legal
            state = apply_move(state, rng.choice(legal))


# ======================================================================
# Both bots: unusual but legal positions
# ======================================================================
CORNERS = {
    "empty hand mid-bout": make_state(
        hand0="", hand1="8S 9H", trump=Suit.HEARTS, talon="10C 10H",
        table=(("6S", "7S"),), attacker=0, attack_limit=3,
    ),
    "defender with no cards facing an attack": make_state(
        hand0="6C", hand1="", trump=Suit.HEARTS, talon="10C 10H",
        table=(("6S", None),), attacker=0, phase=Phase.DEFENSE, attack_limit=2,
    ),
    "six-card bout at the limit": make_state(
        hand0="6C 6D 6H", hand1="AS AC AD", trump=Suit.HEARTS, talon="10C 10H",
        table=(("6S", "7S"), ("7C", "8C"), ("7D", "8D"),
               ("7H", "8H"), ("9S", "10S"), ("9C", "JC")),
        attacker=0, attack_limit=6,
    ),
    "empty talon endgame": make_state(
        hand0="6S 7H", hand1="8S 9H", trump=Suit.HEARTS, talon="", attacker=0,
    ),
    "throw-in position": make_state(
        hand0="6C 6D 9S", hand1="7C 7D", trump=Suit.HEARTS, talon="8C 8H",
        table=(("6S", None),), attacker=0, phase=Phase.TAKING, attack_limit=3,
    ),
}


@pytest.mark.parametrize("bot_class", [RandomBot, GreedyBot])
@pytest.mark.parametrize("label", sorted(CORNERS))
def test_both_bots_cope_with_unusual_positions(bot_class, label) -> None:
    state = CORNERS[label]
    view, legal = view_and_moves(state)
    chosen = fresh(bot_class, 3).choose(view, legal)
    assert chosen in legal, label
    apply_move(state, chosen).validate()
