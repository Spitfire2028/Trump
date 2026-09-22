"""The Podkidnoy Durak rules engine.

This module answers *"is this move legal?"* and *"what does the position
become?"*. It contains **no AI logic whatsoever** — no heuristics, no
preferences, no move ordering. It never answers "what move should I
choose?".

Two independent legality implementations live here on purpose:

* :func:`is_legal_move` checks a single move with direct predicates;
* :func:`get_legal_moves` enumerates every legal move from scratch.

Neither is defined in terms of the other, so the property tests that
assert they always agree are a real cross-check rather than a tautology.
Move generation never emits a move that ``apply_move`` would reject.

The exact rules implemented, and the variant ambiguities resolved along
the way, are documented in ``docs/rules.md``.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from durakfish.cards import Card, Deck, Rank, Suit
from durakfish.exceptions import IllegalMoveError, UnsupportedRuleError
from durakfish.game.history import DrawRecord, EventKind, GameEvent, push_event
from durakfish.game.moves import (
    END_ATTACK,
    TAKE,
    AttackMove,
    DefenseMove,
    EndAttack,
    Move,
    TakeCards,
)
from durakfish.game.ruleset import STANDARD_RULES, FirstAttackerPolicy, RuleSet
from durakfish.game.state import (
    GameState,
    Phase,
    TableSlot,
    add_cards,
    remove_cards,
)

__all__ = [
    "apply_move",
    "beats",
    "choose_first_attacker",
    "get_legal_moves",
    "is_legal_move",
    "legal_attack_cards",
    "legal_defense_cards",
    "new_game",
]


# ----------------------------------------------------------------------
# Card comparison
# ----------------------------------------------------------------------
def beats(defending_card: Card, attacking_card: Card, trump: Suit) -> bool:
    """True if ``defending_card`` beats ``attacking_card`` under ``trump``.

    The complete rule, in three lines:

    * same suit — the higher rank wins (this covers trump vs trump);
    * different suits — only a trump beats a non-trump;
    * a non-trump never beats a trump, whatever its rank.

    A card never beats itself.

    This lives here, not on :class:`~durakfish.cards.Card`, because
    "higher" is meaningless without knowing the trump, and the trump is
    game state.
    """
    if defending_card.suit is attacking_card.suit:
        return defending_card.rank > attacking_card.rank
    return defending_card.suit is trump


# ----------------------------------------------------------------------
# Legal move generation
# ----------------------------------------------------------------------
def _can_add_attack(state: GameState) -> bool:
    """Whether the attacker is allowed to put another card down at all."""
    return state.attacks_made < state.attack_limit


def legal_attack_cards(state: GameState) -> tuple[Card, ...]:
    """Cards the attacker may legally place, in canonical hand order.

    Empty table: any card. Otherwise: only ranks already on the table,
    and only while the bout's attack limit has room.
    """
    if state.phase not in (Phase.ATTACK, Phase.TAKING):
        return ()
    hand = state.hands[state.attacker]
    if not state.table:
        # The opening card of a bout is unrestricted.
        return hand if state.attack_limit > 0 else ()
    if not _can_add_attack(state):
        return ()
    allowed: frozenset[Rank] = state.table_ranks
    return tuple(card for card in hand if card.rank in allowed)


def legal_defense_cards(state: GameState, attacking_card: Card) -> tuple[Card, ...]:
    """Cards in the defender's hand that beat ``attacking_card``."""
    return tuple(
        card
        for card in state.hands[state.defender]
        if beats(card, attacking_card, state.trump_suit)
    )


def get_legal_moves(state: GameState, player: int | None = None) -> tuple[Move, ...]:
    """Every legal move for ``player`` (default: whoever is to act).

    Returns an empty tuple for a player who is not to move and for a
    finished game. Order is deterministic, driven by canonical hand
    order — the search layer decides how to order moves for efficiency,
    not this function.
    """
    if state.is_over:
        return ()
    if player is not None and player != state.current_player:
        return ()

    if state.phase is Phase.DEFENSE:
        open_slot = state.table[-1]
        moves: list[Move] = [
            DefenseMove(open_slot.attacking_card, card)
            for card in legal_defense_cards(state, open_slot.attacking_card)
        ]
        moves.append(TAKE)
        return tuple(moves)

    # ATTACK or TAKING: the attacker may add a card, or end the bout.
    attack_moves: list[Move] = [AttackMove(card) for card in legal_attack_cards(state)]
    if state.table:
        # Ending is only a choice once something is on the table; a bout
        # cannot begin with a pass.
        attack_moves.append(END_ATTACK)
    return tuple(attack_moves)


def is_legal_move(
    state: GameState, move: Move, player: int | None = None
) -> bool:
    """Whether ``move`` is legal right now.

    Written independently of :func:`get_legal_moves` so that the two can
    be cross-checked rather than agreeing by construction.
    """
    if state.is_over:
        return False
    actor = state.current_player
    if player is not None and player != actor:
        return False

    if isinstance(move, AttackMove):
        if state.phase not in (Phase.ATTACK, Phase.TAKING):
            return False
        if move.card not in state.hands[state.attacker]:
            return False
        if not state.table:
            return state.attack_limit > 0
        if not _can_add_attack(state):
            return False
        return move.card.rank in state.table_ranks

    if isinstance(move, DefenseMove):
        if state.phase is not Phase.DEFENSE:
            return False
        if move.defending_card not in state.hands[state.defender]:
            return False
        open_cards = {
            slot.attacking_card for slot in state.table if not slot.is_defended
        }
        if move.attacking_card not in open_cards:
            return False
        return beats(move.defending_card, move.attacking_card, state.trump_suit)

    if isinstance(move, TakeCards):
        return state.phase is Phase.DEFENSE

    if isinstance(move, EndAttack):
        if state.phase not in (Phase.ATTACK, Phase.TAKING):
            return False
        return bool(state.table)

    return False


# ----------------------------------------------------------------------
# State transitions
# ----------------------------------------------------------------------
def apply_move(state: GameState, move: Move) -> GameState:
    """Apply ``move`` and return the resulting state.

    The input state is never modified. Bout resolution — discarding or
    handing over the table, refilling hands and switching roles — happens
    atomically inside the ``EndAttack`` transition, so the returned state
    is always one where somebody has a real decision, or the game is over.

    Raises:
        IllegalMoveError: if the move is not legal in this state.
    """
    if not is_legal_move(state, move):
        raise IllegalMoveError(
            f"illegal move {move} in state {state!r}\n{state.describe()}"
        )

    if isinstance(move, AttackMove):
        return _apply_attack(state, move)
    if isinstance(move, DefenseMove):
        return _apply_defense(state, move)
    if isinstance(move, TakeCards):
        return _apply_take(state, move)
    if isinstance(move, EndAttack):
        return _apply_end_attack(state, move)
    raise IllegalMoveError(f"unknown move type: {move!r}")  # pragma: no cover


def _apply_attack(state: GameState, move: AttackMove) -> GameState:
    hands = list(state.hands)
    hands[state.attacker] = remove_cards(hands[state.attacker], (move.card,))
    event = GameEvent(
        kind=EventKind.MOVE,
        player=state.attacker,
        move=move,
        revealed=(move.card,),
    )
    # A throw-in during a take stays in TAKING; otherwise the defender
    # now owes a response.
    next_phase = Phase.TAKING if state.phase is Phase.TAKING else Phase.DEFENSE
    return state.replace(
        hands=tuple(hands),
        table=state.table + (TableSlot(move.card),),
        phase=next_phase,
        history=push_event(state.history, event),
    )


def _apply_defense(state: GameState, move: DefenseMove) -> GameState:
    hands = list(state.hands)
    hands[state.defender] = remove_cards(hands[state.defender], (move.defending_card,))
    table = list(state.table)
    for index, slot in enumerate(table):
        if not slot.is_defended and slot.attacking_card == move.attacking_card:
            table[index] = TableSlot(slot.attacking_card, move.defending_card)
            break
    event = GameEvent(
        kind=EventKind.MOVE,
        player=state.defender,
        move=move,
        revealed=(move.defending_card,),
    )
    return state.replace(
        hands=tuple(hands),
        table=tuple(table),
        phase=Phase.ATTACK,
        history=push_event(state.history, event),
    )


def _apply_take(state: GameState, move: TakeCards) -> GameState:
    """The defender announces a take. The bout does *not* end here.

    Standard Podkidnoy lets the attacker keep throwing in matching ranks,
    which the defender must also pick up. Cards move only when the
    attacker ends the bout.
    """
    event = GameEvent(kind=EventKind.MOVE, player=state.defender, move=move)
    return state.replace(
        phase=Phase.TAKING,
        history=push_event(state.history, event),
    )


def _refill(
    state: GameState, talon: tuple[Card, ...], hands: list[tuple[Card, ...]]
) -> tuple[tuple[Card, ...], tuple[DrawRecord, ...]]:
    """Refill hands toward ``hand_size``, attacker first, defender last.

    Draw order is a real rule, not bookkeeping: when the talon holds
    fewer cards than the players need, whoever draws first may take the
    last card and thereby avoid being left holding cards at the end.
    """
    order = (state.attacker, state.defender)
    draws: list[DrawRecord] = []
    cursor = 0
    for player in order:
        needed = state.rules.hand_size - len(hands[player])
        if needed <= 0:
            continue
        drawn = talon[cursor : cursor + needed]
        if not drawn:
            continue
        cursor += len(drawn)
        hands[player] = add_cards(hands[player], drawn)
        draws.append(DrawRecord(player, tuple(drawn)))
    return talon[cursor:], tuple(draws)


def _apply_end_attack(state: GameState, move: EndAttack) -> GameState:
    """Resolve the bout: dispose of the table, refill hands, switch roles."""
    hands = list(state.hands)
    table_cards = state.table_cards
    took = state.phase is Phase.TAKING
    taken: tuple[Card, ...]
    discarded: tuple[Card, ...]

    if took:
        # Everything on the table goes to the defender.
        hands[state.defender] = add_cards(hands[state.defender], table_cards)
        discard = state.discard
        # A defender who picked up does not attack; the attacker keeps
        # the initiative and the defender effectively loses a turn.
        next_attacker = state.attacker
        taken, discarded = table_cards, ()
    else:
        # Bito: a complete defence retires the cards from play, and the
        # defender takes over as attacker.
        discard = state.discard | frozenset(table_cards)
        next_attacker = state.defender
        taken, discarded = (), table_cards

    talon, draws = _refill(state, state.talon, hands)

    # Two events, not one: the decision and its consequences are recorded
    # separately so that MOVE events are exactly the sequence of player
    # decisions. Replay and future training data both depend on being able
    # to filter for decisions without knowing which of them end bouts.
    history = push_event(
        state.history,
        GameEvent(kind=EventKind.MOVE, player=state.attacker, move=move),
    )
    history = push_event(
        history,
        GameEvent(
            kind=EventKind.BOUT_END,
            player=state.attacker,
            discarded=discarded,
            taken=taken,
            taken_by=state.defender if took else None,
            draws=draws,
        ),
    )

    resolved = state.replace(
        hands=tuple(hands),
        talon=talon,
        discard=discard,
        table=(),
        attacker=next_attacker,
        phase=Phase.ATTACK,
        history=history,
    )
    return _finish_or_start_bout(resolved)


def _finish_or_start_bout(state: GameState) -> GameState:
    """Detect game end, or set up the limit for the next bout.

    A player is only out once the talon cannot refill them. Holding zero
    cards while the talon still has cards is an ordinary intermediate
    position, not a win.
    """
    if not state.talon:
        empty = [p for p, hand in enumerate(state.hands) if not hand]
        if empty:
            durak: int | None
            if len(empty) == state.num_players:
                durak = None  # everyone went out together
            else:
                remaining = [
                    p for p in range(state.num_players) if p not in empty
                ]
                durak = remaining[0]
            event = GameEvent(kind=EventKind.GAME_OVER, durak=durak)
            return state.replace(
                phase=Phase.GAME_OVER,
                attack_limit=0,
                durak=durak,
                history=push_event(state.history, event),
            )

    limit = min(
        state.rules.max_attacks_per_bout, len(state.hands[state.defender])
    )
    return state.replace(phase=Phase.ATTACK, attack_limit=limit)


# ----------------------------------------------------------------------
# Setting up a game
# ----------------------------------------------------------------------
def choose_first_attacker(
    hands: Iterable[tuple[Card, ...]],
    trump: Suit,
    rules: RuleSet = STANDARD_RULES,
) -> int:
    """Decide who opens.

    Standard rule: the holder of the lowest trump attacks first. If
    neither player was dealt a trump — about one deal in 170 with 36
    cards — the traditional remedy is a redeal, which would break
    reproducibility from a seed. This engine instead falls back to the
    lowest card overall by rank, then by canonical suit order. The
    fallback is deterministic and unbiased; it is documented as a
    deliberate substitute in ``docs/rules.md``.
    """
    if rules.first_attacker_policy is FirstAttackerPolicy.FIXED:
        return 0

    best_player = 0
    best_key: tuple[int, int, int] | None = None
    for player, hand in enumerate(hands):
        trumps = [card for card in hand if card.suit is trump]
        pool = trumps or list(hand)
        if not pool:
            continue
        lowest = min(pool, key=lambda c: (int(c.rank), int(c.suit)))
        key = (0 if trumps else 1, int(lowest.rank), int(lowest.suit))
        if best_key is None or key < best_key:
            best_key, best_player = key, player
    return best_player


def new_game(
    *,
    seed: int | None = None,
    deck: Deck | None = None,
    rules: RuleSet = STANDARD_RULES,
    first_attacker: int | None = None,
) -> GameState:
    """Deal a fresh game.

    Cards are dealt in blocks — ``hand_size`` to player 0, then to player
    1 — rather than alternating one at a time. From a shuffled deck the
    two are distributionally identical, and the block deal is simpler to
    reason about when reproducing a game from a seed.

    The face-up trump card is the bottom card of the talon and stays in
    the talon until drawn; it is public knowledge but is never counted as
    a location of its own.

    Args:
        seed: seeds the shuffle when no deck is supplied.
        deck: an exact deck to deal from, for reproducing positions.
        rules: the variant.
        first_attacker: override the opening attacker, for balanced
            benchmarking where the natural rule would bias results.
    """
    if deck is None:
        deck = Deck.shuffled(seed=seed, size=rules.deck_size)
    else:
        deck = deck.copy()
    if len(deck) != rules.deck_size:
        raise UnsupportedRuleError(
            f"deck has {len(deck)} cards, rules expect {rules.deck_size}"
        )

    hands: list[tuple[Card, ...]] = []
    draws: list[DrawRecord] = []
    for player in range(rules.num_players):
        dealt = tuple(deck.draw_up_to(rules.hand_size))
        hands.append(add_cards((), dealt))
        draws.append(DrawRecord(player, dealt))

    trump_card = deck.bottom_card
    if trump_card is None:  # pragma: no cover - impossible with valid sizes
        raise UnsupportedRuleError("talon is empty after dealing; deck too small")

    attacker = (
        choose_first_attacker(hands, trump_card.suit, rules)
        if first_attacker is None
        else first_attacker
    )
    defender = (attacker + 1) % rules.num_players

    deal_event = GameEvent(
        kind=EventKind.DEAL,
        revealed=(trump_card,),
        draws=tuple(draws),
    )
    state = GameState(
        hands=tuple(hands),
        talon=deck.remaining(),
        discard=frozenset(),
        table=(),
        trump_card=trump_card,
        trump_suit=trump_card.suit,
        attacker=attacker,
        phase=Phase.ATTACK,
        attack_limit=min(rules.max_attacks_per_bout, len(hands[defender])),
        history=push_event(None, deal_event),
        rules=rules,
    )
    state.validate()
    return state


def random_playout(
    state: GameState, rng: random.Random, *, max_moves: int = 5000
) -> GameState:
    """Play uniformly random legal moves until the game ends.

    Lives here rather than in ``ai/`` because it makes no decisions worth
    the name — it is a rules-exercising utility used by fuzz tests and,
    later, as the crudest possible rollout policy baseline.

    Raises:
        RuntimeError: if ``max_moves`` is exceeded, which would indicate
            a non-terminating rules bug rather than a long game.
    """
    for _ in range(max_moves):
        if state.is_over:
            return state
        moves = get_legal_moves(state)
        if not moves:  # pragma: no cover - would be a rules bug
            raise RuntimeError(
            f"no legal moves in a live position:\n{state.describe()}"
        )
        state = apply_move(state, rng.choice(moves))
    raise RuntimeError(f"game did not terminate within {max_moves} moves")
