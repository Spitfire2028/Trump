"""The information boundary.

:class:`InformationSet` is exactly what one player is allowed to know.
:func:`observe` projects a ground-truth :class:`~durakfish.game.GameState`
onto that view, discarding everything hidden. Agents receive an
information set and never a game state, so cheating is not a matter of
self-discipline — the hidden data is simply not reachable from what the
agent holds.

Redaction, not deduction
------------------------
This module **observes**. It does not infer, deduce, count, or estimate.
The distinction is deliberate and load-bearing:

* An observer sees the faces of the cards in their own hand, the cards on
  the table, the discard pile, and the face-up trump card. Those are
  observations, and they are what this module reports.
* An observer can also *work out* things — that the opponent still holds
  the queen they picked up three bouts ago, or that with an empty talon
  the opponent's hand is the complement of everything else. Those are
  deductions. They belong to ``CardTracker`` in Phase 6, which consumes
  this projection.

So :meth:`InformationSet.unobserved_cards` really does mean "not yet
seen", and deliberately still lists cards the observer could deduce the
location of. Shipping an obviously incomplete projection is better than
shipping one that quietly does half a tracker's job.

Information assumptions
-----------------------
The discard pile is **public**: an observer may read the exact cards in
it. This is a stated DurakFish assumption, documented in ``docs/rules.md``
— live Podkidnoy leaves the pile face down and some house rules forbid
reviewing it, but card counting is a real skill of the game and modelling
the pile as observable is what lets an engine exercise it.

The talon's *contents and order* are hidden; only its size is public. The
face-up trump card is public at all times, because it is dealt face up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from durakfish.cards import Card, Rank, Suit, format_cards
from durakfish.cards.deck import standard_cards
from durakfish.game.history import EventKind, GameEvent
from durakfish.game.moves import Move
from durakfish.game.ruleset import RuleSet
from durakfish.game.state import GameState, Phase, TableSlot

__all__ = [
    "InformationSet",
    "PublicDraw",
    "PublicEvent",
    "observe",
    "redact_event",
]


@dataclass(frozen=True, slots=True)
class PublicDraw:
    """What an observer learns about one player's draw from the talon.

    The *number* of cards drawn is public — everyone watches the cards
    leave the talon. Their *faces* are private to the drawer, so
    :attr:`cards` is populated only when the drawer is the observer.
    """

    player: int
    count: int
    cards: tuple[Card, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"player": self.player, "count": self.count}
        if self.cards is not None:
            data["cards"] = [c.short for c in self.cards]
        return data


@dataclass(frozen=True, slots=True)
class PublicEvent:
    """A history event as one observer saw it.

    Every field here is something the observer genuinely witnessed:
    played cards are laid face up, a take moves cards that were already
    face up on the table, and *bito* retires face-up cards to a pile this
    engine treats as readable. Only talon draws are redacted.
    """

    kind: EventKind
    player: int | None = None
    move: Move | None = None
    revealed: tuple[Card, ...] = ()
    discarded: tuple[Card, ...] = ()
    taken: tuple[Card, ...] = ()
    taken_by: int | None = None
    draws: tuple[PublicDraw, ...] = ()
    durak: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind.value}
        if self.player is not None:
            data["player"] = self.player
        if self.move is not None:
            data["move"] = self.move.to_dict()
        for name, cards in (
            ("revealed", self.revealed),
            ("discarded", self.discarded),
            ("taken", self.taken),
        ):
            if cards:
                data[name] = [c.short for c in cards]
        if self.taken_by is not None:
            data["taken_by"] = self.taken_by
        if self.draws:
            data["draws"] = [d.to_dict() for d in self.draws]
        if self.durak is not None:
            data["durak"] = self.durak
        return data


def redact_event(event: GameEvent, observer: int) -> PublicEvent:
    """Project one ground-truth event onto what ``observer`` witnessed.

    Redaction is observer-relative, not globally public: the observer's
    own draws keep their card faces, everybody else's become counts.
    """
    draws = tuple(
        PublicDraw(
            player=record.player,
            count=len(record.cards),
            cards=record.cards if record.player == observer else None,
        )
        for record in event.draws
    )
    return PublicEvent(
        kind=event.kind,
        player=event.player,
        move=event.move,
        revealed=event.revealed,
        discarded=event.discarded,
        taken=event.taken,
        taken_by=event.taken_by,
        draws=draws,
        durak=event.durak,
    )


@dataclass(frozen=True, slots=True)
class InformationSet:
    """Everything one player may legitimately know, and nothing more.

    Deliberately absent, and not obtainable from anything here: other
    players' hands, the talon's contents, and the talon's order.
    """

    player: int
    hand: tuple[Card, ...]
    table: tuple[TableSlot, ...]
    trump_suit: Suit
    trump_card: Card
    talon_size: int
    hand_sizes: tuple[int, ...]
    discard: frozenset[Card]
    attacker: int
    phase: Phase
    attack_limit: int
    public_history: tuple[PublicEvent, ...]
    observed: frozenset[Card]
    rules: RuleSet
    durak: int | None = None

    # ------------------------------------------------------------------
    # Roles, mirroring GameState's derivations so agents read the same API
    # ------------------------------------------------------------------
    @property
    def num_players(self) -> int:
        return len(self.hand_sizes)

    @property
    def defender(self) -> int:
        return (self.attacker + 1) % self.num_players

    @property
    def is_attacker(self) -> bool:
        return self.player == self.attacker

    @property
    def current_player(self) -> int | None:
        if self.phase is Phase.GAME_OVER:
            return None
        if self.phase is Phase.DEFENSE:
            return self.defender
        return self.attacker

    @property
    def is_to_move(self) -> bool:
        return self.current_player == self.player

    @property
    def opponent(self) -> int:
        """The other player. Two-player only, which is the supported variant."""
        return (self.player + 1) % self.num_players

    @property
    def opponent_hand_size(self) -> int:
        return self.hand_sizes[self.opponent]

    @property
    def is_over(self) -> bool:
        return self.phase is Phase.GAME_OVER

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    @property
    def table_cards(self) -> tuple[Card, ...]:
        return tuple(card for slot in self.table for card in slot.cards)

    @property
    def table_ranks(self) -> frozenset[Rank]:
        return frozenset(card.rank for card in self.table_cards)

    @property
    def undefended(self) -> tuple[TableSlot, ...]:
        return tuple(slot for slot in self.table if not slot.is_defended)

    @property
    def attacks_made(self) -> int:
        return len(self.table)

    @property
    def additions_remaining(self) -> int:
        """How many more cards may be added to the attack this bout."""
        return max(0, self.attack_limit - len(self.table))

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def observed_cards(self) -> frozenset[Card]:
        """Every card whose face this observer has ever seen.

        Cumulative, not point-in-time. A card that sat face up on the
        table and was then picked up by the opponent stays observed: the
        observer watched it. Remembering what you have seen is not
        deduction — it is the raw material Phase 6 deduces *from*.

        Comprises: own hand, the current table, the discard pile, the
        face-up trump card, every card ever played or retired, and the
        observer's own draws from the talon.
        """
        return self.observed

    def unobserved_cards(self) -> frozenset[Card]:
        """Cards this observer has **never seen the face of**.

        Read the name literally. This is *not* the set of cards that could
        be in an opponent's hand, and it is not a probability support. No
        deduction of any kind is performed here:

        * it makes no claim about *where* an unobserved card is — only
          that it has not been seen;
        * it is strictly smaller than the opponent's possible holdings,
          because the opponent may also hold cards this observer watched
          them take from the table;
        * an empty talon does not cause the opponent's hand to be
          resolved, though it makes it deducible.

        Combining observations with counting to say where cards actually
        are is ``CardTracker``'s job in Phase 6. Using this set as a
        candidate opponent hand would be a bug.
        """
        universe = frozenset(standard_cards(self.rules.deck_size))
        return universe - self.observed

    @property
    def hidden_card_count(self) -> int:
        """How many cards are physically out of sight: talon plus other hands."""
        return self.talon_size + sum(
            size for player, size in enumerate(self.hand_sizes)
            if player != self.player
        )

    def is_open_information(self) -> bool:
        """True when no hidden information remains in a two-player game.

        A public fact about the position, not a deduction from it: with an
        empty talon and two players, nothing is concealed that could not be
        worked out. Performing that deduction is Phase 6's job; reporting
        that it is possible is fair game here, and is what the endgame
        solver will switch on.
        """
        return self.talon_size == 0 and self.num_players == 2

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    def key(self) -> tuple[Any, ...]:
        """Canonical hashable key for this information set.

        Built from integers so it is stable across processes. History is
        excluded, mirroring ``GameState.transposition_key``; whether
        ISMCTS should key on history as well is a Phase 11 decision and is
        deliberately not pre-empted here.
        """
        return (
            self.player,
            int(self.phase),
            self.attacker,
            self.attack_limit,
            self.talon_size,
            tuple(self.hand_sizes),
            tuple(c.code for c in self.hand),
            tuple(
                (
                    slot.attacking_card.code,
                    -1 if slot.defending_card is None else slot.defending_card.code,
                )
                for slot in self.table
            ),
            tuple(sorted(c.code for c in self.discard)),
            tuple(sorted(c.code for c in self.observed)),
            int(self.trump_suit),
            self.trump_card.code,
            -1 if self.durak is None else self.durak,
        )

    def to_dict(self, *, include_history: bool = True) -> dict[str, Any]:
        """JSON-friendly encoding, useful for debugging and future datasets."""
        data: dict[str, Any] = {
            "player": self.player,
            "hand": [c.short for c in self.hand],
            "table": [
                {
                    "attack": slot.attacking_card.short,
                    "defense": (
                        None
                        if slot.defending_card is None
                        else slot.defending_card.short
                    ),
                }
                for slot in self.table
            ],
            "trump_card": self.trump_card.short,
            "talon_size": self.talon_size,
            "hand_sizes": list(self.hand_sizes),
            "discard": sorted(c.short for c in self.discard),
            "attacker": self.attacker,
            "phase": self.phase.name,
            "attack_limit": self.attack_limit,
            "observed": sorted(c.short for c in self.observed),
            "durak": self.durak,
        }
        if include_history:
            data["public_history"] = [e.to_dict() for e in self.public_history]
        return data

    def describe(self) -> str:
        """Human-readable dump from this player's seat, for debug output."""
        lines = [
            f"P{self.player} view | phase={self.phase.name} "
            f"{'attacking' if self.is_attacker else 'defending'} "
            f"trump={self.trump_suit.symbol} ({self.trump_card})",
            f"  talon={self.talon_size} discard={len(self.discard)} "
            f"opponent={self.opponent_hand_size} cards "
            f"limit={self.attacks_made}/{self.attack_limit}",
            f"  hand: {format_cards(self.hand) or '(empty)'}",
            f"  table: {'  '.join(str(s) for s in self.table) or '(empty)'}",
            f"  unobserved: {len(self.unobserved_cards())} cards",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"InformationSet(P{self.player}, phase={self.phase.name}, "
            f"hand={len(self.hand)}, talon={self.talon_size}, "
            f"table={len(self.table)})"
        )


def observe(
    state: GameState, player: int, *, include_history: bool = True
) -> InformationSet:
    """Project ``state`` onto what ``player`` is allowed to know.

    A pure function: the same state and player always give an identical
    information set, and the state is not touched.

    Args:
        state: ground truth. Never handed to an agent.
        player: the observer's seat.
        include_history: whether to build the redacted event list. The
            log is walked either way to accumulate
            :attr:`InformationSet.observed`, so switching this off saves
            the per-event object construction, not the traversal. That
            traversal is O(moves so far) — negligible once per turn, not
            at search-node rates. Maintaining both incrementally is Phase
            6's job.

    Raises:
        IndexError: if ``player`` is not a seat in this game.
    """
    if not 0 <= player < len(state.hands):
        raise IndexError(f"no such player: {player}")

    # One walk of the event log yields both outputs. The cumulative
    # observation set is built whether or not the redacted history is
    # returned, so `observed_cards()` never depends on a performance flag.
    seen: set[Card] = set(state.hands[player])
    seen.update(state.table_cards)
    seen.update(state.discard)
    seen.add(state.trump_card)

    events: list[PublicEvent] = []
    for event in state.events():
        seen.update(event.revealed)
        seen.update(event.discarded)
        seen.update(event.taken)
        for record in event.draws:
            if record.player == player:
                seen.update(record.cards)
        if include_history:
            events.append(redact_event(event, player))

    return InformationSet(
        player=player,
        hand=state.hands[player],
        table=state.table,
        trump_suit=state.trump_suit,
        trump_card=state.trump_card,
        talon_size=len(state.talon),
        hand_sizes=tuple(len(hand) for hand in state.hands),
        discard=state.discard,
        attacker=state.attacker,
        phase=state.phase,
        attack_limit=state.attack_limit,
        public_history=tuple(events),
        observed=frozenset(seen),
        rules=state.rules,
        durak=state.durak,
    )
