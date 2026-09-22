"""Game state.

:class:`GameState` is an immutable snapshot containing everything needed
to reproduce a position. ``apply_move`` returns a new state; the input is
never touched, which is what lets search explore millions of hypothetical
lines without defensive copying.

State machine
-------------
Only four phases exist:

===============  ==============================================
``ATTACK``       attacker may add a card or end the bout
``DEFENSE``      defender must beat the open attack or take
``TAKING``       defender has announced a take; attacker may
                 still throw in matching ranks
``GAME_OVER``    terminal
===============  ==============================================

There is deliberately no ``DEALING`` or ``RESOLUTION`` phase. Those are
*transitions*, not decision points: dealing happens inside
:func:`durakfish.game.rules.new_game`, and discarding, taking and drawing
all happen atomically inside the ``EndAttack`` transition. The resulting
invariant is worth the omission — **every non-terminal state is one where
somebody has a decision to make**, so search never has to step through
bookkeeping nodes.

Card conservation
-----------------
A card is in exactly one of: a hand, the talon, the table, or the
discard pile. :attr:`trump_card` is a *reference* to a card that lives in
one of those places (the talon at first, later a hand), not a location of
its own — counting it separately would double-count it.
"""

from __future__ import annotations

import bisect
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Any

from durakfish.cards import Card, Rank, Suit, format_cards
from durakfish.exceptions import InvalidStateError
from durakfish.game.history import GameEvent, HistoryNode
from durakfish.game.ruleset import STANDARD_RULES, RuleSet

__all__ = ["GameState", "Phase", "TableSlot", "add_cards", "remove_cards"]


class Phase(IntEnum):
    """Whose decision is pending.

    Not to be confused with the *strategic* game phase (early / middle /
    late / endgame) used for evaluation weighting, which arrives with the
    evaluator and is a different concept entirely.
    """

    ATTACK = 0
    DEFENSE = 1
    TAKING = 2
    GAME_OVER = 3


def _code(card: Card) -> int:
    return card.code


def add_cards(hand: tuple[Card, ...], cards: Iterable[Card]) -> tuple[Card, ...]:
    """Insert cards into a hand, keeping it sorted by card code.

    Hands are kept in canonical order so that two positions differing only
    by the order cards were picked up hash identically. Insertion is a
    bisect rather than a re-sort so the invariant costs O(n) per card
    instead of O(n log n) per state.
    """
    working = list(hand)
    for card in cards:
        bisect.insort(working, card, key=_code)
    return tuple(working)


def remove_cards(hand: tuple[Card, ...], cards: Iterable[Card]) -> tuple[Card, ...]:
    """Remove cards from a hand, preserving sorted order."""
    doomed = set(cards)
    return tuple(card for card in hand if card not in doomed)


@dataclass(frozen=True, slots=True)
class TableSlot:
    """One attacking card and the card that beat it, if any.

    The table is a sequence of these rather than a flat card list, so the
    attack/defence pairing is never lost. That pairing is not decoration:
    the probability engine reads it to infer what the defender could not
    or would not beat.
    """

    attacking_card: Card
    defending_card: Card | None = None

    @property
    def is_defended(self) -> bool:
        return self.defending_card is not None

    @property
    def cards(self) -> tuple[Card, ...]:
        if self.defending_card is None:
            return (self.attacking_card,)
        return (self.attacking_card, self.defending_card)

    def __str__(self) -> str:
        if self.defending_card is None:
            return f"{self.attacking_card}→?"
        return f"{self.attacking_card}→{self.defending_card}"


@dataclass(frozen=True, slots=True)
class GameState:
    """An immutable Durak position.

    Construct via :func:`durakfish.game.rules.new_game`; advance via
    :func:`durakfish.game.rules.apply_move`.
    """

    hands: tuple[tuple[Card, ...], ...]
    talon: tuple[Card, ...]
    discard: frozenset[Card]
    table: tuple[TableSlot, ...]
    trump_card: Card
    trump_suit: Suit
    attacker: int
    phase: Phase
    attack_limit: int
    durak: int | None = None
    history: HistoryNode | None = None
    rules: RuleSet = STANDARD_RULES

    # ------------------------------------------------------------------
    # Roles and turn order
    # ------------------------------------------------------------------
    @property
    def num_players(self) -> int:
        return len(self.hands)

    @property
    def defender(self) -> int:
        return (self.attacker + 1) % self.num_players

    @property
    def current_player(self) -> int | None:
        """Whose decision is pending, or ``None`` if the game is over.

        Derived from the phase rather than stored, so the two can never
        drift apart.
        """
        if self.phase is Phase.GAME_OVER:
            return None
        if self.phase is Phase.DEFENSE:
            return self.defender
        return self.attacker

    def hand(self, player: int) -> tuple[Card, ...]:
        return self.hands[player]

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------
    @property
    def table_cards(self) -> tuple[Card, ...]:
        return tuple(card for slot in self.table for card in slot.cards)

    @property
    def table_ranks(self) -> frozenset[Rank]:
        """Ranks that may be added to the attack.

        Includes ranks of defending cards, which is the standard
        Podkidnoy rule (see ``docs/rules.md``).
        """
        return frozenset(card.rank for card in self.table_cards)

    @property
    def undefended(self) -> tuple[TableSlot, ...]:
        return tuple(slot for slot in self.table if not slot.is_defended)

    @property
    def attacks_made(self) -> int:
        """Attacking cards played this bout; compared against :attr:`attack_limit`."""
        return len(self.table)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    @property
    def is_over(self) -> bool:
        return self.phase is Phase.GAME_OVER

    @property
    def is_draw(self) -> bool:
        return self.phase is Phase.GAME_OVER and self.durak is None

    @property
    def talon_size(self) -> int:
        return len(self.talon)

    def is_open_information(self) -> bool:
        """True when no hidden information remains for either player.

        In two-player Durak, an empty talon means each player can deduce
        the other's hand exactly: it is the deck minus their own hand,
        minus the discard pile, minus the table. From this point the game
        is a finite perfect-information game and can be solved exactly
        rather than sampled — the basis of the endgame solver.

        With three or more players an empty talon only reveals the
        *union* of the opponents' hands, so this deliberately returns
        False there rather than quietly overclaiming.
        """
        return not self.talon and self.num_players == 2

    # ------------------------------------------------------------------
    # Copying
    # ------------------------------------------------------------------
    def replace(self, **changes: Any) -> GameState:
        """Return a copy with the given fields changed."""
        return replace(self, **changes)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------
    def events(self) -> tuple[GameEvent, ...]:
        """Materialise the history in chronological order."""
        out: list[GameEvent] = []
        node = self.history
        while node is not None:
            out.append(node.event)
            node = node.parent
        out.reverse()
        return tuple(out)

    @property
    def event_count(self) -> int:
        return 0 if self.history is None else self.history.length

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------
    def transposition_key(self) -> tuple[Any, ...]:
        """A deterministic, canonical key for this position.

        Built only from integers so it is stable across processes, unlike
        anything derived from string hashing.

        The talon's **order** is included. Two positions with identical
        hands but different talon orders are genuinely different
        positions, because the next draw differs; conflating them would
        make a perfect-information transposition table return wrong
        values. Imperfect-information search keys on information sets
        instead (Phase 6), which is a different key by construction.
        """
        return (
            int(self.phase),
            self.attacker,
            self.attack_limit,
            tuple(tuple(c.code for c in hand) for hand in self.hands),
            tuple(c.code for c in self.talon),
            tuple(
                (
                    slot.attacking_card.code,
                    -1 if slot.defending_card is None else slot.defending_card.code,
                )
                for slot in self.table
            ),
            tuple(sorted(c.code for c in self.discard)),
            int(self.trump_suit),
            self.trump_card.code,
            -1 if self.durak is None else self.durak,
        )

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Raise :class:`InvalidStateError` if any invariant is broken.

        Cheap enough to call from every test and from fuzzing, too
        expensive to call from search.
        """
        rules = self.rules
        if len(self.hands) != rules.num_players:
            raise InvalidStateError(
                f"expected {rules.num_players} hands, got {len(self.hands)}"
            )

        # --- conservation: every card exactly once -----------------------
        counts: Counter[Card] = Counter()
        for hand in self.hands:
            counts.update(hand)
        counts.update(self.talon)
        counts.update(self.table_cards)
        counts.update(self.discard)

        duplicated = [card for card, n in counts.items() if n > 1]
        if duplicated:
            raise InvalidStateError(
                "card in two places at once: "
                + format_cards(sorted(duplicated, key=_code))
            )
        total = sum(counts.values())
        if total != rules.deck_size:
            raise InvalidStateError(
                f"{total} cards accounted for, expected {rules.deck_size}"
            )

        # --- canonical hand order ---------------------------------------
        for player, hand in enumerate(self.hands):
            if list(hand) != sorted(hand, key=_code):
                raise InvalidStateError(f"hand {player} is not in canonical order")

        # --- trump ------------------------------------------------------
        if self.trump_card not in counts:
            raise InvalidStateError("trump card is not anywhere in play")
        if self.trump_suit is not self.trump_card.suit:
            raise InvalidStateError(
                f"trump suit {self.trump_suit} disagrees with trump card "
                f"{self.trump_card}"
            )

        # --- table shape ------------------------------------------------
        seen_undefended = False
        for slot in self.table:
            if slot.is_defended and seen_undefended:
                raise InvalidStateError(
                    "a defended slot follows an undefended one on the table"
                )
            if not slot.is_defended:
                seen_undefended = True

        if self.attack_limit > rules.max_attacks_per_bout:
            raise InvalidStateError(
                f"attack limit {self.attack_limit} exceeds the cap "
                f"{rules.max_attacks_per_bout}"
            )
        if self.attacks_made > self.attack_limit:
            raise InvalidStateError(
                f"{self.attacks_made} attacks made, limit is {self.attack_limit}"
            )

        # --- phase consistency ------------------------------------------
        if self.phase is Phase.ATTACK:
            if self.undefended:
                raise InvalidStateError("ATTACK phase with an undefended attack")
        elif self.phase is Phase.DEFENSE:
            if len(self.undefended) != 1 or self.table[-1].is_defended:
                raise InvalidStateError(
                    "DEFENSE phase must have exactly one open attack, last on the table"
                )
        elif self.phase is Phase.TAKING:
            if not self.undefended:
                raise InvalidStateError("TAKING phase with nothing to take")
        else:  # GAME_OVER
            if self.table:
                raise InvalidStateError("game over with cards still on the table")
            if self.talon:
                raise InvalidStateError("game over with cards still in the talon")

        if self.phase is not Phase.GAME_OVER:
            if self.durak is not None:
                raise InvalidStateError("durak recorded before the game ended")
            if not 0 <= self.attacker < rules.num_players:
                raise InvalidStateError(f"invalid attacker index {self.attacker}")
        elif self.durak is not None:
            if not 0 <= self.durak < rules.num_players:
                raise InvalidStateError(f"invalid durak index {self.durak}")
            if not self.hands[self.durak]:
                raise InvalidStateError("the durak must be the player holding cards")

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self, *, include_history: bool = True) -> dict[str, Any]:
        """JSON-friendly encoding. Compact card notation keeps it readable."""
        data: dict[str, Any] = {
            "hands": [[c.short for c in hand] for hand in self.hands],
            "talon": [c.short for c in self.talon],
            "discard": sorted(c.short for c in self.discard),
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
            "attacker": self.attacker,
            "phase": self.phase.name,
            "attack_limit": self.attack_limit,
            "durak": self.durak,
        }
        if include_history:
            data["history"] = [event.to_dict() for event in self.events()]
        return data

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, rules: RuleSet = STANDARD_RULES
    ) -> GameState:
        """Rebuild a state from :meth:`to_dict` output.

        History is restored as a flat list of events; the linked structure
        is rebuilt in order.
        """
        from durakfish.game.history import push_event  # local: avoids a cycle

        trump_card = Card.parse(data["trump_card"])
        history: HistoryNode | None = None
        for raw in data.get("history", ()):
            history = push_event(history, GameEvent.from_dict(raw))
        return cls(
            hands=tuple(
                tuple(Card.parse(c) for c in hand) for hand in data["hands"]
            ),
            talon=tuple(Card.parse(c) for c in data["talon"]),
            discard=frozenset(Card.parse(c) for c in data["discard"]),
            table=tuple(
                TableSlot(
                    Card.parse(slot["attack"]),
                    None if slot["defense"] is None else Card.parse(slot["defense"]),
                )
                for slot in data["table"]
            ),
            trump_card=trump_card,
            trump_suit=trump_card.suit,
            attacker=int(data["attacker"]),
            phase=Phase[data["phase"]],
            attack_limit=int(data["attack_limit"]),
            durak=data["durak"],
            history=history,
            rules=rules,
        )

    # ------------------------------------------------------------------
    # Debugging
    # ------------------------------------------------------------------
    def describe(self) -> str:
        """Multi-line human-readable dump, for debug mode and failing tests."""
        lines = [
            f"phase={self.phase.name} attacker=P{self.attacker} "
            f"defender=P{self.defender} trump={self.trump_suit.symbol} "
            f"({self.trump_card})",
            f"talon={len(self.talon)} discard={len(self.discard)} "
            f"limit={self.attacks_made}/{self.attack_limit}",
        ]
        for player, hand in enumerate(self.hands):
            marker = "*" if player == self.current_player else " "
            lines.append(f" {marker}P{player}: {format_cards(hand) or '(empty)'}")
        table = "  ".join(str(slot) for slot in self.table) or "(empty)"
        lines.append(f"  table: {table}")
        if self.is_over:
            lines.append(
                "  result: draw" if self.durak is None else f"  durak: P{self.durak}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GameState(phase={self.phase.name}, attacker={self.attacker}, "
            f"hands={[len(h) for h in self.hands]}, talon={len(self.talon)}, "
            f"table={len(self.table)})"
        )
