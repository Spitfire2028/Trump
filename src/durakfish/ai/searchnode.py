"""The search node.

What a node is, exactly
----------------------
A :class:`SearchNode` is **not** a game state and **not** an information
set. It is a third thing, and blurring them would be the easiest way to
get this phase wrong. A node is:

    the searching player's own hand, plus the public position, plus a
    conservative abstraction of what the opponent may do next.

Why an abstraction is unavoidable
---------------------------------
A full-width game tree cannot be built from an information set. Two
things block it, and neither is a shortcoming of the implementation:

* **Opponent replies.** Expanding "the defender beats the attack" needs
  their card identities. Hidden.
* **Bout resolution.** Expanding past *bito* or a take needs the talon
  order, because both players draw. Hidden.

Phase 5 forbids determinization, so the search does not sample hidden
cards, does not enumerate hidden hands, and assigns no probabilities.
Instead it does the only honest thing left: it expands the searcher's own
moves exactly, models the opponent by **action class** rather than by
card, and stops at any frontier where continuing would require knowledge
it does not have.

The opponent abstraction
------------------------
At a node where the opponent moves, the branches are action classes:

============  ==========================================================
Action        Modelled effect
============  ==========================================================
``BEAT``      one card leaves the opponent's hand and covers the open
              attack. Its identity is unknown, so the slot is marked
              defended-by-unknown and contributes **no new rank** to the
              table.
``TAKE``      the opponent picks up; the bout enters its throwing-in
              phase and the searcher moves again.
``ADD``       the opponent puts an unknown card on the table. The
              searcher would then have to beat a card it cannot see, so
              this is a frontier: the branch stops.
``END``       the opponent ends the bout, which resolves it.
============  ==========================================================

Offering both ``BEAT`` and ``TAKE`` is the *opposite* of inference: it
assumes nothing about what the opponent holds and considers both. The
only thing consulted is their **hand size**, which is public — an
opponent with no cards cannot beat.

Not assuming an unknown card's rank is likewise conservative. It can only
cause the search to underestimate its own future throw-in options, never
to invent one.

Where genuine terminals come from
---------------------------------
When the talon is empty, a bout resolution yields exact card *counts* for
both players from public information plus the searcher's own hand — no
identity is inferred. So "I have run out and the opponent has not" is a
real, provable win, and the search finds real forced wins in the endgame.
While the talon still has cards, draws make the continuation unknown and
the node is scored heuristically instead.

What this node deliberately does not contain
--------------------------------------------
No opponent cards, no talon contents or order, no hidden-card
assignment, no event history. Unknown cards are represented by ``None``
and counted, never guessed. Every field below is either public
information or the searcher's own hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from durakfish.cards import Card, Rank, Suit
from durakfish.game.moves import (
    END_ATTACK,
    TAKE,
    AttackMove,
    DefenseMove,
    EndAttack,
    Move,
    TakeCards,
)
from durakfish.game.rules import beats
from durakfish.game.state import Phase
from durakfish.information import InformationSet

__all__ = [
    "Actor",
    "NodeKind",
    "OpponentAction",
    "Outcome",
    "SearchNode",
    "SearchSlot",
]


class Actor(Enum):
    """Whose decision a node represents."""

    SELF = "self"
    OPPONENT = "opponent"


class NodeKind(Enum):
    """Why a node can or cannot be expanded further."""

    #: Somebody has a decision to make; the node has children.
    DECISION = "decision"
    #: The bout ended. Beyond it players draw from a hidden talon, so the
    #: continuation is unknowable and the node is a leaf.
    BOUT_RESOLVED = "bout_resolved"
    #: An unknown card entered the position. The searcher cannot enumerate
    #: its own replies to a card it has not seen, so the branch stops.
    UNKNOWN_FRONTIER = "unknown_frontier"


class Outcome(Enum):
    """A proven result, available only when the talon is empty."""

    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"


class OpponentAction(Enum):
    """An opponent branch. Deliberately not a :class:`Move`.

    Moves name cards. These name *classes of reply*, because the cards are
    hidden. Keeping them a distinct type stops the two ever being confused
    for one another, and makes it impossible to hand one to the rules
    engine by accident.
    """

    BEAT = "beat"
    TAKE = "take"
    ADD = "add"
    END = "end"


@dataclass(frozen=True, slots=True)
class SearchSlot:
    """One attack on the table, as the searcher can see it.

    ``None`` means "a card is there but its face is unknown" — never
    "guess which card". A slot is only ever unknown because a simulated
    opponent action put it there.
    """

    attacking_card: Card | None
    defending_card: Card | None = None
    defended: bool = False

    @property
    def card_count(self) -> int:
        return 1 + (1 if self.defended else 0)

    @property
    def known_cards(self) -> tuple[Card, ...]:
        cards = [c for c in (self.attacking_card, self.defending_card) if c is not None]
        return tuple(cards)

    @property
    def unknown_count(self) -> int:
        unknown = 1 if self.attacking_card is None else 0
        if self.defended and self.defending_card is None:
            unknown += 1
        return unknown


@dataclass(frozen=True, slots=True)
class SearchNode:
    """A position in the search tree. Immutable; transitions return copies.

    Attributes:
        hand: the searcher's own cards, exactly known. Own information.
        unknown_own: cards the searcher would hold whose identity it does
            not know — picked up from an unknown slot, or drawn. Counted,
            never guessed.
        table: attacks in play, with unknown faces marked ``None``.
        known_ranks: ranks visible on the table. Unknown cards contribute
            nothing, which understates the searcher's throw-in options
            rather than inventing them.
        opponent_cards: how many cards the opponent holds. Public.
        talon_size: how many cards remain to be drawn. Public. Contents
            and order are absent by construction.
        trump: public from the face-up trump card.
        refill_to: the hand size players draw back up to. Rules config.
        attack_limit: cards allowed in this bout. Public.
        searcher_is_attacker: fixed for the whole tree, because the tree
            never crosses a bout boundary.
    """

    hand: tuple[Card, ...]
    table: tuple[SearchSlot, ...]
    known_ranks: frozenset[Rank]
    opponent_cards: int
    talon_size: int
    trump: Suit
    refill_to: int
    attack_limit: int
    searcher_is_attacker: bool
    phase: Phase
    to_move: Actor
    kind: NodeKind = NodeKind.DECISION
    unknown_own: int = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_view(cls, view: InformationSet) -> SearchNode:
        """Build the root from an information set.

        Every field is copied from something the observer may legitimately
        know. Nothing is derived, deduced or sampled.
        """
        slots = tuple(
            SearchSlot(
                attacking_card=slot.attacking_card,
                defending_card=slot.defending_card,
                defended=slot.is_defended,
            )
            for slot in view.table
        )
        ranks = frozenset(
            card.rank for slot in slots for card in slot.known_cards
        )
        return cls(
            hand=tuple(view.hand),
            table=slots,
            known_ranks=ranks,
            opponent_cards=view.opponent_hand_size,
            talon_size=view.talon_size,
            trump=view.trump_suit,
            refill_to=view.rules.hand_size,
            attack_limit=view.attack_limit,
            searcher_is_attacker=view.is_attacker,
            phase=view.phase,
            to_move=Actor.SELF if view.is_to_move else Actor.OPPONENT,
        )

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------
    @property
    def own_cards(self) -> int:
        """Total cards the searcher holds, known faces plus unknown ones."""
        return len(self.hand) + self.unknown_own

    @property
    def is_leaf(self) -> bool:
        return self.kind is not NodeKind.DECISION

    @property
    def table_card_count(self) -> int:
        return sum(slot.card_count for slot in self.table)

    @property
    def open_attack(self) -> Card | None:
        """The card awaiting a defence, if it is one the searcher can see."""
        for slot in self.table:
            if not slot.defended:
                return slot.attacking_card
        return None

    def outcome(self) -> Outcome | None:
        """A proven result, or ``None`` if the position is not decided.

        Only ever non-``None`` at a resolved bout with an empty talon,
        where both card counts follow from public information and the
        searcher's own hand. No identity is inferred to reach this.
        """
        if self.kind is not NodeKind.BOUT_RESOLVED or self.talon_size:
            return None
        mine_out = self.own_cards == 0
        theirs_out = self.opponent_cards == 0
        if mine_out and theirs_out:
            return Outcome.DRAW
        if mine_out:
            return Outcome.WIN
        if theirs_out:
            return Outcome.LOSS
        return None

    # ------------------------------------------------------------------
    # The searcher's own moves — exact, never abstracted
    # ------------------------------------------------------------------
    def self_moves(self) -> tuple[Move, ...]:
        """Legal moves for the searcher, in the rules engine's own order.

        Derived from the searcher's own hand and the public table, exactly
        as ``docs/rules.md`` specifies. ``tests/test_search.py`` checks
        this against the authoritative generator on real positions, so a
        divergence from the frozen rules engine fails the build.
        """
        if self.is_leaf or self.to_move is not Actor.SELF:
            return ()

        if self.phase is Phase.DEFENSE:
            open_card = self.open_attack
            if open_card is None:  # pragma: no cover - guarded by callers
                return ()
            moves: list[Move] = [
                DefenseMove(open_card, card)
                for card in self.hand
                if beats(card, open_card, self.trump)
            ]
            moves.append(TAKE)
            return tuple(moves)

        return self._addition_moves(self.hand)

    def _addition_moves(self, hand: tuple[Card, ...]) -> tuple[Move, ...]:
        """Attack or throw-in options, plus ending the bout."""
        room = self.attack_limit - len(self.table)
        if not self.table:
            playable = hand if room > 0 else ()
        elif room > 0:
            playable = tuple(c for c in hand if c.rank in self.known_ranks)
        else:
            playable = ()
        moves: list[Move] = [AttackMove(c) for c in playable]
        if self.table:
            moves.append(END_ATTACK)
        return tuple(moves)

    # ------------------------------------------------------------------
    # The opponent's action classes — abstract, never card-level
    # ------------------------------------------------------------------
    def opponent_actions(self) -> tuple[OpponentAction, ...]:
        """What the opponent might do, by class.

        Only their **hand size** is consulted, which is public. No attempt
        is made to work out whether they can actually beat a given card;
        both possibilities are kept.
        """
        if self.is_leaf or self.to_move is not Actor.OPPONENT:
            return ()

        if self.phase is Phase.DEFENSE:
            if self.opponent_cards <= 0:
                return (OpponentAction.TAKE,)
            return (OpponentAction.BEAT, OpponentAction.TAKE)

        actions: list[OpponentAction] = []
        can_add = (
            self.opponent_cards > 0
            and len(self.table) < self.attack_limit
            and bool(self.table)
        )
        if can_add:
            actions.append(OpponentAction.ADD)
        if self.table:
            actions.append(OpponentAction.END)
        elif self.opponent_cards > 0:
            # Opening a bout is compulsory and always possible.
            actions.append(OpponentAction.ADD)
        return tuple(actions)

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------
    def after_self(self, move: Move) -> SearchNode:
        """Apply one of the searcher's own moves."""
        if isinstance(move, AttackMove):
            return self._with(
                hand=tuple(c for c in self.hand if c != move.card),
                table=(*self.table, SearchSlot(move.card)),
                known_ranks=self.known_ranks | {move.card.rank},
                phase=Phase.DEFENSE if self.phase is Phase.ATTACK else self.phase,
                to_move=Actor.OPPONENT if self.phase is Phase.ATTACK else Actor.SELF,
            )
        if isinstance(move, DefenseMove):
            return self._with(
                hand=tuple(c for c in self.hand if c != move.defending_card),
                table=self._close_open_slot(move.defending_card),
                known_ranks=self.known_ranks | {move.defending_card.rank},
                phase=Phase.ATTACK,
                to_move=Actor.OPPONENT,
            )
        if isinstance(move, TakeCards):
            return self._with(phase=Phase.TAKING, to_move=Actor.OPPONENT)
        if isinstance(move, EndAttack):
            return self._resolve()
        raise TypeError(f"not a move this search understands: {move!r}")

    def after_opponent(self, action: OpponentAction) -> SearchNode:
        """Apply one opponent action class."""
        if action is OpponentAction.BEAT:
            # A card leaves their hand and covers the attack. Its face is
            # unknown, so it contributes no rank to the table.
            return self._with(
                table=self._close_open_slot(None),
                opponent_cards=self.opponent_cards - 1,
                phase=Phase.ATTACK,
                to_move=Actor.SELF,
            )
        if action is OpponentAction.TAKE:
            return self._with(phase=Phase.TAKING, to_move=Actor.SELF)
        if action is OpponentAction.ADD:
            # The searcher would have to answer a card it cannot see.
            return self._with(
                table=(*self.table, SearchSlot(None)),
                opponent_cards=self.opponent_cards - 1,
                phase=Phase.DEFENSE,
                to_move=Actor.SELF,
                kind=NodeKind.UNKNOWN_FRONTIER,
            )
        if action is OpponentAction.END:
            return self._resolve()
        raise TypeError(f"unknown opponent action: {action!r}")

    # ------------------------------------------------------------------
    def _close_open_slot(self, card: Card | None) -> tuple[SearchSlot, ...]:
        slots = list(self.table)
        for index, slot in enumerate(slots):
            if not slot.defended:
                slots[index] = SearchSlot(slot.attacking_card, card, defended=True)
                break
        return tuple(slots)

    def _resolve(self) -> SearchNode:
        """End the bout: move the table, then refill from the talon.

        Card *counts* after refilling follow from public information; card
        *identities* drawn do not, so the result is a leaf. Draw order
        follows ``docs/rules.md`` §7: the attacker refills first.
        """
        hand = self.hand
        unknown_own = self.unknown_own
        opponent = self.opponent_cards

        if self.phase is Phase.TAKING:
            known = tuple(c for slot in self.table for c in slot.known_cards)
            unknown = sum(slot.unknown_count for slot in self.table)
            if self.searcher_is_attacker:
                opponent += len(known) + unknown
            else:
                hand = hand + known
                unknown_own += unknown

        own_total = len(hand) + unknown_own
        # A count of cards left to draw, never their identities. Named to
        # keep the audit's blunt name check meaningful.
        remaining = self.talon_size
        counts = {True: own_total, False: opponent}  # keyed by "is the searcher"
        drawn_by_searcher = 0
        first = self.searcher_is_attacker
        for is_searcher in (first, not first):
            need = max(0, self.refill_to - counts[is_searcher])
            take = min(need, remaining)
            remaining -= take
            counts[is_searcher] += take
            if is_searcher:
                drawn_by_searcher = take

        return self._with(
            hand=hand,
            unknown_own=unknown_own + drawn_by_searcher,
            table=(),
            known_ranks=frozenset(),
            opponent_cards=counts[False],
            talon_size=remaining,
            to_move=Actor.SELF,
            kind=NodeKind.BOUT_RESOLVED,
        )

    def _with(self, **changes: Any) -> SearchNode:
        from dataclasses import replace

        return replace(self, **changes)

    # ------------------------------------------------------------------
    def key(self) -> tuple[Any, ...]:
        """Stable identity built from integers only.

        Used by tests to compare nodes across processes. Deliberately not
        wired into a transposition table — that is a later optimisation
        phase, and correctness comes first.
        """
        return (
            tuple(c.code for c in self.hand),
            self.unknown_own,
            tuple(
                (
                    -1 if s.attacking_card is None else s.attacking_card.code,
                    -1 if s.defending_card is None else s.defending_card.code,
                    s.defended,
                )
                for s in self.table
            ),
            tuple(sorted(int(r) for r in self.known_ranks)),
            self.opponent_cards,
            self.talon_size,
            int(self.trump),
            self.refill_to,
            self.attack_limit,
            self.searcher_is_attacker,
            int(self.phase),
            self.to_move.value,
            self.kind.value,
        )

    def __repr__(self) -> str:
        return (
            f"SearchNode({self.kind.value}, {self.to_move.value} to move, "
            f"hand={self.own_cards}, opp={self.opponent_cards}, "
            f"talon={self.talon_size}, table={len(self.table)})"
        )
