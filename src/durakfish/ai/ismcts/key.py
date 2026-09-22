"""Information-set identity, and the epistemic state carried through search.

Two things live here, and keeping them apart is the whole point of the
phase:

* :class:`InformationSetKey` — *what the observer knows*. It identifies a
  decision point and must never depend on which hidden world was sampled.
* :class:`EpistemicState` — the part of that knowledge which cannot be
  re-derived from a hypothetical position and therefore has to be carried
  forward explicitly.

Why the epistemic state is carried rather than recomputed
---------------------------------------------------------
Phase 7.3.3 established, and this module depends on, a hard constraint: a
:class:`~durakfish.ai.DeterminizedPosition` deliberately carries **no
event history**, because an observer's history has draws redacted and a
hypothesis cannot reconstruct it. Two consequences follow.

First, ``CardTracker.update()`` **cannot** be fed a hypothetical view. It
does not merely give a poor answer — it refuses, with
``TrackerError: event stream shrank from 37 to 2``. Any implementation
that tried would fail loudly, which is the correct behaviour and is
regression-tested.

Second, the cumulative *observed* set — the record of every card whose
face the observer has seen — would silently reset to "whatever is
publicly visible right now" if it were recomputed from a hypothesis. The
observed set is part of information-set identity, so resetting it would
merge decision points that are genuinely different.

So the observed set is threaded through the traversal by hand. It only
ever grows, and it grows by union with whatever the observer can see in
the position they have reached.

What is *not* carried
---------------------
Knowledge and belief are not propagated down the tree. They are needed
only to sample a world, and Phase 7.3.4 fixed re-determinization at the
root, where the real :class:`~durakfish.information.SearchInformation` is
available. Carrying a belief that nothing consumes would be dead weight
that could drift out of step with the observations beside it.

``retained_by_opponent`` is carried anyway, because it is cheap and it is
the one epistemic fact that a future action-conditioned belief would need
first. It does not currently feed the key, and the docstring on
:meth:`EpistemicState.advance` says so rather than implying otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from durakfish.ai.determinized import DeterminizedPosition
from durakfish.cards import Card
from durakfish.game.moves import Move
from durakfish.information import InformationSet, observe

__all__ = ["EpistemicState", "InformationSetKey", "Observation", "information_key"]


@dataclass(frozen=True, slots=True)
class InformationSetKey:
    """Identity of one observer's decision point.

    Composed of the owner and
    :meth:`~durakfish.information.InformationSet.key`, which Phase 7.3.4
    selected over ``SearchInformation.fingerprint()``: the fingerprint
    embeds marginals and allocation counts, so two identical decision
    points would key apart merely because a sampler differed.

    The underlying view key covers own hand, table, trump, talon *size*,
    hand *sizes*, discard, attacker, phase, attack limit, the cumulative
    observed set, and the durak. It contains **no** opponent card, no
    talon contents, and no world allocation — which is what makes two
    observationally identical worlds land on the same node.

    Owner is composed in explicitly: the underlying view key already
    carries ``player``, but relying on that implicitly would make the
    identity silently wrong the day a view is built for another seat.
    """

    owner: int
    view_key: tuple[Any, ...]

    def __repr__(self) -> str:
        return f"InformationSetKey(P{self.owner}, {hash(self.view_key) & 0xFFFF:04x})"


@dataclass(frozen=True, slots=True)
class Observation:
    """What an action reveals to one observer.

    Distinct from the action itself: ``P2 takes`` is a public move, but
    which cards are thereby in their hand, and what else they hold, are
    separate questions. This records only what the observer may legally
    learn.

    Attributes:
        move: the action played. Public — every move in Durak is.
        actor: who played it.
        revealed: cards whose face became visible to this observer as a
            result. Never hidden draws belonging to the opponent.
        terminal: whether the game ended.
    """

    move: Move
    actor: int
    revealed: frozenset[Card]
    terminal: bool

    def __repr__(self) -> str:
        return (
            f"Observation(P{self.actor} {self.move}, "
            f"revealed={len(self.revealed)}, terminal={self.terminal})"
        )


@dataclass(frozen=True, slots=True)
class EpistemicState:
    """The observer's knowledge, carried explicitly through a traversal.

    Immutable. :meth:`advance` returns a new state rather than mutating,
    so a branch of the search cannot corrupt a sibling.

    Attributes:
        owner: whose knowledge this is.
        observed: every card whose face this observer has seen. Grows
            monotonically; never recomputed from a hypothesis.
        retained_by_opponent: cards seen taken by the opponent and not
            seen played since. Carried for future use; see the module
            docstring.
    """

    owner: int
    observed: frozenset[Card]
    retained_by_opponent: frozenset[Card] = frozenset()

    @classmethod
    def from_view(cls, view: InformationSet) -> EpistemicState:
        """Seed from the real root view, where the full history exists."""
        return cls(owner=view.player, observed=frozenset(view.observed))

    def advance(
        self, position: DeterminizedPosition, observation: Observation
    ) -> EpistemicState:
        """Fold one observation in, returning the successor state.

        The observed set grows by union with everything visible in the
        position now reached — the observer's own hand, the table, the
        discard and the face-up trump. Union rather than replacement is
        what preserves the cumulative semantics that a recomputation from
        a hypothesis would destroy.

        ``retained_by_opponent`` is updated on the same rules the Phase 6
        tracker uses, but nothing currently reads it: it does not feed
        :func:`information_key`. It is maintained so that an
        action-conditioned belief, if one is ever added, does not have to
        reconstruct it after the fact.
        """
        visible = observe(position.state, self.owner).observed
        retained = self.retained_by_opponent
        if observation.actor != self.owner:
            retained = retained - observation.revealed
        return EpistemicState(
            owner=self.owner,
            observed=self.observed | visible | observation.revealed,
            retained_by_opponent=retained - visible,
        )

    def view_at(self, position: DeterminizedPosition) -> InformationSet:
        """The observer's view of ``position``, with carried knowledge restored.

        ``observe`` on a hypothesis reports only what is visible right
        now, because the hypothesis has no history. Overlaying the
        carried observed set is what turns that into the observer's
        actual information state.
        """
        local = observe(position.state, self.owner)
        return replace(local, observed=self.observed | local.observed)

    def __repr__(self) -> str:
        return (
            f"EpistemicState(P{self.owner}, observed={len(self.observed)}, "
            f"retained={len(self.retained_by_opponent)})"
        )


def information_key(
    position: DeterminizedPosition, epistemic: EpistemicState
) -> InformationSetKey:
    """Identity of the decision point ``position`` represents to the observer.

    Derived from the observer's view of the hypothesis plus the carried
    knowledge — never from the hypothesis's hidden contents. Two worlds
    that look identical to the observer therefore produce the same key,
    which is the invariant this whole phase exists to establish.
    """
    return InformationSetKey(
        owner=epistemic.owner, view_key=epistemic.view_at(position).key()
    )
