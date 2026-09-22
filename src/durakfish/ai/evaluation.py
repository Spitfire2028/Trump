"""Scoring a search node.

Two separate things live here, and the distinction matters:

* **Terminal scoring** is exact. When a node is a proven win, loss or
  draw, it gets a fixed score and no heuristic is consulted.
* **Heuristic evaluation** is a guess about an undecided position, and is
  deliberately crude in this phase.

Every score is from the **searching player's point of view**: higher is
better for them, at every node, whoever happens to be to move. The
minimax layer takes care of maximising at the searcher's nodes and
minimising at the opponent's. Evaluations are therefore not
side-to-move-relative, which removes an entire category of sign errors.

Information
-----------
An evaluator sees a :class:`~durakfish.ai.searchnode.SearchNode` and
nothing else, so it inherits the node's guarantee: no opponent cards, no
talon contents, no inference. Two positions that are identical to the
searcher score identically — proved in ``tests/test_evaluation.py`` by
permuting hidden cards and comparing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from durakfish.ai.searchnode import Actor, NodeKind, Outcome, SearchNode
from durakfish.exceptions import DurakFishError
from durakfish.game.rules import beats
from durakfish.game.state import Phase

__all__ = [
    "EVALUATORS",
    "MATE_MARGIN",
    "TERMINAL_DRAW",
    "TERMINAL_LOSS",
    "TERMINAL_WIN",
    "BaselineEvaluator",
    "EvaluationWeights",
    "Evaluator",
    "FlexibleEvaluator",
    "MechanisticEvaluator",
    "RepairedBoundedEvaluator",
    "RepairedEvaluator",
    "RepairedObligationEvaluator",
    "RepairedThrowInEvaluator",
    "Score",
    "StructuralEvaluator",
    "StructuralWeights",
    "get_evaluator",
    "terminal_score",
]

Score = float

#: Proven outcomes. Named, not magic: a heuristic must never reach these,
#: so every heuristic term is bounded well inside the margin below.
TERMINAL_WIN: Score = 10_000.0
TERMINAL_LOSS: Score = -10_000.0
TERMINAL_DRAW: Score = 0.0

#: Any score at least this large is a proven win rather than a good guess.
#: Heuristic scores are bounded by roughly +-60, so the gap is wide.
MATE_MARGIN: Score = 1_000.0


def terminal_score(outcome: Outcome, ply: int) -> Score:
    """Score a proven outcome, preferring results that arrive sooner.

    A win at ply 2 scores higher than the same win at ply 6, and a loss
    that can be postponed scores higher than one that cannot. Without this
    the search is indifferent between winning now and winning later, and
    will happily dawdle in a won position.

    The ``ply`` adjustment is small enough that no adjustment can turn a
    win into a loss, or push a proven result inside ``MATE_MARGIN``.
    """
    if outcome is Outcome.WIN:
        return TERMINAL_WIN - ply
    if outcome is Outcome.LOSS:
        return TERMINAL_LOSS + ply
    return TERMINAL_DRAW


@dataclass(frozen=True, slots=True)
class EvaluationWeights:
    """Coefficients of the baseline heuristic.

    Kept in one small object so the heuristic can be restated in a test
    without reaching into the evaluator, and so a future tuning phase has
    somewhere obvious to write to.
    """

    #: Cards the opponent holds minus cards the searcher holds. The
    #: dominant term: in Durak you win by running out.
    card_advantage: float = 1.0
    #: Each trump in the searcher's own hand. Trumps are the scarce
    #: resource; the opponent's are unknown and contribute nothing.
    own_trump: float = 0.30
    #: Each attack still awaiting the searcher's defence. An obligation,
    #: not an asset.
    open_obligation: float = 0.50

    def bound(self) -> float:
        """A loose bound on the magnitude of any heuristic score.

        Used to assert that heuristics can never be mistaken for proven
        results. Thirty-six cards, nine trumps and six open attacks are
        the worst cases the rules allow.
        """
        return (
            36 * abs(self.card_advantage)
            + 9 * abs(self.own_trump)
            + 6 * abs(self.open_obligation)
        )


class Evaluator(Protocol):
    """Scores a node from the searching player's point of view."""

    name: str

    def evaluate(self, node: SearchNode, ply: int) -> Score:
        """Return a score. Higher is better for the searcher.

        Args:
            node: the position. Contains no hidden information.
            ply: distance from the root, used only to prefer sooner
                results at proven terminals.
        """
        ...


class BaselineEvaluator:
    """The simplest evaluation worth having.

    Three terms, all computable from the node:

    1. **Card advantage** — ``opponent_cards - own_cards``. You win by
       running out of cards, so holding fewer is better.
    2. **Own trumps** — a small bonus per trump still in hand. Counted
       only for the searcher; the opponent's trumps are hidden and are
       not guessed at.
    3. **Open obligations** — a penalty per attack the searcher still has
       to answer.

    What it deliberately does not model: which specific cards are strong
    against what remains, whether a card is trapped, tempo, or anything
    requiring a view of the opponent's hand. Those need either a real
    evaluation phase or the hidden-state machinery of later phases.

    Cards whose identity the searcher does not know (drawn, or picked up
    from an unknown slot) count toward the card total but contribute no
    trump bonus — an understatement, never an invention.
    """

    name = "baseline"

    def __init__(self, weights: EvaluationWeights | None = None) -> None:
        self.weights = weights or EvaluationWeights()

    def evaluate(self, node: SearchNode, ply: int = 0) -> Score:
        outcome = node.outcome()
        if outcome is not None:
            return terminal_score(outcome, ply)

        weights = self.weights
        score = weights.card_advantage * (node.opponent_cards - node.own_cards)
        score += weights.own_trump * sum(
            1 for card in node.hand if card.suit == node.trump
        )
        if self._facing_attacks(node):
            open_attacks = sum(1 for slot in node.table if not slot.defended)
            score -= weights.open_obligation * open_attacks
        return score

    @staticmethod
    def _facing_attacks(node: SearchNode) -> bool:
        """True when the undefended cards on the table are the searcher's problem."""
        return not node.searcher_is_attacker and node.phase is Phase.DEFENSE

    def __repr__(self) -> str:
        return f"BaselineEvaluator({self.weights})"


class MechanisticEvaluator:
    """Phase 7.3.10 candidate: material measured by what actually persists.

    Derived from the frozen Phase 2 rules rather than from card-game
    intuition. Two mechanisms drive it.

    **M1 — refill erasure.** At bout resolution, while the talon still
    holds cards, ``_refill`` tops each hand back up to ``rules.hand_size``
    but never removes a card. A deficit below that size is therefore
    *transient* — the refill fills it back in — while a surplus above it
    is permanent until the cards are played. So while the talon lasts the
    persistent material quantity is ``max(0, n - refill_to)``, not ``n``.
    Measured over 4,629 talon-bearing positions from real play, 73.8% had
    at least one player above the refill size, so this term is live
    rather than a corner case.

    **M2 — terminal race.** Once the talon is empty no refill happens,
    ``_finish_or_start_bout`` declares out any player with an empty hand,
    and the remaining player is the durak. Every card must then be shed,
    so raw ``opponent_cards - own_cards`` becomes the right quantity.

    The switch at talon exhaustion is a genuine discontinuity in the
    rules — refill simply stops — not a smoothing parameter.

    Everything else is inherited from :class:`BaselineEvaluator`
    unchanged, **including all three coefficients**. The only difference
    is the structure of the material term, so any measured difference is
    attributable to the mechanism and not to retuning. No coefficient in
    this class was fitted to any dataset.

    Inputs come solely from :class:`SearchNode`, which carries the
    opponent as a *count* and the talon as a *size*, so the evaluator
    cannot read a hidden hand or hidden talon contents.
    """

    name = "mechanistic"

    def __init__(self, weights: EvaluationWeights | None = None) -> None:
        self.weights = weights or EvaluationWeights()

    def material(self, node: SearchNode) -> float:
        """Persistent material difference, per M1/M2 above."""
        if node.talon_size == 0:
            return float(node.opponent_cards - node.own_cards)
        refill = node.refill_to
        own_surplus = max(0, node.own_cards - refill)
        opponent_surplus = max(0, node.opponent_cards - refill)
        return float(opponent_surplus - own_surplus)

    def evaluate(self, node: SearchNode, ply: int = 0) -> Score:
        outcome = node.outcome()
        if outcome is not None:
            return terminal_score(outcome, ply)

        weights = self.weights
        score = weights.card_advantage * self.material(node)
        score += weights.own_trump * sum(
            1 for card in node.hand if card.suit == node.trump
        )
        if not node.searcher_is_attacker and node.phase is Phase.DEFENSE:
            open_attacks = sum(1 for slot in node.table if not slot.defended)
            score -= weights.open_obligation * open_attacks
        return score

    def __repr__(self) -> str:
        return f"MechanisticEvaluator({self.weights})"


@dataclass(frozen=True, slots=True)
class StructuralWeights:
    """Coefficients for the structural candidates, and where they come from.

    None of these was fitted to an outcome. Each is generated by one
    declared constant, ``AUXILIARY_SHARE``: a term's weight is set so that
    its spread over the development split is that share of the material
    term's spread. The rule is

        weight_i = AUXILIARY_SHARE * stdev(material) / stdev(feature_i)

    with development-split standard deviations material 3.7477,
    trump_quality 0.5422, obligation_pressure 0.1891, terminal_proximity
    0.1500, defense_flexibility 1.3094, throw_in_options 0.7773.

    ``AUXILIARY_SHARE = 0.10`` was chosen because the production baseline
    already sits near it: its trump term contributes 0.30 x 0.928 = 0.278
    of spread against material's 3.748, i.e. 7.4%. So this is the
    baseline's own balance stated explicitly, not a new claim about how
    much trumps matter. It is one free parameter, and Phase 7.3.10
    reports the candidate at 0.05, 0.10 and 0.20 rather than hiding it.

    Only the distribution of the features was consulted -- never a game
    result, an oracle value or a search outcome.
    """

    auxiliary_share: float = 0.10
    material: float = 1.0
    trump_quality: float = 0.69
    obligation_pressure: float = 1.98
    terminal_proximity: float = 2.50
    defense_flexibility: float = 0.29
    throw_in_options: float = 0.48

    def scaled(self, share: float) -> StructuralWeights:
        """The same rule at a different auxiliary share."""
        factor = share / self.auxiliary_share
        return StructuralWeights(
            auxiliary_share=share,
            material=self.material,
            trump_quality=self.trump_quality * factor,
            obligation_pressure=self.obligation_pressure * factor,
            terminal_proximity=self.terminal_proximity * factor,
            defense_flexibility=self.defense_flexibility * factor,
            throw_in_options=self.throw_in_options * factor,
        )

    def bound(self) -> float:
        """Loose bound on |score|, to prove a heuristic cannot look proven."""
        return (
            36 * abs(self.material)
            + 9 * abs(self.trump_quality)
            + 6 * abs(self.obligation_pressure)
            + 1 * abs(self.terminal_proximity)
            + 36 * abs(self.defense_flexibility)
            + 36 * abs(self.throw_in_options)
        )


class StructuralEvaluator(MechanisticEvaluator):
    """Phase 7.3.10 candidate E2: persistent material, restated auxiliaries.

    Three changes from :class:`MechanisticEvaluator`, each with a measured
    reason rather than an intuition:

    1. **Trump quality replaces trump count.** The two correlate at
       r = 0.848 on the development split -- overlapping, not redundant --
       so the ace and the six of trumps being worth the same under the
       baseline is a real loss of information, not a rounding.
    2. **The obligation term is ungated and normalised.** The baseline
       applies it only when ``_facing_attacks`` holds; Phase 7.3.9
       measured that guard as satisfied at 0 of 7,591 ISMCTS leaves,
       which makes the production term dead in that entire search path.
       Here the penalty applies wherever undefended slots exist and is
       divided by the hand that must answer them, because two open
       attacks against two cards is not the position two open attacks
       against eight cards is.
    3. **Terminal proximity is added**, active only once the talon is
       empty, where the game becomes a race to shed. It is bounded to
       +-1 before weighting and so cannot approach ``MATE_MARGIN``; it
       does not attempt to detect a decided position, which remains the
       exclusive job of ``terminal_score``.

    Antisymmetry: material and terminal proximity are difference
    quantities and are exactly antisymmetric; trump quality and
    obligation are own-side quantities and are not. The evaluator as a
    whole is therefore *not* antisymmetric, which is a deliberate
    inherited property -- see ``docs/evaluator-design.md``.

    Inputs come solely from :class:`SearchNode`, so no hidden card or
    hidden talon content can enter.
    """

    name = "structural"

    def __init__(
        self,
        weights: EvaluationWeights | None = None,
        structural: StructuralWeights | None = None,
    ) -> None:
        super().__init__(weights)
        self.structural = structural or StructuralWeights()

    def _trump_quality(self, node: SearchNode) -> float:
        return sum(
            (card.rank - 6) / 8.0 for card in node.hand if card.suit == node.trump
        )

    def _obligation_pressure(self, node: SearchNode) -> float:
        if not node.own_cards:
            return 0.0
        open_attacks = sum(1 for slot in node.table if not slot.defended)
        return open_attacks / node.own_cards

    def _terminal_proximity(self, node: SearchNode) -> float:
        if node.talon_size > 0:
            return 0.0
        return (1.0 / (1 + node.own_cards)) - (1.0 / (1 + node.opponent_cards))

    def evaluate(self, node: SearchNode, ply: int = 0) -> Score:
        outcome = node.outcome()
        if outcome is not None:
            return terminal_score(outcome, ply)

        structural = self.structural
        score = structural.material * self.material(node)
        score += structural.trump_quality * self._trump_quality(node)
        score -= structural.obligation_pressure * self._obligation_pressure(node)
        score += structural.terminal_proximity * self._terminal_proximity(node)
        return score

    def __repr__(self) -> str:
        return f"StructuralEvaluator({self.structural})"


class FlexibleEvaluator(StructuralEvaluator):
    """Phase 7.3.10 candidate E3: E2 plus attack and defence flexibility.

    Exists to answer one question the phase brief asks directly: do the
    flexibility quantities of sections 6C and 6D carry information the
    other features do not already have?

    The development split says they might. ``defense_flexibility``
    correlates with the Phase 5 legal-move count at only r = 0.098, so
    counting legal moves is *not* a proxy for defensive capability, and
    with the obligation count at r = 0.409. Both features are computed
    from the hand and the public table without consulting whose turn it
    is, deliberately: ``SearchNode.self_moves()`` returns empty when it
    is not the searcher's move, and Phase 7.3.9 measured that as the case
    at 7,295 of 7,591 ISMCTS leaves.

    If E3 does not beat E2, that is the answer to the question, and this
    class stays a registered experiment rather than becoming a proposal.
    """

    name = "flexible"

    def _defense_flexibility(self, node: SearchNode) -> float:
        open_cards = [
            slot.attacking_card
            for slot in node.table
            if not slot.defended and slot.attacking_card is not None
        ]
        if not open_cards:
            return 0.0
        return float(
            sum(
                1
                for card in node.hand
                if all(beats(card, attack, node.trump) for attack in open_cards)
            )
        )

    def _throw_in_options(self, node: SearchNode) -> float:
        return float(
            sum(1 for card in node.hand if card.rank in node.known_ranks)
        )

    def evaluate(self, node: SearchNode, ply: int = 0) -> Score:
        outcome = node.outcome()
        if outcome is not None:
            return terminal_score(outcome, ply)

        score = super().evaluate(node, ply)
        structural = self.structural
        if node.searcher_is_attacker:
            score += structural.throw_in_options * self._throw_in_options(node)
        else:
            score += structural.defense_flexibility * self._defense_flexibility(
                node
            )
        return score

    def __repr__(self) -> str:
        return f"FlexibleEvaluator({self.structural})"


def _at_share(base: type, share: float, label: str) -> type:
    """A candidate at a different auxiliary share, for the share sweep."""

    class _Shared(base):  # type: ignore[misc]
        name = label

        def __init__(self) -> None:
            super().__init__(structural=StructuralWeights().scaled(share))

    _Shared.__name__ = f"Evaluator_{label}"
    return _Shared


#: Evaluators addressable by name from a search configuration.
class RepairedThrowInEvaluator(FlexibleEvaluator):
    """Phase 7.3.12 R1: attack continuation measured as rank coverage.

    **The defect.** ``FlexibleEvaluator._throw_in_options`` counts cards
    in hand whose rank appears on the table. Playing such a card removes
    it from the hand, so the count falls and the score with it: the
    feature charges the searcher for exercising the very option it is
    supposed to value. Measured over 5,532 attacker shedding moves from
    real play, it opposes shedding in 14.5% of them.

    **What the feature should mean.** The rules permit an addition only
    at a rank already in play, so what carries attacking value is the
    *breadth* of ranks the searcher can still contribute to -- not how
    many duplicate cards are being hoarded against those ranks. Holding
    three sixes is not three times the continuation of holding one; it
    is one rank, available until the last six is gone.

    So the repaired quantity is the number of **distinct ranks in play
    that the hand can still match**:

        |{ r in known_ranks : some card in hand has rank r }|

    Playing one of two matching cards leaves this unchanged, because the
    rank is still coverable and the played card's rank stays on the
    table. Playing the last card of a rank does reduce it, which is a
    genuine loss of continuation rather than an accounting artifact.
    Range is bounded by the number of ranks on the table, so the term no
    longer grows with hand size.
    """

    name = "flex_r1"

    def _throw_in_options(self, node: SearchNode) -> float:
        ranks = {card.rank for card in node.hand}
        return float(len(ranks & set(node.known_ranks)))

    def __repr__(self) -> str:
        return f"RepairedThrowInEvaluator({self.structural})"


class RepairedObligationEvaluator(StructuralEvaluator):
    """Phase 7.3.12 R2: an obligation is only the defender's obligation.

    **The defect, and it is not the one Phase 7.3.11 named.** That phase
    reported the fault as the denominator: ``open_attacks / own_cards``
    rising when the hand shrinks. Reproducing it first, as Phase 7.3.12
    requires, showed that case occurs in **0** real positions -- playing
    a card always changes the open-attack count at the same time, so the
    pure denominator effect never arises within one ply.

    The real fault is the missing seat gate. ``BaselineEvaluator``
    applies its obligation penalty only through ``_facing_attacks``;
    Phase 7.3.10 removed that guard because Phase 7.3.9 had measured it
    as satisfied at 0 of 7,591 ISMCTS leaves. Removing it made the term
    count *every* undefended slot against the searcher -- including the
    attack the searcher has just made, which is the opponent's problem
    and not a liability at all.

    Measured on shedding moves from real play:

    =========  ======  ==================  ===================
    seat       n       mean term change    change is negative
    =========  ======  ==================  ===================
    attacker   5,532   -0.387              99.5%
    defender   1,648   +0.271              0.0%
    =========  ======  ==================  ===================

    Attacking is punished almost every time; defending is scored
    correctly. That is the whole defect, and it is larger than the
    throw-in one.

    **The repair.** Gate on the seat, and only the seat: an undefended
    attack is the searcher's obligation exactly when the searcher is the
    bout's defender. This deliberately keeps Phase 7.3.9's finding
    intact -- what was never satisfied at an ISMCTS leaf was the *phase*
    condition, not the seat condition, so gating on seat alone leaves the
    term reachable where the baseline's version is dead.

    Normalisation is left as the raw count here. Dividing by a quantity
    the player is trying to shrink was never well motivated, and with the
    seat gate in place there is nothing left for it to fix. R4 tests a
    bounded alternative.
    """

    name = "flex_r2"

    def _obligation_pressure(self, node: SearchNode) -> float:
        if node.searcher_is_attacker:
            return 0.0
        return float(sum(1 for slot in node.table if not slot.defended))

    def __repr__(self) -> str:
        return f"RepairedObligationEvaluator({self.structural})"


class RepairedEvaluator(RepairedThrowInEvaluator):
    """Phase 7.3.12 R3: both repairs together.

    Inherits the rank-coverage throw-in from R1 and takes the seat-gated
    obligation from R2 verbatim. No coefficient is changed: every weight
    is the Phase 7.3.10 auxiliary-share value, so any measured difference
    is attributable to the two semantic repairs and to nothing else.
    """

    name = "flex_r3"

    _obligation_pressure = RepairedObligationEvaluator._obligation_pressure

    def __repr__(self) -> str:
        return f"RepairedEvaluator({self.structural})"


class RepairedBoundedEvaluator(RepairedEvaluator):
    """Phase 7.3.12 R4: R3 with obligation normalised by table capacity.

    The one explicitly justified normalisation variant the phase brief
    asks for. R2 and R3 drop normalisation entirely; this asks whether a
    *bounded* obligation reads better than an absolute count.

    The divisor is ``attack_limit`` -- how many attacks this bout can
    hold -- which is public, fixed for the bout, and **independent of the
    searcher's hand**. That is the property the original lacked: it
    cannot be changed by playing a card, so it cannot create an incentive
    about shedding. The result lies in [0, 1], where 1 means every slot
    the bout allows is an unanswered attack on the searcher.
    """

    name = "flex_r4"

    def _obligation_pressure(self, node: SearchNode) -> float:
        if node.searcher_is_attacker or not node.attack_limit:
            return 0.0
        open_attacks = sum(1 for slot in node.table if not slot.defended)
        return open_attacks / node.attack_limit

    def __repr__(self) -> str:
        return f"RepairedBoundedEvaluator({self.structural})"


#:
#: Phase 7.3.10 candidates are registered so they can be selected and
#: validated by name. ``"baseline"`` remains every production default;
#: nothing here changes what an unconfigured search does.
EVALUATORS: dict[str, type] = {
    "baseline": BaselineEvaluator,
    "mechanistic": MechanisticEvaluator,
    "structural": StructuralEvaluator,
    "flexible": FlexibleEvaluator,
    # The auxiliary-share sweep. One declared constant generates every
    # coefficient, so the sweep has one dimension and three points --
    # reported in full rather than searched over silently.
    "structural_005": _at_share(StructuralEvaluator, 0.05, "structural_005"),
    "structural_020": _at_share(StructuralEvaluator, 0.20, "structural_020"),
    "flexible_005": _at_share(FlexibleEvaluator, 0.05, "flexible_005"),
    "flexible_020": _at_share(FlexibleEvaluator, 0.20, "flexible_020"),
    # Phase 7.3.12 incentive repairs. R0 is "flexible", above.
    "flex_r1": RepairedThrowInEvaluator,
    "flex_r2": RepairedObligationEvaluator,
    "flex_r3": RepairedEvaluator,
    "flex_r4": RepairedBoundedEvaluator,
}


def get_evaluator(name: str) -> Evaluator:
    """Look up an evaluator by name.

    Raises:
        DurakFishError: for an unknown name. Silently falling back to a
            default would let a typo in a configuration quietly change
            which engine was benchmarked.
    """
    try:
        factory = EVALUATORS[name]
    except KeyError:
        known = ", ".join(sorted(EVALUATORS))
        raise DurakFishError(
            f"unknown evaluator {name!r}; available: {known}"
        ) from None
    evaluator: Evaluator = factory()
    return evaluator


# Referenced for clarity in the node-kind checks above.
_ = (Actor, NodeKind)
