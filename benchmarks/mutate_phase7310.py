"""Mutation audit for the Phase 7.3.10 evaluator change.

Phase 7.3.10 modified exactly one production file,
``src/durakfish/ai/evaluation.py``, by adding a candidate evaluator and
registering it. Nothing else in production changed, so these mutations
attack that file: both the pre-existing baseline (which must stay pinned)
and the new candidate.

The eight mutation families the phase specification requires are all
represented, and the file marks which is which:

====================================  ==========
sign inversion                        M1, M8
feature removal                       M2, M9
feature sign reversal                 M3, M10
weight corruption                     M4, M11
terminal handling                     M5, M12
perspective reversal                  M6, M13
silent baseline fallback              M14, M15, M26
hidden-information access             M16
====================================  ==========

M18-M26 extend the same families to the structural candidates E2/E3 that
this phase adds (trump quality, ungated obligation pressure, terminal
proximity, the flexibility terms, and the auxiliary-share rule itself).

M17 is an addition of this phase's own: it promotes the candidate to the
production default, which the specification forbids doing silently. It is
here so that a future accidental promotion fails a test rather than
passing unnoticed.

Isolation follows the hardened workflow from Phase 7.3.2a, where an
external kill twice left a mutation behind in the working tree: the
repository is copied to a temporary directory, the mutation is applied
**only to the copy**, tests run there, and the copy is destroyed.
Production files are opened read-only and never written. To make that a
checked claim rather than an assurance, the runner records a SHA-256 of
every production source before the run and verifies it afterwards; a
mismatch is a hard failure.

A hang is not a detection. Timeouts are reported separately and are never
counted as catches.

Run: ``python benchmarks/mutate_phase7310.py``
"""

from __future__ import annotations

import hashlib
import pathlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

REPO = pathlib.Path(__file__).resolve().parent.parent
EVAL = pathlib.Path("src/durakfish/ai/evaluation.py")

TIMEOUT_SECONDS = 240

#: The selection must pin both evaluators. ``test_evaluator.py`` covers
#: the candidate and the shared contract; the rest were written in earlier
#: phases and pin the baseline, which this phase must not disturb.
SELECTION = [
    "tests/test_evaluator.py",
    "tests/test_searchnode.py",
    "tests/test_world_search.py::test_leaf_scoring_is_exactly_the_existing_baseline_evaluator",
    "tests/test_world_search.py::test_values_are_from_the_root_players_point_of_view",
    "tests/test_world_search.py::test_depth_zero_scores_the_position_where_it_stands",
    "tests/test_search.py::test_depth_one_scores_each_child_and_picks_the_best",
    "tests/test_search.py::test_a_sooner_win_outscores_a_later_one",
    "tests/test_ismcts.py::test_a_configured_evaluator_is_actually_consulted",
    "tests/test_ismcts.py::test_the_default_evaluator_path_is_behaviourally_unchanged",
]


SEARCHNODE = pathlib.Path("src/durakfish/ai/searchnode.py")
DETERMINIZED = pathlib.Path("src/durakfish/ai/determinized.py")


@dataclass(frozen=True)
class Edit:
    file: pathlib.Path
    pattern: str
    replacement: str


@dataclass(frozen=True)
class Mutation:
    number: int
    family: str
    target: str
    label: str
    pattern: str = ""
    replacement: str = ""
    file: pathlib.Path = EVAL
    extra: tuple[Edit, ...] = ()

    def edits(self) -> tuple[Edit, ...]:
        first = (Edit(self.file, self.pattern, self.replacement),) if self.pattern \
            else ()
        return first + self.extra


BASE_MATERIAL = (
    "        score = weights.card_advantage * (node.opponent_cards - node.own_cards)"
)
CAND_MATERIAL = "        score = weights.card_advantage * self.material(node)"
BASE_TRUMP = (
    "        score += weights.own_trump * sum(\n"
    "            1 for card in node.hand if card.suit == node.trump\n"
    "        )\n"
    "        if self._facing_attacks(node):"
)
CAND_TRUMP = (
    "        score += weights.own_trump * sum(\n"
    "            1 for card in node.hand if card.suit == node.trump\n"
    "        )\n"
    "        if not node.searcher_is_attacker and node.phase is Phase.DEFENSE:"
)

MUTATIONS = [
    # -------------------------------------------------- baseline, pinned
    Mutation(1, "sign inversion", "baseline",
             "negate the whole baseline score",
             "        if self._facing_attacks(node):\n"
             "            open_attacks = sum(1 for slot in node.table if not slot.defended)\n"  # noqa: E501 (verbatim source anchor)
             "            score -= weights.open_obligation * open_attacks\n"
             "        return score\n"
             "\n"
             "    @staticmethod\n"
             "    def _facing_attacks",
             "        if self._facing_attacks(node):\n"
             "            open_attacks = sum(1 for slot in node.table if not slot.defended)\n"  # noqa: E501 (verbatim source anchor)
             "            score -= weights.open_obligation * open_attacks\n"
             "        return -score\n"
             "\n"
             "    @staticmethod\n"
             "    def _facing_attacks"),
    Mutation(2, "feature removal", "baseline",
             "drop the trump term from the baseline",
             BASE_TRUMP,
             "        score += 0.0 * sum(\n"
             "            1 for card in node.hand if card.suit == node.trump\n"
             "        )\n"
             "        if self._facing_attacks(node):"),
    Mutation(3, "feature sign reversal", "baseline",
             "treat an open obligation as an asset",
             "            score -= weights.open_obligation * open_attacks\n"
             "        return score\n"
             "\n"
             "    @staticmethod",
             "            score += weights.open_obligation * open_attacks\n"
             "        return score\n"
             "\n"
             "    @staticmethod"),
    Mutation(4, "weight corruption", "baseline",
             "corrupt the dominant coefficient at its source",
             "    card_advantage: float = 1.0",
             "    card_advantage: float = 0.25"),
    Mutation(5, "terminal handling", "baseline",
             "heuristic a proven terminal instead of scoring it exactly",
             "    def evaluate(self, node: SearchNode, ply: int = 0) -> Score:\n"
             "        outcome = node.outcome()\n"
             "        if outcome is not None:\n"
             "            return terminal_score(outcome, ply)\n"
             "\n"
             "        weights = self.weights\n"
             "        score = weights.card_advantage * (node.opponent_cards",
             "    def evaluate(self, node: SearchNode, ply: int = 0) -> Score:\n"
             "        outcome = node.outcome()\n"
             "        if outcome is not None and ply == 0:\n"
             "            return terminal_score(outcome, ply)\n"
             "\n"
             "        weights = self.weights\n"
             "        score = weights.card_advantage * (node.opponent_cards"),
    Mutation(6, "perspective reversal", "baseline",
             "score material from the opponent's point of view",
             BASE_MATERIAL,
             "        score = weights.card_advantage * (node.own_cards - node.opponent_cards)"),  # noqa: E501 (verbatim source anchor)
    Mutation(7, "terminal handling", "baseline",
             "drop the ply discount so a later win ties an immediate one",
             "    if outcome is Outcome.WIN:\n        return TERMINAL_WIN - ply",
             "    if outcome is Outcome.WIN:\n        return TERMINAL_WIN"),
    # ------------------------------------------------ candidate, 7.3.10
    Mutation(8, "sign inversion", "candidate",
             "negate the candidate's material term",
             "        return float(opponent_surplus - own_surplus)",
             "        return float(own_surplus - opponent_surplus)"),
    Mutation(9, "feature removal", "candidate",
             "collapse the mechanism: always use raw material",
             "        if node.talon_size == 0:\n"
             "            return float(node.opponent_cards - node.own_cards)\n"
             "        refill = node.refill_to",
             "        if True:\n"
             "            return float(node.opponent_cards - node.own_cards)\n"
             "        refill = node.refill_to"),
    Mutation(10, "feature sign reversal", "candidate",
             "reverse the obligation penalty in the candidate",
             "            score -= weights.open_obligation * open_attacks\n"
             "        return score\n"
             "\n"
             "    def __repr__(self) -> str:\n"
             '        return f"MechanisticEvaluator',
             "            score += weights.open_obligation * open_attacks\n"
             "        return score\n"
             "\n"
             "    def __repr__(self) -> str:\n"
             '        return f"MechanisticEvaluator'),
    Mutation(11, "weight corruption", "candidate",
             "ignore the supplied weights and hardcode the defaults",
             '    name = "mechanistic"\n'
             "\n"
             "    def __init__(self, weights: EvaluationWeights | None = None) -> None:\n"  # noqa: E501 (verbatim source anchor)
             "        self.weights = weights or EvaluationWeights()",
             '    name = "mechanistic"\n'
             "\n"
             "    def __init__(self, weights: EvaluationWeights | None = None) -> None:\n"  # noqa: E501 (verbatim source anchor)
             "        self.weights = EvaluationWeights()"),
    Mutation(12, "terminal handling", "candidate",
             "approximate terminals in the candidate",
             "    def evaluate(self, node: SearchNode, ply: int = 0) -> Score:\n"
             "        outcome = node.outcome()\n"
             "        if outcome is not None:\n"
             "            return terminal_score(outcome, ply)\n"
             "\n"
             "        weights = self.weights\n"
             "        score = weights.card_advantage * self.material(node)",
             "    def evaluate(self, node: SearchNode, ply: int = 0) -> Score:\n"
             "        outcome = node.outcome()\n"
             "        if outcome is not None:\n"
             "            return 500.0 if outcome.name == 'WIN' else -500.0\n"
             "\n"
             "        weights = self.weights\n"
             "        score = weights.card_advantage * self.material(node)"),
    Mutation(13, "perspective reversal", "candidate",
             "swap own and opponent surplus inputs",
             "        own_surplus = max(0, node.own_cards - refill)\n"
             "        opponent_surplus = max(0, node.opponent_cards - refill)",
             "        own_surplus = max(0, node.opponent_cards - refill)\n"
             "        opponent_surplus = max(0, node.own_cards - refill)"),
    # ------------------------------------------- registry and boundaries
    Mutation(14, "silent baseline fallback", "registry",
             "the candidate name silently resolves to the baseline",
             '    "mechanistic": MechanisticEvaluator,',
             '    "mechanistic": BaselineEvaluator,'),
    Mutation(15, "silent baseline fallback", "registry",
             "an unknown evaluator name falls back instead of failing",
             "    try:\n"
             "        factory = EVALUATORS[name]\n"
             "    except KeyError:",
             "    try:\n"
             "        factory = EVALUATORS[name]\n"
             "    except KeyError:\n"
             "        evaluator_: Evaluator = EVALUATORS['baseline']()\n"
             "        return evaluator_\n"
             "    if False:"),
    # The candidate cannot reach hidden cards by construction: a
    # ``SearchNode`` simply has no field to hold them. A mutation confined
    # to ``evaluation.py`` therefore cannot express the leak, and a
    # mutation that cannot express the fault proves nothing. M16 opens the
    # channel first -- it adds a talon-contents field to the node and fills
    # it at the materialisation boundary, which is the only place that can
    # -- and then has the candidate read it. If the structural audits are
    # alive, this dies; if they only ever asserted about a shape that could
    # not vary, it survives. The frozen files are edited on the scratch
    # copy exclusively; production is verified byte-identical afterwards.
    Mutation(16, "hidden-information access", "candidate",
             "leak talon contents into the node and evaluate on them",
             "        if node.talon_size == 0:\n"
             "            return float(node.opponent_cards - node.own_cards)",
             "        if getattr(node, 'talon', ()):\n"
             "            return float(len({c.suit for c in node.talon}))\n"
             "        if node.talon_size == 0:\n"
             "            return float(node.opponent_cards - node.own_cards)",
             EVAL,
             (
                 Edit(SEARCHNODE,
                      "    kind: NodeKind = NodeKind.DECISION\n"
                      "    unknown_own: int = 0",
                      "    kind: NodeKind = NodeKind.DECISION\n"
                      "    unknown_own: int = 0\n"
                      "    talon: tuple[Card, ...] = ()"),
                 Edit(DETERMINIZED,
                      "            talon_size=len(state.talon),",
                      "            talon=tuple(state.talon),\n"
                      "            talon_size=len(state.talon),"),
             )),
    # ---------------------------------- structural candidates (E2/E3)
    Mutation(18, "feature sign reversal", "structural",
             "reward open obligations instead of penalising them",
             "        score -= structural.obligation_pressure * self._obligation_pressure(node)",  # noqa: E501 (verbatim source anchor)
             "        score += structural.obligation_pressure * self._obligation_pressure(node)"),  # noqa: E501 (verbatim source anchor)
    Mutation(19, "feature removal", "structural",
             "drop trump quality, collapsing the ace and the six",
             "        score += structural.trump_quality * self._trump_quality(node)",
             "        score += 0.0 * self._trump_quality(node)"),
    Mutation(20, "weight corruption", "structural",
             "break the auxiliary-share rule for one term only",
             "    terminal_proximity: float = 2.50",
             "    terminal_proximity: float = 7.50"),
    Mutation(21, "sign inversion", "structural",
             "invert trump quality so a high trump becomes a liability",
             "            (card.rank - 6) / 8.0 for card in node.hand if card.suit == node.trump",  # noqa: E501 (verbatim source anchor)
             "            (6 - card.rank) / 8.0 for card in node.hand if card.suit == node.trump"),  # noqa: E501 (verbatim source anchor)
    Mutation(22, "perspective reversal", "structural",
             "reverse terminal proximity's perspective",
             "        return (1.0 / (1 + node.own_cards)) - (1.0 / (1 + node.opponent_cards))",  # noqa: E501 (verbatim source anchor)
             "        return (1.0 / (1 + node.opponent_cards)) - (1.0 / (1 + node.own_cards))"),  # noqa: E501 (verbatim source anchor)
    Mutation(23, "terminal handling", "structural",
             "let terminal proximity run while the talon still refills",
             "    def _terminal_proximity(self, node: SearchNode) -> float:\n"
             "        if node.talon_size > 0:\n"
             "            return 0.0",
             "    def _terminal_proximity(self, node: SearchNode) -> float:\n"
             "        if False:\n"
             "            return 0.0"),
    Mutation(24, "feature removal", "flexible",
             "make the flexibility terms inert",
             "        if node.searcher_is_attacker:\n"
             "            score += structural.throw_in_options * self._throw_in_options(node)",  # noqa: E501 (verbatim source anchor)
             "        if False:\n"
             "            score += structural.throw_in_options * self._throw_in_options(node)"),  # noqa: E501 (verbatim source anchor)
    Mutation(25, "weight corruption", "structural",
             "make the share sweep a no-op, hiding the free parameter",
             "        factor = share / self.auxiliary_share",
             "        factor = 1.0"),
    Mutation(26, "silent baseline fallback", "registry",
             "the flexible name silently resolves to the structural form",
             '    "flexible": FlexibleEvaluator,',
             '    "flexible": StructuralEvaluator,'),
    Mutation(17, "default change", "registry",
             "promote the candidate to the production default",
             '    "baseline": BaselineEvaluator,',
             '    "baseline": MechanisticEvaluator,'),
]


def source_hashes() -> dict[str, str]:
    """SHA-256 of every production source, to prove nothing was written."""
    return {
        str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((REPO / "src").rglob("*.py"))
    }


def build_scratch() -> pathlib.Path:
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="durakfish-7310-"))
    ignore = shutil.ignore_patterns(
        "__pycache__", ".pytest_cache", ".hypothesis", "*.pyc", ".git", "data"
    )
    for item in ("src", "tests", "pyproject.toml"):
        source, target = REPO / item, scratch / item
        if source.is_dir():
            shutil.copytree(source, target, ignore=ignore)
        else:
            shutil.copyfile(source, target)
    return scratch


def run_one(mutation: Mutation) -> tuple[str, str]:
    scratch = build_scratch()
    try:
        for edit in mutation.edits():
            target = scratch / edit.file
            original = target.read_text()
            if original.count(edit.pattern) != 1:
                return "INVALID", (
                    f"{edit.file.name}: anchor matched "
                    f"{original.count(edit.pattern)} times; it must match once"
                )
            mutated = original.replace(edit.pattern, edit.replacement)
            if mutated == original:
                return "INVALID", f"{edit.file.name}: replacement is a no-op"
            target.write_text(mutated)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", *SELECTION, "-q", "-x",
                 "--no-header", "-p", "no:cacheprovider"],
                cwd=str(scratch), capture_output=True, text=True,
                timeout=TIMEOUT_SECONDS,
                env={"PYTHONPATH": "src:.", "PATH": "/usr/bin:/bin", "HOME": "/tmp"},
            )
        except subprocess.TimeoutExpired:
            return "TIMEOUT", f"no verdict within {TIMEOUT_SECONDS}s"
        if result.returncode == 0:
            return "MISSED", "the selection passed with the mutation applied"
        for line in result.stdout.splitlines():
            if line.startswith(("E   ", "FAILED")):
                return "CAUGHT", line.strip()[:88]
        return "CAUGHT", f"pytest exit {result.returncode}"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> None:
    before = source_hashes()
    print("Phase 7.3.10 mutation audit -- src/durakfish/ai/evaluation.py")
    print("mutating scratch copies; production files are never written to")
    print(f"{len(before)} production sources hashed; timeout {TIMEOUT_SECONDS}s\n")

    tally: dict[str, int] = {}
    by_family: dict[str, list[str]] = {}
    for mutation in MUTATIONS:
        verdict, evidence = run_one(mutation)
        tally[verdict] = tally.get(verdict, 0) + 1
        by_family.setdefault(mutation.family, []).append(verdict)
        print(f"  {verdict:<8} M{mutation.number:<3} [{mutation.target}] "
              f"{mutation.label}")
        print(f"           {evidence}")

    print("\nby mutation family:")
    for family, verdicts in sorted(by_family.items()):
        caught = sum(1 for v in verdicts if v == "CAUGHT")
        print(f"  {family:<26} {caught}/{len(verdicts)} caught")

    after = source_hashes()
    drifted = [path for path in before if before[path] != after.get(path)]
    print(f"\nproduction source integrity: "
          f"{'INTACT' if not drifted else 'VIOLATED ' + str(drifted)}")

    caught = tally.get("CAUGHT", 0)
    print(f"caught {caught} of {len(MUTATIONS)}   {tally}")
    sys.exit(0 if caught == len(MUTATIONS) and not drifted else 1)


if __name__ == "__main__":
    main()
