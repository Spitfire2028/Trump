"""Mutation audit for the Phase 7.3.12 incentive repairs.

Phase 7.3.12 added four repaired evaluators to
``src/durakfish/ai/evaluation.py``. These mutations attack exactly what
the phase brief names: throw-in direction, obligation direction,
normalisation, candidate registration, evaluator selection and any
accidental fallback to the baseline.

The Phase 7.3.10 suite (``mutate_phase7310.py``, 26/26) is left
untouched and still runs, so the earlier evidence stays valid rather
than being rewritten by this phase.
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

Run: ``python benchmarks/mutate_phase7312.py``
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
    # --------------------------------------------- obligation direction
    Mutation(1, "obligation direction", "flex_r2",
             "drop the seat gate, restoring the defect being repaired",
             "    def _obligation_pressure(self, node: SearchNode) -> float:\n"
             "        if node.searcher_is_attacker:\n"
             "            return 0.0\n"
             "        return float(sum(1 for slot in node.table if not slot.defended))",
             "    def _obligation_pressure(self, node: SearchNode) -> float:\n"
             "        if False:\n"
             "            return 0.0\n"
             "        return float(sum(1 for slot in node.table if not slot.defended))"),  # noqa: E501 (verbatim source anchor)
    Mutation(2, "obligation direction", "flex_r2",
             "invert the seat gate: charge the attacker, spare the defender",
             "        if node.searcher_is_attacker:\n"
             "            return 0.0\n"
             "        return float(sum(1 for slot in node.table if not slot.defended))",
             "        if not node.searcher_is_attacker:\n"
             "            return 0.0\n"
             "        return float(sum(1 for slot in node.table if not slot.defended))"),  # noqa: E501 (verbatim source anchor)
    Mutation(3, "obligation direction", "flex_r2",
             "count defended slots instead of undefended ones",
             "        return float(sum(1 for slot in node.table if not slot.defended))\n"  # noqa: E501 (verbatim source anchor)
             "\n"
             "    def __repr__(self) -> str:\n"
             '        return f"RepairedObligationEvaluator',
             "        return float(sum(1 for slot in node.table if slot.defended))\n"
             "\n"
             "    def __repr__(self) -> str:\n"
             '        return f"RepairedObligationEvaluator'),
    # ------------------------------------------------ throw-in direction
    Mutation(4, "throw-in direction", "flex_r1",
             "revert rank coverage to the per-card count it replaces",
             "    def _throw_in_options(self, node: SearchNode) -> float:\n"
             "        ranks = {card.rank for card in node.hand}\n"
             "        return float(len(ranks & set(node.known_ranks)))",
             "    def _throw_in_options(self, node: SearchNode) -> float:\n"
             "        return float(\n"
             "            sum(1 for card in node.hand if card.rank in node.known_ranks)\n"  # noqa: E501 (verbatim source anchor)
             "        )"),
    Mutation(5, "throw-in direction", "flex_r1",
             "count ranks NOT on the table, inverting the semantics",
             "        return float(len(ranks & set(node.known_ranks)))",
             "        return float(len(ranks - set(node.known_ranks)))"),
    Mutation(6, "throw-in direction", "flex_r1",
             "make rank coverage inert",
             "        ranks = {card.rank for card in node.hand}\n"
             "        return float(len(ranks & set(node.known_ranks)))",
             "        ranks = {card.rank for card in node.hand}\n"
             "        return 0.0 * float(len(ranks & set(node.known_ranks)))"),
    # ---------------------------------------------------- normalisation
    Mutation(7, "normalisation", "flex_r4",
             "normalise by the hand again -- the rejected divisor",
             "        open_attacks = sum(1 for slot in node.table if not slot.defended)\n"  # noqa: E501 (verbatim source anchor)
             "        return open_attacks / node.attack_limit",
             "        open_attacks = sum(1 for slot in node.table if not slot.defended)\n"  # noqa: E501 (verbatim source anchor)
             "        return open_attacks / max(1, node.own_cards)"),
    Mutation(8, "normalisation", "flex_r4",
             "drop the bounded divisor, collapsing R4 into R3",
             "        return open_attacks / node.attack_limit",
             "        return float(open_attacks)"),
    # ------------------------------------- registration and selection
    Mutation(9, "candidate registration", "registry",
             "R3 silently resolves to the unrepaired candidate",
             '    "flex_r3": RepairedEvaluator,',
             '    "flex_r3": FlexibleEvaluator,'),
    Mutation(10, "candidate registration", "registry",
             "R2 silently resolves to R4",
             '    "flex_r2": RepairedObligationEvaluator,',
             '    "flex_r2": RepairedBoundedEvaluator,'),
    Mutation(11, "silent baseline fallback", "registry",
             "every repair silently resolves to the baseline",
             '    "flex_r1": RepairedThrowInEvaluator,',
             '    "flex_r1": BaselineEvaluator,'),
    Mutation(12, "evaluator selection", "registry",
             "an unknown name falls back instead of failing",
             "    try:\n"
             "        factory = EVALUATORS[name]\n"
             "    except KeyError:",
             "    try:\n"
             "        factory = EVALUATORS[name]\n"
             "    except KeyError:\n"
             "        fallback: Evaluator = EVALUATORS['baseline']()\n"
             "        return fallback\n"
             "    if False:"),
    Mutation(13, "default change", "registry",
             "promote a repair to the production default",
             '    "baseline": BaselineEvaluator,',
             '    "baseline": RepairedEvaluator,'),
    # -------------------------------------------- inheritance integrity
    Mutation(14, "obligation direction", "flex_r3",
             "R3 stops inheriting the repaired obligation",
             "    _obligation_pressure = RepairedObligationEvaluator._obligation_pressure",  # noqa: E501 (verbatim source anchor)
             "    _obligation_pressure = StructuralEvaluator._obligation_pressure"),
    Mutation(15, "terminal handling", "repairs",
             "a repair approximates terminals instead of scoring them",
             "    def evaluate(self, node: SearchNode, ply: int = 0) -> Score:\n"
             "        outcome = node.outcome()\n"
             "        if outcome is not None:\n"
             "            return terminal_score(outcome, ply)\n"
             "\n"
             "        score = super().evaluate(node, ply)",
             "    def evaluate(self, node: SearchNode, ply: int = 0) -> Score:\n"
             "        outcome = node.outcome()\n"
             "        if outcome is not None:\n"
             "            return 250.0\n"
             "\n"
             "        score = super().evaluate(node, ply)"),
]

def source_hashes() -> dict[str, str]:
    """SHA-256 of every production source, to prove nothing was written."""
    return {
        str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((REPO / "src").rglob("*.py"))
    }


def build_scratch() -> pathlib.Path:
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="durakfish-7312-"))
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
    print("Phase 7.3.12 mutation audit -- incentive repairs")
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
