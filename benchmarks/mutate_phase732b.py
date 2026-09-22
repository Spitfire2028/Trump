"""Mutation audit for Phase 7.3.2b, run in a scratch copy of the tree.

Mutation specifications are plain Python data below. An earlier attempt
generated this file by writing Python that emitted Python containing
escaped Python string literals; the escaping went wrong three times and
produced a syntactically invalid runner. The lesson is in the shape of
this file: the specs are literals, the runner is boring, and nothing is
generated.

Isolation
---------
The working tree is never opened for writing. Each mutation is applied to
a fresh temporary copy of the repository, tested there, and the copy is
deleted. A kill at any moment therefore leaves the real sources
untouched — unlike a ``finally`` block, which does not run on ``SIGKILL``
and twice left a mutated source behind earlier in this project.

Verdicts
--------
A hang is not a detection, so timeouts are reported separately and never
counted as catches:

===========  ==================================================
``CAUGHT``   a test failed, and the failure names the fault
``MISSED``   the selection passed with the mutation applied
``TIMEOUT``  no verdict in time; needs investigation
``INVALID``  the anchor is absent, or the edit is a no-op
===========  ==================================================
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

REPO = pathlib.Path(__file__).resolve().parent.parent
AGG = pathlib.Path("src/durakfish/ai/multi_world.py")

TIMEOUT_SECONDS = 120

#: Fast but semantically complete: every property the audit relies on.
SELECTION = [
    f"tests/test_multi_world.py::{name}"
    for name in (
        "test_production_aggregation_matches_the_independent_reference",
        "test_the_reference_agrees_under_unequal_weights_too",
        "test_one_world_reduces_exactly_to_the_single_world_search",
        "test_a_certain_world_reduces_to_single_world_search",
        "test_aggregate_lies_between_the_best_and_worst_world_values",
        "test_scaling_every_weight_changes_nothing",
        "test_duplicating_every_world_changes_nothing",
        "test_dominance_is_respected",
        "test_per_world_values_follow_the_canonical_world_order",
        "test_permuting_the_worlds_cannot_change_the_result",
        "test_ties_break_on_the_search_ordering_earliest_first",
        "test_an_empty_world_set_is_refused",
        "test_zero_total_weight_is_refused",
        "test_the_root_player_must_be_the_player_to_move",
        "test_a_terminal_root_yields_no_move_and_a_terminal_value",
        "test_interleaving_worlds_cannot_contaminate_a_result",
        "test_the_same_worlds_from_two_realities_aggregate_identically",
        "test_restricting_the_root_moves_restricts_the_decision",
    )
]


@dataclass(frozen=True)
class Mutation:
    number: int
    label: str
    file: pathlib.Path
    pattern: str
    replacement: str


WEIGHT_SUM = "sum(entry.weight * value for entry, value in zip(ordered, values, strict=True))"
SORTED_WORLDS = "ordered = sorted(weighted, key=lambda e: (e.world.key(), e.weight))"
MOVE_CALL = "information, entry.world, move, player, settings"
SEARCH_CALL = "position, config, root_player=root_player, root_moves=(move,)"

MUTATIONS = [
    # --- weighting ----------------------------------------------------
    Mutation(1, "remove the weight multiplication", AGG,
             WEIGHT_SUM, "sum(value for value in values)"),
    Mutation(2, "use the wrong weight", AGG,
             "entry.weight * value", "(entry.weight + 1.0) * value"),
    Mutation(3, "divide by world count instead of total weight", AGG,
             "/ total_weight", "/ len(ordered)"),
    Mutation(4, "corrupt normalisation: multiply instead of divide", AGG,
             "/ total_weight", "* total_weight"),
    # --- world handling -----------------------------------------------
    Mutation(5, "skip one world", AGG,
             SORTED_WORLDS, SORTED_WORLDS + "[1:] or [weighted[0]]"),
    Mutation(6, "duplicate one world", AGG,
             "    total_weight = sum(entry.weight for entry in weighted)",
             "    weighted = weighted + weighted[:1]\n"
             "    total_weight = sum(entry.weight for entry in weighted)"),
    Mutation(7, "world ordering dependence: sum in caller order", AGG,
             SORTED_WORLDS, "ordered = list(weighted)"),
    Mutation(8, "reuse a stale world result", AGG,
             MOVE_CALL, "information, ordered[0].world, move, player, settings"),
    Mutation(9, "aggregate only the first world", AGG,
             "        for entry in ordered:", "        for entry in ordered[:1]:"),
    # --- move handling ------------------------------------------------
    Mutation(10, "evaluate the wrong root move", AGG,
             MOVE_CALL, "information, entry.world, candidates[0], player, settings"),
    Mutation(11, "skip a legal root move", AGG,
             "else probe.legal_moves()", "else probe.legal_moves()[1:]"),
    Mutation(12, "choose the minimum instead of the maximum", AGG,
             "if entry.value > best.value:", "if entry.value < best.value:"),
    Mutation(13, "corrupt tie-breaking: last tied move wins", AGG,
             "if entry.value > best.value:", "if entry.value >= best.value:"),
    Mutation(14, "misalign the reported move", AGG,
             "                move=move,", "                move=candidates[0],"),
    # --- perspective and search ---------------------------------------
    Mutation(15, "pass the wrong root player", AGG,
             MOVE_CALL, "information, entry.world, move, 1 - player, settings"),
    Mutation(16, "corrupt per-world value extraction", AGG,
             "return result.value, result.stats.nodes",
             "return 0.0, result.stats.nodes"),
    Mutation(17, "restart ply semantics by searching the child", AGG,
             SEARCH_CALL, "position.apply(move), config, root_player=root_player"),
    Mutation(18, "mishandle a terminal root", AGG,
             "    if probe.is_terminal:", "    if False:"),
    # --- contract guards ----------------------------------------------
    Mutation(19, "drop the root-player-to-move precondition", AGG,
             "    if probe.current_player != player:", "    if False:"),
    Mutation(20, "accept an empty world set", AGG,
             "    if not weighted:", "    if False and not weighted:"),
    Mutation(21, "accept a zero total weight", AGG,
             "    if total_weight <= 0:", "    if False:"),
]


def build_scratch() -> pathlib.Path:
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="durakfish-mutate-b-"))
    ignore = shutil.ignore_patterns(
        "__pycache__", ".pytest_cache", ".hypothesis", "*.pyc", ".git"
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
        target = scratch / mutation.file
        original = target.read_text()
        if mutation.pattern not in original:
            return "INVALID", "anchor text not found in the current source"
        mutated = original.replace(mutation.pattern, mutation.replacement)
        if mutated == original:
            return "INVALID", "replacement equals the original; a no-op"
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
            if line.startswith("E   ") or line.startswith("FAILED"):
                return "CAUGHT", line.strip()[:84]
        return "CAUGHT", f"pytest exit {result.returncode}"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> None:
    print("mutating scratch copies; the working tree is never written to")
    print(f"per-mutation timeout: {TIMEOUT_SECONDS}s\n")
    tally: dict[str, int] = {}
    for mutation in MUTATIONS:
        verdict, evidence = run_one(mutation)
        tally[verdict] = tally.get(verdict, 0) + 1
        print(f"  {verdict:<8} {mutation.number:>2}. {mutation.label}")
        print(f"           {evidence}")
    caught = tally.get("CAUGHT", 0)
    print(f"\ncaught {caught} of {len(MUTATIONS)}   {tally}")
    sys.exit(0 if caught == len(MUTATIONS) else 1)


if __name__ == "__main__":
    main()
