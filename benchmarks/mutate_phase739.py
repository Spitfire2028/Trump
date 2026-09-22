"""Mutation audit for the Phase 7.3.9 evaluator-injection boundary.

Scope is deliberately narrow: Phase 7.3.9 changed exactly one thing in
production — ``ISMCTS`` now resolves and uses the evaluator named by its
configuration, where before it ignored the field and hardcoded the
baseline. These mutations attack that boundary.

Isolation follows the hardened workflow established in Phase 7.3.2a: the
repository is copied to a temporary directory, the mutation is applied
there, tests run there, and the copy is destroyed. The working tree is
never opened for writing, so a kill leaves it untouched — unlike a
``finally`` block, which does not run on ``SIGKILL`` and twice corrupted
the tree earlier in this project.

A hang is not a detection, so timeouts are reported separately and never
counted as catches.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass

REPO = pathlib.Path(__file__).resolve().parent.parent
SEARCH = pathlib.Path("src/durakfish/ai/ismcts/search.py")

TIMEOUT_SECONDS = 180

SELECTION = [
    f"tests/test_ismcts.py::{name}"
    for name in (
        "test_the_default_evaluator_path_is_behaviourally_unchanged",
        "test_a_configured_evaluator_is_actually_consulted",
        "test_an_unknown_evaluator_name_fails_when_the_search_is_built",
        "test_the_search_backs_up_the_observers_perspective",
        "test_terminal_values_carry_the_ply_discount",
        "test_the_search_is_reproducible_under_a_seed",
        "test_the_search_returns_a_legal_root_move",
        "test_indistinguishable_worlds_reach_one_node_with_shared_statistics",
        "test_hidden_permutation_cannot_change_the_key",
        "test_the_ismcts_package_never_names_the_state_layer",
    )
]


@dataclass(frozen=True)
class Mutation:
    number: int
    label: str
    file: pathlib.Path
    pattern: str
    replacement: str


MUTATIONS = [
    Mutation(1, "evaluator bypass: ignore the configuration entirely", SEARCH,
             "        return evaluate_position(\n"
             "            position, observer, ply=ply, evaluator=self._evaluator\n"
             "        )",
             "        return evaluate_position(position, observer, ply=ply)"),
    Mutation(2, "silent baseline fallback regardless of configuration", SEARCH,
             '        self._evaluator = get_evaluator(self._config.evaluator)',
             '        self._evaluator = get_evaluator("baseline")'),
    Mutation(3, "wrong root perspective at the leaf", SEARCH,
             "            position, observer, ply=ply, evaluator=self._evaluator",
             "            position, 1 - observer, ply=ply, evaluator=self._evaluator"),
    Mutation(4, "drop the ply, flattening terminal-distance preference", SEARCH,
             "            position, observer, ply=ply, evaluator=self._evaluator",
             "            position, observer, ply=0, evaluator=self._evaluator"),
    Mutation(5, "accept an unknown evaluator name silently", SEARCH,
             '        self._evaluator = get_evaluator(self._config.evaluator)',
             '        try:\n'
             '            self._evaluator = get_evaluator(self._config.evaluator)\n'
             '        except Exception:\n'
             '            self._evaluator = get_evaluator("baseline")'),
    Mutation(6, "leak the hidden world into node identity", SEARCH,
             "                key = information_key(position, epistemic)",
             "                key = information_key(position, epistemic)\n"
             "                key = type(key)(key.owner,\n"
             "                    key.view_key + (len(position.state.talon),))"),
]


def build_scratch() -> pathlib.Path:
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="durakfish-739-"))
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
        print(f"  {verdict:<8} {mutation.number}. {mutation.label}")
        print(f"           {evidence}")
    caught = tally.get("CAUGHT", 0)
    print(f"\ncaught {caught} of {len(MUTATIONS)}   {tally}")
    sys.exit(0 if caught == len(MUTATIONS) else 1)


if __name__ == "__main__":
    main()
