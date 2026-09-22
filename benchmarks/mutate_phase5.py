"""Mutation-test the Phase 5 audit.

An audit that cannot fail proves nothing. Each mutation below breaks one
property the suite claims to protect; the script applies it, runs the
relevant tests, restores the file byte-for-byte, and reports whether the
break was caught.

Run: ``python benchmarks/mutate_phase5.py``
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "durakfish" / "ai"

MUTATIONS = [
    (
        "action-class generation: drop TAKE from a defender's options",
        SRC / "searchnode.py",
        "            return (OpponentAction.BEAT, OpponentAction.TAKE)",
        "            return (OpponentAction.BEAT,)",
        "tests/test_abstraction_audit.py",
    ),
    (
        "action-class generation: assume a defender with cards can always beat",
        SRC / "searchnode.py",
        "            if self.opponent_cards <= 0:\n                return (OpponentAction.TAKE,)",
        "            if self.opponent_cards <= 0:\n                return (OpponentAction.BEAT,)",
        "tests/test_searchnode.py tests/test_abstraction_audit.py",
    ),
    (
        "terminal detection: claim outcomes while the talon still has cards",
        SRC / "searchnode.py",
        "        if self.kind is not NodeKind.BOUT_RESOLVED or self.talon_size:",
        "        if self.kind is not NodeKind.BOUT_RESOLVED:",
        "tests/test_abstraction_audit.py tests/test_searchnode.py",
    ),
    (
        "terminal detection: swap win and loss",
        SRC / "searchnode.py",
        "        if mine_out:\n            return Outcome.WIN",
        "        if mine_out:\n            return Outcome.LOSS",
        "tests/test_searchnode.py tests/test_abstraction_audit.py",
    ),
    (
        "maximising/minimising: make the opponent maximise too",
        SRC / "search.py",
        "    maximising = node.to_move is Actor.SELF",
        "    maximising = True",
        "tests/test_search.py",
    ),
    (
        "depth: forget to decrement, searching past the budget",
        SRC / "search.py",
        "        value = _minimax_value(child, depth - 1, ply + 1, evaluator, ordering, stats)",
        "        value = _minimax_value(child, depth, ply + 1, evaluator, ordering, stats)",
        "tests/test_search.py",
    ),
    (
        "alpha-beta cutoff: never cut off on either side",
        SRC / "search.py",
        "if beta <= alpha:",
        "if False:",
        "tests/test_search.py",
    ),
    (
        "alpha-beta cutoff: prune too aggressively",
        SRC / "search.py",
        "        beta = min(beta, best)\n        if beta <= alpha:",
        "        beta = min(beta, best)\n        if beta <= alpha + 1.0:",
        "tests/test_search.py",
    ),
    (
        "root move selection: let the last tied move win instead of the first",
        SRC / "search.py",
        "        if best_move is None or value > best_score:",
        "        if best_move is None or value >= best_score:",
        "tests/test_search.py",
    ),
    (
        "root move selection: ignore the supplied legal moves",
        SRC / "search.py",
        "    candidates = (\n        tuple(root_moves) if root_moves is not None else node.self_moves()\n    )",
        "    candidates = node.self_moves()",
        "tests/test_search.py tests/test_phase5_audit.py",
    ),
    (
        "conservative ranks: assume an unknown defence adds its rank",
        SRC / "searchnode.py",
        "                table=self._close_open_slot(None),\n                opponent_cards=self.opponent_cards - 1,",
        "                table=self._close_open_slot(None),\n                known_ranks=self.known_ranks | {Rank.ACE},\n                opponent_cards=self.opponent_cards - 1,",
        "tests/test_abstraction_audit.py tests/test_searchnode.py",
    ),
]

#: Mutations that are semantically unreachable, with the reason. Listed so
#: the record shows they were considered rather than quietly skipped.
UNREACHABLE = [
    (
        "hidden-state dependence inside ai/",
        "A search module cannot depend on hidden state even deliberately: "
        "nothing in ai/ can reach it. The only way to introduce the "
        "dependency is to break the Phase 3 redaction, which is mutated "
        "and caught in the Phase 3 audit instead.",
    ),
]


def run(tests: str) -> bool:
    """True when the test selection passes."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *tests.split(), "-q", "-x", "--no-header"],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src:.", "PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )
    return result.returncode == 0


def main() -> None:
    caught = missed = 0
    print(f"applying {len(MUTATIONS)} mutations\n")
    for label, path, old, new, tests in MUTATIONS:
        original = path.read_text()
        if old not in original:
            print(f"  SKIP    {label}\n          (anchor text not found - "
                  f"did {path.name} change?)")
            continue
        backup = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
        backup.write(original.encode())
        backup.close()
        try:
            path.write_text(original.replace(old, new))
            detected = not run(tests)
        finally:
            shutil.copyfile(backup.name, path)
            assert path.read_text() == original, "failed to restore the source"
        caught += detected
        missed += not detected
        print(f"  {'CAUGHT ' if detected else 'MISSED '} {label}")

    print(f"\ncaught {caught} of {caught + missed}")
    if UNREACHABLE:
        print("\nsemantically unreachable, documented rather than claimed:")
        for label, reason in UNREACHABLE:
            print(f"  - {label}\n      {reason}")
    sys.exit(1 if missed else 0)


if __name__ == "__main__":
    main()
