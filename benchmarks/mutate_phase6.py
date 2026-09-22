"""Mutation-test the Phase 6 knowledge layer.

An audit that cannot fail proves nothing. Each mutation below breaks one
property the suite claims to protect; the script applies it, runs the
relevant tests, restores the file byte-for-byte, and reports whether the
break was caught.

Run: ``python benchmarks/mutate_phase6.py``
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "durakfish" / "information"

MUTATIONS = [
    (
        "observation: stop recording the discard pile",
        SRC / "deduction.py",
        "    for card in observation.discard:\n        place(card, CardLocation.DISCARD, Certainty.KNOWN, RULE_OBSERVED)",
        "    pass",
        "tests/test_knowledge.py",
    ),
    (
        "deduction: forget that the opponent retains what it took",
        SRC / "deduction.py",
        "    for card in observation.retained_by_opponent:",
        "    for card in frozenset():",
        "tests/test_knowledge.py",
    ),
    (
        "deduction: claim the trump card is in the talon even when empty",
        SRC / "deduction.py",
        "    if observation.talon_size > 0 and observation.trump_card not in settled:",
        "    if observation.trump_card not in settled:",
        "tests/test_knowledge.py tests/test_phase6_audit.py",
    ),
    (
        "card elimination: skip the talon-exhausted rule entirely",
        SRC / "deduction.py",
        "    if talon_slots == 0 and candidates:",
        "    if False and candidates:",
        "tests/test_knowledge.py",
    ),
    (
        "hand-size constraint: ignore cards already placed in the opponent's hand",
        SRC / "deduction.py",
        "    opponent_slots = observation.opponent_hand_size - sum(\n        1 for p in settled.values() if p.location is CardLocation.OPPONENT_HAND\n    )",
        "    opponent_slots = observation.opponent_hand_size",
        "tests/test_knowledge.py",
    ),
    (
        "talon constraint: ignore the trump card occupying a talon slot",
        SRC / "deduction.py",
        "    talon_slots = observation.talon_size - sum(\n        1 for p in settled.values() if p.location is CardLocation.TALON\n    )",
        "    talon_slots = observation.talon_size",
        "tests/test_knowledge.py",
    ),
    (
        "known vs deduced: label a deduction as a direct observation",
        SRC / "deduction.py",
        "                Certainty.DEDUCED,\n                RULE_OPPONENT_RETAINS,",
        "                Certainty.KNOWN,\n                RULE_OPPONENT_RETAINS,",
        "tests/test_knowledge.py tests/test_phase6_audit.py",
    ),
    (
        "possible vs impossible: call every hidden placement possible",
        SRC / "knowledge.py",
        "        if location in self.possible_locations(card):\n            return Certainty.POSSIBLE\n        return Certainty.IMPOSSIBLE",
        "        return Certainty.POSSIBLE",
        "tests/test_knowledge.py",
    ),
    (
        "probability: drop normalisation by treating slots as independent",
        SRC / "belief.py",
        "            return self.knowledge.opponent_slots / total",
        "            return min(1.0, self.knowledge.opponent_slots / max(total - 1, 1))",
        "tests/test_knowledge.py",
    ),
    (
        "probability: multiply marginals as if cards were independent",
        SRC / "belief.py",
        "            probability *= slots / remaining\n            slots -= 1\n            remaining -= 1",
        "            probability *= slots / remaining",
        "tests/test_knowledge.py",
    ),
    (
        "incrementality: never drop a card the opponent played back out",
        SRC / "tracker.py",
        "        if event.player == opponent:\n            self._retained.difference_update(event.revealed)",
        "        pass",
        "tests/test_knowledge.py tests/test_phase6_audit.py",
    ),
    (
        "hidden-state leakage: let the tracker keep another player's view",
        SRC / "tracker.py",
        "        if view.player != self._player:",
        "        if False:",
        "tests/test_knowledge.py",
    ),
]

#: Considered and documented rather than claimed as tested.
UNREACHABLE = [
    (
        "reading real hidden state inside the knowledge layer",
        "No module downstream of observe() can reach a GameState: the "
        "tracker's only inputs are an InformationSet and its own record, "
        "both already redacted. Introducing the dependency requires "
        "breaking Phase 3 redaction, which is mutated and caught there.",
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
