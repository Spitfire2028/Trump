"""Mutation-test the Phase 7.2 determinized world model.

An audit that cannot fail proves nothing. Each mutation below breaks one
property the suite claims to protect; the script applies it, runs the
relevant tests, restores the file byte-for-byte, and reports whether the
break was caught.

Run: ``python benchmarks/mutate_phase72.py``
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "durakfish" / "information"

WORLDS = SRC / "worlds.py"
TESTS = "tests/test_worlds.py tests/test_worlds_properties.py"

MUTATIONS = [
    # --- card conservation ---------------------------------------------
    (
        "conservation: drop a card from the generated opponent hand",
        WORLDS,
        "        opponent = self._certain_opponent | frozenset(chosen)",
        "        opponent = self._certain_opponent | frozenset(chosen[1:])",
        TESTS,
    ),
    (
        "conservation: duplicate a talon card",
        WORLDS,
        "            talon = tuple(body) + (self._trump_card,)",
        "            talon = tuple(body) + tuple(body[:1]) + (self._trump_card,)",
        TESTS,
    ),
    (
        "conservation: validator stops checking for phantom/missing cards",
        WORLDS,
        "    placed = public | opponent | talon\n    deck = knowledge.deck\n    if placed != deck:",
        "    placed = public | opponent | talon\n    deck = knowledge.deck\n    if False:",
        TESTS,
    ),
    # --- ownership ------------------------------------------------------
    (
        "ownership: generator ignores cards known to be the opponent's",
        WORLDS,
        "        opponent = self._certain_opponent | frozenset(chosen)",
        "        opponent = frozenset(chosen)",
        TESTS,
    ),
    (
        "ownership: validator stops enforcing known opponent cards",
        WORLDS,
        "    if not certain_opponent <= opponent:",
        "    if False:",
        TESTS,
    ),
    (
        "ownership: validator stops enforcing forced talon cards",
        WORLDS,
        "    if not certain_talon <= talon:",
        "    if False:",
        TESTS,
    ),
    # --- sizes ----------------------------------------------------------
    (
        "size: generator fills one opponent slot too many",
        WORLDS,
        "        chosen = rng.sample(self._candidates, self._opponent_slots)",
        "        chosen = rng.sample(self._candidates, min(len(self._candidates), self._opponent_slots + 1))",
        TESTS,
    ),
    (
        "size: validator stops checking the opponent hand size",
        WORLDS,
        "    if len(opponent) != view.opponent_hand_size:",
        "    if False:",
        TESTS,
    ),
    # --- trump ----------------------------------------------------------
    (
        "trump: generator puts the trump card on top instead of the bottom",
        WORLDS,
        "            talon = tuple(body) + (self._trump_card,)",
        "            talon = (self._trump_card,) + tuple(body)",
        TESTS,
    ),
    (
        "trump: generator shuffles the trump card in with the rest",
        WORLDS,
        "            body = [c for c in body if c != self._trump_card]\n            rng.shuffle(body)\n            talon = tuple(body) + (self._trump_card,)",
        "            rng.shuffle(body)\n            talon = tuple(body)",
        TESTS,
    ),
    (
        "trump: validator stops enforcing trump-at-bottom",
        WORLDS,
        "    if world.talon and world.talon[-1] != view.trump_card:",
        "    if False:",
        TESTS,
    ),
    (
        "trump: generator ignores the empty/non-empty talon distinction",
        WORLDS,
        "        if self._talon_size:",
        "        if True:",
        TESTS,
    ),
    # --- probability ----------------------------------------------------
    (
        "probability: replace joint allocation with independent marginals",
        WORLDS,
        "        chosen = rng.sample(self._candidates, self._opponent_slots)",
        "        p = self._opponent_slots / max(len(self._candidates), 1)\n        chosen = [c for c in self._candidates if rng.random() < p]",
        TESTS,
    ),
    (
        "probability: bias the allocation toward low cards",
        WORLDS,
        "        chosen = rng.sample(self._candidates, self._opponent_slots)",
        "        pool = self._candidates + self._candidates[: len(self._candidates) // 2]\n        chosen = []\n        for c in pool:\n            if len(chosen) < self._opponent_slots and c not in chosen and rng.random() < 0.5:\n                chosen.append(c)\n        chosen += [c for c in self._candidates if c not in chosen][: self._opponent_slots - len(chosen)]",
        TESTS,
    ),
    # --- reproducibility ------------------------------------------------
    (
        "seed: ignore the supplied RNG and use the global module",
        WORLDS,
        "        chosen = rng.sample(self._candidates, self._opponent_slots)",
        "        chosen = random.sample(self._candidates, self._opponent_slots)",
        TESTS,
    ),
    (
        "seed: reseed from entropy inside sample()",
        WORLDS,
        "        chosen = rng.sample(self._candidates, self._opponent_slots)",
        "        rng = random.Random()\n        chosen = rng.sample(self._candidates, self._opponent_slots)",
        TESTS,
    ),
    (
        "determinism: candidate order depends on set iteration order",
        WORLDS,
        "        self._candidates = tuple(\n            sorted(knowledge.candidates, key=lambda c: c.code)\n        )",
        "        self._candidates = tuple(knowledge.candidates)",
        TESTS,
    ),
    # --- mutability -----------------------------------------------------
    (
        "mutability: return worlds that can be modified after construction",
        WORLDS,
        "@dataclass(frozen=True, slots=True)\nclass DeterminizedWorld:",
        "@dataclass(frozen=False, slots=True)\nclass DeterminizedWorld:",
        TESTS,
    ),
    (
        "validator: accept everything",
        WORLDS,
        "def validate_world(world: DeterminizedWorld, information: SearchInformation) -> None:",
        "def validate_world(world: DeterminizedWorld, information: SearchInformation) -> None:\n    return None",
        TESTS,
    ),
]

#: Considered and documented rather than claimed as tested.
UNREACHABLE = [
    (
        "validator: stop checking the talon size",
        "Logically redundant, not untested. With the opponent's hand size "
        "correct and conservation intact, the talon size is fixed by "
        "counting, so any violation trips conservation or the hand-size "
        "check first. Verified by probing which check fires first for "
        "every corruption. Kept as a defensive condition; documented in "
        "worlds.py.",
    ),
    (
        "validator: allow impossible opponent placements",
        "Logically redundant for the same reason: every card impossible "
        "in the opponent's hand is settled elsewhere, so public-trespass, "
        "the overlap check or certain-ownership rejects it earlier. It can "
        "never be the first check to fire, so no corruption can isolate "
        "it. Kept as defensive; documented in worlds.py.",
    ),
    (
        "injecting GameState access into the generator",
        "worlds.py imports only Card, CardLocation and SearchInformation. "
        "Its single input is a SearchInformation, which Phase 7.1 proved "
        "carries no route to a GameState. A mutation would have to add an "
        "import, which the architecture test and the AST test in "
        "test_worlds.py both reject, so the leak is caught at the import "
        "rather than at runtime.",
    ),
    (
        "reading the actual hidden hand or talon",
        "There is no reference to reach them through: the generator never "
        "receives a GameState, and the noninterference test generates from "
        "two different real hidden states and requires byte-identical "
        "output, so any such read would change the result and fail.",
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
