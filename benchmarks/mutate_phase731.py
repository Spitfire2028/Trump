"""Mutation-test the Phase 7.3.1 determinization adapter.

An audit that cannot fail proves nothing. Each mutation below breaks one
property the suite claims to protect; the script applies it, runs the
relevant tests, restores the file byte-for-byte, and reports whether the
break was caught.

Run: ``python benchmarks/mutate_phase731.py``
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "durakfish" / "ai"

ADAPTER = SRC / "determinized.py"
TESTS = "tests/test_determinized.py"

MUTATIONS = [
    (
        "hidden allocation: swap the observer's hand with the opponent's",
        ADAPTER,
        "        add_cards((), view.hand) if seat == observer\n        else add_cards((), world.opponent_hand)",
        "        add_cards((), world.opponent_hand) if seat == observer\n        else add_cards((), view.hand)",
        TESTS,
    ),
    (
        "conservation: drop a card from the hypothetical talon",
        ADAPTER,
        "        talon=world.talon,",
        "        talon=world.talon[1:],",
        TESTS,
    ),
    (
        "talon order: reverse it, moving the trump off the bottom",
        ADAPTER,
        "        talon=world.talon,",
        "        talon=tuple(reversed(world.talon)),",
        TESTS,
    ),
    (
        "trump: take the suit from somewhere other than the trump card",
        ADAPTER,
        "        trump_suit=view.trump_suit,",
        "        trump_suit=view.table[0].attacking_card.suit if view.table else view.trump_suit,",
        TESTS,
    ),
    (
        "public table: drop the table from the hypothetical position",
        ADAPTER,
        "        table=view.table,",
        "        table=(),",
        TESTS,
    ),
    (
        "public discard: drop the discard pile",
        ADAPTER,
        "        discard=view.discard,",
        "        discard=frozenset(),",
        TESTS,
    ),
    (
        "phase: always start in the attack phase",
        ADAPTER,
        "        phase=view.phase,",
        "        phase=list(type(view.phase))[0],",
        TESTS,
    ),
    (
        "attacker: swap which seat is attacking",
        ADAPTER,
        "        attacker=view.attacker,",
        "        attacker=1 - view.attacker,",
        TESTS,
    ),
    (
        "rules: substitute a different attack limit",
        ADAPTER,
        "        attack_limit=view.attack_limit,",
        "        attack_limit=max(1, view.attack_limit - 1),",
        TESTS,
    ),
    (
        "terminal: carry no durak into the hypothetical position",
        ADAPTER,
        "        durak=view.durak,",
        "        durak=None,",
        TESTS,
    ),
    (
        "validation: accept invalid worlds by default",
        ADAPTER,
        "    validate: bool = True,",
        "    validate: bool = False,",
        TESTS,
    ),
    (
        "validation: swallow the validator's verdict",
        ADAPTER,
        "        except DurakFishError as failure:",
        "        except DurakFishError:\n            pass\n        except ValueError as failure:",
        TESTS,
    ),
    (
        "observer check: accept a world belonging to the other player",
        ADAPTER,
        "    if world.player != information.player:",
        "    if False:",
        TESTS,
    ),
    (
        "legal actions: hide one legal move from the caller",
        ADAPTER,
        "        return get_legal_moves(self.state)",
        "        return get_legal_moves(self.state)[:-1] or get_legal_moves(self.state)",
        TESTS,
    ),
    (
        "transition: return the position unchanged instead of applying",
        ADAPTER,
        "            player=self.player, state=apply_move(self.state, move), root_world=None",
        "            player=self.player, state=self.state, root_world=None",
        TESTS,
    ),
    (
        "terminal value: invert win and loss",
        ADAPTER,
        "        return -1 if loser == player else 1",
        "        return 1 if loser == player else -1",
        TESTS,
    ),
    (
        "identity: fingerprint ignores the position entirely",
        ADAPTER,
        "        return (self.player, self.state.transposition_key())",
        "        return (self.player,)",
        TESTS,
    ),
    (
        "provenance: keep a stale root world across transitions",
        ADAPTER,
        "            player=self.player, state=apply_move(self.state, move), root_world=None",
        "            player=self.player, state=apply_move(self.state, move),\n            root_world=self.root_world",
        TESTS,
    ),
]

#: Considered and documented rather than claimed as tested.
UNREACHABLE = [
    (
        "reading the real hidden state inside the adapter",
        "build_position takes exactly two arguments, a SearchInformation "
        "and a DeterminizedWorld, and Phases 7.1 and 7.2 each proved that "
        "neither carries a route to the real GameState. There is no "
        "reference to mutate into a read. The substitution test closes the "
        "gap from the other side: the same world supplied against two "
        "different realities must give an identical position, so any such "
        "read would change the result and fail.",
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
