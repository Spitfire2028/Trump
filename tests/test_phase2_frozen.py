"""The Phase 2 freeze, made executable.

Phase 2 was accepted, audited and frozen. "Frozen" is a promise that
decays the moment somebody runs a formatter across ``src/`` — as nearly
happened during Phase 3, when a repository-wide ``ruff --fix`` was run
over a directory that includes the rules engine.

Timestamps are poor evidence: an edit that reverts itself still bumps
them, and a bulk tool touches everything at once. Content hashes are
exact. Any change to ``game/`` now fails this test by name, which forces
the change to be deliberate and explained rather than incidental.

**If this test fails**, do not simply update the hash. Either revert the
change, or — if the change is genuinely required — justify it, add a
regression test for the behaviour it fixes, and update the hash in the
same commit so the record shows the engine changed and why.

The baseline was recorded during the Phase 3 audit. The evidence that it
equals Phase 2's accepted content is behavioural rather than
archaeological: the 245 Phase 1-2 tests and both Phase 2 audits (3,000
random and 3,000 adversarial-policy games) pass unchanged against exactly
these bytes.
"""

from __future__ import annotations

import hashlib
import pathlib

GAME = pathlib.Path(__file__).resolve().parent.parent / "src" / "durakfish" / "game"

#: SHA-256 of every module in the frozen rules engine.
FROZEN: dict[str, str] = {
    "__init__.py": "96228bc2486f534789f3c83097538bc80ba9ef88f0888fb2c7b7273a3e37b1a5",
    "history.py": "4184acc60a23b89756204d6655c234fde060dfe5cf9b8b22298571ccdad72d64",
    "moves.py": "683e376ec40bb7aaac577af476a1c78add4f3220bcbba479c076d24dab12c2ba",
    "rules.py": "2ba45aef0a129498d86d5bfcff9cd452342106051087167177dd1f98b093cb2a",
    "ruleset.py": "5eb52f7b3eb28f1f3cb7ec6b42ddddecdc76eaf440764e86e4432fddcf71e3bb",
    "state.py": "79656be23b5dd3031d2db932d597adef9bf006d83fa396bc44300b02c30e28a0",
}


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_rules_engine_is_byte_for_byte_unchanged() -> None:
    changed = [
        name
        for name, expected in FROZEN.items()
        if digest(GAME / name) != expected
    ]
    assert not changed, (
        f"frozen Phase 2 modules were modified: {changed}. "
        "See this module's docstring before updating the hashes."
    )


def test_no_module_was_added_to_or_removed_from_the_rules_engine() -> None:
    present = {p.name for p in GAME.glob("*.py")}
    assert present == set(FROZEN), (
        f"rules engine gained {present - set(FROZEN)} and lost "
        f"{set(FROZEN) - present}"
    )
