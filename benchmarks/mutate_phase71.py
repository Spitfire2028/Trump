"""Mutation-test the Phase 7.1 search contract.

An audit that cannot fail proves nothing. Each mutation below breaks one
property the suite claims to protect; the script applies it, runs the
relevant tests, restores the file byte-for-byte, and reports whether the
break was caught.

Run: ``python benchmarks/mutate_phase71.py``
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "durakfish" / "information"

CONTRACT = SRC / "search_contract.py"
KNOWLEDGE = SRC / "knowledge.py"
BELIEF = SRC / "belief.py"
TESTS = "tests/test_search_contract.py"

MUTATIONS = [
    (
        "expose the information set's raw view under a forbidden name",
        CONTRACT,
        "    view: InformationSet\n    knowledge: KnowledgeState",
        "    view: InformationSet\n    hidden_state: object = None\n    knowledge: KnowledgeState",
        TESTS,
    ),
    (
        "replace probability with certainty in classify()",
        CONTRACT,
        "        if certainty is Certainty.DEDUCED:\n            return InformationClass.DEDUCED",
        "        return InformationClass.CERTAIN",
        TESTS,
    ),
    (
        "drop deduction information: report every rule as observation",
        CONTRACT,
        "    def justification(self, card: Card) -> str | None:\n        \"\"\"Which rule established the card's location, if any.\"\"\"\n        return self.knowledge.justification(card)",
        "    def justification(self, card: Card) -> str | None:\n        return \"observed\"",
        TESTS,
    ),
    (
        "remove impossible-card filtering from the contract",
        CONTRACT,
        "    def is_impossible(self, card: Card, location: CardLocation) -> bool:\n        return self.knowledge.is_impossible(card, location)",
        "    def is_impossible(self, card: Card, location: CardLocation) -> bool:\n        return False",
        TESTS,
    ),
    (
        "change probability values",
        CONTRACT,
        "    def probability(self, card: Card, location: CardLocation) -> float:\n        \"\"\"Marginal probability the card is in the location.\"\"\"\n        return self.belief.probability(card, location)",
        "    def probability(self, card: Card, location: CardLocation) -> float:\n        return min(1.0, self.belief.probability(card, location) * 1.5)",
        TESTS,
    ),
    (
        "make two indistinguishable worlds produce different fingerprints",
        CONTRACT,
        "            tuple(sorted(c.code for c in self.undetermined)),\n            self.allocations,",
        "            tuple(sorted(c.code for c in self.undetermined)),\n            id(self.view) % 7,\n            self.allocations,",
        TESTS,
    ),
    (
        "allow post-construction mutation by unfreezing the contract",
        CONTRACT,
        "@dataclass(frozen=True, slots=True)\nclass SearchInformation:",
        "@dataclass(frozen=False, slots=True, eq=False)\nclass SearchInformation:",
        TESTS,
    ),
    (
        "change serialisation: drop the classification block",
        CONTRACT,
        '            "classification": {\n                card.short: self.classify(card).value\n                for card in sorted(self.knowledge.deck, key=lambda c: c.code)\n            },',
        "",
        TESTS,
    ),
    (
        "bypass the boundary: skip the player-mismatch check",
        CONTRACT,
        "        if tracker.player != view.player:",
        "        if False:",
        TESTS,
    ),
    (
        "let a stale tracker be paired with a newer view",
        CONTRACT,
        "        if tracker.events_seen != len(view.public_history):",
        "        if False:",
        TESTS,
    ),
    (
        "hidden-hand leakage: expose the undetermined set as opponent cards",
        CONTRACT,
        "    def certain_cards_in(self, location: CardLocation) -> frozenset[Card]:\n        \"\"\"Cards established to be in ``location``. Never speculative.\"\"\"\n        return self.knowledge.cards_in(location)",
        "    def certain_cards_in(self, location: CardLocation) -> frozenset[Card]:\n        return self.knowledge.cards_in(location) | self.knowledge.candidates",
        TESTS,
    ),
    (
        "collapse certainty into probability in the underlying knowledge",
        KNOWLEDGE,
        "        if location in self.possible_locations(card):\n            return Certainty.POSSIBLE\n        return Certainty.IMPOSSIBLE",
        "        return Certainty.POSSIBLE",
        TESTS + " tests/test_knowledge.py",
    ),
    (
        "break normalisation in the belief the contract exposes",
        BELIEF,
        "            return self.knowledge.talon_slots / total",
        "            return 0.0",
        TESTS,
    ),
]

#: Considered and documented rather than claimed as tested.
UNREACHABLE = [
    (
        "obtaining a GameState inside the contract",
        "search_contract.py imports nothing from durakfish.game beyond "
        "value types, and its inputs are an InformationSet and a "
        "KnowledgeState, both already redacted. A mutation would have to "
        "add an import that the architecture and AST tests both reject, "
        "so the leak is caught at the import rather than at runtime.",
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
