"""Mutation audit for Phase 7.3.5, run in a scratch copy of the tree.

Specs are plain data; the runner is boring. The working tree is never
opened for writing — each mutation is applied to a temporary copy which
is then deleted — so a kill leaves the real sources untouched. That
matters: an earlier in-place runner in this project left a mutated source
behind twice, and every measurement afterwards was silently wrong.

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
KEY = pathlib.Path("src/durakfish/ai/ismcts/key.py")
NODE = pathlib.Path("src/durakfish/ai/ismcts/node.py")
SEARCH = pathlib.Path("src/durakfish/ai/ismcts/search.py")

TIMEOUT_SECONDS = 180

SELECTION = [
    f"tests/test_ismcts.py::{name}"
    for name in (
        "test_indistinguishable_worlds_reach_one_node_with_shared_statistics",
        "test_the_eight_of_spades_position_shares_one_information_set",
        "test_hidden_permutation_cannot_change_the_key",
        "test_two_worlds_with_different_hidden_cards_share_a_root_key",
        "test_the_key_is_deterministic_and_owner_sensitive",
        "test_distinguishable_states_get_different_keys",
        "test_carried_observation_survives_traversal_and_only_grows",
        "test_edges_are_shared_and_statistics_accumulate",
        "test_selection_tries_every_unvisited_edge_before_exploiting",
        "test_the_registry_returns_one_node_per_key",
        "test_the_determinizer_only_produces_valid_worlds",
        "test_the_determinizer_is_reproducible",
        "test_the_determinizer_preserves_deduced_ownership",
        "test_the_search_returns_a_legal_root_move",
        "test_the_search_is_reproducible_under_a_seed",
        "test_the_search_never_touches_the_global_random_module",
        "test_visits_are_conserved_across_the_root_edges",
        "test_a_terminal_root_yields_no_move",
        "test_view_at_restores_carried_knowledge_onto_a_hypothesis",
        "test_the_observer_must_be_to_move",
        "test_the_recommendation_is_the_most_visited_edge",
        "test_the_search_explores_more_than_one_root_action",
        "test_statistics_accumulate_below_the_root",
        "test_the_search_backs_up_the_observers_perspective",
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
    Mutation(1, "include the hidden world in the node key", KEY,
             "        owner=epistemic.owner, view_key=epistemic.view_at(position).key()",
             "        owner=epistemic.owner,\n        view_key=(epistemic.view_at(position).key(),\n                  position.root_world.key() if position.root_world else ())"),
    Mutation(2, "drop the observer from the key", KEY,
             "        owner=epistemic.owner, view_key=epistemic.view_at(position).key()",
             "        owner=0, view_key=epistemic.view_at(position).key()"),
    Mutation(3, "reconstruct observation from the hypothesis instead of carrying it", KEY,
             "            observed=self.observed | visible | observation.revealed,",
             "            observed=visible | observation.revealed,"),
    Mutation(4, "let the carried observation shrink", KEY,
             "            observed=self.observed | visible | observation.revealed,",
             "            observed=frozenset(visible),"),
    Mutation(5, "ignore carried knowledge when building the view", KEY,
             "        return replace(local, observed=self.observed | local.observed)",
             "        return local"),
    Mutation(6, "identify edges by object identity", NODE,
             "        identity = canonical_move_key(move)\n            if identity not in self.edges:",
             "        identity = (id(move), 0, 0)\n            if identity not in self.edges:"),
    Mutation(7, "look edges up by object identity", NODE,
             "        identity = canonical_move_key(move)\n        edge = self.edges.get(identity)",
             "        identity = (id(move), 0, 0)\n        edge = self.edges.get(identity)"),
    Mutation(8, "never share a node: a fresh node per lookup", NODE,
             "        node = self._nodes.get(key)\n        if node is None:",
             "        node = None\n        if node is None:"),
    Mutation(9, "exploit before trying unvisited edges", NODE,
             "        unvisited = [m for m in ordered if self.edge_for(m).visits == 0]\n        if unvisited:\n            return unvisited[0]",
             "        unvisited = []\n        if unvisited:\n            return unvisited[0]"),
    Mutation(10, "recommend by value instead of visits", NODE,
             "            visited, key=lambda e: (e.visits, [-k for k in canonical_move_key(e.move)])",
             "            visited, key=lambda e: (e.mean_value, [-k for k in canonical_move_key(e.move)])"),
    Mutation(11, "back up the opponent's perspective", SEARCH,
             "        return evaluate_position(position, observer, ply=ply)",
             "        return evaluate_position(position, 1 - observer, ply=ply)"),
    Mutation(13, "ignore the injected RNG", SEARCH,
             "            world = determinizer.sample(rng)",
             "            world = determinizer.sample(random.Random())"),
    Mutation(14, "record statistics on the wrong node", SEARCH,
             "        for node, move in path:\n            node.record_visit()\n            node.edge_for(move).record(value)",
             "        for node, move in path[:1]:\n            node.record_visit()\n            node.edge_for(move).record(value)"),
    Mutation(15, "select using world-specific legality only, ignoring shared statistics", SEARCH,
             "                move = node.uct_select(legal, self._config.exploration)",
             "                move = sorted(legal, key=canonical_move_key)[0]"),
    Mutation(16, "skip the observer-to-move precondition", SEARCH,
             "        if probe.current_player != observer:",
             "        if False:"),
    Mutation(17, "never expand: evaluate the root every iteration", SEARCH,
             "                if fresh and not expanded:",
             "                if True:"),
]

#: Considered and documented rather than claimed as tested.
UNREACHABLE = [
    (
        "check the depth limit before terminality (was mutation 12)",
        "Semantically equivalent, and proven so. The next statement asks "
        "the frozen engine for legal moves, and a finished game has none, "
        "so the loop breaks on the following line regardless. Recorded by "
        "test_a_terminal_position_offers_no_moves.",
    ),
    (
        "sample an impossible world / ignore hard deductions",
        "Not expressible as a mutation of this layer. The Determinizer "
        "delegates wholly to the Phase 7.2 WorldGenerator, which validates "
        "every world it returns; there is no code here that could relax a "
        "deduction. The property is covered from the other side by "
        "test_the_determinizer_only_produces_valid_worlds and "
        "test_the_determinizer_preserves_deduced_ownership, and the "
        "generator itself was mutation-audited in Phase 7.2 (19/19).",
    ),
    (
        "feed a hypothetical view to CardTracker",
        "Cannot be introduced silently: CardTracker refuses a shrinking "
        "event stream and raises TrackerError. Any mutation attempting it "
        "fails loudly rather than degrading, which "
        "test_a_hypothetical_view_is_refused_by_the_card_tracker pins down "
        "as a standing regression.",
    ),
]


def build_scratch() -> pathlib.Path:
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="durakfish-ismcts-mut-"))
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
    for label, reason in UNREACHABLE:
        print(f"\nnot expressible here: {label}\n  {reason}")
    sys.exit(0 if caught == len(MUTATIONS) else 1)


if __name__ == "__main__":
    main()
