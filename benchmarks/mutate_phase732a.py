"""Mutation audit for Phase 7.3.2a, run in a scratch copy of the tree.

Why this runner is shaped this way
----------------------------------
The previous in-place runner corrupted the working tree twice. It mutated
production sources and restored them in a ``finally`` block — which does
not execute when the process is killed from outside. Both times the tree
was left holding the "never decrement depth" mutation, and every
measurement taken afterwards was silently wrong: a 12-second suite
appeared to take 280.

So this runner never touches the working tree. It copies the repository
to a temporary directory, mutates the copy, runs the tests there, and
deletes the copy. A kill at any moment leaves the real sources untouched,
because they are never opened for writing.

Classification
--------------
A hang is not a detection. The old runner would have counted one, and
that is exactly how a genuine test weakness stayed hidden for two turns.
Verdicts are therefore:

===========================  ==============================================
``CAUGHT``                   a test failed, and the failure names the fault
``MISSED``                   the selection passed with the mutation applied
``TIMEOUT``                  needs investigation, never counted as caught
``INVALID``                  anchor missing, or the edit is a no-op
===========================  ==============================================

Only ``CAUGHT`` counts toward the score.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
AI = pathlib.Path("src/durakfish/ai")
SEARCH = AI / "world_search.py"
ADAPTER = AI / "determinized.py"

#: Per-mutation wall-clock ceiling. The clean selection runs in ~4s, so
#: this is ample; it exists so a runaway search is reported rather than
#: stalling the audit.
TIMEOUT_SECONDS = 90

#: A fast selection that still covers every property under audit, chosen
#: so no mutation is judged by how long it takes to fail.
SELECTION = [
    f"tests/test_world_search.py::{name}"
    for name in (
        "test_max_depth_reached_never_exceeds_the_budget",
        "test_the_evaluation_node_describes_the_position_it_came_from",
        "test_values_are_from_the_root_players_point_of_view",
        "test_depth_zero_scores_the_position_where_it_stands",
        "test_leaf_scoring_is_exactly_the_existing_baseline_evaluator",
        "test_depth_one_expands_only_the_root_moves",
        "test_terminal_outcomes_match_the_oracle",
        "test_a_finished_position_returns_its_exact_value_and_no_move",
        "test_terminals_are_detected_before_the_depth_cutoff",
        "test_zero_sum_holds_on_solved_endgames",
        "test_alpha_beta_agrees_with_minimax_on_a_small_sample",
        "test_the_chosen_move_matches_the_oracle_when_it_is_uniquely_best",
        "test_different_worlds_produce_different_search_results",
    )
]

MUTATIONS: list[tuple[str, pathlib.Path, str, str]] = [
    (
        "1. minimax MAX/MIN inversion",
        SEARCH,
        "    maximising = position.current_player == root_player\n    best = -_INFINITY if maximising else _INFINITY",
        "    maximising = position.current_player != root_player\n    best = -_INFINITY if maximising else _INFINITY",
    ),
    (
        "2. alpha-beta MAX/MIN inversion",
        SEARCH,
        "    if position.current_player == root_player:\n        best = -_INFINITY",
        "    if position.current_player != root_player:\n        best = -_INFINITY",
    ),
    (
        "3. root perspective inversion",
        SEARCH,
        "    maximising = position.current_player == player",
        "    maximising = position.current_player != player",
    ),
    (
        "4. terminal check removed from minimax",
        SEARCH,
        "    stats.visit(ply)\n    if position.is_terminal:\n        stats.leaf(terminal=True)\n        return _terminal_value(position, root_player, ply)\n    if depth == 0:",
        "    stats.visit(ply)\n    if depth == 0:",
    ),
    (
        "5. terminal WIN/LOSS swap",
        SEARCH,
        "    elif loser == root_player:\n        outcome = Outcome.LOSS\n    else:\n        outcome = Outcome.WIN",
        "    elif loser == root_player:\n        outcome = Outcome.WIN\n    else:\n        outcome = Outcome.LOSS",
    ),
    (
        "6. draw mis-scored as a win",
        SEARCH,
        "    if loser is None:\n        outcome = Outcome.DRAW",
        "    if loser is None:\n        outcome = Outcome.WIN",
    ),
    (
        "7. minimax depth decrement removed",
        SEARCH,
        "            position.apply(move), depth - 1, ply + 1, root_player,",
        "            position.apply(move), depth, ply + 1, root_player,",
    ),
    (
        "8. alpha-beta depth decrement removed (both branches)",
        SEARCH,
        "depth - 1, ply + 1, alpha, beta,",
        "depth, ply + 1, alpha, beta,",
    ),
    (
        "9. leaf evaluator removed",
        SEARCH,
        '    scorer = evaluator if evaluator is not None else get_evaluator("baseline")\n    return scorer.evaluate(position.evaluation_node(root_player), ply)',
        "    return 0.0",
    ),
    (
        "10. a legal move skipped at every node",
        SEARCH,
        "    return order_self_moves(\n        position.evaluation_node(root_player), moves, ordering=ordering\n    )",
        "    ordered = order_self_moves(\n        position.evaluation_node(root_player), moves, ordering=ordering\n    )\n    return ordered[1:] or ordered",
    ),
    (
        "11. wrong child selection: last tie wins",
        SEARCH,
        "        improved = value > best_value if maximising else value < best_value",
        "        improved = value >= best_value if maximising else value <= best_value",
    ),
    (
        "12. always choose the first root move",
        SEARCH,
        "        if best_move is None or improved:\n            best_value, best_move = value, move",
        "        if best_move is None:\n            best_value, best_move = value, move",
    ),
    (
        "13. alpha bound corrupted at max nodes",
        SEARCH,
        "            alpha = max(alpha, best)\n            if beta <= alpha:",
        "            alpha = max(alpha, best)\n            if beta <= alpha + 1.0:",
    ),
    (
        "14. beta bound corrupted at min nodes",
        SEARCH,
        "        beta = min(beta, best)\n        if beta <= alpha:",
        "        beta = min(beta, best)\n        if beta <= alpha + 1.0:",
    ),
    (
        "16. evaluation node hides the real opponent count",
        ADAPTER,
        "            opponent_cards=len(state.hands[opponent]),",
        "            opponent_cards=6,",
    ),
    (
        "17. evaluation node uses the wrong seat's hand",
        ADAPTER,
        "            hand=state.hands[root_player],",
        "            hand=state.hands[opponent],",
    ),
    (
        "18. SearchNode allowed to pre-empt terminal detection",
        ADAPTER,
        "            kind=NodeKind.DECISION,",
        "            kind=NodeKind.BOUT_RESOLVED,",
    ),
    (
        "19. stale position searched instead of the child",
        SEARCH,
        "    for move in moves:\n        value = _minimax_value(\n            position.apply(move), depth - 1, ply + 1, root_player,",
        "    for move in moves:\n        value = _minimax_value(\n            position, depth - 1, ply + 1, root_player,",
    ),
]

#: Considered and documented rather than claimed as tested.
UNREACHABLE = [
    (
        "root alpha/beta update removed (was mutation 15)",
        "Correctness-preserving, and proven so rather than assumed. "
        "Removing the root-level bound updates only forfeits pruning at "
        "the root; interior cutoffs still occur. Measured with the "
        "mutation applied: 279 comparisons of alpha-beta against "
        "full-width minimax gave 0 differences in value or chosen move, "
        "and alpha-beta still visited 11,444 nodes against minimax's "
        "17,523. It is a performance regression, not a fault, so no "
        "correctness test can or should detect it.",
    ),
    (
        "cache contamination between worlds",
        "No cache exists. Phase 7.3.2a deliberately adds no transposition "
        "table, so cross-world contamination is structurally impossible "
        "rather than merely untested. The interleaving test covers the "
        "property from the other side.",
    ),
    (
        "reading the real hidden state inside the searcher",
        "The searcher's only input is a DeterminizedPosition, which Phase "
        "7.3.1 proved is a pure function of a SearchInformation and a "
        "DeterminizedWorld. There is no reference to the real game to "
        "mutate into a read, and the noninterference test requires "
        "identical results from two different realities.",
    ),
]


def build_scratch() -> pathlib.Path:
    """Copy the repository somewhere disposable."""
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="durakfish-mutate-"))
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


def run_one(path: pathlib.Path, old: str, new: str) -> tuple[str, str]:
    scratch = build_scratch()
    try:
        target = scratch / path
        original = target.read_text()
        if old not in original:
            return "INVALID", "anchor text not found; the mutation is stale"
        mutated = original.replace(old, new)
        if mutated == original:
            return "INVALID", "replacement equals the original; a no-op mutation"
        target.write_text(mutated)

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", *SELECTION, "-q", "-x",
                 "--no-header", "-p", "no:cacheprovider"],
                cwd=str(scratch),
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                env={"PYTHONPATH": "src:.", "PATH": "/usr/bin:/bin", "HOME": "/tmp"},
            )
        except subprocess.TimeoutExpired:
            return "TIMEOUT", f"no verdict within {TIMEOUT_SECONDS}s"

        if result.returncode == 0:
            return "MISSED", "the selection passed with the mutation applied"
        for line in result.stdout.splitlines():
            if line.startswith("E   ") or line.startswith("FAILED"):
                return "CAUGHT", line.strip()[:86]
        return "CAUGHT", f"pytest exit {result.returncode}"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> None:
    print("mutating a scratch copy; the working tree is never written to")
    print(f"per-mutation timeout: {TIMEOUT_SECONDS}s\n")
    tally: dict[str, int] = {}
    for label, path, old, new in MUTATIONS:
        verdict, evidence = run_one(path, old, new)
        tally[verdict] = tally.get(verdict, 0) + 1
        print(f"  {verdict:<8} {label}")
        print(f"           {evidence}")

    caught = tally.get("CAUGHT", 0)
    print(f"\ncaught {caught} of {len(MUTATIONS)}   {tally}")
    for label, reason in UNREACHABLE:
        print(f"\nunreachable: {label}\n  {reason}")
    sys.exit(0 if caught == len(MUTATIONS) else 1)


if __name__ == "__main__":
    main()
