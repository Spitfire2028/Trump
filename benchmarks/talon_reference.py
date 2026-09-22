"""Exact reference values for talon-bearing positions.  (Phase 7.3.8)

Diagnostic, not production. Nothing in the engine imports this.

Reconstructed after a container filesystem reset destroyed the original
working tree. The code is reproduced from the Phase 7.3.8 record; its
correctness is re-verified on every run of ``verify`` below rather than
assumed, because a reconstruction is exactly the kind of artefact that
should not be trusted on the strength of its provenance.

Independence
------------
The solver uses only the frozen Phase 2 transitions
(``get_legal_moves``, ``apply_move``, ``is_over``, ``durak``) plus the
engine's own ``transposition_key`` for memoisation. Those define the
game, so reusing them cannot bias a value. It does **not** use
``BaselineEvaluator``, ``SearchBot``, ``world_search``, ISMCTS or
multi-world aggregation, so it cannot reproduce the thing it is meant to
judge.

It is written independently of ``perfect_information_value`` in the test
suite rather than reusing it, because it needs two properties that
function lacks: an explicit node budget, so an intractable position is
*abandoned* rather than hanging the study, and a returned exactness flag
so estimated values can never be silently mixed with proven ones.

Why a fixed talon removes chance
--------------------------------
Talon order is part of ``GameState`` and draws take from a fixed end, so
once the order is specified the game is fully deterministic — verified by
replaying identical move sequences. ``transposition_key`` includes the
talon and excludes history, so memoising on it is sound. There is no
chance node to average over, which is what makes an exact minimax well
defined here.

Why alpha-beta rather than plain minimax
----------------------------------------
The first version of this solver was full-width. With outcomes confined
to {-1, 0, +1} the alpha-beta window closes almost immediately, and
adding pruning moved 12-card positions from 0/10 proven to 8/8 at 2.5 s
mean. That is not an optimisation detail: an unpruned solver understates
what is feasible, and a "domain infeasible" verdict drawn from it would
have been an artefact of the solver rather than a fact about the game.

Run ``python benchmarks/talon_reference.py`` to re-verify the solver
against the independent test oracle and re-measure feasibility.
"""

from __future__ import annotations

import random
import statistics
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from durakfish.game import apply_move, get_legal_moves, new_game  # noqa: E402


class BudgetExceeded(Exception):
    """The position could not be proven within the node budget."""


@dataclass
class SolveResult:
    value: int
    exact: bool
    nodes: int


def solve(state, root_player: int, budget: int = 200_000) -> SolveResult:
    """Exact game-theoretic value, or a budget failure.

    Memo entries store a bound flag, because a value proved under a
    narrow window is not necessarily exact outside it; storing a cutoff
    as though it were exact is the classic way to corrupt a
    transposition table.

    Returns +1 / 0 / -1 from ``root_player``'s perspective. ``exact`` is
    False only when the budget was hit, in which case ``value`` is
    meaningless and must not be used.
    """
    memo: dict = {}
    counter = [0]

    def recurse(node, alpha: int, beta: int) -> int:
        counter[0] += 1
        if counter[0] > budget:
            raise BudgetExceeded
        if node.is_over:
            loser = node.durak
            return 0 if loser is None else (-1 if loser == root_player else 1)

        key = node.transposition_key()
        cached = memo.get(key)
        if cached is not None:
            value, flag = cached
            if flag == 0:
                return value
            if flag < 0 and value <= alpha:
                return value
            if flag > 0 and value >= beta:
                return value

        original_alpha, original_beta = alpha, beta
        mover = node.current_player
        if mover == root_player:
            best = -1
            for move in get_legal_moves(node):
                best = max(best, recurse(apply_move(node, move), alpha, beta))
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
        else:
            best = 1
            for move in get_legal_moves(node):
                best = min(best, recurse(apply_move(node, move), alpha, beta))
                beta = min(beta, best)
                if alpha >= beta:
                    break

        flag = 0
        if best <= original_alpha:
            flag = -1
        elif best >= original_beta:
            flag = 1
        memo[key] = (best, flag)
        return best

    try:
        value = recurse(state, -1, 1)
    except (BudgetExceeded, RecursionError):
        return SolveResult(0, False, counter[0])
    return SolveResult(value, True, counter[0])


# ----------------------------------------------------------------------
def walk(seed: int):
    rng = random.Random(seed)
    state = new_game(seed=seed)
    while not state.is_over:
        yield state
        state = apply_move(state, rng.choice(get_legal_moves(state)))


def verify(limit: int = 40) -> tuple[int, int]:
    """Check the solver against the independent test-suite oracle.

    Only empty-talon endgames, because that is the one region where a
    second, independently written exact solver exists to disagree with.
    """
    from tests.test_abstraction_audit import perfect_information_value

    agree = checked = 0
    for seed in range(40):
        for state in walk(seed):
            if state.talon or state.is_over or state.current_player is None:
                continue
            if len(state.hands[0]) + len(state.hands[1]) > 5:
                continue
            player = state.current_player
            result = solve(state, player, budget=400_000)
            if result.exact:
                checked += 1
                agree += result.value == perfect_information_value(state, player, {})
            if checked >= limit:
                return agree, checked
    return agree, checked


def feasibility(max_cards: int = 13, games: int = 60, budget: int = 400_000) -> None:
    """Re-measure how far exact solving reaches with a talon present."""
    pool: dict[int, list] = {}
    for seed in range(games):
        for state in walk(seed):
            if state.is_over or state.current_player is None or not state.talon:
                continue
            total = len(state.hands[0]) + len(state.hands[1]) + len(state.talon)
            if total <= max_cards:
                pool.setdefault(total, []).append(state)
    print("  cards | found | proven | mean ms | mean nodes")
    for total in sorted(pool):
        sample = pool[total][:8]
        proven = 0
        times: list[float] = []
        nodes: list[int] = []
        for state in sample:
            start = time.perf_counter()
            result = solve(state, state.current_player, budget=budget)
            times.append((time.perf_counter() - start) * 1e3)
            nodes.append(result.nodes)
            proven += result.exact
        print(f"  {total:5d} | {len(pool[total]):5d} | {proven:2d}/{len(sample):<3d} |"
              f" {statistics.mean(times):7.1f} | {statistics.mean(nodes):10.0f}")


def main() -> None:
    sys.setrecursionlimit(60_000)
    print("=== solver correctness vs the independent test oracle ===")
    agree, checked = verify()
    print(f"  agrees on {agree}/{checked} empty-talon endgames")
    if agree != checked:
        raise SystemExit("reconstructed solver disagrees with the oracle")
    print("\n=== exact feasibility with a talon present ===")
    feasibility()


if __name__ == "__main__":
    main()
