"""Perfect-information world search: depth, nodes, time.

Run: ``python benchmarks/bench_world_search.py``

A baseline for Phase 7.3.2b, which will evaluate many worlds per
decision. The figures that matter are how nodes grow with depth and how
much alpha-beta saves; correctness came first and nothing here was tuned.
"""

from __future__ import annotations

import random
import sys
import timeit

sys.path.insert(0, "src")

from durakfish.ai import (  # noqa: E402
    SearchConfig, alpha_beta_world, build_position, minimax_world,
)
from durakfish.game import apply_move, get_legal_moves, new_game  # noqa: E402
from durakfish.information import (  # noqa: E402
    DeterminizedWorld, SearchInformation, observe,
)


def positions(count: int = 40):
    """Representative positions across the phases of a game."""
    out, rng, seed = [], random.Random(5), 0
    while len(out) < count:
        state = new_game(seed=seed)
        seed += 1
        for index, _ in enumerate(range(200)):
            if state.is_over or len(out) >= count:
                break
            player = state.current_player
            assert player is not None
            if index % 9 == 0:
                information = SearchInformation.from_view(observe(state, player))
                world = DeterminizedWorld(
                    player=player,
                    opponent_hand=frozenset(state.hands[1 - player]),
                    talon=tuple(state.talon),
                )
                out.append(build_position(information, world))
            state = apply_move(state, rng.choice(get_legal_moves(state)))
    return out


def main() -> None:
    sample = positions()
    print(f"{len(sample)} positions across early, mid and late game\n")
    print(f"  {'algorithm':<12} {'depth':>5} {'nodes/pos':>10} {'cutoffs':>9} "
          f"{'pruned':>8} {'ms/pos':>9} {'nodes/sec':>12}")

    for depth in (1, 3, 5):
        row = {}
        for label, fn, algorithm in (
            ("minimax", minimax_world, "minimax"),
            ("alpha_beta", alpha_beta_world, "alpha_beta"),
        ):
            config = SearchConfig(depth=depth, algorithm=algorithm)
            start = timeit.default_timer()
            nodes = cuts = 0
            for position in sample:
                result = fn(position, config)
                nodes += result.stats.nodes
                cuts += result.stats.cutoffs
            row[label] = (nodes, cuts, timeit.default_timer() - start)

        baseline = row["minimax"][0]
        for label in ("minimax", "alpha_beta"):
            nodes, cuts, elapsed = row[label]
            share = (
                f"{100 * (1 - nodes / baseline):6.1f}%"
                if label == "alpha_beta" and baseline
                else "     -"
            )
            print(f"  {label:<12} {depth:>5} {nodes / len(sample):>10.1f} "
                  f"{cuts / len(sample):>9.1f} {share:>8} "
                  f"{elapsed / len(sample) * 1e3:>9.3f} {nodes / elapsed:>12,.0f}")

    print("\nfor context")
    print(f"  {'build_position() (7.3.1)':<34} { 56.50:8.2f} us")
    print(f"  {'WorldGenerator.sample (7.2)':<34} { 25.72:8.2f} us")
    print(f"  {'apply_move() (Phase 2)':<34} { 13.60:8.2f} us")


if __name__ == "__main__":
    main()
