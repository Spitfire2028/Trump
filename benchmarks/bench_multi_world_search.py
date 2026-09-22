"""Scaling of multi-world aggregation.

Run: ``python benchmarks/bench_multi_world_search.py``

The expected relationship is simply

    work ~ world_count x root_moves x single_world_search_cost

because every world is searched independently and no cache exists. The
point of measuring is to confirm that and to record the constant, not to
tune anything.
"""

from __future__ import annotations

import random
import sys
import timeit

sys.path.insert(0, "src")

from durakfish.ai import SearchConfig, aggregate_worlds  # noqa: E402
from durakfish.game import apply_move, get_legal_moves, new_game  # noqa: E402
from durakfish.information import (  # noqa: E402
    SearchInformation, WorldGenerator, observe,
)

WORLD_COUNTS = (1, 5, 10, 25)
DEPTHS = (1, 3, 5)


def decisions(count: int = 6):
    """Representative positions where the observer is to move."""
    out, rng, seed = [], random.Random(5), 0
    while len(out) < count:
        state = new_game(seed=seed)
        seed += 1
        for index in range(200):
            if state.is_over or len(out) >= count:
                break
            player = state.current_player
            assert player is not None
            if index % 17 == 0:
                information = SearchInformation.from_view(observe(state, player))
                if information.allocations >= 25:
                    out.append(information)
            state = apply_move(state, rng.choice(get_legal_moves(state)))
    return out


def main() -> None:
    sample = decisions()
    print(f"{len(sample)} decision positions, observer to move\n")
    print(f"  {'depth':>5} {'worlds':>7} {'nodes':>10} {'ms/decision':>12} "
          f"{'ms/world':>10} {'nodes/world':>12}")

    for depth in DEPTHS:
        config = SearchConfig(depth=depth)
        for count in WORLD_COUNTS:
            worlds_per_position = [
                [WorldGenerator(info).sample(random.Random(k)) for k in range(count)]
                for info in sample
            ]
            start = timeit.default_timer()
            nodes = 0
            for info, worlds in zip(sample, worlds_per_position, strict=True):
                nodes += aggregate_worlds(info, worlds, config).total_nodes
            elapsed = timeit.default_timer() - start
            per_decision = elapsed / len(sample)
            print(f"  {depth:>5} {count:>7} {nodes / len(sample):>10.0f} "
                  f"{per_decision * 1e3:>12.2f} {per_decision / count * 1e3:>10.2f} "
                  f"{nodes / len(sample) / count:>12.1f}")

    print("\n  determinism: repeated runs give identical values (asserted in tests)")
    print("  no cache exists, so scaling is linear in world count by construction")


if __name__ == "__main__":
    main()
