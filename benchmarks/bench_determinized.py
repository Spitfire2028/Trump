"""Overhead of the determinization adapter.

Run: ``python benchmarks/bench_determinized.py``

The comparison that matters is the adapter path against searching the
real position directly. Any difference is pure overhead introduced by
Phase 7.3.1, and the point of measuring is to record it rather than to
tune it.
"""

from __future__ import annotations

import random
import sys
import timeit

sys.path.insert(0, "src")

from durakfish.ai import SearchConfig, SearchNode, alpha_beta, build_position  # noqa: E402
from durakfish.game import apply_move, get_legal_moves, new_game  # noqa: E402
from durakfish.information import (  # noqa: E402
    DeterminizedWorld,
    SearchInformation,
    WorldGenerator,
    observe,
)


def sample(count: int = 150):
    out = []
    rng = random.Random(9)
    seed = 0
    while len(out) < count:
        state = new_game(seed=seed)
        seed += 1
        while not state.is_over and len(out) < count:
            player = state.current_player
            assert player is not None
            information = SearchInformation.from_view(observe(state, player))
            world = DeterminizedWorld(
                player=player,
                opponent_hand=frozenset(state.hands[1 - player]),
                talon=tuple(state.talon),
            )
            out.append((state, player, information, world))
            state = apply_move(state, rng.choice(get_legal_moves(state)))
    return out


def report(label: str, seconds: float, iterations: int) -> None:
    per = seconds / iterations
    print(f"  {label:<46} {per * 1e6:8.2f} us   {1 / per:>12,.0f} ops/sec")


def main() -> None:
    rows = sample()
    print(f"sampled {len(rows)} positions\n")

    print("adapter")
    n = 100
    t = timeit.timeit(
        lambda: [build_position(i, w) for _, _, i, w in rows], number=n
    )
    report("build_position()  (validating)", t, n * len(rows))

    t = timeit.timeit(
        lambda: [build_position(i, w, validate=False) for _, _, i, w in rows],
        number=n,
    )
    report("build_position()  (validation off)", t, n * len(rows))

    positions = [build_position(i, w) for _, _, i, w in rows]
    n = 200
    t = timeit.timeit(lambda: [p.legal_moves() for p in positions], number=n)
    report("legal_moves()", t, n * len(positions))

    moves = [p.legal_moves()[0] for p in positions]
    t = timeit.timeit(
        lambda: [p.apply(m) for p, m in zip(positions, moves, strict=True)],
        number=n,
    )
    report("apply()", t, n * len(positions))

    t = timeit.timeit(lambda: [p.fingerprint() for p in positions], number=n)
    report("fingerprint()", t, n * len(positions))

    n = 100
    t = timeit.timeit(
        lambda: [
            SearchNode.from_view(observe(p.state, p.player)) for p in positions
        ],
        number=n,
    )
    report("SearchNode from a hypothetical position", t, n * len(positions))

    print("\nsearch: adapter path vs the real position, same depth")
    for depth in (1, 3, 5):
        config = SearchConfig(depth=depth)

        def direct() -> None:
            for state, player, _, _ in rows:
                stripped = state.replace(history=None)
                alpha_beta(
                    SearchNode.from_view(observe(stripped, player)),
                    config,
                    root_moves=get_legal_moves(stripped),
                )

        def adapted() -> None:
            for _, _, information, world in rows:
                position = build_position(information, world)
                alpha_beta(
                    SearchNode.from_view(observe(position.state, position.player)),
                    config,
                    root_moves=position.legal_moves(),
                )

        n = 5
        direct_time = timeit.timeit(direct, number=n) / (n * len(rows))
        adapted_time = timeit.timeit(adapted, number=n) / (n * len(rows))
        overhead = (adapted_time - direct_time) * 1e6
        print(f"  depth {depth}: direct {direct_time * 1e6:8.2f} us   "
              f"adapter {adapted_time * 1e6:8.2f} us   "
              f"overhead {overhead:+7.2f} us "
              f"({100 * overhead / (direct_time * 1e6):+5.1f}%)")

    print("\nfor comparison")
    print(f"  {'WorldGenerator.sample (7.2, early)':<46} { 25.72:8.2f} us")
    print(f"  {'SearchInformation.from_tracker (7.1)':<46} {  1.27:8.2f} us")
    print(f"  {'observe() with history (Phase 3)':<46} {186.00:8.2f} us")
    print(f"  {'apply_move() (Phase 2)':<46} { 13.60:8.2f} us")


if __name__ == "__main__":
    main()
