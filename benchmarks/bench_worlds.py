"""Cost of generating determinized worlds.

Run: ``python benchmarks/bench_worlds.py``

A future search may want many worlds per decision, so the figures that
matter are the marginal cost of one more world and how much of the total
is one-off setup. Correctness came first in Phase 7.2; this establishes
the baseline that a later phase would optimise against.
"""

from __future__ import annotations

import random
import sys
import timeit

sys.path.insert(0, "src")

from durakfish.game import apply_move, get_legal_moves, new_game  # noqa: E402
from durakfish.information import (  # noqa: E402
    CardTracker,
    SearchInformation,
    WorldGenerator,
    observe,
    validate_world,
)

sys.path.insert(0, ".")
from tests.reference_worlds import enumerate_allocations  # noqa: E402


def positions(seed: int = 11):
    """A mid-game position, a late one, and a fully determined one."""
    rng = random.Random(seed)
    state = new_game(seed=seed)
    found: dict[str, SearchInformation] = {}
    while not state.is_over:
        information = SearchInformation.from_view(observe(state, 0))
        talon = information.view.talon_size
        if talon > 15 and "early" not in found:
            found["early"] = information
        elif 0 < talon <= 6 and "late" not in found:
            found["late"] = information
        elif talon == 0 and "determined" not in found:
            found["determined"] = information
        state = apply_move(state, rng.choice(get_legal_moves(state)))
    return found


def report(label: str, seconds: float, iterations: int) -> None:
    per = seconds / iterations
    print(f"  {label:<44} {per * 1e6:8.2f} us   {1 / per:>12,.0f} ops/sec")


def main() -> None:
    found = positions()
    for name, information in found.items():
        print(f"\n{name}: {len(information.undetermined)} undetermined, "
              f"{information.allocations:,} allocations")
        generator = WorldGenerator(information)
        rng = random.Random(1)

        n = 2000
        t = timeit.timeit(lambda: WorldGenerator(information), number=n)
        report("construct a generator", t, n)

        t = timeit.timeit(lambda: generator.sample(rng), number=n)
        report("sample one world", t, n)

        n = 200
        t = timeit.timeit(lambda: generator.sample_many(10, rng), number=n)
        report("sample 10 worlds", t, n * 10)

        n = 20
        t = timeit.timeit(lambda: generator.sample_many(100, rng), number=n)
        report("sample 100 worlds", t, n * 100)

        world = generator.sample(rng)
        n = 1000
        t = timeit.timeit(lambda: validate_world(world, information), number=n)
        report("validate one world", t, n)

        if information.allocations <= 5000:
            n = 5
            t = timeit.timeit(lambda: enumerate_allocations(information), number=n)
            report("reference enumeration (whole space)", t, n)

    print("\nend-to-end, per decision")
    information = found.get("early")
    if information is not None:
        rng = random.Random(2)
        n = 200
        t = timeit.timeit(
            lambda: WorldGenerator(information).sample_many(32, rng), number=n
        )
        per = t / n
        print(f"  {'build generator + 32 worlds':<44} {per * 1e6:8.2f} us   "
              f"{1 / per:>12,.0f} decisions/sec")

    print("\nfor comparison")
    print(f"  {'SearchInformation.from_tracker (7.1)':<44} {  1.27:8.2f} us")
    print(f"  {'SearchInformation.from_view (7.1)':<44} { 71.86:8.2f} us")
    print(f"  {'CardTracker.update (Phase 6)':<44} { 37.46:8.2f} us")
    print(f"  {'observe() with history (Phase 3)':<44} {186.00:8.2f} us")
    print(f"  {'apply_move() (Phase 2)':<44} { 13.60:8.2f} us")
    _ = CardTracker


if __name__ == "__main__":
    main()
