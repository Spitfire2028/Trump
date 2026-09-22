"""Baseline performance of the rules-engine hot paths.

Run: ``python benchmarks/bench_rules.py``

These numbers are a *baseline to regress against*, not a target. The
engine is written for correctness first; Phase 14 optimises against
profiles, and profiling without a recorded starting point is guesswork.

The positions sampled are real ones taken from random games, not
hand-picked easy cases, so the numbers reflect the mix search will see.
"""

from __future__ import annotations

import random
import sys
import timeit
from dataclasses import replace

sys.path.insert(0, "src")

from durakfish.cards import Suit
from durakfish.cards.deck import standard_cards
from durakfish.game import (
    apply_move,
    beats,
    get_legal_moves,
    is_legal_move,
    new_game,
)

DECK = standard_cards(36)


def sample_positions(count: int = 400) -> list:
    """Collect live positions from random games."""
    positions = []
    rng = random.Random(12345)
    seed = 0
    while len(positions) < count:
        state = new_game(seed=seed)
        seed += 1
        while not state.is_over and len(positions) < count:
            positions.append(state)
            state = apply_move(state, rng.choice(get_legal_moves(state)))
    return positions


def report(label: str, seconds: float, iterations: int) -> None:
    per_op = seconds / iterations
    rate = 1.0 / per_op
    unit = f"{per_op * 1e6:8.2f} us"
    print(f"  {label:<34} {unit}   {rate:>12,.0f} ops/sec")


def main() -> None:
    positions = sample_positions()
    rng = random.Random(7)
    moves = [rng.choice(get_legal_moves(s)) for s in positions]
    print(f"sampled {len(positions)} live positions from random games\n")

    print("rules primitives")
    pairs = [(rng.choice(DECK), rng.choice(DECK)) for _ in range(1000)]
    n = 200
    t = timeit.timeit(
        lambda: [beats(d, a, Suit.HEARTS) for d, a in pairs], number=n
    )
    report("beats()", t, n * len(pairs))

    n = 200
    t = timeit.timeit(lambda: [get_legal_moves(s) for s in positions], number=n)
    report("get_legal_moves()", t, n * len(positions))

    n = 200
    t = timeit.timeit(
        lambda: [is_legal_move(s, m) for s, m in zip(positions, moves)], number=n
    )
    report("is_legal_move()", t, n * len(positions))

    n = 100
    t = timeit.timeit(
        lambda: [apply_move(s, m) for s, m in zip(positions, moves)], number=n
    )
    report("apply_move()  (validating)", t, n * len(positions))

    print("\nstate operations")
    n = 200
    t = timeit.timeit(lambda: [replace(s, attacker=s.attacker) for s in positions], number=n)
    report("state copy (dataclasses.replace)", t, n * len(positions))

    n = 200
    t = timeit.timeit(lambda: [s.transposition_key() for s in positions], number=n)
    report("transposition_key()", t, n * len(positions))

    n = 200
    t = timeit.timeit(lambda: [hash(s.transposition_key()) for s in positions], number=n)
    report("transposition_key() + hash", t, n * len(positions))

    n = 20
    t = timeit.timeit(lambda: [s.validate() for s in positions], number=n)
    report("validate()  (tests only)", t, n * len(positions))

    print("\nwhole games")
    games = 300
    start = timeit.default_timer()
    total_moves = 0
    for seed in range(games):
        rng2 = random.Random(seed)
        state = new_game(seed=seed)
        while not state.is_over:
            state = apply_move(state, rng2.choice(get_legal_moves(state)))
            total_moves += 1
    elapsed = timeit.default_timer() - start
    print(f"  {'random playouts':<34} {elapsed / games * 1e3:8.2f} ms   "
          f"{games / elapsed:>12,.0f} games/sec")
    print(f"  {'':<34} {'':>11}   {total_moves / elapsed:>12,.0f} moves/sec")
    print(f"\n  mean game length: {total_moves / games:.1f} moves")


if __name__ == "__main__":
    main()
