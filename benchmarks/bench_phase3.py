"""Baseline performance of the Phase 3 boundary and driver.

Run: ``python benchmarks/bench_phase3.py``

The number worth watching is the cost of ``observe()`` relative to
``apply_move()``. Redaction walks the whole event log, so it grows with
game length — fine at one call per turn, potentially not at search-node
rates. Recording the shape of that growth now is what will make the Phase
6 decision (maintain the redacted log incrementally) evidence-based.
"""

from __future__ import annotations

import random
import sys
import timeit

sys.path.insert(0, "src")

from durakfish.ai import CallableAgent
from durakfish.game import apply_move, get_legal_moves, new_game
from durakfish.information import observe
from durakfish.simulation import play_game, replay, verify_record


def report(label: str, seconds: float, iterations: int, unit: str = "ops") -> None:
    per_op = seconds / iterations
    print(f"  {label:<40} {per_op * 1e6:8.2f} us   {1 / per_op:>12,.0f} {unit}/sec")


def sample_positions(count: int = 400) -> list:
    positions = []
    rng = random.Random(99)
    seed = 0
    while len(positions) < count:
        state = new_game(seed=seed)
        seed += 1
        while not state.is_over and len(positions) < count:
            positions.append(state)
            state = apply_move(state, rng.choice(get_legal_moves(state)))
    return positions


def main() -> None:
    positions = sample_positions()
    mean_history = sum(p.event_count for p in positions) / len(positions)
    print(f"sampled {len(positions)} live positions "
          f"(mean history length {mean_history:.0f} events)\n")

    print("information boundary")
    n = 100
    t = timeit.timeit(lambda: [observe(s, 0) for s in positions], number=n)
    report("observe()  with history", t, n * len(positions))

    t = timeit.timeit(
        lambda: [observe(s, 0, include_history=False) for s in positions], number=n
    )
    report("observe()  without history", t, n * len(positions))

    views = [observe(s, 0) for s in positions]
    n = 200
    t = timeit.timeit(lambda: [v.key() for v in views], number=n)
    report("InformationSet.key()", t, n * len(views))

    n = 50
    t = timeit.timeit(lambda: [v.unobserved_cards() for v in views], number=n)
    report("unobserved_cards()", t, n * len(views))

    # Cost growth with game length: early vs late positions.
    early = [p for p in positions if p.event_count < 30]
    late = [p for p in positions if p.event_count > 120]
    for label, group in (("early game", early), ("late game", late)):
        if not group:
            continue
        n = 100
        t = timeit.timeit(lambda g=group: [observe(s, 0) for s in g], number=n)
        mean = sum(p.event_count for p in group) / len(group)
        report(f"observe()  {label} (~{mean:.0f} events)", t, n * len(group))

    print("\ndriver")
    agents = [
        CallableAgent(lambda v, legal, r: r.choice(legal), "a"),
        CallableAgent(lambda v, legal, r: r.choice(legal), "b"),
    ]
    games = 200
    start = timeit.default_timer()
    plies = 0
    for seed in range(games):
        record = play_game(agents, seed=seed)
        plies += record.length
    elapsed = timeit.default_timer() - start
    print(f"  {'play_game()':<40} {elapsed / games * 1e3:8.2f} ms   "
          f"{games / elapsed:>12,.0f} games/sec")
    print(f"  {'':<40} {'':>11}   {plies / elapsed:>12,.0f} plies/sec")

    records = [play_game(agents, seed=s) for s in range(50)]
    start = timeit.default_timer()
    for record in records:
        list(replay(record, validate_states=False))
    elapsed = timeit.default_timer() - start
    print(f"  {'replay()  (no validation)':<40} "
          f"{elapsed / len(records) * 1e3:8.2f} ms   "
          f"{len(records) / elapsed:>12,.0f} games/sec")

    start = timeit.default_timer()
    for record in records:
        verify_record(record)
    elapsed = timeit.default_timer() - start
    print(f"  {'verify_record()  (full validation)':<40} "
          f"{elapsed / len(records) * 1e3:8.2f} ms   "
          f"{len(records) / elapsed:>12,.0f} games/sec")

    print(f"\n  mean game length: {plies / games:.1f} plies")


if __name__ == "__main__":
    main()
