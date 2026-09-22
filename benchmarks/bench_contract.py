"""Cost of building the Phase 7.1 search contract.

Run: ``python benchmarks/bench_contract.py``

The contract bundles work that already existed, so the question is
whether bundling it adds anything, and whether a future search should
build one per node or carry one per turn.
"""

from __future__ import annotations

import json
import random
import sys
import timeit

sys.path.insert(0, "src")

from durakfish.game import apply_move, get_legal_moves, new_game  # noqa: E402
from durakfish.information import (  # noqa: E402
    CardTracker,
    SearchInformation,
    observe,
)


def one_game(seed: int):
    rng = random.Random(seed)
    state = new_game(seed=seed)
    views = []
    while not state.is_over:
        views.append(observe(state, 0))
        state = apply_move(state, rng.choice(get_legal_moves(state)))
    return views


def report(label: str, seconds: float, iterations: int) -> None:
    per = seconds / iterations
    print(f"  {label:<40} {per * 1e6:8.2f} us   {1 / per:>12,.0f} ops/sec")


def main() -> None:
    views = one_game(11)
    print(f"one game of {len(views)} views\n")

    print("construction")
    n = 20
    t = timeit.timeit(
        lambda: [SearchInformation.from_view(v) for v in views], number=n
    )
    report("from_view()  (rebuilds the tracker)", t, n * len(views))

    tracker = CardTracker(0)
    pairs = []
    for view in views:
        tracker.update(view)
        pairs.append((CardTracker.rebuild(view), view))

    live = CardTracker(0)
    live.update(views[-1])
    ready = (live, views[-1])
    n = 200
    t = timeit.timeit(
        lambda: SearchInformation.from_tracker(*ready), number=n
    )
    report("from_tracker()  (tracker already current)", t, n)

    contract = SearchInformation.from_view(views[-1])
    n = 200
    t = timeit.timeit(contract.fingerprint, number=n)
    report("fingerprint()", t, n)

    t = timeit.timeit(lambda: hash(contract), number=n)
    report("hash()", t, n)

    t = timeit.timeit(lambda: json.dumps(contract.to_dict()), number=n)
    report("to_dict() + json", t, n)

    n = 2000
    t = timeit.timeit(
        lambda: [contract.classify(c) for c in list(contract.knowledge.deck)[:10]],
        number=n,
    )
    report("classify() per card", t, n * 10)

    print("\nrepeated construction (a search building one per node)")
    n = 20
    t = timeit.timeit(
        lambda: [SearchInformation.from_view(views[-1]) for _ in range(10)], number=n
    )
    report("from_view() x10 on one position", t, n * 10)

    print("\nfor comparison")
    print(f"  {'observe() with history (Phase 3)':<40} {186.00:8.2f} us")
    print(f"  {'CardTracker.update() (Phase 6)':<40} { 37.46:8.2f} us")
    print(f"  {'CardTracker.rebuild() (Phase 6)':<40} { 57.50:8.2f} us")
    print(f"  {'SearchBot depth 3 (Phase 5)':<40} {159.00:8.2f} us")


if __name__ == "__main__":
    main()
