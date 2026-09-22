"""Cost of maintaining knowledge.

Run: ``python benchmarks/bench_knowledge.py``

The question this answers is whether carrying knowledge forward is
actually cheaper than recomputing it. The tracker keeps exactly one fact
between events, so the answer is not obvious in advance and is worth
measuring rather than assuming.
"""

from __future__ import annotations

import json
import random
import sys
import timeit

sys.path.insert(0, "src")

from durakfish.cards.deck import standard_cards  # noqa: E402
from durakfish.game import apply_move, get_legal_moves, new_game  # noqa: E402
from durakfish.information import (  # noqa: E402
    BeliefState,
    CardLocation,
    CardTracker,
    KnowledgeState,
    Observation,
    deduce,
    observe,
)

DECK = frozenset(standard_cards(36))


def one_game(seed: int) -> list:
    """Views from a single game, in order — what a tracker actually sees."""
    rng = random.Random(seed)
    state = new_game(seed=seed)
    views = []
    while not state.is_over:
        views.append(observe(state, 0))
        state = apply_move(state, rng.choice(get_legal_moves(state)))
    views.append(observe(state, 0))
    return views


def sample(count: int = 300):
    """A spread of positions from many games, for the stateless measurements."""
    views = []
    rng = random.Random(7)
    seed = 1000
    while len(views) < count:
        state = new_game(seed=seed)
        seed += 1
        while not state.is_over and len(views) < count:
            views.append(observe(state, 0))
            state = apply_move(state, rng.choice(get_legal_moves(state)))
    return views


def report(label: str, seconds: float, iterations: int) -> None:
    per = seconds / iterations
    print(f"  {label:<36} {per * 1e6:8.2f} us   {1 / per:>12,.0f} ops/sec")


def main() -> None:
    views = sample()
    game = one_game(11)
    mean_history = sum(len(v.public_history) for v in views) / len(views)
    print(f"sampled {len(views)} positions across games "
          f"(mean history {mean_history:.0f} events)")
    print(f"plus one full game of {len(game)} sequential views\n")

    print("knowledge maintenance")
    tracker = CardTracker(0)
    tracker.update(views[-1])
    knowledge = tracker.knowledge
    belief = tracker.belief

    n = 50
    def incremental() -> None:
        live = CardTracker(0)
        for view in game:
            live.update(view)

    t = timeit.timeit(incremental, number=n)
    report("incremental update (per event)", t, n * len(game))

    def rebuild() -> None:
        for view in game:
            CardTracker.rebuild(view)

    t = timeit.timeit(rebuild, number=n)
    report("full rebuild (per position)", t, n * len(game))

    observations = [Observation.from_view(v) for v in views]
    n = 100
    t = timeit.timeit(lambda: [deduce(o) for o in observations], number=n)
    report("deduce()", t, n * len(observations))

    n = 200
    t = timeit.timeit(
        lambda: [belief.probability(c, CardLocation.OPPONENT_HAND) for c in DECK],
        number=n,
    )
    report("belief.probability() per card", t, n * len(DECK))

    n = 2000
    t = timeit.timeit(lambda: knowledge.key(), number=n)
    report("KnowledgeState.key()", t, n)

    t = timeit.timeit(lambda: json.dumps(knowledge.to_dict()), number=n)
    report("knowledge serialisation", t, n)

    t = timeit.timeit(
        lambda: KnowledgeState.from_dict(knowledge.to_dict(), DECK), number=n
    )
    report("knowledge deserialisation", t, n)

    n = 500
    t = timeit.timeit(lambda: BeliefState.from_knowledge(knowledge).to_dict(), number=n)
    report("belief summary", t, n)

    print("\nreplay throughput")
    start = timeit.default_timer()
    for _ in range(20):
        CardTracker.replay(0, game)
    elapsed = timeit.default_timer() - start
    print(f"  {'replay of a whole game':<36} "
          f"{elapsed / 20 * 1e3:8.2f} ms   {20 * len(game) / elapsed:>12,.0f} events/sec")

    print("\nfor comparison")
    print(f"  {'observe() with history (Phase 3)':<36} {186.0:8.2f} us")
    print(f"  {'apply_move() (Phase 2)':<36} {13.6:8.2f} us")
    print(f"  {'SearchBot depth 3 (Phase 5)':<36} {159.0:8.2f} us")


if __name__ == "__main__":
    main()
