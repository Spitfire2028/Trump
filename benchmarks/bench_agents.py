"""Baseline performance of the Phase 4 agents.

Run: ``python benchmarks/bench_agents.py``

The question this answers is whether the agent layer is now the
bottleneck, or whether the Phase 3 information boundary still is. Phase 3
measured ``observe()`` at ~186 us against ~14 us for ``apply_move``; if a
bot's decision is far cheaper than that, the agents are not what needs
optimising and reporting so is more useful than tuning them.
"""

from __future__ import annotations

import random
import sys
import timeit

sys.path.insert(0, "src")

from durakfish.ai import GreedyBot, RandomBot  # noqa: E402
from durakfish.game import apply_move, get_legal_moves, new_game  # noqa: E402
from durakfish.information import observe  # noqa: E402
from durakfish.simulation import play_game, summarize  # noqa: E402


def sample_decisions(count: int = 400):
    """Collect (view, legal) pairs from real positions."""
    pairs = []
    rng = random.Random(99)
    seed = 0
    while len(pairs) < count:
        state = new_game(seed=seed)
        seed += 1
        while not state.is_over and len(pairs) < count:
            player = state.current_player
            assert player is not None
            pairs.append((observe(state, player), get_legal_moves(state)))
            state = apply_move(state, rng.choice(get_legal_moves(state)))
    return pairs


def report(label: str, seconds: float, iterations: int) -> None:
    per = seconds / iterations
    print(f"  {label:<38} {per * 1e6:8.2f} us   {1 / per:>12,.0f} ops/sec")


def main() -> None:
    pairs = sample_decisions()
    mean_legal = sum(len(legal) for _, legal in pairs) / len(pairs)
    print(f"sampled {len(pairs)} real decisions "
          f"({mean_legal:.1f} legal moves on average)\n")

    print("decision latency")
    for name, bot in (("RandomBot.choose()", RandomBot()), ("GreedyBot.choose()", GreedyBot())):
        bot.reset(random.Random(1))
        n = 200
        t = timeit.timeit(
            lambda b=bot: [b.choose(v, legal) for v, legal in pairs], number=n
        )
        report(name, t, n * len(pairs))

    print("\nfor comparison (Phase 3 baseline)")
    print(f"  {'observe() with history':<38} {186.0:8.2f} us   "
          f"{5_377:>12,.0f} ops/sec   (Phase 3 measurement)")
    print(f"  {'apply_move()':<38} {13.6:8.2f} us   "
          f"{73_484:>12,.0f} ops/sec   (Phase 2 measurement)")

    print("\nmatch-ups")
    matchups = {
        "RandomBot vs RandomBot": lambda: [RandomBot("a"), RandomBot("b")],
        "RandomBot vs GreedyBot": lambda: [RandomBot("a"), GreedyBot("b")],
        "GreedyBot vs GreedyBot": lambda: [GreedyBot("a"), GreedyBot("b")],
    }
    games = 200
    for label, make in matchups.items():
        agents = make()
        start = timeit.default_timer()
        records = [play_game(agents, seed=s) for s in range(games)]
        elapsed = timeit.default_timer() - start
        summary = summarize(records)
        print(f"  {label:<38} {elapsed / games * 1e3:8.2f} ms   "
              f"{games / elapsed:>7,.0f} games/sec   "
              f"{summary.mean_plies:5.1f} plies   "
              f"seat0 {summary.win_rate(0):5.1%}")

    print("\n  Phase 3 baseline (CallableAgent random): 39 games/sec")


if __name__ == "__main__":
    main()
