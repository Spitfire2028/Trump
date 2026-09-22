"""Search performance: depth, nodes, latency, and the match-ups.

Run: ``python benchmarks/bench_search.py``

The relationship worth watching is depth -> nodes -> latency, not
games/sec: the second follows from the first, and only the first says
anything about whether the search will scale when the tree grows.
"""

from __future__ import annotations

import random
import sys
import timeit

sys.path.insert(0, "src")

from durakfish.ai import (  # noqa: E402
    BaselineEvaluator,
    GreedyBot,
    RandomBot,
    SearchBot,
    SearchConfig,
    SearchNode,
    alpha_beta,
    minimax,
)
from durakfish.game import apply_move, get_legal_moves, new_game  # noqa: E402
from durakfish.information import observe  # noqa: E402
from durakfish.simulation import play_game, summarize  # noqa: E402

DEPTHS = (1, 2, 3, 4, 5)


def sample(count: int = 200):
    pairs = []
    rng = random.Random(99)
    seed = 0
    while len(pairs) < count:
        state = new_game(seed=seed)
        seed += 1
        while not state.is_over and len(pairs) < count:
            player = state.current_player
            assert player is not None
            view = observe(state, player)
            pairs.append((SearchNode.from_view(view), get_legal_moves(state), view))
            state = apply_move(state, rng.choice(get_legal_moves(state)))
    return pairs


def main() -> None:
    pairs = sample()
    print(f"sampled {len(pairs)} real positions\n")

    evaluator = BaselineEvaluator()
    n = 500
    t = timeit.timeit(lambda: [evaluator.evaluate(p[0], 0) for p in pairs], number=n)
    per = t / (n * len(pairs))
    print(f"  {'evaluator':<28} {per * 1e6:8.2f} us   {1 / per:>12,.0f} ops/sec\n")

    print(f"  {'algorithm':<12} {'depth':>5} {'nodes/pos':>10} {'cutoffs':>9} "
          f"{'pruned':>8} {'latency':>11}")
    for depth in DEPTHS:
        rows = {}
        for label, fn, algo in (
            ("minimax", minimax, "minimax"),
            ("alpha_beta", alpha_beta, "alpha_beta"),
        ):
            config = SearchConfig(depth=depth, algorithm=algo)
            start = timeit.default_timer()
            nodes = cuts = 0
            for node, legal, _ in pairs:
                result = fn(node, config, root_moves=legal)
                nodes += result.stats.nodes
                cuts += result.stats.cutoffs
            elapsed = timeit.default_timer() - start
            rows[label] = (nodes, cuts, elapsed)
        mm_nodes = rows["minimax"][0]
        ab_nodes, ab_cuts, ab_time = rows["alpha_beta"]
        pruned = 100 * (1 - ab_nodes / mm_nodes) if mm_nodes else 0.0
        for label in ("minimax", "alpha_beta"):
            nodes, cuts, elapsed = rows[label]
            share = f"{pruned:6.1f}%" if label == "alpha_beta" else "     -"
            print(f"  {label:<12} {depth:>5} {nodes / len(pairs):>10.1f} "
                  f"{cuts / len(pairs):>9.1f} {share:>8} "
                  f"{elapsed / len(pairs) * 1e6:>9.1f} us")

    print("\ndecision latency")
    for depth in DEPTHS:
        bot = SearchBot(SearchConfig(depth=depth), "s")
        bot.reset(random.Random(0))
        start = timeit.default_timer()
        for _, legal, view in pairs:
            bot.choose(view, legal)
        elapsed = timeit.default_timer() - start
        per = elapsed / len(pairs)
        print(f"  SearchBot depth {depth:<11} {per * 1e6:8.2f} us   "
              f"{1 / per:>12,.0f} ops/sec")
    print(f"  {'RandomBot (Phase 4)':<26} {1.8:8.2f} us")
    print(f"  {'GreedyBot (Phase 4)':<26} {2.9:8.2f} us")
    print(f"  {'observe() (Phase 3)':<26} {186.0:8.2f} us")

    print("\nmatch-ups (sanity checks, not strength certification)")
    config = SearchConfig(depth=3)
    matchups = {
        "SearchBot vs RandomBot": lambda: [SearchBot(config, "s"), RandomBot("r")],
        "SearchBot vs GreedyBot": lambda: [SearchBot(config, "s"), GreedyBot("g")],
        "SearchBot vs SearchBot": lambda: [SearchBot(config, "a"), SearchBot(config, "b")],
    }
    games = 100
    for label, make in matchups.items():
        agents = make()
        start = timeit.default_timer()
        records = [play_game(agents, seed=s, first_attacker=s % 2)
                   for s in range(games)]
        elapsed = timeit.default_timer() - start
        summary = summarize(records)
        print(f"  {label:<26} {games / elapsed:>6,.1f} games/sec   "
              f"{summary.mean_plies:5.1f} plies   "
              f"seat0 {summary.win_rate(0):5.1%}  draws {summary.draws}")


if __name__ == "__main__":
    main()
