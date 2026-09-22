"""ISMCTS scaling: iterations, information sets, worlds.

Run: ``python benchmarks/bench_ismcts.py``

The reported ``reuse`` column is worlds sampled per distinct information
set. Read it carefully: it is dominated by the expansion rule (one new
node per iteration) and by how quickly worlds become *distinguishable*,
not by whether sharing works. In the early game the opponent's very first
card play is observable, so worlds separate immediately and reuse sits
near 1.00 — that is correct behaviour, not absent sharing. In an endgame
with a single compatible world reuse climbs steeply as the tree is
revisited.

Evidence that sharing works is the 8♠ regression in
``tests/test_ismcts.py``, not this column.

No strength claim is made or measured here.
"""

from __future__ import annotations

import random
import sys
import timeit

sys.path.insert(0, "src")

from durakfish.ai.ismcts import ISMCTS, ISMCTSConfig  # noqa: E402
from durakfish.game import apply_move, get_legal_moves, new_game  # noqa: E402
from durakfish.information import SearchInformation, observe  # noqa: E402

ITERATIONS = (10, 100, 1_000)


def positions():
    """One early, one mid, one endgame position, observer to move."""
    found = {}
    rng = random.Random(5)
    seed = 0
    while len(found) < 3:
        state = new_game(seed=seed)
        seed += 1
        while not state.is_over:
            player = state.current_player
            if player is not None:
                info = SearchInformation.from_view(observe(state, player))
                talon = info.view.talon_size
                if talon > 18 and "early" not in found:
                    found["early"] = info
                elif 0 < talon <= 8 and "mid" not in found:
                    found["mid"] = info
                elif talon == 0 and "endgame" not in found:
                    found["endgame"] = info
            state = apply_move(state, rng.choice(get_legal_moves(state)))
    return found


def main() -> None:
    sample = positions()
    print(f"  {'position':<9} {'worlds':>8} {'iters':>7} {'time ms':>9} "
          f"{'iters/s':>9} {'info sets':>10} {'reuse':>8} {'root moves':>11}")
    for label in ("early", "mid", "endgame"):
        info = sample[label]
        for iterations in ITERATIONS:
            search = ISMCTS(ISMCTSConfig(iterations=iterations))
            start = timeit.default_timer()
            result = search.search(info, random.Random(1))
            elapsed = timeit.default_timer() - start
            sets = result.information_sets
            sharing = result.worlds_sampled / sets if sets else 0.0
            print(f"  {label:<9} {info.allocations:>8,} {iterations:>7} "
                  f"{elapsed * 1e3:>9.1f} {iterations / elapsed:>9.0f} "
                  f"{sets:>10} {sharing:>8.2f} {len(result.visit_counts()):>11}")

    print("\n  reuse = worlds sampled / distinct information sets")
    print("  Dominated by the expansion rule and by how fast worlds become")
    print("  distinguishable. Near 1.00 early is expected: the opponent's")
    print("  first observable card separates the worlds. Sharing is proven")
    print("  by the 8-of-spades regression, not by this column.")


if __name__ == "__main__":
    main()
