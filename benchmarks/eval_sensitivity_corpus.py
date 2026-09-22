"""Realistic position corpus for the evaluator-sensitivity study.  (Phase 7.3.9)

Diagnostic, not production. Nothing in the engine imports this.

Positions come from **legal game trajectories** played with the frozen
Phase 2 engine, not from independently synthesised states. That matters:
a synthetic position can be legal yet unreachable, and a corpus of those
would measure evaluator behaviour on states the engine never actually
searches.

Selection bias is the other trap. Phase 7.3.8 established that exactly
solvable positions are ~1% of talon-bearing play and structurally
atypical, so sampling "positions cheap to solve" would quietly answer an
easier question than the one asked. This corpus therefore samples across
whole trajectories and records enough metadata to stratify afterwards,
rather than filtering up front.

Every position is stored with its full serialised ``GameState`` so the
study is rerunnable without regenerating (and without depending on
trajectory generation staying bit-identical).

Run: ``python benchmarks/eval_sensitivity_corpus.py``
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from durakfish.game import apply_move, get_legal_moves, new_game  # noqa: E402

CORPUS_PATH = Path("benchmarks/data/eval_sensitivity_corpus.json")

#: Trajectory seeds. Fixed so the corpus is reproducible.
SEEDS = range(120)

#: Sample every Nth decision along a trajectory. Sampling rather than
#: taking every position keeps successive entries from being trivially
#: correlated (a position and its own child).
STRIDE = 3

#: Cap, to keep the experiment runnable at several depths x 4 evaluators.
TARGET = 900


def phase_of(state) -> str:
    """Coarse game phase, by how much of the talon is left."""
    talon = len(state.talon)
    if talon > 18:
        return "early"
    if talon > 0:
        return "mid"
    return "late"


def describe(state, player: int, seed: int, index: int) -> dict:
    """Metadata used to stratify the results afterwards."""
    opponent = 1 - player
    hand = state.hands[player]
    return {
        "seed": seed,
        "ply_index": index,
        "player": player,
        "card_count": len(state.hands[0]) + len(state.hands[1]) + len(state.talon),
        "own_hand_size": len(hand),
        "opponent_hand_size": len(state.hands[opponent]),
        "talon_size": len(state.talon),
        "trump_suit": state.trump_suit.name,
        "own_trumps": sum(1 for c in hand if c.suit is state.trump_suit),
        "legal_move_count": len(get_legal_moves(state)),
        "phase_name": state.phase.name,
        "game_phase": phase_of(state),
        "is_attacker": state.attacker == player,
        "table_size": len(state.table),
        "attack_limit": state.attack_limit,
        "state": state.to_dict(),
    }


def build() -> list[dict]:
    rows: list[dict] = []
    for seed in SEEDS:
        rng = random.Random(seed)
        state = new_game(seed=seed)
        index = 0
        while not state.is_over:
            player = state.current_player
            if player is not None and index % STRIDE == 0:
                # Only positions with a real choice can show a decision
                # change, so single-move positions are recorded but the
                # experiment will skip them.
                rows.append(describe(state, player, seed, index))
            state = apply_move(state, rng.choice(get_legal_moves(state)))
            index += 1
            if len(rows) >= TARGET:
                break
        if len(rows) >= TARGET:
            break
    return rows


def main() -> None:
    rows = build()
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text(json.dumps(rows))

    print(f"corpus: {len(rows)} positions -> {CORPUS_PATH}")
    for field in ("game_phase", "phase_name", "legal_move_count", "is_attacker"):
        counts: dict = {}
        for row in rows:
            counts[row[field]] = counts.get(row[field], 0) + 1
        shown = dict(sorted(counts.items(), key=lambda kv: str(kv[0]))[:8])
        print(f"  {field:<18} {shown}")
    talons = [r["talon_size"] for r in rows]
    print(f"  talon_size         min {min(talons)} max {max(talons)}")
    multi = sum(1 for r in rows if r["legal_move_count"] >= 2)
    print(f"  positions with >=2 legal moves: {multi}")


if __name__ == "__main__":
    main()
