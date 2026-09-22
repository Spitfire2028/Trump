"""Split corpus for evaluator design and validation.  (Phase 7.3.10)

Diagnostic, not production.

Positions come from legal trajectories played with the frozen Phase 2
engine, as in Phase 7.3.9. The addition here is an explicit, structural
three-way split.

**The split is by trajectory seed, not by position.** Splitting positions
at random would put a position and a near-duplicate of it from the same
game on opposite sides of the split, which inflates held-out agreement
and is exactly how a dataset lies to you. Whole games therefore go to one
side or the other:

====================  ===========
seeds 0–79            development
seeds 80–109          validation
seeds 110–159         held-out test
====================  ===========

The test seeds are disjoint games and are not to be inspected until the
candidate and its coefficients are frozen.

Run: ``python benchmarks/evaluator_corpus.py``
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from durakfish.game import apply_move, get_legal_moves, new_game

CORPUS_PATH = Path("benchmarks/data/evaluator_corpus.json")

SPLITS = {
    "dev": range(0, 80),
    "validation": range(80, 110),
    "test": range(110, 160),
}

STRIDE = 3
PER_SPLIT = 500


def phase_of(state) -> str:
    talon = len(state.talon)
    if talon > 18:
        return "early"
    if talon > 0:
        return "mid"
    return "late"


def describe(state, player: int, seed: int, index: int, split: str) -> dict:
    opponent = 1 - player
    hand = state.hands[player]
    return {
        "split": split,
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
        "attack_limit": state.attack_limit,
        "state": state.to_dict(),
    }


def build() -> list[dict]:
    rows: list[dict] = []
    for split, seeds in SPLITS.items():
        taken = 0
        for seed in seeds:
            if taken >= PER_SPLIT:
                break
            rng = random.Random(seed)
            state = new_game(seed=seed)
            index = 0
            while not state.is_over and taken < PER_SPLIT:
                player = state.current_player
                if player is not None and index % STRIDE == 0:
                    rows.append(describe(state, player, seed, index, split))
                    taken += 1
                state = apply_move(state, rng.choice(get_legal_moves(state)))
                index += 1
    return rows


def main() -> None:
    rows = build()
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows)
    CORPUS_PATH.write_text(payload)

    print(f"corpus: {len(rows)} positions -> {CORPUS_PATH}")
    print(f"  sha256: {hashlib.sha256(payload.encode()).hexdigest()[:32]}")
    for split in SPLITS:
        part = [r for r in rows if r["split"] == split]
        multi = sum(1 for r in part if r["legal_move_count"] >= 2)
        seeds = sorted({r["seed"] for r in part})
        phases: dict = {}
        for row in part:
            phases[row["game_phase"]] = phases.get(row["game_phase"], 0) + 1
        print(f"  {split:<11} n={len(part):<5} >=2 moves={multi:<5} "
              f"seeds {seeds[0]}-{seeds[-1]}  phases {dict(sorted(phases.items()))}")

    overlap = set()
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                overlap |= set(SPLITS[a]) & set(SPLITS[b])
    print(f"  seed overlap between splits: {overlap or 'NONE'}")


if __name__ == "__main__":
    main()
