"""Simulation layer: driving, recording and replaying complete games."""

from __future__ import annotations

from durakfish.simulation.record import (
    RECORD_FORMAT_VERSION,
    GameRecord,
    MoveRecord,
)
from durakfish.simulation.runner import (
    DEFAULT_MOVE_LIMIT,
    derive_seeds,
    play_game,
    replay,
    verify_record,
)
from durakfish.simulation.stats import MatchSummary, summarize

__all__ = [
    "DEFAULT_MOVE_LIMIT",
    "RECORD_FORMAT_VERSION",
    "GameRecord",
    "MatchSummary",
    "MoveRecord",
    "derive_seeds",
    "play_game",
    "replay",
    "summarize",
    "verify_record",
]
