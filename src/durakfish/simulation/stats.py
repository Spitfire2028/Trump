"""Minimal statistics over completed games.

Enough to check that agents behave and that results are reproducible, and
no more. Ratings, tournaments, round-robins and significance testing are
Phase 13; the shape of this module is deliberately too small to grow into
them by accident.

Everything here is derived from :class:`~durakfish.simulation.record.GameRecord`
objects after the fact. Nothing observes a game while it runs, so
collecting statistics cannot perturb one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from durakfish.exceptions import SimulationError
from durakfish.simulation.record import GameRecord

__all__ = ["MatchSummary", "summarize"]


@dataclass(frozen=True, slots=True)
class MatchSummary:
    """Aggregate outcome of a set of games played by the same line-up.

    Attributes:
        agent_names: by seat.
        games: how many records were summarised.
        plies: total decisions taken across all games.
        decisions: decisions taken by each seat.
        wins: games each seat won by going out first.
        losses: games each seat lost as the durak.
        draws: games where both players went out together.
    """

    agent_names: tuple[str, ...]
    games: int
    plies: int
    decisions: tuple[int, ...]
    wins: tuple[int, ...]
    losses: tuple[int, ...]
    draws: int

    @property
    def mean_plies(self) -> float:
        return self.plies / self.games if self.games else 0.0

    def win_rate(self, seat: int) -> float:
        """Share of games won outright by ``seat``. Draws count as neither."""
        return self.wins[seat] / self.games if self.games else 0.0

    def describe(self) -> str:
        lines = [
            f"{self.games} games, {self.mean_plies:.1f} plies on average",
        ]
        for seat, name in enumerate(self.agent_names):
            lines.append(
                f"  seat {seat} {name:<12} "
                f"{self.wins[seat]:>5} won  {self.losses[seat]:>5} lost  "
                f"({self.win_rate(seat):.1%})  "
                f"{self.decisions[seat]:>7} decisions"
            )
        lines.append(f"  draws {self.draws}")
        return "\n".join(lines)


def summarize(records: Sequence[GameRecord]) -> MatchSummary:
    """Aggregate records played by one line-up.

    Raises:
        SimulationError: if the records do not share a seating, since
            summing wins across different line-ups would be meaningless.
    """
    if not records:
        raise SimulationError("cannot summarise an empty set of records")

    names = records[0].agent_names
    seats = len(names)
    wins = [0] * seats
    losses = [0] * seats
    decisions = [0] * seats
    draws = 0
    plies = 0

    for record in records:
        if record.agent_names != names:
            raise SimulationError(
                f"records mix line-ups: {names} and {record.agent_names}"
            )
        plies += record.length
        for entry in record.moves:
            decisions[entry.player] += 1
        if record.durak is None:
            draws += 1
        else:
            losses[record.durak] += 1
            for seat in range(seats):
                if seat != record.durak:
                    wins[seat] += 1

    return MatchSummary(
        agent_names=names,
        games=len(records),
        plies=plies,
        decisions=tuple(decisions),
        wins=tuple(wins),
        losses=tuple(losses),
        draws=draws,
    )
