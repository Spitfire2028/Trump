"""Recording a game.

A :class:`GameRecord` holds everything needed to reproduce a game exactly:
the seeds it was played from, the ruleset, the opening position, and the
moves. It is ground truth — it contains the real deal, including cards no
player could see — so a record is a debugging and training artefact, never
something handed to an agent.

Redundancy on purpose
---------------------
The record stores both the derived seeds *and* the dealt opening position.
Either alone would do: the deal is a pure function of the deal seed. Both
are kept so :func:`~durakfish.simulation.runner.verify_record` can check
they still agree, which turns any future change to dealing or seed
derivation into a failing test instead of a silent divergence in old
records.

:attr:`MoveRecord.annotations` is reserved for search statistics and
evaluations from later phases. It stays empty here rather than being
retrofitted once self-play data matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from durakfish.exceptions import RecordError
from durakfish.game.moves import Move, move_from_dict
from durakfish.game.ruleset import STANDARD_RULES, RuleSet
from durakfish.game.state import GameState

__all__ = ["RECORD_FORMAT_VERSION", "GameRecord", "MoveRecord"]

#: Bumped whenever the on-disk shape changes incompatibly.
RECORD_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class MoveRecord:
    """One decision, with room for whatever later phases want to attach."""

    ply: int
    player: int
    move: Move
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "ply": self.ply,
            "player": self.player,
            "move": self.move.to_dict(),
        }
        if self.annotations:
            data["annotations"] = dict(self.annotations)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MoveRecord:
        return cls(
            ply=int(data["ply"]),
            player=int(data["player"]),
            move=move_from_dict(data["move"]),
            annotations=dict(data.get("annotations", {})),
        )


@dataclass(frozen=True, slots=True)
class GameRecord:
    """A complete, replayable game.

    Attributes:
        seed: the master seed the game was played from.
        deal_seed: derived stream that produced the deal.
        agent_seeds: derived stream handed to each agent, by seat.
        agent_names: who played each seat.
        initial_state: the dealt opening position, ground truth.
        moves: every decision in order.
        durak: the loser, or ``None`` for a draw.
        deck_supplied: True when an explicit deck was dealt from, in which
            case ``deal_seed`` did not determine the cards.
    """

    seed: int
    deal_seed: int
    agent_seeds: tuple[int, ...]
    agent_names: tuple[str, ...]
    rules: RuleSet
    initial_state: GameState
    moves: tuple[MoveRecord, ...]
    durak: int | None
    deck_supplied: bool = False
    version: int = RECORD_FORMAT_VERSION

    @property
    def is_draw(self) -> bool:
        return self.durak is None

    @property
    def length(self) -> int:
        """Number of decisions taken."""
        return len(self.moves)

    def winner(self) -> int | None:
        """The player who went out, or ``None`` for a draw. Two players only."""
        if self.durak is None:
            return None
        return (self.durak + 1) % self.rules.num_players

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "deal_seed": self.deal_seed,
            "agent_seeds": list(self.agent_seeds),
            "agent_names": list(self.agent_names),
            "deck_supplied": self.deck_supplied,
            "rules": {
                "deck_size": self.rules.deck_size,
                "num_players": self.rules.num_players,
                "hand_size": self.rules.hand_size,
                "max_attacks_per_bout": self.rules.max_attacks_per_bout,
                "first_attacker_policy": self.rules.first_attacker_policy.value,
            },
            "initial_state": self.initial_state.to_dict(),
            "moves": [m.to_dict() for m in self.moves],
            "durak": self.durak,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameRecord:
        """Rebuild a record.

        Raises:
            RecordError: if the record was written by an incompatible
                format version, or is missing required fields.
        """
        version = int(data.get("version", -1))
        if version != RECORD_FORMAT_VERSION:
            raise RecordError(
                f"record format version {version} is not supported "
                f"(this build reads version {RECORD_FORMAT_VERSION})"
            )
        try:
            raw_rules = data["rules"]
            rules = RuleSet(
                deck_size=int(raw_rules["deck_size"]),
                num_players=int(raw_rules["num_players"]),
                hand_size=int(raw_rules["hand_size"]),
                max_attacks_per_bout=int(raw_rules["max_attacks_per_bout"]),
            )
            return cls(
                seed=int(data["seed"]),
                deal_seed=int(data["deal_seed"]),
                agent_seeds=tuple(int(s) for s in data["agent_seeds"]),
                agent_names=tuple(str(n) for n in data["agent_names"]),
                rules=rules,
                initial_state=GameState.from_dict(
                    data["initial_state"], rules=rules
                ),
                moves=tuple(MoveRecord.from_dict(m) for m in data["moves"]),
                durak=data["durak"],
                deck_supplied=bool(data.get("deck_supplied", False)),
                version=version,
            )
        except KeyError as missing:
            raise RecordError(f"record is missing field {missing}") from None

    def __repr__(self) -> str:
        outcome = "draw" if self.is_draw else f"P{self.durak} lost"
        names = " vs ".join(self.agent_names)
        return f"GameRecord(seed={self.seed}, {names}, {self.length} plies, {outcome})"


# Referenced by GameRecord.from_dict's default ruleset construction path.
_ = STANDARD_RULES
