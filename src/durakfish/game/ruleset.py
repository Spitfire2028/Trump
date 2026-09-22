"""Rule configuration.

Every number that a Durak variant might change lives here rather than as
a literal buried in the rules engine. This module has no dependencies
beyond the standard library so that both :mod:`durakfish.game.state` and
:mod:`durakfish.game.rules` can import it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

from durakfish.exceptions import UnsupportedRuleError

__all__ = ["STANDARD_RULES", "FirstAttackerPolicy", "RuleSet"]

from enum import Enum


class FirstAttackerPolicy(str, Enum):
    """How the opening attacker is chosen."""

    LOWEST_TRUMP = "lowest_trump"
    """Standard: whoever holds the lowest trump. See :data:`RuleSet` for
    the deterministic fallback when neither player has one."""

    FIXED = "fixed"
    """Player 0 always opens. Useful for reproducible benchmarks."""


@dataclass(frozen=True, slots=True)
class RuleSet:
    """An immutable description of the variant being played.

    Defaults describe **two-player Podkidnoy Durak with 36 cards**, which
    is the only configuration this engine version implements end to end.
    """

    deck_size: int = 36
    num_players: int = 2
    hand_size: int = 6
    """Players refill to this many cards after each bout, talon permitting."""

    max_attacks_per_bout: int = 6
    """Hard cap on attacking cards in one bout, before the defender's
    hand size is taken into account."""

    addition_ranks_include_defenses: bool = True
    """Standard Podkidnoy: a card may be added if its rank appears
    anywhere on the table, including on cards played *in defence*.
    Some house rules count attacking cards only; setting this False
    would implement those (not yet supported)."""

    additions_allowed_after_take: bool = True
    """Standard Podkidnoy: once the defender announces a take, the
    attacker may keep throwing in matching ranks up to the limit, and the
    defender picks those up too."""

    first_attacker_policy: FirstAttackerPolicy = FirstAttackerPolicy.LOWEST_TRUMP

    def __post_init__(self) -> None:
        if self.num_players != 2:
            raise UnsupportedRuleError(
                f"this engine version implements 2-player Durak only, "
                f"got num_players={self.num_players}"
            )
        if self.hand_size < 1:
            raise UnsupportedRuleError("hand_size must be at least 1")
        if self.max_attacks_per_bout < 1:
            raise UnsupportedRuleError("max_attacks_per_bout must be at least 1")
        if not self.addition_ranks_include_defenses:
            raise UnsupportedRuleError(
                "attack-ranks-exclude-defences is not implemented yet"
            )
        if not self.additions_allowed_after_take:
            raise UnsupportedRuleError(
                "disabling throw-ins after a take is not implemented yet"
            )


#: The default variant: 36-card, 2-player Podkidnoy Durak.
STANDARD_RULES = RuleSet()
