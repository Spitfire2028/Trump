"""DurakFish — an imperfect-information game engine for Podkidnoy Durak.

The package root deliberately re-exports the card layer only. Each layer
above it is imported explicitly:

    from durakfish.game import new_game, apply_move, get_legal_moves
    from durakfish.information import observe
    from durakfish.ai import CallableAgent
    from durakfish.simulation import play_game, verify_record

That is not merely a style preference. Re-exporting the upper layers here
would mean ``import durakfish.game`` also loaded the agent and simulation
packages, so the rules engine could no longer be used — or tested — in
isolation from the layers built on top of it. ``tests/test_architecture.py``
enforces exactly that, and this module obeys the same rule it checks.

Layers depend only downward:
``cards -> game -> information -> ai -> simulation``.
"""

from __future__ import annotations

from durakfish.cards import ALL_CARDS, Card, Deck, Rank, Suit, parse_cards
from durakfish.exceptions import DurakFishError

__version__ = "0.3.0"

__all__ = [
    "ALL_CARDS",
    "Card",
    "Deck",
    "DurakFishError",
    "Rank",
    "Suit",
    "__version__",
    "parse_cards",
]
