"""Game layer: moves, rules, state and history. Depends only on cards."""

from __future__ import annotations

from durakfish.game.history import (
    DrawRecord,
    EventKind,
    GameEvent,
    HistoryNode,
    push_event,
)
from durakfish.game.moves import (
    END_ATTACK,
    TAKE,
    AttackMove,
    DefenseMove,
    EndAttack,
    Move,
    MoveType,
    TakeCards,
    move_from_dict,
)
from durakfish.game.rules import (
    apply_move,
    beats,
    choose_first_attacker,
    get_legal_moves,
    is_legal_move,
    legal_attack_cards,
    legal_defense_cards,
    new_game,
    random_playout,
)
from durakfish.game.ruleset import STANDARD_RULES, FirstAttackerPolicy, RuleSet
from durakfish.game.state import GameState, Phase, TableSlot

__all__ = [
    "END_ATTACK",
    "STANDARD_RULES",
    "TAKE",
    "AttackMove",
    "DefenseMove",
    "DrawRecord",
    "EndAttack",
    "EventKind",
    "FirstAttackerPolicy",
    "GameEvent",
    "GameState",
    "HistoryNode",
    "Move",
    "MoveType",
    "Phase",
    "RuleSet",
    "TableSlot",
    "TakeCards",
    "apply_move",
    "beats",
    "choose_first_attacker",
    "get_legal_moves",
    "is_legal_move",
    "legal_attack_cards",
    "legal_defense_cards",
    "move_from_dict",
    "new_game",
    "push_event",
    "random_playout",
]
