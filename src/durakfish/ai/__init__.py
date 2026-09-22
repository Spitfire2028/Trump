"""AI layer: the agent boundary and the baseline agents.

Phase 3 defined the protocol and its non-strategic reference
implementations. Phase 4 adds the first real policies — ``RandomBot`` and
``GreedyBot`` — plus the deterministic tie-breaking they share.

Both bots decide from an ``InformationSet`` and a list of legal moves and
from nothing else. Search, evaluation, card tracking and opponent models
belong to later phases.
"""

from __future__ import annotations

from durakfish.ai.base import Agent, AgentBase, CallableAgent, ScriptedAgent
from durakfish.ai.determinized import (
    DeterminizationError,
    DeterminizedPosition,
    build_position,
)
from durakfish.ai.evaluation import BaselineEvaluator, EvaluationWeights, Evaluator
from durakfish.ai.greedy_bot import GreedyBot, card_cost
from durakfish.ai.multi_world import (
    AggregationError,
    MoveAggregate,
    MultiWorldResult,
    WeightedWorld,
    aggregate_worlds,
    enumerate_worlds,
)
from durakfish.ai.random_bot import RandomBot
from durakfish.ai.search import (
    SearchConfig,
    SearchResult,
    SearchStats,
    alpha_beta,
    minimax,
    search,
)
from durakfish.ai.search_bot import SearchBot
from durakfish.ai.searchnode import SearchNode
from durakfish.ai.tiebreak import canonical_move_key, canonical_order
from durakfish.ai.world_search import (
    WorldSearchResult,
    alpha_beta_world,
    evaluate_position,
    minimax_world,
    search_world,
)

__all__ = [
    "Agent",
    "AgentBase",
    "AggregationError",
    "BaselineEvaluator",
    "CallableAgent",
    "DeterminizationError",
    "DeterminizedPosition",
    "EvaluationWeights",
    "Evaluator",
    "GreedyBot",
    "MoveAggregate",
    "MultiWorldResult",
    "RandomBot",
    "ScriptedAgent",
    "SearchBot",
    "SearchConfig",
    "SearchNode",
    "SearchResult",
    "SearchStats",
    "WeightedWorld",
    "WorldSearchResult",
    "aggregate_worlds",
    "alpha_beta",
    "alpha_beta_world",
    "build_position",
    "canonical_move_key",
    "canonical_order",
    "card_cost",
    "enumerate_worlds",
    "evaluate_position",
    "minimax",
    "minimax_world",
    "search",
    "search_world",
]
