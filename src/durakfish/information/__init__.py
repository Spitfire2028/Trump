"""Information layer: what each player knows, and what follows from it.

Three separable stages, in a strict one-directional pipeline:

    observe()          Phase 3 - redaction. Reports what was seen and
                       performs no inference whatsoever.
        |
    CardTracker        Phase 6 - accumulation. Folds the public event
        |              stream into an observer's running record.
    deduce()           Phase 6 - logic. Turns observations into facts that
        |              follow from the rules and card conservation.
    KnowledgeState     observed / deduced / undetermined, never mixed.
        |
    BeliefState        Phase 6 - probability, computed from knowledge and
        |              unable to write back into it.
    SearchInformation  Phase 7.1 - the single object a search may consume.
        |
    DeterminizedWorld  Phase 7.2 - one concrete hidden world drawn
                       uniformly from those the information permits.

``observe()`` is deliberately unchanged by Phase 6: it stayed a pure
projection, so the Phase 3 boundary is still exactly where it was.
Knowledge is a layer that consumes an information set, never one that
rewrites it.
"""

from __future__ import annotations

from durakfish.information.belief import BeliefState
from durakfish.information.deduction import RULES, Observation, deduce
from durakfish.information.information_set import (
    InformationSet,
    PublicDraw,
    PublicEvent,
    observe,
    redact_event,
)
from durakfish.information.knowledge import (
    HIDDEN_LOCATIONS,
    PUBLIC_LOCATIONS,
    CardLocation,
    Certainty,
    KnowledgeError,
    KnowledgeState,
    Placement,
)
from durakfish.information.search_contract import (
    FORBIDDEN_ATTRIBUTES,
    InformationClass,
    SearchContractError,
    SearchInformation,
)
from durakfish.information.tracker import CardTracker, TrackerError
from durakfish.information.worlds import (
    DeterminizedWorld,
    WorldError,
    WorldGenerator,
    is_valid_world,
    validate_world,
)

__all__ = [
    "FORBIDDEN_ATTRIBUTES",
    "HIDDEN_LOCATIONS",
    "PUBLIC_LOCATIONS",
    "RULES",
    "BeliefState",
    "CardLocation",
    "CardTracker",
    "Certainty",
    "DeterminizedWorld",
    "InformationClass",
    "InformationSet",
    "KnowledgeError",
    "KnowledgeState",
    "Observation",
    "Placement",
    "PublicDraw",
    "PublicEvent",
    "SearchContractError",
    "SearchInformation",
    "TrackerError",
    "WorldError",
    "WorldGenerator",
    "deduce",
    "is_valid_world",
    "observe",
    "redact_event",
    "validate_world",
]
