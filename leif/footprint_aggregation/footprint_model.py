"""
Footprint Data Model and Serialization for LEIF.
Encapsulates structured longitudinal footprints for entities, locations, and sessions.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
import json
from leif.types import FootprintRecord, EnvironmentalChange


@dataclass
class EnvironmentalFootprintReport:
    """
    Comprehensive structured report of the Longitudinal Environmental Interaction Footprint.
    """
    session_id: str
    observation_duration_seconds: float
    total_events_count: int
    entity_footprints: Dict[str, FootprintRecord] = field(default_factory=dict)
    location_footprints: Dict[str, FootprintRecord] = field(default_factory=dict)
    interaction_distribution: Dict[str, int] = field(default_factory=dict)
    observable_changes: List[Dict[str, Any]] = field(default_factory=list)
    overall_confidence: float = 1.0
    uncertain_events_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
