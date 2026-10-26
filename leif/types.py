"""
Data models and core type definitions for LEIF.
Strictly maps to the Longitudinal Environmental Interaction Footprint specifications.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple


class EntityCategory(str, Enum):
    PERSON = "person"
    OBJECT = "object"
    SURFACE = "surface"
    REGION = "environmental_region"


class InteractionType(str, Enum):
    HOLDING = "holding"
    TOUCHING = "touching"
    OPENING = "opening"
    CLOSING = "closing"
    PLACING = "placing"
    MOVING = "moving"
    USING = "using"
    IDLE = "idle"


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def iou(self, other: "BoundingBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih

        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


@dataclass
class Entity:
    """Detected person, object, surface, or environmental region."""
    entity_id: str                      # Global or local identifier
    category: EntityCategory            # person, object, surface, region
    class_name: str                     # e.g., 'person', 'cup', 'laptop', 'desk'
    bbox: BoundingBox                   # Spatial location
    confidence: float                   # Detection confidence
    track_id: Optional[str] = None      # Assigned persistent tracking ID
    location_id: Optional[str] = None   # Current environmental region / surface ID
    state: Dict[str, Any] = field(default_factory=dict) # e.g. {'open': False, 'carried': True}
    feature_vector: Optional[List[float]] = None # Embedding for Re-ID


@dataclass
class InteractionPrediction:
    """Predicted interaction relationship between subject and object."""
    subject_id: str
    object_id: str
    interaction_type: InteractionType
    confidence: float
    timestamp: float
    frame_idx: int
    location_id: Optional[str] = None
    spatial_distance: float = 0.0


@dataclass
class SceneGraphNode:
    node_id: str
    label: str
    category: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneGraphEdge:
    source_id: str
    target_id: str
    relation: str                       # e.g., 'holding', 'on_top_of', 'inside'
    confidence: float
    temporal_frame: int


@dataclass
class SpatioTemporalSceneGraph:
    """Dynamic scene graph for an instantaneous timestamp or short window."""
    timestamp: float
    frame_idx: int
    nodes: Dict[str, SceneGraphNode] = field(default_factory=dict)
    edges: List[SceneGraphEdge] = field(default_factory=list)


@dataclass
class EventRecord:
    """
    Structured persistent event memory record.
    Matches the project specification schema.
    """
    event_id: str
    timestamp_start: float
    timestamp_end: float
    subject_id: str
    object_id: str
    location_id: str
    interaction_type: str
    confidence: float
    object_state_before: Dict[str, Any]
    object_state_after: Dict[str, Any]
    source_clip: str
    session_id: str = "session_001"
    is_repeated: bool = False
    recurrence_index: int = 1
    uncertain_association: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.timestamp_end - self.timestamp_start)


@dataclass
class EnvironmentalChange:
    """Recorded observable physical change resulting from interaction."""
    change_id: str
    event_id: str
    object_id: str
    change_type: str                    # 'moved', 'opened', 'closed', 'placed_on'
    description: str
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]
    confidence: float


@dataclass
class FootprintRecord:
    """
    Aggregated Environmental Interaction Footprint summary for an entity,
    location, or the complete scene.
    """
    target_id: str                      # Object or location identifier
    target_type: str                    # 'object' or 'location'
    interaction_count: int
    total_duration: float
    first_seen: float
    last_seen: float
    recurrence_interval: float          # Average seconds between interactions
    location_frequency: Dict[str, int]  # Breakdown by room location
    interaction_frequency: Dict[str, int] # Breakdown by interaction type (holding, touching, etc.)
    temporal_trends: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    observable_changes: List[str] = field(default_factory=list)
