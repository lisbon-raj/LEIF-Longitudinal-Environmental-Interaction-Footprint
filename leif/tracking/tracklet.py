"""
Tracklet Representation for Multi-Object Tracking.
Maintains state, trajectory history, appearance signature, and uncertainty logs for an entity.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from leif.types import Entity, EntityCategory, BoundingBox


class Tracklet:
    """
    Represents an entity's continuous or re-associated trajectory over time.
    """

    def __init__(
        self,
        track_id: str,
        initial_entity: Entity,
        timestamp_sec: float,
        frame_idx: int,
        appearance_feature: Optional[np.ndarray] = None,
    ):
        self.track_id = track_id
        self.category = initial_entity.category
        self.class_name = initial_entity.class_name
        self.current_bbox = initial_entity.bbox
        self.last_seen_timestamp = timestamp_sec
        self.first_seen_timestamp = timestamp_sec
        self.last_seen_frame = frame_idx
        self.missed_frames = 0
        self.velocity: Tuple[float, float] = (0.0, 0.0) # (vx, vy) in pixels/sec
        self.location_id = initial_entity.location_id
        self.state = dict(initial_entity.state)

        # Trajectory history: (frame_idx, timestamp_sec, bbox, location_id)
        self.history: List[Tuple[int, float, BoundingBox, Optional[str]]] = [
            (frame_idx, timestamp_sec, initial_entity.bbox, initial_entity.location_id)
        ]

        # Appearance signatures: keep recent samples for Re-ID
        self.appearance_signatures: List[np.ndarray] = []
        if appearance_feature is not None:
            self.appearance_signatures.append(appearance_feature)

        # Audit log of uncertain associations
        self.uncertain_associations: List[Dict[str, Any]] = []

    @property
    def total_duration(self) -> float:
        return max(0.0, self.last_seen_timestamp - self.first_seen_timestamp)

    @property
    def mean_appearance_signature(self) -> Optional[np.ndarray]:
        if not self.appearance_signatures:
            return None
        return np.mean(self.appearance_signatures, axis=0)

    def predict_bbox(self, dt: float) -> BoundingBox:
        """Estimates bounding box position based on constant velocity model."""
        vx, vy = self.velocity
        dx = vx * dt
        dy = vy * dt
        return BoundingBox(
            x1=self.current_bbox.x1 + dx,
            y1=self.current_bbox.y1 + dy,
            x2=self.current_bbox.x2 + dx,
            y2=self.current_bbox.y2 + dy,
            confidence=max(0.2, self.current_bbox.confidence - 0.05 * self.missed_frames),
        )

    def update(
        self,
        entity: Entity,
        timestamp_sec: float,
        frame_idx: int,
        appearance_feature: Optional[np.ndarray] = None,
        is_uncertain: bool = False,
        uncertainty_reason: str = "",
    ) -> None:
        """Updates tracklet with matched detection."""
        dt = max(1e-4, timestamp_sec - self.last_seen_timestamp)
        old_cx, old_cy = self.current_bbox.center
        new_cx, new_cy = entity.bbox.center

        # Update smooth velocity
        current_vx = (new_cx - old_cx) / dt
        current_vy = (new_cy - old_cy) / dt
        self.velocity = (
            0.6 * self.velocity[0] + 0.4 * current_vx,
            0.6 * self.velocity[1] + 0.4 * current_vy,
        )

        self.current_bbox = entity.bbox
        self.last_seen_timestamp = timestamp_sec
        self.last_seen_frame = frame_idx
        self.missed_frames = 0
        self.location_id = entity.location_id
        self.state.update(entity.state)

        self.history.append((frame_idx, timestamp_sec, entity.bbox, entity.location_id))

        if appearance_feature is not None:
            self.appearance_signatures.append(appearance_feature)
            if len(self.appearance_signatures) > 10:
                self.appearance_signatures.pop(0)

        if is_uncertain:
            self.uncertain_associations.append({
                "timestamp": timestamp_sec,
                "frame_idx": frame_idx,
                "reason": uncertainty_reason,
            })

    def mark_missed(self) -> None:
        self.missed_frames += 1

    def to_entity(self) -> Entity:
        """Converts current track state to an Entity."""
        return Entity(
            entity_id=self.track_id,
            category=self.category,
            class_name=self.class_name,
            bbox=self.current_bbox,
            confidence=self.current_bbox.confidence,
            track_id=self.track_id,
            location_id=self.location_id,
            state=dict(self.state),
        )
