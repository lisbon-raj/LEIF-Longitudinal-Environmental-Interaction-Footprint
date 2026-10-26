"""
Spatial Relations and Geometric Proximity Utilities for HOI.
Calculates distance, relative velocity, and contact between person and object bounding boxes.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from leif.types import BoundingBox
from leif.tracking.tracklet import Tracklet


class SpatialRelationAnalyzer:
    """
    Computes spatial proximity, bounding box overlap, and velocity co-movement
    between person and object tracklets.
    """

    @staticmethod
    def compute_box_distance(box1: BoundingBox, box2: BoundingBox) -> float:
        """
        Computes minimum Euclidean distance between the edges of two axis-aligned bounding boxes.
        Returns 0.0 if boxes intersect.
        """
        # Horizontal distance
        if box1.x2 < box2.x1:
            dx = box2.x1 - box1.x2
        elif box2.x2 < box1.x1:
            dx = box1.x1 - box2.x2
        else:
            dx = 0.0

        # Vertical distance
        if box1.y2 < box2.y1:
            dy = box2.y1 - box1.y2
        elif box2.y2 < box1.y1:
            dy = box1.y1 - box2.y2
        else:
            dy = 0.0

        return float(np.hypot(dx, dy))

    @staticmethod
    def compute_centroid_distance(box1: BoundingBox, box2: BoundingBox) -> float:
        c1 = box1.center
        c2 = box2.center
        return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))

    @staticmethod
    def compute_relative_velocity(
        track_person: Tracklet, track_object: Tracklet
    ) -> Tuple[float, float, float]:
        """
        Computes velocity alignment and relative speed between two tracklets.

        Returns:
            (relative_speed, person_speed, object_speed)
        """
        pvx, pvy = track_person.velocity
        ovx, ovy = track_object.velocity

        person_speed = float(np.hypot(pvx, pvy))
        object_speed = float(np.hypot(ovx, ovy))
        rel_speed = float(np.hypot(pvx - ovx, pvy - ovy))

        return rel_speed, person_speed, object_speed

    @staticmethod
    def are_co_moving(
        track_person: Tracklet, track_object: Tracklet, speed_threshold: float = 10.0
    ) -> bool:
        """
        Returns True if the object and person are moving with similar speed and direction (e.g. carried).
        """
        rel_speed, person_speed, object_speed = SpatialRelationAnalyzer.compute_relative_velocity(
            track_person, track_object
        )
        if person_speed > speed_threshold and object_speed > speed_threshold:
            # Both moving, check if relative speed is much smaller than individual speeds
            return rel_speed < 0.5 * max(person_speed, object_speed)
        return False
