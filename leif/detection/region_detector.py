"""
Environmental Region and Surface Detector.
Manages fixed and detected spatial zones (desk_zone, coffee_station, door_zone).
Computes spatial containment for people and objects.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from leif.types import Entity, EntityCategory, BoundingBox


class RegionDetector:
    """
    Manages known environmental regions and surfaces in a controlled setting,
    or dynamically registers surface regions.
    """

    def __init__(self, predefined_regions: Optional[Dict[str, Tuple[float, float, float, float]]] = None):
        # Default regions for controlled lab/office
        self.regions: Dict[str, Tuple[float, float, float, float]] = predefined_regions or {
            "desk_zone": (40.0, 140.0, 260.0, 330.0),
            "coffee_station": (390.0, 140.0, 600.0, 330.0),
            "door_zone": (540.0, 30.0, 630.0, 160.0),
        }

    def detect_regions(self, timestamp_sec: float, frame_idx: int) -> List[Entity]:
        """Returns environmental region entities for the scene."""
        entities = []
        for region_id, coords in self.regions.items():
            x1, y1, x2, y2 = coords
            entities.append(
                Entity(
                    entity_id=f"region_{region_id}",
                    category=EntityCategory.REGION,
                    class_name=region_id,
                    bbox=BoundingBox(x1, y1, x2, y2, confidence=1.0),
                    confidence=1.0,
                    location_id=region_id,
                    state={"type": "environmental_region"},
                )
            )
        return entities

    def get_enclosing_region(self, point: Tuple[float, float]) -> Optional[str]:
        """Finds which environmental region encloses a given (x, y) point."""
        px, py = point
        for region_id, (rx1, ry1, rx2, ry2) in self.regions.items():
            if rx1 <= px <= rx2 and ry1 <= py <= ry2:
                return region_id
        return None

    def get_region_for_bbox(self, bbox: BoundingBox) -> Optional[str]:
        """Finds the region containing the bottom-center or center of a bounding box."""
        # For people and standing objects, bottom-center (feet/base) is best
        cx = (bbox.x1 + bbox.x2) / 2.0
        cy = bbox.y2  # base of object
        region = self.get_enclosing_region((cx, cy))
        if region:
            return region
        # Fallback to center
        return self.get_enclosing_region(bbox.center)
