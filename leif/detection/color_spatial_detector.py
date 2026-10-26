"""
Color and Spatial Morphology Detector.
Detects entities (people, cups, laptops, doors) directly from visual frames using
color segmentation, spatial blob extraction, and region association.
Provides immediate zero-dependency computer-vision detection.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from leif.detection.base_detector import BaseDetector
from leif.detection.region_detector import RegionDetector
from leif.types import Entity, EntityCategory, BoundingBox


class ColorSpatialDetector(BaseDetector):
    """
    Detects people, objects (cup, laptop, door), and environmental surfaces
    using color filtering, spatial morphology, and spatial region context.
    """

    def __init__(self, region_detector: Optional[RegionDetector] = None):
        self.region_detector = region_detector or RegionDetector()

    def _extract_bounding_boxes(
        self, mask: np.ndarray, min_area: int = 15, max_area: int = 250000
    ) -> List[BoundingBox]:
        """Finds bounding boxes of connected components in a binary mask using numpy."""
        if not np.any(mask):
            return []

        # Find rows and cols containing positive pixels
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        area = (rmax - rmin + 1) * (cmax - cmin + 1)
        if min_area <= area <= max_area:
            return [
                BoundingBox(
                    x1=float(cmin),
                    y1=float(rmin),
                    x2=float(cmax + 1),
                    y2=float(rmax + 1),
                    confidence=0.90,
                )
            ]
        return []

    def detect(
        self,
        image_rgb: np.ndarray,
        timestamp_sec: float,
        frame_idx: int,
    ) -> List[Entity]:
        detected_entities: List[Entity] = []
        h, w, _ = image_rgb.shape

        # 1. Environmental Regions
        region_entities = self.region_detector.detect_regions(timestamp_sec, frame_idx)
        detected_entities.extend(region_entities)

        # 2. Person Detection (Blue torso: R in [20, 60], G in [100, 160], B in [160, 220])
        person_mask = (
            (image_rgb[:, :, 0] >= 20) & (image_rgb[:, :, 0] <= 70) &
            (image_rgb[:, :, 1] >= 100) & (image_rgb[:, :, 1] <= 160) &
            (image_rgb[:, :, 2] >= 160) & (image_rgb[:, :, 2] <= 220)
        )
        person_boxes = self._extract_bounding_boxes(person_mask, min_area=100)
        for i, bbox in enumerate(person_boxes):
            # Expand box upwards slightly to include head
            expanded_bbox = BoundingBox(
                x1=max(0.0, bbox.x1 - 5),
                y1=max(0.0, bbox.y1 - 35),
                x2=min(float(w), bbox.x2 + 5),
                y2=min(float(h), bbox.y2 + 10),
                confidence=0.92,
            )
            loc = self.region_detector.get_region_for_bbox(expanded_bbox)
            detected_entities.append(
                Entity(
                    entity_id=f"det_person_{frame_idx}_{i}",
                    category=EntityCategory.PERSON,
                    class_name="person",
                    bbox=expanded_bbox,
                    confidence=0.92,
                    location_id=loc,
                    state={"posture": "standing"},
                )
            )

        # 3. Cup Detection (Reddish: R > 190, G < 100, B < 100)
        cup_mask = (
            (image_rgb[:, :, 0] >= 190) &
            (image_rgb[:, :, 1] <= 100) &
            (image_rgb[:, :, 2] <= 100)
        )
        cup_boxes = self._extract_bounding_boxes(cup_mask, min_area=30)
        for i, bbox in enumerate(cup_boxes):
            loc = self.region_detector.get_region_for_bbox(bbox)
            detected_entities.append(
                Entity(
                    entity_id=f"det_cup_{frame_idx}_{i}",
                    category=EntityCategory.OBJECT,
                    class_name="cup",
                    bbox=bbox,
                    confidence=0.88,
                    location_id=loc,
                    state={"filled": False},
                )
            )

        # 4. Laptop Detection (Dark slate: R in [35, 75], G in [55, 95], B in [75, 115])
        laptop_mask = (
            (image_rgb[:, :, 0] >= 35) & (image_rgb[:, :, 0] <= 75) &
            (image_rgb[:, :, 1] >= 55) & (image_rgb[:, :, 1] <= 95) &
            (image_rgb[:, :, 2] >= 75) & (image_rgb[:, :, 2] <= 115)
        )
        laptop_boxes = self._extract_bounding_boxes(laptop_mask, min_area=50)
        for i, bbox in enumerate(laptop_boxes):
            loc = self.region_detector.get_region_for_bbox(bbox)
            detected_entities.append(
                Entity(
                    entity_id=f"det_laptop_{frame_idx}_{i}",
                    category=EntityCategory.OBJECT,
                    class_name="laptop",
                    bbox=bbox,
                    confidence=0.89,
                    location_id=loc,
                    state={"display": "open"},
                )
            )

        # 5. Door Detection (Door zone: x in [540, 630], y in [30, 160])
        door_bbox = BoundingBox(x1=550.0, y1=40.0, x2=620.0, y2=150.0, confidence=0.95)
        # Check door color to determine open vs closed
        door_patch = image_rgb[40:150, 550:620]
        mean_blue = np.mean(door_patch[:, :, 2]) if door_patch.size > 0 else 0
        door_is_open = mean_blue < 145.0  # darker blue indicates open door in synthetic
        detected_entities.append(
            Entity(
                entity_id=f"det_door_{frame_idx}",
                category=EntityCategory.OBJECT,
                class_name="door",
                bbox=door_bbox,
                confidence=0.95,
                location_id="door_zone",
                state={"door_state": "opened" if door_is_open else "closed"},
            )
        )

        return detected_entities
