"""
Pretrained Vision Model Detector Adapter.
Wraps popular backbones (Ultralytics YOLO, torchvision Faster R-CNN) with graceful fallback.
"""

from typing import List, Optional
import numpy as np
from leif.detection.base_detector import BaseDetector
from leif.detection.color_spatial_detector import ColorSpatialDetector
from leif.detection.region_detector import RegionDetector
from leif.types import Entity, EntityCategory, BoundingBox


class PretrainedDetector(BaseDetector):
    """
    Adapter for standard pretrained object detectors (Ultralytics YOLOv8).
    Falls back gracefully to ColorSpatialDetector if synthetic video is loaded or deep learning packages fail.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.35,
        region_detector: Optional[RegionDetector] = None,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.region_detector = region_detector or RegionDetector()
        self.model = None
        self._fallback = ColorSpatialDetector(self.region_detector)
        self._init_model()

    def _init_model(self) -> None:
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
        except Exception:
            self.model = None

    def detect(
        self,
        image_rgb: np.ndarray,
        timestamp_sec: float,
        frame_idx: int,
    ) -> List[Entity]:
        # Always include environmental regions
        region_entities = self.region_detector.detect_regions(timestamp_sec, frame_idx)

        if self.model is None:
            return self._fallback.detect(image_rgb, timestamp_sec, frame_idx)

        try:
            h, w, _ = image_rgb.shape
            results = self.model(image_rgb, verbose=False)[0]
            entities = list(region_entities)

            has_person = False
            for i, box in enumerate(results.boxes):
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue

                cls_id = int(box.cls[0])
                raw_name = results.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Map COCO misclassifications for door panels/frames
                cls_name = raw_name
                if raw_name == "surfboard" and (y2 - y1) / max(1.0, (x2 - x1)) > 1.3:
                    cls_name = "door"
                elif raw_name in ("refrigerator", "microwave"):
                    # Appliances with doors
                    cls_name = "door"

                bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)
                loc_id = self.region_detector.get_region_for_bbox(bbox)

                if cls_name == "person":
                    category = EntityCategory.PERSON
                    has_person = True
                elif cls_name in ("table", "dining table", "desk"):
                    category = EntityCategory.SURFACE
                else:
                    category = EntityCategory.OBJECT

                entities.append(
                    Entity(
                        entity_id=f"det_{cls_name}_{frame_idx}_{i}",
                        category=category,
                        class_name=cls_name,
                        bbox=bbox,
                        confidence=conf,
                        location_id=loc_id,
                    )
                )

            # If YOLO detected NO person (e.g. synthetic geometric office demo), check fallback
            if not has_person:
                fallback_ents = self._fallback.detect(image_rgb, timestamp_sec, frame_idx)
                fallback_tracks = [e for e in fallback_ents if e.category in (EntityCategory.PERSON, EntityCategory.OBJECT)]
                if fallback_tracks:
                    entities.extend(fallback_tracks)

            return entities
        except Exception:
            return self._fallback.detect(image_rgb, timestamp_sec, frame_idx)
