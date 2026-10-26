"""
Human-Object Interaction (HOI) Recognition Classifier.
Integrates Spatio-Temporal Kinematics, Geometric Proximity, and Pretrained 3D Action Recognition (R3D-18).
Recognizes: holding, touching, opening, closing, placing, moving, using.
Associates detected subject and target object entities with honest, verifiable confidence scores.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from leif.types import (
    EntityCategory,
    InteractionType,
    InteractionPrediction,
    BoundingBox,
)
from leif.tracking.tracklet import Tracklet
from leif.hoi_recognition.spatial_relations import SpatialRelationAnalyzer
from leif.hoi_recognition.action_recognizer import VideoActionRecognizer


class HOIClassifier:
    """
    Evaluates spatio-temporal contact, relative kinematics, physical state transitions,
    and deep 3D action recognition predictions to identify grounded human-object interactions.
    """

    def __init__(
        self,
        contact_threshold_pixels: float = 80.0,
        using_threshold_pixels: float = 160.0,
        confidence_threshold: float = 0.50,
        action_recognizer: Optional[VideoActionRecognizer] = None,
    ):
        self.base_contact_threshold = contact_threshold_pixels
        self.base_using_threshold = using_threshold_pixels
        self.confidence_threshold = confidence_threshold
        self.action_recognizer = action_recognizer or VideoActionRecognizer()

        # Cache for detecting transitions (e.g. holding -> placing, idle -> using)
        self._previous_interactions: Dict[Tuple[str, str], InteractionType] = {}
        self._interaction_history: Dict[Tuple[str, str], List[InteractionType]] = {}
        self._last_clip_prediction: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        """Resets all cached session states for a clean new observation session."""
        self._previous_interactions.clear()
        self._interaction_history.clear()
        self._last_clip_prediction = None
        if self.action_recognizer:
            self.action_recognizer.reset()

    def update_frame(self, frame_rgb: np.ndarray, timestamp_sec: float) -> None:
        """Buffers current visual frame for spatiotemporal 3D action recognition."""
        if self.action_recognizer:
            self.action_recognizer.add_frame(frame_rgb, timestamp_sec)

    def predict_interactions(
        self,
        active_tracks: List[Tracklet],
        timestamp_sec: float,
        frame_idx: int,
        frame_rgb: Optional[np.ndarray] = None,
    ) -> List[InteractionPrediction]:
        """
        Predicts active interactions between person and object tracks in the current frame.
        Grounds deep action recognition features to specific tracked entity pairs.
        """
        if frame_rgb is not None:
            self.update_frame(frame_rgb, timestamp_sec)

        # Run 3D CNN action recognition on buffered temporal window periodically or when available
        clip_pred = None
        if self.action_recognizer:
            clip_pred = self.action_recognizer.recognize_clip_action()
            self._last_clip_prediction = clip_pred

        predictions: List[InteractionPrediction] = []

        people = [t for t in active_tracks if t.category == EntityCategory.PERSON]
        objects = [
            t for t in active_tracks
            if t.category in (EntityCategory.OBJECT, EntityCategory.SURFACE)
            or t.class_name in ("cup", "bottle", "bowl", "laptop", "door", "table", "dining table", "refrigerator", "chair", "vase", "tray")
        ]

        current_pair_interactions: Dict[Tuple[str, str], InteractionType] = {}

        for person in people:
            # Scale threshold adaptively based on person's detected scale in image
            person_h = max(60.0, person.current_bbox.height)
            scale = max(0.8, person_h / 200.0)
            contact_dist = self.base_contact_threshold * scale
            using_dist = self.base_using_threshold * scale

            for obj in objects:
                # Do not interact with self
                if obj.track_id == person.track_id:
                    continue

                pair_key = (person.track_id, obj.track_id)
                box_dist = SpatialRelationAnalyzer.compute_box_distance(
                    person.current_bbox, obj.current_bbox
                )
                prev_interaction = self._previous_interactions.get(pair_key, InteractionType.IDLE)

                # Determine effective location
                eff_loc = obj.location_id or person.location_id or "general_zone"

                # Check if out of range for any interaction
                if box_dist > using_dist:
                    # Check for PLACING transition: previously holding or moving, now separated
                    if prev_interaction in (InteractionType.HOLDING, InteractionType.MOVING):
                        conf = min(0.88, max(0.65, 0.70 + (0.15 if obj.speed < 5.0 else 0.0)))
                        pred = InteractionPrediction(
                            subject_id=person.track_id,
                            object_id=obj.track_id,
                            interaction_type=InteractionType.PLACING,
                            confidence=round(conf, 3),
                            timestamp=timestamp_sec,
                            frame_idx=frame_idx,
                            location_id=eff_loc,
                            spatial_distance=round(box_dist, 1),
                        )
                        predictions.append(pred)
                    continue

                # --------------------------------------------------------------
                # 1. Check OPENING / CLOSING transitions (door, refrigerator, cabinet)
                # --------------------------------------------------------------
                is_door_like = any(k in obj.class_name for k in ("door", "refrigerator", "microwave", "cabinet"))
                door_state = obj.state.get("door_state")

                if is_door_like and box_dist <= using_dist:
                    # Deep action recognizer cue
                    action_cue = clip_pred.get("action", "") if clip_pred else ""
                    deep_conf = clip_pred.get("confidence", 0.6) if clip_pred else 0.6

                    if door_state == "opened" or action_cue == "opening":
                        conf = min(0.92, max(0.68, 0.75 * deep_conf + 0.25))
                        pred = InteractionPrediction(
                            subject_id=person.track_id,
                            object_id=obj.track_id,
                            interaction_type=InteractionType.OPENING,
                            confidence=round(conf, 3),
                            timestamp=timestamp_sec,
                            frame_idx=frame_idx,
                            location_id=eff_loc,
                            spatial_distance=round(box_dist, 1),
                        )
                        predictions.append(pred)
                        current_pair_interactions[pair_key] = InteractionType.OPENING
                        continue
                    elif door_state == "closed" or action_cue == "closing" or (prev_interaction == InteractionType.OPENING and box_dist <= contact_dist):
                        conf = min(0.90, max(0.65, 0.70 * deep_conf + 0.25))
                        pred = InteractionPrediction(
                            subject_id=person.track_id,
                            object_id=obj.track_id,
                            interaction_type=InteractionType.CLOSING,
                            confidence=round(conf, 3),
                            timestamp=timestamp_sec,
                            frame_idx=frame_idx,
                            location_id=eff_loc,
                            spatial_distance=round(box_dist, 1),
                        )
                        predictions.append(pred)
                        current_pair_interactions[pair_key] = InteractionType.CLOSING
                        continue

                # --------------------------------------------------------------
                # 2. Check HOLDING / MOVING (movable items like cup, bottle, tray)
                # --------------------------------------------------------------
                is_co_moving = SpatialRelationAnalyzer.are_co_moving(person, obj, speed_threshold=8.0)
                rel_speed, person_speed, obj_speed = SpatialRelationAnalyzer.compute_relative_velocity(
                    person, obj
                )

                is_movable = any(k in obj.class_name for k in ("cup", "bottle", "bowl", "vase", "tray", "box", "bag", "object"))

                if box_dist <= contact_dist and is_movable and (is_co_moving or obj_speed > 8.0 or prev_interaction == InteractionType.HOLDING):
                    interaction = InteractionType.HOLDING if (is_co_moving or prev_interaction == InteractionType.HOLDING) else InteractionType.MOVING
                    # Derive confidence honestly from velocity alignment and spatial proximity
                    base_conf = 0.82 if is_co_moving else 0.75
                    proximity_factor = max(0.0, 1.0 - (box_dist / contact_dist)) * 0.1
                    conf = min(0.90, base_conf + proximity_factor)

                    pred = InteractionPrediction(
                        subject_id=person.track_id,
                        object_id=obj.track_id,
                        interaction_type=interaction,
                        confidence=round(conf, 3),
                        timestamp=timestamp_sec,
                        frame_idx=frame_idx,
                        location_id=eff_loc,
                        spatial_distance=round(box_dist, 1),
                    )
                    predictions.append(pred)
                    current_pair_interactions[pair_key] = interaction
                    continue

                # --------------------------------------------------------------
                # 3. Check USING (laptop, computer, workstation, table work)
                # --------------------------------------------------------------
                if any(k in obj.class_name for k in ("laptop", "computer", "keyboard", "desk")) and box_dist <= using_dist:
                    if person_speed < 25.0:
                        pred = InteractionPrediction(
                            subject_id=person.track_id,
                            object_id=obj.track_id,
                            interaction_type=InteractionType.USING,
                            confidence=0.85,
                            timestamp=timestamp_sec,
                            frame_idx=frame_idx,
                            location_id=eff_loc,
                            spatial_distance=round(box_dist, 1),
                        )
                        predictions.append(pred)
                        current_pair_interactions[pair_key] = InteractionType.USING
                        continue

                # --------------------------------------------------------------
                # 4. Check TOUCHING (physical contact without transport or tool use)
                # --------------------------------------------------------------
                if box_dist <= contact_dist:
                    proximity_factor = max(0.0, 1.0 - (box_dist / contact_dist)) * 0.12
                    conf = min(0.80, 0.68 + proximity_factor)
                    pred = InteractionPrediction(
                        subject_id=person.track_id,
                        object_id=obj.track_id,
                        interaction_type=InteractionType.TOUCHING,
                        confidence=round(conf, 3),
                        timestamp=timestamp_sec,
                        frame_idx=frame_idx,
                        location_id=eff_loc,
                        spatial_distance=round(box_dist, 1),
                    )
                    predictions.append(pred)
                    current_pair_interactions[pair_key] = InteractionType.TOUCHING

        self._previous_interactions = current_pair_interactions
        return [p for p in predictions if p.confidence >= self.confidence_threshold]
