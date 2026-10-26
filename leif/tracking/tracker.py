"""
Multi-Object Tracker and Re-Identification Engine.
Assigns persistent IDs, maintains continuity across gaps, and logs uncertain associations.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from leif.types import Entity, EntityCategory, BoundingBox
from leif.tracking.tracklet import Tracklet
from leif.tracking.reid_manager import ReIDManager


class MultiObjectTracker:
    """
    Maintains persistent track IDs for detected entities over long video durations.
    Handles short occlusions, re-identification across gaps, and uncertainty tracking.
    """

    def __init__(
        self,
        max_disappeared_frames: int = 30,
        iou_threshold: float = 0.25,
        max_centroid_distance: float = 120.0,
        reid_manager: Optional[ReIDManager] = None,
    ):
        self.max_disappeared_frames = max_disappeared_frames
        self.iou_threshold = iou_threshold
        self.max_centroid_distance = max_centroid_distance
        self.reid = reid_manager or ReIDManager()

        self.active_tracks: Dict[str, Tracklet] = {}
        self.lost_tracks: Dict[str, Tracklet] = {}
        self.retired_tracks: List[Tracklet] = []

        self._next_track_id: Dict[str, int] = {}

    def _generate_track_id(self, class_name: str) -> str:
        idx = self._next_track_id.get(class_name, 1)
        self._next_track_id[class_name] = idx + 1
        return f"{class_name}_trk_{idx:03d}"

    def update(
        self,
        detections: List[Entity],
        image_rgb: np.ndarray,
        timestamp_sec: float,
        frame_idx: int,
    ) -> List[Tracklet]:
        """
        Updates multi-object tracks with detections in current frame.

        Returns:
            List of currently confirmed active Tracklets.
        """
        # Separate static regions/surfaces from movable entities
        trackable_detections = [
            d for d in detections if d.category in (EntityCategory.PERSON, EntityCategory.OBJECT)
        ]

        # Extract appearance signatures for current detections
        det_signatures = [
            self.reid.extract_appearance_signature(image_rgb, det.bbox)
            for det in trackable_detections
        ]

        matched_tracks = set()
        matched_detections = set()

        h, w = image_rgb.shape[:2]
        effective_max_dist = max(self.max_centroid_distance, 0.25 * max(h, w))

        # ----------------------------------------------------------------------
        # STAGE 1: Spatial Matching with Active Tracks
        # ----------------------------------------------------------------------
        for track_id, track in list(self.active_tracks.items()):
            best_match_idx = None
            best_score = -1.0

            predicted_box = track.predict_bbox(dt=max(0.1, timestamp_sec - track.last_seen_timestamp))

            for det_idx, det in enumerate(trackable_detections):
                if det_idx in matched_detections:
                    continue
                if det.class_name != track.class_name:
                    continue

                # Compute IoU and centroid distance
                iou = predicted_box.iou(det.bbox)
                c1 = predicted_box.center
                c2 = det.bbox.center
                dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])

                if iou >= self.iou_threshold or dist <= effective_max_dist:
                    score = iou + max(0.0, 1.0 - (dist / effective_max_dist))
                    if score > best_score:
                        best_score = score
                        best_match_idx = det_idx

            if best_match_idx is not None:
                matched_det = trackable_detections[best_match_idx]
                matched_sig = det_signatures[best_match_idx]
                track.update(matched_det, timestamp_sec, frame_idx, matched_sig)
                matched_tracks.add(track_id)
                matched_detections.add(best_match_idx)

        # ----------------------------------------------------------------------
        # STAGE 2: Re-Identification for Lost / Disappeared Tracks
        # ----------------------------------------------------------------------
        unmatched_det_indices = [
            i for i in range(len(trackable_detections)) if i not in matched_detections
        ]

        for det_idx in unmatched_det_indices:
            det = trackable_detections[det_idx]
            det_sig = det_signatures[det_idx]
            det_pos = det.bbox.center

            best_lost_track_id = None
            best_reid_sim = -1.0
            best_is_uncertain = False
            best_reason = ""

            for lost_id, lost_track in list(self.lost_tracks.items()):
                if lost_id in matched_tracks:
                    continue
                if lost_track.class_name != det.class_name:
                    continue

                lost_pos = lost_track.current_bbox.center
                lost_sig = lost_track.mean_appearance_signature

                is_match, is_uncertain, sim, reason = self.reid.evaluate_association(
                    det_sig, lost_sig, det_pos, lost_pos
                )

                if is_match and sim > best_reid_sim:
                    best_reid_sim = sim
                    best_lost_track_id = lost_id
                    best_is_uncertain = is_uncertain
                    best_reason = reason

            if best_lost_track_id is not None:
                # Re-associated track! Restore from lost to active
                restored_track = self.lost_tracks.pop(best_lost_track_id)
                restored_track.update(
                    det,
                    timestamp_sec,
                    frame_idx,
                    det_sig,
                    is_uncertain=best_is_uncertain,
                    uncertainty_reason=best_reason,
                )
                self.active_tracks[best_lost_track_id] = restored_track
                matched_tracks.add(best_lost_track_id)
                matched_detections.add(det_idx)

        # ----------------------------------------------------------------------
        # STAGE 3: Handle Unmatched Active Tracks (Move to Lost or Mark Missed)
        # ----------------------------------------------------------------------
        for track_id, track in list(self.active_tracks.items()):
            if track_id not in matched_tracks:
                track.mark_missed()
                if track.missed_frames > 2:
                    # Move to lost tracks pool
                    self.lost_tracks[track_id] = self.active_tracks.pop(track_id)

        # Handle Lost Tracks Ageing
        for lost_id, lost_track in list(self.lost_tracks.items()):
            lost_track.mark_missed()
            if lost_track.missed_frames > self.max_disappeared_frames:
                # Retire track permanently
                self.retired_tracks.append(self.lost_tracks.pop(lost_id))

        # ----------------------------------------------------------------------
        # STAGE 4: Spawn New Tracks for Unmatched Detections
        # ----------------------------------------------------------------------
        for det_idx in range(len(trackable_detections)):
            if det_idx not in matched_detections:
                det = trackable_detections[det_idx]
                det_sig = det_signatures[det_idx]
                new_track_id = self._generate_track_id(det.class_name)
                new_track = Tracklet(
                    track_id=new_track_id,
                    initial_entity=det,
                    timestamp_sec=timestamp_sec,
                    frame_idx=frame_idx,
                    appearance_feature=det_sig,
                )
                self.active_tracks[new_track_id] = new_track

        return list(self.active_tracks.values())
