"""
Event Validation and Deduplication Module.
Consolidates raw frame-level interaction predictions into discrete, continuous,
validated events with confidence, continuity verification, and noise rejection.
"""

from typing import List, Dict, Tuple, Optional, Any
from leif.types import (
    InteractionPrediction,
    EventRecord,
    InteractionType,
)
from leif.tracking.tracklet import Tracklet


class CandidateEvent:
    """Internal state for an ongoing, accumulating interaction event."""

    def __init__(
        self,
        event_id: str,
        subject_id: str,
        object_id: str,
        interaction_type: str,
        timestamp_start: float,
        location_id: str,
        initial_confidence: float,
        initial_object_state: Dict[str, Any],
        source_clip: str,
        uncertain_association: bool = False,
    ):
        self.event_id = event_id
        self.subject_id = subject_id
        self.object_id = object_id
        self.interaction_type = interaction_type
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_start
        self.location_id = location_id
        self.confidences: List[float] = [initial_confidence]
        self.object_state_before = dict(initial_object_state)
        self.object_state_after = dict(initial_object_state)
        self.source_clip = source_clip
        self.last_updated_time = timestamp_start
        self.uncertain_association = uncertain_association

    def update(
        self,
        timestamp_sec: float,
        confidence: float,
        current_location: str,
        current_object_state: Dict[str, Any],
        is_uncertain: bool = False,
    ) -> None:
        self.timestamp_end = timestamp_sec
        self.last_updated_time = timestamp_sec
        self.confidences.append(confidence)
        if current_location:
            self.location_id = current_location
        self.object_state_after.update(current_object_state)
        if is_uncertain:
            self.uncertain_association = True

    @property
    def duration(self) -> float:
        return max(0.0, self.timestamp_end - self.timestamp_start)

    @property
    def mean_confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0

    def to_event_record(self, session_id: str = "session_001") -> EventRecord:
        return EventRecord(
            event_id=self.event_id,
            timestamp_start=round(self.timestamp_start, 2),
            timestamp_end=round(self.timestamp_end, 2),
            subject_id=self.subject_id,
            object_id=self.object_id,
            location_id=self.location_id or "unknown_zone",
            interaction_type=self.interaction_type,
            confidence=round(self.mean_confidence, 3),
            object_state_before=self.object_state_before,
            object_state_after=self.object_state_after,
            source_clip=self.source_clip,
            session_id=session_id,
            uncertain_association=self.uncertain_association,
        )


class EventValidator:
    """
    Validates interactions across temporal continuity windows, enforces minimum duration,
    filters transient noise, and deduplicates frame observations.
    """

    def __init__(
        self,
        min_duration_seconds: float = 0.5,
        max_temporal_gap_seconds: float = 2.0,
        min_confidence: float = 0.50,
        session_id: str = "session_001",
    ):
        self.min_duration = min_duration_seconds
        self.max_gap = max_temporal_gap_seconds
        self.min_confidence = min_confidence
        self.session_id = session_id

        self._active_candidates: Dict[Tuple[str, str, str], CandidateEvent] = {}
        self._next_event_idx = 1

    def _generate_event_id(self) -> str:
        eid = f"evt_{self._next_event_idx:04d}"
        self._next_event_idx += 1
        return eid

    def process_frame_interactions(
        self,
        predictions: List[InteractionPrediction],
        active_tracks: List[Tracklet],
        timestamp_sec: float,
        source_clip: str = "clip_0000",
    ) -> List[EventRecord]:
        """
        Processes current frame interactions.
        Returns any events that have concluded and satisfied validation criteria.
        """
        completed_events: List[EventRecord] = []
        track_map = {t.track_id: t for t in active_tracks}

        seen_keys = set()

        for pred in predictions:
            key = (pred.subject_id, pred.object_id, pred.interaction_type.value)
            seen_keys.add(key)

            target_track = track_map.get(pred.object_id)
            current_state = dict(target_track.state) if target_track else {}
            loc_id = pred.location_id or (target_track.location_id if target_track else "unknown")

            # Check if subject or object had uncertain association
            is_uncertain = False
            if target_track and target_track.uncertain_associations:
                is_uncertain = True
            subj_track = track_map.get(pred.subject_id)
            if subj_track and subj_track.uncertain_associations:
                is_uncertain = True

            if key not in self._active_candidates:
                # Spawn new candidate event
                self._active_candidates[key] = CandidateEvent(
                    event_id=self._generate_event_id(),
                    subject_id=pred.subject_id,
                    object_id=pred.object_id,
                    interaction_type=pred.interaction_type.value,
                    timestamp_start=timestamp_sec,
                    location_id=loc_id,
                    initial_confidence=pred.confidence,
                    initial_object_state=current_state,
                    source_clip=source_clip,
                    uncertain_association=is_uncertain,
                )
            else:
                # Extend existing candidate event
                self._active_candidates[key].update(
                    timestamp_sec=timestamp_sec,
                    confidence=pred.confidence,
                    current_location=loc_id,
                    current_object_state=current_state,
                    is_uncertain=is_uncertain,
                )

        # Check for candidates that have ceased / timed out (gap > max_gap)
        for key, candidate in list(self._active_candidates.items()):
            if key not in seen_keys:
                gap = timestamp_sec - candidate.last_updated_time
                if gap > self.max_gap:
                    # Finalize candidate
                    del self._active_candidates[key]
                    if (
                        candidate.duration >= self.min_duration
                        and candidate.mean_confidence >= self.min_confidence
                    ):
                        completed_events.append(candidate.to_event_record(self.session_id))

        return completed_events

    def finalize_remaining(self, final_timestamp: float) -> List[EventRecord]:
        """Finalizes any active candidates remaining at end of video."""
        completed_events: List[EventRecord] = []
        for key, candidate in list(self._active_candidates.items()):
            if (
                candidate.duration >= self.min_duration
                and candidate.mean_confidence >= self.min_confidence
            ):
                completed_events.append(candidate.to_event_record(self.session_id))
        self._active_candidates.clear()
        return completed_events
