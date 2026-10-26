"""
Longitudinal Footprint Aggregator.
Calculates interaction count, total duration, first/last seen, recurrence intervals,
location distribution, interaction frequency, and temporal trends from event memory.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np
from leif.types import EventRecord, FootprintRecord, EnvironmentalChange
from leif.footprint_aggregation.footprint_model import EnvironmentalFootprintReport


class FootprintAggregator:
    """
    Synthesizes discrete longitudinal event memory into a structured Environmental Footprint.
    """

    def __init__(self, time_bucket_seconds: float = 60.0):
        self.time_bucket_seconds = time_bucket_seconds

    def aggregate_footprint(
        self,
        events: List[EventRecord],
        changes: Optional[List[EnvironmentalChange]] = None,
        session_id: str = "session_001",
    ) -> EnvironmentalFootprintReport:
        if not events:
            return EnvironmentalFootprintReport(
                session_id=session_id,
                observation_duration_seconds=0.0,
                total_events_count=0,
            )

        changes = changes or []

        # Sort events chronologically
        sorted_events = sorted(events, key=lambda e: e.timestamp_start)
        min_time = sorted_events[0].timestamp_start
        max_time = max(e.timestamp_end for e in sorted_events)
        total_session_duration = max(0.0, max_time - min_time)

        # Overall interaction distribution
        interaction_dist: Dict[str, int] = defaultdict(int)
        uncertain_count = 0
        confidences = []

        # Group by object
        events_by_object: Dict[str, List[EventRecord]] = defaultdict(list)
        # Group by location
        events_by_location: Dict[str, List[EventRecord]] = defaultdict(list)

        for evt in sorted_events:
            interaction_dist[evt.interaction_type] += 1
            if evt.uncertain_association:
                uncertain_count += 1
            confidences.append(evt.confidence)
            events_by_object[evt.object_id].append(evt)
            events_by_location[evt.location_id].append(evt)

        # Build Object Footprints
        object_footprints: Dict[str, FootprintRecord] = {}
        for obj_id, obj_evts in events_by_object.items():
            object_footprints[obj_id] = self._compute_target_footprint(
                target_id=obj_id,
                target_type="object",
                events=obj_evts,
                changes=[c for c in changes if c.object_id == obj_id],
            )

        # Build Location Footprints
        location_footprints: Dict[str, FootprintRecord] = {}
        for loc_id, loc_evts in events_by_location.items():
            location_footprints[loc_id] = self._compute_target_footprint(
                target_id=loc_id,
                target_type="location",
                events=loc_evts,
                changes=[],
            )

        # Format changes
        changes_summary = [
            {
                "change_id": c.change_id,
                "event_id": c.event_id,
                "object_id": c.object_id,
                "type": c.change_type,
                "description": c.description,
                "confidence": c.confidence,
            }
            for c in changes
        ]

        mean_conf = float(np.mean(confidences)) if confidences else 1.0

        return EnvironmentalFootprintReport(
            session_id=session_id,
            observation_duration_seconds=round(total_session_duration, 2),
            total_events_count=len(sorted_events),
            entity_footprints=object_footprints,
            location_footprints=location_footprints,
            interaction_distribution=dict(interaction_dist),
            observable_changes=changes_summary,
            overall_confidence=round(mean_conf, 3),
            uncertain_events_count=uncertain_count,
        )

    def _compute_target_footprint(
        self,
        target_id: str,
        target_type: str,
        events: List[EventRecord],
        changes: List[EnvironmentalChange],
    ) -> FootprintRecord:
        total_duration = sum(e.duration for e in events)
        first_seen = min(e.timestamp_start for e in events)
        last_seen = max(e.timestamp_end for e in events)

        # Recurrence interval calculation
        recurrence_intervals = []
        for i in range(1, len(events)):
            gap = events[i].timestamp_start - events[i - 1].timestamp_end
            if gap >= 0:
                recurrence_intervals.append(gap)

        mean_recurrence = (
            float(np.mean(recurrence_intervals)) if recurrence_intervals else 0.0
        )

        # Distributions
        loc_freq: Dict[str, int] = defaultdict(int)
        act_freq: Dict[str, int] = defaultdict(int)
        conf_list = []

        # Temporal trends (binned by bucket size)
        bucket_counts: Dict[int, int] = defaultdict(int)

        for e in events:
            loc_freq[e.location_id] += 1
            act_freq[e.interaction_type] += 1
            conf_list.append(e.confidence)
            b_idx = int(e.timestamp_start // self.time_bucket_seconds)
            bucket_counts[b_idx] += 1

        trends = [
            {"time_bucket_min": round((b * self.time_bucket_seconds) / 60.0, 1), "event_count": count}
            for b, count in sorted(bucket_counts.items())
        ]

        obs_changes = [c.description for c in changes]

        return FootprintRecord(
            target_id=target_id,
            target_type=target_type,
            interaction_count=len(events),
            total_duration=round(total_duration, 2),
            first_seen=round(first_seen, 2),
            last_seen=round(last_seen, 2),
            recurrence_interval=round(mean_recurrence, 2),
            location_frequency=dict(loc_freq),
            interaction_frequency=dict(act_freq),
            temporal_trends=trends,
            confidence=round(float(np.mean(conf_list)), 3) if conf_list else 1.0,
            observable_changes=obs_changes,
        )
