"""
Longitudinal Session Comparator.
Compares interaction patterns, entity footprints, and environmental usage
across multiple sessions or increasing observation durations (1m, 5m, 15m, 1h).
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
from leif.footprint_aggregation.footprint_model import EnvironmentalFootprintReport


class LongitudinalSessionComparator:
    """
    Compares footprint records across multiple observation sessions or varying durations.
    """

    def compare_sessions(
        self, session_reports: List[EnvironmentalFootprintReport]
    ) -> Dict[str, Any]:
        """
        Computes comparative drift, interaction frequency changes, and duration growth across sessions.
        """
        if not session_reports:
            return {"sessions_count": 0, "comparison": {}}

        session_ids = [r.session_id for r in session_reports]
        durations = [r.observation_duration_seconds for r in session_reports]
        event_counts = [r.total_events_count for r in session_reports]

        # Entity interaction frequencies across sessions
        all_entities = set()
        for r in session_reports:
            all_entities.update(r.entity_footprints.keys())

        entity_trends: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in session_reports:
            for ent in all_entities:
                if ent in r.entity_footprints:
                    fp = r.entity_footprints[ent]
                    entity_trends[ent].append({
                        "session_id": r.session_id,
                        "count": fp.interaction_count,
                        "duration": fp.total_duration,
                        "recurrence_interval": fp.recurrence_interval,
                    })
                else:
                    entity_trends[ent].append({
                        "session_id": r.session_id,
                        "count": 0,
                        "duration": 0.0,
                        "recurrence_interval": 0.0,
                    })

        # Regional usage comparison across sessions
        all_locations = set()
        for r in session_reports:
            all_locations.update(r.location_footprints.keys())

        location_trends: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in session_reports:
            for loc in all_locations:
                if loc in r.location_footprints:
                    lfp = r.location_footprints[loc]
                    location_trends[loc].append({
                        "session_id": r.session_id,
                        "events": lfp.interaction_count,
                        "duration": lfp.total_duration,
                    })
                else:
                    location_trends[loc].append({
                        "session_id": r.session_id,
                        "events": 0,
                        "duration": 0.0,
                    })

        return {
            "sessions_count": len(session_reports),
            "session_ids": session_ids,
            "observation_durations_sec": durations,
            "total_events_per_session": event_counts,
            "entity_trends": dict(entity_trends),
            "location_trends": dict(location_trends),
        }
