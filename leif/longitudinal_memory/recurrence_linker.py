"""
Recurrence Linker for Longitudinal Event Memory.
Discovers and links recurring interactions involving the same entities or locations over time.
Prevents repeated actions from being treated as isolated, disconnected events.
"""

from typing import List, Dict, Tuple, Optional
from leif.types import EventRecord


class RecurrenceLinker:
    """
    Tracks historical observations and links new events to prior occurrences of the same
    entity, location, or interaction pattern.
    """

    def __init__(self):
        # Cache of past events: key = (subject_id, object_id, interaction_type) -> list of EventRecords
        self._history_by_pattern: Dict[Tuple[str, str, str], List[EventRecord]] = {}
        # Cache by object_id: key = object_id -> list of EventRecords
        self._history_by_object: Dict[str, List[EventRecord]] = {}

    def link_event(self, event: EventRecord) -> EventRecord:
        """
        Enriches an incoming event with recurrence linkages and index.
        """
        pattern_key = (event.subject_id, event.object_id, event.interaction_type)
        past_pattern_events = self._history_by_pattern.get(pattern_key, [])

        if past_pattern_events:
            event.is_repeated = True
            event.recurrence_index = len(past_pattern_events) + 1
        else:
            # Check if object was previously interacted with in a different way
            past_obj_events = self._history_by_object.get(event.object_id, [])
            if past_obj_events:
                event.is_repeated = True
                event.recurrence_index = len(past_obj_events) + 1
            else:
                event.is_repeated = False
                event.recurrence_index = 1

        # Register event in history caches
        if pattern_key not in self._history_by_pattern:
            self._history_by_pattern[pattern_key] = []
        self._history_by_pattern[pattern_key].append(event)

        if event.object_id not in self._history_by_object:
            self._history_by_object[event.object_id] = []
        self._history_by_object[event.object_id].append(event)

        return event

    def get_recurrence_interval(self, event: EventRecord) -> Optional[float]:
        """
        Computes elapsed seconds since the previous occurrence of this interaction.
        """
        pattern_key = (event.subject_id, event.object_id, event.interaction_type)
        past_events = self._history_by_pattern.get(pattern_key, [])
        if len(past_events) >= 2:
            # Previous event is index -2 (since event itself was just registered)
            prev_event = past_events[-2]
            interval = event.timestamp_start - prev_event.timestamp_end
            return max(0.0, interval)
        return None
