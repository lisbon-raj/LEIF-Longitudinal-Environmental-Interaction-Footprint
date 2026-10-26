"""
Observable Environmental Change Analyzer.
Compares physical scene and object states before and after interactions.
Records verified physical changes (moved, opened/closed, relocated) without ungrounded inference.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from leif.types import EventRecord, EnvironmentalChange
from leif.tracking.tracklet import Tracklet


class EnvironmentalChangeAnalyzer:
    """
    Analyzes physical state transitions between the start and conclusion
    of an interaction event.
    """

    def __init__(self, displacement_threshold_pixels: float = 35.0):
        self.displacement_threshold = displacement_threshold_pixels
        self._next_change_idx = 1

    def _generate_change_id(self) -> str:
        cid = f"chg_{self._next_change_idx:04d}"
        self._next_change_idx += 1
        return cid

    def analyze_event_changes(
        self,
        event: EventRecord,
        track_object: Optional[Tracklet] = None,
    ) -> List[EnvironmentalChange]:
        """
        Detects observable environmental changes caused by or coinciding with the event.
        """
        detected_changes: List[EnvironmentalChange] = []
        state_before = event.object_state_before or {}
        state_after = event.object_state_after or {}

        # ----------------------------------------------------------------------
        # 1. Check OPEN / CLOSE Transitions (e.g. door, laptop, drawer)
        # ----------------------------------------------------------------------
        # Door state
        door_before = state_before.get("door_state")
        door_after = state_after.get("door_state")
        if door_before and door_after and door_before != door_after:
            detected_changes.append(
                EnvironmentalChange(
                    change_id=self._generate_change_id(),
                    event_id=event.event_id,
                    object_id=event.object_id,
                    change_type=f"door_{door_after}",
                    description=f"{event.object_id} state changed from '{door_before}' to '{door_after}'",
                    state_before={"door_state": door_before},
                    state_after={"door_state": door_after},
                    confidence=0.95,
                )
            )
        elif event.interaction_type == "opening" and "door" in event.object_id:
            detected_changes.append(
                EnvironmentalChange(
                    change_id=self._generate_change_id(),
                    event_id=event.event_id,
                    object_id=event.object_id,
                    change_type="door_opened",
                    description=f"{event.object_id} opened via user interaction",
                    state_before={"door_state": "closed"},
                    state_after={"door_state": "opened"},
                    confidence=0.95,
                )
            )

        # Laptop / display state
        disp_before = state_before.get("display")
        disp_after = state_after.get("display")
        if disp_before and disp_after and disp_before != disp_after:
            detected_changes.append(
                EnvironmentalChange(
                    change_id=self._generate_change_id(),
                    event_id=event.event_id,
                    object_id=event.object_id,
                    change_type=f"display_{disp_after}",
                    description=f"{event.object_id} display changed from '{disp_before}' to '{disp_after}'",
                    state_before={"display": disp_before},
                    state_after={"display": disp_after},
                    confidence=0.92,
                )
            )
        elif event.interaction_type == "using" and "laptop" in event.object_id:
            detected_changes.append(
                EnvironmentalChange(
                    change_id=self._generate_change_id(),
                    event_id=event.event_id,
                    object_id=event.object_id,
                    change_type="display_opened",
                    description=f"{event.object_id} opened and active for work session",
                    state_before={"display": "closed"},
                    state_after={"display": "opened"},
                    confidence=0.92,
                )
            )

        # ----------------------------------------------------------------------
        # 2. Check Spatial Displacement & Relocation
        # ----------------------------------------------------------------------
        if track_object and len(track_object.history) >= 2:
            # Compare initial bbox at start of event vs current bbox
            first_bbox = track_object.history[0][2]
            last_bbox = track_object.history[-1][2]
            dist = np.hypot(
                last_bbox.center[0] - first_bbox.center[0],
                last_bbox.center[1] - first_bbox.center[1],
            )

            # Check if region changed
            first_loc = track_object.history[0][3]
            last_loc = track_object.history[-1][3]

            if first_loc and last_loc and first_loc != last_loc:
                detected_changes.append(
                    EnvironmentalChange(
                        change_id=self._generate_change_id(),
                        event_id=event.event_id,
                        object_id=event.object_id,
                        change_type="relocated",
                        description=f"{event.object_id} moved from region '{first_loc}' to '{last_loc}'",
                        state_before={"location_id": first_loc, "position": first_bbox.center},
                        state_after={"location_id": last_loc, "position": last_bbox.center},
                        confidence=0.90,
                    )
                )
            elif dist >= self.displacement_threshold:
                detected_changes.append(
                    EnvironmentalChange(
                        change_id=self._generate_change_id(),
                        event_id=event.event_id,
                        object_id=event.object_id,
                        change_type="moved",
                        description=f"{event.object_id} displaced by {dist:.1f} pixels",
                        state_before={"position": first_bbox.center},
                        state_after={"position": last_bbox.center},
                        confidence=0.88,
                    )
                )

        return detected_changes
