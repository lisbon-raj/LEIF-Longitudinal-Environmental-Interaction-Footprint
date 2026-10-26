"""
Module 3: Multi-Object Tracking & Re-Identification.
Maintains persistent identities across frames and occlusion gaps.
"""

from leif.tracking.tracklet import Tracklet
from leif.tracking.reid_manager import ReIDManager
from leif.tracking.tracker import MultiObjectTracker

__all__ = [
    "Tracklet",
    "ReIDManager",
    "MultiObjectTracker",
]
