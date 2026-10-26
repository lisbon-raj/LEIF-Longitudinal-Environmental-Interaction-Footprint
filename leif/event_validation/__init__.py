"""
Module 6: Event Validation and Deduplication.
Ensures temporal continuity, attaches confidence, filters noise, and consolidates discrete events.
"""

from leif.event_validation.validator import EventValidator, CandidateEvent

__all__ = [
    "EventValidator",
    "CandidateEvent",
]
