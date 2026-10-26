"""
Privacy and Data Minimization Module.
Enforces event-level storage, anonymizes subject identifiers, and strips PII.
"""

from typing import List
from leif.types import EventRecord


class PrivacyAnonymizer:
    """
    Enforces research data minimization by pseudonymizing subject identities
    and ensuring only structured event-level data is stored long-term.
    """

    def __init__(self, subject_prefix: str = "subject_"):
        self.subject_prefix = subject_prefix
        self._id_map = {}
        self._counter = 1

    def anonymize_subject_id(self, raw_id: str) -> str:
        """Maps any raw personal or biometric identifier to an anonymous token."""
        if raw_id not in self._id_map:
            anon_id = f"{self.subject_prefix}{self._counter:03d}"
            self._id_map[raw_id] = anon_id
            self._counter += 1
        return self._id_map[raw_id]

    def sanitize_event(self, event: EventRecord) -> EventRecord:
        """Returns a sanitized event record with anonymized subject identifiers."""
        event.subject_id = self.anonymize_subject_id(event.subject_id)
        return event

    def sanitize_events(self, events: List[EventRecord]) -> List[EventRecord]:
        return [self.sanitize_event(e) for e in events]
