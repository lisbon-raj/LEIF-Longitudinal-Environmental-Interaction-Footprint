"""
Longitudinal Memory Database Engine.
Provides persistent, indexed SQLite storage for all validated LEIF events and changes.
"""

import os
import json
import sqlite3
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
from leif.types import EventRecord, EnvironmentalChange


class LongitudinalMemoryDB:
    """
    Persistent SQLite Database managing LEIF longitudinal event records
    and environmental change histories.
    """

    def __init__(self, db_path: str = "data/memory_store/leif_longitudinal_memory.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Events Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp_start REAL NOT NULL,
                    timestamp_end REAL NOT NULL,
                    duration REAL NOT NULL,
                    subject_id TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    location_id TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    object_state_before TEXT,
                    object_state_after TEXT,
                    source_clip TEXT,
                    is_repeated INTEGER DEFAULT 0,
                    recurrence_index INTEGER DEFAULT 1,
                    uncertain_association INTEGER DEFAULT 0
                )
            """)

            # 2. Environmental Changes Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS environmental_changes (
                    change_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    state_before TEXT,
                    state_after TEXT,
                    confidence REAL NOT NULL,
                    FOREIGN KEY (event_id) REFERENCES events (event_id)
                )
            """)

            # Indices for rapid querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON events (session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_object ON events (object_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_location ON events (location_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_interaction ON events (interaction_type)")
            conn.commit()

    def insert_event(self, event: EventRecord) -> None:
        """Stores a validated event record into persistent memory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO events (
                    event_id, session_id, timestamp_start, timestamp_end, duration,
                    subject_id, object_id, location_id, interaction_type, confidence,
                    object_state_before, object_state_after, source_clip,
                    is_repeated, recurrence_index, uncertain_association
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.session_id,
                event.timestamp_start,
                event.timestamp_end,
                event.duration,
                event.subject_id,
                event.object_id,
                event.location_id,
                event.interaction_type,
                event.confidence,
                json.dumps(event.object_state_before),
                json.dumps(event.object_state_after),
                event.source_clip,
                1 if event.is_repeated else 0,
                event.recurrence_index,
                1 if event.uncertain_association else 0,
            ))
            conn.commit()

    def insert_change(self, change: EnvironmentalChange) -> None:
        """Stores an observable environmental change linked to an event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO environmental_changes (
                    change_id, event_id, object_id, change_type,
                    description, state_before, state_after, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                change.change_id,
                change.event_id,
                change.object_id,
                change.change_type,
                change.description,
                json.dumps(change.state_before),
                json.dumps(change.state_after),
                change.confidence,
            ))
            conn.commit()

    def get_all_events(self, session_id: Optional[str] = None) -> List[EventRecord]:
        """Retrieves events, optionally filtered by observation session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if session_id:
                cursor.execute("SELECT * FROM events WHERE session_id = ? ORDER BY timestamp_start", (session_id,))
            else:
                cursor.execute("SELECT * FROM events ORDER BY timestamp_start")
            rows = cursor.fetchall()
            return [self._row_to_event(r) for r in rows]

    def get_events_by_object(self, object_id: str) -> List[EventRecord]:
        """Retrieves interaction history for a specific tracked object."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events WHERE object_id = ? ORDER BY timestamp_start", (object_id,))
            return [self._row_to_event(r) for r in cursor.fetchall()]

    def get_events_by_location(self, location_id: str) -> List[EventRecord]:
        """Retrieves all interactions that occurred at a specific environmental location."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM events WHERE location_id = ? ORDER BY timestamp_start", (location_id,))
            return [self._row_to_event(r) for r in cursor.fetchall()]

    def get_changes_for_event(self, event_id: str) -> List[EnvironmentalChange]:
        """Retrieves environmental changes triggered by a specific event."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM environmental_changes WHERE event_id = ?", (event_id,))
            changes = []
            for r in cursor.fetchall():
                changes.append(
                    EnvironmentalChange(
                        change_id=r["change_id"],
                        event_id=r["event_id"],
                        object_id=r["object_id"],
                        change_type=r["change_type"],
                        description=r["description"],
                        state_before=json.loads(r["state_before"] or "{}"),
                        state_after=json.loads(r["state_after"] or "{}"),
                        confidence=r["confidence"],
                    )
                )
            return changes

    def _row_to_event(self, row: sqlite3.Row) -> EventRecord:
        return EventRecord(
            event_id=row["event_id"],
            timestamp_start=row["timestamp_start"],
            timestamp_end=row["timestamp_end"],
            subject_id=row["subject_id"],
            object_id=row["object_id"],
            location_id=row["location_id"],
            interaction_type=row["interaction_type"],
            confidence=row["confidence"],
            object_state_before=json.loads(row["object_state_before"] or "{}"),
            object_state_after=json.loads(row["object_state_after"] or "{}"),
            source_clip=row["source_clip"],
            session_id=row["session_id"],
            is_repeated=bool(row["is_repeated"]),
            recurrence_index=row["recurrence_index"],
            uncertain_association=bool(row["uncertain_association"]),
        )
