"""
Automated Longitudinal Summary and Report Generator.
Generates human-readable narratives, Before/After state change tables,
confidence-aware event audits, and session comparison summaries.
"""

import os
from typing import List, Dict, Any, Optional
from leif.types import EventRecord, EnvironmentalChange
from leif.footprint_aggregation.footprint_model import EnvironmentalFootprintReport


class LongitudinalSummaryGenerator:
    """
    Synthesizes event memory and footprints into human-readable narratives
    and formatted markdown/HTML audit tables.
    """

    def generate_markdown_report(
        self,
        report: EnvironmentalFootprintReport,
        events: List[EventRecord],
        changes: List[EnvironmentalChange],
        output_path: str = "outputs/summaries/longitudinal_summary_report.md",
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        lines: List[str] = []
        lines.append("# LEIF: Longitudinal Environmental Interaction Footprint Report\n")
        lines.append(f"**Session Identifier:** `{report.session_id}`  ")
        lines.append(f"**Total Observation Duration:** {report.observation_duration_seconds:.1f} seconds  ")
        lines.append(f"**Total Verified Events:** {report.total_events_count}  ")
        lines.append(f"**Overall Pipeline Confidence:** {report.overall_confidence * 100:.1f}%  ")
        lines.append(f"**Uncertain Associations Flagged:** {report.uncertain_events_count}\n")

        # ----------------------------------------------------------------------
        # H. Automated Textual Summary
        # ----------------------------------------------------------------------
        lines.append("## 1. Automated Longitudinal Executive Summary\n")
        lines.append(self._generate_narrative(report, events, changes))
        lines.append("\n")

        # ----------------------------------------------------------------------
        # E. Before / After Environmental State Summary
        # ----------------------------------------------------------------------
        lines.append("## 2. Observable Environmental State Change Summary\n")
        if changes:
            lines.append("| Change ID | Event ID | Object | Change Type | Description | Confidence |")
            lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for c in changes:
                lines.append(
                    f"| `{c.change_id}` | `{c.event_id}` | **{c.object_id}** | `{c.change_type}` | {c.description} | {c.confidence:.2f} |"
                )
        else:
            lines.append("*No permanent observable environmental state changes detected during this session.*")
        lines.append("\n")

        # ----------------------------------------------------------------------
        # F. Confidence-Aware Event List
        # ----------------------------------------------------------------------
        lines.append("## 3. Confidence-Aware Longitudinal Event Audit Table\n")
        lines.append("| Event ID | Start (s) | End (s) | Dur (s) | Subject | Object | Location | Interaction | Conf | Status |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for e in events:
            status = "⚠️ UNCERTAIN" if e.uncertain_association else "✅ Confirmed"
            rep_str = f" (x{e.recurrence_index})" if e.is_repeated else ""
            lines.append(
                f"| `{e.event_id}` | {e.timestamp_start:.2f} | {e.timestamp_end:.2f} | {e.duration:.2f} | "
                f"`{e.subject_id}` | **{e.object_id}** | `{e.location_id}` | `{e.interaction_type}`{rep_str} | "
                f"{e.confidence:.2f} | {status} |"
            )
        lines.append("\n")

        # ----------------------------------------------------------------------
        # Entity Footprint Breakdown
        # ----------------------------------------------------------------------
        lines.append("## 4. Entity-Level Environmental Interaction Footprints\n")
        lines.append("| Entity ID | Type | Count | Total Dur (s) | First Seen | Last Seen | Recurrence Int (s) | Primary Location |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for ent_id, fp in report.entity_footprints.items():
            primary_loc = max(fp.location_frequency.items(), key=lambda x: x[1])[0] if fp.location_frequency else "N/A"
            rec_str = f"{fp.recurrence_interval:.1f}s" if fp.recurrence_interval > 0 else "N/A (single)"
            lines.append(
                f"| **{ent_id}** | {fp.target_type} | {fp.interaction_count} | {fp.total_duration:.1f} | "
                f"{fp.first_seen:.1f}s | {fp.last_seen:.1f}s | {rec_str} | `{primary_loc}` |"
            )
        lines.append("\n")

        # ----------------------------------------------------------------------
        # Regional Utilization Footprint
        # ----------------------------------------------------------------------
        lines.append("## 5. Regional Environmental Utilization\n")
        lines.append("| Environmental Region | Total Events | Accumulated Dwell (s) | First Seen | Last Seen |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for loc_id, fp in report.location_footprints.items():
            lines.append(
                f"| **{loc_id}** | {fp.interaction_count} | {fp.total_duration:.1f}s | "
                f"{fp.first_seen:.1f}s | {fp.last_seen:.1f}s |"
            )
        lines.append("\n")

        content = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_path

    def _generate_narrative(
        self,
        report: EnvironmentalFootprintReport,
        events: List[EventRecord],
        changes: List[EnvironmentalChange],
    ) -> str:
        """Constructs an explainable textual narrative."""
        if not events:
            return "No interactions were detected during this observation period."

        most_active_obj = max(
            report.entity_footprints.items(), key=lambda x: x[1].total_duration
        )[0] if report.entity_footprints else "None"

        most_used_loc = max(
            report.location_footprints.items(), key=lambda x: x[1].total_duration
        )[0] if report.location_footprints else "None"

        narrative = (
            f"During this observation session of {report.observation_duration_seconds:.1f} seconds, "
            f"a total of {report.total_events_count} verified interaction events were detected across "
            f"{len(report.entity_footprints)} unique entities. "
            f"The primary zone of human activity was **{most_used_loc}**, "
            f"with the most extensively utilized object being **{most_active_obj}** "
            f"(accumulating {report.entity_footprints[most_active_obj].total_duration:.1f} seconds of interaction). "
        )

        if changes:
            narrative += (
                f"The system recorded {len(changes)} observable environmental state change(s): "
                + "; ".join(c.description for c in changes) + ". "
            )
        else:
            narrative += "No physical displacement or state transitions remained after the interactions. "

        repeated_count = sum(1 for e in events if e.is_repeated)
        if repeated_count > 0:
            narrative += (
                f"Longitudinal memory successfully linked {repeated_count} repeated interaction(s), "
                f"demonstrating recurrent behavioral patterns rather than disconnected observations."
            )

        return narrative
