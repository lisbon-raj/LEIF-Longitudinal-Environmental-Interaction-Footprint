"""
Unit and Integration Tests for Phase 8 (Footprint Aggregation) and Phase 9 (Visualization).
"""

import os
from leif.types import EventRecord, EnvironmentalChange
from leif.footprint_aggregation import FootprintAggregator
from leif.visualization import (
    TimelinePlotter,
    InteractionGraphVisualizer,
    EnvironmentHeatmapGenerator,
    InteractionChartGenerator,
    LongitudinalSummaryGenerator,
)


def test_footprint_and_visualization_suite():
    # 1. Prepare sample events
    events = [
        EventRecord(
            event_id="evt_0001",
            timestamp_start=2.0,
            timestamp_end=7.0,
            subject_id="person_P1",
            object_id="laptop_01",
            location_id="desk_zone",
            interaction_type="using",
            confidence=0.92,
            object_state_before={"display": "closed"},
            object_state_after={"display": "open"},
            source_clip="clip_0000",
            session_id="session_001",
            is_repeated=False,
            recurrence_index=1,
        ),
        EventRecord(
            event_id="evt_0002",
            timestamp_start=8.0,
            timestamp_end=12.0,
            subject_id="person_P1",
            object_id="cup_01",
            location_id="desk_zone",
            interaction_type="holding",
            confidence=0.89,
            object_state_before={"location_id": "desk_zone"},
            object_state_after={"location_id": "coffee_station"},
            source_clip="clip_0001",
            session_id="session_001",
            is_repeated=False,
            recurrence_index=1,
        ),
        EventRecord(
            event_id="evt_0003",
            timestamp_start=15.0,
            timestamp_end=18.0,
            subject_id="person_P1",
            object_id="cup_01",
            location_id="coffee_station",
            interaction_type="holding",
            confidence=0.86,
            object_state_before={"location_id": "coffee_station"},
            object_state_after={"location_id": "coffee_station"},
            source_clip="clip_0002",
            session_id="session_001",
            is_repeated=True,
            recurrence_index=2,
        ),
        EventRecord(
            event_id="evt_0004",
            timestamp_start=19.0,
            timestamp_end=22.0,
            subject_id="person_P1",
            object_id="door_01",
            location_id="door_zone",
            interaction_type="opening",
            confidence=0.95,
            object_state_before={"door_state": "closed"},
            object_state_after={"door_state": "opened"},
            source_clip="clip_0003",
            session_id="session_001",
            is_repeated=False,
            recurrence_index=1,
            uncertain_association=True,  # Test uncertain flag
        ),
    ]

    changes = [
        EnvironmentalChange(
            change_id="chg_0001",
            event_id="evt_0002",
            object_id="cup_01",
            change_type="relocated",
            description="cup_01 moved from region 'desk_zone' to 'coffee_station'",
            state_before={"location_id": "desk_zone"},
            state_after={"location_id": "coffee_station"},
            confidence=0.90,
        ),
        EnvironmentalChange(
            change_id="chg_0002",
            event_id="evt_0004",
            object_id="door_01",
            change_type="door_opened",
            description="door_01 state changed from 'closed' to 'opened'",
            state_before={"door_state": "closed"},
            state_after={"door_state": "opened"},
            confidence=0.95,
        ),
    ]

    # 2. Test Footprint Aggregator
    aggregator = FootprintAggregator()
    report = aggregator.aggregate_footprint(events, changes, session_id="session_001")

    assert report.total_events_count == 4
    assert report.uncertain_events_count == 1
    assert "cup_01" in report.entity_footprints
    cup_fp = report.entity_footprints["cup_01"]
    assert cup_fp.interaction_count == 2
    assert abs(cup_fp.total_duration - 7.0) < 0.1  # 4.0s + 3.0s
    assert cup_fp.recurrence_interval == 3.0  # 15.0 - 12.0 = 3.0s
    assert "coffee_station" in report.location_footprints

    # 3. Test Visualizations
    out_dir = "outputs/tests"
    os.makedirs(out_dir, exist_ok=True)

    # A. Timeline Plotter
    timeline_path = os.path.join(out_dir, "test_timeline.png")
    TimelinePlotter().plot_timeline(events, output_path=timeline_path)
    assert os.path.exists(timeline_path)
    assert os.path.getsize(timeline_path) > 1000

    # B. Graph Visualizer
    graph_path = os.path.join(out_dir, "test_graph.png")
    InteractionGraphVisualizer().plot_graph(events, output_path=graph_path)
    assert os.path.exists(graph_path)
    assert os.path.getsize(graph_path) > 1000

    # C. Heatmap Generator
    heatmap_path = os.path.join(out_dir, "test_heatmap.png")
    EnvironmentHeatmapGenerator().plot_heatmap(events, output_path=heatmap_path)
    assert os.path.exists(heatmap_path)
    assert os.path.getsize(heatmap_path) > 1000

    # D. Chart Generator
    chart_path = os.path.join(out_dir, "test_charts.png")
    InteractionChartGenerator().plot_summary_charts(events, output_path=chart_path)
    assert os.path.exists(chart_path)
    assert os.path.getsize(chart_path) > 1000

    # E & H. Summary Markdown Report
    report_path = os.path.join(out_dir, "test_report.md")
    LongitudinalSummaryGenerator().generate_markdown_report(report, events, changes, output_path=report_path)
    assert os.path.exists(report_path)
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Automated Longitudinal Executive Summary" in content
    assert "Observable Environmental State Change Summary" in content
    assert "UNCERTAIN" in content  # Flags uncertain event

    print("Phase 8 (Footprint Aggregation) & Phase 9 (Visualization) Tests Passed Successfully!")


if __name__ == "__main__":
    test_footprint_and_visualization_suite()
