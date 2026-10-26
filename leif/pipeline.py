"""
Master LEIF Pipeline Orchestrator.
Coordinates the end-to-end execution loop:
Video Input -> Frame Sampling -> Detection -> Tracking -> HOI Recognition
-> Scene Graph -> Event Validation -> Longitudinal Memory -> Change Analysis
-> Footprint Aggregation -> Visualizations & Reports.
"""

import os
from typing import Dict, Any, Optional, List
from leif.video_acquisition import VideoSource, FrameSampler, VideoMetadata
from leif.detection import RegionDetector, ColorSpatialDetector, PretrainedDetector
from leif.tracking import MultiObjectTracker, ReIDManager
from leif.hoi_recognition import HOIClassifier
from leif.scene_graph import SceneGraphBuilder, TemporalSceneGraphSequence
from leif.event_validation import EventValidator
from leif.longitudinal_memory import LongitudinalMemoryDB, RecurrenceLinker
from leif.change_analysis import EnvironmentalChangeAnalyzer
from leif.footprint_aggregation import FootprintAggregator, EnvironmentalFootprintReport
from leif.visualization import (
    TimelinePlotter,
    InteractionGraphVisualizer,
    EnvironmentHeatmapGenerator,
    InteractionChartGenerator,
    LongitudinalSummaryGenerator,
)
from leif.privacy import PrivacyAnonymizer
from leif.types import EventRecord, EnvironmentalChange


class LEIFPipeline:
    """
    End-to-End Longitudinal Environmental Interaction Footprint System.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        db_path: str = "data/memory_store/leif_longitudinal_memory.db",
        output_dir: str = "outputs",
    ):
        self.config = config or {}
        self.output_dir = output_dir

        # Initialize Sub-modules
        self.region_detector = RegionDetector()
        self.detector = ColorSpatialDetector(self.region_detector)
        self.reid = ReIDManager()
        self.tracker = MultiObjectTracker(reid_manager=self.reid)
        self.hoi_classifier = HOIClassifier()
        self.temporal_scene_graph = TemporalSceneGraphSequence()
        self.validator = EventValidator()
        self.recurrence_linker = RecurrenceLinker()
        self.memory_db = LongitudinalMemoryDB(db_path=db_path)
        self.change_analyzer = EnvironmentalChangeAnalyzer()
        self.footprint_aggregator = FootprintAggregator()
        self.anonymizer = PrivacyAnonymizer()

        # Visualizers
        self.timeline_plotter = TimelinePlotter()
        self.graph_visualizer = InteractionGraphVisualizer()
        self.heatmap_generator = EnvironmentHeatmapGenerator()
        self.chart_generator = InteractionChartGenerator()
        self.summary_generator = LongitudinalSummaryGenerator()

    def process_video(
        self,
        video_path: str,
        session_id: str = "session_001",
        target_fps: float = 2.0,
        generate_visuals: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes complete LEIF processing on an input video or directory of frames.
        """
        print(f"=== [LEIF] Initializing Session '{session_id}' on: {video_path} ===")

        # 1. Video Acquisition & Frame Sampling
        source = VideoSource(video_path)
        # Reset session sub-modules for clean state
        self.tracker = MultiObjectTracker(reid_manager=self.reid)
        self.temporal_scene_graph = TemporalSceneGraphSequence()
        self.validator = EventValidator(session_id=session_id)
        self.recurrence_linker = RecurrenceLinker()
        self.change_analyzer = EnvironmentalChangeAnalyzer()

        sampler = FrameSampler(source, target_fps=target_fps)

        all_completed_events: List[EventRecord] = []
        all_detected_changes: List[EnvironmentalChange] = []
        annotated_frames_info = []

        sess_out = os.path.join(self.output_dir, session_id)
        annotated_frames_dir = os.path.join(sess_out, "annotated_frames")
        if generate_visuals:
            os.makedirs(annotated_frames_dir, exist_ok=True)

        last_timestamp = 0.0

        for sampled in sampler.sample_frames():
            t = sampled.timestamp_sec
            f_idx = sampled.frame_idx
            last_timestamp = t

            # 2. Entity Detection
            entities = self.detector.detect(sampled.image_rgb, t, f_idx)
            region_entities = [e for e in entities if e.category.value == "environmental_region"]

            # 3. Multi-Object Tracking & Re-Identification
            active_tracks = self.tracker.update(entities, sampled.image_rgb, t, f_idx)

            # 4. HOI Recognition
            interactions = self.hoi_classifier.predict_interactions(active_tracks, t, f_idx)

            # 5. Spatio-Temporal Scene Graph
            graph = SceneGraphBuilder.build_graph(
                active_tracks, region_entities, interactions, t, f_idx
            )
            self.temporal_scene_graph.add_snapshot(graph)

            # 6. Event Validation
            completed = self.validator.process_frame_interactions(
                predictions=interactions,
                active_tracks=active_tracks,
                timestamp_sec=t,
                source_clip=sampled.clip_id,
            )

            # 7 & 8. Longitudinal Memory Update & Observable Changes
            for evt in completed:
                # Privacy Anonymization
                sanitized_evt = self.anonymizer.sanitize_event(evt)
                # Link Recurrence
                linked_evt = self.recurrence_linker.link_event(sanitized_evt)
                # Store in Persistent DB
                self.memory_db.insert_event(linked_evt)
                all_completed_events.append(linked_evt)

                # Observable Environmental Changes
                target_track = next(
                    (tr for tr in active_tracks if tr.track_id == linked_evt.object_id), None
                )
                changes = self.change_analyzer.analyze_event_changes(linked_evt, target_track)
                for chg in changes:
                    self.memory_db.insert_change(chg)
                    all_detected_changes.append(chg)

            # Render Annotated Visual Frame
            if generate_visuals:
                annotated_rgb = self._annotate_frame(
                    sampled.image_rgb, active_tracks, region_entities, interactions, t, f_idx, session_id
                )
                frame_filename = f"frame_{f_idx:05d}.jpg"
                frame_save_path = os.path.join(annotated_frames_dir, frame_filename)
                try:
                    import cv2
                    cv2.imwrite(frame_save_path, cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))
                    annotated_frames_info.append({
                        "frame_idx": f_idx,
                        "timestamp": round(t, 2),
                        "url": f"outputs/{session_id}/annotated_frames/{frame_filename}",
                        "active_tracks_count": len(active_tracks),
                        "active_interactions": [
                            {"subject": p.subject_id, "object": p.object_id, "action": p.interaction_type.value, "conf": round(p.confidence, 2)}
                            for p in interactions
                        ]
                    })
                except Exception:
                    pass

        # Finalize remaining candidates at end of video
        final_events = self.validator.finalize_remaining(last_timestamp)
        for evt in final_events:
            sanitized_evt = self.anonymizer.sanitize_event(evt)
            linked_evt = self.recurrence_linker.link_event(sanitized_evt)
            self.memory_db.insert_event(linked_evt)
            all_completed_events.append(linked_evt)
            changes = self.change_analyzer.analyze_event_changes(linked_evt, None)
            for chg in changes:
                self.memory_db.insert_change(chg)
                all_detected_changes.append(chg)

        # 9. Footprint Aggregation
        footprint_report = self.footprint_aggregator.aggregate_footprint(
            events=all_completed_events,
            changes=all_detected_changes,
            session_id=session_id,
        )

        visual_outputs = {}
        annotated_video_rel = None
        if generate_visuals:
            # 10. Generate Visual Outputs
            os.makedirs(sess_out, exist_ok=True)

            timeline_file = os.path.join(sess_out, "timeline.png")
            graph_file = os.path.join(sess_out, "interaction_graph.png")
            heatmap_file = os.path.join(sess_out, "environment_heatmap.png")
            chart_file = os.path.join(sess_out, "frequency_charts.png")
            report_file = os.path.join(sess_out, "footprint_summary.md")

            visual_outputs["timeline"] = self.timeline_plotter.plot_timeline(
                all_completed_events, output_path=timeline_file
            )
            visual_outputs["interaction_graph"] = self.graph_visualizer.plot_graph(
                all_completed_events, output_path=graph_file
            )
            visual_outputs["heatmap"] = self.heatmap_generator.plot_heatmap(
                all_completed_events, output_path=heatmap_file
            )
            visual_outputs["frequency_charts"] = self.chart_generator.plot_summary_charts(
                all_completed_events, output_path=chart_file
            )
            visual_outputs["summary_report"] = self.summary_generator.generate_markdown_report(
                footprint_report, all_completed_events, all_detected_changes, output_path=report_file
            )

            # Compile Annotated MP4 Video
            if annotated_frames_info:
                try:
                    import cv2
                    video_out_file = os.path.join(sess_out, "annotated_video.mp4")
                    first_frame_path = os.path.join(annotated_frames_dir, f"frame_{annotated_frames_info[0]['frame_idx']:05d}.jpg")
                    sample_im = cv2.imread(first_frame_path)
                    if sample_im is not None:
                        vh, vw, _ = sample_im.shape
                        v_writer = cv2.VideoWriter(video_out_file, cv2.VideoWriter_fourcc(*'mp4v'), target_fps, (vw, vh))
                        for info in annotated_frames_info:
                            fim = cv2.imread(os.path.join(annotated_frames_dir, f"frame_{info['frame_idx']:05d}.jpg"))
                            if fim is not None:
                                v_writer.write(fim)
                        v_writer.release()
                        annotated_video_rel = f"outputs/{session_id}/annotated_video.mp4"
                except Exception as e:
                    print(f"[Warning] Failed to generate annotated video: {e}")

        # Compile Tracked Entities Summary
        all_tracks_dict = {
            **self.tracker.active_tracks,
            **self.tracker.lost_tracks,
            **{t.track_id: t for t in self.tracker.retired_tracks}
        }
        tracked_entities = []
        for tr_id, tr in all_tracks_dict.items():
            locs = list(dict.fromkeys(h[3] for h in tr.history if h[3]))
            tracked_entities.append({
                "track_id": tr.track_id,
                "class_name": tr.class_name,
                "category": tr.category.value if hasattr(tr.category, "value") else str(tr.category),
                "first_seen": round(tr.first_seen_timestamp, 2),
                "last_seen": round(tr.last_seen_timestamp, 2),
                "total_duration": round(tr.total_duration, 2),
                "location_id": tr.location_id or (locs[-1] if locs else "unknown"),
                "locations_visited": locs,
                "interaction_count": sum(1 for e in all_completed_events if e.object_id == tr.track_id or e.subject_id == tr.track_id or (tr.category.value == "person" and e.subject_id.startswith("subject_"))),
                "state": tr.state,
                "current_bbox": {
                    "x1": round(tr.current_bbox.x1, 1),
                    "y1": round(tr.current_bbox.y1, 1),
                    "x2": round(tr.current_bbox.x2, 1),
                    "y2": round(tr.current_bbox.y2, 1),
                    "confidence": round(tr.current_bbox.confidence, 2)
                }
            })
        tracked_entities.sort(key=lambda x: (0 if x["category"] == "person" else 1, x["first_seen"]))

        # Save complete serializable results to JSON for dashboard / API
        results_json_path = os.path.join(sess_out, "results.json")
        serializable_results = {
            "session_id": session_id,
            "total_events": len(all_completed_events),
            "total_changes": len(all_detected_changes),
            "overall_confidence": round(footprint_report.overall_confidence * 100, 1),
            "events": [
                {
                    "event_id": e.event_id,
                    "timestamp_start": e.timestamp_start,
                    "timestamp_end": e.timestamp_end,
                    "duration": round(e.duration, 2),
                    "subject_id": e.subject_id,
                    "object_id": e.object_id,
                    "location_id": e.location_id,
                    "interaction_type": e.interaction_type,
                    "confidence": round(e.confidence, 3),
                    "is_repeated": e.is_repeated,
                    "recurrence_index": e.recurrence_index,
                    "uncertain_association": e.uncertain_association,
                    "object_state_before": e.object_state_before,
                    "object_state_after": e.object_state_after,
                }
                for e in all_completed_events
            ],
            "changes": [
                {
                    "change_id": c.change_id,
                    "event_id": c.event_id,
                    "object_id": c.object_id,
                    "change_type": c.change_type,
                    "description": c.description,
                    "state_before": c.state_before,
                    "state_after": c.state_after,
                    "confidence": round(c.confidence, 2),
                }
                for c in all_detected_changes
            ],
            "tracked_entities": tracked_entities,
            "annotated_frames": annotated_frames_info,
            "annotated_video": annotated_video_rel,
            "visual_outputs": {
                "timeline": f"outputs/{session_id}/timeline.png",
                "interaction_graph": f"outputs/{session_id}/interaction_graph.png",
                "heatmap": f"outputs/{session_id}/environment_heatmap.png",
                "frequency_charts": f"outputs/{session_id}/frequency_charts.png",
                "summary_report": f"outputs/{session_id}/footprint_summary.md",
            },
            "summary": footprint_report.to_dict(),
            "metadata": {
                "source_path": video_path,
                "target_fps": target_fps,
                "duration_seconds": round(last_timestamp, 2),
                "total_frames": sampler.video_source.metadata.total_frames,
                "width": sampler.video_source.metadata.width,
                "height": sampler.video_source.metadata.height,
            }
        }
        try:
            import json
            with open(results_json_path, "w") as jf:
                json.dump(serializable_results, jf, indent=2)
        except Exception as err:
            print(f"[Warning] Failed to save results.json: {err}")

        print(f"=== [LEIF] Finished: {len(all_completed_events)} events, {len(all_detected_changes)} state changes ===")

        return {
            "session_id": session_id,
            "total_events": len(all_completed_events),
            "total_changes": len(all_detected_changes),
            "events": all_completed_events,
            "changes": all_detected_changes,
            "footprint_report": footprint_report,
            "visual_outputs": visual_outputs,
            "tracked_entities": tracked_entities,
            "annotated_frames": annotated_frames_info,
            "annotated_video": annotated_video_rel,
            "serializable": serializable_results,
            "metadata": serializable_results["metadata"],
        }

    def _annotate_frame(
        self,
        image_rgb,
        active_tracks: List[Any],
        region_entities: List[Any],
        interactions: List[Any],
        timestamp_sec: float,
        frame_idx: int,
        session_id: str,
    ):
        """Renders bounding boxes, tracking IDs, regions, and active interactions onto frame."""
        import cv2
        frame = image_rgb.copy()
        h, w, _ = frame.shape

        PERSON_COLOR = (41, 128, 185)     # Steel Blue
        OBJECT_COLOR = (230, 126, 34)     # Warm Orange
        CUP_COLOR = (231, 76, 60)         # Red
        LAPTOP_COLOR = (52, 73, 94)       # Slate Dark
        DOOR_COLOR = (120, 140, 160)      # Muted Blue-Grey

        # 1. Draw Active Tracks
        for track in active_tracks:
            bbox = track.current_bbox
            x1, y1 = max(0, int(bbox.x1)), max(0, int(bbox.y1))
            x2, y2 = min(w - 1, int(bbox.x2)), min(h - 1, int(bbox.y2))

            if track.category.value == "person":
                color = PERSON_COLOR
                tag = f"PERSON: {track.track_id}"
            elif "cup" in track.class_name:
                color = CUP_COLOR
                tag = f"CUP: {track.track_id}"
            elif "laptop" in track.class_name:
                color = LAPTOP_COLOR
                tag = f"LAPTOP: {track.track_id}"
            elif "door" in track.class_name:
                color = DOOR_COLOR
                tag = f"DOOR: {track.track_id}"
            else:
                color = OBJECT_COLOR
                tag = f"OBJ: {track.track_id}"

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label banner
            label = f"{tag} ({bbox.confidence:.2f})"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - lh - 6)), (x1 + lw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Draw HUD Header
        cv2.rectangle(frame, (0, 0), (w, 30), (20, 25, 35), -1)
        hud_text = f"LEIF TRACKER | {session_id} | T: {timestamp_sec:05.2f}s | Frame: {frame_idx:04d} | Active Tracks: {len(active_tracks)}"
        cv2.putText(frame, hud_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 230, 245), 1, cv2.LINE_AA)

        # 3. Draw Active Interaction Banner
        if interactions:
            action_strs = [f"{p.subject_id} > {p.interaction_type.value.upper()} > {p.object_id} ({p.confidence:.2f})" for p in interactions]
            banner_text = "  |  ".join(action_strs[:2])
            cv2.rectangle(frame, (0, h - 25), (w, h), (180, 40, 40), -1)
            cv2.putText(frame, f"⚡ HOI: {banner_text}", (10, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        return frame
