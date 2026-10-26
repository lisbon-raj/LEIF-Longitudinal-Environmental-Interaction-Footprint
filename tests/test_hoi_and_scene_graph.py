"""
Unit and Integration Tests for Phase 4 (HOI Recognition) and Phase 5 (Scene Graph).
"""

import os
import shutil
from leif.video_acquisition import SyntheticVideoGenerator, VideoSource, FrameSampler
from leif.detection import RegionDetector, ColorSpatialDetector
from leif.tracking import MultiObjectTracker
from leif.hoi_recognition import HOIClassifier, SpatialRelationAnalyzer
from leif.scene_graph import SceneGraphBuilder, TemporalSceneGraphSequence
from leif.types import InteractionType


def test_phase_4_and_5_integration():
    test_dir = "data/sample_clips/test_hoi_session"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)

    # 1. Generate full 20-second synthetic interaction sequence
    generator = SyntheticVideoGenerator(
        output_dir=test_dir,
        fps=5.0,
        duration_seconds=19.0,
        resolution=(640, 360),
    )
    generator.generate()

    # 2. Setup Pipeline components
    source = VideoSource(test_dir)
    sampler = FrameSampler(source, target_fps=2.0) # Sample at 2 FPS (38 frames total)
    reg_det = RegionDetector()
    detector = ColorSpatialDetector(reg_det)
    tracker = MultiObjectTracker(max_disappeared_frames=20)
    hoi_classifier = HOIClassifier()
    temporal_sequence = TemporalSceneGraphSequence()

    recognized_interactions = set()
    total_triplets = []

    for sampled in sampler.sample_frames():
        # Module 2: Detection
        entities = detector.detect(sampled.image_rgb, sampled.timestamp_sec, sampled.frame_idx)
        region_entities = [e for e in entities if e.category.value == "environmental_region"]

        # Module 3: Tracking
        active_tracks = tracker.update(
            entities, sampled.image_rgb, sampled.timestamp_sec, sampled.frame_idx
        )

        # Module 4: HOI Recognition
        hois = hoi_classifier.predict_interactions(
            active_tracks, sampled.timestamp_sec, sampled.frame_idx
        )
        for h in hois:
            recognized_interactions.add(h.interaction_type)

        # Module 5: Spatio-Temporal Scene Graph
        graph = SceneGraphBuilder.build_graph(
            active_tracks, region_entities, hois, sampled.timestamp_sec, sampled.frame_idx
        )
        temporal_sequence.add_snapshot(graph)

        triplets = SceneGraphBuilder.extract_semantic_triplets(graph)
        total_triplets.extend(triplets)

    # Verify HOI interactions recognized
    print(f"Recognized Interactions: {[i.value for i in recognized_interactions]}")
    assert InteractionType.USING in recognized_interactions, "Should recognize USING laptop"
    assert InteractionType.HOLDING in recognized_interactions, "Should recognize HOLDING cup"
    assert InteractionType.OPENING in recognized_interactions, "Should recognize OPENING door"

    # Verify Scene Graph evolution
    print(f"Snapshots captured: {len(temporal_sequence.snapshots)}")
    assert len(temporal_sequence.snapshots) > 0
    edge_summary = temporal_sequence.get_edge_summary()
    assert len(edge_summary) > 0

    print("Sample Edge Lifespan Summaries:")
    for edge in edge_summary[:5]:
        print(f" - {edge['source']} --[{edge['relation']}]--> {edge['target']} | Dur: {edge['total_duration_sec']}s")

    shutil.rmtree(test_dir, ignore_errors=True)
    print("Phase 4 (HOI) & Phase 5 (Scene Graph) Tests Passed Successfully!")


if __name__ == "__main__":
    test_phase_4_and_5_integration()
