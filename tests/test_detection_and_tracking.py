"""
Unit and Integration Tests for Phase 2 (Detection) and Phase 3 (Tracking & Re-ID).
"""

import os
import shutil
import numpy as np
from PIL import Image

from leif.types import BoundingBox, EntityCategory
from leif.video_acquisition import SyntheticVideoGenerator, VideoSource, FrameSampler
from leif.detection import RegionDetector, ColorSpatialDetector
from leif.tracking import MultiObjectTracker, ReIDManager


def test_phase_2_detection():
    # 1. Test RegionDetector
    reg_det = RegionDetector()
    regions = reg_det.detect_regions(timestamp_sec=0.0, frame_idx=0)
    assert len(regions) >= 3
    assert any(r.class_name == "desk_zone" for r in regions)
    assert any(r.class_name == "coffee_station" for r in regions)
    assert any(r.class_name == "door_zone" for r in regions)

    # Test spatial containment
    desk_pt = (100.0, 200.0)
    assert reg_det.get_enclosing_region(desk_pt) == "desk_zone"
    coffee_pt = (500.0, 200.0)
    assert reg_det.get_enclosing_region(coffee_pt) == "coffee_station"

    # 2. Test ColorSpatialDetector with synthetic benchmark
    test_dir = "data/sample_clips/test_det_session"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    gen = SyntheticVideoGenerator(test_dir, fps=5.0, duration_seconds=4.0, resolution=(640, 360))
    gen.generate()

    source = VideoSource(test_dir)
    frame = source.read_frame_at_index(10)  # Frame at 2.0s: person is at desk
    assert frame is not None

    detector = ColorSpatialDetector(reg_det)
    entities = detector.detect(frame, timestamp_sec=2.0, frame_idx=10)

    # Verify detection of entities
    classes = [e.class_name for e in entities]
    assert "person" in classes
    assert "laptop" in classes
    assert "cup" in classes
    assert "door" in classes
    assert "desk_zone" in classes

    shutil.rmtree(test_dir)
    print("Phase 2 Detection Tests Passed!")


def test_phase_3_tracking_and_reid():
    test_dir = "data/sample_clips/test_trk_session"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    gen = SyntheticVideoGenerator(test_dir, fps=5.0, duration_seconds=6.0, resolution=(640, 360))
    gen.generate()

    source = VideoSource(test_dir)
    sampler = FrameSampler(source, target_fps=5.0)

    reg_det = RegionDetector()
    detector = ColorSpatialDetector(reg_det)
    tracker = MultiObjectTracker(max_disappeared_frames=15, iou_threshold=0.2)

    person_track_ids_observed = set()
    cup_track_ids_observed = set()

    for sampled in sampler.sample_frames():
        entities = detector.detect(sampled.image_rgb, sampled.timestamp_sec, sampled.frame_idx)
        active_tracks = tracker.update(
            entities, sampled.image_rgb, sampled.timestamp_sec, sampled.frame_idx
        )

        for track in active_tracks:
            if track.class_name == "person":
                person_track_ids_observed.add(track.track_id)
            elif track.class_name == "cup":
                cup_track_ids_observed.add(track.track_id)

    # Assert persistence: exactly ONE persistent track ID for the person and cup
    # instead of spawning new tracks every frame
    assert len(person_track_ids_observed) == 1, f"Expected 1 persistent person ID, got {person_track_ids_observed}"
    assert len(cup_track_ids_observed) == 1, f"Expected 1 persistent cup ID, got {cup_track_ids_observed}"

    # Verify Re-ID across artificial occlusion gap
    reid = ReIDManager()
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dummy_img[:, :] = (200, 30, 30) # Red cup
    box = BoundingBox(10, 10, 40, 40)
    sig1 = reid.extract_appearance_signature(dummy_img, box)
    sig2 = reid.extract_appearance_signature(dummy_img, box)
    assert sig1 is not None
    sim = reid.compute_similarity(sig1, sig2)
    assert sim > 0.99  # Identical appearance

    # Test uncertainty logging
    sim_noisy = 0.60
    is_match, is_uncertain, _, reason = reid.evaluate_association(
        sig1, sig2 * 0.6, (10, 10), (15, 15)
    )
    assert is_match is True
    print(f"Association evaluation: is_match={is_match}, is_uncertain={is_uncertain}, reason={reason}")

    shutil.rmtree(test_dir)
    print("Phase 3 Tracking & Re-ID Tests Passed!")


if __name__ == "__main__":
    test_phase_2_detection()
    test_phase_3_tracking_and_reid()
