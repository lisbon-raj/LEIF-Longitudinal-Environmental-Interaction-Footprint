"""
LEIF: Longitudinal Environmental Interaction Footprint CLI Runner.
Usage:
  python run_leif.py --demo
  python run_leif.py --video data/raw_videos/office_recording.mp4 --session-id office_session_01
  python run_leif.py --benchmark
"""

import os
import argparse
import json
from leif.video_acquisition import SyntheticVideoGenerator
from leif.pipeline import LEIFPipeline
from evaluation.benchmark_runner import BenchmarkRunner


def run_demo(session_id: str = "demo_session", output_dir: str = "outputs"):
    print("================================================================================")
    print("   LONGITUDINAL ENVIRONMENTAL INTERACTION FOOTPRINT (LEIF) - DEMO EXECUTION     ")
    print("================================================================================")

    # 1. Generate realistic benchmark video sequence in data/sample_clips/office_demo
    demo_video_dir = "data/sample_clips/office_demo"
    print(f"\n[1/4] Generating Controlled Benchmark Office Video in: {demo_video_dir}...")
    generator = SyntheticVideoGenerator(
        output_dir=demo_video_dir,
        fps=5.0,
        duration_seconds=20.0,
        resolution=(640, 360),
    )
    meta = generator.generate()
    print(f"      Rendered {meta['total_frames']} frames ({meta['duration_seconds']}s) at {meta['fps']} FPS.")

    # 2. Run Complete LEIF Pipeline
    print(f"\n[2/4] Executing LEIF Longitudinal Pipeline...")
    pipeline = LEIFPipeline(output_dir=output_dir)
    results = pipeline.process_video(
        video_path=demo_video_dir,
        session_id=session_id,
        target_fps=2.0,
        generate_visuals=True,
    )

    # 3. Print Results Summary
    print(f"\n[3/4] Pipeline Results for Session '{session_id}':")
    print(f"      • Total Consolidated Events: {results['total_events']}")
    print(f"      • Observable State Changes:  {results['total_changes']}")
    print(f"      • Overall System Confidence: {results['footprint_report'].overall_confidence * 100:.1f}%")
    print(f"      • Uncertain Events Flagged:  {results['footprint_report'].uncertain_events_count}")

    print("\n      Observable Changes Detected:")
    for chg in results["changes"]:
        print(f"       -> [{chg.change_type}] {chg.description}")

    # 4. Show Generated Visualizations
    print(f"\n[4/4] Visualizations & Reports Generated in '{output_dir}/{session_id}':")
    for name, path in results["visual_outputs"].items():
        print(f"      • {name.upper()}: {path}")

    print("\n================================================================================")
    print("                      LEIF PROCESSING COMPLETED SUCCESSFULLY                    ")
    print("================================================================================")
    return results


def run_benchmark():
    print("=== Running LEIF Comparative Benchmark Evaluation ===")
    demo_video_dir = "data/sample_clips/office_benchmark"
    generator = SyntheticVideoGenerator(
        output_dir=demo_video_dir, fps=5.0, duration_seconds=20.0, resolution=(640, 360)
    )
    generator.generate()

    runner = BenchmarkRunner()
    results = runner.run_ablation_comparison(demo_video_dir, duration_sec=20.0)

    print("\nBenchmark Comparison Summary:")
    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description="LEIF: Longitudinal Environmental Interaction Footprint System")
    parser.add_argument("--demo", action="store_true", help="Run full end-to-end demo on benchmark office environment")
    parser.add_argument("--benchmark", action="store_true", help="Run ablation comparison (LEIF vs Baselines)")
    parser.add_argument("--video", type=str, help="Path to input video file or image directory")
    parser.add_argument("--session-id", type=str, default="session_001", help="Observation session ID")
    parser.add_argument("--fps", type=float, default=2.0, help="Target temporal frame sampling rate")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for reports & visuals")

    parser.add_argument("--dashboard", action="store_true", help="Launch interactive web dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Dashboard web server port (default: 8050)")

    args = parser.parse_args()

    if args.dashboard:
        from leif.dashboard_server import start_server
        start_server(port=args.port)
    elif args.benchmark:
        run_benchmark()
    elif args.video:
        pipeline = LEIFPipeline(output_dir=args.output_dir)
        pipeline.process_video(
            video_path=args.video,
            session_id=args.session_id,
            target_fps=args.fps,
            generate_visuals=True,
        )
    else:
        # Default action: run demo
        run_demo(session_id=args.session_id, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
