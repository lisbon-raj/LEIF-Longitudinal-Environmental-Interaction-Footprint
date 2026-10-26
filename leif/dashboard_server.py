"""
LEIF Dashboard Server: Real-time Interactive Web Application for
Longitudinal Environmental Interaction Footprint.
Serves full visual dashboard, video player, tracking viewer, and RESTful pipeline endpoints.
"""

import os
import sys
import json
import base64
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from leif.pipeline import LEIFPipeline
from leif.video_acquisition import SyntheticVideoGenerator


class LEIFDashboardHandler(SimpleHTTPRequestHandler):
    """
    HTTP handler serving both the rich LEIF dashboard web client and the pipeline REST API.
    """

    pipeline: Optional[LEIFPipeline] = None
    latest_results: Optional[Dict[str, Any]] = None

    def __init__(self, *args, **kwargs):
        # Serve from project root directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        super().__init__(*args, directory=project_root, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.serve_dashboard_html()
            return
        elif path == "/api/videos":
            self.handle_get_videos()
            return
        elif path == "/api/latest":
            self.handle_get_latest()
            return
        elif path == "/api/session":
            params = urllib.parse.parse_qs(parsed.query)
            session_id = params.get("id", ["session_001"])[0]
            self.handle_get_session(session_id)
            return

        # Fallback to static file serving (for /outputs/..., /data/..., etc.)
        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/process":
            self.handle_process_video()
            return
        elif path == "/api/upload":
            self.handle_upload_video()
            return
        elif path == "/api/generate-demo":
            self.handle_generate_demo()
            return

        self.send_error(404, "Endpoint not found")

    def serve_dashboard_html(self):
        """Serves the rich interactive single-page dashboard."""
        html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
        if not os.path.exists(html_path):
            self.send_error(500, "Dashboard HTML template not found")
            return

        with open(html_path, "rb") as f:
            content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def handle_get_videos(self):
        """Lists available video files and synthetic sample clips."""
        videos = []
        raw_dir = "data/raw_videos"
        clips_dir = "data/sample_clips"

        if os.path.exists(raw_dir):
            for f in sorted(os.listdir(raw_dir)):
                ext = os.path.splitext(f)[1].lower()
                if ext in (".mp4", ".avi", ".mov", ".mkv"):
                    fpath = os.path.join(raw_dir, f).replace("\\", "/")
                    size_mb = os.path.getsize(fpath) / (1024 * 1024)
                    videos.append({
                        "name": f"Video: {f}",
                        "path": fpath,
                        "type": "video_file",
                        "size_mb": round(size_mb, 2),
                        "recommended": "office_demo" in f,
                    })

        if os.path.exists(clips_dir):
            for d in sorted(os.listdir(clips_dir)):
                dpath = os.path.join(clips_dir, d).replace("\\", "/")
                if os.path.isdir(dpath):
                    frames = [fn for fn in os.listdir(dpath) if fn.endswith((".png", ".jpg"))]
                    if frames:
                        videos.append({
                            "name": f"Frame Sequence: {d} ({len(frames)} frames)",
                            "path": dpath,
                            "type": "image_directory",
                            "frame_count": len(frames),
                            "recommended": False,
                        })

        self.send_json_response({"videos": videos})

    def handle_get_latest(self):
        """Returns latest processed session results."""
        if LEIFDashboardHandler.latest_results:
            self.send_json_response(LEIFDashboardHandler.latest_results)
            return

        # Check disk for session_001
        res_file = "outputs/session_001/results.json"
        if os.path.exists(res_file):
            try:
                with open(res_file, "r") as f:
                    data = json.load(f)
                    LEIFDashboardHandler.latest_results = data
                    self.send_json_response(data)
                    return
            except Exception:
                pass

        # If no results yet, run demo once
        self.run_pipeline_for_demo("session_001")

    def handle_get_session(self, session_id: str):
        """Returns results for a specific session ID."""
        res_file = f"outputs/{session_id}/results.json"
        if os.path.exists(res_file):
            try:
                with open(res_file, "r") as f:
                    data = json.load(f)
                    self.send_json_response(data)
                    return
            except Exception as e:
                self.send_error(500, f"Error reading session results: {e}")
                return
        self.send_error(404, f"Session '{session_id}' not found")

    def handle_upload_video(self):
        """Handles video upload (multipart or JSON base64)."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            content_type = self.headers.get("Content-Type", "")
            os.makedirs("data/raw_videos", exist_ok=True)

            if "application/json" in content_type:
                payload = json.loads(body.decode("utf-8"))
                filename = payload.get("filename", "uploaded_video.mp4")
                file_data = base64.b64decode(payload["data"].split(",")[-1])
            else:
                # Handle raw bytes or simple post
                filename = self.headers.get("X-Filename", "uploaded_video.mp4")
                file_data = body

            # Clean filename
            filename = os.path.basename(filename).replace(" ", "_")
            save_path = os.path.join("data/raw_videos", filename)

            with open(save_path, "wb") as f:
                f.write(file_data)

            self.send_json_response({
                "status": "success",
                "message": f"Video uploaded successfully as {filename}",
                "path": save_path.replace("\\", "/"),
                "size_bytes": len(file_data),
            })
        except Exception as e:
            self.send_json_response({"status": "error", "message": str(e)}, status=500)

    def handle_generate_demo(self):
        """Generates a fresh synthetic controlled benchmark office video."""
        try:
            demo_video_dir = "data/sample_clips/office_demo"
            generator = SyntheticVideoGenerator(
                output_dir=demo_video_dir,
                fps=5.0,
                duration_seconds=20.0,
                resolution=(640, 360),
            )
            meta = generator.generate()

            # Compile into MP4
            import cv2, glob
            frames = sorted(glob.glob(f"{demo_video_dir}/frame_*.png"))
            mp4_path = "data/raw_videos/office_demo.mp4"
            if frames:
                os.makedirs("data/raw_videos", exist_ok=True)
                sample = cv2.imread(frames[0])
                h, w, _ = sample.shape
                out = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (w, h))
                for f in frames:
                    out.write(cv2.imread(f))
                out.release()

            self.send_json_response({
                "status": "success",
                "message": "Generated 20.0s controlled benchmark office video",
                "video_path": mp4_path,
                "frames_dir": demo_video_dir,
                "total_frames": meta["total_frames"],
            })
        except Exception as e:
            self.send_json_response({"status": "error", "message": str(e)}, status=500)

    def handle_process_video(self):
        """Executes the complete LEIF processing pipeline on the selected video."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8")) if body else {}

            video_path = payload.get("video_path", "data/raw_videos/office_demo.mp4")
            session_id = payload.get("session_id", "session_001")
            fps = float(payload.get("fps", 2.0))

            if not os.path.exists(video_path):
                # Fallback to demo clip if file not found
                if os.path.exists("data/sample_clips/office_demo"):
                    video_path = "data/sample_clips/office_demo"
                else:
                    self.handle_generate_demo()
                    video_path = "data/raw_videos/office_demo.mp4"

            if LEIFDashboardHandler.pipeline is None:
                LEIFDashboardHandler.pipeline = LEIFPipeline()

            results = LEIFDashboardHandler.pipeline.process_video(
                video_path=video_path,
                session_id=session_id,
                target_fps=fps,
                generate_visuals=True,
            )

            # Retrieve serializable data
            res_json_path = f"outputs/{session_id}/results.json"
            if os.path.exists(res_json_path):
                with open(res_json_path, "r") as f:
                    data = json.load(f)
                    LEIFDashboardHandler.latest_results = data
                    self.send_json_response(data)
                    return

            self.send_json_response({
                "session_id": session_id,
                "total_events": results["total_events"],
                "total_changes": results["total_changes"],
                "status": "completed",
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.send_json_response({"status": "error", "message": str(e)}, status=500)

    def run_pipeline_for_demo(self, session_id: str = "session_001"):
        """Runs the benchmark office demo pipeline if no results exist."""
        try:
            video_path = "data/raw_videos/office_demo.mp4"
            if not os.path.exists(video_path):
                if os.path.exists("data/sample_clips/office_demo"):
                    video_path = "data/sample_clips/office_demo"
                else:
                    self.handle_generate_demo()

            pipeline = LEIFPipeline()
            pipeline.process_video(video_path=video_path, session_id=session_id, target_fps=2.0)
            res_file = f"outputs/{session_id}/results.json"
            if os.path.exists(res_file):
                with open(res_file, "r") as f:
                    data = json.load(f)
                    LEIFDashboardHandler.latest_results = data
                    self.send_json_response(data)
                    return
        except Exception as e:
            self.send_json_response({"status": "error", "message": str(e)}, status=500)

    def send_json_response(self, data: Any, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def start_server(port: int = 8050, open_browser: bool = False):
    """Starts the multithreaded LEIF web dashboard server."""
    host = "127.0.0.1"
    server_address = (host, port)

    # Try binding with fallback ports
    for p in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer((host, p), LEIFDashboardHandler)
            actual_port = p
            break
        except OSError:
            continue
    else:
        raise RuntimeError("No available port found in range.")

    url = f"http://{host}:{actual_port}"
    print("=" * 80)
    print("   LONGITUDINAL ENVIRONMENTAL INTERACTION FOOTPRINT (LEIF) DASHBOARD SERVER   ")
    print("=" * 80)
    print(f" * Dashboard URL:      {url}")
    print(f" * API Endpoints:      {url}/api/latest, {url}/api/process, {url}/api/videos")
    print(f" * Port:               {actual_port}")
    print("=" * 80)
    print(" Press Ctrl+C to stop server.")

    if open_browser:
        import webbrowser
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping LEIF Dashboard Server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
    start_server(port=port)
