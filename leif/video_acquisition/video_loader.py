"""
Video Acquisition Module: Video Source and Metadata Loader.
Supports video files (OpenCV when available), image directories, and synthetic sequences.
"""

import os
from dataclasses import dataclass
from typing import Optional, Generator, Tuple, List
import numpy as np
from PIL import Image

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


@dataclass
class VideoMetadata:
    source_path: str
    fps: float
    total_frames: int
    duration_seconds: float
    width: int
    height: int
    is_synthetic: bool = False


class VideoSource:
    """
    Unified Video Reader supporting:
    1. Standard video files (.mp4, .avi, etc.) via OpenCV or fallback
    2. Image sequences / directories of frames
    3. Programmatically generated synthetic test streams
    """

    def __init__(self, source_path: str):
        self.source_path = source_path
        self._cv2_cap = None
        self._frame_files: List[str] = []
        self._metadata: Optional[VideoMetadata] = None
        self._init_source()

    def _init_source(self) -> None:
        if os.path.isdir(self.source_path):
            # Directory of image frames
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
            files = sorted([
                os.path.join(self.source_path, f)
                for f in os.listdir(self.source_path)
                if os.path.splitext(f)[1].lower() in valid_exts
            ])
            if not files:
                raise ValueError(f"No valid image files found in directory: {self.source_path}")
            self._frame_files = files

            # Read first image for dimensions
            with Image.open(files[0]) as sample:
                w, h = sample.size
            total = len(files)
            fps = 10.0  # Default assumption for image folder if unspecified
            meta_json = os.path.join(self.source_path, "metadata.json")
            if os.path.exists(meta_json):
                try:
                    import json
                    with open(meta_json, "r") as mf:
                        mdata = json.load(mf)
                        if "fps" in mdata:
                            fps = float(mdata["fps"])
                except Exception:
                    pass
            self._metadata = VideoMetadata(
                source_path=self.source_path,
                fps=fps,
                total_frames=total,
                duration_seconds=total / fps,
                width=w,
                height=h,
                is_synthetic=False,
            )

        elif os.path.isfile(self.source_path):
            if OPENCV_AVAILABLE:
                self._cv2_cap = cv2.VideoCapture(self.source_path)
                if not self._cv2_cap.isOpened():
                    raise RuntimeError(f"OpenCV could not open video: {self.source_path}")
                fps = self._cv2_cap.get(cv2.CAP_PROP_FPS) or 30.0
                total = int(self._cv2_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w = int(self._cv2_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self._cv2_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self._metadata = VideoMetadata(
                    source_path=self.source_path,
                    fps=fps,
                    total_frames=total,
                    duration_seconds=total / fps if fps > 0 else 0.0,
                    width=w,
                    height=h,
                    is_synthetic=False,
                )
            else:
                # If OpenCV is not installed, inspect file or raise informative message
                raise ImportError(
                    f"OpenCV (cv2) is not installed to decode '{self.source_path}'. "
                    "Install opencv-python or use an image frame directory or SyntheticVideoGenerator."
                )
        else:
            raise FileNotFoundError(f"Video source not found: {self.source_path}")

    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            raise RuntimeError("Video source metadata not initialized.")
        return self._metadata

    def read_frame_at_index(self, frame_idx: int) -> Optional[np.ndarray]:
        """Read a single frame by exact index as RGB numpy array (H, W, 3)."""
        if self._frame_files:
            if 0 <= frame_idx < len(self._frame_files):
                with Image.open(self._frame_files[frame_idx]) as img:
                    return np.array(img.convert("RGB"), dtype=np.uint8)
            return None

        if self._cv2_cap is not None:
            self._cv2_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cv2_cap.read()
            if ret and frame is not None:
                # OpenCV returns BGR; convert to RGB
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None

        return None

    def read_frame_at_timestamp(self, timestamp_sec: float) -> Tuple[int, Optional[np.ndarray]]:
        """Read frame corresponding to given timestamp in seconds."""
        frame_idx = int(timestamp_sec * self.metadata.fps)
        return frame_idx, self.read_frame_at_index(frame_idx)

    def frame_stream(self) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """Stream all raw consecutive frames: yields (frame_idx, timestamp_sec, rgb_frame)."""
        fps = self.metadata.fps
        if self._frame_files:
            for idx, file_path in enumerate(self._frame_files):
                with Image.open(file_path) as img:
                    arr = np.array(img.convert("RGB"), dtype=np.uint8)
                yield idx, idx / fps, arr
        elif self._cv2_cap is not None:
            self._cv2_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            idx = 0
            while True:
                ret, frame = self._cv2_cap.read()
                if not ret or frame is None:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield idx, idx / fps, rgb
                idx += 1

    def release(self) -> None:
        if self._cv2_cap is not None:
            self._cv2_cap.release()
            self._cv2_cap = None
