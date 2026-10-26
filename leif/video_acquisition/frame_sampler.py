"""
Frame Sampling and Temporal Clip Windowing.
Efficiently downsamples long-duration video to target processing rates
without requiring processing every raw frame.
"""

from dataclasses import dataclass
from typing import Generator, List, Tuple, Optional
import numpy as np
from leif.video_acquisition.video_loader import VideoSource


@dataclass
class SampledFrame:
    frame_idx: int
    timestamp_sec: float
    image_rgb: np.ndarray
    clip_id: str


@dataclass
class VideoClip:
    clip_id: str
    start_time_sec: float
    end_time_sec: float
    start_frame_idx: int
    end_frame_idx: int
    frames: List[SampledFrame]

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time_sec - self.start_time_sec)


class FrameSampler:
    """
    Downsamples video streams according to a target temporal rate (fps)
    and aggregates frames into logical temporal clips.
    """

    def __init__(
        self,
        video_source: VideoSource,
        target_fps: float = 2.0,
        clip_window_seconds: float = 5.0,
        sampling_mode: str = "uniform",
    ):
        self.video_source = video_source
        self.target_fps = target_fps
        self.clip_window_seconds = clip_window_seconds
        self.sampling_mode = sampling_mode
        self.raw_fps = self.video_source.metadata.fps

        # Calculate stride in terms of raw frames
        if self.target_fps >= self.raw_fps or self.target_fps <= 0:
            self.frame_stride = 1
        else:
            self.frame_stride = max(1, int(round(self.raw_fps / self.target_fps)))

    def sample_frames(self) -> Generator[SampledFrame, None, None]:
        """
        Yields sampled frames at the target temporal rate.
        Skips intermediate redundant frames.
        """
        raw_stream = self.video_source.frame_stream()
        for frame_idx, timestamp_sec, rgb_frame in raw_stream:
            if frame_idx % self.frame_stride == 0:
                clip_idx = int(timestamp_sec // self.clip_window_seconds)
                clip_id = f"clip_{clip_idx:04d}"
                yield SampledFrame(
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    image_rgb=rgb_frame,
                    clip_id=clip_id,
                )

    def sample_clips(self) -> Generator[VideoClip, None, None]:
        """
        Aggregates sampled frames into temporal window clips of duration `clip_window_seconds`.
        """
        current_clip_id: Optional[str] = None
        current_clip_frames: List[SampledFrame] = []
        clip_start_time = 0.0
        clip_start_frame = 0

        for sampled_frame in self.sample_frames():
            if current_clip_id is None:
                current_clip_id = sampled_frame.clip_id
                clip_start_time = sampled_frame.timestamp_sec
                clip_start_frame = sampled_frame.frame_idx

            if sampled_frame.clip_id != current_clip_id:
                # Yield finished clip
                if current_clip_frames:
                    end_time = current_clip_frames[-1].timestamp_sec
                    end_frame = current_clip_frames[-1].frame_idx
                    yield VideoClip(
                        clip_id=current_clip_id,
                        start_time_sec=clip_start_time,
                        end_time_sec=end_time,
                        start_frame_idx=clip_start_frame,
                        end_frame_idx=end_frame,
                        frames=current_clip_frames,
                    )

                # Reset for new clip
                current_clip_id = sampled_frame.clip_id
                current_clip_frames = [sampled_frame]
                clip_start_time = sampled_frame.timestamp_sec
                clip_start_frame = sampled_frame.frame_idx
            else:
                current_clip_frames.append(sampled_frame)

        # Yield final remaining clip
        if current_clip_frames and current_clip_id is not None:
            yield VideoClip(
                clip_id=current_clip_id,
                start_time_sec=clip_start_time,
                end_time_sec=current_clip_frames[-1].timestamp_sec,
                start_frame_idx=clip_start_frame,
                end_frame_idx=current_clip_frames[-1].frame_idx,
                frames=current_clip_frames,
            )
