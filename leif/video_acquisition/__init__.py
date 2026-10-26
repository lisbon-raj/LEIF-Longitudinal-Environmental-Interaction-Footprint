"""
Module 1: Video Acquisition and Sampling.
Handles video inputs, adaptive temporal sampling, and clip windowing.
"""

from leif.video_acquisition.video_loader import VideoSource, VideoMetadata
from leif.video_acquisition.frame_sampler import FrameSampler, SampledFrame, VideoClip
from leif.video_acquisition.synthetic_generator import SyntheticVideoGenerator

__all__ = [
    "VideoSource",
    "VideoMetadata",
    "FrameSampler",
    "SampledFrame",
    "VideoClip",
    "SyntheticVideoGenerator",
]
