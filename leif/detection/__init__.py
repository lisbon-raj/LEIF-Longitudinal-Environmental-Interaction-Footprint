"""
Module 2: Scene and Object Detection.
Detects people, objects, surfaces, and environmental regions.
"""

from leif.detection.base_detector import BaseDetector
from leif.detection.region_detector import RegionDetector
from leif.detection.color_spatial_detector import ColorSpatialDetector
from leif.detection.pretrained_detector import PretrainedDetector

__all__ = [
    "BaseDetector",
    "RegionDetector",
    "ColorSpatialDetector",
    "PretrainedDetector",
]
