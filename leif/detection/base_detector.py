"""
Base Detector Interface for LEIF.
Defines abstract interface for detecting people, objects, surfaces, and environmental regions.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from leif.types import Entity, EntityCategory, BoundingBox


class BaseDetector(ABC):
    """Abstract interface for all LEIF entity detectors."""

    @abstractmethod
    def detect(
        self,
        image_rgb: np.ndarray,
        timestamp_sec: float,
        frame_idx: int,
    ) -> List[Entity]:
        """
        Detects entities in a single frame.

        Args:
            image_rgb: RGB image as numpy array (H, W, 3).
            timestamp_sec: Elapsed video time in seconds.
            frame_idx: Frame sequence index.

        Returns:
            List of detected Entity instances.
        """
        pass
