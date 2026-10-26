"""
Re-Identification (Re-ID) Manager.
Extracts appearance signatures and evaluates identity consistency across temporal gaps.
Implements uncertainty logging to avoid false merges.
"""

from typing import Tuple, Optional
import numpy as np
from leif.types import BoundingBox


class ReIDManager:
    """
    Computes visual appearance representations and similarity metrics.
    Enforces confident vs uncertain association boundaries.
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.75,
        uncertain_threshold: float = 0.50,
        max_reid_distance_pixels: float = 400.0,
    ):
        self.high_confidence_threshold = high_confidence_threshold
        self.uncertain_threshold = uncertain_threshold
        self.max_reid_distance_pixels = max_reid_distance_pixels

    def extract_appearance_signature(self, image_rgb: np.ndarray, bbox: BoundingBox) -> Optional[np.ndarray]:
        """
        Extracts a normalized color and spatial-grid histogram signature from the entity crop.
        """
        h, w, _ = image_rgb.shape
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(w, int(bbox.x2))
        y2 = min(h, int(bbox.y2))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Compute 3D color histogram (4 bins per channel -> 64 bins total)
        hist, _ = np.histogramdd(
            crop.reshape(-1, 3),
            bins=(4, 4, 4),
            range=((0, 256), (0, 256), (0, 256)),
        )
        hist = hist.astype(np.float32).flatten()
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        return hist

    def compute_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Cosine similarity between two normalized feature vectors."""
        if sig1 is None or sig2 is None:
            return 0.0
        dot = float(np.dot(sig1, sig2))
        return max(0.0, min(1.0, dot))

    def evaluate_association(
        self,
        sig1: Optional[np.ndarray],
        sig2: Optional[np.ndarray],
        pos1: Tuple[float, float],
        pos2: Tuple[float, float],
    ) -> Tuple[bool, bool, float, str]:
        """
        Evaluates whether two observations belong to the same entity.

        Returns:
            (is_match, is_uncertain, similarity_score, reason)
        """
        # 1. Spatial distance feasibility
        dist = np.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
        if dist > self.max_reid_distance_pixels:
            return False, False, 0.0, f"Spatial distance too large ({dist:.1f}px)"

        # 2. Appearance similarity
        if sig1 is None or sig2 is None:
            # Only spatial information available
            if dist < 80.0:
                return True, True, 0.55, "Spatial proximity only (appearance unavailable)"
            return False, False, 0.0, "Appearance signature missing"

        sim = self.compute_similarity(sig1, sig2)

        if sim >= self.high_confidence_threshold:
            return True, False, sim, "High appearance similarity"
        elif sim >= self.uncertain_threshold:
            # Uncertain association tier
            return True, True, sim, f"Borderline appearance similarity ({sim:.2f})"
        else:
            return False, False, sim, f"Appearance mismatch ({sim:.2f})"
