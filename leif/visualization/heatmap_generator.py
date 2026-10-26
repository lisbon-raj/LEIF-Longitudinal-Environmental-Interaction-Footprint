"""
Environment Utilization Spatial Heatmap Generator.
Renders 2D spatial distribution and dwell duration heatmaps over environmental regions.
"""

import os
from typing import List, Dict, Tuple
from collections import defaultdict
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

import numpy as np
from leif.types import EventRecord


class EnvironmentHeatmapGenerator:
    """
    Renders 2D spatial heatmaps of environmental region interaction frequency and duration.
    """

    def __init__(self, room_dimensions: Tuple[int, int] = (640, 360)):
        self.width, self.height = room_dimensions
        # Default region bounding boxes (x1, y1, x2, y2)
        self.region_boxes: Dict[str, Tuple[float, float, float, float]] = {
            "desk_zone": (40.0, 140.0, 260.0, 330.0),
            "coffee_station": (390.0, 140.0, 600.0, 330.0),
            "door_zone": (540.0, 30.0, 630.0, 160.0),
        }

    def plot_heatmap(
        self,
        events: List[EventRecord],
        output_path: str = "outputs/heatmaps/environment_heatmap.png",
        metric: str = "duration", # 'duration' or 'frequency'
        title: str = "LEIF: Environment Spatial Utilization Heatmap",
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not MATPLOTLIB_AVAILABLE or plt is None:
            print(f"[Warning] matplotlib is not installed in active environment. Heatmap skipped: {output_path}")
            return output_path

        # Accumulate metrics per region
        region_metrics: Dict[str, float] = defaultdict(float)
        for e in events:
            val = e.duration if metric == "duration" else 1.0
            region_metrics[e.location_id] += val

        # Create 2D density grid
        grid = np.zeros((self.height, self.width), dtype=np.float32)

        for loc, val in region_metrics.items():
            if loc in self.region_boxes:
                x1, y1, x2, y2 = [int(v) for v in self.region_boxes[loc]]
                grid[y1:y2, x1:x2] += val

        fig, ax = plt.subplots(figsize=(10, 6))

        # Show heatmap
        cmap = plt.cm.YlOrRd
        im = ax.imshow(grid, cmap=cmap, origin="upper", aspect="auto", alpha=0.85)

        # Overlay region outlines and labels
        for loc, (x1, y1, x2, y2) in self.region_boxes.items():
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor="#2c3e50", linewidth=2.0, linestyle="--"
            )
            ax.add_patch(rect)
            metric_val = region_metrics.get(loc, 0.0)
            label = f"{loc}\n{metric_val:.1f}s" if metric == "duration" else f"{loc}\n{int(metric_val)} events"
            ax.text(
                (x1 + x2) / 2.0, (y1 + y2) / 2.0,
                label,
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#2c3e50", alpha=0.9),
            )

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Total Duration (seconds)" if metric == "duration" else "Event Count", fontweight="bold")

        ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
        ax.set_xlabel("X (Pixels)", fontweight="bold")
        ax.set_ylabel("Y (Pixels)", fontweight="bold")
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0) # Top-down image coordinate system

        plt.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        return output_path
