"""
Person-Object Interaction Graph Visualizer.
Renders network diagrams showing recurring interactions between people, objects, and locations.
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


class InteractionGraphVisualizer:
    """
    Renders recurring relational graphs connecting People, Objects, and Environmental Locations.
    Edge thickness reflects interaction frequency or accumulated duration.
    """

    def plot_graph(
        self,
        events: List[EventRecord],
        output_path: str = "outputs/graphs/person_object_graph.png",
        title: str = "LEIF: Person-Object-Environment Interaction Graph",
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not events:
            return output_path

        if not MATPLOTLIB_AVAILABLE or plt is None:
            print(f"[Warning] matplotlib is not installed in active environment. Graph skipped: {output_path}")
            return output_path

        # Tally edges: (subject, object, relation) -> count, duration
        edges: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(lambda: {"count": 0, "duration": 0.0})
        nodes = set()
        people = set()
        objects = set()
        locations = set()

        for e in events:
            nodes.add(e.subject_id)
            nodes.add(e.object_id)
            nodes.add(e.location_id)
            people.add(e.subject_id)
            objects.add(e.object_id)
            locations.add(e.location_id)

            key = (e.subject_id, e.object_id, e.interaction_type)
            edges[key]["count"] += 1
            edges[key]["duration"] += e.duration

        # Circular layout positioning
        all_nodes = sorted(list(people)) + sorted(list(objects)) + sorted(list(locations))
        n = len(all_nodes)
        angle_step = 2.0 * np.pi / max(1, n)
        pos = {}
        for i, node in enumerate(all_nodes):
            angle = i * angle_step
            # Position people at top/left, objects center/right, locations bottom
            r = 1.0
            pos[node] = (r * np.cos(angle), r * np.sin(angle))

        fig, ax = plt.subplots(figsize=(9, 8))

        # Draw Edges
        max_count = max(d["count"] for d in edges.values()) if edges else 1
        for (src, tgt, rel), d in edges.items():
            if src in pos and tgt in pos:
                x1, y1 = pos[src]
                x2, y2 = pos[tgt]
                weight = 1.0 + (d["count"] / max_count) * 4.0
                ax.plot([x1, x2], [y1, y2], color="#7f8c8d", alpha=0.6, linewidth=weight, zorder=1)
                mid_x = (x1 + x2) / 2.0
                mid_y = (y1 + y2) / 2.0
                ax.text(
                    mid_x, mid_y, f"{rel} (x{d['count']})",
                    fontsize=8, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#ecf0f1", edgecolor="gray", alpha=0.8),
                    zorder=2,
                )

        # Draw Nodes
        for node in all_nodes:
            x, y = pos[node]
            if node in people:
                color, shape_label = "#3498db", "Person"
            elif node in objects:
                color, shape_label = "#e67e22", "Object"
            else:
                color, shape_label = "#2ecc71", "Region"

            ax.scatter(x, y, s=1200, color=color, edgecolors="black", linewidth=1.5, zorder=3)
            ax.text(x, y, node, ha="center", va="center", color="white", fontweight="bold", fontsize=8, zorder=4)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
        ax.axis("off")
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)

        # Legend
        ax.scatter([], [], s=150, color="#3498db", label="Person")
        ax.scatter([], [], s=150, color="#e67e22", label="Object")
        ax.scatter([], [], s=150, color="#2ecc71", label="Region")
        ax.legend(loc="lower right", fontsize=9)

        plt.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        return output_path
