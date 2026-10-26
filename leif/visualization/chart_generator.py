"""
Interaction Frequency and Duration Multi-Panel Chart Generator.
Compares counts and durations across objects, locations, and interaction categories.
"""

import os
from typing import List
from collections import defaultdict
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

from leif.types import EventRecord


class InteractionChartGenerator:
    """
    Renders comparative 4-panel analytical charts for the LEIF footprint.
    """

    def plot_summary_charts(
        self,
        events: List[EventRecord],
        output_path: str = "outputs/charts/footprint_frequency_charts.png",
        title: str = "LEIF: Interaction Frequency & Duration Footprint",
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not events:
            return output_path

        if not MATPLOTLIB_AVAILABLE or plt is None:
            print(f"[Warning] matplotlib is not installed in active environment. Chart skipped: {output_path}")
            return output_path

        # Tally metrics
        obj_counts = defaultdict(int)
        obj_durations = defaultdict(float)
        loc_counts = defaultdict(int)
        loc_durations = defaultdict(float)
        type_counts = defaultdict(int)

        for e in events:
            obj_counts[e.object_id] += 1
            obj_durations[e.object_id] += e.duration
            loc_counts[e.location_id] += 1
            loc_durations[e.location_id] += e.duration
            type_counts[e.interaction_type] += 1

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

        # 1. Object Interaction Counts
        objs = list(obj_counts.keys())
        counts = [obj_counts[o] for o in objs]
        ax1.bar(objs, counts, color="#3498db", edgecolor="black")
        ax1.set_title("Interaction Frequency by Object", fontweight="bold")
        ax1.set_ylabel("Event Count", fontweight="bold")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        # 2. Object Total Durations
        durs = [obj_durations[o] for o in objs]
        ax2.bar(objs, durs, color="#e67e22", edgecolor="black")
        ax2.set_title("Total Interaction Duration by Object", fontweight="bold")
        ax2.set_ylabel("Duration (seconds)", fontweight="bold")
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

        # 3. Location Utilization
        locs = list(loc_counts.keys())
        loc_vals = [loc_counts[l] for l in locs]
        ax3.bar(locs, loc_vals, color="#2ecc71", edgecolor="black")
        ax3.set_title("Interaction Count by Location", fontweight="bold")
        ax3.set_ylabel("Event Count", fontweight="bold")
        ax3.grid(axis="y", linestyle="--", alpha=0.5)

        # 4. Interaction Types Distribution
        itypes = list(type_counts.keys())
        tvals = [type_counts[t] for t in itypes]
        colors = ["#3498db", "#9b59b6", "#2ecc71", "#f1c40f", "#e74c3c", "#34495e"][:len(itypes)]
        ax4.pie(tvals, labels=itypes, autopct="%1.1f%%", colors=colors, startangle=140)
        ax4.set_title("Interaction Type Breakdown", fontweight="bold")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        return output_path
