"""
Longitudinal Interaction Timeline Plotter.
Generates publication-quality Gantt timeline visualizations of interactions over time.
"""

import os
from typing import List, Optional
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    mpatches = None
    MATPLOTLIB_AVAILABLE = False

from leif.types import EventRecord


class TimelinePlotter:
    """
    Renders longitudinal interaction events across time with durations,
    involved entities, locations, and confidence levels.
    """

    COLOR_PALETTE = {
        "holding": "#3498db",
        "touching": "#95a5a6",
        "opening": "#2ecc71",
        "closing": "#e67e22",
        "placing": "#9b59b6",
        "moving": "#f1c40f",
        "using": "#e74c3c",
        "default": "#1abc9c",
    }

    def plot_timeline(
        self,
        events: List[EventRecord],
        output_path: str = "outputs/timelines/longitudinal_timeline.png",
        title: str = "LEIF: Longitudinal Interaction Timeline",
    ) -> str:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not MATPLOTLIB_AVAILABLE or plt is None:
            print(f"[Warning] matplotlib is not installed in active environment. Timeline skipped: {output_path}")
            return output_path

        if not events:
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.text(0.5, 0.5, "No interaction events recorded", ha="center", va="center")
            fig.savefig(output_path)
            plt.close(fig)
            return output_path

        sorted_events = sorted(events, key=lambda e: (e.object_id, e.timestamp_start))

        # Y-axis will be distinct objects or entity pairs
        entities_y = list(dict.fromkeys(e.object_id for e in sorted_events))
        y_map = {ent: i for i, ent in enumerate(entities_y)}

        fig_height = max(4.0, len(entities_y) * 1.2 + 2.0)
        fig, ax = plt.subplots(figsize=(12, fig_height))

        plotted_types = set()

        for evt in sorted_events:
            y = y_map[evt.object_id]
            dur = max(0.4, evt.duration)
            color = self.COLOR_PALETTE.get(evt.interaction_type, self.COLOR_PALETTE["default"])
            plotted_types.add(evt.interaction_type)

            # Draw event bar
            edge_color = "red" if evt.uncertain_association else "black"
            line_width = 1.5 if evt.uncertain_association else 0.8
            bar = ax.barh(
                y=y,
                width=dur,
                left=evt.timestamp_start,
                height=0.55,
                color=color,
                edgecolor=edge_color,
                linewidth=line_width,
                alpha=0.88,
            )

            # Label on bar
            label_text = f"{evt.interaction_type} ({evt.subject_id} @ {evt.location_id})"
            if evt.uncertain_association:
                label_text += " [Uncertain]"

            mid_x = evt.timestamp_start + (dur / 2.0)
            ax.text(
                mid_x,
                y,
                label_text,
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white" if color in ("#3498db", "#9b59b6", "#e74c3c") else "black",
                clip_on=True,
            )

        ax.set_yticks(range(len(entities_y)))
        ax.set_yticklabels(entities_y, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Interacted Entity", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        # Legend
        patches = [
            mpatches.Patch(color=self.COLOR_PALETTE.get(t, self.COLOR_PALETTE["default"]), label=t.capitalize())
            for t in sorted(plotted_types)
        ]
        if any(e.uncertain_association for e in sorted_events):
            patches.append(
                mpatches.Patch(facecolor="white", edgecolor="red", linewidth=1.5, label="Uncertain Association")
            )

        ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.15, 1.05), fontsize=9)
        plt.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        return output_path
