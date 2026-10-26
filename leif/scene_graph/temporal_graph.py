"""
Temporal Scene Graph Sequence.
Maintains history of graph evolution across long-duration video observations.
"""

from typing import List, Dict, Any, Optional, Tuple
from leif.types import SpatioTemporalSceneGraph, SceneGraphEdge


class TemporalSceneGraphSequence:
    """
    Maintains an ordered sequence of Spatio-Temporal Scene Graphs.
    Tracks edge continuity, topological transitions, and interaction evolution.
    """

    def __init__(self):
        self.snapshots: List[SpatioTemporalSceneGraph] = []
        # Edge lifespans: key = (source_id, relation, target_id) -> list of intervals [(start_t, end_t)]
        self.edge_lifespans: Dict[Tuple[str, str, str], List[Tuple[float, float]]] = {}

    def add_snapshot(self, graph: SpatioTemporalSceneGraph) -> None:
        self.snapshots.append(graph)
        t = graph.timestamp

        # Update edge active intervals
        current_edges = {
            (e.source_id, e.relation, e.target_id) for e in graph.edges
        }

        for edge_key in current_edges:
            if edge_key not in self.edge_lifespans:
                self.edge_lifespans[edge_key] = [(t, t)]
            else:
                last_start, last_end = self.edge_lifespans[edge_key][-1]
                # If continuous within 1.5 seconds, extend current interval
                if t - last_end <= 1.5:
                    self.edge_lifespans[edge_key][-1] = (last_start, t)
                else:
                    self.edge_lifespans[edge_key].append((t, t))

    def get_active_relations_at(self, timestamp_sec: float) -> List[Tuple[str, str, str]]:
        """Finds all relationships active at a specific timestamp."""
        active = []
        for edge_key, intervals in self.edge_lifespans.items():
            for start_t, end_t in intervals:
                if start_t <= timestamp_sec <= end_t:
                    active.append(edge_key)
                    break
        return active

    def get_edge_summary(self) -> List[Dict[str, Any]]:
        """Returns structured longitudinal summary of edge relationships."""
        summary = []
        for (src, rel, tgt), intervals in self.edge_lifespans.items():
            total_duration = sum(end - start for start, end in intervals)
            summary.append({
                "source": src,
                "relation": rel,
                "target": tgt,
                "occurrences": len(intervals),
                "total_duration_sec": round(total_duration, 2),
                "first_seen": round(intervals[0][0], 2) if intervals else 0.0,
                "last_seen": round(intervals[-1][1], 2) if intervals else 0.0,
            })
        return summary
