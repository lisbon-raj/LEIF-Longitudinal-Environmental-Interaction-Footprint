"""
Spatio-Temporal Scene Graph Representation.
Encapsulates person, object, environmental region nodes, and interaction/spatial edges.
"""

from typing import List, Dict, Any, Optional, Tuple
from leif.types import (
    Entity,
    EntityCategory,
    InteractionPrediction,
    SceneGraphNode,
    SceneGraphEdge,
    SpatioTemporalSceneGraph,
)
from leif.tracking.tracklet import Tracklet


class SceneGraphBuilder:
    """
    Constructs an instantaneous or window-level Spatio-Temporal Scene Graph
    connecting Person, Object, Location/Region entities with Interaction and Spatial edges.
    """

    @staticmethod
    def build_graph(
        active_tracks: List[Tracklet],
        region_entities: List[Entity],
        interactions: List[InteractionPrediction],
        timestamp_sec: float,
        frame_idx: int,
    ) -> SpatioTemporalSceneGraph:
        graph = SpatioTemporalSceneGraph(timestamp=timestamp_sec, frame_idx=frame_idx)

        # 1. Add Environmental Region Nodes
        for r in region_entities:
            node = SceneGraphNode(
                node_id=r.entity_id,
                label=r.class_name,
                category=r.category.value,
                bbox=(r.bbox.x1, r.bbox.y1, r.bbox.x2, r.bbox.y2),
                state=dict(r.state),
            )
            graph.nodes[r.entity_id] = node

        # 2. Add Tracked Person & Object Nodes
        for track in active_tracks:
            node = SceneGraphNode(
                node_id=track.track_id,
                label=f"{track.class_name}:{track.track_id}",
                category=track.category.value,
                bbox=(
                    track.current_bbox.x1,
                    track.current_bbox.y1,
                    track.current_bbox.x2,
                    track.current_bbox.y2,
                ),
                state=dict(track.state),
            )
            graph.nodes[track.track_id] = node

            # 3. Add Spatial Region Edges ("located_in")
            if track.location_id:
                region_node_id = f"region_{track.location_id}"
                if region_node_id in graph.nodes:
                    graph.edges.append(
                        SceneGraphEdge(
                            source_id=track.track_id,
                            target_id=region_node_id,
                            relation="located_in",
                            confidence=1.0,
                            temporal_frame=frame_idx,
                        )
                    )

        # 4. Add Human-Object Interaction Edges
        for hoi in interactions:
            if hoi.subject_id in graph.nodes and hoi.object_id in graph.nodes:
                graph.edges.append(
                    SceneGraphEdge(
                        source_id=hoi.subject_id,
                        target_id=hoi.object_id,
                        relation=hoi.interaction_type.value,
                        confidence=hoi.confidence,
                        temporal_frame=frame_idx,
                    )
                )

        return graph

    @staticmethod
    def extract_semantic_triplets(graph: SpatioTemporalSceneGraph) -> List[Tuple[str, str, str, float]]:
        """
        Extracts semantic (subject, relation, object, confidence) triplets from the graph.
        """
        triplets = []
        for edge in graph.edges:
            src = graph.nodes.get(edge.source_id)
            tgt = graph.nodes.get(edge.target_id)
            src_label = src.label if src else edge.source_id
            tgt_label = tgt.label if tgt else edge.target_id
            triplets.append((src_label, edge.relation, tgt_label, edge.confidence))
        return triplets
