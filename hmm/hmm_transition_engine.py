
"""Directed road-network transition engine for PathSnap."""

import math
import networkx as nx

from src.utils.geometry import arc_length
from .transition import (
    transition_probability,
    log_transition_probability,
)


class DirectedRoadNetwork:
    """Build and query a directed road network for HMM transitions."""

    def __init__(self, nodes, arcs, proj):
        self.nodes = nodes
        self.arcs = arcs
        self.proj = proj

        self.graph = nx.DiGraph()
        self.arc_lengths = []
        self.shortest_path_cache = {}

        self._build_graph()

    def _build_graph(self):
        """Build a directed graph using the trip's nodes and arcs."""

        for arc_id, (from_id, to_id) in enumerate(self.arcs):

            if not (0 <= from_id < len(self.nodes)):
                raise ValueError(f"Invalid source node in arc {arc_id}")

            if not (0 <= to_id < len(self.nodes)):
                raise ValueError(f"Invalid destination node in arc {arc_id}")

            length = arc_length(
                self.nodes,
                from_id,
                to_id,
                self.proj
            )

            if not math.isfinite(length) or length < 0:
                raise ValueError(f"Invalid length for arc {arc_id}")

            self.arc_lengths.append(length)

            # Preserve road direction: from_id -> to_id
            if self.graph.has_edge(from_id, to_id):
                # Keep the shortest parallel edge for graph routing.
                self.graph[from_id][to_id]["weight"] = min(
                    self.graph[from_id][to_id]["weight"],
                    length
                )
            else:
                self.graph.add_edge(
                    from_id,
                    to_id,
                    weight=length
                )

    def shortest_path_distance(self, source, target):
        """Return directed shortest-path distance in metres."""

        key = (source, target)

        if key not in self.shortest_path_cache:
            try:
                distance = nx.shortest_path_length(
                    self.graph,
                    source=source,
                    target=target,
                    weight="weight"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                distance = float("inf")

            self.shortest_path_cache[key] = distance

        return self.shortest_path_cache[key]

    def network_distance(
        self,
        previous_arc_id,
        previous_t,
        current_arc_id,
        current_t
    ):
        """
        Calculate distance between two positions on directed road arcs.

        Candidate positions are represented by (arc_id, snap_t).
        snap_t is the fraction of the arc traversed from its source.
        """

        if not (0 <= previous_arc_id < len(self.arcs)):
            raise ValueError("Invalid previous_arc_id")

        if not (0 <= current_arc_id < len(self.arcs)):
            raise ValueError("Invalid current_arc_id")

        if not (0.0 <= previous_t <= 1.0):
            raise ValueError("previous_t must be between 0 and 1")

        if not (0.0 <= current_t <= 1.0):
            raise ValueError("current_t must be between 0 and 1")

        prev_source, prev_target = self.arcs[previous_arc_id]
        curr_source, curr_target = self.arcs[current_arc_id]

        prev_length = self.arc_lengths[previous_arc_id]
        curr_length = self.arc_lengths[current_arc_id]

        # Case 1: Both candidates lie on the same directed arc.
        if previous_arc_id == current_arc_id:

            if current_t >= previous_t:
                return (current_t - previous_t) * prev_length

            # Backward movement requires a directed loop.
            loop_distance = self.shortest_path_distance(
                prev_target,
                prev_source
            )

            if math.isinf(loop_distance):
                return float("inf")

            return (
                (1.0 - previous_t) * prev_length
                + loop_distance
                + current_t * curr_length
            )

        # Case 2: Different candidate arcs.
        middle_distance = self.shortest_path_distance(
            prev_target,
            curr_source
        )

        if math.isinf(middle_distance):
            return float("inf")

        return (
            (1.0 - previous_t) * prev_length
            + middle_distance
            + current_t * curr_length
        )

    def transition_score(
        self,
        previous_arc_id,
        previous_t,
        current_arc_id,
        current_t,
        gps_distance,
        beta=50.0
    ):
        """Calculate probability-like transition score."""

        road_distance = self.network_distance(
            previous_arc_id,
            previous_t,
            current_arc_id,
            current_t
        )

        if math.isinf(road_distance):
            return 0.0

        return transition_probability(
            road_distance=road_distance,
            gps_distance=gps_distance,
            beta=beta
        )

    def log_transition_score(
        self,
        previous_arc_id,
        previous_t,
        current_arc_id,
        current_t,
        gps_distance,
        beta=50.0
    ):
        """Calculate log transition score for Viterbi."""

        road_distance = self.network_distance(
            previous_arc_id,
            previous_t,
            current_arc_id,
            current_t
        )

        if math.isinf(road_distance):
            return float("-inf")

        return log_transition_probability(
            road_distance=road_distance,
            gps_distance=gps_distance,
            beta=beta
        )
