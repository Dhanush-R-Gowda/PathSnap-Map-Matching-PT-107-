"""Directed road-network graph with bounded shortest-path queries.

Conventions
-----------
* ``arc_id`` is the row index in the trip's ``.arcs`` file (trip-local).
* An arc is directed ``from_node -> to_node`` and has a metric length.
* A candidate position is ``(arc_id, offset)`` with ``offset = snap_t * arc_length``
  metres from the arc's start node.
* Network distance between candidate positions ``(a1, s1) -> (a2, s2)``:
    - same arc and ``s2 >= s1``:  ``s2 - s1``
    - otherwise: ``(len(a1) - s1) + SP(to(a1), from(a2)) + s2`` where SP is the
      directed shortest path between nodes (Dijkstra).
  Unreachable pairs return ``inf`` -- they are never given a finite probability.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict

import numpy as np

from src.utils.geometry import arc_length, build_projection, project

INF = math.inf


class RoadGraph:
    def __init__(self, nodes, arcs, proj=None, track=None):
        if proj is None:
            if track is None:
                raise ValueError("Provide proj or track")
            proj = build_projection(track)
        self.proj = proj
        self.nodes = nodes
        self.arcs = arcs
        self.n_arcs = len(arcs)
        self.arc_from = np.fromiter((a[0] for a in arcs), dtype=np.int64, count=len(arcs))
        self.arc_to = np.fromiter((a[1] for a in arcs), dtype=np.int64, count=len(arcs))
        self.arc_len = np.array(
            [arc_length(nodes, f, t, proj) for f, t in arcs], dtype=np.float64
        )
        self.out_arcs: dict[int, list[int]] = defaultdict(list)
        for arc_id, (f, _t) in enumerate(arcs):
            self.out_arcs[f].append(arc_id)
        self._cache: dict[tuple[int, int], tuple[dict, dict]] = {}

    # ------------------------------------------------------------------
    def _bounded_dijkstra(self, source: int, limit: float):
        """Dijkstra from a node, exploring only nodes within ``limit`` metres.

        Returns (dist, parent_arc): dist[node] -> metres, parent_arc[node] -> arc used
        to reach it.  Results are cached per (source, limit-bucket).
        """
        bucket = int(math.ceil(limit / 250.0))
        key = (source, bucket)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        lim = bucket * 250.0
        dist = {source: 0.0}
        parent: dict[int, int] = {}
        heap = [(0.0, source)]
        done = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for arc_id in self.out_arcs.get(u, ()):
                nd = d + self.arc_len[arc_id]
                if nd > lim:
                    continue
                v = int(self.arc_to[arc_id])
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    parent[v] = arc_id
                    heapq.heappush(heap, (nd, v))
        if len(self._cache) > 20000:
            self._cache.clear()
        self._cache[key] = (dist, parent)
        return dist, parent

    def node_distance(self, src: int, dst: int, limit: float) -> float:
        if src == dst:
            return 0.0
        dist, _ = self._bounded_dijkstra(src, limit)
        d = dist.get(dst, INF)
        return d if d <= limit else INF

    def network_distance(self, a1, s1, a2, s2, limit: float) -> float:
        """Directed travel distance (metres) from (a1,s1) to (a2,s2); inf if > limit."""
        if a1 == a2 and s2 >= s1:
            return s2 - s1
        head = self.arc_len[a1] - s1
        if head + s2 > limit:
            return INF
        sp = self.node_distance(int(self.arc_to[a1]), int(self.arc_from[a2]), limit - head - s2)
        if sp == INF:
            return INF
        return head + sp + s2

    def connecting_arcs(self, a1: int, a2: int, limit: float) -> list[int] | None:
        """Arcs strictly between a1 and a2 on the shortest path (None if unreachable)."""
        if a1 == a2:
            return []
        src, dst = int(self.arc_to[a1]), int(self.arc_from[a2])
        if src == dst:
            return []
        dist, parent = self._bounded_dijkstra(src, limit)
        if dst not in dist:
            return None
        path = []
        node = dst
        while node != src:
            arc_id = parent[node]
            path.append(arc_id)
            node = int(self.arc_from[arc_id])
        path.reverse()
        return path

    def arc_polyline(self, arc_id: int):
        f, t = self.arcs[arc_id]
        return [tuple(self.nodes[f]), tuple(self.nodes[t])]
