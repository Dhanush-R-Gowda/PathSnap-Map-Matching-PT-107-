"""HMM / graph / Viterbi tests: synthetic networks and real trip 00000001."""

import math
import os

import numpy as np
import pytest

from src.data.loader import load_trip_for_inference
from src.decoder.graph import INF, RoadGraph
from src.decoder.hmm_matcher import HMMConfig, decode, route_continuity
from src.utils.geometry import build_projection

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIP_DIR = os.path.join(REPO_ROOT, "data", "raw", "Student folder", "train", "00000001")

# ~0.001 deg lon ≈ 100 m at the equator. Straight east-west road:
# nodes 0..3 at x=0,100,200,300 (m); arcs: 0:0->1, 1:1->2, 2:2->3, 3:3->2 (reverse),
# 4: 10->11 disconnected island far north.
DEG = 1e-3


def _graph():
    nodes = [(i * DEG, 0.0) for i in range(4)] + [(0.0, 0.01), (DEG, 0.01)]
    arcs = [(0, 1), (1, 2), (2, 3), (3, 2), (4, 5)]
    track = [(0.0, 0.0, 0.0), (DEG, 0.0, 1.0)]
    return RoadGraph(nodes, arcs, track=track)


def test_arc_lengths_and_lookup():
    g = _graph()
    assert g.n_arcs == 5
    assert all(abs(g.arc_len[i] - 111.2) < 2 for i in range(4))
    assert list(g.arc_from[:3]) == [0, 1, 2] and list(g.arc_to[:3]) == [1, 2, 3]


def test_shortest_path_directed_and_unreachable():
    g = _graph()
    L = g.arc_len[0]
    assert g.node_distance(0, 3, 1000) == pytest.approx(3 * L, rel=1e-6)
    assert g.node_distance(3, 0, 1000) == INF              # only 3->2 exists backwards
    assert g.node_distance(0, 4, 100000) == INF            # disconnected island
    assert g.node_distance(0, 3, 100) == INF               # beyond search bound


def test_network_distance_uses_offsets():
    g = _graph()
    L = g.arc_len[0]
    assert g.network_distance(0, 20, 0, 70, 1000) == pytest.approx(50)
    assert g.network_distance(0, 20, 2, 10, 1000) == pytest.approx((L - 20) + L + 10)
    assert g.network_distance(0, 20, 4, 5, 1e6) == INF


def test_connecting_arcs():
    g = _graph()
    assert g.connecting_arcs(0, 2, 1000) == [1]
    assert g.connecting_arcs(0, 4, 1000) is None


def _track(xs):
    return [(x * DEG, 0.0, float(i)) for i, x in enumerate(xs)]


def test_viterbi_prefers_connected_route_over_noisy_emission():
    g = _graph()
    track = _track([0.2, 0.5, 1.5, 2.5])
    cands = [
        [(0, 0.2, 0.6), (3, 0.9, 0.4)],
        [(0, 0.5, 0.5), (3, 0.5, 0.5)],
        [(1, 0.5, 0.45), (3, 0.5, 0.55)],  # classifier slightly prefers reverse arc
        [(2, 0.5, 0.9), (3, 0.5, 0.1)],
    ]
    res = decode(g, track, cands, HMMConfig(beta=30))
    assert res.obs_arc[2] == 1 and res.obs_arc[3] == 2
    assert res.arc_sequence[-2:] == [1, 2]
    assert route_continuity(g, res.arc_sequence) == 1.0


def test_empty_observation_and_chain_break():
    g = _graph()
    track = _track([0.2, 1.5, 2.5])
    track[2] = (track[2][0], 0.01, 2.0)                    # jumps to disconnected island
    cands = [[(0, 0.2, 1.0)], [], [(4, 0.5, 1.0)]]
    res = decode(g, track, cands)
    assert res.unmatched == [1]
    assert res.obs_arc == [0, None, 4]
    assert res.breaks == [2]                               # unreachable: restart, reported
    assert math.isfinite(res.log_score)


def test_no_candidates_at_all():
    g = _graph()
    res = decode(g, _track([0.1, 0.2]), [[], []])
    assert res.obs_arc == [None, None] and res.arc_sequence == []


def test_unreachable_never_gets_probability():
    g = _graph()
    track = _track([0.2, 0.5])
    track[1] = (0.0, 0.01, 1.0)
    res = decode(g, track, [[(0, 0.2, 1.0)], [(4, 0.1, 1.0)]])
    assert res.breaks == [1]


@pytest.mark.skipif(not os.path.isdir(TRIP_DIR), reason="raw data not available")
def test_real_trip_with_distance_heuristic_emissions():
    from src.features.build_features import build_trip_features_inference

    d = load_trip_for_inference(TRIP_DIR)
    df = build_trip_features_inference("00000001", d["track"], d["nodes"], d["arcs"])
    g = RoadGraph(d["nodes"], d["arcs"], track=d["track"])
    cands = [[] for _ in d["track"]]
    for r in df.itertuples():
        cands[r.obs_index].append((r.arc_id, r.snap_t, math.exp(-r.snap_distance_m / 10.0)))
    res = decode(g, d["track"], cands)
    assert len(res.obs_arc) == len(d["track"])
    assert res.stats["n_layers"] > 0.9 * len(d["track"])
    assert route_continuity(g, res.arc_sequence) > 0.9
