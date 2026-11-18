import pandas as pd

from hmm.integrated_decoder import decode_candidate_routes


def test_viterbi_prefers_connected_sequence_over_local_top1():
    track = [(0.0001, 0.0001, 0.0), (0.0009, 0.0001, 1.0)]
    nodes = [(0.0, 0.0001), (0.0005, 0.0001), (0.001, 0.0001), (0.0, 0.001), (0.001, 0.001)]
    arcs = [(0, 1), (1, 2), (3, 4)]
    candidates = pd.DataFrame([
        {"obs_index": 0, "arc_id": 0, "snap_t": 0.2, "probability": 0.4},
        {"obs_index": 0, "arc_id": 2, "snap_t": 0.8, "probability": 0.6},
        {"obs_index": 1, "arc_id": 1, "snap_t": 0.8, "probability": 0.7},
        {"obs_index": 1, "arc_id": 2, "snap_t": 0.2, "probability": 0.3},
    ])

    decoded = decode_candidate_routes(track, nodes, arcs, candidates)

    assert decoded["classifier_arc_ids"] == [2, 1]
    assert decoded["hmm_arc_ids"] == [0, 1]
    assert decoded["decoder_status"] == "completed"


def test_viterbi_leaves_empty_observation_unmatched_and_resets():
    track = [(0.0001, 0.0001, 0.0), (0.0005, 0.0001, 1.0), (0.0009, 0.0001, 2.0)]
    nodes = [(0.0, 0.0001), (0.0005, 0.0001), (0.001, 0.0001)]
    arcs = [(0, 1), (1, 2)]
    candidates = pd.DataFrame([
        {"obs_index": 0, "arc_id": 0, "snap_t": 0.2, "probability": 1.0},
        {"obs_index": 2, "arc_id": 1, "snap_t": 0.8, "probability": 1.0},
    ])

    decoded = decode_candidate_routes(track, nodes, arcs, candidates)

    assert decoded["hmm_arc_ids"] == [0, None, 1]
    assert decoded["decoder_status"] == "completed_with_resets"
    assert "no candidate arcs" in decoded["warnings"][0]
