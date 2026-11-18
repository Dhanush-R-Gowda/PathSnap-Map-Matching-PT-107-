"""Viterbi route decoding over per-observation ML candidate probabilities."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from hmm.transition import log_transition_probability
from hmm.viterbi_decoder import viterbi_decode
from src.utils.geometry import build_projection, project


def _gps_distance_m(a: tuple[float, float, float], b: tuple[float, float, float], projection: dict) -> float:
    ax, ay = project(a[0], a[1], projection)
    bx, by = project(b[0], b[1], projection)
    return math.hypot(bx - ax, by - ay)


def _arc_lengths(nodes: Sequence[tuple[float, float]], arcs: Sequence[tuple[int, int]], projection: dict) -> list[float]:
    lengths = []
    for from_node, to_node in arcs:
        ax, ay = project(nodes[from_node][0], nodes[from_node][1], projection)
        bx, by = project(nodes[to_node][0], nodes[to_node][1], projection)
        lengths.append(math.hypot(bx - ax, by - ay))
    return lengths


def _edge_score(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    arcs: Sequence[tuple[int, int]],
    lengths: Sequence[float],
    gps_distance: float,
    elapsed_seconds: float,
    beta_m: float,
) -> float | None:
    previous_arc = int(previous["arc_id"])
    current_arc = int(current["arc_id"])
    previous_t = float(previous["snap_t"])
    current_t = float(current["snap_t"])

    if previous_arc == current_arc:
        road_distance = abs(current_t - previous_t) * lengths[previous_arc]
    elif arcs[previous_arc][1] == arcs[current_arc][0]:
        road_distance = (1.0 - previous_t) * lengths[previous_arc] + current_t * lengths[current_arc]
    else:
        return None
    # The time interval provides a broad speed ceiling (55 m/s plus a 120 m
    # tolerance for map-matching offsets) so an impossible jump is not scored
    # as a plausible transition.
    if road_distance > max(120.0, 55.0 * elapsed_seconds):
        return None
    return log_transition_probability(road_distance, gps_distance, beta=beta_m, connected=True)


def decode_candidate_routes(
    track: Sequence[tuple[float, float, float]],
    nodes: Sequence[tuple[float, float]],
    arcs: Sequence[tuple[int, int]],
    candidate_probabilities: pd.DataFrame,
    *,
    beta_m: float = 50.0,
    reset_gap_seconds: float = 120.0,
) -> dict[str, Any]:
    """Return classifier top-1 and directed, connected Viterbi arc sequences.

    Gaps with no candidates and time jumps above ``reset_gap_seconds`` split
    the sequence into independent decoding windows. If no direct directed arc
    transition exists between adjacent candidate sets, decoding restarts at
    the current observation and records a warning instead of fabricating a
    connector road.
    """
    if beta_m <= 0 or reset_gap_seconds <= 0:
        raise ValueError("beta_m and reset_gap_seconds must be positive")
    observation_count = len(track)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_probabilities.itertuples(index=False):
        obs_id = int(getattr(row, "obs_index"))
        arc_id = int(getattr(row, "arc_id"))
        probability = float(getattr(row, "probability"))
        snap_t = float(getattr(row, "snap_t"))
        if not 0 <= obs_id < observation_count:
            raise ValueError(f"Candidate observation index {obs_id} is outside the track")
        if not 0 <= arc_id < len(arcs):
            raise ValueError(f"Candidate arc id {arc_id} is outside the road network")
        if not math.isfinite(probability) or probability < 0 or not math.isfinite(snap_t):
            raise ValueError("Candidate probabilities and snap positions must be finite and non-negative")
        groups[obs_id].append({"arc_id": arc_id, "probability": probability, "snap_t": snap_t})

    for candidates in groups.values():
        total = sum(item["probability"] for item in candidates)
        if total <= 0:
            raise ValueError("A non-empty candidate group has zero total probability")
        for item in candidates:
            item["probability"] /= total
        candidates.sort(key=lambda item: (item["arc_id"]))

    classifier = [None] * observation_count
    for obs_id, candidates in groups.items():
        classifier[obs_id] = max(candidates, key=lambda item: (item["probability"], -item["arc_id"]))["arc_id"]

    decoded: list[int | None] = [None] * observation_count
    warnings: list[str] = []
    if observation_count == 0:
        return {"classifier_arc_ids": classifier, "hmm_arc_ids": decoded, "decoder_status": "no_observations", "warnings": warnings}
    projection = build_projection(track)
    lengths = _arc_lengths(nodes, arcs, projection)
    chunk: list[int] = []
    chunk_transitions: list[dict[tuple[int, int], float]] = []

    def flush_chunk() -> None:
        nonlocal chunk, chunk_transitions
        if not chunk:
            return
        states = [[item["arc_id"] for item in groups[obs_id]] for obs_id in chunk]
        emissions = [
            {item["arc_id"]: math.log(max(item["probability"], 1e-12)) for item in groups[obs_id]}
            for obs_id in chunk
        ]
        transition_maps = [
            {
                (int(previous), int(current)): score
                for (previous, current), score in transition_map.items()
            }
            for transition_map in chunk_transitions
        ]
        path, _score = viterbi_decode(states, emissions, transition_maps)
        for obs_id, arc_id in zip(chunk, path):
            decoded[obs_id] = int(arc_id)
        chunk = []
        chunk_transitions = []

    for obs_id in range(observation_count):
        current_candidates = groups.get(obs_id, [])
        if not current_candidates:
            flush_chunk()
            warnings.append(f"Observation {obs_id} has no candidate arcs and remains unmatched.")
            continue
        if not chunk:
            chunk = [obs_id]
            continue

        previous_obs = chunk[-1]
        delta_t = float(track[obs_id][2] - track[previous_obs][2])
        if delta_t <= 0:
            flush_chunk()
            warnings.append(f"Observation {obs_id} has a non-positive time interval; decoder restarted.")
            chunk = [obs_id]
            continue
        if delta_t > reset_gap_seconds:
            flush_chunk()
            warnings.append(f"GPS gap of {delta_t:.1f}s before observation {obs_id}; decoder restarted.")
            chunk = [obs_id]
            continue

        gps_distance = _gps_distance_m(track[previous_obs], track[obs_id], projection)
        transition_map: dict[tuple[int, int], float] = {}
        for previous in groups[previous_obs]:
            for current in current_candidates:
                score = _edge_score(previous, current, arcs, lengths, gps_distance, delta_t, beta_m)
                if score is not None:
                    transition_map[(previous["arc_id"], current["arc_id"])] = score
        if not transition_map:
            flush_chunk()
            warnings.append(f"No directed road transition connects observations {previous_obs} and {obs_id}; decoder restarted.")
            chunk = [obs_id]
            continue
        chunk_transitions.append(transition_map)
        chunk.append(obs_id)

    flush_chunk()
    if not any(value is not None for value in decoded):
        status = "no_candidates"
    elif warnings:
        status = "completed_with_resets"
    else:
        status = "completed"
    return {
        "classifier_arc_ids": classifier,
        "hmm_arc_ids": decoded,
        "decoder_status": status,
        "warnings": warnings,
    }
