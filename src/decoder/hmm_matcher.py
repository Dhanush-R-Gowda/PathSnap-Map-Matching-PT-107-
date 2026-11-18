"""HMM map matching: ML candidate probabilities (emissions) + network transitions + Viterbi.

Model
-----
Hidden state at observation t: a candidate position (arc, offset=snap_t*len).
Emission     : log P(arc | obs) from the ML classifier, renormalised within the
               observation (floor ``eps``), scaled by ``emission_weight``.
Transition   : Newson-Krumm style  P(j|i) ∝ exp(-|d_net(i,j) - d_gps| / beta),
               d_net = directed shortest-path distance, d_gps = straight-line GPS
               displacement.  Normalised over the *reachable* successors of each
               source state, so rows sum to 1.  Unreachable pairs get log-prob -inf.
Chain breaks : if no state at t is reachable from any state at t-1 (GPS gap, bad
               jump, disconnected roads) the chain restarts at t using emissions
               only; the break is reported rather than hidden.
Empty obs    : observations with no candidates are skipped; the next observation is
               linked to the last non-empty one using the summed time gap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.decoder.graph import INF, RoadGraph
from src.utils.geometry import project

NEG_INF = float("-inf")


@dataclass
class HMMConfig:
    beta: float = 30.0            # transition tolerance (m)
    emission_weight: float = 1.0
    eps: float = 1e-6             # probability floor
    max_speed_mps: float = 40.0   # bound on plausible travel
    min_search_m: float = 300.0   # minimum Dijkstra bound
    backward_tol_m: float = 25.0  # GPS-noise tolerance: small backwards offset on same arc = 0 m travel


@dataclass
class MatchResult:
    obs_arc: list                  # per observation: arc_id or None
    arc_sequence: list             # connected arc ids (consecutive duplicates collapsed, gaps bridged)
    breaks: list                   # observation indices where the chain restarted
    unmatched: list                # observation indices without candidates
    log_score: float = 0.0
    stats: dict = field(default_factory=dict)


def _log_norm_emissions(probs, cfg):
    p = np.maximum(np.asarray(probs, dtype=float), cfg.eps)
    p = p / p.sum()
    return cfg.emission_weight * np.log(p)


def decode(graph: RoadGraph, track, candidates, cfg: HMMConfig | None = None) -> MatchResult:
    """Decode the most likely arc sequence.

    candidates: list (len == len(track)) of lists of (arc_id, snap_t, probability).
    """
    cfg = cfg or HMMConfig()
    n = len(track)
    xy = np.array([project(lon, lat, graph.proj) for lon, lat, _ in track], dtype=float)
    ts = np.array([t for _, _, t in track], dtype=float)

    layers = [i for i in range(n) if candidates[i]]
    unmatched = [i for i in range(n) if not candidates[i]]
    obs_arc = [None] * n
    if not layers:
        return MatchResult(obs_arc, [], [], unmatched, stats={"n_obs": n})

    arc_ids = {i: np.array([c[0] for c in candidates[i]], dtype=np.int64) for i in layers}
    offs = {i: np.array([c[1] * graph.arc_len[c[0]] for c in candidates[i]]) for i in layers}
    lem = {i: _log_norm_emissions([c[2] for c in candidates[i]], cfg) for i in layers}

    score = {layers[0]: lem[layers[0]].copy()}
    back = {}
    breaks = []

    for k in range(1, len(layers)):
        p, c = layers[k - 1], layers[k]
        d_gps = float(np.hypot(*(xy[c] - xy[p])))
        dt = max(ts[c] - ts[p], 1.0)
        limit = max(cfg.min_search_m, 3.0 * d_gps + 150.0, 0.0)
        limit = min(limit, max(cfg.min_search_m, cfg.max_speed_mps * dt + 100.0))
        np_, nc = len(arc_ids[p]), len(arc_ids[c])
        trans = np.full((np_, nc), NEG_INF)
        for a in range(np_):
            row = np.full(nc, INF)
            for b in range(nc):
                row[b] = graph.network_distance(
                    int(arc_ids[p][a]), offs[p][a], int(arc_ids[c][b]), offs[c][b], limit
                )
                if (arc_ids[p][a] == arc_ids[c][b] and offs[c][b] < offs[p][a]
                        and offs[p][a] - offs[c][b] <= cfg.backward_tol_m):
                    row[b] = 0.0
            ok = np.isfinite(row)
            if ok.any():
                w = -np.abs(row[ok] - d_gps) / cfg.beta
                w = w - (w.max() + math.log(np.exp(w - w.max()).sum()))  # row-normalise (log-sum-exp)
                trans[a, ok] = w
        total = score[p][:, None] + trans + lem[c][None, :]
        best_prev = np.argmax(total, axis=0)
        best = total[best_prev, np.arange(nc)]
        if not np.isfinite(best).any():
            # Chain break: restart on emissions only.
            breaks.append(c)
            score[c] = lem[c].copy()
            back[c] = None
        else:
            score[c] = best
            back[c] = best_prev

    # Backtrack (segment by segment through breaks)
    chosen = {}
    last = layers[-1]
    idx = int(np.argmax(score[last]))
    total_score = float(score[last][idx])
    for k in range(len(layers) - 1, -1, -1):
        i = layers[k]
        chosen[i] = idx
        if k == 0:
            break
        if back[i] is None:
            idx = int(np.argmax(score[layers[k - 1]]))
        else:
            idx = int(back[i][idx])
    for i in layers:
        obs_arc[i] = int(arc_ids[i][chosen[i]])

    # Route: collapse duplicates, bridge consecutive arcs through shortest paths.
    seq: list[int] = []
    break_set = set(breaks)
    prev_i = None
    for i in layers:
        a = obs_arc[i]
        if prev_i is not None and obs_arc[prev_i] != a and i not in break_set:
            d_gps = float(np.hypot(*(xy[i] - xy[prev_i])))
            lim = max(cfg.min_search_m, 3.0 * d_gps + 150.0)
            bridge = graph.connecting_arcs(obs_arc[prev_i], a, lim)
            if bridge:
                seq.extend(x for x in bridge if not seq or seq[-1] != x)
        if not seq or seq[-1] != a:
            seq.append(a)
        prev_i = i
    return MatchResult(
        obs_arc, seq, breaks, unmatched, total_score,
        stats={"n_obs": n, "n_layers": len(layers), "n_breaks": len(breaks)},
    )


def route_continuity(graph: RoadGraph, arc_sequence) -> float:
    """Fraction of consecutive arc pairs that are directly connected (to(a)==from(b))."""
    if len(arc_sequence) < 2:
        return 1.0
    ok = sum(graph.arc_to[a] == graph.arc_from[b] for a, b in zip(arc_sequence[:-1], arc_sequence[1:]))
    return ok / (len(arc_sequence) - 1)
