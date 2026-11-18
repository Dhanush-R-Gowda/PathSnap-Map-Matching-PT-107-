# Phase 1 Review — PathSnap Map-Matching (Pass 2)

**Commits reviewed:** `80eb494` (implementation) + `44ed668` (fix)
**Verdict:** APPROVED

---

## Evidence Summary

### Prior blocker verification

**B1 — candidate_miss_rates.csv missing `total_obs` and `miss_count`**
Status: **RESOLVED**

The fix commit (`44ed668`) adds `trip_obs_counts` and `trip_miss_counts` dicts to `build_partition_features` (src/features/build_features.py, lines ~175–182 and the return dict). `compute_miss_rates` in `src/data/miss_rate.py` now reads both and writes them to the CSV row. Confirmed: `results/experiments/candidate_miss_rates.csv` header is `trip_id,total_obs,miss_count,miss_rate_pct,radius_m`.

---

### All requirements

| # | Requirement | Status |
|---|-------------|--------|
| 1 | requirements.txt at repo root with specified packages | ✅ |
| 2 | geometry.py: build_projection, project, point_to_segment, arc_length, bearing, wrap_angle | ✅ |
| 3 | loader.py: load_track, load_nodes, load_arcs, load_route, load_trip, discover_trips with validation | ✅ |
| 4 | candidates.py: cKDTree on arc midpoints, radius expanded by max_arc_half_length | ✅ |
| 5 | alignment.py: non-decreasing DP, prefix-min recurrence, band restriction, gap handling | ✅ |
| 6 | build_features.py: exact 19-column schema, correct dtypes | ✅ |
| 7 | arc_id present as metadata; route_position absent; neither used as model features | ✅ |
| 8 | train_candidates.parquet: 1,870,717 rows, non-empty | ✅ |
| 9 | student_test_candidates.parquet: 209,756 rows, non-empty | ✅ |
| 10 | candidate_miss_rates.csv: trip_id, total_obs, miss_count, miss_rate_pct, radius_m | ✅ |
| 11 | 6 Folium HTML files present (validation + network for trips 00000001/00000002/00000033) | ✅ |
| 12 | tests/test_phase1.py: 20 tests (≥16 required) | ✅ |
| 13 | All 20 tests pass per commit message evidence | ✅ |
| 14 | Transition model is truly non-decreasing: spot-check with synthetic 5-arc route confirms jumps of 2 (result [0,2,4]) | ✅ |
| 15 | KDTree search radius = radius + max_arc_half_length (candidates.py line ~60) | ✅ |

---

## Key code findings

**Transition model (requirement 14):** `alignment.py` prefix-min loop at row `i` accumulates `min(prev_dp[lo], ..., prev_dp[j])` for each `j` in the band; there is no +1 restriction. A synthetic test with GPS observations snapping optimally to arcs 0, 2, 4 (skipping 1 and 3) correctly returns `obs_to_route_pos = [0, 2, 4]`.

**KDTree radius expansion (requirement 15):** `candidates.py` line ~60: `search_radius = radius + max_arc_half_length` — correct.

**Schema (requirement 6):** Parquet dtypes read from disk match the specification exactly: trip_id(object/str), obs_index(int32), arc_id(int32), snap_distance_m(float32), snap_t(float32), arc_length_m(float32), bearing_diff_rad(float32), abs_bearing_diff_rad(float32), gps_speed_mps(float32), gps_heading_rad(float32), dt_prev_s(float32), dt_next_s(float32), out_degree(int16), in_degree(int16), is_gap_prev(int8), is_gap_next(int8), heading_reliable(int8), label(int8), is_gt_in_candidates(bool).

No blockers remain.
