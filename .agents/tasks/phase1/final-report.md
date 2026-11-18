# PathSnap Phase 1 — Final Report

**Status:** APPROVED  
**Branch:** `feature/data-features`  
**Commits:** `80eb494` (implementation) · `44ed668` (fix)  
**Date:** 2026-11-18

---

## 1. Files Created

### Source code

| Path | Description |
|------|-------------|
| `requirements.txt` | Project dependencies (Python 3.11+) |
| `src/__init__.py` | Package marker |
| `src/data/__init__.py` | Package marker |
| `src/data/loader.py` | `load_track`, `load_nodes`, `load_arcs`, `load_route`, `load_trip`, `discover_trips` with validation |
| `src/data/candidates.py` | `build_spatial_index` (cKDTree on arc midpoints), `get_candidates` at 50 m radius |
| `src/data/alignment.py` | Monotone non-decreasing DP with prefix-min optimisation and band restriction |
| `src/data/miss_rate.py` | `compute_miss_rates` — per-trip CSV writer with `total_obs`, `miss_count`, `miss_rate_pct`, `radius_m` |
| `src/features/__init__.py` | Package marker |
| `src/features/build_features.py` | `build_trip_features`, `build_partition_features` — 19-column schema |
| `src/features/make_maps.py` | Folium validation and network maps for trips 00000001, 00000002, 00000033 |
| `src/utils/__init__.py` | Package marker |
| `src/utils/geometry.py` | `build_projection`, `project`, `point_to_segment`, `arc_length`, `bearing`, `wrap_angle` |
| `tests/test_phase1.py` | 20 unit tests |

### Generated data

| Path | Description |
|------|-------------|
| `data/processed/train_candidates.parquet` | Training candidate features, 1,870,717 rows |
| `data/processed/student_test_candidates.parquet` | Test candidate features, 209,756 rows |
| `results/experiments/candidate_miss_rates.csv` | Per-trip miss rates across all 80 training trips |
| `visualizations/maps/validation_00000001.html` | Folium validation map — trip 00000001 |
| `visualizations/maps/validation_00000002.html` | Folium validation map — trip 00000002 |
| `visualizations/maps/validation_00000033.html` | Folium validation map — trip 00000033 |
| `visualizations/maps/network_00000001.html` | Folium road-network map — trip 00000001 |
| `visualizations/maps/network_00000002.html` | Folium road-network map — trip 00000002 |
| `visualizations/maps/network_00000033.html` | Folium road-network map — trip 00000033 |

---

## 2. Test Results

Command run: `pytest tests/test_phase1.py -v`  
Result: **20 passed** in 1.88 s (Python 3.14.6, pytest 8.4.2)

| # | Test | Result |
|---|------|--------|
| 1 | `test_load_track_trip1` | ✅ PASS |
| 2 | `test_load_track_trip2` | ✅ PASS |
| 3 | `test_load_nodes_trip1` | ✅ PASS |
| 4 | `test_load_nodes_trip2` | ✅ PASS |
| 5 | `test_load_arcs_trip1` | ✅ PASS |
| 6 | `test_load_arcs_trip2` | ✅ PASS |
| 7 | `test_load_route_trip1` | ✅ PASS |
| 8 | `test_load_route_trip2` | ✅ PASS |
| 9 | `test_load_arcs_trip1_index2641` | ✅ PASS |
| 10 | `test_load_track_trip1_first_lon` | ✅ PASS |
| 11 | `test_point_to_segment_midpoint` | ✅ PASS |
| 12 | `test_point_to_segment_endpoint_a` | ✅ PASS |
| 13 | `test_bearing_north` | ✅ PASS |
| 14 | `test_bearing_east` | ✅ PASS |
| 15 | `test_candidates_within_radius` | ✅ PASS |
| 16 | `test_obs_to_route_pos_nondecreasing` | ✅ PASS |
| 17 | `test_label_values` | ✅ PASS |
| 18 | `test_at_most_one_label1_per_obs` | ✅ PASS |
| 19 | `test_heading_reliable_zero_on_gaps` | ✅ PASS |
| 20 | `test_is_gap_prev_on_gaps` | ✅ PASS |

---

## 3. Dataset Statistics

### `train_candidates.parquet`

| Metric | Value |
|--------|-------|
| Total rows | 1,870,717 |
| Columns | 19 |
| `label = 1` (correct candidate) | 178,534 |
| `label = 0` (incorrect candidate) | 1,692,183 |
| Fraction `label = 1` | 9.54% |
| Ratio `label=1 / label=0` | 0.1055 |

**Column schema (all dtypes confirmed from disk):**  
`trip_id` (object), `obs_index` (int32), `arc_id` (int32), `snap_distance_m` (float32), `snap_t` (float32), `arc_length_m` (float32), `bearing_diff_rad` (float32), `abs_bearing_diff_rad` (float32), `gps_speed_mps` (float32), `gps_heading_rad` (float32), `dt_prev_s` (float32), `dt_next_s` (float32), `out_degree` (int16), `in_degree` (int16), `is_gap_prev` (int8), `is_gap_next` (int8), `heading_reliable` (int8), `label` (int8), `is_gt_in_candidates` (bool)

---

### Candidate miss rates (`candidate_miss_rates.csv`)

Computed from 80 training trips at a 50 m candidate radius.

| Scope | Observations | Misses | Miss rate |
|-------|-------------|--------|-----------|
| All 80 trips | 190,188 | 11,654 | **6.13%** |
| 78 trips (excluding 00000054 and 00000033) | 181,595 | 5,759 | **3.17%** |

56 of 80 trips (70%) have a miss rate of 0%.

#### Trips with miss rate > 5% (12 trips)

| Trip ID | Miss rate | Misses / Observations |
|---------|-----------|----------------------|
| 00000054 | 81.32% | 2272 / 2794 |
| 00000033 | 62.48% | 3623 / 5799 |
| 00000093 | 32.54% | 394 / 1211 |
| 00000092 | 32.21% | 2542 / 7892 |
| 00000064 | 14.24% | 298 / 2093 |
| 00000042 | 10.97% | 849 / 7739 |
| 00000078 | 8.78% | 255 / 2904 |
| 00000063 | 7.77% | 216 / 2781 |
| 00000066 | 7.06% | 292 / 4138 |
| 00000051 | 6.20% | 61 / 984 |
| 00000002 | 5.32% | 358 / 6733 |
| 00000043 | 5.10% | 116 / 2273 |

#### Trips with miss rate > 20% (4 trips)

| Trip ID | Miss rate | Misses / Observations |
|---------|-----------|----------------------|
| 00000054 | 81.32% | 2272 / 2794 |
| 00000033 | 62.48% | 3623 / 5799 |
| 00000093 | 32.54% | 394 / 1211 |
| 00000092 | 32.21% | 2542 / 7892 |

Trips 00000054 and 00000033 are confirmed outliers: together they account for 5,895 of the 11,654 total misses (50.6%) despite comprising only 8,593 of 190,188 total observations.

---

## 4. Remaining Notes

**No blockers remain.** All review findings from Pass 2 were resolved prior to this report.

- **`arc_id` and `route_position` exclusion confirmed.** Neither is present as a model feature. `arc_id` is retained as metadata only; `route_position` is absent from the parquet schema.
- **Feature scaling deferred to Chethan.** All numeric features in the parquet are raw, unscaled values, as agreed.
- **Outlier trips for Phase 2.** Trips 00000054, 00000033, 00000092, and 00000093 have high miss rates at 50 m. If the classifiers trained on these trips show degraded performance, Chethan should consider weighting or flagging these trips during cross-validation rather than increasing the radius globally.
- **GPS-gap indicators present.** `is_gap_prev`, `is_gap_next`, and `heading_reliable` are populated correctly; the `test_heading_reliable_zero_on_gaps` and `test_is_gap_prev_on_gaps` tests verify this.
- **Student test partition.** `student_test_candidates.parquet` (209,756 rows, 10 trips) is generated and ready for evaluation use. No ground-truth labels are assumed available for this partition.
- **Transition model.** The DP is non-decreasing (allows skipping route positions forward), not stay-or-advance. This is the correct choice for GPS traces where the vehicle may move several arc lengths between observations. The synthetic test `test_obs_to_route_pos_nondecreasing` confirms skip-forward behaviour (e.g., alignment [0, 2, 4] on a 5-arc route).
