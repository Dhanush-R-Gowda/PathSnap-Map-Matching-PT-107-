# Phase 2 Dataset Schema

Schema inspected from the tracked Phase 1 Parquet outputs. Identifier and evaluation columns are retained for grouping, evaluation, and HMM export; they are not model features.

## Input dataset summary

| Partition | Trips | Candidate rows | Raw GPS observations | Eligible observations | Excluded observations | Candidate misses | Miss rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training | 80 | 1,870,717 | 208,417 | 178,534 | 29,883 | 29,883 | 14.338% |
| Student test | 10 | 209,756 | 19,160 | 19,133 | 27 | 27 | 0.141% |

Training candidate rows after observation-level filtering: **1,742,906**. The filter keeps every candidate for an eligible observation. The candidate-miss denominator uses raw track observations, so observations with no generated candidate row are counted as misses.

## Field roles

| Role | Columns |
|---|---|
| Identifiers (not features) | `trip_id`, `obs_index`, `arc_id` |
| Target | `label` (`1` is the aligned ground-truth arc; `0` otherwise) |
| Evaluation-only | `is_gt_in_candidates` |
| Approved numeric features | `snap_distance_m`, `snap_t`, `arc_length_m`, `bearing_diff_rad`, `abs_bearing_diff_rad`, `gps_speed_mps`, `gps_heading_rad`, `dt_prev_s`, `dt_next_s`, `out_degree`, `in_degree`, `is_gap_prev`, `is_gap_next`, `heading_reliable` |

Candidate geometry is not stored as a geometry column in this Parquet schema. Candidate-to-road connectivity is supplied by each trip's raw `.arcs` file (directed from-node/to-node pairs), with `arc_id` indexing that trip-local arc list.

## Exact Parquet columns and dtypes

| Column | Type | Role | Missing training values |
|---|---|---|---:|
| `trip_id` | `object` | identifier | 0 |
| `obs_index` | `int32` | identifier | 0 |
| `arc_id` | `int32` | identifier | 0 |
| `snap_distance_m` | `float32` | feature | 0 |
| `snap_t` | `float32` | feature | 0 |
| `arc_length_m` | `float32` | feature | 0 |
| `bearing_diff_rad` | `float32` | feature | 0 |
| `abs_bearing_diff_rad` | `float32` | feature | 0 |
| `gps_speed_mps` | `float32` | feature | 0 |
| `gps_heading_rad` | `float32` | feature | 0 |
| `dt_prev_s` | `float32` | feature | 0 |
| `dt_next_s` | `float32` | feature | 0 |
| `out_degree` | `int16` | feature | 0 |
| `in_degree` | `int16` | feature | 0 |
| `is_gap_prev` | `int8` | feature | 0 |
| `is_gap_next` | `int8` | feature | 0 |
| `heading_reliable` | `int8` | feature | 0 |
| `label` | `int8` | target | 0 |
| `is_gt_in_candidates` | `bool` | evaluation-only | 0 |

## Data quality checks

- Training full-row duplicates: 0; duplicate candidate keys: 0.
- Training non-finite approved feature values: 0.
- Training candidate rows per represented observation: min 1, median 6.0, mean 9.84, max 150.
- Candidate miss rows with a nonempty candidate set: 11,654 observations.
- Raw observations with no candidate rows: 18,229.
- Feature matrix uses only the approved Phase 1 numeric columns listed above. Preprocessing is fitted within each training fold.
