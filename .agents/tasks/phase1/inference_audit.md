# Inference-Time Preprocessing Audit — Phase 1

## Summary Answer

**No inference pipeline exists.** The only end-to-end trip function is `build_trip_features` in
`src/features/build_features.py`, and it unconditionally loads and uses the `.route` file for label
alignment. Every row it produces contains the training-only columns `label` and
`is_gt_in_candidates`, and the route-dependent DP step (`align_observations`) cannot be bypassed.
There is no function that accepts `(track, nodes, arcs)` alone and returns model-ready features.
Adding inference support requires a targeted refactor of `build_features.py` and a matching
loader-level change — both are small and contained.

---

## Evidence by File

### `src/data/loader.py`

**Symbols:** `load_track`, `load_nodes`, `load_arcs`, `load_route`, `load_trip`, `discover_trips`.

`load_trip` (lines 149–157) always calls all four loaders and returns a dict with keys
`track`, `nodes`, `arcs`, **and** `route`. There is no variant that omits the route.
`load_route` raises `FileNotFoundError` when the `.route` file is absent.

**Can run without .route?** No. `load_trip` hardwires the route load; calling it on an
inference-only directory raises immediately.

---

### `src/data/candidates.py`

**Symbols:** `build_spatial_index`, `get_candidates`.

`build_spatial_index(nodes, arcs, proj)` takes only network data — no route, no track labels.
`get_candidates(gps_lon, gps_lat, proj, tree, nodes, arcs, arc_ids_array, max_arc_half_length,
radius)` takes only a GPS point and the spatial index — no route.

Both functions are entirely route-free. They return `(arc_id, snap_distance_m, snap_t)` tuples
sorted by `snap_distance_m` ascending, satisfying requirement 2 (candidate generation) and
requirement 4 (candidate ordering) exactly as needed.

**Can run without .route?** Yes — these two functions are ready for inference as-is.

---

### `src/data/alignment.py`

**Symbols:** `align_observations`, `_cost_row`.

`align_observations(track, nodes, arcs, route, proj, band_width=100)` requires a populated `route`
list. Its sole purpose is label assignment: it returns `obs_to_route_pos` and `obs_to_arc`, both
used only in `build_trip_features` to set `label = int(arc_id == gt_arc)` and
`is_gt_in_candidates`. The function performs O(N×M) DP work and would fail or return meaningless
results with an empty route.

**Can run without .route?** No, and it should not be called at inference time at all.

---

### `src/utils/geometry.py`

**Symbols:** `build_projection`, `project`, `point_to_segment`, `arc_length`, `bearing`,
`wrap_angle`.

All functions operate purely on coordinates. None reference a route.
`build_projection(track)` derives its projection from the GPS track bounding box — it needs only
the track, which is available at inference time.

**Can run without .route?** Yes — no dependency whatsoever.

---

### `src/features/build_features.py`

**Symbols:** `build_trip_features`, `build_partition_features`, `_build_degree_maps`,
`_empty_schema_df`.

`build_trip_features(trip_id, trip_dir, radius=50.0)` is the core function. Its steps are:

1. `load_trip(trip_dir)` — loads **all four** files including `.route`. ← **blocks inference**
2. `build_spatial_index` + `get_candidates` — route-free. ← safe
3. `align_observations(track, nodes, arcs, route, proj)` — requires `.route`. ← **blocks inference**
4. `_build_degree_maps` — route-free. ← safe
5. GPS kinematics loop (dt_prev, dt_next, heading_reliable, gps_speed, gps_heading) — route-free. ← safe
6. Row-building loop — uses `gt_arc` (from step 3) to set `label` and `is_gt_in_candidates`.
   ← **depends on route**

The 14 model-input feature columns are computed entirely in steps 2 and 4–5. Steps 1 and 3 and the
`label`/`is_gt_in_candidates` columns in step 6 are the only route-dependent parts.

**The 14 inference columns** (from the 19-column schema, minus `label`, `is_gt_in_candidates`,
`trip_id`, `obs_index`, `arc_id`) are:

| Column | Computed in |
|---|---|
| `snap_distance_m` | `get_candidates` → `point_to_segment` |
| `snap_t` | `get_candidates` → `point_to_segment` |
| `arc_length_m` | `arc_length()` in row loop |
| `bearing_diff_rad` | `bearing()` + `wrap_angle()` in row loop |
| `abs_bearing_diff_rad` | derived from above |
| `gps_speed_mps` | GPS kinematics loop |
| `gps_heading_rad` | GPS kinematics loop |
| `dt_prev_s` | GPS kinematics loop |
| `dt_next_s` | GPS kinematics loop |
| `out_degree` | `_build_degree_maps` |
| `in_degree` | `_build_degree_maps` |
| `is_gap_prev` | GPS kinematics loop |
| `is_gap_next` | GPS kinematics loop |
| `heading_reliable` | GPS kinematics loop |

All 14 are computed without touching `route` or `obs_to_arc`.

**Does build_features.py have a function that skips the DP step?** No.

**Can any existing function accept (track, nodes, arcs) without route and return a feature
DataFrame?** No.

---

### `src/data/miss_rate.py`

**Symbols:** `compute_miss_rates`.

Operates only on the summary dict returned by `build_partition_features`. It consumes
`is_gt_in_candidates`, a training-only flag. This module is irrelevant at inference time and need
not change.

---

### `src/features/make_maps.py`

**Symbols:** `make_validation_map`, `make_network_map`, `generate_all_maps`.

Both map functions call `load_trip` and `align_observations` to draw the GT route and miss
locations. These are diagnostic/training-time tools and do not need to run during inference.

---

### `tests/test_phase1.py`

No inference-path tests exist. The test suite covers `load_track`, `load_nodes`, `load_arcs`,
`load_route` (all four loaders), geometry utilities, `get_candidates`, `align_observations`
(monotonicity check), and parquet-level schema/label checks. There are no tests for a
route-free feature extraction path because that path does not exist.

---

## Answers to Investigation Questions

**1. Does `build_features.py` have a function that skips label alignment and just produces
features for prediction?**

No. `build_trip_features` is the only trip-level entry point and it calls `align_observations`
unconditionally before the row-building loop.

**2. Does any existing function accept `(track, nodes, arcs)` without route and return a feature
DataFrame?**

No. The only DataFrame-returning functions are `build_trip_features` and `build_partition_features`,
both requiring a `.route` file via `load_trip`.

**3. Are the feature extraction steps cleanly separated from the label assignment steps?**

Mostly, but not fully. The geometry and kinematics logic is factored into separate utility
functions (`candidates.py`, `geometry.py`) and is entirely route-free. However, the label
assignment (the DP call and the `gt_arc` lookup) is embedded inside `build_trip_features` itself
rather than a separate function, so the row-building loop mixes label columns with feature columns.
Separating them requires splitting the row loop into a feature-extraction pass and an optional
label-attachment pass.

**4. Is there a single entry point that could be called from FastAPI to preprocess one trip for
inference?**

No. The closest candidate, `build_trip_features`, cannot be called from FastAPI because it requires
a filesystem directory with a `.route` file. No `predict_features(track, nodes, arcs)` function
exists.

**5. What would need to be added or refactored?**

See recommendations below.

---

## Conclusions

The pipeline is **training-only**. The underlying building blocks — `build_spatial_index`,
`get_candidates`, `_build_degree_maps`, and all of `geometry.py` — are reusable at inference time.
The barrier is that `build_trip_features` unconditionally calls `load_trip` (which raises without
a `.route` file) and `align_observations` (which requires a populated route list). The 14
model-input feature columns are computed by code that never touches the route; they just happen to
be in the same function that also computes labels.

---

## Recommendations

### R1 — Add `load_trip_inference` to `src/data/loader.py`

A new loader that returns only `track`, `nodes`, `arcs` (no route). One option is an optional
`require_route` flag on `load_trip`; a separate function is cleaner and avoids changing the
training-path signature.

```python
def load_trip_inference(trip_dir):
    return {
        "track": load_track(trip_dir),
        "nodes": load_nodes(trip_dir),
        "arcs":  load_arcs(trip_dir),
    }
```

### R2 — Extract `build_trip_features_inference` in `src/features/build_features.py`

A new function with the signature:

```python
def build_trip_features_inference(
    trip_id: str,
    track: list,
    nodes: list,
    arcs: list,
    radius: float = 50.0,
) -> pd.DataFrame:
```

This function runs steps 2, 4, and 5 from `build_trip_features` unchanged, skips step 3
(`align_observations`), and in the row-building loop omits `label` and `is_gt_in_candidates`.
The returned DataFrame has the 14 model-input columns plus `trip_id`, `obs_index`, `arc_id`
(metadata for the HMM decoder), but no label columns.

The existing `build_trip_features` can be refactored to call `build_trip_features_inference`
internally and then attach labels — but this is optional; a standalone inference function is
sufficient and lower risk.

### R3 — Preserve ordering guarantees

`get_candidates` already sorts by `snap_distance_m` ascending. `build_trip_features` iterates
`range(N)`, so `obs_index` is implicit loop order. The inference function must preserve both: sort
candidates by `snap_distance_m` within each obs, and iterate observations in timestamp order
(which `load_track` enforces). No new ordering logic is needed; just document these contracts
in the function docstring.

### R4 — Add an inference entry point in `src/features/build_features.py` or a new
`src/inference/preprocess.py`

A thin wrapper that accepts a directory path (or three in-memory data structures from FastAPI)
and returns the feature DataFrame suitable for `classifier.predict_proba()`. This is what
Chandra Raj's HMM module and Baby's FastAPI route will call.

### R5 — Add tests for the inference path in `tests/test_phase1.py`

At minimum: (a) call `build_trip_features_inference` on trip `00000001` without a `.route` file
present and confirm no exception; (b) confirm the returned DataFrame has exactly the 14 model-
input columns (and the 3 metadata columns); (c) confirm `obs_index` is non-decreasing and
candidates within each `obs_index` are sorted by `snap_distance_m`.

---

## Files Involved

| File | Status |
|---|---|
| `src/data/loader.py` | Needs `load_trip_inference` added |
| `src/features/build_features.py` | Needs `build_trip_features_inference` added |
| `src/data/candidates.py` | Ready for inference — no changes needed |
| `src/utils/geometry.py` | Ready for inference — no changes needed |
| `src/data/alignment.py` | Training-only — no changes needed |
| `src/data/miss_rate.py` | Training-only — no changes needed |
| `src/features/make_maps.py` | Training/diagnostic — no changes needed |
| `tests/test_phase1.py` | Needs inference-path tests added |
