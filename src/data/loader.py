"""Data loaders for .track, .nodes, .arcs, and .route files."""

import os
import warnings


def _trip_id_from_dir(trip_dir):
    """Extract the 8-digit trip_id from a trip directory path."""
    return os.path.basename(os.path.normpath(trip_dir))


def load_track(trip_dir):
    """Load GPS track observations.

    Parameters
    ----------
    trip_dir : str — full path to the trip subdirectory

    Returns
    -------
    list of (lon, lat, ts) tuples (floats)
    """
    trip_id = _trip_id_from_dir(trip_dir)
    path = os.path.join(trip_dir, f"{trip_id}.track")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Track file not found: {path}")

    rows = []
    prev_ts = None
    with open(path, "r") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                warnings.warn(f"{path}:{lineno}: expected 3 columns, got {len(parts)}")
                continue
            try:
                lon, lat, ts = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError as exc:
                warnings.warn(f"{path}:{lineno}: parse error — {exc}")
                continue

            if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                warnings.warn(f"{path}:{lineno}: implausible coords ({lon}, {lat})")

            if prev_ts is not None and ts < prev_ts:
                warnings.warn(
                    f"{path}:{lineno}: non-monotone timestamp {ts} < {prev_ts}"
                )
            prev_ts = ts
            rows.append((lon, lat, ts))

    if not rows:
        warnings.warn(f"{path}: file is empty")

    return rows


def load_nodes(trip_dir):
    """Load road network nodes.

    Parameters
    ----------
    trip_dir : str

    Returns
    -------
    list of (lon, lat) tuples; list index = node_id
    """
    trip_id = _trip_id_from_dir(trip_dir)
    path = os.path.join(trip_dir, f"{trip_id}.nodes")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Nodes file not found: {path}")

    rows = []
    with open(path, "r") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                warnings.warn(f"{path}:{lineno}: expected 2 columns, got {len(parts)}")
                continue
            try:
                lon, lat = float(parts[0]), float(parts[1])
            except ValueError as exc:
                warnings.warn(f"{path}:{lineno}: parse error — {exc}")
                continue

            if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                warnings.warn(f"{path}:{lineno}: implausible coords ({lon}, {lat})")

            rows.append((lon, lat))

    if not rows:
        warnings.warn(f"{path}: file is empty")

    return rows


def load_arcs(trip_dir):
    """Load road network arcs.

    Parameters
    ----------
    trip_dir : str

    Returns
    -------
    list of (from_node_id, to_node_id) tuples (ints); list index = arc_id
    """
    trip_id = _trip_id_from_dir(trip_dir)
    path = os.path.join(trip_dir, f"{trip_id}.arcs")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Arcs file not found: {path}")

    rows = []
    with open(path, "r") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                warnings.warn(f"{path}:{lineno}: expected 2 columns, got {len(parts)}")
                continue
            try:
                from_id, to_id = int(parts[0]), int(parts[1])
            except ValueError as exc:
                warnings.warn(f"{path}:{lineno}: parse error — {exc}")
                continue
            rows.append((from_id, to_id))

    if not rows:
        warnings.warn(f"{path}: file is empty")

    return rows


def load_route(trip_dir):
    """Load ground-truth route arc IDs.

    Parameters
    ----------
    trip_dir : str

    Returns
    -------
    list of int arc_ids
    """
    trip_id = _trip_id_from_dir(trip_dir)
    path = os.path.join(trip_dir, f"{trip_id}.route")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Route file not found: {path}")

    rows = []
    with open(path, "r") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                arc_id = int(line)
            except ValueError as exc:
                warnings.warn(f"{path}:{lineno}: parse error — {exc}")
                continue
            rows.append(arc_id)

    if not rows:
        warnings.warn(f"{path}: file is empty")

    return rows


def load_trip(trip_dir):
    """Load all data for a single trip.

    Parameters
    ----------
    trip_dir : str

    Returns
    -------
    dict with keys: track, nodes, arcs, route
    """
    return {
        "track": load_track(trip_dir),
        "nodes": load_nodes(trip_dir),
        "arcs": load_arcs(trip_dir),
        "route": load_route(trip_dir),
    }


def discover_trips(partition_dir):
    """Discover all trip subdirectories in a partition.

    Parameters
    ----------
    partition_dir : str — path to train/ or student_test/ directory

    Returns
    -------
    sorted list of trip_id strings (8-digit zero-padded)
    """
    if not os.path.isdir(partition_dir):
        raise NotADirectoryError(f"Partition directory not found: {partition_dir}")

    trip_ids = []
    for name in os.listdir(partition_dir):
        full_path = os.path.join(partition_dir, name)
        if os.path.isdir(full_path) and name.isdigit() and len(name) == 8:
            trip_ids.append(name)

    return sorted(trip_ids)
