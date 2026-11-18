"""Generate Folium validation and EDA network maps for Phase 1."""

import os
import sys
import numpy as np
import pandas as pd
import folium
from folium import plugins

from src.data.loader import load_trip
from src.utils.geometry import build_projection, project
from src.data.candidates import build_spatial_index, get_candidates
from src.data.alignment import align_observations


TRIPS_TO_MAP = ["00000001", "00000002", "00000033"]


def _get_node_coords(arc_id, nodes, arcs):
    """Return (lon_a, lat_a, lon_b, lat_b) for an arc."""
    from_id, to_id = arcs[arc_id]
    lon_a, lat_a = nodes[from_id]
    lon_b, lat_b = nodes[to_id]
    return lon_a, lat_a, lon_b, lat_b


def make_validation_map(trip_id, trip_dir, output_path, radius=50.0):
    """Generate a Folium validation map for one trip.

    Shows:
    - Blue circle markers: GPS track
    - Green polyline: reference route midpoints
    - Grey line segments: candidate arcs for every 25th GPS obs
    - Red line segments: GT candidate arc for the same sampled obs
    - Orange markers: miss locations
    - Layer controls and legend
    """
    data = load_trip(trip_dir)
    track = data["track"]
    nodes = data["nodes"]
    arcs = data["arcs"]
    route = data["route"]

    proj = build_projection(track)

    tree, midpoints_array, arc_ids_array, max_arc_half_length = build_spatial_index(
        nodes, arcs, proj
    )

    obs_to_route_pos, obs_to_arc = align_observations(
        track, nodes, arcs, route, proj
    )

    # Map centre
    lats = [p[1] for p in track]
    lons = [p[0] for p in track]
    centre = [np.mean(lats), np.mean(lons)]

    m = folium.Map(location=centre, zoom_start=14, tiles="CartoDB positron")

    # --- Layer: GPS track ---
    gps_layer = folium.FeatureGroup(name="GPS Track", show=True)
    for lon, lat, ts in track:
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="blue",
            fill=True,
            fill_opacity=0.6,
            popup=f"ts={ts:.1f}",
        ).add_to(gps_layer)
    gps_layer.add_to(m)

    # --- Layer: GT route (midpoints of route arcs) ---
    route_layer = folium.FeatureGroup(name="GT Route", show=True)
    route_coords = []
    for arc_id in route:
        lon_a, lat_a, lon_b, lat_b = _get_node_coords(arc_id, nodes, arcs)
        mid_lon = (lon_a + lon_b) / 2.0
        mid_lat = (lat_a + lat_b) / 2.0
        route_coords.append([mid_lat, mid_lon])
    if route_coords:
        folium.PolyLine(
            locations=route_coords,
            color="green",
            weight=3,
            opacity=0.8,
        ).add_to(route_layer)
    route_layer.add_to(m)

    # --- Layers: Candidate arcs and GT arcs for sampled observations ---
    cand_layer = folium.FeatureGroup(name="Candidate Arcs (every 25th obs)", show=True)
    gt_cand_layer = folium.FeatureGroup(name="GT Candidate Arcs", show=True)
    miss_layer = folium.FeatureGroup(name="Miss Locations", show=True)

    for i, (lon, lat, _ts) in enumerate(track):
        cands = get_candidates(
            lon, lat, proj, tree, nodes, arcs, arc_ids_array,
            max_arc_half_length, radius=radius
        )
        gt_arc = int(obs_to_arc[i])
        cand_arc_ids = {c[0] for c in cands}
        is_miss = gt_arc not in cand_arc_ids

        if is_miss:
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color="orange",
                fill=True,
                fill_opacity=0.9,
                popup=f"MISS obs={i}",
            ).add_to(miss_layer)

        if i % 25 != 0:
            continue

        # Draw candidate arcs
        for arc_id, snap_dist, snap_t in cands:
            lon_a, lat_a, lon_b, lat_b = _get_node_coords(arc_id, nodes, arcs)
            folium.PolyLine(
                locations=[[lat_a, lon_a], [lat_b, lon_b]],
                color="grey",
                weight=2,
                opacity=0.5,
                tooltip=f"arc={arc_id} dist={snap_dist:.1f}m",
            ).add_to(cand_layer)

        # Draw GT arc in red
        if gt_arc in cand_arc_ids:
            lon_a, lat_a, lon_b, lat_b = _get_node_coords(gt_arc, nodes, arcs)
            folium.PolyLine(
                locations=[[lat_a, lon_a], [lat_b, lon_b]],
                color="red",
                weight=3,
                opacity=0.9,
                tooltip=f"GT arc={gt_arc}",
            ).add_to(gt_cand_layer)

    cand_layer.add_to(m)
    gt_cand_layer.add_to(m)
    miss_layer.add_to(m)

    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 10px; border: 1px solid grey;
                border-radius: 5px; font-size: 13px;">
        <b>Legend</b><br>
        <span style="color:blue;">&#9679;</span> GPS Track<br>
        <span style="color:green;">&#8212;</span> GT Route<br>
        <span style="color:grey;">&#8212;</span> Candidate Arcs<br>
        <span style="color:red;">&#8212;</span> GT Candidate Arc<br>
        <span style="color:orange;">&#9679;</span> Miss Location<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    print(f"Saved validation map: {output_path}")


def make_network_map(trip_id, trip_dir, output_path):
    """Generate an EDA network map: all arcs as grey lines + GPS + route in green."""
    data = load_trip(trip_dir)
    track = data["track"]
    nodes = data["nodes"]
    arcs = data["arcs"]
    route = data["route"]

    lats = [p[1] for p in track]
    lons = [p[0] for p in track]
    centre = [np.mean(lats), np.mean(lons)]

    m = folium.Map(location=centre, zoom_start=14, tiles="CartoDB positron")

    # Network arcs (grey)
    net_layer = folium.FeatureGroup(name="Road Network", show=True)
    for arc_id, (from_id, to_id) in enumerate(arcs):
        lon_a, lat_a = nodes[from_id]
        lon_b, lat_b = nodes[to_id]
        folium.PolyLine(
            locations=[[lat_a, lon_a], [lat_b, lon_b]],
            color="grey",
            weight=1,
            opacity=0.3,
        ).add_to(net_layer)
    net_layer.add_to(m)

    # GPS track
    gps_layer = folium.FeatureGroup(name="GPS Track", show=True)
    for lon, lat, ts in track:
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color="blue",
            fill=True,
            fill_opacity=0.5,
        ).add_to(gps_layer)
    gps_layer.add_to(m)

    # GT Route
    route_layer = folium.FeatureGroup(name="GT Route", show=True)
    route_coords = []
    for arc_id in route:
        lon_a, lat_a, lon_b, lat_b = _get_node_coords(arc_id, nodes, arcs)
        route_coords.append([(lat_a + lat_b) / 2.0, (lon_a + lon_b) / 2.0])
    if route_coords:
        folium.PolyLine(
            locations=route_coords,
            color="green",
            weight=4,
            opacity=0.9,
        ).add_to(route_layer)
    route_layer.add_to(m)

    folium.LayerControl().add_to(m)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    print(f"Saved network map: {output_path}")


def generate_all_maps(raw_root, maps_dir, radius=50.0):
    """Generate all 6 Folium maps for the 3 target trips."""
    train_dir = os.path.join(raw_root, "train")

    for trip_id in TRIPS_TO_MAP:
        trip_dir = os.path.join(train_dir, trip_id)
        if not os.path.isdir(trip_dir):
            print(f"Trip directory not found: {trip_dir} — skipping")
            continue

        # Validation map
        val_path = os.path.join(maps_dir, f"validation_{trip_id}.html")
        make_validation_map(trip_id, trip_dir, val_path, radius=radius)

        # Network map
        net_path = os.path.join(maps_dir, f"network_{trip_id}.html")
        make_network_map(trip_id, trip_dir, net_path)


if __name__ == "__main__":
    import argparse

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    )))
    raw_root = os.path.join(repo_root, "data", "raw", "Student folder")
    maps_dir = os.path.join(repo_root, "visualizations", "maps")

    parser = argparse.ArgumentParser(description="Generate Phase 1 Folium maps")
    parser.add_argument("--radius", type=float, default=50.0)
    args = parser.parse_args()

    generate_all_maps(raw_root, maps_dir, radius=args.radius)
