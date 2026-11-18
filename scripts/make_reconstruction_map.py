"""Folium map: GPS, road network, reference route (offline only), classifier-only arcs, HMM route."""

import argparse
import os
import sys
from pathlib import Path

import folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_trip
from src.pipeline import PathSnapMatcher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trip_id")
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--raw-root", default=str(ROOT / "data/raw/Student folder"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    d = load_trip(str(Path(a.raw_root) / a.partition / a.trip_id))  # .route used for reference layer only
    matcher = PathSnapMatcher(os.environ.get("PATHSNAP_ARTIFACTS", ROOT / "artifacts"))
    out = matcher.match(a.trip_id, d["track"], d["nodes"], d["arcs"])
    g, res = out["graph"], out["result"]

    ll = lambda arc: [(lat, lon) for lon, lat in g.arc_polyline(arc)]
    gps = [(lat, lon) for lon, lat, _ in d["track"]]
    m = folium.Map(location=gps[len(gps) // 2], zoom_start=14, tiles="OpenStreetMap")

    net = folium.FeatureGroup("Road network (candidate arcs)", show=False)
    cand = {c[0] for cs in out["candidates"] for c in cs}
    for arc in cand:
        folium.PolyLine(ll(arc), color="#999999", weight=2, opacity=0.6).add_to(net)
    ref = folium.FeatureGroup("Reference route (ground truth)")
    for arc in d["route"]:
        folium.PolyLine(ll(arc), color="#d62728", weight=8, opacity=0.5).add_to(ref)
    ml = folium.FeatureGroup("Classifier-only top-1 arcs", show=False)
    for arc in {x for x in out["ml_top1"] if x is not None}:
        folium.PolyLine(ll(arc), color="#ff7f0e", weight=4).add_to(ml)
    hm = folium.FeatureGroup("HMM/Viterbi predicted route")
    for arc in res.arc_sequence:
        folium.PolyLine(ll(arc), color="#2ca02c", weight=4, opacity=0.95).add_to(hm)
    pts = folium.FeatureGroup("Original GPS observations")
    step = max(1, len(gps) // 700)
    for p in gps[::step]:
        folium.CircleMarker(p, radius=2, color="#1f77b4", fill=True, weight=1).add_to(pts)
    for grp in (net, ref, ml, hm, pts):
        grp.add_to(m)
    folium.Marker(gps[0], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(gps[-1], tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = ("<div style='position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 12px;"
              "border:1px solid #888;font:12px sans-serif'><b>Trip %s (%s)</b><br>"
              "<span style='color:#1f77b4'>●</span> GPS &nbsp;<span style='color:#d62728'>━</span> reference<br>"
              "<span style='color:#ff7f0e'>━</span> classifier-only &nbsp;<span style='color:#2ca02c'>━</span> HMM route</div>"
              % (a.trip_id, a.partition))
    m.get_root().html.add_child(folium.Element(legend))
    m.fit_bounds([(min(p[0] for p in gps), min(p[1] for p in gps)), (max(p[0] for p in gps), max(p[1] for p in gps))])
    dest = Path(a.out or ROOT / f"visualizations/maps/reconstruction_{a.partition}_{a.trip_id}.html")
    m.save(str(dest))
    print("saved", dest)


if __name__ == "__main__":
    main()
