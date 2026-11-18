"""Offline evaluation: ML-only vs ML+HMM on a partition. Ground truth used ONLY here."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.alignment import align_observations
from src.data.loader import discover_trips, load_trip
from src.decoder.hmm_matcher import route_continuity
from src.pipeline import PathSnapMatcher
from src.utils.geometry import build_projection

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--raw-root", default=str(ROOT / "data/raw/Student folder"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results/pipeline_eval.json"))
    args = ap.parse_args()

    import os
    matcher = PathSnapMatcher(os.environ.get("PATHSNAP_ARTIFACTS", ROOT / "artifacts"))
    part = Path(args.raw_root) / args.partition
    trips = discover_trips(str(part))
    if args.limit:
        trips = trips[: args.limit]
    rows, tot = [], dict(n=0, ml=0, hmm=0, miss=0)
    for tid in trips:
        d = load_trip(str(part / tid))
        proj = build_projection(d["track"])
        _, gt = align_observations(d["track"], d["nodes"], d["arcs"], d["route"], proj)
        out = matcher.match(tid, d["track"], d["nodes"], d["arcs"])
        hmm = out["result"].obs_arc
        n = len(gt)
        ml_ok = sum(int(a == g) for a, g in zip(out["ml_top1"], gt))
        hmm_ok = sum(int(a == g) for a, g in zip(hmm, gt))
        miss = sum(int(g not in {c[0] for c in cs}) for g, cs in zip(gt, out["candidates"]))
        ml_cont = route_continuity(out["graph"], [a for a in out["ml_top1"] if a is not None])
        rows.append(dict(trip_id=tid, n_obs=n, ml_top1=ml_ok / n, hmm_top1=hmm_ok / n,
                         miss_rate=miss / n, ml_continuity=ml_cont,
                         hmm_continuity=out["continuity"], breaks=len(out["result"].breaks),
                         runtime_s=out["timing_s"]["total"]))
        print(rows[-1], flush=True)
        for k, v in dict(n=n, ml=ml_ok, hmm=hmm_ok, miss=miss).items():
            tot[k] += v
    summary = dict(partition=args.partition, trips=len(trips), observations=tot["n"],
                   ml_top1=tot["ml"] / tot["n"], hmm_top1=tot["hmm"] / tot["n"],
                   candidate_miss_rate=tot["miss"] / tot["n"],
                   mean_ml_continuity=float(np.mean([r["ml_continuity"] for r in rows])),
                   mean_hmm_continuity=float(np.mean([r["hmm_continuity"] for r in rows])),
                   mean_runtime_s=float(np.mean([r["runtime_s"] for r in rows])))
    print(json.dumps(summary, indent=2))
    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    Path(args.out).write_text(json.dumps(dict(summary=summary, per_trip=rows), indent=2))


if __name__ == "__main__":
    main()
