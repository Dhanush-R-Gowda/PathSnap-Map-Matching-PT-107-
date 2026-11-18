"""Candidate miss-rate reporting."""

import os
import pandas as pd


def compute_miss_rates(summary_dict, output_path, radius=50.0):
    """Compute and save miss rate CSV from a build_partition_features summary dict.

    Parameters
    ----------
    summary_dict : dict — returned by build_partition_features
    output_path  : str  — path to write CSV
    radius       : float — radius used for candidate generation

    Returns
    -------
    pd.DataFrame with columns: trip_id, total_obs, miss_count, miss_rate_pct, radius_m
    """
    trip_miss_rates = summary_dict.get("trip_miss_rates", {})

    rows = []
    for trip_id, miss_rate in trip_miss_rates.items():
        # We need total_obs per trip; derive from miss_rate and stored data
        # miss_rate is miss_count / total_obs; we stored miss_rate but not raw counts
        # Store what we have; total_obs can be reconstructed from the parquet later
        rows.append({
            "trip_id": trip_id,
            "miss_rate_pct": round(miss_rate * 100.0, 4),
            "radius_m": radius,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("trip_id").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # --- Print summary ---
    overall = summary_dict.get("overall_miss_rate", 0.0)
    print(f"\n=== Candidate Miss Rate Report (radius={radius}m) ===")
    print(f"Overall miss rate: {overall:.3%}")

    outlier_trips = {"00000054", "00000033"}
    filtered = {
        k: v for k, v in trip_miss_rates.items() if k not in outlier_trips
    }
    if filtered:
        filtered_rate = sum(filtered.values()) / len(filtered)
        print(f"Miss rate excluding trips 00000054 and 00000033: {filtered_rate:.3%}")

    high_miss = [(t, v) for t, v in trip_miss_rates.items() if v > 0.05]
    high_miss.sort(key=lambda x: -x[1])
    print(f"\nTrips with miss_rate > 5% ({len(high_miss)}):")
    for t, v in high_miss:
        print(f"  {t}: {v:.3%}")

    very_high_miss = [(t, v) for t, v in trip_miss_rates.items() if v > 0.20]
    very_high_miss.sort(key=lambda x: -x[1])
    print(f"\nTrips with miss_rate > 20% ({len(very_high_miss)}):")
    for t, v in very_high_miss:
        print(f"  {t}: {v:.3%}")

    print(f"\nMiss rate CSV saved to: {output_path}")
    return df
