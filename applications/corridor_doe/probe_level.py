"""Simulate one corridor at one traffic level; report stability and span time.

Run in its own process by `stable_level_search.py` so a level that exhausts
memory or hangs costs only itself, and so the timeout that marks a level
unusable can actually kill it.

Writes ``result.json`` with:
  median_trip_hr        median simulated trip, whole corridor
  first_day_hr/last_day_hr, growth_ratio   how much trip time grew across the
                        measured week -- the steady-state test
  diverging             True if trip time is still climbing at the end
  span_travel_hr        simulated median over the observed sensor span
  observed_span_hr      the same span, from RailState
  stopped_share         fraction of trip time at zero speed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import run_sweep  # noqa: E402
from speed_calibration import CORRIDORS  # noqa: E402

# Trip time is called DIVERGING when the last measured day's median exceeds the
# first day's by more than this. A settled corridor wanders a few percent either
# way (Toronto: 7.7 -> 8.6 h, ratio 1.12); one over capacity climbs without
# bound (Spokane at 88: 27.8 -> 87.5 h, ratio 3.15). The threshold sits well
# clear of observed settled behaviour.
DIVERGENCE_RATIO = 1.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corridor", required=True)
    ap.add_argument("--trains", type=int, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    corridor = CORRIDORS[args.corridor]
    plan = pl.read_csv(args.plan)
    keep = plan["Train_ID"].unique().sort()[: args.trains]
    plan = plan.filter(pl.col("Train_ID").is_in(keep.implode()))

    row = (
        pl.read_csv(HERE / "experiments.csv")
        .filter((pl.col("Corridor") == args.corridor)
                & (pl.col("Powertrain") == "diesel"))
        .to_dicts()[0]
    )
    metrics, _events, consist_plan, _sidecar = run_sweep.run_simulation(row, plan)

    p = consist_plan.unique(subset=["Train_ID"]).with_columns(
        (pl.col("Arrival_Time_Actual_Hr") - pl.col("Departure_Time_Actual_Hr"))
        .alias("trip_hr"),
        (pl.col("Departure_Time_Planned_Hr") / 24).floor().alias("day"),
    )
    by_day = (p.group_by("day").agg(pl.col("trip_hr").median().alias("med"))
               .sort("day"))
    first = float(by_day["med"][0])
    last = float(by_day["med"][-1])
    ratio = last / first if first > 0 else float("nan")

    def metric(name: str):
        m = metrics.filter(pl.col("Metric") == name)
        if "Subset" in m.columns:
            m = m.filter(pl.col("Subset") == "All")
        return float(m["Value"][0]) if m.height else None

    dwell_s = metric("Dwell_Time") or 0.0
    trip_s = float(p["trip_hr"].sum()) * 3600.0

    result = {
        "corridor": args.corridor,
        "trains": int(p.height),
        "median_trip_hr": round(float(p["trip_hr"].median()), 2),
        "first_day_hr": round(first, 2),
        "last_day_hr": round(last, 2),
        "growth_ratio": round(ratio, 2),
        "diverging": bool(ratio > DIVERGENCE_RATIO),
        "span_travel_hr": (round(metric("Span_Travel_Time"), 2)
                           if metric("Span_Travel_Time") else None),
        "stopped_share": round(dwell_s / trip_s, 3) if trip_s else None,
        "by_day": [round(float(v), 1) for v in by_day["med"].to_list()],
    }
    meta_path = HERE.parent / "corridor_inputs" / f"corridor_meta_{args.corridor}.csv"
    if meta_path.exists():
        meta = pl.read_csv(meta_path)
        if "observed_span_median_hr" in meta.columns:
            result["observed_span_hr"] = round(
                float(meta["observed_span_median_hr"][0]), 2)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "result.json").write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
