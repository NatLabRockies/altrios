"""One-off baseline capture for the legacy `intermodal_rail` mode.

Mirrors `python/altrios/demos/lifts_demo.py` but writes `container_data` and
`vehicle_log` to CSV plus a summary stats JSON. The output directory is
selectable via the first positional argument (default `target/lifts_baseline`)
so the same script can capture the pre-refactor baseline and a post-refactor
snapshot for byte-equivalence checking.
Delete this file after the Phase 1 rebuild is verified.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import polars as pl

from altrios import sim_manager, defaults
import altrios as alt
from altrios.train_planner import planner_config
from altrios.lifts import run_terminal_simulation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "target" / "lifts_baseline"


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    rail_vehicles = [
        alt.RailVehicle.from_file(vf, skip_init=False)
        for vf in Path(alt.resources_root() / "rolling_stock/").glob("*.yaml")
    ]
    location_map = alt.import_locations(
        alt.resources_root() / "networks/default_locations.csv"
    )
    network = alt.Network.from_file(
        alt.resources_root() / "networks/Taconite-NoBalloon.yaml"
    )
    demand_file = (
        pl.read_csv(defaults.DEMAND_FILE)
        .filter(pl.col("Train_Type").str.contains("Intermodal"))
    )
    train_planner_cfg = planner_config.TrainPlannerConfig(
        cars_per_locomotive={"Default": 35},
        target_cars_per_train={"Default": 90},
        loco_type_shares={"BEL": 0.5, "Diesel_Large": 0.5},
        require_diesel=True,
    )

    print(f"[baseline] imports: {time.perf_counter() - t0:.2f}s")
    t1 = time.perf_counter()

    train_consist_plan, *_ = sim_manager.main(
        network=network,
        rail_vehicles=rail_vehicles,
        location_map=location_map,
        train_planner_config=train_planner_cfg,
        debug=True,
        demand_file=demand_file,
    )

    print(f"[baseline] sim_manager.main: {time.perf_counter() - t1:.2f}s")
    t2 = time.perf_counter()

    container_data, vehicle_log, _ = run_terminal_simulation(
        mode="intermodal_rail",
        train_consist_plan=train_consist_plan,
        terminal="Allouez",
    )

    print(f"[baseline] LIFTS sim: {time.perf_counter() - t2:.2f}s")

    container_csv = out_dir / "container_data.csv"
    vehicle_csv = out_dir / "vehicle_log.csv"
    container_data.write_csv(container_csv)
    vehicle_log.write_csv(vehicle_csv)

    summary = {
        "container_rows": container_data.height,
        "container_columns": container_data.columns,
        "vehicle_rows": vehicle_log.height,
        "resource_types": (
            vehicle_log.get_column("resource_type").unique().sort().to_list()
        ),
        "event_types": (
            vehicle_log.get_column("event_type").unique().sort().to_list()
        ),
        "total_energy_gal_or_kWh": (
            float(vehicle_log.get_column("energy_consumption(gal_or_kWh)").sum())
        ),
        "total_emissions_kgCO2": (
            float(vehicle_log.get_column("emissions(kgCO2)").sum())
        ),
        "ic_count": int(
            container_data.filter(pl.col("container_id").str.starts_with("IC")).height
        ),
        "oc_count": int(
            container_data.filter(pl.col("container_id").str.starts_with("OC")).height
        ),
        "sim_end_time": (
            float(vehicle_log.get_column("record_timestamp").max())
            if vehicle_log.height
            else None
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[baseline] wrote outputs to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
