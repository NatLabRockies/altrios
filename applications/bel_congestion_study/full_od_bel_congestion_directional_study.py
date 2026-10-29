"""Rerun each completed case and report one row per train direction.

The combined results supply the case list; this script recalculates the train
histories so energy, travel time, and representative traces can be split by O-D.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import altrios as alt
from altrios import sim_manager
import full_od_bel_congestion_study as base


SOURCE_RESULTS = base.RESULTS_CSV
OUTPUT_CSV = base.RESULTS_DIR / "Full_OD_BEL_Congestion_Study_Directional_Results.csv"
TRACE_ROOT = base.RESULTS_DIR / "full_od_bel_directional_traces"
# Use a slightly above-median trip, rather than the longest outlier, for each
# direction's speed trace. The selection is based on trip duration.
REPRESENTATIVE_PERCENTILE = 0.60

RESULT_COLUMNS = [
    "Directional_Scenario_ID", "Scenario_ID", "OD_Pair", "Origin", "Destination",
    "Direction_Class", "Train_Type", "Cars_Loaded", "Cars_Empty",
    "Trains_Per_Direction_Per_Day", "Directional_Study_Train_Count",
    "Combined_Study_Train_Count", "Consist_Configuration",
    "Diesel_Large_Locomotives_Per_Consist", "BEL_Locomotives_Per_Consist",
    "BEL_Capacity_MWh", "Random_Seed", "Warmup_Days", "Study_Days",
    "Directional_Average_Train_Time_Hr", "Directional_Median_Train_Time_Hr",
    "Directional_Maximum_Train_Time_Hr", "Directional_Total_Train_Time_Hr",
    "Directional_Baseline_Average_Train_Time_Hr", "Directional_Train_Time_Ratio",
    "Combined_Average_Train_Time_Hr", "Combined_Baseline_Average_Train_Time_Hr",
    "Combined_Average_Train_Time_Ratio", "Congestion_Threshold_Reached",
    "Diesel_Energy_Consumption_MWh", "Diesel_Only_Reference_Energy_MWh",
    "Diesel_Saving_MWh", "Diesel_Saving_Percent",
    "Average_Diesel_Saving_Per_Train_MWh", "Net_BEL_Energy_Consumption_MWh",
    "Total_Propulsion_Energy_MWh", "Energy_Per_Train_MWh", "Route_Distance_Miles",
    "Energy_Per_Train_Mile_MWh", "BEL_Start_SOC_Average", "BEL_End_SOC_Average",
    "BEL_Minimum_SOC", "Representative_Percentile", "Representative_Train_ID",
    "Representative_Train_Time_Hr", "Representative_Train_Hold_Time_Hr",
    "Speed_Trace_CSV", "Speed_vs_Time_Plot", "Speed_vs_Distance_Plot",
    "Time_Distance_Meet_Plot", "Run_Status", "Error_Message",
]


def ensure_output(reset: bool = False) -> None:
    if reset or not OUTPUT_CSV.exists():
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=RESULT_COLUMNS).writeheader()


def append_rows(rows: list[dict]) -> None:
    with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})


def completed_scenarios() -> set[str]:
    """Resume only cases with both directional rows saved successfully."""
    if not OUTPUT_CSV.exists():
        return set()
    with OUTPUT_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[str]] = {}
    for row in rows:
        grouped.setdefault(row["Scenario_ID"], []).append(row.get("Run_Status", ""))
    return {
        scenario for scenario, statuses in grouped.items()
        if len(statuses) == 2 and all(status == "completed" for status in statuses)
    }


def find_case(source_row: pd.Series) -> tuple[dict, dict]:
    service = next(
        case for case in base.SERVICE_CASES
        if case["od_pair"] == source_row["OD_Pair"]
        and case["train_type"] == source_row["Train_Type"]
    )
    capacity = None if pd.isna(source_row["BEL_Capacity_MWh"]) else float(source_row["BEL_Capacity_MWh"])
    propulsion = next(
        case for case in base.PROPULSION_CASES
        if case["name"] == source_row["Consist_Configuration"]
        and case["bel_capacity_mwh"] == capacity
    )
    return service, propulsion


def get_direction(sim: dict) -> tuple[str, str]:
    origin, destination = base.sim_direction(sim)
    return origin, destination


def sim_duration_hr(sim: dict) -> float:
    times = sim["history"]["time_seconds"]
    return (times[-1] - times[0]) / base.SECONDS_PER_HOUR


def select_representative(sims: list[dict]) -> dict:
    """Pick the nearest-rank 60th-percentile trip by duration."""
    ordered = sorted(sims, key=sim_duration_hr)
    index = max(0, math.ceil(REPRESENTATIVE_PERCENTILE * len(ordered)) - 1)
    return ordered[index]


def hold_time_hr(sim: dict) -> float:
    """Sum low-speed periods where the route limit still permits movement."""
    history = sim["history"]
    times = np.asarray(history["time_seconds"], dtype=float)
    speeds_mph = np.asarray(history["speed_meters_per_second"], dtype=float) / 0.44704
    limits_mph = np.asarray(history["speed_limit_meters_per_second"], dtype=float) / 0.44704
    if len(times) < 2:
        return 0.0
    dt = np.diff(times, prepend=times[0])
    held = (speeds_mph < 1.0) & (limits_mph > 5.0)
    return float(dt[held].sum() / base.SECONDS_PER_HOUR)


def export_direction_trace(
    representative: dict,
    scenario_id: str,
    origin: str,
    destination: str,
) -> tuple[pd.DataFrame, Path, Path, Path]:
    """Save the chosen trip's samples and its time/distance speed views."""
    history = representative["history"]
    times = np.asarray(history["time_seconds"], dtype=float)
    distances = np.asarray(history["offset_meters"], dtype=float)
    actual = np.asarray(history["speed_meters_per_second"], dtype=float) / 0.44704
    target = np.asarray(history["speed_target_meters_per_second"], dtype=float) / 0.44704
    limits = np.asarray(history["speed_limit_meters_per_second"], dtype=float) / 0.44704
    # A low actual speed with a usable speed limit suggests a dispatch hold.
    # It does not identify the physical siding track from infrastructure data.
    held = (actual < 1.0) & (limits > 5.0)
    dt = np.diff(times, prepend=times[0])
    trace = pd.DataFrame(
        {
            "Scenario_ID": scenario_id,
            "Train_ID": str(representative["train_id"]),
            "Origin": origin,
            "Destination": destination,
            "Absolute_Time_Hr": times / base.SECONDS_PER_HOUR,
            "Elapsed_Time_Hr": (times - times[0]) / base.SECONDS_PER_HOUR,
            "Distance_Miles": distances / 1_609.344,
            "Link_Index": history["link_idx_front"],
            "Actual_Speed_MPH": actual,
            "Target_Speed_MPH": target,
            "Speed_Limit_MPH": limits,
            "Dispatch_Hold": held,
            "Cumulative_Hold_Time_Hr": np.cumsum(np.where(held, dt, 0.0)) / base.SECONDS_PER_HOUR,
        }
    )
    output_dir = TRACE_ROOT / scenario_id
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{origin}_to_{destination}"
    trace_path = output_dir / f"{stem}_speed_trace.csv"
    time_plot = output_dir / f"{stem}_speed_vs_time.png"
    distance_plot = output_dir / f"{stem}_speed_vs_distance.png"
    trace.to_csv(trace_path, index=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(trace["Elapsed_Time_Hr"], trace["Actual_Speed_MPH"], label="Actual", linewidth=1.5)
    ax.plot(trace["Elapsed_Time_Hr"], trace["Target_Speed_MPH"], label="Target", alpha=0.8)
    ax.plot(trace["Elapsed_Time_Hr"], trace["Speed_Limit_MPH"], label="Limit", alpha=0.7)
    if held.any():
        ax.fill_between(
            trace["Elapsed_Time_Hr"].to_numpy(dtype=float),
            0.0,
            trace["Speed_Limit_MPH"].to_numpy(dtype=float),
            where=np.asarray(held, dtype=bool),
            color="tab:red",
            alpha=0.18,
            label="Dispatch hold",
        )
    ax.set(xlabel="Elapsed trip time [hr]", ylabel="Speed [mph]", title=f"{origin} to {destination}: speed versus time")
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(time_plot, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(trace["Distance_Miles"], trace["Actual_Speed_MPH"], label="Actual", linewidth=1.5)
    ax.plot(trace["Distance_Miles"], trace["Target_Speed_MPH"], label="Target", alpha=0.8)
    ax.plot(trace["Distance_Miles"], trace["Speed_Limit_MPH"], label="Limit", alpha=0.7)
    if held.any():
        ax.scatter(trace.loc[held, "Distance_Miles"], trace.loc[held, "Actual_Speed_MPH"], s=8, color="tab:red", label="Dispatch hold")
    ax.set(xlabel="Route distance [mile]", ylabel="Speed [mph]", title=f"{origin} to {destination}: speed versus distance")
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(distance_plot, dpi=180)
    plt.close(fig)
    return trace, trace_path, time_plot, distance_plot


def export_time_distance_plot(
    traces: list[pd.DataFrame], scenario_id: str, service: dict
) -> Path:
    """Plot the two representative trips on a common corridor axis.

    For all trains in the study week, see full_od_bel_meet_pass_diagrams.py.
    """
    output_dir = TRACE_ROOT / scenario_id
    plot_path = output_dir / "opposing_trains_time_distance.png"
    route_length = max(float(trace["Distance_Miles"].max()) for trace in traces)
    fig, ax = plt.subplots(figsize=(9, 5))
    for trace in traces:
        origin = trace["Origin"].iloc[0]
        distance = trace["Distance_Miles"].to_numpy()
        if origin != service["outbound_origin"]:
            distance = route_length - distance
        ax.plot(trace["Absolute_Time_Hr"], distance, label=f"{trace['Origin'].iloc[0]} to {trace['Destination'].iloc[0]}")
    ax.set(xlabel="Absolute simulation time [hr]", ylabel=f"Distance from {service['outbound_origin']} [mile]", title="Opposing representative trains: time–distance diagram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


def directional_metrics(sims: list[dict]) -> dict:
    """Aggregate study-week travel time and propulsion energy for one O-D."""
    durations = np.asarray([sim_duration_hr(sim) for sim in sims])
    fuel_j = 0.0
    distances = []
    for sim in sims:
        fuel = sim["loco_con"]["history"]["energy_fuel_joules"]
        if fuel:
            fuel_j += fuel[-1] - fuel[0]
        offset = sim["history"]["offset_meters"]
        if offset:
            distances.append(offset[-1] - offset[0])
    battery_mwh, start_soc, end_soc, min_soc = base.battery_metrics(sims)
    diesel_mwh = fuel_j / base.JOULES_PER_MWH
    route_miles = float(np.mean(distances) / 1_609.344)
    return {
        "count": len(sims), "average": float(np.mean(durations)),
        "median": float(np.median(durations)), "maximum": float(np.max(durations)),
        "total": float(np.sum(durations)), "diesel": diesel_mwh,
        "battery": battery_mwh, "total_energy": diesel_mwh + battery_mwh,
        "route_miles": route_miles, "start_soc": start_soc,
        "end_soc": end_soc, "min_soc": min_soc,
    }


def run_iteration(source_row: pd.Series) -> list[dict]:
    """Recreate one case and split its study-week trains into two O-D rows."""
    service, propulsion = find_case(source_row)
    traffic = int(source_row["Trains_Per_Direction_Per_Day"])
    scenario_id = source_row["Scenario_ID"]
    print(f"RUNNING {scenario_id}", flush=True)
    config = base.build_planner_config(propulsion["bel_capacity_mwh"])
    plan = base.build_consist_plan(service, propulsion, traffic, config)
    rail_vehicles = [
        alt.RailVehicle.from_file(path, skip_init=False)
        for path in (alt.resources_root() / "rolling_stock").glob("*.yaml")
    ]
    network = alt.Network.from_file(base.NETWORK_PATH)
    locations = alt.import_locations(base.LOCATIONS_PATH)
    (
        consist_plan_out, loco_pool, refuelers, _grid, _prices,
        train_sims, timed_paths, _untrimmed,
    ) = sim_manager.main(
        network=network, rail_vehicles=rail_vehicles, location_map=locations,
        consist_plan_in=plan, train_planner_config=config, debug=False,
    )
    train_sims.set_save_interval(base.SAVE_INTERVAL)
    sims, _sessions = alt.run_speed_limit_train_sims(
        speed_limit_train_sims=train_sims, network=network,
        train_consist_plan_py=consist_plan_out, loco_pool_py=loco_pool,
        refuel_facilities_py=refuelers,
        timed_paths=[alt.TimedLinkPath.from_pydict(path) for path in timed_paths],
    )
    # Preserve the combined study's departure-week definition so both
    # directional rows reconcile to the original case totals.
    selected = base.study_sims(sims.to_pydict())
    directions = [
        (service["outbound_origin"], service["outbound_destination"], service["outbound_load"]),
        (service["outbound_destination"], service["outbound_origin"], service["return_load"]),
    ]
    rows, traces = [], []
    baseline_source = pd.read_csv(SOURCE_RESULTS)
    baseline_combined = float(source_row["Baseline_Average_Train_Time_Hr"])
    for origin, destination, load in directions:
        directional_sims = [sim for sim in selected if get_direction(sim) == (origin, destination)]
        if not directional_sims:
            raise RuntimeError(f"No simulations found for {origin} to {destination}")
        metrics = directional_metrics(directional_sims)
        representative = select_representative(directional_sims)
        trace, trace_path, time_plot, distance_plot = export_direction_trace(
            representative, scenario_id, origin, destination
        )
        traces.append(trace)
        baseline_match = baseline_source.loc[
            (baseline_source["OD_Pair"] == service["od_pair"])
            & (baseline_source["Train_Type"] == service["train_type"])
            & (baseline_source["Consist_Configuration"] == propulsion["name"])
            & (baseline_source["BEL_Capacity_MWh"].fillna(-1) == (propulsion["bel_capacity_mwh"] if propulsion["bel_capacity_mwh"] is not None else -1))
            & (baseline_source["Trains_Per_Direction_Per_Day"] == 1)
        ]
        # Directional baseline is filled accurately during finalization from
        # the two directional one-traffic rows.
        direction_class = "Balanced Manifest"
        if service["train_type"] != "Manifest":
            direction_class = "Loaded" if load[0] > 0 else "Empty Return"
        total_energy = metrics["total_energy"]
        rows.append(
            {
                "Directional_Scenario_ID": f"{scenario_id}__{origin}_to_{destination}",
                "Scenario_ID": scenario_id, "OD_Pair": service["od_pair"],
                "Origin": origin, "Destination": destination,
                "Direction_Class": direction_class, "Train_Type": service["train_type"],
                "Cars_Loaded": load[0], "Cars_Empty": load[1],
                "Trains_Per_Direction_Per_Day": traffic,
                "Directional_Study_Train_Count": metrics["count"],
                "Combined_Study_Train_Count": int(source_row["Study_Train_Count"]),
                "Consist_Configuration": propulsion["name"],
                "Diesel_Large_Locomotives_Per_Consist": 4 if propulsion["name"] == "Diesel" else 3,
                "BEL_Locomotives_Per_Consist": 0 if propulsion["name"] == "Diesel" else 1,
                "BEL_Capacity_MWh": propulsion["bel_capacity_mwh"] or "",
                "Random_Seed": base.RANDOM_SEED, "Warmup_Days": base.WARMUP_DAYS,
                "Study_Days": base.STUDY_DAYS,
                "Directional_Average_Train_Time_Hr": metrics["average"],
                "Directional_Median_Train_Time_Hr": metrics["median"],
                "Directional_Maximum_Train_Time_Hr": metrics["maximum"],
                "Directional_Total_Train_Time_Hr": metrics["total"],
                "Combined_Average_Train_Time_Hr": float(source_row["Average_Train_Time_Hr"]),
                "Combined_Baseline_Average_Train_Time_Hr": baseline_combined,
                "Combined_Average_Train_Time_Ratio": float(source_row["Average_Train_Time_Ratio"]),
                "Congestion_Threshold_Reached": bool(source_row["Congestion_Threshold_Reached"]),
                "Diesel_Energy_Consumption_MWh": metrics["diesel"],
                "Net_BEL_Energy_Consumption_MWh": metrics["battery"],
                "Total_Propulsion_Energy_MWh": total_energy,
                "Energy_Per_Train_MWh": total_energy / metrics["count"],
                "Route_Distance_Miles": metrics["route_miles"],
                "Energy_Per_Train_Mile_MWh": total_energy / metrics["count"] / metrics["route_miles"],
                "BEL_Start_SOC_Average": metrics["start_soc"],
                "BEL_End_SOC_Average": metrics["end_soc"],
                "BEL_Minimum_SOC": metrics["min_soc"],
                "Representative_Percentile": REPRESENTATIVE_PERCENTILE,
                "Representative_Train_ID": str(representative["train_id"]),
                "Representative_Train_Time_Hr": sim_duration_hr(representative),
                "Representative_Train_Hold_Time_Hr": hold_time_hr(representative),
                "Speed_Trace_CSV": str(trace_path.resolve()),
                "Speed_vs_Time_Plot": str(time_plot.resolve()),
                "Speed_vs_Distance_Plot": str(distance_plot.resolve()),
                "Run_Status": "completed", "Error_Message": "",
            }
        )
    meet_plot = export_time_distance_plot(traces, scenario_id, service)
    for row in rows:
        row["Time_Distance_Meet_Plot"] = str(meet_plot.resolve())
    return rows


def finalize_calculations() -> None:
    """Join same-direction baselines and diesel references into the CSV."""
    results = pd.read_csv(OUTPUT_CSV)
    results = (
        results.loc[results["Run_Status"] == "completed"]
        .drop_duplicates(subset=["Directional_Scenario_ID"], keep="last")
        .copy()
    )
    keys = ["OD_Pair", "Train_Type", "Origin", "Destination", "Consist_Configuration", "BEL_Capacity_MWh"]
    # Congestion ratios use the one-train/day row for the same direction,
    # train type, propulsion configuration, and BEL capacity.
    baseline = results.loc[results["Trains_Per_Direction_Per_Day"] == 1, keys + ["Directional_Average_Train_Time_Hr"]].rename(
        columns={"Directional_Average_Train_Time_Hr": "Directional_Baseline_Average_Train_Time_Hr"}
    )
    results = results.drop(columns=["Directional_Baseline_Average_Train_Time_Hr", "Directional_Train_Time_Ratio"], errors="ignore").merge(baseline, on=keys, how="left")
    results["Directional_Train_Time_Ratio"] = results["Directional_Average_Train_Time_Hr"] / results["Directional_Baseline_Average_Train_Time_Hr"]
    # Savings compare the mixed consist with a diesel-only consist at the
    # same traffic level and O-D direction. A negative result means more
    # diesel energy than the corresponding diesel-only run.
    diesel = results.loc[
        results["Consist_Configuration"] == "Diesel",
        ["OD_Pair", "Train_Type", "Origin", "Destination", "Trains_Per_Direction_Per_Day", "Diesel_Energy_Consumption_MWh"],
    ].rename(columns={"Diesel_Energy_Consumption_MWh": "Diesel_Only_Reference_Energy_MWh"})
    results = results.drop(columns=["Diesel_Only_Reference_Energy_MWh"], errors="ignore").merge(
        diesel,
        on=["OD_Pair", "Train_Type", "Origin", "Destination", "Trains_Per_Direction_Per_Day"],
        how="left",
    )
    results["Diesel_Saving_MWh"] = results["Diesel_Only_Reference_Energy_MWh"] - results["Diesel_Energy_Consumption_MWh"]
    results["Diesel_Saving_Percent"] = results["Diesel_Saving_MWh"] / results["Diesel_Only_Reference_Energy_MWh"]
    results["Average_Diesel_Saving_Per_Train_MWh"] = results["Diesel_Saving_MWh"] / results["Directional_Study_Train_Count"]
    results = results[RESULT_COLUMNS]
    results.to_csv(OUTPUT_CSV, index=False)


def main(reset: bool = False) -> None:
    ensure_output(reset=reset)
    done = completed_scenarios()
    source = pd.read_csv(SOURCE_RESULTS)
    source = source.loc[source["Run_Status"] == "completed"]
    for _, source_row in source.iterrows():
        if source_row["Scenario_ID"] in done:
            print(f"SKIPPING {source_row['Scenario_ID']}", flush=True)
            continue
        try:
            rows = run_iteration(source_row)
            append_rows(rows)
            print(f"COMPLETED {source_row['Scenario_ID']}", flush=True)
        except Exception as exc:
            append_rows([
                {"Directional_Scenario_ID": f"{source_row['Scenario_ID']}__failed", "Scenario_ID": source_row["Scenario_ID"], "Run_Status": "failed", "Error_Message": repr(exc)},
                {"Directional_Scenario_ID": f"{source_row['Scenario_ID']}__failed_2", "Scenario_ID": source_row["Scenario_ID"], "Run_Status": "failed", "Error_Message": repr(exc)},
            ])
            raise
    finalize_calculations()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    main(reset=args.reset)
    print(f"Directional study elapsed time: {time.perf_counter() - started:.1f} s")
