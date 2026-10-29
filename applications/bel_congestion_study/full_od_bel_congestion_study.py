"""Run the O-D/BEL traffic sweep and save one combined-direction row per case.

This is the source study for the directional report and meet-pass stringlines.
Each case contains only one train type and one locomotive configuration.
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
import polars as pl

import altrios as alt
from altrios import defaults, sim_manager
from altrios.train_planner.planner_config import TrainPlannerConfig


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_CSV = RESULTS_DIR / "Full_OD_BEL_Congestion_Study_Results.csv"
TRACE_ROOT = RESULTS_DIR / "full_od_bel_congestion_traces"
NETWORK_PATH = alt.resources_root() / "networks/Taconite-NoBalloon.yaml"
LOCATIONS_PATH = alt.resources_root() / "networks/default_locations.csv"

# Dispatch trains for two weeks, but report departures from the second week.
# The first week lets traffic already on the corridor affect the study week.
WARMUP_DAYS = 7
STUDY_DAYS = 7
MAX_TRAINS_PER_DIRECTION_PER_DAY = 20
RANDOM_SEED = 20260921
LOCOMOTIVES_PER_CONSIST = 4
CARS_PER_TRAIN = 100
SAVE_INTERVAL = 100
SECONDS_PER_HOUR = 3_600
JOULES_PER_MWH = 3.6e9

RESULT_COLUMNS = [
    "Scenario_ID", "OD_Pair", "Origin_Loaded_Direction",
    "Destination_Loaded_Direction", "Train_Type", "Cars_Loaded_Outbound",
    "Cars_Empty_Outbound", "Cars_Loaded_Return", "Cars_Empty_Return",
    "Trains_Per_Direction_Per_Day", "Study_Train_Count",
    "Consist_Configuration", "Diesel_Large_Locomotives_Per_Consist",
    "BEL_Locomotives_Per_Consist", "BEL_Capacity_MWh", "Random_Seed",
    "Warmup_Days", "Study_Days", "Average_Train_Time_Hr",
    "Median_Train_Time_Hr", "Maximum_Train_Time_Hr", "Total_Train_Time_Hr",
    "Baseline_Average_Train_Time_Hr", "Average_Train_Time_Ratio",
    "Congestion_Threshold_Reached", "Diesel_Energy_Consumption_MWh",
    "Net_BEL_Energy_Consumption_MWh", "Total_Propulsion_Energy_MWh",
    "Energy_Per_Train_MWh", "Route_Distance_Miles",
    "Energy_Per_Train_Mile_MWh", "BEL_Start_SOC_Average",
    "BEL_End_SOC_Average", "BEL_Minimum_SOC", "Detected_Meet_Count",
    "Total_Meet_Delay_Hr", "Average_Meet_Delay_Min", "Meet_Summary_CSV",
    "Meet_Speed_Trace_CSV", "Meet_Speed_Plot_Directory", "Run_Status",
    "Error_Message",
]

# The tuple in each load field is (loaded cars, empty cars). Manifest trains
# are half loaded in both directions; ore and grain return empty.
SERVICE_CASES = [
    {
        "od_pair": "Hibbing-Allouez",
        "outbound_origin": "Hibbing",
        "outbound_destination": "Allouez",
        "train_type": "Manifest",
        "outbound_load": (50, 50),
        "return_load": (50, 50),
    },
    {
        "od_pair": "Hibbing-Allouez",
        "outbound_origin": "Hibbing",
        "outbound_destination": "Allouez",
        "train_type": "Ironore",
        "outbound_load": (100, 0),
        "return_load": (0, 100),
    },
    {
        "od_pair": "Minneapolis-Superior",
        "outbound_origin": "Minneapolis",
        "outbound_destination": "Superior",
        "train_type": "Manifest",
        "outbound_load": (50, 50),
        "return_load": (50, 50),
    },
    {
        "od_pair": "Minneapolis-Superior",
        "outbound_origin": "Minneapolis",
        "outbound_destination": "Superior",
        "train_type": "Grain",
        "outbound_load": (100, 0),
        "return_load": (0, 100),
    },
]

# Mixed replaces one of the four Diesel_Large locomotives with one BEL.
# The capacity sweep changes BEL storage, not the train's car count.
PROPULSION_CASES = [
    {"name": "Diesel", "bel_capacity_mwh": None},
    {"name": "Mixed", "bel_capacity_mwh": 2.4},
    {"name": "Mixed", "bel_capacity_mwh": 4.8},
    {"name": "Mixed", "bel_capacity_mwh": 9.6},
    {"name": "Mixed", "bel_capacity_mwh": 14.5},
]


def ensure_results_file() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0:
        with RESULTS_CSV.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=RESULT_COLUMNS).writeheader()


def existing_scenarios() -> dict[str, str]:
    ensure_results_file()
    with RESULTS_CSV.open(newline="", encoding="utf-8") as handle:
        return {
            row["Scenario_ID"]: row.get("Run_Status", "")
            for row in csv.DictReader(handle)
        }


def append_result(row: dict) -> None:
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})


def build_planner_config(bel_capacity_mwh: float | None) -> TrainPlannerConfig:
    """Set the BEL energy capacity for this case (input MWh, model joules)."""
    config = TrainPlannerConfig(simulation_days=WARMUP_DAYS + STUDY_DAYS)
    if bel_capacity_mwh is None:
        return config

    idx = config.loco_info.index[
        config.loco_info["Locomotive_Type"] == "BEL"
    ][0]
    bel = config.loco_info.at[idx, "Rust_Loco"]
    bel_dict = bel.to_pydict()
    bel_type = next(iter(bel_dict["loco_type"].values()))
    bel_type["res"]["energy_capacity_joules"] = bel_capacity_mwh * JOULES_PER_MWH
    config.loco_info.at[idx, "Rust_Loco"] = alt.Locomotive.from_pydict(bel_dict)
    return config


def build_consist_plan(
    service: dict,
    propulsion: dict,
    trains_per_direction_per_day: int,
    config: TrainPlannerConfig,
) -> pl.DataFrame:
    """Create evenly spaced opposing departures with reproducible small jitter."""
    # Reuse the seed across cases so traffic/propulsion comparisons share the
    # same departure pattern at a given traffic level.
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    train_id = 0
    interval_hr = 24 / trains_per_direction_per_day
    directions = [
        (
            service["outbound_origin"],
            service["outbound_destination"],
            service["outbound_load"],
            0.0,
        ),
        (
            service["outbound_destination"],
            service["outbound_origin"],
            service["return_load"],
            interval_hr / 2,
        ),
    ]

    for day in range(WARMUP_DAYS + STUDY_DAYS):
        for origin, destination, (cars_loaded, cars_empty), offset_hr in directions:
            for train_number in range(trains_per_direction_per_day):
                train_id += 1
                jitter_hr = rng.uniform(-0.25, 0.25)
                departure_hr = day * 24 + train_number * interval_hr + offset_hr + jitter_hr
                departure_hr = max(day * 24, min((day + 1) * 24 - 0.01, departure_hr))
                arrival_hr = departure_hr + 10
                for loco_number in range(LOCOMOTIVES_PER_CONSIST):
                    is_bel = propulsion["name"] == "Mixed" and loco_number == 3
                    rows.append(
                        {
                            "Train_ID": train_id,
                            "Train_Type": service["train_type"],
                            "Locomotive_ID": train_id * 10 + loco_number,
                            "Locomotive_Type": "BEL" if is_bel else "Diesel_Large",
                            "Origin_ID": origin,
                            "Destination_ID": destination,
                            "Cars_Loaded": float(cars_loaded),
                            "Cars_Empty": float(cars_empty),
                            "Containers_Loaded": 0.0,
                            "Containers_Empty": 0.0,
                            # Bookkeeping only: the manual planner initializes actual
                            # diesel/BEL energy from ``loco_info`` and ignores this value.
                            "Departure_SOC_J": defaults.DIESEL_TANK_CAPACITY_J,
                            "Departure_Time_Planned_Hr": departure_hr,
                            "Arrival_Time_Planned_Hr": arrival_hr,
                            "Refuel_Start_Time_Planned_Hr": arrival_hr,
                            "Refuel_End_Time_Planned_Hr": arrival_hr + 1,
                            "Departure_Time_Actual_Hr": departure_hr,
                            "Arrival_Time_Actual_Hr": arrival_hr,
                            "TrainSimVec_Index": train_id,
                        }
                    )
    return pl.DataFrame(rows)


def location_name(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("Location ID", "location_id", "id", "name"):
            if key in value:
                return str(value[key])
    return str(value)


def study_sims(sims_list: list[dict]) -> list[dict]:
    """Select trains that *depart* during the reporting week."""
    start_seconds = WARMUP_DAYS * 24 * SECONDS_PER_HOUR
    end_seconds = (WARMUP_DAYS + STUDY_DAYS) * 24 * SECONDS_PER_HOUR
    return [
        sim
        for sim in sims_list
        if sim["history"]["time_seconds"]
        and start_seconds <= sim["history"]["time_seconds"][0] < end_seconds
    ]


def battery_metrics(sims_list: list[dict]) -> tuple[float, float | str, float | str, float | str]:
    """Estimate net BEL energy withdrawn from start SOC minus end SOC.

    This is a net battery change, so regenerative charging can reduce it; it
    is not a separate measurement of grid charging at a facility.
    """
    energy_j = 0.0
    starts, ends, minimums = [], [], []
    for sim in sims_list:
        for loco in sim["loco_con"]["loco_vec"]:
            loco_type = next(iter(loco["loco_type"].values()))
            res = loco_type.get("res")
            if res is None or not res["history"]["soc"]:
                continue
            soc = res["history"]["soc"]
            starts.append(soc[0])
            ends.append(soc[-1])
            minimums.append(min(soc))
            energy_j += (soc[0] - soc[-1]) * res["energy_capacity_joules"]
    if not starts:
        return 0.0, "", "", ""
    return energy_j / JOULES_PER_MWH, np.mean(starts), np.mean(ends), min(minimums)


def sim_direction(sim: dict) -> tuple[str, str]:
    origins = sim.get("origs", [])
    destinations = sim.get("dests", [])
    origin = location_name(origins[0]) if origins else ""
    destination = location_name(destinations[-1]) if destinations else ""
    return origin, destination


def detect_and_export_meets(sims_list: list[dict], scenario_id: str) -> tuple[int, float, float, str, str, str]:
    """Export candidate meet events and speed traces for threshold cases.

    A dispatch hold is a proxy when the simulation does not expose an explicit
    siding occupancy event; it should not be presented as a confirmed meet.
    """
    trace_dir = TRACE_ROOT / scenario_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for sim in sims_list:
        history = sim["history"]
        origin, destination = sim_direction(sim)
        train_id = str(sim["train_id"])
        for time_s, link, speed, speed_limit, offset in zip(
            history["time_seconds"],
            history["link_idx_front"],
            history["speed_meters_per_second"],
            history["speed_limit_meters_per_second"],
            history["offset_meters"],
        ):
            records.append(
                {
                    "Train_ID": train_id,
                    "Origin": origin,
                    "Destination": destination,
                    "Time_Seconds": time_s,
                    "Time_Bin": int(round(time_s / SAVE_INTERVAL)),
                    "Link_Index": link,
                    "Offset_Meters": offset,
                    "Speed_MPH": speed / 0.44704,
                    "Speed_Limit_MPH": speed_limit / 0.44704,
                }
            )
    traces = pd.DataFrame(records)
    if traces.empty:
        return 0, 0.0, 0.0, "", "", ""

    meet_trace_path = trace_dir / "meet_speed_traces.csv"
    traces.to_csv(meet_trace_path, index=False)

    candidates = traces.merge(
        traces,
        on=["Time_Bin", "Link_Index"],
        suffixes=("_A", "_B"),
    )
    candidates = candidates.loc[
        (candidates["Train_ID_A"] < candidates["Train_ID_B"])
        & (candidates["Origin_A"] == candidates["Destination_B"])
        & (candidates["Destination_A"] == candidates["Origin_B"])
    ].copy()
    if candidates.empty:
        # The dispatcher normally prevents exact simultaneous occupancy of a
        # single-track link.  Detect the resulting meet-related hold instead:
        # actual speed below 1 mph while the route limit remains above 5 mph.
        holds = traces.loc[
            (traces["Speed_MPH"] < 1.0) & (traces["Speed_Limit_MPH"] > 5.0)
        ].copy()
        holds.sort_values(["Train_ID", "Time_Bin"], inplace=True)
        holds["New_Event"] = (
            holds.groupby("Train_ID")["Time_Bin"].diff().fillna(2) > 1
        )
        holds["Event_Number"] = holds.groupby("Train_ID")["New_Event"].cumsum()
        meets = (
            holds.groupby(["Train_ID", "Event_Number"], as_index=False)
            .agg(
                Train_ID_A=("Train_ID", "first"),
                Link_Index=("Link_Index", "first"),
                Meet_Start_Seconds=("Time_Seconds", "min"),
                Meet_End_Seconds=("Time_Seconds", "max"),
                Minimum_Speed_A_MPH=("Speed_MPH", "min"),
            )
        )
        meets = meets.loc[
            meets["Meet_End_Seconds"] - meets["Meet_Start_Seconds"] >= 300
        ].copy()
        meets["Train_ID_B"] = ""
        meets["Minimum_Speed_B_MPH"] = np.nan
        meets["Detection_Method"] = "dispatch_hold_proxy"
    else:
        candidates["Pair"] = candidates["Train_ID_A"] + "__" + candidates["Train_ID_B"]
        candidates.sort_values(["Pair", "Time_Bin"], inplace=True)
        candidates["New_Event"] = (
            candidates.groupby("Pair")["Time_Bin"].diff().fillna(2) > 1
        )
        candidates["Event_Number"] = candidates.groupby("Pair")["New_Event"].cumsum()
        meets = (
            candidates.groupby(["Pair", "Event_Number"], as_index=False)
            .agg(
                Train_ID_A=("Train_ID_A", "first"),
                Train_ID_B=("Train_ID_B", "first"),
                Link_Index=("Link_Index", "first"),
                Meet_Start_Seconds=("Time_Seconds_A", "min"),
                Meet_End_Seconds=("Time_Seconds_A", "max"),
                Minimum_Speed_A_MPH=("Speed_MPH_A", "min"),
                Minimum_Speed_B_MPH=("Speed_MPH_B", "min"),
            )
        )
        meets["Detection_Method"] = "shared_link_time"

    if meets.empty:
        return 0, 0.0, 0.0, "", str(meet_trace_path.resolve()), str(trace_dir.resolve())
    meets["Meet_Duration_Min"] = (
        meets["Meet_End_Seconds"] - meets["Meet_Start_Seconds"] + SAVE_INTERVAL
    ) / 60

    meet_summary_path = trace_dir / "meet_summary.csv"
    meets.to_csv(meet_summary_path, index=False)
    involved = set(meets["Train_ID_A"]) | {value for value in meets["Train_ID_B"] if value}
    meet_traces = traces[traces["Train_ID"].isin(involved)].copy()

    # Retain every event in the CSVs, but chart only the five longest holds so
    # threshold cases with heavy congestion do not create hundreds of plots.
    plotted_meets = meets.nlargest(5, "Meet_Duration_Min")
    for index, meet in plotted_meets.iterrows():
        start = meet["Meet_Start_Seconds"] - 1_800
        end = meet["Meet_End_Seconds"] + 1_800
        plotted_ids = [meet["Train_ID_A"]]
        if meet["Train_ID_B"]:
            plotted_ids.append(meet["Train_ID_B"])
        subset = meet_traces.loc[
            meet_traces["Train_ID"].isin(plotted_ids)
            & meet_traces["Time_Seconds"].between(start, end)
        ]
        fig, ax = plt.subplots()
        for train_id, train_trace in subset.groupby("Train_ID"):
            ax.plot(
                (train_trace["Time_Seconds"] - meet["Meet_Start_Seconds"]) / 60,
                train_trace["Speed_MPH"],
                label=f"Train {train_id}",
            )
        ax.axvspan(0, meet["Meet_Duration_Min"], color="gray", alpha=0.2, label="Meet")
        ax.set(xlabel="Time relative to meet start [min]", ylabel="Speed [mph]")
        ax.legend()
        fig.tight_layout()
        fig.savefig(trace_dir / f"meet_{index + 1:03d}_speed.png", dpi=160)
        plt.close(fig)

    total_delay_hr = meets["Meet_Duration_Min"].sum() / 60
    average_delay_min = meets["Meet_Duration_Min"].mean()
    return (
        len(meets), total_delay_hr, average_delay_min,
        str(meet_summary_path.resolve()), str(meet_trace_path.resolve()),
        str(trace_dir.resolve()),
    )


def scenario_id(service: dict, propulsion: dict, traffic: int) -> str:
    capacity = "Diesel" if propulsion["bel_capacity_mwh"] is None else f"{propulsion['bel_capacity_mwh']:g}MWh"
    return "__".join(
        [service["od_pair"], service["train_type"], propulsion["name"], capacity, f"{traffic}TPD"]
    ).replace(" ", "_")


def run_scenario(service: dict, propulsion: dict, traffic: int, baseline_hr: float | None) -> dict:
    """Simulate one case and calculate metrics across both train directions."""
    sid = scenario_id(service, propulsion, traffic)
    print(f"RUNNING {sid}", flush=True)
    config = build_planner_config(propulsion["bel_capacity_mwh"])
    plan = build_consist_plan(service, propulsion, traffic, config)
    rail_vehicles = [
        alt.RailVehicle.from_file(path, skip_init=False)
        for path in (alt.resources_root() / "rolling_stock").glob("*.yaml")
    ]
    network = alt.Network.from_file(NETWORK_PATH)
    location_map = alt.import_locations(LOCATIONS_PATH)
    (
        _consist_plan_out, loco_pool, refuel_facilities, _grid_emissions_factors,
        _nodal_energy_prices, speed_limit_train_sims, timed_paths, _untrimmed,
    ) = sim_manager.main(
        network=network,
        rail_vehicles=rail_vehicles,
        location_map=location_map,
        consist_plan_in=plan,
        train_planner_config=config,
        debug=False,
    )
    speed_limit_train_sims.set_save_interval(SAVE_INTERVAL)
    sims, _refuel_sessions = alt.run_speed_limit_train_sims(
        speed_limit_train_sims=speed_limit_train_sims,
        network=network,
        train_consist_plan_py=_consist_plan_out,
        loco_pool_py=loco_pool,
        refuel_facilities_py=refuel_facilities,
        timed_paths=[alt.TimedLinkPath.from_pydict(path) for path in timed_paths],
    )
    selected = study_sims(sims.to_pydict())
    if not selected:
        raise RuntimeError("No simulations departed during the reporting week")

    durations_hr = np.asarray([
        (sim["history"]["time_seconds"][-1] - sim["history"]["time_seconds"][0])
        / SECONDS_PER_HOUR
        for sim in selected
    ])
    average_hr = float(np.mean(durations_hr))
    if baseline_hr is None:
        baseline_hr = average_hr
    # Compare with this same service/propulsion family's one-train-per-day
    # case. A family stops once its combined average time reaches 2x baseline.
    ratio = average_hr / baseline_hr
    threshold = ratio >= 2.0

    fuel_j = 0.0
    distances_m = []
    for sim in selected:
        fuel_history = sim["loco_con"]["history"]["energy_fuel_joules"]
        if fuel_history:
            fuel_j += fuel_history[-1] - fuel_history[0]
        history = sim["history"]
        if history["offset_meters"]:
            distances_m.append(history["offset_meters"][-1] - history["offset_meters"][0])
    # Fuel energy is the change in the consist's cumulative fuel-energy trace.
    # Keep it separate from the BEL's net SOC change before adding them.
    diesel_mwh = fuel_j / JOULES_PER_MWH
    battery_mwh, start_soc, end_soc, min_soc = battery_metrics(selected)
    total_energy_mwh = diesel_mwh + battery_mwh
    route_miles = float(np.mean(distances_m) / 1_609.344) if distances_m else math.nan

    meet_values = (0, 0.0, 0.0, "", "", "")
    if threshold:
        meet_values = detect_and_export_meets(selected, sid)

    return {
        "Scenario_ID": sid,
        "OD_Pair": service["od_pair"],
        "Origin_Loaded_Direction": service["outbound_origin"],
        "Destination_Loaded_Direction": service["outbound_destination"],
        "Train_Type": service["train_type"],
        "Cars_Loaded_Outbound": service["outbound_load"][0],
        "Cars_Empty_Outbound": service["outbound_load"][1],
        "Cars_Loaded_Return": service["return_load"][0],
        "Cars_Empty_Return": service["return_load"][1],
        "Trains_Per_Direction_Per_Day": traffic,
        "Study_Train_Count": len(selected),
        "Consist_Configuration": propulsion["name"],
        "Diesel_Large_Locomotives_Per_Consist": 4 if propulsion["name"] == "Diesel" else 3,
        "BEL_Locomotives_Per_Consist": 0 if propulsion["name"] == "Diesel" else 1,
        "BEL_Capacity_MWh": propulsion["bel_capacity_mwh"] or "",
        "Random_Seed": RANDOM_SEED,
        "Warmup_Days": WARMUP_DAYS,
        "Study_Days": STUDY_DAYS,
        "Average_Train_Time_Hr": average_hr,
        "Median_Train_Time_Hr": float(np.median(durations_hr)),
        "Maximum_Train_Time_Hr": float(np.max(durations_hr)),
        "Total_Train_Time_Hr": float(np.sum(durations_hr)),
        "Baseline_Average_Train_Time_Hr": baseline_hr,
        "Average_Train_Time_Ratio": ratio,
        "Congestion_Threshold_Reached": threshold,
        "Diesel_Energy_Consumption_MWh": diesel_mwh,
        "Net_BEL_Energy_Consumption_MWh": battery_mwh,
        "Total_Propulsion_Energy_MWh": total_energy_mwh,
        "Energy_Per_Train_MWh": total_energy_mwh / len(selected),
        "Route_Distance_Miles": route_miles,
        "Energy_Per_Train_Mile_MWh": total_energy_mwh / len(selected) / route_miles,
        "BEL_Start_SOC_Average": start_soc,
        "BEL_End_SOC_Average": end_soc,
        "BEL_Minimum_SOC": min_soc,
        "Detected_Meet_Count": meet_values[0],
        "Total_Meet_Delay_Hr": meet_values[1],
        "Average_Meet_Delay_Min": meet_values[2],
        "Meet_Summary_CSV": meet_values[3],
        "Meet_Speed_Trace_CSV": meet_values[4],
        "Meet_Speed_Plot_Directory": meet_values[5],
        "Run_Status": "completed",
        "Error_Message": "",
    }


def main(validate_only: bool = False) -> None:
    ensure_results_file()
    existing = existing_scenarios()
    service_cases = SERVICE_CASES[:1] if validate_only else SERVICE_CASES
    propulsion_cases = PROPULSION_CASES[:2] if validate_only else PROPULSION_CASES
    for service in service_cases:
        for propulsion in propulsion_cases:
            baseline_hr = None
            for traffic in range(1, MAX_TRAINS_PER_DIRECTION_PER_DAY + 1):
                sid = scenario_id(service, propulsion, traffic)
                if sid in existing:
                    prior = pd.read_csv(RESULTS_CSV)
                    prior = prior.loc[prior["Scenario_ID"] == sid].iloc[-1]
                    if prior["Run_Status"] == "failed":
                        print(f"SKIPPING INFEASIBLE FAMILY AFTER {sid}", flush=True)
                        break
                    if traffic == 1:
                        baseline_hr = float(prior["Baseline_Average_Train_Time_Hr"])
                    if bool(prior["Congestion_Threshold_Reached"]):
                        break
                    continue
                try:
                    row = run_scenario(service, propulsion, traffic, baseline_hr)
                    append_result(row)
                    existing[sid] = "completed"
                    baseline_hr = float(row["Baseline_Average_Train_Time_Hr"])
                    print(
                        f"COMPLETED {sid}: avg={row['Average_Train_Time_Hr']:.3f} h, "
                        f"ratio={row['Average_Train_Time_Ratio']:.3f}",
                        flush=True,
                    )
                    if row["Congestion_Threshold_Reached"]:
                        break
                except Exception as exc:
                    append_result({
                        "Scenario_ID": sid,
                        "OD_Pair": service["od_pair"],
                        "Train_Type": service["train_type"],
                        "Consist_Configuration": propulsion["name"],
                        "BEL_Capacity_MWh": propulsion["bel_capacity_mwh"] or "",
                        "Trains_Per_Direction_Per_Day": traffic,
                        "Run_Status": "failed",
                        "Error_Message": repr(exc),
                    })
                    existing[sid] = "failed"
                    print(f"INFEASIBLE {sid}: {exc!r}", flush=True)
                    break


def refresh_meet_traces() -> None:
    results = pd.read_csv(RESULTS_CSV)
    for column in ("Meet_Summary_CSV", "Meet_Speed_Trace_CSV", "Meet_Speed_Plot_Directory"):
        results[column] = results[column].astype("object")
    thresholds = results.loc[results["Congestion_Threshold_Reached"] == True]  # noqa: E712
    for result_index, prior in thresholds.iterrows():
        existing_summary = prior.get("Meet_Summary_CSV")
        if isinstance(existing_summary, str) and existing_summary and Path(existing_summary).exists():
            print(f"KEEPING EXISTING MEETS {prior['Scenario_ID']}", flush=True)
            continue
        service = next(
            case for case in SERVICE_CASES
            if case["od_pair"] == prior["OD_Pair"] and case["train_type"] == prior["Train_Type"]
        )
        capacity = None if pd.isna(prior["BEL_Capacity_MWh"]) else float(prior["BEL_Capacity_MWh"])
        propulsion = next(
            case for case in PROPULSION_CASES
            if case["name"] == prior["Consist_Configuration"]
            and case["bel_capacity_mwh"] == capacity
        )
        refreshed = run_scenario(
            service,
            propulsion,
            int(prior["Trains_Per_Direction_Per_Day"]),
            float(prior["Baseline_Average_Train_Time_Hr"]),
        )
        for column in (
            "Detected_Meet_Count", "Total_Meet_Delay_Hr",
            "Average_Meet_Delay_Min", "Meet_Summary_CSV",
            "Meet_Speed_Trace_CSV", "Meet_Speed_Plot_Directory",
        ):
            results.at[result_index, column] = refreshed[column]
        results.to_csv(RESULTS_CSV, index=False)
        print(f"REFRESHED MEETS {prior['Scenario_ID']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--refresh-meet-traces", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.refresh_meet_traces:
        refresh_meet_traces()
    else:
        main(validate_only=args.validate_only)
    print(f"Study elapsed time: {time.perf_counter() - started:.1f} s")
