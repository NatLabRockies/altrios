"""Generate one all-train meet-pass stringline diagram per study iteration.

Each image contains both directions and every study-week train for one case.
The manifest maps its scenario ID to the exported PNG.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import altrios as alt
from altrios import sim_manager
import full_od_bel_congestion_study as base
import full_od_bel_congestion_directional_study as directional


OUTPUT_DIR = base.RESULTS_DIR / "full_od_bel_meet_pass_diagrams"
MANIFEST_CSV = base.RESULTS_DIR / "Full_OD_BEL_Meet_Pass_Diagram_Manifest.csv"


def expanded_mask(mask: np.ndarray) -> np.ndarray:
    """Include neighboring samples so colored segments meet the black line."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    return np.convolve(mask.astype(int), np.ones(3, dtype=int), mode="same") > 0


def masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=float)
    output[mask] = values[mask]
    return output


def export_stringline(sims: list[dict], scenario_id: str, service: dict) -> Path:
    """Draw all train trajectories against common time and corridor axes."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / f"{scenario_id}__meet_pass_stringline.png"
    route_length_miles = max(
        float(np.nanmax(np.asarray(sim["history"]["offset_meters"], dtype=float))) / 1_609.344
        for sim in sims
    )
    # Day zero on the graph is the first day after the seven-day warmup.
    study_start_seconds = base.WARMUP_DAYS * 24.0 * base.SECONDS_PER_HOUR

    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    for sim in sims:
        history = sim["history"]
        times = np.asarray(history["time_seconds"], dtype=float)
        distances = np.asarray(history["offset_meters"], dtype=float) / 1_609.344
        speeds = np.asarray(history["speed_meters_per_second"], dtype=float) / 0.44704
        limits = np.asarray(history["speed_limit_meters_per_second"], dtype=float) / 0.44704
        if len(times) < 2:
            continue
        origin, _destination = directional.get_direction(sim)
        if origin != service["outbound_origin"]:
            # Reverse the return train's trip-relative offset so both
            # directions share distance measured from the outbound origin.
            distances = route_length_miles - distances
        days = (times - study_start_seconds) / (24.0 * base.SECONDS_PER_HOUR)
        links = np.asarray(history.get("link_idx_front", [np.nan] * len(times)), dtype=float)
        # The recorded histories do not include an authoritative track-type
        # field. Orange marks absent links or near-zero limits (typically
        # terminal/off-network time). Red is a dispatch-hold proxy, not a
        # confirmed siding-track occupancy classification.
        off_network = (~np.isfinite(links)) | (limits <= 0.05)
        siding_hold = expanded_mask((speeds < 1.0) & (limits > 5.0)) & ~off_network

        ax.plot(days, distances, color="#202020", linewidth=0.8, alpha=0.82, zorder=1)
        if siding_hold.any():
            ax.plot(days, masked(distances, siding_hold), color="#E31A1C", linewidth=1.5, alpha=0.95, zorder=3)
        if off_network.any():
            ax.plot(days, masked(distances, expanded_mask(off_network)), color="#FF8C00", linewidth=1.5, alpha=0.95, zorder=4)

    handles = [
        Line2D([0], [0], color="#202020", linewidth=1.4, label="Track: Main / movement"),
        Line2D([0], [0], color="#E31A1C", linewidth=1.8, label="Track: Siding / dispatch hold"),
        Line2D([0], [0], color="#FF8C00", linewidth=1.8, label="Track: Terminal / off network"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.set(
        xlabel="Time after study-period start (days)",
        ylabel=f"Distance from {service['outbound_origin']} (miles)",
        title=(
            f"Meet-pass stringline: {service['od_pair']} — {service['train_type']}\n"
            f"{scenario_id}"
        ),
        ylim=(0.0, route_length_miles),
    )
    ax.grid(True, color="#9CA3AF", linewidth=0.65, alpha=0.65)
    ax.set_axisbelow(True)
    fig.subplots_adjust(right=0.79, bottom=0.12, top=0.88)
    fig.savefig(plot_path, dpi=180, facecolor="white")
    plt.close(fig)
    return plot_path


def simulate_and_plot(source_row: pd.Series) -> tuple[Path, int]:
    """Rerun a completed case to retrieve every train's sampled trajectory."""
    service, propulsion = directional.find_case(source_row)
    traffic = int(source_row["Trains_Per_Direction_Per_Day"])
    scenario_id = str(source_row["Scenario_ID"])
    config = base.build_planner_config(propulsion["bel_capacity_mwh"])
    plan = base.build_consist_plan(service, propulsion, traffic, config)
    rail_vehicles = [
        alt.RailVehicle.from_file(path, skip_init=False)
        for path in (alt.resources_root() / "rolling_stock").glob("*.yaml")
    ]
    network = alt.Network.from_file(base.NETWORK_PATH)
    locations = alt.import_locations(base.LOCATIONS_PATH)
    consist_plan_out, loco_pool, refuelers, _grid, _prices, train_sims, timed_paths, _untrimmed = sim_manager.main(
        network=network,
        rail_vehicles=rail_vehicles,
        location_map=locations,
        consist_plan_in=plan,
        train_planner_config=config,
        debug=False,
    )
    train_sims.set_save_interval(base.SAVE_INTERVAL)
    sims, _sessions = alt.run_speed_limit_train_sims(
        speed_limit_train_sims=train_sims,
        network=network,
        train_consist_plan_py=consist_plan_out,
        loco_pool_py=loco_pool,
        refuel_facilities_py=refuelers,
        timed_paths=[alt.TimedLinkPath.from_pydict(path) for path in timed_paths],
    )
    selected = base.study_sims(sims.to_pydict())
    return export_stringline(selected, scenario_id, service), len(selected)


def completed_scenarios() -> set[str]:
    """Use the manifest to skip images already completed on a prior run."""
    if not MANIFEST_CSV.exists():
        return set()
    manifest = pd.read_csv(MANIFEST_CSV)
    return set(manifest.loc[manifest["Run_Status"] == "completed", "Scenario_ID"])


def append_manifest(row: dict) -> None:
    columns = ["Scenario_ID", "Diagram_Path", "Study_Train_Count", "Run_Status", "Error_Message"]
    write_header = not MANIFEST_CSV.exists()
    with MANIFEST_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in columns})


def main(reset: bool = False, limit: int | None = None) -> None:
    """Generate one image per completed combined-study case."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if reset:
        MANIFEST_CSV.unlink(missing_ok=True)
    done = completed_scenarios()
    source = pd.read_csv(base.RESULTS_CSV)
    source = source.loc[source["Run_Status"] == "completed"]
    completed_this_run = 0
    for _, source_row in source.iterrows():
        scenario_id = str(source_row["Scenario_ID"])
        if scenario_id in done:
            print(f"SKIPPING {scenario_id}", flush=True)
            continue
        print(f"RUNNING {scenario_id}", flush=True)
        try:
            diagram, count = simulate_and_plot(source_row)
            append_manifest({
                "Scenario_ID": scenario_id,
                "Diagram_Path": str(diagram.resolve()),
                "Study_Train_Count": count,
                "Run_Status": "completed",
                "Error_Message": "",
            })
            print(f"COMPLETED {scenario_id}", flush=True)
            completed_this_run += 1
            if limit is not None and completed_this_run >= limit:
                break
        except Exception as exc:
            append_manifest({"Scenario_ID": scenario_id, "Run_Status": "failed", "Error_Message": repr(exc)})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    started = time.perf_counter()
    main(reset=args.reset, limit=args.limit)
    print(f"Meet-pass diagram elapsed time: {time.perf_counter() - started:.1f} s")
