#!/usr/bin/env python3
"""
Validate a built corridor network by running ALTRIOS train simulations on it.

Loads the network YAML + locations CSV produced by the network builder, then
dispatches and walks a SpeedLimitTrainSim between the corridor endpoints --
by default one trip in each direction, exercising the forward and reverse
link chains, the endpoint buffers, and the bidirectional location rows.

The endpoint names are read from the locations CSV, so this works unchanged
for any corridor the builder produced.

Usage (uv-managed environment):
    # Validate the default Spokane -> Lethbridge build in results/:
    uv run python applications/network_builder/run_corridor_sim.py

    # Another corridor / location:
    uv run python applications/network_builder/run_corridor_sim.py \
        --prefix applications/network_builder/results/winnipeg_chicago
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import altrios as alt

THIS_DIR = Path(__file__).resolve().parent  # applications/network_builder
DEFAULT_PREFIX = THIS_DIR / "results" / "spokane_lethbridge"


def endpoint_names(locations_csv: Path) -> tuple[str, str]:
    """Origin/destination location IDs from the builder's locations CSV
    (rows are ordered origin first, destination last)."""
    with open(locations_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No location rows in {locations_csv}")
    return rows[0]["Location ID"], rows[-1]["Location ID"]


def run_trip(network, location_map, origin, dest, args) -> dict:
    """Dispatch and walk one train from origin to dest; return summary stats."""
    rail_vehicles = [
        alt.RailVehicle.from_file(
            alt.resources_root() / "rolling_stock/Manifest_Loaded.yaml"
        ),
        alt.RailVehicle.from_file(
            alt.resources_root() / "rolling_stock/Manifest_Empty.yaml"
        ),
    ]
    train_config = alt.TrainConfig(
        rail_vehicles=rail_vehicles,
        n_cars_by_type={
            "Manifest_Loaded": args.cars_loaded,
            "Manifest_Empty": args.cars_empty,
        },
        train_length_meters=None,
        train_mass_kilograms=None,
    )
    loco_con = alt.Consist([alt.Locomotive.default()] * args.locos, args.save_interval)
    tsb = alt.TrainSimBuilder(
        train_id="0",
        origin_id=origin,
        destination_id=dest,
        train_config=train_config,
        loco_con=loco_con,
    )
    train_sim = tsb.make_speed_limit_train_sim(
        location_map=location_map, save_interval=args.save_interval
    )
    train_sim.set_save_interval(args.save_interval)

    t0 = time.perf_counter()
    est_time_net, _consist = alt.make_est_times(train_sim, network)
    timed_link_path = next(
        iter(
            alt.run_dispatch(
                network,
                alt.SpeedLimitTrainSimVec([train_sim]),
                [est_time_net],
                False,
                False,
            )
        )
    )
    train_sim.walk_timed_path(network=network, timed_path=timed_link_path)
    wall_s = time.perf_counter() - t0

    hist = train_sim.to_pydict()["history"]
    return {
        "origin": origin,
        "destination": dest,
        "distance_km": (hist["offset_meters"][-1] - hist["offset_meters"][0]) / 1e3,
        "duration_h": hist["time_seconds"][-1] / 3600,
        "fuel_gj": train_sim.get_energy_fuel_joules(False) / 1e9,
        "wall_s": wall_s,
    }


def main(args: argparse.Namespace) -> None:
    prefix = args.prefix.resolve()
    # The builder writes msgpack by default; older builds may be YAML.
    candidates = [Path(f"{prefix}.msgpack"), Path(f"{prefix}.yaml")]
    network_path = next((p for p in candidates if p.exists()), None)
    loc_path = Path(str(prefix) + "_locations.csv")
    if network_path is None:
        sys.exit(f"Not found: {prefix}.msgpack or .yaml (build the corridor first)")
    if not loc_path.exists():
        sys.exit(f"Not found: {loc_path} (build the corridor first)")

    print(f"Loading {network_path.name}...")
    network = alt.Network.from_file(network_path)
    location_map = alt.import_locations(loc_path)
    origin, dest = endpoint_names(loc_path)

    trips = [(origin, dest)] if args.one_way else [(origin, dest), (dest, origin)]
    for o, d in trips:
        r = run_trip(network, location_map, o, d, args)
        print(
            f"{r['origin']:>12} -> {r['destination']:<12} "
            f"{r['distance_km']:7.1f} km  {r['duration_h']:5.2f} h  "
            f"fuel {r['fuel_gj']:7.1f} GJ  (simulated in {r['wall_s']:.1f} s)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        default=DEFAULT_PREFIX,
        help="Output prefix of a built corridor, i.e. <prefix>.msgpack (or .yaml) and "
        f"<prefix>_locations.csv exist (default: {DEFAULT_PREFIX.relative_to(THIS_DIR.parent.parent)}).",
    )
    parser.add_argument(
        "--one-way",
        action="store_true",
        help="Only simulate origin -> destination (default: both directions).",
    )
    parser.add_argument(
        "--cars-loaded", type=int, default=50, help="Loaded manifest cars (default 50)."
    )
    parser.add_argument(
        "--cars-empty", type=int, default=50, help="Empty manifest cars (default 50)."
    )
    parser.add_argument(
        "--locos", type=int, default=4, help="Conventional locomotives (default 4)."
    )
    parser.add_argument(
        "--save-interval", type=int, default=100, help="History save interval."
    )
    main(parser.parse_args())
