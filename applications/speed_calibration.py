"""
Test whether posted speed limits plus meet-pass delay reproduce observed run times.

Spokane-Lethbridge is currently simulated with its speed limits capped at
40 mph and then scaled by 0.595 (see ``_calibrate_speeds`` in
``catenary_optimization/energization_optimization_corridor.py``). That scaler
was fitted when ``sim_manager`` dispatched one train per ``run_dispatch`` call,
which -- per commit d4f245c -- bypassed meet-pass entirely:

    check_deadlock skips the only train present, so has_deadlock can never
    become true and no conflict resolution runs.

So the scaler was never measuring track speed. With no congestion modelled,
simulated runs came out unrealistically fast, and shrinking every speed limit
was the only available way to slow them down. Dispatch now batches all trains
into one call, so that delay is modelled directly -- and keeping the scaler
would charge for it twice.

This script runs each corridor at ``min(posted limit, observed p99 spot
speed)`` and compares the resulting sensor-to-sensor times against what
RailState actually observed.

The cap is there because a posted limit is not always what freight runs, and
is not always real data. Toronto-Battle Creek's mainline is posted 80 mph --
an FRA Class 4 *passenger* limit, against 60 mph for freight -- and
Spokane-Lethbridge carries the network builder's synthetic 60/15 mph constants
against a maximum spot speed ever observed of 46.8 mph. The cap travels with
the corridor inputs in ``corridor_meta_<slug>.csv``, derived by
``railstate_corridor.corridor_speed_cap_mph``. ``--no-cap`` runs posted limits
alone for comparison.

Two deliberate choices:

- **Timed paths, not walked physics.** ``run_dispatch`` resolves meets and
  passes into a per-link schedule, which is exactly the quantity under test.
  Walking each train through the physics afterwards costs far more and would
  not change the arrival times, so ``walk_timed_paths`` stays off.
- **Every measured segment.** An earlier version dropped sensor pairs next to
  terminals, on the theory that their observed time was yard dwell rather than
  running time. The data does not support that: Spokane-Lethbridge's slow
  segments are slow by a similar margin in *both* directions (13.9 against
  11.4 km/h), which is not what terminal dwell looks like -- dwell accrues to
  the train entering the yard, not to both. Dropping them only removed the
  segments the model fits worst. Every pair the sensors measure is compared.

Usage:
    python speed_calibration.py --corridor toronto_battle_creek
    python speed_calibration.py --corridor toronto_battle_creek --scaler 0.595
    python speed_calibration.py --all
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl

import altrios as alt
from altrios import sim_manager
from altrios.train_planner import planner_config

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
INPUTS = HERE / "corridor_inputs"
NETWORKS = Path(alt.resources_root()) / "networks"
ROLLING_STOCK = Path(alt.resources_root()) / "rolling_stock"


@dataclass
class Corridor:
    """One corridor's inputs for a calibration run."""

    slug: str
    network: str
    locations: str
    # Per-corridor overrides applied before simulating, mirroring what the
    # energization app does today. None for both means posted limits.
    max_mph: Optional[float] = None
    scaler: Optional[float] = None
    # The observed speed cap. None means "read it from corridor_meta"; set
    # apply_cap False to run posted limits alone.
    speed_cap_mph: Optional[float] = None
    apply_cap: bool = True
    # Point at a plan built for a different traffic level, for the density sweep.
    plan_override: Optional[Path] = None

    @property
    def plan_path(self) -> Path:
        return self.plan_override or (INPUTS / f"train_consist_plan_{self.slug}_clean.csv")

    @property
    def sensors_path(self) -> Path:
        return INPUTS / f"corridor_sensors_{self.slug}.csv"

    @property
    def observed_path(self) -> Path:
        return INPUTS / f"validation_travel_times_{self.slug}.csv"

    @property
    def meta_path(self) -> Path:
        return INPUTS / f"corridor_meta_{self.slug}.csv"

    def effective_cap(self) -> Optional[float]:
        """The cap to apply, from the override or the corridor inputs."""
        if not self.apply_cap:
            return None
        if self.speed_cap_mph is not None:
            return self.speed_cap_mph
        if not self.meta_path.exists():
            raise FileNotFoundError(
                f"{self.meta_path} is missing -- regenerate the corridor inputs "
                "with railstate_corridor.py, or pass --no-cap"
            )
        return float(pl.read_csv(self.meta_path)["speed_cap_mph"][0])


CORRIDORS: Dict[str, Corridor] = {
    "toronto_battle_creek": Corridor(
        slug="toronto_battle_creek",
        network="Toronto_Battle_Creek.msgpack",
        locations="Toronto_Battle_Creek_Locations.csv",
    ),
    "winnipeg_chicago": Corridor(
        slug="winnipeg_chicago",
        network="Winnipeg_Chicago.msgpack",
        locations="Winnipeg_Chicago_locations.csv",
    ),
    "spokane_lethbridge": Corridor(
        slug="spokane_lethbridge",
        network="Spokane_Lethbridge/spokane_lethbridge.msgpack",
        locations="Spokane_Lethbridge/spokane_lethbridge_locations.csv",
    ),
}


def load_rail_vehicles() -> List[alt.RailVehicle]:
    return [
        alt.RailVehicle.from_file(str(path), skip_init=False)
        for path in sorted(ROLLING_STOCK.glob("*.yaml"))
    ]


def dual_mode_planner_config() -> planner_config.TrainPlannerConfig:
    """Planner config with a single Dual_Mode locomotive type.

    Matches ``milestone_1.3.py``: the consist plan names ``Dual_Mode`` for
    every locomotive, so the pool has to offer that type.
    """
    # ALTRIOS defaults force_max to 667.2 kN (150,000 lbf), which is uncited and
    # below both figures on the Wabtec ET44AC sheet that the adjacent line cites
    # for mass and horsepower: 200,000 lbf starting, 166,000 lbf continuous.
    # Starting TE is the right value for a hard cap, because the separate power
    # limit already reproduces the continuous rating above about 11 mph.
    loco = alt.Locomotive.default_dual_mode_loco()
    loco.set_force_max_pounds(200_000.0, "UpdateMu")

    config = planner_config.TrainPlannerConfig(
        loco_info=pd.DataFrame(
            {
                "Dual_Mode": {
                    "Fuel_Type": "Diesel",
                    "Min_Servicing_Time_Hr": 3.0,
                    "Rust_Loco": loco,
                    "Cost_USD": alt.defaults.DIESEL_LOCO_COST_USD,
                    "Lifespan_Years": alt.defaults.LOCO_LIFESPAN,
                }
            }
        )
        .transpose()
        .reset_index(names="Locomotive_Type")
    )
    refueler_info = config.refueler_info[
        config.refueler_info["Refueler_Type"] == "Diesel_Fueler"
    ].copy()
    refueler_info["Locomotive_Type"] = "Dual_Mode"
    config.refueler_info.update(refueler_info)
    return config


def prepare_network(corridor: Corridor) -> alt.Network:
    """Load the corridor network, applying any speed override under test."""
    network = alt.Network.from_file(str(NETWORKS / corridor.network))
    cap = corridor.effective_cap()
    if corridor.max_mph is None and corridor.scaler is None and cap is None:
        return network

    network_dict = network.to_pydict()
    if cap is not None:
        network_dict = alt.utilities.set_network_max_speed(
            network_dict, max_speed_mph=cap
        )
        logger.info("track limit = min(posted, %.1f mph observed p99)", cap)
    if corridor.max_mph is not None:
        network_dict = alt.utilities.set_network_max_speed(
            network_dict, max_speed_mph=corridor.max_mph
        )
        logger.info("capped speed limits at %.1f mph", corridor.max_mph)
    if corridor.scaler is not None:
        network_dict = alt.utilities.scale_network_speed_limits(
            network_dict, corridor.scaler
        )
        logger.info("scaled speed limits by %.3f", corridor.scaler)
    return alt.Network.from_pydict(network_dict)


def simulate(corridor: Corridor) -> Tuple[pl.DataFrame, List, pl.DataFrame]:
    """Dispatch the corridor's consist plan and return ``(plan_out, timed_paths, sensors)``."""
    network = prepare_network(corridor)
    location_map = alt.import_locations(str(NETWORKS / corridor.locations))
    consist_plan_in = pl.read_csv(corridor.plan_path)
    sensors = pl.read_csv(corridor.sensors_path)

    import polars as _pl
    _w = consist_plan_in.group_by("Train_ID").agg(_pl.col("Train_Weight_tons").first().alias("w"))
    logger.info(
        "dispatching %d trains over %s | plan %s | weight median %.0f max %.0f t",
        consist_plan_in["Train_ID"].n_unique(), corridor.slug, corridor.plan_path.name,
        _w["w"].median(), _w["w"].max(),
    )
    start = time.perf_counter()
    (plan_out, _loco_pool, _refuelers, _gef, _nep, _sims, timed_paths, _untrimmed) = (
        sim_manager.main(
            network=network,
            rail_vehicles=load_rail_vehicles(),
            location_map=location_map,
            consist_plan_in=consist_plan_in,
            train_planner_config=dual_mode_planner_config(),
            walk_timed_paths=False,
            debug=False,
        )
    )
    logger.info("dispatch completed in %.1f s", time.perf_counter() - start)
    return plan_out, timed_paths, sensors


def link_offsets(network: alt.Network, origin_link: int = 1) -> Dict[int, float]:
    """Corridor offset, in metres, for every mainline link and its reverse twin.

    Needed because a timed path does not name every link a train traverses --
    see ``simulated_segment_times``.
    """
    links = network.to_pydict()[1]
    by_idx = {link["idx_curr"]: link for link in links}
    offsets: Dict[int, float] = {}
    idx, seen, acc = origin_link, set(), 0.0
    while idx and idx not in seen:
        seen.add(idx)
        link = by_idx[idx]
        offsets[idx] = acc
        flip = link["idx_flip"]
        if flip:
            offsets[flip] = acc
        acc += link["length_meters"]
        idx = link["idx_next"]
    return offsets


def simulated_segment_times(
    plan_out: pl.DataFrame,
    timed_paths: Sequence,
    sensors: pl.DataFrame,
    offsets: Dict[int, float],
) -> pl.DataFrame:
    """Sensor-to-sensor times from the dispatched schedule.

    A timed path records one entry per *dispatch node* -- links where
    ``est_type == Arrive`` in the estimated-time network -- not one per link
    traversed, and that node set differs from train to train. Matching a sensor
    to an exact link index therefore finds it only for the trains that happen
    to have a node there: Sparwood was picked up for 25 of 148 Spokane trains,
    and every pair involving it was a biased subsample of trains that were not
    held.

    So each train's path is converted to (corridor offset, time) and the time
    at a sensor is interpolated between the surrounding nodes. That uses every
    train, and is exact wherever a node does coincide with the sensor.
    """
    corridor_km = dict(zip(sensors["sensor"].to_list(), sensors["corridor_km"].to_list()))

    trains = (
        plan_out.select("Train_ID", "TrainSimVec_Index", "Origin_ID", "Destination_ID")
        .unique()
        .sort("TrainSimVec_Index")
    )

    rows = []
    coverage = {name: 0 for name in corridor_km}
    for train in trains.iter_rows(named=True):
        index = int(train["TrainSimVec_Index"])
        if index >= len(timed_paths):
            continue

        # (offset, time) along the route, in offset order.
        seen = {}
        for entry in timed_paths[index]:
            off = offsets.get(int(entry["link_idx"]))
            if off is not None:
                seen.setdefault(off, entry["time_seconds"] / 3600)
        if len(seen) < 2:
            continue
        xs = sorted(seen)
        ys = [seen[x] for x in xs]

        hits = {}
        for name, km in corridor_km.items():
            target = km * 1000
            if target < xs[0] or target > xs[-1]:
                continue  # sensor outside the span this train was recorded over
            hits[name] = float(np.interp(target, xs, ys))
            coverage[name] += 1

        for a, b in ((a, b) for a in hits for b in hits if corridor_km[a] < corridor_km[b]):
            rows.append(
                {
                    "Train_ID": train["Train_ID"],
                    "Origin_ID": train["Origin_ID"],
                    "from_sensor": a,
                    "to_sensor": b,
                    "segment_km": corridor_km[b] - corridor_km[a],
                    "sim_travel_time_hr": abs(hits[b] - hits[a]),
                }
            )

    logger.info(
        "sensor coverage in the simulated paths (of %d trains): %s",
        trains.height,
        dict(sorted(coverage.items())),
    )
    return pl.DataFrame(rows)


def compare(observed: pl.DataFrame, simulated: pl.DataFrame) -> pl.DataFrame:
    """Observed against simulated, one row per sensor pair."""
    obs = (
        observed.group_by("from_sensor", "to_sensor")
        .agg(
            pl.col("segment_km").first().alias("km"),
            pl.len().alias("obs_n"),
            pl.col("travel_time_hr").median().alias("obs_median_hr"),
            pl.col("travel_time_hr").quantile(0.25).alias("obs_p25_hr"),
            pl.col("travel_time_hr").quantile(0.75).alias("obs_p75_hr"),
        )
    )
    sim = (
        simulated.group_by("from_sensor", "to_sensor")
        .agg(
            pl.len().alias("sim_n"),
            pl.col("sim_travel_time_hr").median().alias("sim_median_hr"),
            pl.col("sim_travel_time_hr").quantile(0.25).alias("sim_p25_hr"),
            pl.col("sim_travel_time_hr").quantile(0.75).alias("sim_p75_hr"),
        )
    )
    return (
        obs.join(sim, on=["from_sensor", "to_sensor"], how="inner")
        .with_columns(
            (pl.col("km") / pl.col("obs_median_hr")).alias("obs_kph"),
            (pl.col("km") / pl.col("sim_median_hr")).alias("sim_kph"),
            (
                (pl.col("sim_median_hr") - pl.col("obs_median_hr"))
                / pl.col("obs_median_hr")
            ).alias("pct_error"),
        )
        .sort("km")
    )


def run(corridor: Corridor) -> pl.DataFrame:
    plan_out, timed_paths, sensors = simulate(corridor)
    offsets = link_offsets(prepare_network(corridor))
    simulated = simulated_segment_times(plan_out, timed_paths, sensors, offsets)
    observed = pl.read_csv(corridor.observed_path)
    table = compare(observed, simulated)

    logger.info("\n%s", table.to_pandas().to_string(index=False))
    if table.height:
        weighted = (table["pct_error"] * table["km"]).sum() / table["km"].sum()
        logger.info(
            "distance-weighted median error: %+.1f%% (simulated %s observed)",
            100 * weighted,
            "slower than" if weighted > 0 else "faster than",
        )
    return table


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--corridor", choices=sorted(CORRIDORS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--scaler",
        type=float,
        help="scale every speed limit before simulating, to reproduce the old "
        "calibration for comparison (0.595 for Spokane-Lethbridge)",
    )
    parser.add_argument(
        "--max-mph",
        type=float,
        help="cap speed limits before scaling, as the old calibration did",
    )
    parser.add_argument(
        "--no-cap",
        action="store_true",
        help="run posted speed limits alone, without the observed p99 cap",
    )
    parser.add_argument("--out", type=Path, help="write the comparison table here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    slugs = sorted(CORRIDORS) if args.all else [args.corridor]
    if not slugs or slugs == [None]:
        parser.error("pass --corridor or --all")

    tables = []
    for slug in slugs:
        corridor = CORRIDORS[slug]
        corridor.scaler = args.scaler
        corridor.max_mph = args.max_mph
        corridor.apply_cap = not args.no_cap
        logger.info("=== %s ===", slug)
        table = run(corridor).with_columns(pl.lit(slug).alias("corridor"))
        tables.append(table)

    if args.out and tables:
        pl.concat(tables).write_csv(args.out)
        logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
