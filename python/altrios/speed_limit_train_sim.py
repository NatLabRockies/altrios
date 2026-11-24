"""
Post-simulation driver that runs speed-limited train simulations along their
realized (meet-pass-adjusted) timed paths and regenerates charging sessions.

This is the Python replacement for the former Rust ``run_speed_limit_train_sims``
function. It shares its refueling/charging queue logic with the train planner via
:mod:`altrios.train_planner.refueling`; the only physics that remains in Rust is the
per-simulation :meth:`SpeedLimitTrainSim.walk_timed_path`.

The driver walks each locomotive's trips in arrival order, chaining state-of-charge
across trips: a locomotive departs on its next trip with whatever SOC it charged to
(capped at the scheduled departure), which in turn determines the realized energy
consumption of that trip.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Union

import polars as pl

import altrios as alt
from altrios.train_planner import refueling


def _reset_loco_pool(loco_pool: pl.DataFrame) -> pl.DataFrame:
    """Build a fresh locomotive pool for post-simulation refueling.

    Every locomotive starts `Ready` at full charge with bookkeeping columns reset.

    Args:
        loco_pool: Pool with `Locomotive_ID`, `Locomotive_Type`, `Node`,
            `Fuel_Type`, `SOC_Max_J`, `SOC_Min_J` and `Capacity_J`.

    Returns:
        Pool with the refueling-queue columns added and state reset.
    """
    return loco_pool.select(
        [
            "Locomotive_ID",
            "Locomotive_Type",
            "Node",
            "Fuel_Type",
            "SOC_Max_J",
            "SOC_Min_J",
            "Capacity_J",
        ]
    ).with_columns(
        pl.col("SOC_Max_J").alias("SOC_J"),
        pl.lit(0.0).alias("SOC_Target_J"),
        pl.lit("Ready").alias("Status"),
        pl.lit(math.inf).alias("Arrival_Time_Hr"),
        # No servicing delay is modeled post-simulation.
        pl.lit(-math.inf).alias("Servicing_Done_Time_Hr"),
        pl.lit(0.0).alias("Min_Servicing_Time_Hr"),
        pl.lit(None, dtype=pl.Float64).alias("Refueling_Done_Time_Hr"),
        pl.lit(None, dtype=pl.Float64).alias("Refuel_Start_Time_Hr"),
        pl.lit(0.0).alias("Refuel_Duration_Hr"),
        pl.lit(0.0).alias("Refuel_Energy_J"),
        pl.lit(0.0).alias("Trip_Energy_J"),
        pl.lit(0.0).alias("Refueler_J_Per_Hr"),
        pl.lit(0.0).alias("Refueler_Efficiency"),
        pl.lit(0, dtype=pl.UInt32).alias("Port_Count"),
        pl.lit(0.0).alias("Battery_Headroom_J"),
    )


def _walk_and_read(
    sim: "alt.SpeedLimitTrainSim",
    network: Union["alt.Network", List["alt.Link"]],
    timed_path: "alt.TimedLinkPath",
    departing_soc_pct: List[float],
) -> Tuple["alt.SpeedLimitTrainSim", List[float], List[float]]:
    """Walk a timed path from the given departing SOC.

    Args:
        sim: Simulation to walk.
        network: Rail network the timed path refers to.
        timed_path: Realized link path with arrival times.
        departing_soc_pct: Departing state of charge (fraction) per locomotive,
            in `Locomotive_ID` order.

    Returns:
        The walked simulation, end-of-trip battery energy (J) per locomotive, and
        trip energy consumption (J) per locomotive; both are 0 for locomotives
        without a battery.
    """
    sim_dict = sim.to_pydict()
    loco_vec = sim_dict["loco_con"]["loco_vec"]
    for loco, soc_pct in zip(loco_vec, departing_soc_pct):
        loco_type = next(iter(loco["loco_type"].values()))
        if "res" in loco_type:
            loco_type["res"]["state"]["soc"] = float(soc_pct)
    sim = alt.SpeedLimitTrainSim.from_pydict(sim_dict)

    sim.walk_timed_path(network=network, timed_path=timed_path)

    walked = sim.to_pydict()
    new_soc_j: List[float] = []
    trip_energy_j: List[float] = []
    for loco in walked["loco_con"]["loco_vec"]:
        loco_type = next(iter(loco["loco_type"].values()))
        if "res" in loco_type:
            capacity_j = loco_type["res"]["energy_capacity_joules"]
            new_soc_j.append(loco_type["res"]["state"]["soc"] * capacity_j)
            trip_energy_j.append(loco_type["res"]["state"]["energy_out_chemical_joules"])
        else:
            new_soc_j.append(0.0)
            trip_energy_j.append(0.0)
    return sim, new_soc_j, trip_energy_j


def run_speed_limit_train_sims(
    speed_limit_train_sims: "alt.SpeedLimitTrainSimVec",
    network: Union["alt.Network", List["alt.Link"]],
    train_consist_plan: pl.DataFrame,
    loco_pool: pl.DataFrame,
    refuelers: pl.DataFrame,
    timed_paths: List["alt.TimedLinkPath"],
) -> Tuple["alt.SpeedLimitTrainSimVec", pl.DataFrame]:
    """Run speed-limited train sims and regenerate charging sessions.

    Trips run in arrival order, each departing at the SOC its units last reached.

    Args:
        speed_limit_train_sims: Simulations to run, indexed by `TrainSimVec_Index`.
        network: Rail network used to walk each timed path.
        train_consist_plan: Realized consist plan with `Arrival_Time_Actual_Hr`,
            `Departure_Time_Actual_Hr`, `Locomotive_ID`, `Destination_ID` and
            `TrainSimVec_Index`.
        loco_pool: Locomotive pool; only identity and energy columns are used.
        refuelers: Charging and refueling facilities.
        timed_paths: Realized timed paths, indexed by `TrainSimVec_Index`.

    Returns:
        The walked simulations and a refueling-session frame with the
        `REFUEL_SESSION_SCHEMA` columns.
    """
    sims_list = [
        alt.SpeedLimitTrainSim.from_pydict(d)
        for d in speed_limit_train_sims.to_pydict()
    ]
    loco_pool = _reset_loco_pool(loco_pool)
    refuel_sessions = refueling.empty_refuel_sessions()

    arrivals = train_consist_plan.select(
        [
            "Arrival_Time_Actual_Hr",
            "Locomotive_ID",
            "Destination_ID",
            "TrainSimVec_Index",
        ]
    ).sort(["Arrival_Time_Actual_Hr", "Locomotive_ID"])

    departures = train_consist_plan.select(
        pl.col("Locomotive_ID"),
        pl.col("Departure_Time_Actual_Hr").alias("Departure_Time_Hr"),
    )

    remaining = arrivals
    current_time = arrivals.get_column("Arrival_Time_Actual_Hr").min()

    done = False
    while not done:
        arriving_now = remaining.filter(
            pl.col("Arrival_Time_Actual_Hr") == current_time
        )
        if arriving_now.height > 0:
            upd_ids: List[int] = []
            upd_soc: List[float] = []
            upd_trip: List[float] = []
            upd_dest: List[str] = []

            for idx in arriving_now.get_column("TrainSimVec_Index").unique().to_list():
                train_rows = (
                    train_consist_plan.filter(pl.col("TrainSimVec_Index") == idx)
                    .unique(subset=["Locomotive_ID"])
                    .sort("Locomotive_ID")
                )
                loco_ids = train_rows.get_column("Locomotive_ID").to_list()
                destination = train_rows.get_column("Destination_ID")[0]

                sub = loco_pool.filter(
                    pl.col("Locomotive_ID").is_in(loco_ids)
                ).sort("Locomotive_ID")
                departing_soc_pct = (
                    sub.get_column("SOC_J") / sub.get_column("Capacity_J")
                ).to_list()

                sims_list[idx], new_soc_j, trip_energy_j = _walk_and_read(
                    sims_list[idx], network, timed_paths[idx], departing_soc_pct
                )

                upd_ids.extend(loco_ids)
                upd_soc.extend(new_soc_j)
                upd_trip.extend(trip_energy_j)
                upd_dest.extend([destination] * len(loco_ids))

            updates = pl.DataFrame(
                {
                    "Locomotive_ID": pl.Series(upd_ids, dtype=pl.UInt32),
                    "__new_soc": upd_soc,
                    "__trip_energy": upd_trip,
                    "__destination": upd_dest,
                }
            )
            loco_pool = (
                loco_pool.join(updates, on="Locomotive_ID", how="left")
                .with_columns(
                    pl.when(pl.col("__destination").is_not_null())
                    .then(pl.col("__new_soc"))
                    .otherwise(pl.col("SOC_J"))
                    .alias("SOC_J"),
                    pl.when(pl.col("__destination").is_not_null())
                    .then(pl.col("__trip_energy"))
                    .otherwise(pl.col("Trip_Energy_J"))
                    .alias("Trip_Energy_J"),
                    pl.when(pl.col("__destination").is_not_null())
                    .then(pl.col("__destination"))
                    .otherwise(pl.col("Node"))
                    .alias("Node"),
                    pl.when(pl.col("__destination").is_not_null())
                    .then(pl.lit(current_time))
                    .otherwise(pl.col("Arrival_Time_Hr"))
                    .alias("Arrival_Time_Hr"),
                    pl.when(pl.col("__destination").is_not_null())
                    .then(pl.lit("Dispatched"))
                    .otherwise(pl.col("Status"))
                    .alias("Status"),
                )
                .drop(["__new_soc", "__trip_energy", "__destination"])
            )

        loco_pool, refuel_sessions = refueling.update_refuel_queue(
            loco_pool, refuelers, current_time, refuel_sessions, departures
        )

        remaining = remaining.filter(
            pl.col("Arrival_Time_Actual_Hr") > current_time
        )
        next_times: List[float] = []
        next_arrival = remaining.get_column("Arrival_Time_Actual_Hr").min()
        if next_arrival is not None:
            next_times.append(next_arrival)
        in_progress = (
            loco_pool.filter(
                (pl.col("Status") == "Refuel_Queue")
                & (pl.col("Refueling_Done_Time_Hr") > current_time)
            )
            .get_column("Refueling_Done_Time_Hr")
            .min()
        )
        if in_progress is not None:
            next_times.append(in_progress)

        if not next_times:
            done = True
        else:
            current_time = min(next_times)

    return alt.SpeedLimitTrainSimVec(sims_list), refuel_sessions
