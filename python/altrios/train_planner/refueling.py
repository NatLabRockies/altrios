"""
Shared locomotive refueling/charging queue logic.

This module provides a single implementation of the discrete-event refueling
scheduler used in two places:

1. The train planner (`altrios.train_planner.planner.run_train_planner`), which
   schedules charging during forward planning when future departures are not yet
   known.
2. The post-simulation driver (`altrios.speed_limit_train_sim.run_speed_limit_train_sims`),
   which regenerates charging sessions from *realized* train schedules and realized
   energy consumption (which can differ from the plan because of meet-pass events).

Both regimes share the same locomotive-pool schema and the same port-queue logic;
they differ only in whether charging is capped at each locomotive's next scheduled
departure (planner: no cap, because dispatch waits until a locomotive is `Ready`;
driver: cap, because the realized departure schedule is fixed in advance).

Canonical column names
-----------------------
Locomotive pool (``loco_pool``) columns consumed/updated here:

- ``Locomotive_ID``, ``Locomotive_Type``, ``Node``, ``Fuel_Type``
- ``Status`` in {``Ready``, ``Dispatched``, ``Refuel_Queue``, ``Servicing``}
- ``Arrival_Time_Hr`` -- time the locomotive last arrived at ``Node``
- ``Servicing_Done_Time_Hr`` -- time servicing completes (``Refuel_Queue`` -> ``Ready``/``Servicing``)
- ``Refueling_Done_Time_Hr`` -- scheduled time charging completes (null if not scheduled)
- ``Refuel_Start_Time_Hr`` -- scheduled time charging starts (null if not scheduled)
- ``Refuel_Duration_Hr`` -- realized charging duration
- ``Refuel_Energy_J`` -- grid-side energy delivered in the most recent session
- ``Trip_Energy_J`` -- energy consumed on the trip that just arrived (driver only)
- ``SOC_J``, ``SOC_Max_J``, ``SOC_Min_J``, ``SOC_Target_J``, ``Capacity_J``
- ``Refueler_J_Per_Hr``, ``Refueler_Efficiency``, ``Port_Count``, ``Battery_Headroom_J``
- ``Min_Servicing_Time_Hr``

State-of-charge is updated at charging *start* (when a port is assigned), not at
charging finish. This makes the departing SOC of a locomotive's next trip
available as soon as its charging session is scheduled, independent of the order
in which later events are processed.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import polars as pl

#: Columns (and dtypes) of a refueling-session record.
REFUEL_SESSION_SCHEMA = {
    "Locomotive_ID": pl.UInt32,
    "Node": pl.Utf8,
    "Locomotive_Type": pl.Utf8,
    "Fuel_Type": pl.Utf8,
    "Refueler_J_Per_Hr": pl.Float64,
    "Refueler_Efficiency": pl.Float64,
    "Trip_Energy_J": pl.Float64,
    "SOC_J": pl.Float64,
    "Refuel_Energy_J": pl.Float64,
    "Refuel_Duration_Hr": pl.Float64,
    "Refuel_Start_Time_Hr": pl.Float64,
    "Refuel_End_Time_Hr": pl.Float64,
}

#: Refueler columns joined onto ``loco_pool`` when a locomotive arrives at a node.
_REFUELER_JOIN_COLS = [
    "Node",
    "Locomotive_Type",
    "Fuel_Type",
    "Refueler_J_Per_Hr",
    "Refueler_Efficiency",
    "Port_Count",
    "Battery_Headroom_J",
]


def empty_refuel_sessions() -> pl.DataFrame:
    """Return an empty refueling-session DataFrame with the canonical schema."""
    return pl.DataFrame(schema=REFUEL_SESSION_SCHEMA)


def _next_departure_map(
    departures: Optional[pl.DataFrame], current_time: float
) -> dict:
    """
    Map each locomotive to its next scheduled departure strictly after ``current_time``.

    Returns an empty dict when ``departures`` is ``None`` (planner regime: charging is
    not capped by a known future departure).
    """
    if departures is None:
        return {}
    nd = (
        departures.filter(pl.col("Departure_Time_Hr") > current_time)
        .group_by("Locomotive_ID")
        .agg(pl.col("Departure_Time_Hr").min())
    )
    return dict(
        zip(nd.get_column("Locomotive_ID").to_list(), nd.get_column("Departure_Time_Hr").to_list())
    )


def _schedule_refueling(
    loco_pool: pl.DataFrame,
    current_time: float,
    departures: Optional[pl.DataFrame],
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Assign charging ports to newly queued locomotives and build session records.

    Locomotives already refueling (with a non-null ``Refueling_Done_Time_Hr``) keep
    their existing schedule and continue to occupy ports. Newly queued locomotives
    (null ``Refueling_Done_Time_Hr``) are assigned to the earliest-freeing port,
    processed in arrival-then-ID order within each charger group.

    Returns the updated ``loco_pool`` and a DataFrame of the sessions started here.
    """
    dep_map = _next_departure_map(departures, current_time)
    cap = departures is not None

    # A charger is defined by node, locomotive type, and fuel type.
    groups = loco_pool.filter(pl.col("Status") == "Refuel_Queue").partition_by(
        ["Node", "Locomotive_Type", "Fuel_Type"]
    )

    updated_ids: list = []
    updated_start: list = []
    updated_end: list = []
    updated_duration: list = []
    updated_energy_batt: list = []
    session_rows: list = []

    for group in groups:
        port_count = int(group.get_column("Port_Count")[0])
        if port_count <= 0:
            continue

        # Ports already occupied by in-progress sessions free up at their done times.
        occupied = sorted(
            done
            for done in group.filter(
                pl.col("Refueling_Done_Time_Hr").is_not_null()
            )
            .get_column("Refueling_Done_Time_Hr")
            .to_list()
            if done > current_time
        )
        # Remaining ports are available now; represent availability with -inf.
        ports = occupied[:port_count] + [-math.inf] * max(0, port_count - len(occupied))

        newly_queued = group.filter(
            pl.col("Refueling_Done_Time_Hr").is_null()
        ).sort(["Arrival_Time_Hr", "Locomotive_ID"])

        for row in newly_queued.iter_rows(named=True):
            arrival = row["Arrival_Time_Hr"]
            rate = row["Refueler_J_Per_Hr"]
            efficiency = row["Refueler_Efficiency"]
            ideal_duration = max(0.0, row["Refuel_Duration_Hr"])
            soc_before = row["SOC_J"]

            # Take the earliest-freeing port.
            earliest = min(ports)
            port_idx = ports.index(earliest)
            start = max(earliest, arrival)
            end = start + ideal_duration
            if cap:
                end = min(end, dep_map.get(row["Locomotive_ID"], math.inf))
            end = max(end, start)
            duration = end - start
            ports[port_idx] = end

            energy_batt = duration * rate
            energy_grid = energy_batt / efficiency if efficiency else 0.0

            updated_ids.append(row["Locomotive_ID"])
            updated_start.append(start)
            updated_end.append(end)
            updated_duration.append(duration)
            updated_energy_batt.append(energy_batt)

            session_rows.append(
                {
                    "Locomotive_ID": row["Locomotive_ID"],
                    "Node": row["Node"],
                    "Locomotive_Type": row["Locomotive_Type"],
                    "Fuel_Type": row["Fuel_Type"],
                    "Refueler_J_Per_Hr": rate,
                    "Refueler_Efficiency": efficiency,
                    "Trip_Energy_J": row["Trip_Energy_J"],
                    "SOC_J": soc_before,
                    "Refuel_Energy_J": energy_grid,
                    "Refuel_Duration_Hr": duration,
                    "Refuel_Start_Time_Hr": start,
                    "Refuel_End_Time_Hr": end,
                }
            )

    if not updated_ids:
        return loco_pool, empty_refuel_sessions()

    updates = pl.DataFrame(
        {
            "Locomotive_ID": pl.Series(updated_ids, dtype=pl.UInt32),
            "__start": updated_start,
            "__end": updated_end,
            "__duration": updated_duration,
            "__energy_batt": updated_energy_batt,
        }
    )
    loco_pool = (
        loco_pool.join(updates, on="Locomotive_ID", how="left")
        .with_columns(
            # SOC is raised at charging start so a locomotive's departing SOC is
            # known as soon as the session is scheduled.
            pl.when(pl.col("__energy_batt").is_not_null())
            .then(pl.col("SOC_J") + pl.col("__energy_batt"))
            .otherwise(pl.col("SOC_J"))
            .alias("SOC_J"),
            pl.when(pl.col("__start").is_not_null())
            .then(pl.col("__start"))
            .otherwise(pl.col("Refuel_Start_Time_Hr"))
            .alias("Refuel_Start_Time_Hr"),
            pl.when(pl.col("__end").is_not_null())
            .then(pl.col("__end"))
            .otherwise(pl.col("Refueling_Done_Time_Hr"))
            .alias("Refueling_Done_Time_Hr"),
            pl.when(pl.col("__duration").is_not_null())
            .then(pl.col("__duration"))
            .otherwise(pl.col("Refuel_Duration_Hr"))
            .alias("Refuel_Duration_Hr"),
            pl.when(pl.col("__energy_batt").is_not_null())
            .then(pl.col("__energy_batt") / pl.col("Refueler_Efficiency"))
            .otherwise(pl.col("Refuel_Energy_J"))
            .alias("Refuel_Energy_J"),
        )
        .drop(["__start", "__end", "__duration", "__energy_batt"])
    )

    new_sessions = pl.DataFrame(session_rows, schema=REFUEL_SESSION_SCHEMA)
    return loco_pool, new_sessions


def update_refuel_queue(
    loco_pool: pl.DataFrame,
    refuelers: pl.DataFrame,
    current_time: float,
    refuel_sessions: pl.DataFrame,
    departures: Optional[pl.DataFrame] = None,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Advance the refueling/servicing queue to ``current_time``.

    Processes, in order:

    1. Locomotives that have arrived (``Dispatched`` with ``Arrival_Time_Hr`` <=
       ``current_time``): refresh their charger info for the arrival node, move them
       to ``Refuel_Queue``, compute their charging target and ideal duration, then
       assign ports and record the resulting sessions.
    2. Locomotives whose charging is complete (``Refuel_Queue`` with
       ``Refueling_Done_Time_Hr`` <= ``current_time``): move them to ``Servicing`` or
       ``Ready`` depending on whether servicing is still outstanding.
    3. Locomotives whose servicing is complete: move them to ``Ready``.

    Parameters
    ----------
    loco_pool : pl.DataFrame
        Locomotive pool (see module docstring for the canonical schema).
    refuelers : pl.DataFrame
        Charging/refueling facilities, keyed by ``Node``/``Locomotive_Type``/``Fuel_Type``.
    current_time : float
        Current simulation time, in hours.
    refuel_sessions : pl.DataFrame
        Accumulator of session records (canonical ``REFUEL_SESSION_SCHEMA``); new
        sessions are appended.
    departures : Optional[pl.DataFrame]
        If provided (``Locomotive_ID``, ``Departure_Time_Hr``), charging is capped at
        each locomotive's next departure after ``current_time``. If ``None``,
        locomotives always charge to their target (planner regime).

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame]
        Updated ``loco_pool`` and the (appended) ``refuel_sessions``.
    """
    arrived = loco_pool.select(
        (pl.col("Status") == "Dispatched") & (pl.col("Arrival_Time_Hr") <= current_time)
    ).to_series()

    if arrived.sum() > 0:
        loco_pool = (
            loco_pool.drop(
                ["Refueler_J_Per_Hr", "Refueler_Efficiency", "Port_Count", "Battery_Headroom_J"]
            )
            .join(
                refuelers.select(_REFUELER_JOIN_COLS),
                on=["Node", "Locomotive_Type", "Fuel_Type"],
                how="left",
            )
            .with_columns(
                pl.when(arrived)
                .then(pl.lit("Refuel_Queue"))
                .otherwise(pl.col("Status"))
                .alias("Status"),
                # Charge target respects battery headroom but never falls below the
                # current SOC (no discharging; no charging if already above threshold).
                pl.when(arrived)
                .then(
                    pl.max_horizontal(
                        pl.col("SOC_Max_J") - pl.col("Battery_Headroom_J"),
                        pl.col("SOC_J"),
                    )
                )
                .otherwise(pl.col("SOC_Target_J"))
                .alias("SOC_Target_J"),
            )
            .with_columns(
                pl.when(arrived)
                .then(
                    (pl.col("SOC_Target_J") - pl.col("SOC_J")) / pl.col("Refueler_J_Per_Hr")
                )
                .otherwise(pl.col("Refuel_Duration_Hr"))
                .alias("Refuel_Duration_Hr"),
                # Newly arrived locomotives have not been assigned a port yet.
                pl.when(arrived)
                .then(None)
                .otherwise(pl.col("Refueling_Done_Time_Hr"))
                .alias("Refueling_Done_Time_Hr"),
            )
            .sort(
                ["Node", "Locomotive_Type", "Fuel_Type", "Arrival_Time_Hr", "Locomotive_ID"],
                nulls_last=True,
            )
        )

        loco_pool, new_sessions = _schedule_refueling(loco_pool, current_time, departures)
        if new_sessions.height > 0:
            refuel_sessions = pl.concat([refuel_sessions, new_sessions])

    # Charging complete -> servicing or ready.
    refueling_finished = loco_pool.select(
        (pl.col("Status") == "Refuel_Queue")
        & (pl.col("Refueling_Done_Time_Hr") <= current_time)
    ).to_series()
    if refueling_finished.sum() > 0:
        loco_pool = loco_pool.with_columns(
            pl.when(
                pl.lit(refueling_finished)
                & (pl.col("Servicing_Done_Time_Hr") <= current_time)
            )
            .then(pl.lit("Ready"))
            .when(
                pl.lit(refueling_finished)
                & (pl.col("Servicing_Done_Time_Hr") > current_time)
            )
            .then(pl.lit("Servicing"))
            .otherwise(pl.col("Status"))
            .alias("Status"),
            pl.when(refueling_finished)
            .then(None)
            .otherwise(pl.col("Refueling_Done_Time_Hr"))
            .alias("Refueling_Done_Time_Hr"),
        )

    # Servicing complete -> ready.
    servicing_finished = loco_pool.select(
        (pl.col("Status") == "Servicing")
        & (pl.col("Servicing_Done_Time_Hr") <= current_time)
    ).to_series()
    if servicing_finished.sum() > 0:
        loco_pool = loco_pool.with_columns(
            pl.when(servicing_finished)
            .then(pl.lit("Ready"))
            .otherwise(pl.col("Status"))
            .alias("Status"),
            pl.when(servicing_finished)
            .then(None)
            .otherwise(pl.col("Servicing_Done_Time_Hr"))
            .alias("Servicing_Done_Time_Hr"),
        )

    return loco_pool.sort("Locomotive_ID"), refuel_sessions
