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

import numpy as np
import polars as pl
from numba import njit

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


@njit(cache=True)
def _assign_ports(
    group_id,
    is_new,
    arrival,
    done,
    rate,
    eff,
    ideal,
    next_dep,
    port_count,
    current_time,
):
    """Sequential earliest-freeing-port assignment over contiguous charger groups.

    Rows are pre-sorted so each charger group is contiguous, with in-progress
    sessions (``is_new == 0``, sorted by finish time) first and newly queued
    locomotives (``is_new == 1``, sorted by arrival then ID) after. For each new
    locomotive the earliest-freeing port is chosen; ``next_dep`` caps the finish
    time (``inf`` disables the cap). Results are written per row; ``scheduled``
    flags the rows that received a session.
    """
    n = group_id.shape[0]
    out_start = np.zeros(n)
    out_end = np.zeros(n)
    out_dur = np.zeros(n)
    out_ebatt = np.zeros(n)
    out_egrid = np.zeros(n)
    scheduled = np.zeros(n, dtype=np.int8)

    i = 0
    while i < n:
        gid = group_id[i]
        j = i
        while j < n and group_id[j] == gid:
            j += 1

        pc = port_count[i]
        if pc > 0:
            # Seed ports with the finish times of still-charging sessions (smallest
            # first); the remaining ports are free now (-inf).
            ports = np.full(pc, -np.inf)
            k = 0
            idx = i
            while idx < j and is_new[idx] == 0:
                if done[idx] > current_time and k < pc:
                    ports[k] = done[idx]
                    k += 1
                idx += 1

            # Assign each newly queued locomotive to the earliest-freeing port.
            while idx < j:
                best = 0
                for p in range(1, pc):
                    if ports[p] < ports[best]:
                        best = p
                start = ports[best]
                if arrival[idx] > start:
                    start = arrival[idx]
                dur_ideal = ideal[idx]
                if not (dur_ideal > 0.0):
                    dur_ideal = 0.0
                end = start + dur_ideal
                if end > next_dep[idx]:
                    end = next_dep[idx]
                if end < start:
                    end = start
                ports[best] = end

                dur = end - start
                ebatt = dur * rate[idx]
                if eff[idx] != 0.0:
                    egrid = ebatt / eff[idx]
                else:
                    egrid = 0.0

                out_start[idx] = start
                out_end[idx] = end
                out_dur[idx] = dur
                out_ebatt[idx] = ebatt
                out_egrid[idx] = egrid
                scheduled[idx] = 1
                idx += 1
        i = j

    return out_start, out_end, out_dur, out_ebatt, out_egrid, scheduled


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

    The per-group sequential assignment runs in a single Numba kernel over flat
    arrays, avoiding per-group DataFrame materialization and per-row iteration.

    Returns the updated ``loco_pool`` and a DataFrame of the sessions started here.
    """
    queue = loco_pool.filter(pl.col("Status") == "Refuel_Queue")
    if queue.height == 0:
        return loco_pool, empty_refuel_sessions()

    # Next departure per locomotive (for the departure cap); inf disables the cap.
    if departures is None:
        queue = queue.with_columns(pl.lit(math.inf).alias("__next_dep"))
    else:
        next_dep = (
            departures.filter(pl.col("Departure_Time_Hr") > current_time)
            .group_by("Locomotive_ID")
            .agg(pl.col("Departure_Time_Hr").min().alias("__next_dep"))
        )
        queue = queue.join(next_dep, on="Locomotive_ID", how="left").with_columns(
            pl.col("__next_dep").fill_null(math.inf)
        )

    # Order rows so each charger group is contiguous, with in-progress sessions
    # first (by finish time) and newly queued locomotives after (by arrival, ID).
    queue = (
        queue.with_columns(
            pl.col("Refueling_Done_Time_Hr").is_null().alias("__is_new"),
        )
        .with_columns(
            pl.when(pl.col("__is_new"))
                .then(pl.col("Arrival_Time_Hr"))
                .otherwise(pl.col("Refueling_Done_Time_Hr"))
                .alias("__sort_val"),
        )
        .sort(
            ["Node", "Locomotive_Type", "Fuel_Type", "__is_new", "__sort_val", "Locomotive_ID"]
        )
        .with_columns(
            (
                (pl.col("Node") != pl.col("Node").shift(1))
                | (pl.col("Locomotive_Type") != pl.col("Locomotive_Type").shift(1))
                | (pl.col("Fuel_Type") != pl.col("Fuel_Type").shift(1))
            )
            .fill_null(True)
            .cum_sum()
            .alias("__group_id"),
        )
    )

    out_start, out_end, out_dur, out_ebatt, out_egrid, scheduled = _assign_ports(
        queue.get_column("__group_id").to_numpy().astype(np.int64),
        queue.get_column("__is_new").to_numpy().astype(np.int8),
        queue.get_column("Arrival_Time_Hr").to_numpy().astype(np.float64),
        queue.get_column("Refueling_Done_Time_Hr").to_numpy().astype(np.float64),
        queue.get_column("Refueler_J_Per_Hr").to_numpy().astype(np.float64),
        queue.get_column("Refueler_Efficiency").to_numpy().astype(np.float64),
        queue.get_column("Refuel_Duration_Hr").to_numpy().astype(np.float64),
        queue.get_column("__next_dep").to_numpy().astype(np.float64),
        queue.get_column("Port_Count").fill_null(0).to_numpy().astype(np.int64),
        float(current_time),
    )

    scheduled_rows = queue.with_columns(
        pl.Series("__start", out_start),
        pl.Series("__end", out_end),
        pl.Series("__duration", out_dur),
        pl.Series("__energy_batt", out_ebatt),
        pl.Series("__energy_grid", out_egrid),
        pl.Series("__scheduled", scheduled),
    ).filter(pl.col("__scheduled") == 1)

    if scheduled_rows.height == 0:
        return loco_pool, empty_refuel_sessions()

    # SOC recorded on the session is the pre-charge (arrival) value.
    new_sessions = scheduled_rows.select(
        "Locomotive_ID",
        "Node",
        "Locomotive_Type",
        "Fuel_Type",
        "Refueler_J_Per_Hr",
        "Refueler_Efficiency",
        "Trip_Energy_J",
        "SOC_J",
        pl.col("__energy_grid").alias("Refuel_Energy_J"),
        pl.col("__duration").alias("Refuel_Duration_Hr"),
        pl.col("__start").alias("Refuel_Start_Time_Hr"),
        pl.col("__end").alias("Refuel_End_Time_Hr"),
    ).cast(REFUEL_SESSION_SCHEMA)

    updates = scheduled_rows.select(
        "Locomotive_ID", "__start", "__end", "__duration", "__energy_batt"
    )
    loco_pool = (
        loco_pool.join(updates, on="Locomotive_ID", how="left")
        .with_columns(
            # SOC is raised at charging start so a locomotive's departing SOC is
            # known as soon as the session is scheduled.
            pl.coalesce(pl.col("SOC_J") + pl.col("__energy_batt"), pl.col("SOC_J"))
                .alias("SOC_J"),
            pl.coalesce("__start", "Refuel_Start_Time_Hr").alias("Refuel_Start_Time_Hr"),
            pl.coalesce("__end", "Refueling_Done_Time_Hr").alias("Refueling_Done_Time_Hr"),
            pl.coalesce("__duration", "Refuel_Duration_Hr").alias("Refuel_Duration_Hr"),
            pl.coalesce(
                pl.col("__energy_batt") / pl.col("Refueler_Efficiency"),
                pl.col("Refuel_Energy_J"),
            ).alias("Refuel_Energy_J"),
        )
        .drop(["__start", "__end", "__duration", "__energy_batt"])
    )

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
    arrived = (pl.col("Status") == "Dispatched") & (
        pl.col("Arrival_Time_Hr") <= current_time
    )
    # Charging complete -> servicing (if still needed) or ready.
    charging_done = (pl.col("Status") == "Refuel_Queue") & (
        pl.col("Refueling_Done_Time_Hr") <= current_time
    )

    if loco_pool.select(arrived.any()).item():
        loco_pool = (
            loco_pool.drop(
                ["Refueler_J_Per_Hr", "Refueler_Efficiency", "Port_Count", "Battery_Headroom_J"]
            )
            .join(
                refuelers.select(_REFUELER_JOIN_COLS),
                on=["Node", "Locomotive_Type", "Fuel_Type"],
                how="left",
            )
            # Materialize the arrival mask before mutating Status, so the later
            # stages don't re-evaluate it against the updated Status.
            .with_columns(arrived.alias("__arrived"))
            .with_columns(
                pl.when(pl.col("__arrived"))
                    .then(pl.lit("Refuel_Queue"))
                    .otherwise(pl.col("Status"))
                    .alias("Status"),
                # Charge target respects battery headroom but never falls below the
                # current SOC (no discharging; no charging if already above threshold).
                pl.when(pl.col("__arrived"))
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
                pl.when(pl.col("__arrived"))
                    .then(
                        (pl.col("SOC_Target_J") - pl.col("SOC_J")) / pl.col("Refueler_J_Per_Hr")
                    )
                    .otherwise(pl.col("Refuel_Duration_Hr"))
                    .alias("Refuel_Duration_Hr"),
                # Newly arrived locomotives have not been assigned a port yet.
                pl.when(pl.col("__arrived"))
                    .then(None)
                    .otherwise(pl.col("Refueling_Done_Time_Hr"))
                    .alias("Refueling_Done_Time_Hr"),
            )
            .drop("__arrived")
            .sort(
                ["Node", "Locomotive_Type", "Fuel_Type", "Arrival_Time_Hr", "Locomotive_ID"],
                nulls_last=True,
            )
        )

        loco_pool, new_sessions = _schedule_refueling(loco_pool, current_time, departures)
        if new_sessions.height > 0:
            refuel_sessions = pl.concat([refuel_sessions, new_sessions])

    loco_pool = loco_pool.with_columns(
        pl.when(charging_done & (pl.col("Servicing_Done_Time_Hr") <= current_time))
            .then(pl.lit("Ready"))
            .when(charging_done)
            .then(pl.lit("Servicing"))
            .otherwise(pl.col("Status"))
            .alias("Status"),
        pl.when(charging_done)
            .then(None)
            .otherwise(pl.col("Refueling_Done_Time_Hr"))
            .alias("Refueling_Done_Time_Hr"),
    )

    # Servicing complete -> ready.
    servicing_done = (pl.col("Status") == "Servicing") & (
        pl.col("Servicing_Done_Time_Hr") <= current_time
    )
    loco_pool = loco_pool.with_columns(
        pl.when(servicing_done)
            .then(pl.lit("Ready"))
            .otherwise(pl.col("Status"))
            .alias("Status"),
        pl.when(servicing_done)
            .then(None)
            .otherwise(pl.col("Servicing_Done_Time_Hr"))
            .alias("Servicing_Done_Time_Hr"),
    )

    return loco_pool.sort("Locomotive_ID"), refuel_sessions
