"""Terminal-mode registry and generic simulation dispatcher.

The LIFTS package currently implements two intermodal process flows that share
a single arrival trigger: rail-to-truck (IC, inbound containers) and the
reverse, truck-to-rail (OC, outbound containers). Future work will add
vessel<->truck and vessel<->rail flows that share yard resources (storage
areas, hostlers, ...) but differ in trigger semantics and schedule shape.

This module abstracts those concerns into a `TerminalMode`:
    * `build_schedule(input_plan, terminal_name) -> list[dict]`
        Mode-specific schedule construction. For intermodal_rail this wraps
        `utilities.build_train_timetable`. A future vessel mode would read a
        vessel call list instead.
    * `process_arrival(env, terminal, schedule_entry) -> generator`
        SimPy process kicked off once per scheduled arrival.

Modes register themselves into `_MODES` and are dispatched by name via
`run_terminal_simulation(mode=...)`. A future declarative input file can
select a mode by name without changing this module.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import polars as pl
import simpy

from altrios.lifts import distances, utilities
from altrios.lifts.classes import Terminal, loggingLevel
from altrios.lifts.energy_use import energy_use_records, CO2_KG_PER_UNIT
from altrios.lifts.train_flow import process_train_arrival


# ---------------------------------------------------------------------------
# Mode registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TerminalMode:
    """Bundle of callables that define one terminal process flow family."""
    name: str
    process_arrival: Callable[..., Any]
    build_schedule: Callable[[Any, str], List[dict]]
    description: str = ""


_MODES: Dict[str, TerminalMode] = {}


def register_mode(mode: TerminalMode) -> None:
    """Register a TerminalMode by name. Raises if the name is already taken."""
    if mode.name in _MODES:
        raise ValueError(
            f"Terminal mode '{mode.name}' is already registered "
            f"(registered modes: {list(_MODES)})"
        )
    _MODES[mode.name] = mode


def get_mode(name: str) -> TerminalMode:
    """Look up a registered TerminalMode by name."""
    if name not in _MODES:
        raise KeyError(
            f"Unknown terminal mode '{name}'. Registered modes: {list(_MODES)}"
        )
    return _MODES[name]


def list_modes() -> List[str]:
    """Return the names of all currently registered terminal modes."""
    return list(_MODES)


# ---------------------------------------------------------------------------
# Generic dispatcher
# ---------------------------------------------------------------------------

def run_terminal_simulation(
        mode: str,
        train_consist_plan: pl.DataFrame,
        terminal: str,
        log_level: loggingLevel = loggingLevel.BASIC) -> tuple[pl.DataFrame, pl.DataFrame, Terminal]:
    """Run a terminal simulation using the named mode.

    Parameters
    ----------
    mode
        Name of a registered `TerminalMode` (see `list_modes()`).
    train_consist_plan
        Mode-specific input plan. For `intermodal_rail` this is the train
        consist plan DataFrame. Future modes may interpret this argument
        differently or rename it.
    terminal
        Terminal name (selects entries from the input plan).

    Returns
    -------
    container_data
        One row per container with arrival/handling/departure timestamps.
    vehicle_log
        One row per resource event (crane/hostler/truck), including an
        `energy_consumption(gal)` column for that event.
    terminal_obj
        The fully-populated `Terminal` (config + layout + resource counts +
        post-run state) used for the simulation. Callers can read its
        attributes to recover the parameters that were actually used.
    """
    mode_obj = get_mode(mode)

    # Reset the module-level energy-use buffer so repeat invocations are clean
    energy_use_records.clear()

    terminal_config = utilities.load_config(utilities.resources_root() / "config.yaml")
    terminal_layout = distances.get_layout(terminal_config)

    random.seed(42)

    train_timetable = mode_obj.build_schedule(train_consist_plan, terminal)
    truck_number = max([entry['truck_number'] for entry in train_timetable])
    chassis_count = max([entry['empty_cars'] + entry['full_cars'] for entry in train_timetable])
    env = simpy.Environment()

    terminal_obj = Terminal(env,
        config=terminal_config,
        layout=terminal_layout,
        truck_capacity=truck_number,
        chassis_count=chassis_count,
        log_level=log_level)

    terminal_obj.log(loggingLevel.BASIC, f"[INFO] layout: {terminal_layout}")
    terminal_obj.log(loggingLevel.DEBUG, "\nTrain timetable:")
    for schedule in train_timetable:
        terminal_obj.log(loggingLevel.DEBUG, str(schedule))
        env.process(mode_obj.process_arrival(env, terminal_obj, schedule))

    num_tracks = terminal_obj.track_number
    num_cranes = sum(terminal_obj.cranes_on_track.values())
    num_hostlers = terminal_obj.hostler_number

    terminal_obj.log(loggingLevel.BASIC, "*" * 50)
    terminal_obj.log(loggingLevel.BASIC,
        f"Mode: {mode_obj.name}; Tracks: {num_tracks}; Cranes: {num_cranes}; Hostlers: {num_hostlers}")
    terminal_obj.log(loggingLevel.BASIC, "*" * 50)

    # When a train_consist_plan is supplied, simulate the entire plan regardless
    # of the config's simulation length. Otherwise honor the configured horizon.
    if train_consist_plan is not None:
        env.run()
    else:
        env.run(until=terminal_config["simulation"]["length"])

    # Create DataFrame for container events. container_events is a flat list of
    # (container_id, event_type, timestamp) tuples; pivot it once to wide form
    # here rather than building a dict-of-dicts during the run.
    _CONTAINER_EVENT_TYPES = [
        "train_arrival_expected", "train_arrival_actual",
        "crane_unload", "hostler_pickup", "hostler_dropoff",
        "truck_arrival", "truck_dropoff", "truck_pickup", "truck_exit",
        "crane_load", "train_depart",
    ]
    events_long = pl.DataFrame(
        terminal_obj.state.container_events,
        schema={"container_id": pl.Utf8, "event_type": pl.Utf8, "timestamp": pl.Float64},
        orient="row",
    )
    if events_long.height == 0:
        container_data = pl.DataFrame(
            schema={"container_id": pl.Utf8, **{t: pl.Float64 for t in _CONTAINER_EVENT_TYPES}}
        )
    else:
        container_data = events_long.pivot(
            values="timestamp", index="container_id", on="event_type",
            aggregate_function="last",
        )
    # Ensure all known event-type columns exist even if some never fired,
    # so downstream column references below don't blow up on small runs.
    missing = [t for t in _CONTAINER_EVENT_TYPES if t not in container_data.columns]
    if missing:
        container_data = container_data.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(t) for t in missing]
        )
    container_data = (
        container_data
        .lazy()
        .sort("container_id")
        .select(pl.col("container_id"), pl.exclude("container_id"))
        .with_columns(
            pl.when(pl.col("truck_exit").is_not_null() & pl.col("train_arrival_expected").is_not_null())
                .then(pl.col("truck_exit") - pl.col("train_arrival_expected"))
                .when(pl.col("train_depart").is_not_null())
                .then(pl.col("crane_load") - pl.col("truck_arrival"))
                .otherwise(None)
                .alias("container_processing_time"),
            pl.col("container_id").str.extract(r"Train-(\d+)").cast(pl.Int64).alias("train_id"),
            pl.col("container_id").str.starts_with("IC").alias("is_ic")
        )
    )

    # OC train actual arrival time
    train_arrival_df = (container_data
        .filter(
            pl.col("is_ic"),
            pl.col("train_arrival_actual").is_not_null()
        )
        .group_by("train_id")
        .agg(pl.col("train_arrival_actual").mean())
    )
    # OC train expected arrival time
    train_arrival_expected_df = (container_data
        .filter(
            pl.col("is_ic"),
            pl.col("train_arrival_expected").is_not_null()
        )
        .group_by("train_id")
        .agg(pl.col("train_arrival_expected").mean())
    )
    container_data = (container_data
        .join(train_arrival_df, on="train_id", how="left")
        .join(train_arrival_expected_df, on="train_id", how="left")
        .rename({
            "train_arrival_actual_right": "train_arrival_actual_oc",
            "train_arrival_expected_right": "train_arrival_expected_oc"
        })
        .drop("is_ic", "train_id")
    ).collect()

    vehicle_log = pl.DataFrame(
        energy_use_records,
        schema={
            "resource_type": pl.Utf8,
            "fuel_type": pl.Utf8,
            "resource_id": pl.Utf8,
            "track_id": pl.Utf8,
            "train_id": pl.Utf8,
            "container_id": pl.Utf8,
            "event_type": pl.Utf8,
            "zone": pl.Utf8,
            "energy_consumption(gal_or_kWh)": pl.Float64,
            "load/travel_time(hr)": pl.Float64,
            "record_timestamp": pl.Float64,
        },
    ).with_columns(
        # Coarse CO2-equivalent emissions, computed at end-of-sim from the
        # static per-fuel factors in energy_use.CO2_KG_PER_UNIT. Unknown fuel
        # types yield NaN so they're easy to spot.
        (
            pl.col("energy_consumption(gal_or_kWh)")
            * pl.col("fuel_type").replace_strict(CO2_KG_PER_UNIT, default=float("nan"))
        ).alias("emissions(kgCO2)")
    )
    return container_data, vehicle_log, terminal_obj


# ---------------------------------------------------------------------------
# Built-in modes
# ---------------------------------------------------------------------------

def _build_intermodal_rail_schedule(train_consist_plan: pl.DataFrame, terminal: str) -> List[dict]:
    return utilities.build_train_timetable(train_consist_plan, terminal, as_dicts=True)


register_mode(TerminalMode(
    name="intermodal_rail",
    process_arrival=process_train_arrival,
    build_schedule=_build_intermodal_rail_schedule,
    description=(
        "Intermodal rail terminal: trains arrive on tracks; ICs (rail-to-truck) "
        "and OCs (truck-to-rail) flow through shared chassis, hostlers, parking "
        "slots, and gantry cranes."
    ),
))


if __name__ == "__main__":
    consist_plan = (pl.read_csv(utilities.package_root() / 'resources' / 'train_consist_plan.csv')
        .with_columns(pl.lit("Intermodal").alias("Train_Type"))
    )
    container_data, vehicle_log_df, terminal_obj = run_terminal_simulation(
        mode="intermodal_rail",
        train_consist_plan=consist_plan,
        terminal="Allouez",
    )

