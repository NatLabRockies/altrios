"""Terminal-mode registry and generic simulation dispatcher.

The LIFTS package supports multiple terminal process-flow families ("modes"):
rail-to-truck and truck-to-rail (today the only registered mode,
``intermodal_rail``), and — coming in Phase 1 of the vessel-workflows
rebuild — vessel<->truck and vessel<->rail flows that share yard equipment
but differ in trigger semantics and schedule shape.

A :class:`TerminalMode` bundles the mode-specific pieces:
    * ``build_schedule(input_plan, terminal_name) -> list[dict]``
        Mode-specific schedule construction (e.g. train timetable, vessel
        call list, drayage arrival stream).
    * ``process_arrival(env, terminal, schedule_entry) -> generator``
        SimPy process kicked off once per scheduled arrival.
    * ``resource_specs`` / ``event_specs`` / ``event_types`` /
      ``container_id_pattern`` / ``post_process`` (optional)
        Phase-3-ready metadata that lets the dispatcher itself stay
        mode-agnostic. See :class:`TerminalMode` for details.

Modes register themselves into ``_MODES`` and are dispatched by name via
:func:`run_terminal_simulation`. A future declarative input file can select
a mode by name without changing this module.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import polars as pl
import simpy

from altrios.lifts import distances, utilities
from altrios.lifts.classes import Terminal, loggingLevel
from altrios.lifts.energy_use import energy_use_records, CO2_KG_PER_UNIT
from altrios.lifts.resources_decl import EventSpec, ResourceSpec
from altrios.lifts.train_flow import process_train_arrival


# Type aliases for the optional post-processing hook.
PostProcessFn = Callable[
    [pl.DataFrame, pl.DataFrame], Tuple[pl.DataFrame, pl.DataFrame]
]


# ---------------------------------------------------------------------------
# Mode registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TerminalMode:
    """Bundle that defines one terminal process flow family.

    Required fields describe how arrivals are scheduled and simulated.
    Optional fields are Phase-3-ready metadata: declarative resource specs,
    declared event surface, an id-extraction pattern, and a mode-specific
    post-processing hook applied to the dispatcher's generic outputs.

    Parameters
    ----------
    name
        Unique mode identifier used by :func:`get_mode`.
    process_arrival
        SimPy generator function kicked off once per scheduled arrival.
    build_schedule
        Mode-specific schedule builder; returns a list of arrival-dict
        entries to feed into ``process_arrival``.
    description
        Human-readable description for diagnostics.
    resource_specs
        ``ResourceSpec`` list this mode requires. The dispatcher unions
        these across active modes (dedup by name) and builds primitives
        onto ``TerminalState``. Empty in legacy modes that still use
        ``TerminalState.__init__``'s built-in primitives.
    event_specs
        ``EventSpec`` list advertising the per-arrival events this mode
        emits. Diagnostic in Phase 1.
    event_types
        Container-event type names this mode emits. The dispatcher backfills
        any of these that never fired with all-null columns, so downstream
        consumers see a stable schema regardless of run size.
    container_id_pattern
        Optional precompiled regex used by ``post_process`` to extract
        arrival ids from container ids.
    post_process
        Optional ``(container_data, vehicle_log) -> (container_data,
        vehicle_log)`` hook for mode-specific dataframe shaping. Receives
        DataFrames whose generic preparation (pivot, sort, backfill) is
        already done.
    """
    name: str
    process_arrival: Callable[..., Any]
    build_schedule: Callable[[Any, str], List[dict]]
    description: str = ""
    resource_specs: Tuple[ResourceSpec, ...] = field(default_factory=tuple)
    event_specs: Tuple[EventSpec, ...] = field(default_factory=tuple)
    event_types: Tuple[str, ...] = field(default_factory=tuple)
    container_id_pattern: Optional[re.Pattern] = None
    post_process: Optional[PostProcessFn] = None


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

    container_data, vehicle_log = _build_generic_outputs(terminal_obj, mode_obj)

    if mode_obj.post_process is not None:
        container_data, vehicle_log = mode_obj.post_process(container_data, vehicle_log)

    return container_data, vehicle_log, terminal_obj


# ---------------------------------------------------------------------------
# Generic output assembly (mode-agnostic)
# ---------------------------------------------------------------------------

def _build_generic_outputs(
    terminal_obj: Terminal, mode_obj: TerminalMode
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Pivot container events to wide form and assemble the vehicle log.

    The dispatcher uses ``mode_obj.event_types`` to backfill any declared
    event columns that never fired during this run. Mode-specific derived
    columns (e.g. ``container_processing_time`` for ``intermodal_rail``)
    are added by the mode's ``post_process`` callable, not here.
    """
    # container_events is a flat list of (container_id, event_type, timestamp)
    # tuples; pivot it once to wide form here rather than building a
    # dict-of-dicts during the run.
    declared_event_types = list(mode_obj.event_types)
    events_long = pl.DataFrame(
        terminal_obj.state.container_events,
        schema={
            "container_id": pl.Utf8,
            "event_type": pl.Utf8,
            "timestamp": pl.Float64,
        },
        orient="row",
    )
    if events_long.height == 0:
        container_data = pl.DataFrame(
            schema={
                "container_id": pl.Utf8,
                **{t: pl.Float64 for t in declared_event_types},
            }
        )
    else:
        container_data = events_long.pivot(
            values="timestamp",
            index="container_id",
            on="event_type",
            aggregate_function="last",
        )
    # Backfill any declared event columns that never fired so downstream
    # consumers see a stable schema regardless of run size.
    missing = [t for t in declared_event_types if t not in container_data.columns]
    if missing:
        container_data = container_data.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(t) for t in missing]
        )
    container_data = (
        container_data
        .sort("container_id")
        .select(pl.col("container_id"), pl.exclude("container_id"))
    )

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
    return container_data, vehicle_log


# ---------------------------------------------------------------------------
# Built-in modes
# ---------------------------------------------------------------------------

# Container-event columns the intermodal_rail flow emits. Order is the original
# logical order of the journey (used to backfill any column that never fired).
_INTERMODAL_RAIL_EVENT_TYPES: Tuple[str, ...] = (
    "train_arrival_expected", "train_arrival_actual",
    "crane_unload", "hostler_pickup", "hostler_dropoff",
    "truck_arrival", "truck_dropoff", "truck_pickup", "truck_exit",
    "crane_load", "train_depart",
)

_INTERMODAL_RAIL_TRAIN_ID_PATTERN = re.compile(r"Train-(\d+)")


def _intermodal_rail_post_process(
    container_data: pl.DataFrame, vehicle_log: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Add intermodal_rail-specific derived columns to ``container_data``.

    Computes per-container processing time and joins each container row back
    to its train's mean ``train_arrival_*`` so OC rows carry the train arrival
    info that only IC rows record directly.
    """
    container_data = container_data.lazy().with_columns(
        pl.when(
            pl.col("truck_exit").is_not_null()
            & pl.col("train_arrival_expected").is_not_null()
        )
        .then(pl.col("truck_exit") - pl.col("train_arrival_expected"))
        .when(pl.col("train_depart").is_not_null())
        .then(pl.col("crane_load") - pl.col("truck_arrival"))
        .otherwise(None)
        .alias("container_processing_time"),
        pl.col("container_id")
        .str.extract(r"Train-(\d+)")
        .cast(pl.Int64)
        .alias("train_id"),
        pl.col("container_id").str.starts_with("IC").alias("is_ic"),
    )

    # OC train actual arrival time (averaged across IC rows of that train).
    train_arrival_df = (
        container_data
        .filter(pl.col("is_ic"), pl.col("train_arrival_actual").is_not_null())
        .group_by("train_id")
        .agg(pl.col("train_arrival_actual").mean())
    )
    # OC train expected arrival time
    train_arrival_expected_df = (
        container_data
        .filter(pl.col("is_ic"), pl.col("train_arrival_expected").is_not_null())
        .group_by("train_id")
        .agg(pl.col("train_arrival_expected").mean())
    )
    container_data = (
        container_data
        .join(train_arrival_df, on="train_id", how="left")
        .join(train_arrival_expected_df, on="train_id", how="left")
        .rename({
            "train_arrival_actual_right": "train_arrival_actual_oc",
            "train_arrival_expected_right": "train_arrival_expected_oc",
        })
        .drop("is_ic", "train_id")
    ).collect()
    return container_data, vehicle_log


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
    event_types=_INTERMODAL_RAIL_EVENT_TYPES,
    container_id_pattern=_INTERMODAL_RAIL_TRAIN_ID_PATTERN,
    post_process=_intermodal_rail_post_process,
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

