"""Terminal-mode registry and generic simulation dispatcher.

The LIFTS package supports three terminal process-flow families ("modes"):
``truck_rail``, ``rail_vessel``, and ``vessel_truck``. All three move
containers through the *main container stack* using a shared catalog of
yard equipment (RTGs, top-picks, yard tractors, chassis), differing only
in which endpoint(s) (rail track, vessel berth, gate) the containers
enter and exit through.

A :class:`TerminalMode` bundles the mode-specific pieces:
    * ``build_schedule(terminal_name, inputs) -> list[dict]``
        Mode-specific schedule construction. Each returned dict is one
        arrival entry tagged with ``_kind`` so the dispatcher can route
        it through the right per-arrival generator. ``inputs`` is the
        per-mode input dict supplied to ``run_terminal_simulation`` (e.g.
        ``{"train_consist_plan": df, "drayage_schedule": df}``).
    * ``process_arrival(env, terminal, schedule_entry) -> generator``
        SimPy process kicked off once per scheduled arrival. Dispatches
        internally on ``schedule_entry["_kind"]``.
    * ``resource_specs`` / ``event_specs`` / ``event_types`` /
      ``container_id_pattern`` / ``post_process`` (optional)
        Declarative metadata. See :class:`TerminalMode` for details.

Modes register themselves into ``_MODES``. The dispatcher
:func:`run_terminal_simulation` accepts a list of mode names and a
per-mode ``inputs`` mapping; multiple modes run concurrently against a
single ``Terminal`` with their resource_specs union-merged so any pool
referenced by more than one mode is a *single* SimPy primitive shared
across them.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import polars as pl
import simpy

from altrios.lifts import distances, specs, utilities
from altrios.lifts.classes import Terminal, loggingLevel
from altrios.lifts.drayage_flow import process_drayage_arrival
from altrios.lifts.consumption import consumption_records, CO2_KG_PER_UNIT
from altrios.workflow_engine import EventSpec, ResourceSpec, merge_specs
from altrios.lifts.train_flow import process_train_arrival
from altrios.lifts.vessel_flow import process_vessel_arrival


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
    Optional fields are declarative metadata: resource specs, declared
    event surface, an id-extraction pattern, and a mode-specific
    post-processing hook applied to the dispatcher's generic outputs.

    Parameters
    ----------
    name
        Unique mode identifier used by :func:`get_mode`.
    process_arrival
        SimPy generator function kicked off once per scheduled arrival.
        Receives one entry dict from ``build_schedule`` and dispatches
        internally on ``entry["_kind"]``.
    build_schedule
        Mode-specific schedule builder; returns a list of arrival-dict
        entries to feed into ``process_arrival``. Signature is
        ``(terminal_name: str, inputs: dict) -> List[dict]``;
        ``inputs`` is the per-mode input dict (e.g.
        ``{"train_consist_plan": df, "drayage_schedule": df}``).
    description
        Human-readable description for diagnostics.
    resource_specs
        ``ResourceSpec`` list this mode requires. The dispatcher passes
        these to ``TerminalState`` so only the requested pools are
        instantiated. Empty falls back to ``TerminalState``'s default
        (union of all three modes' specs).
    event_specs
        ``EventSpec`` list advertising the per-arrival SimPy events this
        mode emits. Diagnostic only.
    event_types
        Container-event type names this mode emits. The dispatcher
        backfills any of these that never fired with all-null columns,
        so downstream consumers see a stable schema regardless of run
        size.
    container_id_pattern
        Optional precompiled regex used by ``post_process`` to extract
        arrival ids from container ids.
    post_process
        Optional ``(container_data, resource_log) -> (container_data,
        resource_log)`` hook for mode-specific dataframe shaping. Receives
        DataFrames whose generic preparation (pivot, sort, backfill) is
        already done.
    """
    name: str
    process_arrival: Callable[..., Any]
    build_schedule: Callable[..., List[dict]]
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
        modes: List[str] | str,
        terminal: str,
        inputs: Dict[str, Dict[str, Any]],
        log_level: loggingLevel = loggingLevel.BASIC,
) -> tuple[pl.DataFrame, pl.DataFrame, Terminal]:
    """Run a multi-mode terminal simulation against one ``Terminal``.

    Multiple modes execute concurrently in a single SimPy environment.
    Their ``resource_specs`` are union-merged via
    :func:`resources_decl.merge_specs` so any pool referenced by more
    than one mode becomes a *single* SimPy primitive shared across them
    (cross-mode contention). Disagreements between modes' specs for the
    same ``name`` raise ``ValueError`` from ``merge_specs``.

    Parameters
    ----------
    modes
        One mode name (str) or a list of registered mode names. See
        :func:`list_modes`. When passing a single string, it is wrapped
        into a one-element list internally.
    terminal
        Terminal name (selects rows from each mode's input plans).
    inputs
        Per-mode input dict: ``{mode_name: {input_key: value, ...}}``.
        Recognized ``input_key`` values are mode-specific
        (``train_consist_plan``, ``vessel_schedule``,
        ``drayage_schedule``); see each mode's ``build_schedule`` for
        the keys it consumes. Modes not in this dict get ``{}``.
    log_level
        Logging verbosity passed to ``Terminal``.

    Returns
    -------
    container_data
        One row per container with arrival/handling/departure timestamps.
        Columns are the union of every active mode's declared
        ``event_types`` (missing columns are backfilled with nulls).
    resource_log
        One row per resource event (crane/hostler/truck) across all
        modes. Each row has a ``role`` tag (``equipment`` /
        ``infrastructure`` / ``storage``, mirroring
        ``ResourceSpec.role``), a ``quantity`` tag (currently always
        ``"energy"``; future-proofs the per-row generalization to other
        consumption quantities), a ``consumption_value`` column carrying
        the value in the configured native unit, and a coarse
        ``emissions(kgCO2)`` column for energy rows.
    terminal_obj
        The fully-populated ``Terminal`` used for the simulation. The
        same instance is shared by all active modes; pool primitives on
        ``terminal_obj.state`` (e.g. ``state.berths``,
        ``state.main_stack_rtgs``) are SimPy objects shared across all
        modes that reference them.
    """
    if isinstance(modes, str):
        modes = [modes]
    if not modes:
        raise ValueError("run_terminal_simulation: 'modes' must be non-empty.")

    mode_objs = [get_mode(name) for name in modes]

    # Reset the module-level energy-use buffer so repeat invocations are clean.
    consumption_records.clear()

    terminal_config = utilities.load_config(utilities.resources_root() / "config.yaml")
    terminal_layout = distances.get_layout(terminal_config)

    random.seed(42)

    # ----- Build per-mode schedules and tag each entry with its source mode.
    schedules_by_mode: Dict[str, List[dict]] = {}
    for mode_obj in mode_objs:
        mode_inputs = inputs.get(mode_obj.name, {}) or {}
        entries = mode_obj.build_schedule(terminal, mode_inputs)
        for e in entries:
            e["_mode"] = mode_obj.name
        schedules_by_mode[mode_obj.name] = entries

    # ----- Merge resource specs across active modes. Cross-mode references
    # to the same spec name produce a single shared SimPy primitive.
    merged_specs = list(merge_specs({
        m.name: m.resource_specs for m in mode_objs if m.resource_specs
    }).values())
    resource_specs = merged_specs if merged_specs else None

    env = simpy.Environment()
    terminal_obj = Terminal(
        env,
        config=terminal_config,
        layout=terminal_layout,
        log_level=log_level,
        resource_specs=resource_specs,
    )

    # ----- Diagnostic banner.
    total_entries = sum(len(s) for s in schedules_by_mode.values())
    terminal_obj.log(loggingLevel.BASIC, f"[INFO] layout: {terminal_layout}")
    terminal_obj.log(loggingLevel.BASIC, "*" * 50)
    terminal_obj.log(loggingLevel.BASIC,
        f"Modes: {[m.name for m in mode_objs]}; "
        f"Schedule entries: {total_entries} "
        f"({', '.join(f'{n}={len(s)}' for n, s in schedules_by_mode.items())})"
    )
    terminal_obj.log(loggingLevel.BASIC, "*" * 50)

    # ----- Spawn one SimPy process per arrival, routed by its tagged mode.
    process_arrival_by_mode = {m.name: m.process_arrival for m in mode_objs}
    for mode_name, entries in schedules_by_mode.items():
        terminal_obj.log(loggingLevel.DEBUG, f"\n{mode_name} schedule:")
        proc = process_arrival_by_mode[mode_name]
        for entry in entries:
            terminal_obj.log(loggingLevel.DEBUG, str(entry))
            env.process(proc(env, terminal_obj, entry))

    # Run to completion when at least one schedule entry was generated;
    # otherwise honor the configured simulation horizon.
    if total_entries > 0:
        env.run()
    else:
        env.run(until=terminal_config["simulation"]["length"])

    # ----- Union event_types across modes; chain post_process callables in
    # the user-supplied mode order.
    merged_event_types: Tuple[str, ...] = tuple(dict.fromkeys(
        t for m in mode_objs for t in m.event_types
    ))
    container_data, resource_log = _build_generic_outputs_with_event_types(
        terminal_obj, merged_event_types,
    )
    for mode_obj in mode_objs:
        if mode_obj.post_process is not None:
            container_data, resource_log = mode_obj.post_process(
                container_data, resource_log,
            )

    return container_data, resource_log, terminal_obj


# ---------------------------------------------------------------------------
# Generic output assembly (mode-agnostic)
# ---------------------------------------------------------------------------

def _build_generic_outputs_with_event_types(
    terminal_obj: Terminal, declared_event_types: Tuple[str, ...],
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Pivot container events to wide form and assemble the vehicle log.

    ``declared_event_types`` is the union of every active mode's declared
    event types; any column listed there that never fired during this
    run is backfilled with nulls so downstream consumers see a stable
    schema regardless of run size and active mode mix. Mode-specific
    derived columns (e.g. ``container_processing_time`` for
    ``truck_rail``) are added by each mode's ``post_process`` callable.
    """
    # container_events is a flat list of (container_id, event_type, timestamp)
    # tuples; pivot it once to wide form here rather than building a
    # dict-of-dicts during the run.
    event_types_list = list(declared_event_types)
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
                **{t: pl.Float64 for t in event_types_list},
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
    missing = [t for t in event_types_list if t not in container_data.columns]
    if missing:
        container_data = container_data.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(t) for t in missing]
        )
    container_data = (
        container_data
        .sort("container_id")
        .select(pl.col("container_id"), pl.exclude("container_id"))
    )

    resource_log = pl.DataFrame(
        consumption_records,
        schema={
            "resource_type": pl.Utf8,
            "role": pl.Utf8,
            "fuel_type": pl.Utf8,
            "resource_id": pl.Utf8,
            "track_id": pl.Utf8,
            "train_id": pl.Utf8,
            "container_id": pl.Utf8,
            "event_type": pl.Utf8,
            "zone": pl.Utf8,
            "quantity": pl.Utf8,
            "consumption_value": pl.Float64,
            "load/travel_time(hr)": pl.Float64,
            "record_timestamp": pl.Float64,
        },
    ).with_columns(
        # Coarse CO2-equivalent emissions, computed at end-of-sim from the
        # static per-fuel factors in consumption.CO2_KG_PER_UNIT. Unknown fuel
        # types yield NaN so they're easy to spot.
        (
            pl.col("consumption_value")
            * pl.col("fuel_type").replace_strict(CO2_KG_PER_UNIT, default=float("nan"))
        ).alias("emissions(kgCO2)")
    )
    return container_data, resource_log


# ---------------------------------------------------------------------------
# Built-in modes
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Built-in modes
# ---------------------------------------------------------------------------

# Container-event columns the rail/yard/drayage flows emit. Order is the
# logical order of the journey (used to backfill any column that never fired).
_TRUCK_RAIL_EVENT_TYPES: Tuple[str, ...] = (
    "train_arrival_expected", "train_arrival_actual",
    "rail_track_rtg_unload",
    "yard_tractor_rail_to_stack",
    "main_stack_rtg_stack_in", "top_pick_stack_in", "stack_in",
    "drayage_arrival", "drayage_gate_in",
    "main_stack_rtg_stack_out", "top_pick_stack_out", "stack_out",
    "yard_tractor_stack_to_rail",
    "drayage_gate_out",
    "rail_track_rtg_load",
    "train_depart",
)

_VESSEL_EVENT_TYPES: Tuple[str, ...] = (
    "vessel_arrival_expected", "vessel_arrival_actual",
    "sts_unload",
    "main_stack_rtg_stack_in", "top_pick_stack_in", "stack_in",
    "main_stack_rtg_stack_out", "top_pick_stack_out", "stack_out",
    "sts_load",
    "vessel_depart",
)

_RAIL_VESSEL_EVENT_TYPES: Tuple[str, ...] = tuple(
    dict.fromkeys((*_TRUCK_RAIL_EVENT_TYPES, *_VESSEL_EVENT_TYPES))
)

_VESSEL_TRUCK_EVENT_TYPES: Tuple[str, ...] = tuple(
    dict.fromkeys((
        *_VESSEL_EVENT_TYPES,
        "drayage_arrival", "drayage_gate_in",
        "stack_in", "stack_out",
        "drayage_gate_out",
    ))
)

_TRAIN_ID_PATTERN = re.compile(r"Train-(\d+)")
_VESSEL_ID_PATTERN = re.compile(r"Vessel-(\d+)")


# ---------------------------------------------------------------------------
# Helpers for schedule construction
# ---------------------------------------------------------------------------

def _resolve_table(value: Any, default_path) -> pl.DataFrame:
    """Coerce ``value`` (DataFrame, path, or None) into a polars DataFrame.

    ``None`` falls back to ``default_path`` so demos can omit the auxiliary
    schedule and get the canonical sample data."""
    if value is None:
        return pl.read_csv(default_path)
    if isinstance(value, pl.DataFrame):
        return value
    # Assume it's a path-like.
    return pl.read_csv(value)


def _synthesize_drayage_from_trains(
    train_entries: List[dict],
    pre_arrival_window_hr: float = 0.5,
    post_arrival_window_hr: float = 1.0,
) -> List[dict]:
    """Generate one dropoff per OC (just before each train arrival) and one
    pickup per IC (just after each train arrival). Used by ``truck_rail``
    when the caller does not supply an explicit drayage schedule, so the
    rail side has containers to load/unload at the stack.

    ``truck_id`` ranges are partitioned per train so collisions are
    impossible: dropoffs use ``train_id * 100000 + i`` and pickups use
    ``train_id * 100000 + 50000 + i``.
    """
    out: List[dict] = []
    for entry in train_entries:
        train_id = int(entry["train_id"])
        arrival = float(entry["arrival_time"])
        oc_n = int(entry.get("oc_number") or 0)
        ic_n = int(entry.get("full_cars") or 0)
        for i in range(1, oc_n + 1):
            out.append({
                "_kind": "drayage",
                "truck_id": train_id * 100000 + i,
                "arrival_time": max(0.0, arrival - pre_arrival_window_hr),
                "action": "dropoff",
                "container_id": None,
                "train_id": train_id,
            })
        for i in range(1, ic_n + 1):
            out.append({
                "_kind": "drayage",
                "truck_id": train_id * 100000 + 50000 + i,
                "arrival_time": arrival + post_arrival_window_hr,
                "action": "pickup",
                "container_id": None,
                "train_id": train_id,
            })
    return out


# ---------------------------------------------------------------------------
# truck_rail
# ---------------------------------------------------------------------------

def _build_truck_rail_schedule(
    terminal_name: str, inputs: Dict[str, Any],
) -> List[dict]:
    """Trains + drayage trucks. Inputs:
      * ``train_consist_plan``: pl.DataFrame (required).
      * ``drayage_schedule``: optional pl.DataFrame or path. If omitted,
        one drayage dropoff per outbound and one drayage pickup per
        inbound container is synthesized from the train schedule.
    """
    train_consist_plan = inputs.get("train_consist_plan")
    if train_consist_plan is None:
        raise ValueError(
            "truck_rail.build_schedule: inputs['train_consist_plan'] is required."
        )
    train_entries = utilities.build_train_timetable(
        train_consist_plan, terminal_name, as_dicts=True,
    )
    for entry in train_entries:
        entry["_kind"] = "train"

    drayage_input = inputs.get("drayage_schedule")
    if drayage_input is None:
        drayage_entries = _synthesize_drayage_from_trains(train_entries)
    else:
        drayage_df = _resolve_table(
            drayage_input,
            utilities.resources_root() / "drayage_schedule.csv",
        )
        drayage_entries = utilities.build_drayage_schedule(
            drayage_df, terminal_name, as_dicts=True,
        )
        for e in drayage_entries:
            e["_kind"] = "drayage"

    combined = train_entries + drayage_entries
    combined.sort(key=lambda e: float(e.get("arrival_time") or 0.0))
    return combined


def _truck_rail_process_arrival(env, terminal, entry: dict):
    kind = entry.get("_kind", "train")
    if kind == "train":
        yield from process_train_arrival(env, terminal, entry)
    elif kind == "drayage":
        yield from process_drayage_arrival(env, terminal, entry)
    else:
        raise ValueError(f"truck_rail: unknown _kind {kind!r}")


def _truck_rail_post_process(
    container_data: pl.DataFrame, resource_log: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Add per-container ``container_processing_time`` and join OC rows to
    their train's actual arrival time."""
    if container_data.height == 0:
        return container_data, resource_log

    has_truck_exit = "drayage_gate_out" in container_data.columns
    has_train_arrival_exp = "train_arrival_expected" in container_data.columns
    has_train_depart = "train_depart" in container_data.columns
    has_drayage_arrival = "drayage_arrival" in container_data.columns
    has_rtg_load = "rail_track_rtg_load" in container_data.columns

    if has_truck_exit and has_train_arrival_exp:
        ic_proc = (
            pl.when(
                pl.col("drayage_gate_out").is_not_null()
                & pl.col("train_arrival_expected").is_not_null()
            )
            .then(pl.col("drayage_gate_out") - pl.col("train_arrival_expected"))
        )
    else:
        ic_proc = pl.lit(None)

    if has_train_depart and has_drayage_arrival and has_rtg_load:
        oc_proc = (
            pl.when(pl.col("train_depart").is_not_null())
            .then(pl.col("rail_track_rtg_load") - pl.col("drayage_arrival"))
        )
    else:
        oc_proc = pl.lit(None)

    container_data = container_data.with_columns(
        ic_proc.otherwise(oc_proc).alias("container_processing_time"),
        pl.col("container_id").str.extract(r"Train-(\d+)").cast(pl.Int64).alias("train_id"),
        pl.col("container_id").str.starts_with("IC").alias("is_ic"),
    )

    if has_train_arrival_exp:
        train_actual = (
            container_data
            .filter(pl.col("is_ic"))
            .filter(pl.col("train_arrival_actual").is_not_null())
            if "train_arrival_actual" in container_data.columns
            else container_data.head(0)
        )
        if train_actual.height > 0:
            train_actual = train_actual.group_by("train_id").agg(
                pl.col("train_arrival_actual").mean().alias("train_arrival_actual_oc")
            )
            container_data = container_data.join(
                train_actual, on="train_id", how="left",
            )

    container_data = container_data.drop("is_ic", "train_id")
    return container_data, resource_log


register_mode(TerminalMode(
    name="truck_rail",
    process_arrival=_truck_rail_process_arrival,
    build_schedule=_build_truck_rail_schedule,
    description=(
        "Truck<->rail terminal: trains arrive on tracks; drayage trucks "
        "deliver export OCs and pick up import ICs at the gate. All "
        "containers route through the main container stack via the rail "
        "yard tractor pool (rail<->stack) and the stack RTG/top-pick crane "
        "fleet."
    ),
    resource_specs=specs.TRUCK_RAIL_SPECS,
    event_types=_TRUCK_RAIL_EVENT_TYPES,
    container_id_pattern=_TRAIN_ID_PATTERN,
    post_process=_truck_rail_post_process,
))


# ---------------------------------------------------------------------------
# rail_vessel
# ---------------------------------------------------------------------------

def _build_rail_vessel_schedule(
    terminal_name: str, inputs: Dict[str, Any],
) -> List[dict]:
    """Trains + vessel calls. Inputs:
      * ``train_consist_plan``: optional pl.DataFrame (omit for vessel-only).
      * ``vessel_schedule``: optional pl.DataFrame or path; defaults to
        the bundled sample ``vessel_call_list.csv``.
    """
    train_consist_plan = inputs.get("train_consist_plan")
    train_entries: List[dict] = []
    if train_consist_plan is not None:
        train_entries = utilities.build_train_timetable(
            train_consist_plan, terminal_name, as_dicts=True,
        )
        for entry in train_entries:
            entry["_kind"] = "train"

    vessel_df = _resolve_table(
        inputs.get("vessel_schedule"),
        utilities.resources_root() / "vessel_call_list.csv",
    )
    vessel_entries = utilities.build_vessel_schedule(
        vessel_df, terminal_name, as_dicts=True,
    )
    for e in vessel_entries:
        e["_kind"] = "vessel"

    combined = train_entries + vessel_entries
    combined.sort(key=lambda e: float(e.get("arrival_time") or 0.0))
    return combined


def _rail_vessel_process_arrival(env, terminal, entry: dict):
    kind = entry.get("_kind")
    if kind == "train":
        yield from process_train_arrival(env, terminal, entry)
    elif kind == "vessel":
        yield from process_vessel_arrival(env, terminal, entry)
    else:
        raise ValueError(f"rail_vessel: unknown _kind {kind!r}")


register_mode(TerminalMode(
    name="rail_vessel",
    process_arrival=_rail_vessel_process_arrival,
    build_schedule=_build_rail_vessel_schedule,
    description=(
        "Rail<->vessel terminal: trains and vessels both deliver/receive "
        "containers; the stack mediates the exchange. No drayage gate."
    ),
    resource_specs=specs.RAIL_VESSEL_SPECS,
    event_types=_RAIL_VESSEL_EVENT_TYPES,
    container_id_pattern=_TRAIN_ID_PATTERN,
))


# ---------------------------------------------------------------------------
# vessel_truck
# ---------------------------------------------------------------------------

def _build_vessel_truck_schedule(
    terminal_name: str, inputs: Dict[str, Any],
) -> List[dict]:
    """Vessel calls + drayage trucks. Inputs:
      * ``vessel_schedule``: optional pl.DataFrame or path; defaults to
        the bundled sample ``vessel_call_list.csv``.
      * ``drayage_schedule``: optional pl.DataFrame or path; defaults to
        the bundled sample ``drayage_schedule.csv``.
    """
    vessel_df = _resolve_table(
        inputs.get("vessel_schedule"),
        utilities.resources_root() / "vessel_call_list.csv",
    )
    vessel_entries = utilities.build_vessel_schedule(
        vessel_df, terminal_name, as_dicts=True,
    )
    for e in vessel_entries:
        e["_kind"] = "vessel"

    drayage_df = _resolve_table(
        inputs.get("drayage_schedule"),
        utilities.resources_root() / "drayage_schedule.csv",
    )
    drayage_entries = utilities.build_drayage_schedule(
        drayage_df, terminal_name, as_dicts=True,
    )
    for e in drayage_entries:
        e["_kind"] = "drayage"

    combined = vessel_entries + drayage_entries
    combined.sort(key=lambda e: float(e.get("arrival_time") or 0.0))
    return combined


def _vessel_truck_process_arrival(env, terminal, entry: dict):
    kind = entry.get("_kind")
    if kind == "vessel":
        yield from process_vessel_arrival(env, terminal, entry)
    elif kind == "drayage":
        yield from process_drayage_arrival(env, terminal, entry)
    else:
        raise ValueError(f"vessel_truck: unknown _kind {kind!r}")


register_mode(TerminalMode(
    name="vessel_truck",
    process_arrival=_vessel_truck_process_arrival,
    build_schedule=_build_vessel_truck_schedule,
    description=(
        "Vessel<->truck terminal: vessels deliver/receive containers; "
        "drayage trucks deliver/receive at the gate. All containers route "
        "through the main container stack."
    ),
    resource_specs=specs.VESSEL_TRUCK_SPECS,
    event_types=_VESSEL_TRUCK_EVENT_TYPES,
    container_id_pattern=_VESSEL_ID_PATTERN,
))


if __name__ == "__main__":
    consist_plan = (pl.read_csv(utilities.package_root() / 'resources' / 'train_consist_plan.csv')
        .with_columns(pl.lit("Intermodal").alias("Train_Type"))
    )
    container_data, resource_log_df, terminal_obj = run_terminal_simulation(
        modes=["truck_rail"],
        terminal="Allouez",
        inputs={"truck_rail": {"train_consist_plan": consist_plan}},
    )

