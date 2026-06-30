"""Freight catalog python helpers (registered via :func:`@register`).

Imported by :func:`altrios.workflow_engine.load_catalog` when a catalog
declares ``python_module: altrios.lifts.python_helpers``. Module-level
``@register(...)`` decorators populate the workflow-engine callable
registry; the freight catalog YAML then references the registered
names from ``schedule_mappings``, ``state_init``, and ``python:`` step
``call:`` parameters.

**Phase 3 / Strategy B scope:** this module is the entire "freight
Python surface" the YAML catalog calls into. Helpers are thin wrappers
around existing freight code in :mod:`altrios.lifts.train_flow`,
:mod:`altrios.lifts.drayage_flow`, :mod:`altrios.lifts.vessel_flow`,
:mod:`altrios.lifts.utilities`, and :mod:`altrios.lifts.consumption`.
**No domain logic lives in this module under B** — it only adapts
signatures between the engine's calling conventions and the existing
helpers' calling conventions.

**Phase 3 / Strategy A removes most of this module:** once helpers are
refactored to take ``(env, state, config, output, ...)`` directly,
the arrival-wrapper layer and the adapter constructor go away. Only
the schedule builders and post-process callables survive.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import polars as pl

from altrios.lifts import consumption, drayage_flow, train_flow, utilities, vessel_flow
from altrios.lifts.classes import loggingLevel
from altrios.lifts.consumption import CO2_KG_PER_UNIT, consumption_records
from altrios.lifts.specs import (
    RAIL_VESSEL_SPECS,
    TRUCK_RAIL_SPECS,
    VESSEL_TRUCK_SPECS,
)
from altrios.lifts.terminal_adapter import TerminalAdapter
from altrios.workflow_engine import build_state_from_specs
from altrios.workflow_engine.registry import register


# Container-event surface for each mode. Mirrors the tuples in the legacy
# ``terminal_sim`` module; lifted here so the catalog can reference them
# at post-process time without import gymnastics.
_TRUCK_RAIL_EVENT_TYPES: tuple[str, ...] = (
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

_VESSEL_EVENT_TYPES: tuple[str, ...] = (
    "vessel_arrival_expected", "vessel_arrival_actual",
    "sts_unload",
    "main_stack_rtg_stack_in", "top_pick_stack_in", "stack_in",
    "main_stack_rtg_stack_out", "top_pick_stack_out", "stack_out",
    "sts_load",
    "vessel_depart",
)

_RAIL_VESSEL_EVENT_TYPES: tuple[str, ...] = tuple(
    dict.fromkeys((*_TRUCK_RAIL_EVENT_TYPES, *_VESSEL_EVENT_TYPES))
)

_VESSEL_TRUCK_EVENT_TYPES: tuple[str, ...] = tuple(
    dict.fromkeys((
        *_VESSEL_EVENT_TYPES,
        "drayage_arrival", "drayage_gate_in",
        "stack_in", "stack_out",
        "drayage_gate_out",
    ))
)

# Per-mode event-type surface keyed by catalog mode name. Looked up by
# :func:`assemble_outputs` when the user calls it after :func:`run_site`.
EVENT_TYPES_BY_MODE: Mapping[str, tuple[str, ...]] = {
    "truck_rail": _TRUCK_RAIL_EVENT_TYPES,
    "rail_vessel": _RAIL_VESSEL_EVENT_TYPES,
    "vessel_truck": _VESSEL_TRUCK_EVENT_TYPES,
}


# ---- state_init ----------------------------------------------------


@register("freight.build_freight_state")
def build_freight_state(
    *, env, state, config: Mapping[str, Any], layout=None,
) -> None:
    """Catalog ``state_init`` hook for the freight catalog.

    Responsibilities (B scope):

    1. Seed Python's stdlib :mod:`random` module with the value
       ``freight_constants.legacy_random_seed`` from config (default
       42, matching ``run_terminal_simulation``). The freight
       generators draw from ``random.uniform`` / ``random.expovariate``
       directly; under B, parity REQUIRES this seed reset. **Phase A
       moves generators to inject the RNG explicitly, after which the
       runner's :class:`numpy.random.Generator` is sole source of
       randomness and this branch is deleted.**
    2. Reset the module-level ``consumption_records`` buffer so repeat
       runs in the same process produce clean output.
    3. Attach a fresh ``container_events: list`` to ``state`` (the
       legacy freight helpers ``.append`` to this attribute directly
       via :func:`altrios.lifts.utilities.record_container_event`).
    4. Build the union of every freight mode's :class:`ResourceSpec`
       set, instantiate the SimPy pools, and attach each to ``state``.
       Done here (rather than via the catalog YAML's
       ``resources:`` field) because the existing freight
       :class:`ResourceSpec` callables — capacity, init_items,
       partition_by — are Python functions that the engine's
       YAML resource-spec form doesn't accept directly. **Phase A
       will translate these into YAML-form specs and delete this
       branch.** Until then, freight sites can't override pool
       capacities via ``resource_overrides``; they override via the
       site's ``config:`` block (which the legacy spec callables
       read from) just like the legacy runner.
    5. Build a :class:`TerminalAdapter` and attach it as
       ``state.terminal_adapter`` so per-arrival ``python:`` graphs
       can pass it to the unmodified freight generators.

    The function is registered under ``freight.build_freight_state``
    and named in ``catalog.schedule_mappings.state_init``.
    """
    import random
    random.seed(42)

    consumption_records.clear()

    if not hasattr(state, "container_events"):
        state.container_events = []
    else:
        # Defensive: a second run in the same process must not see
        # stale events. SimpleNamespace.__init__ already created a
        # fresh state object, so this branch only fires if a caller
        # reuses state — rare but cheap to guard.
        state.container_events = []

    # Build the union of every freight mode's specs (capacity callables
    # read from `config` so any site-level overrides on yard.track_number
    # etc. propagate naturally). Deduplicate by name.
    seen: set[str] = set()
    combined: list = []
    for bundle in (TRUCK_RAIL_SPECS, RAIL_VESSEL_SPECS, VESSEL_TRUCK_SPECS):
        for spec in bundle:
            if spec.name not in seen:
                seen.add(spec.name)
                combined.append(spec)

    pools = build_state_from_specs(env, combined, config, {})
    for name, primitive in pools.items():
        setattr(state, name, primitive)

    # The terminal_adapter goes last so it sees the fully-populated
    # state.
    state.terminal_adapter = TerminalAdapter(
        env=env,
        state=state,
        config=config,
        layout=layout,
        log_level=loggingLevel.BASIC,
    )


# ---- schedule builders ---------------------------------------------


def _resolve_table(value: Any, default_path: Path) -> pl.DataFrame:
    """Coerce ``value`` (DataFrame, path, ``None``) into a polars
    DataFrame. ``None`` falls back to ``default_path`` so the canonical
    sample data ships with the catalog."""
    if value is None:
        return pl.read_csv(default_path)
    if isinstance(value, pl.DataFrame):
        return value
    return pl.read_csv(value)


def _resources_root() -> Path:
    """Re-export ``utilities.resources_root()`` for legibility at
    call sites in this module."""
    return utilities.resources_root()


@register("freight.build_train_schedule")
def build_train_schedule(
    *, schedule: Any, terminal_name: str = "Allouez",
    env=None, state=None, config=None, layout=None, rng=None,
) -> list[dict]:
    """Thin wrapper over :func:`utilities.build_train_timetable`.

    ``schedule`` is the site's schedule value: a DataFrame, a path,
    or ``None``. When ``None``, falls back to the canonical
    ``train_consist_plan.csv`` AND force-overrides every row's
    ``Train_Type`` to ``"Intermodal"`` — matching the legacy demo's
    ``with_columns(pl.lit("Intermodal").alias("Train_Type"))`` trick
    so freight-parity baselines line up. **Callers that pass a real
    DataFrame are NOT subject to this override.**
    """
    if schedule is None:
        df = pl.read_csv(_resources_root() / "train_consist_plan.csv")
        df = df.with_columns(pl.lit("Intermodal").alias("Train_Type"))
    else:
        df = _resolve_table(schedule, _resources_root() / "train_consist_plan.csv")
    entries = utilities.build_train_timetable(df, terminal_name, as_dicts=True)
    return [
        {**e, "kind": "train", "id": f"train-{e['train_id']}"}
        for e in entries
    ]


@register("freight.build_drayage_schedule_synth")
def build_drayage_schedule_synth(
    *, schedule: Any, terminal_name: str = "Allouez",
    env=None, state=None, config=None, layout=None, rng=None,
) -> list[dict]:
    """Drayage builder used by ``truck_rail`` mode.

    When ``schedule is None``, **synthesizes** drayage events from the
    train consist plan (one dropoff before each train, one pickup after
    each train; matches the legacy
    ``_synthesize_drayage_from_trains`` default in ``terminal_sim``).
    When ``schedule`` is supplied, delegates to
    :func:`utilities.build_drayage_schedule`.
    """
    if schedule is not None:
        df = _resolve_table(schedule, _resources_root() / "drayage_schedule.csv")
        entries = utilities.build_drayage_schedule(df, terminal_name, as_dicts=True)
        return [
            {**e, "kind": "drayage", "id": f"drayage-{e['truck_id']}"}
            for e in entries
        ]

    # No explicit drayage schedule: synthesize from trains (matches
    # legacy ``_synthesize_drayage_from_trains`` default behavior).
    df = pl.read_csv(_resources_root() / "train_consist_plan.csv")
    df = df.with_columns(pl.lit("Intermodal").alias("Train_Type"))
    train_entries = utilities.build_train_timetable(df, terminal_name, as_dicts=True)
    return [
        {**e, "kind": "drayage", "id": f"drayage-{e['truck_id']}"}
        for e in _synthesize_drayage_from_trains(train_entries)
    ]


@register("freight.build_drayage_schedule_csv")
def build_drayage_schedule_csv(
    *, schedule: Any, terminal_name: str = "Allouez",
    env=None, state=None, config=None, layout=None, rng=None,
) -> list[dict]:
    """Drayage builder used by ``vessel_truck`` mode.

    Falls back to the canonical ``drayage_schedule.csv`` when
    ``schedule is None``. Vessel<->truck flows have no train consist
    plan to synthesize from, so the CSV fallback IS the legacy default
    (see ``terminal_sim._build_vessel_truck_schedule``)."""
    df = _resolve_table(schedule, _resources_root() / "drayage_schedule.csv")
    entries = utilities.build_drayage_schedule(df, terminal_name, as_dicts=True)
    return [
        {**e, "kind": "drayage", "id": f"drayage-{e['truck_id']}"}
        for e in entries
    ]


# ``freight.build_drayage_schedule`` is retained for catalogs that
# already reference the old name; it dispatches to ``_synth`` because
# the original (pre-B) behavior was the synthesize-from-trains path
# (terminal_sim's truck_rail-style default). New catalogs should pick
# one of ``_synth`` or ``_csv`` explicitly.
@register("freight.build_drayage_schedule")
def build_drayage_schedule(
    *, schedule: Any, terminal_name: str = "Allouez",
    env=None, state=None, config=None, layout=None, rng=None,
) -> list[dict]:
    """Default drayage builder. Delegates to
    :func:`build_drayage_schedule_synth`; kept registered under the
    bare name so the legacy-style catalog naming still works."""
    return build_drayage_schedule_synth(
        schedule=schedule, terminal_name=terminal_name,
        env=env, state=state, config=config, layout=layout, rng=rng,
    )


@register("freight.build_vessel_schedule")
def build_vessel_schedule(
    *, schedule: Any, terminal_name: str = "Allouez",
    env=None, state=None, config=None, layout=None, rng=None,
) -> list[dict]:
    """Thin wrapper over :func:`utilities.build_vessel_schedule`."""
    df = _resolve_table(schedule, _resources_root() / "vessel_call_list.csv")
    entries = utilities.build_vessel_schedule(df, terminal_name, as_dicts=True)
    return [
        {**e, "kind": "vessel", "id": f"vessel-{e['vessel_id']}"}
        for e in entries
    ]


def _synthesize_drayage_from_trains(
    train_entries: list[dict],
    pre_arrival_window_hr: float = 0.5,
    post_arrival_window_hr: float = 1.0,
) -> list[dict]:
    """One dropoff per OC just before each train arrival; one pickup per
    IC just after. Truck-id partitioning matches the legacy logic in
    :func:`terminal_sim._synthesize_drayage_from_trains` so train-rail
    parity holds bit-for-bit."""
    out: list[dict] = []
    for entry in train_entries:
        train_id = int(entry["train_id"])
        arrival = float(entry["arrival_time"])
        oc_n = int(entry.get("oc_number") or 0)
        ic_n = int(entry.get("full_cars") or 0)
        for i in range(1, oc_n + 1):
            out.append({
                "truck_id": train_id * 100000 + i,
                "arrival_time": max(0.0, arrival - pre_arrival_window_hr),
                "action": "dropoff",
                "container_id": None,
                "train_id": train_id,
            })
        for i in range(1, ic_n + 1):
            out.append({
                "truck_id": train_id * 100000 + 50000 + i,
                "arrival_time": arrival + post_arrival_window_hr,
                "action": "pickup",
                "container_id": None,
                "train_id": train_id,
            })
    return out


# ---- arrival wrappers ----------------------------------------------


def _entry_from_entity(entity) -> dict:
    """Reconstruct the entry-dict shape the legacy generators expect.

    The engine flattens an :class:`Entity` into a read-only
    SimpleNamespace before passing it to expressions (merging
    ``entity.attrs`` with top-level ``id``/``kind`` keys). The legacy
    generators receive a plain dict, so we ``vars()`` the view. Extra
    keys (``id``, ``kind``) are harmless — the generators read only
    the fields they know about."""
    if hasattr(entity, "attrs") and isinstance(getattr(entity, "attrs"), Mapping):
        return dict(entity.attrs)
    return dict(vars(entity))


@register("freight.process_train_arrival")
def process_train_arrival_yaml(*, env, state, entity):
    """Adapt the engine's ``python:`` calling convention onto
    :func:`train_flow.process_train_arrival`.

    The freight generator yields SimPy events; the engine's ``python``
    handler ``yield from``s any generator result, so simulated time
    inside the generator blocks the workflow correctly. Returns
    ``None`` (no value to bind)."""
    entry = _entry_from_entity(entity)
    yield from train_flow.process_train_arrival(env, state.terminal_adapter, entry)


@register("freight.process_drayage_arrival")
def process_drayage_arrival_yaml(*, env, state, entity):
    """Adapt onto :func:`drayage_flow.process_drayage_arrival`. See
    :func:`process_train_arrival_yaml` for the calling convention."""
    entry = _entry_from_entity(entity)
    yield from drayage_flow.process_drayage_arrival(env, state.terminal_adapter, entry)


@register("freight.process_vessel_arrival")
def process_vessel_arrival_yaml(*, env, state, entity):
    """Adapt onto :func:`vessel_flow.process_vessel_arrival`. See
    :func:`process_train_arrival_yaml` for the calling convention."""
    entry = _entry_from_entity(entity)
    yield from vessel_flow.process_vessel_arrival(env, state.terminal_adapter, entry)


# ---- output assembly -----------------------------------------------


def assemble_outputs(
    result, *, mode_name: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """End-of-run output assembly for B-phase freight runs.

    The runner returns a :class:`RunResult` whose ``output``
    :class:`OutputCollector` receives the consumption rows written by
    the (dual-write) freight Python helpers (Phase A.1). The
    container-events buffer is still on ``state.container_events``
    (legacy attribute; Phase A.2 migrates it onto the collector). This
    helper pivots both buffers into the same
    ``(container_data, resource_log)`` shape the legacy
    :func:`terminal_sim.run_terminal_simulation` returned, so demos
    and parity tests can compare apples to apples.

    Phase A.9 deletes the legacy ``run_terminal_simulation`` path and
    the dual-write fallback, at which point this helper becomes a
    direct ``result.output.to_freight_dataframes()`` call (or
    equivalent).

    Parameters
    ----------
    result
        The :class:`RunResult` returned by :func:`run_site`.
    mode_name
        ``"truck_rail"``, ``"rail_vessel"``, or ``"vessel_truck"``.
        Selects which event-type surface to backfill (so missing
        columns appear as null rather than silently absent).

    Returns
    -------
    (container_data, resource_log)
        Two polars DataFrames matching the legacy return shape.
    """
    event_types = list(EVENT_TYPES_BY_MODE[mode_name])

    # Container events: Phase A.2 dual-writes the legacy
    # ``state.container_events`` tuple list AND the engine's
    # ``result.output.event_log``. Prefer the collector (authoritative
    # for the run_site path); fall back to the legacy buffer if the
    # collector is empty.
    event_rows = list(result.output.event_log)
    if event_rows:
        # Collector rows are dicts; pivot expects long-form columns.
        events_long = pl.DataFrame(
            event_rows,
            schema={
                "container_id": pl.Utf8,
                "event_type": pl.Utf8,
                "timestamp": pl.Float64,
            },
        )
    else:
        container_events = list(getattr(result.state, "container_events", []) or [])
        events_long = pl.DataFrame(
            container_events,
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
                **{t: pl.Float64 for t in event_types},
            }
        )
    else:
        container_data = events_long.pivot(
            values="timestamp",
            index="container_id",
            on="event_type",
            aggregate_function="last",
        )

    missing = [t for t in event_types if t not in container_data.columns]
    if missing:
        container_data = container_data.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(t) for t in missing]
        )
    container_data = (
        container_data
        .sort("container_id")
        .select(pl.col("container_id"), pl.exclude("container_id"))
    )

    # Consumption rows: Phase A.1 dual-writes the legacy module-global
    # ``consumption_records`` AND the engine's
    # ``result.output.consumption_log``. Prefer the collector
    # (authoritative for the run_site path); fall back to the module
    # buffer if the collector is empty (defensive — should not happen
    # under A.1+).
    output_rows = list(result.output.consumption_log)
    if output_rows:
        consumption_rows = output_rows
    else:
        consumption_rows = list(consumption_records)

    resource_log = pl.DataFrame(
        consumption_rows,
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
        (
            pl.col("consumption_value")
            * pl.col("fuel_type").replace_strict(CO2_KG_PER_UNIT, default=float("nan"))
        ).alias("emissions(kgCO2)")
    )

    # Truck_rail derived columns (container_processing_time + train_id
    # extraction) — kept inline rather than as a separate registered
    # post-process callable because B doesn't yet plumb post-process
    # hooks through the runner. Phase A.9 moves this into a registered
    # callable invoked from catalog post_process.
    if mode_name == "truck_rail":
        container_data = _truck_rail_post_process(container_data)

    return container_data, resource_log


def _truck_rail_post_process(container_data: pl.DataFrame) -> pl.DataFrame:
    """Truck_rail-specific derived columns. Lifted verbatim from
    :func:`terminal_sim._truck_rail_post_process` (the resource_log
    branch was a no-op there too, so we drop it).

    Adds:
    - ``container_processing_time``: gate-out minus expected arrival
      (IC) OR rail load minus drayage arrival (OC).
    - ``train_arrival_actual_oc``: joined-in actual arrival time of
      the train the OC was loaded onto."""
    if container_data.height == 0:
        return container_data

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
    return container_data
