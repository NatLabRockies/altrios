"""Freight catalog python helpers (registered via :func:`@register`).

Imported by :func:`altrios.workflow_engine.load_catalog` when a catalog
declares ``python_module: altrios.lifts.python_helpers``. Module-level
``@register(...)`` decorators populate the workflow-engine callable
registry; the freight catalog YAML then references the registered
names from ``schedule_mappings``, ``state_init``, and ``python:`` step
``call:`` parameters.

This module owns the entire freight Python surface that the YAML
catalog calls into:

* ``state_init`` (``freight.build_freight_state``) -- seeds RNG,
  builds the ResourceSpec union, attaches pools / event buffers /
  distance table to ``state``.
* Schedule builders -- ``freight.build_train_schedule``,
  ``freight.build_drayage_schedule_{synth,csv}``,
  ``freight.build_vessel_schedule``.
* Per-arrival ``python:`` escape hatches called from the fine-grained
  YAML graphs in ``catalog.yaml`` (e.g. ``unload_one_ic``,
  ``load_one_oc``, ``vessel_drain_unload``, ``vessel_drain_load``,
  ``drayage_dropoff``, ``drayage_pickup``). These compose the
  irreducibly stateful pieces (SimPy generators for stack-in/stack-out
  /yard_tractor_haul, the STS-crane parallel worker pattern, the
  greedy schedule matcher).
* ``assemble_outputs`` -- post-run helper that materializes the
  freight ``container_data`` / ``resource_log`` DataFrames from a
  :class:`RunResult` (called by demos and the smoke script after
  :func:`run_site`).
"""
from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import polars as pl
import simpy

from altrios.lifts import consumption, utilities
from altrios.lifts.classes import container, loggingLevel, truck
from altrios.lifts.consumption import (
    CO2_KG_PER_UNIT,
    _record_stack_lift_consumption,
    _record_trip_consumption,
    consumption_records,
)
from altrios.lifts.distances import calculate_distances
from altrios.lifts.specs import (
    RAIL_VESSEL_SPECS,
    TRUCK_RAIL_SPECS,
    VESSEL_TRUCK_SPECS,
)
from altrios.lifts.yard_flow import stack_in, stack_out, yard_tractor_haul
from altrios.workflow_engine import build_state_from_specs
from altrios.workflow_engine.registry import register


# Container-event surface for each mode. Looked up by
# :func:`assemble_outputs` so the wide-form pivot has a stable column
# order across runs (and so post-process consumers know which events to
# expect for a given mode).
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

    Responsibilities:

    1. Seed Python's stdlib :mod:`random` module so per-run draws are
       reproducible at the pinned smoke baselines (the freight
       helpers call ``random.uniform``/``random.random`` directly).
    2. Reset the module-level ``consumption_records`` buffer so repeat
       runs in the same process produce clean output.
    3. Attach a fresh ``container_events: list`` to ``state`` (the
       freight helpers call :func:`utilities.record_container_event`,
       which appends to this attribute).
    4. Attach a fresh ``loaded_ocs_by_train: dict`` so the train graph's
       load/depart steps can hand off OC labels.
    5. Build the union of every freight mode's :class:`ResourceSpec`
       set, instantiate the SimPy pools, and attach each to ``state``.
       Done here (rather than via the catalog YAML's ``resources:``
       field) because the existing freight :class:`ResourceSpec`
       callables -- capacity, init_items, partition_by -- are Python
       functions that the engine's YAML resource-spec form doesn't
       accept directly.
    6. Set the freight logging threshold via
       :func:`utilities.set_log_level` so basic-level log lines from
       the catalog helpers fire.
    7. Compute and attach the yard-distance table to
       ``state.distances`` so :func:`yard_flow.yard_tractor_haul` can
       read it without re-running the geometry.

    The function is registered under ``freight.build_freight_state``
    and named in ``catalog.schedule_mappings.state_init``.
    """
    import random
    random.seed(42)

    consumption_records.clear()

    # Defensive on second-run reuse: a SimpleNamespace.__init__ already
    # produced a fresh state, but resetting these here keeps the
    # contract explicit.
    state.container_events = []
    state.loaded_ocs_by_train = {}

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

    # Freight log threshold (consumed by utilities.log; the helpers
    # call utilities.log directly now that the adapter is gone).
    utilities.set_log_level(loggingLevel.BASIC)

    # Yard-distance table. Cached on state so the hot-path yard_flow
    # helpers can read it without recomputing.
    state.distances = calculate_distances(
        config=config, config_path=None, actual_railcars=None,
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
    ``Train_Type`` to ``"Intermodal"`` so freight-parity baselines
    line up. **Callers that pass a real DataFrame are NOT subject to
    this override.**
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
    each train; see :func:`_synthesize_drayage_from_trains`).
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

    # No explicit drayage schedule: synthesize from trains (one
    # dropoff per OC just before each train, one pickup per IC just
    # after; see ``_synthesize_drayage_from_trains``).
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
    plan to synthesize from, so the CSV fallback IS the default."""
    df = _resolve_table(schedule, _resources_root() / "drayage_schedule.csv")
    entries = utilities.build_drayage_schedule(df, terminal_name, as_dicts=True)
    return [
        {**e, "kind": "drayage", "id": f"drayage-{e['truck_id']}"}
        for e in entries
    ]


# ``freight.build_drayage_schedule`` is a convenience alias dispatching
# to ``_synth`` (the truck_rail-style default). New catalogs should
# pick one of ``_synth`` or ``_csv`` explicitly to make the data source
# obvious at the call site.
@register("freight.build_drayage_schedule")
def build_drayage_schedule(
    *, schedule: Any, terminal_name: str = "Allouez",
    env=None, state=None, config=None, layout=None, rng=None,
) -> list[dict]:
    """Default drayage builder. Delegates to
    :func:`build_drayage_schedule_synth` (synthesize-from-trains)."""
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
    IC just after. Truck-id partitioning is chosen so train-rail
    baselines match the pinned smoke values bit-for-bit."""
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


# ---- Inlined drayage / vessel helpers ------------------------------
#
# Small helpers that have no analogue in the workflow_engine primitive
# set (they compose multiple SimPy events that share local state).
# Called from the decomposed YAML graphs through this module:
#
#   * ``_truck_factory`` — eager truck construction inside
#     ``setup_drayage_arrival`` (pins the RNG draw position so smoke
#     baselines stay stable).
#   * ``_gate_in`` / ``_gate_out`` / ``_drayage_zone_travel`` —
#     composed by ``drayage_dropoff`` / ``drayage_pickup``.
#   * ``_sts_unload_worker`` / ``_sts_load_worker`` — spawned per STS
#     crane in ``vessel_drain_unload`` / ``vessel_drain_load``.


def _truck_factory(truck_id: int, config):
    """Build one drayage truck object respecting the configured
    diesel/electric mix. Used because the drayage schedule does not
    pre-allocate truck objects."""
    diesel = random.random() < config["truck_diesel_percentage"]
    return truck(
        type="Diesel" if diesel else "Electric",
        id=truck_id,
        train_id=0,
    )


def _drayage_zone_travel(env, config, label: str):
    """Placeholder timed move between gate and stack zone."""
    travel_time = config["truck_ingate_time"] + random.uniform(
        0, config["truck_ingate_time_dev"]
    )
    yield env.timeout(travel_time)
    return travel_time


def _gate_in(env, state, config, truck_obj):
    req = state.in_gates.request()
    yield req
    travel_time = config["truck_ingate_time"] + random.uniform(
        0, config["truck_ingate_time_dev"]
    )
    yield env.timeout(travel_time)
    state.in_gates.release(req)
    utilities.record_container_event(
        state, f"DrayageTruck-{truck_obj.id}", "drayage_gate_in", env.now,
    )
    _record_trip_consumption(
        getattr(state, "output", None),
        config["energy_use"], truck_obj, "truck", "loaded",
        getattr(truck_obj, "train_id", ""), "", "drayage_gate_in",
        travel_time, env.now,
    )


def _gate_out(env, state, config, truck_obj, container_obj=None):
    req = state.out_gates.request()
    yield req
    travel_time = config["truck_outgate_time"] + random.uniform(
        0, config["truck_outgate_time_dev"]
    )
    yield env.timeout(travel_time)
    state.out_gates.release(req)
    container_label = (
        container_obj.to_string() if container_obj is not None
        else f"DrayageTruck-{truck_obj.id}"
    )
    utilities.record_container_event(
        state, container_label, "drayage_gate_out", env.now,
    )
    _record_trip_consumption(
        getattr(state, "output", None),
        config["energy_use"], truck_obj, "truck",
        "loaded" if container_obj is not None else "empty",
        getattr(truck_obj, "train_id", ""),
        container_obj.to_string() if container_obj is not None else "",
        "drayage_gate_out",
        travel_time, env.now,
    )


def _sts_unload_worker(env, state, config, berth_id: int, vessel_id: int,
                       ic_queue: simpy.Store):
    """One STS crane drains ICs from ``ic_queue`` until empty."""
    sts_pool = state.sts_cranes_by_berth[berth_id]
    sts_obj = yield sts_pool.get()
    try:
        while ic_queue.items:
            ic = yield ic_queue.get()
            lift_time = (
                config["containers_per_crane_move_mean"]
                + random.uniform(0, config["crane_move_dev_time"])
            )
            yield env.timeout(lift_time)
            utilities.record_container_event(
                state, ic, "sts_unload", env.now,
            )
            _record_stack_lift_consumption(
                getattr(state, "output", None),
                config["energy_use"], sts_obj, "sts_crane", status="loaded",
                train_id=vessel_id,
                container_id=ic.to_string(),
                event_type="sts_unload",
                env_now=env.now, zone="berth",
            )
            yield env.process(stack_in(env, state, config, ic, source_chassis=None))
    finally:
        yield sts_pool.put(sts_obj)


def _sts_load_worker(env, state, config, berth_id: int, vessel_id: int,
                     oc_remaining: list):
    """One STS crane loads OCs onto the vessel until ``oc_remaining[0]``
    decrements to zero. ``oc_remaining`` is a single-element list used
    as a mutable counter shared across parallel STS workers."""
    sts_pool = state.sts_cranes_by_berth[berth_id]
    sts_obj = yield sts_pool.get()
    try:
        while oc_remaining[0] > 0:
            oc_remaining[0] -= 1
            oc = yield env.process(stack_out(env, state, config, container_obj=None))
            lift_time = (
                config["containers_per_crane_move_mean"]
                + random.uniform(0, config["crane_move_dev_time"])
            )
            yield env.timeout(lift_time)
            utilities.record_container_event(
                state, oc, "sts_load", env.now,
            )
            _record_stack_lift_consumption(
                getattr(state, "output", None),
                config["energy_use"], sts_obj, "sts_crane", status="loaded",
                train_id=vessel_id,
                container_id=oc.to_string(),
                event_type="sts_load",
                env_now=env.now, zone="berth",
            )
    finally:
        yield sts_pool.put(sts_obj)


# ---- train-arrival decomposition -----------------------------------
#
# The train-arrival graph in ``catalog.yaml`` orchestrates track
# acquisition, the IC/OC fork-join, and arrival/depart events through
# workflow_engine primitives. The inner unload/load body remains a
# ``python:`` escape hatch because (a) it composes the yard_flow
# helpers (stack_in/stack_out/yard_tractor_haul), each its own SimPy
# generator, and (b) the lift_time random.uniform draw is interleaved
# with consumption recording in a way that's clearer in Python than
# YAML.


@register("freight.setup_train_arrival")
def setup_train_arrival(*, env, entity) -> SimpleNamespace:
    """Normalize a ``train``-kind entity into a SimpleNamespace of
    typed fields the train-arrival graph downstream steps consume via
    ``bindings.meta.<field>`` (asteval can't call ``int``/``float``/
    ``list``/``range``, so type coercion has to happen in Python)."""
    attrs = dict(entity.attrs) if hasattr(entity, "attrs") else dict(vars(entity))
    train_id = int(attrs.get("train_id") or 0)
    ic_count = int(attrs.get("full_cars") or 0)
    oc_count = int(attrs.get("oc_number") or 0)
    arrival_time = float(attrs.get("arrival_time") or 0.0)
    departure_time = float(attrs.get("departure_time") or 0.0)
    return SimpleNamespace(
        train_id=train_id,
        ic_count=ic_count,
        oc_count=oc_count,
        arrival_time=arrival_time,
        departure_time=departure_time,
        ic_ids=list(range(1, ic_count + 1)),
        oc_ids=list(range(oc_count)),
    )


@register("freight.record_train_arrival_expected")
def record_train_arrival_expected(*, env, state, meta) -> None:
    """Pre-record ``train_arrival_expected`` for every IC at the
    train's scheduled arrival time."""
    for ic_id in meta.ic_ids:
        ic_label = container(type="Inbound", id=ic_id, train_id=meta.train_id)
        utilities.record_container_event(
            state, ic_label, "train_arrival_expected", meta.arrival_time,
        )


@register("freight.log_train_on_track")
def log_train_on_track(*, env, meta, track_id) -> None:
    """One-liner log statement so ``--log_level basic`` output
    documents each train being placed on its assigned track."""
    utilities.log(
        loggingLevel.BASIC,
        f"Time {env.now:.3f}: Train {meta.train_id} on track {track_id} "
        f"(IC={meta.ic_count}, OC={meta.oc_count}).",
    )


@register("freight.unload_one_ic")
def unload_one_ic(*, env, state, config, track_id, train_id, ic_id):
    """Lift one IC off the train and route it to the stack.

    The ``train_arrival_actual`` container-event is recorded here so
    the spawning ``loop parallel:true`` step sees no Python side
    effects.
    """
    ic = container(type="Inbound", id=int(ic_id), train_id=int(train_id))
    utilities.record_container_event(
        state, ic, "train_arrival_actual", env.now,
    )

    rtg_pool = state.rail_track_rtgs_by_track[int(track_id)]
    rtg_obj = yield rtg_pool.get()
    try:
        lift_time = (
            config["containers_per_crane_move_mean"]
            + random.uniform(0, config["crane_move_dev_time"])
        )
        yield env.timeout(lift_time)
        utilities.record_container_event(
            state, ic, "rail_track_rtg_unload", env.now,
        )
        _record_stack_lift_consumption(
            getattr(state, "output", None),
            config["energy_use"], rtg_obj, "rail_track_rtg", status="loaded",
            train_id=int(train_id), container_id=ic.to_string(),
            event_type="rail_track_rtg_unload", env_now=env.now,
            zone="track",
        )
    finally:
        yield rtg_pool.put(rtg_obj)

    yield env.process(yard_tractor_haul(
        env, state, config, state.rail_yard_tractors,
        ic, from_zone="rail", to_zone="stack",
    ))
    yield env.process(stack_in(env, state, config, ic, source_chassis=None))


@register("freight.load_one_oc")
def load_one_oc(*, env, state, config, track_id, train_id):
    """Pull one OC off the stack and load it onto the train.

    The loaded OC is appended to
    ``state.loaded_ocs_by_train[train_id]`` so the post-load
    ``freight.record_train_depart_events`` step can record the
    ``train_depart`` row for each OC at the actual departure time.
    """
    oc = yield env.process(stack_out(env, state, config, container_obj=None))

    yield env.process(yard_tractor_haul(
        env, state, config, state.rail_yard_tractors,
        oc, from_zone="stack", to_zone="rail",
    ))

    rtg_pool = state.rail_track_rtgs_by_track[int(track_id)]
    rtg_obj = yield rtg_pool.get()
    try:
        lift_time = (
            config["containers_per_crane_move_mean"]
            + random.uniform(0, config["crane_move_dev_time"])
        )
        yield env.timeout(lift_time)
        utilities.record_container_event(
            state, oc, "rail_track_rtg_load", env.now,
        )
        _record_stack_lift_consumption(
            getattr(state, "output", None),
            config["energy_use"], rtg_obj, "rail_track_rtg", status="loaded",
            train_id=int(train_id), container_id=oc.to_string(),
            event_type="rail_track_rtg_load", env_now=env.now,
            zone="track",
        )
    finally:
        yield rtg_pool.put(rtg_obj)

    tid = int(train_id)
    state.loaded_ocs_by_train.setdefault(tid, []).append(oc)


@register("freight.record_train_depart_events")
def record_train_depart_events(*, env, state, train_id) -> None:
    """Record ``train_depart`` for the train and for every OC the load
    phase placed on it. The OCs come from
    ``state.loaded_ocs_by_train[train_id]``, populated by
    ``freight.load_one_oc``. The bucket is cleared after to avoid
    leaking across runs that share the same state.
    """
    tid = int(train_id)
    utilities.record_container_event(
        state, f"Train-{tid}", "train_depart", env.now,
    )
    loaded = state.loaded_ocs_by_train.get(tid, [])
    for oc in loaded:
        utilities.record_container_event(
            state, oc, "train_depart", env.now,
        )
    state.loaded_ocs_by_train.pop(tid, None)


# ---- drayage-arrival decomposition ---------------------------------
#
# Same approach as the train graph: lift wait/branch into YAML so the
# dropoff-vs-pickup decision is visible, but keep each branch body as
# a python: escape hatch. ``setup_drayage_arrival`` also constructs
# the truck_obj here so the ``random`` draw position stays pinned for
# smoke-baseline stability.


@register("freight.setup_drayage_arrival")
def setup_drayage_arrival(*, env, entity, config) -> SimpleNamespace:
    """Normalize a ``drayage``-kind entity into a SimpleNamespace and
    construct the drayage truck object eagerly. The truck-factory call
    happens here (before any ``yield``) so the ``random.random()``
    draw position is locked, keeping pinned smoke baselines stable."""
    attrs = dict(entity.attrs) if hasattr(entity, "attrs") else dict(vars(entity))
    truck_id = int(attrs["truck_id"])
    action = str(attrs["action"])
    arrival_time = float(attrs.get("arrival_time") or 0.0)
    container_id = attrs.get("container_id")
    truck_obj = _truck_factory(truck_id, config)
    return SimpleNamespace(
        truck_id=truck_id,
        truck_obj=truck_obj,
        action=action,
        arrival_time=arrival_time,
        container_id=container_id,
    )


@register("freight.drayage_dropoff")
def drayage_dropoff(*, env, state, config, meta):
    """Dropoff branch of the drayage flow: truck brings an export
    container into the terminal, stack-in onto the main stack, exits
    empty."""
    oc = container(type="Outbound", id=meta.truck_id, train_id=0)
    if meta.container_id:
        utilities.record_container_event(
            state, oc, f"external_id:{meta.container_id}", env.now,
        )
    utilities.record_container_event(state, oc, "drayage_arrival", env.now)
    yield env.process(_gate_in(env, state, config, meta.truck_obj))
    yield env.process(_drayage_zone_travel(env, config, "to_stack"))
    yield env.process(stack_in(env, state, config, oc, source_chassis=None))
    yield env.process(_gate_out(env, state, config, meta.truck_obj, container_obj=None))


@register("freight.drayage_pickup")
def drayage_pickup(*, env, state, config, meta):
    """Pickup branch of the drayage flow: empty truck claims a
    container off the stack and exits loaded."""
    utilities.record_container_event(
        state, f"DrayageTruck-{meta.truck_id}", "drayage_arrival", env.now,
    )
    yield env.process(_gate_in(env, state, config, meta.truck_obj))
    yield env.process(_drayage_zone_travel(env, config, "to_stack"))
    ic = yield env.process(stack_out(env, state, config, container_obj=None))
    yield env.process(_gate_out(env, state, config, meta.truck_obj, container_obj=ic))


# ---- vessel arrival decomposition --------------------------------


@register("freight.setup_vessel_arrival")
def setup_vessel_arrival(*, env, entity):
    """Normalise one vessel-arrival schedule entry into a meta
    SimpleNamespace consumed by the rest of the vessel graph."""
    attrs = dict(entity.attrs) if hasattr(entity, "attrs") else dict(vars(entity))
    return SimpleNamespace(
        vessel_id=int(attrs["vessel_id"]),
        vessel_name=attrs.get("vessel_name", attrs["vessel_id"]),
        arrival_time=float(attrs["arrival_time"]),
        departure_time=float(attrs["departure_time"]),
        ic_count=int(attrs["inbound_containers"]),
        oc_count=int(attrs["outbound_containers"]),
    )


@register("freight.vessel_pre_record_expected")
def vessel_pre_record_expected(*, env, state, meta):
    """Records ``vessel_arrival_expected`` for each IC at the scheduled
    arrival time -- runs before the berth-acquire wait so the
    "expected" timestamps are pinned to the scheduled arrival rather
    than the actual one."""
    for ic_id in range(1, meta.ic_count + 1):
        ic = container(type="Inbound", id=ic_id, train_id=meta.vessel_id)
        utilities.record_container_event(
            state, ic, "vessel_arrival_expected", meta.arrival_time,
        )


@register("freight.prepare_vessel_berth_ctx")
def prepare_vessel_berth_ctx(*, env, state, meta):
    """Once a berth has been acquired, pick a per-berth STS pool key,
    stage the ic_queue, record ``vessel_arrival_actual`` for each IC,
    and emit the basic berth-arrived log line. Returns a SimpleNamespace
    consumed by the unload/load drain helpers."""
    by_berth = state.sts_cranes_by_berth
    berth_id = max(by_berth.keys(), key=lambda k: (len(by_berth[k].items), -k))
    utilities.log(
        loggingLevel.BASIC,
        f"[Vessel] {meta.vessel_name} berth_id={berth_id} arrived at "
        f"{env.now:.3f}",
    )

    ic_queue: simpy.Store = simpy.Store(env)
    for ic_id in range(1, meta.ic_count + 1):
        ic = container(type="Inbound", id=ic_id, train_id=meta.vessel_id)
        ic_queue.put(ic)
        utilities.record_container_event(
            state, ic, "vessel_arrival_actual", env.now,
        )

    return SimpleNamespace(
        berth_id=berth_id,
        ic_queue=ic_queue,
        sts_per_berth=by_berth[berth_id].capacity,
        oc_remaining=[meta.oc_count],
    )


@register("freight.vessel_drain_unload")
def vessel_drain_unload(*, env, state, config, meta, berth_ctx):
    """Spawn one STS unload worker per crane slot at this berth and wait
    for all of them to drain the per-vessel ic_queue."""
    procs = [
        env.process(
            _sts_unload_worker(
                env, state, config, berth_ctx.berth_id, meta.vessel_id,
                berth_ctx.ic_queue,
            )
        )
        for _ in range(berth_ctx.sts_per_berth)
    ]
    yield simpy.events.AllOf(env, procs)
    utilities.log(
        loggingLevel.BASIC,
        f"[Vessel] {meta.vessel_id} discharged {meta.ic_count} ICs at "
        f"{env.now:.3f}",
    )


@register("freight.vessel_drain_load")
def vessel_drain_load(*, env, state, config, meta, berth_ctx):
    """Spawn one STS load worker per crane slot and wait for all of them
    to dequeue ``oc_count`` OCs off the stack. No-op when
    ``meta.oc_count == 0`` (so the caller can skip the conditional in
    YAML)."""
    if meta.oc_count <= 0:
        return
        yield  # unreachable; keeps this a generator for the engine
    procs = [
        env.process(
            _sts_load_worker(
                env, state, config, berth_ctx.berth_id, meta.vessel_id,
                berth_ctx.oc_remaining,
            )
        )
        for _ in range(berth_ctx.sts_per_berth)
    ]
    yield simpy.events.AllOf(env, procs)
    utilities.log(
        loggingLevel.BASIC,
        f"[Vessel] {meta.vessel_id} loaded {meta.oc_count} OCs at "
        f"{env.now:.3f}",
    )


@register("freight.record_vessel_depart")
def record_vessel_depart(*, env, state, vessel_id):
    """Record the final ``vessel_depart`` event for ``Vessel-{id}``."""
    utilities.record_container_event(
        state, f"Vessel-{int(vessel_id)}", "vessel_depart", env.now,
    )


# ---- output assembly -----------------------------------------------


def assemble_outputs(
    result, *, mode_name: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """End-of-run output assembly for B-phase freight runs.

    The runner returns a :class:`RunResult` whose ``output``
    :class:`OutputCollector` receives the consumption rows written by
    the dual-write freight Python helpers. The container-events buffer
    is also dual-written, both to ``state.container_events`` and to
    the collector's ``event_log``. This helper pivots both buffers
    into the ``(container_data, resource_log)`` shape the freight
    demos and tests consume.

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
        Two polars DataFrames.
    """
    event_types = list(EVENT_TYPES_BY_MODE[mode_name])

    # Container events are dual-written to ``state.container_events``
    # AND the engine's ``result.output.event_log``. Prefer the
    # collector (authoritative for the run_site path); fall back to
    # the state buffer if the collector is empty.
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

    # Consumption rows are dual-written to the module-level
    # ``consumption_records`` buffer AND the engine's
    # ``result.output.consumption_log``. Prefer the collector
    # (authoritative for the run_site path); fall back to the module
    # buffer if the collector is empty (defensive).
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
    # extraction) — inlined here rather than as a separate registered
    # post-process callable because the catalog post_process hook
    # isn't yet plumbed through the runner.
    if mode_name == "truck_rail":
        container_data = _truck_rail_post_process(container_data)

    return container_data, resource_log


def _truck_rail_post_process(container_data: pl.DataFrame) -> pl.DataFrame:
    """Truck_rail-specific derived columns.

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
