"""Vessel-side flow for ``rail_vessel`` and ``vessel_truck``.

A vessel call from :func:`altrios.lifts.utilities.build_vessel_schedule` has::

    {"vessel_id": int, "vessel_name": str, "arrival_time": float,
     "departure_time": float, "inbound_containers": int,
     "outbound_containers": int}

The Phase 1 sequence for one vessel call:

1. Wait until ``arrival_time``.
2. Acquire one of the ``berths`` slots.
3. Drain the ``inbound_containers`` count: per-berth STS workers run in
   parallel via ``simpy.events.AllOf``. Each STS worker:
      a. Pulls one IC off a vessel-side IC queue.
      b. :func:`yard_flow.stack_in` (the STS itself acts as the stack lift
         in Phase 1; routing the chassis traversal via
         :func:`yard_flow.yard_tractor_haul` between berth and stack is a
         Phase 1G refinement once the spec-built state is in place).
4. Wait for outbound containers to be ready on the stack (Phase 1 simply
   takes the first ``outbound_containers`` items off the stack; Phase 1G
   may add per-vessel OC queues keyed by vessel_id).
5. Per-berth STS workers run the load loop in parallel via ``AllOf``: each
   :func:`yard_flow.stack_out` pulls one OC off the stack and the worker
   loads it onto the vessel.
6. Wait until ``departure_time`` (vessel does not depart early).
7. Release the berth.

Notes
-----
* Phase 1 records ICs as ``container(type='Inbound', id=ic_id, train_id=vessel_id)``
  and OCs as ``container(type='Outbound', id=oc_id, train_id=vessel_id)``.
  Reusing the ``train_id`` slot keeps to_string() compatible with the
  legacy container-event pivot; this becomes ``arrival_id`` in Phase 2.
* Container-event labels follow ``vessel_arrival_expected`` /
  ``vessel_arrival_actual`` / ``vessel_depart`` / ``sts_unload`` /
  ``sts_load`` patterns symmetric with the rail-side event vocabulary.
"""
from __future__ import annotations

import random
from typing import Any

import simpy

from altrios.lifts import utilities
from altrios.lifts.classes import container, loggingLevel
from altrios.lifts.consumption import _record_stack_lift_consumption
from altrios.lifts.yard_flow import stack_in, stack_out


def _sts_unload_worker(env, terminal, berth_id: int, vessel_id: int,
                       ic_queue: simpy.Store):
    """One STS crane at ``berth_id`` drains ICs from ``ic_queue`` until
    empty. Each IC is recorded as ``sts_unload`` then handed to the stack
    via :func:`stack_in`. The STS object itself is checked out of
    ``state.sts_cranes_by_berth[berth_id]`` for the lift duration so that
    multiple workers serialize on real crane availability."""
    state = terminal.state
    sts_pool = state.sts_cranes_by_berth[berth_id]
    sts_obj = yield sts_pool.get()
    try:
        while ic_queue.items:
            ic = yield ic_queue.get()
            lift_time = (
                terminal.CONTAINERS_PER_CRANE_MOVE_MEAN
                + random.uniform(0, terminal.CRANE_MOVE_DEV_TIME)
            )
            yield env.timeout(lift_time)
            utilities.record_container_event(
                terminal, ic, "sts_unload", env.now,
            )
            _record_stack_lift_consumption(
                getattr(terminal, "output", None),
                terminal, sts_obj, "sts_crane", status="loaded",
                train_id=vessel_id,
                container_id=ic.to_string(),
                event_type="sts_unload",
                env_now=env.now, zone="berth",
            )
            # Hand off to the stack. Inside stack_in we record stack_in +
            # the main_stack_rtg/top_pick energy entry.
            yield env.process(stack_in(env, terminal, ic, source_chassis=None))
    finally:
        yield sts_pool.put(sts_obj)


def _sts_load_worker(env, terminal, berth_id: int, vessel_id: int,
                     oc_remaining: list[int]):
    """One STS crane at ``berth_id`` loads OCs from the stack onto the
    vessel until ``oc_remaining[0]`` decrements to zero. ``oc_remaining``
    is a single-element list used as a mutable counter shared across the
    parallel STS workers for this berth call."""
    state = terminal.state
    sts_pool = state.sts_cranes_by_berth[berth_id]
    sts_obj = yield sts_pool.get()
    try:
        while oc_remaining[0] > 0:
            oc_remaining[0] -= 1
            oc = yield env.process(stack_out(env, terminal, container_obj=None))
            lift_time = (
                terminal.CONTAINERS_PER_CRANE_MOVE_MEAN
                + random.uniform(0, terminal.CRANE_MOVE_DEV_TIME)
            )
            yield env.timeout(lift_time)
            utilities.record_container_event(
                terminal, oc, "sts_load", env.now,
            )
            _record_stack_lift_consumption(
                getattr(terminal, "output", None),
                terminal, sts_obj, "sts_crane", status="loaded",
                train_id=vessel_id,
                container_id=oc.to_string(),
                event_type="sts_load",
                env_now=env.now, zone="berth",
            )
    finally:
        yield sts_pool.put(sts_obj)


def process_vessel_arrival(env, terminal, vessel_call: dict):
    """SimPy generator: one vessel berth call.

    Per the module docstring, the vessel:
      - waits until arrival_time,
      - acquires a berth,
      - discharges ``inbound_containers`` ICs (parallel STS workers),
      - loads ``outbound_containers`` OCs (parallel STS workers),
      - holds the berth until departure_time,
      - releases the berth.
    """
    state = terminal.state
    vessel_id = int(vessel_call["vessel_id"])
    arrival_time = float(vessel_call["arrival_time"])
    departure_time = float(vessel_call["departure_time"])
    ic_count = int(vessel_call["inbound_containers"])
    oc_count = int(vessel_call["outbound_containers"])

    # Record planned arrival/depart for each IC up front so the
    # post_process pivot can join container -> vessel id.
    for ic_id in range(1, ic_count + 1):
        ic_label = container(type="Inbound", id=ic_id, train_id=vessel_id)
        utilities.record_container_event(
            terminal, ic_label, "vessel_arrival_expected", arrival_time,
        )

    if env.now < arrival_time:
        yield env.timeout(arrival_time - env.now)

    # 1. Acquire a berth.
    berth_request = state.berths.request()
    yield berth_request
    try:
        # Berth_id can be derived in Phase 2 from a per-berth resource;
        # today we pick the lowest sts_cranes_by_berth key with idle
        # cranes (or just key 1 if all busy). This keeps the per-berth
        # crane store API stable while the berth ResourceSpec stays a
        # plain simpy.Resource.
        berth_id = _pick_berth(state)
        terminal.log(loggingLevel.BASIC,
                     f"[Vessel] {vessel_call.get('vessel_name', vessel_id)} "
                     f"berth_id={berth_id} arrived at {env.now:.3f}")

        # 2. Stage ICs in a per-vessel queue and spin up STS workers.
        ic_queue: simpy.Store = simpy.Store(env)
        for ic_id in range(1, ic_count + 1):
            ic = container(type="Inbound", id=ic_id, train_id=vessel_id)
            ic_queue.put(ic)
            utilities.record_container_event(
                terminal, ic, "vessel_arrival_actual", env.now,
            )
        sts_per_berth = len(state.sts_cranes_by_berth[berth_id].items) + \
            (state.sts_cranes_by_berth[berth_id].capacity
             - len(state.sts_cranes_by_berth[berth_id].items))
        # sts_per_berth above always equals capacity; spelled out to make
        # the "spawn one worker per STS slot" intent obvious.
        unload_procs = [
            env.process(_sts_unload_worker(env, terminal, berth_id, vessel_id, ic_queue))
            for _ in range(sts_per_berth)
        ]
        yield simpy.events.AllOf(env, unload_procs)
        terminal.log(loggingLevel.BASIC,
                     f"[Vessel] {vessel_id} discharged {ic_count} ICs at "
                     f"{env.now:.3f}")

        # 3. Load OCs. Phase 1 assumes the stack has enough OC items
        # already (placed there by other modes or pre-staging). If not,
        # the workers block on stack_out until items appear -- which
        # never happens in single-mode test runs, so call sites are
        # responsible for pre-seeding the stack with OC containers in
        # Phase 1F unit tests. Phase 1G's rail_vessel mode pre-stages
        # OCs from the rail arrival side.
        if oc_count > 0:
            oc_remaining = [oc_count]
            load_procs = [
                env.process(_sts_load_worker(env, terminal, berth_id, vessel_id, oc_remaining))
                for _ in range(sts_per_berth)
            ]
            yield simpy.events.AllOf(env, load_procs)
            terminal.log(loggingLevel.BASIC,
                         f"[Vessel] {vessel_id} loaded {oc_count} OCs at "
                         f"{env.now:.3f}")

        # 4. Hold the berth until departure_time, then release.
        if env.now < departure_time:
            yield env.timeout(departure_time - env.now)
        utilities.record_container_event(
            terminal, f"Vessel-{vessel_id}", "vessel_depart", env.now,
        )
    finally:
        state.berths.release(berth_request)


def _pick_berth(state) -> int:
    """Pick the berth_id whose STS pool currently has the most idle cranes
    (ties broken by lowest id). The Phase 1 ``berths`` spec is a flat
    ``simpy.Resource`` so it does not itself carry a berth id; this
    helper exists so the per-berth STS Stores can be addressed."""
    by_berth = state.sts_cranes_by_berth
    return max(by_berth.keys(), key=lambda k: (len(by_berth[k].items), -k))


def run_vessel_schedule(env, terminal, vessel_schedule):
    """Spawn one :func:`process_vessel_arrival` per call in the schedule.

    ``vessel_schedule`` is the list-of-dicts return of
    :func:`build_vessel_schedule` (or its DataFrame equivalent, iterated).
    """
    if hasattr(vessel_schedule, "iter_rows"):
        rows: Any = list(vessel_schedule.iter_rows(named=True))
    else:
        rows = list(vessel_schedule)
    for row in rows:
        env.process(process_vessel_arrival(env, terminal, row))
        yield env.timeout(0)
