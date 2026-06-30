"""Drayage-truck flow for ``truck_rail`` and ``vessel_truck``.

A drayage event from :func:`altrios.lifts.utilities.build_drayage_schedule`
has the shape::

    {"truck_id": int, "arrival_time": float,
     "action": "dropoff" | "pickup",
     "container_id": str | None}

``dropoff`` means the truck brings an export container into the terminal
(it becomes an OC on the rail/vessel side). ``pickup`` means the truck
claims an import container from the terminal (it was an IC on the
rail/vessel side). Each drayage truck visits the stack directly --
there is no parking-slot intermediate.

The Phase 1 sequence for one truck:

1. Wait until ``arrival_time``.
2. Acquire an in-gate, drive through, release the in-gate.
3. *(dropoff only)* Optionally claim a road chassis from the
   ``road_chassis_pool`` (with probability
   ``yard_stack.road_chassis_claim_fraction``); otherwise the truck is
   assumed to bring its own chassis (the "bring" half of the
   bring-vs-claim split).
4. Drive into the stack zone (placeholder constant time).
5. *(dropoff)*  :func:`yard_flow.stack_in` lifts the container off the
   truck's chassis onto the stack.
   *(pickup)* :func:`yard_flow.stack_out` lifts the next container off
   the stack onto the truck's chassis.
6. Acquire an out-gate, drive out, release the out-gate, put the truck
   (and, if pickup, return the claimed chassis) back into their pools.
"""
from __future__ import annotations

import random
from typing import Any

from altrios.lifts import utilities
from altrios.lifts.classes import container, truck
from altrios.lifts.consumption import _record_trip_consumption
from altrios.lifts.yard_flow import stack_in, stack_out


def _truck_factory(truck_id: int, terminal) -> "truck":
    """Build one drayage truck object respecting the configured
    diesel/electric mix (``TRUCK_DIESEL_PERCENTAGE``). Used when the
    schedule does not pre-allocate truck objects (today's case)."""
    diesel = random.random() < terminal.TRUCK_DIESEL_PERCENTAGE
    return truck(
        type="Diesel" if diesel else "Electric",
        id=truck_id,
        train_id=0,
    )


def _drayage_zone_travel(env, terminal, label: str):
    """Placeholder timed move between gate and stack zone. Phase 1 uses a
    constant derived from the truck-gate timing; Phase 2 may sample from
    yard geometry."""
    travel_time = terminal.TRUCK_INGATE_TIME + random.uniform(
        0, terminal.TRUCK_INGATE_TIME_DEV
    )
    yield env.timeout(travel_time)
    return travel_time


def _gate_in(env, terminal, truck_obj):
    req = terminal.state.in_gates.request()
    yield req
    travel_time = terminal.TRUCK_INGATE_TIME + random.uniform(
        0, terminal.TRUCK_INGATE_TIME_DEV
    )
    yield env.timeout(travel_time)
    terminal.state.in_gates.release(req)
    utilities.record_container_event(
        terminal, f"DrayageTruck-{truck_obj.id}", "drayage_gate_in", env.now,
    )
    _record_trip_consumption(
        terminal, truck_obj, "truck", "loaded",
        getattr(truck_obj, "train_id", ""), "", "drayage_gate_in",
        travel_time, env.now,
    )


def _gate_out(env, terminal, truck_obj, container_obj=None):
    req = terminal.state.out_gates.request()
    yield req
    travel_time = terminal.TRUCK_OUTGATE_TIME + random.uniform(
        0, terminal.TRUCK_OUTGATE_TIME_DEV
    )
    yield env.timeout(travel_time)
    terminal.state.out_gates.release(req)
    container_label = (
        container_obj.to_string() if container_obj is not None
        else f"DrayageTruck-{truck_obj.id}"
    )
    utilities.record_container_event(
        terminal, container_label, "drayage_gate_out", env.now,
    )
    _record_trip_consumption(
        terminal, truck_obj, "truck",
        "loaded" if container_obj is not None else "empty",
        getattr(truck_obj, "train_id", ""),
        container_obj.to_string() if container_obj is not None else "",
        "drayage_gate_out",
        travel_time, env.now,
    )


def _claim_road_chassis(env, terminal):
    """Maybe claim a chassis from ``road_chassis_pool``; return
    ``(chassis_obj, was_claimed)``. If ``was_claimed=False``, the truck
    brings its own chassis (modeled implicitly)."""
    state = terminal.state
    cfg = terminal.config.get("yard_stack", {}) or {}
    claim_fraction = float(cfg.get("road_chassis_claim_fraction", 0.5))
    if random.random() < claim_fraction:
        chassis_obj = yield state.road_chassis_pool.get()
        return (chassis_obj, True)
    return (None, False)


def process_drayage_arrival(
    env, terminal, drayage_event: dict, truck_obj: "truck | None" = None,
):
    """SimPy generator: one drayage truck visit.

    Parameters
    ----------
    env, terminal
        Standard SimPy + LIFTS Terminal.
    drayage_event : dict
        One row from :func:`build_drayage_schedule` (as_dicts=True).
    truck_obj : optional truck
        Pre-built truck; if ``None``, one is constructed from
        ``drayage_event["truck_id"]`` using the diesel/electric mix.
    """
    arrival_time = float(drayage_event["arrival_time"])
    action = str(drayage_event["action"])
    truck_id = int(drayage_event["truck_id"])
    if truck_obj is None:
        truck_obj = _truck_factory(truck_id, terminal)

    if env.now < arrival_time:
        yield env.timeout(arrival_time - env.now)

    if action == "dropoff":
        # 1. Truck brings export container into the terminal. Build the
        #    container object now (truck_id reused as the synthetic
        #    "arrival id" so downstream joins stay meaningful).
        explicit_id = drayage_event.get("container_id")
        oc = container(type="Outbound", id=truck_id, train_id=0)
        if explicit_id:
            # The schedule may pre-name the container for testing; the
            # downstream pivot still uses our to_string() identity, but
            # we record an `external_id` event so the consumer can join.
            utilities.record_container_event(
                terminal, oc, f"external_id:{explicit_id}", env.now,
            )
        utilities.record_container_event(terminal, oc, "drayage_arrival", env.now)

        # 2. In-gate
        yield env.process(_gate_in(env, terminal, truck_obj))

        # 3. Drive to stack zone
        yield env.process(_drayage_zone_travel(env, terminal, "to_stack"))

        # 4. Stack-in (RTG or top-pick by availability)
        yield env.process(stack_in(env, terminal, oc, source_chassis=None))

        # 5. Empty truck exits
        yield env.process(_gate_out(env, terminal, truck_obj, container_obj=None))

    elif action == "pickup":
        # 1. Truck arrives empty to claim an IC.
        utilities.record_container_event(
            terminal, f"DrayageTruck-{truck_id}", "drayage_arrival", env.now,
        )

        # 2. In-gate (empty)
        yield env.process(_gate_in(env, terminal, truck_obj))

        # 3. Drive to stack zone
        yield env.process(_drayage_zone_travel(env, terminal, "to_stack"))

        # 4. Stack-out: pull next container off the stack onto the truck's
        #    chassis (road chassis claim is moot for pickup -- truck takes
        #    its own container home on its own chassis).
        ic = yield env.process(stack_out(env, terminal, container_obj=None))

        # 5. Loaded truck exits
        yield env.process(_gate_out(env, terminal, truck_obj, container_obj=ic))

    else:
        raise ValueError(
            f"drayage_event['action'] must be 'dropoff' or 'pickup'; "
            f"got {action!r}"
        )


def run_drayage_schedule(env, terminal, drayage_schedule):
    """Spawn one :func:`process_drayage_arrival` per row of the schedule.

    ``drayage_schedule`` is the list-of-dicts return of
    :func:`build_drayage_schedule` (or its DataFrame equivalent, iterated).
    Yields nothing -- the caller is expected to ``env.process(...)`` this
    once when the simulation starts.
    """
    if hasattr(drayage_schedule, "iter_rows"):
        rows: Any = list(drayage_schedule.iter_rows(named=True))
    else:
        rows = list(drayage_schedule)
    for row in rows:
        env.process(process_drayage_arrival(env, terminal, row))
        yield env.timeout(0)
