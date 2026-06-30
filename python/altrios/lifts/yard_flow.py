"""Yard-flow primitives shared by ``truck_rail``, ``rail_vessel``, and
``vessel_truck``.

The seven-step container journey in these three Phase 1 modes always routes
through the main container stack::

    source-endpoint
        -> (source chassis traversal)
        -> stack_in
        -> container_stack
        -> stack_out
        -> (destination chassis traversal)
        -> destination-endpoint

This module owns the four operations that touch the stack and its supporting
equipment, factored out so the three endpoint modules (``train_flow``,
``vessel_flow``, ``drayage_flow``) can compose them without duplicating the
crane-selection and yard-tractor logic:

* :func:`stack_in` -- lift a container off a chassis onto the stack.
* :func:`stack_out` -- lift a container off the stack onto a chassis.
* :func:`_choose_stack_crane` -- dynamic-routing helper that picks between
  the ``main_stack_rtgs`` pool and the ``top_picks`` pool by availability.
* :func:`yard_tractor_haul` -- yard-tractor move of a chassis between two
  zones (e.g. ``"rail" -> "stack"`` or ``"stack" -> "berth"``).

All four are SimPy generators and must be invoked via ``env.process(...)``
or ``yield env.process(...)`` from another generator.

These helpers consume the spec-built ``TerminalState`` attributes
(``state.main_stack_rtgs``, ``state.top_picks``, ``state.container_stack``,
etc.) defined in :mod:`altrios.lifts.specs`. Energy-use entries are
recorded with ``resource_type=`` set to the equipment pool name
(``main_stack_rtg``, ``top_pick``, ``yard_tractor``) and use the per-
equipment rates in ``energy_use.load_consumption`` / ``trip_consumption``
(the config block is still keyed ``energy_use`` in the YAML).

"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from altrios.lifts import distances, utilities
from altrios.lifts.consumption import (
    _record_stack_lift_consumption,
    _record_yard_tractor_trip_consumption,
)

if TYPE_CHECKING:
    import simpy  # noqa: F401


def _choose_stack_crane(env, state, strategy: str = "availability"):
    """Acquire one stack crane from either ``main_stack_rtgs`` or
    ``top_picks`` according to ``strategy``.

    Yields the SimPy ``get`` event and returns a ``(pool_name, crane_obj)``
    tuple via ``StopIteration.value``. The caller is responsible for
    putting ``crane_obj`` back into the correct pool (use
    ``state.main_stack_rtgs`` if ``pool_name == "main_stack_rtg"``, else
    ``state.top_picks``).

    Strategies
    ----------
    ``"availability"`` (default)
        Take from the pool whose store currently has more idle items; on a
        tie (including both-empty), prefer ``main_stack_rtgs``. This avoids
        the SimPy event-cancellation dance of racing two parallel ``get()``
        calls; Phase 2 may revisit this with a true race when both pools
        are starved.
    ``"rtg_only"``
        Always wait on ``main_stack_rtgs``.
    ``"top_pick_only"``
        Always wait on ``top_picks``.
    """
    rtg_pool = state.main_stack_rtgs
    tp_pool = state.top_picks

    if strategy == "rtg_only":
        crane_obj = yield rtg_pool.get()
        return ("main_stack_rtg", crane_obj)
    if strategy == "top_pick_only":
        crane_obj = yield tp_pool.get()
        return ("top_pick", crane_obj)
    if strategy != "availability":
        raise ValueError(f"Unknown routing strategy: {strategy!r}")

    rtg_items = len(rtg_pool.items)
    tp_items = len(tp_pool.items)
    if rtg_items >= tp_items:
        crane_obj = yield rtg_pool.get()
        return ("main_stack_rtg", crane_obj)
    crane_obj = yield tp_pool.get()
    return ("top_pick", crane_obj)


def _stack_crane_strategy(config) -> str:
    """Read the configured routing strategy from ``yard_stack.routing_strategy``;
    default ``"availability"``."""
    cfg = config.get("yard_stack", {}) or {}
    return str(cfg.get("routing_strategy", "availability"))


def stack_in(env, state, config, container_obj, source_chassis=None):
    """Lift ``container_obj`` off ``source_chassis`` onto the main stack.

    ``source_chassis`` may be ``None`` for endpoints that don't carry the
    container on a chassis through the stack lift (the field is recorded
    only as a label on the resulting container_event row). The chassis
    itself is *not* released here -- the caller decides whether to send it
    back to its pool (terminal chassis) or to keep it with the endpoint
    vehicle (drayage road chassis).
    """
    pool_name, crane_obj = yield env.process(
        _choose_stack_crane(env, state, _stack_crane_strategy(config))
    )
    pool = state.main_stack_rtgs if pool_name == "main_stack_rtg" else state.top_picks
    try:
        lift_time = (
            config["containers_per_crane_move_mean"]
            + random.uniform(0, config["crane_move_dev_time"])
        )
        yield env.timeout(lift_time)
        yield state.container_stack.put(container_obj)
        utilities.record_container_event(
            state, container_obj, "stack_in", env.now,
        )
        _record_stack_lift_consumption(
            getattr(state, "output", None),
            config["energy_use"], crane_obj, pool_name, status="loaded",
            train_id=getattr(container_obj, "train_id", ""),
            container_id=container_obj.to_string(),
            event_type=f"{pool_name}_stack_in",
            env_now=env.now,
        )
    finally:
        yield pool.put(crane_obj)


def stack_out(env, state, config, container_obj=None, dest_chassis=None):
    """Pull one container off the main stack.

    Either takes ``container_obj`` as the *specific* container to remove
    (a future enhancement; today the stack is unkeyed and the SimPy
    ``Store`` returns whichever container is at the head), or, when
    ``container_obj is None``, just returns the first container off the
    stack via ``StopIteration.value``.

    ``dest_chassis`` is recorded as a label; the caller manages the chassis
    lifecycle (see :func:`stack_in` for the rationale).
    """
    # Phase 1: stack is a flat Store; pick the head item. Callers that need
    # a specific container should pre-stage it or use a FilterStore in a
    # future phase.
    if container_obj is None:
        container_obj = yield state.container_stack.get()
    else:
        # ``container_obj`` was passed in: still consume one slot from the
        # stack so capacity bookkeeping is correct. The retrieved item is
        # ignored.
        _ = yield state.container_stack.get()

    pool_name, crane_obj = yield env.process(
        _choose_stack_crane(env, state, _stack_crane_strategy(config))
    )
    pool = state.main_stack_rtgs if pool_name == "main_stack_rtg" else state.top_picks
    try:
        lift_time = (
            config["containers_per_crane_move_mean"]
            + random.uniform(0, config["crane_move_dev_time"])
        )
        yield env.timeout(lift_time)
        utilities.record_container_event(
            state, container_obj, "stack_out", env.now,
        )
        _record_stack_lift_consumption(
            getattr(state, "output", None),
            config["energy_use"], crane_obj, pool_name, status="loaded",
            train_id=getattr(container_obj, "train_id", ""),
            container_id=container_obj.to_string(),
            event_type=f"{pool_name}_stack_out",
            env_now=env.now,
        )
    finally:
        yield pool.put(crane_obj)
    return container_obj


def yard_tractor_haul(
    env, state, config, tractor_pool, container_obj, from_zone: str, to_zone: str,
    travel_time: float | None = None,
):
    """Yard tractor pulled from ``tractor_pool`` hauls one container from
    ``from_zone`` to ``to_zone``.

    ``tractor_pool`` is the SimPy ``Store`` -- either
    ``state.main_yard_tractors`` (water<->stack) or
    ``state.rail_yard_tractors`` (rail<->stack). The chassis carrying the
    container is implicit; this helper only models the tractor's trip time
    and energy. Container-event label is
    ``f"yard_tractor_{from_zone}_to_{to_zone}"``.

    If ``travel_time`` is ``None``, the duration is sampled from
    :func:`distances.simulate_hostler_track_travel` as a Phase 1
    placeholder; future phases can model zone-pair geometry.
    """
    tractor = yield tractor_pool.get()
    try:
        if travel_time is None:
            # Pool-local "in-flight" count so the speed-density model has a
            # plausible congestion signal without leaking the legacy
            # ``in_flight_hostler_count`` definition.
            in_flight = tractor_pool.capacity - len(tractor_pool.items)
            travel_time, _, _, _ = distances.simulate_hostler_track_travel(
                tractor, in_flight, params=state.distances,
            )
        yield env.timeout(travel_time)
        event_label = f"yard_tractor_{from_zone}_to_{to_zone}"
        utilities.record_container_event(
            state, container_obj, event_label, env.now,
        )
        _record_yard_tractor_trip_consumption(
            getattr(state, "output", None),
            config["energy_use"], tractor, status="loaded",
            train_id=getattr(container_obj, "train_id", ""),
            container_id=container_obj.to_string(),
            event_type=event_label,
            travel_time=travel_time, env_now=env.now,
        )
    finally:
        yield tractor_pool.put(tractor)
