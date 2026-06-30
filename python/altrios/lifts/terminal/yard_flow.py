"""Yard-flow primitives shared by ``truck_rail``, ``rail_vessel``, and
``vessel_truck``.

The seven-step container journey in these three modes always routes
through the main container stack::

    source-endpoint
        -> (source chassis traversal)
        -> stack_in
        -> container_stack
        -> stack_out
        -> (destination chassis traversal)
        -> destination-endpoint

This module owns the four operations that touch the stack and its supporting
equipment, factored out so the higher-level train / drayage / vessel
arrival handlers in :mod:`altrios.lifts.terminal.python_helpers` can compose them
without duplicating the crane-selection and yard-tractor logic:

* :func:`stack_in` -- lift a container off a chassis onto the stack.
* :func:`stack_out` -- lift a container off the stack onto a chassis.
* :func:`_choose_stack_crane` -- dynamic-routing helper that picks between
  the ``main_stack_rtgs`` pool and the ``top_picks`` pool by availability.
* :func:`yard_tractor_haul` -- yard-tractor move of a chassis between two
  zones (e.g. ``"rail" -> "stack"`` or ``"stack" -> "berth"``).

All four are SimPy generators and must be invoked via ``env.process(...)``
or ``yield env.process(...)`` from another generator.

These helpers consume the spec-built ``state`` attributes
(``state.main_stack_rtgs``, ``state.top_picks``, ``state.container_stack``,
etc.) defined in :mod:`altrios.lifts.terminal.specs`. Energy-use entries are
recorded with ``resource_type=`` set to the equipment pool name
(``main_stack_rtg``, ``top_pick``, ``yard_tractor``) and use the per-
equipment rates in ``energy_use.load_consumption`` / ``trip_consumption``
(the config block is still keyed ``energy_use`` in the YAML).

"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from altrios.lifts.terminal import distances, utilities
from altrios.lifts.terminal.consumption import (
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
        calls.
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

    SimPy generator; invoke as ``yield env.process(stack_in(...))``. The
    chassis itself is **not** released here — the caller decides whether
    to send it back to its pool (terminal chassis) or to keep it with
    the endpoint vehicle (drayage road chassis).

    Parameters
    ----------
    env : simpy.Environment
        The SimPy environment.
    state : TerminalState
        Run state with ``main_stack_rtgs``, ``top_picks``, and
        ``container_stack`` Stores attached.
    config : Mapping
        Site-level config dict; reads
        ``containers_per_crane_move_mean``, ``crane_move_dev_time``,
        ``energy_use``, and ``yard_stack.routing_strategy``.
    container_obj : container
        Container being placed onto the stack.
    source_chassis : chassis, optional
        Chassis the container is being lifted from. ``None`` for
        endpoints that don't carry the container on a chassis through
        the stack lift; the value is currently used only as a label.

    Yields
    ------
    SimPy events
        Crane acquisition, lift timeout, stack put, and crane release.
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

    SimPy generator; invoke as ``yield env.process(stack_out(...))``.
    Either takes ``container_obj`` as the **specific** container to
    remove (a future enhancement; today the stack is unkeyed and the
    SimPy Store returns whichever container is at the head), or, when
    ``container_obj is None``, returns the first container off the
    stack via ``StopIteration.value``.

    Parameters
    ----------
    env : simpy.Environment
        The SimPy environment.
    state : TerminalState
        Run state with ``main_stack_rtgs``, ``top_picks``, and
        ``container_stack`` Stores attached.
    config : Mapping
        Site-level config dict; reads the same keys as :func:`stack_in`.
    container_obj : container, optional
        Specific container to return; if provided, one Store slot is
        still consumed for capacity bookkeeping.
    dest_chassis : chassis, optional
        Chassis the container is being lowered onto. Currently used
        only as a label; the caller manages the chassis lifecycle
        (see :func:`stack_in` for the rationale).

    Yields
    ------
    SimPy events
        Stack get, crane acquisition, lift timeout, and crane release.

    Returns
    -------
    container
        The container removed from the stack (delivered via
        ``StopIteration.value``).
    """
    # The stack is a flat Store; pick the head item.
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
    """Haul one container with a yard tractor between two zones.

    SimPy generator; invoke as
    ``yield env.process(yard_tractor_haul(...))``. The chassis carrying
    the container is implicit; this helper only models the tractor's
    trip time and energy. The container-event label is
    ``f"yard_tractor_{from_zone}_to_{to_zone}"``.

    Parameters
    ----------
    env : simpy.Environment
        The SimPy environment.
    state : TerminalState
        Run state with the tractor pool, ``distances`` cache, and
        ``output`` collector attached.
    config : Mapping
        Site-level config dict; reads ``energy_use``.
    tractor_pool : simpy.Store
        Pool to acquire one tractor from — typically
        ``state.main_yard_tractors`` (water↔stack) or
        ``state.rail_yard_tractors`` (rail↔stack).
    container_obj : container
        Container being hauled. Used for event labelling and energy
        attribution.
    from_zone, to_zone : str
        Origin / destination zone labels for the event tag.
    travel_time : float, optional
        Override duration in hours. When ``None`` (default), the
        duration is sampled from
        :func:`distances.simulate_hostler_track_travel` using the
        tractor pool's in-flight count as the congestion signal.

    Yields
    ------
    SimPy events
        Tractor acquisition, travel timeout, and tractor release.
    """
    tractor = yield tractor_pool.get()
    try:
        if travel_time is None:
            # Pool-local "in-flight" count gives the speed-density model a
            # plausible congestion signal.
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
