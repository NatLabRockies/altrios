"""Per-train orchestration for the spec-based ``truck_rail`` and
``rail_vessel`` modes.

A train arrival exchanges containers with the *main container stack* via
the rail yard tractor pool:

* Discharge: rail_track_rtg lifts IC off the train; rail_yard_tractor hauls
  to the stack; main_stack_rtg / top_pick performs the ``stack_in``.
* Load: ``stack_out`` pulls an OC off the stack; rail_yard_tractor hauls
  it to the rail zone; rail_track_rtg lifts it onto the train.

This module does *not* speak to drayage trucks directly. OCs arrive on the
stack via :mod:`altrios.lifts.drayage_flow` (truck_rail mode) or
:mod:`altrios.lifts.vessel_flow` (rail_vessel mode); ICs depart the stack
via the same routes in reverse. The decoupling through the stack is the
whole point of the rebuild — train and truck timelines are independent.
"""
from __future__ import annotations

import random
from typing import Any

from altrios.lifts import utilities
from altrios.lifts.classes import container, loggingLevel
from altrios.lifts.energy_use import _record_stack_lift_energy
from altrios.lifts.yard_flow import stack_in, stack_out, yard_tractor_haul


def _unload_one_ic(
    env, terminal, track_id: int, train_id: int, ic,
    done_counter: list, done_event, ic_count: int,
):
    """Lift one IC off the train and route it to the stack.

    The rail-track RTG is held only for the lift duration; the chassis
    haul + stack-in continue without holding the RTG so other unload
    tasks can grab it for the next IC.
    """
    state = terminal.state
    rtg_pool = state.rail_track_rtgs_by_track[track_id]
    rtg_obj = yield rtg_pool.get()
    try:
        lift_time = (
            terminal.CONTAINERS_PER_CRANE_MOVE_MEAN
            + random.uniform(0, terminal.CRANE_MOVE_DEV_TIME)
        )
        yield env.timeout(lift_time)
        utilities.record_container_event(
            terminal, ic, "rail_track_rtg_unload", env.now,
        )
        _record_stack_lift_energy(
            terminal, rtg_obj, "rail_track_rtg", status="loaded",
            train_id=train_id, container_id=ic.to_string(),
            event_type="rail_track_rtg_unload", env_now=env.now,
            zone="track",
        )
    finally:
        yield rtg_pool.put(rtg_obj)

    # Now move the chassis to the stack and lift the container onto the
    # stack. These run sequentially in this task but do not block other
    # rail_track_rtg workers from processing more ICs.
    yield env.process(yard_tractor_haul(
        env, terminal, state.rail_yard_tractors,
        ic, from_zone="rail", to_zone="stack",
    ))
    yield env.process(stack_in(env, terminal, ic, source_chassis=None))

    done_counter[0] += 1
    if done_counter[0] == ic_count and not done_event.triggered:
        done_event.succeed()


def _load_one_oc(
    env, terminal, track_id: int, train_id: int,
    done_counter: list, done_event, oc_count: int,
    loaded_ocs: list,
):
    """Pull one OC off the stack and load it onto the train.

    The OC's identity (``to_string()``) is preserved as-is — re-tagging
    ``train_id`` mid-flight would split the OC's container_event rows
    across two ids (the dropoff side recorded events under the old id).
    Train attribution is recovered post-sim by joining on the timing of
    ``rail_track_rtg_load``.
    """
    state = terminal.state
    oc = yield env.process(stack_out(env, terminal, container_obj=None))

    yield env.process(yard_tractor_haul(
        env, terminal, state.rail_yard_tractors,
        oc, from_zone="stack", to_zone="rail",
    ))

    rtg_pool = state.rail_track_rtgs_by_track[track_id]
    rtg_obj = yield rtg_pool.get()
    try:
        lift_time = (
            terminal.CONTAINERS_PER_CRANE_MOVE_MEAN
            + random.uniform(0, terminal.CRANE_MOVE_DEV_TIME)
        )
        yield env.timeout(lift_time)
        utilities.record_container_event(
            terminal, oc, "rail_track_rtg_load", env.now,
        )
        _record_stack_lift_energy(
            terminal, rtg_obj, "rail_track_rtg", status="loaded",
            train_id=train_id, container_id=oc.to_string(),
            event_type="rail_track_rtg_load", env_now=env.now,
            zone="track",
        )
    finally:
        yield rtg_pool.put(rtg_obj)

    loaded_ocs.append(oc)
    done_counter[0] += 1
    if done_counter[0] == oc_count and not done_event.triggered:
        done_event.succeed()


def process_train_arrival(env, terminal, train_schedule: dict[str, Any]):
    """SimPy generator: one train arrival.

    Steps:

    1. Pre-record ``train_arrival_expected`` for each IC.
    2. Wait until ``arrival_time``.
    3. Acquire a track from ``state.tracks``.
    4. Spawn one ``_unload_one_ic`` task per IC; wait for all to complete.
    5. Spawn one ``_load_one_oc`` task per OC; wait for all to complete.
       The OCs are pulled from the main stack; the caller is responsible
       for ensuring the stack has enough OCs by simulation time
       (truck_rail mode auto-synthesizes a drayage schedule; rail_vessel
       expects the vessel side to pre-stage them).
    6. Hold the track until ``departure_time``; record ``train_depart``.
    7. Release the track.
    """
    state = terminal.state
    train_id = int(train_schedule["train_id"])
    arrival_time = float(train_schedule["arrival_time"])
    departure_time = float(train_schedule["departure_time"])
    ic_count = int(train_schedule.get("full_cars") or 0)
    oc_count = int(train_schedule.get("oc_number") or 0)

    for ic_id in range(1, ic_count + 1):
        ic_label = container(type="Inbound", id=ic_id, train_id=train_id)
        utilities.record_container_event(
            terminal, ic_label, "train_arrival_expected", arrival_time,
        )

    if env.now < arrival_time:
        yield env.timeout(arrival_time - env.now)

    track_id = yield state.tracks.get()
    try:
        terminal.log(
            loggingLevel.BASIC,
            f"Time {env.now:.3f}: Train {train_id} on track {track_id} "
            f"(IC={ic_count}, OC={oc_count}).",
        )

        # ---- Unload phase ------------------------------------------------
        if ic_count > 0:
            ic_done = env.event()
            ic_done_counter = [0]
            for ic_id in range(1, ic_count + 1):
                ic = container(type="Inbound", id=ic_id, train_id=train_id)
                utilities.record_container_event(
                    terminal, ic, "train_arrival_actual", env.now,
                )
                env.process(_unload_one_ic(
                    env, terminal, track_id, train_id, ic,
                    ic_done_counter, ic_done, ic_count,
                ))
            yield ic_done

        # ---- Load phase --------------------------------------------------
        loaded_ocs: list = []
        if oc_count > 0:
            oc_done = env.event()
            oc_done_counter = [0]
            for _ in range(oc_count):
                env.process(_load_one_oc(
                    env, terminal, track_id, train_id,
                    oc_done_counter, oc_done, oc_count, loaded_ocs,
                ))
            yield oc_done

        # ---- Hold and depart --------------------------------------------
        if env.now < departure_time:
            yield env.timeout(departure_time - env.now)
        utilities.record_container_event(
            terminal, f"Train-{train_id}", "train_depart", env.now,
        )
        for oc in loaded_ocs:
            utilities.record_container_event(
                terminal, oc, "train_depart", env.now,
            )
        state.time_per_train[train_id] = env.now - arrival_time
        state.train_delay_time[train_id] = max(0.0, env.now - departure_time)
    finally:
        yield state.tracks.put(track_id)
