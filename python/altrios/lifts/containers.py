"""Container-level IC/OC handling: the IC->parking->truck loop and OC->chassis staging."""
import random

from altrios.lifts import distances, utilities
from altrios.lifts.classes import loggingLevel
from altrios.lifts.energy_use import (
    _record_side_energy,
    _record_trip_energy,
)
from altrios.lifts.hostlers import get_hostler, return_hostler
from altrios.lifts.truck_gate import truck_exit


def check_ic_picked_complete(env, terminal, train_schedule):
    train_id = train_schedule['train_id']
    state = terminal.state
    remaining_ic = state.chassis_ic_count_by_train.get(train_id, 0)
    if (remaining_ic == 0
            and state.train_ic_unload_events[train_id].triggered
            and not state.train_ic_picked_events[train_id].triggered):
        state.train_ic_picked_events[train_id].succeed()


def handle_oc(env, terminal, train_schedule):
    """Eager single-OC move kicked off as soon as one IC has been dropped,
    so OC staging can run in parallel with IC delivery instead of waiting
    for all ICs to finish. ``handle_remaining_oc`` is the catch-up sweep
    that drains whatever this eager path didn't get to (and also handles
    the imbalanced-flow case where there are more OCs than ICs)."""
    train_id = train_schedule['train_id']
    state = terminal.state
    # 1) hostler transport OC from parking slots
    assigned_hostler = yield get_hostler(terminal)
    oc = yield state.parking_oc_store.get()
    state.parking_oc_count_by_train[oc.train_id] = (
        state.parking_oc_count_by_train.get(oc.train_id, 0) - 1
    )
    # current_veh_num is a yard-wide congestion measure (hostlers in flight)
    # feeding the hostler-speed model in distances.simulate_*; it is not a
    # per-hostler attribute, so it intentionally doesn't come from
    # assigned_hostler.
    current_veh_num = state.in_flight_hostler_count()
    hostler_reposition_travel_time, _, _, _ = distances.simulate_reposition_travel(
        assigned_hostler, current_veh_num, params=terminal.distances
    )
    yield env.timeout(hostler_reposition_travel_time)
    utilities.record_container_event(terminal, oc.to_string(), 'hostler_pickup', env.now)
    _record_trip_energy(terminal, assigned_hostler, "hostler", "empty",
                           train_id, oc.to_string(), "hostler_pickup",
                           hostler_reposition_travel_time, env.now,
                           track_id="parking_slots")

    # 2) side-pick loads an OC (charged to the assigned hostler; no separate
    # side-loader resource exists yet).
    side_pick_unload_time = 1 / 60 + random.uniform(0, 1 / 600)
    yield env.timeout(side_pick_unload_time)
    _record_side_energy(terminal, assigned_hostler, train_id, oc.to_string(), env.now)

    # 4) hostler loaded with an OC goes from parking -> chassis
    to_chassis_time, _, _, _ = distances.simulate_hostler_track_travel(
        assigned_hostler, current_veh_num, params=terminal.distances
    )
    yield env.timeout(to_chassis_time)
    yield state.chassis_oc_store(oc.train_id).put(oc)
    state.chassis_oc_count_by_train[oc.train_id] = (
        state.chassis_oc_count_by_train.get(oc.train_id, 0) + 1
    )
    utilities.record_container_event(terminal, oc.to_string(), 'hostler_dropoff', env.now)
    _record_trip_energy(terminal, assigned_hostler, "hostler", "loaded",
                            train_id, oc.to_string(), "hostler_to_chassis_oc",
                            to_chassis_time, env.now)

    # 5) hostler travel back
    # TODO mbruchon: why is travel_time_to_parking 0?
    yield from return_hostler(env, terminal, assigned_hostler,
                                travel_time_to_active=0,
                                travel_time_to_parking=0)


def container_process(env, terminal, train_schedule):
    """
    It is designed to transfer both inbound and outbound containers (IC-OC loop).
    IC -> chassis -> hostler -> parking slot -> truck pickup.
    The main simulation process is as follows:
    1. A hostler picks up an IC, and drops off IC at parking slot.
    2. A truck picks up the IC, and leaves the gate
    3. The hostler picks up an OC, and drops off OC at the chassis.
    4. Once all OCs are prepared (all_oc_prepared), triggers the event of crane loading.
    """
    train_id = train_schedule['train_id']
    state = terminal.state

    # 1) Pull an IC off the chassis (not necessarily for this train...)
    ic = yield state.chassis_ic_store.get()
    state.chassis_ic_count_by_train[ic.train_id] = (
        state.chassis_ic_count_by_train.get(ic.train_id, 0) - 1
    )
    assigned_hostler = yield get_hostler(terminal)

    # 2) Empty hostler travels to the chassis
    current_veh_num = state.in_flight_hostler_count()
    pickup_time, _, _, _ = distances.simulate_hostler_track_travel(
        assigned_hostler, current_veh_num, params=terminal.distances
    )
    yield env.timeout(pickup_time)
    _record_trip_energy(terminal, assigned_hostler, "hostler", "empty",
                           train_id, ic.to_string(), "hostler_to_chassis",
                           pickup_time, env.now)

    # 3) Loaded hostler travels to parking slot
    current_veh_num = state.in_flight_hostler_count()
    dropoff_time, _, _, _ = distances.simulate_hostler_track_travel(
        assigned_hostler, current_veh_num, params=terminal.distances
    )
    yield env.timeout(dropoff_time)
    yield state.parking_ic_store(ic.train_id).put(ic)
    utilities.record_container_event(terminal, ic.to_string(), 'hostler_pickup', env.now)
    _record_trip_energy(terminal, assigned_hostler, "hostler", "loaded",
                           train_id, ic.to_string(), "hostler_to_parking",
                           dropoff_time, env.now)

    check_ic_picked_complete(env, terminal, train_schedule)

    # 4) Side-pick / drop off recording
    side_pick_unload_time = 1 / 60 + random.uniform(0, 1 / 600)
    yield env.timeout(side_pick_unload_time)
    utilities.record_container_event(terminal, ic.to_string(), 'hostler_dropoff', env.now)
    _record_side_energy(terminal, assigned_hostler, train_id, ic.to_string(), env.now)

    yield from return_hostler(env, terminal, assigned_hostler,
                              travel_time_to_active=0,
                              travel_time_to_parking=0,
    )
                              # active_hostlers_needed=True)

    # 5) Assign a truck to pick up the IC
    assigned_truck = yield terminal.state.truck_store.get()
    truck_travel_time, _, _, _ = distances.simulate_truck_travel(
        assigned_truck, train_schedule, terminal, params=terminal.distances
    )
    yield env.timeout(truck_travel_time)
    ic = yield state.parking_ic_store(train_id).get()
    utilities.record_container_event(terminal, ic.to_string(), 'truck_pickup', env.now)
    _record_trip_energy(terminal, assigned_truck, "truck", "empty",
                           train_id, ic.to_string(), "truck_to_parking",
                           truck_travel_time, env.now)
    env.process(truck_exit(env, terminal, assigned_truck, ic, train_schedule))

    # 5. reposition and handle OC
    env.process(handle_oc(env, terminal, train_schedule))


def handle_remaining_oc(env, terminal, train_schedule):
    """Move staged OCs from parking slots onto chassis until all of this train's OCs are ready."""
    train_id = train_schedule['train_id']
    state = terminal.state

    while True:
        # 1) how many oc remaining? & how many oc prepared?
        outbound_remaining = state.parking_oc_count_by_train.get(train_id, 0)
        chassis_remaining = state.chassis_oc_count_by_train.get(train_id, 0)

        if outbound_remaining == 0 and chassis_remaining == train_schedule['oc_number']:
            terminal.log(loggingLevel.DEBUG, f"Time {env.now:.3f}: All OC handled for train {train_id}.")
            if not state.train_oc_prepared_events[train_id].triggered:
                state.train_oc_prepared_events[train_id].succeed()
            return

        # 2) hostler transport OC from parking slots
        assigned_hostler = yield get_hostler(terminal)
        oc = yield state.parking_oc_store.get()
        state.parking_oc_count_by_train[oc.train_id] = (
            state.parking_oc_count_by_train.get(oc.train_id, 0) - 1
        )
        utilities.record_container_event(terminal, oc.to_string(), 'hostler_pickup', env.now)


        # 3) assign an empty-loaded hostler
        current_veh_num = state.in_flight_hostler_count()
        to_parking_time, _, _, _ = distances.simulate_hostler_track_travel(
            assigned_hostler, current_veh_num, params=terminal.distances
        )
        yield env.timeout(to_parking_time)
        _record_trip_energy(terminal, assigned_hostler, "hostler", "empty",
                               train_id, oc.to_string(), "hostler_to_parking_oc",
                               to_parking_time, env.now)

        # 4) side-pick loads an OC (charged to the assigned hostler; no
        # separate side-loader resource exists yet).
        side_pick_unload_time = 1 / 60 + random.uniform(0, 1 / 600)
        yield env.timeout(side_pick_unload_time)
        _record_side_energy(terminal, assigned_hostler, train_id, oc.to_string(), env.now)

        # 5) hostler loaded with an OC -> chassis
        to_chassis_time, _, _, _ = distances.simulate_hostler_track_travel(
            assigned_hostler, current_veh_num, params=terminal.distances
        )
        yield env.timeout(to_chassis_time)
        yield state.chassis_oc_store(oc.train_id).put(oc)
        state.chassis_oc_count_by_train[oc.train_id] = (
            state.chassis_oc_count_by_train.get(oc.train_id, 0) + 1
        )
        utilities.record_container_event(terminal, oc.to_string(), 'hostler_dropoff', env.now)
        _record_trip_energy(terminal, assigned_hostler, "hostler", "loaded",
                               train_id, oc.to_string(), "hostler_to_chassis_oc",
                               to_chassis_time, env.now)

        # 6) hostler travel back
        yield from return_hostler(env, terminal, assigned_hostler,
                                  travel_time_to_active=0,
                                  travel_time_to_parking=to_parking_time)
