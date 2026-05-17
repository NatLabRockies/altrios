"""Container-level IC/OC handling: the IC->parking->truck loop and OC->chassis staging."""
import random

from altrios.lifts import distances, utilities
from altrios.lifts.emissions import (
    _record_side_emissions,
    _record_trip_emissions,
)
from altrios.lifts.hostlers import get_hostler, return_hostler
from altrios.lifts.truck_gate import truck_exit


def check_ic_picked_complete(env, terminal, train_schedule):
    train_id = train_schedule['train_id']
    remaining_ic = sum(
        (getattr(item, 'type', None) == 'Inbound') and (getattr(item, 'train_id', None) == train_id)
        for item in terminal.state.chassis.items
    )
    if (remaining_ic == 0
            and terminal.state.train_ic_unload_events[train_id].triggered
            and not terminal.state.train_ic_picked_events[train_id].triggered):
        terminal.state.train_ic_picked_events[train_id].succeed()


def handle_oc(env, terminal, train_schedule):
    '''
    This function is called right after IC dropped off, such that accelerating container processing.
    Note: handle_remaining_oc is designed for imbalanced container flow.
    TODO mbruchon: consolidate with handle_remaining_oc
    '''
    train_id = train_schedule['train_id']
    # 1) hostler transport OC from parking slots
    assigned_hostler = yield get_hostler(terminal)
    oc = yield terminal.state.parking_slots.get(
        lambda x: x.type == 'Outbound'
    )
    # TODO mbruchon: should current_veh_num come from assigned_hostler?
    current_veh_num = len(terminal.state.parked_hostlers.items) + 1
    hostler_reposition_travel_time, _, _, _ = distances.simulate_reposition_travel(
        assigned_hostler, current_veh_num, config=terminal.config
    )
    yield env.timeout(hostler_reposition_travel_time)
    utilities.record_container_event(terminal, oc.to_string(), 'hostler_pickup', env.now)
    _record_trip_emissions(terminal, assigned_hostler, "hostler", "empty",
                           train_id, oc.to_string(), "hostler_pickup",
                           hostler_reposition_travel_time, env.now,
                           track_id="parking_slots")

    # 2) side-pick loads an OC
    side_pick_unload_time = 1 / 60 + random.uniform(0, 1 / 600)
    yield env.timeout(side_pick_unload_time)  # side-pick
    # TODO mbruchon: this should instantiate a side loader crane
    # Qianqian's code:
    # side_pick_ems = emission_calculation(terminal, "loaded", "side", "side_loading_crane", "Diesel", travel_time=side_pick_unload_time)
    # record_emission(emission_records, "side_loading_crane", 'N/A', 'N/A', str(train_schedule['train_id']), oc,"side_unload", "truck_parking", side_pick_ems, side_pick_unload_time, env.now)
    #_record_trip_emissions(terminal, assigned_hostler, "hostler", "empty",
    #                       train_id, oc.to_string(), "hostler_to_parking_oc",
    #                       to_parking_time, env.now)

    # 4) hostler loaded with an OC goes from parking -> chassis
    to_chassis_time, _, _, _ = distances.simulate_hostler_track_travel(
        assigned_hostler, current_veh_num, config=terminal.config
    )
    yield env.timeout(to_chassis_time)
    yield terminal.state.chassis.put(oc)
    utilities.record_container_event(terminal, oc.to_string(), 'hostler_dropoff', env.now)
    _record_trip_emissions(terminal, assigned_hostler, "hostler", "loaded",
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

    # 1) Pull an IC off the chassis (not necessarily for this train...)
    ic = yield terminal.state.chassis.get(
        lambda x: x.type == 'Inbound' # and x.train_id == train_id
    )
    assigned_hostler = yield get_hostler(terminal)

    # 2) Empty hostler travels to the chassis
    current_veh_num = len(terminal.state.parked_hostlers.items) + 1
    pickup_time, _, _, _ = distances.simulate_hostler_track_travel(
        assigned_hostler, current_veh_num, config=terminal.config
    )
    yield env.timeout(pickup_time)
    _record_trip_emissions(terminal, assigned_hostler, "hostler", "empty",
                           train_id, ic.to_string(), "hostler_to_chassis",
                           pickup_time, env.now)

    # 3) Loaded hostler travels to parking slot
    current_veh_num = len(terminal.state.parked_hostlers.items) + 1
    dropoff_time, _, _, _ = distances.simulate_hostler_track_travel(
        assigned_hostler, current_veh_num, config=terminal.config
    )
    yield env.timeout(dropoff_time)
    yield terminal.state.parking_slots.put(ic)
    utilities.record_container_event(terminal, ic.to_string(), 'hostler_pickup', env.now)
    _record_trip_emissions(terminal, assigned_hostler, "hostler", "loaded",
                           train_id, ic.to_string(), "hostler_to_parking",
                           dropoff_time, env.now)

    check_ic_picked_complete(env, terminal, train_schedule)

    # 4) Side-pick / drop off recording
    side_pick_unload_time = 1 / 60 + random.uniform(0, 1 / 600)
    yield env.timeout(side_pick_unload_time)
    utilities.record_container_event(terminal, ic.to_string(), 'hostler_dropoff', env.now)
    _record_side_emissions(terminal, assigned_hostler, train_id, ic.to_string(), env.now)

    yield from return_hostler(env, terminal, assigned_hostler,
                              travel_time_to_active=0,
                              travel_time_to_parking=0,
    )
                              # active_hostlers_needed=True)

    # 5) Assign a truck to pick up the IC
    assigned_truck = yield terminal.state.truck_store.get()
    truck_travel_time, _, _, _ = distances.simulate_truck_travel(
        assigned_truck, train_schedule, terminal, config=terminal.config
    )
    yield env.timeout(truck_travel_time)
    ic = yield terminal.state.parking_slots.get(
        lambda x: x.type == 'Inbound' and x.train_id == train_id
    )
    utilities.record_container_event(terminal, ic.to_string(), 'truck_pickup', env.now)
    _record_trip_emissions(terminal, assigned_truck, "truck", "empty",
                           train_id, ic.to_string(), "truck_to_parking",
                           truck_travel_time, env.now)
    env.process(truck_exit(env, terminal, assigned_truck, ic, train_schedule))

    # 5. reposition and handle OC
    env.process(handle_oc(env, terminal, train_schedule))


def handle_remaining_oc(env, terminal, train_schedule):
    """Move staged OCs from parking slots onto chassis until all of this train's OCs are ready."""
    train_id = train_schedule['train_id']

    while True:
        # 1) how many oc remaining? & how many oc prepared?
        outbound_remaining = sum(
            (item.type == 'Outbound') and (item.train_id == train_id)
            for item in terminal.state.parking_slots.items
        )
        chassis_remaining = sum(
            (item.type == 'Outbound') and (item.train_id == train_id)
            for item in terminal.state.chassis.items
        )

        if outbound_remaining == 0 and chassis_remaining == train_schedule['oc_number']:
            print(f"Time {env.now:.3f}: All OC handled for train {train_id}.")
            if not terminal.state.train_oc_prepared_events[train_id].triggered:
                terminal.state.train_oc_prepared_events[train_id].succeed()
            return

        # 2) hostler transport OC from parking slots
        assigned_hostler = yield get_hostler(terminal)
        oc = yield terminal.state.parking_slots.get(
            lambda x: x.type == 'Outbound'
        )
        utilities.record_container_event(terminal, oc.to_string(), 'hostler_pickup', env.now)


        # 3) assign an empty-loaded hostler
        current_veh_num = len(terminal.state.parked_hostlers.items) + 1
        to_parking_time, _, _, _ = distances.simulate_hostler_track_travel(
            assigned_hostler, current_veh_num, config=terminal.config
        )
        yield env.timeout(to_parking_time)
        _record_trip_emissions(terminal, assigned_hostler, "hostler", "empty",
                               train_id, oc.to_string(), "hostler_to_parking_oc",
                               to_parking_time, env.now)

        # 4) side-pick loads an OC
        side_pick_unload_time = 1 / 60 + random.uniform(0, 1 / 600)
        yield env.timeout(side_pick_unload_time)  # side-pick
        # TODO mbruchon: this should instantiate a side loader crane
        # Qianqian's code:
        # side_pick_ems = emission_calculation(terminal, "loaded", "side", "side_loading_crane", "Diesel", travel_time=side_pick_unload_time)
        # record_emission(emission_records, "side_loading_crane", 'N/A', 'N/A', str(train_schedule['train_id']), oc,"side_unload", "truck_parking", side_pick_ems, side_pick_unload_time, env.now)
        #_record_trip_emissions(terminal, assigned_hostler, "hostler", "empty",
        #                       train_id, oc.to_string(), "hostler_to_parking_oc",
        #                       to_parking_time, env.now)

        # 5) hostler loaded with an OC -> chassis
        to_chassis_time, _, _, _ = distances.simulate_hostler_track_travel(
            assigned_hostler, current_veh_num, config=terminal.config
        )
        yield env.timeout(to_chassis_time)
        yield terminal.state.chassis.put(oc)
        utilities.record_container_event(terminal, oc.to_string(), 'hostler_dropoff', env.now)
        _record_trip_emissions(terminal, assigned_hostler, "hostler", "loaded",
                               train_id, oc.to_string(), "hostler_to_chassis_oc",
                               to_chassis_time, env.now)

        # 6) hostler travel back
        yield from return_hostler(env, terminal, assigned_hostler,
                                  travel_time_to_active=0,
                                  travel_time_to_parking=to_parking_time)
