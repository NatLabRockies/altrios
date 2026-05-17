import simpy
import random
import polars as pl
from altrios.lifts import utilities
from altrios.lifts import distances
from altrios.lifts.classes import Terminal, container, truck

# Module-level emission record buffer; populacxcted by container_process / handle_remaining_oc /
# crane_unload_process / crane_load_process / truck_entry / truck_exit and consumed by
# run_simulation when out_path is supplied.
emission_records: list = []


def _energy_type_for(vehicle_obj) -> str:
    """Normalize hostler/truck/crane .type into the keys used by emission_calculation."""
    t = getattr(vehicle_obj, "type", "Diesel")
    return str(t).capitalize()


def _record_trip_emissions(terminal, vehicle_obj, vehicle_kind, status, train_id,
                           container_id, event_type, travel_time, env_now, track_id=""):
    """Helper to compute + append a trip emission record."""
    energy_type = _energy_type_for(vehicle_obj)
    emission_value = utilities.emission_calculation(
        terminal, status=status, move="trip", vehicle=vehicle_kind,
        energy_type=energy_type, travel_time=travel_time,
    )
    utilities.record_emission(
        emission_records,
        vehicle_type=vehicle_kind,
        resource_id=getattr(vehicle_obj, "id", ""),
        track_id=track_id,
        train_id=train_id,
        container_id=container_id,
        event_type=event_type,
        zone="yard",
        emission_value=emission_value,
        travel_time=travel_time,
        env_now=env_now,
    )


def _record_load_emissions(terminal, crane_obj, status, train_id, container_id,
                           event_type, env_now, track_id):
    """Helper to compute + append a per-lift crane emission record."""
    energy_type = _energy_type_for(crane_obj)
    emission_value = utilities.emission_calculation(
        terminal, status=status, move="load", vehicle="crane",
        energy_type=energy_type, travel_time=0.0,
    )
    utilities.record_emission(
        emission_records,
        vehicle_type="crane",
        resource_id=getattr(crane_obj, "id", ""),
        track_id=track_id,
        train_id=train_id,
        container_id=container_id,
        event_type=event_type,
        zone="track",
        emission_value=emission_value,
        travel_time=0.0,
        env_now=env_now,
    )


def _record_side_emissions(terminal, hostler_obj, train_id, container_id, env_now):
    """Helper to compute + append a side-pick emission record."""
    energy_type = _energy_type_for(hostler_obj)
    emission_value = utilities.emission_calculation(
        terminal, status="loaded", move="side", vehicle="hostler",
        energy_type=energy_type, travel_time=0.0,
    )
    utilities.record_emission(
        emission_records,
        vehicle_type="hostler",
        resource_id=getattr(hostler_obj, "id", ""),
        track_id="",
        train_id=train_id,
        container_id=container_id,
        event_type="side_pick",
        zone="parking",
        emission_value=emission_value,
        travel_time=0.0,
        env_now=env_now,
    )


# ---------------------------------------------------------------------------
# Truck arrival side
# ---------------------------------------------------------------------------

def truck_entry(env, terminal, truck_obj, oc, train_schedule):
    """One loaded truck enters the gate, drops its OC at a parking slot."""
    train_id = train_schedule["train_id"]
    ingate_request = terminal.in_gates.request()
    yield ingate_request
    travel_time = terminal.TRUCK_INGATE_TIME + random.uniform(0, terminal.TRUCK_INGATE_TIME_DEV)
    yield env.timeout(travel_time)
    terminal.in_gates.release(ingate_request)

    utilities.record_container_event(terminal, oc.to_string(), 'truck_arrival', env.now)
    utilities.record_container_event(terminal, oc.to_string(), 'truck_dropoff', env.now)
    yield terminal.parking_slots.put(oc)
    _record_trip_emissions(terminal, truck_obj, "truck", "loaded", train_id,
                           oc.to_string(), "truck_entry", travel_time, env.now)


def empty_truck(env, terminal, truck_obj):
    """Empty truck just enters/leaves the gate to balance the IC>OC case."""
    ingate_request = terminal.in_gates.request()
    yield ingate_request
    travel_time = terminal.TRUCK_INGATE_TIME + random.uniform(0, terminal.TRUCK_INGATE_TIME_DEV)
    yield env.timeout(travel_time)
    terminal.in_gates.release(ingate_request)
    _record_trip_emissions(terminal, truck_obj, "truck", "empty",
                           getattr(truck_obj, "train_id", ""), "",
                           "empty_truck_entry", travel_time, env.now)


def truck_arrival(env, terminal, train_schedule):
    """Spawn all trucks that bring OCs (and any extra empty trucks) for a train."""
    train_id = train_schedule['train_id']
    truck_number = train_schedule["truck_number"]
    diesel_pct = terminal.TRUCK_DIESEL_PERCENTAGE
    num_diesel = round(truck_number * diesel_pct)
    num_electric = truck_number - num_diesel

    trucks = (
        [truck(type="Diesel", id=i, train_id=train_id) for i in range(num_diesel)] +
        [truck(type="Electric", id=i + num_diesel, train_id=train_id) for i in range(num_electric)]
    )

    oc_number = train_schedule["oc_number"]
    terminal.total_oc[train_id] = oc_number

    oc_start = terminal.OC_COUNT[train_id]
    for oc_id in range(oc_start, oc_start + oc_number):
        terminal.oc_store.put(container(type='Outbound', id=oc_id, train_id=train_id))
    terminal.OC_COUNT[train_id] = oc_start + oc_number

    truck_entries_needed = oc_number
    empty_truck_needed = max(0, len(trucks) - truck_entries_needed)

    for _ in range(truck_entries_needed):
        if not trucks:
            break
        this_truck = trucks.pop(0)
        oc = yield terminal.oc_store.get()
        env.process(truck_entry(env, terminal, this_truck, oc, train_schedule))
        yield terminal.truck_store.put(this_truck)

    for _ in range(empty_truck_needed):
        if not trucks:
            break
        this_truck = trucks.pop(0)
        env.process(empty_truck(env, terminal, this_truck))
        yield terminal.truck_store.put(this_truck)

    if not terminal.all_trucks_arrived_events[train_id].triggered:
        terminal.all_trucks_arrived_events[train_id].succeed()


# ---------------------------------------------------------------------------
# Crane processes
# ---------------------------------------------------------------------------

def crane_unload_process(env, terminal, train_schedule, track_id):
    """Drain all ICs for this train from train_ic_stores onto the chassis FilterStore."""
    train_id = train_schedule['train_id']

    def unload_crane_worker(env):
        crane_obj = yield terminal.cranes_by_track[track_id].get()
        try:
            while True:
                if not any(item.train_id == train_id for item in terminal.train_ic_stores.items):
                    break
                ic = yield terminal.train_ic_stores.get(lambda x: x.train_id == train_id)
                crane_unload_time = (terminal.CONTAINERS_PER_CRANE_MOVE_MEAN +
                                     random.uniform(0, terminal.CRANE_MOVE_DEV_TIME))
                yield env.timeout(crane_unload_time)
                yield terminal.chassis.put(ic)
                utilities.record_container_event(terminal, ic.to_string(), 'crane_unload', env.now)
                _record_load_emissions(terminal, crane_obj, "loaded", train_id,
                                       ic.to_string(), "crane_unload", env.now, track_id)
                env.process(container_process(env, terminal, train_schedule))
        finally:
            yield terminal.cranes_by_track[track_id].put(crane_obj)

    num_cranes = terminal.cranes_on_track[track_id]
    unload_processes = [env.process(unload_crane_worker(env)) for _ in range(num_cranes)]
    yield simpy.events.AllOf(env, unload_processes)

    if not terminal.train_ic_unload_events[train_id].triggered:
        terminal.train_ic_unload_events[train_id].succeed()
        print(f"[Event] All ICs for Train-{train_id} on Track-{track_id} unloaded at {env.now:.3f}")


def crane_load_process(env, terminal, track_id, train_schedule):
    """Wait for OCs to be staged on chassis, then load them onto the train."""
    train_id = train_schedule['train_id']
    yield terminal.train_start_load_events[train_id]

    def load_crane_worker(env):
        crane_obj = yield terminal.cranes_by_track[track_id].get()
        try:
            while True:
                if not any(
                    (item.type == 'Outbound' and item.train_id == train_id)
                    for item in terminal.chassis.items
                ):
                    break
                oc = yield terminal.chassis.get(
                    lambda x: x.type == 'Outbound' and x.train_id == train_id
                )
                crane_load_time = (terminal.CONTAINERS_PER_CRANE_MOVE_MEAN +
                                   random.uniform(0, terminal.CRANE_MOVE_DEV_TIME))
                yield env.timeout(crane_load_time)
                yield terminal.train_oc_stores.put(oc)
                utilities.record_container_event(terminal, oc.to_string(), 'crane_load', env.now)
                _record_load_emissions(terminal, crane_obj, "loaded", train_id,
                                       oc.to_string(), "crane_load", env.now, track_id)
        finally:
            yield terminal.cranes_by_track[track_id].put(crane_obj)

    num_cranes = terminal.cranes_on_track[track_id]
    load_processes = [env.process(load_crane_worker(env)) for _ in range(num_cranes)]
    yield simpy.events.AllOf(env, load_processes)

    if not terminal.train_end_load_events[train_id].triggered:
        terminal.train_end_load_events[train_id].succeed()
        print(f"[Event] All OCs for Train-{train_id} on Track-{track_id} loaded at {env.now:.3f}")


# ---------------------------------------------------------------------------
# Hostler dispatch
# ---------------------------------------------------------------------------

def get_hostler(terminal):
    """Prefer an already-active hostler; fall back to a parked one."""
    parked_available = len(terminal.parked_hostlers.items) > 0
    active_available = len(terminal.active_hostlers.items) > 0
    if (not active_available) and parked_available:
        assigned_hostler = terminal.parked_hostlers.get()
    else:
        assigned_hostler = terminal.active_hostlers.get()

    return assigned_hostler


def return_hostler(env, terminal, assigned_hostler, travel_time_to_active,
                   travel_time_to_parking, active_hostlers_needed=None):
    if active_hostlers_needed is None:
        active_hostlers_needed = len(terminal.active_hostlers.get_queue) > 0
    if active_hostlers_needed:
        yield env.timeout(travel_time_to_active)
        yield terminal.active_hostlers.put(assigned_hostler)
    else:
        yield env.timeout(travel_time_to_parking)
        yield terminal.parked_hostlers.put(assigned_hostler)


def check_ic_picked_complete(env, terminal, train_schedule):
    train_id = train_schedule['train_id']
    remaining_ic = sum(
        (getattr(item, 'type', None) == 'Inbound') and (getattr(item, 'train_id', None) == train_id)
        for item in terminal.chassis.items
    )
    if (remaining_ic == 0
            and terminal.train_ic_unload_events[train_id].triggered
            and not terminal.train_ic_picked_events[train_id].triggered):
        terminal.train_ic_picked_events[train_id].succeed()


# ---------------------------------------------------------------------------
# IC + OC handling
# ---------------------------------------------------------------------------
def handle_oc(env, terminal, train_schedule):
    '''
    This function is called right after IC dropped off, such that accelerating container processing.
    Note: handle_remaining_oc is designed for imbalanced container flow.
    TODO mbruchon: consolidate with handle_remaining_oc
    '''
    train_id = train_schedule['train_id']
    # 1) hostler transport OC from parking slots
    assigned_hostler = yield get_hostler(terminal)
    oc = yield terminal.parking_slots.get(
        lambda x: x.type == 'Outbound'
    )
    # TODO mbruchon: should current_veh_num come from assigned_hostler?
    current_veh_num = len(terminal.parked_hostlers.items) + 1
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
    yield terminal.chassis.put(oc)
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
    ic = yield terminal.chassis.get(
        lambda x: x.type == 'Inbound' # and x.train_id == train_id
    )
    assigned_hostler = yield get_hostler(terminal)

    # 2) Empty hostler travels to the chassis
    current_veh_num = len(terminal.parked_hostlers.items) + 1
    pickup_time, _, _, _ = distances.simulate_hostler_track_travel(
        assigned_hostler, current_veh_num, config=terminal.config
    )
    yield env.timeout(pickup_time)
    _record_trip_emissions(terminal, assigned_hostler, "hostler", "empty",
                           train_id, ic.to_string(), "hostler_to_chassis",
                           pickup_time, env.now)

    # 3) Loaded hostler travels to parking slot
    current_veh_num = len(terminal.parked_hostlers.items) + 1
    dropoff_time, _, _, _ = distances.simulate_hostler_track_travel(
        assigned_hostler, current_veh_num, config=terminal.config
    )
    yield env.timeout(dropoff_time)
    yield terminal.parking_slots.put(ic)
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
    assigned_truck = yield terminal.truck_store.get()
    truck_travel_time, _, _, _ = distances.simulate_truck_travel(
        assigned_truck, train_schedule, terminal, config=terminal.config
    )
    yield env.timeout(truck_travel_time)
    ic = yield terminal.parking_slots.get(
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
            for item in terminal.parking_slots.items
        )
        chassis_remaining = sum(
            (item.type == 'Outbound') and (item.train_id == train_id)
            for item in terminal.chassis.items
        )

        if outbound_remaining == 0 and chassis_remaining == train_schedule['oc_number']:
            print(f"Time {env.now:.3f}: All OC handled for train {train_id}.")
            if not terminal.train_oc_prepared_events[train_id].triggered:
                terminal.train_oc_prepared_events[train_id].succeed()
            return

        # 2) hostler transport OC from parking slots
        assigned_hostler = yield get_hostler(terminal)
        oc = yield terminal.parking_slots.get(
            lambda x: x.type == 'Outbound'
        )
        utilities.record_container_event(terminal, oc.to_string(), 'hostler_pickup', env.now)


        # 3) assign an empty-loaded hostler
        current_veh_num = len(terminal.parked_hostlers.items) + 1
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
        yield terminal.chassis.put(oc)
        utilities.record_container_event(terminal, oc.to_string(), 'hostler_dropoff', env.now)
        _record_trip_emissions(terminal, assigned_hostler, "hostler", "loaded",
                               train_id, oc.to_string(), "hostler_to_chassis_oc",
                               to_chassis_time, env.now)

        # 6) hostler travel back
        yield from return_hostler(env, terminal, assigned_hostler,
                                  travel_time_to_active=0,
                                  travel_time_to_parking=to_parking_time)

def truck_exit(env, terminal, truck_obj, ic, train_schedule):
    train_id = train_schedule['train_id']
    out_gate_request = terminal.out_gates.request()
    yield out_gate_request
    travel_time = terminal.TRUCK_OUTGATE_TIME + random.uniform(0, terminal.TRUCK_OUTGATE_TIME_DEV)
    yield env.timeout(travel_time)
    terminal.out_gates.release(out_gate_request)
    utilities.record_container_event(terminal, ic.to_string(), 'truck_exit', env.now)
    _record_trip_emissions(terminal, truck_obj, "truck", "loaded",
                           train_id, ic.to_string(), "truck_exit",
                           travel_time, env.now)
    yield terminal.truck_store.put(truck_obj)


# ---------------------------------------------------------------------------
# Per-train orchestration
# ---------------------------------------------------------------------------

def handle_train_departure(env, terminal, train_schedule, train_id, track_id, arrival_time):
    if env.now < train_schedule["departure_time"]:
        print(f"Time {env.now:.3f}: [EARLY] Train {train_id} departs from track {track_id}.")
    elif env.now == train_schedule["departure_time"]:
        print(f"Time {env.now:.3f}: [In Time] Train {train_id} departs from track {track_id}.")
    else:
        delay_time = env.now - train_schedule["departure_time"]
        print(f"Time {env.now:.3f}: [DELAYED] Train {train_id} delayed {delay_time:.3f}h from track {track_id}.")

    oc_start = train_schedule.get("_oc_id_start", 1)
    for oc_id in range(oc_start, oc_start + train_schedule['oc_number']):
        utilities.record_container_event(
            terminal,
            f"OC-{oc_id}-Train-{train_id}",
            'train_depart',
            env.now,
        )
    terminal.time_per_train[train_id] = env.now - arrival_time

    if not terminal.train_departed_events[train_id].triggered:
        terminal.train_departed_events[train_id].succeed()


def train_process_per_track(env, terminal, track_id, train_schedule, train_id, arrival_time):
    # Crane unload & hostler process ICs
    env.process(crane_unload_process(env, terminal, train_schedule, track_id))

    # check before crane loading
    # condition 1: all ic picked
    yield terminal.train_ic_picked_events[train_id]
    print(f"[Event]: All {train_schedule['full_cars']} ICs picked for train {train_id}.")
    # condition 2 & 3: no OCs on parking slots - OCs remaining -> process rest OCs; all OC prepared
    print(f"check # OCs on parking slots: {sum((item.type == 'Outbound') and (item.train_id == train_id) for item in terminal.parking_slots.items)}")
    if sum((item.type == 'Outbound') and (item.train_id == train_id) for item in terminal.parking_slots.items) >= 0:
	    env.process(handle_remaining_oc(env, terminal, train_schedule))
	    
    yield terminal.train_oc_prepared_events[train_id]
    
    # crane loading
    # only when 1. all_ic_picked (chassis), 2. all_oc_picked (parking slots) & 3. all_oc_prepared (chassis) satisfied -> crane loading starts
    if not terminal.train_start_load_events[train_id].triggered:
        terminal.train_start_load_events[train_id].succeed()

    env.process(crane_load_process(env, terminal, track_id=track_id, train_schedule=train_schedule))
    yield terminal.train_end_load_events[train_id]

    handle_train_departure(env, terminal, train_schedule, train_id, track_id, arrival_time)
    yield terminal.tracks.put(track_id)

def process_train_arrival(env, terminal, train_schedule):
    train_id = train_schedule["train_id"]
    arrival_time = train_schedule["arrival_time"]
    terminal.all_trucks_arrived_events[train_id] = env.event()

    # Initialize per-train counters
    terminal.IC_COUNT.setdefault(train_id, 1)
    terminal.OC_COUNT.setdefault(train_id, 1)
    train_schedule["_oc_id_start"] = terminal.OC_COUNT[train_id]

    # Trucks bring OCs before the train arrives
    env.process(truck_arrival(env, terminal, train_schedule))

    # Wait for trucks, then enforce timetable arrival
    yield terminal.all_trucks_arrived_events[train_id]
    if env.now <= arrival_time:
        yield env.timeout(arrival_time - env.now)
        print(f"Time {env.now:.3f}: [In Time] Train {train_id}.")
        delay_time = 0
    else:
        delay_time = env.now - arrival_time
        print(f"Time {env.now:.3f}: [DELAYED] Train {train_id} delayed {delay_time:.3f}h.")
    terminal.train_delay_time[train_id] = delay_time
    terminal.train_pool_stores.put(train_id)

    # Track assignment
    track_id = yield terminal.tracks.get()
    if track_id is None:
        print(f"Time {env.now:.3f}: Train {train_id} waiting for an available track.")
        return

    assigned_train_id = yield terminal.train_pool_stores.get()
    print(f"Time {env.now:.3f}: Train {assigned_train_id} assigned to track {track_id}.")

    # Stage ICs on this train into the per-train IC store
    ic_start = terminal.IC_COUNT[train_id]
    for ic_id in range(ic_start, ic_start + train_schedule['full_cars']):
        ic = container(type='Inbound', id=ic_id, train_id=train_id)
        terminal.train_ic_stores.put(ic)
        utilities.record_container_event(terminal, ic.to_string(), 'train_arrival_expected', arrival_time)
        utilities.record_container_event(terminal, ic.to_string(), 'train_arrival_actual', env.now)
    terminal.IC_COUNT[train_id] = ic_start + train_schedule['full_cars']

    utilities.initialize_train_events(env, terminal, train_id)
    env.process(train_process_per_track(env, terminal, track_id, train_schedule, train_id, arrival_time))


def run_simulation(
        train_consist_plan: pl.DataFrame,
        terminal: str,
        out_path=None):
    '''
    Run a multi-train LIFTS simulation for the given terminal.
    '''
    # Reset the module-level emissions buffer so repeat invocations are clean
    emission_records.clear()

    terminal_config = utilities.load_config(utilities.resources_root() / "config.yaml")
    terminal_layout = distances.get_layout(terminal_config)

    random.seed(42)

    train_timetable = utilities.build_train_timetable(train_consist_plan, terminal, as_dicts=True)
    truck_number = max([entry['truck_number'] for entry in train_timetable])
    chassis_count = max([entry['empty_cars'] + entry['full_cars'] for entry in train_timetable])
    env = simpy.Environment()

    terminal = Terminal(env, 
        config=terminal_config,
        layout=terminal_layout, 
        truck_capacity=truck_number, 
        chassis_count=chassis_count)

    print("\nTrain timetable:")
    for schedule in train_timetable:
        print(schedule)
        env.process(process_train_arrival(env, terminal, schedule))

    num_tracks = terminal.track_number
    num_cranes = num_tracks * terminal.cranes_per_track
    num_hostlers = terminal.hostler_number

    print("*" * 50)
    print(f"Tracks: {num_tracks}; Cranes: {num_cranes}; Hostlers: {num_hostlers}")
    print("*" * 50)

    # When a train_consist_plan is supplied, simulate the entire plan regardless
    # of the config's simulation length. Otherwise honor the configured horizon.
    if train_consist_plan is not None:
        env.run()
    else:
        env.run(until=terminal_config["simulation"]["length"])

    # Create DataFrame for container events
    container_data = (
        pl.from_dicts(
            [dict(event, **{'container_id': container_id}) for container_id, event in terminal.container_events.items()],
            infer_schema_length=None
        )
        .lazy()
        .sort("container_id")
        .select(pl.col("container_id"), pl.exclude("container_id"))
        .with_columns(
            pl.when(pl.col("truck_exit").is_not_null() & pl.col("train_arrival_expected").is_not_null())
                .then(pl.col("truck_exit") - pl.col("train_arrival_expected"))
                .when(pl.col("train_depart").is_not_null())
                .then(pl.col("crane_load") - pl.col("truck_arrival"))
                .otherwise(None)
                .alias("container_processing_time"),
            pl.col("container_id").str.extract(r"Train-(\d+)").cast(pl.Int64).alias("train_id"),
            pl.col("container_id").str.starts_with("IC").alias("is_ic")
        )
    )

    # OC train actual arrival time
    train_arrival_df = (container_data
        .filter(
            pl.col("is_ic"),
            pl.col("train_arrival_actual").is_not_null()
        )
        .group_by("train_id")
        .agg(pl.col("train_arrival_actual").mean())
    )
    # OC train expected arrival time
    train_arrival_expected_df = (container_data
        .filter(
            pl.col("is_ic"),
            pl.col("train_arrival_expected").is_not_null()
        )
        .group_by("train_id")
        .agg(pl.col("train_arrival_expected").mean())
    )
    container_data = (container_data
        .join(train_arrival_df, on="train_id", how="left")
        .join(train_arrival_expected_df, on="train_id", how="left")
        .rename({
            "train_arrival_actual_right": "train_arrival_actual_oc",
            "train_arrival_expected_right": "train_arrival_expected_oc"
        })
        .drop("is_ic", "train_id")
    ).collect()

    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        daily_throughput = 2 * terminal.train_batch_size * terminal.track_number
        container_data.write_excel(out_path / f"simulation_container_{daily_throughput}_track_{num_tracks}_crane_{num_cranes}_hostler_{num_hostlers}_results.xlsx")
        if emission_records:
            emission_records_df = pl.DataFrame(emission_records)
            utilities.save_emission_results(
                emission_records_df,
                out_path / f"emission_container_{daily_throughput}_track_{num_tracks}_crane_{num_cranes}_hostler_{num_hostlers}_results.xlsx",
                filetype="xlsx",
            )
    return container_data



if __name__ == "__main__":
    consist_plan = (pl.read_csv(utilities.package_root() / 'resources' / 'train_consist_plan.csv')
        .with_columns(pl.lit("Intermodal").alias("Train_Type"))
    )
    run_simulation(
        train_consist_plan=consist_plan,
        terminal = "Allouez",
        out_path = utilities.package_root() / 'demos' / 'lifts' / 'demos' / 'starter_demo' / 'results'
    )
