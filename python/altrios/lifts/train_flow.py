"""Per-train orchestration: arrival, track assignment, unload/load coordination, departure."""
from altrios.lifts import utilities
from altrios.lifts.classes import container
from altrios.lifts.containers import handle_remaining_oc
from altrios.lifts.cranes import crane_load_process, crane_unload_process
from altrios.lifts.truck_gate import truck_arrival


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
    terminal.state.time_per_train[train_id] = env.now - arrival_time

    if not terminal.state.train_departed_events[train_id].triggered:
        terminal.state.train_departed_events[train_id].succeed()


def train_process_per_track(env, terminal, track_id, train_schedule, train_id, arrival_time):
    # Crane unload & hostler process ICs
    env.process(crane_unload_process(env, terminal, train_schedule, track_id))

    # check before crane loading
    # condition 1: all ic picked
    yield terminal.state.train_ic_picked_events[train_id]
    print(f"[Event]: All {train_schedule['full_cars']} ICs picked for train {train_id}.")
    # condition 2 & 3: no OCs on parking slots - OCs remaining -> process rest OCs; all OC prepared
    print(f"check # OCs on parking slots: {sum((item.type == 'Outbound') and (item.train_id == train_id) for item in terminal.state.parking_slots.items)}")
    if sum((item.type == 'Outbound') and (item.train_id == train_id) for item in terminal.state.parking_slots.items) >= 0:
	    env.process(handle_remaining_oc(env, terminal, train_schedule))
	    
    yield terminal.state.train_oc_prepared_events[train_id]
    
    # crane loading
    # only when 1. all_ic_picked (chassis), 2. all_oc_picked (parking slots) & 3. all_oc_prepared (chassis) satisfied -> crane loading starts
    if not terminal.state.train_start_load_events[train_id].triggered:
        terminal.state.train_start_load_events[train_id].succeed()

    env.process(crane_load_process(env, terminal, track_id=track_id, train_schedule=train_schedule))
    yield terminal.state.train_end_load_events[train_id]

    handle_train_departure(env, terminal, train_schedule, train_id, track_id, arrival_time)
    yield terminal.state.tracks.put(track_id)


def process_train_arrival(env, terminal, train_schedule):
    train_id = train_schedule["train_id"]
    arrival_time = train_schedule["arrival_time"]
    terminal.state.all_trucks_arrived_events[train_id] = env.event()

    # Initialize per-train counters
    terminal.state.IC_COUNT.setdefault(train_id, 1)
    terminal.state.OC_COUNT.setdefault(train_id, 1)
    train_schedule["_oc_id_start"] = terminal.state.OC_COUNT[train_id]

    # Trucks bring OCs before the train arrives
    env.process(truck_arrival(env, terminal, train_schedule))

    # Wait for trucks, then enforce timetable arrival
    yield terminal.state.all_trucks_arrived_events[train_id]
    if env.now <= arrival_time:
        yield env.timeout(arrival_time - env.now)
        print(f"Time {env.now:.3f}: [In Time] Train {train_id}.")
        delay_time = 0
    else:
        delay_time = env.now - arrival_time
        print(f"Time {env.now:.3f}: [DELAYED] Train {train_id} delayed {delay_time:.3f}h.")
    terminal.state.train_delay_time[train_id] = delay_time
    terminal.state.train_pool_stores.put(train_id)

    # Track assignment
    track_id = yield terminal.state.tracks.get()
    if track_id is None:
        print(f"Time {env.now:.3f}: Train {train_id} waiting for an available track.")
        return

    assigned_train_id = yield terminal.state.train_pool_stores.get()
    print(f"Time {env.now:.3f}: Train {assigned_train_id} assigned to track {track_id}.")

    # Stage ICs on this train into the per-train IC store
    ic_start = terminal.state.IC_COUNT[train_id]
    for ic_id in range(ic_start, ic_start + train_schedule['full_cars']):
        ic = container(type='Inbound', id=ic_id, train_id=train_id)
        terminal.state.train_ic_stores.put(ic)
        utilities.record_container_event(terminal, ic.to_string(), 'train_arrival_expected', arrival_time)
        utilities.record_container_event(terminal, ic.to_string(), 'train_arrival_actual', env.now)
    terminal.state.IC_COUNT[train_id] = ic_start + train_schedule['full_cars']

    utilities.initialize_train_events(env, terminal, train_id)
    env.process(train_process_per_track(env, terminal, track_id, train_schedule, train_id, arrival_time))
