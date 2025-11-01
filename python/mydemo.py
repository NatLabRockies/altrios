import simpy
import random
import polars as pl
from altrios.lifts import utilities
from demo_parameters import *
from distances import *
from altrios.lifts.dictionary import *
from altrios.lifts.schedule import *
from double_track_vehicle_performance import *


class Terminal:
    def __init__(self, env, truck_capacity, chassis_count):
        self.env = env
        self.tracks = simpy.Store(env, capacity=state.TRACK_NUMBER)
        for track_id in range(1, self.tracks.capacity + 1):
            self.tracks.put(track_id)
        # Example of number of cranes per track
        cranes_per_track = {
            1: 1,
            2: 1
        }
        self.cranes = simpy.FilterStore(env, sum(cranes_per_track.values()))

        self.train_ic_unload_events = {}
        self.train_ic_unload_count = {}  # condition for train_ic_picked
        self.train_ic_picked_events = {}  # condition 1 for crane loading
        self.train_oc_prepared_events = {}  # condition 2 for crane loading
        self.train_start_load_events = {}  # condition 1 for train departure
        self.train_end_load_events = {}  # condition 2 for train departure

        self.IC_COUNT = {}
        self.OC_COUNT = {}
        self.total_ic = {}
        self.total_oc = {}

        for item in cranes_per_track.items():
            track_id = item[0]
            cranes_on_this_track = item[1]
            for crane_number in range(1, cranes_on_this_track + 1):
                self.cranes.put(crane(type='Hybrid', id=crane_number, track_id=track_id))

        self.train_ic_stores = simpy.FilterStore(env, capacity=9999)
        self.train_oc_stores = simpy.Store(env, capacity=9999)
        self.in_gates = simpy.Resource(env, state.IN_GATE_NUMBERS)
        self.out_gates = simpy.Resource(env, state.OUT_GATE_NUMBERS)
        self.oc_store = simpy.Store(env, capacity=9999)
        self.parking_slots = simpy.FilterStore(env)  # store ic and oc in the parking area
        self.chassis = simpy.FilterStore(env, capacity=9999)
        self.hostlers = simpy.Store(env, capacity=state.HOSTLER_NUMBER)
        self.truck_store = simpy.Store(env)
        # Hostler setup
        hostler_diesel = round(state.HOSTLER_NUMBER * state.HOSTLER_DIESEL_PERCENTAGE)
        hostler_electric = state.HOSTLER_NUMBER - hostler_diesel
        hostlers = [hostler(id=i, type="Diesel") for i in range(hostler_diesel)] + \
                   [hostler(id=i + hostler_diesel, type="electric") for i in range(hostler_electric)]
        for hostler_id in hostlers:
            self.hostlers.put(hostler_id)
        print(f"Hostlers: {hostlers}")


def record_container_event(container, event_type, timestamp):
    global state
    if type(container) is str:
        container_string = container
    else:
        container_string = container.to_string()

    if container_string not in state.container_events:
        state.container_events[container_string] = {}
    state.container_events[container_string][event_type] = timestamp


def save_vehicle_and_performance_metrics(state):
    out_path = altrios.lifts.utilities.package_root() / 'demos' / 'double_track_results'

    container_excel_path = out_path / f"double_track_simulation_crane_{state.CRANE_NUMBER}_hostler_{state.HOSTLER_NUMBER}.xlsx"
    vehicle_excel_path = out_path / f"double_track_vehicle_log_crane_{state.CRANE_NUMBER}_hostler_{state.HOSTLER_NUMBER}.xlsx"

    if not container_excel_path.exists():
        print(f"[Error] Container Excel not found: {container_excel_path}")
        return
    if not vehicle_excel_path.exists():
        print(f"[Error] Vehicle Excel not found: {vehicle_excel_path}")
        return

    ic_time, oc_time, total_time = calculate_container_processing_time(container_excel_path)
    ic_energy, oc_energy, total_energy = calculate_vehicle_energy(vehicle_excel_path)
    print_and_save_metrics(ic_time, oc_time, total_time, ic_energy, oc_energy, total_energy)
    return print_and_save_metrics(ic_time, oc_time, total_time, ic_energy, oc_energy, total_energy)


def emission_calculation(status, vehicle_category, vehicle, travel_time):
    global state
    vehicle_category = vehicle_category.capitalize()
    vehicle_type = vehicle.type.capitalize()
    if status == 'loaded':
        emission_unit = state.FULL_EMISSIONS_RATES[vehicle_category][vehicle_type]
    if status == 'empty':
        emission_unit = state.IDLE_EMISSIONS_RATES[vehicle_category][vehicle_type]
    emissions = emission_unit * travel_time

    return emissions


def truck_entry(env, terminal, truck, oc, train_schedule):
    global state
    ingate_request = terminal.in_gates.request()
    yield ingate_request
    if state.log_level > loggingLevel.NONE:
        print(f"Time {env.now:.3f}: {truck} loaded with {oc} passed the in-gate and is entering the terminal.")

    # Assume each truck takes 1 OC, and drop OC to the closest parking lot according to triangular distribution
    # Assign IDs for OCs
    truck_travel_time = state.TRUCK_INGATE_TIME + random.uniform(0, state.TRUCK_INGATE_TIME_DEV)
    yield env.timeout(truck_travel_time)  # truck passing gate time: 1 sec (demo_parameters.TRUCK_INGATE_TIME and TRUCK_INGATE_TIME_DEV)
    terminal.in_gates.release(ingate_request)
    emissions = emission_calculation('loaded', 'truck', truck, truck_travel_time)
    record_vehicle_event('truck', truck, f'entry_OC_{oc.id}', 'loaded', truck_travel_time, emissions, 'end', env.now)
    if state.log_level > loggingLevel.NONE:
        print(f"Time {env.now:.3f}: {truck} placed OC {oc} at parking slot.")
    record_container_event(oc, 'truck_arrival', env.now)

    # Calculate truck speed according to the current density
    truck_travel_time = simulate_truck_travel(truck, train_schedule, terminal, total_lane_length, d_t_min, d_t_max)
    emissions = emission_calculation('loaded', 'truck', truck, truck_travel_time)
    record_vehicle_event('truck', truck, f'entry_OC_{oc.id}', 'loaded',truck_travel_time, emissions, 'end', env.now)

    # Assume each truck takes 1 OC, and drop OC to the closest parking lot according to triangular distribution
    # Assign IDs for OCs
    print(f"Time {env.now}: Truck {truck} placed OC {oc} at parking slot.")
    record_container_event(oc, 'truck_dropoff', env.now)
    yield terminal.parking_slots.put(oc)
    emissions = emission_calculation('loaded', 'truck', truck, truck_travel_time)
    record_vehicle_event('truck', truck, f'dropoff_OC_{oc.id}', 'loaded', truck_travel_time, emissions, 'end', env.now)


def empty_truck(env, terminal, truck_id):
    global state
    ingate_request = terminal.in_gates.request()
    yield ingate_request
    if state.log_level > loggingLevel.NONE:
        print(f"Time {env.now:.3f}: {truck_id} passed the in-gate and is entering the terminal with empty loading")
    truck_travel_time = state.TRUCK_INGATE_TIME + random.uniform(0, state.TRUCK_INGATE_TIME_DEV)
    yield env.timeout(
        truck_travel_time)  # truck passing gate time: 1 sec (demo_parameters.TRUCK_INGATE_TIME and TRUCK_INGATE_TIME_DEV)
    terminal.in_gates.release(ingate_request)
    # Note the arrival of empty trucks will not be recorded due to excel output dimensions
    emissions = emission_calculation('empty', 'truck', truck_id, truck_travel_time)
    record_vehicle_event('truck', truck_id, 'pass_gate', 'empty', truck_travel_time, emissions, 'end', env.now)


def truck_arrival(env, terminal, train_schedule, all_trucks_arrived_event):
    global state
    truck_number = train_schedule["truck_number"]
    num_diesel = round(truck_number * state.TRUCK_DIESEL_PERCENTAGE)
    num_electric = truck_number - num_diesel

    trucks = [truck(type="Diesel", id=i, train_id=train_schedule['train_id']) for i in range(num_diesel)] + \
             [truck(type="Electric", id=i + num_diesel, train_id=train_schedule['train_id']) for i in
              range(num_electric)]

    # only shows trucks for one train, which may cause stuck?
    print("trucks:", trucks)

    terminal.total_oc[train_schedule['train_id']] = train_schedule["oc_number"]

    for oc_id in range(terminal.OC_COUNT[train_schedule['train_id']],
                       terminal.OC_COUNT[train_schedule['train_id']] + terminal.total_oc[train_schedule['train_id']]):
        terminal.oc_store.put(container(type='Outbound', id=oc_id, train_id=train_schedule['train_id']))
        oc_id += 1
    print(f"Time {env.now:.3f}: oc_store has {terminal.oc_store.items}", )

    truck_entries_needed = train_schedule['oc_number']
    empty_truck_needed = len(trucks) - truck_entries_needed

    for i in range(truck_entries_needed):
        this_truck = trucks.pop(0)
        oc = yield terminal.oc_store.get()
        env.process(truck_entry(env, terminal, this_truck, oc, train_schedule))
        yield terminal.truck_store.put(this_truck)

    for i in range(empty_truck_needed):
        this_truck = trucks.pop(0)
        env.process(empty_truck(env, terminal, this_truck))
        yield terminal.truck_store.put(this_truck)

    all_trucks_arrived_event.succeed()  # if all_trucks_arrived_event is triggered, train is allowed to enter


def check_ic_picked_complete(env, terminal, train_schedule):
    inbound_count = sum(getattr(item, 'type', None) == 'Inbound' and getattr(item, 'train_id', None) == train_schedule['train_id'] for
        item in terminal.chassis.items)
    print(f"ic for train-{train_schedule['train_id']} on chassis:", inbound_count)
    if inbound_count == 0:
        if terminal.train_ic_unload_events[train_schedule['train_id']].triggered:
            terminal.train_ic_picked_events[train_schedule['train_id']].succeed()
            print(f"Time {env.now:.3f}: All ICs for {train_schedule['train_id']} are picked up by hostlers.")


def crane_unload_process(env, terminal, train_schedule, track_id):
    global state
    terminal.train_ic_unload_count[train_schedule['train_id']] = 0

    terminal.total_ic[train_schedule['train_id']] = train_schedule["full_cars"]
    print(f"ic_item on train {train_schedule['train_id']} train_ic_stores has {terminal.train_ic_stores.items}")

    start_id = terminal.IC_COUNT[train_schedule['train_id']]
    end_id = terminal.IC_COUNT[train_schedule['train_id']] + terminal.total_ic[train_schedule['train_id']]
    for ic_id in range(start_id, end_id):
        print(f"Crane request {ic_id} for Train-{train_schedule['train_id']}")
        crane_item = yield terminal.cranes.get(lambda x: x.track_id == track_id)
        print(f"Time {env.now:.3f}: Crane {crane_item.to_string()} for Train-{train_schedule['train_id']}")
        ic_item = yield terminal.train_ic_stores.get(
            lambda x: x.train_id == train_schedule['train_id'])  # pick up the first ic_item, all ic there

        print(f"Time {env.now:.3f}: Crane {crane_item.to_string()} starts unloading IC-{ic_item.id}-Train-{train_schedule['train_id']}")
        crane_unload_move_time = state.CONTAINERS_PER_CRANE_MOVE_MEAN + random.uniform(0, state.CRANE_MOVE_DEV_TIME)
        yield env.timeout(crane_unload_move_time)
        record_container_event(ic_item.to_string(), 'crane_unload', env.now)
        record_vehicle_event('crane', crane_item, f"unload_IC_{ic_item.id}-Train-{train_schedule['train_id']}", 'full', crane_unload_move_time,
                             crane_unload_move_time * state.FULL_EMISSIONS_RATES['Crane'][crane_item.type], 'end',
                             env.now)
        print(f"Time {env.now:.3f}: Crane {crane_item.to_string()} finishes unloading IC-{ic_item.id}-Train-{train_schedule['train_id']} onto chassis")
        yield terminal.chassis.put(ic_item)
        print(f"Before put: cranes {len(terminal.cranes.items)} / crane capacity {terminal.cranes.capacity}")
        yield terminal.cranes.put(crane_item)
        print(f"After put: cranes {len(terminal.cranes.items)} / crane capacity {terminal.cranes.capacity}")

        terminal.train_ic_unload_count[train_schedule['train_id']] += 1
        print(f"Train {train_schedule['train_id']} ic unload count: {terminal.train_ic_unload_count[train_schedule['train_id']]}")
        env.process(container_process(env, terminal, train_schedule))

    if terminal.train_ic_unload_count[train_schedule['train_id']] == terminal.total_ic[train_schedule['train_id']]:
        terminal.train_ic_unload_events[train_schedule['train_id']].succeed()
        print(f"Time {env.now:.3f}: All ICs from Train-{train_schedule['train_id']} have been unloaded onto the chassis.")
        check_ic_picked_complete(env, terminal, train_schedule)


def container_process(env, terminal, train_schedule):
    global state
    '''
    It is designed to transfer both inbound and outbound containers.
    The main simulation process is as follows:
    1. A hostler picks up an IC, and drops off IC at parking slot.
    2. A truck picks up the IC, and leaves the gate
    3. The hostler picks up an OC, and drops off OC at the chassis.
    4. Once all OCs are prepared (all_oc_prepared), the crane starts loading (other function).
    '''
    assigned_hostler = yield terminal.hostlers.get()

    # Hostler puts IC to the closest parking lot
    current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
    hostler_travel_time_to_parking = 0 # simulate_hostler_travel(assigned_hostler.to_string(), current_veh_num, total_lane_length, d_h_min, d_h_max)
    yield env.timeout(hostler_travel_time_to_parking)
    # Hostler picks up IC from chassis
    print("terminal.chassis before hostler picking up IC:", terminal.chassis.items)
    ic = yield terminal.chassis.get(lambda x: x.type == 'Inbound')
    print(f"    Time {env.now:.3f}: Hostler {assigned_hostler.to_string()} picked up IC-{ic.id}-Train-{train_schedule['train_id']} and is heading to parking slot.")
    print(f"Chassis (after getting IC-{ic.id} for Train-{ic.train_id}): {terminal.chassis.items}")
    yield terminal.parking_slots.put(ic)
    record_container_event(ic, 'hostler_pickup', env.now)
    emissions = emission_calculation('empty', 'hostler', hostler, hostler_travel_time_to_parking)
    record_vehicle_event('hostler', assigned_hostler, f"pickup_IC_{ic.id}-Train-{train_schedule['train_id']}", 'empty',
                         hostler_travel_time_to_parking, emissions, 'end', env.now)

    yield terminal.train_ic_unload_events[train_schedule['train_id']]
    check_ic_picked_complete(env, terminal, train_schedule)

    # Test: status for chassis
    print("Containers on chassis (hostler picking-up IC):", terminal.chassis.items)

    # Hostler drop off IC to parking slot
    current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
    hostler_travel_time_to_parking = 0 #simulate_hostler_travel(assigned_hostler.to_string(), current_veh_num, total_lane_length, d_h_min, d_h_max)
    yield env.timeout(hostler_travel_time_to_parking)  # update: time calculated by density-travel_time function
    print(f"Time {env.now:.3f}: Hostler {assigned_hostler.to_string()} dropped off IC {ic.id}-Train-{train_schedule['train_id']} at parking slot.")
    record_container_event(ic, 'hostler_dropoff', env.now)
    emissions = emission_calculation('loaded', 'hostler', assigned_hostler, hostler_travel_time_to_parking)
    record_vehicle_event('hostler', assigned_hostler, f"dropoff_IC_{ic.id}-Train-{train_schedule['train_id']}",
                         'loaded', hostler_travel_time_to_parking, emissions, 'end', env.now)

    # Assign a truck to pick up IC
    assigned_truck = yield terminal.truck_store.get()
    ic = yield terminal.parking_slots.get(lambda x: x.type == 'Inbound')
    print(f"Time {env.now:.3f}: Truck {assigned_truck.to_string()} is assigned to {ic.to_string()} for exit.")
    truck_travel_time = simulate_truck_travel(assigned_truck, train_schedule, terminal, total_lane_length, d_t_min, d_t_max)
    record_container_event(ic, 'truck_pickup', env.now)
    emissions = emission_calculation('empty', 'truck', assigned_truck, truck_travel_time)
    record_vehicle_event('truck', assigned_truck, f"pickup_{ic.to_string()}", 'empty', truck_travel_time, emissions, 'end', env.now)

    # Truck queue and exit the gate
    env.process(truck_exit(env, terminal, assigned_truck, ic, train_schedule))

    # Assign a hostler to pick up an OC
    if sum(item.type == 'Outbound' for item in terminal.parking_slots.items) > 0:
        oc = yield terminal.parking_slots.get(lambda x: x.type == 'Outbound')
        print(f"Time {env.now:.3f}: Hostler {assigned_hostler} is going to pick up OC {oc} from a parking slot")

        # The hostler picks up an OC
        hostler_reposition_travel_time = simulate_reposition_travel(assigned_hostler, current_veh_num,total_lane_length,d_h_min, d_h_max)
        yield env.timeout(hostler_reposition_travel_time)
        print(f"Time {env.now:.3f}: Hostler {assigned_hostler} picked up OC {oc} and is returning to the terminal")
        record_container_event(oc, 'hostler_pickup', env.now)
        emissions = emission_calculation('empty', 'hostler', assigned_hostler, hostler_reposition_travel_time)
        record_vehicle_event('hostler', assigned_hostler, f'pickup_OC_{oc}', 'empty',
                             hostler_reposition_travel_time, emissions, 'end', env.now)

        # Test: Containers after hostler picking-up OC
        print("Containers on chassis (after hostler picking-up OC):", terminal.chassis.items)
        print("# of IC", sum(str(item).isdigit() for item in terminal.chassis.items))

        # The hostler drops off OC at the chassis
        current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
        hostler_travel_time_to_chassis = 0 # simulate_hostler_travel(assigned_hostler, current_veh_num, total_lane_length, d_h_min, d_h_max)
        yield env.timeout(hostler_travel_time_to_chassis)
        emissions = emission_calculation('empty', 'hostler', assigned_hostler, hostler_travel_time_to_chassis)
        yield terminal.chassis.put(oc)
        print(f"Time {env.now:.3f}: Hostler {assigned_hostler} dropped off OC {oc} onto chassis")
        record_container_event(oc, 'hostler_dropoff', env.now)
        record_vehicle_event('hostler', assigned_hostler, f'dropoff_{oc}', 'loaded', hostler_travel_time_to_chassis, emissions, 'end', env.now)

        outbound_count = sum((getattr(item, 'type', None) == 'Outbound') and (
                    getattr(item, 'train_id', None) == train_schedule['train_id']) for item in terminal.chassis.items)
        if outbound_count == train_schedule['oc_number'] and (
                terminal.train_oc_prepared_events[train_schedule['train_id']].triggered == False):
            terminal.train_oc_prepared_events[train_schedule['train_id']].succeed()
            print(f"Time {env.now:.3f}: All OCs for {train_schedule['train_id']} are ready on chassis.")

    # IC < OC: ICs are all picked up and still have OCs remaining
    inbound_remaining = sum(
        (item.type == 'Inbound') & (item.train_id == train_schedule['train_id']) for item in terminal.chassis.items)
    outbound_remaining = sum((item.type == 'Outbound') & (item.train_id == train_schedule['train_id']) for item in
                             terminal.parking_slots.items)
    if (inbound_remaining == 0) and (outbound_remaining == 0):
            current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
            hostler_travel_time_to_parking = 0 # simulate_hostler_travel(assigned_hostler, current_veh_num, total_lane_length, d_h_min, d_h_max)
            emissions = emission_calculation('empty', 'hostler', assigned_hostler, hostler_travel_time_to_parking)
            yield env.timeout(hostler_travel_time_to_parking)
            print(f"Time {env.now:.3f}: Hostler {assigned_hostler.to_string()} return to parking slot.")
            yield terminal.hostlers.put(assigned_hostler)
            record_vehicle_event('hostler', assigned_hostler, f'back_parking', 'empty', hostler_travel_time_to_parking, emissions,
                                 'end', env.now)

def handle_remaining_oc(env, terminal, train_schedule):
    outbound_remaining = sum((item.type == 'Outbound') & (item.train_id == train_schedule['train_id']) for item in
        terminal.parking_slots.items)
    
    for i in range(outbound_remaining):
        assigned_hostler = yield terminal.hostlers.get()

        # Hostler puts IC to the closest parking lot
        current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
        hostler_travel_time_to_parking = 0 # simulate_hostler_travel(assigned_hostler.to_string(), current_veh_num, total_lane_length, d_h_min, d_h_max)
        yield env.timeout(hostler_travel_time_to_parking)

        print(f"ICs are prepared, but OCs remaining: {terminal.oc_store.items}")
        # Repeat hostler picking-up OC process only, until all remaining OCs are transported
        oc = yield terminal.parking_slots.get(lambda x: x.type == 'Outbound')
        print(f"The OC is {oc}")
        pickup_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
        hostler_reposition_travel_time = simulate_reposition_travel(assigned_hostler, pickup_veh_num,
                                                                    total_lane_length, d_r_min, d_r_max)
        yield env.timeout(hostler_reposition_travel_time)
        print(f"Time {env.now:.3f}: Hostler {assigned_hostler} picked up OC {oc} and is returning to the terminal")
        record_container_event(oc, 'hostler_pickup', env.now)
        emissions = emission_calculation('empty', 'hostler', assigned_hostler, hostler_reposition_travel_time)
        record_vehicle_event('hostler', assigned_hostler, f'pickup_OC_{oc}', 'empty', hostler_reposition_travel_time, emissions, 'end', env.now)

        dropoff_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
        travel_time_to_chassis = simulate_reposition_travel(assigned_hostler, dropoff_veh_num, total_lane_length, d_h_min, d_h_max)
        yield env.timeout(travel_time_to_chassis)
        print(f"Time {env.now:.3f}: Hostler {assigned_hostler} dropped off OC {oc} onto chassis")
        yield terminal.chassis.put(oc)
        record_container_event(oc, 'hostler_dropoff', env.now)
        emissions = emission_calculation('loaded', 'hostler', assigned_hostler, travel_time_to_chassis)
        record_vehicle_event('hostler', assigned_hostler, f'dropoff_OC_{oc}', 'loaded', travel_time_to_chassis, emissions, 'end', env.now)

        print("chassis (oc_remaining):", terminal.chassis.items)
        print(f"hostler (oc_remaining): {terminal.hostlers.items}")
        oc_count = sum(1 for item in terminal.train_oc_stores.items if item.train_id == train_schedule['train_id'])

        hostler_travel_time_to_parking = 0 # simulate_hostler_travel(assigned_hostler, current_veh_num, total_lane_length, d_h_min, d_h_max)
        emissions = emission_calculation('empty', 'hostler', assigned_hostler, hostler_travel_time_to_parking)
        yield env.timeout(hostler_travel_time_to_parking)
        print(f"Time {env.now:.3f}: Hostler {assigned_hostler.to_string()} return to parking slot.")
        yield terminal.hostlers.put(assigned_hostler)
        
        if oc_count == train_schedule['oc_number']:
            terminal.train_oc_prepared_events[train_schedule['train_id']].succeed()
            print("chassis (all_oc_prepared):", terminal.chassis.items)
            print(f"hostler (all_oc_prepared): {terminal.hostlers.items}")
            print(f"Time {env.now:.3f}: All OCs are ready on chassis.")

def truck_exit(env, terminal, truck, ic, train_schedule):
    global state
    out_gate_request = terminal.out_gates.request()
    yield out_gate_request
    print(
        f"Time {env.now:.3f}: Truck {truck.id} with IC-{ic.id}-Train-{train_schedule['train_id']} is passing through the out-gate and leaving the terminal")
    truck_travel_time = state.TRUCK_OUTGATE_TIME + random.uniform(0, state.TRUCK_OUTGATE_TIME_DEV)
    yield env.timeout(truck_travel_time)
    terminal.out_gates.release(out_gate_request)
    record_container_event(ic, 'truck_exit', env.now)
    emissions = emission_calculation('loaded', 'truck', truck, truck_travel_time)
    record_vehicle_event('truck', truck, f"leave_gate_IC_{ic.id}-Train-{train_schedule['train_id']}", 'loaded', truck_travel_time,
                         emissions, 'end', env.now)

    yield terminal.truck_store.put(truck)


def crane_load_process(env, terminal, track_id, train_schedule):
    global state
    # yield start_load_event
    yield terminal.train_start_load_events[train_schedule['train_id']]
    print(f"Time {env.now:.3f}: Starting loading process onto the train-{train_schedule['train_id']}.")

    while sum(item.type == 'Outbound' and item.train_id == train_schedule['train_id'] for item in terminal.chassis.items) > 0:    # if there still has OC on chassis
        crane_item = yield terminal.cranes.get(lambda x: x.track_id == track_id)
        oc = yield terminal.chassis.get(lambda x: x.type == 'Outbound' and x.train_id == train_schedule['train_id'])  # obtain an OC from chassis
        print(f"Time {env.now:.3f}: Crane {crane_item.to_string()} starts loading {oc} onto the train.")
        crane_load_move_time = state.CONTAINERS_PER_CRANE_MOVE_MEAN + random.uniform(0, state.CRANE_MOVE_DEV_TIME)
        yield env.timeout(crane_load_move_time)  # loading time, depends on container parameter**
        record_container_event(oc, 'crane_load', env.now)
        record_vehicle_event('crane', (crane_item.id, crane_item.type), f'load_{oc}', 'full', crane_load_move_time,
                             crane_load_move_time * state.FULL_EMISSIONS_RATES['Crane'][crane_item.type], 'end', env.now)
        print(f"Time {env.now:.3f}: Crane {crane_item.to_string()} finished loading {oc} onto the train.")
        yield terminal.train_oc_stores.put(oc)
        yield terminal.cranes.put(crane_item)

    terminal.train_end_load_events[train_schedule['train_id']].succeed()
    print(f"Time {env.now:.3f}: All OCs loaded. Train is fully loaded and ready to depart.")
    print("Containers on chassis (after loading OCs):", terminal.chassis.items)


def process_train_arrival(env, terminal, train_schedule):
    global state

    train_id = train_schedule["train_id"]
    arrival_time = train_schedule["arrival_time"]
    departure_time = train_schedule["departure_time"]
    oc_needed = train_schedule["oc_number"]
    total_ic = train_schedule["full_cars"]

    # Create events as processing conditions
    all_trucks_arrived_event = env.event()  # condition for train arrival

    # Initialize dictionary
    delay_list = {}

    # Initialize IC & OC count
    terminal.IC_COUNT[train_id] = 1
    terminal.OC_COUNT[train_id] = 1

    print(f"IC counter: {terminal.IC_COUNT}")
    print(f"OC counter: {terminal.OC_COUNT}")

    # All trucks arrive before train arrives
    env.process(truck_arrival(env, terminal, train_schedule, all_trucks_arrived_event))

    print("Current available track has ", terminal.tracks.items)

    # Wait train arriving
    if env.now <= arrival_time:
        yield env.timeout(arrival_time - env.now)
        print(f"Time {env.now:.3f}: [In Time] Train {train_schedule['train_id']} has arrived, waiting to be assigned to a track.")
        delay_time = 0
    else:
        delay_time = env.now - arrival_time
        if f"train_id_{train_id}" not in delay_list:
            delay_list[f"train_id_{train_id}"] = {}
        delay_list[f"train_id_{train_id}"]["arrival"] = delay_time
        print( f"Time {env.now:.3f}: [DELAYED] Train {train_schedule['train_id']} has been delayed for {delay_time} hours.")
    state.train_delay_time[train_schedule['train_id']] = delay_time

    track_id = yield terminal.tracks.get()
    if track_id is None:
        print(f"Time {env.now:.3f}: Train {train_id} is waiting for an available track.")
        terminal.waiting_trains.append(train_id)
        return
    print(f"Time {env.now:.3f}: Train {train_id} is assigned to Track {track_id}.")

    ic_num = terminal.IC_COUNT[train_schedule['train_id']]
    for ic_id in range(ic_num, ic_num + train_schedule['full_cars']):
        ic = container(type='Inbound', id=ic_id, train_id=train_schedule['train_id'])
        terminal.train_ic_stores.put(ic)
        record_container_event(ic, 'train_arrival', env.now)  # loop: assign container_id range(current_ic, current_ic + train_schedule['full_cars'])

    terminal.train_ic_unload_events[train_schedule['train_id']] = env.event()
    terminal.train_oc_prepared_events[train_schedule['train_id']] = env.event()

    # print(f"terminal train ic unload events: {terminal.train_ic_unload_events}")

    # crane unloading IC
    terminal.train_ic_picked_events[train_schedule['train_id']] = env.event()
    terminal.train_start_load_events[train_schedule['train_id']] = env.event()
    terminal.train_end_load_events[train_schedule['train_id']] = env.event()

    env.process(crane_unload_process(env, terminal, train_schedule, track_id))

    # prepare all OC and pick up all IC before crane loading
    # yield all_ic_picked & all_oc_prepared

    yield terminal.train_ic_picked_events[train_schedule['train_id']]
    env.process(handle_remaining_oc(env, terminal, train_schedule))
    yield terminal.train_oc_prepared_events[train_schedule['train_id']]
    print(f"Time {env.now:.3f}: All {oc_needed} OCs are ready on chassis.")
    terminal.train_start_load_events[train_schedule['train_id']].succeed()

    env.process(crane_load_process(env, terminal, track_id=track_id, train_schedule=train_schedule))
    yield terminal.train_end_load_events[train_schedule['train_id']]

    # train departs
    if env.now < departure_time:
        yield env.timeout(0)
        print(f"Time {env.now:.3f}: [EARLY] Train {train_schedule['train_id']} departs from the track {track_id}.")
    elif env.now == departure_time:
        yield env.timeout(departure_time - env.now)
        print(f"Time {env.now:.3f}: [In Time] Train {train_schedule['train_id']} departs from the track {track_id}.")
    else:
        delay_time = env.now - departure_time
        if f"train_id_{train_id}" not in delay_list:
            delay_list[f"train_id_{train_id}"] = {}
        delay_list[f"train_id_{train_id}"]["departure"] = delay_time
        print(f"Time {env.now:.3f}: [DELAYED] Train {train_schedule['train_id']} has been delayed for {delay_time} hours from the track {track_id}.")

    yield terminal.tracks.put(track_id)
    print(f"Time {env.now:.3f}: Train {train_schedule['train_id']} is departing from the terminal.")

    for oc_id in range(terminal.OC_COUNT[train_schedule['train_id']],
                       terminal.OC_COUNT[train_schedule['train_id']] + train_schedule['oc_number']):
        record_container_event(f"OC-{oc_id}-Train-{train_schedule['train_id']}", 'train_depart',
                               env.now)  # loop: assign container_id range(current_oc, current_oc + train_schedule['full_cars'])

    state.time_per_train[train_schedule['train_id']] = env.now - arrival_time

    # Update various parameters to track inbound and outbound containers
    terminal.IC_COUNT[train_schedule['train_id']] += train_schedule['full_cars']
    terminal.OC_COUNT[train_schedule['train_id']] += train_schedule['oc_number']
    print(f"terminal oc_num for {train_schedule['train_id']}: {terminal.OC_COUNT[train_schedule['train_id']]}")

    # state.IC_NUM = state.IC_NUM + train_schedule['full_cars']
    # state.OC_NUM = state.OC_NUM + train_schedule['oc_number']

    terminal.IC_COUNT[train_schedule['train_id']] = terminal.IC_COUNT[train_schedule['train_id']] + train_schedule[
        'full_cars']
    print(f"train {train_schedule['train_id']} has {terminal.IC_COUNT[train_schedule['train_id']]} ICs.")

    # Check available tracks
    print("Available tracks:", terminal.tracks.items)


def run_simulation(train_consist_plan: pl.DataFrame, terminal: str, out_path=None):
    global state
    state.terminal = terminal
    state.train_consist_plan = train_consist_plan
    state.initialize()

    # Create environment
    env = simpy.Environment()

    # Train timetable: shorter headway
    train_timetable = [
        {"train_id": 19, "arrival_time": 0, "departure_time": 10, "empty_cars": 3, "full_cars": 5, "oc_number": 2,
         "truck_number": 5},  # test: ic > oc
        {"train_id": 12, "arrival_time": 1, "departure_time": 10, "empty_cars": 5, "full_cars": 4, "oc_number": 4,
         "truck_number": 4},  # test: ic = oc
        {"train_id": 70, "arrival_time": 2, "departure_time": 20, "empty_cars": 5, "full_cars": 3, "oc_number": 5,
         "truck_number": 5},  # test: ic < oc
    ]

    # Initialize train status in the terminal
    train_departed_event = env.event()
    train_departed_event.succeed()

    truck_number = max([entry['truck_number'] for entry in train_timetable])
    chassis_count = max([entry['empty_cars'] + entry['full_cars'] for entry in train_timetable])
    terminal = Terminal(env, truck_capacity=truck_number, chassis_count=chassis_count)

    # Trains arrive according to timetable
    for i, train_schedule in enumerate(train_timetable):
        print(train_schedule)
        env.process(process_train_arrival(env, terminal, train_schedule))

    # Simulation hyperparameters
    env.run(until=state.sim_time)

    # Performance Matrix
    # Train processing time
    avg_time_per_train = sum(state.time_per_train.values()) / len(state.time_per_train)
    print( f"Average train processing time: {sum(state.time_per_train) / len(state.time_per_train) if state.time_per_train else 0:.2f}")
    print("Simulation completed. ")
    with open("avg_time_per_train.txt", "w") as f:
        f.write(str(avg_time_per_train))

    # Create DataFrame for container events
    print(state.sim_time)

    container_data = (
        pl.from_dicts(
            [dict(event, **{'container_id': container_id}) for container_id, event in state.container_events.items()],
            infer_schema_length=None
        )
        .with_columns(
            pl.when(
                pl.col("truck_exit").is_not_null() & pl.col("train_arrival").is_not_null()
            )
            .then(
                pl.col("truck_exit") - pl.col("train_arrival")
            )
            .when(
                pl.col("train_depart").is_not_null()
            )
            .then(
                pl.col("crane_load") - pl.col("truck_arrival")
            )
            .otherwise(None)
            .alias("container_processing_time")
        )
        .sort("container_id")
        .select(pl.col("container_id"), pl.all().exclude("container_id"))
    )
    if out_path is not None:
        container_data.write_excel(
            out_path / f"double_track_simulation_crane_{state.CRANE_NUMBER}_hostler_{state.HOSTLER_NUMBER}.xlsx")

    # Use save_average_times and save_vehicle_logs for vehicle related logs
    save_energy_to_excel(state)
    save_vehicle_and_performance_metrics(state)

    print("Done!")
    return None


if __name__ == "__main__":
    run_simulation(
        train_consist_plan=pl.read_csv(utilities.package_root() / 'demos' / 'starter_demo' / 'train_consist_plan.csv'),
        terminal="Allouez",
        out_path=utilities.package_root() / 'demos' / 'double_track_results'
    )