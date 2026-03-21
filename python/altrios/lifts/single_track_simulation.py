import simpy
import random
import time
import json
import polars as pl

from altrios.lifts.distances_single_track import *
from altrios.lifts.dictionary import *
from altrios.lifts.vehicle import *
import altrios.lifts.distances_single_track as layout
import altrios.lifts.utilities

K, k, M, N, n_t, n_p, n_r= layout.load_layout_config_from_json()

class Terminal:
    def __init__(self, env, truck_capacity, chassis_count):
        self.env = env
        self.tracks = simpy.Store(env, capacity=state.TRACK_NUMBER)
        for track_id in range(1, state.TRACK_NUMBER + 1):
            self.tracks.put(track_id)
        self.cranes = simpy.Store(env, state.CRANE_NUMBER)
        self.in_gates = simpy.Resource(env, state.IN_GATE_NUMBERS)
        self.out_gates = simpy.Resource(env, state.OUT_GATE_NUMBERS)
        self.oc_store = simpy.Store(env)
        self.parking_slots = simpy.Store(env, K)
        self.chassis = simpy.FilterStore(env, capacity=chassis_count)
        self.hostlers = simpy.Store(env, capacity=state.HOSTLER_NUMBER)
        self.truck_store = simpy.Store(env, capacity=truck_capacity)

        # resource setting
        # crane
        num_diesel_crane = round(state.CRANE_NUMBER * state.CRANE_DIESEL_PERCENTAGE)
        num_hybrid_crane = state.CRANE_NUMBER - num_diesel_crane
        cranes = [(i, "diesel") for i in range(num_diesel_crane)] + \
                 [(i + num_diesel_crane, "hybrid") for i in range(num_hybrid_crane)]
        for crane_id in cranes:
            self.cranes.put(crane_id)

        # hostler
        num_diesel = round(state.HOSTLER_NUMBER * state.HOSTLER_DIESEL_PERCENTAGE)
        num_electric = state.HOSTLER_NUMBER - num_diesel
        hostlers = [(i, "diesel") for i in range(num_diesel)] + \
                   [(i + num_diesel, "electric") for i in range(num_electric)]
        for hostler_id in hostlers:
            self.hostlers.put(hostler_id)


def record_event(container_id, event_type, timestamp):
    global state
    if container_id not in state.container_events:
        state.container_events[container_id] = {}
    state.container_events[container_id][event_type] = timestamp


def handle_train_departure(env, train_schedule, train_id, track_id):
    global state

    if env.now < train_schedule["departure_time"]:
        print(f"Time {env.now}: [EARLY] Train {train_schedule['train_id']} departs from the track {track_id}.")
    elif env.now == train_schedule["departure_time"]:
        print(f"Time {env.now}: [Ing Time] Train {train_schedule['train_id']} departs from the track {track_id}.")
    else:
        delay_time = env.now - train_schedule["departure_time"]
        print(f"Time {env.now}: [DELAYED] Train {train_schedule['train_id']} has been delayed for {delay_time} hours from the track {track_id}.")


def save_vehicle_and_performance_metrics(state, ic_avg_delay, oc_avg_delay):
    out_path = altrios.lifts.utilities.package_root() / 'output' / 'single_track_results'

    container_excel_path = out_path / f"{state.CRANE_NUMBER}C-{state.HOSTLER_NUMBER}H_container_throughput_{K}_batch_size_{k}.xlsx"
    vehicle_excel_path = out_path / f"{state.CRANE_NUMBER}C-{state.HOSTLER_NUMBER}H_vehicle_throughput_{K}_batch_size_{k}.xlsx"

    if not container_excel_path.exists():
        print(f"[Error] Container Excel not found: {container_excel_path}")
        return
    if not vehicle_excel_path.exists():
        print(f"[Error] Vehicle Excel not found: {vehicle_excel_path}")
        return

    ic_avg_time, oc_avg_time, total_ic_avg_time, total_oc_avg_time = calculate_container_processing_time(
        container_excel_path,
        train_batch_size=k,
        daily_throughput=K,
        num_trains=math.ceil(K/(k*2)),
        ic_delay_time=ic_avg_delay,
        oc_delay_time=oc_avg_delay
    )

    ic_energy, oc_energy, total_energy = calculate_vehicle_energy(vehicle_excel_path)

    summary_df = pl.DataFrame({
        "ic_avg_time": ic_avg_time, 
        "ic_avg_delay": ic_avg_delay, 
        "total_ic_avg_time": total_ic_avg_time, 
        "oc_avg_time": oc_avg_time, 
        "oc_avg_delay": oc_avg_delay, 
        "total_oc_avg_time": total_oc_avg_time, 
        "ic_energy": ic_energy, 
        "oc_energy": oc_energy, 
        "total_energy": total_energy
    })

    return summary_df


def emission_calculation(status, move, vehicle, id, travel_time):
    global state
    vehicle = vehicle.capitalize()
    type = id[1].capitalize()

    # load (unit in lift)
    if move == 'load':
        if status == 'loaded':
            emission_unit = state.ENERGY_CONSUMPTION["LOAD_CONSUMPTION"][f"{vehicle}_Loaded"][type]
        else:  # idle
            emission_unit = state.ENERGY_CONSUMPTION["LOAD_CONSUMPTION"][f"{vehicle}_Idle"][type]
        emissions = emission_unit

    # trip (unit in hr * travel_time）
    elif move == 'trip':
        if status == 'loaded':
            emission_unit = state.ENERGY_CONSUMPTION["TRIP_CONSUMPTION"][f"{vehicle}_Loaded"][type]
        else:  # empty
            emission_unit = state.ENERGY_CONSUMPTION["TRIP_CONSUMPTION"][f"{vehicle}_Empty"][type]
        emissions = emission_unit * travel_time

    # side (unit in lift)
    elif move == 'side':
        emission_unit = state.ENERGY_CONSUMPTION["SIDE_PICK_CONSUMPTION"]["Side"][type]
        emissions = emission_unit

    else:
        raise ValueError(f"Unsupported move type-{move} for vehicle {vehicle}-{id}.")

    return emissions


def truck_entry(env, train_schedule, terminal, truck_id):
    global state
    with terminal.in_gates.request() as gate_request:
        yield gate_request

        oc_id = yield terminal.oc_store.get()
        record_event(oc_id, 'truck_arrival', env.now)
        # truck drops off an OC to the parking slot
        truck_travel_time, truck_dist, truck_avg_speed, truck_avg_density = simulate_truck_travel(truck_id, train_schedule, terminal, total_lane_length, d_t_min, d_t_max)
        emissions = emission_calculation('loaded', 'trip', 'Truck', truck_id, truck_travel_time)
        record_vehicle_event(
            vehicle_category = 'truck',
            vehicle = truck_id,
            container = oc_id,
            state = 'loaded',
            move = 'trip',
            time = truck_travel_time,
            distance = truck_dist,
            speed = truck_avg_speed,
            density = truck_avg_density,
            emission = emissions,
            type = truck_id[1].capitalize(),
            timestamp = env.now
        )

        # side pick picks up an OC
        side_pick_unload_time = 1/60 + random.uniform(0, 1/600)
        emissions = emission_calculation('loaded', 'side', 'Side', truck_id, side_pick_unload_time)
        record_vehicle_event(
            vehicle_category='side',
            vehicle='Side pick',
            container=oc_id,
            state='loaded',
            move='load',
            time=side_pick_unload_time,
            distance='n/a',
            speed='n/a',
            density='n/a',
            emission=emissions,
            type='Diesel',
            timestamp=env.now
        )

        # Truck resource parking
        terminal.parking_slots.put(oc_id)
        record_event(oc_id, 'truck_dropoff', env.now)
        truck_travel_time, truck_dist, truck_avg_speed, truck_avg_density = simulate_truck_travel(truck_id, train_schedule, terminal, total_lane_length, d_t_min, d_t_max)
        emissions = emission_calculation('empty', 'trip', 'Truck', truck_id, truck_travel_time)
        record_vehicle_event(
            vehicle_category='truck',
            vehicle=truck_id,
            container=oc_id,
            state='empty',
            move='trip',
            time=truck_travel_time,
            distance=truck_dist,
            speed=truck_avg_speed,
            density=truck_avg_density,
            emission=emissions,
            type=truck_id[1].capitalize(),
            timestamp=env.now
        )


def empty_truck(env, train_schedule, terminal, truck_id):
    global state
    with terminal.in_gates.request() as gate_request:
        yield gate_request
        # Truck resource parking, no container-based recording for empty truck considering Excel format
        truck_travel_time, truck_dist, truck_avg_speed, truck_avg_density = simulate_truck_travel(truck_id, train_schedule, terminal, total_lane_length, d_t_min, d_t_max)
        emissions = emission_calculation('empty', 'trip', 'Truck', truck_id, truck_travel_time)
        record_vehicle_event(
            vehicle_category='truck',
            vehicle=truck_id,
            container='n/a',
            state='empty',
            move='trip',
            time=truck_travel_time,
            distance=truck_dist,
            speed=truck_avg_speed,
            density=truck_avg_density,
            emission=emissions,
            type=truck_id[1].capitalize(),
            timestamp=env.now
        )


def truck_arrival(env, terminal, train_schedule, all_trucks_arrived_event):
    global state
    truck_number = train_schedule["truck_number"]
    num_diesel = round(truck_number * state.TRUCK_DIESEL_PERCENTAGE)
    num_electric = truck_number - num_diesel

    trucks = [(i, "diesel") for i in range(num_diesel)] + \
             [(i + num_diesel, "electric") for i in range(num_electric)]

    for truck_id in trucks:
        yield env.timeout(0)  # Assume truck arrives not impact on system, not random.expovariate(arrival_rate)
        terminal.truck_store.put(truck_id)
        if len(terminal.oc_store.items) != 0:  # Truck move OC from outside (oc_store) to terminal (parking_slots)
            env.process(truck_entry(env, train_schedule, terminal, truck_id))
        else:
            env.process(empty_truck(env, train_schedule, terminal, truck_id))

    all_trucks_arrived_event.succeed()  # if all_trucks_arrived_event is triggered, train is allowed to enter


def crane_unload_process(env, terminal, train_schedule, all_oc_prepared, oc_needed, total_ic, all_ic_picked, all_ic_unload_event):
    global state

    ic_store = simpy.Store(env)
    for ic_id in range(state.IC_NUM, state.IC_NUM + total_ic):
        ic_store.put(ic_id)

    def unload_ic(env, ic_id):
        crane_id = yield terminal.cranes.get()

        crane_unload_time = state.CONTAINERS_PER_CRANE_MOVE_MEAN + random.uniform(0, state.CRANE_MOVE_DEV_TIME)
        yield env.timeout(crane_unload_time)

        record_event(ic_id, 'crane_unload', env.now)
        emissions = emission_calculation('loaded', 'load', 'Crane', crane_id, crane_unload_time)
        record_vehicle_event(
            vehicle_category='crane',
            vehicle=crane_id,
            container=ic_id,
            state='loaded',
            move='load',
            time=crane_unload_time,
            distance='n/a',
            speed='n/a',
            density='n/a',
            emission=emissions,
            type=crane_id[1].capitalize(),
            timestamp=env.now
        )

        terminal.chassis.put(ic_id)
        terminal.cranes.put(crane_id)
        env.process(container_process(env, terminal, train_schedule, all_oc_prepared, oc_needed, all_ic_unload_event, all_ic_picked))

        if len(ic_store.items) == 0 and not all_ic_unload_event.triggered:
            all_ic_unload_event.succeed()

    # start process for every ic
    while len(ic_store.items) > 0:
        ic_id = yield ic_store.get()
        env.process(unload_ic(env, ic_id))


def container_process(env, terminal, train_schedule, all_oc_prepared, oc_needed, all_ic_unload_event, all_ic_picked):
    global state
    '''
    It is designed to transfer both inbound and outbound containers.
    The main simulation process is as follows:
    1. A hostler picks up an IC, and drops off IC at parking slot.
    2. A truck picks up the IC, and leaves the gate
    3. The hostler picks up an OC, and drops off OC at the chassis.
    4. Once all OCs are prepared (all_oc_prepared), the crane starts loading (other function).
    '''
    hostler_id = yield terminal.hostlers.get()
    ic_id = yield terminal.chassis.get(lambda x: isinstance(x, int))

    # empty hostler going to pick up IC
    current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
    hostler_travel_time_to_parking, hostler_dist, hostler_speed, hostler_density = simulate_hostler_travel(hostler_id,current_veh_num,total_lane_length,d_h_min,d_h_max)

    # side-pick
    side_pick_unload_time = 1 / 60 + random.uniform(0, 1 / 600)
    emissions = emission_calculation('loaded', 'side', 'Side', hostler_id, side_pick_unload_time)
    record_vehicle_event(
        vehicle_category = 'side',
        vehicle = 'Side pick',
        container = ic_id,
        state = 'loaded',
        move = 'load',
        time = side_pick_unload_time,
        distance='n/a',
        speed='n/a',
        density='n/a',
        emission = emissions,
        type = 'Diesel',
        timestamp = env.now
    )

    # Hostler taking the IC to the parking slot
    current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
    hostler_travel_time_to_parking, hostler_dist, hostler_speed, hostler_density = simulate_hostler_travel(hostler_id, current_veh_num, total_lane_length, d_h_min, d_h_max)
    yield env.timeout(hostler_travel_time_to_parking)
    record_event(ic_id, 'hostler_pickup', env.now)
    emissions = emission_calculation('loaded', 'trip', 'Hostler', hostler_id, hostler_travel_time_to_parking)
    record_vehicle_event(
        vehicle_category = 'hostler',
        vehicle = hostler_id,
        container = ic_id,
        state = 'loaded',
        move = 'trip',
        time = hostler_travel_time_to_parking,
        distance = hostler_dist,
        speed = hostler_speed,
        density = hostler_density,
        emission = emissions,
        type = hostler_id[1].capitalize(),
        timestamp = env.now
    )

    # Side pick lifts the IC to the parking slot
    side_pick_unload_time = 1 / 60 + random.uniform(0, 1 / 600)
    record_event(ic_id, 'hostler_loaded', env.now)
    emissions = emission_calculation('loaded', 'side', 'Side', hostler_id, side_pick_unload_time)
    record_vehicle_event(
        vehicle_category='side',
        vehicle='Side pick',
        container=ic_id,
        state='loaded',
        move='load',
        time=side_pick_unload_time,
        distance='n/a',
        speed='n/a',
        density='n/a',
        emission=emissions,
        type='Diesel',
        timestamp=env.now
    )

    # Prepare for crane loading: if chassis has no IC AND all_ic_picked (parking side) is not triggered => trigger all_ic_picked
    if sum(str(item).isdigit() for item in terminal.chassis.items) == 0 and all_ic_unload_event.triggered and not all_ic_picked.triggered:
        all_ic_picked.succeed()

    # Assign a truck to pick up IC
    truck_id = yield terminal.truck_store.get()

    # Truck going to pick up IC
    record_event(ic_id, 'truck_pickup', env.now)
    truck_travel_time, truck_dist, truck_avg_speed, truck_avg_density = simulate_truck_travel(truck_id, train_schedule, terminal, total_lane_length, d_t_min, d_t_max)
    yield env.timeout(truck_travel_time)
    emissions = emission_calculation('empty', 'trip', 'Truck', truck_id, truck_travel_time)
    record_vehicle_event(
        vehicle_category = 'truck',
        vehicle = truck_id,
        container = ic_id,
        state = 'empty',
        move = 'trip',
        time=truck_travel_time,
        distance=truck_dist,
        speed=truck_avg_speed,
        density=truck_avg_density,
        emission = emissions,
        type = truck_id[1].capitalize(),
        timestamp = env.now
    )

    # Side-pick picks up IC to load the truck
    side_pick_unload_time = 1/60 + random.uniform(0, 1/600)
    emissions = emission_calculation('loaded', 'side', 'Side', hostler_id, side_pick_unload_time)
    record_vehicle_event(
        vehicle_category = 'side',
        vehicle = 'Side pick',
        container = ic_id,
        state = 'loaded',
        move = 'load',
        time = side_pick_unload_time,
        distance='n/a',
        speed='n/a',
        density='n/a',
        emission = emissions,
        type = 'Diesel',
        timestamp=env.now
    )

    # Truck going to leave the gate
    truck_travel_time, truck_dist, truck_avg_speed, truck_avg_density = simulate_truck_travel(truck_id, train_schedule, terminal, total_lane_length, d_t_min, d_t_max)
    yield env.timeout(truck_travel_time)
    emissions = emission_calculation('loaded', 'trip', 'Truck', truck_id, truck_travel_time)
    record_vehicle_event(
        vehicle_category='truck',
        vehicle=truck_id,
        container=ic_id,
        state='loaded',
        move='trip',
        time=truck_travel_time,
        distance=truck_dist,
        speed=truck_avg_speed,
        density=truck_avg_density,
        emission=emissions,
        type=truck_id[1].capitalize(),
        timestamp=env.now
    )

    # Truck queue and exit the gate
    env.process(truck_exit(env, terminal, truck_id, ic_id))

    # Assign a hostler to pick up an OC, if OC remains (balanced loop remains)
    if len(terminal.parking_slots.items) > 0:
        oc = yield terminal.parking_slots.get()

        # The hostler going to pick up an OC after finishing the IC trip
        hostler_reposition_travel_time, reposition_dist, hostler_speed, hostler_density = simulate_reposition_travel(hostler_id, current_veh_num, total_lane_length, d_h_min, d_h_max)
        yield env.timeout(hostler_reposition_travel_time)
        record_event(oc, 'hostler_pickup', env.now)
        emissions = emission_calculation('empty', 'trip', 'Hostler', hostler_id, hostler_reposition_travel_time)
        record_vehicle_event(
            vehicle_category='hostler',
            vehicle=hostler_id,
            container=oc,
            state='empty',
            move='trip',
            time=hostler_reposition_travel_time,
            distance=reposition_dist,
            speed=hostler_speed,
            density=hostler_density,
            emission=emissions,
            type=hostler_id[1].capitalize(),
            timestamp=env.now
        )

        # Side-pick picks up OC to load the hostler
        # Hostler going to the chassis
        record_event(ic_id, 'hostler_loaded', env.now)
        hostler_load_time = 1 / 60 + random.uniform(0, 1 / 600)
        emissions = emission_calculation('loaded', 'side', 'Side', hostler_id, hostler_load_time)
        record_vehicle_event(
            vehicle_category='side',
            vehicle='Side pick',
            container=oc,
            state='loaded',
            move='load',
            time=hostler_load_time,
            distance='n/a',
            speed='n/a',
            density='n/a',
            emission=emissions,
            type='Diesel',
            timestamp=env.now
        )

        # Hostler going to drop off OC
        current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
        hostler_travel_time_to_parking, hostler_dist, hostler_speed, hostler_density = simulate_hostler_travel(hostler_id, current_veh_num, total_lane_length, d_h_min, d_h_max)
        yield env.timeout(hostler_travel_time_to_parking)
        yield terminal.chassis.put(oc)
        record_event(oc, 'hostler_dropoff', env.now)
        emissions = emission_calculation('loaded', 'trip', 'Hostler', hostler_id, hostler_travel_time_to_parking)
        record_vehicle_event(
            vehicle_category='hostler',
            vehicle=hostler_id,
            container=oc,
            state='loaded',
            move='trip',
            time=hostler_travel_time_to_parking,
            distance=hostler_dist,
            speed=hostler_speed,
            density=hostler_density,
            emission=emissions,
            type=hostler_id[1].capitalize(),
            timestamp=env.now
        )

        # Side-pick drops off OC to chassis from hostler
        side_pick_unload_time = 1 / 60 + random.uniform(0, 1 / 600)
        record_event(oc, 'hostler_unloaded', env.now)
        emissions = emission_calculation('loaded', 'side', 'Side', hostler_id, side_pick_unload_time)
        record_vehicle_event(
            vehicle_category='side',
            vehicle='Side pick',
            container=oc,
            state='loaded',
            move='load',
            time=side_pick_unload_time,
            distance='n/a',
            speed='n/a',
            density='n/a',
            emission=emissions,
            type=hostler_id[1].capitalize(),
            timestamp=env.now
        )

    # Hostler going back to resource parking
    current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
    hostler_travel_time_to_parking, hostler_dist, hostler_speed, hostler_density = simulate_hostler_travel(hostler_id, current_veh_num, total_lane_length, d_h_min, d_h_max)
    yield env.timeout(hostler_travel_time_to_parking)
    emissions = emission_calculation('empty', 'trip', 'Hostler', hostler_id, hostler_travel_time_to_parking)
    record_vehicle_event(
        vehicle_category = 'hostler',
        vehicle = hostler_id,
        container = "",
        state = 'empty',
        move = 'trip',
        time=hostler_travel_time_to_parking,
        distance=hostler_dist,
        speed=hostler_speed,
        density=hostler_density,
        emission = emissions,
        type = hostler_id[1].capitalize(),
        timestamp = env.now
    )
    yield terminal.hostlers.put(hostler_id)

    # IC < OC: ICs are all picked up and still have OCs remaining
    if sum(str(item).isdigit() for item in terminal.chassis.items) == 0 and len(terminal.parking_slots.items) == \
            train_schedule['oc_number'] - train_schedule['full_cars']:
        remaining_oc = len(terminal.parking_slots.items)
        for i in range(1, remaining_oc + 1):
            oc = yield terminal.parking_slots.get()

            # Hostler going to pick up OC from parking slots
            current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
            hostler_travel_time_to_parking, hostler_dist, hostler_speed, hostler_density = simulate_hostler_travel(hostler_id, current_veh_num, total_lane_length, d_h_min, d_h_max)
            yield env.timeout(hostler_travel_time_to_parking)
            yield terminal.chassis.put(oc)
            record_event(oc, 'hostler_dropoff', env.now)
            emissions = emission_calculation('loaded', 'trip', 'Hostler', hostler_id, hostler_travel_time_to_parking)
            record_vehicle_event(
                vehicle_category='hostler',
                vehicle=hostler_id,
                container=oc,
                state='loaded',
                move='trip',
                time=hostler_travel_time_to_parking,
                distance=hostler_dist,
                speed=hostler_speed,
                density=hostler_density,
                emission=emissions,
                type=hostler_id[1].capitalize(),
                timestamp=env.now
            )

            # Side-pick picks up an OC
            side_pick_unload_time = 1 / 60 + random.uniform(0, 1 / 600)
            record_event(ic_id, 'hostler_unloaded', env.now)
            emissions = emission_calculation('loaded', 'side', 'Side', hostler_id, side_pick_unload_time)
            record_vehicle_event(
                vehicle_category='side',
                vehicle='Side pick',
                container=oc,
                state='loaded',
                move='load',
                time=side_pick_unload_time,
                distance='n/a',
                speed='n/a',
                density='n/a',
                emission=emissions,
                type=hostler_id[1].capitalize(),
                timestamp=env.now
            )

            # Hostler going to drop off OC
            current_veh_num = state.HOSTLER_NUMBER - len(terminal.hostlers.items)
            hostler_travel_time_to_parking, hostler_dist, hostler_speed, hostler_density = simulate_hostler_travel(hostler_id, current_veh_num, total_lane_length, d_h_min, d_h_max)
            yield env.timeout(hostler_travel_time_to_parking)
            yield terminal.chassis.put(oc)
            record_event(oc, 'hostler_dropoff', env.now)
            emissions = emission_calculation('loaded', 'trip', 'Hostler', hostler_id, hostler_travel_time_to_parking)
            record_vehicle_event(
                vehicle_category='hostler',
                vehicle=hostler_id,
                container=oc,
                state='loaded',
                move='trip',
                time=hostler_travel_time_to_parking,
                distance=hostler_dist,
                speed=hostler_speed,
                density=hostler_density,
                emission=emissions,
                type=hostler_id[1].capitalize(),
                timestamp=env.now
            )

            # Hostler drops off OC to chassis
            side_pick_unload_time = 1 / 60 + random.uniform(0, 1 / 600)
            record_event(oc, 'hostler_unloaded', env.now)
            emissions = emission_calculation('loaded', 'side', 'Side', hostler_id, side_pick_unload_time)
            record_vehicle_event(
                vehicle_category='side',
                vehicle='Side pick',
                container=ic_id,
                state='loaded',
                move='load',
                time=side_pick_unload_time,
                distance='n/a',
                speed='n/a',
                density='n/a',
                emission=emissions,
                type=hostler_id[1].capitalize(),
                timestamp=env.now
            )
            yield terminal.chassis.put(oc)

            if sum(1 for item in terminal.chassis.items if isinstance(item, str) and "OC-" in str(item)) == oc_needed:
                all_oc_prepared.succeed()
            i += 1

    # if sum(1 for item in terminal.chassis.items if isinstance(item, str) and "OC-" in str(item)) == oc_needed:
    if (sum(1 for item in terminal.chassis.items if isinstance(item, str) and "OC-" in str(item)) == oc_needed
    and not all_oc_prepared.triggered):
        all_oc_prepared.succeed()
        print(f"Time {env.now}: All OCs are ready on chassis.")


def truck_exit(env, terminal, truck_id, ic_id):
    global state
    with terminal.out_gates.request() as out_gate_request:
        yield out_gate_request
        truck_pass_time = state.TRUCK_OUTGATE_TIME + random.uniform(0, state.TRUCK_OUTGATE_TIME_DEV)
        yield env.timeout(truck_pass_time)
        record_event(ic_id, 'truck_exit', env.now)
    yield terminal.truck_store.put(truck_id)


def crane_load_process(env, terminal, start_load_event, end_load_event):
    global state
    yield start_load_event

    def load_oc(env):
        while True:
            if not any(isinstance(item, str) and "OC" in item for item in terminal.chassis.items):
                if not end_load_event.triggered:
                    end_load_event.succeed()
                break

            crane_id = yield terminal.cranes.get()
            oc = yield terminal.chassis.get(lambda x: isinstance(x, str) and "OC" in x)

            crane_load_time = state.CONTAINERS_PER_CRANE_MOVE_MEAN + random.uniform(0, state.CRANE_MOVE_DEV_TIME)
            yield env.timeout(crane_load_time)

            record_event(oc, 'crane_load', env.now)
            emissions = emission_calculation('loaded', 'load', 'Crane', crane_id, crane_load_time)
            record_vehicle_event(
                vehicle_category='crane',
                vehicle=crane_id,
                container=oc,
                state='loaded',
                move='load',
                time=crane_load_time,
                distance='n/a',
                speed='n/a',
                density='n/a',
                emission=emissions,
                type=crane_id[1].capitalize(),
                timestamp=env.now
            )

            # release crane
            terminal.cranes.put(crane_id)

    num_cranes = len(terminal.cranes.items)
    for _ in range(num_cranes):
        env.process(load_oc(env))


def process_train_arrival(env, terminal, train_departed_event, train_schedule, next_departed_event):
    global state
    # yield train_departed_event
    if train_departed_event is not None:
        yield train_departed_event

    train_id = train_schedule["train_id"]
    arrival_time = train_schedule["arrival_time"]
    oc_needed = train_schedule["oc_number"]
    total_ic = train_schedule["full_cars"]

    print(f"------------------- The current train is {train_id}: scheduled arrival time {arrival_time}, OC {oc_needed}, IC {total_ic} -------------------")

    # create events as processing conditions
    all_trucks_arrived_event = env.event()  # condition for train arrival
    all_ic_unload_event = env.event()  # condition for all ic picked
    all_ic_picked = env.event()  # condition 1 for crane loading
    all_oc_prepared = env.event()  # condition 2 for crane loading
    start_load_event = env.event()  # condition 1 for train departure
    end_load_event = env.event()  # condition 2 for train departure

    oc_id = state.OC_NUM
    # print("start oc_id:", oc_id)
    for oc in range(state.OC_NUM, state.OC_NUM + train_schedule['oc_number']):
        terminal.oc_store.put(f"OC-{oc_id}")
        oc_id += 1

    # All trucks arrive before train arrives
    env.process(truck_arrival(env, terminal, train_schedule, all_trucks_arrived_event))

    # Track assignment for a train
    track_id = yield terminal.tracks.get()
    if track_id is None:
        terminal.waiting_trains.append(train_id)
        return

    # Wait train arriving
    if env.now <= arrival_time:
        yield env.timeout(arrival_time - env.now)
        delay_time = 0
    else:
        delay_time = env.now - arrival_time
    state.train_delay_time[train_schedule['train_id']] = delay_time

    for ic_id in range(state.IC_NUM, state.IC_NUM + train_schedule['full_cars']):
        record_event(ic_id, 'train_arrival_expected', train_schedule['arrival_time'])
        record_event(ic_id, 'train_arrival', env.now)  # loop: assign container_id range(current_ic, current_ic + train_schedule['full_cars'])

    # crane unloading IC
    env.process(crane_unload_process(env, terminal, train_schedule, all_oc_prepared, oc_needed, total_ic, all_ic_picked, all_ic_unload_event))

    # prepare all OC and pick up all IC before crane loading
    yield all_ic_picked & all_oc_prepared
    start_load_event.succeed()  # condition of chassis loading

    # crane loading process
    env.process(crane_load_process(env, terminal, start_load_event=start_load_event, end_load_event=end_load_event))
    yield end_load_event

    # train departs & delay records
    yield env.timeout(state.TRAIN_INSPECTION_TIME)
    handle_train_departure(env, train_schedule, train_id, track_id)

    yield terminal.tracks.put(track_id)

    for oc_id in range(state.OC_NUM, state.OC_NUM + train_schedule['oc_number']):
        record_event(f"OC-{oc_id}", 'train_depart_expected', train_schedule['departure_time'])
        record_event(f"OC-{oc_id}", 'train_depart', env.now)  # loop: assign container_id range(current_oc, current_oc + train_schedule['full_cars'])

    state.time_per_train[train_schedule['train_id']] = env.now - arrival_time

    # Update various parameters to track inbound and outbound containers
    state.IC_NUM = state.IC_NUM + train_schedule['full_cars']
    state.OC_NUM = state.OC_NUM + train_schedule['oc_number']

    # Trigger the departure event for the current train
    next_departed_event.succeed()  # the terminal now is clear and ready to accept the next train


def run_simulation(terminal: str, train_consist_plan: pl.DataFrame = None, out_path=None):
    global state
    state.terminal = terminal
    state.train_consist_plan = train_consist_plan
    state.initialize()

    random.seed(42)

    # Create environment
    env = simpy.Environment()

    # # Train timetable
    if train_consist_plan is not None:
        train_timetable = altrios.lifts.utilities.build_train_timetable(train_consist_plan, terminal, as_dicts = True)
    else:
        with open(altrios.lifts.utilities.resources_root() / "train_timetable.json", "r") as f:
            train_timetable = json.load(f)

    # Initialize train status in the terminal
    train_departed_event = env.event()
    train_departed_event.succeed()

    truck_number = max([entry['truck_number'] for entry in train_timetable])
    chassis_count = max([entry['empty_cars'] + entry['full_cars'] for entry in train_timetable])
    terminal = Terminal(env, truck_capacity=truck_number, chassis_count=chassis_count)

    # Trains arrive according to timetable
    for i, train_schedule in enumerate(train_timetable):
        print(train_schedule)
        next_departed_event = env.event()  # Create a new departure event for the next train
        env.process(process_train_arrival(env, terminal, train_departed_event, train_schedule, next_departed_event))
        train_departed_event = next_departed_event  # Update the departure event for the next train

    # Simulation duration: time includes warm-up and cool down
    with open(altrios.lifts.utilities.resources_root() / "sim_config.json", "r") as f:
        sim_config = json.load(f)
    total_simulation_length = sim_config["vehicles"]["simulation_duration"]
    env.run(until=total_simulation_length)

    # Data analysis
    # ==== 1. container log dataframe ====
    import pandas as pd
    container_records = []
    for container_id, event in state.container_events.items():
        record = {
            "container_id": container_id,
            "train_arrival": event.get("train_arrival"),
            "train_arrival_expected": event.get("train_arrival_expected"),
            "crane_unload": event.get("crane_unload"),
            "hostler_loaded": event.get("hostler_loaded"),
            "hostler_pickup": event.get("hostler_pickup"),
            "truck_pickup": event.get("truck_pickup"),
            "hostler_unloaded": event.get("hostler_unloaded"),
            "truck_exit": event.get("truck_exit"),
            "truck_arrival": event.get("truck_arrival"),
            "truck_dropoff": event.get("truck_dropoff"),
            "hostler_dropoff": event.get("hostler_dropoff"),
            "crane_load": event.get("crane_load"),
            "train_depart": event.get("train_depart"),
            "train_depart_expected": event.get("train_depart_expected"),
        }
        container_records.append(record)

    container_data = pd.DataFrame(container_records)

    # ==== 2. Label container type ====
    container_data["type"] = container_data["container_id"].astype(str).apply(
        lambda x: "OC" if x.startswith("OC-") else "IC" if x.isdigit() else "Unknown")

    # ==== 3. Add container processing time ====
    def compute_processing_time(row):
        if pd.notnull(row["truck_exit"]) and pd.notnull(row["crane_unload"]):
            return row["truck_exit"] - row["train_arrival"]  # IC
        elif pd.notnull(row["train_depart"]) and pd.notnull(row["hostler_pickup"]):
            return row["train_depart"] - row["hostler_pickup"]  # OC
        else:
            return None

    container_data["container_processing_time"] = container_data.apply(compute_processing_time, axis=1)

    # ==== 4. Add first_oc_pickup_time for OC containers ====
    container_data = container_data[container_data["container_id"].notna()].copy()
    container_data["container_id"] = container_data["container_id"].astype(str)

    df_oc = container_data[container_data['container_id'].str.startswith("OC-")].copy()

    first_pickup_per_train = (
        df_oc.groupby("train_depart")["hostler_pickup"]
        .min()
        .reset_index()
        .rename(columns={"hostler_pickup": "first_oc_pickup_time"})
    )

    df_oc = df_oc.merge(first_pickup_per_train, on="train_depart", how="left")

    container_data = container_data.merge(
        df_oc[["container_id", "first_oc_pickup_time"]],
        on="container_id",
        how="left"
    )

    # ==== 5. Sort by numeric ID ====
    def extract_numeric_id(cid):
        digits = ''.join(filter(str.isdigit, str(cid)))
        return int(digits) if digits else -1

    container_data["container_id_numeric"] = container_data["container_id"].apply(extract_numeric_id)
    container_data = container_data.sort_values("container_id_numeric").drop(columns=["container_id_numeric"])

    # ==== 6. Save to excel ====
    if out_path is not None:
        container_data.to_excel(out_path / f"{state.CRANE_NUMBER}C-{state.HOSTLER_NUMBER}H_container_throughput_{K}_batch_size_{k}.xlsx", index=False)
    performance_df, vehicle_log_df = save_energy_to_excel(state)

    # ==== 7. Delay Calculations ====
    # Processing time for IC => avg processing time + delay time
    ic_df = container_data[container_data["type"] == "IC"].copy()
    ic_df["ic_delay_time"] = ic_df["train_arrival"] - ic_df["train_arrival_expected"]
    ic_avg_delay = ic_df["ic_delay_time"].mean()

    oc_df = container_data[container_data["type"] == "OC"].copy()
    oc_df["oc_delay_time"] = oc_df["train_depart"] - oc_df["first_oc_pickup_time"]
    oc_avg_delay = oc_df["oc_delay_time"].mean()

    summary_df = save_vehicle_and_performance_metrics(state, ic_avg_delay, oc_avg_delay)

    return summary_df, performance_df, vehicle_log_df


if __name__ == "__main__":
    start_time = time.time()
    single_run = run_simulation(
        train_consist_plan=pl.read_csv(altrios.lifts.utilities.resources_root() / 'train_consist_plan.csv'),
        terminal="Allouez",
        out_path=altrios.lifts.utilities.package_root() / 'output' / 'single_track_results'
    )
    print("total simulation time", time.time() - start_time)
    print("Done!")
    # # Performance Matrix
    # output = {
    #     "single_run": single_run,
    #     "ic_avg_time": single_run[0],
    #     "ic_avg_delay": single_run[1],
    #     "total_ic_avg_time": single_run[2],
    #     "oc_avg_time": single_run[3],
    #     "oc_avg_delay": single_run[4],
    #     "total_oc_avg_time": single_run[5],
    #     "ic_energy": single_run[6],
    #     "oc_energy": single_run[7],
    #     "total_energy": single_run[8]
    # }
    #
    # with open("performance_matrix.json", "w") as f:
    #     json.dump(output, f)