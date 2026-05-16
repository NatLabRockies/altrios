import random
from distance import *
from vehicle import *
import yaml
from decouple import *
import polars as pl

emission_records = []

@dataclass
class container:
    type: str = 'Outbound'
    id: int = 0
    train_id: int = 0

    def to_string(self) -> str:
        if self.type == 'Outbound':
            prefix = 'OC'
        elif self.type == 'Inbound':
            prefix = 'IC'
        else:
            prefix = 'C'
        return f"{prefix}-{self.id}-Train-{self.train_id}"


@dataclass
class crane:
    type: str = 'Diesel'
    id: int = 0
    track_id: int = 0

    def to_string(self) -> str:
        return f'{self.id}-Track-{self.track_id}-{self.type}'


@dataclass
class truck:
    type: str = 'Diesel'
    id: int = 0
    train_id: int = 0

    def to_string(self) -> str:
        return f'{self.id}-Track-{self.train_id}-{self.type}'


@dataclass
class hostler:
    type: str = 'Diesel'
    id: int = 0

    def to_string(self) -> str:
        return f'{self.id}-{self.type}'


class Terminal:
    def __init__(self, env, config):
        self.env = env
        self.config = config

        sim_cfg = config['simulation']
        yard_cfg = config["yard"]
        term_cfg = config["terminal"]
        gate_cfg = config["gates"]
        ems_cfg = config["emissions"]

        # simulation
        self.simulation_length = sim_cfg["length"]
        self.observation_start = sim_cfg["analyze_start"]
        self.observation_end = sim_cfg["analyze_end"]
        self.train_per_day = sim_cfg["train_number"]
        self.train_batch_size = sim_cfg["train_batch_size"]

        # yard
        self.yard_type = yard_cfg.get("yard_type")
        self.track_number = int(yard_cfg.get("track_number"))
        self.receiving_track_numbers = int(yard_cfg.get("receiving_track_numbers"))
        self.railcar_length = float(yard_cfg.get("railcar_length"))
        self.d_f = float(yard_cfg.get("d_f"))
        self.d_x = float(yard_cfg.get("d_x"))

        # terminal
        self.cranes_per_track = int(term_cfg.get("cranes_per_track"))
        self.hostler_number = int(term_cfg.get("hostler_number"))
        self.hostler_diesel_percentage = float(term_cfg.get("hostler_diesel_percentage"))

        # gates
        self.in_gate_numbers = int(gate_cfg.get("in_gate_numbers"))
        self.out_gate_numbers = int(gate_cfg.get("out_gate_numbers"))

        # layout
        self.layout = layout
        distances = calculate_distances(actual_railcars=None)
        self.distances = distances
        self.yard_length = distances["yard_length"]
        self.track_capacity = distances["n_max"]

        self.tracks = simpy.Store(env, capacity=self.track_number)
        for track_id in range(1, self.tracks.capacity + 1):
            self.tracks.put(track_id)

        # cranes
        self.cranes_on_track = {
            track_id: term_cfg["cranes_per_track"]
            for track_id in range(1, self.track_number + 1)
        }

        self.cranes_by_track = {
            track_id: simpy.Store(env, capacity=num_cranes)
            for track_id, num_cranes in self.cranes_on_track.items()
        }

        for track_id, num_cranes in self.cranes_on_track.items():
            for crane_number in range(1, num_cranes + 1):
                c = crane(type="Hybrid", id=crane_number, track_id=track_id)
                self.cranes_by_track[track_id].put(c)


        # emissions
        self.ems = ems_cfg

        self.container_events = {}
        self.time_per_train = {}
        self.train_delay_time = {}

        self.all_trucks_arrived_events = {}  # condition for train arrival
        self.train_ic_unload_events = {}    # condition for train_ic_picked
        self.train_ic_picked_events = {}  # condition 1 for crane loading
        self.train_oc_prepared_events = {}  # condition 2 for crane loading
        self.train_start_load_events = {}  # condition 1 for train departure
        self.train_end_load_events = {}  # condition 2 for train departure
        self.train_departed_events = {}

        self.IC_COUNT = {}
        self.OC_COUNT = {}
        self.total_ic = {}
        self.total_oc = {}

        self.train_pool_stores = simpy.Store(env, capacity=99999)  # train queue capacity
        self.train_ic_stores = simpy.FilterStore(env, capacity=99999)
        self.train_oc_stores = simpy.Store(env, capacity=99999)
        self.in_gates = simpy.Resource(env, self.in_gate_numbers)
        self.out_gates = simpy.Resource(env, self.out_gate_numbers)
        self.oc_store = simpy.Store(env, capacity=99999)
        self.parking_slots = simpy.FilterStore(env, capacity=99999)  # store ic and oc in the parking area
        self.chassis = simpy.FilterStore(env, capacity=999999)
        self.parked_hostlers = simpy.Store(env, capacity=99999)
        self.active_hostlers = simpy.Store(env, capacity=99999)
        self.truck_store = simpy.Store(env, capacity=999999999)

        # hostler setup
        hostler_total = self.hostler_number
        hostler_diesel = round(hostler_total * self.hostler_diesel_percentage)
        hostler_electric = hostler_total - hostler_diesel

        self.parked_hostlers = simpy.Store(env, capacity=hostler_total)
        self.active_hostlers = simpy.Store(env, capacity=hostler_total)

        hostlers = [hostler(id=i, type="Diesel") for i in range(hostler_diesel)] + \
                   [hostler(id=i + hostler_diesel, type="Electric") for i in range(hostler_electric)]
        for hostler_id in hostlers:
            self.parked_hostlers.put(hostler_id)

        # fixed processing time
        self.CONTAINERS_PER_CRANE_MOVE_MEAN = 2 / 60  # crane movement avg time: distance / speed = hr
        self. CRANE_MOVE_DEV_TIME = 1 / 3600  # crane movement speed deviation value: hr
        self.TRUCK_DIESEL_PERCENTAGE = 1
        self.TRUCK_ARRIVAL_MEAN = 2/60
        self.TRUCK_INGATE_TIME = 2/60
        self.TRUCK_OUTGATE_TIME = 2/60
        self.TRUCK_INGATE_TIME_DEV = 2/60
        self.TRUCK_OUTGATE_TIME_DEV = 2/60


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def generate_timetable(config):
    sim_length = float(config["simulation"]["length"])
    trains_per_day = int(config["simulation"]["train_number"])
    batch_size = int(config["simulation"]["train_batch_size"])

    if trains_per_day <= 0:
        raise ValueError("simulation.train_number must be > 0")

    all_arrivals = []
    day = 0
    while day * 24 < sim_length:
        base_time = day * 24 + 12.0     # start from noon
        day_arrivals = [round(base_time + random.uniform(-12, 12), 2) for _ in range(trains_per_day)]
        day_arrivals.sort()
        all_arrivals.extend(day_arrivals)
        day += 1
    all_arrivals = [t for t in sorted(all_arrivals) if t <= sim_length]

    if not all_arrivals:
        all_arrivals = [min(12.0, round(sim_length - 1.0, 2))]

    timetable = []
    for i, arrival in enumerate(all_arrivals):
        if i < len(all_arrivals) - 1:
            departure = all_arrivals[i + 1]
        else:
            departure = sim_length  # last one

        timetable.append({
            "train_id": i + 1,
            "arrival_time": round(float(arrival), 2),
            "departure_time": round(float(departure), 2),
            "empty_cars": 0,
            "full_cars": batch_size,
            "oc_number": batch_size,
            "truck_number": batch_size,
        })

    return timetable


def record_container_event(terminal, container, event_type, timestamp):
    if type(container) is str:
        container_string = container
    else:
        container_string = container.to_string()

    if container_string not in terminal.container_events:
        terminal.container_events[container_string] = {}
    terminal.container_events[container_string][event_type] = timestamp


def emission_calculation(terminal, status: str, move: str, vehicle: str, energy_type: str, travel_time: float) -> float:
    ems = terminal.ems

    move = move.lower()
    status = status.lower()
    vehicle = vehicle.lower()
    energy_type = energy_type.capitalize()

    # --- load consumption (unit: per lift)
    if move == "load":
        key = "crane_loaded" if status == "loaded" else "crane_idle"
        emission_unit = ems["load_consumption"][key][energy_type]
        emissions = emission_unit

    # --- trip consumption (unit: hr × travel_time)
    elif move == "trip":
        key = f"{vehicle}_{status}"  # e.g. hostler_loaded / truck_empty
        emission_unit = ems["trip_consumption"][key][energy_type]
        emissions = emission_unit * travel_time

    # --- side pick consumption (unit: per lift)
    elif move == "side":
        emission_unit = ems["side_pick_consumption"]["side"][energy_type]
        emissions = emission_unit

    else:
        raise ValueError(f"Unsupported move type '{move}' for vehicle '{vehicle}'.")

    return emissions


def record_emission(emission_records: list, vehicle_type: str, resource_id: str, track_id: str, train_id: str, container_id: str, event_type: str, zone: str, emission_value: float, travel_time: float, env_now: float) -> None:
    emission_records.append({
        "resource_type": vehicle_type.lower(),
        "resource_id": str(resource_id),
        "track_id":str(track_id),
        "train_id": str(train_id),
        "container_id": str(container_id),
        "event_type": event_type,
        "zone": zone,
        "energy_consumption(gal)": float(emission_value),
        "load/travel_time(hr)": float(travel_time),
        "record_timestamp": float(env_now),
    })


def save_emission_results(emission_records: pl.DataFrame, out_path: Path, filetype: str = "csv"):
    if out_path is None:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if filetype == "csv":
        emission_records.write_csv(out_path)
    elif filetype == "xlsx":
        emission_records.to_pandas().to_excel(out_path, index=False)
    else:
        raise ValueError("filetype must be 'csv' or 'xlsx'")


def truck_entry(env, terminal, truck, oc, train_schedule):
    ingate_request = terminal.in_gates.request()
    yield ingate_request

    # Assume each truck takes 1 OC, and drop OC to the closest parking lot according to triangular distribution
    # Assign IDs for OCs
    truck_travel_time = terminal.TRUCK_INGATE_TIME + random.uniform(0, terminal.TRUCK_INGATE_TIME_DEV)
    yield env.timeout(truck_travel_time)
    terminal.in_gates.release(ingate_request)
    record_container_event(terminal, oc.to_string(), 'truck_arrival', env.now)

    truck_trip_ems = emission_calculation(terminal, "loaded", "trip", "truck", "Diesel", travel_time=truck_travel_time)

    # Assume each truck takes 1 OC
    record_container_event(terminal, oc.to_string(), 'truck_dropoff', env.now)
    record_emission(emission_records, "truck", str(truck), 'N/A', str(train_schedule['train_id']), str(oc), "oc_prepare","parking_slots", truck_trip_ems, truck_travel_time, env.now)

    terminal.parking_slots.put(oc)


def empty_truck(env, terminal, truck_id, train_schedule):
    ingate_request = terminal.in_gates.request()
    yield ingate_request
    truck_travel_time = terminal.TRUCK_INGATE_TIME + random.uniform(0, terminal.TRUCK_INGATE_TIME_DEV)
    truck_trip_ems = emission_calculation(terminal, "empty", "trip", "truck", "Diesel", travel_time=truck_travel_time)
    record_emission(emission_records, "truck", str(truck_id), 'N/A', str(train_schedule['train_id']), 'N/A',"empty_truck_arrival","truck_parking", truck_trip_ems, truck_travel_time, env.now)
    yield env.timeout(truck_travel_time)
    terminal.in_gates.release(ingate_request)


def truck_arrival(env, terminal, train_schedule):
    truck_number = train_schedule["truck_number"]
    num_diesel = round(truck_number * terminal.TRUCK_DIESEL_PERCENTAGE)
    num_electric = truck_number - num_diesel

    trucks = [truck(type="Diesel", id=i, train_id=train_schedule['train_id']) for i in range(num_diesel)] + \
             [truck(type="Electric", id=i + num_diesel, train_id=train_schedule['train_id']) for i in
              range(num_electric)]

    terminal.total_oc[train_schedule['train_id']] = train_schedule["oc_number"]

    for oc_id in range(terminal.OC_COUNT[train_schedule['train_id']],
                       terminal.OC_COUNT[train_schedule['train_id']] + terminal.total_oc[train_schedule['train_id']]):
        terminal.oc_store.put(container(type='Outbound', id=oc_id, train_id=train_schedule['train_id']))
        oc_id += 1

    truck_entries_needed = train_schedule['oc_number']
    empty_truck_needed = len(trucks) - truck_entries_needed

    for i in range(truck_entries_needed):
        this_truck = trucks.pop(0)
        oc = yield terminal.oc_store.get()
        env.process(truck_entry(env, terminal, this_truck, oc, train_schedule))
        yield terminal.truck_store.put(this_truck)

    for i in range(empty_truck_needed):
        this_truck = trucks.pop(0)
        env.process(empty_truck(env, terminal, this_truck, train_schedule))
        yield terminal.truck_store.put(this_truck)

    if not terminal.all_trucks_arrived_events[train_schedule['train_id']].triggered:
        terminal.all_trucks_arrived_events[train_schedule['train_id']].succeed()
        print(f"all_trucks_arrived_events for {train_schedule['train_id']} is succeed.")


def check_ic_picked_complete(env, terminal, train_schedule):
    train_id = train_schedule['train_id']

    remaining_ic = sum((getattr(item, 'type', None) == 'Inbound') and (getattr(item, 'train_id', None) == train_id) for item in terminal.chassis.items)

    if (remaining_ic == 0 and terminal.train_ic_unload_events[train_schedule['train_id']].triggered
            and not terminal.train_ic_picked_events[train_schedule['train_id']].triggered):
        terminal.train_ic_picked_events[train_schedule['train_id']].succeed()
        print(f"train_ic_picked_events for {train_schedule['train_id']} is succeed.")


def crane_unload_process(env, terminal, train_schedule, track_id):
    train_id = train_schedule['train_id']

    def unload_crane_worker(env):
        crane = yield terminal.cranes_by_track[track_id].get()

        while True:
            if not any((item.train_id == train_id) for item in terminal.train_ic_stores.items):
                break

            ic = yield terminal.train_ic_stores.get(lambda x: x.train_id == train_id)
            crane_unload_time = terminal.CONTAINERS_PER_CRANE_MOVE_MEAN + random.uniform(0, terminal.CRANE_MOVE_DEV_TIME)
            crane_unload_ems = emission_calculation(terminal, "loaded", "load", "crane", "Diesel", travel_time=crane_unload_time)

            yield env.timeout(crane_unload_time)
            yield terminal.chassis.put(ic)

            record_container_event(terminal, ic.to_string(), f"crane_unload", env.now)
            record_emission(emission_records, "crane", str(crane.id), str(track_id), str(train_id), str(ic), "unload", "train_side", crane_unload_ems, crane_unload_time, env.now)
            env.process(container_process(env, terminal, train_schedule))

        yield terminal.cranes_by_track[track_id].put(crane)

    num_cranes = terminal.cranes_on_track[track_id]
    unload_processes = [env.process(unload_crane_worker(env)) for _ in range(num_cranes)]
    yield simpy.events.AllOf(env, unload_processes)


    if not terminal.train_ic_unload_events[train_id].triggered:
        terminal.train_ic_unload_events[train_id].succeed()


def crane_load_process(env, terminal, track_id, train_schedule):
    train_id = train_schedule['train_id']
    yield terminal.train_start_load_events[train_id]

    def load_crane_worker(env):
        crane = yield terminal.cranes_by_track[track_id].get()

        while True:
            # Check if there are any OCs left for this train
            if not any((item.type == 'Outbound' and item.train_id == train_id) for item in terminal.chassis.items):
                break

            oc = yield terminal.chassis.get(lambda x: x.type == 'Outbound' and x.train_id == train_id)

            crane_load_time = terminal.CONTAINERS_PER_CRANE_MOVE_MEAN + random.uniform(0, terminal.CRANE_MOVE_DEV_TIME)
            yield env.timeout(crane_load_time)

            crane_load_ems = emission_calculation(terminal, "loaded", "load", "crane", "Diesel", travel_time=crane_load_time)
            record_emission(emission_records, "crane", str(crane.id), str(track_id), str(train_id), oc.to_string(), "load", "train_side", crane_load_ems, crane_load_time, env.now)

            yield terminal.train_oc_stores.put(oc)
            record_container_event(terminal, oc.to_string(), f"crane_load", env.now)

        yield terminal.cranes_by_track[track_id].put(crane)

    num_cranes = terminal.cranes_on_track[track_id]
    load_processes = [env.process(load_crane_worker(env)) for _ in range(num_cranes)]
    yield simpy.events.AllOf(env, load_processes)

    if not terminal.train_end_load_events[train_id].triggered:
        terminal.train_end_load_events[train_id].succeed()
        print(f"train_end_load_events for {train_schedule['train_id']} is succeed.")


def get_hostler(terminal):
    parked_available = len(terminal.parked_hostlers.items) > 0
    active_available = len(terminal.active_hostlers.items) > 0
    if (active_available is False) and parked_available:
        assigned_hostler = terminal.parked_hostlers.get()
    else:
        assigned_hostler = terminal.active_hostlers.get()

    return assigned_hostler


def return_hostler(env, terminal, assigned_hostler, travel_time_to_active, travel_time_to_parking, active_hostlers_needed = None):
    if active_hostlers_needed is None:
        active_hostlers_needed = len(terminal.active_hostlers.get_queue) > 0
    if active_hostlers_needed:
        yield env.timeout(travel_time_to_active)
        yield terminal.active_hostlers.put(assigned_hostler)
    else:
        yield env.timeout(travel_time_to_parking)
        yield terminal.parked_hostlers.put(assigned_hostler)


def handle_oc(env, terminal, train_schedule):
    '''
    This function is called right after IC dropped off, such that accelerating container processing.
    Note: handle_remaining_oc is designed for imbalanced container flow.
    '''
    # hostler transport OC from parking slots
    assigned_hostler = yield get_hostler(terminal)
    oc = yield terminal.parking_slots.get(lambda x: (x.type == 'Outbound'))

    hostler_reposition_travel_time, d_r_dist, hostler_speed, veh_density = simulate_reposition_travel(oc, env.now, config=terminal.config)
    yield env.timeout(hostler_reposition_travel_time)
    record_container_event(terminal, oc.to_string(), 'hostler_pickup', env.now)
    hostler_reposition_ems = emission_calculation(terminal, "empty", "trip", "hostler", "Diesel", travel_time=hostler_reposition_travel_time)
    record_emission(emission_records, "hostler", assigned_hostler, 'N/A', str(train_schedule['train_id']), oc.to_string(), "hostler_pickup", "parking_slots", hostler_reposition_ems, hostler_reposition_travel_time, env.now)

    # OC side-pick
    side_pick_unload_time = 1 / 60 + random.uniform(0, 1 / 600)
    yield env.timeout(side_pick_unload_time)  # side-pick
    side_pick_ems = emission_calculation(terminal, "loaded", "side", "side_loading_crane", "Diesel", travel_time=side_pick_unload_time)
    record_emission(emission_records, "side_loading_crane", 'N/A', 'N/A', str(train_schedule['train_id']), oc.to_string(),"side_unload", "parking_slots", side_pick_ems, side_pick_unload_time, env.now)

    # transport OC from parking to chassis
    current_veh_num = len(terminal.parked_hostlers.items) + 1
    hostler_travel_time_to_chassis, hostler_dist, hostler_speed, hostler_density = simulate_hostler_track_travel(assigned_hostler, current_veh_num, config=terminal.config)
    yield env.timeout(hostler_travel_time_to_chassis)
    yield terminal.chassis.put(oc)
    record_container_event(terminal, oc.to_string(), 'hostler_dropoff', env.now)
    hostler_travel_ems = emission_calculation(terminal, "loaded", "trip", "hostler", "Diesel", travel_time=hostler_travel_time_to_chassis)
    record_emission(emission_records, "hostler", assigned_hostler, 'N/A', str(train_schedule['train_id']), oc.to_string(), "hostler_dropoff", "train_side", hostler_travel_ems, hostler_travel_time_to_chassis, env.now)

    # 5) hostler travel back
    yield from return_hostler(env, terminal, assigned_hostler, travel_time_to_active=0, travel_time_to_parking=0)


def container_process(env, terminal, train_schedule):
    """
    It is designed to transfer both inbound and outbound containers (IC-OC loop).
    The main simulation process is as follows:
    1. A hostler picks up an IC, and drops off IC at parking slot.
    2. A truck picks up the IC, and leaves the gate
    3. The hostler picks up an OC, and drops off OC at the chassis.
    4. Once all OCs are prepared (all_oc_prepared), triggers the event of crane loading.
    """

    # 1. An IC is waiting to be picked up by a hostler
    ic = yield terminal.chassis.get(lambda x: x.type == 'Inbound')
    assigned_hostler = yield get_hostler(terminal)
    # print(f"[HOSTLER & IC] hostler {assigned_hostler} picks up ic {ic}.")
    # print(f"parked hostler: \n{terminal.parked_hostlers.items}\n")

    # 2. hostler travel time 1: the empty hostler pick up the IC
    current_veh_num =  len(terminal.parked_hostlers.items)
    hostler_travel_time_to_track, hostler_dist, hostler_speed, hostler_density = simulate_hostler_track_travel(assigned_hostler, current_veh_num, config=terminal.config)
    yield env.timeout(hostler_travel_time_to_track)

    # 3. hostler travel time 2: the loaded hostler going to drop off the IC
    current_veh_num = len(terminal.parked_hostlers.items)
    hostler_travel_time_to_parking, hostler_dist, hostler_speed, hostler_density = simulate_hostler_track_travel(assigned_hostler, current_veh_num, config=terminal.config)
    yield env.timeout(hostler_travel_time_to_parking)
    yield terminal.parking_slots.put(ic)
    record_container_event(terminal, ic.to_string(), 'hostler_pickup', env.now)

    check_ic_picked_complete(env, terminal, train_schedule)

    # 3. Hostler drop off IC to parking slot
    side_pick_unload_time = 1 / 60 + random.uniform(0, 1 / 600)
    yield env.timeout(side_pick_unload_time)  # side-pick
    record_container_event(terminal, ic.to_string(), 'hostler_dropoff', env.now)
    side_pick_ems = emission_calculation(terminal, "loaded", "side", "side_loading_crane", "Diesel", travel_time=side_pick_unload_time)
    record_emission(emission_records, "side_loading_crane", 'N/A', 'N/A', str(train_schedule['train_id']), ic.to_string(),"side_unload", "truck_parking", side_pick_ems, side_pick_unload_time, env.now)
    yield from return_hostler(env, terminal, assigned_hostler, travel_time_to_active=0, travel_time_to_parking=0)
    # print(f"hostler drop off ic: {ic.to_string()}")

    # 4. Assign a truck to pick up IC
    ic_parking = yield terminal.parking_slots.get(lambda x: x.type == 'Inbound')
    # print(f"truck pick up ic: {ic_parking.to_string()}")

    assigned_truck = yield terminal.truck_store.get()
    truck_travel_time, truck_dist, truck_avg_speed, truck_avg_density = simulate_truck_travel(assigned_truck, train_schedule, terminal, config=terminal.config)
    yield env.timeout(truck_travel_time)

    record_container_event(terminal, ic_parking.to_string(), 'truck_pickup', env.now)
    truck_trip_ems = emission_calculation(terminal, "empty", "trip", "truck", "Diesel", travel_time=truck_travel_time)
    record_emission(emission_records, "truck", str(assigned_truck), 'N/A', str(train_schedule['train_id']), ic_parking,"truck_pickup", "truck_parking", truck_trip_ems, truck_travel_time, env.now)
    env.process(truck_exit(env, terminal, assigned_truck, ic_parking, train_schedule))

    # 5. reposition and handle OC
    env.process(handle_oc(env, terminal, train_schedule))


def handle_remaining_oc(env, terminal, train_schedule):
    train_id = train_schedule['train_id']

    while True:
        # 1) how many oc remaining? & how many oc prepared?
        outbound_remaining = sum((item.type == 'Outbound') and (item.train_id == train_id) for item in terminal.parking_slots.items)
        chassis_remaining = sum((item.type == 'Outbound') and (item.train_id == train_id) for item in terminal.chassis.items)

        if outbound_remaining == 0 and chassis_remaining == train_schedule['oc_number']:
            # print(f"Time {env.now:.3f}: All OC handled for train {train_id}.")
            if not terminal.train_oc_prepared_events[train_id].triggered:
                terminal.train_oc_prepared_events[train_id].succeed()
                print(f"train_oc_prepared_events for {train_schedule['train_id']} is succeed.")
            return

        # 2) hostler transport OC from parking slots
        assigned_hostler = yield get_hostler(terminal)
        oc = yield terminal.parking_slots.get(lambda x: (x.type == 'Outbound'))
        record_container_event(terminal, oc.to_string(), 'hostler_pickup', env.now)

        # 3) assign an empty-loaded hostler
        current_veh_num = len(terminal.parked_hostlers.items) + 1
        hostler_travel_time_to_parking, hostler_dist, hostler_speed, hostler_density = simulate_hostler_track_travel(assigned_hostler, current_veh_num, config=terminal.config)
        yield env.timeout(hostler_travel_time_to_parking)
        hostler_travel_ems = emission_calculation(terminal, "empty", "trip", "hostler", "Diesel", travel_time=hostler_travel_time_to_parking)
        record_emission(emission_records, "hostler", assigned_hostler, 'N/A', str(train_schedule['train_id']), oc.to_string(),"hostler_pickup", "parking_slots", hostler_travel_ems, hostler_travel_time_to_parking, env.now)

        # 4) side-pick loads an OC
        side_pick_unload_time = 1 / 60 + random.uniform(0, 1 / 600)
        yield env.timeout(side_pick_unload_time)  # side-pick
        side_pick_ems = emission_calculation(terminal, "loaded", "side", "side_loading_crane", "Diesel", travel_time=side_pick_unload_time)
        record_emission(emission_records, "side_loading_crane", 'N/A', 'N/A', str(train_schedule['train_id']), oc,"side_unload", "truck_parking", side_pick_ems, side_pick_unload_time, env.now)

        # 5) hostler loaded with an OC -> chassis
        hostler_travel_time_to_chassis, hostler_dist, hostler_speed, hostler_density = simulate_hostler_track_travel(assigned_hostler, current_veh_num, config=terminal.config)
        yield env.timeout(hostler_travel_time_to_chassis)
        yield terminal.chassis.put(oc)

        record_container_event(terminal, oc.to_string(), 'hostler_dropoff', env.now)
        hostler_travel_ems = emission_calculation(terminal, "loaded", "trip", "hostler", "Diesel", travel_time=hostler_travel_time_to_parking)
        record_emission(emission_records, "hostler", assigned_hostler, 'N/A', str(train_schedule['train_id']), oc.to_string(), "hostler_pickup", "train_side", hostler_travel_ems, hostler_travel_time_to_parking, env.now)

        # 6) hostler travel back
        yield from return_hostler(env, terminal, assigned_hostler,travel_time_to_active=0, travel_time_to_parking=0)


def truck_exit(env, terminal, truck, ic, train_schedule):
    out_gate_request = terminal.out_gates.request()
    yield out_gate_request
    truck_travel_time = terminal.TRUCK_OUTGATE_TIME + random.uniform(0, terminal.TRUCK_OUTGATE_TIME_DEV)
    yield env.timeout(truck_travel_time)
    truck_trip_ems = emission_calculation(terminal, "loaded", "trip", "truck", "Diesel", travel_time=truck_travel_time)
    record_emission(emission_records, "truck", str(truck), 'N/A', str(train_schedule['train_id']), ic,"truck_exit_gate", "truck_gate", truck_trip_ems, truck_travel_time, env.now)
    record_container_event(terminal, ic.to_string(), 'truck_exit', env.now)

    terminal.out_gates.release(out_gate_request)
    yield terminal.truck_store.put(truck)


def handle_train_departure(env, terminal, train_schedule, train_id, track_id, arrival_time):
    if env.now < train_schedule["departure_time"]:
        print(f"Time {env.now:.3f}: [EARLY] Train {train_id} departs from the track {track_id}.")
    elif env.now == train_schedule["departure_time"]:
        print(f"Time {env.now:.3f}: [In Time] Train {train_id} departs from the track {track_id}.")
    else:
        delay_time = env.now - train_schedule["departure_time"]
        print(f"Time {env.now:.3f}: [DELAYED] Train {train_id} has been delayed for {delay_time} hours from the track {track_id}.")

    for oc_id in range(terminal.OC_COUNT[train_schedule['train_id']],
                       terminal.OC_COUNT[train_schedule['train_id']] + train_schedule['oc_number']):
        record_container_event(terminal, f"OC-{oc_id}-Train-{train_schedule['train_id']}", 'train_depart',env.now)  # loop: assign container_id range(current_oc, current_oc + train_schedule['full_cars'])
    terminal.time_per_train[train_schedule['train_id']] = env.now - arrival_time

    terminal.train_departed_events[train_schedule['train_id']].succeed()
    print(f"train_departed_events for {train_schedule['train_id']} is succeed.")


def train_process_per_track(env, terminal, track_id, train_schedule, train_id, arrival_time):
    # Crane unload & hostler process ICs
    env.process(crane_unload_process(env, terminal, train_schedule, track_id))

    # check before crane loading
    # condition 1: all ic picked
    yield terminal.train_ic_picked_events[train_schedule['train_id']]

    # condition 2 & 3: no OCs on parking slots - OCs remaining -> process rest OCs; all OC prepared
    print(f"check # OCs on parking slots: {sum((item.type == 'Outbound') and (item.train_id == train_id) for item in terminal.parking_slots.items)}")
    if sum((item.type == 'Outbound') and (item.train_id == train_id) for item in terminal.parking_slots.items) >= 0:
        env.process(handle_remaining_oc(env, terminal, train_schedule))
    yield terminal.train_oc_prepared_events[train_schedule['train_id']]

    # crane loading
    # only when 1. all_ic_picked (chassis), 2. all_oc_picked (parking slots) & 3. all_oc_prepared (chassis) satisfied -> crane loading starts
    terminal.train_start_load_events[train_schedule['train_id']].succeed()
    print(f"train_start_load_events for {train_schedule['train_id']} is succeed.")
    env.process(crane_load_process(env, terminal, track_id=track_id, train_schedule=train_schedule))
    yield terminal.train_end_load_events[train_schedule['train_id']]

    # train departure
    handle_train_departure(env, terminal, train_schedule, train_id, track_id, arrival_time)
    yield terminal.tracks.put(track_id)


def initialize_train_events(env, terminal, train_id):
    for name in [
        "train_ic_unload_events",
        "train_oc_prepared_events",
        "train_ic_picked_events",
        "train_start_load_events",
        "train_end_load_events",
        "train_departed_events",
    ]:
        if not hasattr(terminal, name):
            setattr(terminal, name, {})
        d = getattr(terminal, name)

        if train_id not in d or d[train_id].triggered:
            d[train_id] = env.event()


def process_train_arrival(env, terminal, train_schedule):
    train_id = train_schedule["train_id"]
    arrival_time = train_schedule["arrival_time"]
    terminal.all_trucks_arrived_events[train_schedule['train_id']] = env.event() # condition for train arrival

    # Initialize dictionary
    delay_list = {}

    # Initialize IC & OC count (generating container ID)
    terminal.IC_COUNT[train_id] = 1
    terminal.OC_COUNT[train_id] = 1

    # All trucks arrive before train arrives
    env.process(truck_arrival(env, terminal, train_schedule))

    # Train arrival
    yield terminal.all_trucks_arrived_events[train_schedule['train_id']]
    if env.now <= arrival_time:
        yield env.timeout(arrival_time - env.now)
        print(f"Time {env.now:.3f}: [In Time] Train {train_schedule['train_id']}.")
        delay_time = 0
    else:
        delay_time = env.now - arrival_time
        if f"train_id_{train_id}" not in delay_list:
            delay_list[f"train_id_{train_id}"] = {}
        delay_list[f"train_id_{train_id}"]["arrival"] = delay_time
        print(f"Time {env.now:.3f}: [DELAYED] Train {train_schedule['train_id']} has been delayed for {delay_time} hours.")
    terminal.train_delay_time[train_schedule['train_id']] = delay_time
    terminal.train_pool_stores.put(train_schedule['train_id'])

    # Track assignment
    track_id = yield terminal.tracks.get()
    if track_id is None:
        print(f"Time {env.now:.3f}: Train {train_id} is waiting for a next available track.")
        return
    else:
        train_id = yield terminal.train_pool_stores.get()
        print(f"Time {env.now:.3f}: Train {train_id} has been assigned to track {track_id}.")

        # Initialize train with loaded ICs
        ic_num = terminal.IC_COUNT[train_schedule['train_id']]
        for ic_id in range(ic_num, ic_num + train_schedule['full_cars']):
            ic = container(type='Inbound', id=ic_id, train_id=train_schedule['train_id'])
            terminal.train_ic_stores.put(ic)
            record_container_event(terminal, ic.to_string(), 'train_arrival_expected', train_schedule['arrival_time'])
            record_container_event(terminal, ic.to_string(), 'train_arrival_actual', env.now)

        # Initialize events for each train
        initialize_train_events(env, terminal, train_id)
        # Train processed separately on each track
        env.process(train_process_per_track(env, terminal, track_id, train_schedule, train_id, arrival_time))


def run_simulation(train_consist_plan: pl.DataFrame, terminal: str, out_path=None):
    config = load_config("input/config.yaml")
    simulation_length = config["simulation"]["length"]
    layout = get_layout(config)
    train_timetable = generate_timetable(config)

    # Create environment & terminal resources
    env = simpy.Environment()
    terminal = Terminal(env, config=config)

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

    env.run(until=simulation_length)

# basic data structure
    container_data = (pl.from_dicts(
        [dict(event, **{'container_id': container_id}) for container_id, event in terminal.container_events.items()],
        infer_schema_length=None
        )
        .with_columns(
            pl.when(
                pl.col("truck_exit").is_not_null() & pl.col("train_arrival_expected").is_not_null()
            )
            .then(
                pl.col("truck_exit") - pl.col("train_arrival_expected")
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
        .select(pl.col("container_id"), pl.exclude("container_id"))
    )

    # OC stat
    container_data = container_data.with_columns(
        pl.col("container_id").str.extract(r"Train-(\d+)").cast(pl.Int64).alias("train_id"),
        pl.col("container_id").str.starts_with("OC").alias("is_oc"),
        pl.col("container_id").str.starts_with("IC").alias("is_ic")
    )

    # OC train actual arrival time
    train_arrival_df = (
        container_data
        .filter(pl.col("is_ic") & pl.col("train_arrival_actual").is_not_null())
        .group_by("train_id")
        .agg(pl.col("train_arrival_actual").mean()))
    container_data = container_data.join(train_arrival_df, on="train_id", how="left")
    container_data = container_data.rename({"train_arrival_actual_right": "train_arrival_actual_oc"})

    # OC train expected arrival time
    train_arrival_expected_df = (
        container_data
        .filter(pl.col("is_ic") & pl.col("train_arrival_expected").is_not_null())
        .group_by("train_id")
        .agg(pl.col("train_arrival_expected").mean())
    )
    container_data = container_data.join(train_arrival_expected_df, on="train_id", how="left")
    container_data = container_data.rename({"train_arrival_expected_right": "train_arrival_expected_oc"})

    container_data = container_data.drop(["is_oc", "is_ic", "train_id"])

    emission_records_df = pl.DataFrame(emission_records)

    if out_path is not None:
        container_data.write_excel(out_path / f"simulation_container_{daily_throughput}_track_{num_tracks}_crane_{num_cranes}_hostler_{num_hostlers}_results.xlsx")
        save_emission_results(emission_records_df, out_path / f"emission_container_{daily_throughput}_track_{num_tracks}_crane_{num_cranes}_hostler_{num_hostlers}_results.xlsx", filetype="xlsx")
    return container_data


if __name__ == "__main__":
    run_simulation(
        train_consist_plan=pl.read_csv(utilities.package_root() / 'input' / 'train_consist_plan.csv'),
        terminal="Allouez",
        out_path=utilities.package_root() / 'output' / 'double_track_results'
    )