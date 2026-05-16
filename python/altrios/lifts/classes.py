import polars as pl
import simpy
from dataclasses import dataclass, field
from enum import IntEnum
from altrios.lifts.utilities import load_config
from altrios.lifts.distances import calculate_distances

class loggingLevel(IntEnum):
    NONE = 1
    BASIC = 2
    DEBUG = 3

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
    def __init__(self, env, config, layout, truck_capacity, chassis_count):
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
        distances = calculate_distances(config=config, config_path=None, actual_railcars=None)
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
        
@dataclass
class LiftsState:
    # Fixed: Simulation files and hyperparameters
    log_level: loggingLevel = loggingLevel.DEBUG
    random_seed: int = 42
    sim_time: int = 20 * 24
    terminal: str = 'Allouez'  # Choose 'Hibbing' or 'Allouez'
    train_consist_plan: pl.DataFrame = field(default_factory=lambda: pl.DataFrame())


    # Fixed: Train parameters
    ## Train timetable: train_units, train arrival time
    TRAIN_INSPECTION_TIME: float = 1 / 60  # hr

    # Fixed: Yard parameters
    TRACK_NUMBER: int = 1

    # Fixed: Crane parameters
    # Container parameters: calculate crane horizontal and vertical processing time
    # Current 1 TEU = 20 ft long, 8 ft wide, and 8.6 ft tall; optional 2 TEU = 40 ft long, 8 ft wide, and 8.6 ft tall
    CONTAINERS_PER_CAR: int = 1
    CONTAINER_LEN: float = 20
    CONTAINER_WID: float = 8
    CONTAINER_TAL: float = 8.6
    # crane moving distance = 2 * CONTAINER_WID + CONTAINER_WID = 24.6 ft
    # crane movement speed mean value: 10ft/min = 600 ft/hr
    CRANE_NUMBER: int = 2
    CRANE_DIESEL_PERCENTAGE: float = 1
    CONTAINERS_PER_CRANE_MOVE_MEAN: float = 2/60  # crane movement avg time: distance / speed = hr
    CRANE_MOVE_DEV_TIME: float = 1 / 3600  # crane movement speed deviation value: hr

    # Fixed: Hostler parameters
    HOSTLER_NUMBER: int = 1
    HOSTLER_DIESEL_PERCENTAGE: float = 1
    # Fixed hostler travel time (** will update with density-speed/time functions later soon)
    CONTAINERS_PER_HOSTLER: int = 1  # hostler capacity

    # Fixed: Truck parameters
    TRUCK_DIESEL_PERCENTAGE: float = 1
    TRUCK_ARRIVAL_MEAN: float = 2/60  # hr, assume all containers are well-prepared
    TRUCK_INGATE_TIME: float = 2/60 # hr
    TRUCK_OUTGATE_TIME: float = 2/60  # hr
    TRUCK_INGATE_TIME_DEV: float = 2/60  # hr
    TRUCK_OUTGATE_TIME_DEV: float = 2/60  # hr
    TRUCK_TO_PARKING: float = 2/60 # hr

    # Fixed: Gate parameters
    IN_GATE_NUMBERS: int = 60  # test queuing module with 1; normal operations with 6
    OUT_GATE_NUMBERS: int = 60


    # Fixed: Emission matrix (ZANZEFF reports, 2022)
    ENERGY_CONSUMPTION: dict[str, dict[str, dict[str, float]]] = field(
        default_factory=lambda: {
            "LOAD_CONSUMPTION": {
                "Crane_Loaded": {"Diesel": 0.26, "Hybrid": 0.48},  # H idling: kWh/load
                "Crane_Idle": {"Diesel": 0.02, "Hybrid": 0.024},
            },

            "TRIP_CONSUMPTION": {
                "Hostler_Empty": {"Diesel": 1.11, "Electric": 2.78},  # gallons/hr, kWh/hr
                "Hostler_Loaded": {"Diesel": 1.94, "Electric": 3.66},
                "Truck_Empty": {"Diesel": 1.11, "Electric": 2.68},
                "Truck_Loaded": {"Diesel": 1.94, "Electric": 3.66},
            },

            "SIDE_PICK_CONSUMPTION": {
                "Side": {"Diesel": 2.88, "Electric": 0.21},  # per lift
            },
        }
    )

    # Various: tracking container number
    IC_NUM: int = 1
    OC_NUM: int = 1

    # Various
    time_per_train: dict[str, int] = field(default_factory=lambda: {})  # total processing time for a train
    train_delay_time: dict[str, int] = field(default_factory=lambda: {})  # delay time for a train
    ## Notice: Hostler, truck and crane performance are reflected on the excel output
    container_events: dict = field(default_factory=lambda: {})  # Dictionary to store container event data

    def initialize_from_consist_plan(self, train_consist_plan):
        self.train_consist_plan = train_consist_plan

    def initialize(self):
        self.CRANE_LOAD_CONTAINER_TIME_MEAN = (self.CONTAINERS_PER_CAR * (
                    2 * self.CONTAINER_TAL + self.CONTAINER_WID)) / self.CONTAINERS_PER_CRANE_MOVE_MEAN  # hr
        self.CRANE_UNLOAD_CONTAINER_TIME_MEAN = (self.CONTAINERS_PER_CAR * (
                    2 * self.CONTAINER_TAL + self.CONTAINER_WID)) / self.CONTAINERS_PER_CRANE_MOVE_MEAN  # hr
        # Trains
        if self.train_consist_plan.height > 0:
            self.initialize_from_consist_plan(self.train_consist_plan)

    def __post_init__(self):
        config = load_config()
        vehicles = config.get("vehicles", {})
        self.sim_time = vehicles["simulation_duration"]
        self.CRANE_NUMBER = vehicles["CRANE_NUMBER"]
        self.HOSTLER_NUMBER = vehicles["HOSTLER_NUMBER"]
