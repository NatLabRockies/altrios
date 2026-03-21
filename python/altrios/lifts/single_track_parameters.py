from dataclasses import dataclass, field
from altrios.lifts.schedule import *
from enum import IntEnum


def train_arrival_parameters(train_consist_plan, terminal):
    TRAIN_TIMETABLE = build_train_timetable(train_consist_plan, terminal, as_dicts=False)
    return TRAIN_TIMETABLE


class loggingLevel(IntEnum):
    NONE = 1
    BASIC = 2
    DEBUG = 3


@dataclass
class LiftsState:
    # Fixed: Simulation files and hyperparameters
    log_level: loggingLevel = loggingLevel.DEBUG
    random_seed: int = 42
    sim_time: int = 24
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
    CONTAINERS_PER_CRANE_MOVE_MEAN: float = 1/60  # crane movement avg time: distance / speed = hr
    CRANE_MOVE_DEV_TIME: float = 1 / 3600  # crane movement speed deviation value: hr

    # Fixed: Hostler parameters
    HOSTLER_NUMBER: int = 4
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

    # Fixed: Load Emission matrix
    # Diesel unit: gallons/load
    # Electric unit: kMhr/load
    FULL_LOAD_EMISSIONS_RATES: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            'Truck': {'Diesel': 2.00, 'Electric': 14.2},
            'Hostler': {'Diesel': 0.35, 'Electric': 4.76},
            'Crane': {'Diesel': 0.26, 'Hybrid': 0.48},
        }
    )
    IDLE_LOAD_EMISSIONS_RATES: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            'Truck': {'Diesel': 0.60, 'Electric': 12.42},
            'Hostler': {'Diesel': 0.25, 'Electric': 1.28},
            'Crane': {'Diesel': 0.26, 'Hybrid': 0.48},
        }
    )

    # Diesel unit: gallons/meters
    # Electric unit: kMhr/meters
    FULL_TRIP_EMISSIONS_RATES: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            'Truck': {'Diesel': 1.94, 'Electric': 3.66},
            'Hostler': {'Diesel': 0.35, 'Electric': 4.76},
        }
    )

    IDLE_TRIP_EMISSIONS_RATES: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            'Truck': {'Diesel': 0.60, 'Electric': 1.42},
            'Hostler': {'Diesel': 0.25, 'Electric': 1.28},
        }
    )

    # Various: tracking container number
    IC_NUM: int = 1
    OC_NUM: int = 1

    time_per_train: dict[str, int] = field(default_factory=lambda: {})  # total processing time for a train
    train_delay_time: dict[str, int] = field(default_factory=lambda: {})  # delay time for a train
    ## Notice: Hostler, truck and crane performance are reflected on the excel output
    container_events: dict = field(default_factory=lambda: {})  # Dictionary to store container event data

    def initialize_from_consist_plan(self, train_consist_plan):
        self.train_consist_plan = train_consist_plan
        self.TRAIN_TIMETABLE = train_arrival_parameters(self.train_consist_plan, self.terminal)  # a dictionary


    def initialize(self):
        self.CRANE_LOAD_CONTAINER_TIME_MEAN = (self.CONTAINERS_PER_CAR * (
                    2 * self.CONTAINER_TAL + self.CONTAINER_WID)) / self.CONTAINERS_PER_CRANE_MOVE_MEAN  # hr
        self.CRANE_UNLOAD_CONTAINER_TIME_MEAN = (self.CONTAINERS_PER_CAR * (
                    2 * self.CONTAINER_TAL + self.CONTAINER_WID)) / self.CONTAINERS_PER_CRANE_MOVE_MEAN  # hr
        # Trains
        if self.train_consist_plan.height > 0:
            self.initialize_from_consist_plan(self.train_consist_plan)

    def __post_init__(self):
        self.initialize()

state = LiftsState()