import polars as pl
from dataclasses import dataclass, field
from enum import IntEnum
from altrios.lifts.utilities import load_config

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
