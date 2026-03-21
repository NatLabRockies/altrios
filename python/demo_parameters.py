from dataclasses import dataclass, field
from altrios.lifts.schedule import *
from enum import IntEnum

def train_arrival_parameters(train_consist_plan, terminal):
    TRAIN_TIMETABLE = build_train_timetable(train_consist_plan, terminal, as_dicts = False)
    return TRAIN_TIMETABLE

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
    sim_time: int = 1100
    terminal: str = 'Allouez'    # Choose 'Hibbing' or 'Allouez'
    train_consist_plan: pl.DataFrame = field(default_factory=lambda: pl.DataFrame())

    # Fixed: Train parameters
    ## Train timetable: train_units, train arrival time
    TRAIN_INSPECTION_TIME: float = 10/60    # hr

    # Fixed: Yard parameters
    TRACK_NUMBER: int = 2

    # Fixed: Crane parameters
    # Container parameters: calculate crane horizontal and vertical processing time
    # Current 1 TEU = 20 ft long, 8 ft wide, and 8.6 ft tall; optional 2 TEU = 40 ft long, 8 ft wide, and 8.6 ft tall
    CONTAINERS_PER_CAR: int = 1
    CONTAINER_LEN: float = 20
    CONTAINER_WID: float = 8
    CONTAINER_TAL: float = 8.6
    # CRANE_NUMBER = int(input("Enter the number of crane: "))
    CRANE_NUMBER: int = 1
    CRANE_HYBRID_PERCENTAGE: float = 0.5
    CONTAINERS_PER_CRANE_MOVE_MEAN: float = 1/60   # crane movement speed mean value: 10ft/min = 600 ft/hr, crane speed
    CRANE_MOVE_DEV_TIME: float = 1/6000 # crane movement speed deviation value: hr

    # Fixed: Hostler parameters
    # HOSTLER_NUMBER = int(input("Enter the number of hostler: "))
    HOSTLER_NUMBER: int = 1
    HOSTLER_DIESEL_PERCENTAGE: float = 0.6
    # Fixed hostler travel time (** will update with density-speed/time functions later soon)
    CONTAINERS_PER_HOSTLER: int = 1  # hostler capacity
    HOSTLER_SPEED_LIMIT: float = 20*5280   # hostler speed: ft/hr
    HOSTLER_TRANSPORT_CONTAINER_TIME: float = 0.1    # hostler travel time of picking up IC: hr, triangular distribution
    HOSTLER_FIND_CONTAINER_TIME: float = 1/5  # hostler travel time between dropping off IC and picking up OC: hr, triangular distribution

    # Fixed: Truck parameters
    TRUCK_DIESEL_PERCENTAGE: float = 0.6
    # TRUCK_ARRIVAL_MEAN: float = 40/60   # hr, arrival rate depends on the gap between last train departure and next train arrival
    TRUCK_INGATE_TIME: float = 1/60    # hr
    TRUCK_OUTGATE_TIME: float = 2/60    # hr
    TRUCK_INGATE_TIME_DEV: float = 1/60    # hr
    TRUCK_OUTGATE_TIME_DEV: float = 1/60    # hr
    TRUCK_TO_PARKING: float = 2/60    # hr
    TRUCK_SPEED_LIMIT: float = 20*5280   # ft/hr
    TRUCK_TRANSPORT_CONTAINER_TIME: float = 1/6  # hr, triangular distribution, will be updated with truck density-speed function

    # Fixed: Gate parameters
    IN_GATE_NUMBERS: int = 6  # test queuing module with 1; normal operations with 6
    OUT_GATE_NUMBERS: int = 6
    last_leave_time: float = 0
    truck_arrival_time: list[float] = field(default_factory = lambda: [])   # **trucks arrive according to train departure and arrival gap

    # Fixed: Emission matrix
    # Diesel unit: gallons/hr
    # Electric unit: Mhr
    IDLE_EMISSIONS_RATES: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            'Truck': {'Diesel': 5.2, 'Electric': 2.4},
            'Hostler': {'Diesel': 6.2, 'Electric': 2.4},
            'Crane': {'Diesel': 40.3, 'Hybrid': 30.5},
        }
    )

    FULL_EMISSIONS_RATES: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            'Truck': {'Diesel': 20.7, 'Electric': 10.2},
            'Hostler': {'Diesel': 15.7, 'Electric': 10.2},
            'Crane': {'Diesel': 60.3, 'Hybrid': 50.5},
        }
    )

    # Various: tracking container number
    IC_NUM: int = 1     # tracking
    OC_NUM: int = 1

    ## Train performance, dictionary: id, time
    time_per_train: dict[str, int] = field(default_factory=lambda: {})  # total processing time for a train
    train_delay_time: dict[str, int] = field(default_factory=lambda: {})  # delay time for a train
    ## Notice: Hostler, truck and crane performance are reflected on the excel output
    container_events: dict = field(default_factory=lambda: {})  # Dictionary to store container event data


    def initialize_from_consist_plan(self, train_consist_plan):
        self.train_consist_plan = train_consist_plan
        self.TRAIN_TIMETABLE = train_arrival_parameters(self.train_consist_plan, self.terminal) # a dictionary
        # self.CARS_LOADED_ARRIVAL = int(float(self.TRAIN_TIMETABLE['full_cars']))
        # self.CARS_EMPTY_ARRIVAL = int(float(self.TRAIN_TIMETABLE['empty_cars']))
        # self.TRAIN_ARRIVAL_HR = self.TRAIN_TIMETABLE['arrival_time']
        # self.TRAIN_DEPARTURE_HR = self.TRAIN_TIMETABLE['departure_time']
        # self.TRAIN_UNITS = self.CARS_LOADED_ARRIVAL + self.CARS_EMPTY_ARRIVAL
        # self.TRAIN_SPOTS = self.TRAIN_UNITS
        #
        # # Containers
        # self.INBOUND_CONTAINER_NUMBER = self.CARS_LOADED_ARRIVAL
        # #df = outbound_containers()
        # # TODO: confirm expected source of Outbound_Num; expected input file not available
        # self.OUTBOUND_CONTAINER_NUMBER = self.INBOUND_CONTAINER_NUMBER  #df.loc[df['Train_ID'] == TRAIN_ID, 'Outbound_Num'].values[0]
        #
        # # Chassis
        # self.CHASSIS_NUMBER = self.TRAIN_UNITS
        #
        # # Trucks
        # self.TRUCK_NUMBERS = max(self.INBOUND_CONTAINER_NUMBER, self.OUTBOUND_CONTAINER_NUMBER)

    def initialize(self):
        self.CRANE_LOAD_CONTAINER_TIME_MEAN = (self.CONTAINERS_PER_CAR*(2*self.CONTAINER_TAL+self.CONTAINER_WID))/self.CONTAINERS_PER_CRANE_MOVE_MEAN   # hr
        self.CRANE_UNLOAD_CONTAINER_TIME_MEAN = (self.CONTAINERS_PER_CAR*(2*self.CONTAINER_TAL+self.CONTAINER_WID))/self.CONTAINERS_PER_CRANE_MOVE_MEAN # hr
        # Trains
        if self.train_consist_plan.height > 0:
            self.initialize_from_consist_plan(self.train_consist_plan)

    def __post_init__(self):
        self.initialize()

state = LiftsState()