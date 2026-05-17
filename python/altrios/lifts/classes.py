import simpy
from dataclasses import dataclass
from enum import IntEnum

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
    """Static configuration of a terminal: layout, design parameters, operating
    constants, and pre-computed counts. Everything that mutates during the
    simulation (SimPy stores/resources, per-train events, counters, records)
    lives on the ``state`` member (a :class:`TerminalState`).
    """

    def __init__(self, env, config, layout, truck_capacity, chassis_count):
        self.config = config

        sim_cfg = config["simulation"]
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

        # cranes per track (static map; the SimPy stores holding the crane
        # objects themselves live on ``state.cranes_by_track``)
        self.cranes_on_track = {
            track_id: term_cfg["cranes_per_track"]
            for track_id in range(1, self.track_number + 1)
        }

        # emissions config
        self.ems = ems_cfg

        # sizing inputs (informational; live stores sized from these on state)
        self.truck_capacity = truck_capacity
        self.chassis_count = chassis_count

        # fixed processing-time parameters
        self.CONTAINERS_PER_CRANE_MOVE_MEAN = 2 / 60  # hr
        self.CRANE_MOVE_DEV_TIME = 1 / 3600  # hr
        self.TRUCK_DIESEL_PERCENTAGE = 1
        self.TRUCK_ARRIVAL_MEAN = 2 / 60
        self.TRUCK_INGATE_TIME = 2 / 60
        self.TRUCK_OUTGATE_TIME = 2 / 60
        self.TRUCK_INGATE_TIME_DEV = 2 / 60
        self.TRUCK_OUTGATE_TIME_DEV = 2 / 60

        # Mutable simulation state lives here.
        self.state = TerminalState(env, self)


class TerminalState:
    """Mutable per-run simulation state for a :class:`Terminal`.

    Holds the SimPy ``Store``/``Resource`` instances whose contents evolve as
    the simulation runs, the per-train SimPy events, and the dictionaries that
    accumulate container events, counters, and timing records.
    """

    def __init__(self, env, terminal: "Terminal"):
        self.env = env

        # Track availability (one slot per track id)
        self.tracks = simpy.Store(env, capacity=terminal.track_number)
        for track_id in range(1, self.tracks.capacity + 1):
            self.tracks.put(track_id)

        # Cranes: one Store per track holding the live crane objects.
        self.cranes_by_track = {
            track_id: simpy.Store(env, capacity=num_cranes)
            for track_id, num_cranes in terminal.cranes_on_track.items()
        }
        for track_id, num_cranes in terminal.cranes_on_track.items():
            for crane_number in range(1, num_cranes + 1):
                self.cranes_by_track[track_id].put(
                    crane(type="Hybrid", id=crane_number, track_id=track_id)
                )

        # Gates
        self.in_gates = simpy.Resource(env, terminal.in_gate_numbers)
        self.out_gates = simpy.Resource(env, terminal.out_gate_numbers)

        # Container/queue stores
        self.train_pool_stores = simpy.Store(env, capacity=99999)
        self.train_oc_stores = simpy.Store(env, capacity=99999)
        self.oc_store = simpy.Store(env, capacity=99999)
        self.truck_store = simpy.Store(env, capacity=999999999)

        # ----- Containers in flight: keyed dict-of-Store, NOT FilterStore -----
        # FilterStore.get(lambda ...) is O(N) per call; with thousands of
        # containers in flight this dominates runtime. Splitting by the keys
        # the filters were checking (train_id, type) reduces each get/put to
        # O(1). Counters track per-train totals where consumers pull "any"
        # item but need a per-train completion check.

        # ICs awaiting crane unload, one Store per train.
        self.train_ic_stores: dict = {}

        # Chassis: IC side is shared (container_process pulls any Inbound);
        # OC side is per-train (load_crane_worker pulls by train_id).
        self.chassis_ic_store = simpy.Store(env)
        self.chassis_oc_stores: dict = {}
        self.chassis_ic_count_by_train: dict = {}
        self.chassis_oc_count_by_train: dict = {}

        # Parking slots: IC side is per-train (truck pulls by train_id);
        # OC side is shared (hostler pulls any Outbound, regardless of train).
        self.parking_ic_stores: dict = {}
        self.parking_oc_store = simpy.Store(env)
        self.parking_oc_count_by_train: dict = {}

        # Hostlers: split into parked/active pools
        hostler_total = terminal.hostler_number
        hostler_diesel = round(hostler_total * terminal.hostler_diesel_percentage)
        hostler_electric = hostler_total - hostler_diesel
        self.parked_hostlers = simpy.Store(env, capacity=hostler_total)
        self.active_hostlers = simpy.Store(env, capacity=hostler_total)
        hostlers_list = (
            [hostler(id=i, type="Diesel") for i in range(hostler_diesel)] +
            [hostler(id=i + hostler_diesel, type="Electric") for i in range(hostler_electric)]
        )
        for h in hostlers_list:
            self.parked_hostlers.put(h)

        # Per-train SimPy events
        self.all_trucks_arrived_events: dict = {}
        self.train_ic_unload_events: dict = {}
        self.train_ic_picked_events: dict = {}
        self.train_oc_prepared_events: dict = {}
        self.train_start_load_events: dict = {}
        self.train_end_load_events: dict = {}
        self.train_departed_events: dict = {}

        # Per-train counters and totals
        self.IC_COUNT: dict = {}
        self.OC_COUNT: dict = {}
        self.total_ic: dict = {}
        self.total_oc: dict = {}

        # Records accumulated as the simulation runs.
        # container_events is a flat list of (container_id, event_type, timestamp)
        # tuples; it is pivoted to a wide DataFrame at end-of-sim. Keeping it
        # flat avoids per-event dict-of-dict insertions in the hot path.
        self.container_events: list = []
        self.time_per_train: dict = {}
        self.train_delay_time: dict = {}

    # ----- Per-train Store accessors (lazy creation) ----------------------
    # Using helpers rather than dict.setdefault(...) avoids constructing a
    # throwaway simpy.Store every call when the key already exists.

    def train_ic_store(self, train_id):
        s = self.train_ic_stores.get(train_id)
        if s is None:
            s = simpy.Store(self.env)
            self.train_ic_stores[train_id] = s
        return s

    def chassis_oc_store(self, train_id):
        s = self.chassis_oc_stores.get(train_id)
        if s is None:
            s = simpy.Store(self.env)
            self.chassis_oc_stores[train_id] = s
        return s

    def parking_ic_store(self, train_id):
        s = self.parking_ic_stores.get(train_id)
        if s is None:
            s = simpy.Store(self.env)
            self.parking_ic_stores[train_id] = s
        return s
