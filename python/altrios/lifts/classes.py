"""Core LIFTS data model: Terminal config, mutable TerminalState, and small
equipment dataclasses (container/truck/rtg/sts_crane/top_pick/yard_tractor/
chassis/vessel) plus the loggingLevel enum.
"""
import simpy
from dataclasses import dataclass
from enum import IntEnum

from altrios.lifts import utilities
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
class truck:
    """Drayage truck visiting the terminal."""
    type: str = 'Diesel'
    id: int = 0
    train_id: int = 0

    def to_string(self) -> str:
        return f'{self.id}-Track-{self.train_id}-{self.type}'


@dataclass
class rtg:
    """Rubber-tired gantry crane. Used at both the rail tracks
    (``pool='rail_track'``, ``track_id`` set) and the main container stack
    (``pool='main_stack'``, ``track_id=0``)."""
    type: str = 'Diesel'
    id: int = 0
    pool: str = ''            # 'rail_track' or 'main_stack'
    track_id: int = 0         # only meaningful for pool == 'rail_track'

    def to_string(self) -> str:
        if self.pool == 'rail_track':
            return f'{self.id}-Track-{self.track_id}-{self.type}'
        return f'{self.id}-{self.pool}-{self.type}'


@dataclass
class sts_crane:
    """Ship-to-shore crane stationed at a berth (vessel <-> shore lift)."""
    type: str = 'Diesel'
    id: int = 0
    berth_id: int = 0

    def to_string(self) -> str:
        return f'{self.id}-Berth-{self.berth_id}-{self.type}'


@dataclass
class top_pick:
    """Top-pick handler servicing the main stack as a flexible alternative
    to the RTG. Each top_pick carries the id of the safety car it operates
    with (combined Store entry in Phase 1; may be split in Phase 2)."""
    type: str = 'Diesel'
    id: int = 0
    safety_car_id: int = 0

    def to_string(self) -> str:
        return f'{self.id}-Safety-{self.safety_car_id}-{self.type}'


@dataclass
class yard_tractor:
    """Yard tractor (a.k.a. hostler) hauling chassis around the terminal.
    Distinct pools (``main_yard_tractors`` water<->stack, ``rail_yard_tractors``
    rail<->stack) hold yard_tractor instances tagged with their pool."""
    type: str = 'Diesel'
    id: int = 0
    pool: str = ''            # e.g. 'main' or 'rail'

    def to_string(self) -> str:
        return f'{self.id}-{self.pool}-{self.type}'


@dataclass
class chassis:
    """Chassis carrying a container. Two pools exist (terminal vs road) for
    different physical equipment, distinguished by the ``pool`` field; the
    dataclass itself is shared because chassis carry no behavior."""
    type: str = 'Standard'
    id: int = 0
    pool: str = ''            # 'terminal' or 'road'

    def to_string(self) -> str:
        return f'{self.pool}-chassis-{self.id}-{self.type}'


@dataclass
class vessel:
    """Vessel making a berth call; carries an id and container counts."""
    id: int = 0
    name: str = ''
    inbound_containers: int = 0
    outbound_containers: int = 0

    def to_string(self) -> str:
        return f'Vessel-{self.id}-{self.name}'


class Terminal:
    """Static configuration of a terminal: layout, design parameters, operating
    constants, and pre-computed counts. Everything that mutates during the
    simulation (SimPy stores/resources, per-train events, counters, records)
    lives on the ``state`` member (a :class:`TerminalState`).
    """

    def __init__(self, env, config, layout, truck_capacity, chassis_count,
                 log_level: "loggingLevel" = None):
        self.config = config

        # Run-scoped log threshold. Stored on Terminal for inspection and to
        # support future per-terminal differentiation; the utilities module
        # also holds a synchronized copy for sites that run before Terminal
        # construction.
        self.log_level: loggingLevel = log_level if log_level is not None else loggingLevel.BASIC
        utilities.set_log_level(self.log_level)

        yard_cfg = config["yard"]
        term_cfg = config["terminal"]
        gate_cfg = config["gates"]
        energy_use_cfg = config["energy_use"]

        # Only fields that are read from other modules, or needed below to
        # size SimPy resources, are pulled out as flat attributes. Everything
        # else stays accessible via ``self.config``.
        self.track_number = yard_cfg["track_number"]
        self.hostler_number = term_cfg["hostler_number"]
        self.hostler_diesel_percentage = term_cfg["hostler_diesel_percentage"]
        self.in_gate_numbers = gate_cfg["in_gate_numbers"]
        self.out_gate_numbers = gate_cfg["out_gate_numbers"]

        # layout
        self.layout = layout
        distances = calculate_distances(config=config, config_path=None, actual_railcars=None)
        self.distances = distances
        self.yard_length = distances["yard_length"]
        self.track_capacity = distances["n_max"]

        # Cranes per track. The YAML's ``cranes_per_track`` may be either a
        # scalar (broadcast to every track) or a list of length
        # ``track_number`` (one entry per track, 1-indexed in the resulting
        # map). The SimPy stores holding the crane objects themselves live on
        # ``state.cranes_by_track``.
        cpt_cfg = term_cfg["cranes_per_track"]
        if isinstance(cpt_cfg, (list, tuple)):
            if len(cpt_cfg) != self.track_number:
                raise ValueError(
                    f"cranes_per_track has {len(cpt_cfg)} entries but "
                    f"track_number is {self.track_number}"
                )
            self.cranes_on_track = {i + 1: int(n) for i, n in enumerate(cpt_cfg)}
        else:
            self.cranes_on_track = {
                track_id: int(cpt_cfg)
                for track_id in range(1, self.track_number + 1)
            }

        # energy-use config (per-event diesel/electric consumption rates)
        self.energy_use_config = energy_use_cfg

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

    def log(self, level: "loggingLevel", msg: str) -> None:
        """Print msg iff its severity level is <= self.log_level."""
        utilities.log(level, msg)


class TerminalState:
    """Mutable per-run simulation state for a :class:`Terminal`.

    Holds the SimPy ``Store``/``Resource`` instances whose contents evolve as
    the simulation runs, the per-train SimPy events, and the dictionaries that
    accumulate container events, counters, and timing records.
    """

    def __init__(self, env, terminal: "Terminal"):
        self.env = env
        self._terminal = terminal

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
                    rtg(type="Hybrid", id=crane_number,
                        pool="rail_track", track_id=track_id)
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
        # Today's hostlers are rail-side workers; tag them pool='rail'.
        hostlers_list = (
            [yard_tractor(id=i, type="Diesel", pool="rail")
             for i in range(hostler_diesel)] +
            [yard_tractor(id=i + hostler_diesel, type="Electric", pool="rail")
             for i in range(hostler_electric)]
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

    def in_flight_hostler_count(self) -> int:
        """Hostlers currently checked out (neither idling in the parked pool
        nor sitting in the active pool waiting to be re-dispatched). This
        is the yard-wide congestion measure used by the hostler-speed model
        in :mod:`altrios.lifts.distances`; it is the hostler analog of
        ``simulate_truck_travel``'s ``truck_number - truck_store.items``."""
        return (
            self._terminal.hostler_number
            - len(self.parked_hostlers.items)
            - len(self.active_hostlers.items)
        )
