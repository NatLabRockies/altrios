"""Core LIFTS data model: Terminal config, mutable TerminalState, and small
equipment dataclasses (container/truck/rtg/sts_crane/top_pick/yard_tractor/
chassis/vessel) plus the loggingLevel enum.
"""
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
    simulation (SimPy stores/resources, records) lives on the ``state``
    member (a :class:`TerminalState`).

    SimPy primitives are constructed declaratively from the active mode's
    :class:`~altrios.lifts.resources_decl.ResourceSpec` catalog (see
    :mod:`altrios.lifts.specs`). Callers pass ``resource_specs`` to select
    which pools to build; ``None`` (the default) builds the union of all
    three Phase 1 modes' specs so every flow module can run.
    """

    def __init__(self, env, config, layout,
                 log_level: "loggingLevel" = None,
                 resource_specs=None):
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

        # Only fields that are read from other modules are pulled out as
        # flat attributes. Pool capacities now live on the ResourceSpecs.
        self.track_number = yard_cfg["track_number"]
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
        # map). Used for diagnostic logging; the actual rail-track RTG
        # primitives are built from ``RAIL_TRACK_RTGS_BY_TRACK`` spec.
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

        # fixed processing-time parameters
        self.CONTAINERS_PER_CRANE_MOVE_MEAN = 2 / 60  # hr
        self.CRANE_MOVE_DEV_TIME = 1 / 3600  # hr
        self.TRUCK_DIESEL_PERCENTAGE = 1
        self.TRUCK_ARRIVAL_MEAN = 2 / 60
        self.TRUCK_INGATE_TIME = 2 / 60
        self.TRUCK_OUTGATE_TIME = 2 / 60
        self.TRUCK_INGATE_TIME_DEV = 2 / 60
        self.TRUCK_OUTGATE_TIME_DEV = 2 / 60

        # Mutable simulation state. The TerminalState constructor builds
        # SimPy primitives from the supplied (or default) resource specs.
        self.state = TerminalState(env, self, resource_specs=resource_specs)

    def log(self, level: "loggingLevel", msg: str) -> None:
        """Print msg iff its severity level is <= self.log_level."""
        utilities.log(level, msg)


class TerminalState:
    """Mutable per-run simulation state for a :class:`Terminal`.

    SimPy ``Store`` / ``Resource`` / ``Container`` primitives are
    instantiated from a list of
    :class:`~altrios.lifts.resources_decl.ResourceSpec`. Each spec's
    ``name`` becomes an attribute on this object (e.g. ``state.tracks``,
    ``state.main_stack_rtgs``). When a spec is partitioned (e.g. one Store
    per track id), the attribute is a ``dict`` keyed by the partition.

    Cross-cutting records that don't belong to any single pool (container
    events, per-train timing) live as plain attributes.
    """

    def __init__(self, env, terminal: "Terminal", resource_specs=None):
        # Local imports keep classes.py free of the spec catalog at module
        # import time (the spec catalog imports from classes.py).
        from altrios.lifts.resources_decl import build_state_from_specs
        from altrios.lifts import specs as spec_catalog

        self.env = env
        self._terminal = terminal

        # Default spec set: union of every Phase 1 mode's specs, deduped
        # by name. The dispatcher passes a narrower spec list when a mode
        # only needs a subset; tests can also override.
        if resource_specs is None:
            seen: set = set()
            combined: list = []
            for bundle in (
                spec_catalog.TRUCK_RAIL_SPECS,
                spec_catalog.RAIL_VESSEL_SPECS,
                spec_catalog.VESSEL_TRUCK_SPECS,
            ):
                for spec in bundle:
                    if spec.name not in seen:
                        seen.add(spec.name)
                        combined.append(spec)
            resource_specs = combined

        primitives = build_state_from_specs(
            env, resource_specs, terminal.config, {},
        )
        for name, prim in primitives.items():
            setattr(self, name, prim)

        # Records accumulated as the simulation runs.
        # container_events is a flat list of (container_id, event_type,
        # timestamp) tuples; it is pivoted to a wide DataFrame at
        # end-of-sim. Keeping it flat avoids per-event dict-of-dict
        # insertions in the hot path.
        self.container_events: list = []
        self.time_per_train: dict = {}
        self.train_delay_time: dict = {}
