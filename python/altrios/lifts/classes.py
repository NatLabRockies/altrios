"""LIFTS equipment dataclasses (container/truck/rtg/sts_crane/top_pick/
yard_tractor/chassis/vessel) plus the loggingLevel enum.

These are the small value types passed around between the SimPy actor
generators in :mod:`altrios.lifts.python_helpers` and
:mod:`altrios.lifts.yard_flow`. They carry display labels and pool
tags; they have no behaviour beyond ``to_string()`` for logging.
"""
from dataclasses import dataclass
from enum import IntEnum


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
    with."""
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
