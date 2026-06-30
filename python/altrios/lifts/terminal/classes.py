"""LIFTS equipment dataclasses (container/truck/rtg/sts_crane/top_pick/
yard_tractor/chassis/vessel) plus the loggingLevel enum.

These are the small value types passed around between the SimPy actor
generators in :mod:`altrios.lifts.terminal.python_helpers` and
:mod:`altrios.lifts.terminal.yard_flow`. They carry display labels and pool
tags; they have no behaviour beyond ``to_string()`` for logging.
"""
from dataclasses import dataclass
from enum import IntEnum


class loggingLevel(IntEnum):
    """Verbosity level for the terminal SimPy actors' ``log()`` helper.

    Attributes
    ----------
    NONE
        Suppress all log output.
    BASIC
        Emit one line per major actor event (arrival, completion).
    DEBUG
        Emit detailed step-by-step traces.
    """

    NONE = 1
    BASIC = 2
    DEBUG = 3

@dataclass
class container:
    """One intermodal container moving through the terminal.

    Attributes
    ----------
    type : str
        Direction relative to the terminal: ``'Outbound'``,
        ``'Inbound'``, or another mode-specific label. Drives the
        prefix returned by :meth:`to_string`.
    id : int
        Container ordinal within the run.
    train_id : int
        Identifier of the train this container is assigned to (0 for
        non-rail flows).
    """

    type: str = 'Outbound'
    id: int = 0
    train_id: int = 0

    def to_string(self) -> str:
        """Return a human-readable label like ``OC-12-Train-3``."""
        if self.type == 'Outbound':
            prefix = 'OC'
        elif self.type == 'Inbound':
            prefix = 'IC'
        else:
            prefix = 'C'
        return f"{prefix}-{self.id}-Train-{self.train_id}"


@dataclass
class truck:
    """Drayage truck visiting the terminal.

    Attributes
    ----------
    type : str
        Powertrain label (``'Diesel'``, ``'Electric'``, ...) used by
        consumption tracking and the display string.
    id : int
        Truck ordinal within the run.
    train_id : int
        Associated train identifier when the truck is delivering or
        picking up rail freight (0 otherwise).
    """

    type: str = 'Diesel'
    id: int = 0
    train_id: int = 0

    def to_string(self) -> str:
        """Return a label like ``5-Track-2-Diesel``."""
        return f'{self.id}-Track-{self.train_id}-{self.type}'


@dataclass
class rtg:
    """Rubber-tired gantry crane.

    Used at both the rail tracks (``pool='rail_track'``, ``track_id``
    set) and the main container stack (``pool='main_stack'``,
    ``track_id=0``).

    Attributes
    ----------
    type : str
        Powertrain label.
    id : int
        Crane ordinal within the run.
    pool : str
        Either ``'rail_track'`` or ``'main_stack'``; selects the
        physical location.
    track_id : int
        Track index when ``pool == 'rail_track'``; ignored
        otherwise.
    """

    type: str = 'Diesel'
    id: int = 0
    pool: str = ''            # 'rail_track' or 'main_stack'
    track_id: int = 0         # only meaningful for pool == 'rail_track'

    def to_string(self) -> str:
        """Return a pool-specific human-readable label."""
        if self.pool == 'rail_track':
            return f'{self.id}-Track-{self.track_id}-{self.type}'
        return f'{self.id}-{self.pool}-{self.type}'


@dataclass
class sts_crane:
    """Ship-to-shore crane stationed at a berth (vessel ↔ shore lift).

    Attributes
    ----------
    type : str
        Powertrain label.
    id : int
        Crane ordinal within the run.
    berth_id : int
        Berth this crane is assigned to.
    """

    type: str = 'Diesel'
    id: int = 0
    berth_id: int = 0

    def to_string(self) -> str:
        """Return a label like ``1-Berth-3-Diesel``."""
        return f'{self.id}-Berth-{self.berth_id}-{self.type}'


@dataclass
class top_pick:
    """Top-pick handler servicing the main stack as a flexible RTG alternative.

    Attributes
    ----------
    type : str
        Powertrain label.
    id : int
        Top-pick ordinal within the run.
    safety_car_id : int
        Identifier of the safety car this top-pick operates with.
    """

    type: str = 'Diesel'
    id: int = 0
    safety_car_id: int = 0

    def to_string(self) -> str:
        """Return a label like ``2-Safety-7-Diesel``."""
        return f'{self.id}-Safety-{self.safety_car_id}-{self.type}'


@dataclass
class yard_tractor:
    """Yard tractor (a.k.a. hostler) hauling chassis around the terminal.

    Distinct pools (``main_yard_tractors`` water↔stack,
    ``rail_yard_tractors`` rail↔stack) hold ``yard_tractor`` instances
    tagged with their pool.

    Attributes
    ----------
    type : str
        Powertrain label.
    id : int
        Tractor ordinal within the run.
    pool : str
        Tractor pool tag, typically ``'main'`` or ``'rail'``.
    """

    type: str = 'Diesel'
    id: int = 0
    pool: str = ''            # e.g. 'main' or 'rail'

    def to_string(self) -> str:
        """Return a label like ``3-main-Diesel``."""
        return f'{self.id}-{self.pool}-{self.type}'


@dataclass
class chassis:
    """Chassis carrying a container.

    Two pools exist (terminal vs road) for different physical
    equipment, distinguished by the ``pool`` field; the dataclass
    itself is shared because chassis carry no behavior.

    Attributes
    ----------
    type : str
        Chassis class label (e.g. ``'Standard'``).
    id : int
        Chassis ordinal within the run.
    pool : str
        Either ``'terminal'`` or ``'road'``.
    """

    type: str = 'Standard'
    id: int = 0
    pool: str = ''            # 'terminal' or 'road'

    def to_string(self) -> str:
        """Return a label like ``terminal-chassis-4-Standard``."""
        return f'{self.pool}-chassis-{self.id}-{self.type}'


@dataclass
class vessel:
    """Vessel making a berth call.

    Attributes
    ----------
    id : int
        Vessel ordinal within the run.
    name : str
        Display name (e.g. AIS-style vessel name).
    inbound_containers : int
        Count of containers being discharged.
    outbound_containers : int
        Count of containers being loaded.
    """

    id: int = 0
    name: str = ''
    inbound_containers: int = 0
    outbound_containers: int = 0

    def to_string(self) -> str:
        """Return a label like ``Vessel-2-Maersk Edinburgh``."""
        return f'Vessel-{self.id}-{self.name}'
