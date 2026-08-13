"""
Network builder configuration.

Keeps the corridor parameters separate from the builder code (mirrors
altrios.train_planner.planner_config). Edit / override the BuilderConfig fields
-- see network_builder_demo.py -- to target a different corridor without
touching any of the workflow modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class BuilderConfig:
    """
    Dataclass for network-builder configuration parameters.

    Instantiate this (e.g. in network_builder_demo.py) and override only the
    fields you need for a new corridor -- the workflow modules read these values
    as parameters, so they never need editing or copying.

    Attributes:
    ----------
    - `output_name`: Basename for the built network files (written under the results dir).
    - `max_grade`: Maximum track grade (decimal, e.g. 0.04 = 4%) used to smooth elevation spikes.
    - `buffer_deg`: Buffer in degrees around the waypoint line when clipping the corridor (~0.4 deg ~ 45 km).
    - `regions`: List of `{"name", "url"}` Geofabrik regional PBF entries covering the route (used only when downloading the OSM extract).
    - `waypoints`: Route waypoints `(name, lat, lon)`, ordered start -> end. Defines both the OSM clip area and the mainline route through the network.
    """
    output_name: str = "Toronto_Battle Creek"
    max_grade: float = 0.04
    buffer_deg: float = 0.4
    regions: List[Dict[str, str]] = field(default_factory=lambda: [
        # {
        #     "name": "british-columbia",
        #     "url": "https://download.geofabrik.de/north-america/canada/british-columbia-latest.osm.pbf",
        # },
        # {
        #     "name": "alberta",
        #     "url": "https://download.geofabrik.de/north-america/canada/alberta-latest.osm.pbf",
        # },
        {
            "name": "ontario",
            "url": "https://download.geofabrik.de/north-america/canada/ontario-latest.osm.pbf",
        },
        {
            "name": "michigan",
            "url": "https://download.geofabrik.de/north-america/us/michigan-latest.osm.pbf",
        },

    ])
    waypoints: List[Tuple[str, float, float]] = field(default_factory=lambda: [
        ("MacMillan Yard", 43.780, -79.519),
        ("Burlington West", 43.327, -79.822),
        ("Bayview", 43.2857, -79.8882),
        ("London West", 42.9819, -81.2464),
        ("Port Huron", 42.961, -82.444),
        ("McAllister", 42.340, -85.138),
        ("Battle Creek Yard", 42.357, -85.127),
    ])

    @property
    def start_pt(self) -> Tuple[float, float]:
        """(lat, lon) of the first waypoint -- the route start."""
        return (self.waypoints[0][1], self.waypoints[0][2])

    @property
    def end_pt(self) -> Tuple[float, float]:
        """(lat, lon) of the last waypoint -- the route end."""
        return (self.waypoints[-1][1], self.waypoints[-1][2])

    @property
    def route_waypoints(self) -> List[Tuple[float, float]]:
        """(lat, lon) intermediate waypoints the mainline is routed through."""
        return [(lat, lon) for _name, lat, lon in self.waypoints[1:-1]]
