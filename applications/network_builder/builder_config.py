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
    output_name: str = "spokane_lethbridge"
    max_grade: float = 0.04
    buffer_deg: float = 0.4
    regions: List[Dict[str, str]] = field(default_factory=lambda: [
        {
            "name": "washington",
            "url": "https://download.geofabrik.de/north-america/us/washington-latest.osm.pbf",
        },
        {
            "name": "idaho",
            "url": "https://download.geofabrik.de/north-america/us/idaho-latest.osm.pbf",
        },
        {
            "name": "british-columbia",
            "url": "https://download.geofabrik.de/north-america/canada/british-columbia-latest.osm.pbf",
        },
        {
            "name": "alberta",
            "url": "https://download.geofabrik.de/north-america/canada/alberta-latest.osm.pbf",
        },
    ])
    waypoints: List[Tuple[str, float, float]] = field(default_factory=lambda: [
        ("Spokane", 47.666, -117.360),
        ("Sandpoint", 48.282, -116.561),
        ("Bonners Ferry", 48.692, -116.319),
        ("Kingsgate (Border)", 49.000, -116.180),
        ("Cranbrook", 49.508, -115.747),
        ("Fernie", 49.504, -115.063),
        ("Crowsnest", 49.636, -114.682),
        ("Fort Macleod", 49.725, -113.397),
        ("Lethbridge", 49.724, -112.919),
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
