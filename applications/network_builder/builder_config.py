"""
Network builder configuration.

Keeps the corridor parameters separate from the builder code (mirrors
altrios.train_planner.planner_config). Edit / override the BuilderConfig fields
-- see network_builder_demo.py -- to target a different corridor without
touching any of the workflow modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


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
    - `output_format`: Serialization of the built network: `"msgpack"` (default; compact and
      fast to load) or `"yaml"` (human-readable). Either loads with `altrios.Network.from_file`.
    - `max_grade`: Maximum track grade (decimal, e.g. 0.04 = 4%) used to smooth elevation spikes.
    - `buffer_deg`: Buffer in degrees around the waypoint line when clipping the corridor (~0.4 deg ~ 45 km).
    - `endpoint_buffer_meters`: Length of the buffer links prepended/appended to the
      mainline. ALTRIOS starts a train with its TAIL at offset 0 of the origin link
      (front at one train length), so the buffer must be at least train length PLUS the
      friction-braking distance from the buffer's speed limit (ALTRIOS's braking-point
      sweep needs that headroom on the origin link; ~200 m for a typical freight consist
      from 26.8 m/s). The front-of-train travel surplus over the corridor length is
      (buffer - train length), so use train length + ~300 m for near-exact
      front-at-origin -> front-at-destination corridor semantics. The default 3000 m
      accommodates any train up to ~2.8 km.
    - `include_lat_lons`: Embed per-point lat/lon in each link's `headings` entries in the
      network YAML (`Heading.lat` / `Heading.lon` are optional in the ALTRIOS schema and
      ignored by the simulation; useful for downstream geospatial tools, e.g. the catenary
      optimizer).
    - `mapzen_fallback`: Allow falling back to Mapzen (via OpenTopoData) for points outside
      USGS 3DEP coverage. Set False to force USGS-only elevations; points 3DEP cannot
      resolve are then interpolated along the path instead.
    - `regions`: List of `{"name", "url"}` Geofabrik regional PBF entries covering the route (used only when downloading the OSM extract).
    - `waypoints`: Route waypoints `(name, lat, lon)`, ordered start -> end. Defines both the OSM clip area and the mainline route through the network. The first/last waypoint names label the corridor endpoints in the exported locations CSV.
    - `origin_regions` / `destination_regions`: (grid emissions, electricity price, liquid fuel price) region codes for the locations CSV rows of the first/last waypoint.
    """
    output_name: str = "spokane_lethbridge"
    output_format: str = "msgpack"
    max_grade: float = 0.04
    buffer_deg: float = 0.4
    endpoint_buffer_meters: float = 3000.0
    include_lat_lons: bool = True
    mapzen_fallback: bool = True
    origin_regions: Tuple[str, str, str] = ("WECC", "WA", "WA")
    destination_regions: Tuple[str, str, str] = ("WECC", "ab", "ab")
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

    @classmethod
    def from_file(cls, filepath: str | Path) -> "BuilderConfig":
        """Load a corridor config from a YAML file (matching the .yaml format
        of the other ALTRIOS inputs; JSON also parses, being a YAML subset).

        The file holds any subset of the BuilderConfig fields (missing fields
        keep their defaults); unknown keys are an error. ``waypoints`` entries
        are ``[name, lat, lon]`` lists. Example:

            output_name: winnipeg_chicago
            waypoints:
              - [Winnipeg, 49.899, -97.139]
              - [Chicago, 41.879, -87.638]
        """
        data = yaml.safe_load(Path(filepath).read_text())
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(
                f"Unknown BuilderConfig key(s) in {filepath}: {sorted(unknown)}; "
                f"valid keys: {sorted(valid)}"
            )
        if "waypoints" in data:
            data["waypoints"] = [tuple(w) for w in data["waypoints"]]
        for key in ("origin_regions", "destination_regions"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

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
