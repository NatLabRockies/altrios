import numpy as np
import pandas as pd
import json
import yaml
import logging
from typing import List, Optional, Union, Dict, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def smooth_elevation(elevations: List[float], window_size: int = 5) -> List[float]:
    """
    Smooth elevations with a centered rolling mean.

    Args:
        elevations: Elevations in meters.
        window_size: Window length in samples; even values are rounded up to odd.

    Returns:
        Smoothed elevations, same length as the input.
    """
    if not elevations:
        return []

    if len(elevations) < window_size:
        logger.warning(
            f"Elevation list length {len(elevations)} is smaller than window size {window_size}. Returning original list."
        )
        return elevations

    # Ensure window_size is odd to center properly
    if window_size % 2 == 0:
        window_size += 1

    s = pd.Series(elevations)
    # use min_periods=1 to avoid NaNs at edges, center=True to keep phase
    smoothed = s.rolling(window=window_size, center=True, min_periods=1).mean().tolist()

    return smoothed


def is_siding(tags: Dict[str, str]) -> bool:
    """
    Tell whether an OSM railway way is a siding.

    Args:
        tags: OSM tags of the way.

    Returns:
        True for siding, yard, spur or crossover service, or industrial,
        military or test usage.
    """
    service = tags.get("service", "")
    if service in ["siding", "yard", "spur", "crossover"]:
        return True

    usage = tags.get("usage", "")
    if usage in ["industrial", "military", "test"]:
        return True

    return False


def extract_sidings(
    graph_edges: List[Tuple[Any, Any, Dict]],
) -> List[Tuple[Any, Any, Dict]]:
    """
    Keep the graph edges whose tags mark a siding.

    Args:
        graph_edges: (u, v, data) tuples; data carries 'tags'.

    Returns:
        The siding edges, in input order.
    """
    sidings = []
    for u, v, data in graph_edges:
        tags = data.get("tags", {})
        if is_siding(tags):
            sidings.append((u, v, data))
    return sidings


def set_speed_limits(
    links: List[Dict[str, Any]],
    default_speed_mps: float = 26.8,  # ~60 mph
    siding_speed_mps: float = 6.7,  # ~15 mph
    limit_by_grade: bool = False,
    max_grade: float = 0.02,
) -> List[Dict[str, Any]]:
    """
    Give each link one speed limit over its full length.

    Sidings get siding_speed_mps, other links default_speed_mps; dummy link 0 gets None.

    Args:
        links: Link dictionaries in ALTRIOS format.
        default_speed_mps: Main-line speed limit, m/s.
        siding_speed_mps: Siding speed limit, m/s.
        limit_by_grade: Unused.
        max_grade: Unused.

    Returns:
        The same list with 'speed_set' populated.
    """
    for link in links:
        if link.get("idx_curr", -1) == 0:
            # Dummy link 0 must have speed_set = None for ALTRIOS to parse
            link["speed_set"] = None
            continue

        # Determine if link is a siding based on temporary metadata if available
        tags = link.get("osm_tags", {})
        if is_siding(tags):
            current_limit = siding_speed_mps
        else:
            current_limit = default_speed_mps

        # SpeedSet schema must include speed_params (empty list) for ALTRIOS compatibility
        # speed_set: { speed_limits: [...], speed_params: [], is_head_end: bool }

        link_length = link.get("length_meters", 0.0)

        link["speed_set"] = {
            "speed_limits": [
                {
                    "speed_meters_per_second": current_limit,
                    "offset_start_meters": 0.0,
                    "offset_end_meters": link_length,
                }
            ],
            "speed_params": [],
            "is_head_end": False,
        }

    return links


def add_endpoint_buffer(
    geometry_points: List[Tuple[float, float]], train_length_meters: float
) -> List[Tuple[float, float]]:
    """
    Extend a polyline's last segment by the train length.

    The new point continues the final segment's bearing on a spherical earth.

    Args:
        geometry_points: (lon, lat) tuples in degrees.
        train_length_meters: Distance to extend, m.

    Returns:
        The points plus one appended (lon, lat); unchanged if fewer than two points.
    """
    from math import radians, cos, sin, asin, sqrt, atan2, degrees

    if len(geometry_points) < 2:
        return geometry_points

    # Get last two points
    p1 = geometry_points[-2]
    p2 = geometry_points[-1]

    lon1, lat1 = map(radians, p1)
    lon2, lat2 = map(radians, p2)

    # Calculate bearing from p1 to p2
    dLon = lon2 - lon1
    y = sin(dLon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon)
    bearing = atan2(y, x)  # radians

    # Calculate new point given distance and bearing from p2
    # Earth radius
    R = 6371000.0
    d = train_length_meters

    lat3 = asin(sin(lat2) * cos(d / R) + cos(lat2) * sin(d / R) * cos(bearing))
    lon3 = lon2 + atan2(
        sin(bearing) * sin(d / R) * cos(lat2), cos(d / R) - sin(lat2) * sin(lat3)
    )

    new_point_deg = (degrees(lon3), degrees(lat3))

    # Return new list
    return geometry_points + [new_point_deg]


def load_network_file(file_path: Union[str, Path]) -> Any:
    """
    Load a JSON, YAML or CSV network file.

    Args:
        file_path: Path to the file; the suffix selects the parser.

    Returns:
        Parsed JSON or YAML, or a list of row dicts for CSV.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{file_path} not found.")

    ext = path.suffix.lower()

    if ext == ".json":
        with open(path, "r") as f:
            return json.load(f)
    elif ext in [".yaml", ".yml"]:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    elif ext == ".csv":
        # Assumes CSV contains node coordinates or link definitions
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
