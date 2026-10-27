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
    Applies a sliding window smoothing to a list of elevations.
    matches the sliding window logic used in elevation processing.

    Args:
        elevations: List of elevation values in meters.
        window_size: The size of the sliding window. Must be odd.

    Returns:
        List of smoothed elevation values.
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
    Determines if a railway way is a siding based on OSM tags.
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
    Extracts edges that are identified as sidings from a list of graph edges.

    Args:
        graph_edges: List of (u, v, data) tuples where data contains 'tags'.

    Returns:
        List of (u, v, data) tuples for sidings.
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
    Sets the speed limit for a list of links.
    Populates the 'speed_set' field in the link dictionary.

    Args:
        links: List of link dictionaries (ALTRIOS Link format).
        default_speed_mps: Default speed limit in meters per second.
        siding_speed_mps: Speed limit for sidings.
        limit_by_grade: If True, reduces speed limit based on grade.
        max_grade: The grade threshold.

    Returns:
        List of links with updated speed_set.
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
    Adds an end point to the geometry to account for train length.
    Extends the last segment by train_length_meters.

    Args:
        geometry_points: List of (lon, lat) tuples.
        train_length_meters: The length to extend in meters.

    Returns:
        Updated list of geometry points.
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
    Loads network data from various file types (JSON, YAML, CSV).
    Returns parsed data.
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
