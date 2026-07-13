"""
Build an ALTRIOS network YAML from local OSM XML, including siding branches.

Builds an ALTRIOS network YAML from a local OSM XML extract:
  1. Finding the mainline shortest-path as before.
  2. Detecting siding ways (service=siding, etc.) that share nodes with the mainline.
  3. Emitting siding links with proper idx_next_alt / idx_prev_alt topology
     so the ALTRIOS dispatcher can route trains through sidings for meet/pass.

Usage:
    python build_altrios_with_sidings.py [--fast]
"""

import xml.etree.ElementTree as ET
import networkx as nx
import math
import logging
import yaml
import json
import os
import sys
import numpy as np
from scipy.signal import savgol_filter
from scipy import interpolate

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
import urllib3
import time

try:
    from builder_config import BuilderConfig
    from altrios_builder_utils import (
        set_speed_limits,
        is_siding,
        add_endpoint_buffer,
    )
except ImportError:
    from .builder_config import BuilderConfig
    from .altrios_builder_utils import (
        set_speed_limits,
        is_siding,
        add_endpoint_buffer,
    )

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Point REQUESTS_CA_BUNDLE at a CA bundle (e.g. an exported corporate root) to
# verify TLS on elevation requests; unset, verification is skipped as for the
# other requests in this module (corporate-proxy environments).
SSL_VERIFY = os.environ.get("REQUESTS_CA_BUNDLE") or False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

R_EARTH = 6371000  # metres

# Maximum allowed grade (as decimal, e.g., 0.04 = 4%)
MAX_GRADE = 0.04


# ── helpers ──────────────────────────────────────────────────────────────────


def smooth_elevation_savgol(
    elevations, distances, window_length=200, order=3, segment_length=50
):
    """
    Apply Savitzky-Golay filter to smooth elevations, similar to network_builder.py.
    This helps eliminate grade spikes from noisy elevation data.

    Args:
        elevations: List of elevation values in meters
        distances: Cumulative distances along the path (for proper interpolation)
        window_length: Window size for the Savgol filter
        order: Polynomial order for the filter
        segment_length: Resample interval in meters

    Returns:
        Smoothed elevation values
    """
    if len(elevations) < 5:
        return elevations

    elevations = np.array(elevations)
    distances = np.array(distances)

    # Resample to evenly spaced points for the filter
    total_dist = distances[-1]
    n_segments = max(int(total_dist / segment_length), len(elevations))
    interp_distances = np.linspace(0, total_dist, n_segments)

    # Interpolate elevations to even spacing
    f = interpolate.interp1d(
        distances, elevations, kind="linear", fill_value="extrapolate"
    )
    interp_elevs = f(interp_distances)

    # Apply Savitzky-Golay filter
    effective_window = min(len(interp_elevs), window_length)
    if effective_window % 2 == 0:
        effective_window -= 1
    effective_window = max(effective_window, order + 2)

    if len(interp_elevs) > effective_window:
        smoothed_interp = savgol_filter(
            interp_elevs,
            effective_window,
            min(order, effective_window - 1),
            mode="nearest",
        )
    else:
        smoothed_interp = interp_elevs

    # Interpolate back to original distances
    f2 = interpolate.interp1d(
        interp_distances, smoothed_interp, kind="linear", fill_value="extrapolate"
    )
    smoothed = f2(distances)

    # Preserve original endpoints
    smoothed[0] = elevations[0]
    smoothed[-1] = elevations[-1]

    return list(smoothed)


def filter_grade_spikes(elevations, distances, max_grade=MAX_GRADE, edge_buffer_m=500):
    """
    Filter out grade spikes that exceed the maximum allowable grade.
    Uses the approach from network_builder.py: exclude points near edges
    and cap grades that exceed the threshold.

    Args:
        elevations: List of elevation values in meters
        distances: Cumulative distances along the path
        max_grade: Maximum allowed grade as decimal (0.04 = 4%)
        edge_buffer_m: Buffer distance from edges to exclude (meters)

    Returns:
        Filtered elevation values with grade spikes removed
    """
    if len(elevations) < 3:
        return elevations

    elevations = np.array(elevations, dtype=float)
    distances = np.array(distances, dtype=float)
    total_length = distances[-1]

    # Calculate grades between consecutive points
    for i in range(1, len(elevations)):
        dist_diff = distances[i] - distances[i - 1]
        if dist_diff < 1.0:
            continue

        elev_diff = elevations[i] - elevations[i - 1]
        grade = abs(elev_diff / dist_diff)

        # Skip edge points (within buffer of start/end)
        if (
            distances[i] < edge_buffer_m
            or (total_length - distances[i]) < edge_buffer_m
        ):
            continue

        # If grade exceeds max, interpolate to cap it
        if grade > max_grade:
            # Cap the elevation change to max_grade
            max_elev_change = max_grade * dist_diff
            if elev_diff > 0:
                elevations[i] = elevations[i - 1] + max_elev_change
            else:
                elevations[i] = elevations[i - 1] - max_elev_change

    return list(elevations)


def fetch_elevations_usgs_epqs(locations, max_workers=8):
    """Fetch elevations for [(lat, lon), …] from the USGS 3DEP Elevation Point
    Query Service (EPQS), one request per point.

    Returns a list the same length as ``locations`` with ``None`` wherever
    EPQS has no coverage (it reports a large negative sentinel outside the
    3DEP footprint).
    """
    from concurrent.futures import ThreadPoolExecutor

    session = requests.Session()

    def _one(pt):
        lat, lon = pt
        for attempt in range(4):
            try:
                r = session.get(
                    "https://epqs.nationalmap.gov/v1/json",
                    params={"x": lon, "y": lat, "wkid": 4326, "units": "Meters"},
                    verify=SSL_VERIFY,
                    timeout=30,
                )
                if r.status_code == 200:
                    try:
                        v = float(r.json().get("value"))
                    except (TypeError, ValueError):
                        return None
                    # Sentinel / garbage outside 3DEP coverage
                    if v < -11000 or v > 9000:
                        return None
                    return v
            except Exception as e:
                logger.debug(f"EPQS request failed ({lat}, {lon}): {e}")
            time.sleep(1.5 * (attempt + 1))
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_one, locations))


def fetch_elevations_3dep_bulk(locations, mapzen_fallback=True):
    """Fetch elevations for [(lat, lon), …].

    US points (lat < 49.001) are fetched in bulk from OpenTopoData's ned10m
    (USGS NED 10 m). Points at/above the border are fetched directly from the
    USGS 3DEP EPQS service, whose coverage extends past the border and shares
    the US-side vertical datum — this avoids a dataset seam (and artificial
    grade step) at border crossings. Mapzen via OpenTopoData is the fallback
    for points beyond 3DEP coverage; pass ``mapzen_fallback=False`` for
    USGS-only elevations (uncovered points are interpolated along the path).
    """
    elevations = [None] * len(locations)

    us_indices = [i for i, (lat, _) in enumerate(locations) if lat < 49.001]
    ca_indices = [i for i, (lat, _) in enumerate(locations) if lat >= 49.001]

    logger.info(
        f"Elevation: {len(us_indices)} US (ned10m), "
        f"{len(ca_indices)} CA (USGS 3DEP EPQS, mapzen fallback)"
    )

    def _fetch(indices, dataset="mapzen"):
        if not indices:
            return
        chunk = 80
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 Chrome/91.0.4472.114 Safari/537.36"
            )
        }
        for i in range(0, len(indices), chunk):
            ci = indices[i : i + chunk]
            loc_str = "|".join(f"{locations[j][0]},{locations[j][1]}" for j in ci)
            url = f"https://api.opentopodata.org/v1/{dataset}?locations={loc_str}"
            for attempt in range(5):
                try:
                    r = requests.get(url, headers=headers, verify=False, timeout=30)
                    if r.status_code == 200:
                        for k, res in enumerate(r.json().get("results", [])):
                            elev = res.get("elevation")
                            if elev is not None and k < len(ci):
                                elevations[ci[k]] = elev
                        break
                    elif r.status_code == 429:
                        time.sleep(5 * (attempt + 1))
                    else:
                        logger.warning(f"OpenTopoData {r.status_code}: {r.text}")
                        time.sleep(2)
                except Exception as e:
                    logger.warning(f"Request failed: {e}")
                    time.sleep(2)
            time.sleep(1.5)

    _fetch(us_indices, "ned10m")

    if ca_indices:
        epqs = fetch_elevations_usgs_epqs([locations[i] for i in ca_indices])
        uncovered = []
        for k, i in enumerate(ca_indices):
            if epqs[k] is not None:
                elevations[i] = epqs[k]
            else:
                uncovered.append(i)
        if uncovered:
            if mapzen_fallback:
                logger.info(
                    f"USGS 3DEP has no coverage for {len(uncovered)} points; "
                    "falling back to mapzen"
                )
                _fetch(uncovered, "mapzen")
            else:
                logger.info(
                    f"USGS 3DEP has no coverage for {len(uncovered)} points; "
                    "mapzen fallback disabled, will interpolate along the path"
                )

    # Fill any still-missing points by interpolating along the point sequence
    # (a global-average fill would paint long flat plateaus with steps at
    # each end wherever a whole region's requests failed).
    missing = [i for i, e in enumerate(elevations) if e is None]
    if missing:
        logger.warning(
            f"{len(missing)} points have no elevation from any source; "
            "interpolating along the path"
        )
        valid_idx = [i for i, e in enumerate(elevations) if e is not None]
        if valid_idx:
            filled = np.interp(
                missing, valid_idx, [elevations[i] for i in valid_idx]
            )
            for i, v in zip(missing, filled):
                elevations[i] = float(v)
        else:
            elevations = [0.0] * len(elevations)
    return elevations


def haversine_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * math.asin(math.sqrt(a)) * R_EARTH


def get_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dphi = math.log(math.tan(lat2 / 2 + math.pi / 4) / math.tan(lat1 / 2 + math.pi / 4))
    if abs(dlon) > math.pi:
        dlon = -(2 * math.pi - dlon) if dlon > 0 else 2 * math.pi + dlon
    return (math.degrees(math.atan2(dlon, dphi)) + 360) % 360


# ── OSM parse ────────────────────────────────────────────────────────────────


def parse_osm_xml_to_graph(xml_file):
    """Parse OSM XML → NetworkX graph.  Preserves *all* rail edges so sidings
    are in the graph even when they are not on the mainline shortest path."""
    logger.info(f"Parsing {xml_file}…")
    tree = ET.parse(xml_file)
    root = tree.getroot()

    G = nx.Graph()
    nodes = {}

    for node in root.findall("node"):
        nid = node.get("id")
        lat = float(node.get("lat"))
        lon = float(node.get("lon"))
        nodes[nid] = {"lat": lat, "lon": lon}
        G.add_node(nid, lat=lat, lon=lon)

    for way in root.findall("way"):
        way_id = way.get("id")
        nd_refs = [nd.get("ref") for nd in way.findall("nd")]
        tags = {t.get("k"): t.get("v") for t in way.findall("tag")}
        if len(nd_refs) < 2:
            continue
        for i in range(len(nd_refs) - 1):
            u, v = nd_refs[i], nd_refs[i + 1]
            if u in nodes and v in nodes:
                d = haversine_distance(
                    nodes[u]["lon"],
                    nodes[u]["lat"],
                    nodes[v]["lon"],
                    nodes[v]["lat"],
                )
                # Keep first edge only (simple graph)
                if not G.has_edge(u, v):
                    G.add_edge(u, v, weight=d, osm_way_id=way_id, tags=tags)

    logger.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def find_nearest_node(G, lat, lon):
    best_d, best_n = float("inf"), None
    for n, d in G.nodes(data=True):
        dd = haversine_distance(lon, lat, d["lon"], d["lat"])
        if dd < best_d:
            best_d, best_n = dd, n
    return best_n, best_d


# ── siding detection ─────────────────────────────────────────────────────────


def _identify_sidings(G, mainline_nodes):
    """Return a list of siding *segments* that branch off the mainline.

    Each siding is a list of OSM node IDs forming a contiguous path whose
    first and last nodes are on the mainline (a passing siding) **or** whose
    first node is on the mainline and the path dead-ends.

    We only include sidings where the branch node is on the mainline, so
    that we can wire alternate link indices.
    """

    mainline_set = set(mainline_nodes)

    # Collect all edges tagged as sidings
    siding_edges = []
    for u, v, data in G.edges(data=True):
        if is_siding(data.get("tags", {})):
            siding_edges.append((u, v, data))

    if not siding_edges:
        logger.warning("No siding edges found in OSM graph.")
        return []

    # Build a sub-graph of siding edges
    S = nx.Graph()
    for u, v, data in siding_edges:
        S.add_edge(u, v, **data)

    logger.info(
        f"Siding sub-graph: {S.number_of_nodes()} nodes, {S.number_of_edges()} edges"
    )

    # Find connected components in the siding graph
    sidings = []
    for comp_nodes in nx.connected_components(S):
        # Which nodes in this component touch the mainline?
        junction_nodes = comp_nodes & mainline_set
        if not junction_nodes:
            continue  # siding not connected to mainline – skip

        # For a passing siding we expect exactly 2 junction nodes.
        # For a stub siding we expect 1.
        # If there are >2, just take the pair that is farthest apart on the mainline
        # (or degenerate; still usable).

        # Order junction nodes by their position along the mainline
        junction_list = sorted(
            junction_nodes,
            key=lambda n: mainline_nodes.index(n) if n in mainline_nodes else 999999,
        )

        # Build a path through the siding between the first and last junction node
        sub = S.subgraph(comp_nodes).copy()

        if len(junction_list) >= 2:
            start_j, end_j = junction_list[0], junction_list[-1]
            try:
                siding_path = nx.shortest_path(
                    sub, source=start_j, target=end_j, weight="weight"
                )
            except nx.NetworkXNoPath:
                # Fall back: treat as stub from the first junction node
                siding_path = list(nx.dfs_preorder_nodes(sub, source=junction_list[0]))
        else:
            # Stub siding – just DFS from the single junction node
            siding_path = list(nx.dfs_preorder_nodes(sub, source=junction_list[0]))

        # Only keep if the siding has at least 2 nodes (one edge)
        if len(siding_path) >= 2:
            sidings.append(siding_path)

    logger.info(f"Detected {len(sidings)} siding segments touching the mainline.")
    return sidings


# ── ALTRIOS export ───────────────────────────────────────────────────────────


def export_path_with_sidings(
    G,
    mainline_nodes,
    sidings,
    output_prefix,
    fast_mode=False,
    max_grade=MAX_GRADE,
    mapzen_fallback=True,
):
    """Build an ALTRIOS network YAML that includes the mainline plus
    siding branches with proper idx_next_alt / idx_prev_alt wiring."""

    # ────── Elevations ──────
    # Collect unique nodes we need elevations for
    all_nodes_ordered = list(mainline_nodes)
    siding_extra_nodes = []
    for sp in sidings:
        for n in sp:
            if n not in set(all_nodes_ordered):
                all_nodes_ordered.append(n)
                siding_extra_nodes.append(n)

    node_to_idx = {n: i for i, n in enumerate(all_nodes_ordered)}
    coords = [(G.nodes[n]["lat"], G.nodes[n]["lon"]) for n in all_nodes_ordered]

    # Calculate cumulative distances along the mainline for proper smoothing
    mainline_distances = [0.0]
    for i in range(1, len(mainline_nodes)):
        u, v = mainline_nodes[i - 1], mainline_nodes[i]
        d = haversine_distance(
            G.nodes[u]["lon"], G.nodes[u]["lat"], G.nodes[v]["lon"], G.nodes[v]["lat"]
        )
        mainline_distances.append(mainline_distances[-1] + d)

    if fast_mode:
        elevations = [0.0] * len(coords)
    else:
        elevations = fetch_elevations_3dep_bulk(
            coords, mapzen_fallback=mapzen_fallback
        )

    # Apply Savitzky-Golay smoothing to mainline elevations (more sophisticated than rolling mean)
    mainline_elevs = elevations[: len(mainline_nodes)]
    if len(mainline_elevs) > 5 and not fast_mode:
        logger.info("Applying Savitzky-Golay smoothing to elevations...")
        mainline_elevs = smooth_elevation_savgol(
            mainline_elevs, mainline_distances, window_length=200
        )

        # Filter out grade spikes that exceed max allowable grade
        logger.info(f"Filtering grade spikes (max grade: {max_grade * 100:.1f}%)...")
        mainline_elevs = filter_grade_spikes(
            mainline_elevs, mainline_distances, max_grade=max_grade
        )

        # Update the elevations list with smoothed mainline values
        for i in range(len(mainline_nodes)):
            elevations[i] = mainline_elevs[i]

    # Convert all elevations to native Python floats so yaml.dump writes plain numbers,
    # not numpy binary blobs (which ALTRIOS cannot parse).
    elevations = [float(e) for e in elevations]

    def elev_of(node):
        return elevations[node_to_idx[node]]

    # ────── Link index management ──────
    # We assign link indices sequentially.  Every OSM edge produces two links
    # (fwd and rev).  idx 0 is the dummy sentinel.
    next_idx = [1]  # mutable counter (use list so inner funcs can mutate)

    def alloc_pair():
        fwd = next_idx[0]
        rev = next_idx[0] + 1
        next_idx[0] += 2
        return fwd, rev

    links = []  # list of link dicts
    geometry = []  # for JSON

    # Dummy link 0 - speed_set must be None (null/~) for ALTRIOS to parse correctly
    links.append(
        {
            "idx_curr": 0,
            "idx_flip": 0,
            "idx_next": 0,
            "idx_next_alt": 0,
            "idx_prev": 0,
            "idx_prev_alt": 0,
            "osm_id": 0,
            "length_meters": 0.0,
            "elevs": [],
            "headings": [],
            "speed_set": None,
            "cat_power_limits": [],
            "link_idxs_lockout": [],
        }
    )

    # ────── Helper to make a link pair for an edge u→v ──────
    def make_link_pair(u, v, osm_tags, fwd_idx, rev_idx):
        u_d, v_d = G.nodes[u], G.nodes[v]
        length = max(
            haversine_distance(u_d["lon"], u_d["lat"], v_d["lon"], v_d["lat"]), 1.0
        )
        heading_rad = math.radians(
            get_bearing(u_d["lat"], u_d["lon"], v_d["lat"], v_d["lon"])
        )
        u_e, v_e = elev_of(u), elev_of(v)

        fwd = {
            "idx_curr": fwd_idx,
            "idx_flip": rev_idx,
            "osm_tags": osm_tags,
            "length_meters": length,
            "osm_id": G[u][v].get("osm_way_id", "siding"),
            "idx_next": 0,
            "idx_next_alt": 0,
            "idx_prev": 0,
            "idx_prev_alt": 0,
            "elevs": [
                {"offset_meters": 0.0, "elev_meters": u_e},
                {"offset_meters": length, "elev_meters": v_e},
            ],
            "headings": [
                {"offset_meters": 0.0, "heading_radians": heading_rad},
                {"offset_meters": length, "heading_radians": heading_rad},
            ],
            "speed_set": [],
            "cat_power_limits": [],
            "link_idxs_lockout": [],
        }
        rev_heading = (heading_rad + math.pi) % (2 * math.pi)
        rev = {
            "idx_curr": rev_idx,
            "idx_flip": fwd_idx,
            "osm_tags": osm_tags,
            "length_meters": length,
            "osm_id": G[u][v].get("osm_way_id", "siding"),
            "idx_next": 0,
            "idx_next_alt": 0,
            "idx_prev": 0,
            "idx_prev_alt": 0,
            "elevs": [
                {"offset_meters": 0.0, "elev_meters": v_e},
                {"offset_meters": length, "elev_meters": u_e},
            ],
            "headings": [
                {"offset_meters": 0.0, "heading_radians": rev_heading},
                {"offset_meters": length, "heading_radians": rev_heading},
            ],
            "speed_set": [],
            "cat_power_limits": [],
            "link_idxs_lockout": [],
        }
        geometry.append(
            {
                "link_idx": fwd_idx,
                "coordinates": [[u_d["lon"], u_d["lat"]], [v_d["lon"], v_d["lat"]]],
            }
        )
        geometry.append(
            {
                "link_idx": rev_idx,
                "coordinates": [[v_d["lon"], v_d["lat"]], [u_d["lon"], u_d["lat"]]],
            }
        )
        return fwd, rev

    # ────── Build mainline links ──────
    # We need to know fwd/rev indices before creating links so we can wire
    # connectivity later.  Pre-allocate indices for the start buffer,
    # each mainline segment, and the end buffer.

    buf_fwd, buf_rev = alloc_pair()  # start buffer: 1, 2
    n_main_edges = len(mainline_nodes) - 1

    mainline_indices = []  # [(fwd_idx, rev_idx), …]
    for _ in range(n_main_edges):
        mainline_indices.append(alloc_pair())

    # Pre-allocate siding link indices too
    # siding_indices[s] = [(fwd, rev), …] for each edge in siding s
    siding_indices = []
    for sp in sidings:
        seg = []
        for _ in range(len(sp) - 1):
            seg.append(alloc_pair())
        siding_indices.append(seg)

    # End buffer
    endbuf_fwd, endbuf_rev = alloc_pair()

    # ── Start buffer ──
    start_elev = elev_of(mainline_nodes[0])
    sbl = 3000.0
    links.append(
        {
            "idx_curr": buf_fwd,
            "idx_flip": buf_rev,
            "osm_tags": {"note": "start_buffer"},
            "length_meters": sbl,
            "osm_id": "start_buffer_fwd",
            "idx_next": mainline_indices[0][0],
            "idx_next_alt": 0,
            "idx_prev": 0,
            "idx_prev_alt": 0,
            "elevs": [
                {"offset_meters": 0.0, "elev_meters": start_elev},
                {"offset_meters": sbl, "elev_meters": start_elev},
            ],
            "headings": [
                {"offset_meters": 0.0, "heading_radians": 0.0},
                {"offset_meters": sbl, "heading_radians": 0.0},
            ],
            "speed_set": [],
            "cat_power_limits": [],
            "link_idxs_lockout": [],
        }
    )
    links.append(
        {
            "idx_curr": buf_rev,
            "idx_flip": buf_fwd,
            "osm_tags": {"note": "start_buffer"},
            "length_meters": sbl,
            "osm_id": "start_buffer_rev",
            "idx_next": 0,
            "idx_next_alt": 0,
            "idx_prev": mainline_indices[0][1],
            "idx_prev_alt": 0,
            "elevs": [
                {"offset_meters": 0.0, "elev_meters": start_elev},
                {"offset_meters": sbl, "elev_meters": start_elev},
            ],
            "headings": [
                {"offset_meters": 0.0, "heading_radians": 0.0},
                {"offset_meters": sbl, "heading_radians": 0.0},
            ],
            "speed_set": [],
            "cat_power_limits": [],
            "link_idxs_lockout": [],
        }
    )

    # Start buffer geometry
    u0, u1 = mainline_nodes[0], mainline_nodes[1]
    p0 = (G.nodes[u0]["lon"], G.nodes[u0]["lat"])
    p1 = (G.nodes[u1]["lon"], G.nodes[u1]["lat"])
    p_buf = add_endpoint_buffer([p1, p0], sbl)[-1]
    geometry.append(
        {"link_idx": buf_fwd, "coordinates": [[p_buf[0], p_buf[1]], [p0[0], p0[1]]]}
    )

    # ── Mainline links ──
    for i in range(n_main_edges):
        u = mainline_nodes[i]
        v = mainline_nodes[i + 1]
        fwd_idx, rev_idx = mainline_indices[i]
        edge_data = G.get_edge_data(u, v) or {}
        tags = edge_data.get("tags", {})

        fwd_link, rev_link = make_link_pair(u, v, tags, fwd_idx, rev_idx)

        # Chain connectivity along mainline
        fwd_link["idx_prev"] = mainline_indices[i - 1][0] if i > 0 else buf_fwd
        fwd_link["idx_next"] = (
            mainline_indices[i + 1][0] if i < n_main_edges - 1 else endbuf_fwd
        )

        rev_link["idx_prev"] = (
            mainline_indices[i + 1][1] if i < n_main_edges - 1 else endbuf_rev
        )
        rev_link["idx_next"] = mainline_indices[i - 1][1] if i > 0 else buf_rev

        links.append(fwd_link)
        links.append(rev_link)

    # ── Siding links ──
    # Map: mainline node → index along the mainline (for quick lookup)
    mainline_pos = {n: i for i, n in enumerate(mainline_nodes)}

    for s_idx, sp in enumerate(sidings):
        sindices = siding_indices[s_idx]

        for j in range(len(sp) - 1):
            u, v = sp[j], sp[j + 1]
            fwd_idx, rev_idx = sindices[j]

            if G.has_edge(u, v):
                edge_data = G.get_edge_data(u, v) or {}
            else:
                # Nodes exist but direct edge may not – synthesize
                edge_data = {"tags": {"service": "siding"}}
                d = haversine_distance(
                    G.nodes[u]["lon"],
                    G.nodes[u]["lat"],
                    G.nodes[v]["lon"],
                    G.nodes[v]["lat"],
                )
                G.add_edge(
                    u,
                    v,
                    weight=d,
                    osm_way_id="siding_synth",
                    tags={"service": "siding"},
                )

            tags = edge_data.get("tags", {})
            # Make sure tags mark this as siding (for speed limit)
            if "service" not in tags:
                tags["service"] = "siding"

            fwd_link, rev_link = make_link_pair(u, v, tags, fwd_idx, rev_idx)

            # Internal siding chaining
            fwd_link["idx_prev"] = sindices[j - 1][0] if j > 0 else 0
            fwd_link["idx_next"] = sindices[j + 1][0] if j < len(sp) - 2 else 0

            rev_link["idx_prev"] = sindices[j + 1][1] if j < len(sp) - 2 else 0
            rev_link["idx_next"] = sindices[j - 1][1] if j > 0 else 0

            links.append(fwd_link)
            links.append(rev_link)

    # ── End buffer ──
    end_elev = elev_of(mainline_nodes[-1])
    ebl = 3000.0
    last_main_fwd = mainline_indices[-1][0]
    last_main_rev = mainline_indices[-1][1]

    links.append(
        {
            "idx_curr": endbuf_fwd,
            "idx_flip": endbuf_rev,
            "osm_tags": {"note": "end_buffer"},
            "length_meters": ebl,
            "osm_id": "end_buffer_fwd",
            "idx_next": 0,
            "idx_next_alt": 0,
            "idx_prev": last_main_fwd,
            "idx_prev_alt": 0,
            "elevs": [
                {"offset_meters": 0.0, "elev_meters": end_elev},
                {"offset_meters": ebl, "elev_meters": end_elev},
            ],
            "headings": [
                {"offset_meters": 0.0, "heading_radians": 0.0},
                {"offset_meters": ebl, "heading_radians": 0.0},
            ],
            "speed_set": [],
            "cat_power_limits": [],
            "link_idxs_lockout": [],
        }
    )
    links.append(
        {
            "idx_curr": endbuf_rev,
            "idx_flip": endbuf_fwd,
            "osm_tags": {"note": "end_buffer"},
            "length_meters": ebl,
            "osm_id": "end_buffer_rev",
            "idx_next": last_main_rev,
            "idx_next_alt": 0,
            "idx_prev": 0,
            "idx_prev_alt": 0,
            "elevs": [
                {"offset_meters": 0.0, "elev_meters": end_elev},
                {"offset_meters": ebl, "elev_meters": end_elev},
            ],
            "headings": [
                {"offset_meters": 0.0, "heading_radians": 0.0},
                {"offset_meters": ebl, "heading_radians": 0.0},
            ],
            "speed_set": [],
            "cat_power_limits": [],
            "link_idxs_lockout": [],
        }
    )

    # End buffer geometry
    last_u = mainline_nodes[-2]
    last_v = mainline_nodes[-1]
    p_b = (G.nodes[last_u]["lon"], G.nodes[last_u]["lat"])
    p_l = (G.nodes[last_v]["lon"], G.nodes[last_v]["lat"])
    p_end = add_endpoint_buffer([p_b, p_l], ebl)[-1]
    geometry.append(
        {
            "link_idx": endbuf_fwd,
            "coordinates": [[p_l[0], p_l[1]], [p_end[0], p_end[1]]],
        }
    )

    # ─── Wire siding branches to mainline via idx_next_alt / idx_prev_alt ───
    #
    # Convention (must be consistent for ALTRIOS validation):
    #   If  A.idx_next_alt = B  →  B.idx_prev == A  or  B.idx_prev_alt == A
    #   If  A.idx_prev_alt = B  →  B.idx_next == A  or  B.idx_next_alt == A
    #
    # For a passing siding  sp[0] … sp[-1]  with mainline positions ml_i, ml_j:
    #
    #   Forward trains (Spokane→Lethbridge direction):
    #     ENTER siding at sp[0]:  arriving_fwd  (mainline before sp[0])  branches alt → siding_fwd_first
    #     EXIT  siding at sp[-1]: siding_fwd_last.idx_next → departing_fwd (mainline after sp[-1])
    #                             departing_fwd.idx_prev_alt = siding_fwd_last
    #
    #   Reverse trains (Lethbridge→Spokane direction, reverse links go the other way):
    #     ENTER siding at sp[-1]: arriving_rev_at_last (mainline rev arriving at sp[-1]) branches alt → siding_rev_last
    #     EXIT  siding at sp[0]:  siding_rev_first.idx_next → departing_rev_at_first (mainline rev leaving sp[0])
    #                             departing_rev_at_first.idx_prev_alt = siding_rev_first
    #
    link_by_idx = {lnk["idx_curr"]: lnk for lnk in links}

    for s_idx, sp in enumerate(sidings):
        sindices = siding_indices[s_idx]
        if not sindices:
            continue

        first_siding_node = sp[0]
        last_siding_node = sp[-1]
        siding_fwd_first = sindices[0][0]
        siding_rev_first = sindices[0][1]
        siding_fwd_last = sindices[-1][0]
        siding_rev_last = sindices[-1][1]

        # ── sp[0]: forward ENTRY + reverse EXIT ──────────────────────────────
        if first_siding_node in mainline_pos:
            ml_i = mainline_pos[first_siding_node]

            # Forward entry: mainline fwd link arriving at sp[0] branches alt to siding
            if ml_i > 0:
                arriving_fwd = mainline_indices[ml_i - 1][0]
                lnk = link_by_idx[arriving_fwd]
                if lnk["idx_next_alt"] == 0:
                    lnk["idx_next_alt"] = siding_fwd_first
                sflink = link_by_idx[siding_fwd_first]
                # siding_fwd_first.idx_prev is initialised to 0 in the siding loop
                if sflink["idx_prev"] == 0:
                    sflink["idx_prev"] = arriving_fwd

            # Reverse exit: siding_rev_first (sp[1]→sp[0]) exits to mainline rev
            # mainline_indices[ml_i-1][1] is the rev link from sp[0] → sp[0]-1
            if ml_i > 0:
                departing_rev = mainline_indices[ml_i - 1][1]
                sfrev = link_by_idx[siding_rev_first]
                # siding_rev_first.idx_next is initialised to 0 in the siding loop
                if sfrev["idx_next"] == 0:
                    sfrev["idx_next"] = departing_rev
                ml_lnk = link_by_idx[departing_rev]
                if ml_lnk["idx_prev_alt"] == 0:
                    ml_lnk["idx_prev_alt"] = siding_rev_first

        # ── sp[-1]: forward EXIT + reverse ENTRY ─────────────────────────────
        if last_siding_node in mainline_pos:
            ml_j = mainline_pos[last_siding_node]

            # Forward exit: siding_fwd_last (sp[-2]→sp[-1]) continues to mainline fwd
            # mainline_indices[ml_j][0] is the fwd link departing sp[-1]
            if ml_j < n_main_edges:
                departing_fwd = mainline_indices[ml_j][0]
                sfl = link_by_idx[siding_fwd_last]
                # siding_fwd_last.idx_next is initialised to 0 in the siding loop
                if sfl["idx_next"] == 0:
                    sfl["idx_next"] = departing_fwd
                ml_lnk = link_by_idx[departing_fwd]
                if ml_lnk["idx_prev_alt"] == 0:
                    ml_lnk["idx_prev_alt"] = siding_fwd_last

            # Reverse entry: mainline rev link arriving at sp[-1] branches alt to siding_rev_last
            # mainline_indices[ml_j][1] is the rev link from sp[-1]+1 → sp[-1]
            if ml_j < n_main_edges:
                arriving_rev = mainline_indices[ml_j][1]
                lnk = link_by_idx[arriving_rev]
                if lnk["idx_next_alt"] == 0:
                    lnk["idx_next_alt"] = siding_rev_last
                srl = link_by_idx[siding_rev_last]
                # siding_rev_last.idx_prev is initialised to 0 in the siding loop
                if srl["idx_prev"] == 0:
                    srl["idx_prev"] = arriving_rev

    # ── Speed limits ──
    logger.info("Setting speed limits (mainline=26.8 m/s, siding=6.7 m/s)…")
    set_speed_limits(links, default_speed_mps=26.8, siding_speed_mps=6.7)

    # Remove temp osm_tags
    for l in links:
        l.pop("osm_tags", None)

    # ── Write YAML ──
    network_config = [
        {
            "max_grade": 1000.0,
            "max_curv_radians_per_meter": 1.0,
            "max_heading_step_radians": 3.5,
            "max_elev_step_meters": 100.0,
        },
        links,
    ]

    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)
    yaml_path = output_prefix + ".yaml"
    geo_path = output_prefix + "_geometry.json"
    loc_path = output_prefix + "_locations.csv"

    logger.info(f"Saving {yaml_path}  ({len(links)} links)")
    with open(yaml_path, "w") as f:
        yaml.dump(network_config, f)

    logger.info(f"Saving {geo_path}")
    with open(geo_path, "w") as f:
        json.dump(geometry, f)

    logger.info(f"Saving {loc_path}")
    with open(loc_path, "w") as f:
        f.write(
            "Location ID,Link Index,Offset (m),Is Front End,"
            "Grid Emissions Region,Electricity Price Region,"
            "Liquid Fuel Price Region\n"
        )
        f.write(f"Spokane,{buf_fwd},0,FALSE,WECC,WA,WA\n")
        f.write(f"Lethbridge,{last_main_fwd},0,FALSE,WECC,ab,ab\n")

    # Summary
    n_siding_links = sum(len(si) * 2 for si in siding_indices)
    logger.info(
        f"Done. Mainline links: {n_main_edges * 2}, "
        f"Siding links: {n_siding_links}, "
        f"Buffer links: 4, Total: {len(links)}"
    )


# ── build entry point ────────────────────────────────────────────────────────


def build_network(
    input_xml,
    output_prefix,
    start_pt,
    end_pt,
    waypoints,
    fast=False,
    max_grade=MAX_GRADE,
    mapzen_fallback=True,
):
    """Build an ALTRIOS network (mainline + sidings) from a local OSM XML extract.

    Parses ``input_xml`` into a rail graph, finds the shortest mainline path from
    ``start_pt`` through ``waypoints`` to ``end_pt``, detects sidings, and writes
    ``{output_prefix}.yaml``, ``{output_prefix}_geometry.json`` and
    ``{output_prefix}_locations.csv``. Set ``fast=True`` to skip the elevation
    fetch (no network access required). ``max_grade`` caps the track grade
    (decimal, e.g. 0.04 = 4%) used to smooth elevation spikes.

    Returns the path to the written ``.yaml`` file.
    """
    G = parse_osm_xml_to_graph(input_xml)

    logger.info("Finding nearest graph nodes…")
    start_node, d0 = find_nearest_node(G, *start_pt)
    end_node, d1 = find_nearest_node(G, *end_pt)
    logger.info(f"Start: {start_node} ({d0:.0f} m), End: {end_node} ({d1:.0f} m)")

    # Compute mainline path through waypoints
    current = start_node
    full_path = [current]
    for pt in list(waypoints) + [end_pt]:
        target, _ = find_nearest_node(G, pt[0], pt[1])
        seg = nx.shortest_path(G, source=current, target=target, weight="weight")
        full_path.extend(seg[1:])
        current = target

    # Prune out-and-back excursions (…, a, b, …, x, …, b, a, …): a waypoint
    # that snaps to a node just off the through line (e.g. a dead-end border
    # spur) makes the path double back over the same nodes. Trains cannot
    # reverse mid-route, and revisited nodes corrupt the node-keyed elevation
    # lookup (grade spikes). Cancelling any step that immediately retraces the
    # previous edge removes such excursions entirely.
    pruned = [full_path[0]]
    for n in full_path[1:]:
        if len(pruned) >= 2 and pruned[-2] == n:
            pruned.pop()
        else:
            pruned.append(n)
    if len(pruned) != len(full_path):
        logger.info(
            f"Pruned {len(full_path) - len(pruned)} out-and-back nodes "
            "from the mainline path (waypoints snapped off the through line)"
        )
    full_path = pruned

    logger.info(f"Mainline path: {len(full_path)} nodes")

    # Detect sidings
    sidings = _identify_sidings(G, full_path)

    # Export
    export_path_with_sidings(
        G,
        full_path,
        sidings,
        output_prefix,
        fast_mode=fast,
        max_grade=max_grade,
        mapzen_fallback=mapzen_fallback,
    )
    return output_prefix + ".yaml"


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="Skip elevation fetch")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))  # applications/network_builder
    repo_root = os.path.dirname(os.path.dirname(here))
    input_xml = os.path.join(
        repo_root, "networks", "spokane_lethbridge_local.osm.xml"
    )
    output_prefix = os.path.join(here, "results", "spokane_lethbridge_with_sidings")

    cfg = BuilderConfig()
    build_network(
        input_xml,
        output_prefix,
        start_pt=cfg.start_pt,
        end_pt=cfg.end_pt,
        waypoints=cfg.route_waypoints,
        fast=args.fast,
        max_grade=cfg.max_grade,
    )


if __name__ == "__main__":
    main()
