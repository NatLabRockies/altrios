"""
Script to download railway network data for a corridor using Overpass API
and convert it to ALTRIOS-compatible network files.

Usage:
    python download_and_build_corridor.py --start_lat 47.65 --start_lon -117.42 --end_lat 51.04 --end_lon -114.07 --output_name spokane_calgary

Dependencies:
    overpy
    networkx
    pyyaml
"""

import overpy
import math
import time
import json
import yaml
import os
import argparse
import networkx as nx

try:
    from altrios_builder_utils import (
        set_speed_limits,
        is_siding,
        smooth_elevation,
        add_endpoint_buffer,
    )
except ImportError:
    # Try relative import if running as module
    from .altrios_builder_utils import (
        set_speed_limits,
        is_siding,
        smooth_elevation,
        add_endpoint_buffer,
    )

import logging
import plotly.graph_objects as go
from math import radians, cos, sin, asin, sqrt, atan2, degrees

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000  # meters


def get_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate initial bearing between two points.
    """
    startLat = math.radians(lat1)
    startLong = math.radians(lon1)
    endLat = math.radians(lat2)
    endLong = math.radians(lon2)

    dLong = endLong - startLong

    dPhi = math.log(
        math.tan(endLat / 2.0 + math.pi / 4.0)
        / math.tan(startLat / 2.0 + math.pi / 4.0)
    )
    if abs(dLong) > math.pi:
        if dLong > 0.0:
            dLong = -(2.0 * math.pi - dLong)
        else:
            dLong = 2.0 * math.pi + dLong

    bearing = (math.degrees(math.atan2(dLong, dPhi)) + 360.0) % 360.0
    return bearing


def get_point_at_distance(lat1, lon1, d, bearing, R=6371):
    """
    lat: initial latitude, in degrees
    lon: initial longitude, in degrees
    d: distance, in km
    bearing: bearing, in degrees
    R: earth radius, in km
    """
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    a = math.radians(bearing)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(d / R)
        + math.cos(lat1) * math.sin(d / R) * math.cos(a)
    )
    lon2 = lon1 + math.atan2(
        math.sin(a) * math.sin(d / R) * math.cos(lat1),
        math.cos(d / R) - math.sin(lat1) * math.sin(lat2),
    )

    return (math.degrees(lat2), math.degrees(lon2))


def fetch_osm_data(api, bbox):
    """
    Fetch railway data for a bounding box.
    bbox: (south, west, north, east)
    """
    query = f"""
    [out:json][timeout:25];
    (
      way["railway"="rail"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    (._;>;);
    out body;
    """
    try:
        # Retry loop for server errors
        max_retries = 5
        base_wait = 5

        for i in range(max_retries):
            try:
                result = api.query(query)
                return result
            except (
                overpy.exception.OverpassTooManyRequests,
                overpy.exception.OverpassGatewayTimeout,
            ) as e:
                wait_time = base_wait * (2**i)
                logger.warning(f"Overpass API error: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            except Exception as e:
                # For other errors, maybe just retry once or fail
                if "Server load too high" in str(e):
                    wait_time = base_wait * (2**i)
                    logger.warning(
                        f"Server load high: {e}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise e

        # If we get here, we failed after retries
        logger.error(
            f"Failed to fetch data for bbox {bbox} after {max_retries} retries."
        )
        return None

    except Exception as e:
        logger.error(f"Error fetching data for bbox {bbox}: {e}")
        return None


def download_corridor(start_coords, end_coords, step_km=25, width_km=25):
    """
    Download OSM data along a corridor.
    start_coords: (lat, lon)
    end_coords: (lat, lon)
    """
    api = overpy.Overpass()

    total_dist_meters = haversine_distance(
        start_coords[1], start_coords[0], end_coords[1], end_coords[0]
    )
    total_dist_km = total_dist_meters / 1000.0

    bearing = get_bearing(
        start_coords[0], start_coords[1], end_coords[0], end_coords[1]
    )

    logger.info(f"Total distance: {total_dist_km:.2f} km, Bearing: {bearing:.2f}")

    num_steps = int(math.ceil(total_dist_km / step_km))

    all_ways = {}
    all_nodes = {}

    for i in range(num_steps + 1):
        dist = i * step_km
        center_pt = get_point_at_distance(
            start_coords[0], start_coords[1], dist, bearing
        )

        # Create bbox
        delta_deg = (width_km / 111.0) / 2.0  # Approx degrees
        bbox = (
            center_pt[0] - delta_deg,  # south
            center_pt[1] - delta_deg,  # west
            center_pt[0] + delta_deg,  # north
            center_pt[1] + delta_deg,  # east
        )

        logger.info(
            f"Downloading chunk {i + 1}/{num_steps + 1} centered at {center_pt}"
        )

        result = fetch_osm_data(api, bbox)
        if result:
            for way in result.ways:
                all_ways[way.id] = way
            for node in result.nodes:
                all_nodes[node.id] = node

        time.sleep(1)  # Be nice to the API

    logger.info(f"Downloaded {len(all_ways)} ways and {len(all_nodes)} nodes.")
    return all_ways, all_nodes


def build_graph(ways, nodes):
    G = nx.Graph()

    for way_id, way in ways.items():
        # Filter for valid rail types if needed
        tags = way.tags

        way_nodes = way.nodes
        for i in range(len(way_nodes) - 1):
            n1 = way_nodes[i]
            n2 = way_nodes[i + 1]

            # Check if nodes exist in downloaded data (sometimes nodes are missing if at edge of bbox)
            if n1.id in nodes and n2.id in nodes:
                node1_data = nodes[n1.id]
                node2_data = nodes[n2.id]

                dist = haversine_distance(
                    float(node1_data.lon),
                    float(node1_data.lat),
                    float(node2_data.lon),
                    float(node2_data.lat),
                )

                G.add_node(n1.id, lat=float(node1_data.lat), lon=float(node1_data.lon))
                G.add_node(n2.id, lat=float(node2_data.lat), lon=float(node2_data.lon))
                G.add_edge(n1.id, n2.id, weight=dist, osm_way_id=way_id, tags=tags)

    return G


def plot_network(geometry_collection, output_path: str):
    """
    Plot the network links using Plotly.
    geometry_collection: List of dicts with 'coordinates' key [[lon, lat], [lon, lat]]
    """
    logger.info(f"Generating plot to {output_path}...")

    lats = []
    lons = []

    for item in geometry_collection:
        coords = item["coordinates"]
        # Add line segment
        lons.extend([coords[0][0], coords[1][0], None])
        lats.extend([coords[0][1], coords[1][1], None])

    fig = go.Figure(
        go.Scattermapbox(
            mode="lines",
            lon=lons,
            lat=lats,
            marker={"size": 10},
            line={"width": 2, "color": "blue"},
            name="Rail Network",
        )
    )

    # Center map
    if lats:
        center_lat = sum([l for l in lats if l is not None]) / len(
            [l for l in lats if l is not None]
        )
        center_lon = sum([l for l in lons if l is not None]) / len(
            [l for l in lons if l is not None]
        )
    else:
        center_lat = 0
        center_lon = 0

    fig.update_layout(
        margin={"l": 0, "t": 0, "b": 0, "r": 0},
        mapbox={
            "style": "open-street-map",
            "center": {"lon": center_lon, "lat": center_lat},
            "zoom": 8,
        },
        title="ALTRIOS Network Corridor",
    )

    fig.write_html(output_path)
    logger.info("Plot saved.")


def simplify_and_export_altrios(G, output_prefix: str, plot_html: bool = False):
    """
    Convert graph to ALTRIOS format.
    Simple approach: Each edge in the simplified graph becomes a link.
    """
    logger.info("Building ALTRIOS network structure...")

    # Get largest connected component
    if len(G) == 0:
        logger.error("Graph is empty!")
        return

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    largest_comp = G.subgraph(components[0]).copy()
    logger.info(f"Largest component has {len(largest_comp)} nodes.")

    # Convert to directed for ALTRIOS (links are directional)
    # ALTRIOS expects links to be indexed 1..N
    # Creates a simplified link list

    links_list = []
    # Dummy link 0
    links_list.append(
        {
            "idx_curr": 0,
            "idx_flip": 0,
            "idx_next": 0,
            "idx_next_alt": 0,
            "idx_prev": 0,
            "idx_prev_alt": 0,
            "osm_id": 0,
            "length_meters": 0,
            "elevs": [],
            "headings": [],
            "speed_set": None,
            "cat_power_limits": [],
            "link_idxs_lockout": [],
        }
    )

    # Iterate edges and create links
    # Note: A graph edge (u, v) becomes TWO links (u->v and v->u)

    edge_list = list(largest_comp.edges(data=True))

    # Map edges to link indices
    # Edge i (u, v) -> Link 2*i+1 (u->v), Link 2*i+2 (v->u)

    edge_to_link_idx = {}

    geometry_collection = []

    for i, (u, v, data) in enumerate(edge_list):
        # Indices
        link_idx_fwd = 2 * i + 1
        link_idx_rev = 2 * i + 2

        length = data.get("weight", 0)

        # Get coordinates for geometry file
        u_node = largest_comp.nodes[u]
        v_node = largest_comp.nodes[v]

        geometry_collection.append(
            {
                "link_idx": link_idx_fwd,
                "coordinates": [
                    [u_node["lon"], u_node["lat"]],
                    [v_node["lon"], v_node["lat"]],
                ],
            }
        )
        geometry_collection.append(
            {
                "link_idx": link_idx_rev,
                "coordinates": [
                    [v_node["lon"], v_node["lat"]],
                    [u_node["lon"], u_node["lat"]],
                ],
            }
        )

        # Store mapping for connectivity building
        # For node u, outgoing link is link_idx_fwd (to v)
        # For node v, incoming link is link_idx_fwd (from u)

        # To build idx_next/prev, we need to know which links connect to which nodes
        if u not in edge_to_link_idx:
            edge_to_link_idx[u] = []
        if v not in edge_to_link_idx:
            edge_to_link_idx[v] = []

        edge_to_link_idx[u].append(
            {"to": v, "id": link_idx_fwd, "rev_id": link_idx_rev}
        )
        edge_to_link_idx[v].append(
            {"to": u, "id": link_idx_rev, "rev_id": link_idx_fwd}
        )

        # Add Forward Link
        links_list.append(
            {
                "idx_curr": link_idx_fwd,
                "osm_tags": data.get("tags", {}),
                "idx_flip": link_idx_rev,
                "length_meters": length,
                "osm_id": data.get("osm_way_id", 0),
                # Placeholders for complex logic not implemented in simple download
                "idx_next": 0,
                "idx_next_alt": 0,
                "idx_prev": 0,
                "idx_prev_alt": 0,
                # FIXED: Providing dummy elevs/headings so Rust serde doesn't panic
                "elevs": [
                    {"offset_meters": 0.0, "elev_meters": 1000.0},
                    {"offset_meters": length, "elev_meters": 1000.0},
                ],
                "headings": [
                    {"offset_meters": 0.0, "heading_radians": 0.0},
                    {"offset_meters": length, "heading_radians": 0.0},
                ],
                "speed_set": [],
                "cat_power_limits": [],
                "link_idxs_lockout": [],
            }
        )

        # Add Reverse Link
        links_list.append(
            {
                "idx_curr": link_idx_rev,
                "osm_tags": data.get("tags", {}),
                "idx_flip": link_idx_fwd,
                "length_meters": length,
                "osm_id": data.get("osm_way_id", 0),
                "idx_next": 0,
                "idx_next_alt": 0,
                "idx_prev": 0,
                "idx_prev_alt": 0,
                "elevs": [
                    {"offset_meters": 0.0, "elev_meters": 1000.0},
                    {"offset_meters": length, "elev_meters": 1000.0},
                ],
                "headings": [
                    {"offset_meters": 0.0, "heading_radians": 0.0},
                    {"offset_meters": length, "heading_radians": 0.0},
                ],
                "speed_set": [],
                "cat_power_limits": [],
                "link_idxs_lockout": [],
            }
        )

    # Fix Connectivity (First Pass)
    # This is a naive connectivity builder.
    # For each link u->v (fwd), idx_next is a link starting at v.

    # We need a lookup: Node -> List of Outgoing Links
    node_outgoing = {}
    node_incoming = {}  # Node -> List of Incoming Links

    for link in links_list:
        if link["idx_curr"] == 0:
            continue
        # Recover u,v from somewhere?
        # Easier: Re-iterate edges and use the map we built
        pass

    # Re-iterate edges to set connectivity
    for i, (u, v, data) in enumerate(edge_list):
        link_idx_fwd = 2 * i + 1
        link_idx_rev = 2 * i + 2

        # Fwd Link (u->v): Next connect to v's outgoing. Prev connect to u's incoming.

        # Find links starting at v (excluding the reverse of current)
        v_outgoing = [x["id"] for x in edge_to_link_idx[v] if x["to"] != u]
        # Find links ending at u (excluding reverse of current)
        u_incoming = [
            x["rev_id"] for x in edge_to_link_idx[u] if x["to"] != v
        ]  # x['to'] is neighbor of u. rev_id is link neighbor->u

        # Assign
        if v_outgoing:
            links_list[link_idx_fwd]["idx_next"] = v_outgoing[0]
            if len(v_outgoing) > 1:
                links_list[link_idx_fwd]["idx_next_alt"] = v_outgoing[1]

        if u_incoming:
            links_list[link_idx_fwd]["idx_prev"] = u_incoming[
                0
            ]  # Actually detailed logic needed here usually

        # Rev Link (v->u): Next connect to u's outgoing
        u_outgoing = [x["id"] for x in edge_to_link_idx[u] if x["to"] != v]
        if u_outgoing:
            links_list[link_idx_rev]["idx_next"] = u_outgoing[0]
            if len(u_outgoing) > 1:
                links_list[link_idx_rev]["idx_next_alt"] = u_outgoing[1]

    # Save outputs
    # os.makedirs(os.path.dirname(output_prefix), exist_ok=True) # Assuming dir exists or handled by caller

    # Apply Speed Limits using Utility
    logger.info("Setting speed limits based on OSM tags (siding detection)...")
    set_speed_limits(links_list, default_speed_mps=26.8, siding_speed_mps=6.7)

    # Clean up temporary tags before saving
    for link in links_list:
        link.pop("osm_tags", None)

    yaml_path = output_prefix + ".yaml"
    geo_path = output_prefix + "_geometry.json"

    network_config = [
        {
            "max_grade": 0.02,
            "max_curv_radians_per_meter": 0.01,
            "max_heading_step_radians": 0.1,
            "max_elev_step_meters": 10.0,
        },
        links_list,
    ]

    logger.info(f"Saving network to {yaml_path}")
    with open(yaml_path, "w") as f:
        yaml.dump(network_config, f)

    logger.info(f"Saving geometry to {geo_path}")
    with open(geo_path, "w") as f:
        json.dump(geometry_collection, f)

    if plot_html:
        plot_path = output_prefix + "_plot.html"
        plot_network(geometry_collection, plot_path)

    return


def main():
    parser = argparse.ArgumentParser(description="Download Corridor Rail Network")
    parser.add_argument("--start_lat", type=float, required=True, help="Start Latitude")
    parser.add_argument(
        "--start_lon", type=float, required=True, help="Start Longitude"
    )
    parser.add_argument("--end_lat", type=float, required=True, help="End Latitude")
    parser.add_argument("--end_lon", type=float, required=True, help="End Longitude")
    parser.add_argument(
        "--output_name",
        type=str,
        default="corridor_network",
        help="Output filename prefix (no extension)",
    )
    parser.add_argument(
        "--width_km", type=float, default=20.0, help="Width of search corridor in km"
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate HTML plot of the network"
    )

    args = parser.parse_args()

    start = (args.start_lat, args.start_lon)
    end = (args.end_lat, args.end_lon)

    logger.info(f"Starting download for corridor from {start} to {end}")

    ways, nodes = download_corridor(start, end, step_km=25, width_km=args.width_km)

    if not ways:
        logger.error("No data found!")
        return

    logger.info("Building graph...")
    G = build_graph(ways, nodes)

    output_path = os.path.join("networks", args.output_name)
    simplify_and_export_altrios(G, output_path, plot_html=args.plot)
    logger.info("Done.")


if __name__ == "__main__":
    main()
