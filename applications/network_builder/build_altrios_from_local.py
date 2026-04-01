
import xml.etree.ElementTree as ET
import networkx as nx
import math
import logging
import yaml
import json
import os
import sys
import os
# Ensure local modules (like altrios_builder_utils) can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
import urllib3
import time

try:
    from altrios_builder_utils import set_speed_limits, is_siding, smooth_elevation, add_endpoint_buffer
except ImportError:
    # Try relative import if running as module
    from .altrios_builder_utils import set_speed_limits, is_siding, smooth_elevation, add_endpoint_buffer

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
R_EARTH = 6371000  # meters

def fetch_elevations_3dep_bulk(locations):
    """
    Fetch elevations for a list of (lat, lon) using USGS EPQS (for US) 
    and OpenTopoData (for Canada/fallback).
    
    locations: list of (lat, lon)
    Returns: list of elevations in meters (float)
    """
    elevations = [None] * len(locations)
    
    # 1. Split into US and CA based on Latitude
    # < 49.0 is definitely US (mostly), > 49.0 is CA
    # Actually, the border is exactly 49.0.
    
    us_indices = []
    ca_indices = []
    
    for i, (lat, lon) in enumerate(locations):
        if lat < 49.001:  # Include border in US check first
            us_indices.append(i)
        else:
            ca_indices.append(i)
            
    logger.info(f"Elevation Fetch: {len(us_indices)} US points, {len(ca_indices)} CA points.")
    
    # 2. Fetch US points using USGS EPQS (One by one? Or bulk?)
    # EPQS is one by one. Bulk is via National Map API but complex.
    # We will use OpenTopoData 'ned10m' for US (Dataset from 3DEP) which supports bulk.
    # Or 'mapzen' which is global.
    
    # Let's use OpenTopoData for EVERYTHING because it supports bulk and is faster.
    # 'ned10m' for US, 'cdem' for Canada?
    # OpenTopoData public API is rate limited (1 call/sec, 100 locations/call).
    # We have ~7000 points. 70 calls. 70 seconds. Acceptable.
    
    # We will use 'mapzen' dataset as it covers both seamless.
    # Or 'ned10m' for US and 'mapzen' for CA? 'mapzen' is usually 30m.
    # 3DEP is 10m (ned10m).
    
    def fetch_opentopo(indices, dataset='mapzen'):
        if not indices:
            return
            
        chunk_size = 80 # Safe limit
        for i in range(0, len(indices), chunk_size):
            chunk_indices = indices[i:i+chunk_size]
            chunk_locs = [locations[idx] for idx in chunk_indices]
            
            # Format: lat,lon|lat,lon
            loc_str = "|".join([f"{lat},{lon}" for lat, lon in chunk_locs])
            url = f"https://api.opentopodata.org/v1/{dataset}?locations={loc_str}"
            
            retries = 5
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
            }
            
            for attempt in range(retries):
                try:
                    r = requests.get(url, headers=headers, verify=False, timeout=30)
                    if r.status_code == 200:
                        data = r.json()
                        results = data.get('results', [])
                        
                        # Validate length match
                        valid_cnt = 0
                        for j, res in enumerate(results):
                            elev = res.get('elevation')
                            if elev is not None and j < len(chunk_indices):
                                elevations[chunk_indices[j]] = elev
                                valid_cnt += 1
                        
                        # If excessive Nones, maybe retry?
                        if valid_cnt < len(chunk_indices) * 0.5:
                            logger.warning(f"Low valid yield ({valid_cnt}/{len(chunk_indices)}) from OpenTopoData. Retrying...")
                            time.sleep(2 * (attempt + 1))
                            continue
                            
                        break
                    elif r.status_code == 429: # Rate limit
                        wait_time = 5 * (attempt + 1)
                        logger.warning(f"Rate limited (429). Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.warning(f"OpenTopoData Error {r.status_code}: {r.text}")
                        time.sleep(2)
                except Exception as e:
                    logger.warning(f"Request failed: {e}")
                    time.sleep(2)
            
            logger.info(f"Fetched chunk {i}-{i+chunk_size}/{len(indices)}")
            time.sleep(1.5) # Rate limit padding
    
    # Fetch US with high res if possible, but mapzen is safer for consistency
    # User asked for 3DEP. 'ned10m' IS 3DEP.
    logger.info("Fetching elevations from OpenTopoData (ned10m for US)...")
    fetch_opentopo(us_indices, dataset='ned10m')
    
    logger.info("Fetching elevations from OpenTopoData (mapzen for CA)...")
    fetch_opentopo(ca_indices, dataset='mapzen')
    
    # Fill Nones
    valid_elevs = [e for e in elevations if e is not None]
    avg_elev = sum(valid_elevs)/len(valid_elevs) if valid_elevs else 0
    
    for i in range(len(elevations)):
        if elevations[i] is None:
            elevations[i] = avg_elev # Rough fallback
            
    return elevations

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return c * R_EARTH

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

def parse_osm_xml_to_graph(xml_file):
    """
    Parses OSM XML and builds a NetworkX graph.
    """
    logger.info(f"Parsing {xml_file}...")
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"Failed to parse XML: {e}")
        sys.exit(1)

    G = nx.Graph()
    
    # Node cache
    nodes = {} # id -> {lat, lon}

    # 1. Parse Nodes
    for node in root.findall('node'):
        nid = node.get('id')
        lat = float(node.get('lat'))
        lon = float(node.get('lon'))
        nodes[nid] = {'lat': lat, 'lon': lon}
        G.add_node(nid, lat=lat, lon=lon)

    # 2. Parse Ways (Edges)
    ways_count = 0
    for way in root.findall('way'):
        # Filter logic can go here (e.g. check for railway tag)
        # Assuming input is already filtered for railway=rail
        
        way_id = way.get('id')
        nd_refs = [nd.get('ref') for nd in way.findall('nd')]
        
        # Get tags for metadata
        tags = {tag.get('k'): tag.get('v') for tag in way.findall('tag')}
        
        if len(nd_refs) < 2:
            continue
            
        ways_count += 1
        
        for i in range(len(nd_refs) - 1):
            u = nd_refs[i]
            v = nd_refs[i+1]
            
            if u in nodes and v in nodes:
                # Calculate weight (distance)
                u_data = nodes[u]
                v_data = nodes[v]
                dist = haversine_distance(u_data['lon'], u_data['lat'], v_data['lon'], v_data['lat'])
                
                # Add edge
                # If edge already exists (parallel ways merging nodes), keep min distance or add metadata
                if G.has_edge(u, v):
                    # Multi-track or error. sum tags?
                    pass
                else:
                    G.add_edge(u, v, weight=dist, osm_way_id=way_id, tags=tags)

    logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

def find_nearest_node(G, lat, lon):
    """
    Finds the node in G closest to (lat, lon).
    """
    best_dist = float('inf')
    best_node = None
    
    for node, data in G.nodes(data=True):
        dist = haversine_distance(lon, lat, data['lon'], data['lat'])
        if dist < best_dist:
            best_dist = dist
            best_node = node
            
    return best_node, best_dist

def export_path_to_altrios(G, path_nodes, output_prefix, fast_mode=False):
    """
    Converts a linear sequence of nodes into an ALTRIOS network file.
    """
    logger.info("Building ALTRIOS network structure...")
    
    # 1. Fetch Elevations
    logger.info(f"Fetching elevations for {len(path_nodes)} nodes...")
    path_coords = [(G.nodes[n]['lat'], G.nodes[n]['lon']) for n in path_nodes]
    
    if fast_mode:
        logger.info("FAST MODE: Skipping elevation fetch, setting to 0.")
        node_elevations = [0.0] * len(path_coords)
    else:
        node_elevations = fetch_elevations_3dep_bulk(path_coords)
    
    # Smooth elevations
    logger.info("Smoothing elevations (Window=31)...")
    node_elevations = smooth_elevation(node_elevations, window_size=31)

    links_list = []
    # Dummy link 0 - speed_set must be None (null/~) for ALTRIOS to parse correctly
    links_list.append({
        "idx_curr": 0, "idx_flip": 0, "idx_next": 0, "idx_next_alt": 0, 
        "idx_prev": 0, "idx_prev_alt": 0, "osm_id": 0, "length_meters": 0.0,
        "elevs": [], "headings": [], "speed_set": None, "cat_power_limits": [], 
        "link_idxs_lockout": []
    })
    
    geometry_collection = []

    # --- START BUFFER ---
    # Add a 3km buffer BEFORE the first node
    start_buffer_len = 3000.0
    u_first = path_nodes[0]
    v_first = path_nodes[1]
    
    p_u = (G.nodes[u_first]['lon'], G.nodes[u_first]['lat'])
    p_v = (G.nodes[v_first]['lon'], G.nodes[v_first]['lat'])
    
    # Project backwards: Vector v->u, extend from u
    # add_endpoint_buffer takes [p_prev, p_curr] and extends p_curr
    # So we pass [v, u] to extend u
    p_buffer_start = add_endpoint_buffer([p_v, p_u], start_buffer_len)[-1]
    
    # Indices logic:
    # 0: Dummy
    # 1: Start Buffer Fwd
    # 2: Start Buffer Rev
    # 3: First Real Link Fwd (Offset 2)
    # ...
    
    IDX_OFFSET = 2
    
    # Geometry for Start Buffer
    geometry_collection.append({
        "link_idx": 1,
        "coordinates": [[p_buffer_start[0], p_buffer_start[1]], [p_u[0], p_u[1]]]
    })
    # Rev is same coords reversed? No, visualisation usually just wants one line or specific order.
    # But usually geometry json is just for display.
    
    elev_start = node_elevations[0]
    
    # Add Buffer Links
    # Fwd (1): Prev=0, Next=3
    links_list.append({
        "idx_curr": 1, "idx_flip": 2, "osm_tags": {"note": "start_buffer"},
        "length_meters": start_buffer_len, "osm_id": "start_buffer_fwd",
        "idx_next": 3, "idx_next_alt": 0, "idx_prev": 0, "idx_prev_alt": 0,
        "elevs": [{"offset_meters": 0.0, "elev_meters": elev_start}, {"offset_meters": start_buffer_len, "elev_meters": elev_start}],
        "headings": [{"offset_meters": 0.0, "heading_radians": 0.0}, {"offset_meters": start_buffer_len, "heading_radians": 0.0}],
        "speed_set": [], "cat_power_limits": [], "link_idxs_lockout": []
    })
    # Rev (2): Prev=4, Next=0
    links_list.append({
        "idx_curr": 2, "idx_flip": 1, "osm_tags": {"note": "start_buffer"},
        "length_meters": start_buffer_len, "osm_id": "start_buffer_rev",
        "idx_next": 0, "idx_next_alt": 0, "idx_prev": 4, "idx_prev_alt": 0, # 4 is first real Rev link
        "elevs": [{"offset_meters": 0.0, "elev_meters": elev_start}, {"offset_meters": start_buffer_len, "elev_meters": elev_start}],
        "headings": [{"offset_meters": 0.0, "heading_radians": 0.0}, {"offset_meters": start_buffer_len, "heading_radians": 0.0}],
        "speed_set": [], "cat_power_limits": [], "link_idxs_lockout": []
    })
    
    # --- MAIN PATH ---
    
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i+1]
        
        # Get edge data
        edge_data = G.get_edge_data(u, v)
        length = edge_data.get('weight', 0)
        
        # Enforce minimum length and consistency
        if length < 1.0:
            length = 1.0
        
        # Consistent float
        length = float(length)
            
        osm_id = edge_data.get('osm_way_id', 0)
        tags = edge_data.get('tags', {})
        
        # Link Indices
        link_idx_fwd = 2 * i + 1 + IDX_OFFSET
        link_idx_rev = 2 * i + 2 + IDX_OFFSET
        
        # Connectivity
        # Fwd Next: 2*(i+1)+1 + Offset
        idx_next_fwd = (2 * (i + 1) + 1 + IDX_OFFSET) if i < len(path_nodes) - 2 else 0
        # Fwd Prev: 2*(i-1)+1 + Offset. IF i=0, Prev is 1 (Start Buffer)
        idx_prev_fwd = (2 * (i - 1) + 1 + IDX_OFFSET) if i > 0 else 1
        
        # Rev Next: 2*(i-1)+2 + Offset. IF i=0, Next is 2 (Start Buffer Rev)
        idx_next_rev = (2 * (i - 1) + 2 + IDX_OFFSET) if i > 0 else 2
        # Rev Prev: 2*(i+1)+2 + Offset
        idx_prev_rev = (2 * (i + 1) + 2 + IDX_OFFSET) if i < len(path_nodes) - 2 else 0

        
        # Coordinates
        u_node = G.nodes[u]
        v_node = G.nodes[v]
        
        u_elev = node_elevations[i]
        v_elev = node_elevations[i+1]
        
        # Geometry
        geometry_collection.append({
            "link_idx": link_idx_fwd,
            "coordinates": [[u_node["lon"], u_node["lat"]], [v_node["lon"], v_node["lat"]]]
        })
        geometry_collection.append({
            "link_idx": link_idx_rev,
            "coordinates": [[v_node["lon"], v_node["lat"]], [u_node["lon"], u_node["lat"]]]
        })
        
        # Calculate Heading
        heading = math.atan2(v_node['lon'] - u_node['lon'], v_node['lat'] - u_node['lat']) # Approx
        # Better:
        bearing = get_bearing(u_node['lat'], u_node['lon'], v_node['lat'], v_node['lon'])
        heading_rad = math.radians(bearing)
        
        # Link Defs
        # Fwd
        links_list.append({
            "idx_curr": link_idx_fwd,
            "idx_flip": link_idx_rev,
            "osm_tags": tags,
            "length_meters": length,
            "osm_id": osm_id,
            "idx_next": idx_next_fwd,
            "idx_next_alt": 0,
            "idx_prev": idx_prev_fwd,
            "idx_prev_alt": 0,
            "elevs": [
                {"offset_meters": 0.0, "elev_meters": u_elev},
                {"offset_meters": length, "elev_meters": v_elev},
            ],
            "headings": [
                {"offset_meters": 0.0, "heading_radians": heading_rad},
                {"offset_meters": length, "heading_radians": heading_rad},
            ],
            "speed_set": [],
            "cat_power_limits": [],
            "link_idxs_lockout": [],
        })
        
        # Rev
        links_list.append({
            "idx_curr": link_idx_rev,
            "idx_flip": link_idx_fwd,
            "osm_tags": tags,
            "length_meters": length,
            "osm_id": osm_id,
            "idx_next": idx_next_rev,
            "idx_next_alt": 0,
            "idx_prev": idx_prev_rev,
            "idx_prev_alt": 0,
            "elevs": [
                {"offset_meters": 0.0, "elev_meters": v_elev},
                {"offset_meters": length, "elev_meters": u_elev},
            ],
            "headings": [
                {"offset_meters": 0.0, "heading_radians": (heading_rad + math.pi) % (2*math.pi)},
                {"offset_meters": length, "heading_radians": (heading_rad + math.pi) % (2*math.pi)},
            ],
            "speed_set": [],
            "cat_power_limits": [],
            "link_idxs_lockout": [],
        })

    # --- END BUFFER ---
    # Add Buffer Link at the End
    buffer_dist = 3000.0
    last_u = path_nodes[-2]
    last_v = path_nodes[-1]
    
    p_before = (G.nodes[last_u]['lon'], G.nodes[last_u]['lat'])
    p_last = (G.nodes[last_v]['lon'], G.nodes[last_v]['lat'])
    
    p_new = add_endpoint_buffer([p_before, p_last], buffer_dist)[-1]
    
    link_idx_buffer = len(links_list) + 1 # Next avail index (0 based list but dummy 0 is there)
    # Actually just use length of list if dummy is included? 
    # List: [0, 1, 2, 3, 4 ... N]. len = N+1. Next idx = N+1.
    link_idx_buffer = len(links_list)
    
    # Correction: The logic above generates IDs 1, 2... M. 
    # links_list indices match if sorted. But let's trust explicit ID.
    
    buffer_osm_tags = {'service': 'siding', 'note': 'end_buffer'}
    
    # Update last real link's next
    # Last loop i = len(nodes)-2.
    # Last Fwd ID = 2*(len-2)+1 + OFFSET
    last_fwd_idx = 2 * (len(path_nodes) - 2) + 1 + IDX_OFFSET
    
    found = False
    for l in links_list:
        if l['idx_curr'] == last_fwd_idx:
            l['idx_next'] = link_idx_buffer
            found = True
            break
            
    if not found:
        logger.warning(f"Could not find last link {last_fwd_idx} to connect buffer.")
            
    # Add buffer link
    links_list.append({
        "idx_curr": link_idx_buffer,
        "idx_flip": 0,
        "osm_tags": buffer_osm_tags,
        "length_meters": buffer_dist,
        "osm_id": "end_buffer",
        "idx_next": 0,
        "idx_next_alt": 0,
        "idx_prev": last_fwd_idx,
        "idx_prev_alt": 0,
        "elevs": [
             {"offset_meters": 0.0, "elev_meters": node_elevations[-1]},
             {"offset_meters": buffer_dist, "elev_meters": node_elevations[-1]}
        ],
        "headings": [
            {"offset_meters": 0.0, "heading_radians": 0.0},
            {"offset_meters": buffer_dist, "heading_radians": 0.0}
        ],
        "speed_set": [],
        "cat_power_limits": [],
        "link_idxs_lockout": [],
    })
    
    geometry_collection.append({
        "link_idx": link_idx_buffer,
        "coordinates": [[p_last[0], p_last[1]], [p_new[0], p_new[1]]]
    })

    # Writing Files
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    yaml_path = output_prefix + ".yaml"
    geo_path = output_prefix + "_geometry.json"

    # Set Speed Limits
    logger.info("Setting speed limits based on OSM tags (siding detection)...")
    set_speed_limits(links_list, default_speed_mps=26.8, siding_speed_mps=6.7)
    
    # Remove temporary tags
    for link in links_list:
        link.pop("osm_tags", None)
    
    network_config = [
        {
            "max_grade": 1000.0,
            "max_curv_radians_per_meter": 1.0,
            "max_heading_step_radians": 3.5,
            "max_elev_step_meters": 100.0,
        },
        links_list,
    ]
    
    logger.info(f"Saving network to {yaml_path}")
    with open(yaml_path, "w") as f:
        yaml.dump(network_config, f)

    logger.info(f"Saving geometry to {geo_path}")
    with open(geo_path, "w") as f:
        json.dump(geometry_collection, f)

    # --- SAVE LOCATIONS CSV ---
    # Locations:
    # "Spokane" -> Start of Start Buffer (Link 1, offset 0)
    # Note: Altrios requires Origin Offset to be 0 and relative to Tail.
    # So the train Tail will be at the start of Link 1 (Buffer).
    # Since Link 1 is 3km long and flat, the train (approx 2km max) will fit entirely on Link 1.
    # This prevents geometry errors at the start.
    
    loc_csv_path = output_prefix + "_locations.csv"
    logger.info(f"Saving locations map to {loc_csv_path}")
    
    # Note: CSV format for altrios: Location ID, Link Index, Offset (m), ...
    with open(loc_csv_path, "w") as f:
        f.write("Location ID,Link Index,Offset (m),Is Front End,Grid Emissions Region,Electricity Price Region,Liquid Fuel Price Region\n")
        f.write(f"Spokane,{1},0,FALSE,WECC,WA,WA\n") 
        f.write(f"Lethbridge,{last_fwd_idx},0,FALSE,WECC,ab,ab\n")
        
    return geometry_collection


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Skip elevation fetch for fast testing")
    args = parser.parse_args()

    input_xml = "networks/spokane_lethbridge_local.osm.xml"
    output_name = "spokane_lethbridge_path"
    output_path = os.path.join("networks", output_name)
    
    # 1. Load Graph
    G = parse_osm_xml_to_graph(input_xml)
    
    # 2. Define Points (Lat, Lon)
    # Updated coordinates per user request
    # Start: Spokane area (47.668505, -117.342123)
    start_pt = (47.668505, -117.342123) 
    
    # End: Lethbridge area (49.698540, -112.796653)
    end_pt = (49.698540, -112.796653)
    
    # Intermediate points to force the route
    # UP Spokane Subdivision / Sandpoint
    # Eastport/Kingsgate Border
    # Crowsnest Pass
    waypoints = [
        (48.282, -116.561), # Sandpoint
        (49.000, -116.180), # Kingsgate
        (49.636, -114.682), # Crowsnest
    ]
    
    logger.info("Finding nearest nodes...")
    start_node, d_start = find_nearest_node(G, start_pt[0], start_pt[1])
    end_node, d_end = find_nearest_node(G, end_pt[0], end_pt[1])
    
    logger.info(f"Start Node: {start_node} ({d_start:.0f}m from target)")
    logger.info(f"End Node: {end_node} ({d_end:.0f}m from target)")
    
    if d_start > 5000:
        logger.warning(f"Start node is far from target! {d_start}m")
    if d_end > 5000:
        logger.warning(f"End node is far from target! {d_end}m")
        
    # 3. Compute Path
    logger.info("Computing shortest path...")
    
    full_path = []
    
    # Start -> Waypoint 1
    current_start = start_node
    
    path_segments = waypoints + [(end_pt[0], end_pt[1])]
    
    try:
        current_node = start_node
        full_path = [current_node]
        
        for pt in path_segments:
            target_node, dist = find_nearest_node(G, pt[0], pt[1])
            logger.info(f"Routing to waypoint {target_node} ({dist:.0f}m)...")
            
            segment = nx.shortest_path(G, source=current_node, target=target_node, weight='weight')
            
            # Append segment (excluding the first node as it's already in path)
            full_path.extend(segment[1:])
            current_node = target_node
            
    except nx.NetworkXNoPath:
        logger.error("No path found between points!")
        sys.exit(1)
        
    logger.info(f"Path found! Total nodes: {len(full_path)}")
    
    # 4. Export
    export_path_to_altrios(G, full_path, output_path, fast_mode=args.fast)
    logger.info("Done.")

if __name__ == "__main__":
    main()
