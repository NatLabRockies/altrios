
import sys
import os
import time

# Ensure we can import from the component script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from download_and_build_corridor import download_corridor, build_graph, simplify_and_export_altrios, logger
import logging

# Define Route Waypoints to ensure we follow the rail corridor through the mountains
WAYPOINTS = [
    ("Spokane", 47.666, -117.360),
    ("Sandpoint", 48.282, -116.561),
    ("Bonners Ferry", 48.692, -116.319),
    ("Kingsgate (Border)", 49.000, -116.180),
    ("Cranbrook", 49.508, -115.747),
    ("Fernie", 49.504, -115.063),
    ("Crowsnest", 49.636, -114.682),
    ("Fort Macleod", 49.725, -113.397),
    ("Lethbridge", 49.724, -112.919)
]

if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    
    all_ways = {}
    all_nodes = {}
    
    for i in range(len(WAYPOINTS) - 1):
        p1_name, p1_lat, p1_lon = WAYPOINTS[i]
        p2_name, p2_lat, p2_lon = WAYPOINTS[i+1]
        
        logger.info(f"--- Segment {i+1}/{len(WAYPOINTS)-1}: {p1_name} -> {p2_name} ---")
        
        # Use wider corridor (30km) to ensure we catch winding tracks
        ways, nodes = download_corridor(
            (p1_lat, p1_lon), 
            (p2_lat, p2_lon), 
            step_km=20, 
            width_km=30
        )
        
        if ways:
            all_ways.update(ways)
            all_nodes.update(nodes)
        else:
            logger.warning(f"No data found for segment {p1_name} -> {p2_name}")
        
        # Small pause between segments
        time.sleep(2)

    logger.info("Merging segments...")
    logger.info(f"Total unique ways: {len(all_ways)}")
    logger.info(f"Total unique nodes: {len(all_nodes)}")
    
    if not all_ways:
        logger.error("No data found!")
        sys.exit(1)

    logger.info("Building graph...")
    G = build_graph(all_ways, all_nodes)
    
    output_name = "spokane_lethbridge_full"
    output_path = os.path.join("networks", output_name)
    
    # Ensure networks directory exists
    os.makedirs("networks", exist_ok=True)
    
    simplify_and_export_altrios(G, output_path, plot_html=True)
    logger.info(f"Done. Network saved to {output_path}")
