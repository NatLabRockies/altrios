
import xml.etree.ElementTree as ET
import plotly.graph_objects as go
import logging
import os
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_osm_xml(xml_file):
    """
    Parses OSM XML to extract way geometries.
    """
    logger.info(f"Parsing {xml_file}...")
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML: {e}")
        sys.exit(1)
    
    # Extract Nodes: ID -> {lat, lon}
    nodes = {}
    for node in root.findall('node'):
        try:
            nodes[node.get('id')] = {
                'lat': float(node.get('lat')),
                'lon': float(node.get('lon'))
            }
        except (ValueError, TypeError):
            continue
        
    # Extract Ways: List of [ {lat, lon}, ... ]
    ways = []
    for way in root.findall('way'):
        way_nodes = []
        # Check tags if we want to filter specific rail types, 
        # but we assume the input is already filtered.
        
        for nd in way.findall('nd'):
            ref = nd.get('ref')
            if ref in nodes:
                way_nodes.append(nodes[ref])
        
        if len(way_nodes) > 1:
            ways.append(way_nodes)
            
    logger.info(f"Loaded {len(nodes)} nodes and parsed {len(ways)} ways.")
    return ways

def plot_ways(ways, output_path):
    """
    Plots the ways on an interactive HTML map using Plotly.
    """
    logger.info(f"Generating plot to {output_path}...")
    
    lats = []
    lons = []
    
    # Convert list of ways (segments) into a single trace with None gaps
    for way_nodes in ways:
        w_lats = [n['lat'] for n in way_nodes]
        w_lons = [n['lon'] for n in way_nodes]
        
        lats.extend(w_lats)
        lats.append(None) # Break the line between ways
        lons.extend(w_lons)
        lons.append(None)
        
    fig = go.Figure(go.Scattermapbox(
        mode="lines",
        lon=lons,
        lat=lats,
        marker={'size': 5},
        line={'width': 2, 'color': 'blue'},
        name='Rail Network'
    ))
    
    # Calculate map center
    valid_lats = [l for l in lats if l is not None]
    valid_lons = [l for l in lons if l is not None]
    
    if valid_lats:
        center_lat = sum(valid_lats) / len(valid_lats)
        center_lon = sum(valid_lons) / len(valid_lons)
    else:
        center_lat = 0
        center_lon = 0
        
    fig.update_layout(
        margin={'l':0,'t':0,'b':0,'r':0},
        mapbox={
            'style': "open-street-map",
            'center': {'lon': center_lon, 'lat': center_lat},
            'zoom': 7
        },
        title="Spokane to Lethbridge Rail Corridor (Local Extract)"
    )
    
    fig.write_html(output_path)
    logger.info(f"Plot saved to {output_path}")

def main():
    input_xml = "networks/spokane_lethbridge_local.osm.xml"
    output_html = "networks/spokane_lethbridge_local_plot.html"
    
    if not os.path.exists(input_xml):
        logger.error(f"Input file {input_xml} not found. Please run the build_corridor_from_pbf.py script first.")
        sys.exit(1)
        
    ways = parse_osm_xml(input_xml)
    plot_ways(ways, output_html)

if __name__ == "__main__":
    main()
