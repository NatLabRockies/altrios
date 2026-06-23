import os
import sys
import requests
import subprocess
import logging
import json
from shapely.geometry import LineString, mapping
import time
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CACHE_DIR = "osmium_cache"
NETWORKS_DIR = "networks"

# The corridor (output_name / regions / waypoints) is supplied by the caller.
# See BuilderConfig in altrios_builder_utils.py for the demo defaults.


def check_osmium_installed():
    try:
        subprocess.run(
            ["osmium", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def download_file(url, dest_path):
    if os.path.exists(dest_path):
        logger.info(f"File {dest_path} already exists. Skipping download.")
        return

    logger.info(f"Downloading {url} to {dest_path}...")
    try:
        # verify=False added to bypass SSL errors (e.g. corporate proxies)
        r = requests.get(url, stream=True, verify=False)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Download complete.")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        # Clean up partial file
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise e


def create_corridor_geojson(waypoints, output_path, buffer_deg=0.5):
    """
    Create a buffered polygon around the waypoints and save as GeoJSON.
    0.5 degrees approx 55 km radius (110km total width)
    """
    logger.info("Creating corridor polygon...")

    # Swap to (lon, lat) for GeoJSON/Shapely
    coords = [(lon, lat) for name, lat, lon in waypoints]

    line = LineString(coords)
    # Simplify slightly to reduce complexity if needed, but buffering handles it
    poly = line.buffer(buffer_deg)

    feature = {
        "type": "Feature",
        "properties": {"name": "Rail Corridor"},
        "geometry": mapping(poly),
    }

    feature_collection = {"type": "FeatureCollection", "features": [feature]}

    with open(output_path, "w") as f:
        json.dump(feature_collection, f, indent=2)

    logger.info(f"Saved corridor GeoJSON to {output_path}")


def build_corridor(
    output_name,
    regions,
    waypoints,
    networks_dir=NETWORKS_DIR,
    cache_dir=CACHE_DIR,
    buffer_deg=0.4,  # ~45 km radius around the waypoint line
):
    """Download regional OSM PBFs, clip them to a buffered corridor polygon
    around ``waypoints``, filter to ``railway=rail``, and write
    ``{networks_dir}/{output_name}.osm.pbf`` and ``.osm.xml``.

    Pass ``output_name`` / ``regions`` / ``waypoints`` to target a different
    corridor without editing this file. ``regions`` is a list of
    ``{"name", "url"}`` Geofabrik PBF entries; ``waypoints`` is a list of
    ``(name, lat, lon)`` points. Requires the osmium-tool CLI. Returns the path
    to the written ``.osm.xml`` extract (the input to build_altrios_with_sidings).
    """
    if not check_osmium_installed():
        logger.error("Error: 'osmium' command line tool not found.")
        logger.error(
            "Please install osmium-tool (e.g. 'brew install osmium-tool', 'apt install osmium-tool', or via conda)."
        )
        sys.exit(1)

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(networks_dir, exist_ok=True)

    # 1. Download PBFs
    pbf_paths = []
    for region in regions:
        filename = f"{region['name']}-latest.osm.pbf"
        dest_path = os.path.join(cache_dir, filename)
        download_file(region["url"], dest_path)
        pbf_paths.append(dest_path)

    # 2. Generate Corridor Polygon
    geojson_path = os.path.join(cache_dir, "corridor.geojson")
    create_corridor_geojson(waypoints, geojson_path, buffer_deg=buffer_deg)

    # 3. Merge PBFs
    merged_pbf_path = os.path.join(cache_dir, "merged_region.osm.pbf")
    logger.info("Merging PBF files...")
    # osmium merge inputs... -o output
    cmd_merge = ["osmium", "merge"] + pbf_paths + ["-o", merged_pbf_path, "--overwrite"]
    subprocess.run(cmd_merge, check=True)

    # 4. Extract Corridor
    corridor_pbf_path = os.path.join(cache_dir, "corridor_raw.osm.pbf")
    logger.info("Clipping to corridor polygon...")
    # osmium extract -p poly input -o output
    cmd_extract = [
        "osmium",
        "extract",
        "-p",
        geojson_path,
        merged_pbf_path,
        "-o",
        corridor_pbf_path,
        "--overwrite",
    ]
    subprocess.run(cmd_extract, check=True)

    # 5. Filter for Rail
    final_output_path = os.path.join(networks_dir, f"{output_name}.osm.pbf")
    logger.info("Filtering for railway=rail...")
    # osmium tags-filter input w/railway=rail -o output
    cmd_filter = [
        "osmium",
        "tags-filter",
        corridor_pbf_path,
        "w/railway=rail",
        "-o",
        final_output_path,
        "--overwrite",
    ]
    subprocess.run(cmd_filter, check=True)

    logger.info(f"Success! Final network saved to: {final_output_path}")

    # Convert to XML (the text format consumed by build_altrios_with_sidings.py)
    final_xml_path = os.path.join(networks_dir, f"{output_name}.osm.xml")
    logger.info(f"Converting to XML: {final_xml_path}")
    cmd_xml = [
        "osmium",
        "cat",
        "-f",
        "xml",
        final_output_path,
        "-o",
        final_xml_path,
        "--overwrite",
    ]
    subprocess.run(cmd_xml, check=True)

    logger.info("Done.")
    return final_xml_path


def main():
    # Builds the default demo corridor from BuilderConfig; callers wanting a
    # different corridor should pass their own output_name/regions/waypoints.
    try:
        from altrios_builder_utils import BuilderConfig
    except ImportError:
        from .altrios_builder_utils import BuilderConfig

    cfg = BuilderConfig()
    build_corridor(
        output_name=f"{cfg.output_name}_local",
        regions=cfg.regions,
        waypoints=cfg.waypoints,
        buffer_deg=cfg.buffer_deg,
    )


if __name__ == "__main__":
    main()
