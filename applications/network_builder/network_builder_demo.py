#!/usr/bin/env python3
"""
Network builder demo.

Builds a routable ALTRIOS network (mainline + sidings) from an OSM rail extract
and plots the resulting geometry as an interactive HTML map. This is the single
entry point for the network-building pipeline:

    OSM PBFs ─build_corridor─▶ OSM XML ─build_network─▶ ALTRIOS YAML + geometry + locations
                                       ─plot_json─────▶ interactive HTML map

Everything corridor-specific lives in the CONFIG block below — edit those values
(or copy just this file) to build a different corridor. The workflow modules
(build_corridor_from_pbf.py, build_altrios_with_sidings.py) never need editing,
so there's no reason to keep multiple versions of them.

The demo ships with a clipped Spokane -> Lethbridge extract so it runs offline by
default. Use --download to (re)generate the extract from Geofabrik PBFs for the
configured corridor (requires the osmium-tool CLI).

Usage (uv-managed environment):
    uv sync                                            # one-time setup

    # Build + plot from the bundled extract (offline, fast):
    uv run python applications/network_builder/network_builder_demo.py

    # Fetch elevation/grade from an external API:
    uv run python applications/network_builder/network_builder_demo.py --with-elevation

    # (Re)download + clip the OSM extract for the configured corridor, then build:
    uv run python applications/network_builder/network_builder_demo.py --download
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent  # applications/network_builder
REPO_ROOT = THIS_DIR.parent.parent

# Make the sibling builder/plot modules importable regardless of cwd.
sys.path.insert(0, str(THIS_DIR))

from build_altrios_with_sidings import build_network  # noqa: E402
from plot_altrios_path import plot_json  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# CORRIDOR CONFIG -- edit this block to target a different corridor.
# These values are passed into the workflow modules as parameters; nothing in
# build_corridor_from_pbf.py or build_altrios_with_sidings.py needs to change.
# ============================================================================

# Basename for the built network files (written under the results dir).
OUTPUT_NAME = "spokane_lethbridge"

# Maximum track grade (decimal, e.g. 0.04 = 4%) used to smooth elevation spikes.
MAX_GRADE = 0.04

# Regional OSM PBFs (Geofabrik) covering the route -- only used with --download.
REGIONS = [
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
]

# Route waypoints (name, lat, lon), ordered start -> end. These define both the
# OSM clip area (--download) and the mainline route through the built network.
WAYPOINTS = [
    ("Spokane", 47.666, -117.360),
    ("Sandpoint", 48.282, -116.561),
    ("Bonners Ferry", 48.692, -116.319),
    ("Kingsgate (Border)", 49.000, -116.180),
    ("Cranbrook", 49.508, -115.747),
    ("Fernie", 49.504, -115.063),
    ("Crowsnest", 49.636, -114.682),
    ("Fort Macleod", 49.725, -113.397),
    ("Lethbridge", 49.724, -112.919),
]
# ============================================================================

# Bundled demo extract + default (gitignored) results location.
DEFAULT_INPUT = REPO_ROOT / "networks" / f"{OUTPUT_NAME}_local.osm.xml"
DEFAULT_OUTPUT_DIR = THIS_DIR / "results"


def main(args: argparse.Namespace) -> None:
    # 1. Obtain the OSM rail extract -- download fresh, or use the bundled one.
    if args.download:
        # Lazy import: only --download needs shapely + the osmium CLI.
        from build_corridor_from_pbf import build_corridor  # noqa: E402

        logger.info("Downloading + clipping OSM extract for the configured corridor...")
        input_xml = Path(
            build_corridor(
                output_name=f"{OUTPUT_NAME}_local",
                regions=REGIONS,
                waypoints=WAYPOINTS,
                networks_dir=str(REPO_ROOT / "networks"),
            )
        )
    else:
        input_xml = args.input.resolve()
        if not input_xml.exists():
            logger.error(f"Input OSM extract not found: {input_xml}")
            logger.error("Generate it with --download, or pass --input.")
            sys.exit(1)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(output_dir / OUTPUT_NAME)

    # 2. Derive the routing from the configured waypoints (start -> ... -> end).
    start_pt = (WAYPOINTS[0][1], WAYPOINTS[0][2])
    end_pt = (WAYPOINTS[-1][1], WAYPOINTS[-1][2])
    route_waypoints = [(lat, lon) for _name, lat, lon in WAYPOINTS[1:-1]]

    logger.info("=" * 70)
    logger.info(f"Network builder demo -- {OUTPUT_NAME}")
    logger.info("=" * 70)
    logger.info(f"Input extract : {input_xml}")
    logger.info(f"Output dir    : {output_dir}")
    logger.info(f"Max grade     : {MAX_GRADE * 100:.1f}%")
    logger.info(
        f"Elevation     : {'fetched' if args.with_elevation else 'skipped (fast)'}"
    )

    # 3. Build the ALTRIOS network YAML (+ geometry JSON + locations CSV).
    yaml_path = Path(
        build_network(
            str(input_xml),
            output_prefix,
            start_pt=start_pt,
            end_pt=end_pt,
            waypoints=route_waypoints,
            fast=not args.with_elevation,
            max_grade=MAX_GRADE,
        )
    )
    logger.info(f"Built network : {yaml_path}")

    # 4. Plot the built geometry as an interactive HTML map.
    if not args.no_plot:
        geom_path = output_prefix + "_geometry.json"
        html_path = output_prefix + "_plot.html"
        plot_json(geom_path, html_path)
        logger.info(f"Plot          : {html_path}")

    logger.info("=" * 70)
    logger.info("Demo complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"OSM XML rail extract to build from (default: {DEFAULT_INPUT.relative_to(REPO_ROOT)}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for built network + plot (default: {DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)}, gitignored).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="(Re)download Geofabrik PBFs and clip the OSM extract for the configured corridor (needs osmium-tool).",
    )
    parser.add_argument(
        "--with-elevation",
        action="store_true",
        help="Fetch elevation/grade data from an external API (default: skip for a fast, offline run).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the interactive HTML plot step.",
    )
    main(parser.parse_args())
