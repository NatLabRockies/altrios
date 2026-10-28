#!/usr/bin/env python3
"""
Network builder demo.

Builds a routable ALTRIOS network (mainline + sidings) from an OSM rail extract
and plots the resulting geometry as an interactive HTML map. This is the single
entry point for the network-building pipeline:

    OSM PBFs ─build_corridor─▶ OSM XML ─build_network─▶ ALTRIOS network (.msgpack) + geometry + locations
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

from builder_config import BuilderConfig  # noqa: E402
from build_altrios_with_sidings import build_network  # noqa: E402
from plot_altrios_path import plot_json  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================================================
# CORRIDOR CONFIG -- the default builds the bundled Spokane -> Lethbridge demo
# corridor. For a different corridor, pass --config <corridor.yaml> (see
# corridors/ for examples and BuilderConfig.from_file for the format), or
# override BuilderConfig fields here in code, e.g.:
#   CONFIG = BuilderConfig(
#       output_name="vancouver_kamloops",
#       max_grade=0.03,
#       regions=[{"name": "british-columbia",
#                 "url": ".../british-columbia-latest.osm.pbf"}],
#       waypoints=[("Vancouver", 49.28, -123.12), ("Hope", 49.38, -121.44),
#                  ("Kamloops", 50.67, -120.33)],
#   )
# ============================================================================
CONFIG = BuilderConfig()
# ============================================================================

# Default (gitignored) results location.
DEFAULT_OUTPUT_DIR = THIS_DIR / "results"


def main(args: argparse.Namespace) -> None:
    """Build the configured corridor network and plot it.

    Args:
        args: Parsed options: config, download, input, output_dir,
            with_elevation, no_plot and format.
    """
    global CONFIG
    if args.config:
        logger.info(f"Loading corridor config from {args.config}")
        CONFIG = BuilderConfig.from_file(args.config)

    # 1. Obtain the OSM rail extract -- download fresh, or use the bundled one.
    if args.download:
        # Lazy import: only --download needs shapely + the osmium CLI.
        from build_corridor_from_pbf import build_corridor  # noqa: E402

        logger.info("Downloading + clipping OSM extract for the configured corridor...")
        input_xml = Path(
            build_corridor(
                output_name=f"{CONFIG.output_name}_local",
                regions=CONFIG.regions,
                waypoints=CONFIG.waypoints,
                networks_dir=str(REPO_ROOT / "networks"),
                buffer_deg=CONFIG.buffer_deg,
            )
        )
    else:
        input_xml = (
            args.input or REPO_ROOT / "networks" / f"{CONFIG.output_name}_local.osm.xml"
        ).resolve()
        if not input_xml.exists():
            logger.error(f"Input OSM extract not found: {input_xml}")
            logger.error("Generate it with --download, or pass --input.")
            sys.exit(1)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(output_dir / CONFIG.output_name)

    logger.info("=" * 70)
    logger.info(f"Network builder demo -- {CONFIG.output_name}")
    logger.info("=" * 70)
    logger.info(f"Input extract : {input_xml}")
    logger.info(f"Output dir    : {output_dir}")
    logger.info(f"Max grade     : {CONFIG.max_grade * 100:.1f}%")
    logger.info(
        f"Elevation     : {'fetched' if args.with_elevation else 'skipped (fast)'}"
    )

    # 2. Build the ALTRIOS network file (+ geometry JSON + locations CSV).
    #    Routing (start -> ... -> end) is derived from CONFIG.waypoints.
    network_path = Path(
        build_network(
            str(input_xml),
            output_prefix,
            start_pt=CONFIG.start_pt,
            end_pt=CONFIG.end_pt,
            waypoints=CONFIG.route_waypoints,
            fast=not args.with_elevation,
            max_grade=CONFIG.max_grade,
            include_lat_lons=CONFIG.include_lat_lons,
            mapzen_fallback=CONFIG.mapzen_fallback,
            endpoint_buffer_meters=CONFIG.endpoint_buffer_meters,
            origin_name=CONFIG.waypoints[0][0],
            destination_name=CONFIG.waypoints[-1][0],
            origin_regions=CONFIG.origin_regions,
            destination_regions=CONFIG.destination_regions,
            output_format=args.format or CONFIG.output_format,
        )
    )
    logger.info(f"Built network : {network_path}")

    # 3. Plot the built geometry as an interactive HTML map.
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
        "--config",
        type=Path,
        default=None,
        help="Corridor config YAML (any subset of BuilderConfig fields; see corridors/). "
        "Default: the in-code CONFIG (bundled Spokane -> Lethbridge corridor).",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="OSM XML rail extract to build from "
        "(default: networks/<output_name>_local.osm.xml for the active config).",
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
    parser.add_argument(
        "--format",
        choices=("msgpack", "yaml"),
        default=None,
        help="Network file format (default: the config's output_format, msgpack).",
    )
    main(parser.parse_args())
