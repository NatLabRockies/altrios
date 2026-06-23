#!/usr/bin/env python3
"""
Network builder demo.

Builds a routable ALTRIOS network (mainline + sidings) from a bundled OSM rail
extract and plots the resulting geometry as an interactive HTML map. This is the
single entry point for the network-building pipeline:

    OSM XML extract  ──build_network──▶  ALTRIOS YAML + geometry JSON + locations CSV
                     ──plot_json──────▶  interactive HTML map

The demo ships with a clipped Spokane → Lethbridge rail corridor
(networks/spokane_lethbridge_local.osm.xml) so it runs without downloading any
OSM data. To build a different corridor, generate a new clipped extract with
build_corridor_from_pbf.py (requires the osmium-tool CLI) and pass it via
--input, optionally overriding the route in build_altrios_with_sidings.py.

Usage (uv-managed environment):
    # One-time setup: sync deps + build altrios:
    uv sync

    # Run the demo (fast: skips elevation fetch, no network access needed):
    uv run python applications/network_builder/network_builder_demo.py

    # Include elevation/grade data (fetches from an external elevation API):
    uv run python applications/network_builder/network_builder_demo.py --with-elevation

    # Custom input extract / output location:
    uv run python applications/network_builder/network_builder_demo.py \
        --input path/to/extract.osm.xml --output-dir path/to/out
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

# Bundled demo data + default (gitignored) results location.
DEFAULT_INPUT = REPO_ROOT / "networks" / "spokane_lethbridge_local.osm.xml"
DEFAULT_OUTPUT_DIR = THIS_DIR / "results"
OUTPUT_BASENAME = "spokane_lethbridge"


def main(args: argparse.Namespace) -> None:
    input_xml = args.input.resolve()
    if not input_xml.exists():
        logger.error(f"Input OSM extract not found: {input_xml}")
        logger.error("Generate one with build_corridor_from_pbf.py, or pass --input.")
        sys.exit(1)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(output_dir / OUTPUT_BASENAME)

    logger.info("=" * 70)
    logger.info("Network builder demo")
    logger.info("=" * 70)
    logger.info(f"Input extract : {input_xml}")
    logger.info(f"Output dir    : {output_dir}")
    logger.info(
        f"Elevation     : {'fetched' if args.with_elevation else 'skipped (fast)'}"
    )

    # 1. Build the ALTRIOS network YAML (+ geometry JSON + locations CSV).
    yaml_path = Path(
        build_network(str(input_xml), output_prefix, fast=not args.with_elevation)
    )
    logger.info(f"Built network : {yaml_path}")

    # 2. Plot the built geometry as an interactive HTML map.
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
