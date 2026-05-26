#!/usr/bin/env python3
"""
Unified Network Builder Workflow

Single script that orchestrates the full network-building pipeline:
  1. Download:      Fetch regional OSM PBF files, clip to corridor, filter rail
                    → networks/<corridor>.osm.pbf + .osm.xml
                    (requires the osmium-tool CLI; e.g. `brew install osmium-tool`)
  2. Visualize_raw: Plot all raw OSM ways as an interactive HTML map (debug)
                    → networks/<corridor>_local_plot.html
  3. Build:         Build an ALTRIOS network YAML from local OSM XML
                    (mainline + sidings)
                    → networks/<corridor>_with_sidings.yaml + _geometry.json
  4. Plot:          Plot the resulting ALTRIOS path as an interactive HTML map
                    → networks/<corridor>_*_plot.html
  5. Simulate:      Run a SpeedLimitTrainSim on the built network YAML
                    → results saved under applications/network_builder/results

Usage (uv-managed environment):
    # One-time setup: sync deps + build altrios (auto-selects Python 3.10/3.11):
    uv sync

    # Full pipeline (download → build → plot → simulate):
    uv run python applications/network_builder/network_builder_workflow.py

    # Skip the download step (use existing local OSM XML):
    uv run python applications/network_builder/network_builder_workflow.py \
        --steps build plot simulate

    # Fast mode (skip elevation fetch):
    uv run python applications/network_builder/network_builder_workflow.py \
        --fast --steps build plot

    # Just rebuild the simulation on an existing YAML:
    uv run python applications/network_builder/network_builder_workflow.py \
        --steps simulate

NOTE on hardcoded corridor:
    The build scripts currently hardcode the Spokane → Sandpoint → Kingsgate →
    Crowsnest → Lethbridge waypoints. Use --corridor-name only to change the
    output file basename for steps that accept it (plot/simulate). To target a
    different corridor, edit waypoints in build_altrios_with_sidings.py and
    build_corridor_from_pbf.py, or use download_and_build_corridor.py directly.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent  # applications/network_builder
WORKSPACE_ROOT = THIS_DIR.parent.parent  # repo root
NETWORKS_DIR = WORKSPACE_ROOT / "networks"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_script(script: Path, extra_args: list[str] | None = None) -> None:
    """Run a Python script as a subprocess, with cwd = workspace root."""
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    cmd = [sys.executable, str(script)] + (extra_args or [])
    logger.info(f"  $ {' '.join(cmd)}  (cwd={WORKSPACE_ROOT})")
    subprocess.run(cmd, check=True, cwd=str(WORKSPACE_ROOT))


def _header(title: str) -> None:
    logger.info("=" * 70)
    logger.info(title)
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# Step 1: Download (regional PBFs → clipped + filtered corridor)
# ---------------------------------------------------------------------------


def step_download(corridor_name: str) -> None:
    _header(f"STEP 1: Download + clip corridor → networks/{corridor_name}.osm.*")
    # build_corridor_from_pbf.py has hardcoded OUTPUT_NAME / REGIONS / WAYPOINTS
    # (Spokane→Lethbridge). corridor_name is informational only here.
    _run_script(THIS_DIR / "build_corridor_from_pbf.py")


# ---------------------------------------------------------------------------
# Step 2: Visualize raw OSM extract (optional debug)
# ---------------------------------------------------------------------------


def step_visualize_raw(corridor_name: str) -> None:
    _header(
        f"STEP 2: Visualize raw OSM extract → networks/{corridor_name}_local_plot.html"
    )
    _run_script(THIS_DIR / "visualize_local_network.py")


# ---------------------------------------------------------------------------
# Step 3: Build ALTRIOS network YAML
# ---------------------------------------------------------------------------


def step_build(fast: bool) -> Path:
    """Build the ALTRIOS YAML. Returns the path to the produced .yaml file."""
    _header("STEP 3: Build ALTRIOS network (mainline + sidings)")
    script = THIS_DIR / "build_altrios_with_sidings.py"
    out_yaml = NETWORKS_DIR / "spokane_lethbridge_with_sidings.yaml"

    extra = ["--fast"] if fast else []
    _run_script(script, extra)
    return out_yaml


# ---------------------------------------------------------------------------
# Step 4: Plot the built ALTRIOS geometry
# ---------------------------------------------------------------------------


def step_plot(yaml_path: Path) -> None:
    _header("STEP 4: Plot ALTRIOS path geometry")
    geom_path = yaml_path.with_name(yaml_path.stem + "_geometry.json")
    html_path = yaml_path.with_name(yaml_path.stem + "_plot.html")
    if not geom_path.exists():
        logger.error(f"  Geometry JSON not found: {geom_path}")
        logger.error("  Run the 'build' step first.")
        return
    # Import plot_json directly (the script's __main__ has hardcoded paths)
    sys.path.insert(0, str(THIS_DIR))
    try:
        from plot_altrios_path import plot_json  # type: ignore

        plot_json(str(geom_path), str(html_path))
    finally:
        if str(THIS_DIR) in sys.path:
            sys.path.remove(str(THIS_DIR))
    logger.info(f"  → {html_path}")


# ---------------------------------------------------------------------------
# Step 5: Simulate
# ---------------------------------------------------------------------------


def step_simulate(yaml_path: Path, show_plots: bool) -> None:
    _header(f"STEP 5: Simulate train on {yaml_path.name}")
    if not yaml_path.exists():
        logger.error(f"  Network YAML not found: {yaml_path}")
        logger.error("  Run the 'build' step first.")
        return
    extra = [str(yaml_path)]
    if show_plots:
        extra.append("--plots")
    _run_script(THIS_DIR / "simulate_corridor.py", extra)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_STEPS = ["download", "visualize_raw", "build", "plot", "simulate"]
DEFAULT_STEPS = ["download", "build", "plot", "simulate"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified network builder workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=None,
        help=f"Steps to run (default: {' '.join(DEFAULT_STEPS)})",
    )
    parser.add_argument(
        "--corridor-name",
        default="spokane_lethbridge",
        help="Corridor basename for output files (default: spokane_lethbridge). "
        "Used to locate the YAML for plot/simulate steps.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Pass --fast to the build script (skip elevation fetch).",
    )
    parser.add_argument(
        "--sim-plots",
        action="store_true",
        help="Show matplotlib plots during the simulate step.",
    )
    parser.add_argument(
        "--yaml-path",
        default=None,
        help="Explicit ALTRIOS YAML path for plot/simulate steps. "
        "Defaults to networks/<corridor-name>_with_sidings.yaml.",
    )

    args = parser.parse_args()

    steps: List[str] = args.steps if args.steps else list(DEFAULT_STEPS)

    # Resolve target YAML for plot/simulate
    if args.yaml_path:
        yaml_path = Path(args.yaml_path).resolve()
    else:
        yaml_path = NETWORKS_DIR / f"{args.corridor_name}_with_sidings.yaml"

    logger.info(f"Workspace root : {WORKSPACE_ROOT}")
    logger.info(f"Corridor name  : {args.corridor_name}")
    logger.info(f"Steps          : {', '.join(steps)}")
    logger.info(f"Build mode     : with sidings{' (fast)' if args.fast else ''}")
    logger.info(f"Target YAML    : {yaml_path}")

    if "download" in steps:
        step_download(args.corridor_name)
    if "visualize_raw" in steps:
        step_visualize_raw(args.corridor_name)
    if "build" in steps:
        yaml_path = step_build(args.fast)
    if "plot" in steps:
        step_plot(yaml_path)
    if "simulate" in steps:
        step_simulate(yaml_path, args.sim_plots)

    _header("Network builder workflow complete!")


if __name__ == "__main__":
    main()
