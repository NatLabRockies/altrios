"""
Map how simulated run time responds to traffic density, and correct for the fact
that real density is not uniform.

Two facts motivate this. Meet-pass delay is *convex* in density -- flat while
the corridor is below saturation, then rising steeply -- and real density varies
along a corridor: Toronto-Battle Creek carries 331 trains/week past Malport and
123 past Port Huron. Because every simulated train runs end to end, one density
has to stand for the whole corridor, and for a convex response the average of
the delays exceeds the delay at the average density. So a corridor run at its
train-km-matched average systematically under-predicts congestion.

This sweeps density across each corridor's *observed* per-point range and
records the resulting run-time error. Two products:

- **A sensitivity band.** How much of the calibration answer is the density
  assumption rather than the physics.
- **A congestion-equivalent density**: the uniform level whose delay matches
  what the real, varying density profile would produce, obtained by averaging
  the fitted curve over the observed per-point distribution.

What this deliberately does *not* do is tune density until run times match
observation. That is the 0.595 speed scaler in another costume: it would
silently absorb the speed residual, and Spokane-Lethbridge's multi-hour
discrete holds, into a congestion parameter -- and energy results would inherit
the error while looking right on the clock. The target here is the observed
density profile, never the observed travel time.

Usage:
    python density_sweep.py --corridor toronto_battle_creek
    python density_sweep.py --all --out sweep.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import polars as pl

# The input pipeline is analysis code and lives in the analysis repo, not here.
# ALTRIOS_ANALYSIS_SCRIPTS points at it on a machine that checks it out
# somewhere else.
sys.path.append(
    os.environ.get(
        "ALTRIOS_ANALYSIS_SCRIPTS",
        str(
            Path.home()
            / "Documents"
            / "Repos"
            / "Enterprise"
            / "ALTRIOS_Analysis_Code"
            / "Scripts"
        ),
    )
)

from railstate_corridor import CORRIDOR_SPECS, corridor_config
from railstate_corridor import run as build_corridor

from speed_calibration import (
    CORRIDORS,
    INPUTS,
    compare,
    link_offsets,
    prepare_network,
    simulate,
    simulated_segment_times,
)

logger = logging.getLogger(__name__)

# Per-point observed traffic, trains/week, from the corridor's sensors. The
# sweep stays inside this range: end-to-end traffic cannot exceed what the
# busiest point carries, nor be meaningfully below what the quietest does.
OBSERVED_RANGE: Dict[str, tuple] = {
    "toronto_battle_creek": (123, 331),
    "winnipeg_chicago": (147, 157),
    "spokane_lethbridge": (52, 124),
}

# Density that matches observed train-km -- the primary configuration, and
# each corridor's own default, so the sweep reads it rather than restating it.
MEASURED: Dict[str, int] = {
    slug: int(spec["weekly_count_override"])  # type: ignore[arg-type]
    for slug, spec in CORRIDOR_SPECS.items()
}

# Per-point observed rates, used to weight the curve into a
# congestion-equivalent density.
PER_POINT: Dict[str, List[float]] = {
    "toronto_battle_creek": [330.8, 155.3, 155.5, 123.0, 124.5, 138.5, 122.7],
    "winnipeg_chicago": [147.3, 156.6],
    "spokane_lethbridge": [52.4, 101.8, 106.1, 124.0, 112.8],
}


def build_plan(slug: str, n_trains: int, out_dir: Path) -> Optional[Path]:
    """Generate a consist plan at a given weekly train count."""
    config = corridor_config(slug, weekly_count_override=n_trains)
    plan = build_corridor(config)["plan"]
    path = out_dir / f"sweep_{slug}_{n_trains}.csv"
    plan.write_csv(path)
    return path


def weighted_error(corridor) -> Optional[float]:
    """Distance-weighted run-time error for one simulated corridor."""
    plan_out, timed_paths, sensors = simulate(corridor)
    offsets = link_offsets(prepare_network(corridor))
    simulated = simulated_segment_times(plan_out, timed_paths, sensors, offsets)
    observed = pl.read_csv(corridor.observed_path)
    table = compare(observed, simulated)
    if not table.height:
        return None
    return float(
        (table["pct_error"] * table["km"]).sum() / table["km"].sum()
    )


def congestion_equivalent(levels: Sequence[float], errors: Sequence[float],
                          per_point: Sequence[float]) -> Optional[float]:
    """Uniform density whose delay equals the mean delay over the real profile.

    Interpolates the measured curve at each observed per-point density, averages
    those delays, then inverts the curve. This is where the convexity is paid
    for: the answer sits above the simple average of the profile.
    """
    pts = sorted(zip(levels, errors))
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    if len(xs) < 2:
        return None
    target = float(np.mean(np.interp(np.asarray(per_point, dtype=float), xs, ys)))
    if not (ys.min() <= target <= ys.max()):
        return None
    order = np.argsort(ys)
    return float(np.interp(target, ys[order], xs[order]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--corridor", choices=sorted(CORRIDORS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--levels", type=int, default=4, help="density levels per corridor"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    slugs = sorted(CORRIDORS) if args.all else [args.corridor]
    if not slugs or slugs == [None]:
        parser.error("pass --corridor or --all")

    rows = []
    for slug in slugs:
        lo, hi = OBSERVED_RANGE[slug]
        levels = sorted(
            {int(round(v)) for v in np.linspace(lo, hi, args.levels)}
            | {MEASURED[slug]}
        )
        logger.info("=== %s: densities %s (measured %d)", slug, levels, MEASURED[slug])
        errs = []
        for n in levels:
            corridor = CORRIDORS[slug]
            corridor.plan_override = build_plan(slug, n, INPUTS)
            try:
                err = weighted_error(corridor)
            except Exception as exc:  # noqa: BLE001 - a level may be infeasible
                logger.error("  %4d trains: FAILED (%s)", n, str(exc)[:110])
                err = None
            corridor.plan_override = None
            if err is None:
                continue
            errs.append((n, err))
            logger.info("  %4d trains/wk -> %+.1f%% run-time error", n, 100 * err)
            rows.append({"corridor": slug, "trains_per_week": n, "pct_error": err,
                         "is_measured": n == MEASURED[slug]})

        if len(errs) >= 2:
            eq = congestion_equivalent(
                [e[0] for e in errs], [e[1] for e in errs], PER_POINT[slug]
            )
            if eq is None:
                logger.info("  congestion-equivalent density: outside the swept range")
            else:
                logger.info(
                    "  congestion-equivalent density: %.0f trains/wk "
                    "(measured %d, i.e. %.2fx)",
                    eq, MEASURED[slug], eq / MEASURED[slug],
                )

    if rows:
        table = pl.DataFrame(rows)
        logger.info("\n%s", table.to_pandas().to_string(index=False))
        if args.out:
            table.write_csv(args.out)
            logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
