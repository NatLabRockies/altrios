"""
Combine the per-simulation outputs written by ``run_all.sh`` into one set.

Each simulation runs in its own process and writes its own files, so that a
cell which fails -- or aborts outright, as an out-of-memory Rust allocation
does -- costs only itself. This gathers whatever succeeded.

It is safe to run at any point: partial results combine as readily as complete
ones, and the summary says which cells of the design are still missing.

Usage:
    python collect_results.py
    python collect_results.py --missing   # just list what has not run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import polars as pl

HERE = Path(__file__).parent
OUT = HERE / "out"

logger = logging.getLogger(__name__)

GROUPS = {
    "sim_metrics.csv": "*_metrics.csv",
    "stop_events.csv": "*_stops.csv",
    "consist_plans.csv": "*_plan.csv",
}


def combine(per_sim_dir: Path, out_dir: Path) -> dict:
    written = {}
    for name, pattern in GROUPS.items():
        frames: List[pl.DataFrame] = []
        for path in sorted(per_sim_dir.glob(pattern)):
            try:
                frames.append(pl.read_csv(path))
            except Exception as exc:  # noqa: BLE001 - a truncated file is not fatal
                logger.warning("skipping %s: %s", path.name, str(exc)[:120])
        if not frames:
            continue
        table = pl.concat(frames, how="diagonal_relaxed")
        table.write_csv(out_dir / name)
        written[name] = table.height
        logger.info("wrote %s (%d rows from %d files)", name, table.height, len(frames))
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--per-sim-dir", type=Path, default=OUT / "sims")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--experiments", type=Path, default=HERE / "experiments.csv")
    parser.add_argument("--missing", action="store_true", help="list gaps and stop")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    wanted = set(pl.read_csv(args.experiments)["Sim_Key"].unique().to_list())
    done = set()
    for path in args.per_sim_dir.glob("*_metrics.csv"):
        try:
            done.update(pl.read_csv(path)["Sim_Key"].unique().to_list())
        except Exception:  # noqa: BLE001
            continue
    missing = sorted(wanted - done)

    if not args.missing:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        combine(args.per_sim_dir, args.out_dir)

    logger.info("%d of %d simulations present", len(wanted) - len(missing), len(wanted))
    for key in missing:
        logger.info("  missing: %s", key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
