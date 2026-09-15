"""
Run every simulation in the design, each in its own process.

Two reasons this exists rather than a single in-process loop or a shell script.

**Process isolation.** ALTRIOS keeps the est-times DAG and the walked train
histories in Rust memory. Running several corridors back to back in one process
accumulates enough of it that a later allocation fails -- and a failed Rust
allocation *aborts* rather than raising, so an in-process loop loses every
result after the failure rather than just that cell. One process per simulation
lets the OS reclaim everything in between, and a cell that still cannot fit
costs only itself.

**Python rather than bash.** The shell version of this passed simulation keys
through a command substitution, and ``print`` emits CRLF on Windows: every key
but the last arrived with a trailing carriage return, matched no ``Sim_Key``,
and its process exited immediately having done nothing. ``subprocess`` with an
argument list has no quoting or line-ending layer to get wrong.

Rerunnable: ``--skip-existing`` is passed to each child, so a second invocation
picks up wherever the first stopped. Results land in ``out/sims/`` and are
combined by ``collect_results.py``.

Usage:
    python run_all.py
    python run_all.py --corridor winnipeg_chicago
    python run_all.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import polars as pl

HERE = Path(__file__).parent
POWERTRAIN_ORDER = {"diesel": 0, "hybrid": 1, "catenary_dual": 2, "catenary_bel": 3}

# Winnipeg-Chicago is run last because it is the one corridor that does not fit
# in memory: 162 trains over 1,237 km exhausted the heap after 3.6 hours even
# with the process to itself, aborting on a 1 GB allocation. Its cells are not
# excluded -- they may succeed on a machine with more memory, and the failure
# is worth reproducing -- but they run after everything else, so a corridor
# that cannot complete cannot starve the ones that can.
CORRIDOR_ORDER = {"spokane_lethbridge": 0, "toronto_battle_creek": 1,
                  "winnipeg_chicago": 2}


def simulation_keys(
    experiments: Path,
    corridor: Optional[str] = None,
    skip: Sequence[str] = (),
) -> List[str]:
    """Every distinct simulation, baselines first.

    Ordered so that a sweep which runs out of time still holds the comparisons
    the study turns on: the three diesel baselines, then hybrids by battery
    size, then the catenary cells.
    """
    table = pl.read_csv(experiments).unique(subset=["Sim_Key"], keep="first")
    if corridor:
        table = table.filter(pl.col("Corridor") == corridor)
    if skip:
        table = table.filter(~pl.col("Corridor").is_in(list(skip)))
    return (
        table.with_columns(
            pl.col("Powertrain")
            .replace_strict(POWERTRAIN_ORDER, default=9)
            .alias("_order"),
            pl.col("Corridor")
            .replace_strict(CORRIDOR_ORDER, default=9)
            .alias("_corridor"),
        )
        .sort("_corridor", "_order", "Battery_MWh", "Architecture", nulls_last=False)
        .get_column("Sim_Key")
        .to_list()
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--experiments", type=Path, default=HERE / "experiments.csv")
    parser.add_argument("--corridor")
    parser.add_argument(
        "--skip-corridor", action="append", default=[],
        help="omit a corridor entirely; repeatable",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fresh", action="store_true", help="re-run cells that already have output"
    )
    args = parser.parse_args(argv)

    keys = simulation_keys(args.experiments, args.corridor, args.skip_corridor)
    print(f"{len(keys)} simulations", flush=True)
    if args.dry_run:
        for key in keys:
            print(f"  {key}", flush=True)
        return 0

    done, failed = [], []
    for i, key in enumerate(keys, start=1):
        started = time.perf_counter()
        print(f"[{time.strftime('%H:%M:%S')}] ({i}/{len(keys)}) {key}", flush=True)
        cmd = [args.python, str(HERE / "run_sweep.py"), "--all", "--sim-key", key]
        if not args.fresh:
            cmd.append("--skip-existing")
        result = subprocess.run(
            cmd, cwd=HERE, capture_output=True, text=True, errors="replace"
        )
        # A Rust OOM abort produces no Python traceback and a non-zero return
        # code, so the return code is what decides success, not the log text.
        interesting = [
            line
            for line in (result.stdout + result.stderr).splitlines()
            if any(
                token in line
                for token in ("done in", "FAILED", "memory allocation", "Error", "error:")
            )
        ]
        for line in interesting[-4:]:
            print(f"    {line.strip()}", flush=True)
        if result.returncode == 0:
            done.append(key)
        else:
            detail = next(
                (ln.strip() for ln in reversed(interesting)
                 if "FAILED" in ln or "memory allocation" in ln),
                "",
            )
            failed.append((key, result.returncode, detail))
            print(
                f"    exited {result.returncode} after "
                f"{time.perf_counter() - started:.0f} s",
                flush=True,
            )

    print(f"\n{len(done)} succeeded, {len(failed)} failed", flush=True)
    for key, code, detail in failed:
        print(f"  FAILED ({code}): {key}  {detail}", flush=True)

    # Persist failures. A cell that cannot be simulated is a RESULT, not an
    # absence -- a battery-electric locomotive on a corridor the optimizer
    # declined to electrify genuinely cannot move -- and the analysis table has
    # to say so. Without this, an impossible configuration and one nobody has
    # run yet look identical: both are simply missing rows.
    if failed:
        out = HERE / "out" / "failures.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({
            "Sim_Key": [f[0] for f in failed],
            "return_code": [f[1] for f in failed],
            "detail": [f[2] for f in failed],
        }).write_csv(out)
        print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
