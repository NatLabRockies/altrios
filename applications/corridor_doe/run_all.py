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

**Concurrency.** Cells run SEVERAL at a time (``--jobs``). They are
independent -- each is its own process with its own consist plan and network
-- and the dominant cost, ``make_est_times``, is single-threaded, so a
sequential sweep left most of the machine idle. What bounds the parallelism
is memory rather than cores, so the scheduler also caps how many cells of one
corridor run together and refuses to start a cell that would take free memory
below a floor. Cells are ordered heaviest-corridor-first, after the diesel
baselines, since the longest cell decides when the sweep ends.

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
from typing import List, Optional, Sequence, Tuple

import polars as pl

try:  # optional: without it the memory guard falls back to the job cap alone
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

HERE = Path(__file__).parent
POWERTRAIN_ORDER = {"diesel": 0, "hybrid": 1, "catenary_dual": 2, "catenary_bel": 3}

# HEAVIEST corridor first, which is the opposite of what a sequential sweep
# wanted. Winnipeg-Chicago used to run last so a corridor that might exhaust
# memory could not starve the ones that can finish. Running cells
# concurrently removes that risk -- the others are already in flight -- and
# inverts the scheduling problem: the longest cell decides when the sweep
# ends, so it has to start first or everything waits on its tail.
#
# The diesel baselines still jump the queue regardless of corridor (see
# `_baseline` below). They are the cheapest cells and the comparison every
# other cell is read against, so a sweep cut short still holds them.
CORRIDOR_ORDER = {"winnipeg_chicago": 0, "toronto_battle_creek": 1,
                  "spokane_lethbridge": 2}

# Rough peak resident memory of ONE cell, by corridor, in GB. Winnipeg-Chicago
# is the corridor that has actually aborted on allocation, at ~10 GB observed;
# the reserve is set above the observation because an abort loses the cell
# outright and costs hours, while over-reserving only leaves a core idle.
# These gate STARTS only -- a live reading of free memory (psutil) is the
# real guard, and this exists because a process that started seconds ago has
# not grown into its footprint yet, so free memory alone would happily start
# several heavy cells at once.
RESERVE_GB = {"winnipeg_chicago": 18.0, "toronto_battle_creek": 9.0,
              "spokane_lethbridge": 5.0}
DEFAULT_RESERVE_GB = 9.0
# How long to let a newly started cell grow before judging free memory again.
RAMP_SECONDS = 60.0
POLL_SECONDS = 5.0


def simulation_keys(
    experiments: Path,
    corridor: Optional[str] = None,
    skip: Sequence[str] = (),
    powertrains: Sequence[str] = (),
    skip_powertrains: Sequence[str] = (),
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
    # Powertrain filters exist so the cells that depend on a rebuilt catenary
    # layout can be deferred without holding up the ones that do not: diesel and
    # hybrid need only the consist plan, while the catenary cells need
    # `build_layouts.py` to have re-optimized against the current plan first.
    if powertrains:
        table = table.filter(pl.col("Powertrain").is_in(list(powertrains)))
    if skip_powertrains:
        table = table.filter(~pl.col("Powertrain").is_in(list(skip_powertrains)))
    return (
        table.with_columns(
            pl.col("Powertrain")
            .replace_strict(POWERTRAIN_ORDER, default=9)
            .alias("_order"),
            pl.col("Corridor")
            .replace_strict(CORRIDOR_ORDER, default=9)
            .alias("_corridor"),
            # Baselines first, everything else after; see CORRIDOR_ORDER.
            (pl.col("Powertrain") != pl.lit("diesel")).cast(pl.Int8)
            .alias("_baseline"),
        )
        .sort("_baseline", "_corridor", "_order", "Battery_MWh", "Architecture", nulls_last=False)
        .get_column("Sim_Key")
        .to_list()
    )


def _corridor_of(key: str) -> str:
    """The corridor a simulation key belongs to."""
    return key.split("|", 1)[0]


def _available_gb():
    """Free physical memory in GB, or None when psutil is unavailable."""
    if psutil is None:
        return None
    return psutil.virtual_memory().available / 1e9


def _cell_report(text: str) -> List[str]:
    """The lines worth putting on the console for one finished cell."""
    tokens = ("done in", "FAILED", "memory allocation", "Error", "error:")
    return [ln for ln in text.splitlines() if any(t in ln for t in tokens)]


def run_cells(keys: Sequence[str], args) -> Tuple[List[str], List[tuple]]:
    """Run every cell, up to `args.jobs` at once, and return what happened.

    Concurrency is bounded three ways, because the binding constraint here is
    memory rather than cores: the job cap, a per-corridor cap so the heaviest
    corridor cannot fill every slot with its own cells, and a live free-memory
    floor. A cell that does not fit is SKIPPED rather than waited for, so one
    heavy cell at the head of the queue cannot block lighter ones behind it.

    The first cell always starts, whatever the floor says. Otherwise a machine
    merely short of memory would run nothing at all and report success, which is
    worse than trying a cell and having it fail.

    Each child writes straight to its own log file rather than through a pipe:
    with several running, pipes fill and children block on write, which
    deadlocks the sweep in a way that looks exactly like a slow simulation.
    """
    log_dir = HERE / "out" / "cell_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(keys)
    running: List[dict] = []
    done: List[str] = []
    failed: List[tuple] = []
    total = len(keys)
    started_n = 0
    last_start = 0.0

    while pending or running:
        free = _available_gb()
        while pending and len(running) < args.jobs:
            if running and time.perf_counter() - last_start < RAMP_SECONDS:
                break  # let the last start grow before judging memory again
            chosen = None
            for candidate in pending:
                corridor = _corridor_of(candidate)
                same = sum(1 for r in running if r["corridor"] == corridor)
                if same >= args.jobs_per_corridor:
                    continue
                reserve = RESERVE_GB.get(corridor, DEFAULT_RESERVE_GB)
                if running and free is not None:
                    if free - reserve < args.memory_floor_gb:
                        continue
                chosen = candidate
                break
            if chosen is None:
                break
            pending.remove(chosen)
            safe = chosen.replace("|", "__").replace(".", "p")
            log_path = log_dir / (safe + ".log")
            handle = open(log_path, "w", encoding="utf-8", errors="replace")
            cmd = [args.python, str(HERE / "run_sweep.py"), "--all",
                   "--sim-key", chosen]
            if not args.fresh:
                cmd.append("--skip-existing")
            proc = subprocess.Popen(cmd, cwd=HERE, stdout=handle,
                                    stderr=subprocess.STDOUT)
            started_n += 1
            last_start = time.perf_counter()
            running.append({"proc": proc, "key": chosen, "log": log_path,
                            "handle": handle, "started": time.perf_counter(),
                            "corridor": _corridor_of(chosen)})
            note = "" if free is None else ", {:.0f} GB free".format(free)
            print("[{}] start ({}/{}) {} | {} running, {} queued{}".format(
                time.strftime("%H:%M:%S"), started_n, total, chosen,
                len(running), len(pending), note), flush=True)
            free = _available_gb()

        if not running:
            continue
        time.sleep(POLL_SECONDS)
        for r in list(running):
            code = r["proc"].poll()
            if code is None:
                continue
            r["handle"].close()
            running.remove(r)
            elapsed = (time.perf_counter() - r["started"]) / 60.0
            text = r["log"].read_text(encoding="utf-8", errors="replace")
            interesting = _cell_report(text)
            status = "done" if code == 0 else "EXIT {}".format(code)
            print("[{}] {} {} after {:.1f} min".format(
                time.strftime("%H:%M:%S"), status, r["key"], elapsed), flush=True)
            for line in interesting[-3:]:
                print("    " + line.strip(), flush=True)
            if code == 0:
                done.append(r["key"])
            else:
                detail = next(
                    (ln.strip() for ln in reversed(interesting)
                     if "FAILED" in ln or "memory allocation" in ln),
                    "",
                )
                failed.append((r["key"], code, detail))
    return done, failed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--experiments", type=Path, default=HERE / "experiments.csv")
    parser.add_argument("--corridor")
    parser.add_argument(
        "--skip-corridor", action="append", default=[],
        help="omit a corridor entirely; repeatable",
    )
    parser.add_argument(
        "--powertrain", action="append", default=[],
        help="run only these powertrains; repeatable",
    )
    parser.add_argument(
        "--skip-powertrain", action="append", default=[],
        help="omit a powertrain entirely; repeatable",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--jobs", type=int, default=6,
        help="cells to run at once; memory, not cores, is the binding "
        "constraint (default 6)",
    )
    parser.add_argument(
        "--jobs-per-corridor", type=int, default=2,
        help="cap on concurrent cells from ONE corridor, so the heaviest "
        "corridor cannot fill every slot (default 2)",
    )
    parser.add_argument(
        "--memory-floor-gb", type=float, default=10.0,
        help="keep at least this much physical memory free when starting a "
        "new cell (default 10)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fresh", action="store_true", help="re-run cells that already have output"
    )
    args = parser.parse_args(argv)

    keys = simulation_keys(args.experiments, args.corridor, args.skip_corridor,
                           args.powertrain, args.skip_powertrain)
    print(f"{len(keys)} simulations", flush=True)
    if args.dry_run:
        for key in keys:
            print(f"  {key}", flush=True)
        return 0

    done, failed = run_cells(keys, args)

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
