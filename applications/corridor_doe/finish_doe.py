"""Finish the DoE once the catenary layouts have been rebuilt.

Not part of `run_overnight.py` because this is a REPAIR path, not the design's
normal order. The nine diesel and hybrid cells are already simulated and valid;
what is stale is everything downstream of the catenary layouts, which were
never rebuilt after the consist plans changed (`build_layouts` failed every
cell and still exited 0). So this waits for the rebuild, re-keys the experiment
table to the new buildouts, drops results that no longer belong to any cell,
re-runs only what the content fingerprint says has changed, and rebuilds the
analysis tables.

Runs alongside the catenary deep-dive, which shares nothing with it: the
deep-dive optimizes its own buildouts from its own demand levels and writes to
its own results directory.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import polars as pl

try:
    import psutil
except ImportError:
    psutil = None

HERE = Path(__file__).parent
LOGS = HERE / "out" / "logs"
SIMS = HERE / "out" / "sims"


def _running(needle: str) -> bool:
    """Is a python process running this script?"""
    if psutil is None:
        return False
    for proc in psutil.process_iter(["cmdline"]):
        try:
            line = " ".join(proc.info["cmdline"] or ())
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            continue
        if needle in line and "finish_doe" not in line:
            return True
    return False


def wait_for(needle: str, label: str) -> None:
    """Block until no process is running `needle`."""
    announced = False
    while _running(needle):
        if not announced:
            print(f"[{datetime.now():%H:%M:%S}] waiting for {label} ...", flush=True)
            announced = True
        time.sleep(20)
    print(f"[{datetime.now():%H:%M:%S}] {label} is not running; continuing",
          flush=True)


def stage(name: str, argv) -> int:
    """Run one stage to its own log and return its exit code."""
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{name}.log"
    argv = [sys.executable if a == "python" else a for a in argv]
    started = time.perf_counter()
    print(f"[{datetime.now():%H:%M:%S}] === {name} ===  (log: {log_path})",
          flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(argv, cwd=HERE, stdout=log,
                                stderr=subprocess.STDOUT)
    mins = (time.perf_counter() - started) / 60
    status = "ok" if result.returncode == 0 else f"FAILED ({result.returncode})"
    print(f"[{datetime.now():%H:%M:%S}]     {name} {status} in {mins:.1f} min",
          flush=True)
    if result.returncode != 0:
        for line in log_path.read_text(encoding="utf-8",
                                       errors="replace").splitlines()[-12:]:
            print(f"      | {line}", flush=True)
    return result.returncode


def prune_orphans() -> int:
    """Delete per-simulation output for cells the design no longer contains.

    Rebuilt buildouts get new fingerprints, so their simulation keys change and
    the old files are left behind with nothing pointing at them. `collect` globs
    the directory, so an orphan is not inert -- it is a row in the analysis
    table describing a buildout that no longer exists.
    """
    keys = set(
        pl.read_csv(HERE / "experiments.csv")["Sim_Key"].unique().to_list()
    )
    safe = {k.replace("|", "__").replace(".", "p") for k in keys}
    removed = 0
    for path in sorted(SIMS.glob("*")):
        stem = path.stem
        # Every per-cell artifact, including the sims bundle and its sidecars;
        # a suffix missing here makes that file look like an orphan and deletes it.
        for suffix in ("_metrics", "_plan", "_stops", "_inputs", "_sims",
                       "_loco_pool", "_sidecar"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        if stem not in safe:
            path.unlink()
            removed += 1
    print(f"[{datetime.now():%H:%M:%S}] pruned {removed} orphaned output file(s)",
          flush=True)
    return removed


def main() -> int:
    wait_for("build_layouts.py", "layout rebuild")

    if stage("build_experiments", ["python", "build_experiments.py"]):
        return 1
    prune_orphans()
    # No `--fresh`: the content fingerprint decides. The diesel and hybrid cells
    # have unchanged inputs and are skipped; every catenary cell keys on a
    # rebuilt buildout and re-runs.
    if stage("run_sweep", ["python", "run_all.py"]):
        return 1
    # `collect` first: `postprocess_antiidling` reads the tables it writes.
    # Multi-year discounted costs, from the metrics calculator's own DCF over
    # the saved sims. Independent of collect/postprocess; the notebook reads it.
    if stage("rollout_costs", ["python", "rollout_costs.py"]):
        return 1
    if stage("collect", ["python", "collect_results.py"]):
        return 1
    if stage("postprocess", ["python", "postprocess_antiidling.py"]):
        return 1
    print(f"\n[{datetime.now():%H:%M:%S}] DoE tables rebuilt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
