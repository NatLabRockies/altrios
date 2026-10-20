"""Run the whole design of experiments, end to end, in one sequential pass.

Everything the study needs, in dependency order, one stage at a time. Stages run
SEQUENTIALLY and never in parallel: several of them saturate memory on their own
(Winnipeg-Chicago is the largest walked plan in the design) and the catenary
optimizer already runs its MILP across several cores, so overlapping them trades
a little wall-clock for a risk of an out-of-memory abort that loses hours of
completed work.

The stages
----------
1. ``build_layouts``    Re-optimize a catenary buildout per architecture x cost
                        cell against the CURRENT consist plans, and save each as
                        a simulation-ready network. Must run first: the DoE's
                        catenary cells are keyed by layout fingerprint, and the
                        existing layouts were optimized against the previous,
                        larger plans.
2. ``build_experiments`` Regenerate ``experiments.csv`` so its fingerprints match
                        the layouts just built.
3. ``run_sweep``        Every distinct simulation in the design -- three
                        corridors x (diesel, hybrid x4 battery sizes, and on
                        Spokane the catenary dual-mode and battery-electric
                        cells). Each runs in its own process (see ``run_all``).
4. ``rollout_staged``   The staged-adoption counterpart of every rollout cell: a
                        linear 15-year ramp, blended against the diesel baseline
                        year by year and re-discounted. A sensitivity, not the
                        headline, and nothing downstream depends on it.
4. ``rollout_costs``    Multi-year discounted costs for every cell -- fleet,
                        catenary infrastructure and 30 years of energy -- from
                        ``altrios.metric_calculator``'s own DCF over the walked
                        sims ``run_sweep`` persisted. Re-runnable and re-priceable
                        without re-simulating; the notebook reads its output.
5. ``collect``          Combine per-simulation outputs into the analysis tables
                        (``sim_metrics.csv``, ``stop_events.csv``).
6. ``postprocess``      Expand those cells across cost scenarios and anti-idling
                        levels, both dwell scopes, into ``results.csv``. Runs
                        AFTER ``collect``, which is the order the data flows:
                        `postprocess_antiidling` READS the two tables `collect`
                        writes. With the stages the other way round it silently
                        analysed the PREVIOUS run's outputs -- `results.csv`
                        reported 1 simulation as `ok` and 17 as `NOT RUN` while
                        18 sat finished on disk.
7. ``catenary_deepdive`` The catenary sub-experiment: the placement optimization
                        swept over architecture x cost cell x DEMAND LEVEL
                        (1x / 2x / 4x, as distinct consist-plan samples). It runs
                        LAST because it is the most expensive stage and the least
                        depended-upon -- nothing else reads its outputs, so a
                        failure here still leaves the main DoE complete.

Rerunnable. ``run_all`` skips simulation cells that already have output, so
re-invoking after an interruption picks up where it stopped rather than repeating
finished cells; ``--fresh`` forces them all to re-run. Use ``--from`` to resume at
a named stage and ``--only`` to run one.

Usage:
    python run_overnight.py                  # everything, in order
    python run_overnight.py --dry-run        # print the plan and stop
    python run_overnight.py --from run_sweep # resume from a stage
    python run_overnight.py --only collect
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

HERE = Path(__file__).parent
CATENARY = HERE.parent / "catenary_optimization"
LOGS = HERE / "out" / "logs"

# (name, working directory, argv). `sys.executable` is substituted for "python"
# so every stage runs under the same interpreter as this driver.
STAGES = [
    ("build_layouts", HERE, ["python", "build_layouts.py"]),
    ("build_experiments", HERE, ["python", "build_experiments.py"]),
    ("run_sweep", HERE, ["python", "run_all.py"]),
    ("rollout_costs", HERE, ["python", "rollout_costs.py"]),
    # Staged adoption, as a sensitivity beside the instant-conversion headline.
    # Reads `rollout_costs.csv` and blends each cell against its diesel baseline
    # year by year, so it must follow `rollout_costs` and nothing follows it.
    ("rollout_staged", HERE, ["python", "postprocess_staged.py"]),
    ("collect", HERE, ["python", "collect_results.py"]),
    ("postprocess", HERE, ["python", "postprocess_antiidling.py"]),
    ("catenary_deepdive", CATENARY,
     ["python", "energization_optimization_corridor.py"]),
]

# The catenary deep-dive resumes into the newest unfinished sweep directory by
# default, which would mix rows written under an earlier consist plan with rows
# written under this one. Overnight runs always start clean.
#
# It also has to run on the SAME speed rule as the rest of the DoE. Its own
# defaults are a 40 mph cap then a 0.595 scaler -- 23.8 mph effective -- fitted
# on the 77 km Jaffray-Sparwood segment back when dispatch bypassed meet-pass
# entirely, so the scaler was standing in for congestion that was not modelled.
# `build_layouts.py` already overrides both for the layouts this study is
# compared against (`min(posted, observed p99)`, no scaler), and leaving the
# deep-dive on its defaults would optimize and simulate at a different line
# speed than the layouts and metrics beside it. The observed free-running speed
# on this corridor -- 51-55 kph on trips that met no opposing train -- is 32-34
# mph, which is the cap, so the cap is the better-supported number besides.
STAGE_ENV = {
    "catenary_deepdive": {
        "ENERGIZATION_FRESH_SWEEP": "1",
        "ALTRIOS_SPEED_CAP_MPH": "33.983779",   # corridor_meta p99, Spokane
        "ALTRIOS_SPEED_SCALER": "none",
    }
}


def _run(name: str, cwd: Path, argv: Sequence[str], log_dir: Path) -> int:
    """Run one stage, streaming to its own log, and return its exit code."""
    argv = [sys.executable if a == "python" else a for a in argv]
    log_path = log_dir / f"{name}.log"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **STAGE_ENV.get(name, {})}
    started = time.perf_counter()
    print(f"[{datetime.now():%H:%M:%S}] === {name} ===", flush=True)
    print(f"    {' '.join(Path(a).name if a.endswith('.py') else a for a in argv)}"
          f"  (log: {log_path})", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(argv, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                env=env)
    elapsed = time.perf_counter() - started
    status = "ok" if result.returncode == 0 else f"FAILED ({result.returncode})"
    print(f"[{datetime.now():%H:%M:%S}]     {name} {status} in {elapsed / 60:.1f} min",
          flush=True)
    if result.returncode != 0:
        # Surface the tail of the failing log here so the console transcript is
        # enough to see what broke without opening the file.
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in tail[-15:]:
                print(f"      | {line}", flush=True)
        except OSError:
            pass
    return result.returncode


def main(argv: Optional[Sequence[str]] = None) -> int:
    names = [s[0] for s in STAGES]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="start", choices=names,
                        help="resume at this stage")
    parser.add_argument("--only", choices=names, help="run just this stage")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-going", action="store_true",
        help="continue to later stages after a failure (default: stop, since "
             "each stage consumes the previous one's outputs)",
    )
    parser.add_argument("--fresh", action="store_true",
                        help="re-run simulation cells that already have output")
    args = parser.parse_args(argv)

    stages = list(STAGES)
    if args.only:
        stages = [s for s in stages if s[0] == args.only]
    elif args.start:
        stages = stages[names.index(args.start):]

    # `run_all.py` already skips cells that have output; `--fresh` is its opt-out
    # (it forwards the decision to each child `run_sweep.py`). Passing
    # `--skip-existing` here instead was an argparse error, not a no-op.
    stages = [
        (n, cwd, list(a) + (["--fresh"] if args.fresh else []))
        if n == "run_sweep" else (n, cwd, a)
        for n, cwd, a in stages
    ]

    print(f"{len(stages)} stage(s): {', '.join(n for n, _, _ in stages)}", flush=True)
    if args.dry_run:
        for n, cwd, a in stages:
            print(f"  {n}: (cd {cwd}) {' '.join(a)}", flush=True)
        return 0

    LOGS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    failures: List[str] = []
    for name, cwd, stage_argv in stages:
        code = _run(name, cwd, stage_argv, LOGS)
        if code != 0:
            failures.append(name)
            if not args.keep_going:
                print(f"\nstopping at {name}; later stages depend on it. "
                      f"Fix and resume with --from {name}", flush=True)
                break

    total = (time.perf_counter() - t0) / 60
    print(f"\n[{datetime.now():%H:%M:%S}] {len(stages) - len(failures)}"
          f"/{len(stages)} stage(s) ok in {total:.1f} min", flush=True)
    for name in failures:
        print(f"  FAILED: {name}  (log: {LOGS / f'{name}.log'})", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
