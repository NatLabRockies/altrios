"""Find each corridor's highest usable traffic level.

Two criteria, per the sizing rule:

1. **Steady state.** A simulated week is usable only if trip times level off
   across departure days. Above capacity they do not: on Spokane-Lethbridge at
   88 trains/week the median trip grew 27.8 -> 87.5 h from the first measured
   day to the last, still climbing at the end. That is an unbounded queue, and
   neither its travel times nor its dwell mean anything.

2. **Not slower than observation.** A level can be technically stable and still
   absurd -- trains idling for meets far longer than any real train does. So the
   level is also capped where simulated time over the observed sensor span
   reaches the observed median. In practice this binds only where the model can
   get that slow before going unstable.

The level chosen is the highest that satisfies both. Runs are diesel only: the
question is corridor capacity, which the powertrain does not change, and diesel
is the cheapest cell.

A point that exceeds `--timeout` is recorded as UNUSABLE rather than unknown.
That is deliberate and not a measurement artifact: a level that cannot be
simulated in bounded time cannot be used for the study either way.

Usage:
    python stable_level_search.py --corridor spokane_lethbridge --levels 52,65,78,91
    python stable_level_search.py --all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import polars as pl

HERE = Path(__file__).parent
INPUTS = HERE.parent / "corridor_inputs"
OUT = HERE / "out" / "stable_search"

# Plans to draw subsets from. Oversized where the corridor's own plan is too
# small to probe ABOVE its current level; the corridor's own plan otherwise.
SOURCE_PLAN = {
    "spokane_lethbridge": "big_train_consist_plan_spokane_lethbridge_clean.csv",
    "toronto_battle_creek": "big_train_consist_plan_toronto_battle_creek_clean.csv",
    "winnipeg_chicago": "train_consist_plan_winnipeg_chicago_clean.csv",
}

# Ladders start from a level already known to settle and climb. Spokane is
# bracketed 65 stable / 75 diverging from the earlier probe; Winnipeg is known
# to settle at 64 and to be unusable at 161, a wide bracket that wants bisecting;
# Toronto settles at 128 and its headroom is unmeasured.
DEFAULT_LEVELS = {
    "spokane_lethbridge": [52, 65, 78, 91, 104],
    "toronto_battle_creek": [128, 144, 160, 176],
    "winnipeg_chicago": [64, 96, 128],
}


def run_point(slug: str, n: int, timeout_s: int) -> Dict[str, object]:
    """Simulate one traffic level and report stability and span time."""
    tag = f"{slug}__n{n}"
    out_dir = OUT / tag
    cmd = [
        sys.executable, str(HERE / "probe_level.py"),
        "--corridor", slug, "--trains", str(n),
        "--plan", str(INPUTS / SOURCE_PLAN[slug]),
        "--out", str(out_dir),
    ]
    started = time.perf_counter()
    try:
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                           errors="replace", timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"corridor": slug, "trains": n, "status": "TIMEOUT",
                "seconds": timeout_s}
    elapsed = time.perf_counter() - started
    log = out_dir / "probe.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(r.stdout + r.stderr, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        return {"corridor": slug, "trains": n, "status": "FAILED",
                "seconds": round(elapsed, 1)}
    result = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    result.update({"status": "ok", "seconds": round(elapsed, 1)})
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corridor", choices=sorted(DEFAULT_LEVELS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--levels", help="comma-separated train counts")
    parser.add_argument("--timeout", type=int, default=2400,
                        help="seconds per point before it counts as unusable")
    args = parser.parse_args(argv)

    slugs = sorted(DEFAULT_LEVELS) if args.all else [args.corridor]
    if not slugs or slugs == [None]:
        parser.error("pass --corridor or --all")

    OUT.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for slug in slugs:
        levels = ([int(x) for x in args.levels.split(",")] if args.levels
                  else DEFAULT_LEVELS[slug])
        for n in levels:
            print(f"[{time.strftime('%H:%M:%S')}] {slug} @ {n} trains ...",
                  flush=True)
            row = run_point(slug, n, args.timeout)
            # `by_day` is a list; flatten it so the table stays CSV-writable
            # while keeping the per-day trace visible, which is the evidence
            # the stability call rests on.
            if isinstance(row.get("by_day"), list):
                row["by_day"] = " ".join(str(v) for v in row["by_day"])
            rows.append(row)
            print(f"    {row}", flush=True)
            pl.DataFrame(rows, infer_schema_length=None).write_csv(
                OUT / "stable_search.csv")
            # Climbing past an unusable level tells us nothing and costs the most.
            if row.get("status") != "ok" or row.get("diverging"):
                print(f"    stopping {slug}: level {n} is not usable", flush=True)
                break
    print("\n" + str(pl.DataFrame(rows, infer_schema_length=None)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
