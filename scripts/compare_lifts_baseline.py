"""Compare current `intermodal_rail` outputs against the saved baseline.

Reads the post-refactor CSVs produced by re-running
``capture_lifts_baseline.py`` and compares each row/column to the baseline
captured before the Phase 1C refactor. Exits 0 on identical outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl


BASE_DIR = Path(__file__).resolve().parents[1] / "target" / "lifts_baseline"
CURR_DIR = Path(__file__).resolve().parents[1] / "target" / "lifts_current"


def _compare(name: str) -> bool:
    base = pl.read_csv(BASE_DIR / f"{name}.csv")
    curr = pl.read_csv(CURR_DIR / f"{name}.csv")
    if base.columns != curr.columns:
        print(f"[{name}] COLUMNS DIFFER")
        print(f"  base: {base.columns}")
        print(f"  curr: {curr.columns}")
        return False
    if base.shape != curr.shape:
        print(f"[{name}] SHAPE DIFFERS base={base.shape} curr={curr.shape}")
        return False
    try:
        diff = base.equals(curr)
    except Exception as e:
        print(f"[{name}] equals() raised: {e}")
        return False
    if diff:
        print(f"[{name}] IDENTICAL ({base.shape})")
        return True
    print(f"[{name}] differs in cell values; computing first few differences:")
    for col in base.columns:
        b = base.get_column(col)
        c = curr.get_column(col)
        if b.dtype != c.dtype:
            print(f"  column {col!r} dtype: {b.dtype} vs {c.dtype}")
            continue
        try:
            mask = ~(b.eq_missing(c))
        except Exception:
            mask = b != c
        n_diff = int(mask.sum())
        if n_diff:
            idx = mask.arg_true().head(3).to_list()
            print(f"  column {col!r}: {n_diff} differences; sample idx={idx} base={b.gather(idx).to_list()} curr={c.gather(idx).to_list()}")
    return False


def main() -> int:
    ok = True
    ok &= _compare("container_data")
    ok &= _compare("vehicle_log")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
