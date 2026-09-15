"""Measure peak memory of one simulation as a function of train count.

Winnipeg-Chicago aborts on a 1 GiB allocation during the walk phase, on a
machine with 63.7 GB of RAM and ~39 GB free -- so the question is not whether
memory is scarce but what is consuming it and how it scales. Its dispatch is
fine: the calibration walks no trains and completes all 162.
"""
import argparse, gc, sys
from pathlib import Path
import polars as pl, psutil

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import run_sweep as rs
from speed_calibration import CORRIDORS

p = argparse.ArgumentParser()
p.add_argument("--corridor", default="winnipeg_chicago")
p.add_argument("--key")
p.add_argument("--trains", type=int, nargs="+", default=[20, 40, 80])
a = p.parse_args()

exp = pl.read_csv(Path(__file__).parent / "experiments.csv")
key = a.key or exp.filter(
    (pl.col("Corridor") == a.corridor) & (pl.col("Powertrain") == "diesel")
)["Sim_Key"][0]
row = exp.filter(pl.col("Sim_Key") == key).row(0, named=True)
full = pl.read_csv(CORRIDORS[a.corridor].plan_path)
proc = psutil.Process()

print(f"{'trains':>7} {'peak_MB':>9} {'delta_MB':>9} {'MB/train':>9}")
base = proc.memory_info().rss / 1e6
prev_peak = base
for n in a.trains:
    keep = full["Train_ID"].unique().sort().head(n)
    plan = full.filter(pl.col("Train_ID").is_in(keep.implode()))
    try:
        rs.run_simulation(row, plan)
    except Exception as exc:
        print(f"{n:>7} FAILED {str(exc)[:70]}")
        break
    peak = proc.memory_info().peak_wset / 1e6
    print(f"{n:>7} {peak:9.0f} {peak - base:9.0f} {(peak - base) / n:9.1f}")
    gc.collect()
