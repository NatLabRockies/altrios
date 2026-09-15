"""Per-VECTOR history inventory at the current settings.

Component-level totals said what to disable; this says what is left inside the
components that must stay, which is where any further saving has to come from.
"""
import sys
from collections import defaultdict
from pathlib import Path
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import altrios as alt
from altrios import sim_manager
import run_sweep as rs
from speed_calibration import CORRIDORS, NETWORKS, load_rail_vehicles

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
CORR = sys.argv[2] if len(sys.argv) > 2 else "winnipeg_chicago"
exp = pl.read_csv(Path(__file__).parent / "experiments.csv")
row = exp.filter(
    (pl.col("Corridor") == CORR) & (pl.col("Powertrain") == "diesel")
).row(0, named=True)
c = CORRIDORS[CORR]
full = pl.read_csv(c.plan_path)
keep = full["Train_ID"].unique().sort().head(N)
plan = full.filter(pl.col("Train_ID").is_in(keep.implode()))

(*_h, sims, _t, _u) = sim_manager.main(
    network=rs.network_for(row), rail_vehicles=load_rail_vehicles(),
    location_map=alt.import_locations(str(NETWORKS / c.locations)),
    consist_plan_in=plan,
    train_planner_config=rs.planner_config_for(row["Powertrain"], row["Battery_MWh"]),
    walk_timed_paths=True, save_interval=rs.SAVE_INTERVAL,
    configure_sim=rs.trim_history, debug=False,
)

vecs = defaultdict(int)


def walk(node, path):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        if node and isinstance(node[0], (int, float)) and len(node) > 8:
            vecs[path] += len(node)
        else:
            for v in node[:40]:
                walk(v, f"{path}[]")


for slts in sims.to_pydict():
    walk(slts, "")

total = sum(vecs.values())
needed = {
    "history.time_seconds", "history.speed_meters_per_second",
    "history.offset_meters",
    "loco_con.history.pwr_fuel_watts", "loco_con.history.energy_fuel_joules",
    "loco_con.loco_vec[].history.pwr_aux_watts",
}
print(f"\n{CORR} | {N} trains | interval {rs.SAVE_INTERVAL}s | "
      f"{total:,} slots (~{total*8/1e6:.0f} MB)")
print(f"{'vector':<58}{'values':>12}{'share':>8}  used?")
kept = 0
for k, v in sorted(vecs.items(), key=lambda kv: -kv[1])[:26]:
    mark = "YES" if k in needed else ""
    if k in needed:
        kept += v
    print(f"{k[-57:]:<58}{v:>12,}{100*v/total:>7.1f}%  {mark}")
print(f"\nvectors needed downstream: {kept:,} slots = {100*kept/total:.1f}% of what is saved")
