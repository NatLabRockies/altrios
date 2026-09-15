"""Which saved-history vectors actually carry the memory.

Option 4 -- save fewer variables -- is only worth the trouble if a few vectors
dominate. This walks a small Winnipeg case and measures every history vector in
the returned sims, grouped by the component it hangs off, so the answer is
measured rather than assumed.
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
exp = pl.read_csv(Path(__file__).parent / "experiments.csv")
row = exp.filter(
    (pl.col("Corridor") == "winnipeg_chicago") & (pl.col("Powertrain") == "diesel")
).row(0, named=True)
corridor = CORRIDORS["winnipeg_chicago"]
full = pl.read_csv(corridor.plan_path)
keep = full["Train_ID"].unique().sort().head(N)
plan = full.filter(pl.col("Train_ID").is_in(keep.implode()))

(*_head, sims, _tp, _un) = sim_manager.main(
    network=rs.network_for(row),
    rail_vehicles=load_rail_vehicles(),
    location_map=alt.import_locations(str(NETWORKS / corridor.locations)),
    consist_plan_in=plan,
    train_planner_config=rs.planner_config_for(row["Powertrain"], row["Battery_MWh"]),
    walk_timed_paths=True,
    save_interval=(60 if len(sys.argv) > 2 and sys.argv[2] == "baseline"
                   else rs.SAVE_INTERVAL),
    configure_sim=(None if len(sys.argv) > 2 and sys.argv[2] == "baseline"
                   else rs.trim_history),
    debug=False,
)

by_component = defaultdict(lambda: [0, 0])


def walk(node, path):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        if node and isinstance(node[0], (int, float)) and len(node) > 8:
            comp = path.rsplit(".history.", 1)[0] if ".history." in path else path
            by_component[comp][0] += len(node)
            by_component[comp][1] += 1
        else:
            for v in node[:40]:
                walk(v, f"{path}[]")


for slts in sims.to_pydict():
    walk(slts, "")

rows = sorted(by_component.items(), key=lambda kv: -kv[1][0])
total = sum(v[0] for v in by_component.values())
print(f"\n{N} trains | {total:,} float slots (~{total * 8 / 1e6:,.0f} MB at 8 B/slot)")
print(f"{'component':<50}{'vectors':>9}{'values':>13}{'share':>8}")
for comp, (n_vals, n_vecs) in rows[:16]:
    print(f"{comp[-60:]:<62}{n_vecs:>9,}{n_vals:>13,}{100*n_vals/total:>7.1f}%")
