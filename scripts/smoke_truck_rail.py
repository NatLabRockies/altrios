"""Deeper inspection of the truck_rail run to compare against legacy baseline.
"""
import polars as pl
from altrios.lifts import run_terminal_simulation
from altrios.lifts import utilities

consist_plan = (
    pl.read_csv(utilities.package_root() / "resources" / "train_consist_plan.csv")
    .with_columns(pl.lit("Intermodal").alias("Train_Type"))
)

cd, vl, term = run_terminal_simulation(
    modes=["truck_rail"],
    terminal="Allouez",
    inputs={"truck_rail": {"train_consist_plan": consist_plan}},
)

# IC containers that came in on a train and were picked up by a drayage truck
ic_full = cd.filter(
    pl.col("container_id").str.starts_with("IC")
    & pl.col("train_arrival_expected").is_not_null()
    & pl.col("drayage_gate_out").is_not_null()
)
oc_full = cd.filter(
    pl.col("container_id").str.starts_with("OC")
    & pl.col("drayage_arrival").is_not_null()
    & pl.col("rail_track_rtg_load").is_not_null()
)
print(f"IC full lifecycle: {ic_full.height}")
print(f"OC full lifecycle: {oc_full.height}")

ic_total = cd.filter(pl.col("container_id").str.starts_with("IC"))
oc_total = cd.filter(pl.col("container_id").str.starts_with("OC"))
print(f"IC total rows: {ic_total.height}")
print(f"OC total rows: {oc_total.height}")

train_rows = cd.filter(pl.col("container_id").str.starts_with("Train-"))
print(f"Train rows: {train_rows.height}")
if train_rows.height > 0:
    mx = train_rows["train_depart"].max()
    mn = train_rows["train_depart"].min()
    print(f"max train_depart: {mx:.2f}")
    print(f"min train_depart: {mn:.2f}")

energy_by_resource = vl.group_by("resource_type").agg(
    pl.col("consumption_value").sum().alias("total_energy"),
).sort("resource_type")
print("\nEnergy by resource_type:")
for row in energy_by_resource.iter_rows(named=True):
    print(f"  {row['resource_type']:>22}: {row['total_energy']:10.2f}")

print(f"\ntotal energy: {vl['consumption_value'].sum():.2f}")
print(f"total emissions: {vl['emissions(kgCO2)'].sum():.2f}")
