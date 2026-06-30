"""LIFTS Phase 1 demo: ``truck_rail`` mode.

A truck<->rail intermodal terminal where containers exchange between the
rail tracks (via rail-track RTGs) and the gate (via drayage trucks), with
all containers routing through the main container stack.

This demo reads the bundled ``train_consist_plan.csv`` and runs the
rebuilt ``truck_rail`` mode end-to-end. No explicit drayage schedule is
passed in: ``truck_rail`` synthesizes one from the train arrivals so the
stack always has enough OCs for the trains and enough ICs for the
pickup trucks (~1 drayage truck per container).

To exercise the *explicit* drayage path instead, pass an
``extra_inputs={"drayage_schedule": <DataFrame>}`` of shape
(``Terminal_ID``, ``Truck_ID``, ``Arrival_Time_Hr``, ``Action``,
``Container_ID``); see ``utilities.build_drayage_schedule`` and the
bundled ``drayage_schedule.csv`` for the format. With Phase 1's single
flat ``container_stack``, the explicit schedule must supply enough
drayage volume to cover the trains' OC demand or some trains will not
finish loading before the sim ends.

Run from the repo root with::

    python -m altrios.lifts.demos.truck_rail_demo

or directly with::

    python python/altrios/lifts/demos/truck_rail_demo.py
"""
from __future__ import annotations

import time

import polars as pl

from altrios.lifts import run_terminal_simulation, utilities


TERMINAL = "Allouez"


def _print_summary(container_data: pl.DataFrame, resource_log: pl.DataFrame) -> None:
    ic_total = container_data.filter(pl.col("container_id").str.starts_with("IC"))
    oc_total = container_data.filter(pl.col("container_id").str.starts_with("OC"))
    train_rows = container_data.filter(pl.col("container_id").str.starts_with("Train-"))

    print()
    print("=" * 60)
    print(f"truck_rail @ {TERMINAL} — summary")
    print("=" * 60)
    print(f"  container rows   : {container_data.height}")
    print(f"    IC (inbound)  : {ic_total.height}")
    print(f"    OC (outbound) : {oc_total.height}")
    print(f"  train rows       : {train_rows.height}")
    if train_rows.height > 0:
        print(f"    train_depart   : "
              f"{train_rows['train_depart'].min():.2f} -> "
              f"{train_rows['train_depart'].max():.2f} h")

    if resource_log.height > 0:
        energy_by_resource = (
            resource_log.group_by("resource_type")
            .agg(pl.col("consumption_value").sum().alias("total"))
            .sort("resource_type")
        )
        print()
        print("  energy by resource_type (gal/kWh):")
        for row in energy_by_resource.iter_rows(named=True):
            print(f"    {row['resource_type']:>22}: {row['total']:>10.2f}")
        print(f"  total energy     : "
              f"{resource_log['consumption_value'].sum():.2f}")
        print(f"  total emissions  : "
              f"{resource_log['emissions(kgCO2)'].sum():.2f} kgCO2e")


def main() -> None:
    resources = utilities.resources_root()

    consist_plan = (
        pl.read_csv(resources / "train_consist_plan.csv")
        .with_columns(pl.lit("Intermodal").alias("Train_Type"))
    )

    t0 = time.perf_counter()
    container_data, resource_log, _ = run_terminal_simulation(
        modes=["truck_rail"],
        terminal=TERMINAL,
        inputs={"truck_rail": {"train_consist_plan": consist_plan}},
    )
    elapsed = time.perf_counter() - t0
    print(f"\nLIFTS truck_rail run: {elapsed:.2f} s")

    _print_summary(container_data, resource_log)


if __name__ == "__main__":
    main()
