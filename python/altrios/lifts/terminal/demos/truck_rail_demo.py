"""LIFTS demo: ``truck_rail`` mode (workflow-engine path).

A truck<->rail intermodal terminal where containers exchange between
the rail tracks (via rail-track RTGs) and the gate (via drayage trucks),
with all containers routing through the main container stack.

This demo runs the ``allouez_truck_rail.yaml`` site definition through
:func:`altrios.lifts.workflow_engine.run_site` and then materializes the
freight-flavoured ``container_data`` / ``resource_log`` DataFrames via
:func:`altrios.lifts.terminal.python_helpers.assemble_outputs`.

Run from the repo root with::

    python -m altrios.lifts.demos.truck_rail_demo

or directly with::

    python python/altrios/lifts/demos/truck_rail_demo.py
"""
from __future__ import annotations

import time

import polars as pl

from altrios.lifts import terminal
from altrios.lifts.terminal.python_helpers import assemble_outputs


TERMINAL = "Allouez"
SITE_NAME = "allouez_truck_rail"


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
    if train_rows.height > 0 and "train_depart" in train_rows.columns:
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
    """Run the ``truck_rail`` mode freight demo end-to-end.

    Loads the truck-rail site YAML, runs the simulation with a fixed
    seed of ``42``, assembles container-data / resource-log outputs,
    prints the wall-clock runtime, and dumps a summary of throughput
    and energy.
    """
    t0 = time.perf_counter()
    result = terminal.run(SITE_NAME, seed=42)
    container_data, resource_log = assemble_outputs(result, mode_name="truck_rail")
    elapsed = time.perf_counter() - t0
    print(f"\nLIFTS truck_rail run: {elapsed:.2f} s")

    _print_summary(container_data, resource_log)


if __name__ == "__main__":
    main()

