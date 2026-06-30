"""LIFTS demo: ``vessel_truck`` mode (workflow-engine path).

A vessel<->truck terminal where vessels and drayage trucks exchange
containers through the main container stack. No rail tracks.

This demo runs the ``allouez_vessel_truck.yaml`` site definition through
:func:`altrios.lifts.workflow_engine.run_site` and then materializes the
freight-flavoured ``container_data`` / ``resource_log`` DataFrames via
:func:`altrios.lifts.terminal.python_helpers.assemble_outputs`.

Run from the repo root with::

    python -m altrios.lifts.demos.vessel_truck_demo

or directly with::

    python python/altrios/lifts/demos/vessel_truck_demo.py
"""
from __future__ import annotations

import time
from pathlib import Path

import polars as pl

from altrios.lifts.terminal.python_helpers import assemble_outputs
from altrios.lifts.workflow_engine import run_site


TERMINAL = "Allouez"
SITE_FILE = (
    Path(__file__).resolve().parent.parent / "sites" / "allouez_vessel_truck.yaml"
)


def _print_summary(container_data: pl.DataFrame, resource_log: pl.DataFrame) -> None:
    ic_total = container_data.filter(pl.col("container_id").str.starts_with("IC"))
    oc_total = container_data.filter(pl.col("container_id").str.starts_with("OC"))

    print()
    print("=" * 60)
    print(f"vessel_truck @ {TERMINAL} — summary")
    print("=" * 60)
    print(f"  container rows   : {container_data.height}")
    print(f"    IC (inbound)  : {ic_total.height}")
    print(f"    OC (outbound) : {oc_total.height}")

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

        sts_events = resource_log.filter(pl.col("resource_type") == "sts_crane")
        if sts_events.height > 0:
            print()
            print(f"  STS crane lifts  : {sts_events.height}")
            for row in (
                sts_events.group_by("event_type")
                .agg(pl.len().alias("count"))
                .sort("event_type")
                .iter_rows(named=True)
            ):
                print(f"    {row['event_type']:>22}: {row['count']:>6}")

        drayage_events = resource_log.filter(pl.col("resource_type") == "truck")
        if drayage_events.height > 0:
            print()
            print(f"  drayage truck trips: {drayage_events.height}")

        print(f"  total energy     : "
              f"{resource_log['consumption_value'].sum():.2f}")
        print(f"  total emissions  : "
              f"{resource_log['emissions(kgCO2)'].sum():.2f} kgCO2e")


def main() -> None:
    t0 = time.perf_counter()
    result = run_site(str(SITE_FILE), seed=42)
    container_data, resource_log = assemble_outputs(result, mode_name="vessel_truck")
    elapsed = time.perf_counter() - t0
    print(f"\nLIFTS vessel_truck run: {elapsed:.2f} s")

    _print_summary(container_data, resource_log)


if __name__ == "__main__":
    main()
