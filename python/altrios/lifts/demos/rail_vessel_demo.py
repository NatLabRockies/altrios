"""LIFTS Phase 1 demo: ``rail_vessel`` mode.

A rail<->vessel terminal where trains and vessels both exchange
containers through the main container stack. No drayage gate.

Inputs:

  * ``train_consist_plan.csv`` — trains arriving at terminal ``Allouez``.
  * ``vessel_call_list.csv``   — vessel calls at terminal ``Allouez`` (a
    call appears in the schedule iff its ``Origin_ID`` or ``Destination_ID``
    is ``Allouez``).

Run from the repo root with::

    python -m altrios.lifts.demos.rail_vessel_demo

or directly with::

    python python/altrios/lifts/demos/rail_vessel_demo.py
"""
from __future__ import annotations

import time
from pathlib import Path

import polars as pl

from altrios.lifts.python_helpers import assemble_outputs
from altrios.workflow_engine import run_site


TERMINAL = "Allouez"
SITE_FILE = (
    Path(__file__).resolve().parent.parent / "sites" / "allouez_rail_vessel.yaml"
)


def _print_summary(container_data: pl.DataFrame, resource_log: pl.DataFrame) -> None:
    ic_total = container_data.filter(pl.col("container_id").str.starts_with("IC"))
    oc_total = container_data.filter(pl.col("container_id").str.starts_with("OC"))
    train_rows = container_data.filter(pl.col("container_id").str.starts_with("Train-"))

    print()
    print("=" * 60)
    print(f"rail_vessel @ {TERMINAL} — summary")
    print("=" * 60)
    print(f"  container rows   : {container_data.height}")
    print(f"    IC (inbound)  : {ic_total.height}")
    print(f"    OC (outbound) : {oc_total.height}")
    print(f"  train rows       : {train_rows.height}")

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

        # STS-specific event-type breakdown (highlight vessel activity)
        sts_events = resource_log.filter(pl.col("resource_type") == "sts_crane")
        if sts_events.height > 0:
            print()
            print(f"  STS crane lifts  : {sts_events.height}")
            event_breakdown = (
                sts_events.group_by("event_type")
                .agg(pl.len().alias("count"))
                .sort("event_type")
            )
            for row in event_breakdown.iter_rows(named=True):
                print(f"    {row['event_type']:>22}: {row['count']:>6}")

        print(f"  total energy     : "
              f"{resource_log['consumption_value'].sum():.2f}")
        print(f"  total emissions  : "
              f"{resource_log['emissions(kgCO2)'].sum():.2f} kgCO2e")


def main() -> None:
    t0 = time.perf_counter()
    result = run_site(str(SITE_FILE), seed=42)
    container_data, resource_log = assemble_outputs(result, mode_name="rail_vessel")
    elapsed = time.perf_counter() - t0
    print(f"\nLIFTS rail_vessel run: {elapsed:.2f} s")

    _print_summary(container_data, resource_log)


if __name__ == "__main__":
    main()
