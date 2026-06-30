"""LIFTS Phase 1 demo: ``vessel_truck`` mode.

A vessel<->truck terminal where vessels and drayage trucks exchange
containers through the main container stack. No rail tracks.

Inputs (constructed inline so the demo is self-contained and the
vessel/drayage schedule is mutually feasible):

  * A small vessel call list with two calls at ``Allouez``: an early
    "delivery" call (inbound only) that seeds the stack, and a later
    "loading" call (outbound only) that depletes containers the drayage
    dropoffs have accumulated.
  * A drayage schedule with a burst of dropoffs (build OC stock) and a
    burst of pickups (consume IC stock delivered by the first vessel).

Run from the repo root with::

    python -m altrios.lifts.demos.vessel_truck_demo

or directly with::

    python python/altrios/lifts/demos/vessel_truck_demo.py
"""
from __future__ import annotations

import time

import polars as pl

from altrios.lifts import run_terminal_simulation


TERMINAL = "Allouez"


def _build_vessel_calls() -> pl.DataFrame:
    """Two vessel calls at Allouez.

    * Vessel 1 (t=20h, depart 50h): delivers 80 inbound, loads 0 outbound.
      Drayage will pick those 80 ICs up later.
    * Vessel 2 (t=120h, depart 150h): delivers 0 inbound, loads 60 outbound.
      Drayage dropoffs through t=100h supply the OC stock.
    """
    return pl.DataFrame({
        "Vessel_ID": [1, 2],
        "Vessel_Name": ["Northland Spirit", "Boreal Crest"],
        "Origin_ID": ["Duluth", TERMINAL],
        "Destination_ID": [TERMINAL, "Duluth"],
        "Arrival_Time_Hr": [20.0, 120.0],
        "Departure_Time_Hr": [50.0, 150.0],
        "Inbound_Containers": [80, 0],
        "Outbound_Containers": [0, 60],
    })


def _build_drayage_schedule() -> pl.DataFrame:
    """Two bursts of trucks:

    * Dropoffs t=30..100h supply OC stock for Vessel 2's outbound load.
    * Pickups t=60..120h consume ICs Vessel 1 delivered at t=20h.
    """
    rows: list[dict] = []
    truck_id = 1
    # Burst 1: 80 dropoffs over 70 hours (≈1.1 trucks/hr)
    for i in range(80):
        rows.append({
            "Terminal_ID": TERMINAL,
            "Truck_ID": truck_id,
            "Arrival_Time_Hr": 30.0 + i * (70.0 / 80.0),
            "Action": "dropoff",
            "Container_ID": "",
        })
        truck_id += 1
    # Burst 2: 80 pickups over 60 hours
    for i in range(80):
        rows.append({
            "Terminal_ID": TERMINAL,
            "Truck_ID": truck_id,
            "Arrival_Time_Hr": 60.0 + i * (60.0 / 80.0),
            "Action": "pickup",
            "Container_ID": "",
        })
        truck_id += 1
    return pl.DataFrame(rows)


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
    vessel_calls = _build_vessel_calls()
    drayage_schedule = _build_drayage_schedule()

    print(f"Vessel calls   : {vessel_calls.height} at {TERMINAL}")
    print(f"Drayage trucks : {drayage_schedule.height} at {TERMINAL}")

    t0 = time.perf_counter()
    container_data, resource_log, _ = run_terminal_simulation(
        modes=["vessel_truck"],
        terminal=TERMINAL,
        inputs={"vessel_truck": {
            "vessel_schedule": vessel_calls,
            "drayage_schedule": drayage_schedule,
        }},
    )
    elapsed = time.perf_counter() - t0
    print(f"\nLIFTS vessel_truck run: {elapsed:.2f} s")

    _print_summary(container_data, resource_log)


if __name__ == "__main__":
    main()
