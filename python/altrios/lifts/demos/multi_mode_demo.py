"""LIFTS Phase 2 demo: ``truck_rail`` + ``rail_vessel`` + ``vessel_truck``
concurrent at one terminal.

Runs *all three* terminal modes against the same ``Terminal`` instance
in one SimPy environment. The dispatcher unions their ``resource_specs``
via ``resources_decl.merge_specs``, so every pool referenced by more
than one mode (``main_stack_rtgs``, ``main_yard_tractors``,
``berths``, ``sts_cranes_by_berth``, ``rail_track_rtgs_by_track``,
``in_gates``, ``out_gates``, ``container_stack``, ...) is a *single
shared* SimPy primitive — trains, vessels, and drayage trucks all
contend for the same physical equipment.

Inputs:
  * ``truck_rail``: bundled train consist plan + synthesized drayage
    (one dropoff per OC and one pickup per IC). This drives the
    rail/yard/gate side end-to-end and is the side the LIFTS pipeline
    was originally built for.
  * ``rail_vessel``: long-haul vessel calls only (``train_consist_plan``
    omitted so we don't double-schedule the same trains that
    ``truck_rail`` is already running).
  * ``vessel_truck``: short-haul vessel calls + drayage trucks for
    local pickup/delivery.

The two vessel schedules use disjoint ``Vessel_ID`` ranges (1..2 for
``rail_vessel``, 101..102 for ``vessel_truck``) to avoid collisions in
the per-vessel state.

Run with::

    python -m altrios.lifts.demos.multi_mode_demo

or directly::

    python python/altrios/lifts/demos/multi_mode_demo.py
"""
from __future__ import annotations

import time

import polars as pl

from altrios.lifts import run_terminal_simulation, utilities


TERMINAL = "Allouez"


def _build_rail_vessel_vessels() -> pl.DataFrame:
    """Long-haul vessels (IDs 1..2)."""
    return pl.DataFrame({
        "Vessel_ID": [1, 2],
        "Vessel_Name": ["Northland Spirit", "Boreal Crest"],
        "Origin_ID": ["Duluth", TERMINAL],
        "Destination_ID": [TERMINAL, "Duluth"],
        "Arrival_Time_Hr": [40.0, 220.0],
        "Departure_Time_Hr": [80.0, 260.0],
        "Inbound_Containers": [80, 0],
        "Outbound_Containers": [0, 60],
    })


def _build_vessel_truck_vessels() -> pl.DataFrame:
    """Short-haul / coastal vessels (IDs 101..102)."""
    return pl.DataFrame({
        "Vessel_ID": [101, 102],
        "Vessel_Name": ["Local Tide", "Bay Runner"],
        "Origin_ID": ["Duluth", TERMINAL],
        "Destination_ID": [TERMINAL, "Duluth"],
        "Arrival_Time_Hr": [30.0, 190.0],
        "Departure_Time_Hr": [55.0, 215.0],
        "Inbound_Containers": [40, 0],
        "Outbound_Containers": [0, 30],
    })


def _build_vessel_truck_drayage() -> pl.DataFrame:
    """Local drayage for the vessel_truck side.

    * Dropoffs t=80..170h supply OC stock for vessel 102's outbound load.
    * Pickups t=60..150h consume ICs vessel 101 delivered at t=30h.
    """
    rows: list[dict] = []
    truck_id = 1
    for i in range(40):
        rows.append({
            "Terminal_ID": TERMINAL,
            "Truck_ID": truck_id,
            "Arrival_Time_Hr": 80.0 + i * (90.0 / 40.0),
            "Action": "dropoff",
            "Container_ID": "",
        })
        truck_id += 1
    for i in range(40):
        rows.append({
            "Terminal_ID": TERMINAL,
            "Truck_ID": truck_id,
            "Arrival_Time_Hr": 60.0 + i * (90.0 / 40.0),
            "Action": "pickup",
            "Container_ID": "",
        })
        truck_id += 1
    return pl.DataFrame(rows)


def _print_summary(
    container_data: pl.DataFrame, resource_log: pl.DataFrame, terminal_obj,
) -> None:
    print()
    print("=" * 64)
    print(f"multi_mode (3 modes concurrent) @ {TERMINAL} — summary")
    print("=" * 64)

    # Container counts.
    ic_total = container_data.filter(pl.col("container_id").str.starts_with("IC"))
    oc_total = container_data.filter(pl.col("container_id").str.starts_with("OC"))
    print(f"  container rows : {container_data.height}")
    print(f"    IC (inbound) : {ic_total.height}")
    print(f"    OC (outbound): {oc_total.height}")

    # Activity columns to demonstrate every mode's events landed in the same
    # output frame.
    activity_cols = [
        ("train_arrival_actual", "train arrivals"),
        ("rail_track_rtg_unload", "rail-track RTG unloads"),
        ("rail_track_rtg_load", "rail-track RTG loads"),
        ("train_depart", "train departures"),
        ("sts_unload", "STS unload events"),
        ("sts_load", "STS load events"),
        ("drayage_gate_in", "drayage gate-in events"),
        ("drayage_gate_out", "drayage gate-out events"),
    ]
    print()
    print("  event-column non-null counts (proves cross-mode merge):")
    for col, label in activity_cols:
        if col in container_data.columns:
            cnt = container_data[col].drop_nulls().len()
            print(f"    {label:>30}: {cnt:>6}")

    # Energy / emissions totals.
    if resource_log.height > 0:
        print()
        print("  energy by resource_type (gal/kWh):")
        for row in (
            resource_log.group_by("resource_type")
            .agg(pl.col("consumption_value").sum().alias("total"))
            .sort("resource_type")
            .iter_rows(named=True)
        ):
            print(f"    {row['resource_type']:>22}: {row['total']:>10.2f}")

        total_energy = resource_log["consumption_value"].sum()
        total_co2 = resource_log["emissions(kgCO2)"].sum()
        print()
        print(f"  total energy   : {total_energy:>10.2f}")
        print(f"  total emissions: {total_co2:>10.2f} kgCO2e")

    # Cross-mode pool-sharing evidence: every union-merged spec landed on
    # the single shared Terminal state.
    state = terminal_obj.state
    print()
    print("  shared-state pools (union of all 3 modes' specs):")
    for attr in (
        "berths", "sts_cranes_by_berth",
        "main_stack_rtgs", "top_picks", "container_stack",
        "rail_track_rtgs_by_track", "tracks",
        "main_yard_tractors", "rail_yard_tractors",
        "in_gates", "out_gates",
        "parking_chassis_slots", "terminal_chassis_pool", "road_chassis_pool",
    ):
        present = hasattr(state, attr) and getattr(state, attr) is not None
        print(f"    {attr:>26}: {'yes' if present else 'no'}")


def main() -> None:
    resources = utilities.resources_root()

    consist_plan = (
        pl.read_csv(resources / "train_consist_plan.csv")
        .with_columns(pl.lit("Intermodal").alias("Train_Type"))
    )
    rail_vessels = _build_rail_vessel_vessels()
    short_vessels = _build_vessel_truck_vessels()
    drayage = _build_vessel_truck_drayage()

    print(f"Trains (truck_rail)        : (from consist plan, with synth drayage)")
    print(f"Vessels (rail_vessel)      : {rail_vessels.height}")
    print(f"Vessels (vessel_truck)     : {short_vessels.height}")
    print(f"Drayage trucks (vessel_truck): {drayage.height}")

    t0 = time.perf_counter()
    container_data, resource_log, terminal_obj = run_terminal_simulation(
        modes=["truck_rail", "rail_vessel", "vessel_truck"],
        terminal=TERMINAL,
        inputs={
            "truck_rail": {
                "train_consist_plan": consist_plan,
            },
            "rail_vessel": {
                "vessel_schedule": rail_vessels,
            },
            "vessel_truck": {
                "vessel_schedule": short_vessels,
                "drayage_schedule": drayage,
            },
        },
    )
    elapsed = time.perf_counter() - t0
    print(f"\nLIFTS multi-mode (3-way) run: {elapsed:.2f} s")

    _print_summary(container_data, resource_log, terminal_obj)


if __name__ == "__main__":
    main()
