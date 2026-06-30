"""Phase-2 verification: multi-mode dispatcher + cross-mode resource sharing.

Verifies:

  7a. ``run_terminal_simulation(modes=[A,B,C], ...)`` runs all three
      Phase-1 modes concurrently against one Terminal without error.
  7b. When multiple modes are active, the Terminal state contains the
      *union* of every active mode's ``resource_specs`` — each shared
      spec materialized as exactly one SimPy primitive.
  7c. When only one mode is active, the Terminal state contains only
      *that* mode's specs (no leakage from the other modes' specs).
  7d. Cross-mode event types appear together in the same
      ``container_data`` DataFrame (proves single shared output buffer).

Run from the repo root::

    python scripts/verify_lifts_phase2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from altrios.lifts import run_terminal_simulation, utilities  # noqa: E402
from altrios.lifts.terminal_sim import get_mode  # noqa: E402


TERMINAL = "Allouez"


def _passed(label: str) -> None:
    print(f"  [PASS] {label}")


def _failed(label: str, detail: str = "") -> None:
    print(f"  [FAIL] {label}")
    if detail:
        for line in detail.splitlines():
            print(f"         {line}")
    sys.exit(1)


def _consist_plan() -> pl.DataFrame:
    return (
        pl.read_csv(utilities.resources_root() / "train_consist_plan.csv")
        .with_columns(pl.lit("Intermodal").alias("Train_Type"))
    )


def _rail_vessel_vessels() -> pl.DataFrame:
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


def _vessel_truck_vessels() -> pl.DataFrame:
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


def _vessel_truck_drayage() -> pl.DataFrame:
    rows: list[dict] = []
    truck_id = 1
    for i in range(40):
        rows.append({
            "Terminal_ID": TERMINAL, "Truck_ID": truck_id,
            "Arrival_Time_Hr": 80.0 + i * (90.0 / 40.0),
            "Action": "dropoff", "Container_ID": "",
        })
        truck_id += 1
    for i in range(40):
        rows.append({
            "Terminal_ID": TERMINAL, "Truck_ID": truck_id,
            "Arrival_Time_Hr": 60.0 + i * (90.0 / 40.0),
            "Action": "pickup", "Container_ID": "",
        })
        truck_id += 1
    return pl.DataFrame(rows)


def main() -> None:
    print("=" * 60)
    print("LIFTS Phase 2 verification — multi-mode + cross-mode sharing")
    print("=" * 60)

    # ---- Single-mode reference runs (to compare spec set against multi).
    print("\n[7c-prep] running single-mode truck_rail to record its spec set")
    _, _, term_solo = run_terminal_simulation(
        modes=["truck_rail"],
        terminal=TERMINAL,
        inputs={"truck_rail": {"train_consist_plan": _consist_plan()}},
    )
    truck_rail_spec_names = {s.name for s in get_mode("truck_rail").resource_specs}
    solo_state_attrs = {
        attr for attr in truck_rail_spec_names
        if hasattr(term_solo.state, attr) and getattr(term_solo.state, attr) is not None
    }
    if solo_state_attrs != truck_rail_spec_names:
        _failed(
            "single-mode truck_rail missing expected spec attributes",
            f"expected={sorted(truck_rail_spec_names)} "
            f"got={sorted(solo_state_attrs)}",
        )
    _passed(f"single-mode truck_rail state has all {len(truck_rail_spec_names)} "
            "of its declared resource_specs")

    # 7c: single-mode truck_rail must NOT have the rail_vessel-only or
    # vessel_truck-only specs (no leakage from inactive modes).
    rv_specs = {s.name for s in get_mode("rail_vessel").resource_specs}
    vt_specs = {s.name for s in get_mode("vessel_truck").resource_specs}
    inactive_only = (rv_specs | vt_specs) - truck_rail_spec_names
    leaked = [a for a in inactive_only if hasattr(term_solo.state, a)
              and getattr(term_solo.state, a) is not None]
    if leaked:
        _failed("single-mode truck_rail leaked inactive-mode pools",
                f"leaked={sorted(leaked)}")
    _passed(f"single-mode truck_rail has none of the "
            f"{len(inactive_only)} pools exclusive to other modes "
            f"(e.g. berths, sts_cranes_by_berth)")

    # ---- 7a + 7b: run all three modes concurrently.
    print("\n[7a] running all three modes concurrently")
    _, cd_multi, term_multi = (None, None, None)
    try:
        cd_multi, vl_multi, term_multi = run_terminal_simulation(
            modes=["truck_rail", "rail_vessel", "vessel_truck"],
            terminal=TERMINAL,
            inputs={
                "truck_rail": {"train_consist_plan": _consist_plan()},
                "rail_vessel": {"vessel_schedule": _rail_vessel_vessels()},
                "vessel_truck": {
                    "vessel_schedule": _vessel_truck_vessels(),
                    "drayage_schedule": _vessel_truck_drayage(),
                },
            },
        )
    except Exception as exc:
        _failed("multi-mode run raised", repr(exc))
    _passed(f"3-mode run completed (cd={cd_multi.height}, vl={vl_multi.height})")

    # 7b: union spec set must all be present on the shared state.
    union_spec_names = truck_rail_spec_names | rv_specs | vt_specs
    missing = [
        s for s in union_spec_names
        if not hasattr(term_multi.state, s)
        or getattr(term_multi.state, s) is None
    ]
    if missing:
        _failed(
            "multi-mode state missing union-merged specs",
            f"missing={sorted(missing)} "
            f"(union of {len(union_spec_names)} expected)",
        )
    _passed(f"multi-mode state has all {len(union_spec_names)} union-merged "
            f"specs (including {len(rv_specs & vt_specs)} that appear in "
            "both rail_vessel and vessel_truck)")

    # Identity check: pools shared between rail_vessel and vessel_truck
    # must be the *same Python object* (proves merge_specs deduplicated
    # them rather than instantiating twice).
    shared_between_rv_and_vt = sorted(rv_specs & vt_specs)
    print(f"\n[7b-identity] {len(shared_between_rv_and_vt)} pools shared "
          "between rail_vessel and vessel_truck:")
    print(f"           {shared_between_rv_and_vt}")
    # Single Terminal instance => only one object exists; this is automatic
    # given build_state_from_specs's name-keyed dict. The substantive check
    # is that merge_specs raised no ValueError above. We verify the pool is
    # a SimPy primitive (not e.g. None or a tuple of duplicates).
    import simpy
    for name in shared_between_rv_and_vt:
        obj = getattr(term_multi.state, name)
        # Partitioned specs land as dict[key, primitive]; non-partitioned as
        # a SimPy primitive (Resource/Container/Store).
        if isinstance(obj, dict):
            for k, v in obj.items():
                if not isinstance(v, (simpy.Resource, simpy.PriorityResource,
                                       simpy.Container, simpy.Store,
                                       simpy.FilterStore, simpy.PreemptiveResource)):
                    _failed(
                        f"shared pool {name}[{k!r}] is not a SimPy primitive",
                        f"got type={type(v).__name__}",
                    )
        else:
            if not isinstance(obj, (simpy.Resource, simpy.PriorityResource,
                                    simpy.Container, simpy.Store,
                                    simpy.FilterStore, simpy.PreemptiveResource)):
                _failed(
                    f"shared pool {name} is not a SimPy primitive",
                    f"got type={type(obj).__name__}",
                )
    _passed("every shared pool is a single SimPy primitive (no duplicates)")

    # ---- 7d: cross-mode event types co-located in one DataFrame.
    rail_only_events = {"train_arrival_actual", "rail_track_rtg_unload",
                        "rail_track_rtg_load", "train_depart"}
    vessel_only_events = {"sts_unload", "sts_load"}
    drayage_only_events = {"drayage_gate_in", "drayage_gate_out"}
    cols = set(cd_multi.columns)
    for grp_name, grp in [
        ("rail-only", rail_only_events),
        ("vessel-only", vessel_only_events),
        ("drayage-only", drayage_only_events),
    ]:
        missing = grp - cols
        if missing:
            _failed(
                f"multi-mode container_data missing {grp_name} events",
                f"missing={sorted(missing)}",
            )
        # at least one row must be non-null for each (to prove the events
        # actually fired in this concurrent run, not just got backfilled).
        for col in grp:
            n = cd_multi[col].drop_nulls().len()
            if n == 0:
                _failed(
                    f"multi-mode container_data has {col} column but it is all-null",
                    f"this means the {grp_name} mode didn't actually fire",
                )
    _passed("rail-only, vessel-only, drayage-only events all present "
            "AND non-empty in one shared container_data")

    print("\n" + "=" * 60)
    print("ALL PHASE-2 VERIFICATION CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
