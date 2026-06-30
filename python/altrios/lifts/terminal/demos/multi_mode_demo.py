"""LIFTS demo: combined ``truck_rail`` + ``vessel_truck`` modes.

Demonstrates the workflow-engine multi-mode dispatch unblock. The
``allouez_combined.yaml`` site activates both modes simultaneously
against shared resource pools (one yard-tractor fleet, one container
stack, one chassis pool, one main-stack RTG fleet). The dispatcher
routes ``train`` arrivals to ``truck_rail``, ``vessel`` arrivals to
``vessel_truck``, and disambiguates ``drayage`` arrivals via the
``mode`` key stamped on each by the catalog's schedule builders.

This is a smoke-style illustrative run: it confirms the combined
site loads and runs without errors and prints a per-mode arrival
breakdown plus the shared aggregate consumption. It also exercises
:func:`assemble_outputs` in multi-mode shape (passing the active
mode names as a sequence so the event-type surface is unioned).

Run from the repo root with::

    python -m altrios.lifts.demos.multi_mode_demo

or directly with::

    python python/altrios/lifts/demos/multi_mode_demo.py
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

from altrios.lifts.terminal.python_helpers import assemble_outputs
from altrios.lifts.workflow_engine import run_site


SITE_FILE = (
    Path(__file__).resolve().parent.parent / "sites" / "allouez_combined.yaml"
)
ACTIVE_MODES = ("truck_rail", "vessel_truck")


def _print_summary(result) -> None:
    by_kind: Counter[str] = Counter(e.kind for e in result.entities)
    event_count = len(result.output.event_log)
    consumption_count = len(result.output.consumption_log)
    consumption_total = sum(
        float(r.get("consumption_value") or 0.0)
        for r in result.output.consumption_log
    )

    print()
    print("=" * 60)
    print("allouez_combined (truck_rail + vessel_truck) — summary")
    print("=" * 60)
    print(f"  active modes        : {result.site.modes}")
    print(f"  total entities      : {len(result.entities)}")
    for kind, n in sorted(by_kind.items()):
        print(f"    kind={kind:<8}    : {n}")
    print(f"  event rows          : {event_count}")
    print(f"  consumption rows    : {consumption_count}")
    print(f"  total consumption   : {consumption_total:.2f}")
    print(f"  sim end time (h)    : {result.env.now:.2f}")

    # Show resource sharing in action: count distinct (resource_type)
    # rows. If both modes are sharing the main_stack_rtg fleet, both
    # train- and vessel-driven activities will appear under one
    # resource_type.
    by_resource: Counter[str] = Counter(
        r.get("resource_type", "?") for r in result.output.consumption_log
    )
    if by_resource:
        print()
        print("  consumption by resource_type:")
        for resource_type, n in sorted(by_resource.items()):
            print(f"    {resource_type:<22}: {n} rows")

    # Demonstrate multi-mode assemble_outputs: the event-type surface
    # is unioned across both active modes and truck_rail-specific
    # derived columns (container_processing_time, train_arrival_actual_oc)
    # are added because truck_rail is in the active set.
    cd, rl = assemble_outputs(result, mode_name=ACTIVE_MODES)
    print()
    print(f"  container_data shape : {cd.shape}")
    print(f"  resource_log shape   : {rl.shape}")


def main() -> None:
    t0 = time.perf_counter()
    result = run_site(str(SITE_FILE), seed=42)
    elapsed = time.perf_counter() - t0
    print(f"\nLIFTS multi-mode run: {elapsed:.2f} s")
    _print_summary(result)


if __name__ == "__main__":
    main()
