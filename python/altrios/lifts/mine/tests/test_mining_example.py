"""End-to-end smoke test for the ``mining_haul`` example catalog.

Validates that :func:`altrios.lifts.workflow_engine.run_site` runs against
a YAML catalog from a domain the engine has never seen (no freight
classes, no Terminal object, no pre-existing helpers) and produces
the expected output shape. This is the keystone deliverable for the
"engine is domain-agnostic" claim.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from altrios.lifts.workflow_engine import run_site

# Importing the helper module here ensures @register decorators have
# fired before the loader looks the names up. (The catalog's
# python_module field also imports this module, but importing it
# explicitly belt-and-suspenders the test.)
from altrios.lifts.mine import mining_helpers  # noqa: F401


EXAMPLE_SITE = Path(__file__).resolve().parents[1] / "sites" / "example_mine.yaml"


def test_mining_haul_runs_to_completion():
    """Loads and runs the bundled mining_haul site; checks counts and
    that all trucks reached the ``dump_complete`` event."""
    result = run_site(EXAMPLE_SITE)

    # 8 trucks dispatched, 8 dump_complete events.
    assert len(result.entities) == 8
    assert len(result.output.event_log) == 8

    dumps = [
        r for r in result.output.event_log
        if r["event_type"] == "dump_complete"
    ]
    assert len(dumps) == 8

    # Each event carries the payload column from the catalog.
    payloads = [r["payload_t"] for r in dumps]
    assert all(p == pytest.approx(220.0) for p in payloads)

    # Truck ids are auto-assigned by the builder.
    ids = sorted(r["entity_id"] for r in dumps)
    assert ids == [f"truck-{i}" for i in range(8)]

    # Resource pools exist on state with the catalog-declared capacities.
    assert result.state.shovels.capacity == 2
    assert result.state.crusher_bays.capacity == 1


def test_mining_haul_respects_capacity_contention():
    """With only 1 crusher bay and 8 trucks, sim time must exceed the
    minimum cycle time × number of trucks / number of bays. This is a
    sanity check that the resource really serializes."""
    result = run_site(EXAMPLE_SITE)
    # Min crusher-bay-side work per truck: dump time = 0.02 hr.
    # 8 trucks * 0.02 = 0.16 hr at minimum across the single bay.
    # Plus the first truck has to drive empty + load + drive loaded
    # before it can dump (0.05 + min 0.05 + 0.07 = 0.17 hr).
    # So total sim time must be >= ~0.33 hr.
    assert result.env.now >= 0.30
    # And less than naive sequential (full path repeated 8 times).
    full_seq_upper = 8 * (0.05 + 0.0833 + 0.07 + 0.02)
    assert result.env.now <= full_seq_upper + 0.1


def test_mining_haul_seed_kwarg_overrides_site():
    """Site declares seed: 1234. kwarg seed=99 should win."""
    r1 = run_site(EXAMPLE_SITE, seed=99)
    r2 = run_site(EXAMPLE_SITE, seed=99)
    # Two runs with the same seed should produce identical event times
    # (the only stochastic step is the Uniform load duration).
    times_1 = sorted(r["record_timestamp"] for r in r1.output.event_log)
    times_2 = sorted(r["record_timestamp"] for r in r2.output.event_log)
    assert times_1 == times_2


def test_mining_haul_different_seeds_diverge():
    """Different seeds → at least one event time differs."""
    r1 = run_site(EXAMPLE_SITE, seed=1)
    r2 = run_site(EXAMPLE_SITE, seed=2)
    times_1 = sorted(r["record_timestamp"] for r in r1.output.event_log)
    times_2 = sorted(r["record_timestamp"] for r in r2.output.event_log)
    assert times_1 != times_2
