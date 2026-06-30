"""Freight smoke tests for the workflow-engine-driven LIFTS path.

A regression smoke test that pins the ``run_site`` output for each
Allouez freight demo within ±0.5% of a recorded baseline.

Baselines below were produced by running ``run_site`` against the three
``allouez_*.yaml`` site files at seed=42. Drift outside the tolerance
signals a real divergence worth investigating.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from altrios.lifts.python_helpers import assemble_outputs
from altrios.workflow_engine import run_site


_SITES_DIR = Path(__file__).parent.parent / "sites"


# (cd_rows, rl_rows, energy, env_now) baseline per mode. Recorded after
# Phase A.6 completed; see this module's docstring for context.
_BASELINES: dict[str, tuple[int, int, float, float]] = {
    "truck_rail": (3808, 11364, 2468.898, 336.264),
    "rail_vessel": (644, 2140, 749.000, 272.000),
    "vessel_truck": (719, 2290, 772.658, 272.000),
}

_REL_TOL = 0.005  # ±0.5% — same tolerance as the prior parity bar.


def _approx(actual: float, expected: float, rel_tol: float = _REL_TOL) -> bool:
    if expected == 0.0:
        return abs(actual) < 1e-9
    return abs(actual - expected) / abs(expected) <= rel_tol


def _run(site_filename: str, mode_name: str) -> tuple[pl.DataFrame, pl.DataFrame, float]:
    result = run_site(str(_SITES_DIR / site_filename), seed=42)
    cd, rl = assemble_outputs(result, mode_name=mode_name)
    return cd, rl, float(result.env.now)


def _assert_baseline(mode_name: str, site_filename: str) -> None:
    cd_rows_exp, rl_rows_exp, energy_exp, env_now_exp = _BASELINES[mode_name]
    cd, rl, env_now = _run(site_filename, mode_name)

    assert cd.height == cd_rows_exp, (
        f"{mode_name}: container_data row count drifted: "
        f"got {cd.height}, baseline {cd_rows_exp}"
    )
    assert _approx(rl.height, rl_rows_exp), (
        f"{mode_name}: resource_log row count drifted: "
        f"got {rl.height}, baseline {rl_rows_exp}"
    )
    energy = float(rl["consumption_value"].sum())
    assert _approx(energy, energy_exp), (
        f"{mode_name}: total energy drifted: got {energy:.3f}, "
        f"baseline {energy_exp:.3f} "
        f"(diff {(energy - energy_exp) / energy_exp * 100:+.3f}%)"
    )
    assert _approx(env_now, env_now_exp), (
        f"{mode_name}: env.now drifted: got {env_now:.3f}, "
        f"baseline {env_now_exp:.3f}"
    )


def test_truck_rail_smoke():
    """truck_rail run_site path produces expected counts + energy."""
    _assert_baseline("truck_rail", "allouez_truck_rail.yaml")


def test_rail_vessel_smoke():
    """rail_vessel run_site path produces expected counts + energy."""
    _assert_baseline("rail_vessel", "allouez_rail_vessel.yaml")


def test_vessel_truck_smoke():
    """vessel_truck run_site path produces expected counts + energy."""
    _assert_baseline("vessel_truck", "allouez_vessel_truck.yaml")


# ---- Multi-mode smoke ----------------------------------------------
#
# The ``allouez_combined.yaml`` site activates ``truck_rail`` +
# ``vessel_truck`` simultaneously, sharing the yard-crane / chassis /
# stack / yard-tractor pools. The smoke pins entity counts and
# aggregate consumption so resource-sharing regressions (e.g. accidental
# double-counting from a dispatcher change) surface immediately. It
# does NOT depend on :func:`assemble_outputs` since v1 of that helper
# is single-mode; we read totals straight off the engine collector.

_COMBINED_BASELINE = {
    "trains": 20,
    "vessels": 4,
    "drayage": 1944,
    "event_rows": 19710,
    "consumption_rows": 13654,
    "consumption_total": 3247.34,
    "env_now": 336.264,
}


def test_combined_truck_rail_vessel_truck_smoke():
    """allouez_combined activates two modes against shared pools; pin
    arrival counts, event/consumption row counts, total consumption,
    and sim-end time."""
    result = run_site(
        str(_SITES_DIR / "allouez_combined.yaml"), seed=42,
    )

    by_kind: dict[str, int] = {}
    for ent in result.entities:
        by_kind[ent.kind] = by_kind.get(ent.kind, 0) + 1

    assert by_kind.get("train", 0) == _COMBINED_BASELINE["trains"], (
        f"train count drifted: got {by_kind.get('train', 0)}, "
        f"baseline {_COMBINED_BASELINE['trains']}"
    )
    assert by_kind.get("vessel", 0) == _COMBINED_BASELINE["vessels"], (
        f"vessel count drifted: got {by_kind.get('vessel', 0)}, "
        f"baseline {_COMBINED_BASELINE['vessels']}"
    )
    assert by_kind.get("drayage", 0) == _COMBINED_BASELINE["drayage"], (
        f"drayage count drifted: got {by_kind.get('drayage', 0)}, "
        f"baseline {_COMBINED_BASELINE['drayage']}"
    )

    assert _approx(
        len(result.output.event_log), _COMBINED_BASELINE["event_rows"]
    ), (
        f"event_log row count drifted: got {len(result.output.event_log)}, "
        f"baseline {_COMBINED_BASELINE['event_rows']}"
    )
    assert _approx(
        len(result.output.consumption_log),
        _COMBINED_BASELINE["consumption_rows"],
    ), (
        f"consumption_log row count drifted: "
        f"got {len(result.output.consumption_log)}, "
        f"baseline {_COMBINED_BASELINE['consumption_rows']}"
    )
    consumption_total = sum(
        float(r.get("consumption_value") or 0.0)
        for r in result.output.consumption_log
    )
    assert _approx(consumption_total, _COMBINED_BASELINE["consumption_total"]), (
        f"total consumption drifted: got {consumption_total:.2f}, "
        f"baseline {_COMBINED_BASELINE['consumption_total']:.2f}"
    )
    assert _approx(float(result.env.now), _COMBINED_BASELINE["env_now"]), (
        f"env.now drifted: got {result.env.now:.3f}, "
        f"baseline {_COMBINED_BASELINE['env_now']:.3f}"
    )
