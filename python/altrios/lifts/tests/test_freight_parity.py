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
