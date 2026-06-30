"""Freight smoke tests for the workflow-engine-driven LIFTS path.

A regression smoke test that pins the ``run_site`` output for each
Allouez freight demo within ±0.5% of a recorded baseline.

Baselines below were produced by running ``run_site`` against the three
``allouez_*.yaml`` site files at seed=42. Drift outside the tolerance
signals a real divergence worth investigating.
"""
from __future__ import annotations

import polars as pl

from altrios.lifts import terminal
from altrios.lifts.terminal.python_helpers import assemble_outputs


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


def _run(site_name: str, mode_name: str) -> tuple[pl.DataFrame, pl.DataFrame, float]:
    result = terminal.run(site_name, seed=42)
    cd, rl = assemble_outputs(result, mode_name=mode_name)
    return cd, rl, float(result.env.now)


def _assert_baseline(mode_name: str, site_name: str) -> None:
    cd_rows_exp, rl_rows_exp, energy_exp, env_now_exp = _BASELINES[mode_name]
    cd, rl, env_now = _run(site_name, mode_name)

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
    _assert_baseline("truck_rail", "allouez_truck_rail")


def test_rail_vessel_smoke():
    """rail_vessel run_site path produces expected counts + energy."""
    _assert_baseline("rail_vessel", "allouez_rail_vessel")


def test_vessel_truck_smoke():
    """vessel_truck run_site path produces expected counts + energy."""
    _assert_baseline("vessel_truck", "allouez_vessel_truck")


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

# ``allouez_all_modes.yaml`` activates all three freight modes; trains
# and vessels are duplicated across the contended modes
# (truck_rail + rail_vessel for trains; rail_vessel + vessel_truck for
# vessels). Drayage is contended between truck_rail and vessel_truck.
# See the site file's header comment for the illustrative-only caveat
# (no per-arrival routing data → duplication, not partitioning).
#
# Note: this site exhibits run-to-run noise in event/consumption row
# counts and end-of-sim time because duplicated trains and vessels
# contend for shared SimPy resource pools (tracks, sts_workers,
# cranes); SimPy's tie-breaking is sensitive to dict / set iteration
# order, which Python randomises per process. Per-test tolerance is
# widened below to ``_ALL_MODES_REL_TOL`` to absorb this noise while
# still catching a meaningful regression.
#
# Baselines centred on empirical midpoints from a 10-run in-process
# probe (seed=42); ±1.5 % tolerance covers each observed range with
# ~50 % budget headroom (worst case: consumption_total at ~74 %).
_ALL_MODES_REL_TOL = 0.015  # ±1.5 %
_ALL_MODES_BASELINE = {
    "trains": 40,           # 20 truck_rail + 20 rail_vessel
    "vessels": 8,           # 4 rail_vessel + 4 vessel_truck
    "drayage": 1944,        # unchanged
    "event_rows": 28490,        # observed range [28406, 28575] (±0.30%)
    "consumption_rows": 18967,  # observed range [18882, 19051] (±0.45%)
    "consumption_total": 4875.0,    # observed range [4840.8, 4908.2] (±0.69%)
    "env_now": 336.0,           # observed range [334.7, 337.4] (±0.40%)
}


def test_combined_truck_rail_vessel_truck_smoke():
    """allouez_combined activates two modes against shared pools; pin
    arrival counts, event/consumption row counts, total consumption,
    and sim-end time."""
    result = terminal.run("allouez_combined", seed=42)

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


def test_combined_all_modes_smoke():
    """allouez_all_modes activates all three freight modes; pin
    arrival counts (trains/vessels duplicated across contended
    modes), event/consumption row counts, total consumption, and
    sim-end time. Also exercise multi-mode assemble_outputs by passing
    the active mode list."""
    result = terminal.run("allouez_all_modes", seed=42)

    by_kind: dict[str, int] = {}
    for ent in result.entities:
        by_kind[ent.kind] = by_kind.get(ent.kind, 0) + 1

    assert by_kind.get("train", 0) == _ALL_MODES_BASELINE["trains"], (
        f"train count drifted: got {by_kind.get('train', 0)}, "
        f"baseline {_ALL_MODES_BASELINE['trains']}"
    )
    assert by_kind.get("vessel", 0) == _ALL_MODES_BASELINE["vessels"], (
        f"vessel count drifted: got {by_kind.get('vessel', 0)}, "
        f"baseline {_ALL_MODES_BASELINE['vessels']}"
    )
    assert by_kind.get("drayage", 0) == _ALL_MODES_BASELINE["drayage"], (
        f"drayage count drifted: got {by_kind.get('drayage', 0)}, "
        f"baseline {_ALL_MODES_BASELINE['drayage']}"
    )

    assert _approx(
        len(result.output.event_log),
        _ALL_MODES_BASELINE["event_rows"],
        rel_tol=_ALL_MODES_REL_TOL,
    ), (
        f"event_log row count drifted: got {len(result.output.event_log)}, "
        f"baseline {_ALL_MODES_BASELINE['event_rows']} "
        f"±{_ALL_MODES_REL_TOL * 100:.1f}%"
    )
    assert _approx(
        len(result.output.consumption_log),
        _ALL_MODES_BASELINE["consumption_rows"],
        rel_tol=_ALL_MODES_REL_TOL,
    ), (
        f"consumption_log row count drifted: "
        f"got {len(result.output.consumption_log)}, "
        f"baseline {_ALL_MODES_BASELINE['consumption_rows']} "
        f"±{_ALL_MODES_REL_TOL * 100:.1f}%"
    )
    consumption_total = sum(
        float(r.get("consumption_value") or 0.0)
        for r in result.output.consumption_log
    )
    assert _approx(
        consumption_total,
        _ALL_MODES_BASELINE["consumption_total"],
        rel_tol=_ALL_MODES_REL_TOL,
    ), (
        f"total consumption drifted: got {consumption_total:.2f}, "
        f"baseline {_ALL_MODES_BASELINE['consumption_total']:.2f} "
        f"±{_ALL_MODES_REL_TOL * 100:.1f}%"
    )
    assert _approx(
        float(result.env.now),
        _ALL_MODES_BASELINE["env_now"],
        rel_tol=_ALL_MODES_REL_TOL,
    ), (
        f"env.now drifted: got {result.env.now:.3f}, "
        f"baseline {_ALL_MODES_BASELINE['env_now']:.3f} "
        f"±{_ALL_MODES_REL_TOL * 100:.1f}%"
    )

    # Exercise multi-mode assemble_outputs: union the event-type
    # surfaces across all three modes; sanity-check that the
    # container_data wide frame includes columns from each surface.
    cd, rl = assemble_outputs(
        result, mode_name=("truck_rail", "rail_vessel", "vessel_truck"),
    )
    assert "train_arrival_actual" in cd.columns, (
        "truck_rail / rail_vessel event surface not present"
    )
    assert "sts_unload" in cd.columns, (
        "vessel event surface not present"
    )
    assert "drayage_gate_out" in cd.columns, (
        "drayage event surface not present"
    )
    assert rl.height == len(result.output.consumption_log), (
        f"resource_log row count diverged from collector: "
        f"got {rl.height}, expected {len(result.output.consumption_log)}"
    )
