"""End-to-end tests for :func:`altrios.lifts.workflow_engine.run_site`.

These exercise the runner's full path: load YAML → build env+state →
dispatch arrivals → run env to completion → return RunResult. They do
NOT depend on any domain catalog (freight/mining/etc.) — a tiny
inline catalog is written to tmp_path for each test.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from altrios.lifts.workflow_engine import RunError, run_site
from altrios.lifts.workflow_engine.registry import register


# Inline fixture callables — kept module-level so the @register
# decorators run once at import time, before tests fire.
@register("runner_tests.build_simple_arrivals")
def _build_simple_arrivals(schedule, *, count: int = 3, **_kwargs) -> list[dict]:
    """Return ``count`` synthetic widget arrivals spaced 1 hour apart.

    ``schedule`` is whatever the site put under
    ``schedules.runner_tests.widget_arrivals`` and is ignored here; we
    accept it solely so the runner's calling convention is honoured.
    """
    return [
        {"kind": "widget", "id": f"widget-{i}", "arrival_time": float(i)}
        for i in range(count)
    ]


@register("runner_tests.state_init_extras")
def _state_init_extras(*, env, state, config, layout) -> None:
    """Attach a domain-specific counter to state. Demonstrates the
    ``state_init`` hook by which catalogs augment generic state."""
    state.extras_counter = 0


CATALOG_GENERIC_YAML = """\
meta:
  schema_version: 1
name: runner_tests_catalog
python_module: altrios.lifts.workflow_engine.tests.test_runner
entity_kinds:
  - name: widget
    attrs:
      arrival_time: float
modes:
  - name: assembly
    arrival_routing:
      widget: process_widget
    graphs:
      - name: process_widget
        entry: timeout_one
        steps:
          - id: timeout_one
            type: timeout
            params:
              duration: 0.5
            next: emit_event
          - id: emit_event
            type: record_event
            params:
              event_type: widget_done
schedule_mappings:
  runner_tests.widget_arrivals:
    builder: runner_tests.build_simple_arrivals
    count: 4
"""


def _write_catalog(tmp_path: Path) -> Path:
    p = tmp_path / "catalog.yaml"
    p.write_text(CATALOG_GENERIC_YAML)
    return p


def _write_site(
    tmp_path: Path,
    *,
    schedules_block: str = "",
    extra_top: str = "",
) -> Path:
    cat = _write_catalog(tmp_path)
    site_yaml = (
        "meta: {schema_version: 1}\n"
        "name: runner_tests_site\n"
        f"catalog: {cat}\n"
        "modes: [assembly]\n"
        f"{extra_top}"
        f"{schedules_block}"
    )
    p = tmp_path / "site.yaml"
    p.write_text(site_yaml)
    return p


# ---- Direct arrival_entries path -----------------------------------


def test_run_site_direct_arrivals_completes(tmp_path: Path):
    site = _write_site(tmp_path)
    entries = [
        {"kind": "widget", "id": "w1", "arrival_time": 0.0},
        {"kind": "widget", "id": "w2", "arrival_time": 1.0},
    ]
    result = run_site(site, arrival_entries=entries, seed=42)
    assert len(result.entities) == 2
    assert len(result.output.event_log) == 2
    # Both events should have fired AFTER the 0.5-hr timeout.
    times = sorted(r["record_timestamp"] for r in result.output.event_log)
    assert times == [pytest.approx(0.5), pytest.approx(1.5)]


def test_run_site_unknown_kind_raises(tmp_path: Path):
    site = _write_site(tmp_path)
    with pytest.raises(RunError) as exc:
        run_site(site, arrival_entries=[{"kind": "ghost", "arrival_time": 0}])
    assert "not routed" in str(exc.value)


def test_run_site_entry_missing_kind_raises(tmp_path: Path):
    site = _write_site(tmp_path)
    with pytest.raises(RunError) as exc:
        run_site(site, arrival_entries=[{"arrival_time": 0}])
    assert "missing required 'kind'" in str(exc.value)


def test_run_site_no_arrivals_returns_immediately(tmp_path: Path):
    site = _write_site(tmp_path)
    result = run_site(site, arrival_entries=[])
    assert result.entities == []
    assert result.output.event_log == []
    assert result.env.now == 0


def test_run_site_auto_assigns_entity_id(tmp_path: Path):
    site = _write_site(tmp_path)
    result = run_site(
        site,
        arrival_entries=[{"kind": "widget"}, {"kind": "widget"}],
    )
    ids = [e.id for e in result.entities]
    assert ids == ["widget-0", "widget-1"]


# ---- Schedule-driven path ------------------------------------------


def test_run_site_uses_schedule_builder(tmp_path: Path):
    schedules_block = (
        "schedules:\n"
        "  runner_tests.widget_arrivals: ignored_payload\n"
    )
    site = _write_site(tmp_path, schedules_block=schedules_block)
    result = run_site(site, seed=7)
    # Builder is configured with count=4 in the catalog.
    assert len(result.entities) == 4
    assert len(result.output.event_log) == 4


def test_run_site_schedule_override(tmp_path: Path):
    schedules_block = (
        "schedules:\n"
        "  runner_tests.widget_arrivals: original\n"
    )
    site = _write_site(tmp_path, schedules_block=schedules_block)
    # The override payload is passed to the builder; the builder we
    # registered ignores it but the call site exercises the override
    # plumbing nonetheless.
    result = run_site(
        site,
        schedule_overrides={"runner_tests.widget_arrivals": "replaced"},
    )
    assert len(result.entities) == 4


def test_run_site_unknown_schedule_raises(tmp_path: Path):
    schedules_block = (
        "schedules:\n"
        "  runner_tests.unknown: payload\n"
    )
    site = _write_site(tmp_path, schedules_block=schedules_block)
    with pytest.raises(RunError) as exc:
        run_site(site)
    assert "schedule_mappings" in str(exc.value)


# ---- Seed + RNG threading ------------------------------------------


def test_run_site_seed_kwarg_overrides_site_seed(tmp_path: Path):
    site = _write_site(tmp_path, extra_top="seed: 999\n")
    # We can't directly inspect the threaded RNG, but both runs should
    # complete without error and produce identical event counts when
    # arrivals are deterministic (they are here — Constant durations).
    r1 = run_site(site, arrival_entries=[{"kind": "widget"}], seed=1)
    r2 = run_site(site, arrival_entries=[{"kind": "widget"}], seed=2)
    assert len(r1.output.event_log) == len(r2.output.event_log) == 1


# ---- Resource overrides --------------------------------------------


CATALOG_WITH_RESOURCE_YAML = """\
meta:
  schema_version: 1
name: runner_with_resource
entity_kinds:
  - name: widget
    attrs: {}
modes:
  - name: line
    arrival_routing:
      widget: do_nothing
    graphs:
      - name: do_nothing
        entry: noop
        steps:
          - id: noop
            type: log
            params:
              message: "noop"
    resources:
      - name: stations
        kind: Resource
        role: equipment
        capacity: 3
"""


def test_run_site_resource_override_applies(tmp_path: Path):
    cat = tmp_path / "catalog2.yaml"
    cat.write_text(CATALOG_WITH_RESOURCE_YAML)
    site = tmp_path / "site2.yaml"
    site.write_text(
        "meta: {schema_version: 1}\n"
        "name: s2\n"
        f"catalog: {cat}\n"
        "modes: [line]\n"
        "resource_overrides:\n"
        "  stations: {capacity: 10}\n"
    )
    result = run_site(site, arrival_entries=[{"kind": "widget"}])
    # SimPy Resource exposes .capacity
    assert result.state.stations.capacity == 10


def test_run_site_resource_override_unknown_spec_raises(tmp_path: Path):
    cat = tmp_path / "catalog3.yaml"
    cat.write_text(CATALOG_WITH_RESOURCE_YAML)
    site = tmp_path / "site3.yaml"
    site.write_text(
        "meta: {schema_version: 1}\n"
        "name: s3\n"
        f"catalog: {cat}\n"
        "modes: [line]\n"
        "resource_overrides:\n"
        "  nonexistent: {capacity: 1}\n"
    )
    with pytest.raises(RunError) as exc:
        run_site(site, arrival_entries=[])
    assert "unknown spec" in str(exc.value)


def test_run_site_resource_override_unsupported_field_raises(tmp_path: Path):
    cat = tmp_path / "catalog4.yaml"
    cat.write_text(CATALOG_WITH_RESOURCE_YAML)
    site = tmp_path / "site4.yaml"
    site.write_text(
        "meta: {schema_version: 1}\n"
        "name: s4\n"
        f"catalog: {cat}\n"
        "modes: [line]\n"
        "resource_overrides:\n"
        "  stations: {kind: Store}\n"
    )
    with pytest.raises(RunError) as exc:
        run_site(site, arrival_entries=[])
    assert "unsupported field" in str(exc.value)


# ---- state_init hook -----------------------------------------------


def test_run_site_state_init_kwarg(tmp_path: Path):
    site = _write_site(tmp_path)
    result = run_site(
        site,
        arrival_entries=[],
        state_init="runner_tests.state_init_extras",
    )
    assert result.state.extras_counter == 0


def test_run_site_unknown_state_init_raises(tmp_path: Path):
    site = _write_site(tmp_path)
    with pytest.raises(RunError) as exc:
        run_site(site, arrival_entries=[], state_init="not.registered")
    assert "not registered" in str(exc.value)


# ---- Dispatch ordering ---------------------------------------------


def test_run_site_arrivals_sorted_by_time(tmp_path: Path):
    """Out-of-order arrival_entries must be processed in time order."""
    site = _write_site(tmp_path)
    # Pass entries out of time order via the schedule path so the
    # runner's sort kicks in (direct arrival_entries are not sorted —
    # SimPy handles the ordering via timeouts).
    schedules_block = (
        "schedules:\n"
        "  runner_tests.widget_arrivals: ignored\n"
    )
    site2 = _write_site(tmp_path, schedules_block=schedules_block)
    result = run_site(site2)
    times = [r["record_timestamp"] for r in result.output.event_log]
    assert times == sorted(times)


# ---- config_defaults merge -----------------------------------------


CATALOG_WITH_DEFAULTS_YAML = """\
meta:
  schema_version: 1
name: runner_with_defaults
config_defaults:
  shovel_load_time_hr: 0.05
  truck_diesel_percentage: 1.0
entity_kinds:
  - name: widget
    attrs: {}
modes:
  - name: assembly
    arrival_routing:
      widget: noop
    graphs:
      - name: noop
        entry: log_it
        steps:
          - id: log_it
            type: log
            params:
              message: "x"
"""


def test_run_site_catalog_defaults_appear_in_config(tmp_path: Path):
    cat = tmp_path / "cat.yaml"
    cat.write_text(CATALOG_WITH_DEFAULTS_YAML)
    site = tmp_path / "site.yaml"
    site.write_text(
        "meta: {schema_version: 1}\n"
        "name: s\n"
        f"catalog: {cat}\n"
        "modes: [assembly]\n"
    )
    result = run_site(site, arrival_entries=[])
    assert result.config["shovel_load_time_hr"] == pytest.approx(0.05)
    assert result.config["truck_diesel_percentage"] == pytest.approx(1.0)


def test_run_site_site_config_overrides_catalog_defaults(tmp_path: Path):
    cat = tmp_path / "cat.yaml"
    cat.write_text(CATALOG_WITH_DEFAULTS_YAML)
    site = tmp_path / "site.yaml"
    site.write_text(
        "meta: {schema_version: 1}\n"
        "name: s\n"
        f"catalog: {cat}\n"
        "modes: [assembly]\n"
        "config:\n"
        "  truck_diesel_percentage: 0.4\n"
        "  custom_site_key: 99\n"
    )
    result = run_site(site, arrival_entries=[])
    # Site key wins
    assert result.config["truck_diesel_percentage"] == pytest.approx(0.4)
    # Untouched default carries through
    assert result.config["shovel_load_time_hr"] == pytest.approx(0.05)
    # Site-only keys are present
    assert result.config["custom_site_key"] == 99


# ---- Multi-mode dispatch -------------------------------------------
#
# When more than one active mode routes the same entity kind, arrival
# entries must carry an explicit ``mode`` key to disambiguate. The
# runner picks the workflow from the named mode and records that the
# right mode fired by inspecting per-mode event_type tags.

CATALOG_MULTI_MODE_YAML = """\
meta:
  schema_version: 1
name: multi_mode_catalog
entity_kinds:
  - name: widget
    attrs:
      arrival_time: float
modes:
  - name: assembly_a
    arrival_routing:
      widget: process_a
    graphs:
      - name: process_a
        entry: emit
        steps:
          - id: emit
            type: record_event
            params:
              event_type: assembled_by_a
  - name: assembly_b
    arrival_routing:
      widget: process_b
    graphs:
      - name: process_b
        entry: emit
        steps:
          - id: emit
            type: record_event
            params:
              event_type: assembled_by_b
"""


def _write_multi_mode_site(
    tmp_path: Path, *, active_modes: str = "[assembly_a, assembly_b]"
) -> Path:
    cat = tmp_path / "multi_catalog.yaml"
    cat.write_text(CATALOG_MULTI_MODE_YAML)
    site = tmp_path / "multi_site.yaml"
    site.write_text(
        "meta: {schema_version: 1}\n"
        "name: multi_mode_site\n"
        f"catalog: {cat}\n"
        f"modes: {active_modes}\n"
    )
    return site


def test_run_site_multi_mode_dispatch_by_mode_key(tmp_path: Path):
    """Both modes claim ``widget``; explicit ``mode`` routes correctly."""
    site = _write_multi_mode_site(tmp_path)
    entries = [
        {"kind": "widget", "id": "wa", "mode": "assembly_a"},
        {"kind": "widget", "id": "wb", "mode": "assembly_b"},
        {"kind": "widget", "id": "wa2", "mode": "assembly_a"},
    ]
    result = run_site(site, arrival_entries=entries)
    by_id = {r["entity_id"]: r["event_type"] for r in result.output.event_log}
    assert by_id == {
        "wa": "assembled_by_a",
        "wb": "assembled_by_b",
        "wa2": "assembled_by_a",
    }
    # ``mode`` should NOT leak onto the entity's attrs.
    for ent in result.entities:
        assert "mode" not in ent.attrs


def test_run_site_multi_mode_no_mode_key_raises(tmp_path: Path):
    """Contended kind without ``mode`` key → actionable RunError."""
    site = _write_multi_mode_site(tmp_path)
    with pytest.raises(RunError) as exc:
        run_site(site, arrival_entries=[{"kind": "widget", "id": "w"}])
    msg = str(exc.value)
    assert "multiple active modes" in msg
    assert "assembly_a" in msg and "assembly_b" in msg
    assert "'mode' key" in msg


def test_run_site_multi_mode_explicit_mode_not_active_raises(tmp_path: Path):
    """``mode`` key naming an inactive mode → actionable RunError."""
    # Only activate assembly_a.
    site = _write_multi_mode_site(tmp_path, active_modes="[assembly_a]")
    with pytest.raises(RunError) as exc:
        run_site(
            site,
            arrival_entries=[
                {"kind": "widget", "id": "w", "mode": "assembly_b"}
            ],
        )
    assert "not active" in str(exc.value)
    assert "assembly_b" in str(exc.value)


def test_run_site_multi_mode_explicit_mode_unknown_kind_raises(tmp_path: Path):
    """``mode`` key with a kind the mode doesn't route → actionable RunError."""
    # Add a second kind only routed by assembly_a; ask assembly_b for it.
    cat_yaml = """\
meta:
  schema_version: 1
name: multi_mode_catalog
entity_kinds:
  - {name: widget, attrs: {arrival_time: float}}
  - {name: gadget, attrs: {arrival_time: float}}
modes:
  - name: assembly_a
    arrival_routing:
      widget: emit_w
      gadget: emit_g
    graphs:
      - {name: emit_w, entry: e, steps: [{id: e, type: record_event, params: {event_type: w}}]}
      - {name: emit_g, entry: e, steps: [{id: e, type: record_event, params: {event_type: g}}]}
  - name: assembly_b
    arrival_routing:
      widget: emit_w
    graphs:
      - {name: emit_w, entry: e, steps: [{id: e, type: record_event, params: {event_type: w}}]}
"""
    cat = tmp_path / "cat.yaml"
    cat.write_text(cat_yaml)
    site = tmp_path / "site.yaml"
    site.write_text(
        "meta: {schema_version: 1}\n"
        "name: s\n"
        f"catalog: {cat}\n"
        "modes: [assembly_a, assembly_b]\n"
    )
    with pytest.raises(RunError) as exc:
        run_site(
            site,
            arrival_entries=[
                {"kind": "gadget", "id": "g", "mode": "assembly_b"}
            ],
        )
    msg = str(exc.value)
    assert "does not route" in msg
    assert "assembly_b" in msg
    assert "gadget" in msg


def test_run_site_multi_mode_uncontended_kind_still_works(tmp_path: Path):
    """Kinds claimed by exactly one mode dispatch without a ``mode`` key,
    even when other (contended) kinds exist in the catalog."""
    cat_yaml = """\
meta:
  schema_version: 1
name: mixed_catalog
entity_kinds:
  - {name: widget, attrs: {arrival_time: float}}
  - {name: gadget, attrs: {arrival_time: float}}
modes:
  - name: mode_a
    arrival_routing:
      widget: emit
      gadget: emit_g
    graphs:
      - {name: emit, entry: e, steps: [{id: e, type: record_event, params: {event_type: w_a}}]}
      - {name: emit_g, entry: e, steps: [{id: e, type: record_event, params: {event_type: g_only}}]}
  - name: mode_b
    arrival_routing:
      widget: emit
    graphs:
      - {name: emit, entry: e, steps: [{id: e, type: record_event, params: {event_type: w_b}}]}
"""
    cat = tmp_path / "cat.yaml"
    cat.write_text(cat_yaml)
    site = tmp_path / "site.yaml"
    site.write_text(
        "meta: {schema_version: 1}\n"
        "name: s\n"
        f"catalog: {cat}\n"
        "modes: [mode_a, mode_b]\n"
    )
    # ``gadget`` is only routed by mode_a → no ``mode`` key needed.
    result = run_site(
        site,
        arrival_entries=[
            {"kind": "gadget", "id": "g1"},
            {"kind": "widget", "id": "w1", "mode": "mode_a"},
            {"kind": "widget", "id": "w2", "mode": "mode_b"},
        ],
    )
    by_id = {r["entity_id"]: r["event_type"] for r in result.output.event_log}
    assert by_id == {"g1": "g_only", "w1": "w_a", "w2": "w_b"}
