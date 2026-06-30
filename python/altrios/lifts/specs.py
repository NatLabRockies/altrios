"""Catalog of declarative :class:`ResourceSpec` instances for Phase 1 modes.

Each spec is a module-level constant referenced from one or more
:class:`TerminalMode` registrations (see ``terminal_sim.py``). When two
modes reference the same spec object (by identity, not just by name), the
dispatcher's :func:`merge_specs` deduplicates them and the resulting
SimPy primitive is shared across both modes — see ``resources_decl.py``.

Capacity callables read from the ``config`` mapping. The ``vessel:`` and
``yard_stack:`` sections are canonical (see ``resources/config.yaml``);
the ``.get(...)`` defaults below remain as a defensive fallback so this
module can still import against a stripped-down config (e.g. in tests).
"""
from __future__ import annotations

from typing import Any, Mapping

from altrios.lifts.classes import (
    chassis,
    rtg,
    sts_crane,
    top_pick,
    yard_tractor,
)
from altrios.workflow_engine import ResourceSpec


# ---------------------------------------------------------------------------
# Capacity helpers (Phase 1H may rewrite these to remove the fallbacks)
# ---------------------------------------------------------------------------

def _vessel_cfg(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("vessel", {}) or {}


def _yard_stack_cfg(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("yard_stack", {}) or {}


def _gates_cfg(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("gates", {}) or {}


def _yard_cfg(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("yard", {}) or {}


def _terminal_cfg(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return config.get("terminal", {}) or {}


def _cranes_per_track(config: Mapping[str, Any], track_id: int) -> int:
    """Match :class:`Terminal.__init__`'s scalar-or-list interpretation."""
    cfg = _terminal_cfg(config).get("cranes_per_track", 1)
    if isinstance(cfg, (list, tuple)):
        return int(cfg[track_id - 1])
    return int(cfg)


# ---------------------------------------------------------------------------
# Track / gate specs (currently live on TerminalState as direct attributes;
# migrated to spec form in Phase 1G)
# ---------------------------------------------------------------------------

TRACKS = ResourceSpec(
    name="tracks",
    kind="Store",
    role="infrastructure",
    capacity=lambda c, s: _yard_cfg(c).get("track_number", 5),
    init_items=lambda c, s: list(
        range(1, _yard_cfg(c).get("track_number", 5) + 1)
    ),
)

IN_GATES = ResourceSpec(
    name="in_gates",
    kind="Resource",
    role="infrastructure",
    capacity=lambda c, s: _gates_cfg(c).get("in_gate_numbers", 3),
)

OUT_GATES = ResourceSpec(
    name="out_gates",
    kind="Resource",
    role="infrastructure",
    capacity=lambda c, s: _gates_cfg(c).get("out_gate_numbers", 3),
)

# Rail-track RTGs: one Store per track, partitioned by track_id. Holds the
# rtg objects today's ``cranes_by_track`` holds (no behavioral change).
RAIL_TRACK_RTGS_BY_TRACK = ResourceSpec(
    name="rail_track_rtgs_by_track",
    kind="Store",
    role="equipment",
    capacity=lambda c, s: _cranes_per_track(c, s["_partition_key"]),
    partition_by=lambda c, s: range(1, _yard_cfg(c).get("track_number", 5) + 1),
    init_items=lambda c, s: [
        rtg(type="Hybrid", id=i, pool="rail_track",
            track_id=s["_partition_key"])
        for i in range(1, _cranes_per_track(c, s["_partition_key"]) + 1)
    ],
)

# ---------------------------------------------------------------------------
# Vessel-side specs (referenced by rail_vessel and vessel_truck)
# ---------------------------------------------------------------------------

BERTHS = ResourceSpec(
    name="berths",
    kind="Resource",
    role="infrastructure",
    capacity=lambda c, s: _vessel_cfg(c).get("berth_number", 2),
)

STS_CRANES_BY_BERTH = ResourceSpec(
    name="sts_cranes_by_berth",
    kind="Store",
    role="equipment",
    capacity=lambda c, s: _vessel_cfg(c).get("sts_cranes_per_berth", 2),
    partition_by=lambda c, s: range(1, _vessel_cfg(c).get("berth_number", 2) + 1),
    init_items=lambda c, s: [
        sts_crane(
            type="Diesel", id=i, berth_id=s["_partition_key"],
        )
        for i in range(1, _vessel_cfg(c).get("sts_cranes_per_berth", 2) + 1)
    ],
)

# ---------------------------------------------------------------------------
# Main stack specs (referenced by all three Phase 1 modes)
# ---------------------------------------------------------------------------

MAIN_STACK_RTGS = ResourceSpec(
    name="main_stack_rtgs",
    kind="Store",
    role="equipment",
    capacity=lambda c, s: _yard_stack_cfg(c).get("main_stack_rtg_count", 6),
    init_items=lambda c, s: [
        rtg(type="Diesel", id=i, pool="main_stack")
        for i in range(1, _yard_stack_cfg(c).get("main_stack_rtg_count", 6) + 1)
    ],
)

TOP_PICKS = ResourceSpec(
    name="top_picks",
    kind="Store",
    role="equipment",
    capacity=lambda c, s: _yard_stack_cfg(c).get("top_pick_count", 2),
    init_items=lambda c, s: [
        top_pick(type="Diesel", id=i, safety_car_id=i)
        for i in range(1, _yard_stack_cfg(c).get("top_pick_count", 2) + 1)
    ],
)

CONTAINER_STACK = ResourceSpec(
    name="container_stack",
    kind="Store",
    role="storage",
    capacity=lambda c, s: _yard_stack_cfg(c).get("stack_capacity", 500),
)

PARKING_CHASSIS_SLOTS = ResourceSpec(
    name="parking_chassis_slots",
    kind="Resource",
    role="infrastructure",
    capacity=lambda c, s: _yard_stack_cfg(c).get("parking_chassis_slot_count", 30),
)

# ---------------------------------------------------------------------------
# Yard tractor pools
# ---------------------------------------------------------------------------

MAIN_YARD_TRACTORS = ResourceSpec(
    name="main_yard_tractors",
    kind="Store",
    role="equipment",
    capacity=lambda c, s: _yard_stack_cfg(c).get("main_yard_tractor_count", 12),
    init_items=lambda c, s: [
        yard_tractor(type="Diesel", id=i, pool="main")
        for i in range(1, _yard_stack_cfg(c).get("main_yard_tractor_count", 12) + 1)
    ],
)

RAIL_YARD_TRACTORS = ResourceSpec(
    name="rail_yard_tractors",
    kind="Store",
    role="equipment",
    capacity=lambda c, s: _yard_stack_cfg(c).get("rail_yard_tractor_count", 8),
    init_items=lambda c, s: [
        yard_tractor(type="Diesel", id=i, pool="rail")
        for i in range(1, _yard_stack_cfg(c).get("rail_yard_tractor_count", 8) + 1)
    ],
)

# ---------------------------------------------------------------------------
# Chassis pools
# ---------------------------------------------------------------------------

TERMINAL_CHASSIS_POOL = ResourceSpec(
    name="terminal_chassis_pool",
    kind="Store",
    role="equipment",
    capacity=lambda c, s: _yard_stack_cfg(c).get("terminal_chassis_count", 50),
    init_items=lambda c, s: [
        chassis(type="Standard", id=i, pool="terminal")
        for i in range(1, _yard_stack_cfg(c).get("terminal_chassis_count", 50) + 1)
    ],
)

ROAD_CHASSIS_POOL = ResourceSpec(
    name="road_chassis_pool",
    kind="Store",
    role="equipment",
    capacity=lambda c, s: _yard_stack_cfg(c).get("road_chassis_pool_count", 30),
    init_items=lambda c, s: [
        chassis(type="Standard", id=i, pool="road")
        for i in range(1, _yard_stack_cfg(c).get("road_chassis_pool_count", 30) + 1)
    ],
)


# ---------------------------------------------------------------------------
# Mode -> spec bundles. Each tuple is the ResourceSpec list that one Phase 1
# mode will pass as its ``TerminalMode.resource_specs``. Cross-mode sharing
# happens through repeated spec objects in different bundles; the dispatcher
# deduplicates by name with the agreement check in ``merge_specs``.
# ---------------------------------------------------------------------------

TRUCK_RAIL_SPECS: tuple[ResourceSpec, ...] = (
    TRACKS,
    IN_GATES,
    OUT_GATES,
    RAIL_TRACK_RTGS_BY_TRACK,
    MAIN_STACK_RTGS,
    TOP_PICKS,
    CONTAINER_STACK,
    PARKING_CHASSIS_SLOTS,
    RAIL_YARD_TRACTORS,
    TERMINAL_CHASSIS_POOL,
    ROAD_CHASSIS_POOL,
)

RAIL_VESSEL_SPECS: tuple[ResourceSpec, ...] = (
    TRACKS,
    BERTHS,
    STS_CRANES_BY_BERTH,
    RAIL_TRACK_RTGS_BY_TRACK,
    MAIN_STACK_RTGS,
    TOP_PICKS,
    CONTAINER_STACK,
    PARKING_CHASSIS_SLOTS,
    MAIN_YARD_TRACTORS,
    RAIL_YARD_TRACTORS,
    TERMINAL_CHASSIS_POOL,
)

VESSEL_TRUCK_SPECS: tuple[ResourceSpec, ...] = (
    BERTHS,
    STS_CRANES_BY_BERTH,
    IN_GATES,
    OUT_GATES,
    MAIN_STACK_RTGS,
    TOP_PICKS,
    CONTAINER_STACK,
    PARKING_CHASSIS_SLOTS,
    MAIN_YARD_TRACTORS,
    TERMINAL_CHASSIS_POOL,
    ROAD_CHASSIS_POOL,
)
