"""``altrios.lifts.workflow_engine`` — domain-neutral discrete-event workflow engine.

This package holds the simulation primitives that are not tied to any
particular industry. Freight intermodal terminal modeling lives in
:mod:`altrios.lifts.terminal` and is the first *catalog* against this
engine; an open-pit mining example lives in :mod:`altrios.lifts.mine`.
The engine itself knows nothing about cranes, trains, ships, or
containers — it only knows about :class:`ResourceSpec`-described capacity
pools and :class:`Entity`-described flow objects driven by step-graphs
declared in YAML.

See ``workflow-engine-plan.md`` alongside this package for the design
and implementation plan that governs it.
"""
from __future__ import annotations

from altrios.lifts.workflow_engine.catalog import Catalog, WorkflowMode
from altrios.lifts.workflow_engine.entities import (
    Entity,
    EntityKindSpec,
    KNOWN_ATTR_TYPES,
    merge_entity_kinds,
)
from altrios.lifts.workflow_engine.layout import Layout, LayoutNode
from altrios.lifts.workflow_engine.loader import (
    LoaderError,
    load_catalog,
    load_site,
    make_runner,
    make_site_path,
)
from altrios.lifts.workflow_engine.output import OutputCollector
from altrios.lifts.workflow_engine.resources import (
    CapacitySpec,
    EventSpec,
    ItemFactory,
    KNOWN_ROLES,
    PartitionKeys,
    ResourceSpec,
    build_state_from_specs,
    merge_specs,
)
from altrios.lifts.workflow_engine.runner import RunError, RunResult, run_site

__all__ = [
    "CapacitySpec",
    "Catalog",
    "Entity",
    "EntityKindSpec",
    "EventSpec",
    "ItemFactory",
    "KNOWN_ATTR_TYPES",
    "KNOWN_ROLES",
    "Layout",
    "LayoutNode",
    "LoaderError",
    "OutputCollector",
    "PartitionKeys",
    "ResourceSpec",
    "RunError",
    "RunResult",
    "WorkflowMode",
    "build_state_from_specs",
    "load_catalog",
    "load_site",
    "make_runner",
    "make_site_path",
    "merge_entity_kinds",
    "merge_specs",
    "run_site",
]
