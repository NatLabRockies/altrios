"""``altrios.workflow_engine`` — domain-neutral discrete-event workflow engine.

This package holds the simulation primitives that are not tied to any
particular industry. Freight intermodal terminal modeling lives in
:mod:`altrios.lifts` and is the first *catalog* against this engine, but
the engine itself knows nothing about cranes, trains, ships, or
containers — it only knows about :class:`ResourceSpec`-described capacity
pools and :class:`Entity`-described flow objects driven by step-graphs
declared in YAML.

Phase 3A.3 lifts the previously freight-coupled
``altrios.lifts.resources_decl`` and a newly-added entity module into
this package. Subsequent Phase 3 sub-phases add the step interpreter,
expression evaluator, YAML loader, distribution library, and python:-
callable registry.

See ``WORKFLOW_ENGINE_PLAN.md`` at the repo root for the design and
implementation plan that governs this package.
"""
from __future__ import annotations

from altrios.workflow_engine.catalog import Catalog, WorkflowMode
from altrios.workflow_engine.entities import (
    Entity,
    EntityKindSpec,
    KNOWN_ATTR_TYPES,
    merge_entity_kinds,
)
from altrios.workflow_engine.layout import Layout, LayoutNode
from altrios.workflow_engine.loader import LoaderError, load_catalog, load_site
from altrios.workflow_engine.output import OutputCollector
from altrios.workflow_engine.resources import (
    CapacitySpec,
    EventSpec,
    ItemFactory,
    KNOWN_ROLES,
    PartitionKeys,
    ResourceSpec,
    build_state_from_specs,
    merge_specs,
)
from altrios.workflow_engine.runner import RunError, RunResult, run_site

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
    "merge_entity_kinds",
    "merge_specs",
    "run_site",
]
