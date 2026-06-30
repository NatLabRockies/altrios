"""Engine-level ``Catalog`` and ``WorkflowMode`` dataclasses.

A :class:`Catalog` is the runtime form of a parsed-and-validated
catalog YAML bundle: the set of :class:`ResourceSpec`,
:class:`EntityKindSpec`, and :class:`WorkflowMode` instances that
together define one *site type* (freight terminal, mining operation,
airport, ...).

These are constructed by the YAML loader (Phase 3C.6 / ``loader.py``)
after schema validation. The interpreter consumes a Catalog plus a
:class:`Site` to build the per-run execution state.

``Catalog`` and ``WorkflowMode`` are immutable. ``Site`` is **not**
defined here — it lives next to its own pydantic model in
:mod:`schemas` for now, since Phase 3C does not yet need a runtime
representation distinct from the parsed model. A future
``runtime.py`` may add one when the run orchestration layer lands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .entities import EntityKindSpec
from .resources import ResourceSpec
from .steps import StepGraph


@dataclass(frozen=True)
class WorkflowMode:
    """One process-flow family within a catalog.

    Examples in the freight catalog: ``truck_rail``, ``rail_vessel``,
    ``vessel_truck``. A catalog typically defines several modes that
    can run concurrently against shared resource pools.

    Attributes
    ----------
    name
        Identifier (used as the namespace prefix for schedule mappings,
        e.g. ``"truck_rail.train_arrivals"``).
    arrival_routing
        Map from entity kind name to step-graph name. The dispatcher
        looks up the arriving entity's ``kind`` and starts the named
        graph for it.
    graphs
        Map from graph name to :class:`StepGraph`. All graphs the mode
        owns, including those reached only via ``spawn``.
    resource_specs
        Resource pools this mode declares. The engine unions specs
        across active modes; identical names from different modes must
        have identical specs (see :func:`resources.merge_specs`).
    """

    name: str
    arrival_routing: Mapping[str, str]
    graphs: Mapping[str, StepGraph]
    resource_specs: tuple[ResourceSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"WorkflowMode.name must be a non-empty str, got {self.name!r}."
            )
        for kind, graph_name in self.arrival_routing.items():
            if graph_name not in self.graphs:
                raise ValueError(
                    f"WorkflowMode {self.name!r}: arrival_routing maps "
                    f"entity kind {kind!r} to unknown graph {graph_name!r}. "
                    f"Available graphs: {sorted(self.graphs)}."
                )
        # Cross-graph spawn references — every ``spawn`` step's
        # ``graph`` param must name a graph in this mode (or in another
        # mode of the catalog, but cross-mode spawns are linted at
        # catalog-build time, not here).
        object.__setattr__(self, "arrival_routing", MappingProxyType(dict(self.arrival_routing)))
        object.__setattr__(self, "graphs", MappingProxyType(dict(self.graphs)))


@dataclass(frozen=True)
class Catalog:
    """A parsed, validated bundle of one site type's definitions.

    Catalogs are domain-agnostic by construction: the engine doesn't
    distinguish a freight catalog from a mining catalog. A site file
    selects which catalog to load and which modes to activate.

    Attributes
    ----------
    name
        Catalog identifier, e.g. ``"freight_intermodal"``.
    schema_version
        Always 1 in Phase 3; future migrations will add other versions.
    modes
        Tuple of :class:`WorkflowMode` instances declared by the
        catalog. A site selects a subset of these to activate.
    entity_kinds
        Map from kind name to :class:`EntityKindSpec`. Each mode's
        ``arrival_routing`` keys must appear here.
    consumption_rates
        Free-form dict of consumption-rate tables (keyed by resource
        type or by some catalog-defined scheme). Consumed by
        ``record_consumption`` steps via expression access to
        ``config.consumption_rates`` (the loader stitches the table
        into the run's config dict).
    schedule_mappings
        Free-form dict describing how external schedules (CSV files,
        DataFrames) map to entity arrival streams. Interpreted by the
        catalog's ``python_module`` rather than by the engine itself.
    python_module
        Dotted Python module path; the loader imports this module so
        its module-level ``@register(...)`` decorators run and populate
        the callable registry. ``None`` for catalogs that don't need
        Python helpers.
    """

    name: str
    schema_version: int
    modes: tuple[WorkflowMode, ...]
    entity_kinds: Mapping[str, EntityKindSpec]
    consumption_rates: Mapping[str, Any] = field(default_factory=dict)
    schedule_mappings: Mapping[str, Any] = field(default_factory=dict)
    python_module: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"Catalog.name must be a non-empty str, got {self.name!r}."
            )
        if self.schema_version != 1:
            raise ValueError(
                f"Catalog {self.name!r}: schema_version must be 1 (only v1 "
                f"is supported in Phase 3), got {self.schema_version!r}."
            )
        seen_modes: set[str] = set()
        for mode in self.modes:
            if mode.name in seen_modes:
                raise ValueError(
                    f"Catalog {self.name!r}: duplicate mode name {mode.name!r}."
                )
            seen_modes.add(mode.name)
            # Every routed entity kind must be declared.
            for kind in mode.arrival_routing:
                if kind not in self.entity_kinds:
                    raise ValueError(
                        f"Catalog {self.name!r}: mode {mode.name!r} routes "
                        f"unknown entity kind {kind!r}. Declared kinds: "
                        f"{sorted(self.entity_kinds)}."
                    )
        object.__setattr__(
            self, "entity_kinds", MappingProxyType(dict(self.entity_kinds))
        )
        object.__setattr__(
            self, "consumption_rates", MappingProxyType(dict(self.consumption_rates))
        )
        object.__setattr__(
            self, "schedule_mappings", MappingProxyType(dict(self.schedule_mappings))
        )

    def mode(self, name: str) -> WorkflowMode:
        """Look up a mode by name; raise ``KeyError`` with the list of
        available modes if missing."""
        for m in self.modes:
            if m.name == name:
                return m
        raise KeyError(
            f"No mode named {name!r} in catalog {self.name!r}. "
            f"Available: {sorted(m.name for m in self.modes)}."
        )
