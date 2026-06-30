"""Pydantic v2 schemas for YAML-loaded workflow definitions.

These models bridge raw parsed YAML (with brace-expressions already
converted to :class:`Expression` instances by ``yaml_expressions``)
and the engine's frozen dataclasses (:class:`Step`, :class:`StepGraph`,
:class:`ResourceSpec`).

Validation responsibilities split:

- **Schema (here)**: shape of the YAML — required keys present, types
  correct, enums in the allowed set, no unknown extras.
- **Engine dataclasses**: cross-step invariants — entry exists, step
  ids match dict keys, next-pointers resolve, etc. (already enforced
  by ``StepGraph.__post_init__``).
- **Interpreter (runtime)**: per-primitive param semantics.

This split keeps each layer focused: pydantic catches typos and shape
issues at load time with rich error messages; the dataclasses catch
graph-integrity bugs; the interpreter catches semantic errors at run
time.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .catalog import Catalog, WorkflowMode
from .entities import EntityKindSpec
from .expressions import Expression
from .resources import KNOWN_ROLES, ResourceSpec
from .steps import Step, StepGraph


# Schema version v1 is the only supported version in Phase 3.
SCHEMA_VERSION_V1 = 1


# Subset of step.type names that the engine knows how to execute.
# Keep in sync with ``interpreter.build_default_primitives``; catalog
# YAML using an unknown type is rejected at load time so authors get a
# clear "did you mean ..." instead of a deferred runtime error.
KNOWN_PRIMITIVES: frozenset[str] = frozenset({
    "bind", "set_attr", "branch", "assert", "log", "timeout",
    "request", "release", "transfer",
    "record_event", "record_resource_event", "record_consumption",
    "parallel", "loop", "spawn",
    "make_event", "wait_event", "trigger_event",
    "python",
})


# Subset of ResourceSpec.kind values accepted by the engine.
KNOWN_RESOURCE_KINDS: frozenset[str] = frozenset({
    "Store", "Resource", "Container",
})


class StepModel(BaseModel):
    """Schema for one step inside a workflow graph.

    YAML shape::

        - id: load_crane
          type: request
          params:
            pool: cranes
            bind: crane
          next: hold
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    id: str = Field(min_length=1, description="Unique step id within graph.")
    type: str = Field(min_length=1, description="Primitive name.")
    params: dict[str, Any] = Field(default_factory=dict)
    next: Optional[str] = Field(default=None, min_length=1)

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v not in KNOWN_PRIMITIVES:
            close = sorted(
                p for p in KNOWN_PRIMITIVES if v.lower() in p or p in v.lower()
            )
            hint = f" Did you mean one of {close}?" if close else (
                f" Known primitives: {sorted(KNOWN_PRIMITIVES)}."
            )
            raise ValueError(f"Unknown step type {v!r}.{hint}")
        return v

    def to_engine(self) -> Step:
        return Step(id=self.id, type=self.type, params=self.params, next=self.next)


class StepGraphModel(BaseModel):
    """Schema for one named workflow graph.

    YAML shape::

        name: container_arrival
        entry: receive
        steps:
          - {id: receive, type: timeout, params: {duration: 5}}
          - ...
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    entry: str = Field(min_length=1)
    steps: list[StepModel] = Field(min_length=1)

    @field_validator("steps")
    @classmethod
    def _ids_unique(cls, v: list[StepModel]) -> list[StepModel]:
        ids = [s.id for s in v]
        seen: set[str] = set()
        dups: list[str] = []
        for sid in ids:
            if sid in seen:
                dups.append(sid)
            seen.add(sid)
        if dups:
            raise ValueError(
                f"Duplicate step id(s) within graph: {sorted(set(dups))}."
            )
        return v

    def to_engine(self) -> StepGraph:
        # The engine dataclass enforces entry-exists and next-resolves;
        # the pydantic layer only checks intra-step shape. Catch
        # mismatches between the two layers with a clear wrapper.
        engine_steps = {s.id: s.to_engine() for s in self.steps}
        return StepGraph(name=self.name, entry=self.entry, steps=engine_steps)


class ResourceSpecModel(BaseModel):
    """Schema for one declared resource pool.

    YAML shape::

        - name: cranes
          kind: Resource
          role: equipment
          capacity: 4

    Partitioned pools and custom ``init_items`` factories require a
    Python callable; those are wired in at catalog-load time (3C.6)
    by resolving ``partition_by_python:`` / ``init_items_python:``
    against the catalog's registered ``python_module``. For 3C.3 we
    only validate the YAML-native fields; the Python-hook fields are
    declared here as optional dotted-name strings.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    role: str = Field(min_length=1)
    capacity: int = Field(default=1, ge=0)
    partition_by_python: Optional[str] = Field(default=None, min_length=1)
    init_items_python: Optional[str] = Field(default=None, min_length=1)

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, v: str) -> str:
        if v not in KNOWN_RESOURCE_KINDS:
            raise ValueError(
                f"Unknown resource kind {v!r}; must be one of "
                f"{sorted(KNOWN_RESOURCE_KINDS)}."
            )
        return v

    @field_validator("role")
    @classmethod
    def _warn_or_accept_role(cls, v: str) -> str:
        # ``role`` is open-ended per the engine's docstring, but we at
        # least require non-empty (Field above) and surface a hint for
        # likely typos of conventional values.
        if v not in KNOWN_ROLES:
            # Don't reject — open vocab — but the loader could log a
            # debug message. Silent acceptance for now keeps the
            # schema permissive.
            pass
        return v

    def to_engine(
        self,
        *,
        partition_by_resolver=None,
        init_items_resolver=None,
    ) -> ResourceSpec:
        """Build the engine dataclass.

        ``partition_by_resolver`` and ``init_items_resolver`` are
        callables ``(dotted_name: str) -> Callable | None`` supplied
        by the catalog loader; if either of the Python-hook fields is
        set on the model, the corresponding resolver MUST also be
        provided. Used to defer python-helper lookup until the
        catalog's python_module has been imported and registered.
        """
        partition_by = None
        init_items = None
        if self.partition_by_python is not None:
            if partition_by_resolver is None:
                raise ValueError(
                    f"ResourceSpec {self.name!r} declares "
                    f"partition_by_python={self.partition_by_python!r} but no "
                    "resolver was provided to to_engine()."
                )
            partition_by = partition_by_resolver(self.partition_by_python)
        if self.init_items_python is not None:
            if init_items_resolver is None:
                raise ValueError(
                    f"ResourceSpec {self.name!r} declares "
                    f"init_items_python={self.init_items_python!r} but no "
                    "resolver was provided to to_engine()."
                )
            init_items = init_items_resolver(self.init_items_python)
        return ResourceSpec(
            name=self.name,
            kind=self.kind,
            role=self.role,
            capacity=self.capacity,
            partition_by=partition_by,
            init_items=init_items,
        )


# ---- Entity kinds ----------------------------------------------------


class EntityKindSpecModel(BaseModel):
    """Schema for one entity kind in a catalog.

    YAML shape::

        - name: container
          description: One intermodal container moving through the terminal.
          attrs:
            arrival_time: float
            origin: str
            destination: str
            weight_t: float
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    attrs: dict[str, str] = Field(default_factory=dict)
    description: str = ""

    def to_engine(self) -> EntityKindSpec:
        return EntityKindSpec(
            name=self.name, attrs=self.attrs, description=self.description
        )


# ---- Workflow mode --------------------------------------------------


class WorkflowModeModel(BaseModel):
    """Schema for one process-flow family within a catalog.

    YAML shape::

        - name: truck_rail
          arrival_routing:
            container: container_arrival
            train: train_arrival
          graphs:
            - name: container_arrival
              entry: receive
              steps: [...]
            - name: train_arrival
              entry: arrive_yard
              steps: [...]
          resources:
            - {name: cranes, kind: Resource, role: equipment, capacity: 4}
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    arrival_routing: dict[str, str] = Field(default_factory=dict)
    graphs: list[StepGraphModel] = Field(min_length=1)
    resources: list[ResourceSpecModel] = Field(default_factory=list)

    @field_validator("graphs")
    @classmethod
    def _graph_names_unique(cls, v: list[StepGraphModel]) -> list[StepGraphModel]:
        names = [g.name for g in v]
        if len(names) != len(set(names)):
            dups = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate graph name(s) in mode: {dups}.")
        return v

    @field_validator("resources")
    @classmethod
    def _resource_names_unique(
        cls, v: list[ResourceSpecModel]
    ) -> list[ResourceSpecModel]:
        names = [r.name for r in v]
        if len(names) != len(set(names)):
            dups = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate resource name(s) in mode: {dups}.")
        return v

    @model_validator(mode="after")
    def _routing_targets_exist(self) -> "WorkflowModeModel":
        graph_names = {g.name for g in self.graphs}
        for kind, graph_name in self.arrival_routing.items():
            if graph_name not in graph_names:
                raise ValueError(
                    f"Mode {self.name!r}: arrival_routing[{kind!r}]"
                    f"={graph_name!r} names no declared graph. "
                    f"Available: {sorted(graph_names)}."
                )
        return self

    def to_engine(
        self,
        *,
        partition_by_resolver=None,
        init_items_resolver=None,
    ) -> WorkflowMode:
        graphs = {g.name: g.to_engine() for g in self.graphs}
        resource_specs = tuple(
            r.to_engine(
                partition_by_resolver=partition_by_resolver,
                init_items_resolver=init_items_resolver,
            )
            for r in self.resources
        )
        return WorkflowMode(
            name=self.name,
            arrival_routing=dict(self.arrival_routing),
            graphs=graphs,
            resource_specs=resource_specs,
        )


# ---- Catalog --------------------------------------------------------


class MetaModel(BaseModel):
    """The ``meta:`` block required at the top of every YAML file."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(description="Must equal 1 in Phase 3.")

    @field_validator("schema_version")
    @classmethod
    def _v1_only(cls, v: int) -> int:
        if v != SCHEMA_VERSION_V1:
            raise ValueError(
                f"meta.schema_version must be {SCHEMA_VERSION_V1} "
                f"(only v1 is supported in Phase 3); got {v!r}."
            )
        return v


class CatalogModel(BaseModel):
    """Schema for a catalog YAML file.

    YAML shape::

        meta: {schema_version: 1}
        name: freight_intermodal
        description: Intermodal terminal catalog.
        python_module: altrios.lifts.python_helpers
        entity_kinds:
          - name: container
            attrs: {weight_t: float, origin: str}
        modes:
          - name: truck_rail
            ...
        consumption_rates: {...}
        schedule_mappings: {...}
    """

    model_config = ConfigDict(extra="forbid")

    meta: MetaModel
    name: str = Field(min_length=1)
    description: str = ""
    python_module: Optional[str] = Field(default=None, min_length=1)
    entity_kinds: list[EntityKindSpecModel] = Field(default_factory=list)
    modes: list[WorkflowModeModel] = Field(default_factory=list)
    consumption_rates: dict[str, Any] = Field(default_factory=dict)
    schedule_mappings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("entity_kinds")
    @classmethod
    def _kind_names_unique(
        cls, v: list[EntityKindSpecModel]
    ) -> list[EntityKindSpecModel]:
        names = [k.name for k in v]
        if len(names) != len(set(names)):
            dups = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate entity_kind name(s): {dups}.")
        return v

    @field_validator("modes")
    @classmethod
    def _mode_names_unique(
        cls, v: list[WorkflowModeModel]
    ) -> list[WorkflowModeModel]:
        names = [m.name for m in v]
        if len(names) != len(set(names)):
            dups = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate mode name(s): {dups}.")
        return v

    def to_engine(
        self,
        *,
        partition_by_resolver=None,
        init_items_resolver=None,
    ) -> Catalog:
        kinds = {k.name: k.to_engine() for k in self.entity_kinds}
        modes = tuple(
            m.to_engine(
                partition_by_resolver=partition_by_resolver,
                init_items_resolver=init_items_resolver,
            )
            for m in self.modes
        )
        return Catalog(
            name=self.name,
            schema_version=self.meta.schema_version,
            modes=modes,
            entity_kinds=kinds,
            consumption_rates=dict(self.consumption_rates),
            schedule_mappings=dict(self.schedule_mappings),
            python_module=self.python_module,
        )


# ---- Site -----------------------------------------------------------


class LayoutNodeModel(BaseModel):
    """One named node in a 2-D site layout. ``z`` is parsed but unused
    in v1 (reserved for future gradient/lift modeling)."""

    model_config = ConfigDict(extra="forbid")

    x: float
    y: float
    z: Optional[float] = None


class LayoutModel(BaseModel):
    """Schema for the optional ``layout:`` block in a site file.

    YAML shape::

        layout:
          nodes:
            berth_1: {x: 0,   y: 0}
            stack_A: {x: 380, y: 50}
            track_1: {x: 600, y: 50, z: 2.5}
    """

    model_config = ConfigDict(extra="forbid")

    nodes: dict[str, LayoutNodeModel] = Field(default_factory=dict)


class SiteModel(BaseModel):
    """Schema for a site YAML file.

    YAML shape::

        meta: {schema_version: 1}
        name: rotterdam_intermodal
        catalog: altrios.lifts
        modes: [truck_rail, rail_vessel]
        config:
          crane_count: 4
          shift_hours: 8
        layout:
          nodes:
            berth_1: {x: 0,   y: 0}
            stack_A: {x: 380, y: 50}
        resource_overrides:
          sts_crane: {capacity: 8}
        schedules:
          truck_rail.train_arrivals: data/trains.csv
        seed: 42

    ``extends:`` (for site-extends-site inheritance) is recognized
    here but resolved by the loader at file-load time, not by the
    schema. By the time a SiteModel is constructed, any extends-merge
    has already happened.
    """

    model_config = ConfigDict(extra="forbid")

    meta: MetaModel
    name: str = Field(min_length=1)
    catalog: str = Field(min_length=1)
    modes: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    layout: Optional[LayoutModel] = None
    resource_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    schedules: dict[str, Any] = Field(default_factory=dict)
    seed: Optional[int] = None
    extends: Optional[str] = Field(default=None, min_length=1)

    @field_validator("modes")
    @classmethod
    def _modes_unique(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            dups = sorted({m for m in v if v.count(m) > 1})
            raise ValueError(f"Duplicate mode(s) in site.modes: {dups}.")
        return v
