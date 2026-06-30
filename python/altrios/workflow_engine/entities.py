"""Declarative entity definitions for the workflow engine.

An :class:`Entity` is a flow object — something with identity that moves
through a workflow, accrues events, and is acted upon by resources. In
freight terms: containers, trains, vessels, drayage trucks. In mining: ore
loads, haul trucks. In airports: aircraft, passenger groups.

Entities are **distinct from**
:class:`~altrios.workflow_engine.resources.ResourceSpec` instances.
Resources are seized capacity (cranes, tracks, berths). Entities flow.
The two interact: a workflow ``request``s a resource, uses it on an
entity, and ``release``s it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


# Conventional Python type names that may appear in an entity kind's
# ``attrs:`` declaration. The YAML loader maps these strings to actual
# Python types for runtime coercion; the interpreter uses the same set
# to validate ``{entity.foo}`` expression references at load time.
# Catalogs may declare custom kind-attr types — the engine does not
# enforce this set.
KNOWN_ATTR_TYPES: frozenset[str] = frozenset({
    "str",
    "int",
    "float",
    "bool",
    "list",
    "dict",
    "datetime",   # parsed as ISO 8601 by the YAML loader
})


@dataclass
class Entity:
    """One flow object travelling through a workflow.

    Entities are mutable (workflows update ``attrs`` via ``set_attr`` steps
    or via Python helpers). Identity is by ``id`` within a single run.

    Parameters
    ----------
    id
        Unique within a run. Catalogs typically assemble this from the
        entity kind + an arrival/source index (e.g. ``"container-123"``,
        ``"train-7"``) but the engine itself only requires uniqueness.
    kind
        Name of the :class:`EntityKindSpec` this entity instantiates
        (e.g. ``"container"``, ``"train"``, ``"vessel"``).
    attrs
        Free-form attribute bag. Keys typically correspond to the names
        declared in the matching :class:`EntityKindSpec`; the engine does
        not currently enforce this at runtime.
    parent_id
        Optional reference to a parent entity. Used by ``spawn`` steps
        that create child entities (e.g. an arriving train spawns one
        entity per container in its consist). The engine does not chase
        this reference automatically; helpers and ``python:`` callables
        may use it.
    """

    id: str
    kind: str
    attrs: dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError(f"Entity.id must be a non-empty string; got {self.id!r}")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError(
                f"Entity(id={self.id!r}).kind must be a non-empty string; "
                f"got {self.kind!r}"
            )


@dataclass(frozen=True)
class EntityKindSpec:
    """Declarative type for one entity kind in a catalog.

    Each catalog declares the entity kinds its workflows manipulate.
    Used by tooling for static validation and by Python-side code that
    needs to attach kind names without engine churn.

    Parameters
    ----------
    name
        Kind identifier, e.g. ``"container"``. Referenced by
        ``arrival_routing:`` blocks and by ``Entity.kind``.
    attrs
        Mapping of attribute name to type name. Type names are typically
        drawn from :data:`KNOWN_ATTR_TYPES` (``"str"`` / ``"int"`` /
        ``"float"`` / ...). The engine does not enforce membership;
        catalogs may declare custom types.
    description
        Human-readable purpose statement for diagnostics and generated
        documentation.
    """

    name: str
    attrs: Mapping[str, str] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"EntityKindSpec.name must be a non-empty string; got {self.name!r}"
            )
        for attr_name, type_name in self.attrs.items():
            if not isinstance(attr_name, str) or not attr_name:
                raise ValueError(
                    f"EntityKindSpec(name={self.name!r}) has an invalid attr "
                    f"name: {attr_name!r}"
                )
            if not isinstance(type_name, str) or not type_name:
                raise ValueError(
                    f"EntityKindSpec(name={self.name!r}) attr {attr_name!r} "
                    f"must declare a non-empty type name; got {type_name!r}"
                )


def merge_entity_kinds(
    kinds_by_source: Mapping[str, "Mapping[str, EntityKindSpec]"],
) -> dict[str, EntityKindSpec]:
    """Union ``EntityKindSpec`` mappings across multiple sources (typically
    multiple modes within one catalog, or multiple catalogs in a future
    multi-catalog scenario). Two specs with the same ``name`` must agree
    on ``attrs`` and ``description``; otherwise a ``ValueError`` is
    raised so the caller knows which two sources conflict.
    """
    merged: dict[str, EntityKindSpec] = {}
    contributors: dict[str, list[str]] = {}
    for source_name, kinds in kinds_by_source.items():
        for spec in kinds.values():
            existing = merged.get(spec.name)
            if existing is None:
                merged[spec.name] = spec
                contributors[spec.name] = [source_name]
                continue
            if (
                dict(existing.attrs) != dict(spec.attrs)
                or existing.description != spec.description
            ):
                raise ValueError(
                    f"EntityKindSpec(name={spec.name!r}) disagreement between "
                    f"sources {contributors[spec.name]} and {source_name!r}: "
                    f"existing={existing!r}; new={spec!r}"
                )
            contributors[spec.name].append(source_name)
    return merged
