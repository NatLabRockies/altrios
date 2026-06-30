"""Declarative SimPy resource specs for workflow modes.

A :class:`ResourceSpec` describes one SimPy primitive (``Store``,
``Resource``, or ``Container``). Modes declare the specs they need; the
dispatcher unions specs from all active modes, dedups by name (via
:func:`merge_specs`), validates agreement on kind / capacity /
partition_by / role, and asks :func:`build_state_from_specs` to
instantiate the primitives.

Cross-mode sharing emerges from shared names — there is no
``private | shared`` flag. If two modes both declare a spec named
``"main_stack_rtgs"``, exactly one SimPy primitive is created and both
modes refer to it through the same attribute on ``TerminalState``.

This module lives in :mod:`altrios.lifts.workflow_engine` and knows nothing
about freight, mining, or any other domain. Catalogs define
domain-specific :class:`ResourceSpec` instances (e.g. ``TRACKS``,
``STS_CRANES_BY_BERTH``) referencing the engine-level dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Union

import simpy

# A capacity can be a literal int or a callable resolving against the
# terminal config + schedules dict produced by the mode's build_schedule.
CapacitySpec = Union[int, Callable[[Mapping[str, Any], Mapping[str, Any]], int]]
ItemFactory = Callable[[Mapping[str, Any], Mapping[str, Any]], Iterable[Any]]
PartitionKeys = Callable[[Mapping[str, Any], Mapping[str, Any]], Iterable[Any]]


# Conventional role values for ResourceSpec.role. Catalogs are free to
# introduce additional role strings; the engine does not enforce
# membership, but tooling (validation, default output columns,
# documentation) understands these three.
KNOWN_ROLES: frozenset[str] = frozenset({
    "equipment",         # cranes, hostlers, yard tractors, chassis — actors that perform work
    "infrastructure",    # tracks, berths, gate lanes, parking slots — capacity-limited spatial locations
    "storage",           # container stack — holds entities being processed
})


@dataclass(frozen=True)
class ResourceSpec:
    """Declarative description of one SimPy primitive.

    Parameters
    ----------
    name
        Identifier used both as the merge key for cross-mode sharing and as
        the attribute name on ``TerminalState`` (e.g. ``state.tracks``).
    kind
        SimPy class to instantiate: ``"Store"``, ``"Resource"``, or
        ``"Container"``.
    role
        Conceptual category for tooling — typically one of
        ``"equipment"``, ``"infrastructure"``, or ``"storage"`` (see
        :data:`KNOWN_ROLES`). Catalogs may use additional strings; the
        engine does not enforce membership. The role drives default
        output-schema columns, load-time validation of consumption-rate
        attachment, and human-facing error/documentation messages, but
        **does not** affect simulation behavior — all roles are still
        requested/released identically.
    capacity
        Either an int or a callable ``(config, schedules) -> int``.
    partition_by
        Optional callable ``(config, schedules) -> iterable of keys``. When
        provided, the attribute on ``TerminalState`` is a ``dict`` mapping
        each key to one primitive (e.g. one Store per track id).
    init_items
        Optional callable ``(config, schedules) -> iterable of items`` to
        ``put`` into a Store at construction. For partitioned Stores the
        callable can read the partition key via ``schedules['_partition_key']``.
    """

    name: str
    kind: str
    role: str
    capacity: CapacitySpec = 1
    partition_by: Optional[PartitionKeys] = None
    init_items: Optional[ItemFactory] = None

    def __post_init__(self) -> None:
        if self.kind not in _SIMPY_CLASSES:
            raise ValueError(
                f"ResourceSpec(name={self.name!r}).kind must be one of "
                f"{list(_SIMPY_CLASSES)}; got {self.kind!r}"
            )
        if not isinstance(self.role, str) or not self.role:
            raise ValueError(
                f"ResourceSpec(name={self.name!r}).role must be a non-empty "
                f"string; got {self.role!r}. Conventional values are "
                f"{sorted(KNOWN_ROLES)}, but catalogs may use other strings."
            )


@dataclass(frozen=True)
class EventSpec:
    """Declarative description of per-arrival SimPy events a mode emits.

    Modes can advertise their event surface for diagnostics and runtime
    introspection.
    """

    name: str
    per_arrival: bool = True
    description: str = ""


_SIMPY_CLASSES: dict[str, type] = {
    "Store": simpy.Store,
    "Resource": simpy.Resource,
    "Container": simpy.Container,
}


def _resolve_capacity(
    spec: ResourceSpec, config: Mapping[str, Any], schedules: Mapping[str, Any]
) -> int:
    cap = spec.capacity
    return int(cap(config, schedules)) if callable(cap) else int(cap)


def _resolve_partition_keys(
    spec: ResourceSpec, config: Mapping[str, Any], schedules: Mapping[str, Any]
) -> Optional[list[Any]]:
    if spec.partition_by is None:
        return None
    keys = list(spec.partition_by(config, schedules))
    if not keys:
        raise ValueError(
            f"ResourceSpec(name={spec.name!r}).partition_by returned no keys"
        )
    return keys


def _make_one(
    env: simpy.Environment,
    spec: ResourceSpec,
    capacity: int,
    config: Mapping[str, Any],
    schedules: Mapping[str, Any],
) -> Any:
    obj = _SIMPY_CLASSES[spec.kind](env, capacity=capacity)
    if spec.init_items is not None:
        if spec.kind != "Store":
            raise ValueError(
                f"ResourceSpec(name={spec.name!r}) has init_items but "
                f"kind={spec.kind!r}; only Store supports initial items"
            )
        for item in spec.init_items(config, schedules):
            obj.put(item)
    return obj


def merge_specs(
    specs_by_mode: Mapping[str, Iterable[ResourceSpec]],
) -> dict[str, ResourceSpec]:
    """Union ``ResourceSpec`` lists across active modes; assert agreement.

    Two specs agree iff they have the same ``kind``, the same ``role``,
    identical ``partition_by``/``init_items`` callables (compared by
    object identity), and equal ``capacity``. Raise ``ValueError`` on
    disagreement so the caller knows which two modes conflict.
    """
    merged: dict[str, ResourceSpec] = {}
    contributors: dict[str, list[str]] = {}
    for mode_name, mode_specs in specs_by_mode.items():
        for spec in mode_specs:
            existing = merged.get(spec.name)
            if existing is None:
                merged[spec.name] = spec
                contributors[spec.name] = [mode_name]
                continue
            if (
                existing.kind != spec.kind
                or existing.role != spec.role
                or existing.partition_by is not spec.partition_by
                or existing.init_items is not spec.init_items
                or existing.capacity != spec.capacity
            ):
                raise ValueError(
                    f"ResourceSpec(name={spec.name!r}) disagreement between "
                    f"modes {contributors[spec.name]} and {mode_name!r}: "
                    f"existing={existing!r}; new={spec!r}"
                )
            contributors[spec.name].append(mode_name)
    return merged


def build_state_from_specs(
    env: simpy.Environment,
    specs: Iterable[ResourceSpec],
    config: Mapping[str, Any],
    schedules: Mapping[str, Any],
) -> dict[str, Any]:
    """Instantiate primitives for the given specs.

    Returns a name -> primitive (or ``dict[key, primitive]`` for partitioned
    specs) map. ``TerminalState.__init__`` sets each entry as an explicit
    attribute on the state object so call sites read e.g. ``state.tracks``
    rather than ``state._pools['tracks']``.
    """
    out: dict[str, Any] = {}
    for spec in specs:
        keys = _resolve_partition_keys(spec, config, schedules)
        if keys is None:
            out[spec.name] = _make_one(
                env,
                spec,
                _resolve_capacity(spec, config, schedules),
                config,
                schedules,
            )
            continue
        partitioned: dict[Any, Any] = {}
        for key in keys:
            schedules_view = dict(schedules)
            schedules_view["_partition_key"] = key
            partitioned[key] = _make_one(
                env,
                spec,
                _resolve_capacity(spec, config, schedules_view),
                config,
                schedules_view,
            )
        out[spec.name] = partitioned
    return out
