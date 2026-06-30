"""Site layout helper exposed in workflow expressions as ``layout``.

A :class:`Layout` is a small read-only collection of named 2-D points
(plus an optional, currently-unused ``z``). It exists so workflow
authors can write things like::

    timeout:
      duration: "{layout.distance('berth_1', 'stack_A') / config.truck_speed_mps}"

without hand-coding distance tables. The metric is Manhattan distance
in meters. Future versions may add Euclidean or routed distance — but
to stay domain-agnostic the engine itself only knows about coordinates,
not the modeled infrastructure that connects them.

:class:`Layout` is built from a validated
:class:`~altrios.workflow_engine.schemas.LayoutModel` by
``Layout.from_model``; it can also be constructed directly from a dict
of coordinates for tests and ad-hoc use.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional


@dataclass(frozen=True)
class LayoutNode:
    """One named coordinate in a site layout.

    Coordinates are in meters. ``z`` is parsed by the schema but
    currently unused by the engine — it is reserved for future
    elevation/gradient modeling.
    """

    name: str
    x: float
    y: float
    z: Optional[float] = None


@dataclass(frozen=True)
class Layout:
    """Read-only collection of named coordinates with distance helpers.

    Exposed in workflow expression strings as the ``layout`` namespace.
    All public methods raise :class:`KeyError` with the full list of
    available node names when called with an unknown name; this makes
    typo'd node references surface immediately at expression-eval time
    instead of producing garbled distances.

    Examples
    --------
    >>> lay = Layout.from_dict({"a": (0, 0), "b": (3, 4)})
    >>> lay.distance("a", "b")
    7.0
    >>> lay.node("b").y
    4.0
    """

    nodes: Mapping[str, LayoutNode]

    def __post_init__(self) -> None:
        # Freeze the mapping so workflow expressions can't mutate it
        # via, e.g., ``layout.nodes['a'] = ...`` (asteval would reject
        # the assignment anyway, but defending the invariant here keeps
        # Python callers honest as well).
        for k, v in self.nodes.items():
            if not isinstance(v, LayoutNode):
                raise TypeError(
                    f"Layout.nodes[{k!r}] must be a LayoutNode, "
                    f"got {type(v).__name__}."
                )
            if v.name != k:
                raise ValueError(
                    f"Layout.nodes mapping key {k!r} disagrees with "
                    f"LayoutNode.name={v.name!r}; keys must match."
                )
        object.__setattr__(self, "nodes", MappingProxyType(dict(self.nodes)))

    @classmethod
    def from_dict(
        cls, coords: Mapping[str, tuple[float, float] | tuple[float, float, float]]
    ) -> "Layout":
        """Build a Layout from a ``{name: (x, y)}`` or
        ``{name: (x, y, z)}`` mapping. Convenience for tests."""
        nodes: dict[str, LayoutNode] = {}
        for name, xy in coords.items():
            if len(xy) == 2:
                nodes[name] = LayoutNode(name=name, x=float(xy[0]), y=float(xy[1]))
            elif len(xy) == 3:
                nodes[name] = LayoutNode(
                    name=name,
                    x=float(xy[0]),
                    y=float(xy[1]),
                    z=float(xy[2]),
                )
            else:
                raise ValueError(
                    f"Layout.from_dict: entry {name!r} must be (x, y) or "
                    f"(x, y, z); got length {len(xy)}."
                )
        return cls(nodes=nodes)

    @classmethod
    def from_model(cls, model) -> "Layout":
        """Build a Layout from a
        :class:`~altrios.workflow_engine.schemas.LayoutModel` (or any
        object with a ``nodes`` mapping of
        ``x``/``y``/``z``-attributed objects)."""
        nodes: dict[str, LayoutNode] = {}
        for name, n in model.nodes.items():
            nodes[name] = LayoutNode(
                name=name,
                x=float(n.x),
                y=float(n.y),
                z=None if n.z is None else float(n.z),
            )
        return cls(nodes=nodes)

    def node(self, name: str) -> LayoutNode:
        """Return the named node; raise ``KeyError`` listing
        available nodes if missing."""
        if name not in self.nodes:
            raise KeyError(
                f"No layout node named {name!r}. "
                f"Available: {sorted(self.nodes)}."
            )
        return self.nodes[name]

    def distance(self, a: str, b: str) -> float:
        """Manhattan distance in meters between two named nodes
        (locked decision §6). Only ``x`` and ``y`` are used; ``z`` is
        ignored in v1."""
        na = self.node(a)
        nb = self.node(b)
        return abs(na.x - nb.x) + abs(na.y - nb.y)

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, name: object) -> bool:
        return name in self.nodes
