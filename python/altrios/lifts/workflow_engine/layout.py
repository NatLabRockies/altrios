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
:class:`~altrios.lifts.workflow_engine.schemas.LayoutModel` by
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
        """Build a :class:`Layout` from a coordinate mapping.

        Parameters
        ----------
        coords : Mapping[str, tuple of float]
            Mapping of node name to either ``(x, y)`` or
            ``(x, y, z)``. Coordinates are interpreted as meters.

        Returns
        -------
        Layout
            New layout containing one :class:`LayoutNode` per entry in
            ``coords``.

        Raises
        ------
        ValueError
            If any tuple has a length other than 2 or 3.
        """
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
        """Build a :class:`Layout` from a validated layout schema model.

        Parameters
        ----------
        model
            Any object with a ``nodes`` mapping whose values expose
            ``x``, ``y``, and ``z`` attributes — typically a
            :class:`~altrios.lifts.workflow_engine.schemas.LayoutModel`.

        Returns
        -------
        Layout
            New layout with one :class:`LayoutNode` per ``model.nodes``
            entry. ``z`` is preserved when present and set to ``None``
            otherwise.
        """
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
        """Look up a layout node by name.

        Parameters
        ----------
        name : str
            The :class:`LayoutNode` name to look up.

        Returns
        -------
        LayoutNode
            The named node.

        Raises
        ------
        KeyError
            If ``name`` is not in :attr:`nodes`. The error message
            includes the sorted list of available names so typos are
            immediately diagnosable.
        """
        if name not in self.nodes:
            raise KeyError(
                f"No layout node named {name!r}. "
                f"Available: {sorted(self.nodes)}."
            )
        return self.nodes[name]

    def distance(self, a: str, b: str) -> float:
        """Return the Manhattan distance in meters between two nodes.

        Manhattan (L1) distance is hard-coded per locked decision
        §6; ``z`` is ignored in v1 even when populated. Catalogs that
        need Euclidean or routed distance should expose a helper of
        their own via ``python:`` callables.

        Parameters
        ----------
        a : str
            Name of the first node.
        b : str
            Name of the second node.

        Returns
        -------
        float
            ``|a.x - b.x| + |a.y - b.y|`` in meters.

        Raises
        ------
        KeyError
            If either ``a`` or ``b`` is not a known node name.
        """
        na = self.node(a)
        nb = self.node(b)
        return abs(na.x - nb.x) + abs(na.y - nb.y)

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, name: object) -> bool:
        return name in self.nodes
