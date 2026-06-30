"""Workflow ``Step`` and ``StepGraph`` dataclasses.

A ``Step`` is the smallest unit of work in a workflow — a single
primitive (``bind``, ``timeout``, ``request``, ...) with its parameters.
A ``StepGraph`` is a named, frozen collection of steps with a designated
entry step; the interpreter walks the graph at simulation time.

Step graphs are produced by the YAML catalog loader. They are immutable
once constructed so that the same graph can be safely reused across
many entities and across many simulation runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class Step:
    """One node in a workflow graph.

    Attributes:
        id: Unique identifier within the enclosing :class:`StepGraph`.
        type: Name of the primitive (``"timeout"``, ``"request"``, ...).
            Must match a key in the interpreter's primitive table.
        params: Per-primitive keyword arguments. Values may be literals
            (numbers, strings, dicts) or :class:`Expression` instances —
            the interpreter calls ``.evaluate(ctx)`` on any Expression
            it sees in a param slot.
        next: ID of the step to execute after this one, or ``None`` for
            the terminal step. Primitives that branch (``branch``,
            ``loop``, ``parallel`` join, ``spawn`` wait, ...) return
            their own next-id from the handler and the interpreter
            takes that in preference to ``next``.
    """

    id: str
    type: str
    params: Mapping[str, Any] = field(default_factory=dict)
    next: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError(f"Step.id must be a non-empty str, got {self.id!r}.")
        if not isinstance(self.type, str) or not self.type:
            raise ValueError(
                f"Step.type must be a non-empty str, got {self.type!r}."
            )
        if self.next is not None and (
            not isinstance(self.next, str) or not self.next
        ):
            raise ValueError(
                f"Step.next must be None or a non-empty str, got {self.next!r}."
            )
        # Freeze params into a read-only mapping so a graph can't be
        # mutated through a handler grabbing a reference to params.
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True)
class StepGraph:
    """An immutable named collection of :class:`Step` objects.

    Attributes:
        name: Human-readable identifier (e.g. ``"truck_inbound"``).
        entry: ID of the step at which execution starts. Must appear
            in ``steps``.
        steps: Mapping from step id to :class:`Step`.

    Raises ``ValueError`` if ``entry`` is missing, if any step's id
    doesn't match its key, or if any ``next``-reference points at a
    step id that doesn't exist in this graph. Cross-graph references
    (``spawn`` of another graph) are checked at workflow-load time,
    not here.
    """

    name: str
    entry: str
    steps: Mapping[str, Step]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"StepGraph.name must be a non-empty str, got {self.name!r}."
            )
        if not isinstance(self.entry, str) or not self.entry:
            raise ValueError(
                f"StepGraph.entry must be a non-empty str, got {self.entry!r}."
            )
        # Make sure every value is a Step and matches its key.
        frozen: dict[str, Step] = {}
        for key, value in self.steps.items():
            if not isinstance(value, Step):
                raise TypeError(
                    f"StepGraph.steps[{key!r}] must be a Step, "
                    f"got {type(value).__name__}."
                )
            if value.id != key:
                raise ValueError(
                    f"StepGraph.steps[{key!r}].id={value.id!r} "
                    f"does not match its dict key."
                )
            frozen[key] = value
        if self.entry not in frozen:
            raise ValueError(
                f"StepGraph entry {self.entry!r} is not among defined "
                f"steps: {sorted(frozen)}."
            )
        # Validate intra-graph next pointers. Per-primitive params that
        # carry step ids (``branch.true``, ``loop.do``, ...) are
        # validated by the YAML loader against the primitive's schema,
        # not here, because this dataclass doesn't know primitive
        # semantics.
        for step in frozen.values():
            if step.next is not None and step.next not in frozen:
                raise ValueError(
                    f"Step {step.id!r}.next={step.next!r} does not name "
                    f"a step in graph {self.name!r}."
                )
        object.__setattr__(self, "steps", MappingProxyType(frozen))
