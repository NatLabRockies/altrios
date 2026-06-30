"""Sandboxed expression evaluator for workflow-engine step parameters.

Catalogs write step parameters as Python-like expressions referencing the
running workflow's state (e.g. ``"entity.weight_t / 3.0"``, ``"min(
config.max_speed, layout.distance(entity.from_node, entity.to_node) /
2.0)"``). At interpret time the engine resolves these against an
:class:`ExpressionContext` exposing six namespaces (``entity``,
``bindings``, ``state``, ``config``, ``layout``, ``env``) and a fixed
allowlist of pure math/comparison helpers.

Implementation is **asteval** (sandboxed AST interpreter). asteval
already restricts the AST to a safe subset of Python; we further
tighten it by stripping the default symbol table down to a math-only
allowlist and exposing only the six context namespaces.

**No general function calls.** Catalog authors get exactly the
allowlist below; anything else uses the :class:`Step` primitive
``python`` (see :mod:`altrios.lifts.workflow_engine.registry`).

Expression source format: a raw Python expression string (no ``{ }``
braces). The YAML loader is responsible for stripping any template
braces before constructing an :class:`Expression`.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, TYPE_CHECKING

import asteval

if TYPE_CHECKING:  # pragma: no cover
    from altrios.lifts.workflow_engine.entities import Entity


# Math/utility helpers exposed inside expressions. Arithmetic /
# comparison / Boolean operators come from Python itself (handled by
# asteval's AST visitor); these add the few function calls catalog
# authors are allowed to make.
_ALLOWED_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "len": len,
    "sum": sum,
    "round": round,
    "pow": pow,
    "divmod": divmod,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "atan2": math.atan2,
    "hypot": math.hypot,
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "nan": math.nan,
}

# Forbid expressions containing dunder access to head off the most common
# asteval-sandbox-escape pattern (``obj.__class__.__bases__`` etc.). This
# is a belt-and-suspenders check; asteval restricts the AST to a safe
# subset on top, and catalogs are not adversarial inputs.
_DUNDER_PATTERN = re.compile(r"__[A-Za-z_]")


class ExpressionError(Exception):
    """Raised when an expression fails to parse or evaluate."""


@dataclass
class ExpressionContext:
    """Per-evaluation symbol-table material.

    Each attribute corresponds to one namespace visible to expression
    source. ``entity``, ``state``, ``config``, and ``layout`` are arbitrary
    objects (any duck-typed value works — at interpret time the engine
    populates them with proxy wrappers around real domain objects).
    ``bindings`` is the mutable per-workflow local-variable scope set by
    earlier ``bind`` / ``request`` steps. ``env`` is the SimPy environment
    (only ``env.now`` is in normal use).

    Use :meth:`as_symtable` to materialize the dict asteval consumes.
    """
    entity: Any = None
    bindings: Optional[Mapping[str, Any]] = None
    state: Any = None
    config: Any = None
    layout: Any = None
    env: Any = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def as_symtable(self) -> dict[str, Any]:
        """Return the symtable additions for this context.

        ``bindings`` is exposed both as the ``bindings`` namespace
        (``bindings.foo``) and via a :class:`NamespaceProxy` so that
        catalog authors can choose either style. Other entries follow
        the namespace-attribute convention.
        """
        table: dict[str, Any] = {}
        if self.entity is not None:
            table["entity"] = self.entity
        if self.bindings is not None:
            # Both bracket-style (`bindings["foo"]`) and attr-style
            # (`bindings.foo`) work because NamespaceProxy supports
            # both.
            table["bindings"] = NamespaceProxy(self.bindings)
        if self.state is not None:
            table["state"] = self.state
        if self.config is not None:
            table["config"] = self.config
        if self.layout is not None:
            table["layout"] = self.layout
        if self.env is not None:
            table["env"] = self.env
        if self.extra:
            for k, v in self.extra.items():
                table[k] = v
        return table


class NamespaceProxy:
    """Lightweight ``Mapping`` → attribute-access adapter.

    Wraps a dict-like ``data`` so that both ``proxy.foo`` and
    ``proxy["foo"]`` work. Used for the ``bindings`` namespace and for
    wrapping config/layout dicts loaded from YAML.

    Read-only at expression-evaluation time. Writes happen through the
    interpreter's ``bind`` / ``set_attr`` primitives, which mutate the
    underlying dict directly.
    """
    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, name: str) -> Any:
        return self._data[name]

    def __contains__(self, name: object) -> bool:
        return name in self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        keys = sorted(self._data.keys()) if isinstance(self._data, dict) else list(self._data)
        return f"NamespaceProxy({keys!r})"


def _make_sandbox() -> asteval.Interpreter:
    """Construct a fresh asteval Interpreter with the math allowlist
    and no other globals.

    A fresh Interpreter is built per :meth:`Expression.evaluate` call
    so we don't have to worry about residual variables leaking between
    evaluations. asteval's parse step is cached per-:class:`Expression`,
    which is the expensive bit; building the symtable is cheap.
    """
    ae = asteval.Interpreter(
        minimal=True,
        use_numpy=False,
        builtins_readonly=True,
        max_statement_length=10_000,
    )
    # Wipe the symtable and rebuild with our allowlist. ``minimal=True``
    # still pulls in ``open``, ``print``, etc.; we strip all of them and
    # replace with the locked allowlist.
    ae.symtable.clear()
    ae.symtable.update({
        "True": True,
        "False": False,
        "None": None,
    })
    ae.symtable.update(_ALLOWED_FUNCTIONS)
    return ae


class Expression:
    """A validated, sandboxed workflow-parameter expression.

    Construct once at workflow build time (syntax validation); evaluate
    many times against fresh contexts during simulation. Immutable.

    asteval's high-level ``ae(source)`` call is used at evaluate time
    rather than the lower-level ``ae.run(node)`` API: the latter
    silently swallows attribute-lookup errors when nodes are passed
    across interpreter instances, while the former surfaces them
    cleanly via ``ae.error``.
    """
    __slots__ = ("source",)

    def __init__(self, source: str) -> None:
        if not isinstance(source, str):
            raise ExpressionError(
                f"Expression source must be a str, got {type(source).__name__}."
            )
        source = source.strip()
        if not source:
            raise ExpressionError("Expression source is empty.")
        if _DUNDER_PATTERN.search(source):
            raise ExpressionError(
                f"Expression {source!r} contains a dunder reference (``__``); "
                "dunder access is forbidden in workflow expressions."
            )
        # Syntax validation: try parsing with a throwaway sandbox.
        # Stores no compiled AST — asteval's high-level path re-parses
        # internally and that's where its error machinery lives.
        ae = _make_sandbox()
        try:
            ae.parse(source)
        except Exception as exc:  # asteval raises various exception types
            raise ExpressionError(
                f"Failed to parse expression {source!r}: {exc}"
            ) from exc
        if ae.error:
            raise ExpressionError(
                f"Failed to parse expression {source!r}: "
                f"{_format_asteval_error(ae.error[0])}"
            )
        object.__setattr__(self, "source", source)

    def __setattr__(self, name, value):
        raise AttributeError("Expression is immutable")

    def evaluate(self, ctx: ExpressionContext) -> Any:
        """Evaluate the expression against ``ctx`` and return the result.

        Raises :class:`ExpressionError` on any runtime failure (name
        not found, attribute error, type error, etc.).
        """
        ae = _make_sandbox()
        ae.symtable.update(ctx.as_symtable())
        try:
            result = ae(self.source, show_errors=False, raise_errors=False)
        except Exception as exc:
            # asteval normally suppresses exceptions and stores them on
            # ``ae.error``; if one leaks out, wrap it.
            raise ExpressionError(
                f"Expression {self.source!r} raised: {exc}"
            ) from exc
        if ae.error:
            raise ExpressionError(
                f"Expression {self.source!r} raised: "
                f"{_format_asteval_error(ae.error[0])}"
            )
        return result

    def __repr__(self) -> str:
        return f"Expression({self.source!r})"


def _format_asteval_error(err: Any) -> str:
    """Best-effort formatter for asteval ExceptionHolder objects.

    asteval's ``get_error()`` crashes when an error is raised from
    ``ae.run(node)`` without source text; fall back to ``exc`` + ``msg``
    attributes, which are always populated.
    """
    exc_name = getattr(err.exc, "__name__", str(err.exc))
    msg = getattr(err, "msg", "") or ""
    return f"{exc_name}: {msg}".rstrip(": ")


def evaluate(source: str, ctx: ExpressionContext) -> Any:
    """One-shot convenience wrapper: parse + evaluate.

    Prefer constructing an :class:`Expression` once at workflow build
    time when the same source string will be evaluated many times.
    """
    return Expression(source).evaluate(ctx)
