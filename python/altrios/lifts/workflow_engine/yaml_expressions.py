"""Expression-string detection for YAML-loaded catalogs.

Catalog YAML uses the convention that **a scalar string whose entire
content is wrapped in curly braces** is an :class:`Expression`:

.. code-block:: yaml

    - {type: bind, name: dist, value: "{layout.distance('a', 'b')}"}
    - {type: timeout, duration: "{entity.weight_t / 3.0}"}

Plain strings (``status: "loading"``) and partially-braced strings
(``message: "loaded {n} items"``) are **NOT** treated as expressions —
the former is just a literal, the latter is a template (handled
separately by the ``log`` / ``record_event`` primitives if they ever
support templates).

This module exposes :func:`convert_expressions` which walks a parsed
YAML structure (dict / list / scalar) and returns a new structure
where matching strings have been replaced with :class:`Expression`
instances. Walk is recursive and preserves container types.

The whole-string-only rule keeps the YAML grammar dead-simple: there
is exactly one syntactic form for "this is an expression", and it
cannot ever be ambiguous with a string literal that just happens to
include braces.
"""
from __future__ import annotations

from typing import Any

from .expressions import Expression, ExpressionError


class ExpressionConversionError(Exception):
    """Raised when a brace-wrapped string fails to parse as a valid
    :class:`Expression`. Wraps the underlying ``ExpressionError`` with
    a JSON-path-style location string for debugging."""


def is_expression_string(value: object) -> bool:
    """Return ``True`` iff ``value`` is a string wrapped in outer braces.

    Whitespace around the braces is tolerated (YAML often inserts
    leading or trailing spaces depending on block style); the interior
    is not inspected here — syntax validation happens when the
    :class:`~altrios.lifts.workflow_engine.expressions.Expression` is
    constructed.

    Parameters
    ----------
    value : object
        The candidate value (any type accepted; non-strings return
        ``False`` immediately).

    Returns
    -------
    bool
        ``True`` iff ``value`` is a non-empty ``str`` whose stripped
        form starts with ``{`` and ends with ``}``.
    """
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return (
        len(stripped) >= 2
        and stripped.startswith("{")
        and stripped.endswith("}")
    )


def extract_expression_source(value: str) -> str:
    """Strip the outer braces and surrounding whitespace.

    Caller must ensure :func:`is_expression_string` returns ``True``
    for ``value`` first; this function does no validation.

    Parameters
    ----------
    value : str
        A brace-wrapped expression string such as ``"{entity.weight}"``.

    Returns
    -------
    str
        The inner source with the outer braces and any inner whitespace
        stripped — e.g. ``"entity.weight"``.
    """
    stripped = value.strip()
    # Drop one layer of braces and any inner whitespace.
    return stripped[1:-1].strip()


def convert_expressions(data: Any, *, path: str = "$") -> Any:
    """Walk ``data`` recursively; return an equivalent structure with
    every brace-wrapped string replaced by an :class:`Expression`.

    Parameters
    ----------
    data
        A value parsed from YAML — typically a dict / list / scalar
        tree.
    path
        JSON-path-style breadcrumb used in error messages. Callers
        pass the default; the recursion appends ``.key`` and ``[i]``
        suffixes as it descends, so failures point at the offending
        node.

    Returns
    -------
    A new structure where strings matching :func:`is_expression_string`
    have been replaced with :class:`Expression`. Other strings, lists,
    dicts, and scalars are returned unchanged (containers are
    shallow-copied so the caller's input is never mutated).

    Raises
    ------
    ExpressionConversionError
        If a brace-wrapped string fails to parse as a valid expression
        (syntax error, dunder use, empty body, etc.).
    """
    if isinstance(data, str):
        if is_expression_string(data):
            source = extract_expression_source(data)
            try:
                return Expression(source)
            except ExpressionError as exc:
                raise ExpressionConversionError(
                    f"At {path}: failed to parse expression "
                    f"{source!r}: {exc}"
                ) from exc
        return data
    if isinstance(data, list):
        return [
            convert_expressions(item, path=f"{path}[{i}]")
            for i, item in enumerate(data)
        ]
    if isinstance(data, dict):
        out: dict[Any, Any] = {}
        for k, v in data.items():
            child_path = f"{path}.{k}" if isinstance(k, str) else f"{path}[{k!r}]"
            out[k] = convert_expressions(v, path=child_path)
        return out
    return data
