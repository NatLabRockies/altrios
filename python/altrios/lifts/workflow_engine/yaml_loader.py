"""YAML loader with ``!include`` support.

A thin layer over ``yaml.safe_load`` that:

1. Resolves a custom ``!include <relative-path>`` tag, recursively
   inlining the included file's parsed YAML into the parent document.
2. Tracks an include-stack so cyclic includes are detected with a
   clear error message (rather than infinite recursion).
3. Resolves include paths relative to the *including* file's
   directory, matching the intuition that a YAML's includes are
   sibling files unless an absolute path is given.

Only ``yaml.safe_load`` is used (never ``yaml.load`` without
``SafeLoader``) so untrusted YAML cannot construct arbitrary Python
objects.

The loader returns a plain Python dict (or list / scalar — whatever the
YAML's top-level type is). Expression-string detection and schema
validation are layered on top in subsequent modules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class YamlLoaderError(Exception):
    """Raised for include resolution failures, cyclic includes, and
    YAML parse errors."""


def load_yaml_file(path: str | Path) -> Any:
    """Load a YAML file from ``path`` with ``!include`` support.

    Parameters
    ----------
    path
        Filesystem path to a YAML file. Relative paths are resolved
        against the current working directory before loading begins;
        subsequent ``!include`` directives are resolved relative to
        each file's own directory (NOT the original CWD), so an
        included file can itself include further files using its
        local layout.

    Returns
    -------
    The parsed YAML document — typically a ``dict``, but may be a
    ``list``, scalar, or ``None`` for an empty file.

    Raises
    ------
    YamlLoaderError
        If a file cannot be opened, a ``!include`` cycle is detected,
        or the YAML itself is invalid.
    """
    return _load_resolved(Path(path).resolve(), include_stack=())


def load_yaml_string(source: str, *, base_dir: str | Path | None = None) -> Any:
    """Parse YAML text with ``!include`` support.

    Used chiefly by tests; production callers should prefer
    :func:`load_yaml_file` so cycle detection has a real path anchor
    and ``!include`` resolution starts from a meaningful directory.

    Parameters
    ----------
    source : str
        Raw YAML text.
    base_dir : str or pathlib.Path, optional
        Directory against which any ``!include`` directives in
        ``source`` are resolved. Defaults to the current working
        directory.

    Returns
    -------
    Any
        The parsed YAML document — typically a ``dict``, but may be a
        ``list``, scalar, or ``None`` for empty input.

    Raises
    ------
    YamlLoaderError
        If the YAML is invalid or an ``!include`` directive cannot be
        resolved.
    """
    base = Path(base_dir).resolve() if base_dir is not None else Path.cwd()
    loader = _make_loader(base, include_stack=())
    try:
        return yaml.load(source, Loader=loader)
    except yaml.YAMLError as exc:
        raise YamlLoaderError(f"YAML parse error: {exc}") from exc


def _load_resolved(abs_path: Path, *, include_stack: tuple[Path, ...]) -> Any:
    """Internal: load a fully-resolved absolute path with the given
    include stack already validated for cycles."""
    if not abs_path.is_file():
        raise YamlLoaderError(
            f"YAML file not found: {abs_path}"
            + (
                f" (included from {include_stack[-1]})"
                if include_stack
                else ""
            )
        )
    new_stack = include_stack + (abs_path,)
    loader = _make_loader(abs_path.parent, include_stack=new_stack)
    try:
        with abs_path.open("r", encoding="utf-8") as fh:
            return yaml.load(fh, Loader=loader)
    except yaml.YAMLError as exc:
        raise YamlLoaderError(
            f"YAML parse error in {abs_path}: {exc}"
        ) from exc


def _make_loader(
    base_dir: Path, *, include_stack: tuple[Path, ...]
) -> type[yaml.SafeLoader]:
    """Build a SafeLoader subclass with ``!include`` registered.

    The loader class is built fresh per call so the ``base_dir`` /
    ``include_stack`` captured in the closure is unique to this
    invocation and not shared between unrelated loads.
    """

    class _IncludingSafeLoader(yaml.SafeLoader):
        pass

    def _include_constructor(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
        if not isinstance(node, yaml.ScalarNode):
            raise YamlLoaderError(
                f"!include expects a scalar string path, got "
                f"{type(node).__name__} at {_node_mark(node)}."
            )
        raw_path = loader.construct_scalar(node)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise YamlLoaderError(
                f"!include path must be a non-empty string at "
                f"{_node_mark(node)}, got {raw_path!r}."
            )
        included = Path(raw_path)
        if not included.is_absolute():
            included = base_dir / included
        included = included.resolve()
        if included in include_stack:
            cycle = " -> ".join(str(p) for p in include_stack + (included,))
            raise YamlLoaderError(f"Cyclic !include detected: {cycle}")
        return _load_resolved(included, include_stack=include_stack)

    _IncludingSafeLoader.add_constructor("!include", _include_constructor)
    return _IncludingSafeLoader


def _node_mark(node: yaml.Node) -> str:
    """Format a YAML node's source location for error messages."""
    start = getattr(node, "start_mark", None)
    if start is None:
        return "<unknown location>"
    return f"line {start.line + 1}, column {start.column + 1}"
