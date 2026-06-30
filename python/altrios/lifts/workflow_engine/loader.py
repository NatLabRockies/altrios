"""High-level entry points for loading catalogs and sites from YAML.

This module composes the lower-level pieces:

1. :mod:`altrios.lifts.workflow_engine.yaml_loader` — safe YAML parse with
   ``!include`` support.
2. :mod:`altrios.lifts.workflow_engine.yaml_expressions` — convert ``"{...}"``
   strings into :class:`Expression` objects.
3. :mod:`altrios.lifts.workflow_engine.schemas` — pydantic validation.
4. ``.to_engine(...)`` on the validated models — build frozen
   dataclasses (:class:`Catalog`, :class:`WorkflowMode`, etc).

It also handles two concerns the lower layers deliberately do not:

* **Catalog references** — a site's ``catalog:`` field may name either
  a filesystem path or a Python dotted-module that ships a
  ``catalog.yaml`` resource. We resolve the reference here.
* **Site ``extends:``** — single-level deep-merge of a parent site
  before validation, so authors can keep a base ``site.yaml`` plus
  thin per-scenario override files (locked decision §13).
* **Python module registration** — a catalog's ``python_module``
  field is imported so its module-level
  ``@register(...)`` decorators run; resource specs that reference
  Python helpers (``partition_by_python``, ``init_items_python``) are
  then resolved via the default callable registry.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

from .catalog import Catalog
from .registry import RegistryError, get_registry
from .schemas import CatalogModel, SiteModel
from .yaml_expressions import convert_expressions
from .yaml_loader import load_yaml_file


class LoaderError(Exception):
    """Raised when catalog or site loading fails for reasons outside
    of pure schema validation (catalog resolution, module import,
    extends-merge conflicts, ...). Pydantic-level errors are surfaced
    as ``pydantic.ValidationError``; YAML syntax errors as
    ``YamlLoaderError``."""


PathLike = Union[str, os.PathLike]


# ---- Catalog site-path resolver --------------------------------------


def make_site_path(
    package_file: PathLike, *, kind: str
) -> Callable[[str], Path]:
    """Build a ``site_path(name) -> Path`` resolver for a catalog package.

    Convention: every catalog package ships its bundled site YAMLs in a
    ``sites/`` subdirectory next to its ``__init__.py``. This factory
    captures that directory once and returns a small lookup function the
    catalog re-exports for callers (demos, tests, downstream users).

    Typical use in a catalog ``__init__.py``::

        from altrios.lifts.workflow_engine import make_site_path

        site_path = make_site_path(__file__, kind="terminal")

    Parameters
    ----------
    package_file : str or os.PathLike
        Usually ``__file__`` of the catalog package's ``__init__.py``.
        The ``sites/`` directory is resolved relative to its parent.
    kind : str
        Short label for the catalog (e.g. ``"terminal"``, ``"mine"``).
        Used only in the :class:`FileNotFoundError` message so a missing
        site name surfaces a useful hint.

    Returns
    -------
    Callable[[str], Path]
        A function that accepts a site name (with or without the
        ``.yaml`` extension) and returns the absolute path to the
        matching file. Raises :class:`FileNotFoundError` listing the
        available sites if the name does not match a bundled YAML.
    """
    sites_dir = Path(os.fspath(package_file)).resolve().parent / "sites"

    def site_path(name: str) -> Path:
        filename = name if name.endswith(".yaml") else f"{name}.yaml"
        path = sites_dir / filename
        if not path.is_file():
            available = sorted(p.stem for p in sites_dir.glob("*.yaml"))
            raise FileNotFoundError(
                f"No {kind} site named {filename!r} under {sites_dir}. "
                f"Available: {available}"
            )
        return path

    site_path.__doc__ = (
        f"Return the absolute path to a bundled {kind} site YAML by name. "
        f"Accepts ``name`` with or without the ``.yaml`` extension."
    )
    return site_path


def make_runner(
    site_path_fn: Callable[[str], Path],
) -> Callable[..., "RunResult"]:
    """Build a name-based site runner that composes :func:`run_site`.

    The returned ``run(name, **kwargs)`` resolves ``name`` via
    ``site_path_fn`` and forwards the resulting path plus all kwargs
    to :func:`altrios.lifts.workflow_engine.runner.run_site`. Pair
    with :func:`make_site_path` in a catalog ``__init__.py`` so callers
    can launch bundled sites without juggling paths::

        from altrios.lifts.workflow_engine import make_site_path, make_runner

        site_path = make_site_path(__file__, kind="terminal")
        run = make_runner(site_path)

    Then::

        from altrios.lifts import terminal
        result = terminal.run("allouez_truck_rail", seed=42)

    The runner import is deferred so ``import <catalog>`` stays cheap.

    Parameters
    ----------
    site_path_fn
        A callable produced by :func:`make_site_path` (or any
        ``str -> Path`` resolver).

    Returns
    -------
    Callable[..., RunResult]
        A function ``run(name, **kwargs) -> RunResult``.
    """
    def run(name: str, **kwargs: Any) -> "RunResult":
        # Lazy import — pulls in simpy and the full engine runtime.
        from altrios.lifts.workflow_engine.runner import run_site
        return run_site(site_path_fn(name), **kwargs)

    run.__doc__ = (
        "Run a bundled site by name. Resolves ``name`` via the catalog's "
        "site_path resolver and forwards kwargs to "
        ":func:`altrios.lifts.workflow_engine.run_site`."
    )
    return run


# ---- Catalog --------------------------------------------------------


def load_catalog(path: PathLike) -> Catalog:
    """Load and validate a catalog YAML file; return a :class:`Catalog`.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem path to a catalog YAML file.

    Returns
    -------
    Catalog
        Frozen catalog dataclass with all expression strings parsed,
        Python helpers resolved, and the catalog's ``python_module``
        imported (if any).

    Raises
    ------
    YamlLoaderError
        If the file cannot be parsed (syntax error, cyclic include,
        unsafe tag).
    pydantic.ValidationError
        If the parsed structure does not match :class:`CatalogModel`.
    LoaderError
        If the top-level YAML is not a mapping, if ``python_module``
        cannot be imported, or if a ``partition_by_python`` /
        ``init_items_python`` name is not registered after the
        module loads.
    """
    raw = load_yaml_file(path)
    if not isinstance(raw, dict):
        raise LoaderError(
            f"Catalog file {os.fspath(path)!r} must contain a YAML mapping "
            f"at the top level; got {type(raw).__name__}."
        )
    converted = convert_expressions(raw)
    model = CatalogModel.model_validate(converted)

    if model.python_module is not None:
        try:
            importlib.import_module(model.python_module)
        except ImportError as exc:
            raise LoaderError(
                f"Catalog {model.name!r}: failed to import "
                f"python_module={model.python_module!r}: {exc}"
            ) from exc

    registry = get_registry()

    def _resolver(field_name: str):
        def resolve(dotted_name: str):
            try:
                return registry.get(dotted_name)
            except RegistryError as exc:
                raise LoaderError(
                    f"Catalog {model.name!r}: resource spec referenced "
                    f"{field_name}={dotted_name!r} but no such callable "
                    f"is registered. Did you forget to import the "
                    f"catalog's python_module, or to decorate the "
                    f"helper with @register({dotted_name!r})? "
                    f"Available: {sorted(registry.names())}."
                ) from exc
        return resolve

    return model.to_engine(
        partition_by_resolver=_resolver("partition_by_python"),
        init_items_resolver=_resolver("init_items_python"),
    )


# ---- Site -----------------------------------------------------------


def load_site(path: PathLike) -> tuple[SiteModel, Catalog]:
    """Load and validate a site YAML file together with its catalog.

    ``extends:`` is supported at one level only in v1: the named
    parent file is loaded first, then the current file's fields
    deep-merge on top. The parent's own ``extends:`` field is
    silently ignored if present (chained inheritance is deferred).
    The parent's relative ``catalog:`` is re-anchored to the parent
    file's directory before merging, so a base site keeps working
    when an extends-child lives in a different directory.

    Parameters
    ----------
    path : str or os.PathLike
        Filesystem path to a site YAML file.

    Returns
    -------
    tuple of (SiteModel, Catalog)
        ``site_model`` is the pydantic representation (not an engine
        dataclass); ``catalog`` is the fully resolved engine
        :class:`Catalog`. The run-orchestration layer consumes both
        halves directly.

    Raises
    ------
    YamlLoaderError
        If any file in the include / extends chain cannot be parsed.
    pydantic.ValidationError
        If the merged site structure does not match
        :class:`SiteModel`.
    LoaderError
        If the top-level YAML is not a mapping, if ``extends`` is
        malformed, if the catalog reference cannot be resolved, or
        if the site activates a mode not declared by the catalog.
    """
    base_dir = os.path.dirname(os.fspath(Path(path).resolve()))
    raw = _load_with_extends(path)
    if not isinstance(raw, dict):
        raise LoaderError(
            f"Site file {os.fspath(path)!r} must contain a YAML mapping "
            f"at the top level; got {type(raw).__name__}."
        )
    converted = convert_expressions(raw)
    site = SiteModel.model_validate(converted)

    catalog_path = _resolve_catalog_reference(site.catalog, site_dir=base_dir)
    catalog = load_catalog(catalog_path)

    # Cross-validate that every mode the site activates is actually
    # in the catalog. (The site model alone can't enforce this.)
    catalog_mode_names = {m.name for m in catalog.modes}
    for mode_name in site.modes:
        if mode_name not in catalog_mode_names:
            raise LoaderError(
                f"Site {site.name!r}: activated mode {mode_name!r} is "
                f"not present in catalog {catalog.name!r}. "
                f"Available: {sorted(catalog_mode_names)}."
            )

    return site, catalog


# ---- Helpers --------------------------------------------------------


def _load_with_extends(path: PathLike) -> Mapping[str, Any]:
    """Load a YAML file, applying a single ``extends:`` override pass.

    The parent is resolved relative to the child file's directory.
    The child's fields deep-merge on top of the parent's. ``extends``
    itself is stripped from the result so it never reaches pydantic.
    """
    child = load_yaml_file(path)
    if not isinstance(child, dict) or "extends" not in child:
        return child

    parent_ref = child["extends"]
    if not isinstance(parent_ref, str) or not parent_ref:
        raise LoaderError(
            f"Site file {os.fspath(path)!r}: extends must be a non-empty "
            f"string path; got {parent_ref!r}."
        )

    child_dir = os.path.dirname(os.fspath(Path(path).resolve()))
    parent_path = (
        parent_ref
        if os.path.isabs(parent_ref)
        else os.path.join(child_dir, parent_ref)
    )
    parent = load_yaml_file(parent_path)
    if not isinstance(parent, dict):
        raise LoaderError(
            f"Site file {os.fspath(path)!r}: extends parent "
            f"{parent_path!r} must contain a YAML mapping; got "
            f"{type(parent).__name__}."
        )

    # Drop the parent's own extends (no chained inheritance in v1).
    parent = {k: v for k, v in parent.items() if k != "extends"}
    # Anchor the parent's relative ``catalog:`` field to the parent
    # file's directory. Otherwise it would be resolved relative to the
    # child after merging, which is almost never what the author wants
    # — a base site file should keep working when an extends-child
    # lives in a different directory.
    parent_dir = os.path.dirname(os.fspath(Path(parent_path).resolve()))
    cat_ref = parent.get("catalog")
    if (
        isinstance(cat_ref, str)
        and ("/" in cat_ref or os.sep in cat_ref or cat_ref.endswith((".yaml", ".yml")))
        and not os.path.isabs(cat_ref)
    ):
        parent["catalog"] = os.path.normpath(os.path.join(parent_dir, cat_ref))

    merged = _deep_merge(parent, {k: v for k, v in child.items() if k != "extends"})
    return merged


def _deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``over`` onto ``base``.

    Dict values are merged; lists and scalars are replaced wholesale.
    The result is a fresh dict; neither input is mutated.
    """
    out = dict(base)
    for k, v in over.items():
        if (
            k in out
            and isinstance(out[k], Mapping)
            and isinstance(v, Mapping)
        ):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_catalog_reference(
    ref: str, *, site_dir: Optional[str] = None
) -> str:
    """Resolve a site's ``catalog:`` field to a filesystem path.

    Rules:

    * If ``ref`` contains a path separator OR ends in ``.yaml`` /
      ``.yml``, it is treated as a filesystem path (absolute or
      relative to ``site_dir`` when relative).
    * Otherwise it is treated as a Python dotted-module path; the
      module is imported and a sibling file named ``catalog.yaml`` is
      expected next to its ``__init__.py``.
    """
    looks_like_path = (
        os.sep in ref
        or "/" in ref
        or ref.endswith(".yaml")
        or ref.endswith(".yml")
    )
    if looks_like_path:
        if os.path.isabs(ref):
            return ref
        if site_dir is None:
            return os.path.abspath(ref)
        return os.path.join(site_dir, ref)

    # Dotted-module form.
    try:
        module = importlib.import_module(ref)
    except ImportError as exc:
        raise LoaderError(
            f"Catalog reference {ref!r} could not be imported as a Python "
            f"module: {exc}. If you meant to reference a file, include a "
            f"path separator (e.g. './my_catalog.yaml')."
        ) from exc

    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise LoaderError(
            f"Catalog reference {ref!r}: module has no __file__ attribute, "
            f"so the sibling 'catalog.yaml' cannot be located."
        )
    candidate = os.path.join(os.path.dirname(module_file), "catalog.yaml")
    if not os.path.exists(candidate):
        raise LoaderError(
            f"Catalog reference {ref!r}: expected a 'catalog.yaml' file "
            f"next to {module_file!r}, but it does not exist."
        )
    return candidate
