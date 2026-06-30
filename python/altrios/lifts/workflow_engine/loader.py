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
from typing import Any, Mapping, Optional, Union

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


# ---- Catalog --------------------------------------------------------


def load_catalog(path: PathLike) -> Catalog:
    """Load and validate a catalog YAML file; return a :class:`Catalog`.

    Parameters
    ----------
    path
        Filesystem path to a catalog YAML file.

    Raises
    ------
    YamlLoaderError
        If the file cannot be parsed (syntax error, cyclic include,
        unsafe tag).
    pydantic.ValidationError
        If the parsed structure doesn't match :class:`CatalogModel`.
    LoaderError
        If ``python_module`` cannot be imported, or if a
        ``partition_by_python`` / ``init_items_python`` name is not
        registered after the module loads.
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
    """Load and validate a site YAML file, including its catalog.

    Returns a tuple ``(site_model, catalog)``. The site model is the
    pydantic representation (not an engine dataclass); the
    run-orchestration layer consumes both halves directly.

    ``extends:`` is supported at one level only in v1: the named
    parent file is loaded first, then the current file's fields
    deep-merge on top. The parent's own ``extends:`` field is
    silently ignored if present (deliberately — chained inheritance
    is a future feature).
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
