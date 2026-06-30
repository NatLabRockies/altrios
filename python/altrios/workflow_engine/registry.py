"""Catalog-supplied Python callable registry.

The step interpreter's ``python:`` primitive (Phase 3B) and the
distributions library's ``python:`` custom-distribution hook (Phase 3D)
look up callables through this module. A catalog declares its Python
helpers module via its YAML ``python_module:`` field; the YAML loader
imports that module, which uses :func:`register` at import-time to
publish individual callables.

Usage from a catalog's ``python_helpers.py``::

    from altrios.workflow_engine.registry import register

    @register("freight.choose_track")
    def choose_track(env, terminal, train_id: int) -> int:
        ...

YAML reference from a step::

    - type: python
      call: freight.choose_track
      args:
        env: env
        terminal: state.terminal
        train_id: entity.train_id
      bind: chosen_track

Names are flat (no nested namespaces); a dotted prefix is convention
only. Names are global to a single ``CallableRegistry`` instance — by
default the engine uses a module-level singleton accessible via
:func:`get_registry`, but tests and the YAML loader can construct
isolated registries (e.g. one per loaded catalog) to prevent test
crosstalk.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


class RegistryError(Exception):
    """Raised for duplicate registrations, unknown names, or arity
    mismatches at call time."""


@dataclass
class CallableRegistry:
    """An isolated, mutable registry of named Python callables.

    Each entry maps ``name -> callable``. Registrations are append-only
    by default; ``unregister`` is provided for tests but should not be
    used by catalog code in production.
    """
    name: str = "default"
    _callables: dict[str, Callable[..., Any]] = field(default_factory=dict)

    def register(
        self,
        name: str,
        fn: Optional[Callable[..., Any]] = None,
    ) -> Callable[..., Any]:
        """Register ``fn`` under ``name``. Usable as a decorator factory
        (``@registry.register("foo")``) or a direct call
        (``registry.register("foo", fn)``).

        Raises :class:`RegistryError` if ``name`` is already taken; this
        prevents two catalogs (or two reloads of the same module) from
        silently shadowing each other.
        """
        if not isinstance(name, str) or not name:
            raise RegistryError(
                f"Callable name must be a non-empty str, got {name!r}."
            )

        def _do_register(target: Callable[..., Any]) -> Callable[..., Any]:
            if not callable(target):
                raise RegistryError(
                    f"register({name!r}): target {target!r} is not callable."
                )
            if name in self._callables:
                existing = self._callables[name]
                raise RegistryError(
                    f"Callable name {name!r} already registered "
                    f"to {existing!r}; would shadow with {target!r}."
                )
            self._callables[name] = target
            return target

        if fn is None:
            return _do_register
        return _do_register(fn)

    def unregister(self, name: str) -> None:
        """Remove ``name`` from the registry. Test-utility only."""
        if name not in self._callables:
            raise RegistryError(f"Cannot unregister unknown name {name!r}.")
        del self._callables[name]

    def get(self, name: str) -> Callable[..., Any]:
        """Look up a registered callable. Raises :class:`RegistryError`
        when the name is unknown (includes the list of registered names
        in the error to aid debugging typos)."""
        try:
            return self._callables[name]
        except KeyError:
            available = sorted(self._callables)
            raise RegistryError(
                f"No callable registered under {name!r}. "
                f"Registered names: {available}."
            )

    def __contains__(self, name: object) -> bool:
        return name in self._callables

    def __len__(self) -> int:
        return len(self._callables)

    def names(self) -> list[str]:
        """Return the sorted list of currently registered names."""
        return sorted(self._callables)

    def call(self, name: str, /, **kwargs: Any) -> Any:
        """Invoke the named callable with keyword arguments.

        Validates argument names against the callable's signature so a
        typo in a YAML ``args:`` dict surfaces as a clear
        :class:`RegistryError` rather than a misleading TypeError.
        """
        fn = self.get(name)
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            # Some C-implemented callables can't be introspected; fall
            # back to a direct call.
            return fn(**kwargs)
        try:
            bound = sig.bind(**kwargs)
        except TypeError as exc:
            raise RegistryError(
                f"call({name!r}) argument mismatch: {exc}. "
                f"Signature: {sig}; provided keys: {sorted(kwargs)}."
            ) from exc
        return fn(*bound.args, **bound.kwargs)


# Module-level default registry. The YAML loader populates this when it
# imports a catalog's python_module; the interpreter looks here unless a
# per-run registry override is supplied.
_DEFAULT_REGISTRY = CallableRegistry(name="default")


def get_registry() -> CallableRegistry:
    """Return the module-level default registry."""
    return _DEFAULT_REGISTRY


def register(
    name: str,
    fn: Optional[Callable[..., Any]] = None,
) -> Callable[..., Any]:
    """Shorthand for ``get_registry().register(name, fn)``.

    Usable as a decorator (``@register("name")``) or a direct call.
    """
    return _DEFAULT_REGISTRY.register(name, fn)
