"""Catalog-supplied Python callable registry.

The step interpreter's ``python:`` primitive and the distributions
library's ``python:`` custom-distribution hook look up callables
through this module. A catalog declares its Python helpers module via
its YAML ``python_module:`` field; the YAML loader imports that module,
which uses :func:`register` at import-time to publish individual
callables.

Usage from a catalog's ``python_helpers.py``::

    from altrios.lifts.workflow_engine.registry import register

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
        """Register ``fn`` under ``name``.

        Usable either as a decorator factory
        (``@registry.register("foo")``) or as a direct call
        (``registry.register("foo", fn)``). Duplicate registrations
        are refused so that two catalogs (or two reloads of the same
        module) cannot silently shadow each other.

        Parameters
        ----------
        name : str
            The non-empty registry name. Dotted prefixes
            (``"freight.choose_track"``) are convention only; the
            registry treats names as flat strings.
        fn : Callable, optional
            The callable to register. Omit to use as a decorator
            factory; pass directly to register an existing function.

        Returns
        -------
        Callable
            When ``fn`` is provided, returns ``fn`` unchanged (for
            chaining). When ``fn`` is ``None``, returns a decorator
            that registers and returns its argument.

        Raises
        ------
        RegistryError
            If ``name`` is empty, the target is not callable, or
            ``name`` is already registered.
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
        """Remove ``name`` from the registry.

        Intended for use in tests that build up isolated registries;
        catalog production code should not call this.

        Parameters
        ----------
        name : str
            The registered name to remove.

        Raises
        ------
        RegistryError
            If ``name`` is not currently registered.
        """
        if name not in self._callables:
            raise RegistryError(f"Cannot unregister unknown name {name!r}.")
        del self._callables[name]

    def get(self, name: str) -> Callable[..., Any]:
        """Look up a registered callable.

        Parameters
        ----------
        name : str
            The registered name to look up.

        Returns
        -------
        Callable
            The callable registered under ``name``.

        Raises
        ------
        RegistryError
            If ``name`` is not registered. The message lists the sorted
            set of currently registered names so typos surface
            immediately.
        """
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
        """Return the sorted list of currently registered names.

        Returns
        -------
        list of str
            Registered names in alphabetical order.
        """
        return sorted(self._callables)

    def call(self, name: str, /, **kwargs: Any) -> Any:
        """Invoke the named callable with keyword arguments.

        Validates ``kwargs`` against the target callable's signature so
        a YAML ``args:`` typo surfaces as a clear
        :class:`RegistryError` rather than a misleading ``TypeError``
        deep inside the callable.

        Parameters
        ----------
        name : str
            The registered callable name to invoke. Positional-only
            so it cannot collide with a callable kwarg named ``name``.
        **kwargs
            Keyword arguments forwarded to the registered callable.

        Returns
        -------
        Any
            Whatever the registered callable returns.

        Raises
        ------
        RegistryError
            If ``name`` is not registered, or if ``kwargs`` do not
            match the callable's signature.
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
    """Return the module-level default :class:`CallableRegistry` singleton.

    Returns
    -------
    CallableRegistry
        The shared registry that the YAML loader and step interpreter
        consult unless a per-run override is supplied.
    """
    return _DEFAULT_REGISTRY


def register(
    name: str,
    fn: Optional[Callable[..., Any]] = None,
) -> Callable[..., Any]:
    """Shorthand for ``get_registry().register(name, fn)``.

    Usable as a decorator (``@register("name")``) or as a direct call.

    Parameters
    ----------
    name : str
        The non-empty registry name to register under.
    fn : Callable, optional
        The callable to register. Omit to use as a decorator factory.

    Returns
    -------
    Callable
        When ``fn`` is provided, returns ``fn`` unchanged. When ``fn``
        is ``None``, returns a decorator that registers and returns
        its argument.

    Raises
    ------
    RegistryError
        Propagated from :meth:`CallableRegistry.register`.
    """
    return _DEFAULT_REGISTRY.register(name, fn)
