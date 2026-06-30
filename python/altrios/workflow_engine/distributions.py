"""Probability distributions for scalar workflow parameters.

Any scalar param (``duration``, ``quantity``, ...) in a workflow YAML
may be specified as either a literal value (handled as a constant) or
as a distribution dict::

    duration: 30                                       # → Constant(30)
    duration: {dist: uniform, low: 28, high: 32}

**Scope (v1):** The engine ships only the distributions actually used
by the freight catalog today — :class:`Constant`, :class:`Uniform`,
and :class:`Poisson`. Adding a new distribution is a four-line change
(subclass :class:`Distribution`, register it in :data:`_DIST_TYPES`)
so we intentionally don't pre-ship distributions that no catalog
uses; YAGNI keeps the engine surface small until a real catalog
needs more.

Sampling is decoupled from random-state ownership: every
:meth:`Distribution.sample` method takes a ``numpy.random.Generator``
as input and returns a ``float``. The engine owns the RNG; this
module owns only distribution shape and parameter validation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


class DistributionError(Exception):
    """Raised when a distribution spec is malformed or sampling fails."""


# ---- Base class -----------------------------------------------------


@dataclass(frozen=True)
class Distribution(ABC):
    """Abstract base for scalar-sampling distributions.

    Subclasses are frozen dataclasses that validate their parameters
    in ``__post_init__`` and implement :meth:`sample`.
    """

    @abstractmethod
    def sample(self, rng: np.random.Generator) -> float:
        """Draw one sample using the provided RNG."""


# ---- Concrete distributions ----------------------------------------


@dataclass(frozen=True)
class Constant(Distribution):
    """Degenerate distribution returning ``value`` on every sample.

    Used both as the explicit ``{dist: constant, value: 30}`` spec and
    as the implicit form for bare-scalar params (``duration: 30``).
    """

    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
            raise DistributionError(
                f"Constant.value must be numeric, got {self.value!r} "
                f"({type(self.value).__name__})."
            )
        object.__setattr__(self, "value", float(self.value))

    def sample(self, rng: np.random.Generator) -> float:
        return self.value


@dataclass(frozen=True)
class Uniform(Distribution):
    """Continuous Uniform on ``[low, high]``.

    Models the ``random.uniform(low, high)`` jitter pattern used
    throughout the freight catalog's service-time generators.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        for nm, v in (("low", self.low), ("high", self.high)):
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise DistributionError(f"Uniform.{nm} must be numeric, got {v!r}.")
        if self.low > self.high:
            raise DistributionError(
                f"Uniform: low ({self.low}) must be <= high ({self.high})."
            )
        object.__setattr__(self, "low", float(self.low))
        object.__setattr__(self, "high", float(self.high))

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass(frozen=True)
class Poisson(Distribution):
    """Poisson distribution with mean ``rate``. Returns a float-typed
    non-negative integer so the engine's "sample returns float"
    contract holds; callers that need an int should coerce.

    Models the ``rng.poisson(rate)`` calls in
    :mod:`altrios.lifts.utilities` that draw per-window arrival
    counts.
    """

    rate: float

    def __post_init__(self) -> None:
        if not isinstance(self.rate, (int, float)) or isinstance(self.rate, bool):
            raise DistributionError(f"Poisson.rate must be numeric, got {self.rate!r}.")
        if self.rate < 0:
            raise DistributionError(f"Poisson.rate must be >= 0, got {self.rate}.")
        object.__setattr__(self, "rate", float(self.rate))

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.poisson(self.rate))


# ---- Parse / dispatch ----------------------------------------------


# YAML ``dist:`` tag → class. Keep alphabetical for ease of audit.
# Adding a new distribution is a one-line change here plus a new
# subclass above.
_DIST_TYPES: dict[str, type[Distribution]] = {
    "constant": Constant,
    "poisson": Poisson,
    "uniform": Uniform,
}


def known_distribution_names() -> tuple[str, ...]:
    """Sorted tuple of supported ``dist:`` tag names."""
    return tuple(sorted(_DIST_TYPES))


def parse_distribution(spec: Any) -> Distribution:
    """Convert a YAML scalar or distribution dict into a
    :class:`Distribution`.

    * Numeric scalar (``int`` / ``float``) → :class:`Constant`.
    * Mapping with ``dist:`` key → the corresponding class, with the
      remaining keys passed as constructor kwargs.

    Booleans and strings are rejected — both forms are common typos
    that we want to fail loud on rather than silently coerce.
    """
    if isinstance(spec, bool):
        raise DistributionError(
            f"Cannot parse {spec!r} as a distribution: booleans are "
            "not numeric. Use 0 or 1 explicitly if you mean a constant."
        )
    if isinstance(spec, (int, float)):
        return Constant(value=spec)
    if not isinstance(spec, Mapping):
        raise DistributionError(
            f"Distribution spec must be a number or a dict, "
            f"got {type(spec).__name__}: {spec!r}."
        )
    if "dist" not in spec:
        raise DistributionError(
            f"Distribution dict is missing required 'dist' key. "
            f"Got keys: {sorted(spec)}."
        )
    name = spec["dist"]
    if name not in _DIST_TYPES:
        raise DistributionError(
            f"Unknown distribution {name!r}. "
            f"Known: {list(known_distribution_names())}."
        )
    cls = _DIST_TYPES[name]
    kwargs = {k: v for k, v in spec.items() if k != "dist"}
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise DistributionError(
            f"Bad arguments for distribution {name!r}: {exc}"
        ) from exc
