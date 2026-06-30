"""Tests for :mod:`altrios.lifts.workflow_engine.distributions`."""
from __future__ import annotations

import numpy as np
import pytest

from altrios.lifts.workflow_engine.distributions import (
    Constant,
    Distribution,
    DistributionError,
    Poisson,
    Uniform,
    known_distribution_names,
    parse_distribution,
)


# ---- Constant -------------------------------------------------------


def test_constant_basic():
    c = Constant(value=42)
    assert c.value == 42.0
    rng = np.random.default_rng(0)
    assert c.sample(rng) == 42.0


def test_constant_accepts_int_or_float():
    assert Constant(value=3).value == 3.0
    assert Constant(value=3.5).value == 3.5


def test_constant_rejects_non_numeric():
    with pytest.raises(DistributionError):
        Constant(value="hi")  # type: ignore[arg-type]


def test_constant_rejects_bool():
    with pytest.raises(DistributionError):
        Constant(value=True)


# ---- Uniform --------------------------------------------------------


def test_uniform_basic():
    u = Uniform(low=0, high=1)
    rng = np.random.default_rng(0)
    samples = [u.sample(rng) for _ in range(100)]
    assert all(0 <= s <= 1 for s in samples)


def test_uniform_rejects_low_above_high():
    with pytest.raises(DistributionError) as exc:
        Uniform(low=5, high=2)
    assert "low" in str(exc.value)


def test_uniform_low_equals_high_degenerate():
    u = Uniform(low=5, high=5)
    rng = np.random.default_rng(0)
    assert u.sample(rng) == 5.0


def test_uniform_rejects_non_numeric():
    with pytest.raises(DistributionError):
        Uniform(low="a", high=1)  # type: ignore[arg-type]


# ---- Poisson --------------------------------------------------------


def test_poisson_basic():
    p = Poisson(rate=3.0)
    rng = np.random.default_rng(0)
    samples = [p.sample(rng) for _ in range(2000)]
    assert all(s >= 0 and s == int(s) for s in samples)
    assert abs(float(np.mean(samples)) - 3.0) < 0.2


def test_poisson_zero_rate_returns_zero():
    p = Poisson(rate=0.0)
    rng = np.random.default_rng(0)
    assert all(p.sample(rng) == 0.0 for _ in range(20))


def test_poisson_rejects_negative_rate():
    with pytest.raises(DistributionError):
        Poisson(rate=-1.0)


# ---- parse_distribution --------------------------------------------


def test_parse_scalar_becomes_constant():
    d = parse_distribution(30)
    assert isinstance(d, Constant)
    assert d.value == 30.0


def test_parse_float_scalar():
    assert isinstance(parse_distribution(3.14), Constant)


def test_parse_rejects_bool():
    with pytest.raises(DistributionError):
        parse_distribution(True)


def test_parse_rejects_string():
    with pytest.raises(DistributionError):
        parse_distribution("30")


def test_parse_rejects_dict_missing_dist():
    with pytest.raises(DistributionError) as exc:
        parse_distribution({"low": 0, "high": 1})
    assert "'dist' key" in str(exc.value)


def test_parse_rejects_unknown_dist():
    """Unknown name surfaces the known-list to aid debugging typos."""
    with pytest.raises(DistributionError) as exc:
        parse_distribution({"dist": "weibull", "shape": 1})
    msg = str(exc.value)
    assert "Unknown distribution 'weibull'" in msg
    assert "uniform" in msg


def test_parse_rejects_bad_arguments():
    with pytest.raises(DistributionError) as exc:
        parse_distribution({"dist": "uniform", "low": 0, "high": 1, "extra": 9})
    assert "Bad arguments" in str(exc.value)


def test_parse_dispatch_all_three():
    cases = [
        (30, Constant),
        ({"dist": "constant", "value": 30}, Constant),
        ({"dist": "uniform", "low": 0, "high": 1}, Uniform),
        ({"dist": "poisson", "rate": 5}, Poisson),
    ]
    for spec, expected in cases:
        d = parse_distribution(spec)
        assert isinstance(d, expected), f"{spec} -> {type(d).__name__}"


def test_known_distribution_names():
    assert known_distribution_names() == ("constant", "poisson", "uniform")


# ---- Immutability / hashing ----------------------------------------


def test_distributions_are_hashable():
    """Frozen dataclasses are hashable; useful if a workflow
    de-duplicates parsed distributions."""
    assert hash(Uniform(low=0, high=1)) == hash(Uniform(low=0, high=1))


def test_distributions_are_immutable():
    d = Uniform(low=0, high=1)
    with pytest.raises(Exception):
        d.low = 5  # type: ignore[misc]
