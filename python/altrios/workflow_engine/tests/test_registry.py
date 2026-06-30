"""Unit tests for :mod:`altrios.workflow_engine.registry`."""
from __future__ import annotations

import pytest

from altrios.workflow_engine.registry import (
    CallableRegistry,
    RegistryError,
    get_registry,
    register,
)


@pytest.fixture
def reg() -> CallableRegistry:
    """Per-test isolated registry to avoid module-level singleton crosstalk."""
    return CallableRegistry(name="test")


def test_register_and_call_direct(reg: CallableRegistry):
    reg.register("add", lambda a, b: a + b)
    assert reg.call("add", a=3, b=4) == 7
    assert "add" in reg
    assert len(reg) == 1
    assert reg.names() == ["add"]


def test_register_as_decorator(reg: CallableRegistry):
    @reg.register("mul")
    def mul(a: int, b: int) -> int:
        return a * b

    assert reg.call("mul", a=2, b=5) == 10


def test_register_duplicate_name_raises(reg: CallableRegistry):
    reg.register("x", lambda: 1)
    with pytest.raises(RegistryError, match="already registered"):
        reg.register("x", lambda: 2)


def test_register_non_callable_raises(reg: CallableRegistry):
    with pytest.raises(RegistryError, match="not callable"):
        reg.register("oops", 42)  # type: ignore[arg-type]


def test_register_invalid_name_raises(reg: CallableRegistry):
    with pytest.raises(RegistryError):
        reg.register("", lambda: 1)
    with pytest.raises(RegistryError):
        reg.register(None, lambda: 1)  # type: ignore[arg-type]


def test_get_unknown_name_lists_available(reg: CallableRegistry):
    reg.register("alpha", lambda: 1)
    reg.register("beta", lambda: 2)
    with pytest.raises(RegistryError, match="alpha.*beta|beta.*alpha"):
        reg.get("gamma")


def test_call_argument_mismatch_message_clear(reg: CallableRegistry):
    def fn(a: int, b: int) -> int:
        return a + b
    reg.register("fn", fn)
    with pytest.raises(RegistryError, match="argument mismatch"):
        reg.call("fn", a=1, c=99)  # 'c' doesn't exist, 'b' missing


def test_unregister(reg: CallableRegistry):
    reg.register("x", lambda: 1)
    reg.unregister("x")
    assert "x" not in reg
    with pytest.raises(RegistryError):
        reg.unregister("x")


def test_module_level_register_and_get():
    """The default registry singleton works through the module-level
    helpers — used by catalogs in ``python_helpers.py`` modules."""
    default = get_registry()
    # Use a globally-unique name to avoid stomping on prior runs.
    name = "test_registry_module_level_unique_name_xyz"
    # If a prior test left it lying around (e.g. interactive reload),
    # clean up first.
    if name in default:
        default.unregister(name)
    try:
        @register(name)
        def fn(x):
            return x * 2

        assert default.call(name, x=5) == 10
    finally:
        if name in default:
            default.unregister(name)


def test_call_passes_only_signature_args(reg: CallableRegistry):
    """When the registered callable defines a typed signature, the
    registry validates that the kwargs match before invoking; surplus
    kwargs are rejected."""
    def two_args(a: int, b: int) -> int:
        return a - b
    reg.register("two", two_args)
    with pytest.raises(RegistryError, match="argument mismatch"):
        reg.call("two", a=1, b=2, c=3)


def test_kwargs_only_call_works(reg: CallableRegistry):
    def f(*, x: int, y: int = 5) -> int:
        return x + y
    reg.register("kw", f)
    assert reg.call("kw", x=3) == 8
    assert reg.call("kw", x=3, y=10) == 13
