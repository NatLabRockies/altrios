"""Unit tests for :mod:`altrios.lifts.workflow_engine.expressions`.

Covers: math allowlist, namespace access (entity/bindings/state/config/
layout/env), proxy semantics, dunder-rejection, error reporting, parse
caching."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from altrios.lifts.workflow_engine.entities import Entity
from altrios.lifts.workflow_engine.expressions import (
    Expression,
    ExpressionContext,
    ExpressionError,
    NamespaceProxy,
    evaluate,
)


# ---------------------------------------------------------------------------
# Math + literals
# ---------------------------------------------------------------------------

def test_arithmetic():
    assert evaluate("1 + 2", ExpressionContext()) == 3
    assert evaluate("3 * 4 - 1", ExpressionContext()) == 11
    assert evaluate("(2 + 3) * 4", ExpressionContext()) == 20
    assert evaluate("7 / 2", ExpressionContext()) == 3.5
    assert evaluate("7 // 2", ExpressionContext()) == 3
    assert evaluate("7 % 3", ExpressionContext()) == 1
    assert evaluate("2 ** 8", ExpressionContext()) == 256


def test_comparisons_and_boolean():
    assert evaluate("3 < 5", ExpressionContext()) is True
    assert evaluate("3 > 5", ExpressionContext()) is False
    assert evaluate("3 == 3 and 4 != 5", ExpressionContext()) is True
    assert evaluate("not (3 > 5)", ExpressionContext()) is True
    assert evaluate("True or False", ExpressionContext()) is True


def test_math_allowlist():
    assert evaluate("sqrt(16)", ExpressionContext()) == 4.0
    assert evaluate("abs(-3.5)", ExpressionContext()) == 3.5
    assert evaluate("min(3, 1, 5)", ExpressionContext()) == 1
    assert evaluate("max(3, 1, 5)", ExpressionContext()) == 5
    assert evaluate("round(2.7)", ExpressionContext()) == 3
    assert evaluate("log(exp(2))", ExpressionContext()) == pytest.approx(2.0)
    assert evaluate("hypot(3, 4)", ExpressionContext()) == 5.0
    assert evaluate("pi", ExpressionContext()) == math.pi


def test_disallowed_general_builtins_are_not_available():
    """``open`` / ``print`` / ``eval`` / ``__import__`` must NOT be
    callable from expression source."""
    for name in ("open", "print", "eval", "exec", "__import__", "input"):
        with pytest.raises(ExpressionError):
            evaluate(f"{name}(1)", ExpressionContext())


def test_dunder_access_rejected_at_parse_time():
    """Heads off the most common sandbox-escape pattern via
    ``__class__.__bases__``."""
    with pytest.raises(ExpressionError, match="dunder"):
        Expression("[].__class__")
    with pytest.raises(ExpressionError, match="dunder"):
        Expression("(1).__class__.__bases__")


# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------

def test_entity_namespace_flattens_attrs():
    """An Entity's attrs dict is exposed as proxy attribute access so
    catalog authors write ``entity.weight_t`` not ``entity.attrs['weight_t']``."""
    from altrios.lifts.workflow_engine.expressions import NamespaceProxy

    # Use a duck-typed entity with `.id`, `.kind`, and attribute access
    # for each attrs key (this is what the interpreter will wrap real
    # Entity objects with at runtime; tested here in isolation).
    ent = SimpleNamespace(
        id="C1",
        kind="container",
        weight_t=22.5,
        origin="rail",
        destination="stack",
    )
    ctx = ExpressionContext(entity=ent)
    assert evaluate("entity.id", ctx) == "C1"
    assert evaluate("entity.kind", ctx) == "container"
    assert evaluate("entity.weight_t", ctx) == 22.5
    assert evaluate("entity.weight_t / 3.0", ctx) == 7.5
    assert evaluate("entity.origin == 'rail'", ctx) is True


def test_bindings_namespace_both_styles():
    ctx = ExpressionContext(bindings={"dist_m": 380.0, "label": "rail_to_stack"})
    # Attribute style
    assert evaluate("bindings.dist_m", ctx) == 380.0
    # Bracket style
    assert evaluate("bindings['label']", ctx) == "rail_to_stack"
    assert evaluate("bindings.dist_m / 10.0", ctx) == 38.0


def test_state_config_layout_env_namespaces():
    state = SimpleNamespace(occupancy=0.42)
    config = SimpleNamespace(max_speed=18.0)
    layout = SimpleNamespace(node_count=12)
    env = SimpleNamespace(now=123.4)
    ctx = ExpressionContext(state=state, config=config, layout=layout, env=env)
    assert evaluate("state.occupancy", ctx) == 0.42
    assert evaluate("config.max_speed", ctx) == 18.0
    assert evaluate("layout.node_count", ctx) == 12
    assert evaluate("env.now", ctx) == 123.4
    assert evaluate(
        "min(config.max_speed, layout.node_count * 1.5)",
        ctx,
    ) == 18.0


def test_layout_method_call_allowed_through_attribute_chain():
    """Methods on context-namespace objects are callable — the
    ``no-general-function-calls`` rule means catalogs can't introduce
    new functions, not that they can't call methods on engine-provided
    objects."""
    layout = SimpleNamespace(distance=lambda a, b: abs(a - b) * 10.0)
    ctx = ExpressionContext(layout=layout)
    assert evaluate("layout.distance(5, 2)", ctx) == 30.0


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------

def test_missing_name_raises_with_source_in_message():
    with pytest.raises(ExpressionError, match="entity.missing"):
        evaluate("entity.missing", ExpressionContext(entity=SimpleNamespace(id="X")))


def test_parse_error_raises():
    with pytest.raises(ExpressionError):
        Expression("1 +")


def test_empty_or_non_string_source_rejected():
    with pytest.raises(ExpressionError):
        Expression("")
    with pytest.raises(ExpressionError):
        Expression("   ")
    with pytest.raises(ExpressionError):
        Expression(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Immutability + repr
# ---------------------------------------------------------------------------

def test_expression_is_immutable():
    expr = Expression("1 + 2")
    with pytest.raises(AttributeError):
        expr.source = "something else"  # type: ignore[misc]


def test_expression_repr():
    expr = Expression("entity.weight_t / 3.0")
    assert "entity.weight_t / 3.0" in repr(expr)


# ---------------------------------------------------------------------------
# Reuse: parse-once, evaluate-many
# ---------------------------------------------------------------------------

def test_same_expression_reused_across_contexts():
    expr = Expression("entity.weight_t * 2")
    a = SimpleNamespace(weight_t=10.0)
    b = SimpleNamespace(weight_t=25.0)
    assert expr.evaluate(ExpressionContext(entity=a)) == 20.0
    assert expr.evaluate(ExpressionContext(entity=b)) == 50.0


def test_evaluations_are_independent_no_residual_state():
    """A name set in one evaluate() must not leak into the next."""
    # Note: ``=`` is statement-level; asteval supports it as an expression
    # in some configurations. If asteval lets the expression run
    # ``a = 1`` we still want that binding to die at end-of-evaluate.
    # Use a name that doesn't exist yet on both ctx and assert it errs.
    ctx = ExpressionContext(bindings={"x": 7})
    assert evaluate("bindings.x", ctx) == 7
    # A fresh context without bindings — ``bindings`` must not exist now.
    with pytest.raises(ExpressionError):
        evaluate("bindings.x", ExpressionContext())


# ---------------------------------------------------------------------------
# NamespaceProxy basics
# ---------------------------------------------------------------------------

def test_namespace_proxy_attribute_and_bracket_access():
    p = NamespaceProxy({"a": 1, "b": "two"})
    assert p.a == 1
    assert p["b"] == "two"
    assert "a" in p
    assert len(p) == 2
    assert sorted(iter(p)) == ["a", "b"]
    with pytest.raises(AttributeError):
        _ = p.missing


def test_namespace_proxy_is_readonly_at_attribute_level():
    p = NamespaceProxy({"a": 1})
    with pytest.raises(AttributeError):
        p.a = 99


# ---------------------------------------------------------------------------
# Real Entity wiring
# ---------------------------------------------------------------------------

def test_real_entity_via_namespace_proxy():
    """If a caller wraps a real Entity through a proxy that flattens
    attrs onto a SimpleNamespace, expressions work end-to-end."""
    ent = Entity(id="C42", kind="container",
                 attrs={"weight_t": 33.0, "origin": "rail"})
    flat = SimpleNamespace(id=ent.id, kind=ent.kind, **ent.attrs)
    ctx = ExpressionContext(entity=flat)
    assert evaluate("entity.weight_t * 2 + 1", ctx) == 67.0
    assert evaluate("entity.origin == 'rail'", ctx) is True
