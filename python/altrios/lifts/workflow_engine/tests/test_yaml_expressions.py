"""Tests for the expression-string converter used by the YAML loader."""
from __future__ import annotations

import pytest

from altrios.lifts.workflow_engine.expressions import Expression
from altrios.lifts.workflow_engine.yaml_expressions import (
    ExpressionConversionError,
    convert_expressions,
    extract_expression_source,
    is_expression_string,
)


# ---- is_expression_string ------------------------------------------


def test_is_expression_string_basic():
    assert is_expression_string("{a}")
    assert is_expression_string("{ entity.weight_t / 3.0 }")
    assert is_expression_string("  {expr}  ")  # outer whitespace tolerated


def test_is_expression_string_negatives():
    assert not is_expression_string("plain")
    assert not is_expression_string("loaded {n} items")  # partial brace
    assert not is_expression_string("{open")
    assert not is_expression_string("close}")
    assert not is_expression_string("")
    assert not is_expression_string("{")
    assert not is_expression_string("}")
    # Non-strings are never expression strings.
    assert not is_expression_string(123)
    assert not is_expression_string(None)
    assert not is_expression_string({"a": 1})


def test_extract_expression_source():
    assert extract_expression_source("{a}") == "a"
    assert extract_expression_source("{ entity.x }") == "entity.x"
    assert extract_expression_source("  {a + b}  ") == "a + b"


# ---- convert_expressions: scalars ---------------------------------


def test_convert_scalar_expression():
    result = convert_expressions("{x + 1}")
    assert isinstance(result, Expression)
    assert result.source == "x + 1"


def test_convert_scalar_literal_returned_unchanged():
    assert convert_expressions("plain") == "plain"
    assert convert_expressions(42) == 42
    assert convert_expressions(3.14) == 3.14
    assert convert_expressions(True) is True
    assert convert_expressions(None) is None


def test_convert_template_string_not_an_expression():
    """Strings with text outside the braces stay literal — they are
    templates, not expressions."""
    template = "loaded {n} items"
    assert convert_expressions(template) == template


# ---- convert_expressions: containers ------------------------------


def test_convert_in_dict():
    data = {
        "type": "bind",
        "name": "dist",
        "value": "{layout.distance('a', 'b')}",
        "literal": "hello",
    }
    out = convert_expressions(data)
    assert out["type"] == "bind"
    assert out["name"] == "dist"
    assert isinstance(out["value"], Expression)
    assert out["value"].source == "layout.distance('a', 'b')"
    assert out["literal"] == "hello"


def test_convert_in_list():
    data = ["a", "{x}", 42, {"v": "{y * 2}"}]
    out = convert_expressions(data)
    assert out[0] == "a"
    assert isinstance(out[1], Expression)
    assert out[1].source == "x"
    assert out[2] == 42
    assert isinstance(out[3]["v"], Expression)


def test_convert_nested_deep():
    data = {
        "steps": [
            {"id": "s1", "type": "timeout", "params": {"duration": "{2 * 3}"}},
            {"id": "s2", "type": "bind", "params": {"value": "{x + y}"}},
        ]
    }
    out = convert_expressions(data)
    assert isinstance(out["steps"][0]["params"]["duration"], Expression)
    assert isinstance(out["steps"][1]["params"]["value"], Expression)
    assert out["steps"][0]["id"] == "s1"  # untouched scalar


def test_convert_input_not_mutated():
    """The input structure must not be mutated; convert always returns
    a fresh container at each level."""
    data = {"v": "{x}"}
    _ = convert_expressions(data)
    assert isinstance(data["v"], str)  # original still a string


def test_convert_path_in_error_message():
    data = {"steps": [{"params": {"duration": "{1 +}"}}]}
    with pytest.raises(ExpressionConversionError) as excinfo:
        convert_expressions(data)
    msg = str(excinfo.value)
    assert "$.steps[0].params.duration" in msg


def test_convert_dunder_expression_rejected():
    """Dunder expressions are rejected at Expression construction,
    which the converter wraps with location info."""
    with pytest.raises(ExpressionConversionError, match="dunder"):
        convert_expressions("{__import__('os')}")


def test_convert_empty_expression_rejected():
    """The empty brace expression ``{}`` parses as a string then raises
    when Expression sees an empty source."""
    with pytest.raises(ExpressionConversionError):
        convert_expressions("{}")
