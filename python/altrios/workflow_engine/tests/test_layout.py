"""Tests for :mod:`altrios.workflow_engine.layout`."""
from __future__ import annotations

import pytest

from altrios.workflow_engine.layout import Layout, LayoutNode
from altrios.workflow_engine.schemas import LayoutModel


def test_layout_node_basic():
    n = LayoutNode(name="a", x=1.0, y=2.0)
    assert n.z is None


def test_layout_node_with_z():
    n = LayoutNode(name="a", x=1.0, y=2.0, z=3.0)
    assert n.z == 3.0


def test_from_dict_xy():
    lay = Layout.from_dict({"a": (0, 0), "b": (3, 4)})
    assert lay.node("a").x == 0
    assert lay.node("b").y == 4


def test_from_dict_xyz():
    lay = Layout.from_dict({"a": (0, 0, 1.5)})
    assert lay.node("a").z == 1.5


def test_from_dict_rejects_bad_arity():
    with pytest.raises(ValueError) as exc:
        Layout.from_dict({"bad": (1.0,)})
    assert "must be (x, y) or (x, y, z)" in str(exc.value)


def test_distance_manhattan_2d():
    lay = Layout.from_dict({"a": (0, 0), "b": (3, 4)})
    # Manhattan: |3-0| + |4-0| = 7 (NOT Euclidean 5)
    assert lay.distance("a", "b") == 7.0


def test_distance_symmetric():
    lay = Layout.from_dict({"a": (1, 2), "b": (10, 20)})
    assert lay.distance("a", "b") == lay.distance("b", "a")


def test_distance_zero_to_self():
    lay = Layout.from_dict({"a": (5, 5)})
    assert lay.distance("a", "a") == 0.0


def test_distance_ignores_z():
    """v1 explicitly ignores z (decision §6 — 2-D Manhattan)."""
    lay = Layout.from_dict({"a": (0, 0, 0), "b": (3, 4, 1000)})
    assert lay.distance("a", "b") == 7.0


def test_distance_negative_coords():
    lay = Layout.from_dict({"a": (-5, -10), "b": (5, 10)})
    assert lay.distance("a", "b") == 30.0


def test_unknown_node_raises_with_available_list():
    lay = Layout.from_dict({"a": (0, 0), "b": (1, 1)})
    with pytest.raises(KeyError) as exc:
        lay.node("c")
    msg = str(exc.value)
    assert "'c'" in msg
    assert "'a'" in msg and "'b'" in msg


def test_distance_unknown_node_raises():
    lay = Layout.from_dict({"a": (0, 0)})
    with pytest.raises(KeyError):
        lay.distance("a", "nope")


def test_from_model_roundtrip():
    model = LayoutModel(
        nodes={
            "berth_1": {"x": 0, "y": 0},
            "stack_A": {"x": 380, "y": 50, "z": 2.5},
        }
    )
    lay = Layout.from_model(model)
    assert lay.node("berth_1").x == 0
    assert lay.node("stack_A").z == 2.5
    assert lay.distance("berth_1", "stack_A") == 430.0  # 380 + 50


def test_len_and_contains():
    lay = Layout.from_dict({"a": (0, 0), "b": (1, 1)})
    assert len(lay) == 2
    assert "a" in lay
    assert "z" not in lay


def test_nodes_is_immutable():
    """workflow expressions must not be able to mutate the layout."""
    lay = Layout.from_dict({"a": (0, 0)})
    with pytest.raises(TypeError):
        lay.nodes["b"] = LayoutNode(name="b", x=1, y=1)  # type: ignore[index]


def test_direct_construction_rejects_key_mismatch():
    with pytest.raises(ValueError) as exc:
        Layout(nodes={"a": LayoutNode(name="b", x=0, y=0)})
    assert "disagrees with" in str(exc.value)


def test_direct_construction_rejects_wrong_type():
    with pytest.raises(TypeError) as exc:
        Layout(nodes={"a": (0, 0)})  # type: ignore[dict-item]
    assert "must be a LayoutNode" in str(exc.value)


def test_layout_usable_in_expression():
    """Smoke test: Layout exposed under ``layout`` name is queryable
    via asteval the same way the interpreter will use it."""
    from altrios.workflow_engine.expressions import Expression, ExpressionContext

    lay = Layout.from_dict({"a": (0, 0), "b": (3, 4)})
    ctx = ExpressionContext(layout=lay)
    assert Expression("layout.distance('a', 'b')").evaluate(ctx) == 7.0
    assert Expression("layout.node('b').x").evaluate(ctx) == 3.0
