"""Tests for the pydantic schemas — Step, StepGraph,
ResourceSpec models and their conversion to engine dataclasses."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from altrios.workflow_engine.expressions import Expression
from altrios.workflow_engine.resources import ResourceSpec
from altrios.workflow_engine.schemas import (
    KNOWN_PRIMITIVES,
    ResourceSpecModel,
    StepGraphModel,
    StepModel,
)
from altrios.workflow_engine.steps import Step, StepGraph


# ---- StepModel -----------------------------------------------------


def test_step_model_basic():
    m = StepModel(id="s1", type="timeout", params={"duration": 5})
    engine = m.to_engine()
    assert isinstance(engine, Step)
    assert engine.id == "s1"
    assert engine.type == "timeout"
    assert engine.params["duration"] == 5
    assert engine.next is None


def test_step_model_with_next():
    m = StepModel(id="s1", type="bind", params={"name": "x", "value": 1}, next="s2")
    assert m.to_engine().next == "s2"


def test_step_model_rejects_unknown_type():
    with pytest.raises(ValidationError, match="Unknown step type"):
        StepModel(id="s1", type="not_a_real_primitive")


def test_step_model_rejects_extra_fields():
    with pytest.raises(ValidationError):
        StepModel(id="s1", type="bind", params={}, unexpected="oops")


def test_step_model_rejects_empty_id():
    with pytest.raises(ValidationError):
        StepModel(id="", type="bind", params={"name": "x", "value": 1})


def test_step_model_accepts_expression_in_params():
    """Expression objects (produced by yaml_expressions.convert) pass
    through the params dict untouched."""
    expr = Expression("entity.weight_t / 3.0")
    m = StepModel(id="t", type="timeout", params={"duration": expr})
    engine = m.to_engine()
    assert engine.params["duration"] is expr


def test_all_known_primitives_pass():
    """Sanity: every primitive in KNOWN_PRIMITIVES is accepted by
    StepModel without surprise."""
    for prim in KNOWN_PRIMITIVES:
        StepModel(id="x", type=prim)


# ---- StepGraphModel ------------------------------------------------


def test_step_graph_model_basic():
    m = StepGraphModel(
        name="g",
        entry="a",
        steps=[
            StepModel(id="a", type="timeout", params={"duration": 1}, next="b"),
            StepModel(id="b", type="log", params={"message": "done"}),
        ],
    )
    graph = m.to_engine()
    assert isinstance(graph, StepGraph)
    assert graph.name == "g"
    assert graph.entry == "a"
    assert sorted(graph.steps) == ["a", "b"]


def test_step_graph_rejects_duplicate_ids():
    with pytest.raises(ValidationError, match="Duplicate step id"):
        StepGraphModel(
            name="g",
            entry="a",
            steps=[
                StepModel(id="a", type="bind", params={"name": "x", "value": 1}),
                StepModel(id="a", type="bind", params={"name": "y", "value": 2}),
            ],
        )


def test_step_graph_rejects_empty_steps():
    with pytest.raises(ValidationError):
        StepGraphModel(name="g", entry="a", steps=[])


def test_step_graph_entry_missing_raises_at_to_engine():
    """Pydantic accepts the model (per-step shape is OK); the engine
    dataclass catches the cross-step invariant."""
    m = StepGraphModel(
        name="g",
        entry="missing",
        steps=[
            StepModel(id="a", type="log", params={"message": "x"}),
        ],
    )
    with pytest.raises(ValueError, match="entry"):
        m.to_engine()


# ---- ResourceSpecModel --------------------------------------------


def test_resource_spec_model_basic():
    m = ResourceSpecModel(
        name="cranes", kind="Resource", role="equipment", capacity=4
    )
    engine = m.to_engine()
    assert isinstance(engine, ResourceSpec)
    assert engine.name == "cranes"
    assert engine.kind == "Resource"
    assert engine.role == "equipment"
    assert engine.capacity == 4
    assert engine.partition_by is None
    assert engine.init_items is None


def test_resource_spec_default_capacity_is_1():
    m = ResourceSpecModel(name="r", kind="Resource", role="equipment")
    assert m.to_engine().capacity == 1


def test_resource_spec_rejects_unknown_kind():
    with pytest.raises(ValidationError, match="Unknown resource kind"):
        ResourceSpecModel(name="r", kind="NotAClass", role="equipment")


def test_resource_spec_accepts_custom_role():
    """Roles are open-ended (catalog vocab), so non-canonical roles are
    accepted without warning."""
    m = ResourceSpecModel(
        name="r", kind="Resource", role="aircraft_gate", capacity=2
    )
    assert m.to_engine().role == "aircraft_gate"


def test_resource_spec_rejects_extra_fields():
    with pytest.raises(ValidationError):
        ResourceSpecModel(
            name="r", kind="Resource", role="equipment", surprise="oops"
        )


def test_resource_spec_partition_by_python_requires_resolver():
    m = ResourceSpecModel(
        name="tracks",
        kind="Resource",
        role="infrastructure",
        partition_by_python="freight.track_keys",
    )
    with pytest.raises(ValueError, match="resolver"):
        m.to_engine()  # no resolver provided


def test_resource_spec_partition_by_python_resolved():
    m = ResourceSpecModel(
        name="tracks",
        kind="Resource",
        role="infrastructure",
        partition_by_python="freight.track_keys",
    )

    def resolver(name: str):
        assert name == "freight.track_keys"
        return lambda config, schedules: ["t1", "t2"]

    engine = m.to_engine(partition_by_resolver=resolver)
    assert engine.partition_by is not None
    assert engine.partition_by({}, {}) == ["t1", "t2"]


def test_resource_spec_init_items_python_resolved():
    m = ResourceSpecModel(
        name="stack",
        kind="Store",
        role="storage",
        capacity=10,
        init_items_python="freight.stack_init",
    )

    def resolver(name: str):
        return lambda config, schedules: ["c1", "c2", "c3"]

    engine = m.to_engine(init_items_resolver=resolver)
    assert engine.init_items({}, {}) == ["c1", "c2", "c3"]


def test_capacity_must_be_non_negative():
    with pytest.raises(ValidationError):
        ResourceSpecModel(
            name="r", kind="Resource", role="equipment", capacity=-1
        )
