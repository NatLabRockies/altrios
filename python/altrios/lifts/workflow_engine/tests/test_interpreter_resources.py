"""Tests for resource primitives: ``request``, ``release``,
``transfer``, ``record_event``, ``record_resource_event``,
``record_consumption``."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import simpy

from altrios.lifts.workflow_engine import (
    Entity,
    OutputCollector,
    ResourceSpec,
    build_state_from_specs,
)
from altrios.lifts.workflow_engine.expressions import Expression
from altrios.lifts.workflow_engine.interpreter import (
    ExecutionContext,
    InterpreterError,
    build_default_primitives,
    execute,
)
from altrios.lifts.workflow_engine.steps import Step, StepGraph


# ---- Test helpers ----------------------------------------------------


def _make_state(env: simpy.Environment, specs) -> SimpleNamespace:
    primitives = build_state_from_specs(env, specs, config={}, schedules={})
    state = SimpleNamespace()
    for name, prim in primitives.items():
        setattr(state, name, prim)
    return state


def _make_ctx(env, state=None, entity=None) -> ExecutionContext:
    return ExecutionContext(
        env=env,
        primitives=build_default_primitives(),
        state=state,
        entity=entity,
    )


def _run(graph, ctx, env):
    proc = env.process(execute(graph, ctx))
    env.run(until=proc)


# ---- request / release on Resource ----------------------------------


def test_request_release_resource():
    env = simpy.Environment()
    state = _make_state(env, [
        ResourceSpec(name="cranes", kind="Resource", role="equipment", capacity=2)
    ])
    ctx = _make_ctx(env, state)
    g = StepGraph(
        name="g",
        entry="req",
        steps={
            "req": Step(
                id="req",
                type="request",
                params={"pool": "cranes", "bind": "crane"},
                next="hold",
            ),
            "hold": Step(
                id="hold", type="timeout", params={"duration": 3.0}, next="rel"
            ),
            "rel": Step(
                id="rel",
                type="release",
                params={"pool": "cranes", "bind": "crane"},
            ),
        },
    )
    _run(g, ctx, env)
    assert env.now == 3.0
    assert state.cranes.count == 0  # released
    assert "crane" in ctx.bindings  # request object still bound


def test_request_resource_capacity_serializes_processes():
    """Two processes contend for a capacity-1 Resource; the second
    waits until the first releases."""
    env = simpy.Environment()
    state = _make_state(env, [
        ResourceSpec(name="single", kind="Resource", role="equipment", capacity=1)
    ])
    g = StepGraph(
        name="g",
        entry="req",
        steps={
            "req": Step(
                id="req",
                type="request",
                params={"pool": "single", "bind": "r"},
                next="work",
            ),
            "work": Step(
                id="work", type="timeout", params={"duration": 5.0}, next="rel"
            ),
            "rel": Step(
                id="rel", type="release", params={"pool": "single", "bind": "r"}
            ),
        },
    )
    ctx_a = _make_ctx(env, state)
    ctx_b = _make_ctx(env, state)
    env.process(execute(g, ctx_a))
    env.process(execute(g, ctx_b))
    env.run()
    assert env.now == 10.0  # serialized


# ---- request / release on Store -------------------------------------


def test_request_store_returns_item_and_release_puts_back():
    env = simpy.Environment()
    state = _make_state(env, [
        ResourceSpec(
            name="bag",
            kind="Store",
            role="storage",
            capacity=10,  # ResourceSpec defaults to 1, which would overfill the init.
            init_items=lambda c, s: ["apple", "banana"],
        )
    ])
    ctx = _make_ctx(env, state)
    g = StepGraph(
        name="g",
        entry="get",
        steps={
            "get": Step(
                id="get",
                type="request",
                params={"pool": "bag", "bind": "item"},
                next="put",
            ),
            "put": Step(
                id="put",
                type="release",
                params={"pool": "bag", "bind": "item"},
            ),
        },
    )
    _run(g, ctx, env)
    assert ctx.bindings["item"] == "apple"
    assert len(state.bag.items) == 2  # apple returned


def test_release_store_with_value_param():
    env = simpy.Environment()
    state = _make_state(env, [
        ResourceSpec(name="basket", kind="Store", role="storage")
    ])
    ctx = _make_ctx(env, state)
    g = StepGraph(
        name="g",
        entry="put",
        steps={
            "put": Step(
                id="put",
                type="release",
                params={"pool": "basket", "value": "egg"},
            )
        },
    )
    _run(g, ctx, env)
    assert state.basket.items == ["egg"]


# ---- request / release on Container ---------------------------------


def test_request_release_container():
    env = simpy.Environment()
    state = _make_state(env, [
        ResourceSpec(
            name="fuel",
            kind="Container",
            role="storage",
            capacity=100,
            init_items=None,
        )
    ])
    # Container has no init_items; pre-fill via direct put for test.
    state.fuel.put(80)
    ctx = _make_ctx(env, state)
    g = StepGraph(
        name="g",
        entry="get",
        steps={
            "get": Step(
                id="get",
                type="request",
                params={"pool": "fuel", "qty": 30, "bind": "qty"},
                next="put",
            ),
            "put": Step(
                id="put",
                type="release",
                params={"pool": "fuel", "qty": 10},
            ),
        },
    )
    _run(g, ctx, env)
    assert state.fuel.level == 60  # 80 - 30 + 10
    assert ctx.bindings["qty"] == 30


# ---- partitioned pools ---------------------------------------------


def test_request_partitioned_pool_by_key():
    env = simpy.Environment()
    state = _make_state(env, [
        ResourceSpec(
            name="tracks",
            kind="Resource",
            role="infrastructure",
            capacity=1,
            partition_by=lambda c, s: ["t1", "t2"],
        )
    ])
    ctx = _make_ctx(env, state)
    ctx.bindings["my_track"] = "t2"
    g = StepGraph(
        name="g",
        entry="req",
        steps={
            "req": Step(
                id="req",
                type="request",
                params={
                    "pool": "tracks",
                    "partition_key": Expression("bindings.my_track"),
                    "bind": "track",
                },
            )
        },
    )
    _run(g, ctx, env)
    assert state.tracks["t2"].count == 1
    assert state.tracks["t1"].count == 0


# ---- transfer ------------------------------------------------------


def test_transfer_moves_item_between_stores():
    env = simpy.Environment()
    state = _make_state(env, [
        ResourceSpec(
            name="source",
            kind="Store",
            role="storage",
            init_items=lambda c, s: ["pkg-7"],
        ),
        ResourceSpec(name="dest", kind="Store", role="storage"),
    ])
    ctx = _make_ctx(env, state)
    g = StepGraph(
        name="g",
        entry="t",
        steps={
            "t": Step(
                id="t",
                type="transfer",
                params={"from": "source", "to": "dest"},
            )
        },
    )
    _run(g, ctx, env)
    assert state.source.items == []
    assert state.dest.items == ["pkg-7"]


def test_transfer_specific_entity_matches_head():
    env = simpy.Environment()
    state = _make_state(env, [
        ResourceSpec(
            name="src",
            kind="Store",
            role="storage",
            init_items=lambda c, s: ["X"],
        ),
        ResourceSpec(name="dst", kind="Store", role="storage"),
    ])
    ctx = _make_ctx(env, state)
    ctx.bindings["target"] = "X"
    g = StepGraph(
        name="g",
        entry="t",
        steps={
            "t": Step(
                id="t",
                type="transfer",
                params={
                    "entity": Expression("bindings.target"),
                    "from": "src",
                    "to": "dst",
                },
            )
        },
    )
    _run(g, ctx, env)
    assert state.dst.items == ["X"]


def test_transfer_specific_entity_mismatch_raises():
    env = simpy.Environment()
    state = _make_state(env, [
        ResourceSpec(
            name="src",
            kind="Store",
            role="storage",
            init_items=lambda c, s: ["WRONG"],
        ),
        ResourceSpec(name="dst", kind="Store", role="storage"),
    ])
    ctx = _make_ctx(env, state)
    ctx.bindings["target"] = "RIGHT"
    g = StepGraph(
        name="g",
        entry="t",
        steps={
            "t": Step(
                id="t",
                type="transfer",
                params={
                    "entity": Expression("bindings.target"),
                    "from": "src",
                    "to": "dst",
                },
            )
        },
    )
    with pytest.raises(InterpreterError, match="expected entity"):
        _run(g, ctx, env)


# ---- record_event --------------------------------------------------


def test_record_event_populates_envelope():
    env = simpy.Environment()
    ctx = _make_ctx(env, entity=Entity(id="C-1", kind="container", attrs={}))
    env.process(_advance_then_record(env, ctx))
    env.run()
    rows = ctx.output.event_log
    assert len(rows) == 1
    r = rows[0]
    assert r["record_timestamp"] == 2.0
    assert r["event_type"] == "arrived"
    assert r["entity_id"] == "C-1"
    assert r["entity_kind"] == "container"


def _advance_then_record(env, ctx):
    yield env.timeout(2.0)
    g = StepGraph(
        name="g",
        entry="r",
        steps={
            "r": Step(
                id="r",
                type="record_event",
                params={"event_type": "arrived"},
            )
        },
    )
    yield from execute(g, ctx)


def test_record_event_with_extra_columns():
    env = simpy.Environment()
    ctx = _make_ctx(env, entity=Entity(id="C-2", kind="container", attrs={"weight_t": 12.0}))
    g = StepGraph(
        name="g",
        entry="r",
        steps={
            "r": Step(
                id="r",
                type="record_event",
                params={
                    "event_type": "loaded",
                    "columns": {
                        "zone": "stack_A",
                        "weight_t": Expression("entity.weight_t"),
                    },
                },
            )
        },
    )
    _run(g, ctx, env)
    row = ctx.output.event_log[0]
    assert row["zone"] == "stack_A"
    assert row["weight_t"] == 12.0


# ---- record_resource_event ----------------------------------------


def test_record_resource_event_captures_optionals():
    env = simpy.Environment()
    state = _make_state(env, [
        ResourceSpec(name="crane", kind="Resource", role="equipment", capacity=1)
    ])
    ctx = _make_ctx(env, state, entity=Entity(id="C-1", kind="container"))
    ctx.bindings["the_crane"] = state.crane
    g = StepGraph(
        name="g",
        entry="r",
        steps={
            "r": Step(
                id="r",
                type="record_resource_event",
                params={
                    "resource": Expression("bindings.the_crane"),
                    "event_type": "busy",
                    "duration": 4.5,
                    "status": "loading",
                    "role": "equipment",
                },
            )
        },
    )
    _run(g, ctx, env)
    row = ctx.output.resource_log[0]
    assert row["event_type"] == "busy"
    assert row["duration"] == 4.5
    assert row["status"] == "loading"
    assert row["role"] == "equipment"
    assert row["entity_id"] == "C-1"


# ---- record_consumption -------------------------------------------


def test_record_consumption_row_shape():
    env = simpy.Environment()
    state = _make_state(env, [
        ResourceSpec(name="rtg", kind="Resource", role="equipment", capacity=1)
    ])
    ctx = _make_ctx(env, state)
    ctx.bindings["the_rtg"] = state.rtg
    g = StepGraph(
        name="g",
        entry="c",
        steps={
            "c": Step(
                id="c",
                type="record_consumption",
                params={
                    "resource": Expression("bindings.the_rtg"),
                    "quantity": "energy",
                    "value": 1.42,
                    "status": "loading",
                    "duration": 3.0,
                    "fuel_type": "Diesel",
                    "role": "equipment",
                },
            )
        },
    )
    _run(g, ctx, env)
    row = ctx.output.consumption_log[0]
    assert row["quantity"] == "energy"
    assert row["consumption_value"] == 1.42
    assert row["fuel_type"] == "Diesel"
    assert row["status"] == "loading"
    assert row["duration"] == 3.0


# ---- OutputCollector basics --------------------------------------


def test_output_collector_independent_lists():
    oc = OutputCollector()
    oc.record_event({"event_type": "a"})
    oc.record_resource_event({"resource_id": 1})
    oc.record_consumption({"value": 1.0})
    assert len(oc.event_log) == 1
    assert len(oc.resource_log) == 1
    assert len(oc.consumption_log) == 1
