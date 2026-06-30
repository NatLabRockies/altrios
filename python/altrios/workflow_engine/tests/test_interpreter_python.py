"""Tests for the Phase 3B.6 ``python`` escape-hatch primitive."""
from __future__ import annotations

import pytest
import simpy

from altrios.workflow_engine.expressions import Expression
from altrios.workflow_engine.interpreter import (
    ExecutionContext,
    InterpreterError,
    build_default_primitives,
    execute,
)
from altrios.workflow_engine.registry import CallableRegistry
from altrios.workflow_engine.steps import Step, StepGraph


def _ctx(env, registry, **kwargs) -> ExecutionContext:
    return ExecutionContext(
        env=env,
        primitives=build_default_primitives(),
        registry=registry,
        **kwargs,
    )


def _run(graph, ctx, env):
    proc = env.process(execute(graph, ctx))
    env.run(until=proc)


def test_python_calls_registered_function_and_binds_return():
    env = simpy.Environment()
    reg = CallableRegistry()
    reg.register("math.add", lambda a, b: a + b)
    ctx = _ctx(env, reg)
    g = StepGraph(
        name="g",
        entry="py",
        steps={
            "py": Step(
                id="py",
                type="python",
                params={
                    "call": "math.add",
                    "args": {"a": 3, "b": 4},
                    "bind": "result",
                },
            )
        },
    )
    _run(g, ctx, env)
    assert ctx.bindings["result"] == 7


def test_python_args_can_be_expressions():
    env = simpy.Environment()
    reg = CallableRegistry()
    reg.register("math.add", lambda a, b: a + b)
    ctx = _ctx(env, reg)
    ctx.bindings["x"] = 10
    g = StepGraph(
        name="g",
        entry="py",
        steps={
            "py": Step(
                id="py",
                type="python",
                params={
                    "call": "math.add",
                    "args": {"a": Expression("bindings.x"), "b": 5},
                    "bind": "result",
                },
            )
        },
    )
    _run(g, ctx, env)
    assert ctx.bindings["result"] == 15


def test_python_callable_can_yield_simpy_events():
    """A registered callable returning a generator is treated as a
    SimPy sub-process: the engine yields through it so the callable
    can perform timed work."""
    env = simpy.Environment()
    reg = CallableRegistry()

    def slow_double(env, x):
        yield env.timeout(2.5)
        return x * 2

    reg.register("slow.double", slow_double)
    ctx = _ctx(env, reg)
    g = StepGraph(
        name="g",
        entry="py",
        steps={
            "py": Step(
                id="py",
                type="python",
                params={
                    "call": "slow.double",
                    "args": {"env": Expression("env"), "x": 7},
                    "bind": "out",
                },
            )
        },
    )
    _run(g, ctx, env)
    assert env.now == 2.5
    assert ctx.bindings["out"] == 14


def test_python_without_bind_runs_for_side_effect_only():
    env = simpy.Environment()
    reg = CallableRegistry()
    log: list = []
    reg.register("side.append", lambda v: log.append(v) or None)
    ctx = _ctx(env, reg)
    g = StepGraph(
        name="g",
        entry="py",
        steps={
            "py": Step(
                id="py",
                type="python",
                params={"call": "side.append", "args": {"v": "hello"}},
            )
        },
    )
    _run(g, ctx, env)
    assert log == ["hello"]


def test_python_missing_callable_raises_interpreter_error():
    env = simpy.Environment()
    reg = CallableRegistry()
    ctx = _ctx(env, reg)
    g = StepGraph(
        name="g",
        entry="py",
        steps={
            "py": Step(
                id="py", type="python", params={"call": "nowhere"}
            )
        },
    )
    with pytest.raises(InterpreterError, match="nowhere"):
        _run(g, ctx, env)


def test_python_signature_mismatch_raises_interpreter_error():
    env = simpy.Environment()
    reg = CallableRegistry()
    reg.register("two_args", lambda a, b: a + b)
    ctx = _ctx(env, reg)
    g = StepGraph(
        name="g",
        entry="py",
        steps={
            "py": Step(
                id="py",
                type="python",
                params={"call": "two_args", "args": {"a": 1, "c": 99}},
            )
        },
    )
    with pytest.raises(InterpreterError, match="argument mismatch"):
        _run(g, ctx, env)


def test_python_without_registry_raises():
    env = simpy.Environment()
    ctx = ExecutionContext(
        env=env, primitives=build_default_primitives()
    )  # no registry
    g = StepGraph(
        name="g",
        entry="py",
        steps={
            "py": Step(
                id="py", type="python", params={"call": "noop"}
            )
        },
    )
    with pytest.raises(InterpreterError, match="registry"):
        _run(g, ctx, env)


def test_python_generator_callable_propagates_inner_errors():
    env = simpy.Environment()
    reg = CallableRegistry()

    def bad_helper(env):
        yield env.timeout(1.0)
        raise ValueError("boom")

    reg.register("bad", bad_helper)
    ctx = _ctx(env, reg)
    g = StepGraph(
        name="g",
        entry="py",
        steps={
            "py": Step(
                id="py",
                type="python",
                params={"call": "bad", "args": {"env": Expression("env")}},
            )
        },
    )
    with pytest.raises(InterpreterError, match="generator raised"):
        _run(g, ctx, env)
