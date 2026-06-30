"""Tests for the simple primitives: ``bind``, ``set_attr``,
``branch``, ``assert``, ``log``, ``timeout``, plus the StepGraph
validators and the interpreter's outer loop."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import simpy

from altrios.lifts.workflow_engine.expressions import Expression
from altrios.lifts.workflow_engine.interpreter import (
    AssertionFailure,
    ExecutionContext,
    InterpreterError,
    build_default_primitives,
    execute,
)
from altrios.lifts.workflow_engine.steps import Step, StepGraph


# ---- StepGraph / Step validators -------------------------------------


def test_step_validates_id_and_type():
    with pytest.raises(ValueError):
        Step(id="", type="bind")
    with pytest.raises(ValueError):
        Step(id="x", type="")
    with pytest.raises(ValueError):
        Step(id="x", type="bind", next="")


def test_step_params_frozen():
    s = Step(id="x", type="bind", params={"a": 1})
    with pytest.raises(TypeError):
        s.params["a"] = 2  # type: ignore[index]


def test_stepgraph_entry_must_exist():
    s = Step(id="x", type="log", params={"message": "hi"})
    with pytest.raises(ValueError, match="entry"):
        StepGraph(name="g", entry="missing", steps={"x": s})


def test_stepgraph_id_must_match_key():
    s = Step(id="actual", type="log", params={"message": "hi"})
    with pytest.raises(ValueError, match="does not match"):
        StepGraph(name="g", entry="x", steps={"x": s})


def test_stepgraph_next_must_exist():
    s1 = Step(id="a", type="log", params={"message": "hi"}, next="b")
    with pytest.raises(ValueError, match="does not name"):
        StepGraph(name="g", entry="a", steps={"a": s1})


def test_stepgraph_freezes_steps_mapping():
    s = Step(id="a", type="log", params={"message": "hi"})
    g = StepGraph(name="g", entry="a", steps={"a": s})
    with pytest.raises(TypeError):
        g.steps["a"] = s  # type: ignore[index]


# ---- Test helpers ----------------------------------------------------


def _make_ctx(env, *, entity=None) -> ExecutionContext:
    return ExecutionContext(
        env=env,
        primitives=build_default_primitives(),
        entity=entity,
    )


def _run(graph: StepGraph, ctx: ExecutionContext, env: simpy.Environment):
    """Schedule the graph as a SimPy process and run to completion."""
    proc = env.process(execute(graph, ctx))
    env.run(until=proc)


# ---- bind ------------------------------------------------------------


def test_bind_literal():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    g = StepGraph(
        name="g",
        entry="b",
        steps={"b": Step(id="b", type="bind", params={"name": "x", "value": 7})},
    )
    _run(g, ctx, env)
    assert ctx.bindings == {"x": 7}


def test_bind_expression_evaluates_against_state():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    ctx.state = SimpleNamespace(count=4)
    g = StepGraph(
        name="g",
        entry="b",
        steps={
            "b": Step(
                id="b",
                type="bind",
                params={"name": "doubled", "value": Expression("state.count * 2")},
            )
        },
    )
    _run(g, ctx, env)
    assert ctx.bindings["doubled"] == 8


def test_bind_requires_name_and_value():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    g = StepGraph(
        name="g",
        entry="b",
        steps={"b": Step(id="b", type="bind", params={"value": 1})},
    )
    with pytest.raises(InterpreterError, match="missing required param 'name'"):
        _run(g, ctx, env)


# ---- set_attr --------------------------------------------------------


def test_set_attr_on_entity_attrs_dict():
    env = simpy.Environment()
    entity = SimpleNamespace(id="E1", attrs={"status": "queued"})
    ctx = _make_ctx(env, entity=entity)
    g = StepGraph(
        name="g",
        entry="s",
        steps={
            "s": Step(
                id="s",
                type="set_attr",
                params={"attr": "status", "value": "loaded"},
            )
        },
    )
    _run(g, ctx, env)
    assert entity.attrs["status"] == "loaded"


def test_set_attr_on_plain_object_via_setattr():
    env = simpy.Environment()
    entity = SimpleNamespace(id="E1")  # no .attrs dict
    ctx = _make_ctx(env, entity=entity)
    g = StepGraph(
        name="g",
        entry="s",
        steps={
            "s": Step(
                id="s",
                type="set_attr",
                params={"attr": "color", "value": "red"},
            )
        },
    )
    _run(g, ctx, env)
    assert entity.color == "red"


def test_set_attr_with_no_target_raises():
    env = simpy.Environment()
    ctx = _make_ctx(env)  # no entity
    g = StepGraph(
        name="g",
        entry="s",
        steps={
            "s": Step(
                id="s",
                type="set_attr",
                params={"attr": "x", "value": 1},
            )
        },
    )
    with pytest.raises(InterpreterError, match="no target"):
        _run(g, ctx, env)


# ---- branch ----------------------------------------------------------


def test_branch_takes_true_arm():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    g = StepGraph(
        name="g",
        entry="br",
        steps={
            "br": Step(
                id="br",
                type="branch",
                params={"condition": True, "true": "tt", "false": "ff"},
            ),
            "tt": Step(id="tt", type="bind", params={"name": "took", "value": "T"}),
            "ff": Step(id="ff", type="bind", params={"name": "took", "value": "F"}),
        },
    )
    _run(g, ctx, env)
    assert ctx.bindings["took"] == "T"


def test_branch_takes_false_arm_via_expression():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    ctx.bindings["x"] = 0
    g = StepGraph(
        name="g",
        entry="br",
        steps={
            "br": Step(
                id="br",
                type="branch",
                params={
                    "condition": Expression("bindings.x > 5"),
                    "true": "tt",
                    "false": "ff",
                },
            ),
            "tt": Step(id="tt", type="bind", params={"name": "took", "value": "T"}),
            "ff": Step(id="ff", type="bind", params={"name": "took", "value": "F"}),
        },
    )
    _run(g, ctx, env)
    assert ctx.bindings["took"] == "F"


def test_branch_with_null_target_ends_workflow():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    g = StepGraph(
        name="g",
        entry="br",
        steps={
            "br": Step(
                id="br",
                type="branch",
                params={"condition": True, "true": None, "false": "ff"},
            ),
            "ff": Step(id="ff", type="bind", params={"name": "took", "value": "F"}),
        },
    )
    _run(g, ctx, env)
    assert "took" not in ctx.bindings


# ---- assert ----------------------------------------------------------


def test_assert_passes_silently_when_true():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    g = StepGraph(
        name="g",
        entry="a",
        steps={"a": Step(id="a", type="assert", params={"condition": True})},
    )
    _run(g, ctx, env)  # no exception


def test_assert_fails_with_message():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    g = StepGraph(
        name="g",
        entry="a",
        steps={
            "a": Step(
                id="a",
                type="assert",
                params={"condition": False, "message": "values mismatch"},
            )
        },
    )
    with pytest.raises(AssertionFailure, match="values mismatch"):
        _run(g, ctx, env)


def test_assert_message_supports_expression():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    ctx.bindings["actual"] = 7
    g = StepGraph(
        name="g",
        entry="a",
        steps={
            "a": Step(
                id="a",
                type="assert",
                params={
                    "condition": False,
                    "message": Expression("'got ' + str(bindings.actual)"),
                },
            )
        },
    )
    # 'str' isn't in our expression allowlist, so message resolution
    # itself fails — but assert wraps that, keeping the user-visible
    # error a clear AssertionFailure.
    with pytest.raises(AssertionFailure, match="assert message"):
        _run(g, ctx, env)


# ---- log -------------------------------------------------------------


def test_log_emits_at_configured_level(caplog):
    env = simpy.Environment()
    ctx = _make_ctx(env)
    g = StepGraph(
        name="g",
        entry="l",
        steps={
            "l": Step(
                id="l",
                type="log",
                params={"level": "warning", "message": "hello"},
            )
        },
    )
    with caplog.at_level(logging.WARNING, logger="altrios.lifts.workflow_engine.interpreter"):
        _run(g, ctx, env)
    assert any("hello" in rec.message for rec in caplog.records)


def test_log_rejects_unknown_level():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    g = StepGraph(
        name="g",
        entry="l",
        steps={
            "l": Step(
                id="l",
                type="log",
                params={"level": "scream", "message": "hi"},
            )
        },
    )
    with pytest.raises(InterpreterError, match="not a valid"):
        _run(g, ctx, env)


# ---- timeout ---------------------------------------------------------


def test_timeout_advances_simulated_time():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    g = StepGraph(
        name="g",
        entry="t",
        steps={"t": Step(id="t", type="timeout", params={"duration": 5.0})},
    )
    _run(g, ctx, env)
    assert env.now == 5.0


def test_timeout_supports_expression_param():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    ctx.bindings["dur"] = 3.5
    g = StepGraph(
        name="g",
        entry="t",
        steps={
            "t": Step(
                id="t", type="timeout", params={"duration": Expression("bindings.dur")}
            )
        },
    )
    _run(g, ctx, env)
    assert env.now == 3.5


def test_timeout_rejects_negative():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    g = StepGraph(
        name="g",
        entry="t",
        steps={"t": Step(id="t", type="timeout", params={"duration": -1.0})},
    )
    with pytest.raises(InterpreterError, match="non-negative"):
        _run(g, ctx, env)


def test_timeout_distribution_dict_constant(monkeypatch):
    """A ``{dist: constant, value: N}`` duration should sample to N and
    advance the clock just like a bare literal would."""
    import numpy as np
    env = simpy.Environment()
    ctx = _make_ctx(env)
    ctx.rng = np.random.default_rng(0)
    g = StepGraph(
        name="g",
        entry="t",
        steps={
            "t": Step(
                id="t",
                type="timeout",
                params={"duration": {"dist": "constant", "value": 7.0}},
            )
        },
    )
    _run(g, ctx, env)
    assert env.now == 7.0


def test_timeout_distribution_dict_uniform_seeded():
    """Uniform[10, 20] sampled with a fixed-seed RNG matches the
    underlying ``rng.uniform`` call. Validates that ``parse_distribution``
    plus ``ctx.rng`` wire correctly."""
    import numpy as np
    expected_rng = np.random.default_rng(42)
    expected = expected_rng.uniform(10, 20)

    env = simpy.Environment()
    ctx = _make_ctx(env)
    ctx.rng = np.random.default_rng(42)
    g = StepGraph(
        name="g",
        entry="t",
        steps={
            "t": Step(
                id="t",
                type="timeout",
                params={"duration": {"dist": "uniform", "low": 10, "high": 20}},
            )
        },
    )
    _run(g, ctx, env)
    assert env.now == pytest.approx(expected)


def test_timeout_distribution_dict_requires_rng():
    """If ``ctx.rng`` is None, a distribution-typed duration should
    fail loud rather than silently fall back to anything."""
    env = simpy.Environment()
    ctx = _make_ctx(env)  # rng defaults to None
    g = StepGraph(
        name="g",
        entry="t",
        steps={
            "t": Step(
                id="t",
                type="timeout",
                params={"duration": {"dist": "uniform", "low": 0, "high": 1}},
            )
        },
    )
    with pytest.raises(InterpreterError, match="ExecutionContext.rng is None"):
        _run(g, ctx, env)


def test_timeout_unknown_distribution_propagates_error():
    import numpy as np
    env = simpy.Environment()
    ctx = _make_ctx(env)
    ctx.rng = np.random.default_rng(0)
    g = StepGraph(
        name="g",
        entry="t",
        steps={
            "t": Step(
                id="t",
                type="timeout",
                params={"duration": {"dist": "weibull", "shape": 1}},
            )
        },
    )
    from altrios.lifts.workflow_engine.distributions import DistributionError
    with pytest.raises(DistributionError, match="Unknown distribution 'weibull'"):
        _run(g, ctx, env)


# ---- interpreter outer loop -----------------------------------------


def test_unknown_primitive_raises_with_step_id():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    g = StepGraph(
        name="g",
        entry="x",
        steps={"x": Step(id="x", type="not_a_real_primitive", params={})},
    )
    with pytest.raises(InterpreterError, match="No handler"):
        _run(g, ctx, env)


def test_branch_to_undefined_step_id_raises():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    # The branch destination 'ghost' is built dynamically by branch's
    # handler at runtime, so StepGraph can't statically detect it —
    # interpreter catches it on the jump.
    g = StepGraph(
        name="g",
        entry="br",
        steps={
            "br": Step(
                id="br",
                type="branch",
                params={"condition": True, "true": "ghost", "false": None},
            ),
        },
    )
    with pytest.raises(InterpreterError, match="undefined step 'ghost'"):
        _run(g, ctx, env)


def test_sequential_chain_with_timeouts_and_bindings():
    """End-to-end: a 4-step graph that times, binds, branches, and
    terminates. Verifies the interpreter actually walks ``step.next``
    pointers and threads bindings across steps."""
    env = simpy.Environment()
    ctx = _make_ctx(env)
    g = StepGraph(
        name="multi",
        entry="t1",
        steps={
            "t1": Step(id="t1", type="timeout", params={"duration": 2.0}, next="b1"),
            "b1": Step(
                id="b1",
                type="bind",
                params={"name": "n", "value": Expression("env.now")},
                next="br",
            ),
            "br": Step(
                id="br",
                type="branch",
                params={
                    "condition": Expression("bindings.n >= 2"),
                    "true": "t2",
                    "false": None,
                },
            ),
            "t2": Step(id="t2", type="timeout", params={"duration": 1.0}),
        },
    )
    _run(g, ctx, env)
    assert env.now == 3.0
    assert ctx.bindings["n"] == 2.0
