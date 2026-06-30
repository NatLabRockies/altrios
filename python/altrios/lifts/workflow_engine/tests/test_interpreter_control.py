"""Tests for control-flow primitives:
``parallel``, ``loop``, ``spawn``, ``make_event``, ``wait_event``,
``trigger_event``."""
from __future__ import annotations

import pytest
import simpy

from altrios.lifts.workflow_engine.expressions import Expression
from altrios.lifts.workflow_engine.interpreter import (
    ExecutionContext,
    InterpreterError,
    build_default_primitives,
    execute,
)
from altrios.lifts.workflow_engine.steps import Step, StepGraph


def _make_ctx(env, **kwargs) -> ExecutionContext:
    return ExecutionContext(env=env, primitives=build_default_primitives(), **kwargs)


def _run(graph, ctx, env):
    proc = env.process(execute(graph, ctx))
    env.run(until=proc)


# ---- parallel --------------------------------------------------------


def test_parallel_join_all_takes_longest_branch():
    """Three parallel timeouts of 2s, 5s, and 3s with join=all should
    finish at t=5."""
    env = simpy.Environment()
    ctx = _make_ctx(env)
    g = StepGraph(
        name="g",
        entry="par",
        steps={
            "par": Step(
                id="par",
                type="parallel",
                params={"branches": ["a", "b", "c"], "join": "all"},
                next="done",
            ),
            "a": Step(id="a", type="timeout", params={"duration": 2.0}),
            "b": Step(id="b", type="timeout", params={"duration": 5.0}),
            "c": Step(id="c", type="timeout", params={"duration": 3.0}),
            "done": Step(
                id="done",
                type="bind",
                params={"name": "finished_at", "value": Expression("env.now")},
            ),
        },
    )
    _run(g, ctx, env)
    assert ctx.bindings["finished_at"] == 5.0
    assert env.now == 5.0


def test_parallel_join_any_takes_shortest_branch():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    g = StepGraph(
        name="g",
        entry="par",
        steps={
            "par": Step(
                id="par",
                type="parallel",
                params={"branches": ["a", "b"], "join": "any"},
                next="done",
            ),
            "a": Step(id="a", type="timeout", params={"duration": 1.0}),
            "b": Step(id="b", type="timeout", params={"duration": 10.0}),
            "done": Step(
                id="done",
                type="bind",
                params={"name": "finished_at", "value": Expression("env.now")},
            ),
        },
    )
    _run(g, ctx, env)
    assert ctx.bindings["finished_at"] == 1.0


def test_parallel_branches_have_isolated_bindings():
    """Two branches write to a same-named binding; the parent's
    binding is unaffected (forked dict)."""
    env = simpy.Environment()
    ctx = _make_ctx(env)
    ctx.bindings["x"] = "parent"
    g = StepGraph(
        name="g",
        entry="par",
        steps={
            "par": Step(
                id="par",
                type="parallel",
                params={"branches": ["a", "b"]},
            ),
            "a": Step(id="a", type="bind", params={"name": "x", "value": "A"}),
            "b": Step(id="b", type="bind", params={"name": "x", "value": "B"}),
        },
    )
    _run(g, ctx, env)
    assert ctx.bindings["x"] == "parent"


def test_parallel_rejects_empty_branches():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    g = StepGraph(
        name="g",
        entry="par",
        steps={
            "par": Step(
                id="par", type="parallel", params={"branches": []}
            )
        },
    )
    with pytest.raises(InterpreterError, match="non-empty list"):
        _run(g, ctx, env)


# ---- loop ------------------------------------------------------------


def test_loop_sequential_executes_each_iteration():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    ctx.bindings["items"] = [1.0, 2.0, 3.0]
    # Use the shared output collector to count iterations rather than
    # the bindings dict (which is forked per iteration).
    g = StepGraph(
        name="g",
        entry="lp",
        steps={
            "lp": Step(
                id="lp",
                type="loop",
                params={
                    "over": Expression("bindings.items"),
                    "as": "x",
                    "do": "rec",
                },
            ),
            "rec": Step(
                id="rec",
                type="record_event",
                params={
                    "event_type": "tick",
                    "columns": {"value": Expression("bindings.x")},
                },
            ),
        },
    )
    _run(g, ctx, env)
    vals = [r["value"] for r in ctx.output.event_log]
    assert vals == [1.0, 2.0, 3.0]


def test_loop_sequential_accumulates_time():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    ctx.bindings["durations"] = [1.5, 2.5]
    g = StepGraph(
        name="g",
        entry="lp",
        steps={
            "lp": Step(
                id="lp",
                type="loop",
                params={
                    "over": Expression("bindings.durations"),
                    "as": "d",
                    "do": "to",
                },
            ),
            "to": Step(
                id="to",
                type="timeout",
                params={"duration": Expression("bindings.d")},
            ),
        },
    )
    _run(g, ctx, env)
    assert env.now == 4.0


def test_loop_parallel_overlaps_iterations():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    ctx.bindings["durations"] = [1.5, 2.5, 0.5]
    g = StepGraph(
        name="g",
        entry="lp",
        steps={
            "lp": Step(
                id="lp",
                type="loop",
                params={
                    "over": Expression("bindings.durations"),
                    "as": "d",
                    "do": "to",
                    "parallel": True,
                },
            ),
            "to": Step(
                id="to",
                type="timeout",
                params={"duration": Expression("bindings.d")},
            ),
        },
    )
    _run(g, ctx, env)
    assert env.now == 2.5  # max, not sum


def test_loop_over_empty_iterable_completes_immediately():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    g = StepGraph(
        name="g",
        entry="lp",
        steps={
            "lp": Step(
                id="lp",
                type="loop",
                params={"over": [], "as": "x", "do": "noop"},
            ),
            "noop": Step(id="noop", type="bind", params={"name": "_", "value": 1}),
        },
    )
    _run(g, ctx, env)
    assert env.now == 0.0


# ---- spawn -----------------------------------------------------------


def test_spawn_waits_for_subgraph_by_default():
    env = simpy.Environment()
    sub = StepGraph(
        name="sub",
        entry="t",
        steps={"t": Step(id="t", type="timeout", params={"duration": 4.0})},
    )
    parent = StepGraph(
        name="parent",
        entry="sp",
        steps={
            "sp": Step(
                id="sp",
                type="spawn",
                params={"graph": "sub"},
                next="done",
            ),
            "done": Step(
                id="done",
                type="bind",
                params={"name": "t", "value": Expression("env.now")},
            ),
        },
    )
    ctx = _make_ctx(env, graphs={"sub": sub})
    _run(parent, ctx, env)
    assert ctx.bindings["t"] == 4.0


def test_spawn_wait_false_does_not_block_parent():
    env = simpy.Environment()
    sub = StepGraph(
        name="sub",
        entry="t",
        steps={"t": Step(id="t", type="timeout", params={"duration": 10.0})},
    )
    parent = StepGraph(
        name="parent",
        entry="sp",
        steps={
            "sp": Step(
                id="sp",
                type="spawn",
                params={"graph": "sub", "wait": False},
                next="done",
            ),
            "done": Step(
                id="done",
                type="bind",
                params={"name": "t", "value": Expression("env.now")},
            ),
        },
    )
    ctx = _make_ctx(env, graphs={"sub": sub})
    proc = env.process(execute(parent, ctx))
    env.run(until=proc)
    # Parent finished at t=0; the spawned sub-graph is still in flight.
    assert ctx.bindings["t"] == 0.0
    # Continue running to drain the simulation; the sub-graph completes
    # at t=10. (Use a fresh until=10 to advance time.)
    env.run(until=10.0)
    assert env.now == 10.0


def test_spawn_unknown_graph_raises():
    env = simpy.Environment()
    parent = StepGraph(
        name="parent",
        entry="sp",
        steps={
            "sp": Step(id="sp", type="spawn", params={"graph": "ghost"})
        },
    )
    ctx = _make_ctx(env, graphs={})
    with pytest.raises(InterpreterError, match="not in ctx.graphs"):
        _run(parent, ctx, env)


# ---- make_event / wait_event / trigger_event ------------------------


def test_make_and_trigger_single_event():
    env = simpy.Environment()
    ctx = _make_ctx(env)

    # Producer: timeout 2s, then trigger the event.
    producer = StepGraph(
        name="prod",
        entry="t",
        steps={
            "t": Step(
                id="t", type="timeout", params={"duration": 2.0}, next="tr"
            ),
            "tr": Step(
                id="tr", type="trigger_event", params={"event_var": "ready"}
            ),
        },
    )

    # Consumer: make event, wait, record finish time.
    consumer = StepGraph(
        name="cons",
        entry="mk",
        steps={
            "mk": Step(
                id="mk",
                type="make_event",
                params={"bind": "ready"},
                next="wt",
            ),
            "wt": Step(
                id="wt",
                type="wait_event",
                params={"event_var": "ready"},
                next="rec",
            ),
            "rec": Step(
                id="rec",
                type="bind",
                params={"name": "done_at", "value": Expression("env.now")},
            ),
        },
    )

    # Step the consumer until it creates the event and blocks; then run
    # the producer using the same bindings dict (so it sees ``ready``).
    cons_proc = env.process(execute(consumer, ctx))
    env.step()  # mk + wt; now blocked on ``ready``
    env.process(execute(producer, ctx))
    env.run(until=cons_proc)
    assert ctx.bindings["done_at"] == 2.0


def test_make_event_count_and_wait_all():
    """Three independent events; wait_event mode='all' should join
    when all three have triggered."""
    env = simpy.Environment()
    ctx = _make_ctx(env)

    g = StepGraph(
        name="g",
        entry="mk",
        steps={
            "mk": Step(
                id="mk",
                type="make_event",
                params={"bind": "locks", "count": 3},
                next="par",
            ),
            "par": Step(
                id="par",
                type="parallel",
                params={"branches": ["t1", "t2", "t3"]},
                next="join",
            ),
            "t1": Step(
                id="t1",
                type="timeout",
                params={"duration": 1.0},
                next="tr1",
            ),
            "tr1": Step(
                id="tr1",
                type="trigger_event",
                params={"event_var": "locks", "index": 0},
            ),
            "t2": Step(
                id="t2",
                type="timeout",
                params={"duration": 2.0},
                next="tr2",
            ),
            "tr2": Step(
                id="tr2",
                type="trigger_event",
                params={"event_var": "locks", "index": 1},
            ),
            "t3": Step(
                id="t3",
                type="timeout",
                params={"duration": 3.0},
                next="tr3",
            ),
            "tr3": Step(
                id="tr3",
                type="trigger_event",
                params={"event_var": "locks", "index": 2},
            ),
            "join": Step(
                id="join",
                type="wait_event",
                params={"event_var": "locks", "mode": "all"},
                next="done",
            ),
            "done": Step(
                id="done",
                type="bind",
                params={"name": "t", "value": Expression("env.now")},
            ),
        },
    )
    _run(g, ctx, env)
    # parallel branches use forked bindings (siblings can't see
    # ``locks``); rewrite — trigger_event needs the events to be in
    # the executing branch's bindings. Since each forked branch copies
    # bindings at fork time, ``locks`` is present in all of them.
    # The list of events itself is shared (list is by reference, fork
    # only deep-copies the dict shell). So triggers in branches do
    # mutate the shared events.
    assert ctx.bindings["t"] == 3.0


def test_trigger_event_already_triggered_is_noop():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    g = StepGraph(
        name="g",
        entry="mk",
        steps={
            "mk": Step(
                id="mk", type="make_event", params={"bind": "e"}, next="t1"
            ),
            "t1": Step(
                id="t1", type="trigger_event", params={"event_var": "e"}, next="t2"
            ),
            "t2": Step(
                id="t2", type="trigger_event", params={"event_var": "e"}
            ),
        },
    )
    _run(g, ctx, env)  # second trigger does not raise


def test_wait_event_mode_one_blocks_on_single_event():
    env = simpy.Environment()
    ctx = _make_ctx(env)
    g = StepGraph(
        name="g",
        entry="mk",
        steps={
            "mk": Step(
                id="mk",
                type="make_event",
                params={"bind": "e"},
                next="par",
            ),
            "par": Step(
                id="par",
                type="parallel",
                params={"branches": ["wait", "fire"], "join": "all"},
            ),
            "wait": Step(
                id="wait",
                type="wait_event",
                params={"event_var": "e", "mode": "one"},
            ),
            "fire": Step(
                id="fire",
                type="timeout",
                params={"duration": 4.0},
                next="tr",
            ),
            "tr": Step(
                id="tr",
                type="trigger_event",
                params={"event_var": "e"},
            ),
        },
    )
    _run(g, ctx, env)
    assert env.now == 4.0
