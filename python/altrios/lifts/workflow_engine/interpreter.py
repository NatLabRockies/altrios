"""SimPy step-graph interpreter.

The interpreter walks a :class:`StepGraph` one step at a time, looking
each step's ``type`` up in a primitive table and delegating execution
to the handler.

Primitive handlers are stateless callables of signature
``(step: Step, ctx: ExecutionContext) -> StepResult``.

A :class:`StepResult` is one of:

- ``None`` — fall through to ``step.next``.
- ``str`` — jump to the named step id.
- ``Generator`` — a SimPy generator to yield from; its return value
  (delivered via ``StopIteration``) is itself a :class:`StepResult` and
  is recursed into.

This split lets the simple primitives (``bind``, ``branch``, ``log``)
stay as plain functions while letting time-consuming primitives
(``timeout``, ``request``) be generator functions that yield SimPy
events. The interpreter's outer loop ``yield from``s any generator
handler results so the whole graph is itself a SimPy process.
"""
from __future__ import annotations

import logging
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Generator, Mapping, Union

import simpy

from .expressions import Expression, ExpressionContext, ExpressionError
from .distributions import parse_distribution
from .output import OutputCollector
from .steps import Step, StepGraph


_log = logging.getLogger("altrios.lifts.workflow_engine.interpreter")


# A handler may return one of these. ``Generator`` is for SimPy yields;
# its return value (delivered via StopIteration) must itself be a
# StepResult.
StepResult = Union[None, str, Generator]
StepHandler = Callable[["Step", "ExecutionContext"], StepResult]


class InterpreterError(Exception):
    """Raised when the interpreter cannot execute a step (unknown
    primitive, missing required param, type error, failed assertion)."""


class AssertionFailure(InterpreterError):
    """Raised when an ``assert`` step's condition evaluates falsy."""


@dataclass
class ExecutionContext:
    """Per-process mutable state threaded through every handler.

    A fresh ``ExecutionContext`` is constructed for each workflow
    invocation (one per entity going through one graph). The ``env``,
    ``state``, ``config``, ``layout``, ``registry``, and ``primitives``
    fields are typically the same shared instances across all
    contexts within a single simulation run; ``entity`` and
    ``bindings`` are per-invocation.

    Attributes:
        env: SimPy environment (or any object with a ``.now`` attribute
            for the ``env`` expression namespace).
        primitives: Map from primitive type name to handler. Built by
            :func:`build_default_primitives` plus catalog-supplied
            additions.
        bindings: Mutable local-variable scope. Updated by ``bind``,
            ``request``, ``python`` (when ``bind`` is set), etc. Read
            by expressions via the ``bindings`` namespace.
        entity: The :class:`Entity` flowing through this workflow.
            ``None`` for graphs that operate purely on resources/state.
        state: Live engine state (resource pools, container stacks, ...).
            Exposed read-only in expressions as the ``state`` namespace.
        config: Site-level scalar config dict. Read-only in expressions.
        layout: Layout helper exposing ``.distance(a, b)`` and
            ``.node(name)``. Read-only in expressions.
        registry: Catalog-supplied Python callables for ``python:``
            steps. Optional (3B.3 has no ``python`` primitive yet).
    """

    env: Any
    primitives: Mapping[str, StepHandler]
    bindings: dict[str, Any] = field(default_factory=dict)
    entity: Any = None
    state: Any = None
    config: Mapping[str, Any] = field(default_factory=dict)
    layout: Any = None
    registry: Any = None
    output: OutputCollector = field(default_factory=OutputCollector)
    # Single run-wide RNG used by every distribution-typed param
    # (``timeout.duration``, ``record_consumption.quantity``, ...).
    # Sharing one stream means adding a step in the middle of a graph
    # will perturb downstream samples — acceptable for v1 since
    # reproducibility is anchored by passing a seeded
    # ``numpy.random.Generator`` here. Per-stream isolation can be
    # added later if a catalog needs it.
    rng: Any = None  # numpy.random.Generator
    # The currently-executing graph and the registry of available
    # graphs. Threaded onto the context (rather than passed positionally
    # through every handler) so the control-flow primitives can
    # introspect the surrounding graph (``parallel``, ``loop``) and
    # dispatch to other graphs (``spawn``).
    current_graph: Any = None  # StepGraph
    graphs: Mapping[str, Any] = field(default_factory=dict)

    def to_expression_context(self) -> ExpressionContext:
        """Snapshot the current state as an :class:`ExpressionContext`
        for evaluating a single expression. Cheap; called for every
        expression-typed param resolution.

        If ``self.entity`` is a dataclass-style object with an
        ``.attrs`` dict (e.g. :class:`Entity`), the attrs are
        **flattened** into a read-only SimpleNamespace view so catalog
        authors can write ``entity.weight_t`` rather than
        ``entity.attrs['weight_t']``. The original object is left
        untouched; mutations from ``set_attr`` go through
        :attr:`entity` directly.
        """
        entity_view = self.entity
        if entity_view is not None:
            attrs = getattr(entity_view, "attrs", None)
            if isinstance(attrs, Mapping):
                flat_kwargs: dict[str, Any] = {}
                for top_attr in ("id", "kind", "parent_id"):
                    if hasattr(entity_view, top_attr):
                        flat_kwargs[top_attr] = getattr(entity_view, top_attr)
                # attrs win over base names only when there's no
                # collision; ``id`` etc. are part of Entity's API and
                # shouldn't be overridable from attrs in expressions.
                for k, v in attrs.items():
                    if k not in flat_kwargs:
                        flat_kwargs[k] = v
                entity_view = SimpleNamespace(**flat_kwargs)
        return ExpressionContext(
            entity=entity_view,
            bindings=self.bindings,
            state=self.state,
            config=self.config,
            layout=self.layout,
            env=self.env,
        )


def _resolve(value: Any, ctx: ExecutionContext) -> Any:
    """Resolve a param value: evaluate an :class:`Expression`, leave
    everything else (literals, dicts, callables) untouched."""
    if isinstance(value, Expression):
        return value.evaluate(ctx.to_expression_context())
    return value


def _require_param(step: Step, name: str) -> Any:
    """Pull a required param off a Step, raising
    :class:`InterpreterError` with the step id in the message if it's
    missing. Cheap; called from the hot path of every primitive."""
    try:
        return step.params[name]
    except KeyError:
        raise InterpreterError(
            f"Step {step.id!r} ({step.type!r}) is missing required "
            f"param {name!r}. Got params: {sorted(step.params)}."
        ) from None


# ---- Primitive handlers ----------------------------------------------


def _h_bind(step: Step, ctx: ExecutionContext) -> StepResult:
    """``bind`` — evaluate ``value`` and store under ``name``.

    YAML::

        - {type: bind, name: dist_m, value: "{layout.distance('a', 'b')}"}
    """
    name = _require_param(step, "name")
    if not isinstance(name, str) or not name:
        raise InterpreterError(
            f"Step {step.id!r}: bind.name must be a non-empty str, got {name!r}."
        )
    value = _resolve(_require_param(step, "value"), ctx)
    ctx.bindings[name] = value
    return None


def _h_set_attr(step: Step, ctx: ExecutionContext) -> StepResult:
    """``set_attr`` — mutate an attribute on an entity (or any object).

    YAML::

        - {type: set_attr, entity: entity, attr: status, value: "loaded"}

    The ``entity`` param defaults to ``ctx.entity`` when omitted; pass
    an Expression to target some other object (e.g. a bound resource).
    """
    if "entity" in step.params:
        target = _resolve(step.params["entity"], ctx)
    else:
        target = ctx.entity
    if target is None:
        raise InterpreterError(
            f"Step {step.id!r}: set_attr has no target — neither "
            f"ctx.entity nor step.params['entity'] is set."
        )
    attr = _require_param(step, "attr")
    if not isinstance(attr, str) or not attr:
        raise InterpreterError(
            f"Step {step.id!r}: set_attr.attr must be a non-empty str, got {attr!r}."
        )
    value = _resolve(_require_param(step, "value"), ctx)
    # Entity has an ``attrs`` dict; ordinary objects get setattr.
    attrs = getattr(target, "attrs", None)
    if isinstance(attrs, dict):
        attrs[attr] = value
    else:
        setattr(target, attr, value)
    return None


def _h_branch(step: Step, ctx: ExecutionContext) -> StepResult:
    """``branch`` — evaluate ``condition``, jump to ``true`` or ``false``.

    Either branch may be ``None`` to mean "end the workflow on this
    side"; the param itself must still be present in the YAML so the
    intent is explicit.
    """
    cond = _resolve(_require_param(step, "condition"), ctx)
    true_id = _require_param(step, "true")
    false_id = _require_param(step, "false")
    target = true_id if cond else false_id
    if target is not None and not isinstance(target, str):
        raise InterpreterError(
            f"Step {step.id!r}: branch.{'true' if cond else 'false'} "
            f"must be a step id (str) or None, got {target!r}."
        )
    if target is None:
        return None  # explicit end of this branch
    return target


def _h_assert(step: Step, ctx: ExecutionContext) -> StepResult:
    """``assert`` — raise :class:`AssertionFailure` if condition is
    falsy. ``message`` is optional and is evaluated only on failure.
    """
    cond = _resolve(_require_param(step, "condition"), ctx)
    if cond:
        return None
    if "message" in step.params:
        try:
            message = _resolve(step.params["message"], ctx)
        except ExpressionError as exc:
            message = f"<assert message itself failed to evaluate: {exc}>"
    else:
        message = "(no message)"
    raise AssertionFailure(
        f"Assertion failed at step {step.id!r}: {message}"
    )


def _h_log(step: Step, ctx: ExecutionContext) -> StepResult:
    """``log`` — emit a diagnostic message via Python ``logging``.

    ``level`` defaults to ``"info"``. ``message`` is evaluated each
    call (no caching) so it can include current binding values.
    """
    level_name = step.params.get("level", "info")
    if not isinstance(level_name, str):
        raise InterpreterError(
            f"Step {step.id!r}: log.level must be a str, got {level_name!r}."
        )
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise InterpreterError(
            f"Step {step.id!r}: log.level={level_name!r} is not a valid "
            f"logging level (try 'debug', 'info', 'warning', 'error')."
        )
    message = _resolve(_require_param(step, "message"), ctx)
    _log.log(level, "[step %s] %s", step.id, message)
    return None


def _h_timeout(step: Step, ctx: ExecutionContext) -> StepResult:
    """``timeout`` — advance simulated time by ``duration``.

    ``duration`` is either a scalar (literal or Expression) or a
    distribution dict (``{dist: uniform, low: 0, high: 1}``). Dicts
    are parsed each call via :func:`parse_distribution` and sampled
    from ``ctx.rng``. The handler is a generator so the interpreter's
    ``yield from`` chains it into the SimPy process.
    """
    duration = _resolve(_require_param(step, "duration"), ctx)
    if isinstance(duration, Mapping):
        if ctx.rng is None:
            raise InterpreterError(
                f"Step {step.id!r}: duration is a distribution dict but "
                "ExecutionContext.rng is None; pass a numpy.random.Generator "
                "when building the context to use distribution-typed params."
            )
        duration = parse_distribution(duration).sample(ctx.rng)
    if not isinstance(duration, (int, float)) or duration < 0:
        raise InterpreterError(
            f"Step {step.id!r}: timeout.duration must resolve to a "
            f"non-negative number, got {duration!r}."
        )
    return _yield_timeout(ctx.env, duration)


def _yield_timeout(env: Any, duration: float) -> Generator:
    """Generator wrapper for ``env.timeout(duration)``. Kept separate
    so :func:`_h_timeout` can return it without being a generator
    function itself (which would force callers into the yield-from
    path even for the validation-error case)."""
    yield env.timeout(duration)
    return None


# ---- Resource-pool helpers (3B.4) -----------------------------------


def _lookup_pool(step: Step, ctx: ExecutionContext) -> Any:
    """Resolve ``step.params['pool']`` to a SimPy primitive.

    The ``pool`` param can be:

    - A string — looked up by name via ``getattr`` then dict-style on
      ``ctx.state``.
    - An :class:`Expression` — evaluated; the result is used directly.

    If ``partition_key`` is also present, the resolved primitive is
    expected to be a dict and the key is used to index it.
    """
    if ctx.state is None:
        raise InterpreterError(
            f"Step {step.id!r}: ExecutionContext.state is None; "
            "resource primitives need a state container."
        )
    raw = _require_param(step, "pool")
    if isinstance(raw, Expression):
        pool = raw.evaluate(ctx.to_expression_context())
    elif isinstance(raw, str):
        pool = getattr(ctx.state, raw, None)
        if pool is None and hasattr(ctx.state, "__getitem__"):
            try:
                pool = ctx.state[raw]
            except (KeyError, TypeError):
                pool = None
        if pool is None:
            raise InterpreterError(
                f"Step {step.id!r}: no pool named {raw!r} on state."
            )
    else:
        raise InterpreterError(
            f"Step {step.id!r}: pool must be a str or Expression, "
            f"got {type(raw).__name__}: {raw!r}."
        )
    if "partition_key" in step.params:
        key = _resolve(step.params["partition_key"], ctx)
        if not isinstance(pool, Mapping):
            raise InterpreterError(
                f"Step {step.id!r}: partition_key={key!r} given but pool "
                f"{type(pool).__name__} is not a Mapping."
            )
        try:
            pool = pool[key]
        except KeyError:
            raise InterpreterError(
                f"Step {step.id!r}: partition_key={key!r} not present in "
                f"pool (have: {sorted(pool)[:10]}...)."
            ) from None
    return pool


def _h_request(step: Step, ctx: ExecutionContext) -> StepResult:
    """``request`` — acquire from a SimPy Store / Resource / Container.

    YAML::

        - {type: request, pool: cranes, bind: crane}
        - {type: request, pool: tracks, partition_key: "{entity.track_id}", bind: track}
        - {type: request, pool: fuel_tank, qty: 50, bind: amount}

    Binds:
        - For ``Resource``: the ``Request`` object (release-able).
        - For ``Store``: the item obtained.
        - For ``Container``: the requested quantity.
    """
    pool = _lookup_pool(step, ctx)
    bind_name = step.params.get("bind")
    if bind_name is not None and (
        not isinstance(bind_name, str) or not bind_name
    ):
        raise InterpreterError(
            f"Step {step.id!r}: request.bind must be a non-empty str, "
            f"got {bind_name!r}."
        )
    return _yield_request(step, ctx, pool, bind_name)


def _yield_request(
    step: Step,
    ctx: ExecutionContext,
    pool: Any,
    bind_name: str | None,
) -> Generator:
    if isinstance(pool, simpy.Resource):
        req = pool.request()
        yield req
        if bind_name:
            ctx.bindings[bind_name] = req
    elif isinstance(pool, simpy.Store):
        # FilterStore exposes the same .get() with a filter kwarg.
        filt = step.params.get("filter")
        if filt is None:
            item = yield pool.get()
        elif callable(filt):
            item = yield pool.get(filt)
        else:
            raise InterpreterError(
                f"Step {step.id!r}: request.filter must be callable, "
                f"got {type(filt).__name__}."
            )
        if bind_name:
            ctx.bindings[bind_name] = item
    elif isinstance(pool, simpy.Container):
        qty = _resolve(step.params.get("qty", 1), ctx)
        if not isinstance(qty, (int, float)) or qty <= 0:
            raise InterpreterError(
                f"Step {step.id!r}: request.qty for a Container must be "
                f"positive, got {qty!r}."
            )
        yield pool.get(qty)
        if bind_name:
            ctx.bindings[bind_name] = qty
    else:
        raise InterpreterError(
            f"Step {step.id!r}: request pool has unsupported type "
            f"{type(pool).__name__}; expected simpy.Resource, Store, "
            "or Container."
        )
    return None


def _h_release(step: Step, ctx: ExecutionContext) -> StepResult:
    """``release`` — return capacity to the pool.

    YAML::

        - {type: release, pool: cranes, bind: crane}        # Resource
        - {type: release, pool: stack_A, value: entity}     # Store
        - {type: release, pool: fuel_tank, qty: 50}         # Container
    """
    pool = _lookup_pool(step, ctx)
    if isinstance(pool, simpy.Resource):
        bind_name = _require_param(step, "bind")
        if not isinstance(bind_name, str):
            raise InterpreterError(
                f"Step {step.id!r}: release.bind must be a str, got {bind_name!r}."
            )
        try:
            req = ctx.bindings[bind_name]
        except KeyError:
            raise InterpreterError(
                f"Step {step.id!r}: release.bind={bind_name!r} is not in "
                f"bindings; available: {sorted(ctx.bindings)}."
            ) from None
        pool.release(req)
        # Don't auto-delete the binding — a later step may inspect it
        # (e.g. to log resource id). Catalog can ``bind`` over it if
        # needed.
        return None
    if isinstance(pool, simpy.Store):
        if "value" in step.params:
            item = _resolve(step.params["value"], ctx)
        elif "bind" in step.params:
            try:
                item = ctx.bindings[step.params["bind"]]
            except KeyError:
                raise InterpreterError(
                    f"Step {step.id!r}: release.bind={step.params['bind']!r} "
                    f"is not in bindings; available: {sorted(ctx.bindings)}."
                ) from None
        else:
            raise InterpreterError(
                f"Step {step.id!r}: release for a Store needs either "
                "'value' or 'bind' to know what to put back."
            )
        yield_token = pool.put(item)
        # pool.put returns a PutEvent on Stores with capacity limits;
        # yield it so we block if the store is full.
        return _yield_event(yield_token)
    if isinstance(pool, simpy.Container):
        qty = _resolve(_require_param(step, "qty"), ctx)
        if not isinstance(qty, (int, float)) or qty <= 0:
            raise InterpreterError(
                f"Step {step.id!r}: release.qty for a Container must be "
                f"positive, got {qty!r}."
            )
        yield_token = pool.put(qty)
        return _yield_event(yield_token)
    raise InterpreterError(
        f"Step {step.id!r}: release pool has unsupported type "
        f"{type(pool).__name__}."
    )


def _yield_event(event) -> Generator:
    """Wrap a single SimPy event in a generator so the interpreter
    can ``yield from`` it uniformly."""
    yield event
    return None


def _h_transfer(step: Step, ctx: ExecutionContext) -> StepResult:
    """``transfer`` — atomically move an entity between two Stores.

    YAML::

        - {type: transfer, entity: entity, from: gate_queue, to: stack_A}

    The transfer is "atomic" only in the cooperative SimPy sense — no
    other coroutine runs between the get and the put — because both
    happen inside a single generator step without yielding to ``env``
    between them when the destination has room. If the destination is
    full, this generator blocks (yielding the put event) until space
    opens up; the source has already been emptied at that point.
    """
    src = _lookup_pool_named(step, ctx, "from")
    dst = _lookup_pool_named(step, ctx, "to")
    if not isinstance(src, simpy.Store) or not isinstance(dst, simpy.Store):
        raise InterpreterError(
            f"Step {step.id!r}: transfer requires Store-typed 'from' and "
            f"'to'; got {type(src).__name__} -> {type(dst).__name__}."
        )
    if "entity" in step.params:
        wanted = _resolve(step.params["entity"], ctx)

        def _match(item, _wanted=wanted):
            return item is _wanted or item == _wanted
        # Best-effort filter: simpy.Store doesn't take a filter, so fall
        # back to a get-then-validate pattern. If the head item isn't
        # what we wanted, that's an error in the workflow design — the
        # engine doesn't reorder Stores.
        return _yield_transfer_match(src, dst, wanted)
    return _yield_transfer_any(src, dst)


def _yield_transfer_match(src, dst, wanted) -> Generator:
    item = yield src.get()
    if not (item is wanted or item == wanted):
        # Put it back and raise. This is an error in workflow design.
        yield src.put(item)
        raise InterpreterError(
            f"transfer expected entity {wanted!r} at head of source, "
            f"got {item!r}. Use a filter_store if you need selection."
        )
    yield dst.put(item)
    return None


def _yield_transfer_any(src, dst) -> Generator:
    item = yield src.get()
    yield dst.put(item)
    return None


def _lookup_pool_named(step: Step, ctx: ExecutionContext, key: str) -> Any:
    """``_lookup_pool`` but with a configurable param name (used by
    ``transfer`` which has ``from``/``to`` instead of ``pool``)."""
    if key not in step.params:
        raise InterpreterError(
            f"Step {step.id!r} ({step.type!r}) is missing required "
            f"param {key!r}."
        )
    raw = step.params[key]
    if isinstance(raw, Expression):
        return raw.evaluate(ctx.to_expression_context())
    if isinstance(raw, str):
        pool = getattr(ctx.state, raw, None)
        if pool is None and hasattr(ctx.state, "__getitem__"):
            try:
                pool = ctx.state[raw]
            except (KeyError, TypeError):
                pool = None
        if pool is None:
            raise InterpreterError(
                f"Step {step.id!r}: no pool named {raw!r} on state."
            )
        return pool
    raise InterpreterError(
        f"Step {step.id!r}: {key!r} must be a str or Expression, "
        f"got {type(raw).__name__}: {raw!r}."
    )


# ---- Recording primitives (3B.4) -------------------------------------


def _h_record_event(step: Step, ctx: ExecutionContext) -> StepResult:
    """``record_event`` — append a row to the per-entity event log.

    YAML::

        - {type: record_event, entity: entity, event_type: "loaded"}

    The handler injects ``record_timestamp`` from ``env.now`` and (when
    available) ``entity_id``/``entity_kind`` from ``ctx.entity``. The
    catalog can pass extra columns via ``columns: {...}``.
    """
    entity = (
        _resolve(step.params["entity"], ctx)
        if "entity" in step.params
        else ctx.entity
    )
    event_type = _resolve(_require_param(step, "event_type"), ctx)
    row: dict[str, Any] = {
        "record_timestamp": ctx.env.now,
        "event_type": event_type,
    }
    if entity is not None:
        row["entity_id"] = getattr(entity, "id", None)
        row["entity_kind"] = getattr(entity, "kind", None)
    if "columns" in step.params:
        extras = _resolve(step.params["columns"], ctx)
        if not isinstance(extras, Mapping):
            raise InterpreterError(
                f"Step {step.id!r}: record_event.columns must be a "
                f"mapping, got {type(extras).__name__}."
            )
        for k, v in extras.items():
            row[str(k)] = _resolve(v, ctx)
    ctx.output.record_event(row)
    return None


def _h_record_resource_event(step: Step, ctx: ExecutionContext) -> StepResult:
    """``record_resource_event`` — append a resource-level status row.

    YAML::

        - {type: record_resource_event, resource: crane, event_type: "busy",
           duration: 4.2, status: "loading", entity: entity}
    """
    resource = _resolve(_require_param(step, "resource"), ctx)
    event_type = _resolve(_require_param(step, "event_type"), ctx)
    row: dict[str, Any] = {
        "record_timestamp": ctx.env.now,
        "event_type": event_type,
        "resource_id": getattr(resource, "id", id(resource)),
        "resource_type": type(resource).__name__,
    }
    for optional in ("duration", "status", "role", "fuel_type", "zone"):
        if optional in step.params:
            row[optional] = _resolve(step.params[optional], ctx)
    if "entity" in step.params:
        ent = _resolve(step.params["entity"], ctx)
    else:
        ent = ctx.entity
    if ent is not None:
        row["entity_id"] = getattr(ent, "id", None)
    if "columns" in step.params:
        extras = _resolve(step.params["columns"], ctx)
        if not isinstance(extras, Mapping):
            raise InterpreterError(
                f"Step {step.id!r}: record_resource_event.columns must be "
                f"a mapping, got {type(extras).__name__}."
            )
        for k, v in extras.items():
            row[str(k)] = _resolve(v, ctx)
    ctx.output.record_resource_event(row)
    return None


def _h_record_consumption(step: Step, ctx: ExecutionContext) -> StepResult:
    """``record_consumption`` — append an energy/fuel/emissions row.

    YAML::

        - {type: record_consumption, resource: crane, quantity: "energy",
           status: "loading", duration: 4.2, value: 0.85}

    ``value`` is the consumption amount itself (rate × duration is the
    catalog's responsibility — typically computed in a preceding
    ``python:`` step or via an Expression).
    """
    resource = _resolve(_require_param(step, "resource"), ctx)
    quantity = _resolve(_require_param(step, "quantity"), ctx)
    value = _resolve(_require_param(step, "value"), ctx)
    row: dict[str, Any] = {
        "record_timestamp": ctx.env.now,
        "resource_id": getattr(resource, "id", id(resource)),
        "resource_type": type(resource).__name__,
        "quantity": quantity,
        "consumption_value": value,
    }
    for optional in ("status", "duration", "role", "fuel_type", "zone"):
        if optional in step.params:
            row[optional] = _resolve(step.params[optional], ctx)
    if "entity" in step.params:
        ent = _resolve(step.params["entity"], ctx)
    else:
        ent = ctx.entity
    if ent is not None:
        row["entity_id"] = getattr(ent, "id", None)
    if "columns" in step.params:
        extras = _resolve(step.params["columns"], ctx)
        if not isinstance(extras, Mapping):
            raise InterpreterError(
                f"Step {step.id!r}: record_consumption.columns must be a "
                f"mapping, got {type(extras).__name__}."
            )
        for k, v in extras.items():
            row[str(k)] = _resolve(v, ctx)
    ctx.output.record_consumption(row)
    return None


# ---- Control-flow primitives (3B.5) ----------------------------------


def _fork_context(ctx: ExecutionContext, **overrides: Any) -> ExecutionContext:
    """Shallow-copy ``ctx`` with a private ``bindings`` dict.

    All other fields (env, state, output, registry, graphs, ...) are
    shared by reference — this is what we want for parallel branches
    and loop iterations: side-effects on shared SimPy primitives and
    the output log are visible across branches, but a binding written
    in one branch doesn't clobber a same-named binding in a sibling.

    ``overrides`` keyword args override specific fields after the
    copy, used to inject the loop variable into a fresh iteration.
    """
    new_bindings = dict(ctx.bindings)
    forked = ExecutionContext(
        env=ctx.env,
        primitives=ctx.primitives,
        bindings=new_bindings,
        entity=ctx.entity,
        state=ctx.state,
        config=ctx.config,
        layout=ctx.layout,
        registry=ctx.registry,
        output=ctx.output,
        current_graph=ctx.current_graph,
        graphs=ctx.graphs,
        rng=ctx.rng,
    )
    for k, v in overrides.items():
        setattr(forked, k, v)
    return forked


def _h_parallel(step: Step, ctx: ExecutionContext) -> StepResult:
    """``parallel`` — execute several branches concurrently.

    YAML::

        - type: parallel
          branches: [load_crane, prep_chassis, route_lookup]
          join: all          # or 'any'

    Each branch is the chain of steps reachable from its named step id
    via ``step.next``. Branches see a forked bindings dict (changes
    don't leak across siblings), but share the rest of the context
    (state, env, output collector). The parent's ``step.next`` is
    taken after the join completes.
    """
    branches = _require_param(step, "branches")
    if not isinstance(branches, (list, tuple)) or not branches:
        raise InterpreterError(
            f"Step {step.id!r}: parallel.branches must be a non-empty "
            f"list of step ids, got {branches!r}."
        )
    for br in branches:
        if not isinstance(br, str) or not br:
            raise InterpreterError(
                f"Step {step.id!r}: parallel.branches entries must be "
                f"non-empty step ids, got {br!r}."
            )
    join = step.params.get("join", "all")
    if join not in ("all", "any"):
        raise InterpreterError(
            f"Step {step.id!r}: parallel.join must be 'all' or 'any', "
            f"got {join!r}."
        )

    branch_ctxs = [_fork_context(ctx) for _ in branches]
    procs = [
        ctx.env.process(_execute_from(br, bctx))
        for br, bctx in zip(branches, branch_ctxs)
    ]
    if join == "all":
        combined = simpy.events.AllOf(ctx.env, procs)
    else:
        combined = simpy.events.AnyOf(ctx.env, procs)
    return _yield_event(combined)


def _h_loop(step: Step, ctx: ExecutionContext) -> StepResult:
    """``loop`` — iterate ``over`` an iterable, executing ``do`` each time.

    YAML::

        - type: loop
          over: "{bindings.container_list}"
          as: container
          do: handle_one
          parallel: false        # default false (sequential)

    The loop variable is exposed in the forked branch's bindings under
    ``as`` (so the sub-chain can reference ``bindings.container``).
    The original ``ctx.bindings`` is not modified.
    """
    iterable = _resolve(_require_param(step, "over"), ctx)
    if iterable is None:
        return None
    name = _require_param(step, "as")
    if not isinstance(name, str) or not name:
        raise InterpreterError(
            f"Step {step.id!r}: loop.as must be a non-empty str, got {name!r}."
        )
    do_step = _require_param(step, "do")
    if not isinstance(do_step, str) or not do_step:
        raise InterpreterError(
            f"Step {step.id!r}: loop.do must be a non-empty step id, got {do_step!r}."
        )
    is_parallel = bool(step.params.get("parallel", False))

    try:
        items = list(iterable)
    except TypeError as exc:
        raise InterpreterError(
            f"Step {step.id!r}: loop.over result is not iterable: {exc}."
        ) from exc

    if is_parallel:
        return _yield_loop_parallel(items, name, do_step, ctx)
    return _yield_loop_sequential(items, name, do_step, ctx)


def _yield_loop_sequential(items, name, do_step, ctx) -> Generator:
    for item in items:
        iter_ctx = _fork_context(ctx)
        iter_ctx.bindings[name] = item
        yield from _execute_from(do_step, iter_ctx)
    return None


def _yield_loop_parallel(items, name, do_step, ctx) -> Generator:
    procs = []
    for item in items:
        iter_ctx = _fork_context(ctx)
        iter_ctx.bindings[name] = item
        procs.append(ctx.env.process(_execute_from(do_step, iter_ctx)))
    if procs:
        yield simpy.events.AllOf(ctx.env, procs)
    return None


def _h_spawn(step: Step, ctx: ExecutionContext) -> StepResult:
    """``spawn`` — start another StepGraph as a sub-process.

    YAML::

        - type: spawn
          graph: post_arrival_report
          entity: entity                # optional; defaults to current
          wait: true                    # block on completion (default true)

    The spawned graph uses a forked bindings dict (so sub-graph
    bindings don't leak back) but shares state/env/output.
    """
    graph_name = _require_param(step, "graph")
    if not isinstance(graph_name, str):
        raise InterpreterError(
            f"Step {step.id!r}: spawn.graph must be a str, got {graph_name!r}."
        )
    try:
        sub_graph = ctx.graphs[graph_name]
    except KeyError:
        raise InterpreterError(
            f"Step {step.id!r}: spawn.graph={graph_name!r} not in "
            f"ctx.graphs. Available: {sorted(ctx.graphs)}."
        ) from None
    if "entity" in step.params:
        sub_entity = _resolve(step.params["entity"], ctx)
    else:
        sub_entity = ctx.entity
    wait = bool(step.params.get("wait", True))

    sub_ctx = _fork_context(ctx, entity=sub_entity, bindings={})
    proc = ctx.env.process(execute(sub_graph, sub_ctx))
    if wait:
        return _yield_event(proc)
    return None


def _h_make_event(step: Step, ctx: ExecutionContext) -> StepResult:
    """``make_event`` — mint one or more fresh SimPy events.

    YAML::

        - {type: make_event, bind: door_open}              # single event
        - {type: make_event, bind: locks, count: 3}        # list of 3 events
    """
    name = _require_param(step, "bind")
    if not isinstance(name, str) or not name:
        raise InterpreterError(
            f"Step {step.id!r}: make_event.bind must be a non-empty str, "
            f"got {name!r}."
        )
    count = _resolve(step.params.get("count", 1), ctx)
    if not isinstance(count, int) or count < 1:
        raise InterpreterError(
            f"Step {step.id!r}: make_event.count must be a positive int, "
            f"got {count!r}."
        )
    if count == 1:
        ctx.bindings[name] = simpy.Event(ctx.env)
    else:
        ctx.bindings[name] = [simpy.Event(ctx.env) for _ in range(count)]
    return None


def _h_wait_event(step: Step, ctx: ExecutionContext) -> StepResult:
    """``wait_event`` — block on a previously-made event (or list).

    YAML::

        - {type: wait_event, event_var: door_open}
        - {type: wait_event, event_var: locks, mode: all}    # AllOf
        - {type: wait_event, event_var: locks, mode: any}    # AnyOf
    """
    name = _require_param(step, "event_var")
    try:
        ev = ctx.bindings[name]
    except KeyError:
        raise InterpreterError(
            f"Step {step.id!r}: wait_event.event_var={name!r} not in "
            f"bindings; available: {sorted(ctx.bindings)}."
        ) from None
    mode = step.params.get("mode", "one")
    if mode == "one":
        if not isinstance(ev, simpy.events.Event):
            raise InterpreterError(
                f"Step {step.id!r}: wait_event mode='one' expected a single "
                f"Event for binding {name!r}, got {type(ev).__name__}."
            )
        return _yield_event(ev)
    if not isinstance(ev, (list, tuple)):
        raise InterpreterError(
            f"Step {step.id!r}: wait_event mode={mode!r} expects a list of "
            f"Events for binding {name!r}, got {type(ev).__name__}."
        )
    if mode == "all":
        return _yield_event(simpy.events.AllOf(ctx.env, list(ev)))
    if mode == "any":
        return _yield_event(simpy.events.AnyOf(ctx.env, list(ev)))
    raise InterpreterError(
        f"Step {step.id!r}: wait_event.mode must be 'one', 'all', or 'any', "
        f"got {mode!r}."
    )


def _h_trigger_event(step: Step, ctx: ExecutionContext) -> StepResult:
    """``trigger_event`` — call ``succeed()`` on a bound event.

    YAML::

        - {type: trigger_event, event_var: door_open}
        - {type: trigger_event, event_var: locks, index: 0}   # one of a list
        - {type: trigger_event, event_var: locks, all: true}  # every one

    Triggering an already-succeeded event is a no-op (silently
    ignored) rather than an error — common in workflows where the
    same event may be triggered defensively from multiple paths.
    """
    name = _require_param(step, "event_var")
    try:
        target = ctx.bindings[name]
    except KeyError:
        raise InterpreterError(
            f"Step {step.id!r}: trigger_event.event_var={name!r} not in "
            f"bindings; available: {sorted(ctx.bindings)}."
        ) from None
    if "index" in step.params:
        idx = _resolve(step.params["index"], ctx)
        if not isinstance(idx, int):
            raise InterpreterError(
                f"Step {step.id!r}: trigger_event.index must be int, "
                f"got {idx!r}."
            )
        try:
            ev = target[idx]
        except (TypeError, IndexError) as exc:
            raise InterpreterError(
                f"Step {step.id!r}: trigger_event.index={idx} invalid for "
                f"binding {name!r}: {exc}."
            ) from exc
        _safe_succeed(ev)
    elif step.params.get("all", False):
        if not isinstance(target, (list, tuple)):
            raise InterpreterError(
                f"Step {step.id!r}: trigger_event.all=true expects a list "
                f"of Events for binding {name!r}, got {type(target).__name__}."
            )
        for ev in target:
            _safe_succeed(ev)
    else:
        _safe_succeed(target)
    return None


def _safe_succeed(ev: Any) -> None:
    if not isinstance(ev, simpy.events.Event):
        raise InterpreterError(
            f"trigger_event target is not a simpy.Event: {type(ev).__name__}."
        )
    if not ev.triggered:
        ev.succeed()


# ---- Escape-hatch primitive (3B.6) -----------------------------------


def _h_python(step: Step, ctx: ExecutionContext) -> StepResult:
    """``python`` — call a registered Python helper.

    YAML::

        - type: python
          call: freight.choose_track
          args:
            env: env
            terminal: state.terminal
            train_id: entity.train_id
          bind: chosen_track

    ``call`` is looked up in :attr:`ExecutionContext.registry`. ``args``
    is a mapping of keyword args; each value is passed through
    :func:`_resolve` (so Expression values are evaluated; literals pass
    through unchanged). If the callable returns a generator (a SimPy
    sub-process), the engine ``yield from``s it so the workflow can
    block on simulated time inside Python helpers.

    The optional ``bind`` param stores the return value (or, for
    generators, the StopIteration value) under that name in
    :attr:`ExecutionContext.bindings`.
    """
    if ctx.registry is None:
        raise InterpreterError(
            f"Step {step.id!r}: python step requires "
            "ExecutionContext.registry to be set (use "
            "altrios.lifts.workflow_engine.registry.get_registry() or a "
            "catalog-scoped CallableRegistry)."
        )
    call_name = _require_param(step, "call")
    if not isinstance(call_name, str):
        raise InterpreterError(
            f"Step {step.id!r}: python.call must be a str, got {call_name!r}."
        )
    args_spec = step.params.get("args", {})
    if not isinstance(args_spec, Mapping):
        raise InterpreterError(
            f"Step {step.id!r}: python.args must be a mapping, got "
            f"{type(args_spec).__name__}."
        )
    bind_name = step.params.get("bind")
    if bind_name is not None and (
        not isinstance(bind_name, str) or not bind_name
    ):
        raise InterpreterError(
            f"Step {step.id!r}: python.bind must be a non-empty str, "
            f"got {bind_name!r}."
        )
    resolved_args = {k: _resolve(v, ctx) for k, v in args_spec.items()}
    # ``registry.call`` validates the signature before invoking; surface
    # its RegistryError as an InterpreterError so workflow errors all
    # share one exception type.
    try:
        result = ctx.registry.call(call_name, **resolved_args)
    except Exception as exc:
        raise InterpreterError(
            f"Step {step.id!r}: python.call={call_name!r} failed: {exc}"
        ) from exc
    if isinstance(result, types.GeneratorType):
        return _yield_python_call(result, ctx, bind_name)
    if bind_name:
        ctx.bindings[bind_name] = result
    return None


def _yield_python_call(
    gen: Generator, ctx: ExecutionContext, bind_name: str | None
) -> Generator:
    """Yield a Python helper's generator into the SimPy schedule.

    The generator's return value (via StopIteration.value) is bound
    under ``bind_name`` if provided. The helper can yield any SimPy
    event (e.g. ``env.timeout(2.0)``) to simulate work that takes
    simulated time.
    """
    try:
        ret = yield from gen
    except Exception as exc:
        raise InterpreterError(
            f"python helper generator raised: {exc}"
        ) from exc
    if bind_name:
        ctx.bindings[bind_name] = ret
    return None


# ---- Primitive table assembly ----------------------------------------


def build_default_primitives() -> dict[str, StepHandler]:
    """Return a fresh dict of the built-in primitive handlers.

    Caller may extend this dict before stashing it in
    :class:`ExecutionContext` to add catalog-specific primitives or
    test doubles.
    """
    return {
        "bind": _h_bind,
        "set_attr": _h_set_attr,
        "branch": _h_branch,
        "assert": _h_assert,
        "log": _h_log,
        "timeout": _h_timeout,
        "request": _h_request,
        "release": _h_release,
        "transfer": _h_transfer,
        "record_event": _h_record_event,
        "record_resource_event": _h_record_resource_event,
        "record_consumption": _h_record_consumption,
        "parallel": _h_parallel,
        "loop": _h_loop,
        "spawn": _h_spawn,
        "make_event": _h_make_event,
        "wait_event": _h_wait_event,
        "trigger_event": _h_trigger_event,
        "python": _h_python,
    }


# ---- Interpreter -----------------------------------------------------


def execute(graph: StepGraph, ctx: ExecutionContext) -> Generator:
    """Walk ``graph`` to completion as a SimPy generator process.

    Yields SimPy events from time-consuming primitives. Returns
    ``None`` when the workflow finishes naturally (a step with
    ``next=None`` runs and its handler returns ``None`` or a
    null-jump).
    """
    # Stash the graph on the context so control-flow primitives
    # (``parallel``, ``loop``) can look up step definitions by id.
    # Restore the prior value afterwards so nested ``spawn`` calls
    # don't clobber the outer caller's graph reference.
    prev_graph = ctx.current_graph
    ctx.current_graph = graph
    try:
        yield from _execute_from(graph.entry, ctx)
    finally:
        ctx.current_graph = prev_graph
    return None


def _execute_from(step_id: str | None, ctx: ExecutionContext) -> Generator:
    """Walk a sub-chain starting at ``step_id`` until the chain ends
    naturally. Used by both :func:`execute` and the control-flow
    primitives (``parallel`` / ``loop``) that need to run a sub-chain
    of the current graph.
    """
    graph = ctx.current_graph
    if graph is None:
        raise InterpreterError(
            "_execute_from called with ctx.current_graph=None."
        )
    visited_count = 0
    MAX_STEPS = 1_000_000

    while step_id is not None:
        visited_count += 1
        if visited_count > MAX_STEPS:
            raise InterpreterError(
                f"Step graph {graph.name!r} executed more than "
                f"{MAX_STEPS} steps; aborting (likely infinite loop)."
            )
        try:
            step = graph.steps[step_id]
        except KeyError:
            raise InterpreterError(
                f"Jump to undefined step {step_id!r} in graph "
                f"{graph.name!r}."
            ) from None
        try:
            handler = ctx.primitives[step.type]
        except KeyError:
            raise InterpreterError(
                f"No handler registered for primitive {step.type!r} "
                f"(step {step.id!r} in graph {graph.name!r}). "
                f"Known primitives: {sorted(ctx.primitives)}."
            ) from None
        result = handler(step, ctx)
        while isinstance(result, types.GeneratorType):
            result = yield from result
        if result is None:
            step_id = step.next
        elif isinstance(result, str):
            step_id = result
        else:
            raise InterpreterError(
                f"Handler for primitive {step.type!r} (step {step.id!r}) "
                f"returned {result!r}; expected None, a step id (str), "
                f"or a generator."
            )
    return None
