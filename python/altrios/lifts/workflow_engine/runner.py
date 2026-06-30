"""Canonical entry point for running a workflow-engine site.

:func:`run_site` is the user-facing one-stop call:

.. code-block:: python

    from altrios.lifts.workflow_engine import run_site

    result = run_site("sites/rotterdam.yaml", seed=42)
    print(result.event_log)
    print(result.resource_log)

Internally it composes:

1. :func:`load_site` parses YAML, validates against pydantic
   schemas, imports the catalog's ``python_module``, and returns a
   ``(SiteModel, Catalog)`` pair.
2. Resource specs from every site-activated mode are union-merged
   (catalog-level :func:`merge_specs`); site-supplied
   ``resource_overrides`` are applied on top.
3. :func:`build_state_from_specs` instantiates SimPy primitives for
   the merged spec set; the result is wrapped as a
   :class:`SimpleNamespace` so existing freight-style ``state.tracks``
   attribute access continues to work.
4. A SimPy environment plus a seeded ``numpy.random.Generator``
   (driven by ``site.seed`` or the ``seed=`` kwarg override) are
   built and threaded onto an :class:`ExecutionContext`.
5. Arrival entries come from one of two sources:
   - The site's ``schedules:`` block, processed by builder callables
     registered in the catalog's python_module.
   - Direct entries passed via ``arrival_entries=`` to
     :func:`run_site` (test/programmatic use).
   Each entry is wrapped in an :class:`Entity` and scheduled as a
   SimPy process running the graph named in
   ``mode.arrival_routing[entity.kind]``. When more than one active
   mode routes the same kind, the entry's optional ``mode`` key
   selects which one fires.
6. ``env.run()`` is called; on completion, a :class:`RunResult`
   bundles the populated :class:`OutputCollector` plus references
   to the catalog, site, env, and state.

The runner is intentionally domain-agnostic: it knows nothing about
freight, mining, or airports. All domain-specific behavior (state
augmentation, schedule synthesis, output post-processing) lives in
the catalog's python_module and the workflow YAML.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import simpy

from .catalog import Catalog, WorkflowMode
from .entities import Entity
from .interpreter import (
    ExecutionContext,
    build_default_primitives,
    execute,
)
from .layout import Layout
from .loader import LoaderError, load_site
from .output import OutputCollector
from .registry import get_registry
from .resources import ResourceSpec, build_state_from_specs, merge_specs
from .schemas import SiteModel


PathLike = Union[str, os.PathLike]


class RunError(Exception):
    """Raised when a site run fails for an engine-level reason
    (missing schedule builder, unroutable entity kind, ...).

    Workflow-step exceptions (``InterpreterError``,
    ``AssertionFailure``, ``DistributionError``) propagate untouched
    so test surfaces see the original cause.
    """


@dataclass
class RunResult:
    """Bundle returned by :func:`run_site`.

    Attributes
    ----------
    site
        The validated :class:`SiteModel`.
    catalog
        The built engine :class:`Catalog`.
    env
        The SimPy environment that just finished running.
    state
        The :class:`types.SimpleNamespace` with one attribute per
        resource pool, mirroring the freight ``terminal.state.tracks``
        access convention.
    output
        The shared :class:`OutputCollector` populated by every step
        that fired during the run. Convert to dataframes via
        :meth:`OutputCollector.to_event_df` / ``to_resource_df`` /
        ``to_consumption_df``.
    config
        The flat config dict merged from catalog + site (site wins).
    entities
        Snapshot of every :class:`Entity` that was scheduled. Useful
        for post-run inspection and for builder pipelines that map
        from entity ids to richer records.
    """

    site: SiteModel
    catalog: Catalog
    env: simpy.Environment
    state: SimpleNamespace
    output: OutputCollector
    config: dict[str, Any]
    entities: list[Entity] = field(default_factory=list)


# ---- Top-level entry -----------------------------------------------


def run_site(
    site_path: PathLike,
    *,
    seed: Optional[int] = None,
    arrival_entries: Optional[Sequence[Mapping[str, Any]]] = None,
    until: Optional[float] = None,
    state_init: Optional[str] = None,
    schedule_overrides: Optional[Mapping[str, Any]] = None,
) -> RunResult:
    """Load and run one site to completion.

    Parameters
    ----------
    site_path
        Filesystem path to a site YAML file (resolves ``extends:``,
        catalog references, and validates schema).
    seed
        Master RNG seed. Overrides any ``seed:`` in the site file when
        not None. The resulting ``numpy.random.Generator`` is the
        single shared RNG threaded into every workflow
        :class:`ExecutionContext` (distribution-typed params draw
        from it).
    arrival_entries
        Direct list of arrival entries (each a mapping with at
        minimum ``kind``, ``arrival_time``, and an entity ``id``).
        When supplied, the site's ``schedules:`` block is ignored.
        Used by tests and programmatic drivers that already have
        synthesized arrivals.
    until
        Optional simulation-time horizon. When ``None``, the run
        terminates naturally when every spawned process finishes
        (the common case for finite arrival schedules).
    state_init
        Dotted name of a callable registered in the catalog's
        python_module that will be invoked after
        :func:`build_state_from_specs` with kwargs
        ``(env, state, config, layout)``. Catalogs use this hook to
        attach domain-specific state extras (e.g. the freight
        catalog's ``container_events`` list). Resolved via the
        default :class:`CallableRegistry`. When omitted, the
        catalog's ``schedule_mappings`` is checked for a
        ``state_init`` key as a fallback.
    schedule_overrides
        Per-schedule-name dict that overrides the site's ``schedules:``
        block entries before they are dispatched to builder callables.
        Useful for scenario-style ``run_site(path, schedule_overrides={
        "truck_rail.train_arrivals": "data/busy_month.csv"})``.

    Returns
    -------
    :class:`RunResult`

    Raises
    ------
    RunError
        For engine-level failures (missing builder, unroutable kind).
    YamlLoaderError, pydantic.ValidationError, LoaderError
        Propagated from :func:`load_site`.
    InterpreterError, AssertionFailure, DistributionError
        Propagated from step execution.
    """
    site, catalog = load_site(site_path)

    # Resolve active modes — defaults to every mode in the catalog
    # when the site doesn't filter.
    active_mode_names = site.modes if site.modes else [m.name for m in catalog.modes]
    active_modes = [catalog.mode(n) for n in active_mode_names]
    if not active_modes:
        raise RunError(
            f"Site {site.name!r} activates no modes and the catalog "
            f"{catalog.name!r} has none. Nothing to run."
        )

    # Merge resource specs across active modes (single SimPy primitive
    # per shared name; disagreements raise inside merge_specs).
    merged_specs = list(merge_specs({
        m.name: m.resource_specs for m in active_modes if m.resource_specs
    }).values())
    merged_specs = _apply_resource_overrides(merged_specs, site.resource_overrides)

    # Merge config: catalog ships defaults; site keys win.
    config: dict[str, Any] = dict(catalog.config_defaults)
    config.update(site.config)

    # Seed resolution: explicit kwarg wins over site.seed.
    effective_seed = seed if seed is not None else site.seed
    rng = (
        np.random.default_rng(effective_seed)
        if effective_seed is not None
        else np.random.default_rng()
    )

    env = simpy.Environment()
    # Build state primitives from specs.
    state_map = build_state_from_specs(
        env=env,
        specs=merged_specs,
        config=config,
        schedules=dict(site.schedules),
    )
    state = SimpleNamespace(**state_map)

    layout = Layout.from_model(site.layout) if site.layout is not None else None

    output = OutputCollector()

    # Stash output on state so state_init hooks and python: step helpers
    # can write into it without needing to thread it through every
    # signature. The runner's per-graph ExecutionContext.output (built
    # below) is the authoritative reference, but ``state.output`` is the
    # ergonomic access path for Python escape-hatch helpers that don't
    # otherwise see the ExecutionContext.
    state.output = output

    # Run the catalog's state initializer, if one is registered.
    registry = get_registry()
    init_call = state_init or catalog.schedule_mappings.get("state_init")
    if init_call:
        try:
            init_fn = registry.get(init_call)
        except Exception as exc:
            raise RunError(
                f"Site {site.name!r}: state_init callable {init_call!r} "
                f"is not registered. Did the catalog's python_module "
                f"forget to @register it? Registered: "
                f"{sorted(registry.names())}."
            ) from exc
        init_fn(env=env, state=state, config=config, layout=layout)

    # Resolve arrivals. arrival_entries arg wins; otherwise pull from
    # site.schedules via per-stream builders.
    if arrival_entries is None:
        entries = _resolve_schedules(
            site,
            catalog,
            registry,
            overrides=schedule_overrides or {},
            env=env,
            state=state,
            config=config,
            layout=layout,
            rng=rng,
        )
    else:
        entries = list(arrival_entries)

    # Dispatch one SimPy process per entry. Each entry MUST carry a
    # ``kind`` matching an entity kind declared in the catalog and
    # routed by at least one active mode. ``arrival_time`` (hours) is
    # consumed by the wrapper to delay the workflow start. An optional
    # ``mode`` key selects the workflow mode explicitly when more than
    # one active mode routes the same kind.
    #
    # Two dispatch tables:
    #   unique_kind_to_mode: kind -> mode, only when exactly one active
    #     mode routes the kind. Used when the entry omits ``mode``.
    #   mode_kind_to_mode: (mode_name, kind) -> mode, for entries that
    #     specify ``mode`` explicitly.
    # ``contended_kinds`` records which modes claim each kind that
    # appears in more than one active mode, to produce actionable
    # error messages.
    unique_kind_to_mode: dict[str, WorkflowMode] = {}
    contended_kinds: dict[str, list[str]] = {}
    mode_kind_to_mode: dict[tuple[str, str], WorkflowMode] = {}
    active_mode_names = {m.name for m in active_modes}
    for mode in active_modes:
        for kind in mode.arrival_routing:
            mode_kind_to_mode[(mode.name, kind)] = mode
            if kind in contended_kinds:
                contended_kinds[kind].append(mode.name)
            elif kind in unique_kind_to_mode:
                contended_kinds[kind] = [
                    unique_kind_to_mode[kind].name,
                    mode.name,
                ]
                del unique_kind_to_mode[kind]
            else:
                unique_kind_to_mode[kind] = mode

    primitives = build_default_primitives()
    entities: list[Entity] = []
    for raw in entries:
        if "kind" not in raw:
            raise RunError(
                f"Arrival entry is missing required 'kind' key. "
                f"Got: {raw!r}."
            )
        kind = raw["kind"]
        explicit_mode = raw.get("mode")
        if explicit_mode is not None:
            mode = mode_kind_to_mode.get((explicit_mode, kind))
            if mode is None:
                if explicit_mode not in active_mode_names:
                    raise RunError(
                        f"Arrival entry {raw!r}: mode {explicit_mode!r} "
                        f"is not active in site {site.name!r}. Active "
                        f"modes: {sorted(active_mode_names)}."
                    )
                routed = list(catalog.mode(explicit_mode).arrival_routing)
                raise RunError(
                    f"Arrival entry {raw!r}: mode {explicit_mode!r} does "
                    f"not route entity kind {kind!r}. Kinds routed by "
                    f"{explicit_mode!r}: {routed}."
                )
        else:
            mode = unique_kind_to_mode.get(kind)
            if mode is None:
                if kind in contended_kinds:
                    raise RunError(
                        f"Arrival entry {raw!r}: entity kind {kind!r} is "
                        f"routed by multiple active modes "
                        f"({contended_kinds[kind]}). Add a 'mode' key to "
                        f"the arrival entry to disambiguate."
                    )
                raise RunError(
                    f"Entity kind {kind!r} is not routed by any active "
                    f"mode in site {site.name!r}. Active routing: "
                    f"{ {m.name: list(m.arrival_routing) for m in active_modes} }."
                )
        graph_name = mode.arrival_routing[kind]
        graph = mode.graphs[graph_name]

        arrival_time = float(raw.get("arrival_time", 0.0))
        ent_id = raw.get("id") or _auto_id(kind, len(entities))
        # ``mode`` is a dispatch directive, not an entity attribute —
        # strip it before storing attrs.
        attrs = {
            k: v for k, v in raw.items()
            if k not in ("kind", "id", "arrival_time", "mode")
        }
        attrs.setdefault("arrival_time", arrival_time)
        entity = Entity(id=str(ent_id), kind=kind, attrs=attrs)
        entities.append(entity)

        ctx = ExecutionContext(
            env=env,
            primitives=primitives,
            entity=entity,
            state=state,
            config=config,
            layout=layout,
            registry=registry,
            output=output,
            graphs=mode.graphs,
            rng=rng,
        )
        env.process(_arrival_wrapper(env, graph, ctx, arrival_time))

    # Run.
    if until is not None:
        env.run(until=until)
    elif entries:
        env.run()
    # else: nothing scheduled, leave env at time 0 untouched.

    return RunResult(
        site=site,
        catalog=catalog,
        env=env,
        state=state,
        output=output,
        config=config,
        entities=entities,
    )


# ---- Helpers --------------------------------------------------------


def _arrival_wrapper(env, graph, ctx: ExecutionContext, arrival_time: float):
    """Delay the workflow until ``arrival_time`` then ``yield from``
    the graph executor. SimPy needs a generator function here so we
    can ``env.process(...)`` the result."""
    delay = arrival_time - env.now
    if delay > 0:
        yield env.timeout(delay)
    yield from execute(graph, ctx)


def _apply_resource_overrides(
    specs: list[ResourceSpec],
    overrides: Mapping[str, Mapping[str, Any]],
) -> list[ResourceSpec]:
    """Apply per-spec overrides from the site file.

    Currently only ``capacity`` is supported as an override (the
    common scenario-tweak). Extending to other fields is a one-line
    addition; we keep the surface small until a real catalog needs
    more.
    """
    if not overrides:
        return specs
    by_name = {s.name: s for s in specs}
    for spec_name, fields in overrides.items():
        spec = by_name.get(spec_name)
        if spec is None:
            raise RunError(
                f"resource_overrides references unknown spec {spec_name!r}. "
                f"Known: {sorted(by_name)}."
            )
        unknown = set(fields) - {"capacity"}
        if unknown:
            raise RunError(
                f"resource_overrides[{spec_name!r}]: unsupported field(s) "
                f"{sorted(unknown)}. Only 'capacity' may be overridden in v1."
            )
        if "capacity" in fields:
            # ResourceSpec is frozen — build a fresh one with the
            # overridden capacity.
            from dataclasses import replace
            by_name[spec_name] = replace(spec, capacity=fields["capacity"])
    return list(by_name.values())


def _resolve_schedules(
    site: SiteModel,
    catalog: Catalog,
    registry,
    *,
    overrides: Mapping[str, Any],
    env, state, config, layout, rng,
) -> list[dict]:
    """Resolve every site ``schedules:`` entry into a flat arrival list.

    Each schedule entry's name must appear as a key in the catalog's
    ``schedule_mappings`` dict. The mapping is itself a dict with at
    least a ``builder`` key (dotted name in the registry); other keys
    become keyword args to the builder. The site's value for the
    schedule name is passed as the ``schedule`` kwarg, so a YAML like::

        # catalog
        schedule_mappings:
          truck_rail.train_arrivals:
            builder: altrios.lifts.helpers.build_train_arrivals
            entity_kind: train

        # site
        schedules:
          truck_rail.train_arrivals: data/trains.csv

    Calls::

        build_train_arrivals(
            schedule="data/trains.csv",
            entity_kind="train",
            env=env, state=state, config=config, layout=layout, rng=rng,
        )
    """
    if not site.schedules:
        return []
    all_entries: list[dict] = []
    for name, raw_schedule in site.schedules.items():
        if name in overrides:
            raw_schedule = overrides[name]
        mapping = catalog.schedule_mappings.get(name)
        if mapping is None:
            raise RunError(
                f"Site {site.name!r}: schedule {name!r} has no entry in "
                f"catalog {catalog.name!r}.schedule_mappings. Known: "
                f"{sorted(catalog.schedule_mappings)}."
            )
        if not isinstance(mapping, Mapping) or "builder" not in mapping:
            raise RunError(
                f"Catalog {catalog.name!r}: schedule_mappings[{name!r}] "
                f"must be a mapping with a 'builder' key, got {mapping!r}."
            )
        builder_name = mapping["builder"]
        builder_kwargs = {k: v for k, v in mapping.items() if k != "builder"}
        try:
            builder = registry.get(builder_name)
        except Exception as exc:
            raise RunError(
                f"Schedule builder {builder_name!r} for schedule {name!r} "
                f"is not registered. Did the catalog's python_module "
                f"forget to @register it?"
            ) from exc
        try:
            built = builder(
                schedule=raw_schedule,
                env=env,
                state=state,
                config=config,
                layout=layout,
                rng=rng,
                **builder_kwargs,
            )
        except Exception as exc:
            raise RunError(
                f"Schedule builder {builder_name!r} for schedule {name!r} "
                f"failed: {exc}"
            ) from exc
        if not isinstance(built, Iterable):
            raise RunError(
                f"Schedule builder {builder_name!r} must return an "
                f"iterable of arrival dicts, got {type(built).__name__}."
            )
        all_entries.extend(built)
    # Sort by arrival_time for deterministic dispatch (SimPy would
    # process them by env.timeout anyway, but pre-sorting keeps logs
    # readable).
    all_entries.sort(key=lambda e: float(e.get("arrival_time", 0.0)))
    return all_entries


def _auto_id(kind: str, idx: int) -> str:
    """Fallback entity id when an arrival entry doesn't supply one."""
    return f"{kind}-{idx}"
