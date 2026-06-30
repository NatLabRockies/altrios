# Workflow Engine — Phase 3 Refactor Plan

**Status:** Phase 3A + 3B + 3C + 3D complete; next: Phase 3E (translate `truck_rail` to YAML)
**Last updated:** 2026-06-29
**Owner:** LIFTS subsystem maintainers

This document records the design and implementation plan for replacing the
hand-coded LIFTS `TerminalMode`s with a declarative, YAML-driven workflow
engine. The engine is intended to be **domain-neutral**; the freight
intermodal use case becomes the first of several possible *catalogs* that
plug into it (mining, airports, hospitals, factories, parcel hubs, etc.).

Keep this file synchronized with code: when an implementation decision changes,
update the corresponding section below.

---

## 1. Quick status

| Phase | Name | Status |
|---|---|---|
| 3A.1 | Add `role` tag to `ResourceSpec`; annotate 14 specs | **Done** (2026-06-29) |
| 3A.2 | Add `Entity` + `EntityKindSpec` dataclasses | **Done** (2026-06-29) |
| 3A.3 | Create `altrios.workflow_engine` package; move generic pieces | **Done** (2026-06-29) — imports repointed, no backwards-compatible re-exports per project policy |
| 3A.4 | Rename `energy_use*` → `consumption*` | **Done** (2026-06-29) |
| 3A.5 | Rename `vehicle_log_df` → `resource_log_df`; add `role` + `quantity` cols | **Done** (2026-06-29) |
| 3A.6 | Re-run Phase 1+2 verification matrices | **Done** (2026-06-29) |
| 3B.1 | `expressions.py` (asteval sandbox + `ExpressionContext`) | **Done** (2026-06-29) — 19 tests |
| 3B.2 | `registry.py` (`@register` decorator for `python:` callables) | **Done** (2026-06-29) — 11 tests |
| 3B.3 | `interpreter.py` Step/StepGraph + simple primitives (bind, set_attr, branch, assert, log, timeout) | **Done** (2026-06-29) — 27 tests |
| 3B.4 | Resource primitives (request, release, transfer, record_event, record_resource_event, record_consumption) | **Done** (2026-06-29) — 14 tests + `output.py` |
| 3B.5 | Control-flow primitives (parallel, loop, spawn, make_event/wait_event/trigger_event) | **Done** (2026-06-29) — 15 tests |
| 3B.6 | `python` escape hatch | **Done** (2026-06-29) — 8 tests; **94 total workflow_engine tests pass** |
| 3C.1 | `yaml_loader.py` (safe_load + `!include` constructor + cycle detection) | **Done** (2026-06-29) — 14 tests |
| 3C.2 | `yaml_expressions.py` (`"{expr}"` detection + recursive walk) | **Done** (2026-06-29) — 13 tests |
| 3C.3 | `schemas.py`: `StepModel` / `StepGraphModel` / `ResourceSpecModel` | **Done** (2026-06-29) — 20 tests |
| 3C.4 | `schemas.py`: `EntityKindSpecModel` / `WorkflowModeModel` / `CatalogModel` / `SiteModel` + engine `Catalog` / `WorkflowMode` | **Done** (2026-06-29) — 38 tests |
| 3C.5 | `layout.py` (`Layout.distance` / `Layout.node`) | **Done** (2026-06-29) — 18 tests |
| 3C.6 | `loader.py` (`load_catalog` / `load_site` + `extends:` merge + python_module import) | **Done** (2026-06-29) — 22 tests; **219 total workflow_engine tests pass** |
| 3D | `distributions.py` (Constant + Uniform + Poisson — only what LIFTS uses today; extensible registry) + wire into `timeout.duration` | **Done** (2026-06-29) — 25 tests + 4 timeout-integration tests; **244 total workflow_engine tests pass** |
| 3E | Translate `truck_rail` to YAML; parity test | Planned (next) |
| 3F | Translate `rail_vessel` + `vessel_truck` to YAML | Planned |
| 3G | Retire Python-coded modes | Planned |
| 3H | Docs + non-freight (mining) smoke test | Planned |

---

## 2. Motivation

After Phases 1–2 the LIFTS dispatcher already has most of the structure of a
generic engine: declarative `ResourceSpec`s, union-merging across modes, a
generic output assembler, per-arrival routing on `_kind`. Roughly 80 % of the
code is already domain-neutral. What remains domain-specific:

- `process_arrival` generators (`process_train_arrival`, `process_vessel_arrival`,
  `process_drayage_arrival`) hard-code freight workflows in Python.
- Output column names (`container_id`, `train_id`, `vehicle_log`) leak freight
  vocabulary into the engine schema.
- Energy lookup keys and rate tables are freight-flavored constants.

The goals of Phase 3 are:

1. **Declarative workflows.** Replace `process_arrival` generators with YAML
   step-graphs that the engine interprets.
2. **Domain-neutral engine.** Move all freight-specific concepts (entity
   names, energy key conventions, schedule column mappings) into a *catalog*.
3. **Pluggable catalogs.** Make adding a new domain (e.g., mining) a matter
   of writing YAML + a small Python helpers module, with zero engine changes.
4. **Stochastic complexity.** Allow durations, travel speeds, and other
   numeric parameters to be distributions, sampled from a seeded RNG.

### Non-goals (deferred to Phase 4+)

- Multi-site networks (one engine simulating multiple sites with traffic
  between them).
- Graphical workflow editor / live debugger.
- Distributed simulation across processes or machines.
- Real-time data ingestion.
- Optimization-in-the-loop.

---

## 3. Target architecture

### 3.1 Two-tier file model

The engine consumes two YAML file types:

- **Catalog** = a reusable *domain library* ("site type"). One catalog covers
  one industry or terminal-class. Versioned with the engine repo. Defines:
  workflows, entity kinds, default `ResourceSpec`s, consumption rates,
  schedule column mappings, and a Python helpers module.
- **Site** = a per-physical-place instantiation. References one catalog,
  selects which modes are active, overrides resource counts, declares the
  spatial layout, and points at default arrival schedules.

```
python/altrios/                            # engine + freight catalog
├── workflow_engine/                       # DOMAIN-NEUTRAL ENGINE
│   ├── __init__.py
│   ├── resources.py                       # ResourceSpec + merge/build helpers
│   ├── entities.py                        # Entity + EntityKindSpec dataclasses
│   ├── steps.py                           # Step + StepGraph dataclasses
│   ├── interpreter.py                     # SimPy generator built from a graph
│   ├── expressions.py                     # asteval-based {entity.x} resolver
│   ├── distributions.py                   # constant / uniform / lognormal / ...
│   ├── yaml_loader.py                     # PyYAML loader + !include + schema
│   ├── registry.py                        # @register'd python: callables
│   └── output.py                          # resource_log + entity_data assembly
│
└── lifts/                                 # FREIGHT CATALOG
    ├── catalog/                           # the actual YAML files
    │   ├── truck_rail.yaml
    │   ├── rail_vessel.yaml
    │   ├── vessel_truck.yaml
    │   └── common/
    │       ├── equipment.yaml             # 14 default ResourceSpec defs
    │       ├── consumption_rates.yaml
    │       ├── distributions.yaml         # named distribution presets
    │       └── schedule_mappings.yaml
    ├── python_helpers.py                  # @register'd freight callables
    └── demos/sites/                       # example Site files for the demo
        ├── single_track_demo.yaml
        └── double_track_demo.yaml
```

User-provided catalogs and sites can live anywhere on disk.

### 3.2 Operating-scenario axis

The conceptual hierarchy is three-tier:

```
Site Type      = catalog       (truck-rail terminal, mining operation, airport)
Site           = site file     (Rotterdam terminal at coordinates X)
Scenario at Site = run config  (busy month vs quiet month at Rotterdam)
```

Only **two file types** (catalog and site) exist in YAML. The third tier
("scenario at site") is handled by either:

1. **Python overrides** on the entry call:
   ```python
   from altrios.workflow_engine import run_site
   run_site(
       "sites/rotterdam.yaml",
       seed=42,
       schedules={"truck_rail.train_arrivals": "data/busy_month.csv"},
       resource_overrides={"sts_crane": {"capacity": 8}},
   )
   ```
2. **Thin `extends:` override files** when users want each scenario in version
   control:
   ```yaml
   # sites/rotterdam/busy_month.yaml
   extends: ./site.yaml
   schedules:
     truck_rail.train_arrivals: data/busy_month_trains.csv
   seed: 100
   ```

---

## 4. Key abstractions

### 4.1 `ResourceSpec` (extends existing, adds `role`)

Already exists in Phase 2 (`altrios/lifts/resources_decl.py`). Phase 3A.1
adds a required **`role`** string field:

| Role | Used for | Logged how |
|---|---|---|
| `equipment` | STS crane, RTG, hostler, stack lift, chassis, yard tractor | Consumption rows + status transitions |
| `infrastructure` | Track, berth, gate lane, parking slot | Enter/exit timestamps only |
| `storage` | Container stack | Level traces |

`role` is an open-ended string in the engine; the three above are conventions.
The engine uses it for: (a) load-time validation (e.g., refuse to attach
`consumption_rates` to a non-equipment resource by default), (b) default
output-schema columns, (c) error messages and docs. **No behavioral branching
in the SimPy core** — all three roles are still requested/released identically.

### 4.2 `Entity` (new in Phase 3A.2)

```python
@dataclass
class Entity:
    id: str                          # unique within run
    kind: str                        # e.g., 'container', 'train', 'aircraft'
    attrs: dict[str, Any]            # typed by entity_kinds YAML
    parent_id: str | None = None
```

Entities **flow through workflows**. They have identity, attributes, and
event trails. They are *not* `Resource`s and do not get `request`ed —
they are moved between Stores via the `transfer` step primitive.

**Important distinction:** Individual intermodal containers are `Entity`
instances of kind `container`. The container *stack* (the physical pile)
is a `ResourceSpec` with `role: storage`, backed by a SimPy `Store` that
holds container entities.

### 4.3 `EntityKindSpec` (new in Phase 3A.2)

Declared per catalog. Statically types `Entity.attrs` so expression
references like `{entity.weight_t}` can be validated at YAML load time
rather than at simulation runtime.

```yaml
entity_kinds:
  container:
    attrs:
      arrival_time: float
      origin: str
      destination: str
      weight_t: float
      kind: str        # 'IC' (inbound) | 'OC' (outbound)
```

### 4.4 `Step`, `StepGraph`, `WorkflowMode` (new in Phase 3B)

```python
@dataclass(frozen=True)
class Step:
    id: str
    type: str                    # one of the 17 primitive names
    params: dict[str, Any]
    next: str | list[str] | None

@dataclass(frozen=True)
class StepGraph:
    name: str
    entry: str
    steps: dict[str, Step]
```

`WorkflowMode` is a `TerminalMode` subclass that replaces the
`process_arrival` callable with a `graphs: dict[str, StepGraph]` map and
an `arrival_routing: dict[str, str]` (mapping `_kind` to graph name).
The engine's generic dispatcher invokes the named graph through the step
interpreter.

### 4.5 `Catalog`

YAML-loaded bundle:

```python
@dataclass(frozen=True)
class Catalog:
    name: str
    schema_version: int
    resource_specs: tuple[ResourceSpec, ...]
    entity_kinds: dict[str, EntityKindSpec]
    consumption_rates: dict
    modes: tuple[WorkflowMode, ...]
    schedule_mappings: dict[str, ScheduleMapping]
    python_module: str | None
```

`load_catalog(path) -> Catalog` parses YAML, validates with pydantic,
registers any `python:` callables it references.

### 4.6 `ExpressionContext`

Expressions in workflow params (`duration: "{entity.weight_t / 3.0}"`)
are evaluated by **asteval** with a constrained symbol table:

| Namespace | Contents |
|---|---|
| `entity` | The current entity's `id`, `kind`, and `attrs.*` |
| `bindings` | Local variables set by earlier `bind` / `request` steps |
| `state` | Live `TerminalState` view: `state.container_stack.level`, etc. |
| `config` | Scalars from the site file's `config:` block |
| `layout` | Site coordinates and Manhattan distances |
| `env` | SimPy `env.now` |

Allowed operations: arithmetic, comparison, Boolean, `min`/`max`/`abs`/
`sqrt`/`log`/`exp`. **No general function calls in expressions.** Anything
beyond this uses a `python:` step.

---

## 5. Step primitives (v1 — 17 primitives)

Locked-down primitive set for Phase 3B. Catalogs author workflows
exclusively from these primitives plus the `python:` escape hatch.

| Primitive | Params (key ones) | Purpose |
|---|---|---|
| `request` | `pool`, `qty=1`, `partition_key`, `bind`, `priority`, `filter` | Acquire a Resource or take from a Store |
| `release` | `pool`, `bind` | Return a Resource or `put` back into a Store |
| `timeout` | `duration` (scalar OR distribution dict) | Advance simulated time |
| `record_event` | `entity`, `event_type`, `timestamp?` | Append to entity's event trail |
| `record_resource_event` | `resource`, `event_type`, `entity?`, `duration?`, `status?` | Resource-level status row |
| `record_consumption` | `resource`, `quantity` (energy/co2/fuel/...), `status`, `duration`, `entity?` | Add row to consumption log; rate × duration |
| `transfer` | `entity`, `from` (pool expr), `to` (pool expr) | Atomic Store-to-Store move |
| `bind` | `name`, `value` (expr) | Set a local variable |
| `set_attr` | `entity`, `attr`, `value` | Mutate cross-step entity state |
| `branch` | `condition` (expr), `true` (step id), `false` (step id) | Conditional jump |
| `parallel` | `branches` (step ids), `join` (`all`/`any`) | AllOf / AnyOf |
| `loop` | `over` (expr), `as`, `do` (step id), `parallel` (bool) | Sequential or parallel iteration |
| `spawn` | `graph`, `entity`, `wait` (bool) | Subprocess |
| `trigger_event` | `event_var` | Resolve a named SimPy event |
| `wait_event` | `event_var`, `mode` (`one`/`all`/`any`) | Block on a named event |
| `make_event` | `bind`, `count` | Mint events for AllOf patterns |
| `python` | `call` (dotted name), `args` (expr dict), `bind?` | Escape hatch — invoke a registered Python helper |
| `assert` | `condition`, `message?` | Invariant check |
| `log` | `level`, `message` (template) | Diagnostic |

(That's actually 19 — `assert` and `log` were added as utility primitives.
v1 implements all of them.)

---

## 6. Site `layout:` schema (v1 — locked)

The simplest schema that supports mining/airport/etc. without
over-engineering.

```yaml
layout:
  nodes:
    berth_1: {x: 0,    y: 0}
    stack_A: {x: 380,  y: 50}
    track_1: {x: 600,  y: 50, z: 2.5}     # z parsed but UNUSED in v1
```

**v1 rules:**
- Each node has `x` and `y` in **meters** (required) and optional `z` in
  meters (parsed but ignored — reserved for future gradient/lift modeling).
- Travel between any two named nodes uses **Manhattan distance**:
  `|x_a − x_b| + |y_a − y_b|`.
- No graph/edge form, no explicit distance matrix, no Euclidean option in v1.

**Expression access:**
```yaml
- {type: bind, name: dist, value: "{layout.distance('berth_1', 'track_1')}"}
- {type: bind, name: x, value: "{layout.node('stack_A').x}"}
```

If a future use case needs route-dependent travel (road segments, gradient
effects, congestion), it lives in `python_helpers.py` callables for now;
the engine remains layout-agnostic.

---

## 7. Distributions

All scalar params accept either a literal value or a distribution dict:

```yaml
duration: {dist: uniform, low: 28, high: 32}
```

**Shipped v1 distributions:** `constant`, `uniform`, `poisson` — only
what the freight catalog actually uses today (`random.uniform` jitter
on service times, `np.random.poisson` for per-window arrival counts,
plus bare scalars). Additional distributions (normal, lognormal,
triangular, exponential, discrete, empirical_cdf, etc.) are a
four-line addition each (subclass `Distribution`, register in
`_DIST_TYPES`) and will be added when a catalog needs them.

**Seeding:** v1 uses a single shared `numpy.random.Generator` per run,
threaded through `ExecutionContext.rng`. Reproducibility is anchored
by passing a seeded generator from the run setup. Per-stream RNGs
(so that adding/removing a step doesn't perturb unrelated
distributions) are a known follow-up; the engine boundary is already
distribution-agnostic so this is purely an `ExecutionContext` /
`_h_timeout` change when needed.

---

## 8. Implementation phases — detail

### 3A — Engine schema, extraction, renames (in progress)

Pure refactor; no new behavior visible to existing demos. Each sub-step is
verifiable independently via the Phase 1 + Phase 2 verification matrices
(`scripts/verify_lifts_phase1.py`, `scripts/verify_lifts_phase2.py`).

| Sub-step | Description |
|---|---|
| **3A.1** | Add required `role: str` field to `ResourceSpec`; annotate all 14 freight specs (`equipment`/`infrastructure`/`storage`). Validate role at construction. |
| **3A.2** | Add `Entity` and `EntityKindSpec` dataclasses in a new `entities.py`. Declarative only — not yet wired into workflows. |
| **3A.3** | Create `python/altrios/workflow_engine/` package. Move `resources_decl.py` → `workflow_engine/resources.py`. Move `entities.py` → `workflow_engine/entities.py`. **Update all import sites in `altrios.lifts` to the new path** (no backwards-compatible re-exports, per project policy). |
| **3A.4** | Rename `energy_use.py` → `consumption.py`; `record_energy_use` → `record_consumption`; `compute_energy_use` → `compute_consumption`; `energy_use_records` → `consumption_records`; the five `_record_*_energy` helpers → `_record_*_consumption`; `_energy_type_for` → `_fuel_type_for`. The `energy_value` function parameter and the output column key `"energy_consumption(gal_or_kWh)"` are deferred to 3A.5 (column-rename phase). `terminal.energy_use_config` attribute and YAML config block `energy_use:` are unchanged in 3A.4 — they refer to the config-data schema and will get their own rename phase later. |
| **3A.5** | Rename `vehicle_log` / `vehicle_log_df` → `resource_log` / `resource_log_df` everywhere (engine, demos, scripts). Rename the value column `energy_consumption(gal_or_kWh)` → `consumption_value`. Add a `role` column populated from the recording helper (currently every freight helper passes `role="equipment"`). Add a `quantity` column populated by `record_consumption` (default `"energy"`); this future-proofs the dataframe for non-energy quantities (water, CO2 direct, NOx, etc.) in Phase 3D without further schema changes. The emissions post-processing branch is now keyed off `consumption_value` and is conceptually scoped to `quantity == "energy"` rows (no filter needed today because that's the only quantity emitted). |
| **3A.6** | Re-run both verification matrices. Truck-rail counts must match exactly (IC=914 / OC=980 / energy=2468). Cross-mode demo must still produce 1998 entries. |

### 3B — Step interpreter + expressions

New code only. Lands `workflow_engine/interpreter.py`, `expressions.py`,
`registry.py`. Per-primitive unit tests under `python/altrios/
workflow_engine/tests/`. Expression evaluator is **asteval** (sandboxed
AST interpreter; allows `sqrt`/`log`/`min`/`max`/etc. but no general
function calls).

### 3C — YAML loader

`workflow_engine/yaml_loader.py`. Uses **PyYAML** with `yaml.safe_load`.
`!include` implemented as a custom constructor (~30 lines). Schema
validation via **pydantic v2**.

### 3D — Distributions + python-callable registry

`workflow_engine/distributions.py` (8 distributions listed above).
`workflow_engine/registry.py` exposes a `@register("name")` decorator
that catalogs use to expose Python helpers callable from `python:` steps.

### 3E — Author `truck_rail.yaml`

Translate `train_flow.py` + `drayage_flow.py` into a YAML catalog plus a
trim `python_helpers.py` containing only the irreducibly complex bits
(equipment factories, RTG dispatch heuristic, greedy schedule matcher).
**Parity test:** when the YAML uses constant-distribution durations
matching the Python scalars, the YAML mode must produce bit-exact
results under a fixed seed (IC count, OC count, total consumption, sim
end time).

### 3F — Author `rail_vessel.yaml` + `vessel_truck.yaml`

Vessel `AllOf` pattern uses `parallel: {join: all}`. Drayage subgraphs
are shared with `truck_rail` via `!include`. Parity tests per mode.

### 3G — Retire Python `TerminalMode`s

Once 3E and 3F parity tests pass, delete `train_flow.py`, `vessel_flow.py`,
`drayage_flow.py`. The `TerminalMode` class still exists in the engine but
no longer has Python-coded subclasses in the freight catalog.

User-facing API: introduce `run_site(site_path, **overrides)` as the new
canonical entry. The dict-based `run_terminal_simulation(...)` shim is
deleted (no backwards-compatible aliases per project policy).

### 3H — Docs + non-freight smoke test

Author `workflow_engine/examples/mining_haul.yaml` as a ~60-line example
demonstrating shovels → haul trucks → crusher / waste dump cycles with
stochastic durations. Smoke test confirms the engine works for a
non-freight domain without engine modification.

---

## 9. Decisions log

Each entry is locked. Changing a locked decision requires updating this
table and the implementation.

| # | Decision | Date | Rationale |
|---|---|---|---|
| 1 | **Expression evaluator** = `asteval` | 2026-06-29 | Sandboxed, active maintenance, gives math functions out of the box. |
| 2 | **YAML library** = `PyYAML` (`yaml.safe_load`) | 2026-06-29 | Already in repo deps; `!include` is a small custom constructor either way. Accept YAML 1.1 footguns (document them). |
| 3 | **Schema validation** = `pydantic v2` | 2026-06-29 | Better error messages than `jsonschema`; already in dep graph. |
| 4 | **Distribution seeding** = `numpy.random.SeedSequence` | 2026-06-29 | Reproducible under structural workflow changes. |
| 5 | **Error attribution** = `WorkflowStepError(workflow, step_id, entity_id, env_now, original_exc)` | 2026-06-29 | Wraps each step's exceptions; locality first. |
| 6 | **Parity tolerance for 3E/3F** | 2026-06-29 | Bit-exact under fixed seed when YAML uses constant-distribution durations matching Python scalars. Stochastic-vs-Python parity allowed ±0.5 % on aggregate metrics; counts must remain exact. |
| 7 | **Test framework** = `pytest` under per-package `tests/` dirs | 2026-06-29 | Phase 1 and 2 verification scripts stay as regression matrices until 3G; then removed. |
| 8 | **`Entity` vs `Resource` boundary** = orthogonal | 2026-06-29 | Entity = flow object with narrative; Resource = seized capacity. Containers are Entities; the stack is a Resource. Agentive machinery is modeled as an Entity-in-Store, no new abstraction. |
| 9 | **Resource `role` tag** = required, conventional set `{equipment, infrastructure, storage}` | 2026-06-29 | Author-facing semantic clarity; load-time validation; no behavioral branching in the SimPy core. |
| 10 | **Output schema** = collapse `vehicle_log` → `resource_log` with a `role` column; rename `container_id` to a catalog-configurable `entity_id_column` (freight catalog keeps the label `container_id`) | 2026-06-29 | Engine-neutral column names; freight retains familiar labels via catalog config. |
| 11 | **Consumption rename** = `record_energy_use` → `record_consumption(quantity: energy\|co2\|fuel\|...)`; `energy_rates` → `consumption_rates`; `energy_log` → `consumption_log` | 2026-06-29 | Generalizes to emissions and other quantities. Rename churn happens once, before 3E. |
| 12 | **Travel as composite** = deferred | 2026-06-29 | Expressible today as `timeout` + `record_consumption` + optional road-segment `request`/`release`. Add a `travel` macro only if patterns demand it. |
| 13 | **Two-tier file structure** = catalog + site (NOT three tiers in files) | 2026-06-29 | "Scenario at a site" axis handled by Python overrides or thin `extends:` files; no third YAML file type. Renamed from "scenario" to "site" terminology. |
| 14 | **Site layout schema** = 2-D Manhattan, meters, optional unused `z` | 2026-06-29 | Smallest schema that supports mining/airport/etc. without over-engineering. No graph/edge form in v1. |
| 15 | **Engine package location** = `python/altrios/workflow_engine/` | 2026-06-29 | Stays inside the `altrios` package; not yet a standalone distribution. |
| 16 | **Catalog reference syntax** = Python import path (e.g., `altrios.lifts`) for shipped catalogs; filesystem paths for user catalogs | 2026-06-29 | Both styles supported in the YAML loader. No named registry. |
| 17 | **Demo migration policy** = rewrite demos to use `run_site()` | 2026-06-29 | After 3G, the YAML scenario IS the contract. No transitional shims. |
| 18 | **Schema versioning** = mandatory `meta.schema_version: 1` on every YAML | 2026-06-29 | Enables future migrations. v1 is the only version in Phase 3. |

---

## 10. Open questions (resolve mid-implementation)

| # | Question | Default if no decision is made |
|---|---|---|
| 1 | One-arrival → N-entities patterns: handled via `spawn` or a multi-arrival type? | Use `spawn` (confirmed in 3F when vessel workflow is authored). |
| 2 | `init_items` form: pure DSL or Python callback? | Start with `init_items_python:` callback; add DSL only if asked. |
| 3 | Schedule mapping richness: how rich a DSL for matching arrivals to departures (e.g., greedy gap matcher)? | Expose as a `pairing_python:` callback; do not build a YAML DSL for it. |
| 4 | Hot-path perf: ~500 K step invocations expected for freight runs. | Measure in 3E. Consider a step-fusion pre-pass if measurable. |

---

## 11. Out of scope

Explicitly deferred to Phase 4 or later:

- Multi-site networks (one engine, multiple sites with traffic between them).
- Live data ingestion (real-time sensor / AIS / GPS feeds).
- Optimization-in-the-loop (engine inside a Bayesian optimizer's inner call).
- Distributed simulation across processes or machines.
- Graphical workflow editor or live debugger.

---

## 12. Glossary

| Term | Definition |
|---|---|
| **Engine** | The domain-neutral `altrios.workflow_engine` package — primitives, interpreter, loader, output assembly. Knows nothing about cranes, trains, or ships. |
| **Catalog** | A YAML + Python bundle defining one *site type* (freight terminal, mining operation, ...). Reusable. Ships with the engine or supplied by a user. |
| **Site** | A YAML file naming one catalog, selecting active modes, declaring layout, and overriding default resource counts. One per physical place modeled. |
| **Mode** | One process-flow family within a catalog (e.g., `truck_rail`, `rail_vessel`). A catalog typically defines several modes that can run concurrently. |
| **Entity** | A flow object — has identity, attributes, an event trail. Containers, trains, vessels, drayage trucks, aircraft, ore loads. |
| **Resource** | A seized capacity — has a SimPy `Resource`/`Store`/`Container` backing, queues, capacity limit. Cranes, tracks, berths, chassis pools. |
| **Workflow** | A `StepGraph` — the procedure run when an entity arrives. |
| **Step** | One node in a workflow — invokes a primitive with parameters. |
| **Primitive** | One of the 19 built-in step types the interpreter knows how to execute. |
| **Run** | One end-to-end simulation invocation of a site over some duration with some seed. |

---

## 13. How to update this document

When implementation diverges from this plan:

1. **Decisions log (section 9)**: append a new row if a previously open
   question is resolved; *do not* edit a locked decision in place — instead
   add a row noting the override, with the new date and the reason.
2. **Status table (section 1)**: update sub-step status as work proceeds.
3. **Architecture sections (3–8)**: edit in place; this is the canonical
   description of *what is*.
4. **Open questions (section 10)**: remove rows that get answered; add
   rows for newly-surfaced ambiguities.

The session-scoped working notes that drove this design live in chat /
session memory (not committed). This file is the long-lived record.
