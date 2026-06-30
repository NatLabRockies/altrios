# Workflow Engine

The `altrios.lifts.workflow_engine` package is the domain-neutral simulation
core that drives the LIFTS intermodal terminal model and any other
catalog that plugs into it. It interprets YAML-declared workflows
against a SimPy event loop and emits structured event / consumption
logs.

This page is an orientation. For full design rationale, decision
history, and primitive reference see `workflow-engine-plan.md` in the
`python/altrios/lifts/workflow_engine/` package.

## Package layout

The whole workflow-engine subsystem lives under `altrios.lifts`:

- `altrios.lifts.workflow_engine` — the domain-neutral engine
  (primitives, interpreter, loader, output collector). Knows nothing
  about freight, mining, or any specific domain.
- `altrios.lifts.terminal` — the freight intermodal rail-terminal
  catalog (the original LIFTS use case): trains, drayage trucks,
  vessels, containers, RTGs, top-picks, yard tractors.
- `altrios.lifts.mine` — a small open-pit mining haul-cycle example
  catalog. Demonstrates the engine is genuinely domain-neutral.

Additional domain catalogs (airports, hospitals, parcel hubs…) would
land as further sibling sub-packages.

## Two-tier file model

Every workflow-engine run consumes exactly two YAML file types:

- **Catalog** (e.g. `python/altrios/lifts/terminal/catalog.yaml`) —
  defines a reusable *site type*: the entity kinds it knows about,
  the workflow step-graphs that process them, the modes that group
  those graphs into operational families, and the schedule-builder
  registrations. A catalog ships with a Python helpers module that
  registers the callables the YAML references.

- **Site** (e.g.
  `python/altrios/lifts/terminal/sites/allouez_truck_rail.yaml`) —
  names a catalog, picks which modes are active, supplies layout
  and configuration, and points at the schedule data the run should
  use. One site file per physical place being modeled.

The split exists so a single catalog (the LIFTS intermodal-terminal
domain knowledge) can be reused unchanged across many sites and
scenarios.

## Modes

A *mode* is one process-flow family inside a catalog (for LIFTS:
`truck_rail`, `rail_vessel`, `vessel_truck`). Each mode declares
which entity kinds it routes (e.g. `train`, `vessel`, `drayage`)
and which workflow graph handles each. A site activates one or more
modes.

When a site activates a single mode the dispatcher routes each
arrival entry by its `kind`. When a site activates multiple modes
that route the same `kind` (e.g. drayage is processed differently by
`truck_rail` and by `vessel_truck`), each arrival entry must carry
an explicit `mode` key. The catalog's schedule builders stamp this
key automatically for mode-specific schedule entries — see
`schedule_mappings` in
`python/altrios/lifts/terminal/catalog.yaml`.

The current LIFTS sample sites (all under
`python/altrios/lifts/terminal/sites/`):

| Site file | Active modes | Notes |
|---|---|---|
| `allouez_truck_rail.yaml` | `truck_rail` | Single-mode; train + drayage. |
| `allouez_rail_vessel.yaml` | `rail_vessel` | Single-mode; train + vessel exchange. |
| `allouez_vessel_truck.yaml` | `vessel_truck` | Single-mode; vessel + drayage. |
| `allouez_combined.yaml` | `truck_rail`, `vessel_truck` | Two modes sharing one yard / one stack / one tractor fleet. |
| `allouez_all_modes.yaml` | all three | Illustrative; trains and vessels duplicated across contended modes (see site file header for caveats). |

## Running a site

For bundled catalogs, the canonical entry point is the catalog's
`run` helper, which resolves the site by name:

```python
from altrios.lifts import terminal

result = terminal.run("allouez_truck_rail", seed=42)
print(result.env.now)               # simulation end time (hours)
print(len(result.entities))         # arrivals scheduled
print(len(result.output.event_log)) # container-event rows recorded
```

Under the hood, `terminal.run` (and `mine.run`) is a thin wrapper
around the engine's `run_site`, which also accepts a path or a
string directly:

```python
from altrios.lifts.workflow_engine import run_site

result = run_site("python/altrios/lifts/terminal/sites/allouez_truck_rail.yaml",
                  seed=42)
```

The returned `RunResult` exposes the active `SiteModel`, the
catalog, the SimPy `Environment`, the `state` namespace, the
`OutputCollector` (`event_log`, `consumption_log`, `resource_log`),
the resolved `config` dict, and the list of `Entity` objects that
were scheduled.

## Per-catalog post-processing

The engine output schema is intentionally loose: `record_event`,
`record_resource_event`, and `record_consumption` primitives accept
arbitrary row dicts. Wide-table assembly (pivoting event rows into
the familiar `container_data` shape, joining derived columns) is
the catalog's responsibility, performed in Python after the run.

For the LIFTS catalog, that helper is
`altrios.lifts.terminal.python_helpers.assemble_outputs`. It accepts either
a single `mode_name` for a single-mode site or a sequence of mode
names for a combined multi-mode site; the expected event-type
surface is then the union across all named modes.

```python
from altrios.lifts.terminal.python_helpers import assemble_outputs

# Single-mode:
cd, rl = assemble_outputs(result, mode_name="truck_rail")

# Multi-mode (combined site):
cd, rl = assemble_outputs(result, mode_name=("truck_rail", "vessel_truck"))
```

## Demos and smoke tests

Working examples live under `python/altrios/lifts/terminal/demos/`:

- `truck_rail_demo.py`, `rail_vessel_demo.py`, `vessel_truck_demo.py`
  — one per single-mode site.
- `multi_mode_demo.py` — drives `allouez_combined.yaml` and
  illustrates resource pool sharing plus multi-mode `assemble_outputs`.

Regression baselines for all five sites live in
`python/altrios/lifts/terminal/tests/test_freight_parity.py`
(`±0.5 %` tolerance on aggregate metrics for single/two-mode sites;
`±1.5 %` for the all-three-mode site; exact on entity counts).

## Adding a new catalog

A new catalog needs:

1. A YAML file declaring `entity_kinds`, `modes` (each with
   `arrival_routing` and `graphs`), `schedule_mappings`, and a
   `python_module` reference.
2. A Python module that registers the schedule builders, the
   `state_init` hook, and any per-arrival `python:` escape-hatch
   callables the workflow graphs invoke. Use the
   `@altrios.lifts.workflow_engine.registry.register("name")` decorator.

`python/altrios/lifts/terminal/catalog.yaml` and
`python/altrios/lifts/terminal/python_helpers.py` together form the
canonical reference catalog. A second small example used in the
engine tests lives at
`python/altrios/lifts/mine/mining_haul.yaml`.
