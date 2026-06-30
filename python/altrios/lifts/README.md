# `altrios.lifts` — intermodal terminal simulation

Container-flow simulation for intermodal terminals (truck<->rail,
rail<->vessel, vessel<->truck).

## ✨ New entry point: `run_site`

> **All new code should call** [`altrios.workflow_engine.run_site`][run_site]
> **with one of the sites in [`./sites/`](./sites/).**
>
> The legacy `run_terminal_simulation` entry point is still available
> for backwards compatibility but will be removed at the end of
> Phase 3. See `WORKFLOW_ENGINE_PLAN.md` §8.A for the deletion plan.

```python
from altrios.workflow_engine import run_site
from altrios.lifts.python_helpers import assemble_outputs

result = run_site(
    "python/altrios/lifts/sites/allouez_truck_rail.yaml",
    seed=42,
)
container_data, resource_log = assemble_outputs(result, mode_name="truck_rail")
print(f"IC count = {container_data.filter(...)}")
```

The three bundled sites — `allouez_truck_rail.yaml`,
`allouez_rail_vessel.yaml`, `allouez_vessel_truck.yaml` — match the
demos that historically lived in `lifts/demos/*.py` exactly (modulo
sub-percent stochastic drift from a different RNG stream; see
`tests/test_freight_parity.py`).

## Legacy entry point (transitional)

```python
from altrios.lifts import run_terminal_simulation

container_data, resource_log, terminal_obj = run_terminal_simulation(
    modes=["truck_rail"],
    terminal="Allouez",
    inputs={"truck_rail": {"train_consist_plan": df}},
)
```

This still works today and is bit-for-bit identical to past releases.
Internally, the catalog YAML + `python_helpers.py` wrappers delegate to
the same `train_flow.process_train_arrival` / `drayage_flow` /
`vessel_flow` generators, so `run_site` shares the same domain logic.

## Files

| File | Role |
|------|------|
| `catalog.yaml` | Catalog declaration: 3 modes + schedule builders |
| `python_helpers.py` | Registered Python callables (B-phase wrappers) |
| `terminal_adapter.py` | Attribute proxy over the new engine state for legacy generators |
| `sites/*.yaml` | Per-demo site files (consumed by `run_site`) |
| `tests/test_freight_parity.py` | New-vs-legacy parity assertions |
| `train_flow.py`, `drayage_flow.py`, `vessel_flow.py` | Per-arrival SimPy generators (unchanged) |
| `specs.py` | Declarative `ResourceSpec` instances for SimPy pools |
| `consumption.py` | Module-level `consumption_records` buffer (legacy) |
| `terminal_sim.py` | Legacy `run_terminal_simulation` entry point + mode registry |

## Phase A roadmap

Phase A (full refactor) is scheduled immediately after Phase B's
git checkpoint. It will:

- Translate `specs.py` `ResourceSpec` callables into YAML form so
  freight site `resource_overrides:` becomes functional.
- Migrate the freight generators to take `(env, state, config,
  output, ...)` and inject the runner's RNG (eliminating the
  `random.seed(42)` hack in `python_helpers.build_freight_state`).
- Delete `terminal_sim.py`, the `TerminalMode` registry, and the
  `TerminalAdapter` shim.
- Move `state.container_events` and `consumption_records` onto
  `OutputCollector` so `assemble_outputs` collapses into
  `result.output.to_freight_dataframes()`.

[run_site]: ../workflow_engine/runner.py
