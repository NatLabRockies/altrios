# `altrios.lifts` — intermodal terminal simulation

Container-flow simulation for intermodal terminals (truck<->rail,
rail<->vessel, vessel<->truck). The freight workflows are authored as
YAML step graphs that the
[`altrios.lifts.workflow_engine`](../workflow_engine/) executes; this package
provides the freight-specific catalog, Python helpers, and demo sites.

## Entry point: `run_site`

```python
from altrios.lifts.workflow_engine import run_site
from altrios.lifts.terminal.python_helpers import assemble_outputs

result = run_site(
    "python/altrios/lifts/sites/allouez_truck_rail.yaml",
    seed=42,
)
container_data, resource_log = assemble_outputs(result, mode_name="truck_rail")
```

The three bundled sites — `allouez_truck_rail.yaml`,
`allouez_rail_vessel.yaml`, `allouez_vessel_truck.yaml` — are exercised
end-to-end by the demos in [`demos/`](./demos/) and pinned by the
smoke test in [`tests/test_freight_parity.py`](./tests/test_freight_parity.py).

## Files

| File | Role |
|------|------|
| `catalog.yaml` | Freight catalog: 3 modes (truck_rail / rail_vessel / vessel_truck), entity kinds, schedule builder bindings, fine-grained arrival graphs |
| `python_helpers.py` | Registered Python callables invoked from the catalog (state_init, schedule builders, per-arrival escape hatches, `assemble_outputs`) |
| `specs.py` | Declarative `ResourceSpec` instances for the SimPy pools used by the freight modes |
| `consumption.py` | Per-event consumption recording helpers and `consumption_records` buffer; dual-writes to the runner's `OutputCollector` |
| `utilities.py` | Shared helpers: timetable / drayage / vessel schedule builders, container-event recording, log gate |
| `yard_flow.py` | SimPy generators for `stack_in` / `stack_out` / `yard_tractor_haul` (the irreducibly stateful yard moves) |
| `distances.py` | Yard geometry (distance table) computed from site layout |
| `classes.py` | Small equipment dataclasses (`container`, `truck`, `rtg`, `sts_crane`, ...) and `loggingLevel` enum |
| `sites/*.yaml` | Per-demo site files (consumed by `run_site`) |
| `resources/*` | Canonical CSVs (train_consist_plan, drayage_schedule, vessel_call_list) and `config.yaml` (yard/terminal/gates/vessel/yard_stack/energy_use/layout) |
| `demos/*.py` | Standalone demo scripts driving `run_site` against the three site files |
| `tests/test_freight_parity.py` | Smoke test pinning IC/OC counts, total energy, and sim end time per mode |
