# Plan: lifts package vessel workflows + framework with flexible resource sharing

> **Authored:** 2026-06-29 (planning session)
> **Status:** Approved by user for implementation
> **Scope:** Phase 1 actionable; Phase 2 sketched; Phase 3 design-only.

## TL;DR
Replace the existing `intermodal_rail` mode with three new modes — `truck_rail`, `rail_vessel`, `vessel_truck` — each defining its own resource references and arrival process. Pull forward two abstractions from later phases: (a) declarative resource specs where resources are declared once and any mode can reference them by name, with cross-mode references implicitly meaning a shared SimPy primitive; and (b) per-mode metadata on `TerminalMode` so the dispatcher is mode-agnostic. The framework makes **no assumption** that all modes share any particular resource or follow any particular journey shape — sharing is per-resource and emerges from which modes reference which name.

Net result: Phase 2 (concurrent multi-mode) is mostly state-renaming; Phase 3 (YAML DSL) becomes a file-format swap. New use cases can introduce modes that don't share a stack, use entirely different equipment, or share only a single gate pool — all without framework changes.

---

## Confirmed model — equipment chains for the three Phase 1 modes

These chains are **properties of these specific modes**, not framework assumptions. A future mode is free to follow any other shape.

All three Phase 1 modes happen to route containers through a shared container stack; each container's journey within these modes is:

1. **Source-endpoint lift** (mode-specific)
2. **Source chassis traversal** (terminal-chassis-by-yard-tractor for vessel/rail; drayage trucks drive their own road chassis into the stack)
3. **Stack-in lift** by main-stack RTG/top-pick (dynamic routing by availability)
4. **Container stack** (buffered storage)
5. **Stack-out lift** by main-stack RTG/top-pick (dynamic routing)
6. **Destination chassis traversal**
7. **Destination-endpoint lift**

The three modes differ at the endpoints:

| Mode | Source endpoint | Destination endpoint |
|---|---|---|
| `truck_rail` | drayage @ stack | train @ rail tracks |
| `rail_vessel` | train @ rail tracks | vessel @ berths |
| `vessel_truck` | vessel @ berths | drayage @ stack |

### Equipment and their cross-mode references

Each resource is declared once and referenced by name from one or more mode specs. The sharing pattern below is **specific to these three modes**, not a framework constraint:

| Resource | Referenced by |
|---|---|
| `berths` | rail_vessel, vessel_truck |
| `sts_cranes_by_berth` (partition_by=berth) | rail_vessel, vessel_truck |
| `rail_track_rtgs_by_track` (partition_by=track) — i.e. current `cranes_on_track` | truck_rail, rail_vessel |
| `main_stack_rtgs` (flat pool) | truck_rail, rail_vessel, vessel_truck |
| `top_picks` (flat pool, each carrying attached safety_car id) | truck_rail, rail_vessel, vessel_truck |
| `main_yard_tractors` (water↔stack) | rail_vessel, vessel_truck |
| `rail_yard_tractors` (rail↔stack) | truck_rail, rail_vessel |
| `terminal_chassis_pool` | truck_rail, rail_vessel, vessel_truck |
| `road_chassis_pool` | truck_rail, vessel_truck |
| `container_stack` | truck_rail, rail_vessel, vessel_truck |
| `parking_chassis_slots` (yard-tractor staging) | truck_rail, rail_vessel, vessel_truck |
| `tracks` | truck_rail, rail_vessel |
| `in_gates`, `out_gates` | truck_rail, vessel_truck |

Notable detail: rail-track RTGs and main-stack RTGs are two **independent** resources. Top-pick/safety-car alternative applies only to main-stack RTGs; the rail-track RTGs have no top-pick alternative. Both directions (source→dest and dest→source) within a mode are assumed symmetric; validate during implementation.

### Notable consequences for the existing code

- The existing `truck_rail` (current `intermodal_rail`) flow is **restructured**, not just renamed: drayage no longer interacts with `parking_*_stores`; containers route through the container stack; `chassis_*_stores` are reframed as terminal-chassis Stores.
- Byte-identical regression for legacy `truck_rail` is dropped. Functional equivalence (same input → same total container count, similar throughput within a sanity band) replaces it.
- Per-train OC/IC counters and `all_trucks_arrived_events` synchronization are largely **superseded** by stack buffering: trucks arrive on their own schedule; the stack provides decoupling.
- The `intermodal_rail` name is **removed cleanly**; no deprecation alias.

---

## Phase 1 — Three-mode rebuild with Phase-3-ready abstractions

### Phase 1A — Reorganize demos out of package internals (*parallel*)
- Move `python/altrios/lifts/demos/` to a new top-level `python/demos/lifts/`.
- Move the rail-truck `__main__` block from `python/altrios/lifts/terminal_sim.py` into a demo file under the new directory.
- Add a `README.md` describing each demo.

### Phase 1B — Declarative resource specs (*pulled forward from Phase 3*)
New module `python/altrios/lifts/resources_decl.py`:
- `ResourceSpec(name, kind, capacity, partition_by, init_items)` — `kind` is `Store|Resource|Container`; `capacity` may be an int or callable taking `(config, schedules)`; `partition_by` enables per-track / per-berth dicts; `init_items` is an optional factory that populates a Store. **No `private|shared` flag** — sharing emerges from cross-mode references to the same `name`.
- `EventSpec(name, per_arrival, ...)` — declares container-event-stream contributions.
- `build_state_from_specs(env, specs, sizing) -> dict[str, simpy primitive | dict]` — single factory consumed by `TerminalState.__init__`.
- When multiple modes contribute a `ResourceSpec` with the same `name`, the dispatcher dedups them and asserts the specs agree (same kind, capacity expression, partition_by). One SimPy primitive is created; both modes use it.
- The factory sets resulting primitives as **explicit attributes** on `TerminalState` (e.g. `state.tracks`, `state.in_gates`) for grep-ability.

### Phase 1C — Per-mode metadata on `TerminalMode` (*pulled forward from Phase 3*)
Extend the `TerminalMode` dataclass in `python/altrios/lifts/terminal_sim.py` with optional fields:
- `resource_specs: list[ResourceSpec]` — pools this mode requires. Dispatcher unions across active modes, dedup by name.
- `event_specs: list[EventSpec]` — per-arrival events this mode creates lazily.
- `event_types: list[str]` — container-event types this mode emits (for the pivot step).
- `container_id_pattern: re.Pattern | None` — regex used by `post_process` to extract arrival ids.
- `post_process: Callable[(LazyFrame, LazyFrame), (LazyFrame, LazyFrame)] | None` — mode-specific dataframe shaping.
- Defaults are empty/no-op. Dispatcher unions `event_types` across active modes and chains `post_process` calls. **All rail-specific logic in `run_terminal_simulation` moves into `truck_rail`'s `post_process`.**

### Phase 1D — New SimPy pools and entity dataclasses
Add to `python/altrios/lifts/classes.py`:
- Dataclasses: `sts_crane`, `main_stack_rtg`, `top_pick` (carries `safety_car` id — combined Store entry in Phase 1), `yard_tractor` (canonical name; `hostler` aliased for backward compat in call sites), `terminal_chassis`, `road_chassis`, `vessel`.
- New pools declared via `ResourceSpec` (referenced by name from the relevant modes per the table above). Existing `in_gates`/`out_gates`/`tracks` also migrate to specs.
- The legacy `parking_ic_stores`/`parking_oc_store`/`chassis_ic_store`/`chassis_oc_stores` are **retired** (their roles fold into the new `terminal_chassis_pool`, `container_stack`, and `parking_chassis_slots` specs).

### Phase 1E — Schedule builders and sample inputs
- Keep `build_train_timetable` in `python/altrios/lifts/utilities.py` for `truck_rail` and `rail_vessel`.
- Add `build_vessel_schedule(vessel_call_list, terminal_name, as_dicts)`.
- Add `build_drayage_schedule(...)` — Poisson trickle or CSV-driven; decouples drayage arrivals from train arrivals.
- New samples: `python/altrios/lifts/resources/vessel_call_list.csv` and `python/altrios/lifts/resources/drayage_schedule.csv`.

### Phase 1F — Endpoint adapters and shared helpers used by these three modes
New modules implementing the shared helpers (these are utilities the three Phase 1 modes happen to find useful; not framework primitives):
- `python/altrios/lifts/yard_flow.py`:
  - `stack_in(env, terminal, container, source_chassis)`, `stack_out(env, terminal, container, dest_chassis)` — Choose between main-stack RTG and top-pick by availability; transfer container; record events.
  - `_choose_stack_crane(terminal)` — Dynamic routing helper; configurable strategy.
  - `yard_tractor_haul(env, terminal, tractor_pool, container, from_zone, to_zone)`.
- New endpoint adapters:
  - `python/altrios/lifts/vessel_flow.py`: berth/STS orchestration (per-berth parallel workers, AllOf sync, per-arrival events).
  - `python/altrios/lifts/drayage_flow.py`: gate ingress, optional road-chassis claim, drive-to-stack, wait for stack service, gate egress.
- `python/altrios/lifts/train_flow.py` and `python/altrios/lifts/cranes.py` are **substantially simplified**: rail endpoint exchanges with the rail yard tractor pool, not directly with drayage; `all_trucks_arrived_events` synchronization is removed; the AllOf-based crane worker sync pattern is preserved where useful.
- `python/altrios/lifts/containers.py` is largely subsumed by `yard_flow.py`; retire or retain whichever yields the cleanest call graph.
- `python/altrios/lifts/truck_gate.py` refocused on pure gate ingress/egress; container handoff logic moves to `drayage_flow.py`.

### Phase 1G — Register three modes
In `python/altrios/lifts/terminal_sim.py`:
- `truck_rail`: drayage origin, rail destination (and reverse). `resource_specs` references the pools listed in the table above for `truck_rail`. `process_arrival` orchestrates train arrivals; drayage trucks run on their independent process loop driven by `drayage_schedule`.
- `rail_vessel`: rail origin, vessel destination (and reverse).
- `vessel_truck`: vessel origin, drayage destination (and reverse).
- `intermodal_rail` is **removed** — no alias, no deprecation shim.

### Phase 1H — Config extensions
Extend `python/altrios/lifts/resources/config.yaml`:
```yaml
vessel:
  berth_number: 2
  sts_cranes_per_berth: 2

yard_stack:
  main_stack_rtg_count: 6
  top_pick_count: 2
  terminal_chassis_count: 50
  main_yard_tractor_count: 12
  rail_yard_tractor_count: 8
  road_chassis_pool_count: 30
  stack_capacity: 500
  routing_strategy: availability   # availability | rtg_only | split

energy_use:
  trip_consumption:
    sts_loaded: {...}
    sts_empty: {...}
    main_stack_rtg_loaded: {...}
    top_pick_loaded: {...}
    yard_tractor_empty: {...}
    yard_tractor_loaded: {...}
```
Existing rail-track-RTG config entries are kept and possibly renamed for clarity.

### Phase 1I — Demos
Under `python/demos/lifts/`:
- `truck_rail_demo.py` — exercises rebuilt `truck_rail` against original `train_consist_plan.csv` plus a sample `drayage_schedule.csv`.
- `vessel_demo.py` — runs `rail_vessel` and `vessel_truck` sequentially against `vessel_call_list.csv` plus (for vessel_truck) `drayage_schedule.csv`.

---

## Phase 2 — Concurrent multi-mode at one terminal

With Phase 1's declarative resource specs and per-mode metadata in place, Phase 2 is small:
1. `run_terminal_simulation(modes: list[str], schedules: dict[str, Any], ...)` — multi-mode dispatcher; keeps `mode=` as deprecated alias.
2. Rename per-train state to per-arrival keyed by `(mode, native_id)` in `python/altrios/lifts/classes.py`, `train_flow.py`, `containers.py`, `cranes.py`.
3. Combined schedule input (dict-of-mode-to-input or unified DataFrame with `Mode` column).
4. New demo: all three Phase 1 modes concurrent at one terminal, exercising cross-mode pool sharing.

---

## Phase 3 — YAML DSL (design-only)

With Phase 1+2 complete, each mode is already pure data (`resource_specs`, `event_specs`, `event_types`, `container_id_pattern`) plus a generator. Phase 3:
1. Define YAML schemas for resources, events, schedule binding, and a process-step graph (built-in primitives: `request`, `release`, `timeout`, `transfer-container`, `record-event`, `record-energy`, `trigger-event`, `wait-event`, `branch`, `parallel`).
2. Implement YAML→`TerminalMode` loader. Resource section loads straight into the Phase 1 factory; only the step graph needs interpretation.
3. Python escape hatch: a step of `type: python` invokes a registered function. Dynamic routing (RTG vs top-pick) and other custom logic plug in here.
4. Re-express Phase 1's three modes in YAML as parity tests. Hand-written and YAML registrations coexist.

Phase 3 is design-only in this plan; no implementation steps are scoped.

---

## Relevant files

- `python/altrios/lifts/terminal_sim.py` — Extend `TerminalMode` (1C); dispatcher made mode-agnostic; `intermodal_rail` removed.
- `python/altrios/lifts/classes.py` — `TerminalState.__init__` consumes spec factory (1B); add new entity dataclasses (1D); retire legacy `parking_*` and `chassis_*` stores.
- New `python/altrios/lifts/resources_decl.py` — `ResourceSpec`, `EventSpec`, `build_state_from_specs`.
- `python/altrios/lifts/utilities.py` — Add `build_vessel_schedule`, `build_drayage_schedule`.
- `python/altrios/lifts/distances.py` — Add yard-tractor and STS/RTG sampling.
- `python/altrios/lifts/train_flow.py`, `cranes.py` — Substantially simplified; drop truck-sync events.
- `python/altrios/lifts/containers.py` — Largely subsumed by `yard_flow.py`.
- `python/altrios/lifts/truck_gate.py` — Refocused for pure gate ingress/egress.
- `python/altrios/lifts/energy_use.py` — No code changes; new `resource_type` values flow through.
- `python/altrios/lifts/resources/config.yaml` — New `vessel`, `yard_stack` sections.
- New: `vessel_flow.py`, `yard_flow.py`, `drayage_flow.py`, `resources/vessel_call_list.csv`, `resources/drayage_schedule.csv`.
- New demo dir `python/demos/lifts/`.

---

## Verification

1. **All three demos run end-to-end** — `truck_rail_demo.py`, `vessel_demo.py` (rail_vessel + vessel_truck) complete with non-empty `container_data` and `vehicle_log`.
2. **Per-mode event coverage** — For each Phase 1 mode, every container in `container_data` has the timestamps the mode's `event_types` declares it should produce. (Per-mode check; no framework-level "every container must visit the stack" assertion.)
3. **Equipment usage visible** — `vehicle_log.resource_type` distinct values include `sts_crane`, `main_stack_rtg`, `top_pick`, yard tractors (distinguished by pool), drayage trucks, rail-track RTG.
4. **Dynamic routing fires** — `vehicle_log` contains both `main_stack_rtg_*` and `top_pick_*` events under load.
5. **Energy / CO2 coverage** — All new equipment rows have non-NaN `energy_consumption(gal_or_kWh)` and `emissions(kgCO2)`.
6. **Functional truck_rail sanity** — Rebuilt `truck_rail` on original `train_consist_plan.csv` processes the same total IC/OC counts as the legacy run; throughput and wait-time metrics fall within a documented sanity band (baseline captured before refactor).
7. **Cross-mode resource sharing wired correctly** — When two Phase 1 modes both reference the same `ResourceSpec` name, `state.<name>` is a single SimPy primitive (object identity check). When only one mode of a pair is active (e.g. only `truck_rail`), the resource is still created (since `truck_rail` references it) and unreferenced specs from inactive modes are *not* created.
8. **Dispatcher mode-agnostic** — Grep `run_terminal_simulation` body for `train_`, `IC`, `OC`, `Train-`; none should remain.
9. **Imports** — `list_modes() == {"truck_rail", "rail_vessel", "vessel_truck"}`; importing `intermodal_rail` (by name through `get_mode`) raises `KeyError`.

---

## Decisions captured from interviews

- **Mode naming**: `truck_rail`, `rail_vessel`, `vessel_truck`. `intermodal_rail` removed entirely (no alias).
- **Resource sharing model**: each resource declared once with a name; modes reference resources by name. Cross-mode references to the same name automatically share one SimPy primitive. No binary `private/shared` flag. Sharing pattern is per-resource and per-mode — different parts may be shared with different subsets of workflows.
- **No framework assumption about container journey shape**: the seven-step journey applies to these three Phase 1 modes but is not baked into the dispatcher or `TerminalMode`. A future mode is free to follow any topology (no shared stack, no chassis, different equipment, etc.).
- **Two RTG pools** (for these three Phase 1 modes): rail-track RTGs (per-track partitioned; referenced by `truck_rail` and `rail_vessel`) and main-stack RTGs (flat pool; referenced by all three). Top-pick alternative applies only to main-stack RTGs.
- **Top-pick + safety car**: combined Store entry in Phase 1; can split in Phase 2 if needed.
- **Dynamic routing**: by availability; strategy configurable.
- **Chassis pools**: terminal chassis (yard-tractor-hauled, referenced by all three Phase 1 modes) and road chassis (drayage-pulled Store; referenced by `truck_rail` and `vessel_truck`). Never mixed.
- **Yard tractor pools**: `main_yard_tractors` (water↔stack) and `rail_yard_tractors` (rail↔stack); two distinct pools.
- **Parking is yard-tractor-only**: drayage trucks do not enter parking. (Property of these modes; framework imposes no parking concept.)
- **Drayage interacts at the stack**: drayage trucks drive into the stack area; main-stack RTG/top-pick services them in place. (Property of these modes.)
- **Schedule sources**: train consist plan, vessel call list, drayage schedule — three separate CSV inputs.
- **Pulled-forward variant**: declarative resource specs and per-mode metadata land in Phase 1.
- **Phase 1 demo scope**: two demos — `truck_rail_demo.py` and combined vessel demo. Multi-mode concurrent demo is Phase 2.
- **Demo location**: `python/demos/lifts/`, out of package internals.
- **Byte-identical regression for legacy truck_rail is dropped**: functional-equivalence sanity bands replace it.
- **Long-term DSL**: declarative YAML + Python escape hatch; no ceiling on simulation complexity.

---

## Further considerations (resolve during Phase 1)

1. **Stack-buffered decoupling of truck and train arrivals** — Recommend a simple Poisson-process drayage generator from a CSV of expected hourly volumes in Phase 1; richer scheduling later.
2. **Existing per-train events** — Keep `train_arrival_*`/`train_depart_*` (and add per-vessel analogues); retire OC-preparation sync; replace IC pickup sync with a stack-residence record.
3. **Road-chassis bring-vs-claim ratio** — Config-driven probability, default 0.5 in Phase 1.
4. **RTG / top-pick stack-block partitioning** — Flat pools in Phase 1; per-block partitioning via `ResourceSpec.partition_by` later.
5. **Direction symmetry per mode** — Validate during 1F; split into two generators per mode if asymmetric.
6. **Retiring `process_train_arrival` & friends** — Replace with a leaner train-endpoint adapter; preserve genuinely useful patterns (e.g. AllOf-based crane worker sync).
7. **Spec deduplication semantics** — When two modes contribute a `ResourceSpec` for the same name, what constitutes "agreement"? Recommend: same `kind`, same `partition_by`, and either identical `capacity` literals or a configured combine rule (e.g., `max`, `sum`, or "must match"). Default to "must match" with an explicit override mechanism for future modes that legitimately need to extend a shared pool.
