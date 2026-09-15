# Three-corridor design of experiments

Powertrain comparison across Spokane–Lethbridge, Winnipeg–Chicago and
Toronto–Battle Creek, with Spokane–Lethbridge carrying a catenary deep dive.

```
build_experiments.py   in/*.csv          ->  experiments.csv   (the full design)
run_sweep.py           experiments.csv   ->  out/sim_metrics.csv, out/stop_events.csv
postprocess_antiidling.py                ->  out/results.csv   (every cell)
```

**279 analysis cells, 21 simulations.** Cost scenarios and anti-idling levels
change no physics, so they expand a finished result rather than adding runs.
`Sim_Key` marks which cells share a simulation; `Cell_ID` identifies the cell.

## Factors

| # | Factor | Levels | Affects |
|---|---|---|---|
| 1 | Corridor | Toronto–Battle Creek, Winnipeg–Chicago, Spokane–Lethbridge | simulation |
| 2 | Powertrain | diesel, hybrid (all corridors); catenary + battery-electric, catenary + dual-mode (Spokane only) | simulation |
| 3 | Catenary architecture | 25 kV AC, 2×25 kV AC autotransformer, 9 kV DC | simulation |
| 4 | Infrastructure cost | low ×0.5, med ×1.0, high ×2.0 (catenary cells only) | cost only |
| 5 | Energy prices | low / med / high, harmonized across all cells | cost only |
| 6 | Hybrid battery | 0.1, 0.5, 1.0, 2.0 MWh | simulation |
| 7 | Anti-idling | none, AESS without APU, AESS with APU, battery-as-APU (hybrid only) | post-processing |

Simulations per corridor: Toronto 5, Winnipeg 5, Spokane 11.

Architecture, infrastructure and energy levels reproduce `ARCH_MEDIUM_COSTS`,
`INFRA_LEVELS` and `ENERGY_LEVELS` in
`catenary_optimization/energization_optimization_corridor.py`.

## Choices worth knowing about

**Traffic and speed limits come from the RailState calibration.** Each corridor
runs its own measured consist plan from `../corridor_inputs/`, built by
`railstate_corridor.py` in the analysis repo: trains per week sized so simulated
carload-km matches observed carload-km over the sensor span (138 / 161 / 88),
consists sampled month-by-month in proportion to real volume, loading split by
BTS gateway. Track limits are `min(posted, corridor p99 observed spot speed)` —
52.2 / 54.1 / 34.0 mph — carried in `corridor_meta_<slug>.csv`. The equipment
ceiling is held non-binding at 70 mph so the track governs everywhere.

**Every powertrain gets the same starting tractive effort** (200,000 lbf, the
Wabtec ET44AC figure ALTRIOS already cites for locomotive mass and horsepower),
so a difference between cells is a difference in powertrain, not in how hard the
locomotive can pull.

**Energy prices are harmonized by construction.** Both catenary and
non-catenary cells price from the same EIA AEO tables that `metric_calculator`
loads, scaled by the cell's multipliers. `Energy_Level` is therefore on every
row, not only catenary rows — without that, the central comparison of the study
would partly be an artifact of which price source each arm happened to use.
Note the two AEO tables carry different units (diesel $/gal, electricity
cents/kWh, both in 2025 dollars); `postprocess_antiidling.py` converts.

**Anti-idling acts on individual stops, not on corridor dwell totals.** A
shutdown threshold is a per-stop rule, so `run_sweep.stop_events` emits one row
per stop — duration and fuel burned — and the post-processor applies the
threshold to the real distribution. Applying it to an aggregate would assume a
stop-duration distribution instead of measuring one.

## Open items

**The anti-idling parameters are placeholders.** `Idle_Fuel_Fraction`,
`Shutdown_Threshold_Seconds` and `Restart_Penalty_MJ_Per_Event` in
`in/anti_idling.csv` are plausible values, not sourced ones — the `Source`
column says so on each row. They must be replaced before any result leaves the
team. They are isolated in data so that replacing them is a one-line CSV edit.

**Catenary cells currently use uniform energization.** `network_for()` energizes
the whole corridor at the architecture's voltage. The existing catenary analysis
instead *optimizes* the buildout per architecture and cost cell. Wiring the
optimized layouts in is the remaining integration; until then the catenary arms
are an upper bound on electrified route-miles and should not be read as the
optimized buildout.

**Simulated run times are ~30% shorter than observed** on the corridors with
real posted limits, attributed to non-running time ALTRIOS does not model
(crew changes, signal holds, terminal dwell). This affects any metric
denominated in time, including locomotive cycle time and fleet count. Energy per
car-mile is much less exposed. See `speed_calibration.py`.

**Spokane–Lethbridge's posted limits are synthetic** (the network builder's
60/15 mph constants). The p99 cap is what makes the corridor usable at all;
its speed results are still the least trustworthy of the three.
