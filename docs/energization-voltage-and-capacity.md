# Substation Capacity and Line Voltage: What the Code Actually Supports

This note documents, precisely and from the current source, (1) what options
exist for **substation capacity** and **line (catenary) voltage**, (2) whether
voltage is a discrete choice or a continuous value, and (3) exactly where and how
these two quantities are consumed — in the **catenary placement optimization
(MILP)** versus the **train simulation / post‑processing line‑loss model**.

Primary sources:
- [python/altrios/optimization/energization_placement.py](../python/altrios/optimization/energization_placement.py) — the MILP.
- [python/altrios/energization.py](../python/altrios/energization.py) — placement helpers and the AC line‑loss model (`apply_line_losses`, `compute_line_losses`).
- [altrios-core/src/track/link/cat_power.rs](../altrios-core/src/track/link/cat_power.rs) — the `CatPowerLimit` struct the simulation reads.
- [altrios-core/src/consist/consist_model.rs](../altrios-core/src/consist/consist_model.rs) — where the simulation applies the power limit and line‑loss factor.

---

## 1. Substation capacity

### How it is specified
Substation capacity is the **maximum real power** a substation can deliver,
expressed in watts on the `Substation` dataclass:

- `Substation.max_power_watts: Optional[float]` — see [energization_placement.py](../python/altrios/optimization/energization_placement.py#L94). Any positive float (continuous). `None` falls back to a global default.
- Global fallback: `DEFAULT_SUBSTATION_MAX_POWER_WATTS = 20.0e6` (20 MVA) in [energization.py](../python/altrios/energization.py#L20), overridable per call via the `default_substation_max_power_watts` argument of `apply_line_losses`.

There is **no discrete list** of allowed capacities. Capacity is an arbitrary,
continuous watt value per substation.

### Where it is used
Substation capacity is **not used by the placement MILP at all**. The optimizer's
decision variables are only:

- `x[p]` — electrify track piece `p`,
- `s[c]` — install a shared structure in corridor cell `c`,
- `y[j]` — build candidate substation `j` (a lump‑sum capital decision).

Building a substation `y[j]` is a binary decision driven by **cost** and the
**feeding/spacing constraint** (each energized piece must lie within
`substation_feed_distance_miles` of a built substation). The MILP never reads
`max_power_watts`; a substation's capacity has zero effect on what gets
electrified or which substations are built. (Confirmed: `max_power_watts` appears
only in the `Substation` dataclass definition within the optimization module,
never in the model‑building or extraction code.)

Substation capacity **is** used later, in the separate AC line‑loss
post‑processing step `apply_line_losses` in [energization.py](../python/altrios/energization.py#L598). There it **derates the deliverable pantograph power** of each energized
interval. For a feeding substation with resistance `r_eff` to the interval and
voltage `v`, the capacity‑limited pantograph power solves `P + k·P² = Pmax` with
`k = r_eff / v²`:

```
p_pow = (-1 + sqrt(1 + 4·k·Pmax)) / (2·k)
```

The interval's final `power_limit_watts` is then set to
`min(base_rating, p_pow, p_volt)` (see below for `p_volt`). So capacity influences
the **simulation inputs** (the per‑interval power ceiling), but only if you run
`apply_line_losses`; it never influences the optimizer's placement/build choices.

---

## 2. Line (catenary) voltage

### Is it discrete or continuous?
**Continuous.** Voltage is a free floating‑point value in volts. There is no enum
or discrete option set constraining it. Two entry points:

- Global nominal voltage: `apply_line_losses(..., system_voltage_volts: float = 25_000.0, ...)` — any float ([energization.py](../python/altrios/energization.py#L602)). It is also written to the network config as `system_voltage_volts`.
- Per‑substation override: `Substation.voltage_volts: Optional[float]` ([energization_placement.py](../python/altrios/optimization/energization_placement.py#L101)). `None` uses the global value.

The resolved voltage per interval is `sub.voltage_volts if set else system_voltage_volts`.

### The one place that *looks* discrete — but is not the voltage itself
`apply_line_losses` accepts `electrification_type: str = "ac_25kv_1x25"`, which is
a key into `ELECTRIFICATION_IMPEDANCE_DEFAULTS` in [energization.py](../python/altrios/energization.py#L35):

| `electrification_type` | (R Ω/km, X Ω/km) |
|---|---|
| `ac_25kv_1x25` | (0.20, 0.50) |
| `ac_2x25_at`   | (0.075, 0.15) |
| `ac_50kv`      | (0.20, 0.50) |

This discrete choice **only selects default per‑km loop impedance**
(resistance/reactance) — it does **not** set the voltage. The names reference
common systems (25 kV, 2×25 kV autotransformer, 50 kV) but the numeric voltage is
still whatever you pass in `system_voltage_volts` / `voltage_volts`. You may also
bypass the table entirely by passing `resistance_ohms_per_meter` and
`reactance_ohms_per_meter` directly (both continuous). Passing an unknown
`electrification_type` (without explicit impedances) raises `ValueError`.

So: **impedance defaults are a discrete menu; voltage is fully continuous.**

### Where voltage is used
Like capacity, voltage is **not used by the placement MILP**. It appears only in
the `Substation` dataclass within the optimizer and is never read by the model.

Voltage is used in two places downstream:

1. **`apply_line_losses` derating (post‑processing).** Voltage drives both limits
   that can lower an interval's `power_limit_watts`:
   - Resistive‑loss/capacity limit via `k = r_eff / v²` (higher voltage → less
     loss → more deliverable power).
   - **Voltage‑drop limit**: with `z_eff = hypot(r_eff, x_eff)` and minimum
     pantograph voltage fraction `v_min_frac`
     (`DEFAULT_MIN_PANTOGRAPH_VOLTAGE_FRACTION = 0.7`, EN 50163‑style),
     `p_volt = v·(v − v_min_frac·v) / z_eff`.
   The interval also gets `voltage_volts`, `resistance_to_source_ohms`,
   `reactance_to_source_ohms`, and `district_id` stamped on it.

2. **The Rust train simulation.** Each `CatPowerLimit` carries `voltage`,
   `resistance_to_source`, `reactance_to_source`, and `power_limit`
   ([cat_power.rs](../altrios-core/src/track/link/cat_power.rs#L7)). At each step
   `Consist::set_cat_power_limit` in [consist_model.rs](../altrios-core/src/consist/consist_model.rs#L260)
   picks the active interval for the train's offset and computes a resistive
   line‑loss factor:

   ```
   loss_factor k = resistance_to_source / voltage²   (1/W), loss = k · pwr_elec_in²
   ```

   A non‑positive `voltage` disables loss. Reactance (`reactance_to_source`) is
   **not** used inside the simulation; it only affects the voltage‑drop capacity
   limit computed earlier in `apply_line_losses`.

---

## 3. Summary matrix

| Quantity | Type of choice | Set where | Used by MILP placement? | Used by line‑loss post‑proc (`apply_line_losses`)? | Used by Rust simulation? |
|---|---|---|---|---|---|
| Substation capacity (`max_power_watts`) | Continuous watts (per substation; global default 20 MVA) | `Substation.max_power_watts`, `default_substation_max_power_watts` | **No** | **Yes** — derates deliverable power via `P + kP² = Pmax` | Indirectly (via the derated `power_limit` it produces) |
| Nominal line voltage (`voltage_volts` / `system_voltage_volts`) | **Continuous volts** (no enum) | `system_voltage_volts` arg; `Substation.voltage_volts` override | **No** | **Yes** — sets `k = r/v²` and voltage‑drop limit | **Yes** — `loss = (r/v²)·P²` per step |
| Electrification type (`electrification_type`) | **Discrete** menu of 3 keys (impedance defaults only) | `apply_line_losses(electrification_type=...)` | No | Yes — picks default R/X per km (not voltage) | Via stamped `resistance_to_source` / `reactance_to_source` |
| Loop impedance (`resistance_/reactance_ohms_per_meter`) | Continuous (overrides the discrete menu) | `apply_line_losses(...)` | No | Yes | Via stamped values |

### Key takeaways
- **Voltage is continuous**, not a discrete selection. The only discrete
  electrification input is `electrification_type`, and it selects **default
  impedance**, not voltage.
- **Substation capacity is continuous** watts with a 20 MVA default.
- **The catenary‑placement optimizer (MILP) ignores both voltage and capacity.**
  It decides *where* to electrify and *which* substations to build based on cost,
  budget, high‑grade priority weighting, and the distance‑based feeding/spacing
  constraint — not on any electrical rating. `uniform_power_limit_watts` is
  applied as a constant to every energized segment and is likewise not derived
  from voltage or capacity.
- Voltage and capacity **do** matter to the physics — but only through the
  optional `apply_line_losses` step, which rewrites each energized interval's
  `power_limit_watts` and stamps voltage/resistance so the train simulation can
  enforce the ceiling and integrate resistive line loss (`loss = (R/V²)·P²`) each
  time step.

---

## 4. High‑voltage DC (HVDC) substations

**Bottom line: DC / HVDC substations are already technically enabled by the
current data structures and physics**, and the two DC ergonomics — a substation
**conversion‑efficiency** field and `dc_*` impedance defaults (including a
high‑voltage **9 kV** entry) — are now built in. Implementing an HVDC feed is
therefore a matter of choosing DC input values; no structural code change is
required.

### Why the current structure already supports DC

1. **Voltage is continuous and unit‑agnostic.** `system_voltage_volts` and
   `Substation.voltage_volts` are free floats in volts (see §2). Nothing
   constrains them to AC magnitudes; set them to a DC level (e.g. **9 kV** HVDC,
   or conventional 1.5/3 kV) directly.

2. **The loss law is literally the DC formula.** The simulation computes
   `loss = (resistance_to_source / voltage²)·P²` in
   [consist_model.rs](../altrios-core/src/consist/consist_model.rs#L260), and
   `apply_line_losses` uses `k = r_eff / v²`. That is exactly `I²R` with
   `I = P/V` — physically *exact* for DC, whereas for AC it is an approximation
   that neglects power factor. DC is, if anything, the more natural fit. At 9 kV
   the `1/V²` scaling gives far lower loss (and hence much wider substation
   spacing) than 1.5/3 kV DC for the same delivered power.

3. **Reactance degrades gracefully to zero.** The voltage‑drop limit uses
   `z_eff = hypot(r_eff, x_eff)`. DC has no reactance, so the `dc_*` defaults set
   reactance to `0.0`; then `z_eff = r_eff`, the correct DC voltage‑drop behavior.
   Reactance is never read by the Rust simulation anyway, so nothing is lost.

4. **No AC‑specific physics is baked in.** There is no VAr / power‑factor /
   load‑flow modeling that would misbehave under DC. The `electrification_type`
   menu only supplies *default impedances*; if you pass `resistance_ohms_per_meter`
   and `reactance_ohms_per_meter` explicitly, the string is ignored entirely.

### Built‑in DC ergonomics

**`dc_*` impedance defaults** — new rows in `ELECTRIFICATION_IMPEDANCE_DEFAULTS`
(`(R Ω/km, X Ω/km)`; reactance is `0.0` for all DC):

| `electrification_type` | (R Ω/km, X Ω/km) | Notes |
|---|---|---|
| `dc_1500v` | (0.03, 0.0) | Conventional 1.5 kV DC; heavy conductors → low R |
| `dc_3kv`   | (0.04, 0.0) | Conventional 3 kV DC |
| `dc_9kv`   | (0.20, 0.0) | **HVDC (9 kV).** Lower current at 9 kV allows lighter, standard‑catenary conductors, so R is near the 25 kV AC value while `1/V²` and zero reactance enable wide substation spacing |

**`Substation.efficiency`** — new optional field (grid → catenary conversion
efficiency in `(0, 1]`). Default is `DEFAULT_SUBSTATION_EFFICIENCY = 0.99` (AC,
transformer only). For a 9 kV HVDC substation set it lower (≈0.97) to capture
AC→DC rectifier loss. `apply_line_losses` records it in `substation_meta`, and
`compute_line_losses` scales each district's `substation_energy_joules`
(grid‑side input) up by `1/efficiency`, so rectifier losses flow through to the
grid‑emissions accounting.

### DC input assumptions to supply (9 kV HVDC example)

| Input | HVDC (9 kV) value |
|---|---|
| `system_voltage_volts` / `Substation.voltage_volts` | `9000.0` |
| `electrification_type` | `"dc_9kv"` (or pass explicit `resistance_/reactance_ohms_per_meter`) |
| `reactance_ohms_per_meter` | `0.0` (already the `dc_9kv` default) |
| `Substation.max_power_watts` | DC substation power rating (deliverable to catenary) |
| `Substation.efficiency` | ≈`0.97` (transformer + rectifier) |
| `min_pantograph_voltage_fraction` | Per the DC design standard (EN 50163 defines DC minimums, typically ~0.67) |
| `substation_feed_distance_miles` (MILP) | Can be **wider** than 1.5/3 kV DC thanks to the higher voltage, though still an independent input; the optimizer's constraint is a distance rule only |

With those, both the MILP placement (which is electrically agnostic anyway) and
the line‑loss/simulation path run unchanged. Example:

```python
net2, meta = en.apply_line_losses(
    net,
    substations,                       # each may set efficiency=0.97, voltage_volts=9000
    system_voltage_volts=9_000.0,
    electrification_type="dc_9kv",     # (0.20 Ω/km, 0.0 Ω/km) defaults
    default_substation_efficiency=0.97,
)
```

### Summary
HVDC substations are **enabled and ergonomically supported**: the `Substation`
(continuous voltage + capacity + `efficiency`) and `CatPowerLimit`
(resistance/reactance/voltage) structures and the `loss = R·(P/V)²` physics are
voltage‑agnostic and reactance‑optional, and the DC loss law is a special case the
code already implements exactly. Select `dc_9kv` (or pass explicit impedances),
set a 9 kV voltage, a substation `efficiency` (~0.97 for rectifier loss), and a
DC‑appropriate feed spacing — no structural changes needed.

---

## 5. Configuration presets and the pre‑processing helper

Rather than hand‑tuning `substation_feed_distance_miles`, `uniform_power_limit_watts`,
and the substation rating for each voltage class, use the pre‑processing helper
`suggest_optimization_inputs` in [energization.py](../python/altrios/energization.py).
It turns an electrification *choice* into a consistent set of optimizer and
line‑loss inputs.

### 5.1 Presets (`ELECTRIFICATION_CONFIGS`)

`ElectrificationConfig` bundles voltage, per‑meter loop impedance, conversion
efficiency, and minimum‑voltage fraction. Built‑in presets (keys, case‑insensitive):

| Preset | Voltage | Impedance key | Efficiency | Min‑V fraction |
|---|---|---|---|---|
| `ac_25kv`  | 25 kV | `ac_25kv_1x25` | 0.99 | 0.70 |
| `ac_2x25`  | 25 kV (2×25 AT) | `ac_2x25_at` | 0.99 | 0.70 |
| `ac_50kv`  | 50 kV | `ac_50kv` | 0.99 | 0.70 |
| `dc_9kv`   | 9 kV  | `dc_9kv` | 0.97 | 0.67 |
| `dc_3kv`   | 3 kV  | `dc_3kv` | 0.975 | 0.67 |
| `dc_1500v` | 1.5 kV| `dc_1500v` | 0.975 | 0.67 |

### 5.2 How the feed distance is derived

Feed distance is sized from the **voltage‑drop limit** (the same physics as
`apply_line_losses`). For a train drawing the design power $P_d$ at the
single‑sided coverage boundary (distance $d$ from its only feeding substation),
the pantograph voltage stays at or above the minimum fraction $f$ of nominal $V$
when

$$
V\,(1-f)\,V \;\ge\; |Z|\,d\,\frac{P_d}{V}
\quad\Longrightarrow\quad
d \;\le\; \frac{V^{2}\,(1-f)}{|Z|\,P_d},
\qquad |Z| = \sqrt{R^{2}+X^{2}},
$$

with $R, X$ the per‑meter loop impedance. The suggested
`substation_feed_distance_miles` is this $d$. Interior track fed from two
substations obtains roughly double the capability, so the single‑sided value is
conservative. Because $d \propto V^{2}$, higher voltages permit far wider
spacing; DC's zero reactance further lowers $|Z|$. The helper also sets:

- `uniform_power_limit_watts = ` $P_d$;
- `substation_max_power_watts = ` $P_d \times$ `concurrent_trains_per_substation` (default 2);
- the electrical fields (`system_voltage_volts`, `electrification_type`, R/X per meter, `efficiency`, min‑V fraction) from the preset.

### 5.3 Derived values at the default design point

With `design_train_power_watts = 10 MW`, `concurrent_trains_per_substation = 2`
(so every substation is rated 20 MW), and `min_feed_distance_miles = 1`:

| Preset | Feed distance (mi) | Per‑train limit | Substation rating |
|---|---|---|---|
| `ac_25kv`  | 21.6 | 10 MW | 20 MW |
| `ac_2x25`  | 69.5 | 10 MW | 20 MW |
| `ac_50kv`  | 86.5 | 10 MW | 20 MW |
| `dc_9kv`   | 8.3  | 10 MW | 20 MW |
| `dc_3kv`   | 4.6  | 10 MW | 20 MW |
| `dc_1500v` | 1.5 (clamped) | 10 MW | 20 MW |

These track real practice (25 kV AC substations tens of km apart; 3 kV DC every
~10 km). The very wide 50 kV / 2×25 figures are the *voltage‑drop* ceiling only;
thermal and protection limits apply in practice, so clamp with
`max_feed_distance_miles` when needed.

### 5.4 Usage

```python
from altrios import energization as en

s = en.suggest_optimization_inputs("dc_9kv", design_train_power_watts=10e6)

# 1) Placement optimizer
result = en.energize_optimized(
    network, substations,
    **s.optimizer_kwargs(),          # uniform_power_limit_watts + substation_feed_distance_miles
    budget=budget, costs=costs,
)

# 2) Line-loss post-processing (consistent electrical assumptions)
net2, meta = en.apply_line_losses(
    result.network, substations,
    **s.line_loss_kwargs(),          # voltage, impedance, ratings, efficiency, min-V fraction
)
```

The two `*_kwargs()` helpers guarantee the optimizer and the loss model see the
same voltage class. Per‑substation `max_power_watts`, `voltage_volts`, and
`efficiency` still override the suggested defaults where set.

---

## 6. Illustrative capital & energy cost scenarios (sweep assumptions)

The cost‑based (lifecycle `min_cost`) sweep in
[applications/catenary_optimization/energization_optimization_corridor.py](../applications/catenary_optimization/energization_optimization_corridor.py)
evaluates each architecture over a 3 × 3 matrix of **infrastructure‑cost** and
**diesel‑vs‑electricity‑cost** scenarios (9 scenarios per architecture, 3
architectures = 27 runs). A single discount rate and horizon are used throughout.

> The dollar figures below are **illustrative, order‑of‑magnitude** values chosen
> as reasonable medium point estimates with deliberately generous low/high
> ranges; they are not a costing study. They are consistent with commonly cited
> ranges for overhead electrification (roughly \$2–5M per route‑mile for
> catenary, and \$10–30M for traction/rectifier substations) and with
> representative recent U.S. diesel and industrial‑electricity prices. Replace
> them with project‑specific figures for real analysis.

### 6.1 Fixed assumptions

| Parameter | Value | Basis |
|---|---|---|
| Discount rate | 7% | `altrios.defaults.DISCOUNT_RATE`; matches the U.S. OMB Circular A‑94 default real discount rate for public‑investment analysis and is the rate already used elsewhere in ALTRIOS (`metric_calculator`). |
| Project horizon | 25 yr | Representative catenary/substation service life (overhead line equipment is commonly depreciated over ~25–40 yr); PVAF ≈ 11.65 at 7%. |
| Traffic | 24 trains/day | Illustrative moderate‑density corridor. In production this comes from the ALTRIOS demand / train‑planner pipeline, not a fixed guess. |

### 6.2 Architecture medium capital costs

Per‑architecture medium point estimates (the low/high scenarios scale these).
Feed distance comes from `suggest_optimization_inputs` and governs how many
substations get built.

> **Basis and honesty note.** These are 2025‑USD engineering syntheses calibrated to
> the published range for greenfield mainline electrification, not a costing study.
> OCS (shared structure + one track's contact/messenger wire) totals ~\$2.7–3.1M per
> single‑track route‑mile, within the ~\$2.5–3.5M/mi band spanned by international
> figures (IEA 2019), U.S. corridor estimates (Solutionary Rail 2016), and
> engineering guidance (Network Rail 2015). AC traction substations are anchored near
> recent feeder‑station contracts (~\$20M; Balfour Beatty/Crossrail 2013) and
> discounted for open‑country siting. The 2×25 kV autotransformer premium follows
> INECO (2011) and Courtois (1993). The 9 kV MVDC substation is modeled as a modular,
> containerized VSC (active‑front‑end) converter station — the form underpinning
> MVDC's competitive case (Gómez‑Expósito et al. 2014; Simiyu & Davidson 2021;
> UIC 2019) — priced conservatively for the power‑electronics premium, with the
> low‑infra case representing a mature, mass‑produced modular deployment. The exact
> split between the shared *structure* and per‑track *catenary* terms remains
> engineering judgment — replace with project quotes for a real analysis.

| Architecture | Feed dist | Structure \$/mi | Catenary \$/mi | Substation \$ each | Rationale for the delta |
|---|---|---|---|---|---|
| 25 kV AC (`ac_25kv`) | 21.6 mi | 2.2M | 0.50M | 15M | Baseline single‑phase 25 kV catenary and traction substation. |
| 2×25 kV AT (`ac_2x25`) | 69.5 mi | 2.4M | 0.65M | 17M | Autotransformer feed adds a negative feeder + periodic line autotransformers (higher \$/mi) but transmits at an effective 50 kV, so its ~70 mi feed builds fewer, larger substations — the dominant effect — while keeping standard 25 kV rolling stock. |
| 9 kV MVDC (`dc_9kv`) | 8.3 mi | 2.2M | 0.55M | 10M | Lighter, standard‑catenary conductors (lower current than legacy DC) keep catenary \$/mi near the 25 kV AC value. Substations are modular, containerized VSC (active‑front‑end) converter stations — the competitive future form — priced conservatively for the converter premium; its short (~8 mi) feed still forces many to be built, so DC lifecycle cost is highly leveraged on this figure. |

### 6.3 Scenario axes

**Infrastructure‑cost level** — multiplier on the architecture's medium capital
(applied to structure, catenary, and substation alike). The range brackets the wide
project‑to‑project dispersion in reported electrification unit costs (>2× across
projects due to terrain, clearance/double‑stack catenary‑height upgrades, urban
complexity, labor market, and overrun risk): low ≈ favorable open‑country greenfield
with economies of scale; high ≈ difficult terrain / complex corridors near the upper
end of reported OCS costs (Network Rail 2015; IEA 2019):

| Level | Multiplier |
|---|---|
| low  | 0.5× |
| med  | 1.0× |
| high | 2.0× |

**Diesel‑vs‑electricity level** — the energy‑price spread over the horizon; sets
how strongly operating cost favors electrification. The medium row reflects
representative recent U.S. prices (EIA retail on‑highway diesel has run
roughly \$3–4/gal; EIA industrial electricity roughly \$0.07–0.09/kWh); the
low/high rows widen this to bracket unfavorable and favorable futures:

| Level | Diesel \$/gal | Electricity \$/kWh | Interpretation |
|---|---|---|---|
| low  | 2.50 | 0.12 | Cheap diesel, dear power — electrification barely favored |
| med  | 3.50 | 0.09 | Representative recent U.S. prices (EIA range) |
| high | 5.00 | 0.06 | Dear diesel, cheap power — electrification strongly favored |

### 6.4 What the matrix shows

The results behave as economics dictate and expose the core architecture
trade‑off:

- **50 kV AC** builds the fewest substations (0–1, thanks to its 86.5 mi feed) —
  the leanest infrastructure footprint.
- **9 kV HVDC** builds the most (5–11, on its 8.3 mi feed) — the classic DC
  penalty of close substation spacing, partly offset by cheaper rectifier
  stations.
- **25 kV AC** sits in between (≈3–4 substations).
- Raising the **energy spread** (low→high) monotonically increases electrified
  mileage and lifecycle savings; raising **infrastructure cost** (low→high)
  suppresses electrification, and under high infra + low energy spread the NPV
  test electrifies nothing.

The rule‑based (`max_benefit`) sweep in the same script uses a single fixed
(medium 25 kV AC) cost set and instead varies budget, grade emphasis, feed
distance, and minimum‑run length.

### 6.5 Energy columns and on‑board conversion (AC vs DC)

Each cost‑matrix row also reports annual energy for the chosen plan, in TJ
(1 TJ = 10⁶ MJ):

- **dieselTJ** — diesel fuel (LHV) burned on the *unelectrified* remainder of the
  network.
- **elecTJ** — grid‑side electricity consumed on the *electrified* track,
  computed from traction (wheel) energy divided by the full grid→wheel chain
  (substation × on‑board front‑end × inverter+motor × nominal line loss).
- **totalTJ** — their sum. Note this mixes fuel LHV with delivered electrical
  energy, which are not thermodynamically equivalent (diesel traction is ~40%
  efficient, electric ~90%+), so the total falls sharply as electrification rises.

**On‑board conversion is now differentiated for AC vs DC in the energy/opex
model.** Each `ElectrificationConfig` carries an `onboard_efficiency` for the
**front‑end** stage (pantograph → traction DC link) *only*: AC locomotives include
a step‑down transformer (~1.5%) plus rectifier (~2.0%) → ~0.965, while DC needs
only a chopper/filter (~1.0%) → ~0.99.

The scenario runner completes the electricity chain so it is comparable with the
end‑to‑end diesel fuel figure, dividing traction (wheel) energy by the **full
grid → wheel** product, with each factor applied **exactly once** (so nothing is
double‑counted):

$$
\eta_{\text{grid}\to\text{wheel}} = \underbrace{\eta_{\text{sub}}}_{\text{grid}\to\text{catenary}}
\times \underbrace{\eta_{\text{onboard}}}_{\text{front‑end}}
\times \underbrace{\eta_{\text{drive}}}_{\text{inverter}+\text{motor}}
\times \underbrace{\eta_{\text{line}}}_{\text{nominal catenary}}.
$$

| Architecture | substation | on‑board front‑end | inverter+motor | line (nominal) | grid→wheel |
|---|---|---|---|---|---|
| 25 kV AC | 0.99 | 0.965 | 0.93 | 0.97 | 0.8618 |
| 50 kV AC | 0.99 | 0.965 | 0.93 | 0.97 | 0.8618 |
| 9 kV DC  | 0.97 | 0.990 | 0.93 | 0.97 | 0.8653 |

Only the substation and on‑board front‑end differ by architecture; the inverter +
motor (`DRIVETRAIN_EFFICIENCY`, the standalone‑sweep stand‑in for the core sim's
`ElectricDrivetrain`) and the nominal line factor (`NOMINAL_LINE_EFFICIENCY`) are
common. With the transformer+rectifier penalty properly counted, **DC still comes
out slightly ahead** (0.8653 vs 0.8618): its lack of an on‑board transformer more
than offsets its off‑board rectifier‑substation penalty, a net ~0.35‑point edge.

Three caveats: (1) at 9 kV a real high‑power DC‑DC step‑down may be needed (vs a
light filter), which would raise the DC front‑end loss toward 1.5–3% and narrow or
erase this edge; (2) `line_efficiency` here is a flat nominal average — the *full
pipeline* computes catenary line loss **spatially** via `apply_line_losses`
instead, so the two paths should not both be applied to the same energy; (3) the
ALTRIOS *core simulation* still applies a flat `CatenaryInterface.ETA = 0.99` for
all catenary types and its own `ElectricDrivetrain.eta`, so the AC/DC front‑end
split and the drivetrain/line factors above live only in the illustrative
energy/opex layer (`energy_densities`). Because the scenario runner does **not**
invoke the core sim, `CatenaryInterface.ETA` and `ElectricDrivetrain.eta` are never
applied there — there is no overlap or double counting between the two layers.

**Per‑stage loss basis.** The chain decomposes as follows. These per‑component
figures sit within accepted ranges for modern IGBT traction power electronics,
traction transformers, and motors.

| Stage | AC (25 kV) | DC (9 kV) | Where applied | Basis for the range |
|---|---|---|---|---|
| Step‑down transformer | ~1.5% | — (none) | `onboard_efficiency` | Traction transformers ~97–98.5% efficient; the transformer is the well‑known on‑board penalty unique to AC. |
| Line converter / rectifier (AC→DC) | ~2.0% | — | `onboard_efficiency` | Modern active front‑end / 4‑quadrant IGBT converters ~98–99%. |
| DC chopper / input filter | — | ~1.0% | `onboard_efficiency` | Light filter <0.5%; a real high‑power step‑down chopper 1.5–3% (see 9 kV caveat). |
| **Front‑end subtotal (`onboard_efficiency`)** | **~3.5% → 0.965** | **~1.0% → 0.99** | — | Product of the above. |
| Traction inverter + motor | ~7% (0.93) | ~7% (0.93) | `DRIVETRAIN_EFFICIENCY` | IGBT inverters ~97–98.5% × traction motors ~95–96%; common to both. In the full pipeline this is the core sim's `ElectricDrivetrain`, not this factor. |
| Catenary line loss | ~3% (0.97) | ~3% (0.97) | `NOMINAL_LINE_EFFICIENCY` | Nominal average; the full pipeline computes it spatially via `apply_line_losses`. |

The user‑provided ~6% (AC) / ~3.5% (DC) *front‑of‑motor stack totals* are consistent
with the transformer+rectifier+inverter (AC) and chopper+inverter (DC) rows above;
the scenario runner keeps the inverter+motor in a separate factor
(`DRIVETRAIN_EFFICIENCY`) rather than in `onboard_efficiency`, so each stage is
counted once.

### 6.6 Sources and basis of assumptions

The assumptions above are grounded as follows (strongest‑grounded first). No
figure here should be treated as a project cost estimate; the dollar values in
particular are illustrative. Citation keys refer to the BibTeX entries in
[§7 References](#7-references-bibtex).

- **Nominal / minimum catenary voltages** (voltage classes and the
  minimum‑voltage fractions in §5): standardized — EN 50163 / IEC 60850 (railway
  supply voltages) [`en50163`, `iec60850`], which define nominal and permissible
  min/max for 1.5 kV and 3 kV DC and 15 kV / 25 kV AC systems; the ~0.67 DC and
  0.7 AC fractions follow their lowest‑permanent‑voltage limits.
- **Discount rate (7%)**: U.S. OMB Circular A‑94 default real discount rate for
  public‑investment analysis [`omb_a94`]; also `altrios.defaults.DISCOUNT_RATE`.
- **Energy prices**: representative ranges from U.S. EIA published series (retail
  on‑highway diesel [`eia_diesel`]; industrial retail electricity [`eia_elec`]).
  Point values are picked
  within/around those ranges to bracket the decision.
- **Power‑electronics / transformer efficiencies** (§6.5): standard ranges from
  electric‑traction engineering references (e.g. A. Steimel, *Electric Traction —
  Motive Power and Energy Supply* [`steimel2014`]; general IGBT
  converter/transformer efficiency data). The AC transformer + rectifier
  front‑end and the DC chopper/filter front‑end are the physically distinguishing
  stages.
- **Loop impedance defaults** (§2, `ELECTRIFICATION_IMPEDANCE_DEFAULTS`):
  literature per‑km catenary‑loop R/X for the respective geometries ("Route A"
  values), with DC reactance set to zero.
- **Electrification capital costs (§6.2) and their ±2× spread**: order‑of‑magnitude,
  consistent with the ~\$2–5M per route‑mile range for mainline overhead
  electrification and \$10–30M traction substations reported across recent U.S.
  and international programs (see the techno‑economic freight‑electrification
  analysis of Zenith et al. [`zenith2020`], already referenced by
  `altrios.defaults`, and public program cost data such as the Caltrain corridor
  electrification project [`caltrain_pcep`]). These are the **softest**
  assumptions and are the first thing to replace with project‑specific quotes.
- **Diesel energy content** (used for the energy columns): computed from
  `altrios.defaults` (`LHV_DIESEL_KJ_PER_KG`, `RHO_DIESEL_KG_PER_M3`), so it is
  consistent with the rest of ALTRIOS.

---

## 7. References (BibTeX)

> **Verification required.** The entries below were compiled from model knowledge
> to give you a manuscript‑ready starting point. Before submission, **verify each
> field against the primary source** — in particular DOIs, ISBNs, publication
> years/editions, volume/issue/page numbers, and access dates — and confirm the
> current URL for the government/agency data series (deep links change; only
> stable landing pages are given here). Standard numbers (EN/IEC) and agency
> document numbers are the stable, citable identifiers.

```bibtex
@techreport{en50163,
  author      = {{CENELEC}},
  title       = {Railway Applications --- Supply Voltages of Traction Systems},
  number      = {EN 50163:2004+A1:2007},
  institution = {European Committee for Electrotechnical Standardization (CENELEC)},
  address     = {Brussels},
  year        = {2007},
  note        = {Defines nominal and permanent/non-permanent min/max voltages for
                 DC (0.6/0.75/1.5/3 kV) and AC (15/25 kV) traction supplies}
}

@techreport{iec60850,
  author      = {{IEC}},
  title       = {Railway Applications --- Supply Voltages of Traction Systems},
  number      = {IEC 60850},
  edition     = {4.0},
  institution = {International Electrotechnical Commission},
  address     = {Geneva},
  year        = {2014}
}

@techreport{omb_a94,
  author      = {{U.S. Office of Management and Budget}},
  title       = {Guidelines and Discount Rates for Benefit-Cost Analysis of
                 Federal Programs},
  number      = {Circular No. A-94},
  institution = {Executive Office of the President},
  address     = {Washington, DC},
  year        = {1992},
  note        = {7\% real discount rate for public-investment analysis; appendix
                 discount rates updated periodically},
  url         = {https://www.whitehouse.gov/omb/}
}

@misc{eia_diesel,
  author       = {{U.S. Energy Information Administration}},
  title        = {Gasoline and Diesel Fuel Update --- Retail On-Highway Diesel Prices},
  howpublished = {\url{https://www.eia.gov/petroleum/gasdiesel/}},
  institution  = {U.S. Department of Energy},
  year         = {2024},
  note         = {Accessed 2026; verify access date}
}

@misc{eia_elec,
  author       = {{U.S. Energy Information Administration}},
  title        = {Electric Power Monthly --- Average Retail Price of Electricity,
                  Industrial Sector},
  howpublished = {\url{https://www.eia.gov/electricity/monthly/}},
  institution  = {U.S. Department of Energy},
  year         = {2024},
  note         = {Accessed 2026; verify access date}
}

@book{steimel2014,
  author    = {Andreas Steimel},
  title     = {Electric Traction --- Motive Power and Energy Supply:
               Basics and Practical Experience},
  edition   = {2},
  publisher = {DIV Deutscher Industrieverlag},
  address   = {Munich},
  year      = {2014},
  note      = {ISBN and edition to be verified}
}

@article{zenith2020,
  author  = {Zenith, Federico and Isaac, Roger and Hoffrichter, Andreas and
             Thomassen, Magnus S. and M{\o}ller-Holst, Steffen},
  title   = {Techno-economic Analysis of Freight Railway Electrification by
             Overhead Line, Hydrogen and Batteries: Case Studies in Norway and USA},
  journal = {Proceedings of the Institution of Mechanical Engineers, Part F:
             Journal of Rail and Rapid Transit},
  year    = {2020},
  volume  = {234},
  number  = {7},
  pages   = {791--802},
  doi     = {10.1177/0954409719861741},
  note    = {Referenced by altrios.defaults; verify author list, volume/pages, DOI}
}

@misc{caltrain_pcep,
  author       = {{Peninsula Corridor Joint Powers Board (Caltrain)}},
  title        = {Peninsula Corridor Electrification Project (PCEP)},
  howpublished = {\url{https://www.caltrain.com/}},
  year         = {2024},
  note         = {Public program cost data point for mainline overhead
                  electrification (~51 route-miles); verify current figures and URL}
}
```
