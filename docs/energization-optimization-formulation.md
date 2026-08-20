# Optimal Energization Placement: Inputs, Outputs, and Mathematical Formulation

## Abstract

This document specifies the mixed-integer linear program (MILP) implemented in
[python/altrios/optimization/energization_placement.py](../python/altrios/optimization/energization_placement.py)
for the joint selection of (i) track segments to electrify (via catenary or
wireless power transfer) and (ii) transmission substations to construct. The
model maximizes priority-weighted electrified track mileage subject to a capital
budget, a substation feeding/spacing requirement, an optional shared-structure
coupling between parallel tracks, and an optional minimum contiguous-run
constraint. The formulation is deliberately kept linear and, apart from a single
budget knapsack row, near totally unimodular, so that the linear-programming (LP)
relaxation is tight and the program solves efficiently with the HiGHS solver
through the Pyomo APPSI interface.

The same core supports **two electric powertrains**, which the corridor study
optimizes separately (§7): a **dual-mode** locomotive, whose wiring value is
fully separable per site (draw from the wire wherever it exists), and a
**battery-electric locomotive (BEL)**, which additionally requires an
energy-based range limit so its pack never depletes between charges. Both are run
in the **whole-site** reformulation (energizing a structure site wires all its
track pieces), which shrinks the model and closes the optimality gap.

The optimizer is *electrically agnostic*: nominal line voltage, substation power
capacity, and conductor impedance play no role in placement. Those quantities
enter only in the downstream line-loss post-processing
(`apply_line_losses`/`compute_line_losses`); see
[energization-voltage-and-capacity.md](energization-voltage-and-capacity.md).

---

## 1. Inputs

All inputs are carried by the dataclass `EnergizationPlacementInputs`. They fall
into four groups.

### 1.1 Network and candidate assets

| Symbol / field | Meaning |
|---|---|
| `network` | ALTRIOS `Network` with per-link geometry (`headings` lat/lon, `elevs`, `length_meters`, `idx_next`/`idx_next_alt`). Not mutated. |
| `substations : List[Substation]` | Candidate and/or pre-existing substations. Each is located by explicit `(lat, lon)` or by `link_idx` (whence the link midpoint is used). |
| `est_time_nets : List[EstTimeNet]` | Optional. In per-link mode, restricts candidate segments to the links these trains traverse. Unused by structure-site mode. |

A `Substation` contributes a location, a capital cost, and a flag `existing`.
Pre-existing substations are treated as always-built at zero cost; candidate
substations are binary build decisions.

### 1.2 Cost coefficients (`EnergizationOptCosts`)

Let the following per-mile coefficients be resolved with the precedence "specific
overrides general" (`_first_not_none`):

- $c^{\text{en}}$ — marginal electrification cost per track-mile
  (`energization_cost_per_mile`, falling back to `cost_per_mile`);
- $c^{\text{st}}$ — shared corridor **structure** cost per corridor-mile
  (`structure_cost_per_mile`, default $0$);
- $c^{\text{su}}$ — substation interconnection cost per mile
  (`substation_cost_per_mile`, falling back to `cost_per_mile`);
- $F_j$ — lump-sum fixed cost of substation $j$ (`Substation.fixed_cost`,
  falling back to `substation_fixed_cost`);
- $\ell_j$ — substation interconnection distance (`connection_miles`).

### 1.3 Objective and constraint parameters

| Symbol / field | Meaning | Default |
|---|---|---|
| $B$ — `budget` | Capital budget cap ($). `None` $\Rightarrow$ unconstrained | `None` |
| $g^{*}$ — `grade_threshold` | Grade (rise/run) at/above which a segment is "high priority" | `0.01` |
| $w$ — `high_grade_benefit_weight` | Objective multiplier on high-grade segments | `10.0` |
| $D$ — `substation_feed_distance_miles` | Max distance from an energized piece to a built substation | `20.0` |
| `require_high_grade_coverage` | Force (vs. merely prioritize) high-grade electrification | `False` |
| $L_{\min}$ — `min_energized_length_miles` | Minimum contiguous energized run (corridor-miles) | `None` |
| $G_{\max}$ — `max_energized_gap_miles` | Maximum contiguous **un**electrified gap (corridor-miles); *distance* battery-range limit (legacy) | `None` |
| $E$ — `max_energized_gap_kwh` | Usable battery **energy** (kWh) a BEL may spend between charges. Supersedes $G_{\max}$: turns the range limit from distance into drain-energy. BEL only | `None` |
| $\delta$ — `gap_energy_kwh_per_mile_by_link` | Per-link single-locomotive traction **drain** (kWh per corridor-mile). Required whenever $E$ is set | `None` |
| $\gamma$ — `charge_energy_kwh_per_mile_by_link` | Per-link **net charge** banked per energized corridor-mile (kWh/mi); enables charge-credited (Tier 1) and SOC crediting | `None` |
| $\rho$ — `regen_energy_kwh_per_mile_by_link` | Per-link **regen** recovery (kWh per corridor-mile); range then uses net drain $\delta-\rho$ | `None` |
| `battery_soc_state_model` | Use the explicit **SOC recursion** (Tier 2) range model instead of the covering gap rows. BEL only | `False` |
| `electrify_whole_site` | **Whole-site** reformulation: energizing a site wires all its pieces ($x \equiv s$), collapsing the per-piece binaries onto the site indicators | `False` |
| `uniform_power_limit_watts` | Power limit stamped on every energized segment (a constant; not optimized) | `0.0` |

Reasonable values for $D$ and `uniform_power_limit_watts` (and the matching
substation rating and electrical assumptions) can be generated from an
electrification choice — e.g. `dc_9kv`, `ac_25kv`, `ac_50kv` — with the
pre-processing helper `energization.suggest_optimization_inputs`, which sizes the
feeding radius from the voltage-drop limit $d \le V^{2}(1-f)/(|Z|\,P_d)$. See
[energization-voltage-and-capacity.md](energization-voltage-and-capacity.md) §5.

### 1.4 Discretization mode

The model is expressed over corridor **cells** (shared structure sites) and track
**pieces** (individually electrifiable link sub-intervals).

- **Structure-site (corridor) mode** — enabled when `structure_spacing_miles`
  $=\Delta$ is set. Every candidate link is densified (`_densify_links`) into
  uniform along-track pieces of length $\approx \Delta$. Piece midpoints are
  projected to a local equirectangular meter frame and clustered by
  connected components (union–find over a KD-tree radius query) whenever they lie
  within `corridor_width_miles` of one another. Each resulting cluster is one
  **cell** (a shared structure site); parallel tracks and sidings at a location
  therefore share one structure cost. Clustering is purely geometric, requiring
  neither centerline, topology, nor est-time nets.
- **Per-link mode** — the default (`structure_spacing_miles = None`). Each
  candidate link is simultaneously one cell and one whole-link piece.

Orthogonally, **whole-site mode** (`electrify_whole_site=True`) removes the
per-piece electrification decision entirely: energizing a structure site wires
*all* of its pieces, so $x_p$ is eliminated and the site indicator $s_c$ alone
represents electrification (§7.1). Both corridor-study powertrains run in this
mode. The per-piece model of §3–§6 is the finer generalization.

---

## 2. Index sets and derived data

| Set | Description |
|---|---|
| $\mathcal{C}$ | corridor cells (shared structure sites); `model.C` |
| $\mathcal{P}$ | track pieces (electrifiable link sub-intervals); `model.P` |
| $\mathcal{J}$ | candidate substations; `model.J` |

For each piece $p \in \mathcal{P}$ let

- $c(p) \in \mathcal{C}$ — the cell containing $p$ (`piece_cell`);
- $m_p$ — track length of $p$ in miles (`piece_miles`);
- $h_{c}\in\{0,1\}$ — whether cell $c$ is high-grade, i.e. contains a piece whose
  maximum grade (`_link_max_grade`) is $\ge g^{*}$ (`cell_high_grade`);
- $b_p = m_p \cdot \big(w \, h_{c(p)} + (1 - h_{c(p)})\big)$ — the priority-weighted
  benefit of $p$ (`piece_benefit`): $m_p w$ if its cell is high-grade, else $m_p$;
- $\kappa_p = c^{\text{en}} m_p$ — marginal electrification cost (`piece_marginal_cost`);
- $\mathrm{cov}(p)\subseteq\mathcal{J}$ — candidate substations within $D$ miles of
  $p$ (great-circle; `_piece_coverage`);
- $a_p\in\{0,1\}$ — whether $p$ is "auto-covered" by a pre-existing substation
  within $D$ miles (`piece_auto_covered`).

For each cell $c$: $S_c = c^{\text{st}} \, \lambda_c$ is the structure cost
(`struct_cost`), where $\lambda_c$ is the along-track corridor length of the cell
in miles (`cell_len_miles`; in structure-site mode the maximum piece length in the
cell, so parallel tracks do not multiply it). For each substation $j$:
$G_j = F_j + c^{\text{su}} \ell_j$ (`sub_cost`).

Distances use the haversine formula on mean Earth radius
$R_\oplus = 3958.7613$ mi.

### 2.1 Battery-electric range data and variables (BEL variant only)

When a battery range limit is active, the mainline is tiled into contiguous
corridor **buckets** and the following per-bucket data and variables are derived
(energies in kWh; per-corridor-mile densities aggregated to buckets via
`_cell_energy_from_density`). They are inert for the dual-mode variant.

| Symbol | Field / source | Meaning |
|---|---|---|
| $\mathcal{B} = \{0,\dots,n_B-1\}$ | `model.RunBucket` | Ordered contiguous mainline buckets (~`bucket_mi` wide, capped by `_BUCKET_MAX_MILES = 2` mi). |
| $u_b \in \{0,1\}$ | `model.u` | Bucket $b$ energized; coupled to cells by $s_c = u_{b(c)}$ (`run_couple`). |
| $\ell_b$ | `bucket_len_miles` | Corridor length of bucket $b$ (mi). |
| $\delta_b$ | $\delta$ `gap_energy_kwh_per_mile_by_link` $\to$ `cell_drain_kwh` | Traction **drain** over bucket $b$ (kWh). |
| $\gamma_b$ | $\gamma$ `charge_energy_kwh_per_mile_by_link` $\to$ `cell_charge_kwh` | **Net charge** banked if bucket $b$ is energized (kWh). |
| $\rho_b$ | $\rho$ `regen_energy_kwh_per_mile_by_link` $\to$ `cell_regen_kwh` | **Regen** recovery over bucket $b$ (kWh). |
| $\delta^{\text{net}}_b = \delta_b - \rho_b$ | derived (`bucket_net`) | Net drain used by every range family. |
| $E$ | $E$ `max_energized_gap_kwh` | Usable pack energy (kWh) — the range budget. |
| $\mathrm{soc}_b \in [0,E]$ | `model.soc` | SOC-recursion state of charge (Tier 2). |
| $\mathrm{curtail}_b \ge 0$ | `model.curtail` | Charge-saturation slack (Tier 2). |

---

## 3. Decision variables

All variables are binary:

$$
x_p \in \{0,1\}\ \ (\text{electrify piece } p),\qquad
s_c \in \{0,1\}\ \ (\text{install structure in cell } c),\qquad
y_j \in \{0,1\}\ \ (\text{build substation } j).
$$

Pre-existing substations are not variables; they enter only through $a_p$ and
$\mathrm{cov}(\cdot)$.

---

## 4. Objective

Maximize total priority-weighted electrified track mileage. Because each parallel
track is its own piece, wiring both directions of a double-track corridor is
rewarded separately once the shared structure is paid for:

$$
\max \quad Z \;=\; \sum_{p \in \mathcal{P}} b_p \, x_p .
$$

(`model.obj`, `sense = maximize`.)

### 4.1 Pluggable benefit model

The optimizer is agnostic to how the per-piece coefficient $b_p$ is produced: it
enters only as a linear objective weight, so alternative, less grade-specific
value models can be substituted **without any change to the variables,
constraints, or solver**. By default $b_p$ uses the grade heuristic
$b_p = m_p\,(w\,h_{c(p)} + (1 - h_{c(p)}))$. Supplying
`EnergizationPlacementInputs.piece_benefit_fn` — a callable
`PieceBenefitContext -> float` invoked once per piece (via `_compute_piece_benefit`) —
overrides it. This is the seam for future objectives such as **gallons of diesel
displaced** by a dual-mode locomotive, computed per segment from a first-pass
simulation. Because a dual-mode locomotive draws from catenary wherever wire
exists (no battery state-of-charge or contiguity coupling), a separable
per-piece benefit is a physically faithful, and still linear, objective.

### 4.2 Lifecycle cost-minimizing objective (`objective="min_cost"`)

An alternative objective **minimizes total lifecycle cost** = capital plus
discounted operating cost, rather than maximizing a benefit proxy. Let
$\Delta\text{opex}_p$ be the **annual** net operating cost of electrifying piece
$p$ (electricity purchased, including line and conversion loss, minus the diesel a
dual-mode consist would otherwise burn; negative means a saving), supplied via
`EnergizationPlacementInputs.piece_opex_fn`. With discount rate $r$ and horizon
$N$ years (an `EconomicParameters`), the present-value annuity factor is

$$
\mathrm{PVAF}(r,N) = \sum_{t=1}^{N}(1+r)^{-t} = \frac{1-(1+r)^{-N}}{r},
$$

and the objective is

$$
\min \;\; \sum_{c} S_c s_c + \sum_p \kappa_p x_p + \sum_j G_j y_j
\;+\; \mathrm{PVAF}(r,N)\sum_p \Delta\text{opex}_p\, x_p .
$$

The horizon and discount rate enter **only** as the scalar $\mathrm{PVAF}$
multiplying the per-piece annual operating cost, so the program remains the same
linear MILP over the same variables and constraints (C1–C7); no solver change is
needed. Each piece is now decided by a genuine net-present-value test — electrify
iff its lifetime fuel savings outweigh its capital cost — with structure and
substations pulled in through C1/C2 as usual. The optional budget C3 still applies
(a financing cap even when electrification is NPV-positive). This is the objective
used by **both** corridor-study powertrains (§7); it differs only in the operating
term $\Delta\text{opex}_p$ (dual-mode net fuel cost vs. BEL round-trip-loss
saving) and, for the BEL, the added range limit C7.

All traffic/loading/consist assumptions (trains per day, tonnage, number and type
of locomotives) live in $\Delta\text{opex}_p$ — i.e. in how `piece_opex_fn` is
built (see `energization.make_piece_opex_fn`), typically from annual per-link
energy aggregated over first-pass dual-mode simulations. The optimizer stays
agnostic to them. The result reports `total_cost` (capital), `annual_opex_cost`,
`discounted_opex_cost`, and `lifecycle_cost` (the minimized objective).

---

## 5. Constraints

### 5.1 Structure linking (strong, per piece)

A piece may be electrified only if its cell's shared structure is installed:

$$
x_p \le s_{c(p)} \qquad \forall p \in \mathcal{P}. \tag{C1}
$$

This disaggregated form (`model.structure`) is preferred over an aggregated
Big-M row because it yields a tight LP relaxation.

### 5.2 Substation feeding / coverage (strong, disaggregated)

Let $a_p$ mark auto-coverage by an existing substation. Then

$$
\begin{cases}
\text{(skip; unconstrained)} & \text{if } a_p = 1,\\[2pt]
x_p = 0 & \text{if } a_p = 0 \text{ and } \mathrm{cov}(p) = \varnothing,\\[4pt]
x_p \le \displaystyle\sum_{j \in \mathrm{cov}(p)} y_j & \text{otherwise.}
\end{cases}
\tag{C2}
$$

(`model.coverage`.) A piece with no reachable substation is forced off; a piece
reachable only by candidates requires at least one of them to be built.

### 5.3 Capital budget (knapsack)

When $B$ is finite,

$$
\sum_{c \in \mathcal{C}} S_c\, s_c
\;+\; \sum_{p \in \mathcal{P}} \kappa_p\, x_p
\;+\; \sum_{j \in \mathcal{J}} G_j\, y_j
\;\le\; B. \tag{C3}
$$

(`model.budget`.) This single row is the only constraint that breaks total
unimodularity.

### 5.4 Required high-grade coverage (optional)

If `require_high_grade_coverage` is set, every high-grade *mainline* piece is
forced on:

$$
x_p = 1 \qquad \forall p \in \mathcal{R}, \qquad
\mathcal{R} = \{\, p : \text{mainline}(p) \wedge h_{c(p)} = 1 \,\}. \tag{C4}
$$

(`model.require_high_grade`.) Feasibility is checked in advance: if any
$p\in\mathcal{R}$ has $a_p = 0$ and $\mathrm{cov}(p)=\varnothing$, the build raises
`ValueError` identifying the offending links.

### 5.5 Minimum contiguous run length (optional)

When $L_{\min}$ is set (structure-site mode only), the model additionally ties
structure to energization and forbids interior energized runs shorter than
$L_{\min}$ **corridor-miles** (parallel tracks in a site count once).

First, structure is tied to actual electrification:

$$
s_c \le \sum_{p:\, c(p)=c} x_p \qquad \forall c \in \mathcal{C}, \tag{C5}
$$

so that $s_c = 1$ if and only if some track in the cell is energized (with
$s_c = 0$ for empty cells). Combined with (C1), $s_c$ is a faithful indicator of
"corridor site $c$ is energized," even when $c^{\text{st}} = 0$.

Next, cells are ordered into corridor site-sequences. Pieces are chained across
link boundaries via `idx_next` where the connection is unambiguous
(`_build_link_chains`), and each chain is collapsed to its ordered sequence of
distinct cells $\sigma = (\sigma_0, \sigma_1, \dots, \sigma_{n-1})$ with per-cell
lengths $\lambda_{\sigma_k}$. For every interior start index $i \ge 1$, define the
reach
$$
r(i) = \min\Big\{\, k \ge i : \textstyle\sum_{t=i}^{k} \lambda_{\sigma_t} \ge L_{\min} \,\Big\},
$$
skipping starts for which the remaining corridor is too short (those runs terminate
at a corridor end/junction and are exempt). For each such $i$ and every
$k \in \{i+1, \dots, r(i)\}$ the model imposes

$$
s_{\sigma_k} \;\ge\; s_{\sigma_i} - s_{\sigma_{i-1}}. \tag{C6}
$$

(`model.min_run`.) The right-hand side equals $1$ exactly when an energized run
*starts* at $\sigma_i$ (i.e. $\sigma_i$ energized, its predecessor $\sigma_{i-1}$
not); the constraint then forces every site out to the reach $r(i)$ to be
energized as well, guaranteeing a run of at least $L_{\min}$. Runs may begin at any
site boundary, and duplicate triples arising from parallel tracks are removed.

### 5.6 Battery-electric range limit (optional; BEL only)

This constraint family applies **only to the battery-electric variant**; the
dual-mode variant sets none of it (it draws from the wire wherever it exists,
with no state-of-charge coupling, so its placement is fully separable). A BEL
recharges from the wire over energized track and drains its pack over gaps, so
its **usable pack energy** must never be exhausted between charges.

The model tiles the mainline into contiguous corridor buckets $b = 0,\dots,n_B-1$
(§2.1), couples each cell to its bucket $s_c = u_{b(c)}$, and — via the tie
$s_c \le \sum_{p:\,c(p)=c} x_p$ (C5), or $s_c$ directly in whole-site mode — makes
$u_b \in \{0,1\}$ a faithful “bucket $b$ energized” indicator. The limit is imposed
by **one** of four families, in increasing fidelity (`build_model` selects based
on which inputs are supplied):

**(a) Distance gap (legacy).** With $G_{\max}$ = `max_energized_gap_miles` and
bucket lengths $\ell_b$: for every start $i$ let $\rho(i)$ be the smallest index
with $\sum_{t=i}^{\rho(i)} \ell_t > G_{\max}$; require at least one energized
bucket in every minimal over-length window,
$$
\sum_{b=i}^{\rho(i)} u_b \;\ge\; 1 \qquad \forall\, i:\ \rho(i) < n_B. \tag{C7a}
$$
An interval (consecutive-ones) covering matrix — totally unimodular.

**(b) Energy gap.** Identical rows, but windows are bounded by cumulative **net
drain energy** exceeding the usable pack $E$ = `max_energized_gap_kwh` (window
weights $\delta^{\text{net}}_b$ instead of $\ell_b$), so wire is required where
the pack drains fastest (steep/heavy track) rather than on a flat per-mile
budget,
$$
\sum_{b=i}^{\rho(i)} u_b \;\ge\; 1, \qquad \rho(i) = \min\Big\{ j : \textstyle\sum_{t=i}^{j} \delta^{\text{net}}_t > E \Big\}. \tag{C7b}
$$
Still an interval covering (TU). It is the charge-credited form (c) with
$\gamma_b \equiv 0$ and each row relaxed to $\sum u \ge 1$.

**(c) Charge-credited gap (Tier 1).** Drops the “any energized bucket fully
recharges” assumption of (a)/(b). For every contiguous window $S = [i,j]$ whose
cumulative net drain exceeds the pack ($\sum_{b \in S} \delta^{\text{net}}_b - E > 0$),
the energized buckets must bank at least the overrun:
$$
\sum_{b \in S} \max\!\big(\delta^{\text{net}}_b + \gamma_b,\, 0\big)\, u_b
\;\ge\; \sum_{b \in S} \delta^{\text{net}}_b - E. \tag{C7c}
$$
One weighted-covering row per over-length window (windows capped at
$\kappa_{\text{win}} E$ of drain, `_GAP_WINDOW_KWH_FACTOR` $= 2$). All weights are
non-negative over contiguous support, so the LP stays tight though not strictly
TU. Emitting the row for every window ending at each bucket covers the unknown
“last full charge” point, so it is sufficient for SOC feasibility with no
continuous state, no saturation, and no big-$M$.

**(d) SOC recursion (Tier 2) — the model actually run in the study.** Replaces the
covering rows with an explicit per-bucket state of charge that evolves along the
mainline. With continuous $\mathrm{soc}_b \in [0,E]$ and free curtailment slack
$\mathrm{curtail}_b \ge 0$, starting full ($\mathrm{soc}_{-1} = E$):
$$
\mathrm{soc}_b = \mathrm{soc}_{b-1} + \gamma_b\, u_b - \delta^{\text{net}}_b - \mathrm{curtail}_b,
\qquad 0 \le \mathrm{soc}_b \le E,\ \ \mathrm{curtail}_b \ge 0. \tag{C7d}
$$
(`model.soc`, `model.curtail`, `model.soc_balance`.) The lower bound forbids
depletion; the upper bound is charge **saturation**, made exact and LINEAR (no
big-$M$) by the $\mathrm{curtail}_b$ slack — the solver discards exactly the
charge that would overflow the pack because nothing rewards a lower SOC.
Path-ordered and faithful, it subsumes (a)–(c) (their windows are its
relaxations), at the cost of continuous state: no longer TU, but it adds no
integers and no big-$M$, so the relaxation stays tractable.

**Feasibility.** The covering families (a)–(c) can **force** electrification, so a
pre-check verifies every binding window contains at least one coverable bucket
(auto-covered by an existing substation or reachable by a candidate) and
otherwise raises `ValueError` naming the offending corridor miles; the SOC
recursion (d) reports an un-closeable stretch through the solver. In every
variant the battery is assumed full at each corridor end (yard charging), so end
gaps obey the same limit (no exemption — contrast the minimum-run C6, whose
boundary runs are exempt).

---

## 6. Complete model

$$
\begin{aligned}
\max_{x,s,y}\quad & \sum_{p \in \mathcal{P}} b_p\, x_p \\
\text{s.t.}\quad
& x_p \le s_{c(p)} && \forall p \in \mathcal{P} && \text{(C1)}\\
& x_p \le \textstyle\sum_{j \in \mathrm{cov}(p)} y_j \ \ (\text{or } x_p{=}0,\ \text{or skip}) && \forall p \in \mathcal{P} && \text{(C2)}\\
& \textstyle\sum_c S_c s_c + \sum_p \kappa_p x_p + \sum_j G_j y_j \le B && && \text{(C3)}\\
& x_p = 1 && \forall p \in \mathcal{R} && \text{(C4, optional)}\\
& s_c \le \textstyle\sum_{p:\,c(p)=c} x_p && \forall c \in \mathcal{C} && \text{(C5, optional)}\\
& s_{\sigma_k} \ge s_{\sigma_i} - s_{\sigma_{i-1}} && \forall (\sigma_k,\sigma_i,\sigma_{i-1}) && \text{(C6, optional)}\\
& \textstyle\sum_{b\in S}\max(\delta^{\text{net}}_b+\gamma_b,0)\,u_b \ge \textstyle\sum_{b\in S}\delta^{\text{net}}_b - E && \forall\ \text{windows } S && \text{(C7a/b/c, BEL)}\\
& \big[\text{or}\big]\ \ \mathrm{soc}_b = \mathrm{soc}_{b-1}+\gamma_b u_b-\delta^{\text{net}}_b-\mathrm{curtail}_b,\ 0\le\mathrm{soc}_b\le E && \forall b && \text{(C7d, BEL)}\\
& x_p,\, s_c,\, y_j \in \{0,1\}. &&&&
\end{aligned}
$$

This is the general **per-piece** template with the `max_benefit` objective and
all optional rows shown together. The range rows C7 apply to the
**battery-electric** variant only. The corridor study instead runs the
**whole-site**, `min_cost` forms of §7, which specialize this template per
powertrain.

---

## 7. Powertrain-specific formulations (as run in the corridor study)

The corridor study
([applications/catenary_optimization/energization_optimization_corridor.py](../applications/catenary_optimization/energization_optimization_corridor.py))
runs the model **twice** per (architecture, demand) case — once for each electric
powertrain — sharing the same core (C1–C6) but differing in the objective's
operating-cost term and in whether the battery range limit (C7) applies. Both
variants run in **whole-site** mode with the `min_cost` objective and a 5-mile
minimum energized run.

### 7.1 Whole-site reformulation (both variants)

With `electrify_whole_site=True`, energizing a structure site wires **every**
track piece at it, so the per-piece binaries $x_p$ are eliminated and the site
indicator $s_c$ alone represents electrification. Each per-cell coefficient
aggregates its pieces:
$$
\kappa_c = \sum_{p:\,c(p)=c}\kappa_p, \qquad
\Delta\text{opex}_c = \sum_{p:\,c(p)=c}\Delta\text{opex}_p, \qquad
b_c = \sum_{p:\,c(p)=c} b_p,
$$
coverage uses the union of the cell's piece coverages, and the structure-linking
row C1 becomes vacuous. This collapses roughly one binary per short link onto one
per corridor site and tightens the LP relaxation, so the optimality gap closes
instead of stalling near the weak per-piece bound. Results are reported per
energized site.

### 7.2 Dual-mode variant

A dual-mode consist draws from the catenary wherever wire exists and burns diesel
elsewhere — there is **no** battery state or contiguity coupling, so wiring value
is fully **separable** per site. The operating term is the net annual fuel cost of
energizing site $c$,
$$
\Delta\text{opex}^{\text{DM}}_c
= \underbrace{(\text{grid energy incl. line \& conversion loss})\times p_{\text{elec}}}_{\text{electricity purchased}}
\;-\; \underbrace{(\text{gallons displaced})\times p_{\text{diesel}}}_{\text{diesel avoided}}
\quad (\$/\text{yr}),
$$
built by `energization.make_piece_opex_fn` (negative where electrification saves
fuel). The dual-mode program is
$$
\begin{aligned}
\min_{s,\,y}\quad
& \sum_c S_c s_c + \sum_c \kappa_c s_c + \sum_j G_j y_j
  + \mathrm{PVAF}(r,N)\sum_c \Delta\text{opex}^{\text{DM}}_c\, s_c \\
\text{s.t.}\quad
& s_c \le \textstyle\sum_{j\in\mathrm{cov}(c)} y_j\ \ (\text{or } s_c{=}0,\ \text{or skip}) && \forall c && \text{(C2)}\\
& s_c = u_{b(c)},\quad s_{\sigma_k} \ge s_{\sigma_i} - s_{\sigma_{i-1}} && && \text{(C5–C6, min run)}\\
& \textstyle\sum_c S_c s_c + \sum_c \kappa_c s_c + \sum_j G_j y_j \le B && && \text{(C3, optional)}\\
& s_c,\ y_j \in \{0,1\}. &&&&
\end{aligned}
$$
**No range constraint (C7).**

### 7.3 Battery-electric (BEL) variant

A BEL is a pure battery locomotive: it recharges only over energized track and
must never deplete. Its operating term rewards charging over wire by the battery
**round-trip loss avoided** (`energization.make_bel_piece_opex_fn`, a negative
opex = saving), and it adds the range limit §5.6 — in the study, the **SOC
recursion (Tier 2)** (C7d) with net drain, catenary charge crediting, and regen.
The BEL program adds the SOC state to the core:
$$
\begin{aligned}
\min_{s,\,y,\,\mathrm{soc},\,\mathrm{curtail}}\quad
& \sum_c S_c s_c + \sum_c \kappa_c s_c + \sum_j G_j y_j
  + \mathrm{PVAF}(r,N)\sum_c \Delta\text{opex}^{\text{BEL}}_c\, s_c \\
\text{s.t.}\quad
& \text{(C2), (C3 optional), (C5–C6) as in §7.2, with } s_c = u_{b(c)} \\
& \mathrm{soc}_b = \mathrm{soc}_{b-1} + \gamma_b u_b - \delta^{\text{net}}_b - \mathrm{curtail}_b && \forall b && \text{(C7d)}\\
& 0 \le \mathrm{soc}_b \le E,\quad \mathrm{curtail}_b \ge 0,\quad \mathrm{soc}_{-1} = E \\
& s_c,\ y_j \in \{0,1\}. &&&&
\end{aligned}
$$
The covering families C7a/b/c are the lighter alternatives (C7a/b totally
unimodular, C7c a tight weighted covering; selected when
`battery_soc_state_model=False`).

### 7.4 Which constraints apply to which powertrain

| Constraint | Dual-mode | BEL |
|---|:---:|:---:|
| C1 structure linking $x_p \le s_{c(p)}$ (per-piece mode) | ✓ | ✓ |
| C2 substation coverage | ✓ | ✓ |
| C3 capital budget (optional) | ✓ | ✓ |
| C4 required high-grade coverage (optional) | ✓ | ✓ |
| C5 structure ↔ energization tie | ✓ | ✓ |
| C6 minimum energized run (5 mi in the study) | ✓ | ✓ |
| C7 battery range limit (a/b/c covering **or** d SOC recursion) | — | ✓ (d) |
| Objective operating term | net fuel cost $\Delta\text{opex}^{\text{DM}}$ | round-trip-loss saving $\Delta\text{opex}^{\text{BEL}}$ |

The only *structural* difference is C7: the dual-mode layout is unconstrained by
range (separable per site), while the BEL layout must keep every mainline stretch
within the pack's usable energy $E$. Both otherwise share the same sites,
substations, coverage, budget, and minimum-run logic.

---

## 8. Structural / computational properties

- **Tight relaxation.** Constraints (C1), (C2), (C4)–(C6) have $\pm 1$
  coefficients; the corresponding coefficient matrix is (near) totally unimodular,
  so the LP relaxation is integral in the absence of the budget row.
- **Gap constraint stays tight.** The covering range rows (C7a/b) are *covering*
  rows whose support is a **contiguous block** of the ordered bucket variables — a
  consecutive-ones (interval) matrix, which is totally unimodular. Added over the
  1-D mainline together with the interval-structured (C1)/(C5) coupling, they
  introduce **no** fractionality of their own (no big-$M$, no general
  coefficients), so the LP relaxation remains integral apart from the pre-existing
  budget row. The charge-credited form (C7c/Tier 1) uses non-negative weights over
  the same contiguous support (weighted covering — tight, though not strictly TU).
  The **SOC recursion (C7d/Tier 2)** used in the study is *not* TU (it carries
  continuous state) but adds no integers and no big-$M$, so its relaxation stays
  tractable. The practical effect of the range rows is on *feasibility*, not
  relaxation looseness: because they can force electrification they can tighten a
  binding budget (C3) or render the model infeasible (caught by the pre-check /
  solver) — but they do not loosen the bound or trigger a branching blow-up.
- **Single non-TU row.** Only the budget knapsack (C3) couples the decisions with
  general coefficients, which is what makes the MILP nontrivial yet fast.
- **Solver.** HiGHS via Pyomo APPSI (`solve_model`), with optional
  `time_limit_seconds`, relative `mip_gap`, and `tee` (solver streaming).

---

## 9. Outputs (`EnergizationPlacementResult`)

`extract_result` reads the solved model (thresholding each binary at $0.5$) and
returns:

| Field | Definition |
|---|---|
| `network` | Copy of the input network with `cat_power_limits` applied to every energized link sub-interval (overlaps merged via `_merge_intervals`); each interval carries `power_limit_watts = uniform_power_limit_watts` and `district_id = None`. |
| `energized_link_idxs` | Sorted link indices that received any energized interval. |
| `energized_link_intervals` | Mapping `link_idx -> [(start_m, end_m), ...]` of merged energized sub-intervals. |
| `built_substation_ids` | Candidate substations $j$ with $y_j = 1$. |
| `installed_structure_cell_ids` | Cells $c$ with $s_c = 1$. |
| `total_cost` | $\sum_{c: s_c=1} S_c + \sum_{p: x_p=1}\kappa_p + \sum_{j: y_j=1} G_j$ ($). |
| `energized_track_miles` | $\sum_{p: x_p = 1} m_p$ (summed across parallel tracks). |
| `objective_value` | $Z = \sum_p b_p x_p$, the optimal priority-weighted mileage. |
| `status` | HiGHS termination-condition string. |

The convenience wrapper `energization.energize_optimized(...)` constructs
`EnergizationPlacementInputs`, invokes `optimize_energization_placement`, and
returns the same result; the returned network is directly consumable by the train
simulation and by the optional line-loss post-processing.

---

## 10. Modeling assumptions and scope

1. **Electrical neutrality of placement.** Voltage, substation capacity, and
   conductor impedance do not appear above. Placement responds only to geometry
   (length, grade, position), cost, budget, and the distance-based feeding rule.
   Physical power/loss limits are applied afterward by `apply_line_losses`.
2. **Feeding as a covering rule.** The substation feeding requirement is modeled
   as a fixed radius $D$ (great-circle), not a load-flow. It captures spacing, not
   electrical adequacy.
3. **Geometry dependence.** Structure-site mode and all coverage tests require
   lat/lon geometry on links. Per-link mode with est-time-net restriction is
   available where geometry-based corridor clustering is not desired.
4. **Uniform power limit.** `uniform_power_limit_watts` is stamped identically on
   all energized intervals; heterogeneous ratings, if required, are introduced in
   post-processing.
5. **Battery-electric range model.** The BEL range limit is enforced in **energy**,
   not pure distance: buckets carry traction **drain** $\delta_b$, catenary
   **charge** $\gamma_b$, and **regen** $\rho_b$ densities (from a first-pass
   simulation), and the study uses the explicit **SOC recursion** (C7d/Tier 2),
   which tracks pack state along the mainline with charge saturation encoded by a
   linear curtailment slack. It still treats the pack as full at each corridor end
   (yard charging) and aggregates drain/charge to coarse corridor buckets;
   sub-bucket SOC excursions and true instantaneous power limits are resolved by
   the downstream train simulation, not in placement. The simpler distance,
   energy, and charge-credited covering families (C7a/b/c) remain available; C7a/b
   are totally unimodular. The dual-mode variant carries none of this — its
   placement is fully separable per site.