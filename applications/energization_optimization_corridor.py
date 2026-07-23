"""Parameter sweep of energization-placement optimizations on the Spokane-Lethbridge
network (real WGS84 lat/lon geometry).

Runs BOTH optimizer objectives across a grid of settings, printing each result to
the terminal and writing a CSV per sweep (see ``RESULTS_DIR``):

- Rule-based (``objective="max_benefit"``): maximize priority-weighted, grade-
  prioritized track mileage subject to a capital budget. Swept over budget,
  grade emphasis, feed distance, and minimum-run length.

- Cost-based (``objective="min_cost"``): minimize lifecycle cost = capital +
  PVAF(discount_rate, lifespan) * annual operating cost, where operating cost is
  electricity purchased minus diesel a dual-mode consist would otherwise burn.
  Swept over architecture, train-traffic demand case (``central`` mirrors the
  milestone_1.3 Spokane-Lethbridge consist plan; ``high`` adds trains/day to
  illustrate catenary utilization tradeoffs), infrastructure cost, and energy
  prices. The per-link diesel/electricity energy feeding this opex comes from
  FIRST-PASS calibrated dual-mode simulations (all-diesel + all-catenary) over
  the corridor when ``USE_SIM_DERIVED_DENSITIES`` (default) -- so it reflects
  actual speed/grade/mass physics; set it False for fast illustrative
  grade-scaled constants.

Traffic assumptions (trains/day, loading, consist makeup) live in the per-link
annual energy densities feeding ``make_piece_opex_fn``. By default these are
derived from first-pass dual-mode simulations over the corridor (speed-,
grade-, and mass-aware), scaled to the annual train set by trains/day; the
illustrative grade-scaled constants remain as a fast fallback. Each train is
pulled by a
multi-locomotive dual-mode consist (``LOCOS_PER_TRAIN``), so the quantity that
drives catenary power draw / utilization is locomotives/day =
trains/day * LOCOS_PER_TRAIN.

Finally, a SIMULATION STEP runs an actual ALTRIOS train simulation over the
corridor with a chosen plan's optimized catenary applied -- first derating
deliverable pantograph power for AC line losses from the BUILT substation
locations (``en.apply_line_losses``, the piece milestone_2.3b lacked) -- and
writes the simulated diesel + grid-electricity + line-loss energy (vs. an
all-diesel baseline) to ``energization_simulation_results.csv``.

Run:
    python applications/energization_optimization_corridor.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import altrios as alt
from altrios import energization as en
from altrios import defaults as _defaults, utilities as _utils
from altrios.optimization import energization_placement as cp

# ---------------------------------------------------------------------------
# Diagnostics / progress logging
# ---------------------------------------------------------------------------
_T0 = time.perf_counter()


def _log(msg: str) -> None:
    """Print a timestamped (elapsed-seconds) progress line, flushed immediately.

    Flushing matters when the script appears to hang: without it stdout is block-
    buffered when piped to a file, so the last line before a stall may never be
    seen. With it, the most recent ``_log`` line pinpoints where execution is.
    """
    print(f"[{time.perf_counter() - _T0:8.1f}s] {msg}", flush=True)
    sys.stdout.flush()


# Directory for CSV result output (written next to this script by default).
RESULTS_DIR = Path(__file__).resolve().parent


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write ``rows`` (a list of dicts sharing the same keys) to ``path`` as CSV."""
    if not rows:
        _log(f"no rows to write -> {path}")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _log(f"wrote {len(rows)} rows -> {path}")


# Illustrative grid->wheel electric-chain efficiencies applied (once) in
# energy_densities to complete the comparison with end-to-end diesel fuel. These
# are common to AC and DC (only the substation + on-board front-end differ by
# architecture). The traction inverter + motor is the standalone-sweep analog of
# the core sim's ElectricDrivetrain (not run here); the line factor is a NOMINAL
# average -- the full pipeline computes catenary line loss spatially via
# apply_line_losses instead.
DRIVETRAIN_EFFICIENCY = 0.93      # traction inverter (~2.5%) + motor (~4.5%)
NOMINAL_LINE_EFFICIENCY = 0.97    # ~3% average catenary I2R loss

# ---------------------------------------------------------------------------
# Shared network, substations, and capital costs
# ---------------------------------------------------------------------------
_NET_PATH = alt.resources_root() / "networks" / "Spokane_Lethbridge" / "spokane_lethbridge.yaml"
_log(f"loading network from {_NET_PATH} ...")
_t = time.perf_counter()
network = alt.Network.from_file(str(_NET_PATH))
links = network.to_pydict()[1][1:]
total_miles = sum(L["length_meters"] * _utils.MI_PER_M for L in links)
_log(f"network loaded: {len(links)} links, {total_miles:.1f} track miles "
     f"({time.perf_counter() - _t:.1f}s)")


def midpoint(link):
    hs = link.get("headings") or []
    if not hs:
        return None
    m = hs[len(hs) // 2]
    if m.get("lat") is None or m.get("lon") is None:
        return None
    return float(m["lat"]), float(m["lon"])


def max_grade(link) -> float:
    elevs = link.get("elevs") or []
    if len(elevs) < 2:
        return 0.0
    g = 0.0
    for a, b in zip(elevs[:-1], elevs[1:]):
        run = b["offset_meters"] - a["offset_meters"]
        if run:
            g = max(g, abs(b["elev_meters"] - a["elev_meters"]) / run)
    return g


# Order links W->E by longitude and place substations at fractional positions.
pts = [(midpoint(L), L["length_meters"]) for L in links]
pts = [(p, ln) for p, ln in pts if p is not None]
pts.sort(key=lambda t: t[0][1])


def at_fraction(frac: float):
    idx = min(len(pts) - 1, int(frac * (len(pts) - 1)))
    return pts[idx][0]


# Dense candidate substations along the corridor so that each architecture's
# feeding radius (not candidate availability) governs how many get built: DC
# (short feed) builds many, 50 kV AC (long feed) builds few. One pre-existing
# (free) substation anchors the west terminus.
SUBSTATION_MAX_POWER_WATTS = 50.0e6
N_CANDIDATES = 30
substations = [
    cp.Substation(
        id="west_terminus", lat=at_fraction(0.0)[0], lon=at_fraction(0.0)[1],
        existing=True, max_power_watts=SUBSTATION_MAX_POWER_WATTS,
        grid_emissions_region="NWPP",
    ),
]
for _k in range(1, N_CANDIDATES + 1):
    _frac = _k / (N_CANDIDATES + 1)
    _lat, _lon = at_fraction(_frac)
    substations.append(
        cp.Substation(
            id=f"c{_k:02d}", lat=_lat, lon=_lon,
            max_power_watts=SUBSTATION_MAX_POWER_WATTS,
            grid_emissions_region="NWPP" if _frac < 0.5 else "RMPA",
        )
    )

# Precompute per-link max grade once (used to scale traffic energy for cost mode).
GRADE_BY_LINK = {L["idx_curr"]: max_grade(L) for L in links}
# Per-link track length (miles), for splitting energy into electrified (grid) vs
# unelectrified (diesel) portions of each plan.
LINK_MILES = {L["idx_curr"]: L["length_meters"] * _utils.MI_PER_M for L in links}

# Solver diagnostics. ``SOLVER_LOG`` streams the HiGHS/pyomo solver log for every
# run so a hang during the SOLVE phase is visible; ``SOLVER_TIME_LIMIT_SECONDS``
# caps solver wall time so a pathological run terminates (with a time-limit
# status) instead of hanging forever. NOTE: the solver log only covers solving --
# if a run's ``START`` line prints but no solver log follows, the hang is in
# problem SETUP (piece densification + geometric clustering in ``_prepare``),
# which scales with network size.
SOLVER_LOG = True
SOLVER_TIME_LIMIT_SECONDS = 600.0

# Common structure-site (corridor) settings shared by every sweep run. Capital
# costs are passed per-sweep (the cost-based matrix varies them).
COMMON = dict(
    uniform_power_limit_watts=50.0e6,
    structure_spacing_miles=1.0,
    corridor_width_miles=0.05,
    mip_gap=0.01,
    tee=SOLVER_LOG,
    time_limit_seconds=SOLVER_TIME_LIMIT_SECONDS,
)


def run_opt(label: str, **kwargs):
    """Run one optimization with progress + timing diagnostics.

    Prints a ``START`` line before problem setup (so a hang in SETUP -- before any
    solver log appears -- is attributable to this run) and a ``DONE`` line with the
    total wall time and solver status. The solver log itself (when ``tee`` is set)
    streams between the two.
    """
    _log(f"START  {label}")
    t0 = time.perf_counter()
    r = en.energize_optimized(network, substations, **kwargs)
    _log(f"DONE   {label}  ({time.perf_counter() - t0:.1f}s, "
         f"status={str(r.status).split('.')[-1]})")
    return r

# All monetary inputs to the optimizer are expressed in MILLIONS of dollars ($M)
# so the MILP stays well-scaled: raw-dollar budgets (~1e8) and capital costs
# (~1e7) trip HiGHS's "excessively large row bounds" warning and slow the LP
# relaxations. Scaling every money term by the same factor is an exact
# reformulation -- the optimal plan is identical and results come back in $M.
# Energy (gallons/kWh/MJ) and mileage terms are unaffected.
USD_PER_MUSD = 1.0e6

# Rule-based (max_benefit) sweep uses a single fixed (medium 25 kV AC) cost set.
# Costs in $M (see USD_PER_MUSD).
RULE_COSTS = cp.EnergizationOptCosts(
    structure_cost_per_mile=2.5,   # $2.5M/mi shared structure (paid once per site)
    energization_cost_per_mile=0.5,   # $0.5M/mi marginal per electrified track
    substation_fixed_cost=20.0,    # $20M per built substation
)


def energy_densities(trains_per_day, substation_efficiency, onboard_efficiency,
                     *, gal_per_train_mi=3.0, kwh_traction_per_train_mi=33.0,
                     drivetrain_efficiency=DRIVETRAIN_EFFICIENCY,
                     line_efficiency=NOMINAL_LINE_EFFICIENCY,
                     grade_multiplier=40.0):
    """Per-link annual energy densities (per track-mile), grade-scaled.

    Returns ``(diesel_gal_by_link, elec_kwh_by_link)``: annual diesel gallons and
    annual grid-side electricity (kWh) per track-mile. ``kwh_traction_per_train_mi``
    is traction energy at the wheels; the grid-side electricity divides it by the
    FULL grid->wheel electric chain so it is comparable with the end-to-end diesel
    fuel figure:

        grid_eff = substation_efficiency   (grid -> catenary; AC vs DC)
                 * onboard_efficiency       (pantograph -> DC link front-end; AC vs DC)
                 * drivetrain_efficiency     (inverter + traction motor; common)
                 * line_efficiency           (nominal catenary I2R loss; common)

    Each factor is applied exactly ONCE, so there is no double counting. The
    ``drivetrain_efficiency`` is the standalone-sweep stand-in for the core sim's
    ``ElectricDrivetrain`` (the core sim is not run here); ``line_efficiency`` is a
    nominal average -- the full pipeline computes catenary line loss spatially via
    ``apply_line_losses`` instead of this flat factor.
    """
    annual_trains = trains_per_day * 365
    grid_eff = (
        substation_efficiency * onboard_efficiency
        * drivetrain_efficiency * line_efficiency
    )
    diesel_by_link, elec_by_link = {}, {}
    for idx, g in GRADE_BY_LINK.items():
        scale = 1.0 + grade_multiplier * g  # steeper track displaces more diesel
        diesel_by_link[idx] = annual_trains * gal_per_train_mi * scale
        elec_by_link[idx] = annual_trains * kwh_traction_per_train_mi * scale / grid_eff
    return diesel_by_link, elec_by_link


def build_opex_fn(diesel_by_link, elec_by_link, diesel_price, elec_price):
    """Per-piece annual net operating-cost fn ($M/yr) from densities + prices."""
    base = en.make_piece_opex_fn(
        diesel_gallons_per_track_mile_by_link=diesel_by_link,
        electricity_kwh_per_track_mile_by_link=elec_by_link,
        diesel_price_usd_per_gallon=diesel_price,
        electricity_price_usd_per_kwh=elec_price,
    )
    # Optimizer money unit is $M (see USD_PER_MUSD); convert $/yr -> $M/yr so the
    # operating cost is consistent with the capital costs in the objective.
    return lambda ctx: base(ctx) / USD_PER_MUSD


def plan_energy_mj(result, diesel_by_link, elec_by_link):
    """Annual (diesel_MJ, electricity_MJ) consumed under a chosen plan.

    Electrified track runs on grid electricity (incl. substation conversion loss);
    the unelectrified remainder of every link still burns diesel. Electricity is
    grid-side delivered energy; diesel is fuel (LHV) energy -- the two are not
    thermodynamically equivalent, so ``total`` mixes fuel and delivered energy.
    """
    energized = result.energized_link_intervals or {}
    diesel_gal = 0.0
    elec_kwh = 0.0
    for idx, miles in LINK_MILES.items():
        e_mi = 0.0
        if idx in energized:
            e_mi = sum((b - a) for a, b in energized[idx]) * _utils.MI_PER_M
        u_mi = max(miles - e_mi, 0.0)
        diesel_gal += diesel_by_link.get(idx, 0.0) * u_mi
        elec_kwh += elec_by_link.get(idx, 0.0) * e_mi
    # diesel gallons -> MJ: (LHV MJ/kg) * (kg/m3) * (m3/gallon), all from
    # altrios.defaults / altrios.utilities; kWh -> MJ via _utils.KWH_PER_MJ.
    diesel_mj = (
        diesel_gal
        * (_defaults.LHV_DIESEL_KJ_PER_KG / 1000.0)
        * _defaults.RHO_DIESEL_KG_PER_M3
        / (_utils.GALLONS_PER_LITER * _utils.LITER_PER_M3)
    )
    return diesel_mj, elec_kwh / _utils.KWH_PER_MJ


# ---------------------------------------------------------------------------
# Sweep 1: rule-based (max_benefit)
# ---------------------------------------------------------------------------
RULE_CONFIGS = [
    dict(label="baseline",          budget=150, high_grade_benefit_weight=5.0,
         grade_threshold=0.02, substation_feed_distance_miles=30.0,
         min_energized_length_miles=2.0),
    dict(label="tight budget $75M", budget=75,  high_grade_benefit_weight=5.0,
         grade_threshold=0.02, substation_feed_distance_miles=30.0,
         min_energized_length_miles=2.0),
    dict(label="steep-focused",     budget=150, high_grade_benefit_weight=20.0,
         grade_threshold=0.015, substation_feed_distance_miles=30.0,
         min_energized_length_miles=2.0),
    dict(label="long runs 25mi",    budget=150, high_grade_benefit_weight=5.0,
         grade_threshold=0.02, substation_feed_distance_miles=30.0,
         min_energized_length_miles=25.0),
    dict(label="wide feed 60mi",    budget=150, high_grade_benefit_weight=5.0,
         grade_threshold=0.02, substation_feed_distance_miles=60.0,
         min_energized_length_miles=2.0),
]


def run_rule_sweep():
    print("\n" + "=" * 96)
    print("RULE-BASED SWEEP (objective = max_benefit): maximize grade-priority miles s.t. budget")
    print("=" * 96)
    hdr = f"{'config':<20}{'status':<12}{'obj(pw-mi)':>11}{'miles':>8}{'capex$M':>9}{'substations':>26}"
    print(hdr)
    print("-" * 96)
    rows = []
    results = {}
    for cfg in RULE_CONFIGS:
        label = cfg.pop("label")
        r = run_opt(
            f"rule/{label}", objective="max_benefit", costs=RULE_COSTS,
            **COMMON, **cfg,
        )
        cfg["label"] = label  # restore for possible reuse
        results[label] = r
        status = str(r.status).split(".")[-1]
        subs = str(len(r.built_substation_ids))
        print(
            f"{label:<20}{status:<12}{(r.objective_value or 0):>11.1f}"
            f"{(r.energized_track_miles or 0):>8.1f}{(r.total_cost or 0):>9.1f}"
            f"{subs:>26}"
        )
        rows.append({
            "config": label,
            "status": status,
            "objective_pw_miles": r.objective_value or 0,
            "energized_track_miles": r.energized_track_miles or 0,
            "capex_musd": r.total_cost or 0,
            "num_substations": len(r.built_substation_ids),
        })
    _write_csv(RESULTS_DIR / "energization_rule_results.csv", rows)
    return results


# ---------------------------------------------------------------------------
# Sweep 2: cost-based (min_cost)
#   3 architectures x 2 demand cases x (3 infra x 3 energy) = 54 runs
# ---------------------------------------------------------------------------
# Fixed assumptions across the cost matrix (single discount rate + horizon; see
# the .md for cost-assumption sources and ranges). Demand (trains/day) is swept.
DISCOUNT_RATE = 0.07
PROJECT_LIFESPAN_YEARS = 25

# Train traffic (demand) cases for the cost sweep. "central" mirrors the
# milestone_1.3 Spokane-Lethbridge dual-mode consist plan; "high" stresses
# catenary utilization with more trains/day. Every train is pulled by a
# multi-locomotive consist, so the quantity that drives catenary power draw /
# utilization is locomotives/day = trains/day * LOCOS_PER_TRAIN. The per-train
# energy densities already sum over a consist's locomotives, so LOCOS_PER_TRAIN
# is reported (not multiplied back into energy) to avoid double counting.
# NOTE: set these to the actual milestone_1.3 consist-plan trains/day and consist
# size; the values here are representative placeholders.
LOCOS_PER_TRAIN = 3
DEMAND_LEVELS = {
    "central": 12,  # trains/day -- representative milestone_1.3 corridor traffic
    "high":    24,  # trains/day -- utilization-stress case
}

# Architecture -> (short name, medium capital costs $). Medium values are
# order-of-magnitude figures consistent with published overhead-electrification
# cost ranges (~$2-5M/route-mile catenary; traction/rectifier substations
# ~$10-30M); see docs/energization-voltage-and-capacity.md.
ARCH_MEDIUM_COSTS = {
    # key into en.suggest_optimization_inputs -> feed distance, voltage, efficiency
    # costs in $M (see USD_PER_MUSD)
    "ac_25kv": dict(structure=2.5, energization=0.50, substation=20.0),
    "ac_50kv": dict(structure=2.7, energization=0.55, substation=25.0),
    "dc_9kv":  dict(structure=2.6, energization=0.70, substation=12.0),
}

# Infrastructure-cost scenario: multiplier on the architecture's medium capital.
# Generous low/high range around the medium point estimate.
INFRA_LEVELS = {"low": 0.5, "med": 1.0, "high": 2.0}

# Diesel-vs-electricity scenario over the project lifespan: (diesel $/gal,
# electricity $/kWh). "low" barely favors electrification (cheap diesel, dear
# power); "high" strongly favors it (dear diesel, cheap power).
ENERGY_LEVELS = {
    "low":  dict(diesel=2.50, elec=0.12),
    "med":  dict(diesel=3.50, elec=0.09),
    "high": dict(diesel=5.00, elec=0.06),
}


def run_cost_sweep():
    econ = cp.EconomicParameters(
        project_lifespan_years=PROJECT_LIFESPAN_YEARS, discount_rate=DISCOUNT_RATE
    )
    print("\n" + "=" * 118)
    print("COST-BASED SWEEP (objective = min_cost): minimize capital + discounted operating cost")
    demand_desc = " | ".join(   
        f"{name}={tpd}tr/day ({tpd * LOCOS_PER_TRAIN}loco/day)"
        for name, tpd in DEMAND_LEVELS.items()
    )
    print(f"  horizon {PROJECT_LIFESPAN_YEARS} yr | discount {DISCOUNT_RATE:.0%} | "
          f"PVAF {econ.pvaf():.2f} | {LOCOS_PER_TRAIN} locos/train | demand: {demand_desc}")
    print("  energy columns are annual TJ (1 TJ = 1e6 MJ): diesel = fuel LHV on "
          "unelectrified track; elec = grid-side on electrified track")
    print("  (elec chain = substation x on-board front-end x inverter+motor x "
          "nominal line loss; each applied once)")
    print("=" * 118)
    hdr = (f"{'infra':<6}{'energy':<7}{'miles':>7}{'capex$M':>9}{'opex$M/yr':>11}"
           f"{'lifecyc$M':>11}{'#subs':>6}{'dieselTJ':>10}{'elecTJ':>9}{'totalTJ':>9}")
    rows = []
    # First-pass calibrated dual-mode sims give the per-link per-train energy that
    # feeds the min_cost opex (run once, then scaled per (arch, demand)); else
    # fall back to the illustrative grade-scaled constants.
    sim_gal_mi, sim_kwh_mi = (
        first_pass_densities() if USE_SIM_DERIVED_DENSITIES else (None, None)
    )
    for arch in ("ac_25kv", "ac_50kv", "dc_9kv"):
        sug = en.suggest_optimization_inputs(arch)
        med = ARCH_MEDIUM_COSTS[arch]
        for demand_lvl, trains_per_day in DEMAND_LEVELS.items():
            locos_per_day = trains_per_day * LOCOS_PER_TRAIN
            if USE_SIM_DERIVED_DENSITIES:
                # Sim-derived annual per-track-mile densities. Drivetrain +
                # on-board losses are already in the simulated pantograph draw,
                # so only the substation (grid->catenary) efficiency is applied
                # to convert to grid-side electricity purchased.
                annual = trains_per_day * 365
                diesel_by_link = {li: annual * v for li, v in sim_gal_mi.items()}
                elec_by_link = {
                    li: annual * sim_kwh_mi[li] / sug.substation_efficiency
                    for li in sim_kwh_mi
                }
            else:
                # Illustrative grade-scaled densities (full efficiency chain
                # applied inside energy_densities).
                diesel_by_link, elec_by_link = energy_densities(
                    trains_per_day, sug.substation_efficiency, sug.onboard_efficiency
                )
            print("-" * 118)
            print(f"--- {arch} | demand={demand_lvl}: {trains_per_day} trains/day "
                  f"x {LOCOS_PER_TRAIN} locos = {locos_per_day} locos/day | "
                  f"feed {sug.substation_feed_distance_miles:.1f} mi, "
                  f"sub_eff {sug.substation_efficiency}, onboard_eff {sug.onboard_efficiency} ---")
            print(hdr)
            for infra_lvl, mult in INFRA_LEVELS.items():
                these_costs = cp.EnergizationOptCosts(
                    structure_cost_per_mile=med["structure"] * mult,
                    energization_cost_per_mile=med["energization"] * mult,
                    substation_fixed_cost=med["substation"] * mult,
                )
                for energy_lvl, e in ENERGY_LEVELS.items():
                    opex_fn = build_opex_fn(
                        diesel_by_link, elec_by_link, e["diesel"], e["elec"]
                    )
                    r = run_opt(
                        f"cost/{arch}/demand={demand_lvl}/infra={infra_lvl}/energy={energy_lvl}",
                        objective="min_cost",
                        economics=econ, piece_opex_fn=opex_fn, costs=these_costs,
                        substation_feed_distance_miles=sug.substation_feed_distance_miles,
                        min_energized_length_miles=2.0, **COMMON,
                    )
                    diesel_mj, elec_mj = plan_energy_mj(r, diesel_by_link, elec_by_link)
                    print(
                        f"{infra_lvl:<6}{energy_lvl:<7}"
                        f"{(r.energized_track_miles or 0):>7.0f}{(r.total_cost or 0):>9.1f}"
                        f"{(r.annual_opex_cost or 0):>11.2f}"
                        f"{(r.lifecycle_cost or 0):>11.1f}"
                        f"{len(r.built_substation_ids):>6}"
                        f"{diesel_mj/1e6:>10.1f}{elec_mj/1e6:>9.1f}"
                        f"{(diesel_mj + elec_mj)/1e6:>9.1f}"
                    )
                    rows.append({
                        "architecture": arch,
                        "demand_level": demand_lvl,
                        "trains_per_day": trains_per_day,
                        "locos_per_day": locos_per_day,
                        "infra_level": infra_lvl,
                        "energy_level": energy_lvl,
                        "status": str(r.status).split(".")[-1],
                        "energized_track_miles": r.energized_track_miles or 0,
                        "capex_musd": r.total_cost or 0,
                        "annual_opex_musd_per_yr": r.annual_opex_cost or 0,
                        "lifecycle_musd": r.lifecycle_cost or 0,
                        "num_substations": len(r.built_substation_ids),
                        "diesel_tj": diesel_mj / 1e6,
                        "elec_tj": elec_mj / 1e6,
                        "total_tj": (diesel_mj + elec_mj) / 1e6,
                    })
    _write_csv(RESULTS_DIR / "energization_cost_results.csv", rows)
    print("\n(negative lifecycle cost => electrification is net cost-saving over the horizon)")


# ---------------------------------------------------------------------------
# Simulation step: actual train-sim energy on the optimized catenary network
# ---------------------------------------------------------------------------
# After optimizing placement, run a real ALTRIOS train simulation over the
# corridor with the chosen plan's catenary applied, so energy consumed is the
# SIMULATED diesel + grid electricity (not the illustrative densities above).
# Unlike milestone_2.3b, the catenary power limits are first DERATED for AC line
# losses via ``en.apply_line_losses`` using the BUILT substation locations, so
# deliverable pantograph power (and thus the fuel/electric split) reflects
# distance to the nearest feeding substation.
RUN_SIMULATION = True
# When True, the min_cost opex per-link energy comes from first-pass calibrated
# dual-mode simulations (all-diesel + all-catenary) instead of the illustrative
# grade-scaled constants. Adds two full-corridor sims per cost sweep.
USE_SIM_DERIVED_DENSITIES = True
SIM_SAVE_INTERVAL = 60
SIM_ORIGIN, SIM_DESTINATION = "Spokane", "Lethbridge"
SIM_LOCATIONS_CSV = (alt.resources_root() / "networks" / "Spokane_Lethbridge"
                     / "spokane_lethbridge_locations.csv")
SIM_N_CARS = {"Manifest_Loaded": 75, "Manifest_Empty": 75}
# Dual-mode consist sized to haul the modeled train over the ruling grade
# (matches milestone_2.3b's working setup); distinct from LOCOS_PER_TRAIN, which
# is the cost-sweep's demand-accounting figure.
SIM_N_LOCOS = 10
# Proof-of-concept: also simulate a battery-electric consist equipped with a
# pantograph (BEL + catenary). On energized track the BEL draws catenary power
# for traction first, then charges its battery with any leftover capacity; off
# catenary it runs on the battery. Single-train PoC (robustify later). Toggle off
# with RUN_BEL_POC = False. SIM_BEL_CAT_PWR_MAX_WATTS is the per-locomotive
# maximum deliverable pantograph power.
RUN_BEL_POC = True
SIM_BEL_CAT_PWR_MAX_WATTS = 5.0e6
# Electrical assumptions for line losses (25 kV AC medium; matches RULE_COSTS).
SIM_SYSTEM_VOLTAGE_VOLTS = 25_000.0
SIM_ELECTRIFICATION_TYPE = "ac_25kv_1x25"
# Speed-limit calibration from milestone_1.3 / milestone_2.3b: cap the link speed
# limits then scale them so simulated travel speed matches OBSERVED corridor
# speeds (uncalibrated limits give unrealistically fast runs, which distorts the
# fuel/electric split). 0.595 is the calibrated scaler; 40 mph is the pre-scaling
# cap used during that calibration. Set SPEED_CALIBRATION_MAX_MPH = None to skip
# the cap and apply the scaler alone.
SPEED_CALIBRATION_MAX_MPH = 40.0
SPEED_LIMIT_SCALER = 0.595


def _calibrate_speeds(net):
    """Return a copy of ``net`` with calibrated speed limits (cap, then scale)."""
    d = net.to_pydict()
    if SPEED_CALIBRATION_MAX_MPH is not None:
        d = _utils.set_network_max_speed(d, max_speed_mph=SPEED_CALIBRATION_MAX_MPH)
    d = _utils.scale_network_speed_limits(d, SPEED_LIMIT_SCALER)
    return alt.Network.from_pydict(d)


def _build_and_walk(sim_network, location_map, label, loco_factory=None):
    """Build the standard train, dispatch, and walk the timed path over
    ``sim_network``; return the solved SpeedLimitTrainSim.

    ``loco_factory`` is a zero-arg callable returning one ``alt.Locomotive``; the
    consist is ``SIM_N_LOCOS`` copies of it. It defaults to
    ``default_dual_mode_loco``: on catenary a dual-mode loco draws from the grid
    rather than a battery, so per-link catenary energy reflects genuine grid draw
    (a BEL could hide draw in its battery). Pass a BEL-with-pantograph factory for
    the battery-electric proof of concept.
    """
    if loco_factory is None:
        loco_factory = alt.Locomotive.default_dual_mode_loco
    rvl = alt.RailVehicle.from_file(alt.resources_root() / "rolling_stock/Manifest_Loaded.yaml")
    rve = alt.RailVehicle.from_file(alt.resources_root() / "rolling_stock/Manifest_Empty.yaml")
    train_config = alt.TrainConfig(
        rail_vehicles=[rvl, rve], n_cars_by_type=SIM_N_CARS,
        train_length_meters=None, train_mass_kilograms=None,
    )
    loco = loco_factory()
    loco.set_save_interval(SIM_SAVE_INTERVAL)
    loco_con = alt.Consist([loco.copy() for _ in range(SIM_N_LOCOS)], SIM_SAVE_INTERVAL)
    tsb = alt.TrainSimBuilder(
        train_id="0", origin_id=SIM_ORIGIN, destination_id=SIM_DESTINATION,
        train_config=train_config, loco_con=loco_con,
    )
    ts = tsb.make_speed_limit_train_sim(location_map=location_map, save_interval=SIM_SAVE_INTERVAL)
    ts.set_save_interval(SIM_SAVE_INTERVAL)
    est_time_net, _consist = alt.make_est_times(ts, sim_network)
    timed_path = next(iter(alt.run_dispatch(
        sim_network, alt.SpeedLimitTrainSimVec([ts]), [est_time_net], False, False)))
    ts.walk_timed_path(network=sim_network, timed_path=timed_path)
    return ts


def _bel_with_cat_loco():
    """Factory for a battery-electric locomotive equipped with a pantograph."""
    return alt.Locomotive.default_battery_electric_loco_with_cat(
        SIM_BEL_CAT_PWR_MAX_WATTS
    )


def _loco_cat_int(loco: dict):
    """Return a locomotive's catenary-interface dict, or ``None``.

    Works for both dual-mode locomotives (always have ``cat_int``) and
    battery-electric locomotives (``cat_int`` is optional / may be ``None``).
    """
    loco_type = loco["loco_type"]
    for key in ("DualModeLoco", "BatteryElectricLoco"):
        sub = loco_type.get(key)
        if sub is not None:
            return sub.get("cat_int")
    return None


def _simulate_once(sim_network, location_map, label, loco_factory=None):
    """Run one SpeedLimitTrainSim over ``sim_network`` and return actual energy.

    Returns a dict with diesel fuel J, grid electricity-in J, catenary line-loss
    J, net battery-discharge J, and travel time (hr). Diesel is the built-in
    consist fuel getter (zero for a battery-electric consist); grid and line-loss
    energy come from each locomotive's catenary interface cumulative energy
    history (dual-mode always, battery-electric when a pantograph is fitted).
    """
    ts = _build_and_walk(sim_network, location_map, label, loco_factory)
    d = ts.to_pydict()
    times = d["history"]["time_seconds"]
    dts = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
    # Pantograph-side electrical energy drawn by the consist and the ADDITIONAL
    # resistive line-loss energy the substation must supply (grid-side draw =
    # pantograph + line loss). Prefer the exact cumulative energy history; fall
    # back to integrating power if an older build lacks the cumulative fields.
    pantograph_j = line_loss_j = 0.0
    battery_j = 0.0
    for loco in d["loco_con"]["loco_vec"]:
        ch = _loco_cat_int(loco)
        if ch is not None:
            ch = ch["history"]
            e_in = ch.get("energy_elec_in_joules")
            e_ll = ch.get("energy_line_loss_joules")
            if e_in:
                pantograph_j += e_in[-1]
                line_loss_j += e_ll[-1] if e_ll else 0.0
            else:  # fallback: left-rectangle integration of power over time steps
                grid = ch.get("pwr_elec_in_watts") or []
                loss = ch.get("pwr_line_loss_watts") or [0.0] * len(grid)
                pantograph_j += sum(p * dt for p, dt in zip(grid[:-1], dts))
                line_loss_j += sum(p * dt for p, dt in zip(loss[:-1], dts))
        # Net chemical energy discharged from the battery (positive) / charged
        # (negative); makes the battery's role in the energy balance visible for
        # the BEL proof of concept.
        bel = loco["loco_type"].get("BatteryElectricLoco")
        if bel is not None:
            res_hist = bel["res"]["history"]
            e_chem = res_hist.get("energy_out_chemical_joules")
            if e_chem:
                battery_j += e_chem[-1]
    grid_j = pantograph_j + line_loss_j  # substation-side (electricity purchased)
    diesel_j = ts.get_energy_fuel_joules(False)
    _log(f"  sim[{label}]: diesel {diesel_j/1e9:.2f} GJ, grid {grid_j/1e9:.2f} GJ "
         f"(pantograph {pantograph_j/1e9:.2f} + line-loss {line_loss_j/1e9:.3f}), "
         f"battery {battery_j/1e9:.2f} GJ, {times[-1]/3600:.2f} hr")
    return {"diesel_j": diesel_j, "pantograph_elec_j": pantograph_j,
            "line_loss_j": line_loss_j, "grid_elec_j": grid_j,
            "battery_j": battery_j, "ts": ts,
            "travel_hr": times[-1] / 3600.0}


def _per_link_energy_from_sim(ts):
    """Aggregate per-link consist fuel (J) and pantograph electricity (J) from a
    walked sim, attributing each time step's energy to the link under the train
    head (``link_idx_front``). Returns ``(diesel_j_by_link, elec_j_by_link)``.
    """
    d = ts.to_pydict()
    hist = d["history"]
    times = hist["time_seconds"]
    front = hist["link_idx_front"]
    n = len(times)
    dts = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
    fuel_w = [0.0] * n
    grid_w = [0.0] * n
    for loco in d["loco_con"]["loco_vec"]:
        dm = loco["loco_type"].get("DualModeLoco")
        if dm is None:
            continue
        fh = dm["fc"]["history"].get("pwr_fuel_watts") or []
        ch = dm["cat_int"]["history"].get("pwr_elec_in_watts") or []
        for i in range(n):
            if i < len(fh):
                fuel_w[i] += fh[i]
            if i < len(ch):
                grid_w[i] += ch[i]
    diesel_j_by_link = {}
    elec_j_by_link = {}
    for i, dt in enumerate(dts):
        li = front[i]
        if not li:  # skip 0 / None (off-network / pre-departure steps)
            continue
        diesel_j_by_link[li] = diesel_j_by_link.get(li, 0.0) + fuel_w[i] * dt
        elec_j_by_link[li] = elec_j_by_link.get(li, 0.0) + grid_w[i] * dt
    return diesel_j_by_link, elec_j_by_link


def first_pass_densities():
    """Run first-pass calibrated dual-mode sims to get per-link, per-train energy.

    Two runs over the calibrated corridor: an all-diesel run (no catenary) for
    the diesel a consist burns per link, and an all-catenary run (network fully
    energized) for the pantograph electricity per link if that link were
    electrified. Returns ``(diesel_gal_per_track_mile_by_link,
    pantograph_kwh_per_track_mile_by_link)`` -- intrinsic per-train densities.
    Substation efficiency and placement-dependent line losses are applied later
    (per architecture / in the confirmation sim), not here.
    """
    _log("FIRST-PASS calibrated sims (dual-mode: all-diesel + all-catenary) ...")
    location_map = alt.import_locations(str(SIM_LOCATIONS_CSV))
    net_diesel = _calibrate_speeds(network)
    net_cat = _calibrate_speeds(en.energize_uniform(
        network, uniform_power_limit_watts=COMMON["uniform_power_limit_watts"]))
    ts_die = _build_and_walk(net_diesel, location_map, "first-pass/all-diesel")
    ts_cat = _build_and_walk(net_cat, location_map, "first-pass/all-catenary")
    diesel_j_by_link, _e = _per_link_energy_from_sim(ts_die)   # fuel from diesel run
    _d, elec_j_by_link = _per_link_energy_from_sim(ts_cat)     # grid from catenary run

    # J/gallon diesel (LHV * density * m3/gallon), from altrios defaults/utils.
    j_per_gal = ((_defaults.LHV_DIESEL_KJ_PER_KG * 1000.0)
                 * _defaults.RHO_DIESEL_KG_PER_M3
                 / (_utils.GALLONS_PER_LITER * _utils.LITER_PER_M3))
    diesel_gal_per_mi = {}
    elec_kwh_per_mi = {}
    for li, miles in LINK_MILES.items():
        if miles <= 0:
            continue
        diesel_gal_per_mi[li] = (diesel_j_by_link.get(li, 0.0) / j_per_gal) / miles
        elec_kwh_per_mi[li] = (elec_j_by_link.get(li, 0.0) / 3.6e6) / miles
    _log(f"  first-pass per-train totals: "
         f"{sum(diesel_j_by_link.values())/j_per_gal:.0f} gal diesel (all-diesel), "
         f"{sum(elec_j_by_link.values())/3.6e6:.0f} kWh pantograph (all-catenary)")
    return diesel_gal_per_mi, elec_kwh_per_mi


def _write_bel_draw_series(ts, path: Path) -> None:
    """Emit the BEL catenary draw time series to ``path`` as CSV.

    One row per saved time step with the train position (time, corridor offset,
    link under the head) and the consist-summed catenary/battery power split:

    - ``grid_draw_w``: total power drawn from the catenary (pantograph, incl. any
      routed into the battery). Substation-side draw also includes line loss.
    - ``cat_out_w``: catenary power delivered on-board (after interface losses).
    - ``traction_w``: electrical propulsion power at the drivetrain input.
    - ``cat_to_prop_w`` / ``cat_to_aux_w`` / ``cat_to_charge_w``: catenary power
      split into traction, auxiliaries, then battery charging (same priority
      order the model uses), reconstructed from existing histories.
    - ``batt_discharge_w``: traction the catenary did not cover (battery supplied).

    The split is a per-step reconstruction from existing histories (no extra core
    state); it is exact except at simultaneous power limits.
    """
    d = ts.to_pydict()
    hist = d["history"]
    times = hist["time_seconds"]
    offsets = hist.get("offset_back_meters") or hist.get("offset_meters") or []
    front = hist.get("link_idx_front") or []
    n = len(times)
    grid_w = [0.0] * n
    line_loss_w = [0.0] * n
    cat_out_w = [0.0] * n
    trac_w = [0.0] * n
    aux_w = [0.0] * n
    for loco in d["loco_con"]["loco_vec"]:
        ci = _loco_cat_int(loco)
        if ci is not None:
            ch = ci["history"]
            gh = ch.get("pwr_elec_in_watts") or []
            oh = ch.get("pwr_elec_out_watts") or []
            lh = ch.get("pwr_line_loss_watts") or []
            for i in range(min(n, len(gh))):
                grid_w[i] += gh[i]
            for i in range(min(n, len(oh))):
                cat_out_w[i] += oh[i]
            for i in range(min(n, len(lh))):
                line_loss_w[i] += lh[i]
        bel = loco["loco_type"].get("BatteryElectricLoco")
        if bel is not None:
            eh = bel["edrv"]["history"].get("pwr_elec_prop_in_watts") or []
            for i in range(min(n, len(eh))):
                trac_w[i] += max(eh[i], 0.0)
            ah = (loco.get("history") or {}).get("pwr_aux_watts") or []
            for i in range(min(n, len(ah))):
                aux_w[i] += ah[i]
    rows = []
    for i in range(n):
        # Allocate delivered catenary power in the same order the model does:
        # traction first, then aux, then the remainder charges the battery.
        cat_to_prop = min(cat_out_w[i], trac_w[i])
        cat_to_aux = min(cat_out_w[i] - cat_to_prop, max(aux_w[i], 0.0))
        cat_to_charge = max(cat_out_w[i] - cat_to_prop - cat_to_aux, 0.0)
        batt_discharge = max(trac_w[i] - cat_to_prop, 0.0)
        rows.append({
            "time_s": times[i],
            "offset_m": offsets[i] if i < len(offsets) else None,
            "link_idx_front": front[i] if i < len(front) else None,
            "grid_draw_w": grid_w[i],
            "line_loss_w": line_loss_w[i],
            "cat_out_w": cat_out_w[i],
            "traction_w": trac_w[i],
            "cat_to_prop_w": cat_to_prop,
            "cat_to_aux_w": cat_to_aux,
            "cat_to_charge_w": cat_to_charge,
            "batt_discharge_w": batt_discharge,
        })
    _write_csv(path, rows)


def run_simulation_step(plan_result, plan_label, built_subs):
    """Simulate ``plan_result``'s catenary plan (with AC line losses) plus an
    all-diesel baseline, and write the actual simulated energy to CSV."""
    _log(f"SIMULATION STEP on plan '{plan_label}': {len(built_subs)} feeding "
         f"substations, {plan_result.energized_track_miles or 0:.0f} energized "
         f"track-miles")
    location_map = alt.import_locations(str(SIM_LOCATIONS_CSV))

    # Derate the optimized catenary for AC line losses from substation locations.
    net_cat, _meta = en.apply_line_losses(
        plan_result.network, built_subs,
        system_voltage_volts=SIM_SYSTEM_VOLTAGE_VOLTS,
        electrification_type=SIM_ELECTRIFICATION_TYPE,
        base_power_limit_watts=COMMON["uniform_power_limit_watts"],
    )
    # Apply the calibrated speed limits (cap + scaler) so simulated travel speed
    # -- and thus the fuel/electric split -- matches observed corridor speeds.
    net_cat = _calibrate_speeds(net_cat)
    net_diesel = _calibrate_speeds(network)

    e_cat = _simulate_once(net_cat, location_map, f"{plan_label}/with-catenary")
    e_die = _simulate_once(net_diesel, location_map, f"{plan_label}/all-diesel")
    rows = [
        {"scenario": "with_catenary", "plan": plan_label,
         "diesel_gj": e_cat["diesel_j"] / 1e9,
         "grid_elec_gj": e_cat["grid_elec_j"] / 1e9,
         "pantograph_elec_gj": e_cat["pantograph_elec_j"] / 1e9,
         "line_loss_gj": e_cat["line_loss_j"] / 1e9,
         "battery_gj": e_cat["battery_j"] / 1e9,
         "travel_hr": e_cat["travel_hr"]},
        {"scenario": "all_diesel", "plan": plan_label,
         "diesel_gj": e_die["diesel_j"] / 1e9,
         "grid_elec_gj": 0.0, "pantograph_elec_gj": 0.0, "line_loss_gj": 0.0,
         "battery_gj": e_die["battery_j"] / 1e9,
         "travel_hr": e_die["travel_hr"]},
        {"scenario": "delta_electrification", "plan": plan_label,
         "diesel_gj": (e_cat["diesel_j"] - e_die["diesel_j"]) / 1e9,
         "grid_elec_gj": e_cat["grid_elec_j"] / 1e9,
         "pantograph_elec_gj": e_cat["pantograph_elec_j"] / 1e9,
         "line_loss_gj": e_cat["line_loss_j"] / 1e9,
         "battery_gj": e_cat["battery_j"] / 1e9,
         "travel_hr": e_cat["travel_hr"] - e_die["travel_hr"]},
    ]

    # Battery-electric + pantograph proof of concept: single-train run of a BEL
    # consist with pantographs over the same energized network. On catenary it
    # draws grid power for traction (and charges the battery with surplus); off
    # catenary it runs on the battery. Emits the when/where catenary-draw time
    # series so grid draw can be located along the corridor even when some of it
    # charges the battery.
    if RUN_BEL_POC:
        try:
            e_bel = _simulate_once(
                net_cat, location_map, f"{plan_label}/with-catenary-bel",
                loco_factory=_bel_with_cat_loco,
            )
            rows.append({
                "scenario": "with_catenary_bel", "plan": plan_label,
                "diesel_gj": e_bel["diesel_j"] / 1e9,
                "grid_elec_gj": e_bel["grid_elec_j"] / 1e9,
                "pantograph_elec_gj": e_bel["pantograph_elec_j"] / 1e9,
                "line_loss_gj": e_bel["line_loss_j"] / 1e9,
                "battery_gj": e_bel["battery_j"] / 1e9,
                "travel_hr": e_bel["travel_hr"]})
            _write_bel_draw_series(
                e_bel["ts"],
                RESULTS_DIR / "energization_bel_draw_series.csv",
            )
        except Exception as exc:  # noqa: BLE001 - PoC must not sink the main outputs
            _log(f"BEL proof-of-concept FAILED: {exc}")

    _write_csv(RESULTS_DIR / "energization_simulation_results.csv", rows)


if __name__ == "__main__":
    _log(f"Network: {len(links)} links, {total_miles:.1f} track miles")
    _log(f"solver log={'on' if SOLVER_LOG else 'off'}, "
         f"time limit={SOLVER_TIME_LIMIT_SECONDS:.0f}s per run")
    _t = time.perf_counter()
    rule_results = run_rule_sweep()
    _log(f"rule sweep complete ({time.perf_counter() - _t:.1f}s)")
    _t = time.perf_counter()
    run_cost_sweep()
    _log(f"cost sweep complete ({time.perf_counter() - _t:.1f}s)")
    _log("all sweeps complete")

    if RUN_SIMULATION:
        # Simulate the rule 'baseline' plan end-to-end (line losses + train sim).
        try:
            plan = rule_results["baseline"]
            built_ids = set(plan.built_substation_ids)
            built_subs = [
                s for s in substations
                if getattr(s, "existing", False) or s.id in built_ids
            ]
            _t = time.perf_counter()
            run_simulation_step(plan, "rule/baseline", built_subs)
            _log(f"simulation step complete ({time.perf_counter() - _t:.1f}s)")
        except Exception as exc:  # noqa: BLE001 - keep sweep outputs on sim failure
            _log(f"simulation step FAILED: {exc}")
