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

Traffic assumptions come from the imported train consist plan
(``CONSIST_PLAN_PATH``): the per-link annual energy densities feeding
``make_piece_opex_fn`` are derived by simulating EVERY train in the plan over the
corridor (speed-, grade-, and mass-aware), summing per-link energy across all
trains, and annualizing (the plan is treated as one representative week). Each
train keeps its plan attributes (origin/destination, train type, car counts,
weight, length, consist size) while its locomotives are overridden to be all
dual-mode (cost analysis) or n-1 BEL-with-pantograph + 1 diesel (battery
analysis). The demand sweep is a MULTIPLIER on that annualized plan (central=1x,
high=2x). The illustrative grade-scaled constants remain as a fast fallback.

Finally, a SIMULATION STEP re-runs the full consist plan over a chosen plan's
optimized catenary -- first derating deliverable pantograph power for AC line
losses from the BUILT substation locations (``en.apply_line_losses``, the piece
milestone_2.3b lacked) -- and writes the aggregated simulated diesel +
grid-electricity + line-loss energy (vs. an all-diesel baseline) to
``energization_simulation_results.csv``.

Run:
    python applications/catenary_optimization/energization_optimization_corridor.py
"""

from __future__ import annotations

import os

# Core budget for this run: cap native thread pools (BLAS/OpenMP + ALTRIOS' Rust
# rayon + the HiGHS solver, set further below) to at most four cores. Set BEFORE
# importing numeric/altrios libraries, which spin up thread pools at import time.
_MAX_CORES = 4
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_var, str(_MAX_CORES))

# Imports below intentionally follow the thread-cap env setup above (E402), which
# must run before any numeric/altrios library initializes its thread pools.
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import hashlib  # noqa: E402
from pathlib import Path  # noqa: E402

import altrios as alt  # noqa: E402
from altrios import energization as en  # noqa: E402
from altrios import defaults as _defaults, utilities as _utils  # noqa: E402
from altrios import metric_calculator as _mc  # noqa: E402
from altrios import sim_manager as _sim_manager  # noqa: E402
from altrios.optimization import energization_placement as cp  # noqa: E402
from altrios.train_planner import data_prep as _data_prep  # noqa: E402
from altrios.train_planner import planner_config as _planner_config  # noqa: E402
from altrios.speed_limit_train_sim import (  # noqa: E402
    run_speed_limit_train_sims as _run_speed_limit_train_sims,
)

import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402

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


# Directory for CSV/PDF result output. Each run writes into its own timestamped
# subfolder (``outputs/energization_results/sweep_<YYYYmmdd_HHMMSS>``) next to this
# script so previous runs' outputs are preserved rather than overwritten. Override
# with the ``ENERGIZATION_RESULTS_DIR`` environment variable.
_RESULTS_ROOT = Path(__file__).resolve().parent / "outputs" / "energization_results"
# Dropped in a sweep dir only once the WHOLE sweep finished; its ABSENCE marks an
# incomplete sweep that can be RESUMED.
_SWEEP_COMPLETE_MARKER = "SWEEP_COMPLETE"


def _resolve_results_dir() -> Path:
    """Pick the sweep output dir, RESUMING an interrupted run when possible.

    - ``ENERGIZATION_RESULTS_DIR`` (if set) always wins.
    - else, unless ``ENERGIZATION_FRESH_SWEEP`` is truthy, reuse the most recent
      checkpoint-aware ``sweep_*`` dir that has NO ``SWEEP_COMPLETE`` marker (so
      re-running after a crash continues where it died);
    - else start a fresh timestamped dir.
    """
    override = os.environ.get("ENERGIZATION_RESULTS_DIR")
    if override:
        return Path(override)
    fresh = os.environ.get("ENERGIZATION_FRESH_SWEEP", "").strip().lower() not in (
        "", "0", "false", "no")
    if not fresh and _RESULTS_ROOT.exists():
        resumable = sorted(
            (d for d in _RESULTS_ROOT.glob("sweep_*")
             if d.is_dir() and (d / "_checkpoints").is_dir()
             and not (d / _SWEEP_COMPLETE_MARKER).exists()),
            key=lambda d: d.stat().st_mtime,
        )
        if resumable:
            return resumable[-1]
    return _RESULTS_ROOT / f"sweep_{time.strftime('%Y%m%d_%H%M%S')}"


RESULTS_DIR = _resolve_results_dir()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# Per-step checkpoint markers (one JSON per completed heavy step) live here; their
# presence lets a re-run SKIP finished steps and resume at the one that died.
_CKPT_DIR = RESULTS_DIR / "_checkpoints"
_CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Stable cache directory (persists across timestamped runs) for the expensive
# full-consist-plan simulation passes. The pre-optimization first-pass densities
# are keyed by a hash of their inputs and reused if already computed, so repeated
# sweeps don't re-simulate all ~160 trains (see ``first_pass_densities``).
_CACHE_DIR = _RESULTS_ROOT / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


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


# ---------------------------------------------------------------------------
# Checkpoint / resume: each heavy sweep step drops a JSON marker when it FULLY
# completes, so a re-run (e.g. after the session is killed) skips finished steps
# and restarts at the start of the one that died. The combined metrics CSV is
# rebuilt from the per-variant ``metrics.parquet`` shards (idempotent, so resumed
# work never double-counts).
# ---------------------------------------------------------------------------
def _step_done(step_id: str) -> bool:
    """True if ``step_id`` already completed in this (resumable) sweep dir."""
    return (_CKPT_DIR / f"{step_id}.json").exists()


def _mark_step_done(step_id: str, payload: dict) -> None:
    """Record ``step_id`` as complete, storing its summary ``payload``."""
    (_CKPT_DIR / f"{step_id}.json").write_text(json.dumps(payload))


def _completed_summary_rows() -> list[dict]:
    """Summary rows of every completed ``layout_*`` step, to reseed the summary
    table on a resumed run (diesel-step markers are excluded)."""
    out = []
    for p in sorted(_CKPT_DIR.glob("layout_*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:  # noqa: BLE001 - a corrupt marker just forces a redo
            pass
    return out


def _rebuild_metrics_csv(path: Path) -> None:
    """Rebuild the combined metrics CSV from every variant folder's
    ``metrics.parquet`` (one row-set per folder, so it is resume-safe)."""
    parts = []
    for pq in sorted(RESULTS_DIR.glob("*/metrics.parquet")):
        try:
            parts.append(pl.read_parquet(pq))
        except Exception:  # noqa: BLE001 - skip a half-written shard
            pass
    if parts:
        pl.concat(parts, how="diagonal_relaxed").write_csv(path)


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
# (short feed) builds many, 2x25 kV AT (long feed) builds few. Two pre-existing
# (free) substations anchor the corridor termini: west_terminus (Spokane) and
# east_terminus (Lethbridge).
SUBSTATION_MAX_POWER_WATTS = 50.0e6
N_CANDIDATES = 30
substations = [
    cp.Substation(
        id="west_terminus", lat=at_fraction(0.0)[0], lon=at_fraction(0.0)[1],
        existing=True, max_power_watts=SUBSTATION_MAX_POWER_WATTS,
        grid_emissions_region="NorthernGrid_West",
    ),
    cp.Substation(
        id="east_terminus", lat=at_fraction(1.0)[0], lon=at_fraction(1.0)[1],
        existing=True, max_power_watts=SUBSTATION_MAX_POWER_WATTS,
        grid_emissions_region="NorthernGrid_East",
    ),
]
for _k in range(1, N_CANDIDATES + 1):
    _frac = _k / (N_CANDIDATES + 1)
    _lat, _lon = at_fraction(_frac)
    substations.append(
        cp.Substation(
            id=f"c{_k:02d}", lat=_lat, lon=_lon,
            max_power_watts=SUBSTATION_MAX_POWER_WATTS,
            # Existing west_terminus catenary charging is attributed to
            # NorthernGrid West; the existing east_terminus and every candidate
            # substation are NorthernGrid East.
            grid_emissions_region="NorthernGrid_East",
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
    # Cap HiGHS parallelism to the run's core budget (see ``_MAX_CORES``).
    solver_threads=_MAX_CORES,
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
    # Restrict candidate electrification to trafficked track (both directions);
    # links no train uses are excluded so the optimizer can't electrify
    # trafficless spurs/branches just for their grade. Callers may override.
    kwargs.setdefault("est_time_nets", _traffic_est_time_nets())
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
# Kept in sync with ARCH_MEDIUM_COSTS["ac_25kv"] below. Costs in $M (see USD_PER_MUSD).
RULE_COSTS = cp.EnergizationOptCosts(
    structure_cost_per_mile=2.2,   # $2.2M/mi shared structure (paid once per site)
    energization_cost_per_mile=0.50,  # $0.50M/mi marginal per electrified track
    substation_fixed_cost=15.0,    # $15M per built substation
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
# Battery-electric locomotive (BEL) range parameterization for the maximum-gap
# constraint. A BEL recharges from the wire over electrified track and draws down
# its battery over un-electrified gaps, so the plan must keep every contiguous
# gap short enough that the battery does not deplete. Under the optimizer's
# simplifying assumptions (any electrified segment recharges the pack; a constant
# average draw per mile), a usable pack of ``BEL_USABLE_BATTERY_KWH`` at
# ``BEL_AVG_KWH_PER_MILE`` bridges ``en.max_gap_miles_from_battery`` miles. The
# ``reserve_fraction`` holds back part of the pack (SOC protection / accessory
# load). Tune the two energy numbers to a specific locomotive to re-scale the gap.
BEL_USABLE_BATTERY_KWH = 9000.0   # usable traction energy between charges (kWh)
BEL_AVG_KWH_PER_MILE = 190.0      # avg per-locomotive traction draw (kWh/mile)
BEL_RESERVE_FRACTION = 0.15       # withheld reserve (SOC protection + accessories)
BEL_MAX_GAP_MILES = en.max_gap_miles_from_battery(
    BEL_USABLE_BATTERY_KWH,
    BEL_AVG_KWH_PER_MILE,
    reserve_fraction=BEL_RESERVE_FRACTION,
)

RULE_CONFIGS = [
    dict(label="baseline",          budget=150, high_grade_benefit_weight=5.0,
         grade_threshold=0.02, substation_feed_distance_miles=30.0,
         min_energized_length_miles=5.0),
    dict(label="tight budget $75M", budget=75,  high_grade_benefit_weight=5.0,
         grade_threshold=0.02, substation_feed_distance_miles=30.0,
         min_energized_length_miles=5.0),
    dict(label="steep-focused",     budget=150, high_grade_benefit_weight=20.0,
         grade_threshold=0.015, substation_feed_distance_miles=30.0,
         min_energized_length_miles=5.0),
    dict(label="long runs 25mi",    budget=150, high_grade_benefit_weight=5.0,
         grade_threshold=0.02, substation_feed_distance_miles=30.0,
         min_energized_length_miles=25.0),
    dict(label="wide feed 60mi",    budget=150, high_grade_benefit_weight=5.0,
         grade_threshold=0.02, substation_feed_distance_miles=60.0,
         min_energized_length_miles=5.0),
    # Battery-electric: cap the maximum unelectrified gap to the BEL's range so
    # the pack never depletes between charges (gap derived from battery size
    # above). No budget cap -- report the spend needed to guarantee the range.
    dict(label=f"BEL gap<={BEL_MAX_GAP_MILES:.0f}mi", budget=None,
         high_grade_benefit_weight=5.0, grade_threshold=0.02,
         substation_feed_distance_miles=30.0,
         max_energized_gap_miles=BEL_MAX_GAP_MILES),
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

# Train traffic (demand) cases for the cost sweep. The corridor's actual traffic
# comes from the imported consist plan (``CONSIST_PLAN_PATH``): every train in the
# plan is simulated (pre-optimization, to derive per-link energy densities; and
# post-optimization, to test the buildout). The plan is treated as one
# representative WEEK, so its energy annualizes by ``ANNUAL_SCALER`` = 365.25 /
# ``PLAN_DAYS``. The demand sweep no longer sets trains/day directly; instead each
# level is a MULTIPLIER on the annualized plan energy: "central" = the plan as-is,
# "high" = double the traffic (catenary utilization-stress case). Per-train consist
# size + makeup come from the plan; each train's locomotives are overridden to be
# either all dual-mode (cost analysis) or n-1 BEL-with-pantograph + 1 diesel
# (battery analysis).
PLAN_DAYS = 7.0                     # consist plan treated as one representative week
ANNUAL_SCALER = 365.25 / PLAN_DAYS  # week -> year
DEMAND_LEVELS = {
    "central": 1.0,  # the imported consist plan as-is
    "high":    2.0,  # utilization-stress case: double the plan's traffic
}

# Architecture -> medium (central) capital-cost point estimates, in $M (see
# USD_PER_MUSD). Values are 2025-USD engineering syntheses calibrated to published
# overhead-electrification cost ranges for greenfield main-line freight:
#   - OCS (shared structure + one track's contact/messenger wire): ~$2.5-3.5M per
#     single-track route-mile [IEA2019; SolutionaryRail2016; NetworkRail2015].
#   - Traction substation + grid interconnection: ~$10-20M each for AC feeder
#     stations (cf. Crossrail ~GBP16M/substation [Crossrail2013]); DC rectifier
#     substations are individually cheaper (~$6-10M) but sited more densely.
#   - 2x25 kV AT adds a negative feeder + line autotransformers (higher $/mi) while
#     transmitting at 50 kV, so fewer/larger substations suffice [INECO2011;
#     Courtois1993].
#   - 9 kV HVDC (MVDC) uses lighter OCS than low-voltage DC and is modeled with
#     MODULAR, containerized power-electronic (VSC / active-front-end) converter
#     substations -- the form that gives MVDC its competitive rationale
#     (bidirectional regeneration-to-grid, grid support, factory-built rollout).
#     VSC conversion carries an equipment premium over legacy diode rectifiers,
#     only partly offset by reduced civil works and factory production, so the
#     central estimate is set conservatively (NOT cheap); the low-infra case (x0.5)
#     represents the mature, mass-produced modular future -- i.e. DC at its
#     realistic competitive best [UIC2019; Simiyu2021].
# See docs/energization-voltage-and-capacity.md for the full source table.
ARCH_MEDIUM_COSTS = {
    # key into en.suggest_optimization_inputs -> feed distance, voltage, efficiency
    # costs in $M (see USD_PER_MUSD): structure ($M/route-mi), energization
    # ($M/track-mi), substation ($M each)
    "ac_25kv": dict(structure=2.2, energization=0.50, substation=15.0),
    "ac_2x25": dict(structure=2.4, energization=0.65, substation=17.0),
    "dc_9kv":  dict(structure=2.2, energization=0.55, substation=10.0),
}

# Infrastructure-cost scenario: multiplier bracketing the wide project-to-project
# spread in electrification capital (terrain, clearance/double-stack catenary-height
# upgrades, urban complexity, and cost-overrun risk). "low" ~ favorable open-country
# greenfield with economies of scale; "high" ~ difficult terrain / complex corridors,
# consistent with the upper end of reported OCS costs [NetworkRail2015; IEA2019].
INFRA_LEVELS = {"low": 0.5, "med": 1.0, "high": 2.0}

# Diesel-vs-electricity "gap" scenario over the project lifespan. The CENTRAL
# ("med") case is the AEO2026 reference (cb2026) price TRAJECTORY -- transportation
# diesel ($/gal) and industrial electricity ($/kWh) at each rollout year (see
# _aeo_diesel_price / _aeo_elec_price, which read the updated EIA price files via
# metric_calculator). The low/high cases WIDEN the gap by ENERGY_GAP_WIDTH, moving
# diesel and electricity in opposite directions: "low" barely favors electrification
# (cheaper diesel, dearer power), "high" strongly favors it. Values are MULTIPLIERS
# on the AEO trajectory.
ENERGY_GAP_WIDTH = 1.0 / 3.0
ENERGY_LEVELS = {
    "low":  dict(diesel=1.0 - ENERGY_GAP_WIDTH, elec=1.0 + ENERGY_GAP_WIDTH),
    "med":  dict(diesel=1.0, elec=1.0),
    "high": dict(diesel=1.0 + ENERGY_GAP_WIDTH, elec=1.0 - ENERGY_GAP_WIDTH),
}


def _aeo_price_at(df, year):
    """Nearest-year ``Price`` from an AEO price table, clamped to its year range."""
    yrs = df.get_column("Year")
    y = min(max(int(year), int(yrs.min())), int(yrs.max()))
    return float(df.filter(pl.col("Year") == y).get_column("Price")[0])


def _aeo_diesel_price(year):
    """AEO2026 transportation diesel price ($/gal) at ``year``."""
    return _aeo_price_at(_mc.liquid_fuel_prices_eia, year)


def _aeo_elec_price(year):
    """AEO2026 industrial electricity price ($/kWh) at ``year`` (table is c/kWh)."""
    return _aeo_price_at(_mc.electricity_prices_eia, year) / 100.0


def _levelized_aeo_price(price_fn, start_year, lifespan, discount):
    """Discount-weighted average AEO price over ``lifespan`` years: one representative
    price for the buildout optimization, annualized consistently with the DCF."""
    num = sum(price_fn(start_year + t) / (1.0 + discount) ** t for t in range(lifespan))
    den = sum(1.0 / (1.0 + discount) ** t for t in range(lifespan))
    return num / den


# ---------------------------------------------------------------------------
# Metrics-calculator rollout settings (itemized loco / capital / fuel costs)
# ---------------------------------------------------------------------------
# Following the rollout_demo.py / sim_manager_demo.py pattern, the itemized-cost
# report builds a list of per-year ScenarioInfo objects and hands them to
# altrios.metric_calculator.main, which owns the discounted-cash-flow accounting
# (fleet rollover by asset age, residual value, discounting). For now we assume
# METRICS_ROLLOUT_YEARS IDENTICAL operating years (a true multi-year, phased
# catenary investment scheme is future work); the calculator still rolls the
# locomotive fleet and infrastructure over as assets reach end of life.
METRICS_ROLLOUT_YEARS = 30
METRICS_START_YEAR = _defaults.BASE_ANALYSIS_YEAR
# Infrastructure asset lifespans (years) fed to the rollover engine.
STRUCTURE_LIFESPAN_YEARS = 50      # civil structures / foundations / poles
ENERGIZATION_LIFESPAN_YEARS = 40   # contact + messenger wire / OCS hardware
SUBSTATION_LIFESPAN_YEARS = 30     # traction substation / power electronics
# The itemized-cost report is built for the architecture, demand, and energy
# prices below. The architecture matches the confirming simulation's 25 kV AC
# electrification (SIM_ELECTRIFICATION_TYPE); the demand multiplier scales the
# annualized full-consist-plan traffic (1.0 = the plan as-is).
ITEMIZED_ARCH = "ac_25kv"
ITEMIZED_DEMAND_MULT = DEMAND_LEVELS["central"]
ITEMIZED_ENERGY_LEVEL = "med"

# Grid emissions regions (Cambium GEA) used to attribute electricity emissions.
# The existing (west_terminus) substation's catenary charging and any Spokane
# stationary charging are NorthernGrid West; every other substation is East.
# These strings match the ``gea`` column of ``defaults.GRID_EMISSIONS_FILE`` and
# the "Grid Emissions Region" column of the corridor locations CSV.
GRID_REGION_WEST = "NorthernGrid_West"
GRID_REGION_EAST = "NorthernGrid_East"
STATIONARY_CHARGING_REGION = GRID_REGION_WEST  # Spokane terminus
CAMBIUM_SCENARIO = "MidCase"


def _region_fractions(dual_mode_energy, substation_meta):
    """Fraction of catenary grid electricity by grid-emissions region.

    Runs ``compute_line_losses`` on every dual-mode plan sim (``dual_mode_energy
    ["sims"]``), sums each feeding substation/district's grid-side energy across
    all trains, then groups by the substation's ``grid_emissions_region`` (existing
    west_terminus -> West, all others -> East). Falls back to all-East if the split
    is unavailable.
    """
    sims = (dual_mode_energy or {}).get("sims") or []
    if sims and substation_meta:
        parts = []
        for ts in sims:
            df = en.compute_line_losses(ts, substation_meta)
            df = df.filter(
                (pl.col("district_id") != pl.lit("__total__"))
                & pl.col("grid_emissions_region").is_not_null()
            )
            if df.height > 0:
                parts.append(df)
        if parts:
            g = (pl.concat(parts, how="diagonal_relaxed")
                 .group_by("grid_emissions_region")
                 .agg(pl.col("substation_energy_joules").sum()))
            total = g.get_column("substation_energy_joules").sum()
            if total and total > 0:
                return {
                    r["grid_emissions_region"]: r["substation_energy_joules"] / total
                    for r in g.iter_rows(named=True)
                }
    return {GRID_REGION_EAST: 1.0}


def _split_electricity(energy, region_frac):
    """Split a variant's grid electricity across grid-emissions regions.

    Catenary charging follows ``region_frac`` (by feeding substation); any
    stationary charging would be attributed to the Spokane (West) region -- none
    is modeled in this catenary workflow, so it is omitted here.
    """
    catenary_j = float(energy.get("grid_elec_j") or 0.0)
    region_elec = {r: catenary_j * f for r, f in region_frac.items()}
    return region_elec or {GRID_REGION_EAST: catenary_j}


def _built_sub_objs(result):
    """Substation objects feeding a plan: every existing one plus those the
    optimizer selected (``result.built_substation_ids``)."""
    ids = set(result.built_substation_ids or [])
    return [s for s in substations if s.existing or s.id in ids]


def _layout_fingerprint(result) -> str:
    """Stable hash of a plan's energized intervals + built substations, so two
    scenarios yielding the same buildout reuse a single (expensive) plan sim."""
    intervals = sorted(
        (li, tuple(sorted((round(a, 3), round(b, 3)) for a, b in ivs)))
        for li, ivs in (result.energized_link_intervals or {}).items()
    )
    subs = tuple(sorted(str(x) for x in (result.built_substation_ids or [])))
    return hashlib.sha1(repr((intervals, subs)).encode()).hexdigest()[:16]


def run_pipeline():
    """Full pipeline. The train SIMULATION depends only on the physical buildout,
    set by ARCHITECTURE x DEMAND: per (arch, demand) the NPV layout is optimized
    ONCE -- at reference (med) infrastructure cost + (med) energy prices -- both
    unconstrained (dual-mode target) and with the BEL max-gap constraint (battery
    target), and each of the three variants is simulated ONCE:

      1. ``dual_mode``  -- unconstrained layout, 100% dual-mode consists.
      2. ``bel``        -- gap-constrained layout, (n-1) BEL-with-cat + 1 diesel.
      3. ``diesel``     -- NO catenary / NO optimization, 100% conventional diesel.

    INFRASTRUCTURE COST x ENERGY PRICE are then swept as a pure ECONOMIC sensitivity
    on those fixed simulated buildouts: they re-run only the (cheap) metrics
    calculator, NEVER the simulation. Every (scenario x variant) still gets its own
    folder with the optimization outputs, the (catenary-)network + substation
    locations (for later PDF rebuilds), the metrics, the concatenated per-train sim
    series, and (catenary variants) a buildout PDF. The conventional-diesel baseline
    is architecture- and infrastructure-independent (and its raw sim is demand-
    independent), so it is simulated ONCE and emitted one run PER DEMAND LEVEL, its
    metrics spanning the energy-price sweep."""
    econ = cp.EconomicParameters(
        project_lifespan_years=PROJECT_LIFESPAN_YEARS, discount_rate=DISCOUNT_RATE
    )
    ps = _plan_summary()
    location_map = alt.import_locations(str(SIM_LOCATIONS_CSV))
    net_diesel = _calibrate_speeds(network)  # fixed no-catenary baseline network
    print("\n" + "=" * 118)
    print("FULL PIPELINE (min_cost): optimize + simulate per (arch, demand); sweep "
          "infra x energy as an economics-only re-pricing (no sim re-runs)")
    demand_desc = " | ".join(
        f"{name}=x{mult:g} ({ps['trains_per_day'] * mult:.1f}tr/day)"
        for name, mult in DEMAND_LEVELS.items()
    )
    print(f"  horizon {PROJECT_LIFESPAN_YEARS} yr | discount {DISCOUNT_RATE:.0%} | "
          f"PVAF {econ.pvaf():.2f} | consist plan: {ps['n_trains']} trains over "
          f"{PLAN_DAYS:.0f}d | demand: {demand_desc}")
    print("=" * 118)
    # First-pass full-consist-plan sims give the ANNUAL per-link per-track-mile
    # energy that feeds the min_cost opex (run once, cached, then multiplied by the
    # demand multiplier); else fall back to the illustrative grade-scaled constants.
    sim_gal_mi, sim_kwh_mi = (
        first_pass_densities() if USE_SIM_DERIVED_DENSITIES else (None, None)
    )
    # Reseed the summary table from any completed steps so a RESUMED sweep keeps
    # prior rows; per-variant metrics are rebuilt from disk (glob), not memory.
    summary_rows = _completed_summary_rows()
    metrics_csv = RESULTS_DIR / "energization_metrics_all.csv"
    if summary_rows:
        _log(f"RESUME: {len(summary_rows)} completed (arch,demand) step(s); "
             f"continuing in {RESULTS_DIR.name}")
        _write_csv(RESULTS_DIR / "energization_cost_results.csv", summary_rows)
    # Cache heavy full-plan sims by (traction_kind, layout fingerprint); the diesel
    # baseline network is constant so its sim runs once and is reused everywhere.
    sim_cache = {}

    def _sim(net, traction_kind, fingerprint, label):
        key = (traction_kind, fingerprint)
        cached = sim_cache.get(key)
        if cached is None:  # reload a bundle serialized before an earlier kill
            cached = _load_sim_bundle(traction_kind, fingerprint)
        if cached is None:
            cached = _run_full_plan(net, location_map, label, traction_kind,
                                    keep_sims=True)
            _save_sim_bundle(traction_kind, fingerprint, cached)
        sim_cache[key] = cached
        return cached

    # Reference cost point that determines the SIMULATED buildout. Infra cost and
    # energy price are swept later ONLY in the metrics accounting, so they never
    # change the layout being simulated (and thus never re-run the simulation).
    # The buildout is decided at the CENTRAL energy case: the discount-weighted
    # average AEO price over the project lifespan (so lifetime opex sits
    # consistently alongside the annualized capital in the objective's PVAF).
    ref_diesel = _levelized_aeo_price(_aeo_diesel_price, METRICS_START_YEAR,
                                      PROJECT_LIFESPAN_YEARS, DISCOUNT_RATE)
    ref_elec = _levelized_aeo_price(_aeo_elec_price, METRICS_START_YEAR,
                                    PROJECT_LIFESPAN_YEARS, DISCOUNT_RATE)
    ref_infra_mult = INFRA_LEVELS["med"]

    for arch in ("ac_25kv", "ac_2x25", "dc_9kv"):
        sug = en.suggest_optimization_inputs(arch)
        med = ARCH_MEDIUM_COSTS[arch]
        for demand_lvl, demand_mult in DEMAND_LEVELS.items():
            trains_per_day = ps["trains_per_day"] * demand_mult
            if USE_SIM_DERIVED_DENSITIES:
                diesel_by_link = {
                    li: demand_mult * v for li, v in sim_gal_mi.items()}
                elec_by_link = {
                    li: demand_mult * sim_kwh_mi[li] / sug.substation_efficiency
                    for li in sim_kwh_mi
                }
            else:
                diesel_by_link, elec_by_link = energy_densities(
                    trains_per_day, sug.substation_efficiency, sug.onboard_efficiency
                )
            feed_mi = sug.substation_feed_distance_miles
            layout = f"cost_{arch}_{demand_lvl}"
            step_id = f"layout_{layout}"
            if _step_done(step_id):
                _log(f"RESUME: skipping completed step {layout}")
                continue
            # Optimize the SIMULATED buildout once, at the reference cost point.
            ref_costs = cp.EnergizationOptCosts(
                structure_cost_per_mile=med["structure"] * ref_infra_mult,
                energization_cost_per_mile=med["energization"] * ref_infra_mult,
                substation_fixed_cost=med["substation"] * ref_infra_mult,
            )
            common_opt = dict(
                objective="min_cost", economics=econ,
                piece_opex_fn=build_opex_fn(diesel_by_link, elec_by_link,
                                            ref_diesel, ref_elec),
                costs=ref_costs, substation_feed_distance_miles=feed_mi,
                min_energized_length_miles=5.0, **COMMON,
            )
            # Variant-1 layout: NPV-optimal, NO max-gap constraint.
            r_dual = run_opt(f"{layout}/dual", **common_opt)
            # Variant-2 layout: NPV-optimal, WITH the BEL max-gap constraint
            # (can be infeasible -> skip that variant for this layout).
            try:
                r_bel = run_opt(f"{layout}/bel",
                                max_energized_gap_miles=BEL_MAX_GAP_MILES,
                                **common_opt)
            except Exception as exc:  # noqa: BLE001 - keep pipeline on infeasible gap
                _log(f"BEL layout FAILED for {layout}: {exc}")
                r_bel = None

            summary_row = {
                "layout": layout, "architecture": arch,
                "demand_level": demand_lvl, "demand_multiplier": demand_mult,
                "dual_status": str(r_dual.status).split(".")[-1],
                "dual_miles": r_dual.energized_track_miles or 0.0,
                "dual_capex_musd": r_dual.total_cost or 0.0,
                "dual_lifecycle_musd": r_dual.lifecycle_cost or 0.0,
                "dual_substations": len(r_dual.built_substation_ids),
                "bel_status": str(r_bel.status).split(".")[-1] if r_bel else "skipped",
                "bel_miles": (r_bel.energized_track_miles or 0.0) if r_bel else 0.0,
                "bel_capex_musd": (r_bel.total_cost or 0.0) if r_bel else 0.0,
                "bel_substations": len(r_bel.built_substation_ids) if r_bel else 0,
            }
            summary_rows.append(summary_row)
            _write_csv(RESULTS_DIR / "energization_cost_results.csv", summary_rows)

            # Simulate each variant ONCE (cached); the infra x energy sweep below
            # reuses these -- it never re-runs a simulation.
            subs_dual = _built_sub_objs(r_dual)
            net_dual, meta_dual = en.apply_line_losses(
                r_dual.network, subs_dual,
                system_voltage_volts=SIM_SYSTEM_VOLTAGE_VOLTS,
                electrification_type=SIM_ELECTRIFICATION_TYPE,
                base_power_limit_watts=COMMON["uniform_power_limit_watts"])
            net_dual = _calibrate_speeds(net_dual)
            e_dual = _sim(net_dual, "dual_mode", _layout_fingerprint(r_dual),
                          f"{layout}/dual")
            region_frac_dual = _region_fractions(e_dual, meta_dual)

            net_bel = subs_bel = e_bel = region_frac_bel = None
            if r_bel is not None:
                subs_bel = _built_sub_objs(r_bel)
                net_bel, meta_bel = en.apply_line_losses(
                    r_bel.network, subs_bel,
                    system_voltage_volts=SIM_SYSTEM_VOLTAGE_VOLTS,
                    electrification_type=SIM_ELECTRIFICATION_TYPE,
                    base_power_limit_watts=COMMON["uniform_power_limit_watts"])
                net_bel = _calibrate_speeds(net_bel)
                e_bel = _sim(net_bel, "battery_electric",
                             _layout_fingerprint(r_bel), f"{layout}/bel")
                region_frac_bel = _region_fractions(e_bel, meta_bel)

            # Train-planner charging pass: schedules stationary charging sessions
            # (BEL only) and yields actual departure times to place each train's
            # load curve on a common wall-clock. The dispatch schedule is traction-
            # agnostic, so its departures are reused for the dual-mode curves too.
            sched_net = net_bel if r_bel is not None else net_dual
            refuel_sessions_bel, dep_by_index = _planner_charge_pass(
                sched_net, location_map)
            if r_bel is None:
                refuel_sessions_bel = None
            dep_seconds = _departure_seconds(dep_by_index, len(e_dual["sims"]))

            def _do_variant(tag, net, result, built_objs, energy, region_frac,
                            loco_type, loco_cost, render_pdf, *, scenario,
                            infra_lvl, infra_mult, energy_lvl,
                            substation_meta=None, refuel_sessions=None):
                try:
                    m = _variant_metrics_df(
                        tag, loco_type, loco_cost, energy, result, built_objs,
                        region_frac, arch=arch, demand_lvl=demand_lvl,
                        demand_mult=demand_mult, infra_lvl=infra_lvl,
                        infra_mult=infra_mult, energy_lvl=energy_lvl)
                    _serialize_variant_run(
                        RESULTS_DIR / f"{scenario}__{tag}", tag=tag, net=net,
                        result=result, metrics_df=m, sims=energy["sims"],
                        feed_distance_miles=feed_mi, render_pdf=render_pdf,
                        plan_label=f"{scenario} {tag}",
                        departure_seconds=dep_seconds,
                        substation_meta=substation_meta,
                        refuel_sessions=refuel_sessions)
                except Exception as exc:  # noqa: BLE001 - keep pipeline alive
                    _log(f"variant [{tag}] FAILED for {scenario}: {exc}")

            # ECONOMIC sensitivity: sweep infra cost x energy price over the FIXED
            # simulated buildouts. Only the metrics calculator re-runs here.
            for infra_lvl, infra_mult in INFRA_LEVELS.items():
                for energy_lvl in ENERGY_LEVELS:
                    scenario = f"{layout}_infra-{infra_lvl}_energy-{energy_lvl}"
                    _do_variant("dual_mode", net_dual, r_dual, subs_dual, e_dual,
                                region_frac_dual, "Dual_Mode",
                                lambda _y: _defaults.DUAL_MODE_LOCO_COST_USD, True,
                                scenario=scenario, infra_lvl=infra_lvl,
                                infra_mult=infra_mult, energy_lvl=energy_lvl,
                                substation_meta=meta_dual)
                    if r_bel is not None:
                        _do_variant("bel", net_bel, r_bel, subs_bel, e_bel,
                                    region_frac_bel, "BEL",
                                    _bel_with_cat_loco_cost, True,
                                    scenario=scenario, infra_lvl=infra_lvl,
                                    infra_mult=infra_mult, energy_lvl=energy_lvl,
                                    substation_meta=meta_bel,
                                    refuel_sessions=refuel_sessions_bel)

            # Step complete: rebuild the combined metrics CSV and checkpoint so a
            # RESUMED run skips this whole (arch, demand) block next time.
            _rebuild_metrics_csv(metrics_csv)
            _mark_step_done(step_id, summary_row)

    # Conventional-diesel baseline: NO catenary / NO optimization, so it is
    # independent of ARCHITECTURE and INFRASTRUCTURE cost -- and the raw plan sim is
    # demand-independent too (demand is a downstream annualization multiplier). So
    # it is simulated ONCE (cached) and emitted ONE run PER DEMAND LEVEL, its metrics
    # spanning the energy-price sweep (diesel price is the only economic dimension
    # that touches the baseline).
    e_die = None  # simulated lazily, only if a diesel step still needs running
    for demand_lvl, demand_mult in DEMAND_LEVELS.items():
        d_step = f"diesel_{demand_lvl}"
        if _step_done(d_step):
            _log(f"RESUME: skipping completed step diesel {demand_lvl}")
            continue
        if e_die is None:
            e_die = _sim(net_diesel, "conventional", "no_catenary",
                         "diesel-baseline")
        die_metrics = []
        for energy_lvl in ENERGY_LEVELS:
            try:
                die_metrics.append(_variant_metrics_df(
                    "diesel", "Diesel", lambda _y: _defaults.DIESEL_LOCO_COST_USD, e_die,
                    None, [], {GRID_REGION_EAST: 1.0}, arch="none",
                    demand_lvl=demand_lvl, demand_mult=demand_mult,
                    infra_lvl="na", infra_mult=1.0, energy_lvl=energy_lvl))
            except Exception as exc:  # noqa: BLE001 - keep pipeline alive
                _log(f"diesel metrics FAILED {demand_lvl}/{energy_lvl}: {exc}")
        if die_metrics:
            m = pl.concat(die_metrics, how="diagonal_relaxed")
            _serialize_variant_run(
                RESULTS_DIR / f"cost_diesel_{demand_lvl}__diesel", tag="diesel",
                net=net_diesel, result=None, metrics_df=m, sims=e_die["sims"],
                feed_distance_miles=None, render_pdf=False,
                plan_label=f"diesel {demand_lvl}")
        _rebuild_metrics_csv(metrics_csv)
        _mark_step_done(d_step, {"demand": demand_lvl})

    _rebuild_metrics_csv(metrics_csv)
    (RESULTS_DIR / _SWEEP_COMPLETE_MARKER).write_text(
        time.strftime("%Y-%m-%d %H:%M:%S"))
    n_econ = len(INFRA_LEVELS) * len(ENERGY_LEVELS)
    _log(f"sweep COMPLETE: {len(summary_rows)} catenary layouts x 2 variants x "
         f"{n_econ} economic points + diesel x {len(DEMAND_LEVELS)} demand -> "
         f"{RESULTS_DIR}")



# ---------------------------------------------------------------------------
# Full-plan simulation config (shared by the density pass + the pipeline variants)
# ---------------------------------------------------------------------------
# When True, the min_cost opex per-link energy comes from full-consist-plan
# calibrated dual-mode simulations (all-diesel + all-catenary) instead of the
# illustrative grade-scaled constants. Adds two full-plan passes (cached) per run.
USE_SIM_DERIVED_DENSITIES = True
SIM_SAVE_INTERVAL = 60
SIM_ORIGIN, SIM_DESTINATION = "Spokane", "Lethbridge"
SIM_LOCATIONS_CSV = (alt.resources_root() / "networks" / "Spokane_Lethbridge"
                     / "spokane_lethbridge_locations.csv")
SIM_N_CARS = {"Manifest_Loaded": 75, "Manifest_Empty": 75}
# Fallback consist size for the loco-independent traffic est-time nets only
# (``_make_corridor_ts``); the energy sims size every consist from the plan.
SIM_N_LOCOS = 10
# BEL variant: each plan train is rebuilt as n-1 BEL-with-pantograph units + 1
# diesel. On energized track a BEL draws catenary power for traction first, then
# charges its battery with leftover capacity; off catenary it runs on the battery.
# SIM_BEL_CAT_PWR_MAX_WATTS is the per-locomotive max deliverable pantograph power.
SIM_BEL_CAT_PWR_MAX_WATTS = 5.0e6
# Imported train consist plan (``_load_full_consist_plan``): one row per
# locomotive assignment, collapsed to one record per train. EVERY train is
# simulated (pre-optimization for densities, post-optimization to test the
# buildout) with its locomotives overridden to all dual-mode (cost) or n-1 BEL +
# 1 diesel (battery). Defaults to the milestone_1.3 Spokane-Lethbridge consist
# plan sitting next to this script.
CONSIST_PLAN_PATH = (Path(__file__).resolve().parent
                     / "train_consist_plan_spokane_lethbridge_clean.csv")
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


def load_consist_plan(path=None):
    """Deprecated single-consist shim; see ``_load_full_consist_plan``.

    Kept only so external callers importing this name keep working. Returns the
    binding consist size (max locomotives on any plan train) as ``{"n_locos": N}``.
    """
    plan = _load_full_consist_plan(path)
    n = max((int(t["n_locos"]) for t in plan), default=SIM_N_LOCOS)
    return {"n_locos": n}


# Rolling stock, consist plan, and plan summary, each loaded once and reused for
# every simulated train (both the pre-optimization density pass and the
# post-optimization buildout test run the FULL plan).
_RAIL_VEHICLES = None
_FREIGHT_TO_CAR = None
_FULL_CONSIST_PLAN = None
_PLAN_SUMMARY = None


def _load_rail_vehicles():
    """Load the corridor rolling stock (all freight types) once, with the
    freight-type -> car-type mapping the train planner uses."""
    global _RAIL_VEHICLES, _FREIGHT_TO_CAR
    if _RAIL_VEHICLES is None:
        _RAIL_VEHICLES = [
            alt.RailVehicle.from_file(rv_file)
            for rv_file in sorted(
                (alt.resources_root() / "rolling_stock").glob("*.yaml"))
        ]
        _FREIGHT_TO_CAR = _data_prep.get_freight_to_car_mapping(_RAIL_VEHICLES)
    return _RAIL_VEHICLES, _FREIGHT_TO_CAR


def _load_full_consist_plan(path=None):
    """Parse the imported consist plan into one record per TRAIN (loaded once).

    The CSV has one row per locomotive assignment (keyed by ``Train_ID``); this
    collapses it to a per-train list carrying origin/destination, train type,
    loaded/empty car counts, physical length/weight, and the number of assigned
    locomotives (consist size). The full-plan simulators then override each
    train's locomotives to be all dual-mode or n-1 BEL + 1 diesel while preserving
    that consist size and the rest of the train's attributes.
    """
    global _FULL_CONSIST_PLAN
    if _FULL_CONSIST_PLAN is None:
        path = path if path is not None else CONSIST_PLAN_PATH
        df = pl.read_csv(path)
        _FULL_CONSIST_PLAN = (
            df.group_by("Train_ID")
            .agg(
                pl.len().alias("n_locos"),
                pl.first("Origin_ID").alias("Origin_ID"),
                pl.first("Destination_ID").alias("Destination_ID"),
                pl.first("Train_Type").alias("Train_Type"),
                pl.first("Cars_Loaded").alias("Cars_Loaded"),
                pl.first("Cars_Empty").alias("Cars_Empty"),
                pl.first("Train_Length_meters").alias("Train_Length_meters"),
                pl.first("Train_Weight_tons").alias("Train_Weight_tons"),
                # Fallback wall-clock offset when the planner schedule is unavailable.
                pl.first("Departure_Time_Planned_Hr").alias("Departure_Time_Planned_Hr"),
            )
            .sort("Train_ID")
            .to_dicts()
        )
    return _FULL_CONSIST_PLAN


def _plan_summary():
    """Traffic summary of the imported consist plan (treated as one week).

    Returns trains/day, locomotives/day, and average locomotives/train over
    ``PLAN_DAYS``; used for reporting and fleet sizing now that the demand sweep
    is a multiplier on the plan rather than a trains/day setting.
    """
    global _PLAN_SUMMARY
    if _PLAN_SUMMARY is None:
        plan = _load_full_consist_plan()
        n_trains = len(plan)
        total_locos = sum(int(t["n_locos"]) for t in plan)
        _PLAN_SUMMARY = {
            "n_trains": n_trains,
            "total_locos": total_locos,
            "trains_per_day": n_trains / PLAN_DAYS,
            "locos_per_day": total_locos / PLAN_DAYS,
            "avg_locos_per_train": total_locos / max(n_trains, 1),
        }
    return _PLAN_SUMMARY


def _build_locos(n_locos, traction_kind, save_interval=SIM_SAVE_INTERVAL):
    """Build one train's locomotive list with composition set by ``traction_kind``.

    - ``"dual_mode"`` -- all ``n_locos`` units are dual-mode (diesel + catenary),
      for the COST analysis (diesel displaced by grid electricity on wire).
    - ``"battery_electric"`` -- ``n_locos - 1`` BEL-with-pantograph units plus ONE
      conventional diesel (non-dual-mode "get-home" unit), for the BATTERY
      analysis. For ``n_locos == 1`` a minimal 1 BEL + 1 diesel pair is built.

    The consist SIZE comes from the plan; only the unit composition is set here.
    """
    n = max(int(n_locos), 1)
    if traction_kind == "dual_mode":
        locos = [alt.Locomotive.default_dual_mode_loco() for _ in range(n)]
    elif traction_kind == "conventional":
        locos = [alt.Locomotive.default() for _ in range(n)]  # 100% diesel baseline
    elif traction_kind == "battery_electric":
        n_bel = max(n - 1, 1)
        locos = [_bel_with_cat_loco() for _ in range(n_bel)]
        locos.append(alt.Locomotive.default())  # conventional diesel (not dual-mode)
    else:
        raise ValueError(
            "traction_kind must be 'dual_mode', 'conventional', or "
            f"'battery_electric', got {traction_kind!r}"
        )
    for loco in locos:
        loco.set_save_interval(save_interval)
    return locos


def _make_plan_train_ts(train, location_map, traction_kind,
                        save_interval=SIM_SAVE_INTERVAL):
    """Build one SpeedLimitTrainSim for a consist-plan train.

    Uses the train's actual plan attributes -- origin/destination, train type,
    loaded/empty car counts, and physical weight/length (rail-vehicle mass and
    length are scaled to match) -- with the locomotive composition overridden per
    ``traction_kind`` (all dual-mode, or n-1 BEL + 1 diesel).
    """
    rail_vehicles, freight_map = _load_rail_vehicles()
    rv_to_use, n_cars_by_type = _data_prep.configure_rail_vehicles(
        train["Train_Type"], int(train["Cars_Loaded"]), int(train["Cars_Empty"]),
        rail_vehicles, freight_map,
    )
    locos = _build_locos(train["n_locos"], traction_kind, save_interval)
    rv_to_use = _data_prep.scale_rv_mass_and_length(
        rv_to_use, n_cars_by_type, locos,
        total_mass_kilograms=train["Train_Weight_tons"] * _utils.KG_PER_TON,
        total_length_meters=train["Train_Length_meters"],
    )
    train_config = alt.TrainConfig(
        rail_vehicles=rv_to_use, n_cars_by_type=n_cars_by_type,
        train_length_meters=None, train_mass_kilograms=None,
    )
    loco_con = alt.Consist(locos, save_interval)
    tsb = alt.TrainSimBuilder(
        train_id=str(train["Train_ID"]),
        origin_id=train["Origin_ID"], destination_id=train["Destination_ID"],
        train_config=train_config, loco_con=loco_con,
    )
    ts = tsb.make_speed_limit_train_sim(
        location_map=location_map, save_interval=save_interval)
    ts.set_save_interval(save_interval)
    return ts


def _make_corridor_ts(location_map, origin_id, destination_id, loco_factory=None):
    """Build (but do not dispatch/walk) a fixed corridor SpeedLimitTrainSim.

    Used only for the up-front traffic est-time nets that restrict the optimizer's
    candidate links (see ``_traffic_est_time_nets``); which links a train is routed
    over is locomotive-independent, so this builds a simple ``SIM_N_LOCOS`` consist
    of ``loco_factory`` (defaulting to ``default_dual_mode_loco``) with the fixed
    ``SIM_N_CARS``. The actual energy sims build every plan train instead (see
    ``_make_plan_train_ts`` / ``_run_full_plan``).
    """
    rvl = alt.RailVehicle.from_file(alt.resources_root() / "rolling_stock/Manifest_Loaded.yaml")
    rve = alt.RailVehicle.from_file(alt.resources_root() / "rolling_stock/Manifest_Empty.yaml")
    train_config = alt.TrainConfig(
        rail_vehicles=[rvl, rve], n_cars_by_type=SIM_N_CARS,
        train_length_meters=None, train_mass_kilograms=None,
    )
    if loco_factory is None:
        loco_factory = alt.Locomotive.default_dual_mode_loco
    loco = loco_factory()
    loco.set_save_interval(SIM_SAVE_INTERVAL)
    loco_con = alt.Consist([loco.copy() for _ in range(SIM_N_LOCOS)], SIM_SAVE_INTERVAL)
    tsb = alt.TrainSimBuilder(
        train_id="0", origin_id=origin_id, destination_id=destination_id,
        train_config=train_config, loco_con=loco_con,
    )
    ts = tsb.make_speed_limit_train_sim(location_map=location_map, save_interval=SIM_SAVE_INTERVAL)
    ts.set_save_interval(SIM_SAVE_INTERVAL)
    return ts


# Cached est-time nets (both corridor directions) that define which links carry
# train traffic; passed to every optimization so candidate electrification is
# restricted to trafficked track (see ``_traffic_est_time_nets``).
_TRAFFIC_ETNS = None


def _traffic_est_time_nets():
    """Est-time nets for trains in BOTH corridor directions (built once, cached).

    Restricts every optimization to links that carry train traffic in either
    direction; the optimizer additionally treats each traversed link's flip as
    trafficked, so no-traffic spurs/branches are excluded from consideration.
    Built on the raw (uncalibrated) ``network`` -- speed calibration changes
    timing, not which links a train is routed over.
    """
    global _TRAFFIC_ETNS
    if _TRAFFIC_ETNS is None:
        location_map = alt.import_locations(str(SIM_LOCATIONS_CSV))
        etns = []
        for o, d in ((SIM_ORIGIN, SIM_DESTINATION), (SIM_DESTINATION, SIM_ORIGIN)):
            # Which links a train is routed over is loco-independent, so use the
            # always-available default loco (not the dual-mode sim consist) to
            # keep the traffic mask decoupled from locomotive availability.
            ts = _make_corridor_ts(location_map, o, d, alt.Locomotive.default)
            etn, _consist = alt.make_est_times(ts, network)
            etns.append(etn)
        _TRAFFIC_ETNS = etns
    return _TRAFFIC_ETNS


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


def _energy_from_walked(ts):
    """Extract energy totals from a single walked SpeedLimitTrainSim.

    Returns a dict with diesel fuel J, pantograph electricity-in J, catenary
    line-loss J, grid electricity-in J (pantograph + line loss), net
    battery-discharge J, and travel time (hr). Diesel is the built-in consist fuel
    getter (zero for a pure battery-electric consist); grid and line-loss energy
    come from each locomotive's catenary-interface cumulative energy history
    (dual-mode always, battery-electric when a pantograph is fitted). Shared by the
    full-plan aggregator ``_run_full_plan``.
    """
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
        # the BEL analysis.
        bel = loco["loco_type"].get("BatteryElectricLoco")
        if bel is not None:
            e_chem = bel["res"]["history"].get("energy_out_chemical_joules")
            if e_chem:
                battery_j += e_chem[-1]
    return {"diesel_j": ts.get_energy_fuel_joules(False),
            "pantograph_elec_j": pantograph_j, "line_loss_j": line_loss_j,
            "grid_elec_j": pantograph_j + line_loss_j, "battery_j": battery_j,
            "travel_hr": times[-1] / 3600.0}


def _run_full_plan(sim_network, location_map, label, traction_kind,
                   want_per_link=False, keep_sims=True):
    """Build, dispatch, and walk EVERY train in the consist plan over
    ``sim_network``; return energy summed over all trains.

    Each train is built from its plan attributes (``_make_plan_train_ts``) with the
    locomotive composition set by ``traction_kind`` (all dual-mode, or n-1 BEL + 1
    diesel). Trains are dispatched independently (no meet/pass interaction), which
    matches how the per-link densities / aggregate energy are used here. The
    returned totals are for the plan's ~one-week traffic; callers annualize with
    ``ANNUAL_SCALER``.

    Parameters
    ----------
    want_per_link : bool
        Also return per-link diesel/pantograph energy maps summed over all trains
        (dual-mode only; feeds the optimizer density inputs).
    keep_sims : bool
        Retain the walked sims in the result (needed for battery sizing / the
        itemized-cost rollout; set False to save memory for baseline-only runs).
    """
    plan = _load_full_consist_plan()
    n = len(plan)
    totals = {"diesel_j": 0.0, "pantograph_elec_j": 0.0, "line_loss_j": 0.0,
              "battery_j": 0.0, "travel_hr": 0.0}
    diesel_by_link, elec_by_link = {}, {}
    sims = []
    _log(f"  {label}: simulating {n} plan trains ({traction_kind}) ...")
    t0 = time.perf_counter()
    for k, train in enumerate(plan, 1):
        ts = _make_plan_train_ts(train, location_map, traction_kind)
        etn, _consist = alt.make_est_times(ts, sim_network)
        timed_path = next(iter(alt.run_dispatch(
            sim_network, alt.SpeedLimitTrainSimVec([ts]), [etn], False, False)))
        ts.walk_timed_path(network=sim_network, timed_path=timed_path)
        e = _energy_from_walked(ts)
        for key in totals:
            totals[key] += e[key]
        if want_per_link:
            dbl, ebl = _per_link_energy_from_sim(ts)
            for li, v in dbl.items():
                diesel_by_link[li] = diesel_by_link.get(li, 0.0) + v
            for li, v in ebl.items():
                elec_by_link[li] = elec_by_link.get(li, 0.0) + v
        if keep_sims:
            sims.append(ts)
        if k % 20 == 0 or k == n:
            _log(f"    {label}: {k}/{n} trains ({time.perf_counter() - t0:.0f}s)")
    totals["grid_elec_j"] = totals["pantograph_elec_j"] + totals["line_loss_j"]
    _log(f"  {label} TOTAL over {n} trains: diesel {totals['diesel_j']/1e9:.1f} GJ, "
         f"grid {totals['grid_elec_j']/1e9:.1f} GJ, "
         f"battery {totals['battery_j']/1e9:.1f} GJ")
    return {**totals, "sims": sims, "n_trains": n,
            "diesel_j_by_link": diesel_by_link, "elec_j_by_link": elec_by_link}


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


# ---------------------------------------------------------------------------
# Walked-sim disk cache: the ~full-plan sim is the most expensive step, so each
# bundle is serialized to ``_CACHE_DIR`` the instant it finishes (keyed by
# traction kind + layout fingerprint). A killed/restarted process reloads it
# instead of re-simulating ~all trains. Persists across sweep dirs (the diesel
# baseline, keyed "no_catenary", is thus only ever simulated once).
# ---------------------------------------------------------------------------
def _sim_bundle_path(traction_kind: str, fingerprint: str) -> Path:
    return _CACHE_DIR / f"simbundle_{traction_kind}_{fingerprint}.msgpack"


def _aggregate_walked_energy(sims: list) -> dict:
    """Rebuild the ``_run_full_plan`` energy totals from ALREADY-walked sims (no
    dispatch), so a reloaded bundle matches a freshly simulated one."""
    totals = {"diesel_j": 0.0, "pantograph_elec_j": 0.0, "line_loss_j": 0.0,
              "battery_j": 0.0, "travel_hr": 0.0}
    for ts in sims:
        e = _energy_from_walked(ts)
        for k in totals:
            totals[k] += e[k]
    totals["grid_elec_j"] = totals["pantograph_elec_j"] + totals["line_loss_j"]
    return {**totals, "sims": list(sims), "n_trains": len(sims),
            "diesel_j_by_link": {}, "elec_j_by_link": {}}


def _load_sim_bundle(traction_kind: str, fingerprint: str):
    """Reload a cached walked sim bundle from disk, or ``None`` if absent/corrupt."""
    path = _sim_bundle_path(traction_kind, fingerprint)
    if not path.exists():
        return None
    try:
        vec = alt.SpeedLimitTrainSimVec.from_file(str(path))
        sims = [alt.SpeedLimitTrainSim.from_pydict(d) for d in vec.to_pydict()]
        _log(f"  reusing cached sim bundle <- {path.name} ({len(sims)} sims)")
        return _aggregate_walked_energy(sims)
    except Exception as exc:  # noqa: BLE001 - a bad cache just forces a re-sim
        _log(f"  sim-bundle cache read FAILED [{path.name}]: {exc}")
        return None


def _save_sim_bundle(traction_kind: str, fingerprint: str, bundle: dict) -> None:
    """Serialize a walked sim bundle to disk IMMEDIATELY after simulating, so a
    killed process never loses the (expensive) full-plan sim."""
    try:
        alt.SpeedLimitTrainSimVec(list(bundle["sims"])).to_file(
            str(_sim_bundle_path(traction_kind, fingerprint)))
    except Exception as exc:  # noqa: BLE001 - caching is best-effort
        _log(f"  sim-bundle cache write FAILED "
             f"[{traction_kind}/{fingerprint}]: {exc}")


def _bel_planner_config():
    """A ``TrainPlannerConfig`` that maps the plan's ``Dual_Mode`` locomotives to
    battery-electric (with pantograph) units charged by an electric refueler, so
    the planner schedules stationary charging sessions for the imported plan."""
    cfg = _planner_config.TrainPlannerConfig(
        loco_info=pd.DataFrame({
            "Dual_Mode": {
                "Fuel_Type": "Electricity",
                "Min_Servicing_Time_Hr": 3.0,
                "Rust_Loco": _bel_with_cat_loco(),
                "Cost_USD": _defaults.BEL_MINUS_BATTERY_COST_USD,
                "Lifespan_Years": _defaults.LOCO_LIFESPAN,
            },
        }).transpose().reset_index(names="Locomotive_Type"))
    # Point the electric charger at the plan's (Dual_Mode) locomotive type.
    charger = cfg.refueler_info[
        cfg.refueler_info["Refueler_Type"] == "BEL_Charger"].copy()
    charger["Locomotive_Type"] = "Dual_Mode"
    cfg.refueler_info.update(charger)
    return cfg


def _planner_charge_pass(sim_network, location_map):
    """Run the train planner + post-sim charging driver over the imported plan.

    The planner schedules the plan onto ``sim_network`` (battery-electric consists),
    then :func:`run_speed_limit_train_sims` walks the realized timed paths and
    regenerates charging sessions from realized energy use. Returns
    ``(refuel_sessions, departure_hr_by_index)`` where the second maps each
    ``TrainSimVec_Index`` to its actual departure hour (for aligning load curves on
    a common wall-clock); ``(None, {})`` if the planner pass fails.
    """
    try:
        rail_vehicles, _ = _load_rail_vehicles()
        consist_plan_in = pl.read_csv(CONSIST_PLAN_PATH)
        (consist_plan_out, loco_pool, refuelers, _gef, _nep,
         train_sims, timed_paths, _untrimmed) = _sim_manager.main(
            rail_vehicles=rail_vehicles, location_map=location_map,
            network=sim_network, consist_plan_in=consist_plan_in,
            train_planner_config=_bel_planner_config(),
            scenario_year=METRICS_START_YEAR, walk_timed_paths=False, debug=False)
        # sim_manager returns timed paths as pydicts; rebuild the objects the
        # driver's walk_timed_path expects.
        timed_path_objs = [alt.TimedLinkPath.from_pydict(tp) for tp in timed_paths]
        _walked, refuel_sessions = _run_speed_limit_train_sims(
            train_sims, sim_network, consist_plan_out, loco_pool, refuelers,
            timed_path_objs)
        dep = (consist_plan_out
               .group_by("TrainSimVec_Index")
               .agg(pl.first("Departure_Time_Actual_Hr").alias("dep_hr")))
        dep_by_index = {int(r["TrainSimVec_Index"]): float(r["dep_hr"])
                        for r in dep.iter_rows(named=True)
                        if r["dep_hr"] is not None}
        return refuel_sessions, dep_by_index
    except Exception as exc:  # noqa: BLE001 - charging export is non-fatal
        _log(f"  planner charging pass FAILED: {exc}")
        return None, {}


def _departure_seconds(dep_by_index, n_sims):
    """Per-train wall-clock departure offset (s) aligned to the ``_run_full_plan``
    sim order, from the planner schedule (``dep_by_index``) with the plan's planned
    departures as a fallback when the planner schedule is missing an index."""
    plan = _load_full_consist_plan()
    out = []
    for i in range(n_sims):
        hr = dep_by_index.get(i)
        if hr is None and i < len(plan):
            hr = plan[i].get("Departure_Time_Planned_Hr")
        out.append(float(hr) * 3600.0 if hr is not None else 0.0)
    return out


def _first_pass_cache_path():
    """Stable cache path for the first-pass densities, keyed by their inputs.

    The key hashes the consist-plan file contents plus the network file, speed
    calibration, save interval, and catenary power limit so the cache is reused
    only when those inputs are unchanged; edit any of them and a fresh pass runs.
    """
    h = hashlib.sha1()
    h.update(Path(CONSIST_PLAN_PATH).read_bytes())
    h.update(str(_NET_PATH).encode())
    h.update(repr((SPEED_CALIBRATION_MAX_MPH, SPEED_LIMIT_SCALER, SIM_SAVE_INTERVAL,
                   COMMON["uniform_power_limit_watts"], PLAN_DAYS)).encode())
    return _CACHE_DIR / f"first_pass_densities_{h.hexdigest()[:12]}.parquet"


def first_pass_densities():
    """First-pass calibrated sims over the FULL consist plan -> per-link densities.

    Runs every train in the imported consist plan (all-dual-mode composition)
    twice over the calibrated corridor: an all-diesel pass (no catenary) for the
    diesel each train burns per link, and an all-catenary pass (network fully
    energized) for the pantograph electricity per link if that link were
    electrified. Per-link energy is summed over ALL trains and annualized
    (``ANNUAL_SCALER``, the plan treated as one representative week), giving
    ``(diesel_gal_per_track_mile_by_link, pantograph_kwh_per_track_mile_by_link)``
    -- ANNUAL densities that already embed the plan's directional traffic. The
    demand sweep applies its multiplier on top; substation efficiency and
    placement-dependent line losses are applied later.

    The result is cached to ``_CACHE_DIR`` (keyed by the inputs) and reused on
    subsequent runs so the ~160-train pass isn't repeated unnecessarily.
    """
    cache = _first_pass_cache_path()
    if cache.exists():
        _log(f"FIRST-PASS: reusing cached densities <- {cache}")
        cdf = pl.read_parquet(cache)
        gal = dict(zip(cdf.get_column("link_idx").to_list(),
                       cdf.get_column("diesel_gal_per_mi").to_list()))
        kwh = dict(zip(cdf.get_column("link_idx").to_list(),
                       cdf.get_column("elec_kwh_per_mi").to_list()))
        return gal, kwh

    _log("FIRST-PASS calibrated sims over FULL consist plan "
         "(dual-mode: all-diesel + all-catenary) ...")
    location_map = alt.import_locations(str(SIM_LOCATIONS_CSV))
    net_diesel = _calibrate_speeds(network)
    net_cat = _calibrate_speeds(en.energize_uniform(
        network, uniform_power_limit_watts=COMMON["uniform_power_limit_watts"]))
    die = _run_full_plan(net_diesel, location_map, "first-pass/all-diesel",
                         "dual_mode", want_per_link=True, keep_sims=False)
    cat = _run_full_plan(net_cat, location_map, "first-pass/all-catenary",
                         "dual_mode", want_per_link=True, keep_sims=False)
    diesel_j_by_link = die["diesel_j_by_link"]   # fuel from all-diesel pass
    elec_j_by_link = cat["elec_j_by_link"]       # grid from all-catenary pass

    # J/gallon diesel (LHV * density * m3/gallon), from altrios defaults/utils.
    j_per_gal = ((_defaults.LHV_DIESEL_KJ_PER_KG * 1000.0)
                 * _defaults.RHO_DIESEL_KG_PER_M3
                 / (_utils.GALLONS_PER_LITER * _utils.LITER_PER_M3))
    diesel_gal_per_mi = {}
    elec_kwh_per_mi = {}
    for li, miles in LINK_MILES.items():
        if miles <= 0:
            continue
        # Sum over all plan trains -> annualize (week -> year) -> per track-mile.
        diesel_gal_per_mi[li] = (
            (diesel_j_by_link.get(li, 0.0) / j_per_gal) * ANNUAL_SCALER / miles)
        elec_kwh_per_mi[li] = (
            (elec_j_by_link.get(li, 0.0) / 3.6e6) * ANNUAL_SCALER / miles)

    _log(f"  first-pass plan totals ({die['n_trains']} trains, x{ANNUAL_SCALER:.1f} "
         f"-> annual): {sum(diesel_j_by_link.values())/j_per_gal*ANNUAL_SCALER:.0f} "
         f"gal diesel/yr (all-diesel), "
         f"{sum(elec_j_by_link.values())/3.6e6*ANNUAL_SCALER:.0f} kWh pantograph/yr "
         f"(all-catenary)")
    # Serialize the annual per-track-mile densities for reuse.
    pl.DataFrame({
        "link_idx": list(diesel_gal_per_mi.keys()),
        "diesel_gal_per_mi": list(diesel_gal_per_mi.values()),
        "elec_kwh_per_mi": [elec_kwh_per_mi[li] for li in diesel_gal_per_mi],
    }).write_parquet(cache)
    _log(f"  cached first-pass densities -> {cache}")
    return diesel_gal_per_mi, elec_kwh_per_mi


# ---------------------------------------------------------------------------
# Itemized-cost report: locomotives + capital (catenary) + fuel, via the metrics
# calculator's discounted-cash-flow engine (rollout_demo.py / sim_manager_demo.py
# pattern), assuming METRICS_ROLLOUT_YEARS identical operating years.
# ---------------------------------------------------------------------------
def _bel_with_cat_loco_cost(year: int) -> float:
    """BEL-with-catenary unit cost ($): the usual battery-electric cost (NREL ATB
    battery pack via ``metric_calculator.add_battery_costs``) plus the pantograph
    premium (``defaults.PANTOGRAPH_COST_USD``)."""
    import pandas as pd

    info_df = pd.DataFrame({"Rust_Loco": [_bel_with_cat_loco()],
                            "Cost_USD": [float("nan")]})
    info_df = _mc.add_battery_costs(info_df, year)
    return float(info_df.at[0, "Cost_USD"]) + _defaults.PANTOGRAPH_COST_USD


def _infrastructure_costs_df(plan_result, built_subs, arch=ITEMIZED_ARCH,
                             infra_mult=1.0) -> pl.DataFrame:
    """Per-asset catenary infrastructure capital ($) for the metrics rollover,
    using ``arch``'s medium costs scaled by ``infra_mult`` (the infrastructure-cost
    sweep level): one aggregate structure asset, one aggregate energization asset,
    and one row per built (non-existing) substation.

    Structure is charged per corridor route-mile and energization per track-mile;
    on this single-track corridor both use the energized track-miles. This is the
    illustrative decomposition documented in
    docs/energization-voltage-and-capacity.md, not a costing study.

    The ``Distance_Miles`` / ``Distance_Units`` columns feed the metrics
    calculator's ``Distance_Energized`` metric: energization carries the built
    catenary wire length (track-miles). Structure would carry corridor-miles, but
    that is not stored on the plan result, so it is left null here.
    """
    med = ARCH_MEDIUM_COSTS[arch]
    miles = plan_result.energized_track_miles or 0.0
    n_subs = sum(1 for s in built_subs if not getattr(s, "existing", False))
    rows = [
        {"Infrastructure_Type": "Catenary_Structure",
         "Cost_USD": med["structure"] * infra_mult * USD_PER_MUSD * miles,
         "Lifespan_Years": STRUCTURE_LIFESPAN_YEARS,
         "Distance_Miles": None, "Distance_Units": None},
        {"Infrastructure_Type": "Catenary_Energization",
         "Cost_USD": med["energization"] * infra_mult * USD_PER_MUSD * miles,
         "Lifespan_Years": ENERGIZATION_LIFESPAN_YEARS,
         "Distance_Miles": miles, "Distance_Units": "track-miles"},
    ]
    rows += [
        {"Infrastructure_Type": "Substation",
         "Cost_USD": med["substation"] * infra_mult * USD_PER_MUSD,
         "Lifespan_Years": SUBSTATION_LIFESPAN_YEARS,
         "Distance_Miles": None, "Distance_Units": None}
        for _ in range(n_subs)
    ]
    return pl.DataFrame(rows)


def _build_fleet_tables(sims_list, loco_type, loco_cost):
    """Build the ``loco_pool`` + ``consist_plan`` tables spanning the full plan.

    Enumerates every locomotive in every plan train (``sims_list`` order == the
    ``SpeedLimitTrainSimVec`` index) so the metrics calculator can attribute
    purchases and energy per train. As an approximation the whole variant fleet is
    priced/typed as ``loco_type`` (for the BEL variant this also prices the single
    per-train diesel unit as BEL); the itemized report is illustrative, not a
    costing study.
    """
    plan = _load_full_consist_plan()
    pool_rows, cp_rows = [], []
    loco_id = 0
    for idx, ts in enumerate(sims_list):
        n_locos = len(ts.to_pydict()["loco_con"]["loco_vec"])
        train = plan[idx] if idx < len(plan) else {}
        train_id = str(train.get("Train_ID", idx))
        train_type = train.get("Train_Type", "Manifest")
        for _ in range(n_locos):
            loco_id += 1
            pool_rows.append({
                "Locomotive_ID": loco_id, "Locomotive_Type": loco_type,
                "Cost_USD": float(loco_cost),
                "Lifespan_Years": _defaults.LOCO_LIFESPAN,
            })
            cp_rows.append({
                "Locomotive_ID": loco_id, "Train_ID": train_id,
                "TrainSimVec_Index": idx, "Train_Type": train_type,
            })
    return pl.DataFrame(pool_rows), pl.DataFrame(cp_rows)


def _year_scenario(scenario_year, sims, loco_pool, consist_plan, region_elec_j,
                   energy_lvl, emissions_factors, simulation_days, infra_df):
    """One year's ScenarioInfo for the itemized-cost rollout over the full plan.

    The full consist plan stands in for one week of traffic (``simulation_days`` =
    ``PLAN_DAYS`` / demand multiplier); the metrics calculator annualizes the
    per-sim diesel and the ``refuel_sessions`` electricity by ``365.25 /
    simulation_days`` so both scale consistently. ``region_elec_j`` maps each
    grid-emissions region to its share of the plan's grid electricity (joules, week
    total); one ``refuel_sessions`` node per region lets the calculator price and
    (via ``emissions_factors``) attribute emissions by region.
    """
    mult = ENERGY_LEVELS[energy_lvl]
    # Central ("med") prices are the AEO2026 trajectory at this rollout year; the
    # gap multipliers widen low/high. Electricity is $/kWh here.
    diesel_price = _aeo_diesel_price(scenario_year) * mult["diesel"]
    elec_price = _aeo_elec_price(scenario_year) * mult["elec"]
    regions = list(region_elec_j.keys())
    # Electricity priced per grid-region node; diesel priced once (mean is used).
    # calculate_energy_cost multiplies electricity kWh by Price/100, so the
    # electricity price is in CENTS/kWh; diesel is $/gallon.
    nodal_energy_prices = pl.DataFrame(
        [{"Node": r, "Fuel": "Electricity", "Price": elec_price * 100.0}
         for r in regions]
        + [{"Node": "corridor", "Fuel": "Diesel", "Price": diesel_price}]
    )
    refuel_sessions = pl.DataFrame({
        "Locomotive_ID": [1] * len(regions),
        "Locomotive_Type": [loco_pool.get_column("Locomotive_Type")[0]] * len(regions),
        "Fuel_Type": ["Electricity"] * len(regions),
        "Node": regions,
        "Refuel_Energy_J": [float(region_elec_j[r]) for r in regions],
    })
    return _mc.ScenarioInfo(
        sims=sims, simulation_days=simulation_days, annualize=True,
        scenario_year=scenario_year, loco_pool=loco_pool,
        consist_plan=consist_plan, refuel_facilities=None,
        refuel_sessions=refuel_sessions, emissions_factors=emissions_factors,
        nodal_energy_prices=nodal_energy_prices, count_unused_locomotives=True,
        train_planner_config=None, infrastructure_costs=infra_df,
    )


# Union of the metrics pulled by this workflow, rollout_demo (rollout.main ->
# metric_calculator.main default list), and sim_manager_demo; metrics.main drops
# any that evaluate to null (e.g. Count_Refuelers when none are modeled here).
_ANNUAL_METRICS = [
    ("Meet_Pass_Events", "count"),
    ("Freight_Moved", "million tonne-mi"),
    ("Freight_Moved", "million tonne-km"),
    ("Freight_Moved", "car-miles"),
    ("Freight_Moved", "cars"),
    ("GHG", "tonne CO2-eq"),
    ("Count_Locomotives", "assets"),
    ("Count_Refuelers", "assets"),
    ("Count_Infrastructure", "assets"),
    ("Energy_Costs", "USD"),
    ("Energy", "MJ"),
    ("Energy_Per_Freight_Moved", "kWh per car-mile"),
    ("Distance_Energized", "track-miles"),
]


def _variant_metrics_df(tag, loco_type, loco_cost_fn, energy, plan_result, built_subs,
                        region_frac, *, arch, demand_lvl, demand_mult, infra_lvl,
                        infra_mult, energy_lvl):
    """Run the metrics calculator for ONE simulated variant at ONE economic point
    (infra cost x energy price); return its tidy long-format metrics cross-joined
    with a single-row provenance frame.

    ``energy`` is the walked-sim bundle from ``_run_full_plan`` (dual-mode, BEL, or
    conventional diesel) -- the SAME cached sim is re-priced across the economic
    sweep, so infra/energy never re-run the simulation. Catenary variants pass their
    infrastructure (scaled by ``infra_mult``) + built substations; the diesel
    baseline (``tag == "diesel"``) has no infrastructure.
    """
    mult = ENERGY_LEVELS[energy_lvl]
    ps = _plan_summary()
    rollout_years = [METRICS_START_YEAR + k for k in range(METRICS_ROLLOUT_YEARS)]
    emissions_factors = _mc.interpolate_emissions_factors_cambium(
        [GRID_REGION_WEST, GRID_REGION_EAST], rollout_years, CAMBIUM_SCENARIO)
    infra_df = (None if tag == "diesel"
                else _infrastructure_costs_df(plan_result, built_subs, arch,
                                              infra_mult))
    # Full plan == one week of traffic; the demand multiplier scales it by
    # shortening the represented window (annualization = 365.25 / simulation_days),
    # scaling sim diesel and refuel electricity together.
    simulation_days = PLAN_DAYS / max(demand_mult, 1e-9)
    n_locos = int(round(ps["total_locos"] * demand_mult))
    loco_cost_ref = float(loco_cost_fn(METRICS_START_YEAR))  # year-1 purchase cost
    _log(f"METRICS [{tag}] {arch}/{demand_lvl}/infra-{infra_lvl}/energy-{energy_lvl}: "
         f"{METRICS_ROLLOUT_YEARS}-yr rollout of {ps['n_trains']} trains "
         f"(loco ${loco_cost_ref/1e6:.2f}M @{METRICS_START_YEAR})")
    sims = alt.SpeedLimitTrainSimVec(list(energy["sims"]))
    # Consist plan is year-invariant; the loco pool's Cost_USD is set per rollout
    # year so purchases are priced at their year of purchase in the DCF rollover.
    loco_pool_base, consist_plan = _build_fleet_tables(
        energy["sims"], loco_type, loco_cost_ref)
    region_elec_j = _split_electricity(energy, region_frac)
    scenarios = [
        _year_scenario(
            year, sims,
            loco_pool_base.with_columns(
                pl.lit(float(loco_cost_fn(year))).alias("Cost_USD")),
            consist_plan, region_elec_j, energy_lvl, emissions_factors,
            simulation_days, infra_df)
        for year in rollout_years
    ]
    metrics = _mc.main(scenarios, annual_metrics=_ANNUAL_METRICS,
                       calculate_multiyear_metrics=True)
    # Cross-join onto a single-row provenance frame so every metric row carries its
    # full context (variant, architecture, demand/infra/energy case, prices, buildout).
    params = pl.DataFrame([{
        "variant": tag,
        "loco_type": loco_type,
        "loco_cost_usd": loco_cost_ref,
        "architecture": arch,
        "demand_level": demand_lvl,
        "demand_multiplier": demand_mult,
        "infra_level": infra_lvl,
        "infra_multiplier": infra_mult,
        "energy_level": energy_lvl,
        "diesel_price_usd_per_gal": _aeo_diesel_price(METRICS_START_YEAR) * mult["diesel"],
        "elec_price_usd_per_kwh": _aeo_elec_price(METRICS_START_YEAR) * mult["elec"],
        "discount_rate": DISCOUNT_RATE,
        "rollout_years": METRICS_ROLLOUT_YEARS,
        "start_year": METRICS_START_YEAR,
        "n_trains": ps["n_trains"],
        "n_locos": n_locos,
        "energized_track_miles": (plan_result.energized_track_miles or 0.0)
        if plan_result is not None else 0.0,
        "num_substations": len(built_subs),
    }])
    return metrics.join(params, how="cross")


# ---------------------------------------------------------------------------
# Per-run serialization: every metrics-calculator call gets its own folder with
# the optimization outputs, the (catenary-)network + substation locations (so its
# map can be reconstructed later), the metrics, and the concatenated per-train
# simulation time series -- all as Parquet.
# ---------------------------------------------------------------------------
def _plot_electrified_track():
    """Import (once) the sibling plotter, adding this script's dir to sys.path."""
    try:
        from plot_electrified_network import plot_electrified_track
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from plot_electrified_network import plot_electrified_track
    return plot_electrified_track


def _network_links_df(net) -> pl.DataFrame:
    """Flatten a network's links (with any energized ``cat_power_limits``) to a
    per-link dataframe: midpoint lat/lon, length, max grade, energized meters."""
    rows = []
    for L in net.to_pydict()[1][1:]:
        cpl = L.get("cat_power_limits") or []
        e_m = sum((c.get("offset_end_meters", 0.0) - c.get("offset_start_meters", 0.0))
                  for c in cpl)
        mp = midpoint(L)
        rows.append({
            "idx_curr": L["idx_curr"],
            "length_meters": L["length_meters"],
            "max_grade": GRADE_BY_LINK.get(L["idx_curr"], 0.0),
            "energized_meters": e_m,
            "n_cat_intervals": len(cpl),
            "lat": mp[0] if mp else None,
            "lon": mp[1] if mp else None,
        })
    return pl.DataFrame(rows)


def _substations_df(built_ids) -> pl.DataFrame:
    """Candidate + existing substation locations with a ``built`` flag so a map can
    be reconstructed later. ``built_ids`` = the ids the optimizer selected."""
    built = set(built_ids or [])
    return pl.DataFrame([{
        "id": str(s.id),
        "lat": s.lat,
        "lon": s.lon,
        "existing": bool(s.existing),
        "built": bool(s.existing) or (s.id in built),
        "max_power_watts": s.max_power_watts,
        "grid_emissions_region": s.grid_emissions_region,
    } for s in substations])


def _opt_outputs_df(result) -> pl.DataFrame:
    """Energized sub-link intervals from an optimization result, each row carrying
    the plan's scalar summary (status / costs / miles) for a self-describing table."""
    intervals = (result.energized_link_intervals or {}) if result is not None else {}
    rows = [
        {"link_idx": li, "offset_start_meters": a, "offset_end_meters": b,
         "miles": (b - a) * _utils.MI_PER_M}
        for li, ivs in intervals.items() for a, b in ivs
    ]
    df = pl.DataFrame(rows) if rows else pl.DataFrame(schema={
        "link_idx": pl.Int64, "offset_start_meters": pl.Float64,
        "offset_end_meters": pl.Float64, "miles": pl.Float64})
    summary = pl.DataFrame([{
        "status": str(result.status).split(".")[-1] if result is not None else "none",
        "energized_track_miles": (result.energized_track_miles or 0.0)
        if result is not None else 0.0,
        "total_cost_musd": (result.total_cost or 0.0) if result is not None else 0.0,
        "annual_opex_musd": (result.annual_opex_cost or 0.0)
        if result is not None else 0.0,
        "lifecycle_musd": (result.lifecycle_cost or 0.0) if result is not None else 0.0,
        "num_substations": len(result.built_substation_ids)
        if result is not None else 0,
    }])
    return df.join(summary, how="cross")


def _train_timeseries_df(sims) -> pl.DataFrame:
    """Concatenate every walked sim's ``to_dataframe()`` with ``train_id`` /
    ``train_index`` columns (the plan train id when available, else the index)."""
    plan = _load_full_consist_plan()
    parts = []
    for idx, ts in enumerate(sims):
        tid = str(plan[idx]["Train_ID"]) if idx < len(plan) else str(idx)
        parts.append(ts.to_dataframe().with_columns(
            pl.lit(tid).alias("train_id"), pl.lit(idx).alias("train_index")))
    return pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()


def _substation_load_series_df(sims, departure_seconds, substation_meta):
    """Sparse per-substation catenary load time series, summed over all trains.

    Each train's :func:`en.substation_load_curves` (power vs. its own
    dispatch-relative time) is shifted to the common wall-clock by its actual
    departure time, binned to ``SIM_SAVE_INTERVAL`` seconds, and summed across
    every train and locomotive feeding each substation. Returns long-format rows
    ``time_seconds`` / ``substation_id`` / ``power_watts`` with zero-power bins
    excluded (a sparse series per substation).
    """
    bin_s = float(SIM_SAVE_INTERVAL)
    acc: dict = {}  # (bin_time_seconds, substation_id) -> summed watts
    for ts, dep_s in zip(sims, departure_seconds):
        curves = en.substation_load_curves(ts, substation_meta)
        cols = [c for c in curves.columns
                if c.startswith("load_watts__") and c != "load_watts__total"]
        if not cols:
            continue
        rel_t = curves.get_column("time_seconds").to_list()
        for c in cols:
            sub_id = c[len("load_watts__"):]
            watts = curves.get_column(c).to_list()
            for t_rel, w in zip(rel_t, watts):
                if not w:
                    continue
                t_bin = round((float(dep_s) + float(t_rel)) / bin_s) * bin_s
                key = (t_bin, sub_id)
                acc[key] = acc.get(key, 0.0) + float(w)
    rows = [{"time_seconds": t, "substation_id": s, "power_watts": w}
            for (t, s), w in sorted(acc.items()) if w]
    if not rows:
        return pl.DataFrame(schema={"time_seconds": pl.Float64,
                                    "substation_id": pl.Utf8,
                                    "power_watts": pl.Float64})
    return pl.DataFrame(rows)


def _stationary_charging_df(refuel_sessions) -> pl.DataFrame:
    """Train-planner-scheduled stationary charging sessions as start/end time,
    total energy, and (constant) charger power. ``refuel_sessions`` is the
    :data:`refueling.REFUEL_SESSION_SCHEMA` frame from
    :func:`run_speed_limit_train_sims`; grid-side ``Refuel_Energy_J`` and the
    ``Refueler_J_Per_Hr`` charge rate are converted to kWh / kW."""
    if refuel_sessions is None or refuel_sessions.height == 0:
        return pl.DataFrame(schema={
            "locomotive_id": pl.UInt32, "locomotive_type": pl.Utf8,
            "node": pl.Utf8, "fuel_type": pl.Utf8,
            "start_hr": pl.Float64, "end_hr": pl.Float64,
            "total_kwh": pl.Float64, "peak_kw": pl.Float64})
    return refuel_sessions.select(
        pl.col("Locomotive_ID").alias("locomotive_id"),
        pl.col("Locomotive_Type").alias("locomotive_type"),
        pl.col("Node").alias("node"),
        pl.col("Fuel_Type").alias("fuel_type"),
        pl.col("Refuel_Start_Time_Hr").alias("start_hr"),
        pl.col("Refuel_End_Time_Hr").alias("end_hr"),
        (pl.col("Refuel_Energy_J") / 3.6e6).alias("total_kwh"),
        (pl.col("Refueler_J_Per_Hr") / 3.6e6).alias("peak_kw"),
    )


def _serialize_variant_run(folder, *, tag, net, result, metrics_df, sims,
                           feed_distance_miles, render_pdf, plan_label,
                           departure_seconds=None, substation_meta=None,
                           refuel_sessions=None):
    """Write one run's outputs to ``folder`` as Parquet: optimization outputs
    (catenary variants only), the network + substation locations (+ a network YAML
    for later PDF rebuilds), the metrics, and the concatenated per-train sim series.
    Catenary variants also get a sparse per-substation dynamic load series; the BEL
    variant additionally gets the planner-scheduled stationary charging sessions.
    Renders the buildout PDF for catenary variants."""
    folder.mkdir(parents=True, exist_ok=True)
    built_ids = result.built_substation_ids if result is not None else []
    if result is not None:  # diesel baseline has no optimization to serialize
        _opt_outputs_df(result).write_parquet(folder / "optimization.parquet")
    _network_links_df(net).write_parquet(folder / "network_links.parquet")
    _substations_df(built_ids).write_parquet(folder / "substations.parquet")
    try:
        net.to_file(str(folder / "network.yaml"))
    except Exception as exc:  # noqa: BLE001 - network file is a convenience artifact
        _log(f"  network.yaml write FAILED for {folder.name}: {exc}")
    metrics_df.write_parquet(folder / "metrics.parquet")
    _train_timeseries_df(sims).write_parquet(folder / "train_timeseries.parquet")
    if substation_meta is not None and departure_seconds is not None:
        try:
            _substation_load_series_df(
                sims, departure_seconds, substation_meta).write_parquet(
                    folder / "substation_load_curves.parquet")
        except Exception as exc:  # noqa: BLE001 - keep pipeline on export failure
            _log(f"  substation load curves FAILED for {folder.name}: {exc}")
    if refuel_sessions is not None:
        try:
            _stationary_charging_df(refuel_sessions).write_parquet(
                folder / "stationary_charging.parquet")
        except Exception as exc:  # noqa: BLE001 - keep pipeline on export failure
            _log(f"  stationary charging FAILED for {folder.name}: {exc}")
    if render_pdf and result is not None:
        try:
            _plot_electrified_track()(
                network, result=result, substations=substations,
                out_path=str(folder / "map.pdf"),
                substation_feed_distance_miles=feed_distance_miles,
                title=f"Optimized catenary -- {SIM_ORIGIN} to {SIM_DESTINATION} "
                      f"({plan_label})")
        except Exception as exc:  # noqa: BLE001 - keep pipeline on plot failure
            _log(f"  map rendering FAILED for {folder.name}: {exc}")
    _log(f"  serialized [{tag}] -> {folder}")


if __name__ == "__main__":
    _log(f"Network: {len(links)} links, {total_miles:.1f} track miles")
    _log(f"results dir: {RESULTS_DIR}")
    _log(f"solver log={'on' if SOLVER_LOG else 'off'}, "
         f"time limit={SOLVER_TIME_LIMIT_SECONDS:.0f}s per run, "
         f"cores<= {_MAX_CORES}")
    # Full pipeline: per sweep scenario, optimize the dual-mode (unconstrained) and
    # BEL (max-gap) catenary layouts, then simulate + serialize three variants
    # (dual-mode, BEL, conventional diesel) each into its own folder with the
    # optimization outputs, network + substation locations, metrics, per-train sim
    # series, and (catenary variants) a buildout PDF.
    _t = time.perf_counter()
    run_pipeline()
    _log(f"pipeline complete ({time.perf_counter() - _t:.1f}s)")
    _log("done")
