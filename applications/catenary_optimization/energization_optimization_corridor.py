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
dual-mode (cost analysis) or all BEL-with-pantograph (100% battery-electric, no
diesel-assist; battery analysis). The demand sweep is a MULTIPLIER on that
annualized plan (central=1x,
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

# Receptive wire: braking energy the pack cannot take goes back through the
# pantograph to the substation instead of the dynamic-brake grids. Every
# substation costed below is receptive hardware (AC transformers; MVDC VSC
# converters, whose price premium is justified by exactly this), so the default
# is on. False restores the pre-2026-09 behaviour. Returned energy counts in
# full for energy and emissions and at EXPORT_CREDIT_FRACTION of the purchase
# price for cost. The DoE's `build_layouts` overrides both from its policy.
CATENARY_RECEPTIVE = True
CATENARY_EXPORT_CREDIT_FRACTION = 0.5


def _with_receptivity(loco):
    """Stamp `CATENARY_RECEPTIVE` on a locomotive's catenary interface."""
    d = loco.to_pydict()
    kind = next(iter(d["loco_type"]))
    cat = (d["loco_type"][kind] or {}).get("cat_int")
    if not isinstance(cat, dict) or bool(cat.get("receptive", False)) == CATENARY_RECEPTIVE:
        return loco
    cat["receptive"] = CATENARY_RECEPTIVE
    return alt.Locomotive.from_pydict(d)
NOMINAL_LINE_EFFICIENCY = 0.97    # ~3% average catenary I2R loss

# ---------------------------------------------------------------------------
# Shared network, substations, and capital costs
# ---------------------------------------------------------------------------
_NET_PATH = alt.resources_root() / "networks" / "Spokane_Lethbridge" / "spokane_lethbridge.msgpack"
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

# ---------------------------------------------------------------------------
# Energy price zones: regional prices carried on national / provincial trajectories
# ---------------------------------------------------------------------------
# The corridor's lat/lon price-zone boxes (WA, ID, BC, AB) name the electricity and
# diesel zones of metrics_inputs/Energy_Price_Zones.csv, and
# `metric_calculator.zone_energy_price` turns a zone into a price at a year (a 2025
# reference price scaled by the AEO2026 trajectory for US zones and the province's
# Canada's Energy Future 2026 trajectory for Canadian zones). Links, substations and
# terminals are all placed with the same boxes, so the optimizer's per-piece opex,
# the metrics rollout and the corridor DoE all price energy on one basis: catenary
# electricity where the wire is fed from, diesel as the mean of the terminal zones
# (a run-through train fuels at both ends).
PRICE_ZONES_PATH = (alt.resources_root() / "networks" / "Spokane_Lethbridge"
                    / "spokane_lethbridge_price_zones.csv")
PRICE_ZONES = pl.read_csv(PRICE_ZONES_PATH)
NODE_SEP = "|"  # pricing node: "<grid emissions region>|<electricity price zone>"


def zone_at(lat, lon, boundaries=None):
    """The first price-zone box (row of ``PRICE_ZONES``) containing (lat, lon); a
    null bound is open."""
    for b in (PRICE_ZONES if boundaries is None else boundaries).iter_rows(named=True):
        if b["Lat_Min"] is not None and lat < b["Lat_Min"]:
            continue
        if b["Lat_Max"] is not None and lat >= b["Lat_Max"]:
            continue
        if b["Lon_Min"] is not None and lon < b["Lon_Min"]:
            continue
        if b["Lon_Max"] is not None and lon >= b["Lon_Max"]:
            continue
        return b
    raise ValueError(f"({lat:.4f}, {lon:.4f}) falls in no price zone")


def grid_node(region, zone):
    return f"{region}{NODE_SEP}{zone}"


def node_region(node):
    return node.split(NODE_SEP)[0]


def node_zone(node):
    return node.split(NODE_SEP)[-1]


def composite_emissions(ef_by_region, nodes):
    """Cambium factors keyed by pricing node, copying each node's region's rows."""
    return pl.concat([
        ef_by_region.filter(pl.col("Node") == node_region(n)).with_columns(pl.lit(n).alias("Node"))
        for n in nodes
    ])


SUBSTATIONS_BY_ID = {str(s.id): s for s in substations}


def substation_node(sub):
    """Pricing node of a substation: its grid-emissions region plus the electricity
    zone its position falls in."""
    return grid_node(sub.grid_emissions_region,
                     zone_at(float(sub.lat), float(sub.lon))["Electricity_Price_Region"])


STATIONARY_CHARGING_NODE = substation_node(SUBSTATIONS_BY_ID["west_terminus"])  # Spokane
DEFAULT_CATENARY_NODE = substation_node(SUBSTATIONS_BY_ID["east_terminus"])  # Lethbridge


def _elec_zone_by_link():
    zones = {}
    for L in links:
        m = midpoint(L)
        zones[L["idx_curr"]] = (zone_at(m[0], m[1])["Electricity_Price_Region"]
                                if m is not None else None)
    known = [z for z in zones.values() if z is not None]
    fill = max(set(known), key=known.count) if known else None
    return {idx: (z if z is not None else fill) for idx, z in zones.items()}


# Electricity price zone of every link, by its midpoint (links without geometry take
# the corridor's most common zone).
ELEC_ZONE_BY_LINK = _elec_zone_by_link()
_TERMINAL_ZONES = None


def terminal_price_zones():
    """Node, Electricity_Zone, Diesel_Zone of the corridor's terminals (location file)."""
    global _TERMINAL_ZONES
    if _TERMINAL_ZONES is None:
        _TERMINAL_ZONES = _mc.location_price_zones(
            alt.import_locations(str(SIM_LOCATIONS_CSV)))
    return _TERMINAL_ZONES


def diesel_price_zones():
    return sorted(set(terminal_price_zones().get_column("Diesel_Zone").to_list()))


def zone_elec_price(zone, year):
    """Electricity price ($/kWh) of a zone at ``year``."""
    return _mc.zone_energy_price(zone, "Electricity", year) / 100.0


def corridor_diesel_price(year):
    """Diesel price ($/gal) at ``year``: mean over the terminals' diesel zones."""
    zones = diesel_price_zones()
    return sum(_mc.zone_energy_price(z, "Diesel", year) for z in zones) / len(zones)


def levelized_diesel_price(mult=1.0):
    """Discount-weighted corridor diesel price over the project lifespan ($/gal)."""
    return mult * _levelized_aeo_price(corridor_diesel_price, METRICS_START_YEAR,
                                       PROJECT_LIFESPAN_YEARS, DISCOUNT_RATE)


def levelized_elec_price_by_link(mult=1.0):
    """Discount-weighted electricity price over the project lifespan, per link, from
    the link's price zone ($/kWh) -- what the optimizer's per-piece opex sees."""
    by_zone = {
        z: _levelized_aeo_price(lambda y, _z=z: zone_elec_price(_z, y), METRICS_START_YEAR,
                                PROJECT_LIFESPAN_YEARS, DISCOUNT_RATE)
        for z in set(ELEC_ZONE_BY_LINK.values())
    }
    return {idx: mult * by_zone[z] for idx, z in ELEC_ZONE_BY_LINK.items()}


def corridor_mean_elec_price(prices_by_link):
    """Track-length-weighted mean of a per-link electricity price map ($/kWh)."""
    num = sum(p * LINK_MILES.get(idx, 0.0) for idx, p in prices_by_link.items())
    den = sum(LINK_MILES.get(idx, 0.0) for idx in prices_by_link)
    return num / den if den > 0 else 0.0

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
# WHOLE-SITE electrification: wire every track piece at an energized structure
# site (x == s) instead of choosing per piece. Collapses ~18k link-driven piece
# binaries onto ~2k site binaries and strengthens the LP relaxation, so the
# optimality gap CLOSES (publishable) instead of stalling near the weak per-piece
# bound. Applied to both the dual-mode and BEL layouts. Set False for the fine
# per-piece formulation (much larger optimality gap within a time limit).
ELECTRIFY_WHOLE_SITE = True
# Back-end direction exclusion: the whole-site optimizer wires every track at an
# energized site (full cost), but a parallel track/direction the fleet never
# charges from is wasted. When True, the STUDY (metrics) subtracts the marginal
# wire cost of energized links whose plan-wide catenary draw is ~0 (structure and
# substations, shared per site, are kept). Optimization is unchanged.
EXCLUDE_UNUSED_DIRECTION_WIRE = True
# Feed the direction exclusion INTO the buildout (optional, gated). When True, the
# zero-catenary-draw links found after simulating a layout are removed from the
# optimizer's candidate set and the layout is RE-OPTIMIZED (and re-simulated), so
# the exclusion factors into the NPV/objective rather than only the reported cost.
# A bounded fixed point: applied identically to EACH powertrain using its OWN
# realized draw (fair -- same method, each fleet's own information; no grade/regen
# heuristic). Only links a fleet never charged from are removed, so a BEL's SOC
# feasibility is preserved. Each extra iteration ~doubles that variant's heavy sim
# cost; the loop stops early once the excluded set stops growing. Requires
# EXCLUDE_UNUSED_DIRECTION_WIRE. False (default) keeps the pure back-end refund.
DIRECTION_EXCLUSION_IN_BUILDOUT = True
# Max extra optimize+simulate feedback rounds when the above is enabled.
DIRECTION_EXCLUSION_MAX_ITERS = 1

COMMON = dict(
    uniform_power_limit_watts=50.0e6,
    structure_spacing_miles=1.0,
    corridor_width_miles=0.05,
    electrify_whole_site=ELECTRIFY_WHOLE_SITE,
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


def bel_drain_by_link(traction_by_link=None, grade_multiplier=40.0):
    """Per-link single-locomotive traction DRAIN (kWh per corridor-mile).

    The relative per-link SHAPE comes from ``traction_by_link`` when supplied --
    the sim-derived per-track-mile traction intensity (grade/speed/mass aware,
    e.g. ``first_pass_densities`` output) -- else from a grade proxy
    ``1 + grade_multiplier * grade``. Either shape is normalized (length-weighted)
    so the corridor-average equals ``design_drain_kwh_per_mile()`` -- the DESIGN
    train's draw, not the fleet mean (see `BEL_DESIGN_DRAIN_PERCENTILE`)
    (Option B sizing), keeping the per-loco scale grounded while the shape reflects
    real physics. Traffic-independent (one locomotive traversing once).
    """
    if traction_by_link:
        shape = {idx: float(traction_by_link.get(idx, 0.0)) for idx in GRADE_BY_LINK}
    else:
        shape = {idx: 1.0 + grade_multiplier * g for idx, g in GRADE_BY_LINK.items()}
    wsum = sum(shape[idx] * LINK_MILES.get(idx, 0.0) for idx in shape)
    wmi = sum(LINK_MILES.get(idx, 0.0) for idx in shape)
    mean_shape = (wsum / wmi) if wmi > 0 else 1.0
    if mean_shape <= 0:
        mean_shape = 1.0
    return {idx: design_drain_kwh_per_mile() * s / mean_shape
            for idx, s in shape.items()}


def bel_charge_by_link(drain=None):
    """Per-link NET CHARGE banked per energized corridor-mile (kWh/mi, Tier 1).

    ``min(gross wire delivery - local draw, battery charge-acceptance)``, floored:
    steep track (high draw) banks less, and no link banks faster than the pack can
    accept -- both matched to ALTRIOS's physical power caps.
    """
    d_by = drain if drain is not None else bel_drain_by_link()
    return {
        idx: min(
            max(BEL_GROSS_CHARGE_KWH_PER_MILE - d, BEL_MIN_NET_CHARGE_KWH_PER_MILE),
            BEL_CHARGE_ACCEPT_KWH_PER_MILE,
        )
        for idx, d in d_by.items()
    }


def bel_regen_by_link(drain=None):
    """Per-link regen recovery per corridor-mile (kWh/mi).

    Modeled as a flat fraction (``BEL_REGEN_FRACTION``, measured from simulation)
    of the local drain -- ``max_grade`` carries no sign, so regen cannot be
    localized to specific downgrades; the flat fraction captures the average range
    extension. Feeds NET drain (drain - regen) in the gap/SOC constraints.
    """
    d_by = drain if drain is not None else bel_drain_by_link()
    return {idx: BEL_REGEN_FRACTION * d for idx, d in d_by.items()}


def bel_traction_annual_by_link(trains_per_day, grade_multiplier=40.0,
                                kwh_traction_per_train_mi=33.0):
    """Per-link ANNUAL traction (at-the-wheel) energy per track-mile (kWh).

    The traffic-scaled quantity behind the BEL round-trip-loss opex (Item 3); it
    mirrors the traction term of ``energy_densities`` but is NOT divided by the
    grid->wheel chain (it is wheel energy, the amount that would pass through the
    battery).
    """
    annual_trains = trains_per_day * 365
    return {
        idx: annual_trains * kwh_traction_per_train_mi * (1.0 + grade_multiplier * g)
        for idx, g in GRADE_BY_LINK.items()
    }


def build_bel_opex_fn(trains_per_day, elec_price):
    """Per-piece annual net operating-cost fn ($M/yr) for a battery-electric fleet.

    Unlike the dual-mode opex (electricity minus displaced diesel), the BEL opex is
    the round-trip battery loss avoided by drawing a piece's traction energy
    straight from the wire -- a saving proportional to the piece's energy, which is
    what makes "prioritize highest energy/mile" the correct objective gradient.
    """
    base = en.make_bel_piece_opex_fn(
        traction_kwh_per_track_mile_by_link=bel_traction_annual_by_link(trains_per_day),
        electricity_price_usd_per_kwh=elec_price,
        battery_roundtrip_efficiency=BEL_ROUNDTRIP_EFFICIENCY,
    )
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
# Nameplate pack energy (kWh) overriding the ALTRIOS default (2.4 MWh); applied to
# the sim loco in ``_bel_with_cat_loco`` and used to size the optimizer gap.
SIM_BEL_BATTERY_CAPACITY_KWH = 9600.0   # 9.6 MWh nameplate
# Usable SOC span (default RES min_soc=0.05, max_soc=0.95 -> 0.90 usable).
SIM_BEL_SOC_WINDOW = 0.90
BEL_USABLE_BATTERY_KWH = SIM_BEL_BATTERY_CAPACITY_KWH * SIM_BEL_SOC_WINDOW  # kWh
# Per-locomotive traction draw, regen, and charge rates RECONCILED against a
# dual-mode simulation of representative plan trains over this corridor (per-loco
# electrical traction output; avg speed ~32 mph on the 3.28 MW-rated loco):
#   gross draw ~49 kWh/mi/loco (24 light .. 62 heavy), regen ~9% of draw,
#   5 MW pantograph @ 32 mph -> ~156 kWh/mi gross charge capability.
BEL_AVG_KWH_PER_MILE = 49.0       # measured avg per-locomotive traction draw (kWh/mi)
# The gap constraint is a FEASIBILITY constraint, so it has to hold for the train
# that decides feasibility, not the average one. Sizing it on the fleet mean is
# what let the optimizer leave gaps no heavy train can cross:
#
#   7,344 kWh budget / 49 kWh per loco-mile (mean)  = 150 miles permitted
#   7,344 kWh budget / 66 kWh per loco-mile (heavy) = 111 miles actual range
#
# and every gap between those is a train that needs a diesel helper. Measured
# layouts left gaps of 110-208 miles, and the helper rate tracked that single
# number almost perfectly: 110 mi -> 0 helpers, 143 -> 30, 208 -> 41 of ~80.
#
# Before 2026-07-30 this did not arise because the constraint was a flat
# `max_energized_gap_miles` of 40, derived from a 190 kWh/loco-mile assumption
# that exceeded what a 3.28 MW locomotive can draw in a mile at 32 mph (~102
# kWh). The energy-based gap that replaced it is better physics -- wire where
# the pack drains fastest rather than on a flat ruler -- but it was recalibrated
# to a mean where the old number had been an (impossible) worst case.
#
# Drain scales with how much train each locomotive is pulling, so the design
# figure is the mean scaled by the ratio of the design train's tons-per-
# locomotive to the fleet's. Same percentile idea as
# `DESIGN_TRAIN_POWER_PERCENTILE`, applied to range instead of substation
# spacing.
# Base for the design figure: the drain MEASURED from the all-catenary first
# pass, not the 49 the opex model carries. Summing the fully-energized per-link
# pantograph densities over the calibrated samples and dividing by the
# locomotive-miles that produced them gives 38 kWh per locomotive-mile
# (2026-09-16, Spokane-Lethbridge, 104 trains/week, 3.48 locos/train).
#
# Using 49 as the base instead double-counts conservatism -- it is already 29%
# above measurement -- and at p90 that compounded to 90 kWh/loco-mile, an 81-mile
# permitted gap, and a MILP with no feasible buildout on any architecture.
BEL_MEASURED_MEAN_KWH_PER_MILE = 38.0
BEL_DESIGN_DRAIN_PERCENTILE = 90.0


def design_drain_kwh_per_mile():
    """Per-locomotive drain of the design train, for sizing the range gap."""
    import numpy as _np

    tons, locos, per_loco = 0.0, 0, []
    for s in range(len(CONSIST_PLAN_PATHS)):
        for t in _load_full_consist_plan(sample=s):
            w = float(t.get("Train_Weight_tons") or 0.0)
            n = int(t.get("n_locos") or 0)
            if w > 0 and n > 0:
                tons += w
                locos += n
                per_loco.append(w / n)
    if not per_loco or locos == 0:
        return BEL_MEASURED_MEAN_KWH_PER_MILE
    mean_tpl = tons / locos
    design_tpl = float(_np.percentile(per_loco, BEL_DESIGN_DRAIN_PERCENTILE))
    return BEL_MEASURED_MEAN_KWH_PER_MILE * design_tpl / mean_tpl
BEL_REGEN_FRACTION = 0.09         # regen recovered as a fraction of gross draw
BEL_RESERVE_FRACTION = 0.15       # withheld reserve (SOC protection + accessories)
BEL_MAX_GAP_MILES = en.max_gap_miles_from_battery(
    BEL_USABLE_BATTERY_KWH,
    BEL_AVG_KWH_PER_MILE,
    reserve_fraction=BEL_RESERVE_FRACTION,
)
# ENERGY-based range parameterization (replaces the flat per-mile gap for the BEL
# layout). ``BEL_MAX_GAP_KWH`` is the usable pack energy; the optimizer bounds
# each unelectrified window by its per-link DRAIN energy (``bel_drain_by_link``)
# so wire is required where the battery depletes fastest, not on a flat distance.
BEL_MAX_GAP_KWH = en.max_gap_kwh_from_battery(
    BEL_USABLE_BATTERY_KWH, reserve_fraction=BEL_RESERVE_FRACTION,
)
BEL_ROUNDTRIP_EFFICIENCY = 0.90   # pack round-trip eff (RES eta curve, one-way^2)
# Charge model matched to ALTRIOS's physical caps. Per-mile charge is bounded by
# BOTH the pantograph power (5 MW, gross wire delivery) and the battery
# charge-acceptance = RES pwr_in_max_watts (9.6 MW = 1C; the pack's own limit,
# no longer the traction rating), each converted to kWh/mi at the representative
# corridor speed. NET charge = min(gross - draw, charge-acceptance), floored.
BEL_REPRESENTATIVE_SPEED_MPH = 32.0     # measured avg corridor speed
# CHARGE acceptance of the BEL pack. This used to be the RES's single
# `pwr_out_max` (3.281 e6), which is the locomotive's 4,400 hp TRACTION rating
# and not a battery property at all. The sim now carries an explicit
# `res.pwr_in_max` (run_sweep.BEL_CHARGE_C_RATE = 1C on the 9.6 MWh pack), so
# this must track it or the layout sizing and the simulation disagree about how
# fast a train refills between energized sections.
SIM_BEL_RES_PWR_IN_MAX_WATTS = 9.6e6  # 1C on the 9.6 MWh pack
SIM_BEL_RES_PWR_OUT_MAX_WATTS = 3.281e6  # traction rating; discharge only


def _pwr_w_to_kwh_per_mi(power_w, mph):
    """Energy per mile (kWh) a constant power (W) delivers at speed ``mph``."""
    return power_w / (mph * 0.44704) * 1609.344 / 3.6e6


# 5.0e6 matches SIM_BEL_CAT_PWR_MAX_WATTS (defined with the sim config below).
BEL_GROSS_CHARGE_KWH_PER_MILE = _pwr_w_to_kwh_per_mi(
    5.0e6, BEL_REPRESENTATIVE_SPEED_MPH)
BEL_CHARGE_ACCEPT_KWH_PER_MILE = _pwr_w_to_kwh_per_mi(
    SIM_BEL_RES_PWR_IN_MAX_WATTS, BEL_REPRESENTATIVE_SPEED_MPH)
BEL_MIN_NET_CHARGE_KWH_PER_MILE = 20.0
BEL_AVG_NET_CHARGE_KWH_PER_MILE = max(
    min(BEL_GROSS_CHARGE_KWH_PER_MILE - BEL_AVG_KWH_PER_MILE,
        BEL_CHARGE_ACCEPT_KWH_PER_MILE),
    BEL_MIN_NET_CHARGE_KWH_PER_MILE,
)
# Tier-0 recharge: minimum energized run (miles) sized so a run refills
# ``BEL_RECHARGE_FRACTION`` of the usable pack at the average net charge rate
# (fraction < 1 since not every gap fully depletes the pack). Combined (max) with
# the anti-sliver minimum run.
BEL_RECHARGE_FRACTION = 0.5
BEL_RECHARGE_MILES = en.min_recharge_miles_from_battery(
    BEL_USABLE_BATTERY_KWH, BEL_AVG_NET_CHARGE_KWH_PER_MILE,
    recharge_fraction=BEL_RECHARGE_FRACTION, reserve_fraction=BEL_RESERVE_FRACTION,
)
# When True the BEL layout uses the charge-credited (Tier 1) gap family: an
# energized bucket credits only (charge + drain), so short slivers on steep track
# no longer count as a full recharge. Under WHOLE-SITE it solves to PROVEN OPTIMAL
# fast (~23s at 2mi buckets); the recharge physics is then also carried by the
# Tier-0 min-run below. False (default) uses the fast ``sum u>=1`` energy gap (any
# single energized bucket resets the gap; totally unimodular) + Tier-0 recharge.
BEL_RECHARGE_TIER1 = False
# When True the BEL layout uses the explicit SOC-recursion (Tier 2) range model
# instead of the covering families: a continuous per-bucket state of charge with a
# curtailment slack (linear, no big-M) that tracks charge/drain/regen exactly and
# forbids depletion. The MOST FAITHFUL model, and under WHOLE-SITE it solves to
# PROVEN OPTIMAL fast (~14s at 2mi buckets, resolution-robust). Recommended for a
# rigorous study; it needs MORE wire than the optimistic default (slivers do not
# fully recharge), and it supersedes the Tier-0 recharge min-run (only the
# anti-sliver floor is applied when on). False uses the covering gap above.
BEL_SOC_STATE_MODEL = True

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
# either all dual-mode (cost analysis) or all BEL-with-pantograph (100% BEL)
# (battery analysis).
PLAN_DAYS = 7.0                     # consist plan treated as one representative week
ANNUAL_SCALER = 365.25 / PLAN_DAYS  # week -> year

# Warm-up and cool-down, in days either side of the measured week. The plan is a
# representative week, so the warm-up traffic IS that week: its last two days
# are replayed before it and its first two after, which keeps day-of-week
# structure and, critically, does not invent demand that would change the
# calibrated density.
#
# These trains are dispatched and walked so they congest the corridor, but their
# energy is excluded from the per-link densities -- only the measured week is
# summed, and ANNUAL_SCALER then annualizes exactly seven days.
WARMUP_DAYS = 2.0
COOLDOWN_DAYS = 2.0
# Demand levels as SAMPLE SETS, not multipliers. Each entry lists which of the
# four independent consist-plan samples the level comprises; a level's energy,
# emissions, cost and load are the sums over its samples, and each sample is
# dispatched and walked as its own meet-pass problem.
#
# A scalar multiplier (the previous design) scales one week's energy but cannot
# change which trains are on the corridor or when, so it produces no new
# consists, no new departure profile, and a load curve that is just the 1x curve
# stretched. Distinct samples give genuinely different weeks, which is what the
# grid-impact question needs.
#
# What this deliberately does NOT model is the congestion of 2x or 4x traffic:
# the samples never share a dispatch, so no train in S2 ever meets a train in
# S1. Measured on this corridor, 4x traffic in one dispatch is an unbounded
# meet-pass queue. These levels answer economics and grid impact at higher
# throughput, and results must be caveated as not representing higher congestion.
def _calibrated_sample_count() -> int:
    """How many independent weeks ONE calibrated corridor-week takes.

    The sizing rule caps how many trains may share a single meet-pass
    problem, so a corridor whose observed volume exceeds that cap is carried
    by several samples: Spokane-Lethbridge's 104 trains/week are two weeks of
    52. Read from the corridor inputs rather than restated here, so the
    demand levels move with the rule instead of drifting from it -- when this
    was hard-coded at one sample, `1x` meant HALF the calibrated volume and
    nothing in the output said so.
    """
    meta = (Path(__file__).resolve().parent.parent / "corridor_inputs"
            / "corridor_meta_spokane_lethbridge.csv")
    if not meta.exists():
        return 1
    frame = pl.read_csv(meta)
    if "n_samples" not in frame.columns:
        return 1
    return max(1, int(frame["n_samples"][0]))


_CALIBRATED_SAMPLES = _calibrated_sample_count()
# 1x IS the corridor's calibrated volume, whatever number of samples that
# takes; 2x and 4x are that many again, drawn as further independent weeks.
DEMAND_LEVELS = {
    f"{m}x": tuple(range(m * _CALIBRATED_SAMPLES)) for m in (1, 2, 4)
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
#   - FINDING (2026-09): at this corridor's design-train power (p95 consist,
#     ~29 MW) a 9 kV wire carries ~4.9 kA at its 6 kV floor and the voltage-drop
#     feed distance collapses to ~2.8 mi, so dc_9kv never builds. That is not the
#     0.67 voltage floor (the EN 50163 two-thirds convention, which the MVDC
#     literature keeps) nor the conductor (the 9 kV proposals use LIGHTER wire,
#     266 mm2/track, and reach ~100 km spacing with few-MW trains); it is the
#     train power. Kept in the design so the infeasibility is reported, not
#     assumed away. A 24 kV DC architecture (~1.2 kA) would be the fair MVDC
#     comparator. Sources: SNCF/UIC "Go for 9 kV" (2019); Sharifi et al. 2022,
#     IET Electr. Syst. Transp. (7.5-24 kV window).
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
# ("med") case is each price zone's 2025 reference price on its trajectory (AEO2026
# reference case for US zones, Canada's Energy Future 2026 for Canadian zones; see
# the price-zone section above and metric_calculator.zone_energy_price). The
# low/high cases WIDEN the gap by ENERGY_GAP_WIDTH, moving diesel and electricity in
# opposite directions: "low" barely favors electrification (cheaper diesel, dearer
# power), "high" strongly favors it. Values are MULTIPLIERS on the zone trajectory.
ENERGY_GAP_WIDTH = 1.0 / 3.0
ENERGY_LEVELS = {
    "low":  dict(diesel=1.0 - ENERGY_GAP_WIDTH, elec=1.0 + ENERGY_GAP_WIDTH),
    "med":  dict(diesel=1.0, elec=1.0),
    "high": dict(diesel=1.0 + ENERGY_GAP_WIDTH, elec=1.0 - ENERGY_GAP_WIDTH),
}

# Buildout-map PDFs are the most expensive per-cell serialization step, so render
# them only for a few representative cost cells (the reference and the two extremes
# that bracket electrification) instead of all 9. Every cell still writes its
# metrics/network/sim outputs. Set to None to render every cell.
PDF_RENDER_CELLS = {("med", "med"), ("low", "high"), ("high", "low")}


def _should_render_pdf(infra_lvl, energy_lvl):
    """Whether to render the buildout PDF for this (infra, energy) cost cell."""
    return PDF_RENDER_CELLS is None or (infra_lvl, energy_lvl) in PDF_RENDER_CELLS


def _aeo_price_at(df, year):
    """Nearest-year ``Price`` from an AEO price table, clamped to its year range."""
    yrs = df.get_column("Year")
    y = min(max(int(year), int(yrs.min())), int(yrs.max()))
    return float(df.filter(pl.col("Year") == y).get_column("Price")[0])


def _aeo_diesel_price(year):
    """AEO2026 national transportation diesel price ($/gal) at ``year``. Kept for
    reference; pricing now goes through the zone functions above."""
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
# First year of the cost rollout. Deliberately LATER than the simulation's own
# base analysis year: the simulated week is a physical result and does not move,
# while the investment it is priced against is assumed to begin in 2030. With a
# 20-year locomotive life that puts the fleet replacement in 2050.
#
# CAVEAT the outputs must carry: every input trajectory ends in 2050 -- the ATB
# battery series, the AEO diesel and electricity series, and the Cambium grid
# emissions factors. Each holds its 2050 value flat thereafter (verified, not
# assumed: none extrapolates or errors). So the last nine years of a 2030-2059
# horizon, 30% of it, run on frozen 2050 prices and grid carbon.
METRICS_START_YEAR = 2030
# Infrastructure asset lifespans (years) fed to the rollover engine.
STRUCTURE_LIFESPAN_YEARS = 50      # civil structures / foundations / poles
ENERGIZATION_LIFESPAN_YEARS = 40   # contact + messenger wire / OCS hardware
SUBSTATION_LIFESPAN_YEARS = 30     # traction substation / power electronics
# The itemized-cost report is built for the architecture, demand level, and
# energy prices below. The architecture matches the confirming simulation's
# 25 kV AC electrification (SIM_ELECTRIFICATION_TYPE); the demand level names
# which consist-plan samples it covers (see DEMAND_LEVELS).
ITEMIZED_ARCH = "ac_25kv"
ITEMIZED_DEMAND_LEVEL = "1x"
ITEMIZED_DEMAND_SAMPLES = DEMAND_LEVELS[ITEMIZED_DEMAND_LEVEL]
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
    """Fraction of catenary grid electricity by pricing node
    ``"<grid emissions region>|<electricity price zone>"``.

    Runs ``compute_line_losses`` on every dual-mode plan sim (``dual_mode_energy
    ["sims"]``), sums each feeding substation/district's grid-side energy across
    all trains, then groups by the substation's node (its grid-emissions region
    plus the price zone its position falls in, see `substation_node`). So a
    train's draw is priced where the wire under it is fed from. Falls back to the
    Lethbridge terminus node if the split is unavailable.
    """
    sims = (dual_mode_energy or {}).get("sims") or []
    if sims and substation_meta:
        by_node = {}
        for ts in sims:
            df = en.compute_line_losses(ts, substation_meta)
            df = df.filter(pl.col("district_id") != pl.lit("__total__"))
            for r in df.iter_rows(named=True):
                did = r["district_id"]
                if did is None or str(did) not in SUBSTATIONS_BY_ID:
                    continue
                node = substation_node(SUBSTATIONS_BY_ID[str(did)])
                by_node[node] = by_node.get(node, 0.0) + float(r["substation_energy_joules"])
        total = sum(by_node.values())
        if total > 0:
            return {n: v / total for n, v in sorted(by_node.items())}
    return {DEFAULT_CATENARY_NODE: 1.0}


def _split_electricity(energy, region_frac, stationary_j=0.0):
    """Split a variant's grid electricity across pricing nodes
    ``"<grid emissions region>|<electricity price zone>"``.

    Catenary charging follows ``region_frac`` (by feeding substation); train-planner
    stationary charging (``stationary_j`` grid joules, week total) is attributed to
    the Spokane terminus node (West region, Washington prices). Feeding both into the
    metrics calculator's electricity total lets it back out
    ``Electricity_Stationary_Charging`` (total grid electricity minus the sim's
    catenary grid draw) and price the energy by zone and attribute its emissions by
    region.
    """
    catenary_j = float(energy.get("grid_elec_j") or 0.0)
    region_elec = {r: catenary_j * f for r, f in region_frac.items()}
    if stationary_j:
        region_elec[STATIONARY_CHARGING_NODE] = (
            region_elec.get(STATIONARY_CHARGING_NODE, 0.0) + float(stationary_j))
    return region_elec or {DEFAULT_CATENARY_NODE: catenary_j}


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


MAX_SIM_CACHE = 6
# Bound the in-memory sim bundles (each a full walked plan) so repeated
# re-optimization cannot accumulate every distinct layout and exhaust memory;
# the disk bundle cache still allows reload.


def simulate_layout(net, traction_kind, fingerprint, label,
                    samples, location_map, sim_cache):
    """Walk the plan on ``net`` once per demand sample, and merge.

    Cached per (traction_kind, fingerprint, sample) so a level that shares
    samples with a smaller level -- 4x contains 2x contains 1x -- reuses
    their sims rather than re-walking them.
    """
    parts = []
    for s_ in samples:
        key = (traction_kind, fingerprint, s_)
        tag = f"{fingerprint}_s{s_}"
        cached = sim_cache.get(key)
        if cached is None:  # reload a bundle serialized before an earlier kill
            cached = _load_sim_bundle(traction_kind, tag)
        if cached is None:
            cached = _run_full_plan(net, location_map,
                                    f"{label} S{s_ + 1}", traction_kind,
                                    keep_sims=True, sample=s_)
            _save_sim_bundle(traction_kind, tag, cached)
        sim_cache[key] = cached
        parts.append(cached)
    while len(sim_cache) > MAX_SIM_CACHE * len(CONSIST_PLAN_PATHS):
        sim_cache.pop(next(iter(sim_cache)))  # FIFO evict oldest layout
    return _merge_plan_runs(parts)

def optimize_and_sim(label, traction_kind, opt_kwargs, samples,
                     location_map, sim_cache=None, derive_cache=None):
    """Optimize a layout and simulate it, optionally feeding the realized
    zero-catenary-draw exclusion back into the buildout as a bounded fixed
    point (gated by ``DIRECTION_EXCLUSION_IN_BUILDOUT``).

    Each round removes the links this fleet actually drew ~0 catenary from
    (its OWN realized draw -- no grade/regen heuristic) from the optimizer's
    candidate set and re-optimizes, so the exclusion informs the NPV/objective
    rather than only the reported cost. Only never-charged links are removed,
    so a BEL's SOC feasibility is preserved. Stops after
    ``DIRECTION_EXCLUSION_MAX_ITERS`` rounds or once the excluded set stops
    growing. A failed re-optimization keeps the prior (feasible) layout.

    Returns ``(result, sim_net, energy, excl, meta, subs, region_frac)``; all
    but ``result`` are ``None`` when the optimizer yields no usable layout.
    """
    if sim_cache is None:
        sim_cache = {}
    if derive_cache is None:
        derive_cache = {}
    denylist: list = []
    best = None
    n_iters = (DIRECTION_EXCLUSION_MAX_ITERS
               if DIRECTION_EXCLUSION_IN_BUILDOUT else 0)
    for it in range(n_iters + 1):
        try:
            r = run_opt(label, excluded_link_idxs=(denylist or None),
                        **opt_kwargs)
        except Exception as exc:  # noqa: BLE001
            if best is None:
                raise
            _log(f"  {label}: re-optimize failed ({exc}); keeping prior layout")
            break
        if r is None or r.network is None:
            break
        subs = _built_sub_objs(r)
        net, meta = en.apply_line_losses(
            r.network, subs,
            system_voltage_volts=SIM_SYSTEM_VOLTAGE_VOLTS,
            electrification_type=SIM_ELECTRIFICATION_TYPE,
            base_power_limit_watts=COMMON["uniform_power_limit_watts"])
        net = _calibrate_speeds(net)
        fp = _layout_fingerprint(r)
        e = simulate_layout(net, traction_kind, fp, label, samples,
                        location_map, sim_cache)
        derived = derive_cache.get((traction_kind, fp))
        if derived is None:
            region_frac = _region_fractions(e, meta)
            excl = (_excluded_energization_miles(
                        r, e["sims"], e.get("draw_by_link"))
                    if EXCLUDE_UNUSED_DIRECTION_WIRE else (0.0, []))
            derived = (region_frac, excl)
            derive_cache[(traction_kind, fp)] = derived
        region_frac, excl = derived
        best = (r, net, e, excl, meta, subs, region_frac)
        if it == n_iters:
            break
        new_excl = [li for li in excl[1] if li not in set(denylist)]
        if not new_excl:
            _log(f"  {label}: direction exclusion converged after "
                 f"{it} feedback iter(s)")
            break
        denylist = list(set(denylist) | set(excl[1]))
        _log(f"  {label}: feedback iter {it + 1}: dropping {len(new_excl)} "
             f"zero-draw links ({excl[0]:.1f} mi), re-optimizing "
             f"(denylist now {len(denylist)} links)")
    if best is None:
        return None, None, None, (0.0, []), None, None, None
    return best

def run_pipeline():
    """Full pipeline. The train SIMULATION depends only on the physical buildout,
    set by ARCHITECTURE x DEMAND: per (arch, demand) the NPV layout is optimized
    ONCE -- at reference (med) infrastructure cost + (med) energy prices -- both
    unconstrained (dual-mode target) and with the BEL max-gap constraint (battery
    target), and each of the three variants is simulated ONCE:

      1. ``dual_mode``  -- unconstrained layout, 100% dual-mode consists.
      2. ``bel``        -- gap-constrained layout, 100% BEL-with-cat consists; a
                           diesel helper is swapped in per train ONLY if a pure-BEL
                           consist cannot pull the grade (``_walk_plan_train``).
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
        f"{name}={len(smp)} sample(s), "
        f"{_level_plan_summary(smp)['trains_per_day']:.1f}tr/day"
        for name, smp in DEMAND_LEVELS.items()
    )
    print(f"  horizon {PROJECT_LIFESPAN_YEARS} yr | discount {DISCOUNT_RATE:.0%} | "
          f"PVAF {econ.pvaf():.2f} | consist plan: {ps['n_trains']} trains over "
          f"{PLAN_DAYS:.0f}d | demand: {demand_desc}")
    print("=" * 118)
    # First-pass full-consist-plan sims give the ANNUAL per-link per-track-mile
    # energy that feeds the min_cost opex (run once, cached, then multiplied by the
    # demand multiplier); else fall back to the illustrative grade-scaled constants.
    # One density pass PER SAMPLE. A demand level sums the samples it contains,
    # which is what makes 2x and 4x carry genuinely different weeks of traffic
    # rather than one week scaled up.
    sim_density_by_sample = {}
    if USE_SIM_DERIVED_DENSITIES:
        for _s in sorted({s_ for samples in DEMAND_LEVELS.values() for s_ in samples}):
            sim_density_by_sample[_s] = first_pass_densities(_s)
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
    # Derived post-processing (region fractions + direction-exclusion), cached by
    # (traction_kind, layout fingerprint) so cost scenarios sharing a buildout only
    # pay for it once.
    derive_cache = {}
    # Bound the in-memory sim bundles (each ~160 walked sims) so per-cell dual
    # re-optimization can't accumulate every distinct layout and exhaust memory;
    # the disk bundle cache still allows reload, and BEL stays alive via ``e_bel``.
    _MAX_SIM_CACHE = 6

    # Bound to this run's caches and location map; the loop itself is module
    # level so `build_layouts.py` can use the same refinement.
    def _sim(net, traction_kind, fingerprint, label, samples=(0,)):
        return simulate_layout(net, traction_kind, fingerprint, label, samples,
                               location_map, sim_cache)

    def _optimize_and_sim(label, traction_kind, opt_kwargs, samples=(0,)):
        return optimize_and_sim(label, traction_kind, opt_kwargs, samples,
                                location_map, sim_cache, derive_cache)

    # Reference cost point that determines the SIMULATED buildout. Infra cost and
    # energy price are swept later ONLY in the metrics accounting, so they never
    # change the layout being simulated (and thus never re-run the simulation).
    # The buildout is decided at the CENTRAL energy case: the discount-weighted
    # average zone price over the project lifespan (so lifetime opex sits
    # consistently alongside the annualized capital in the objective's PVAF).
    # Electricity is a price PER LINK from the link's zone, so a piece in Alberta is
    # judged at Alberta power; diesel is the corridor's terminal-zone mean.
    ref_diesel = levelized_diesel_price()
    ref_elec = levelized_elec_price_by_link()
    ref_infra_mult = INFRA_LEVELS["med"]

    for arch in ("ac_25kv", "ac_2x25", "dc_9kv"):
        # Feed distance (substation spacing) sized to the 95th-pct consist draw.
        sug = en.suggest_optimization_inputs(
            arch, design_train_power_watts=_design_train_power_watts())
        med = ARCH_MEDIUM_COSTS[arch]
        for demand_lvl, demand_samples in DEMAND_LEVELS.items():
            lvl = _level_plan_summary(demand_samples)
            trains_per_day = lvl["trains_per_day"]
            if USE_SIM_DERIVED_DENSITIES:
                diesel_by_link = _sum_link_densities(
                    sim_density_by_sample[s_][0] for s_ in demand_samples)
                elec_raw = _sum_link_densities(
                    sim_density_by_sample[s_][1] for s_ in demand_samples)
                elec_by_link = {
                    li: v / sug.substation_efficiency for li, v in elec_raw.items()
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
            # Reference cost point: fixes the BEL buildout and serves as the base
            # that per-cell energy-price multipliers scale from.
            ref_costs = cp.EnergizationOptCosts(
                structure_cost_per_mile=med["structure"] * ref_infra_mult,
                energization_cost_per_mile=med["energization"] * ref_infra_mult,
                substation_fixed_cost=med["substation"] * ref_infra_mult,
            )
            common_opt = dict(
                objective="min_cost", economics=econ,
                substation_feed_distance_miles=feed_mi,
                **COMMON,
            )
            # Variant-2 layout (BEL) is optimized ONCE at the reference cost point:
            # its energized mileage is pinned by the battery-range (max-gap)
            # constraint, so it is far less cost-sensitive than the dual-mode NPV
            # layout -- only RE-PRICED across the cost grid, not re-optimized.
            # ENERGY-based range gap (wire where the pack drains fastest),
            # round-trip-loss opex, energy-sized recharge min run. BEL drain shape
            # from the sim-derived per-link traction intensity when available.
            bel_drain = bel_drain_by_link(elec_raw if USE_SIM_DERIVED_DENSITIES
                                          else None)
            r_bel = None
            net_bel = subs_bel = e_bel = region_frac_bel = meta_bel = None
            excl_bel = (0.0, [])
            net_bel_out = None
            try:
                (r_bel, net_bel, e_bel, excl_bel, meta_bel, subs_bel,
                 region_frac_bel) = _optimize_and_sim(
                    f"{layout}/bel", "battery_electric",
                    dict(piece_opex_fn=build_bel_opex_fn(trains_per_day, ref_elec),
                         costs=ref_costs,
                         max_energized_gap_kwh=BEL_MAX_GAP_KWH,
                         gap_energy_kwh_per_mile_by_link=bel_drain,
                         # Charge densities feed BOTH the Tier-1 covering and the
                         # SOC (Tier-2) recursion; either needs them.
                         charge_energy_kwh_per_mile_by_link=(
                             bel_charge_by_link(bel_drain)
                             if (BEL_RECHARGE_TIER1 or BEL_SOC_STATE_MODEL) else None),
                         regen_energy_kwh_per_mile_by_link=bel_regen_by_link(bel_drain),
                         battery_soc_state_model=BEL_SOC_STATE_MODEL,
                         # SOC (Tier 2) tracks recharge exactly, so only the
                         # anti-sliver floor is needed; the covering tiers use the
                         # Tier-0 recharge run.
                         min_energized_length_miles=(
                             5.0 if BEL_SOC_STATE_MODEL
                             else max(5.0, BEL_RECHARGE_MILES)),
                         warmstart=True, **common_opt),
                    samples=demand_samples)
                if r_bel is not None and r_bel.network is None:
                    _log(f"BEL layout for {layout}: no incumbent "
                         f"(status={str(r_bel.status).split('.')[-1]}); skipping")
                    r_bel = None
            except Exception as exc:  # noqa: BLE001 - keep pipeline on infeasible gap
                _log(f"BEL layout FAILED for {layout}: {exc}")
                r_bel = None
            if r_bel is None:
                net_bel = subs_bel = e_bel = region_frac_bel = meta_bel = None
                excl_bel = (0.0, [])
                net_bel_out = None
            else:
                net_bel_out = _prune_unused_catenary(net_bel, excl_bel[1])

            # Train-planner charging pass (BEL only): stationary-charging sessions +
            # actual departure times to align load curves on a common wall-clock.
            # The dispatch schedule is traction-agnostic, so its departures are
            # reused for the dual-mode curves too; computed ONCE per block.
            sched_net = net_bel if r_bel is not None else net_diesel
            # One charging pass per demand sample, concatenated: stationary
            # charging is grid load and has to scale with the level's traffic
            # like everything else. Sim indices are offset per sample so the
            # departure map lines up with the concatenated sim vector.
            _refuel_parts, dep_by_index, _offset = [], {}, 0
            for _s in demand_samples:
                _rs, _dep = _planner_charge_pass(sched_net, location_map, _s)
                if _rs is not None and getattr(_rs, "height", 0):
                    _refuel_parts.append(_rs)
                for _k, _v in (_dep or {}).items():
                    dep_by_index[int(_k) + _offset] = _v
                _offset += len(_level_measured_plan((_s,)))
            refuel_sessions_bel = (
                pl.concat(_refuel_parts, how="diagonal_relaxed")
                if _refuel_parts else None)
            if r_bel is None:
                refuel_sessions_bel = None
            dep_seconds = _departure_seconds(dep_by_index, lvl["n_trains"])

            def _do_variant(tag, net, result, built_objs, energy, region_frac,
                            loco_type, cost_fns, render_pdf, *, scenario,
                            infra_lvl, infra_mult, energy_lvl,
                            substation_meta=None, refuel_sessions=None,
                            opt_infra_mult=None):
                try:
                    m = _variant_metrics_df(
                        tag, loco_type, cost_fns, energy, result, built_objs,
                        region_frac, arch=arch, demand_lvl=demand_lvl,
                        demand_samples=demand_samples, infra_lvl=infra_lvl,
                        infra_mult=infra_mult, energy_lvl=energy_lvl,
                        net_purchased=net, stationary_sessions=refuel_sessions,
                        opt_infra_mult=opt_infra_mult)
                    _serialize_variant_run(
                        RESULTS_DIR / f"{scenario}__{tag}", tag=tag, net=net,
                        result=result, metrics_df=m, sims=energy["sims"],
                        feed_distance_miles=feed_mi, render_pdf=render_pdf,
                        plan_label=f"{scenario} {tag}",
                        departure_seconds=dep_seconds,
                        substation_meta=substation_meta,
                        refuel_sessions=refuel_sessions, samples=demand_samples)
                except Exception as exc:  # noqa: BLE001 - keep pipeline alive
                    _log(f"variant [{tag}] FAILED for {scenario}: {exc}")

            # COST-SCENARIO sweep. The dual-mode buildout is RE-OPTIMIZED per cell
            # (its NPV layout genuinely depends on infra cost x fuel-price spread);
            # the fingerprint sim/derive caches make cells yielding an identical
            # layout nearly free. BEL is re-priced from its single reference buildout.
            ref_r_dual = None
            for infra_lvl, infra_mult in INFRA_LEVELS.items():
                cell_costs = cp.EnergizationOptCosts(
                    structure_cost_per_mile=med["structure"] * infra_mult,
                    energization_cost_per_mile=med["energization"] * infra_mult,
                    substation_fixed_cost=med["substation"] * infra_mult,
                )
                for energy_lvl in ENERGY_LEVELS:
                    emult = ENERGY_LEVELS[energy_lvl]
                    cell_diesel = ref_diesel * emult["diesel"]
                    cell_elec = {i: p * emult["elec"] for i, p in ref_elec.items()}
                    scenario = f"{layout}_infra-{infra_lvl}_energy-{energy_lvl}"
                    render_pdf = _should_render_pdf(infra_lvl, energy_lvl)
                    # Dual-mode buildout at THIS cost scenario (NPV-optimal, anti-
                    # sliver min run, NO max-gap constraint).
                    (r_dual, net_dual, e_dual, excl_dual, meta_dual, subs_dual,
                     region_frac_dual) = _optimize_and_sim(
                        f"{scenario}/dual", "dual_mode",
                        dict(piece_opex_fn=build_opex_fn(
                                 diesel_by_link, elec_by_link, cell_diesel, cell_elec),
                             costs=cell_costs, min_energized_length_miles=5.0,
                             warmstart=True, **common_opt),
                        samples=demand_samples)
                    net_dual_out = _prune_unused_catenary(net_dual, excl_dual[1])
                    if infra_lvl == "med" and energy_lvl == "med":
                        ref_r_dual = r_dual
                    _do_variant("dual_mode", net_dual_out, r_dual, subs_dual, e_dual,
                                region_frac_dual, "Dual_Mode",
                                {"Dual_Mode": lambda _y: _defaults.DUAL_MODE_LOCO_COST_USD},
                                render_pdf,
                                scenario=scenario, infra_lvl=infra_lvl,
                                infra_mult=infra_mult, energy_lvl=energy_lvl,
                                substation_meta=meta_dual)
                    if r_bel is not None:
                        _do_variant("bel", net_bel_out, r_bel, subs_bel, e_bel,
                                    region_frac_bel, "BEL",
                                    {"BEL": _bel_with_cat_loco_cost,
                                     "Diesel": lambda _y: _defaults.DIESEL_LOCO_COST_USD},
                                    render_pdf,
                                    scenario=scenario, infra_lvl=infra_lvl,
                                    infra_mult=infra_mult, energy_lvl=energy_lvl,
                                    # BEL layouts are optimized at reference
                                    # costs whatever the cell's multiplier.
                                    opt_infra_mult=1.0,
                                    substation_meta=meta_bel,
                                    refuel_sessions=refuel_sessions_bel)

            # Summary row: dual-mode reported at the reference (med/med) cell; BEL
            # from its single buildout.
            summary_row = {
                "layout": layout, "architecture": arch,
                "demand_level": demand_lvl,
                "demand_multiplier": float(len(demand_samples)),
                "dual_status": (str(ref_r_dual.status).split(".")[-1]
                                if ref_r_dual else "failed"),
                "dual_gap_pct": (ref_r_dual.optimality_gap_rel * 100.0
                                 if ref_r_dual is not None
                                 and ref_r_dual.optimality_gap_rel is not None
                                 else None),
                "dual_gap_abs_musd": (ref_r_dual.optimality_gap_abs
                                      if ref_r_dual else None),
                "dual_miles": ((ref_r_dual.energized_track_miles or 0.0)
                               if ref_r_dual else 0.0),
                "dual_capex_musd": ((ref_r_dual.total_cost or 0.0)
                                    if ref_r_dual else 0.0),
                "dual_lifecycle_musd": ((ref_r_dual.lifecycle_cost or 0.0)
                                        if ref_r_dual else 0.0),
                "dual_substations": (len(ref_r_dual.built_substation_ids)
                                     if ref_r_dual else 0),
                "bel_status": str(r_bel.status).split(".")[-1] if r_bel else "skipped",
                "bel_gap_pct": (r_bel.optimality_gap_rel * 100.0
                                if r_bel and r_bel.optimality_gap_rel is not None
                                else None),
                "bel_gap_abs_musd": r_bel.optimality_gap_abs if r_bel else None,
                "bel_miles": (r_bel.energized_track_miles or 0.0) if r_bel else 0.0,
                "bel_capex_musd": (r_bel.total_cost or 0.0) if r_bel else 0.0,
                "bel_substations": len(r_bel.built_substation_ids) if r_bel else 0,
            }
            summary_rows.append(summary_row)
            _write_csv(RESULTS_DIR / "energization_cost_results.csv", summary_rows)

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
    for demand_lvl, demand_samples in DEMAND_LEVELS.items():
        d_step = f"diesel_{demand_lvl}"
        if _step_done(d_step):
            _log(f"RESUME: skipping completed step diesel {demand_lvl}")
            continue
        if e_die is None:
            e_die = _sim(net_diesel, "conventional", "no_catenary",
                         "diesel-baseline", demand_samples)
        die_metrics = []
        for energy_lvl in ENERGY_LEVELS:
            try:
                die_metrics.append(_variant_metrics_df(
                    "diesel", "Diesel",
                    {"Diesel": lambda _y: _defaults.DIESEL_LOCO_COST_USD}, e_die,
                    None, [], {DEFAULT_CATENARY_NODE: 1.0}, arch="none",
                    demand_lvl=demand_lvl, demand_samples=demand_samples,
                    infra_lvl="na", infra_mult=1.0, energy_lvl=energy_lvl))
            except Exception as exc:  # noqa: BLE001 - keep pipeline alive
                _log(f"diesel metrics FAILED {demand_lvl}/{energy_lvl}: {exc}")
        if not die_metrics:
            # Every energy level raised: don't checkpoint an empty step (which would
            # let SWEEP_COMPLETE be written with the diesel baseline silently missing).
            raise RuntimeError(
                f"diesel metrics produced nothing for demand {demand_lvl}; "
                "see the 'diesel metrics FAILED' log lines above")
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
# BEL variant: each plan train is rebuilt as all BEL-with-pantograph units (100%
# battery-electric, no diesel-assist). On energized track a BEL draws catenary
# power for traction first, then charges its battery with leftover capacity; off
# catenary it runs on the battery.
# SIM_BEL_CAT_PWR_MAX_WATTS is the per-locomotive max deliverable pantograph power.
SIM_BEL_CAT_PWR_MAX_WATTS = 5.0e6
# Imported train consist plan (``_load_full_consist_plan``): one row per
# locomotive assignment, collapsed to one record per train. EVERY train is
# simulated (pre-optimization for densities, post-optimization to test the
# buildout) with its locomotives overridden to all dual-mode (cost) or all BEL
# (battery). Defaults to the milestone_1.3 Spokane-Lethbridge consist
# plan sitting next to this script.
#
# There is one per demand SAMPLE. Each is an independent draw of the same
# representative week at the same per-sample traffic level (52 trains/week,
# the level at which this corridor's trip times still settle), differing only
# in which observed consists were drawn -- pairwise overlap runs 15-23%, so
# they are different weeks rather than copies of one.
#
# `DEMAND_LEVELS` composes them in units of the CALIBRATED volume, which is
# itself two samples here: 1x is samples 1-2, 2x is 1-4, 4x is 1-8.
#
# They are never concatenated. A 4x level is EIGHT meet-pass problems of 52
# trains each, solved separately and summed, not one problem of 416. That is
# deliberate: it raises energy throughput for the economics and grid-impact
# questions without asserting the congestion that 416 trains on this corridor
# would actually cause -- which, measured, is an unbounded queue. Results at 2x
# and 4x therefore describe demand, not congestion, and must be caveated so.
_CORRIDOR_INPUTS = Path(__file__).resolve().parent.parent / "corridor_inputs"
_PLAN_BASENAME = "train_consist_plan_spokane_lethbridge_clean.csv"
CONSIST_PLAN_PATHS = tuple(
    _CORRIDOR_INPUTS / (_PLAN_BASENAME if k == 1
                        else f"sample{k}_{_PLAN_BASENAME}")
    for k in range(1, max(max(s_) for s_ in DEMAND_LEVELS.values()) + 2)
)
# Single-sample entry point, for the paths that are not demand-swept.
CONSIST_PLAN_PATH = CONSIST_PLAN_PATHS[0]
# Electrical assumptions for line losses (25 kV AC medium; matches RULE_COSTS).
SIM_SYSTEM_VOLTAGE_VOLTS = 25_000.0
SIM_ELECTRIFICATION_TYPE = "ac_25kv_1x25"
# Speed-limit treatment. This is an experiment factor, not a fixed property of
# the corridor, so both numbers are overridable from the environment
# (ALTRIOS_SPEED_CAP_MPH, ALTRIOS_SPEED_SCALER; either may be "none") and both
# are already hashed into the run cache key, so varying them cannot silently
# reuse another cell's results.
#
# The default -- cap at 40 mph, then scale by 0.595 -- is the milestone_1.3 /
# milestone_2.3b calibration, fitted so simulated travel time matched RailState
# observations on Jaffray<->Sparwood. It was fitted while `sim_manager`
# dispatched one train per `run_dispatch` call, which per commit d4f245c
# bypassed meet-pass entirely: `check_deadlock` skips the only train present, so
# no conflict resolution ever ran. The scaler was therefore never measuring
# track speed; it was standing in for congestion delay that was not modelled.
# Dispatch now batches all trains into one call and models that delay directly,
# so keeping the scaler charges for it twice -- on Winnipeg-Chicago,
# `speed_calibration.py --scaler 0.595` overshoots observed run time by +40.4%.
#
# It is nonetheless still the default here, for a reason specific to THIS
# corridor: Spokane-Lethbridge's network carries synthetic speed limits (the
# builder's 60/15 mph constants, not surveyed data). RailState's 99th-percentile
# spot speeds come to 0.34-0.51 of them, against 0.66-1.00 on
# Toronto-Battle Creek and 0.82-0.91 on Winnipeg-Chicago, where the posted
# limits are real. Dropping the scaler here would run trains at fabricated line
# speeds. Until real speed-limit data arrives for this corridor the calibration
# stays, with its provenance stated rather than implied.
def _env_float(name, default):
    """Environment override for a numeric factor; "none"/"" means no limit."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    return None if raw.lower() in ("", "none") else float(raw)


SPEED_CALIBRATION_MAX_MPH = _env_float("ALTRIOS_SPEED_CAP_MPH", 40.0)
SPEED_LIMIT_SCALER = _env_float("ALTRIOS_SPEED_SCALER", 0.595)


def _calibrate_speeds(net):
    """Return a copy of ``net`` with calibrated speed limits (cap, then scale)."""
    d = net.to_pydict()
    if SPEED_CALIBRATION_MAX_MPH is not None:
        d = _utils.set_network_max_speed(d, max_speed_mph=SPEED_CALIBRATION_MAX_MPH)
    if SPEED_LIMIT_SCALER is not None:
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
# Keyed by demand-sample index (or by explicit path), not singletons: the
# demand levels need several plans loaded at once and must not share a cache.
_FULL_CONSIST_PLAN = {}
_PLAN_SUMMARY = {}


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


def _load_full_consist_plan(path=None, sample=0):
    """Parse the imported consist plan into one record per TRAIN (loaded once).

    The CSV has one row per locomotive assignment (keyed by ``Train_ID``); this
    collapses it to a per-train list carrying origin/destination, train type,
    loaded/empty car counts, physical length/weight, and the number of assigned
    locomotives (consist size). The full-plan simulators then override each
    train's locomotives to be all dual-mode or all BEL while preserving
    that consist size and the rest of the train's attributes.
    """
    key = str(path) if path is not None else int(sample)
    if key not in _FULL_CONSIST_PLAN:
        path = path if path is not None else CONSIST_PLAN_PATHS[sample]
        df = pl.read_csv(path)
        _FULL_CONSIST_PLAN[key] = (
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
    return _FULL_CONSIST_PLAN[key]


def _extended_plan(sample=0):
    """The consist plan wrapped into warm-up + measured week + cool-down.

    Each record carries ``_measured``: True for the week being reported on,
    False for the traffic that exists only to congest it.
    """
    base = _load_full_consist_plan(sample=sample)
    if WARMUP_DAYS <= 0 and COOLDOWN_DAYS <= 0:
        return [dict(t, _measured=True) for t in base]

    week_hr = PLAN_DAYS * 24.0
    warm_hr, cool_hr = WARMUP_DAYS * 24.0, COOLDOWN_DAYS * 24.0
    span = max(int(t["Train_ID"]) for t in base) + 1

    def block(records, delta_hr, id_offset, measured):
        out = []
        for t in records:
            r = dict(t)
            r["Departure_Time_Planned_Hr"] = t["Departure_Time_Planned_Hr"] + delta_hr
            r["Train_ID"] = int(t["Train_ID"]) + id_offset
            r["_measured"] = measured
            out.append(r)
        return out

    tail = [t for t in base if t["Departure_Time_Planned_Hr"] >= week_hr - warm_hr]
    head = [t for t in base if t["Departure_Time_Planned_Hr"] < cool_hr]
    records = (
        block(tail, -week_hr, 0, False)
        + block(base, 0.0, span, True)
        + block(head, week_hr, 2 * span, False)
    )
    floor = min(r["Departure_Time_Planned_Hr"] for r in records)
    for r in records:
        r["Departure_Time_Planned_Hr"] -= floor
    return records


def _in_measure_window(train) -> bool:
    """Whether this train's energy counts toward the reported week."""
    return bool(train.get("_measured", True))


def _plan_summary(sample=0):
    """Traffic summary of one demand sample's consist plan (one week).

    Returns trains/day, locomotives/day, and average locomotives/train over
    ``PLAN_DAYS``, for reporting and fleet sizing. A demand LEVEL's traffic is
    the sum over its samples -- see `_level_plan_summary`.
    """
    if sample not in _PLAN_SUMMARY:
        plan = _load_full_consist_plan(sample=sample)
        n_trains = len(plan)
        total_locos = sum(int(t["n_locos"]) for t in plan)
        _PLAN_SUMMARY[sample] = {
            "n_trains": n_trains,
            "total_locos": total_locos,
            "trains_per_day": n_trains / PLAN_DAYS,
            "locos_per_day": total_locos / PLAN_DAYS,
            "avg_locos_per_train": total_locos / max(n_trains, 1),
        }
    return _PLAN_SUMMARY[sample]


def _level_plan_summary(samples):
    """Traffic of a demand LEVEL: its samples' traffic, summed.

    Counts and rates add across samples because each sample is a separate week
    of trains on the same corridor. Averages (locomotives per train) do not add
    and are re-derived from the summed totals.
    """
    totals = {"n_trains": 0, "total_locos": 0, "trains_per_day": 0.0,
              "locos_per_day": 0.0}
    for s in samples:
        ps = _plan_summary(s)
        for k in totals:
            totals[k] += ps[k]
    totals["avg_locos_per_train"] = (
        totals["total_locos"] / max(totals["n_trains"], 1))
    totals["n_samples"] = len(samples)
    return totals


# Substation SPACING is sized from a voltage-drop model at a DESIGN train power:
# the aggregate pantograph draw of a high-percentile consist = percentile over
# trains of (locos/train) x per-loco pantograph rating. Consist sizes come from
# the imported plan; a higher percentile -> heavier design train -> shorter max
# feed distance (more/closer substations). Replaces the 10 MW library default,
# which understated real per-consist draws now that line loss is aggregated per
# consist rather than per locomotive.
DESIGN_TRAIN_POWER_PERCENTILE = 95.0
_DESIGN_TRAIN_POWER_WATTS = None


def _design_train_power_watts():
    """Aggregate pantograph power of the ``DESIGN_TRAIN_POWER_PERCENTILE``-th
    percentile consist, feeding the substation-spacing voltage-drop model."""
    global _DESIGN_TRAIN_POWER_WATTS
    if _DESIGN_TRAIN_POWER_WATTS is None:
        import numpy as _np
        # Pooled over every demand sample: substation spacing is a property
        # of the corridor's consists, not of how many weeks of them are
        # simulated, and pooling gives the percentile a larger, stabler base.
        n_locos = [int(t["n_locos"])
                   for s in range(len(CONSIST_PLAN_PATHS))
                   for t in _load_full_consist_plan(sample=s)]
        pct_locos = float(_np.percentile(n_locos, DESIGN_TRAIN_POWER_PERCENTILE))
        _DESIGN_TRAIN_POWER_WATTS = pct_locos * SIM_BEL_CAT_PWR_MAX_WATTS
        _log(f"substation-spacing design train power: "
             f"p{DESIGN_TRAIN_POWER_PERCENTILE:.0f} consist = {pct_locos:.1f} locos "
             f"x {SIM_BEL_CAT_PWR_MAX_WATTS / 1e6:.1f} MW = "
             f"{_DESIGN_TRAIN_POWER_WATTS / 1e6:.1f} MW")
    return _DESIGN_TRAIN_POWER_WATTS


def _build_locos(n_locos, traction_kind, save_interval=SIM_SAVE_INTERVAL,
                 n_diesel_helpers=0):
    """Build one train's locomotive list with composition set by ``traction_kind``.

    - ``"dual_mode"`` -- all ``n_locos`` units are dual-mode (diesel + catenary),
      for the COST analysis (diesel displaced by grid electricity on wire).
    - ``"battery_electric"`` -- all ``n_locos`` units are BEL-with-pantograph
      (100% battery-electric, no diesel-assist), for the BATTERY analysis. The
      Tier-2 SOC layout guarantees the pack never depletes and simulation confirms
      every plan train completes with SOC margin, so no get-home diesel is needed.
      ``n_diesel_helpers`` swaps that many BELs for conventional diesel units --
      the per-train fallback when a pure-BEL consist cannot pull a grade (see
      ``_walk_plan_train``); the resulting mixed consist is carried on the sim.
    - ``"conventional"`` -- all ``n_locos`` units are conventional diesel, the
      all-diesel baseline for comparison.

    The consist SIZE comes from the plan; only the unit composition is set here.
    """
    n = max(int(n_locos), 1)
    if traction_kind == "dual_mode":
        locos = [alt.Locomotive.default_dual_mode_loco() for _ in range(n)]
    elif traction_kind == "conventional":
        locos = [alt.Locomotive.default() for _ in range(n)]  # 100% diesel baseline
    elif traction_kind == "battery_electric":
        k = min(max(int(n_diesel_helpers), 0), n)  # diesel helpers swapped in
        locos = [_bel_with_cat_loco() for _ in range(n - k)]
        locos += [alt.Locomotive.default() for _ in range(k)]
    else:
        raise ValueError(
            "traction_kind must be 'dual_mode', 'conventional', or "
            f"'battery_electric', got {traction_kind!r}"
        )
    locos = [_with_receptivity(loco) for loco in locos]
    for loco in locos:
        loco.set_save_interval(save_interval)
    return locos


def _make_plan_train_ts(train, location_map, traction_kind,
                        save_interval=SIM_SAVE_INTERVAL, n_diesel_helpers=0):
    """Build one SpeedLimitTrainSim for a consist-plan train.

    Uses the train's actual plan attributes -- origin/destination, train type,
    loaded/empty car counts, and physical weight/length (rail-vehicle mass and
    length are scaled to match) -- with the locomotive composition overridden per
    ``traction_kind`` (all dual-mode, or all BEL). ``n_diesel_helpers`` swaps that
    many BELs for conventional diesel units (battery-electric fallback only).
    """
    rail_vehicles, freight_map = _load_rail_vehicles()
    rv_to_use, n_cars_by_type = _data_prep.configure_rail_vehicles(
        train["Train_Type"], int(train["Cars_Loaded"]), int(train["Cars_Empty"]),
        rail_vehicles, freight_map,
    )
    locos = _build_locos(train["n_locos"], traction_kind, save_interval,
                         n_diesel_helpers=n_diesel_helpers)
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
    # Depart at the train's planned hour. Without this every train starts at
    # t=0, so a batched dispatch would put a whole week of traffic on the
    # corridor simultaneously and meet-pass would be meaningless -- which is why
    # these trains were previously dispatched one at a time, each seeing an
    # empty corridor.
    departure_hr = train.get("Departure_Time_Planned_Hr")
    tsb = alt.TrainSimBuilder(
        train_id=str(train["Train_ID"]),
        origin_id=train["Origin_ID"], destination_id=train["Destination_ID"],
        train_config=train_config, loco_con=loco_con,
        init_train_state=(
            alt.InitTrainState(time_seconds=float(departure_hr) * 3600.0)
            if departure_hr is not None else None
        ),
    )
    # The plan is one representative week; stamping it on the sim keeps the Rust
    # annualize getters consistent with the metrics calculator's own scaler
    # (which no longer depends on this field, but downstream readers may).
    ts = tsb.make_speed_limit_train_sim(
        location_map=location_map, save_interval=save_interval,
        simulation_days=int(PLAN_DAYS))
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
    loco = _with_receptivity(loco_factory())
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
    """Factory for a battery-electric locomotive equipped with a pantograph.

    Overrides the ALTRIOS default pack (2.4 MWh) to
    ``SIM_BEL_BATTERY_CAPACITY_KWH`` by round-tripping the locomotive through a
    pydict (there is no direct setter for ``res.energy_capacity_joules``).
    """
    loco = alt.Locomotive.default_battery_electric_loco_with_cat(
        SIM_BEL_CAT_PWR_MAX_WATTS
    )
    d = loco.to_pydict()
    d["loco_type"]["BatteryElectricLoco"]["res"]["energy_capacity_joules"] = (
        SIM_BEL_BATTERY_CAPACITY_KWH * 3.6e6  # kWh -> J
    )
    return alt.Locomotive.from_pydict(d)


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


def _extract_from_pydict(ts, d):
    """Single pass over a walked sim's pydict: energy totals + per-link consist
    fuel (J) + per-link pantograph draw (J).

    Consolidates what used to be three separate ``ts.to_pydict()`` passes
    (``_energy_from_walked``, ``_per_link_energy_from_sim``, and
    ``_per_link_catenary_draw``) into one, since each full-history serialization is
    the dominant per-sim cost. Returns ``{"totals", "diesel_j_by_link",
    "draw_by_link"}``. ``draw_by_link`` (pantograph energy per link, any loco with a
    catenary interface) feeds the direction-exclusion pass; ``diesel_j_by_link``
    (consist fuel per link) feeds the first-pass densities.
    """
    hist = d["history"]
    times = hist["time_seconds"]
    front = hist["link_idx_front"]
    n = len(times)
    dts = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
    fuel_w = [0.0] * n
    panto_w = [0.0] * n
    pantograph_j = line_loss_j = battery_j = regen_j = 0.0
    for loco in d["loco_con"]["loco_vec"]:
        lt = loco["loco_type"]
        ci = _loco_cat_int(loco)
        if ci is not None:
            ch = ci["history"]
            cs = ci.get("state") or {}
            # Cumulative totals come from the terminal STATE, matching
            # `metric_calculator._sim_catenary_joules`. The history tail is the
            # last sample on a multiple of `save_interval`, not the value at
            # arrival, so it drops up to `save_interval - 1` seconds of delivery
            # per locomotive per trip -- always downward, and growing with the
            # interval. Measured at 60 s on a catenary sim: 0.011% low.
            e_in = cs.get("energy_elec_in_joules")
            e_ll = cs.get("energy_line_loss_joules")
            # Returned to a receptive wire; `energy_elec_in` is NET of it, so
            # the gross draw is their sum. Absent on sims that predate it.
            e_rg = cs.get("energy_elec_regen_joules")
            if e_in is None:
                e_in = (ch.get("energy_elec_in_joules") or [None])[-1]
                e_ll = (ch.get("energy_line_loss_joules") or [None])[-1]
                e_rg = (ch.get("energy_elec_regen_joules") or [0.0])[-1]
            if e_in is not None:
                pantograph_j += e_in + (e_rg or 0.0)
                regen_j += e_rg or 0.0
                line_loss_j += e_ll or 0.0
            else:  # fallback: left-rectangle integration of power over time steps
                grid = ch.get("pwr_elec_in_watts") or []
                loss = ch.get("pwr_line_loss_watts") or [0.0] * len(grid)
                pantograph_j += sum(p * dt for p, dt in zip(grid[:-1], dts))
                line_loss_j += sum(p * dt for p, dt in zip(loss[:-1], dts))
            pin = ch.get("pwr_elec_in_watts") or []
            for i in range(min(n, len(pin))):
                panto_w[i] += pin[i]
        for key in ("DualModeLoco", "ConventionalLoco"):
            sub = lt.get(key)
            if sub is not None:
                fh = sub["fc"]["history"].get("pwr_fuel_watts") or []
                for i in range(min(n, len(fh))):
                    fuel_w[i] += fh[i]
                break
        # Net chemical energy discharged from the battery (positive) / charged
        # (negative); makes the battery's role in the energy balance visible.
        bel = lt.get("BatteryElectricLoco")
        if bel is not None:
            # Terminal state for the same reason as the catenary totals above.
            e_chem = (bel["res"].get("state") or {}).get("energy_out_chemical_joules")
            if e_chem is None:
                e_chem = (
                    bel["res"]["history"].get("energy_out_chemical_joules") or [None]
                )[-1]
            if e_chem:
                battery_j += e_chem
    diesel_j_by_link = {}
    draw_by_link = {}
    net_by_link = {}
    # `pwr_elec_in` is signed: positive drawing, negative returning. The gross
    # draw per link says which wire is USED (direction exclusion); the net,
    # with the return credited at the export fraction, is what the optimizer's
    # opex should see.
    credit = CATENARY_EXPORT_CREDIT_FRACTION
    for i, dt in enumerate(dts):
        li = front[i]
        if not li:  # skip 0 / None (off-network / pre-departure steps)
            continue
        if fuel_w[i]:
            diesel_j_by_link[li] = diesel_j_by_link.get(li, 0.0) + fuel_w[i] * dt
        if panto_w[i] > 0.0:
            draw_by_link[li] = draw_by_link.get(li, 0.0) + panto_w[i] * dt
            net_by_link[li] = net_by_link.get(li, 0.0) + panto_w[i] * dt
        elif panto_w[i] < 0.0:
            net_by_link[li] = net_by_link.get(li, 0.0) + credit * panto_w[i] * dt
    # Grid electricity is NET of what a receptive wire took back (the line loss
    # on the return leg comes out of the returned energy).
    loss_on_return = (line_loss_j * regen_j / (pantograph_j + regen_j)
                      if pantograph_j + regen_j > 0 else 0.0)
    totals = {"diesel_j": ts.get_energy_fuel_joules(False),
              "pantograph_elec_j": pantograph_j, "regen_elec_j": regen_j,
              "line_loss_j": line_loss_j,
              "grid_elec_j": pantograph_j + line_loss_j - max(regen_j - loss_on_return, 0.0),
              "battery_j": battery_j,
              "travel_hr": times[-1] / 3600.0}
    return {"totals": totals, "diesel_j_by_link": diesel_j_by_link,
            "draw_by_link": draw_by_link, "net_by_link": net_by_link}


def _energy_from_walked(ts):
    """Energy totals for one walked sim (thin wrapper over ``_extract_from_pydict``)."""
    return _extract_from_pydict(ts, ts.to_pydict())["totals"]


def _walk_plan_train(train, location_map, sim_network, traction_kind):
    """Build, dispatch, and walk ONE plan train; return ``(walked_ts, n_helpers)``.

    For ``battery_electric`` consists this adds a diesel-helper fallback: if a pure
    -BEL consist fails to run -- ``make_est_times`` (or the subsequent dispatch /
    walk) raises, typically a grade too steep for the available off-wire BEL power
    -- one BEL is swapped for a conventional diesel unit and the run is retried,
    escalating up to an all-diesel consist (which always runs, matching the diesel
    baseline). The realized mixed consist is carried on the returned sim, so the
    exact BEL/diesel composition is recoverable from
    ``ts.to_pydict()["loco_con"]["loco_vec"]`` for downstream costing.

    Non-BEL traction kinds never add helpers (``max_help == 0``); a failure there
    propagates as before.
    """
    n_locos = max(int(train["n_locos"]), 1)
    max_help = n_locos if traction_kind == "battery_electric" else 0
    last_exc = None
    for n_help in range(0, max_help + 1):
        try:
            ts = _make_plan_train_ts(train, location_map, traction_kind,
                                     n_diesel_helpers=n_help)
            etn, _consist = alt.make_est_times(ts, sim_network)
            timed_path = next(iter(alt.run_dispatch(
                sim_network, alt.SpeedLimitTrainSimVec([ts]), [etn], False, False)))
            ts.walk_timed_path(network=sim_network, timed_path=timed_path)
            if n_help:
                _log(f"    train {train.get('Train_ID')}: pure-BEL run infeasible; "
                     f"succeeded with {n_help} diesel helper(s) of {n_locos} units")
            return ts, n_help
        except Exception as exc:  # noqa: BLE001 - escalate to a diesel helper
            last_exc = exc
    raise RuntimeError(
        f"train {train.get('Train_ID')} failed to run even with {max_help} diesel "
        f"helper(s): {last_exc}"
    )


# Walk all non-BEL plan trains in one parallel rayon batch (core-side) instead of
# one-at-a-time; set False to force the serial per-train walk.
WALK_PARALLELIZE = True


def _run_full_plan(sim_network, location_map, label, traction_kind,
                   want_per_link=False, keep_sims=True, sample=0):
    """Build, dispatch, and walk EVERY train in the consist plan over
    ``sim_network``; return energy summed over all trains.

    Each train is built from its plan attributes (``_make_plan_train_ts``) with the
    locomotive composition set by ``traction_kind`` (all dual-mode, or all BEL
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
    plan = _extended_plan(sample)
    n = len(plan)
    n_measured = sum(1 for t in plan if _in_measure_window(t))
    totals = {"diesel_j": 0.0, "pantograph_elec_j": 0.0, "regen_elec_j": 0.0,
              "line_loss_j": 0.0, "grid_elec_j": 0.0, "battery_j": 0.0, "travel_hr": 0.0}
    diesel_by_link, elec_by_link, draw_by_link = {}, {}, {}
    sims = []
    trains_with_helpers = 0
    total_diesel_helpers = 0
    _log(f"  {label}: simulating {n} plan trains ({traction_kind}); "
         f"{n_measured} measured, {n - n_measured} warm-up/cool-down ...")
    t0 = time.perf_counter()

    def _accumulate(ts):
        ex = _extract_from_pydict(ts, ts.to_pydict())
        for key in totals:
            totals[key] += ex["totals"][key]
        # Per-link pantograph draw (any loco) always, so the direction-exclusion
        # pass never has to re-serialize the sims.
        for li, v in ex["draw_by_link"].items():
            draw_by_link[li] = draw_by_link.get(li, 0.0) + v
        if want_per_link:
            for li, v in ex["diesel_j_by_link"].items():
                diesel_by_link[li] = diesel_by_link.get(li, 0.0) + v
            # Net electricity per link (return credited), for the opex densities.
            for li, v in ex["net_by_link"].items():
                elec_by_link[li] = elec_by_link.get(li, 0.0) + v
        if keep_sims:
            sims.append(ts)

    # Non-BEL consists have no diesel-helper escalation (a walk failure is fatal),
    # so build + dispatch serially, then walk EVERY train in one parallel rayon
    # batch -- the dominant cost. BEL keeps the per-train retry path.
    if WALK_PARALLELIZE and traction_kind in ("dual_mode", "conventional"):
        # ONE `run_dispatch` over every train, so meet-pass conflict resolution
        # actually runs. Dispatching a `SpeedLimitTrainSimVec` of one -- as this
        # did -- means `check_deadlock` skips the only train present and no
        # conflict is ever resolved, so each train saw an empty corridor and the
        # per-link densities carried no congestion at all. That is the same
        # defect commit d4f245c fixed in `sim_manager`.
        # `built`, NOT `sims`: `sims` is the accumulator of RETAINED walked sims
        # that this function returns, and `_accumulate` appends to it below.
        # Rebinding it here as the pre-dispatch build list and then deleting it
        # left the return statement reading a deleted local.
        built, etns = [], []
        for k, train in enumerate(plan, 1):
            ts = _make_plan_train_ts(train, location_map, traction_kind)
            etn, _consist = alt.make_est_times(ts, sim_network)
            built.append(ts)
            etns.append(etn)
            if k % 20 == 0 or k == n:
                _log(f"    {label}: built {k}/{n} "
                     f"({time.perf_counter() - t0:.0f}s)")
        vec = alt.SpeedLimitTrainSimVec(built)
        timed = list(alt.run_dispatch(sim_network, vec, etns, False, False))
        del built, etns
        _log(f"    {label}: dispatched {n} trains together "
             f"({time.perf_counter() - t0:.0f}s)")
        vec.walk_timed_paths(sim_network, timed, True)
        del timed
        _log(f"    {label}: parallel-walked {n} trains "
             f"({time.perf_counter() - t0:.0f}s)")
        walked = vec.tolist()
        del vec
        # Warm-up and cool-down trains exist to congest the corridor for the
        # measured week; their own energy is not part of the week being
        # annualized, so it is excluded here rather than scaled away later.
        for train, ts in zip(plan, walked):
            if _in_measure_window(train):
                _accumulate(ts)
    else:
        for k, train in enumerate(plan, 1):
            ts, n_help = _walk_plan_train(train, location_map, sim_network,
                                          traction_kind)
            if n_help:
                trains_with_helpers += 1
                total_diesel_helpers += n_help
            if _in_measure_window(train):
                _accumulate(ts)
            if k % 20 == 0 or k == n:
                _log(f"    {label}: {k}/{n} trains "
                     f"({time.perf_counter() - t0:.0f}s)")

    # `grid_elec_j` was summed per train above, net of returned energy.
    _log(f"  {label} TOTAL over {n} trains: diesel {totals['diesel_j']/1e9:.1f} GJ, "
         f"grid {totals['grid_elec_j']/1e9:.1f} GJ (returned {totals['regen_elec_j']/1e9:.1f} GJ), "
         f"battery {totals['battery_j']/1e9:.1f} GJ")
    if trains_with_helpers:
        _log(f"  {label}: {trains_with_helpers}/{n} trains required a diesel helper "
             f"({total_diesel_helpers} helper units total)")
    return {**totals, "sims": sims, "n_trains": n_measured, "n_dispatched": n,
            "diesel_j_by_link": diesel_by_link, "elec_j_by_link": elec_by_link,
            "draw_by_link": draw_by_link,
            "trains_with_helpers": trains_with_helpers,
            "total_diesel_helpers": total_diesel_helpers}


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
        # Signed pantograph power: a return (negative) is credited at the
        # export fraction, a draw in full.
        w = grid_w[i] if grid_w[i] >= 0.0 else CATENARY_EXPORT_CREDIT_FRACTION * grid_w[i]
        elec_j_by_link[li] = elec_j_by_link.get(li, 0.0) + w * dt
    return diesel_j_by_link, elec_j_by_link


def _per_link_catenary_draw(sims) -> dict:
    """Per-link catenary (pantograph) energy drawn (J), summed over all trains.

    Works for any loco type (reads each loco's ``cat_int`` pantograph power). A link
    that was energized but drew ~0 over the whole plan is never used -- e.g. a
    parallel track / direction whose trains stay on battery -- so its marginal wire
    cost can be excluded on the back-end (see ``_excluded_energization_miles``).
    """
    draw: dict = {}
    for ts in sims:
        d = ts.to_pydict()
        hist = d["history"]
        front = hist["link_idx_front"]
        times = hist["time_seconds"]
        n = len(times)
        dts = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]
        panto_w = [0.0] * n
        for loco in d["loco_con"]["loco_vec"]:
            ci = _loco_cat_int(loco)
            if ci is None:
                continue
            ph = ci["history"].get("pwr_elec_in_watts") or []
            for i in range(min(n, len(ph))):
                panto_w[i] += ph[i]
        for i, dt in enumerate(dts):
            li = front[i]
            if not li:
                continue
            if panto_w[i] > 0.0:
                draw[li] = draw.get(li, 0.0) + panto_w[i] * dt
    return draw


def _excluded_energization_miles(result, sims, draw_by_link=None, threshold_j=1.0e6):
    """Track-miles of energized wire that NEVER drew catenary over the whole plan.

    The optimizer wires every track at an energized site (whole-site, full cost);
    this back-end pass finds energized links whose plan-wide catenary draw is below
    ``threshold_j`` (a parallel track/direction the fleet never charges from) so the
    STUDY can subtract that marginal wire cost. Structure (shared per site) and
    substations are unaffected. Returns ``(excluded_miles, excluded_link_idxs)``.

    ``draw_by_link`` (precomputed during the sim walk) is used when supplied to
    avoid re-serializing every sim; otherwise it is derived here.
    """
    if result is None or (not sims and not draw_by_link):
        return 0.0, []
    draw = draw_by_link if draw_by_link is not None else _per_link_catenary_draw(sims)
    intervals = result.energized_link_intervals or {}
    excl_miles = 0.0
    excl_links = []
    for li, ivs in intervals.items():
        if draw.get(li, 0.0) <= threshold_j:
            excl_miles += sum((b - a) for a, b in ivs) * _utils.MI_PER_M
            excl_links.append(li)
    return excl_miles, excl_links


def _prune_unused_catenary(net, excl_links):
    """Return a network copy with catenary removed on ``excl_links``.

    The optimizer wires every track at an energized site (whole-site); this strips
    ``cat_power_limits`` from links the fleet never charges from so the serialized
    network file reflects the AS-PURCHASED catenary, consistent with the back-end
    direction-exclusion cost subtraction. ``net`` is returned unchanged when there
    is nothing to prune.
    """
    if not excl_links:
        return net
    excl = set(excl_links)
    d = net.to_pydict()
    for link in d[1][1:]:
        if link.get("idx_curr") in excl:
            link["cat_power_limits"] = []
    return alt.Network.from_pydict(d)


def _net_energized_track_miles(net) -> float:
    """Total energized track-miles actually present in a network's cat_power_limits.

    The single source of truth for AS-PURCHASED energization -- reads the network
    object directly, so a pruned (direction-excluded) network reports its reduced
    wire without any separate bookkeeping.
    """
    total_m = 0.0
    for L in net.to_pydict()[1][1:]:
        for c in (L.get("cat_power_limits") or []):
            total_m += (c.get("offset_end_meters", 0.0)
                        - c.get("offset_start_meters", 0.0))
    return total_m * _utils.MI_PER_M


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
    totals = {"diesel_j": 0.0, "pantograph_elec_j": 0.0, "regen_elec_j": 0.0,
              "line_loss_j": 0.0, "grid_elec_j": 0.0, "battery_j": 0.0, "travel_hr": 0.0}
    draw_by_link = {}
    for ts in sims:
        ex = _extract_from_pydict(ts, ts.to_pydict())
        for k in totals:
            totals[k] += ex["totals"][k]
        for li, v in ex["draw_by_link"].items():
            draw_by_link[li] = draw_by_link.get(li, 0.0) + v
    totals["grid_elec_j"] = totals["pantograph_elec_j"] + totals["line_loss_j"] - totals["regen_elec_j"]
    return {**totals, "sims": list(sims), "n_trains": len(sims),
            "diesel_j_by_link": {}, "elec_j_by_link": {},
            "draw_by_link": draw_by_link}


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


def _planner_charge_pass(sim_network, location_map, sample=0):
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
        consist_plan_in = pl.read_csv(CONSIST_PLAN_PATHS[sample])
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


def _departure_seconds(dep_by_index, n_sims, sample=0):
    """Per-train wall-clock departure offset (s) aligned to the ``_run_full_plan``
    sim order, from the planner schedule (``dep_by_index``) with the plan's planned
    departures as a fallback when the planner schedule is missing an index."""
    plan = _load_full_consist_plan(sample=sample)
    out = []
    for i in range(n_sims):
        hr = dep_by_index.get(i)
        if hr is None and i < len(plan):
            hr = plan[i].get("Departure_Time_Planned_Hr")
        out.append(float(hr) * 3600.0 if hr is not None else 0.0)
    return out


def _level_measured_plan(samples):
    """Per-train records in the order `_merge_plan_runs` concatenated the sims.

    `_run_full_plan` retains a sim only for MEASURED trains (warm-up and
    cool-down congest the corridor but are not part of the annualized week), in
    plan order, and the merge concatenates samples in order. Anything that maps
    a sim index back to its train -- fleet tables, load-curve departure
    alignment -- has to walk the same sequence or it will attribute one
    sample's locomotives to another sample's train.
    """
    return [t for s in samples
            for t in _extended_plan(s) if _in_measure_window(t)]


def _merge_plan_runs(parts):
    """Combine one `_run_full_plan` result per demand sample into one.

    A demand level is several independent weeks of traffic on the same
    corridor, so its energy, emissions, cost basis and train counts are the
    sums over its samples, its per-link dictionaries add link by link, and its
    walked sims concatenate -- which is what carries the extra traffic into the
    metrics calculator and the substation load curves.

    The samples are NOT merged before dispatch. Each was dispatched and walked
    on its own, so no train in one sample ever met a train in another. This
    raises throughput without asserting the congestion that the combined
    traffic would cause; see `DEMAND_LEVELS`.
    """
    parts = list(parts)
    if len(parts) == 1:
        return parts[0]
    merged = {}
    for key, sample_value in parts[0].items():
        if key == "sims":
            continue
        if isinstance(sample_value, dict):
            merged[key] = _sum_link_densities(part[key] for part in parts)
        elif isinstance(sample_value, bool) or not isinstance(sample_value, (int, float)):
            merged[key] = sample_value          # labels, networks, flags
        else:
            merged[key] = sum(part[key] for part in parts)
    # A plain list, matching what `_run_full_plan` returns. Wrapping it in a
    # `SpeedLimitTrainSimVec` here made the merged result behave differently from
    # an unmerged one: the vec is not iterable from Python, so every consumer
    # that walks `energy["sims"]` directly -- `_region_fractions`,
    # `_build_fleet_tables`, `_train_timeseries_df` -- broke, but only at 2x and
    # 4x, since 1x returns its single part untouched. Callers that need a vec
    # build one themselves.
    merged["sims"] = [ts for part in parts for ts in list(part.get("sims") or [])]
    return merged


def _sum_link_densities(dicts):
    """Add per-link energy densities across demand samples.

    Each sample contributes a full week of traffic over the same links, so the
    densities add. Links absent from one sample (no train reached them in that
    week) simply contribute zero there rather than dropping out of the union.
    """
    out = {}
    for d in dicts:
        for link_idx, value in d.items():
            out[link_idx] = out.get(link_idx, 0.0) + value
    return out


def _first_pass_cache_path(sample=0):
    """Stable cache path for the first-pass densities, keyed by their inputs.

    The key hashes the consist-plan file contents plus the network file, speed
    calibration, save interval, and catenary power limit so the cache is reused
    only when those inputs are unchanged; edit any of them and a fresh pass runs.
    """
    h = hashlib.sha1()
    h.update(Path(CONSIST_PLAN_PATHS[sample]).read_bytes())
    h.update(str(_NET_PATH).encode())
    h.update(repr((SPEED_CALIBRATION_MAX_MPH, SPEED_LIMIT_SCALER, SIM_SAVE_INTERVAL,
                   COMMON["uniform_power_limit_watts"], PLAN_DAYS)).encode())
    return _CACHE_DIR / f"first_pass_densities_{h.hexdigest()[:12]}.parquet"


def first_pass_densities(sample=0):
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
    cache = _first_pass_cache_path(sample)
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
                         "dual_mode", want_per_link=True, keep_sims=False,
                         sample=sample)
    cat = _run_full_plan(net_cat, location_map, "first-pass/all-catenary",
                         "dual_mode", want_per_link=True, keep_sims=False,
                         sample=sample)
    diesel_j_by_link = die["diesel_j_by_link"]   # fuel from all-diesel pass
    elec_j_by_link = cat["elec_j_by_link"]       # grid from all-catenary pass

    # These maps are the optimizer's entire view of where electrification pays,
    # and they are assembled from history series this study never configures
    # explicitly -- it saves everything. The sibling DoE (`corridor_doe`) now
    # records a deliberately parsimonious five-series set, and applying that
    # here would empty these maps: `link_idx_front` keys the per-link
    # attribution, `fc.history.pwr_fuel_watts` supplies the diesel, and
    # `cat_int.history.pwr_elec_in_watts` the pantograph draw. Two of the three
    # degrade to empty lists rather than raising, so the failure would reach the
    # MILP as "nothing to save anywhere" instead of as an error.
    for name, by_link in (("diesel", diesel_j_by_link), ("electric", elec_j_by_link)):
        if not by_link:
            raise RuntimeError(
                f"first-pass {name} per-link density map is empty -- the walked "
                "sims did not carry the history this needs (train "
                "`link_idx_front`, per-locomotive `fc.pwr_fuel_watts`, "
                "`cat_int.pwr_elec_in_watts`). Check that save intervals were "
                "not trimmed for these components."
            )

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
def _bel_with_cat_loco_cost(year: int, battery_scenario: str = "Moderate") -> float:
    """BEL-with-catenary unit cost ($): the usual battery-electric cost (NREL ATB
    battery pack via ``metric_calculator.add_battery_costs``) plus the pantograph
    premium (``defaults.PANTOGRAPH_COST_USD``)."""
    import pandas as pd

    info_df = pd.DataFrame({"Rust_Loco": [_bel_with_cat_loco()],
                            "Cost_USD": [float("nan")]})
    info_df = _mc.add_battery_costs(info_df, year, scenario=battery_scenario)
    return float(info_df.at[0, "Cost_USD"]) + _defaults.PANTOGRAPH_COST_USD


def _infrastructure_costs_df(plan_result, built_subs, arch=ITEMIZED_ARCH,
                             infra_mult=1.0, net_purchased=None,
                             opt_infra_mult=None) -> pl.DataFrame:
    """Per-asset catenary infrastructure capital ($) for the metrics rollover,
    using ``arch``'s medium costs scaled by ``infra_mult`` (the infrastructure-cost
    sweep level): one aggregate structure asset, one aggregate energization asset,
    and one row per built (non-existing) substation.

    Structure is charged per corridor route-mile -- the optimizer's own basis,
    once per site however many parallel tracks share the masts, read from
    ``plan_result.structure_route_miles`` -- and energization per track-mile.
    A plan solved before the optimizer recorded route-miles has that field
    ``None``; then the mileage is backed out exactly from the optimizer's capital
    total at the multiplier the plan was OPTIMIZED at (``opt_infra_mult``,
    defaulting to ``infra_mult``; BEL plans are optimized at reference costs, so
    their callers pass 1.0). When ``net_purchased`` is given, the energization (marginal
    wire) miles are read straight from that network's ``cat_power_limits`` -- i.e.
    the AS-PURCHASED wire after direction exclusion strips catenary the fleet never
    charges from -- while structure keeps the full optimizer mileage (masts serve
    the site regardless of direction). ``None`` uses the full mileage for both.

    The ``Distance_Miles`` / ``Distance_Units`` columns feed the metrics
    calculator's ``Distance_Energized`` metric: energization carries the built
    catenary wire length (track-miles). Structure's is left null on purpose so
    that metric keeps reporting wire only.
    """
    med = ARCH_MEDIUM_COSTS[arch]
    track_miles = plan_result.energized_track_miles or 0.0
    n_subs = sum(1 for s in built_subs if not getattr(s, "existing", False))
    struct_miles = getattr(plan_result, "structure_route_miles", None)
    if struct_miles is None:
        m = infra_mult if opt_infra_mult is None else opt_infra_mult
        struct_miles = ((plan_result.total_cost or 0.0)
                        - med["energization"] * m * track_miles
                        - med["substation"] * m * n_subs) / (med["structure"] * m)
        if not (-1e-6 <= struct_miles <= track_miles + 1e-6):
            raise ValueError(f"backed-out structure {struct_miles:.2f} route-mi is "
                             f"outside [0, {track_miles:.2f}] track-mi")
        struct_miles = max(struct_miles, 0.0)
    energ_miles = (_net_energized_track_miles(net_purchased)
                   if net_purchased is not None else track_miles)
    rows = [
        {"Infrastructure_Type": "Catenary_Structure",
         "Cost_USD": med["structure"] * infra_mult * USD_PER_MUSD * struct_miles,
         "Lifespan_Years": STRUCTURE_LIFESPAN_YEARS,
         "Distance_Miles": None, "Distance_Units": None},
        {"Infrastructure_Type": "Catenary_Energization",
         "Cost_USD": med["energization"] * infra_mult * USD_PER_MUSD * energ_miles,
         "Lifespan_Years": ENERGIZATION_LIFESPAN_YEARS,
         "Distance_Miles": energ_miles, "Distance_Units": "track-miles"},
    ]
    rows += [
        {"Infrastructure_Type": "Substation",
         "Cost_USD": med["substation"] * infra_mult * USD_PER_MUSD,
         "Lifespan_Years": SUBSTATION_LIFESPAN_YEARS,
         "Distance_Miles": None, "Distance_Units": None}
        for _ in range(n_subs)
    ]
    return pl.DataFrame(rows)


def _classify_loco_type(loco_dict) -> str:
    """Canonical fleet-costing label for one locomotive from its ``to_pydict``
    ``loco_type`` variant: ``"BEL"`` (battery-electric, incl. pantograph-equipped),
    ``"Dual_Mode"`` (diesel + catenary), or ``"Diesel"`` (conventional). These match
    the keys of the per-variant ``cost_fns`` maps so mixed consists (e.g. a BEL
    train carrying a diesel helper) are priced and counted per actual unit."""
    lt = loco_dict.get("loco_type", {})
    if "BatteryElectricLoco" in lt:
        return "BEL"
    if "DualModeLoco" in lt:
        return "Dual_Mode"
    return "Diesel"  # ConventionalLoco


def _price_pool_for_year(pool, cost_fns, year):
    """Re-price a ``loco_pool`` for one rollout ``year``, per ``Locomotive_Type``.

    ``cost_fns`` maps each canonical label (see :func:`_classify_loco_type`) to a
    ``year -> USD`` cost function, so each unit is priced at its year-of-purchase
    cost by TYPE -- exact for mixed BEL + diesel-helper consists.
    """
    price_map = pl.DataFrame({
        "Locomotive_Type": list(cost_fns.keys()),
        "Cost_USD": [float(fn(year)) for fn in cost_fns.values()],
    })
    return pool.drop("Cost_USD").join(price_map, on="Locomotive_Type", how="left")


def _build_fleet_tables(sims_list, cost_fns, samples=(0,)):
    """Build the ``loco_pool`` + ``consist_plan`` tables spanning the full plan.

    Enumerates every locomotive in every plan train (``sims_list`` order == the
    ``SpeedLimitTrainSimVec`` index) so the metrics calculator can attribute
    purchases and energy per train. Each unit is typed by its ACTUAL locomotive
    variant (:func:`_classify_loco_type`) and priced from ``cost_fns`` at the
    start-year cost, so a mixed BEL consist carrying a diesel helper contributes a
    correctly-typed, correctly-priced diesel unit rather than an approximated BEL.
    Per-year re-pricing over the rollout is applied later by
    :func:`_price_pool_for_year`.
    """
    plan = _level_measured_plan(samples)
    pool_rows, cp_rows = [], []
    loco_id = 0
    for idx, ts in enumerate(sims_list):
        loco_vec = ts.to_pydict()["loco_con"]["loco_vec"]
        train = plan[idx] if idx < len(plan) else {}
        train_id = str(train.get("Train_ID", idx))
        train_type = train.get("Train_Type", "Manifest")
        for loco in loco_vec:
            label = _classify_loco_type(loco)
            loco_id += 1
            pool_rows.append({
                "Locomotive_ID": loco_id, "Locomotive_Type": label,
                "Cost_USD": float(cost_fns[label](METRICS_START_YEAR)),
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
    # Central ("med") prices are each zone's trajectory at this rollout year; the
    # gap multipliers widen low/high. Electricity is priced per node from the node's
    # zone; diesel once, as the terminal-zone mean.
    diesel_price = corridor_diesel_price(scenario_year) * mult["diesel"]
    regions = list(region_elec_j.keys())
    # calculate_energy_cost multiplies electricity kWh by Price/100, so the
    # electricity price is in CENTS/kWh; diesel is $/gallon.
    nodal_energy_prices = pl.DataFrame(
        [{"Node": r, "Fuel": "Electricity",
          "Price": zone_elec_price(node_zone(r), scenario_year) * mult["elec"] * 100.0}
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


def _variant_metrics_df(tag, loco_type, cost_fns, energy, plan_result, built_subs,
                        region_frac, *, arch, demand_lvl, demand_samples, infra_lvl,
                        infra_mult, energy_lvl, net_purchased=None,
                        stationary_sessions=None, opt_infra_mult=None):
    """Run the metrics calculator for ONE simulated variant at ONE economic point
    (infra cost x energy price); return its tidy long-format metrics cross-joined
    with a single-row provenance frame.

    ``energy`` is the walked-sim bundle from ``_run_full_plan`` (dual-mode, BEL, or
    conventional diesel) -- the SAME cached sim is re-priced across the economic
    sweep, so infra/energy never re-run the simulation. Catenary variants pass their
    infrastructure (scaled by ``infra_mult``) + built substations; the diesel
    baseline (``tag == "diesel"``) has no infrastructure.
    ``net_purchased`` is the AS-PURCHASED (direction-excluded) network; the
    energization cost is read straight from its ``cat_power_limits``, so no separate
    exclusion bookkeeping is threaded in.

    ``cost_fns`` maps each canonical locomotive label (see
    :func:`_classify_loco_type`) to a ``year -> USD`` cost function; each unit is
    priced by its ACTUAL type, so a BEL consist carrying a diesel helper is costed
    exactly (BEL units at the BEL price, the helper at the diesel price).
    ``loco_type`` is the representative label used for provenance reporting.
    """
    mult = ENERGY_LEVELS[energy_lvl]
    ps = _plan_summary()
    rollout_years = [METRICS_START_YEAR + k for k in range(METRICS_ROLLOUT_YEARS)]
    ef_by_region = _mc.interpolate_emissions_factors_cambium(
        [GRID_REGION_WEST, GRID_REGION_EAST], rollout_years, CAMBIUM_SCENARIO)
    infra_df = (None if tag == "diesel"
                else _infrastructure_costs_df(plan_result, built_subs, arch,
                                              infra_mult, net_purchased=net_purchased,
                                              opt_infra_mult=opt_infra_mult))
    full_miles = (plan_result.energized_track_miles or 0.0
                  if plan_result is not None else 0.0)
    purchased_miles = (_net_energized_track_miles(net_purchased)
                       if net_purchased is not None else full_miles)
    excl_miles = max(full_miles - purchased_miles, 0.0)
    # A demand level's sims ALREADY contain its extra trains -- one sample's
    # worth per sample, concatenated by `_merge_plan_runs` -- so the represented
    # window stays one week and annualization is the plain 365.25 / PLAN_DAYS.
    # The previous design faked demand by shrinking this window instead, which
    # scaled energy without ever putting more trains on the corridor.
    _lvl = _level_plan_summary(demand_samples)
    simulation_days = PLAN_DAYS
    n_locos = int(_lvl["total_locos"])
    loco_cost_ref = float(cost_fns[loco_type](METRICS_START_YEAR))  # representative
    _log(f"METRICS [{tag}] {arch}/{demand_lvl}/infra-{infra_lvl}/energy-{energy_lvl}: "
         f"{METRICS_ROLLOUT_YEARS}-yr rollout of {_lvl['n_trains']} trains "
         f"(loco ${loco_cost_ref/1e6:.2f}M @{METRICS_START_YEAR})")
    sims = alt.SpeedLimitTrainSimVec(list(energy["sims"]))
    # Consist plan is year-invariant; the loco pool's Cost_USD is set per rollout
    # year (by locomotive TYPE) so purchases are priced at their year of purchase in
    # the DCF rollover -- exact for mixed BEL + diesel-helper consists.
    loco_pool_base, consist_plan = _build_fleet_tables(
        energy["sims"], cost_fns, demand_samples)
    # Grid-side stationary (terminal) charging energy, week total, so the metrics
    # calculator sees it in the electricity total and reports it as
    # Electricity_Stationary_Charging (total grid electricity minus catenary draw).
    stationary_j = 0.0
    if stationary_sessions is not None and stationary_sessions.height > 0:
        stationary_j = float(
            stationary_sessions.get_column("Refuel_Energy_J").sum() or 0.0)
    region_elec_j = _split_electricity(energy, region_frac, stationary_j)
    # Emissions factors keyed by the same "<region>|<zone>" nodes the electricity is
    # priced on, copying each node's region.
    emissions_factors = composite_emissions(ef_by_region, list(region_elec_j))
    elec_total_j = sum(region_elec_j.values())
    elec_price_start = (
        sum(zone_elec_price(node_zone(n), METRICS_START_YEAR) * j for n, j in region_elec_j.items())
        / elec_total_j if elec_total_j > 0
        else sum(zone_elec_price(node_zone(n), METRICS_START_YEAR) for n in region_elec_j)
        / len(region_elec_j)) * mult["elec"]
    scenarios = [
        _year_scenario(
            year, sims,
            _price_pool_for_year(loco_pool_base, cost_fns, year),
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
        "demand_samples": len(demand_samples),
        "demand_multiplier": float(len(demand_samples)),
        "infra_level": infra_lvl,
        "infra_multiplier": infra_mult,
        "energy_level": energy_lvl,
        # Start-year prices: diesel is the terminal-zone mean; electricity is the
        # draw-weighted mean over the pricing nodes, which are listed with shares.
        "diesel_price_usd_per_gal": corridor_diesel_price(METRICS_START_YEAR) * mult["diesel"],
        "elec_price_usd_per_kwh": elec_price_start,
        "elec_price_nodes": ";".join(
            f"{n}:{(j / elec_total_j if elec_total_j > 0 else 0.0):.3f}"
            for n, j in region_elec_j.items()),
        "discount_rate": DISCOUNT_RATE,
        "rollout_years": METRICS_ROLLOUT_YEARS,
        "start_year": METRICS_START_YEAR,
        "n_trains": _lvl["n_trains"],
        "n_locos": n_locos,
        "energized_track_miles": (plan_result.energized_track_miles or 0.0)
        if plan_result is not None else 0.0,
        "excluded_direction_miles": excl_miles,
        # Diesel (arch "none") has no catenary, hence no energization to exclude.
        "direction_exclusion_savings_musd": (
            ARCH_MEDIUM_COSTS[arch]["energization"] * infra_mult * excl_miles
            if arch in ARCH_MEDIUM_COSTS else 0.0),
        "optimality_gap_pct": (plan_result.optimality_gap_rel * 100.0
                               if plan_result is not None
                               and plan_result.optimality_gap_rel is not None
                               else None),
        "solve_status": (str(plan_result.status).split(".")[-1]
                         if plan_result is not None else None),
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
        "optimality_gap_abs_musd": (result.optimality_gap_abs
                                    if result is not None else None),
        "optimality_gap_pct": (result.optimality_gap_rel * 100.0
                               if result is not None
                               and result.optimality_gap_rel is not None else None),
        "objective_bound_musd": (result.objective_bound
                                 if result is not None else None),
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


def _train_timeseries_df(sims, samples=(0,)) -> pl.DataFrame:
    """Concatenate every walked sim's ``to_dataframe()`` with ``train_id`` /
    ``train_index`` columns (the plan train id when available, else the index)."""
    plan = _level_measured_plan(samples)
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
                           refuel_sessions=None, samples=(0,)):
    """Write one run's outputs to ``folder`` as Parquet: optimization outputs
    (catenary variants only), the network + substation locations (+ a network msgpack
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
        # msgpack rather than yaml: ~5x smaller and far faster to write, which matters
        # when a sweep writes one of these per run. Lossless; read back with
        # `alt.Network.from_file`.
        net.to_file(str(folder / "network.msgpack"))
    except Exception as exc:  # noqa: BLE001 - network file is a convenience artifact
        _log(f"  network.msgpack write FAILED for {folder.name}: {exc}")
    metrics_df.write_parquet(folder / "metrics.parquet")
    _train_timeseries_df(sims, samples).write_parquet(
        folder / "train_timeseries.parquet")
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
