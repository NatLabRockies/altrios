"""Multi-year discounted costs for every DoE cell, from the metrics calculator.

The sweep simulates one representative week per cell. This stage turns each
week into a project: ``METRICS_ROLLOUT_YEARS`` identical operating years, priced
year by year and discounted to the base year by ``altrios.metric_calculator``'s
own discounted-cash-flow engine -- the SAME engine and the SAME assumptions the
catenary deep-dive uses, so the two studies' lifecycle costs are comparable.

Nothing is re-simulated. ``run_sweep`` persists each cell's walked sims, loco
pool and catenary draw; this reads them back and builds, per cost cell, the
list of per-year ``ScenarioInfo`` the calculator expects:

- **Locomotive fleet**, priced per unit by its ACTUAL type at its year of
  purchase (so a battery-electric consist carrying a diesel helper is costed as
  BEL units plus one diesel), with ``LOCO_LIFESPAN`` replacements inside the
  horizon handled by the calculator.
- **Catenary infrastructure** as ``infrastructure_costs`` -- structure and
  energization per track-mile plus one row per substation, each with its own
  lifespan, scaled by the cell's infrastructure-cost multiplier.
- **Energy** from the simulated diesel and the catenary grid draw, priced by
  region: each terminal and each feeding substation sits in a price zone
  (US state / PADD region / Canadian province, see ``Energy_Price_Zones.csv``)
  whose 2025 reference price rides the AEO2026 (US) or Canada's Energy Future
  2026 (Canada) trajectory for each rollout year, widened by the cell's
  energy-price level. Catenary draw is split by feeding substation, so it is
  priced where the ton-miles are; diesel is the mean of the terminal zones.
  Grid emissions come from Cambium for the corridor's regions.

Output: ``out/rollout_costs.csv`` -- the calculator's tidy metrics
(``Cost_Total_Discounted``, ``Cost_Purchases``, ``Cost_Energy``, ``LCOTKM``,
...) cross-joined with each cell's provenance. Cells whose bundle is missing are
skipped and named, never silently zeroed.

Usage:
    python rollout_costs.py                 # every cell with a saved bundle
    python rollout_costs.py --sim-key KEY   # one cell
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import polars as pl

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "catenary_optimization"))

# The catenary app's speed rule has to be set before it is imported; its price
# trajectories and cost constants are what this stage reuses.
os.environ.setdefault("ALTRIOS_SPEED_CAP_MPH", "33.983779")
os.environ.setdefault("ALTRIOS_SPEED_SCALER", "none")

import altrios as alt  # noqa: E402
from altrios import defaults, utilities  # noqa: E402
from altrios import energization as en  # noqa: E402
from altrios import metric_calculator as mc  # noqa: E402
import energization_optimization_corridor as cat  # noqa: E402

logger = logging.getLogger("rollout_costs")
OUT = HERE / "out"
SIMS = OUT / "sims"

# What the sweep simulated: one representative week.
SIMULATION_DAYS = 7.0

# One row per rollout year; the calculator's cross-year functions need the
# per-year fleet counts, unit costs and lifespans these produce.
ANNUAL_METRICS = [
    ("Freight_Moved", "million tonne-mi"),
    ("Freight_Moved", "million tonne-km"),
    ("Freight_Moved", "car-miles"),
    ("Count_Locomotives", "assets"),
    ("Count_Infrastructure", "assets"),
    # Terminal chargers (battery-electric and hybrid cells): counts, unit cost
    # and lifespan from the refuel facilities each ScenarioInfo carries, so the
    # calculator's rollout prices them like locomotives and infrastructure.
    ("Count_Refuelers", "assets"),
    ("Energy_Costs", "USD"),
    ("Energy", "MJ"),
    # Use-phase CO2-eq per year: GREET diesel factor, Cambium grid factor by
    # region and year (the emissions factors handed to each ScenarioInfo).
    ("GHG", "tonne CO2-eq"),
]


# ------------------------------------------------------------------ pricing
def _atb_pack_usd(loco: alt.Locomotive, year: int,
                  battery_scenario: str = "Moderate") -> float:
    """The NREL battery-system cost (energy $/kWh + power $/kW components) for this
    locomotive's pack in ``year``, restated to the 2025 base year, as
    `metric_calculator.add_battery_costs` prices it -- without that function's BEL
    chassis term."""
    prices = mc.battery_prices(battery_scenario).filter(
        (pl.col("Year") - pl.lit(year)).abs().min()
        == (pl.col("Year") - pl.lit(year)).abs())
    def value(metric):
        return prices.filter(pl.col("Metric") == metric).get_column("Value")[0]
    inner = next(iter(loco.to_pydict()["loco_type"].values()))
    res = inner.get("res")
    if not res:
        return 0.0
    kw = res["pwr_out_max_watts"] / 1000
    kwh = res["energy_capacity_joules"] * 1e-6 * utilities.KWH_PER_MJ
    return ((kw * value("USD_Per_kW") + kwh * value("USD_Per_kWh"))
            * utilities.USD_2024_TO_2025 * defaults.RETAIL_PRICE_EQUIVALENT_MULTIPLIER)


def cost_functions(row: dict, battery_scenario: str = "Moderate"):
    """``Locomotive_Type -> (year -> USD)`` for every type in this cell's pool.

    The sweep's pool labels its main units `Dual_Mode` (a planner convention,
    not the powertrain) and helpers `Diesel_Helper`; the powertrain decides what
    a main unit actually is and therefore costs.
    """
    import run_sweep  # the same locomotive definitions the sweep simulated

    powertrain = row["Powertrain"]
    battery = row.get("Battery_MWh")
    battery = None if battery in (None, "", "nan") else float(battery)
    diesel = lambda _y: defaults.DIESEL_LOCO_COST_USD  # noqa: E731
    if powertrain == "diesel":
        main = diesel
    elif powertrain == "catenary_dual":
        main = lambda _y: defaults.DUAL_MODE_LOCO_COST_USD  # noqa: E731
    elif powertrain == "catenary_bel":
        main = lambda y, _s=battery_scenario: cat._bel_with_cat_loco_cost(y, _s)  # noqa: E731
    elif powertrain == "hybrid":
        # No sourced hybrid price exists anywhere in the codebase. The stated
        # assumption: a conventional diesel unit plus its pack at the ATB price
        # for the purchase year -- the calculator's own battery pricing, applied
        # on top of the diesel chassis instead of the BEL chassis. Labelled as
        # an assumption in the outputs.
        loco = run_sweep.build_locomotive("hybrid", battery)
        main = lambda y, _l=loco, _s=battery_scenario: (  # noqa: E731
            defaults.DIESEL_LOCO_COST_USD + _atb_pack_usd(_l, y, _s))
    else:
        raise ValueError(f"unknown powertrain {powertrain!r}")
    return {run_sweep.PLAN_LOCO_TYPE: main, run_sweep.HELPER_LOCO_TYPE: diesel}


def price_pool(pool: pl.DataFrame, cost_fns, year: int) -> pl.DataFrame:
    """Re-price the pool for one rollout year, per locomotive type."""
    price_map = pl.DataFrame({
        "Locomotive_Type": list(cost_fns),
        "Cost_USD": [float(fn(year)) for fn in cost_fns.values()],
    })
    pool = pool.with_columns(pl.col("Locomotive_Type").cast(pl.Utf8))
    if "Cost_USD" in pool.columns:
        pool = pool.drop("Cost_USD")
    pool = pool.join(price_map, on="Locomotive_Type", how="left")
    if "Lifespan_Years" not in pool.columns:
        pool = pool.with_columns(pl.lit(defaults.LOCO_LIFESPAN).alias("Lifespan_Years"))
    return pool


# ------------------------------------------------------------------ infrastructure
def _n_existing_substations() -> int:
    """Pre-existing (free) substations in the corridor module. The optimizer never
    prices them, but `_built_sub_objs` -- and so `substations_built` in
    layouts.csv -- lists them alongside the candidates it selected."""
    return sum(1 for s in cat.substations if getattr(s, "existing", False))


def _candidate_substations(lay: dict) -> int:
    """Substations the optimizer actually bought for a layout."""
    direct = lay.get("candidate_substations")
    if direct is not None and direct == direct:
        return int(direct)
    n = int(lay["substations_built"] or 0) - _n_existing_substations()
    if n < 0:
        raise ValueError(f"layout {lay['fingerprint']}: {lay['substations_built']} "
                         f"substations listed, fewer than the corridor's existing ones")
    return n


def structure_route_miles(lay: dict, med: dict) -> float:
    """Shared-structure route-miles the optimizer charged for a layout.

    Structure (masts, foundations) is priced once per corridor site however many
    parallel tracks share it, so its basis is route-miles, not the track-miles
    `energized_track_miles` reports -- on this corridor's yards and terminals the
    two differ by more than 3x. Layouts built after the optimizer began recording
    `structure_route_miles` carry it directly; older ones back it out exactly
    from the optimizer's own capital total, which is
    structure*route_mi + energization*track_mi + substation*n_candidates, all at
    the multiplier the layout was optimized at (dual-mode at its cell's, BEL at
    reference).
    """
    direct = lay.get("structure_route_miles")
    if direct is not None and direct == direct:  # present and not NaN
        return float(direct)
    m = float(lay.get("infra_multiplier") or 1.0)
    miles = float(lay["energized_track_miles"] or 0.0)
    capex = float(lay["capex_musd"] or 0.0)
    route = (capex - med["energization"] * m * miles
             - med["substation"] * m * _candidate_substations(lay)) / (med["structure"] * m)
    if not (-1e-6 <= route <= miles + 1e-6):
        raise ValueError(f"layout {lay['fingerprint']}: backed-out structure "
                         f"{route:.2f} route-mi is outside [0, {miles:.2f}] track-mi")
    return max(route, 0.0)


def infrastructure_costs(row: dict, layouts: pl.DataFrame) -> Optional[pl.DataFrame]:
    """Per-asset catenary capital for this cell, or None for the diesel arms.

    Mirrors the deep-dive's `_infrastructure_costs_df`: structure per shared
    route-mile, energization per energized track-mile, one row per substation
    the optimizer bought, each with its own lifespan, all scaled by the cell's
    infra multiplier. Built from `layouts.csv` rather than the optimizer result
    object, which the sweep does not keep; the row is matched on architecture
    and powertrain as well as fingerprint because one buildout can be shared
    across architectures, and its recorded capital is architecture-priced.
    """
    if row["Powertrain"] not in ("catenary_dual", "catenary_bel"):
        return None
    fp = row.get("Layout_Fingerprint")
    if fp in (None, "", "nobuild", "na"):
        return None  # unelectrified: nothing was built
    lay = layouts.filter(pl.col("fingerprint") == fp,
                         pl.col("Architecture") == row["Architecture"],
                         pl.col("Powertrain") == row["Powertrain"])
    if lay.height == 0:
        raise FileNotFoundError(f"layout {fp} ({row['Architecture']}, "
                                f"{row['Powertrain']}) not in layouts.csv")
    lay = lay.row(0, named=True)
    miles = float(lay["energized_track_miles"] or 0.0)
    med = cat.ARCH_MEDIUM_COSTS[row["Architecture"]]
    route_miles = structure_route_miles(lay, med)
    n_subs = _candidate_substations(lay)
    mult = float(row["Infra_Multiplier"])
    # Structure's distance is left null on purpose: the calculator's
    # Distance_Energized metric then keeps reporting wire track-miles only, as
    # the deep-dive's does.
    rows = [
        {"Infrastructure_Type": "Catenary_Structure",
         "Cost_USD": med["structure"] * mult * cat.USD_PER_MUSD * route_miles,
         "Lifespan_Years": cat.STRUCTURE_LIFESPAN_YEARS,
         "Distance_Miles": None, "Distance_Units": None},
        {"Infrastructure_Type": "Catenary_Energization",
         "Cost_USD": med["energization"] * mult * cat.USD_PER_MUSD * miles,
         "Lifespan_Years": cat.ENERGIZATION_LIFESPAN_YEARS,
         "Distance_Miles": miles, "Distance_Units": "track-miles"},
    ] + [
        {"Infrastructure_Type": "Substation",
         "Cost_USD": med["substation"] * mult * cat.USD_PER_MUSD,
         "Lifespan_Years": cat.SUBSTATION_LIFESPAN_YEARS,
         "Distance_Miles": None, "Distance_Units": None}
        for _ in range(n_subs)
    ]
    return pl.DataFrame(rows)


# ------------------------------------------------------------------ regions and price zones
_REGION_CACHE: Dict[str, List[str]] = {}
_LOCATION_CACHE: Dict[str, dict] = {}


def _locations(corridor_slug: str) -> dict:
    if corridor_slug not in _LOCATION_CACHE:
        from speed_calibration import CORRIDORS, NETWORKS
        _LOCATION_CACHE[corridor_slug] = alt.import_locations(
            str(NETWORKS / CORRIDORS[corridor_slug].locations))
    return _LOCATION_CACHE[corridor_slug]


def cambium_regions(corridor_slug: str) -> List[str]:
    """The Cambium grid regions this corridor's locations sit in."""
    if corridor_slug not in _REGION_CACHE:
        _REGION_CACHE[corridor_slug] = sorted({
            loc.to_pydict()["Grid Emissions Region"]
            for locs in _locations(corridor_slug).values() for loc in locs
        })
    return _REGION_CACHE[corridor_slug]


def terminal_zones(corridor_slug: str) -> pl.DataFrame:
    """One row per terminal: ``Node``, ``Grid_Region``, ``Electricity_Zone``,
    ``Diesel_Zone`` -- the price zones the corridor's location file assigns."""
    rows = []
    for locs in _locations(corridor_slug).values():
        for loc in locs:
            d = loc.to_pydict()
            rows.append({"Node": d["Location ID"],
                         "Grid_Region": d["Grid Emissions Region"],
                         "Electricity_Zone": d["Electricity Price Region"],
                         "Diesel_Zone": d["Liquid Fuel Price Region"]})
    return pl.DataFrame(rows).unique(maintain_order=True)


def price_zone_boundaries(corridor_slug: str) -> Optional[pl.DataFrame]:
    """The corridor's lat/lon price-zone boxes (``in/corridors.csv`` column
    ``Price_Zones_Path``, a file under the networks resources), or None when the
    corridor has none. Rows are tried in order; a null bound is open."""
    table = pl.read_csv(HERE / "in" / "corridors.csv", infer_schema_length=None)
    row = table.filter(pl.col("Corridor") == corridor_slug)
    if row.height == 0:
        raise KeyError(f"{corridor_slug} is not in in/corridors.csv")
    if "Price_Zones_Path" not in row.columns:
        return None
    path = row.get_column("Price_Zones_Path")[0]
    if path in (None, "", "nan"):
        return None
    from speed_calibration import NETWORKS
    return pl.read_csv(NETWORKS / path)


# Zone geometry and pricing-node helpers are the catenary app's, so the deep-dive
# and the DoE place substations and build "<region>|<zone>" nodes identically.
zone_at = cat.zone_at
NODE_SEP = cat.NODE_SEP
grid_node = cat.grid_node
node_region = cat.node_region
node_zone = cat.node_zone


def grid_draw_shares(sims, corridor_slug: str, safe: str) -> Dict[str, float]:
    """Share of the week's catenary grid draw by pricing node ``<region>|<zone>``.

    Substation-resolved: `energization.compute_line_losses` apportions each dual-mode
    sim's pantograph draw plus line loss to the feeding substation (district); that
    substation's position (the catenary module's ``substations``, by id) is placed in
    a price zone by the corridor's zone boxes, and its grid-emissions region rides
    along. A train's draw is therefore priced where the wire under it is fed from --
    ton-miles in Alberta buy Alberta power. Cached in ``out/sims/<safe>_grid_nodes.json``
    (delete it to recompute after changing the zone boxes).

    Falls back to an even split across the terminals' (region, zone) pairs when the
    corridor has no zone boxes, no sims are given, or the sims carry no district
    split (diesel and hybrid cells, whose grid draw is zero anyway).
    """
    cache = SIMS / f"{safe}_grid_nodes.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    shares: Optional[Dict[str, float]] = None
    boundaries = price_zone_boundaries(corridor_slug)
    if boundaries is not None and sims is not None:
        subs = {str(s.id): s for s in cat.substations}
        by_node: Dict[str, float] = {}
        for ts in sims.to_pydict():
            df = en.compute_line_losses(ts).filter(pl.col("district_id") != "__total__")
            for r in df.iter_rows(named=True):
                did = r["district_id"]
                if did is None or str(did) not in subs:
                    continue  # unfed track or an unknown district: nothing to price
                s = subs[str(did)]
                zone = zone_at(float(s.lat), float(s.lon), boundaries)["Electricity_Price_Region"]
                node = grid_node(s.grid_emissions_region, zone)
                by_node[node] = by_node.get(node, 0.0) + float(r["substation_energy_joules"])
        total = sum(by_node.values())
        if total > 0:
            shares = {n: v / total for n, v in sorted(by_node.items())}
    if shares is None:
        tz = terminal_zones(corridor_slug)
        nodes = sorted({grid_node(r["Grid_Region"], r["Electricity_Zone"])
                        for r in tz.iter_rows(named=True)})
        shares = {n: 1.0 / len(nodes) for n in nodes}
    if sims is not None:  # only cache a split that came from the sims themselves
        cache.write_text(json.dumps(shares, indent=1), encoding="utf-8")
    return shares


def diesel_zones(corridor_slug: str) -> List[str]:
    """The distinct diesel price zones of the corridor's terminals."""
    return sorted(set(terminal_zones(corridor_slug).get_column("Diesel_Zone").to_list()))


composite_emissions = cat.composite_emissions


def energy_prices(year: int, energy_level: str, elec_nodes: Sequence[str],
                  diesel_zone_list: Sequence[str]):
    """``(diesel USD/gal, {node: electricity US cents/kWh})`` at ``year`` for one
    energy-price level: each zone's reference price on its own trajectory
    (`metric_calculator.zone_energy_price`), widened by the level's gap multipliers.
    Diesel is the mean over the corridor's terminal zones -- the calculator prices
    diesel with one network-average price, and a run-through train fuels at both ends."""
    mult = cat.ENERGY_LEVELS[energy_level]
    diesel = (sum(mc.zone_energy_price(z, "Diesel", year) for z in diesel_zone_list)
              / len(diesel_zone_list) * mult["diesel"])
    elec = {n: mc.zone_energy_price(node_zone(n), "Electricity", year) * mult["elec"]
            for n in elec_nodes}
    return diesel, elec


# ------------------------------------------------------------------ one cell
def year_scenario(year, sims, pool, consist_plan, grid_j_by_node, energy_level,
                  emissions_factors, infra_df, diesel_zone_list, refuel_facilities=None):
    """One rollout year's ScenarioInfo, priced by zone at that year's trajectories."""
    nodes = list(grid_j_by_node)
    diesel_price, elec_cents = energy_prices(year, energy_level, nodes, diesel_zone_list)
    # The calculator prices electricity in CENTS per kWh and diesel in $/gal.
    nodal_prices = pl.DataFrame(
        [{"Node": n, "Fuel": "Electricity", "Price": elec_cents[n]} for n in nodes]
        + [{"Node": "corridor", "Fuel": "Diesel", "Price": diesel_price}])
    # Grid electricity enters as refuel sessions, one per pricing node
    # ("<region>|<zone>"), so the calculator prices it by zone and attributes its
    # emissions by region.
    sessions = pl.DataFrame({
        "Locomotive_ID": pl.Series([1] * len(nodes), dtype=pl.UInt32),
        "Locomotive_Type": pl.Series([pool["Locomotive_Type"][0]] * len(nodes), dtype=pl.Utf8),
        "Fuel_Type": pl.Series(["Electricity"] * len(nodes), dtype=pl.Utf8),
        "Node": pl.Series(nodes, dtype=pl.Utf8),
        "Refuel_Energy_J": pl.Series([float(grid_j_by_node[n]) for n in nodes],
                                     dtype=pl.Float64),
    })
    return mc.ScenarioInfo(
        sims=sims, simulation_days=SIMULATION_DAYS, annualize=True,
        scenario_year=year, loco_pool=pool, consist_plan=consist_plan,
        refuel_facilities=refuel_facilities, refuel_sessions=sessions,
        emissions_factors=emissions_factors, nodal_energy_prices=nodal_prices,
        count_unused_locomotives=True, train_planner_config=None,
        infrastructure_costs=infra_df,
    )


# ------------------------------------------------------------------ terminal charging
# Hybrid charger counts are scaled from a battery-electric cell's (see
# `terminal_charging`). A corridor with no battery-electric cell borrows this
# one's installed ports per unit of weekly terminal charging energy.
REFERENCE_BEL_CORRIDOR = "spokane_lethbridge"
REFERENCE_BEL_ARCHITECTURE = "ac_25kv"
_TERMINAL_CACHE: Dict[str, dict] = {}


def _safe(sim_key: str) -> str:
    return sim_key.replace("|", "__").replace(".", "p")


def terminal_charging_j(row: dict, safe: str, sims=None) -> Dict[str, float]:
    """Grid joules of terminal charging per terminal for one simulation (week).

    From the sidecar when the sweep recorded it; otherwise reconstructed from the
    persisted plan and sims the same way the sweep now does, so cells simulated
    before terminal charging was tracked are priced consistently."""
    if safe in _TERMINAL_CACHE:
        return _TERMINAL_CACHE[safe]
    import run_sweep  # noqa: F401 -- puts the DoE modules on the path
    import terminal_charging as tc
    sidecar = json.loads((SIMS / f"{safe}_sidecar.json").read_text(encoding="utf-8"))
    if "terminal_charging_j" in sidecar:
        out = {k: float(v) for k, v in (sidecar["terminal_charging_j"] or {}).items()}
    elif row["Powertrain"] not in ("hybrid", "catenary_bel"):
        out = {}
    else:
        plan = pl.read_csv(SIMS / f"{safe}_plan.csv", infer_schema_length=None)
        if sims is None:
            sims = alt.SpeedLimitTrainSimVec.from_file(str(SIMS / f"{safe}_sims.msgpack"))
        # Departure SOC is read straight off the persisted plan: the planner
        # stamped it under its headroom cap, so no reconstruction is needed.
        sessions = tc.charging_sessions(plan, sims.to_pydict(),
                                        defaults.BEL_CHARGER_EFFICIENCY)
        out = tc.energy_by_node(sessions)
    _TERMINAL_CACHE[safe] = out
    return out


def _bel_chargers(safe: str) -> Optional[pl.DataFrame]:
    """The battery-electric cell's own chargers: the planner's electric refuelers
    (persisted by the sweep), else rebuilt from the persisted pool's port count
    with the planner's charger template."""
    import run_sweep
    path = SIMS / f"{safe}_refuelers.csv"
    if path.exists():
        ref = pl.read_csv(path, infer_schema_length=None)
        ref = ref.filter(pl.col("Fuel_Type") == "Electricity")
        return ref if ref.height else None
    pool = pl.read_csv(SIMS / f"{safe}_loco_pool.csv", infer_schema_length=None)
    elec = pool.filter(pl.col("Fuel_Type") == "Electricity")
    if elec.height == 0:
        return None
    ports = float(elec["Port_Count"].max())
    tpl = run_sweep.planner_config_for("catenary_bel", None).refueler_info
    tpl = pl.from_pandas(tpl[tpl["Fuel_Type"] == "Electricity"].head(1))
    nodes = sorted(pool["Node"].unique().to_list())
    return pl.concat([tpl.with_columns(pl.lit(n).alias("Node"),
                                       pl.lit(ports).alias("Port_Count")) for n in nodes],
                     how="diagonal_relaxed")


def charger_facilities(row: dict, safe: str, experiments: pl.DataFrame,
                       sims=None) -> Optional[pl.DataFrame]:
    """Refuel facilities whose capital the rollout prices: the battery-electric
    cell's own chargers, or for a hybrid the reference battery-electric chargers
    scaled by terminal charging energy (`terminal_charging.scaled_chargers`)."""
    import terminal_charging as tc
    if row["Powertrain"] == "catenary_bel":
        return _bel_chargers(safe)
    if row["Powertrain"] != "hybrid":
        return None
    own = terminal_charging_j(row, safe, sims)
    if not own:
        return None
    bels = experiments.filter((pl.col("Powertrain") == "catenary_bel")
                              & (pl.col("Architecture") == REFERENCE_BEL_ARCHITECTURE))
    same = bels.filter(pl.col("Corridor") == row["Corridor"])
    ref = same if same.height else bels.filter(pl.col("Corridor") == REFERENCE_BEL_CORRIDOR)
    if ref.height == 0:
        logger.warning("  no reference battery-electric cell; hybrid chargers unpriced")
        return None
    ref_row = ref.unique(subset=["Sim_Key"]).to_dicts()[0]
    ref_safe = _safe(ref_row["Sim_Key"])
    if not (SIMS / f"{ref_safe}_sidecar.json").exists():
        logger.warning("  reference %s has no saved sims; hybrid chargers unpriced",
                       ref_row["Sim_Key"])
        return None
    ref_chargers = _bel_chargers(ref_safe)
    ref_j = terminal_charging_j(ref_row, ref_safe)
    if ref_chargers is None or not ref_j:
        return None
    ref_ports = {r["Node"]: float(r["Port_Count"]) for r in ref_chargers.iter_rows(named=True)}
    out = tc.scaled_chargers(ref_chargers, own, ref_j, ref_ports,
                             same_corridor=bool(same.height))
    logger.info("  hybrid chargers (installed-power equivalent ports): %s",
                {r["Node"]: round(r["Port_Count"], 3) for r in out.iter_rows(named=True)})
    return out if out.height else None


def rollout_cell(row: dict, safe: str, layouts: pl.DataFrame,
                 emissions_cache: Dict[tuple, pl.DataFrame]) -> Optional[pl.DataFrame]:
    """Every cost cell's discounted rollout for one simulation, or None if the
    sweep left no bundle for it."""
    bundle = SIMS / f"{safe}_sims.msgpack"
    if not bundle.exists():
        return None
    sidecar = json.loads((SIMS / f"{safe}_sidecar.json").read_text(encoding="utf-8"))
    pool = pl.read_csv(SIMS / f"{safe}_loco_pool.csv", infer_schema_length=None)
    plan = pl.read_csv(SIMS / f"{safe}_plan.csv", infer_schema_length=None)
    sims = alt.SpeedLimitTrainSimVec.from_file(str(bundle))

    years = [cat.METRICS_START_YEAR + k for k in range(cat.METRICS_ROLLOUT_YEARS)]
    # The week's grid draw (pantograph plus line loss, exact totals from the sidecar)
    # split across pricing nodes "<Cambium region>|<electricity price zone>" by the
    # substations that fed it (see `grid_draw_shares`).
    # Net of energy returned to a receptive wire when the sidecar records it.
    # The calculator prices what it is handed, so the return is credited at the
    # full price here; the post-processed table applies the export fraction.
    grid_j = float(sidecar.get("grid_draw_j",
                               float(sidecar["pantograph_j"]) + float(sidecar["line_loss_j"])))
    grid_j -= float(sidecar.get("grid_return_j", 0.0))
    shares = grid_draw_shares(sims, row["Corridor"], safe)
    grid_by_node = {n: grid_j * s for n, s in shares.items()} if grid_j else {}
    # Terminal (stationary) charging, priced and attributed at each terminal's
    # own "<region>|<zone>" node. Battery-electric and hybrid cells only.
    tz = {r["Node"]: grid_node(r["Grid_Region"], r["Electricity_Zone"])
          for r in terminal_zones(row["Corridor"]).iter_rows(named=True)}
    for term, j in terminal_charging_j(row, safe, sims).items():
        if term in tz and j > 0:
            grid_by_node[tz[term]] = grid_by_node.get(tz[term], 0.0) + j
    if not grid_by_node:  # keep at least one pricing node (zero energy)
        grid_by_node = {n: 0.0 for n in shares}
    total_j = sum(grid_by_node.values())
    shares = ({n: j / total_j for n, j in grid_by_node.items()} if total_j > 0
              else {n: 1.0 / len(grid_by_node) for n in grid_by_node})
    dz = diesel_zones(row["Corridor"])
    # Cambium GEA regions, NOT location names. sim_manager keys its emissions
    # factors by location Node ("Spokane", "Lethbridge") after mapping each
    # location's "Grid Emissions Region"; `interpolate_emissions_factors_cambium`
    # wants the region names themselves and silently omits anything it does not
    # recognise -- which would have left every electrified cell with no factors.
    regions = sorted(set(cambium_regions(row["Corridor"])) | {node_region(n) for n in shares})
    key = tuple(regions)
    if key not in emissions_cache:
        emissions_cache[key] = (
            mc.interpolate_emissions_factors_cambium(regions, years, cat.CAMBIUM_SCENARIO)
            if regions else None)
    emissions = (composite_emissions(emissions_cache[key], list(shares))
                 if emissions_cache[key] is not None else None)

    cost_fns = cost_functions(row)
    experiments = pl.read_csv(HERE / "experiments.csv", infer_schema_length=None)
    chargers = charger_facilities(row, safe, experiments, sims)
    frames = []
    # Cost cells: infra x energy for catenary; energy only for the rest.
    cells = (pl.read_csv(HERE / "experiments.csv", infer_schema_length=None)
             .filter(pl.col("Sim_Key") == row["Sim_Key"])
             .unique(subset=["Infra_Level", "Energy_Level"]))
    for cell in cells.to_dicts():
        started = time.perf_counter()
        infra_df = infrastructure_costs(cell, layouts)
        scenarios = [
            year_scenario(y, sims, price_pool(pool, cost_fns, y), plan,
                          grid_by_node, cell["Energy_Level"], emissions, infra_df, dz,
                          chargers)
            for y in years
        ]
        diesel_start, elec_start = energy_prices(cat.METRICS_START_YEAR, cell["Energy_Level"],
                                                 list(shares), dz)
        values = mc.main(scenarios, annual_metrics=ANNUAL_METRICS,
                         calculate_multiyear_metrics=True)
        provenance = pl.DataFrame([{
            "Sim_Key": row["Sim_Key"], "Corridor": row["Corridor"],
            "Powertrain": row["Powertrain"], "Label": row["Label"],
            "Battery_MWh": row.get("Battery_MWh"),
            "Architecture": row.get("Architecture"),
            "Layout_Fingerprint": row.get("Layout_Fingerprint"),
            "Infra_Level": cell["Infra_Level"],
            "Infra_Multiplier": float(cell["Infra_Multiplier"]),
            "Energy_Level": cell["Energy_Level"],
            "Rollout_Years": cat.METRICS_ROLLOUT_YEARS,
            "Start_Year": cat.METRICS_START_YEAR,
            "Discount_Rate": cat.DISCOUNT_RATE,
            "Loco_Unit_Cost_USD_Start": float(cost_fns[list(cost_fns)[0]](cat.METRICS_START_YEAR)),
            "Loco_Cost_Basis": ("diesel unit + ATB pack (assumption)"
                                if row["Powertrain"] == "hybrid" else "altrios defaults"),
            "Grid_Regions": ";".join(regions),
            # Where the energy was priced: grid draw share by "<region>|<zone>" node,
            # the terminal diesel zones, and the resulting start-year prices.
            "Elec_Price_Nodes": ";".join(f"{n}:{s:.3f}" for n, s in shares.items()),
            "Diesel_Price_Zones": ";".join(dz),
            "Diesel_Price_USD_per_gal_Start": diesel_start,
            "Elec_Price_cents_per_kWh_Start": sum(elec_start[n] * s for n, s in shares.items()),
            "n_trains": int(sidecar.get("n_trains") or 0),
        }])
        frames.append(provenance.join(values, how="cross"))
        logger.info("  %s | infra=%s energy=%s: %d rows in %.0f s",
                    row["Sim_Key"], cell["Infra_Level"], cell["Energy_Level"],
                    values.height, time.perf_counter() - started)
    return pl.concat(frames, how="diagonal_relaxed")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sim-key", help="run one simulation only")
    parser.add_argument("--out", type=Path, default=OUT / "rollout_costs.csv")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    experiments = pl.read_csv(HERE / "experiments.csv", infer_schema_length=None)
    sims = experiments.unique(subset=["Sim_Key"], keep="first")
    if args.sim_key:
        sims = sims.filter(pl.col("Sim_Key") == args.sim_key)
    layouts = pl.read_csv(OUT / "layouts" / "layouts.csv", infer_schema_length=None)

    emissions_cache: Dict[tuple, pl.DataFrame] = {}
    frames: List[pl.DataFrame] = []
    missing: List[str] = []
    for row in sims.to_dicts():
        safe = row["Sim_Key"].replace("|", "__").replace(".", "p")
        logger.info("=== %s ===", row["Sim_Key"])
        frame = rollout_cell(row, safe, layouts, emissions_cache)
        if frame is None:
            missing.append(row["Sim_Key"])
            continue
        frames.append(frame)
        # Written after every simulation so an interruption keeps what finished.
        pl.concat(frames, how="diagonal_relaxed").write_csv(args.out)

    if frames:
        logger.info("wrote %s (%d simulations)", args.out, len(frames))
    for key in missing:
        logger.warning("no saved sims for %s -- re-run the sweep for it", key)
    return 0 if frames else 1


if __name__ == "__main__":
    raise SystemExit(main())
