"""ONE-OFF: multi-year discounted costs for the EXISTING DoE runs, without sims.

Delete this once the sweep has been re-run with sim persistence; the permanent
path is `rollout_costs.py`, which hands the metrics calculator each cell's saved
simulation as per-year ``ScenarioInfo``s -- the catenary deep-dive's method.

The existing runs kept no sims, so that path cannot reach them. But the
calculator's three rollout functions (``calculate_rollout_investments``,
``calculate_rollout_total_costs``, ``calculate_rollout_lcotkm``) never touch
sims: they consume the table of ANNUAL metric rows that the per-year step emits.
For thirty identical operating years every one of those rows is a constant that
the per-cell outputs already hold -- weekly diesel and grid draw, freight moved,
the locomotive pool by type -- annualized exactly as the calculator would and
priced exactly as `rollout_costs.py` prices them (it imports that module's cost,
infrastructure and region functions, so the two paths cannot drift).

Every annual row is constructed to be what the calculator's own annual functions
would have produced from a simulation of the same week:

- Diesel gallons via the same GREET lower heating value and density.
- Electricity in kWh per Cambium region, priced in cents/kWh like
  ``calculate_energy_cost``; diesel in $/gal.
- Fleet counts, unit costs and lifespans per locomotive type; infrastructure
  counts, unit costs and lifespans per asset type.
- Freight in million tonne-km and million tonne-mi, and car-miles.

Anti-idling is carried the way the sweep carries it: post-processed, not
simulated. For every (measure, dwell scope) the sweep recorded for a cell in
``results.csv``, the same annual rows are built again with the cell's diesel
replaced by its ``Diesel_MJ_Adjusted``; fleet, infrastructure and freight are
unchanged (no APU or hotel-load hardware is priced). The baseline rows carry
``Anti_Idling`` none / not_applicable and ``Dwell_Scope`` All, and are what the
lifecycle cells read.

``--check-against`` compares one cell to a sim-based output of the same cell and
reports the largest relative difference -- the alignment test this exists to
pass. Writes ONLY ``out/rollout_costs_sim_free.csv``; nothing else is touched.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import polars as pl

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import rollout_costs as rc  # noqa: E402  -- SAME pricing, infrastructure, regions
import catenary_policy  # noqa: E402
from rollout_costs import alt, cat, defaults, mc, utilities  # noqa: E402

logger = logging.getLogger("rollout_costs_sim_free")
OUT = HERE / "out"
SIMS = OUT / "sims"
SIMULATION_DAYS = 7.0
ANNUALIZER = 365.25 / SIMULATION_DAYS  # the calculator's own factor

# GREET ultra-low-sulfur diesel, as `calculate_diesel_use` reads it.
_GREET = mc.emissions_factors_greet.filter(pl.col("Name") == "ultra low sulfur diesel")
LHV_KJ_PER_KG = float(_GREET["LowHeatingValue"][0]) * 1e3
RHO_G_PER_GALLON = float(_GREET["Density"][0])
# GREET 2020 well-to-wheel CO2-eq per MJ of diesel, as `calculate_ghg` reads it
# (the table has one year; the calculator applies it to every rollout year).
DIESEL_G_CO2E_PER_MJ = float(_GREET.filter(
    (pl.col("Units") == "gCO2 eq/MJ") & (pl.col("Year") == 2020))["Value"][0])
_EF_CACHE: Dict[tuple, Dict[tuple, float]] = {}


def grid_factor(nodes, year: int) -> Dict[str, float]:
    """Cambium CO2-eq kg/MWh per pricing node ("<region>|<zone>") for one rollout
    year, from the same interpolation the workflow stage hands the calculator."""
    regions = sorted({rc.node_region(n) for n in nodes})
    key = tuple(regions)
    if key not in _EF_CACHE:
        years = [cat.METRICS_START_YEAR + k for k in range(cat.METRICS_ROLLOUT_YEARS)]
        ef = mc.interpolate_emissions_factors_cambium(list(regions), years, cat.CAMBIUM_SCENARIO)
        _EF_CACHE[key] = {(r["Node"], int(r["Year"])): float(r["CO2eq_kg_per_MWh"])
                          for r in ef.to_dicts()}
    table = _EF_CACHE[key]
    return {n: table[(rc.node_region(n), year)] for n in nodes
            if (rc.node_region(n), year) in table}


def _row(metric, units, value, subset="All", year="All"):
    return {"Metric": metric, "Units": units, "Value": float(value),
            "Subset": subset, "Year": year}


def weekly_inputs(safe: str) -> Optional[dict]:
    """What the sweep recorded for one cell, straight from its per-cell files."""
    metrics_path = SIMS / f"{safe}_metrics.csv"
    plan_path = SIMS / f"{safe}_plan.csv"
    if not (metrics_path.exists() and plan_path.exists()):
        return None
    m = pl.read_csv(metrics_path, infer_schema_length=None).filter(
        pl.col("Subset") == "All")

    def val(metric, units=None):
        q = m.filter(pl.col("Metric") == metric)
        if units:
            q = q.filter(pl.col("Units") == units)
        return float(q["Value"][0]) if q.height and q["Value"][0] is not None else 0.0

    plan = pl.read_csv(plan_path, infer_schema_length=None)
    fleet = (plan.group_by("Locomotive_Type")
             .agg(pl.col("Locomotive_ID").n_unique().alias("n"))
             .to_dicts())
    return {
        "diesel_mj": val("Diesel_Usage", "MJ"),
        "grid_mj": val("Catenary_Grid_Draw", "MJ"),
        # Returned to a receptive wire (0 on runs that predate it). Energy and
        # emissions net it in full; cost credits it at the export fraction.
        "grid_return_mj": val("Catenary_Grid_Return", "MJ"),
        "mtkm": val("Freight_Moved", "million tonne-km"),
        "car_miles": val("Freight_Moved", "car-miles"),
        "fleet": {f["Locomotive_Type"]: int(f["n"]) for f in fleet},
        "n_trains": int(plan["Train_ID"].n_unique()),
    }


def annual_rows(year: int, w: dict, cost_fns, infra_df, nodes: Dict[str, float],
                diesel_zone_list, energy_level) -> List[dict]:
    """The rows the calculator's annual step would emit for this year."""
    rows = []
    # --- fleet: `calculate_locomotive_counts` with count_unused_locomotives ----
    total_units = sum(w["fleet"].values())
    for loco_type, n in w["fleet"].items():
        rows += [
            _row("Count_Locomotives", "assets", n, loco_type, year),
            _row("Unit_Cost", "USD", cost_fns[loco_type](year), loco_type, year),
            _row("Lifespan", "years", defaults.LOCO_LIFESPAN, loco_type, year),
            _row("Pct_Locomotives", "fraction (0-1)", n / total_units, loco_type, year),
        ]
    rows.append(_row("Count_Locomotives", "assets", total_units, "All", year))
    # --- infrastructure: `calculate_infrastructure_counts` ----------------------
    if infra_df is not None:
        agg = infra_df.group_by("Infrastructure_Type").agg(
            pl.col("Cost_USD").mean().alias("Unit_Cost"),
            pl.len().alias("Count_Infrastructure"),
            pl.col("Lifespan_Years").mean().alias("Lifespan"))
        for r in agg.to_dicts():
            t = r["Infrastructure_Type"]
            rows += [
                _row("Count_Infrastructure", "assets", r["Count_Infrastructure"], t, year),
                _row("Unit_Cost", "USD", r["Unit_Cost"], t, year),
                _row("Lifespan", "years", r["Lifespan"], t, year),
            ]
    # --- energy and its cost: `calculate_energy_cost` -------------------------
    # Zone prices: diesel $/gal (mean of terminal zones), electricity cents/kWh per
    # "<region>|<zone>" node, exactly as `rollout_costs.energy_prices` builds them.
    diesel_price, elec_cents = rc.energy_prices(year, energy_level, list(nodes), diesel_zone_list)
    diesel_j = w["diesel_mj"] * 1e6 * ANNUALIZER
    gallons = diesel_j / 1e3 / LHV_KJ_PER_KG * 1e3 / RHO_G_PER_GALLON
    # Net of energy returned to a receptive wire for usage and emissions; for
    # cost the return is credited at the policy's export fraction.
    grid_net_mj = w["grid_mj"] - w.get("grid_return_mj", 0.0)
    grid_priced_mj = (w["grid_mj"]
                      - catenary_policy.EXPORT_CREDIT_FRACTION * w.get("grid_return_mj", 0.0))
    grid_kwh = grid_net_mj * utilities.KWH_PER_MJ * ANNUALIZER
    priced_kwh = grid_priced_mj * utilities.KWH_PER_MJ * ANNUALIZER
    elec_cost = 0.0
    for node, share in nodes.items():
        kwh = grid_kwh * share
        rows.append(_row("Electricity_Usage", "kWh", kwh, node, year))
        rows.append(_row("Cost_Electricity", "USD", priced_kwh * share * elec_cents[node] / 100.0, node, year))
        elec_cost += priced_kwh * share * elec_cents[node] / 100.0
    # --- emissions: `calculate_ghg` --------------------------------------------
    diesel_ghg = w["diesel_mj"] * ANNUALIZER * DIESEL_G_CO2E_PER_MJ / utilities.G_PER_TONNE
    factors = grid_factor(list(nodes), year) if grid_kwh > 0 else {}
    elec_ghg = sum((grid_kwh * share) / 1000.0 * factors[n] / 1000.0
                   for n, share in nodes.items() if n in factors) if grid_kwh > 0 else 0.0
    rows += [
        _row("GHG_Diesel", "tonne CO2-eq", diesel_ghg, "All", year),
        _row("GHG_Electricity", "tonne CO2-eq", elec_ghg, "All", year),
        _row("GHG_Energy", "tonne CO2-eq", diesel_ghg + elec_ghg, "All", year),
        _row("Electricity_Usage", "kWh", grid_kwh, "All", year),
        _row("Cost_Electricity", "USD", elec_cost, "All", year),
        _row("Diesel_Usage", "gallons", gallons, "All", year),
        _row("Cost_Diesel", "USD", gallons * diesel_price, "All", year),
        _row("Cost_Energy", "USD", gallons * diesel_price + elec_cost, "All", year),
        _row("Energy", "MJ", w["diesel_mj"] * ANNUALIZER, "Diesel", year),
        _row("Energy", "MJ", grid_net_mj * ANNUALIZER, "Electricity", year),
    ]
    # --- freight: `calculate_freight_moved` -----------------------------------
    mtkm = w["mtkm"] * ANNUALIZER
    rows += [
        _row("Freight_Moved", "million tonne-km", mtkm, "All", year),
        _row("Freight_Moved", "million tonne-mi", mtkm * utilities.MI_PER_KM, "All", year),
        _row("Freight_Moved", "car-miles", w["car_miles"] * ANNUALIZER, "All", year),
    ]
    return rows


BASELINE_IDLING = ("none", "not_applicable")
_RESULTS: Optional[pl.DataFrame] = None


def idling_variants(sim_key: str, infra_level: str, energy_level: str) -> List[dict]:
    """The (measure, scope, adjusted diesel MJ) triples the sweep post-processed
    for one cost cell of one simulation, read from results.csv. The baseline is
    identical under both dwell scopes, so it is kept once, under ``All``."""
    global _RESULTS
    if _RESULTS is None:
        _RESULTS = pl.read_csv(OUT / "results.csv", infer_schema_length=None)
    rows = (_RESULTS.filter(pl.col("Sim_Key") == sim_key,
                            pl.col("Infra_Level") == infra_level,
                            pl.col("Energy_Level") == energy_level,
                            pl.col("Result_Status") == "ok")
            .filter(~(pl.col("Anti_Idling").is_in(BASELINE_IDLING)
                      & (pl.col("Dwell_Scope") != "All")))
            .select("Anti_Idling", "Dwell_Scope", "Diesel_Usage__MJ", "Diesel_MJ_Adjusted")
            .unique(subset=["Anti_Idling", "Dwell_Scope"])
            .sort("Anti_Idling", "Dwell_Scope"))
    return rows.to_dicts()


def rollout_cell(row: dict, safe: str, layouts: pl.DataFrame) -> Optional[pl.DataFrame]:
    w = weekly_inputs(safe)
    if w is None:
        return None
    years = [cat.METRICS_START_YEAR + k for k in range(cat.METRICS_ROLLOUT_YEARS)]
    # Grid-draw split by pricing node, from the sim-based stage's cache when it has
    # run (out/sims/<safe>_grid_nodes.json); otherwise the terminal-zone fallback.
    nodes = rc.grid_draw_shares(None, row["Corridor"], safe)
    dz = rc.diesel_zones(row["Corridor"])
    cost_fns = rc.cost_functions(row)
    cells = (pl.read_csv(HERE / "experiments.csv", infer_schema_length=None)
             .filter(pl.col("Sim_Key") == row["Sim_Key"])
             .unique(subset=["Infra_Level", "Energy_Level"]))
    frames = []
    for cell in cells.to_dicts():
        infra_df = rc.infrastructure_costs(cell, layouts)
        variants = idling_variants(row["Sim_Key"], cell["Infra_Level"], cell["Energy_Level"])
        if not variants:
            logger.warning("  %s | infra=%s energy=%s: no results.csv rows, baseline only",
                           row["Sim_Key"], cell["Infra_Level"], cell["Energy_Level"])
            variants = [{"Anti_Idling": "none", "Dwell_Scope": "All",
                         "Diesel_Usage__MJ": w["diesel_mj"], "Diesel_MJ_Adjusted": w["diesel_mj"]}]
        for v in variants:
            frames.append(_rollout_variant(row, cell, w, v, cost_fns, infra_df, nodes, dz, years))
    return pl.concat(frames, how="diagonal_relaxed")


def _rollout_variant(row, cell, w, v, cost_fns, infra_df, nodes, dz, years) -> pl.DataFrame:
    """One (cost cell, anti-idling variant): the calculator's DCF over annual rows
    built with the variant's adjusted diesel. The sweep's unadjusted diesel must
    match the per-cell file the baseline is built from, or the variants would be
    scaling a different number than they were derived from."""
    base = float(v["Diesel_Usage__MJ"] or 0.0)
    if abs(base - w["diesel_mj"]) > 1e-6 * max(abs(base), 1.0):
        raise ValueError(f"{row['Sim_Key']}: results.csv diesel {base} != per-cell {w['diesel_mj']}")
    adjusted = float(v["Diesel_MJ_Adjusted"]) if v["Diesel_MJ_Adjusted"] is not None else base
    w = dict(w, diesel_mj=adjusted)
    values = pl.DataFrame(
        [r for y in years
         for r in annual_rows(y, w, cost_fns, infra_df, nodes, dz, cell["Energy_Level"])],
        # Int32: the calculator stamps scenario years as Int32 and its
        # purchase rows inherit that, so a diagonal concat needs a match.
        schema={"Metric": pl.Utf8, "Units": pl.Utf8, "Value": pl.Float64,
                "Subset": pl.Utf8, "Year": pl.Int32},
    )
    # The calculator's own DCF, exactly as `metric_calculator.main` runs it.
    values = mc.calculate_rollout_investments(values)
    values = mc.calculate_rollout_total_costs(values)
    values = mc.calculate_rollout_lcotkm(values)
    values = (values.filter(pl.col("Value").is_not_null()).unique()
              .sort(["Metric", "Units", "Year", "Subset"],
                    descending=[False, False, False, True]))
    provenance = pl.DataFrame([{
        "Sim_Key": row["Sim_Key"], "Corridor": row["Corridor"],
        "Powertrain": row["Powertrain"], "Label": row["Label"],
        "Battery_MWh": row.get("Battery_MWh"),
        "Architecture": row.get("Architecture"),
        "Layout_Fingerprint": row.get("Layout_Fingerprint"),
        "Infra_Level": cell["Infra_Level"],
        "Infra_Multiplier": float(cell["Infra_Multiplier"]),
        "Energy_Level": cell["Energy_Level"],
        "Anti_Idling": v["Anti_Idling"],
        "Dwell_Scope": v["Dwell_Scope"],
        "Diesel_MJ_Weekly": adjusted,
        "Rollout_Years": cat.METRICS_ROLLOUT_YEARS,
        "Start_Year": cat.METRICS_START_YEAR,
        "Discount_Rate": cat.DISCOUNT_RATE,
        "Loco_Unit_Cost_USD_Start": float(cost_fns[list(cost_fns)[0]](cat.METRICS_START_YEAR)),
        "Loco_Cost_Basis": ("diesel unit + ATB pack (assumption)"
                            if row["Powertrain"] == "hybrid" else "altrios defaults"),
        "Grid_Regions": ";".join(sorted({rc.node_region(n) for n in nodes})),
        "Elec_Price_Nodes": ";".join(f"{n}:{s:.3f}" for n, s in nodes.items()),
        "Diesel_Price_Zones": ";".join(dz),
        "n_trains": w["n_trains"],
        "Source": "sim-free one-off (rollout_costs_sim_free.py)",
    }])
    logger.info("  %s | infra=%s energy=%s | %s/%s: %d rows", row["Sim_Key"],
                cell["Infra_Level"], cell["Energy_Level"], v["Anti_Idling"],
                v["Dwell_Scope"], values.height)
    return provenance.join(values, how="cross")


def check_against(mine: pl.DataFrame, reference_path: Path) -> None:
    """Compare per-year discounted totals and LCOTKM with a sim-based output."""
    ref = pl.read_csv(reference_path, infer_schema_length=None)
    keys = ["Sim_Key", "Infra_Level", "Energy_Level", "Metric", "Units", "Subset", "Year"]
    mine = mine.filter(pl.col("Anti_Idling").is_in(BASELINE_IDLING), pl.col("Dwell_Scope") == "All")
    a = mine.with_columns(pl.col("Year").cast(pl.Utf8)).select(keys + ["Value"])
    b = ref.with_columns(pl.col("Year").cast(pl.Utf8)).select(keys + [pl.col("Value").alias("Ref")])
    j = a.join(b, on=keys, how="inner").filter(
        pl.col("Metric").is_in(["Cost_Total_Discounted", "Cost_Energy", "Cost_Purchases",
                                "Cost_Retired_Assets", "LCOTKM", "Freight_Moved",
                                "Diesel_Usage", "Count_Locomotives",
                                "GHG_Energy", "GHG_Diesel", "GHG_Electricity"]))
    j = j.with_columns(
        ((pl.col("Value") - pl.col("Ref")).abs()
         / pl.max_horizontal(pl.col("Ref").abs(), pl.lit(1.0))).alias("rel"))
    print(f"alignment vs {reference_path.name}: {j.height} shared rows, "
          f"max relative difference {j['rel'].max():.2e}")
    worst = j.sort("rel", descending=True).head(5)
    print(worst.select(["Metric", "Subset", "Year", "Value", "Ref", "rel"]).to_pandas().to_string(index=False))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sim-key", help="run one simulation only")
    parser.add_argument("--out", type=Path, default=OUT / "rollout_costs_sim_free.csv")
    parser.add_argument("--check-against", type=Path,
                        help="a sim-based rollout CSV to compare against")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sims = pl.read_csv(HERE / "experiments.csv", infer_schema_length=None).unique(
        subset=["Sim_Key"], keep="first")
    if args.sim_key:
        sims = sims.filter(pl.col("Sim_Key") == args.sim_key)
    layouts = pl.read_csv(OUT / "layouts" / "layouts.csv", infer_schema_length=None)

    frames, missing = [], []
    for row in sims.to_dicts():
        safe = row["Sim_Key"].replace("|", "__").replace(".", "p")
        logger.info("=== %s ===", row["Sim_Key"])
        frame = rollout_cell(row, safe, layouts)
        (frames if frame is not None else missing).append(frame if frame is not None else row["Sim_Key"])
    if not frames:
        logger.error("no cells had per-cell outputs")
        return 1
    result = pl.concat(frames, how="diagonal_relaxed")
    result.write_csv(args.out)
    logger.info("wrote %s (%d simulations)", args.out, len(frames))
    for key in missing:
        logger.warning("no per-cell outputs for %s", key)
    if args.check_against:
        check_against(result, args.check_against)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
