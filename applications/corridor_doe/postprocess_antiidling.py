"""
Expand simulation results into the full design: anti-idling and energy prices.

Anti-idling is the one factor in this study that is applied entirely after the
fact. That is a deliberate choice, not a shortcut: stopping the prime mover
during a hold changes fuel burned at rest, and changes nothing about how the
train moves, which links it occupies, or when it meets another train. So the
21 simulations stand, and each anti-idling level is arithmetic on their output.

The arithmetic acts on **individual stops**, not on the corridor dwell total.
A shutdown threshold is a per-stop rule -- a system set to act after 15 minutes
leaves a 40-second meet completely alone -- so applying it to an aggregate would
assume a stop-duration distribution instead of using the real one.
``run_sweep.stop_events`` writes one row per stop for exactly this reason.

For each cell, ``eligible`` is the set of stops lasting at least
``Shutdown_Threshold_Seconds`` -- a system set to act after 15 minutes leaves a
40-second meet running -- and ``Idle_Source`` says what burns during them:

    baseline   fuel stays as simulated; nothing changes
    shutdown   eligible stops burn nothing, plus a restart penalty each
    apu        eligible stops burn APU fuel instead of main-engine idle fuel
    battery    eligible stops burn nothing; the battery carries the hotel load

``apu`` uses ``APU_Fuel_Consumption`` from ``metric_calculator``, which
integrates each locomotive's real auxiliary-power history through an APU fuel
curve (0.063 gal/hr per aux kW, plus 0.4115). That metric covers *all* stopped
time, so it is prorated here by the eligible share of dwell seconds -- an
approximation, and the only one in this step worth flagging.

Energy prices are harmonized across catenary and non-catenary cells: both draw
from the same EIA tables ``metric_calculator`` already loads, scaled by the
cell's ``Diesel_Multiplier`` and ``Electricity_Multiplier``. Pricing a diesel
cell from one source and a catenary cell from another would make the central
comparison of the study an artifact of bookkeeping.

Usage:
    python postprocess_antiidling.py
    python postprocess_antiidling.py --out out/results.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import polars as pl

from altrios import defaults, metric_calculator

HERE = Path(__file__).parent
OUT = HERE / "out"

logger = logging.getLogger(__name__)

# Diesel lower heating value and density, from the same GREET table
# metric_calculator uses to convert fuel energy to gallons, so an adjusted
# energy converts back to gallons on identical assumptions.
_DIESEL = metric_calculator.emissions_factors_greet.filter(
    pl.col("Name") == pl.lit("ultra low sulfur diesel")
)
LHV_KJ_PER_KG = float(_DIESEL["LowHeatingValue"][0]) * 1e3
RHO_G_PER_GALLON = float(_DIESEL["Density"][0])

# The two AEO tables carry EIA's own units, and they differ: the liquid fuel
# table is dollars per gallon, the electricity table is CENTS per kilowatt-hour
# (2025 dollars in both cases -- the same real basis, which is what makes the
# harmonization meaningful). metric_calculator never trips over this because it
# uses both tables only as ratios to a reference year, where units cancel. Used
# as absolute prices here, electricity has to be converted.
ELEC_CENTS_PER_KWH_TO_USD = 0.01

# GREET ultra-low-sulfur diesel, gCO2-eq per MJ. The table carries one value,
# for 2020, which metric_calculator applies to every year.
DIESEL_G_CO2E_PER_MJ = float(
    metric_calculator.emissions_factors_greet.filter(
        (pl.col("Name") == "ultra low sulfur diesel")
        & (pl.col("Units") == "gCO2 eq/MJ")
        & (pl.col("Year") == 2020)
    )["Value"][0]
)


def _price_at(table: pl.DataFrame, year: int) -> float:
    """Nearest-year price, clamped to the table's range."""
    years = table["Year"]
    y = min(max(int(year), int(years.min())), int(years.max()))
    return float(table.filter(pl.col("Year") == y)["Price"][0])


# "All" counts every second a train stands still; "En_Route" excludes the hold
# at the origin before dispatch and the stand at the destination.
DWELL_SCOPES = ("All", "En_Route")


def diesel_mj_to_gallons(mj: float) -> float:
    return (mj * 1e6 / 1e3 / LHV_KJ_PER_KG) * 1e3 / RHO_G_PER_GALLON


def wide_metrics(metrics: pl.DataFrame, dwell_scope: str = "All") -> pl.DataFrame:
    """One row per Sim_Key, one column per (metric, units).

    `dwell_scope` picks which version of the dwell metrics to use. The dwell
    family is reported twice by `metric_calculator`: "All" counts every second
    a train stands still, including the hold at its origin before the
    dispatcher releases it and the time it stands at its destination;
    "En_Route" counts only stops between the two terminals. Metrics that carry
    no en-route variant (energy, freight, GHG) fall back to "All" either way.
    """
    subset = metrics
    if "Subset" in subset.columns:
        scoped = subset.filter(pl.col("Subset") == pl.lit(dwell_scope))
        scoped_names = set(scoped["Metric"].to_list())
        subset = pl.concat([
            scoped,
            subset.filter(
                (pl.col("Subset") == pl.lit("All"))
                & ~pl.col("Metric").is_in(list(scoped_names))
            ),
        ], how="diagonal_relaxed")
    return (
        subset.with_columns(
            pl.concat_str([pl.col("Metric"), pl.col("Units")], separator="__")
            .str.replace_all(r"[ /]", "_")
            .alias("key")
        )
        .select("Sim_Key", "key", "Value")
        .pivot(on="key", index="Sim_Key", values="Value", aggregate_function="first")
    )


def stop_summary(
    stops: pl.DataFrame, thresholds: Sequence[float], dwell_scope: str = "All"
) -> pl.DataFrame:
    """Eligible stop count and fuel per (Sim_Key, threshold).

    Computed once per distinct threshold rather than per cell, since many cells
    share one. Under the "En_Route" scope, stops at the origin and destination
    terminals are excluded, matching the dwell metrics of the same scope --
    `run_sweep.stop_events` splits its runs at the terminal boundary so the two
    agree exactly rather than approximately.
    """
    if dwell_scope != "All" and "terminal" in stops.columns:
        stops = stops.filter(~pl.col("terminal"))
    frames = []
    for threshold in sorted(set(float(t) for t in thresholds)):
        frames.append(
            stops.filter(pl.col("duration_seconds") >= threshold)
            .group_by("Sim_Key")
            .agg(
                pl.len().alias("eligible_stops"),
                pl.col("fuel_MJ").sum().alias("eligible_fuel_MJ"),
                pl.col("duration_seconds").sum().alias("eligible_dwell_seconds"),
            )
            .with_columns(pl.lit(threshold).alias("Shutdown_Threshold_Seconds"))
        )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def expand(
    experiments: pl.DataFrame,
    metrics: pl.DataFrame,
    stops: pl.DataFrame,
    year: int,
    failures: Optional[pl.DataFrame] = None,
    dwell_scope: str = "All",
) -> pl.DataFrame:
    wide = wide_metrics(metrics, dwell_scope)
    # LEFT, not inner: a cell whose simulation failed stays in the table with a
    # reason. Dropping it makes an infeasible configuration -- a battery-electric
    # locomotive on a corridor the optimizer declined to electrify, which cannot
    # move at all -- indistinguishable from one nobody has run yet.
    cells = experiments.join(wide, on="Sim_Key", how="left")
    simulated = set(wide["Sim_Key"].to_list())
    if not simulated:
        raise ValueError(
            "no experiment cell matched a simulated Sim_Key -- run run_sweep.py first"
        )
    if failures is not None and failures.height:
        cells = cells.join(
            failures.select(
                "Sim_Key",
                pl.concat_str([
                    pl.lit("FAILED ("), pl.col("return_code").cast(pl.Utf8),
                    pl.lit("): "), pl.col("detail").fill_null(""),
                ]).alias("_failure"),
            ),
            on="Sim_Key", how="left",
        )
    else:
        cells = cells.with_columns(pl.lit(None, dtype=pl.Utf8).alias("_failure"))

    # The threshold is written as an integer in the CSV but computed as a float
    # in the summary, so cast before joining on it.
    cells = cells.with_columns(
        pl.col("Shutdown_Threshold_Seconds").cast(pl.Float64)
    )
    summary = stop_summary(
        stops, cells["Shutdown_Threshold_Seconds"].to_list(), dwell_scope
    )
    cells = cells.join(
        summary,
        on=["Sim_Key", "Shutdown_Threshold_Seconds"],
        how="left",
    ).with_columns(
        pl.col("eligible_stops").fill_null(0),
        pl.col("eligible_fuel_MJ").fill_null(0.0),
        pl.col("eligible_dwell_seconds").fill_null(0.0),
    )

    diesel_col = "Diesel_Usage__MJ"
    stationary_col = "Electricity_Usage__MJ"
    catenary_col = "Catenary_Grid_Draw__MJ"
    for col in (diesel_col, stationary_col, catenary_col,
                "Grid_Emissions_Factor__kg_CO2-eq_per_MWh"):
        if col not in cells.columns:
            cells = cells.with_columns(pl.lit(0.0).alias(col))
    # Total grid electricity = stationary charging + everything drawn from the
    # wire (delivered + resistive loss). The two are disjoint: the catenary
    # figure comes from the locomotives' catenary interfaces, the stationary one
    # from battery net energy, which is ~0 whenever the wire is doing the work.
    elec_col = "Electricity_Total__MJ"
    cells = cells.with_columns(
        (pl.col(stationary_col).fill_null(0.0) + pl.col(catenary_col).fill_null(0.0))
        .alias(elec_col)
    )

    diesel_price = _price_at(metric_calculator.liquid_fuel_prices_eia, year)
    elec_price = (
        _price_at(metric_calculator.electricity_prices_eia, year)
        * ELEC_CENTS_PER_KWH_TO_USD
    )
    logger.info(
        "harmonized %d prices: diesel $%.3f/gal, electricity $%.4f/kWh "
        "(EIA tables, before per-cell multipliers)",
        year, diesel_price, elec_price,
    )

    dwell_col = "Dwell_Time__seconds"
    apu_col = "APU_Fuel_Consumption__MJ"
    for col in (dwell_col, apu_col):
        if col not in cells.columns:
            cells = cells.with_columns(pl.lit(0.0).alias(col))

    # Share of all stopped time that the anti-idling system actually acts on.
    eligible_share = (
        pl.when(pl.col(dwell_col) > 0)
        .then(pl.col("eligible_dwell_seconds") / pl.col(dwell_col))
        .otherwise(0.0)
        .clip(upper_bound=1.0)
    )

    return (
        cells.with_columns(eligible_share.alias("Eligible_Dwell_Share"))
        .with_columns(
            # What the eligible stops burn under this level, replacing the
            # main-engine idle fuel they burned in the simulation.
            pl.when(pl.col("Idle_Source") == pl.lit("baseline"))
            .then(pl.col("eligible_fuel_MJ"))
            .when(pl.col("Idle_Source") == pl.lit("apu"))
            .then(pl.col(apu_col) * pl.col("Eligible_Dwell_Share"))
            .otherwise(pl.lit(0.0))
            .alias("Idle_Fuel_Replacement_MJ")
        )
        .with_columns(
            (pl.col("eligible_fuel_MJ") - pl.col("Idle_Fuel_Replacement_MJ"))
            .clip(lower_bound=0.0)
            .alias("Idle_Fuel_Saved_MJ"),
            pl.when(pl.col("Idle_Source") == pl.lit("shutdown"))
            .then(pl.col("eligible_stops") * pl.col("Restart_Penalty_MJ_Per_Event"))
            .otherwise(pl.lit(0.0))
            .alias("Restart_Fuel_MJ"),
        )
        .with_columns(
            (
                pl.col(diesel_col)
                - pl.col("Idle_Fuel_Saved_MJ")
                + pl.col("Restart_Fuel_MJ")
            )
            .clip(lower_bound=0.0)
            .alias("Diesel_MJ_Adjusted")
        )
        .with_columns(
            (
                pl.col("Diesel_MJ_Adjusted") * 1e6 / 1e3 / LHV_KJ_PER_KG
                * 1e3 / RHO_G_PER_GALLON
            ).alias("Diesel_Gallons_Adjusted")
        )
        .with_columns(
            (
                pl.col("Diesel_Gallons_Adjusted")
                * pl.lit(diesel_price)
                * pl.col("Diesel_Multiplier")
            ).alias("Cost_Diesel_USD"),
            (
                pl.col(elec_col)
                / 3.6  # MJ -> kWh
                * pl.lit(elec_price)
                * pl.col("Electricity_Multiplier")
            ).alias("Cost_Electricity_USD"),
        )
        .with_columns(
            # GHG, computed here rather than by `calculate_ghg`, which cannot
            # run when charging was not explicitly modelled (see run_sweep).
            # Diesel uses the GREET ultra-low-sulfur-diesel factor; electricity
            # uses the grid factor the simulation itself was given.
            (
                pl.col("Diesel_MJ_Adjusted")
                * pl.lit(DIESEL_G_CO2E_PER_MJ)
                / 1e6
            ).alias("GHG_Diesel_tonne"),
            (
                pl.col(elec_col) / 3600.0  # MJ -> MWh
                * pl.col("Grid_Emissions_Factor__kg_CO2-eq_per_MWh").fill_null(0.0)
                / 1e3
            ).alias("GHG_Electricity_tonne"),
        )
        .with_columns(
            # A catenary cell is only valid once it records energy drawn from
            # the wire. Two separate defects used to make that impossible, and
            # this flag stays as the guard that both remain fixed: the network
            # was energized without voltage (so no power flowed), and the
            # pantograph energy was computed but never reported. A cell that
            # still shows zero catenary draw has regressed on one of them, and
            # its energy, GHG and cost columns cannot be used.
            pl.when(~pl.col("Sim_Key").is_in(list(simulated)))
            .then(pl.col("_failure").fill_null(pl.lit("NOT RUN")))
            .when(
                # An unelectrified cell legitimately draws nothing from a wire
                # that was never built, so only a cell WITH a buildout and no
                # catenary energy indicates the accounting has regressed.
                pl.col("Powertrain").str.starts_with("catenary")
                & (pl.col(catenary_col).fill_null(0.0) <= 0.0)
                & (pl.col("Layout_Fingerprint") != pl.lit("nobuild"))
            )
            .then(pl.lit("INVALID: catenary energy not accounted"))
            .otherwise(pl.lit("ok"))
            .alias("Result_Status"),
            (pl.col("GHG_Diesel_tonne") + pl.col("GHG_Electricity_tonne"))
            .alias("GHG_Energy_tonne"),
            (pl.col("Cost_Diesel_USD") + pl.col("Cost_Electricity_USD"))
            .alias("Cost_Energy_USD"),
            (
                pl.when(pl.col(diesel_col) > 0)
                .then(pl.col("Idle_Fuel_Saved_MJ") / pl.col(diesel_col))
                .otherwise(0.0)
            ).alias("Idle_Fuel_Saved_Fraction_Of_Total"),
        )
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--experiments", type=Path, default=HERE / "experiments.csv")
    parser.add_argument("--metrics", type=Path, default=OUT / "sim_metrics.csv")
    parser.add_argument("--stops", type=Path, default=OUT / "stop_events.csv")
    parser.add_argument("--out", type=Path, default=OUT / "results.csv")
    parser.add_argument(
        "--year", type=int, default=defaults.BASE_ANALYSIS_YEAR,
        help="price year for the harmonized energy costs",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    for path in (args.experiments, args.metrics, args.stops):
        if not path.exists():
            parser.error(f"{path} is missing -- run build_experiments.py and run_sweep.py")

    failures_path = OUT / "failures.csv"
    experiments = pl.read_csv(args.experiments)
    metrics = pl.read_csv(args.metrics)
    stops = pl.read_csv(args.stops)
    failures = pl.read_csv(failures_path) if failures_path.exists() else None

    # Both readings of "how long does a train sit still", stacked and labelled.
    # Which one an anti-idling policy should act on is a judgement about the
    # policy, not about the simulation: a locomotive parked at a terminal is a
    # candidate for shutdown under some operating practices and out of scope
    # under others, and terminal holds dominate the totals on a congested
    # corridor. Reporting one would bury that choice.
    blocks = []
    for scope in DWELL_SCOPES:
        block = expand(
            experiments, metrics, stops, args.year, failures, dwell_scope=scope
        ).drop("_failure")
        blocks.append(block.with_columns(pl.lit(scope).alias("Dwell_Scope")))
    results = pl.concat(blocks, how="diagonal_relaxed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.write_csv(args.out)
    logger.info("wrote %s (%d cells)", args.out, results.height)

    for status, n in (
        results.group_by("Result_Status").len().sort("len", descending=True)
    ).iter_rows():
        (logger.info if status == "ok" else logger.warning)(
            "%5d of %d cells: %s", n, results.height, status
        )

    saved = results.group_by("Anti_Idling").agg(
        pl.col("Idle_Fuel_Saved_Fraction_Of_Total").mean().alias("mean_fuel_saved"),
        pl.len().alias("cells"),
    ).sort("Anti_Idling")
    logger.info("\n%s", saved.to_pandas().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
