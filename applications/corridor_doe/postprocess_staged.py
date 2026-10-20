"""Staged-adoption variant of every rollout cell, as a sensitivity.

The headline rollout converts a corridor instantly: year one of the horizon has
the whole fleet and the whole buildout in place. That is the right comparison
BETWEEN technologies -- it isolates the technology and not a deployment plan --
but it is not how anyone would actually do it, and the difference is large
enough that reporting only the instant case invites the obvious objection.

This stage produces the staged counterpart WITHOUT re-simulating anything. The
rollout table already carries per-year, undiscounted values for every cell and
the matching diesel baseline, so a staged corridor in year ``y`` is the blend

    staged(y) = f(y) * new(y) + (1 - f(y)) * diesel(y)

with ``f`` the converted share. Everything additive blends: capital, energy,
emissions, freight, usage. The result is re-discounted with the calculator's own
per-year ``Discounting_Factor``, so the discounting convention cannot drift from
the headline table.

**What the blend assumes.** A corridor converted a segment at a time: in year
``y`` a share ``f(y)`` of its traffic, its fleet AND its buildout is the new
technology, the rest is diesel. That is why infrastructure ramps on the same
schedule as the locomotives rather than being built up front -- staging the
fleet but not the wire would price a corridor that is fully electrified and
mostly running diesel under it, which is neither the instant case nor a staged
one. It also means the premium this reports is a *deferral* effect: the same
assets, bought later, discounted harder.

**What it does not capture.** Learning (later purchases ride the cost
trajectory, which IS captured, but no extra volume-driven decline), the
transition cost of running two fleets, and any efficiency loss from partial
electrification. The first leans against the staged case, the last two lean for
the instant one.

Usage:
    python postprocess_staged.py                      # default 15-year ramp
    python postprocess_staged.py --ramp-years 10
    python postprocess_staged.py --in out/rollout_costs.csv --out out/rollout_costs_staged.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

import polars as pl

HERE = Path(__file__).parent
OUT = HERE / "out"

logger = logging.getLogger("postprocess_staged")

# Metrics that are sums over the year and therefore blend linearly. Anything not
# listed is a derived quantity (a rate, a factor, a count of asset types) and is
# recomputed or carried from the headline cell instead of blended.
ADDITIVE_METRICS = (
    "Cost_Total",
    "Cost_Purchases",
    "Cost_Energy",
    "Cost_Diesel",
    "Cost_Electricity",
    "Cost_Retired_Assets",
    "Diesel_Usage",
    "Electricity_Usage",
    "Energy",
    "Freight_Moved",
    "GHG_Diesel",
    "GHG_Electricity",
    "GHG_Energy",
    "Count_Locomotives",
    # Already-discounted per-year series. Blending these directly is exact
    # rather than approximate, because discounting is linear in the value:
    #   disc(y) * (f(y)*new(y) + (1-f(y))*diesel(y))
    #     == f(y)*disc(y)*new(y) + (1-f(y))*disc(y)*diesel(y)
    # so there is no need to undo and redo the calculator's factors, and no way
    # for this stage's discount convention to drift from the headline table's.
    "Cost_Total_Discounted",
    "Mt-km_Discounted",
    "Mt-mi_Discounted",
)

# The levelized denominators. `Freight_Moved` is NOT one of them -- it is
# car-miles, which is why an early version of this stage produced an LCOTKM four
# orders of magnitude out.
LCOT_NUMERATOR = "Cost_Total_Discounted"
LCOT_DENOMINATORS = {"LCOTKM": "Mt-km_Discounted", "LCOTMI": "Mt-mi_Discounted"}

# The cell identity a staged variant must share with its diesel baseline. Infra
# and powertrain are deliberately absent: those are what differ.
#
# `REQUIRED_MATCH_KEYS` must be present or the blend is meaningless -- a staged
# cell has to be compared against the same corridor at the same energy prices.
# The rest are matched only when the table carries them: `rollout_costs.py`
# emits a narrower schema than the anti-idling post-process adds on top, so
# insisting on all of them would refuse the very table this normally runs after.
REQUIRED_MATCH_KEYS = ("Corridor", "Energy_Level")
OPTIONAL_MATCH_KEYS = ("Anti_Idling", "Dwell_Scope", "Battery_Cost_Scenario",
                       "Capital_Basis")


def converted_share(year: int, start_year: int, ramp_years: int) -> float:
    """Fraction of the corridor converted at the START of `year`.

    Linear from nothing in `start_year` to everything in
    `start_year + ramp_years`. Linear rather than S-shaped on purpose: an
    S-curve needs two more shape parameters that nothing in this study pins
    down, and it would move the answer while looking more sophisticated.
    """
    if ramp_years <= 0:
        return 1.0
    return min(max((year - start_year) / float(ramp_years), 0.0), 1.0)


def _year_int(frame: pl.DataFrame) -> pl.DataFrame:
    """`Year` as an integer, dropping the aggregate ("All") rows -- those are
    the calculator's own multi-year summaries, which this stage recomputes."""
    return (frame.filter(pl.col("Year").cast(pl.Utf8).str.contains(r"^\d{4}$"))
            .with_columns(pl.col("Year").cast(pl.Utf8).cast(pl.Int32).alias("Year_Int")))


def stage_one_cell(new: pl.DataFrame, diesel: pl.DataFrame,
                   ramp_years: int) -> Optional[pl.DataFrame]:
    """Blend one cell against its diesel baseline, year by year, and
    re-discount. Returns tidy rows in the input's own schema."""
    start_year = int(new["Start_Year"][0])
    n = _year_int(new)
    d = _year_int(diesel)
    if n.height == 0 or d.height == 0:
        return None

    def series(frame: pl.DataFrame, metric: str, subset=None) -> dict:
        """Year -> value for one metric, optionally within one subset."""
        sub = frame.filter(pl.col("Metric") == metric)
        if subset is not None:
            sub = sub.filter(pl.col("Subset").cast(pl.Utf8).fill_null("") == subset)
        if sub.height == 0:
            return {}
        return {int(y): float(v) for y, v in
                sub.group_by("Year_Int").agg(pl.col("Value").sum()).iter_rows()}

    def subsets_of(metric: str) -> list:
        """Every `Subset` the metric is broken out by, across both cells.

        The breakdown MUST be preserved: the lifecycle decomposition separates
        fleet from infrastructure by `Subset` on `Cost_Purchases`, and a staged
        table that summed them would collapse that split. A subset present in
        one cell and not the other blends against zero, which is what a corridor
        that has catenary only after conversion should do.
        """
        out = []
        for frame in (n, d):
            sub = frame.filter(pl.col("Metric") == metric)
            if sub.height:
                out += sub.get_column("Subset").cast(pl.Utf8).fill_null("").to_list()
        return sorted(set(out))

    years = sorted(set(n["Year_Int"].to_list()) & set(d["Year_Int"].to_list()))
    if not years:
        return None
    shares = {y: converted_share(y, start_year, ramp_years) for y in years}

    rows = []
    blended = {}
    for metric in ADDITIVE_METRICS:
        parts = subsets_of(metric)
        if not parts:
            continue
        units_sub = n.filter(pl.col("Metric") == metric)
        units = units_sub["Units"][0] if units_sub.height else None
        totals = {y: 0.0 for y in years}
        for part in parts:
            sn, sd = series(n, metric, part), series(d, metric, part)
            for y in years:
                f = shares[y]
                v = f * sn.get(y, 0.0) + (1.0 - f) * sd.get(y, 0.0)
                totals[y] += v
                rows.append(dict(Metric=metric, Units=units, Value=v,
                                 Subset=(part or None), Year=str(y)))
        blended[metric] = totals

    # The discounting factors ride along unblended: they are a property of the
    # year and the rate, identical in both cells, and `pv()`-style present-value
    # code downstream joins on them.
    for y, v in series(n, "Discounting_Factor").items():
        rows.append(dict(Metric="Discounting_Factor", Units="fraction (0-1)",
                         Value=v, Subset="All", Year=str(y)))

    for y in years:
        rows.append(dict(Metric="Converted_Share", Units="fraction",
                         Value=shares[y], Subset=None, Year=str(y)))

    # Horizon totals: sums of the blended per-year series, so the headline
    # table's own discounting carries straight through.
    def units_of(metric: str):
        sub = n.filter(pl.col("Metric") == metric)
        return sub["Units"][0] if sub.height else None

    cost_disc = sum(blended.get(LCOT_NUMERATOR, {}).get(y, 0.0) for y in years)
    rows.append(dict(Metric=LCOT_NUMERATOR, Units=units_of(LCOT_NUMERATOR),
                     Value=cost_disc, Subset=None, Year="All"))
    for lcot, denom in LCOT_DENOMINATORS.items():
        if denom not in blended:
            continue
        total = sum(blended[denom].get(y, 0.0) for y in years)
        rows.append(dict(Metric=denom, Units=units_of(denom), Value=total,
                         Subset=None, Year="All"))
        rows.append(dict(Metric=lcot, Units=units_of(lcot),
                         Value=(cost_disc / total) if total else None,
                         Subset=None, Year="All"))
    for metric in ("Cost_Total", "Cost_Purchases", "Cost_Energy", "GHG_Energy"):
        if metric in blended:
            rows.append(dict(Metric=metric, Units=units_of(metric),
                             Value=sum(blended[metric].get(y, 0.0) for y in years),
                             Subset=None, Year="All"))
    rows.append(dict(Metric="Ramp_Years", Units="years",
                     Value=float(ramp_years), Subset=None, Year="All"))

    provenance = {k: new[k][0] for k in new.columns
                  if k not in ("Metric", "Units", "Value", "Subset", "Year")}
    prov = pl.DataFrame([provenance])
    return prov.join(pl.DataFrame(rows), how="cross")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="inp", type=Path, default=OUT / "rollout_costs.csv")
    ap.add_argument("--out", dest="outp", type=Path,
                    default=OUT / "rollout_costs_staged.csv")
    ap.add_argument("--ramp-years", type=int, default=15,
                    help="years from no conversion to full conversion (default 15, "
                         "i.e. 2030-2045)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.inp.exists():
        logger.error("%s is missing; run rollout_costs.py first", args.inp)
        return 1
    table = pl.read_csv(args.inp, infer_schema_length=None)
    # `Anti_Idling` names the same thing two ways: a diesel cell with no
    # anti-idling measure is `none`, an electrified cell is `not_applicable`
    # (it has no main-engine idle to suppress). Matching on the raw value
    # therefore never pairs an electrified cell with its diesel baseline, so
    # they are normalized to one label before matching.
    if "Anti_Idling" in table.columns:
        table = table.with_columns(
            pl.when(pl.col("Anti_Idling").cast(pl.Utf8)
                    .is_in(["not_applicable", "aess_none"]))
            .then(pl.lit("none"))
            .otherwise(pl.col("Anti_Idling").cast(pl.Utf8))
            .alias("Anti_Idling"))
    missing = [k for k in REQUIRED_MATCH_KEYS if k not in table.columns]
    if missing:
        logger.error("rollout table lacks %s", missing)
        return 1
    match_keys = tuple(REQUIRED_MATCH_KEYS) + tuple(
        k for k in OPTIONAL_MATCH_KEYS if k in table.columns)
    logger.info("matching baselines on %s", ", ".join(match_keys))

    diesel = table.filter(pl.col("Powertrain") == "diesel")
    others = table.filter(pl.col("Powertrain") != "diesel")
    if diesel.height == 0:
        logger.error("no diesel baseline rows: a staged blend has nothing to "
                     "ramp away from")
        return 1

    frames, skipped = [], []
    group_keys = ["Sim_Key", "Infra_Level", "Energy_Level"] + [
        k for k in ("Anti_Idling", "Dwell_Scope") if k in table.columns]
    for key, cell in others.group_by(group_keys, maintain_order=True):
        base = diesel
        for k in match_keys:
            base = base.filter(pl.col(k) == cell[k][0])
        if base.height == 0:
            skipped.append((key, "no matching diesel baseline"))
            continue
        # One diesel cell per match: if several survive (different infra levels
        # on the baseline), take the cheapest-infra one, which is the no-build
        # diesel the study compares against.
        if base["Sim_Key"].n_unique() > 1:
            base = base.filter(pl.col("Sim_Key") == sorted(base["Sim_Key"].unique())[0])
        out = stage_one_cell(cell, base, args.ramp_years)
        if out is None:
            skipped.append((key, "no overlapping years"))
            continue
        frames.append(out)

    if not frames:
        logger.error("nothing staged")
        return 1
    result = pl.concat(frames, how="diagonal_relaxed")
    args.outp.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(args.outp)
    logger.info("wrote %s: %d rows over %d cells (%d-year ramp)",
                args.outp, result.height, len(frames), args.ramp_years)
    for key, why in skipped:
        logger.warning("skipped %s: %s", key, why)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
