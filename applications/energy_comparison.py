"""
Does it matter, for energy, whether missing run time is modelled as slow track or as dwell?

``speed_calibration`` shows posted speed limits plus meet-pass reproduce about
70% of observed run time on the two corridors whose posted limits are real
data (Toronto-Battle Creek -31.0%, Winnipeg-Chicago -29.8%). Two ways of
closing that gap both match the observed *time*:

- **Scale the speed limits** (the historical approach, refit here to 0.70).
  Every train runs slower everywhere.
- **Keep posted limits and add dwell.** Trains run at line speed and then wait.

They are not interchangeable for the thing the corridor is actually for.
Aerodynamic resistance goes as speed squared, so a train held to 75% of line
speed spends a materially different amount of energy from one that runs at line
speed and then idles -- and the second draws full power in bursts rather than
reduced power continuously, which is what sizing catenary depends on.

This script runs both and reports the difference. Dwell contributes no traction
energy at all (same distance, same speeds); it adds locomotive idle fuel, which
is charged here at the fuel converter's own ``pwr_idle_fuel_watts``.

If the two totals agree closely, the distinction does not matter for the
deliverable and the scaler is simply fine. If they diverge, the gap is what the
scaler costs, and is the argument for modelling dwell explicitly.

Usage:
    python energy_comparison.py --corridor winnipeg_chicago
    python energy_comparison.py --all --trains 40
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import polars as pl

import altrios as alt
from altrios import sim_manager

from speed_calibration import (
    CORRIDORS,
    Corridor,
    INPUTS,
    NETWORKS,
    dual_mode_planner_config,
    load_rail_vehicles,
    prepare_network,
)

logger = logging.getLogger(__name__)

# Fuel burned by an idling locomotive, from the default dual-mode model's
# fuel converter. Dwell costs this per locomotive per second.
IDLE_FUEL_W = 19_703.2784

# Refit from the calibration: simulated run time is ~0.70 of observed, so
# scaling every speed limit by this reproduces the observed time. (An earlier
# 0.75 was fitted before `simulated_segment_times` interpolated at corridor
# offset; exact-link matching registered only the minority of trains whose
# dispatch nodes fell on a sensor link, and that subsample ran biased fast.)
REFIT_SCALER = 0.70


def run_variant(
    corridor: Corridor,
    plan: pl.DataFrame,
    scaler: Optional[float],
) -> Tuple[float, float, int]:
    """Walk the plan and return ``(fuel_joules, train_hours, n_trains)``.

    ``walk_timed_paths`` is on here, unlike in the calibration: energy only
    exists once the train is actually walked through the physics.
    """
    network = alt.Network.from_file(str(NETWORKS / corridor.network))
    if scaler is not None:
        d = network.to_pydict()
        d = alt.utilities.scale_network_speed_limits(d, scaler)
        network = alt.Network.from_pydict(d)

    start = time.perf_counter()
    (plan_out, _lp, _rf, _gef, _nep, sims, _tp, _un) = sim_manager.main(
        network=network,
        rail_vehicles=load_rail_vehicles(),
        location_map=alt.import_locations(str(NETWORKS / corridor.locations)),
        consist_plan_in=plan,
        train_planner_config=dual_mode_planner_config(),
        walk_timed_paths=True,
        debug=False,
    )
    fuel = sims.get_energy_fuel_joules(annualize=False)
    hours = (
        plan_out.select(
            (pl.col("Arrival_Time_Actual_Hr") - pl.col("Departure_Time_Actual_Hr"))
        )
        .to_series()
        .sum()
    )
    logger.info(
        "  %-28s %6.1f s | fuel %.4e J | train-hours %.1f",
        "scaler %.2f" % scaler if scaler else "posted limits",
        time.perf_counter() - start,
        fuel,
        hours,
    )
    return fuel, float(hours), plan_out["Train_ID"].n_unique()


def observed_ratio(corridor: Corridor) -> float:
    """Observed run time divided by simulated, from the calibration output.

    Distance-weighted across segments, so a corridor with several sensor pairs
    is not dominated by its shortest one.
    """
    path = INPUTS / f"calibration_{corridor.slug}.csv"
    table = pl.read_csv(path)
    weight = table["km"]
    ratio = (table["obs_median_hr"] / table["sim_median_hr"] * weight).sum() / weight.sum()
    return float(ratio)


def compare(corridor: Corridor, n_trains: Optional[int]) -> dict:
    plan = pl.read_csv(corridor.plan_path)
    if n_trains:
        # Energy per train-km is what is being compared, so a subset suffices
        # and keeps the physics walk affordable.
        keep = plan["Train_ID"].unique().sort()[:n_trains]
        plan = plan.filter(pl.col("Train_ID").is_in(keep))

    ratio = observed_ratio(corridor)
    logger.info(
        "%s: %d trains | observed/simulated run time = %.3f",
        corridor.slug,
        plan["Train_ID"].n_unique(),
        ratio,
    )

    scaled_fuel, scaled_hours, n = run_variant(corridor, plan, REFIT_SCALER)
    posted_fuel, posted_hours, _ = run_variant(corridor, plan, None)

    # Dwell needed so posted-limit runs take as long as observed. Traction
    # energy is unchanged by waiting; only idle fuel is added.
    dwell_hours = posted_hours * (ratio - 1.0)
    locos = plan.height / max(plan["Train_ID"].n_unique(), 1)
    idle_fuel = dwell_hours * 3600 * IDLE_FUEL_W * locos
    posted_total = posted_fuel + idle_fuel

    return {
        "corridor": corridor.slug,
        "trains": n,
        "obs_over_sim": ratio,
        "scaled_fuel_GJ": scaled_fuel / 1e9,
        "posted_traction_GJ": posted_fuel / 1e9,
        "dwell_idle_GJ": idle_fuel / 1e9,
        "posted_total_GJ": posted_total / 1e9,
        "scaled_hours": scaled_hours,
        "posted_hours": posted_hours,
        "dwell_hours": dwell_hours,
        "locos_per_train": locos,
        "pct_diff": 100 * (scaled_fuel - posted_total) / posted_total,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--corridor", choices=sorted(CORRIDORS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--trains",
        type=int,
        default=40,
        help="cap the number of trains walked (0 for all); the comparison is "
        "between two variants of the same set, so a subset is representative",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    slugs = sorted(CORRIDORS) if args.all else [args.corridor]
    if not slugs or slugs == [None]:
        parser.error("pass --corridor or --all")

    rows = []
    for slug in slugs:
        try:
            rows.append(compare(CORRIDORS[slug], args.trains or None))
        except Exception as exc:  # noqa: BLE001 - report and continue to the next corridor
            logger.error("%s failed: %s", slug, exc)

    if rows:
        table = pl.DataFrame(rows)
        logger.info("\n%s", table.to_pandas().to_string(index=False))
        for row in rows:
            logger.info(
                "%s: scaler %.2f burns %+.1f%% the fuel of posted-limits-plus-dwell "
                "(%.1f vs %.1f GJ, of which %.1f GJ is idle)",
                row["corridor"],
                REFIT_SCALER,
                row["pct_diff"],
                row["scaled_fuel_GJ"],
                row["posted_total_GJ"],
                row["dwell_idle_GJ"],
            )
        if args.out:
            table.write_csv(args.out)
            logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
