"""
Optimize a catenary buildout for every architecture x cost cell, and save each
as a simulation-ready network.

This replicates the catenary study's own design rather than approximating it.
Three things follow from that, and each was wrong in an earlier version:

**Dual-mode is re-optimized per cost cell; battery-electric is not.** This
mirrors the catenary study exactly, and the asymmetry is the point.

A DUAL-MODE buildout is an NPV decision -- wire pays for itself by displacing
diesel -- so it genuinely moves with infrastructure capital and with the
diesel-to-electricity spread, and some cost cells rationally build nothing. Nine
cost cells per architecture, re-optimized each time.

A BATTERY-ELECTRIC buildout is a FEASIBILITY decision. Its extent is pinned by
how far the pack can run between energized stretches (``max_energized_gap_kwh``
against per-link drain), so it is optimized ONCE per architecture at the
reference cost point and only re-priced across the grid -- never re-optimized,
and never "no build". Treating it like dual-mode, as an earlier version of this
script did, manufactures cost cells with a battery-electric locomotive and no
wire, which cannot move at all: those runs abort in ``est_times`` after seconds.
That is not a finding about electrification economics, it is a miscopied
experiment design.

**Energy densities come from simulation, not from constants.** The optimizer
decides by comparing displaced diesel against capital, so the diesel figure
decides the answer. ``energy_densities`` defaults to 3 gal/train-mile; this
corridor's own diesel baseline burns about 10. Using the constants understated
the saving roughly threefold and made ``min_cost`` build nothing anywhere --
which contradicted the catenary study, where buildout appears in several cost
cells. ``first_pass_densities`` runs the real thing: every plan train, twice
(all-diesel and all-catenary), for per-link annual densities.

**The prices are the study's prices.** The first pass is driven by *this* DoE's
consist plan and *this* DoE's speed rule (min(posted, observed p99)), and the
optimizer prices diesel and electricity from the same harmonized EIA figures the
results are scored with. Optimizing a buildout against one set of assumptions
and scoring it against another would make the comparison meaningless.

Layouts that come out identical across cost cells are written once and shared;
``layouts.csv`` maps every cell to its layout file, so ``run_sweep`` can dedupe
simulations the same way the catenary study does.

Usage:
    python build_layouts.py                    # all 27 cells
    python build_layouts.py --arch dc_9kv
    python build_layouts.py --densities-only   # just the first pass (cached)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import polars as pl

HERE = Path(__file__).parent
LAYOUTS = HERE / "out" / "layouts"
CATENARY_DIR = HERE.parent / "catenary_optimization"
PLAN_PATH = (
    HERE.parent / "corridor_inputs" / "train_consist_plan_spokane_lethbridge_clean.csv"
)

# Drive the catenary app's speed calibration to the DoE's rule before importing
# it: a hard cap at the corridor's observed p99, no legacy scaler. The first
# pass simulates through `_calibrate_speeds`, so this has to be set for the
# densities to describe the corridor the DoE actually runs.
SPEED_CAP_MPH = 34.0
os.environ.setdefault("ALTRIOS_SPEED_CAP_MPH", str(SPEED_CAP_MPH))
os.environ.setdefault("ALTRIOS_SPEED_SCALER", "none")

sys.path.insert(0, str(CATENARY_DIR))

from altrios import energization as en  # noqa: E402
from altrios.optimization import energization_placement as cp  # noqa: E402

logger = logging.getLogger(__name__)

# Harmonized base prices: the same EIA AEO figures postprocess_antiidling uses.
DIESEL_USD_PER_GALLON = 3.65
ELECTRICITY_USD_PER_KWH = 0.091

MIN_ENERGIZED_LENGTH_MILES = 5.0


def _cost_levels() -> Tuple[pl.DataFrame, pl.DataFrame]:
    infra = pl.read_csv(HERE / "in" / "cost_cells.csv").filter(
        pl.col("Applies_To") == pl.lit("catenary")
    )
    energy = pl.read_csv(HERE / "in" / "energy_cells.csv")
    return infra, energy


def trains_per_day(plan_path: Path, plan_days: float = 7.0) -> float:
    """Traffic level from this DoE's own consist plan, not the app's copy.

    The BEL opex model is per-train, so this has to describe the traffic the
    study actually simulates rather than whatever plan the catenary app ships.
    """
    return pl.read_csv(plan_path)["Train_ID"].n_unique() / plan_days


def prepare_app(plan_path: Path):
    """Import the catenary app and point it at this DoE's consist plan.

    The app caches the first pass by a key that hashes the plan bytes and the
    speed calibration, so repointing it invalidates correctly rather than
    silently reusing densities computed for different traffic.
    """
    import energization_optimization_corridor as cat

    cat.CONSIST_PLAN_PATH = plan_path
    cat._FULL_CONSIST_PLAN = None
    cat._PLAN_SUMMARY = None
    logger.info(
        "app speed calibration: cap=%s mph scaler=%s | plan=%s",
        cat.SPEED_CALIBRATION_MAX_MPH, cat.SPEED_LIMIT_SCALER, plan_path.name,
    )
    return cat


def layout_fingerprint(result) -> str:
    """Stable hash of energized intervals + built substations.

    Cost cells that yield the same buildout share one network file and one
    simulation, which is how the catenary study keeps 27 optimizations from
    becoming 27 full-plan sims.
    """
    intervals = sorted(
        (int(li), tuple(sorted((round(a, 3), round(b, 3)) for a, b in ivs)))
        for li, ivs in (result.energized_link_intervals or {}).items()
    )
    payload = json.dumps(
        {"intervals": intervals, "subs": sorted(result.built_substation_ids or [])},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def build_bel_layout(cat, arch, costs_row, densities, out_dir):
    """The battery-electric buildout for one architecture.

    Optimized once, at the reference (med) cost point, under the battery-range
    constraint: every unelectrified gap must be one the pack can cross. The cost
    grid re-prices this layout rather than re-deriving it, so a BEL cell always
    has wire under it.
    """
    sug = en.suggest_optimization_inputs(arch)
    _gal, kwh = densities
    costs = cp.EnergizationOptCosts(
        structure_cost_per_mile=float(costs_row["Structure_MUSD_Per_Route_Mi"]),
        energization_cost_per_mile=float(costs_row["Energization_MUSD_Per_Track_Mi"]),
        substation_fixed_cost=float(costs_row["Substation_MUSD_Each"]),
    )
    econ = cp.EconomicParameters(
        project_lifespan_years=cat.PROJECT_LIFESPAN_YEARS,
        discount_rate=cat.DISCOUNT_RATE,
    )
    drain = cat.bel_drain_by_link(kwh)
    tpd = trains_per_day(PLAN_PATH)
    started = time.perf_counter()
    result = cat.run_opt(
        f"doe/{arch}/bel",
        objective="min_cost",
        economics=econ,
        substation_feed_distance_miles=sug.substation_feed_distance_miles,
        costs=costs,
        piece_opex_fn=cat.build_bel_opex_fn(tpd, ELECTRICITY_USD_PER_KWH),
        max_energized_gap_kwh=cat.BEL_MAX_GAP_KWH,
        gap_energy_kwh_per_mile_by_link=drain,
        charge_energy_kwh_per_mile_by_link=(
            cat.bel_charge_by_link(drain)
            if (cat.BEL_RECHARGE_TIER1 or cat.BEL_SOC_STATE_MODEL) else None
        ),
        regen_energy_kwh_per_mile_by_link=cat.bel_regen_by_link(drain),
        battery_soc_state_model=cat.BEL_SOC_STATE_MODEL,
        min_energized_length_miles=(
            5.0 if cat.BEL_SOC_STATE_MODEL
            else max(5.0, cat.BEL_RECHARGE_MILES)
        ),
        warmstart=True,
        **cat.COMMON,
    )
    row = {
        "Architecture": arch,
        "Powertrain": "catenary_bel",
        "Infra_Level": "ref",
        "Energy_Level": "ref",
        "feed_distance_miles": sug.substation_feed_distance_miles,
        "system_voltage_volts": sug.system_voltage_volts,
        "electrification_type": sug.electrification_type,
        "optimize_seconds": time.perf_counter() - started,
        "max_gap_kwh": cat.BEL_MAX_GAP_KWH,
    }
    return _finalize(cat, result, sug, row, out_dir)


def _finalize(cat, result, sug, row, out_dir):
    """Apply line losses, write the network, and record what was built."""
    if result is None or result.network is None:
        row.update(fingerprint=None, path=None, energized_track_miles=0.0,
                   substations_built=0, capex_musd=0.0,
                   intervals_with_voltage=0, status="no layout")
        return row
    miles = float(getattr(result, "energized_track_miles", 0.0) or 0.0)
    fp = layout_fingerprint(result)
    row.update(
        fingerprint=fp,
        energized_track_miles=miles,
        capex_musd=float(getattr(result, "total_cost", 0.0) or 0.0),
        status=str(getattr(result, "status", "")).split(".")[-1],
    )
    if miles <= 0.0:
        row.update(path=None, substations_built=0, intervals_with_voltage=0)
        return row
    path = out_dir / f"{fp}.msgpack"
    subs = cat._built_sub_objs(result)
    if not path.exists():
        network, _meta = en.apply_line_losses(
            result.network, subs,
            system_voltage_volts=sug.system_voltage_volts,
            electrification_type=sug.electrification_type,
            base_power_limit_watts=cat.COMMON["uniform_power_limit_watts"],
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        network.to_file(str(path))
        lims = [l for L in network.to_pydict()[1]
                for l in (L.get("cat_power_limits") or [])]
        row["intervals_with_voltage"] = sum(
            1 for l in lims if (l.get("voltage_volts") or 0) > 0
        )
    else:
        row["intervals_with_voltage"] = -1
    row.update(path=str(path), substations_built=len(subs))
    return row


def build_cell(cat, arch, costs_row, infra_row, energy_row, densities, out_dir):
    """Optimize one dual-mode (architecture, infra, energy) cell."""
    sug = en.suggest_optimization_inputs(arch)
    infra_mult = float(infra_row["Infra_Multiplier"])
    costs = cp.EnergizationOptCosts(
        structure_cost_per_mile=float(costs_row["Structure_MUSD_Per_Route_Mi"]) * infra_mult,
        energization_cost_per_mile=float(costs_row["Energization_MUSD_Per_Track_Mi"]) * infra_mult,
        substation_fixed_cost=float(costs_row["Substation_MUSD_Each"]) * infra_mult,
    )
    diesel_by_link, elec_by_link = densities
    econ = cp.EconomicParameters(
        project_lifespan_years=cat.PROJECT_LIFESPAN_YEARS,
        discount_rate=cat.DISCOUNT_RATE,
    )
    diesel_price = DIESEL_USD_PER_GALLON * float(energy_row["Diesel_Multiplier"])
    elec_price = ELECTRICITY_USD_PER_KWH * float(energy_row["Electricity_Multiplier"])

    started = time.perf_counter()
    result = cat.run_opt(
        f"doe/{arch}/infra-{infra_row['Infra_Level']}/energy-{energy_row['Energy_Level']}",
        objective="min_cost",
        economics=econ,
        substation_feed_distance_miles=sug.substation_feed_distance_miles,
        costs=costs,
        piece_opex_fn=cat.build_opex_fn(
            diesel_by_link, elec_by_link, diesel_price, elec_price
        ),
        min_energized_length_miles=MIN_ENERGIZED_LENGTH_MILES,
        **cat.COMMON,
    )
    elapsed = time.perf_counter() - started
    row = {
        "Architecture": arch,
        "Powertrain": "catenary_dual",
        "Infra_Level": infra_row["Infra_Level"],
        "Energy_Level": energy_row["Energy_Level"],
        "diesel_usd_per_gal": diesel_price,
        "electricity_usd_per_kwh": elec_price,
        "infra_multiplier": infra_mult,
        "feed_distance_miles": sug.substation_feed_distance_miles,
        "system_voltage_volts": sug.system_voltage_volts,
        "electrification_type": sug.electrification_type,
        "optimize_seconds": elapsed,
    }
    return _finalize(cat, result, sug, row, out_dir)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--arch", action="append", default=[])
    parser.add_argument("--out-dir", type=Path, default=LAYOUTS)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--densities-only", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cat = prepare_app(args.plan)
    logger.info("first pass over the full consist plan (cached after the first run)")
    densities = cat.first_pass_densities()
    gal, kwh = densities
    import statistics as st
    logger.info(
        "sim-derived densities: diesel median %.0f gal/track-mi/yr (max %.0f), "
        "electricity median %.0f kWh/track-mi/yr",
        st.median(gal.values()), max(gal.values()), st.median(kwh.values()),
    )
    if args.densities_only:
        return 0

    arches = pl.read_csv(HERE / "in" / "catenary_architectures.csv")
    if args.arch:
        arches = arches.filter(pl.col("Architecture").is_in(args.arch))
    infra, energy = _cost_levels()

    rows = []
    # Battery-electric: one feasibility-pinned layout per architecture, at the
    # reference cost point. The cost grid re-prices it; it is never rebuilt.
    ref_infra = infra.filter(pl.col("Infra_Level") == pl.lit("med"))
    ref_row = ref_infra.row(0, named=True) if ref_infra.height else None
    for arch_row in arches.iter_rows(named=True):
        try:
            row = build_bel_layout(
                cat, arch_row["Architecture"], arch_row, densities, args.out_dir
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("BEL cell FAILED: %s", str(exc)[:160])
            continue
        rows.append(row)
        logger.info(
            "  %-8s BEL (ref cost)      -> %7.1f track-mi, %2d subs, "
            "$%7.1fM, %s (%.0f s)",
            row["Architecture"], row["energized_track_miles"],
            row["substations_built"], row["capex_musd"],
            row["fingerprint"] or "NO BUILD", row["optimize_seconds"],
        )
        pl.DataFrame(rows).write_csv(args.out_dir / "layouts.csv")

    # Dual-mode: an NPV decision, re-optimized for every cost cell.
    for arch_row in arches.iter_rows(named=True):
        for infra_row in infra.iter_rows(named=True):
            for energy_row in energy.iter_rows(named=True):
                try:
                    row = build_cell(
                        cat, arch_row["Architecture"], arch_row,
                        infra_row, energy_row, densities, args.out_dir,
                    )
                except Exception as exc:  # noqa: BLE001 - one cell is not all
                    logger.exception("cell FAILED: %s", str(exc)[:160])
                    continue
                rows.append(row)
                logger.info(
                    "  %-8s infra=%-4s energy=%-4s -> %7.1f track-mi, "
                    "%2d subs, $%7.1fM, %s (%.0f s)",
                    row["Architecture"], row["Infra_Level"], row["Energy_Level"],
                    row["energized_track_miles"], row["substations_built"],
                    row["capex_musd"], row["fingerprint"] or "NO BUILD",
                    row["optimize_seconds"],
                )
                pl.DataFrame(rows).write_csv(args.out_dir / "layouts.csv")

    if rows:
        table = pl.DataFrame(rows)
        for pt in table["Powertrain"].unique().sort():
            sub_t = table.filter(pl.col("Powertrain") == pt)
            built = sub_t.filter(pl.col("energized_track_miles") > 0)
            logger.info(
                "%-14s %d of %d cells electrify; %d distinct layouts",
                pt, built.height, sub_t.height,
                built["fingerprint"].n_unique() if built.height else 0,
            )
        logger.info("wrote %s", args.out_dir / "layouts.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
