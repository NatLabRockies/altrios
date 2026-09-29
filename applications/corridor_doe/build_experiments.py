"""
Build the full experiment table for the three-corridor design of experiments.

Every row is one *analysis cell*. Rows sharing a ``Sim_Key`` share a single
simulation run: only corridor, powertrain, battery size and catenary
architecture change the physics, so cost scenarios and anti-idling levels are
expansions of an already-computed result rather than new runs. Keeping both in
one table -- with ``Sim_Key`` saying which is which -- means the sweep can group
by ``Sim_Key`` to find its real work, while the table still states the complete
design rather than half of it.

Factors, per the DoE specification:

1. **Corridor** -- three of them. Spokane-Lethbridge is the catenary deep dive.
2. **Powertrain** -- conventional diesel and hybrid everywhere; catenary with
   battery-electric and catenary with dual-mode on the deep-dive corridor only.
3. **Catenary architecture** -- line voltage, replicating the existing
   ``catenary_optimization`` analysis (25 kV AC, 2x25 kV AC autotransformer,
   9 kV DC), crossed with its infrastructure-cost levels.
4. **Energy prices** -- one harmonized set of diesel and electricity price
   scenarios applied to catenary and non-catenary cells alike, so the two are
   comparable. This is why ``Energy_Level`` is on every row rather than only on
   catenary rows.
5. **Battery size** -- 0.5 and 1.0 MWh, on the hybrid, every corridor. Two
   points, not four: `make_est_times` builds each train's DAG by simulating
   it, and a hybrid consist costs roughly 5.7x a diesel one per train, so
   each extra battery size is hours of wall-clock per corridor for a
   sensitivity that two points already show the direction of.
6. **Anti-idling** -- post-processing only; see ``postprocess_antiidling.py``.
   Battery-as-APU applies to hybrids alone, and no anti-idling level applies to
   the catenary battery-electric locomotive, which burns no fuel to idle.

Usage:
    python build_experiments.py            # writes experiments.csv
    python build_experiments.py --summary  # also print the factor breakdown
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import polars as pl

HERE = Path(__file__).parent
IN = HERE / "in"
LAYOUTS = HERE / "out" / "layouts" / "layouts.csv"
NA = "na"


def _layout_map() -> Optional[Dict[str, str]]:
    """(powertrain, architecture, infra, energy) -> layout fingerprint.

    The two catenary powertrains are keyed differently on purpose, following the
    catenary study. A DUAL-MODE buildout is an NPV decision, re-optimized for
    every cost cell, so its key carries the cost levels and some cells map to
    "nobuild". A BATTERY-ELECTRIC buildout is a feasibility decision pinned by
    battery range: optimized once at the reference cost point and only re-priced,
    so every cost cell maps to that one layout. A BEL cell never resolves to
    "nobuild" -- a battery-electric locomotive with no wire cannot move, and
    such a cell is a design error rather than an economic finding.
    """
    if not LAYOUTS.exists():
        return None
    table = pl.read_csv(LAYOUTS)
    out: Dict[str, str] = {}
    for row in table.iter_rows(named=True):
        fp = row.get("fingerprint")
        miles = row.get("energized_track_miles") or 0.0
        resolved = fp if (fp and miles > 0) else "nobuild"
        powertrain = row.get("Powertrain") or "catenary_dual"
        if powertrain == "catenary_bel":
            out[f"catenary_bel|{row['Architecture']}"] = resolved
        else:
            out["|".join((powertrain, row["Architecture"],
                          row["Infra_Level"], row["Energy_Level"]))] = resolved
    return out


def _layout_expr(layouts: Optional[Dict[str, str]]) -> pl.Expr:
    """Layout fingerprint per row."""
    if layouts is None:
        return pl.col("Architecture")
    key = (
        pl.when(pl.col("Powertrain") == pl.lit("catenary_bel"))
        .then(
            pl.concat_str(
                [pl.lit("catenary_bel"), pl.col("Architecture")], separator="|"
            )
        )
        .otherwise(
            pl.concat_str(
                [pl.col("Powertrain"), pl.col("Architecture"),
                 pl.col("Infra_Level"), pl.col("Energy_Level")],
                separator="|",
            )
        )
    )
    return key.replace_strict(layouts, default="nobuild")


def build() -> pl.DataFrame:
    corridors = pl.read_csv(IN / "corridors.csv")
    powertrains = pl.read_csv(IN / "powertrains.csv")
    batteries = pl.read_csv(IN / "battery_sizes.csv")
    arches = pl.read_csv(IN / "catenary_architectures.csv")
    infra = pl.read_csv(IN / "cost_cells.csv")
    energy = pl.read_csv(IN / "energy_cells.csv")
    idling = pl.read_csv(IN / "anti_idling.csv")

    # 1-2. Corridor x powertrain, dropping catenary powertrains from the
    # corridors that are not the deep dive.
    cells = corridors.join(powertrains, how="cross").filter(
        pl.col("Requires_Catenary") <= pl.col("Catenary_Deep_Dive")
    )

    # 5. Battery size, on the swept powertrains only. Everything else carries a
    # null, meaning "not a factor here" rather than "zero".
    first_battery = batteries["Battery_MWh"][0]
    cells = (
        cells.join(batteries, how="cross")
        .filter(
            (pl.col("Sweep_Battery") == 1) | (pl.col("Battery_MWh") == first_battery)
        )
        .with_columns(
            pl.when(pl.col("Sweep_Battery") == 1)
            .then(pl.col("Battery_MWh"))
            .otherwise(None)
            .alias("Battery_MWh")
        )
    )

    # 3. Catenary architecture, on catenary powertrains only.
    arch_cols = [c for c in arches.columns if c != "Architecture"]
    first_arch = arches["Architecture"][0]
    cells = (
        cells.join(arches, how="cross")
        .filter(
            (pl.col("Requires_Catenary") == 1)
            | (pl.col("Architecture") == first_arch)
        )
        .with_columns(
            [
                pl.when(pl.col("Requires_Catenary") == 1)
                .then(pl.col(c))
                .otherwise(None)
                .alias(c)
                for c in arch_cols
            ]
            + [
                pl.when(pl.col("Requires_Catenary") == 1)
                .then(pl.col("Architecture"))
                .otherwise(pl.lit(NA))
                .alias("Architecture")
            ]
        )
    )

    # 3-4. Cost scenarios. Infrastructure multipliers only mean something where
    # infrastructure is built; energy prices apply everywhere, harmonized.
    cells = (
        cells.join(infra, how="cross")
        .filter(
            pl.when(pl.col("Requires_Catenary") == 1)
            .then(pl.col("Applies_To") == pl.lit("catenary"))
            .otherwise(pl.col("Applies_To") == pl.lit("non_catenary"))
        )
        .drop("Applies_To")
        .join(energy, how="cross")
    )

    # The simulation key.
    #
    # For every powertrain but catenary, cost is pure re-pricing: diesel getting
    # dearer does not change how a train moves, so one simulation serves all
    # nine cost cells. A catenary buildout is different -- it is an NPV decision,
    # so the layout itself moves with capital cost and with the diesel-to-power
    # spread, and each cost cell is genuinely a different network.
    #
    # Cells whose optimization lands on the SAME buildout still share a
    # simulation: the key carries the layout fingerprint from build_layouts, not
    # the cost labels. That is what keeps 27 optimizations from becoming 27
    # full-plan sims, and it is how the catenary study does it too. Cells where
    # the optimizer declines to electrify share a single "nobuild" key -- the
    # corridor is simply unelectrified, whatever the prices that led there.
    layouts = _layout_map()
    if layouts is None:
        logging.warning(
            "out/layouts/layouts.csv not found -- catenary cells will key on "
            "architecture alone, which assumes one buildout per architecture. "
            "Run build_layouts.py, then rebuild this table."
        )
    cells = cells.with_columns(
        pl.when(pl.col("Requires_Catenary") == 1)
        .then(
            # Architecture is deliberately absent: the fingerprint IS the
            # network's identity, so two architectures whose optimizer declined
            # to electrify are the same unelectrified corridor and must share
            # one simulation rather than three identical ones. Architecture is
            # still on the row for reporting and cost.
            pl.concat_str(
                [
                    pl.col("Corridor"),
                    pl.col("Powertrain"),
                    pl.col("Battery_MWh").fill_null(-1.0).cast(pl.Utf8),
                    _layout_expr(layouts),
                ],
                separator="|",
            )
        )
        .otherwise(
            pl.concat_str(
                [
                    pl.col("Corridor"),
                    pl.col("Powertrain"),
                    pl.col("Battery_MWh").fill_null(-1.0).cast(pl.Utf8),
                    pl.col("Architecture"),
                ],
                separator="|",
            )
        )
        .alias("Sim_Key")
    )

    # 6. Anti-idling, filtered to the powertrains each level applies to.
    cells = (
        cells.join(idling, how="cross")
        .filter(
            pl.col("Applies_To_Powertrains")
            .str.split("|")
            .list.contains(pl.col("Powertrain"))
        )
        .drop("Applies_To_Powertrains")
    )

    return (
        cells.with_columns(
            pl.concat_str(
                [
                    pl.col("Sim_Key"),
                    pl.col("Infra_Level"),
                    pl.col("Energy_Level"),
                    pl.col("Anti_Idling"),
                ],
                separator="|",
            ).alias("Cell_ID")
        )
        .with_columns(
            pl.when(pl.col("Requires_Catenary") == 1)
            .then(_layout_expr(layouts))
            .otherwise(pl.lit(NA))
            .alias("Layout_Fingerprint")
        )
        .sort(
            "Corridor", "Powertrain", "Battery_MWh", "Architecture",
            "Infra_Level", "Energy_Level", "Anti_Idling",
        )
        .select(
            "Cell_ID", "Sim_Key", "Corridor", "Powertrain", "Label",
            "Battery_MWh", "Architecture", "Infra_Level", "Infra_Multiplier",
            "Energy_Level", "Diesel_Multiplier", "Electricity_Multiplier",
            "Layout_Fingerprint",
            "Anti_Idling", "Idle_Source", "Shutdown_Threshold_Seconds",
            "Restart_Penalty_MJ_Per_Event", "Cycle_Off_Minutes", "Cycle_On_Minutes",
            "Network_Path", "Locations_Path", "System_Voltage_V",
            "Electrification_Type", "Structure_MUSD_Per_Route_Mi",
            "Energization_MUSD_Per_Track_Mi", "Substation_MUSD_Each",
            "Consumes_Diesel",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", type=Path, default=HERE / "experiments.csv")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    table = build()
    table.write_csv(args.out)
    print(
        f"{table.height} analysis cells, "
        f"{table['Sim_Key'].n_unique()} simulation runs"
    )
    print(f"wrote {args.out}")

    if args.summary:
        print()
        for col in (
            "Corridor", "Powertrain", "Battery_MWh", "Architecture",
            "Infra_Level", "Energy_Level", "Anti_Idling",
        ):
            counts = table.group_by(col).len().sort(col)
            print(f"{col}: " + ", ".join(f"{k}={v}" for k, v in counts.iter_rows()))
        print()
        print("simulation runs per corridor:")
        sims = (
            table.unique(subset=["Sim_Key"])
            .group_by("Corridor")
            .len()
            .sort("Corridor")
        )
        for corridor, n in sims.iter_rows():
            print(f"  {corridor}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
