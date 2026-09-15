"""
Run the three-corridor design of experiments as a parametric sweep.

Reads ``experiments.csv`` (built by ``build_experiments.py``), groups it by
``Sim_Key``, and runs each distinct simulation exactly once. Cost scenarios and
anti-idling levels do not change the physics, so they are expansions applied
afterwards -- the 279-cell design needs only 21 simulations, and the anti-idling
factor needs none of its own (see ``postprocess_antiidling.py``).

Shape follows ``ALTRIOS-Parallel-Systems/applications/milestone 6.2/
test_case_sweep.py``: input CSVs cross-joined into a case table, a loop over
partitions, ``sim_manager.main`` then ``run_speed_limit_train_sims``, and
``metric_calculator`` over a fixed metric list, with results written
incrementally so a long sweep can be interrupted without losing what it has.

Two things differ from that template, both because this study starts from
measured traffic rather than a demand scenario:

- **The consist plan is an input, not an output.** Each corridor's plan comes
  from ``corridor_inputs/train_consist_plan_<slug>_clean.csv``, built by
  ``railstate_corridor.py`` in the analysis repo from RailState observations.
  The planner is handed that plan rather than a demand file.
- **The network carries the observed speed cap.** Track limits are
  ``min(posted, corridor p99 observed spot speed)``, read from
  ``corridor_inputs/corridor_meta_<slug>.csv`` via ``speed_calibration``.

``Dwell_Time`` and ``Dwell_Fuel_Energy`` are in the metric list because the
anti-idling factor is computed from them downstream. They require saved
sim history, so ``SAVE_INTERVAL`` must not be None.

Usage:
    python run_sweep.py --dry-run                   # list the simulations
    python run_sweep.py --corridor winnipeg_chicago
    python run_sweep.py --all
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import polars as pl

import altrios as alt
from altrios import history, metric_calculator, sim_manager
from altrios.train_planner import planner_config

HERE = Path(__file__).parent
OUT = HERE / "out"
LAYOUTS = OUT / "layouts"
APPLICATIONS = HERE.parent
sys.path.insert(0, str(APPLICATIONS))

from speed_calibration import (  # noqa: E402
    CORRIDORS,
    INPUTS,
    NETWORKS,
    load_rail_vehicles,
    prepare_network,
)

logger = logging.getLogger(__name__)

# Saved history is the dominant memory cost of a walked plan, and it is the
# reason Winnipeg-Chicago could not be simulated at all: at the 60 s default it
# reaches ~560 MB/train and projects to ~90 GB for 162 trains on a 64 GB machine.
#
# `HISTORY` below is the complete statement of what this study records -- every
# component, every series. Anything not named here is not saved, so a metric
# added later that needs a new series fails at configuration time with the
# series named, rather than yielding a null column several steps downstream.
# `altrios.history.configure` enforces that and the interval rules; see its
# module docstring.
#
# Five series carry the whole study:
#   train time, speed and offset  -> dwell time, stop events, meet-pass counts,
#                                    and which stops are at a terminal
#   consist cumulative fuel energy -> dwell fuel
#   locomotive cumulative aux energy -> APU fuel
# Everything else -- the twenty-odd consist power/energy series, the eight per
# locomotive, the electric drive, fuel converter, generator, battery and
# friction brake histories -- is recorded by default and read by nothing.
#
# Catenary energy is deliberately absent. `_sim_catenary_joules` reads the
# cumulative total off `cat_int`'s terminal STATE, which is both exact and
# independent of any save interval, so `cat_int` records nothing at all.
#
# One interval throughout, and that is a physics constraint rather than a
# limitation: the consist and locomotive series are attributed to INDIVIDUAL
# stops by alignment with train speed, and the anti-idling factor applies a
# 900 s shutdown threshold to the distribution of those stops. A coarser
# interval on either would stop resolving the threshold. (They would still be
# correctly aligned -- `metric_calculator` joins on the step counter `i`, which
# is never maskable -- just too coarse to be useful.)
SAVE_INTERVAL = 120

# Cumulative series, never the snapshot rates. Every quantity this study takes
# from a component is an ENERGY over an interval, and `E[i+1] - E[i]` is exact
# at any save interval because ALTRIOS accumulates it every one-second step.
# The rate form (`pwr_x_watts` times the interval) instead assumes the rate held
# across the whole 120 s, which is wrong exactly where it matters -- the sample
# where a train comes to a stand or pulls away -- and would silently carry a
# moving-power value into a stopped sample if a component were ever coarsened.
HISTORY = {
    "<root>": dict(
        interval=SAVE_INTERVAL,
        keep=["time_seconds", "speed_meters_per_second", "offset_meters"],
    ),
    # Diesel is taken at CONSIST level: per-locomotive fuel energy lives on the
    # `fc` sub-history, which records nothing, and the consist total is what the
    # study reports anyway. (The corridor TOTAL needs no history at all --
    # `get_energy_fuel` reads the fuel converter's terminal state -- so this
    # series exists only to attribute fuel to individual stops.)
    "loco_con": dict(interval=SAVE_INTERVAL, keep=["energy_fuel_joules"]),
    "loco_vec": dict(interval=SAVE_INTERVAL, keep=["energy_aux_joules"]),
}

# Warm-up and cool-down, in days either side of the measured week.
#
# A measured train has to meet realistic traffic at both ends of its window. Run
# a bare 7-day plan and the first trains enter an empty corridor and the last
# ones run into a draining one, so both edges are optimistically fast.
#
# Two days covers the median end-to-end transit on every corridor (37.1 h
# Spokane-Lethbridge, 25.1 h Winnipeg-Chicago, 13.4 h Toronto-Battle Creek) and
# the p95 on Toronto. It does NOT cover the p95 on the other two (57.6 h and
# 39.0 h), so the longest-running trains at the window edges still see a
# partially filled corridor -- a deliberate trade, since every extra day is more
# trains through meet-pass, whose cost grows faster than train count.
#
# The extra trains are dispatched but never walked (see `measure_window_hr` in
# sim_manager), so they cost conflict-resolution time and no memory.
WARMUP_DAYS = 2
COOLDOWN_DAYS = 2
HOURS_PER_DAY = 24.0
PLAN_WEEK_HOURS = 7 * HOURS_PER_DAY

# Starting tractive effort per locomotive, from the Wabtec ET44AC sheet that
# ALTRIOS itself cites for locomotive mass and horsepower. Applied to every
# powertrain so the comparison is not confounded by differing adhesion limits.
FORCE_MAX_POUNDS = 200_000.0

# The consist plan names one locomotive type for every locomotive, so the sweep
# varies the powertrain by rebinding the Rust locomotive behind that name.
PLAN_LOCO_TYPE = "Dual_Mode"

# Catenary battery-electric settings, matching SIM_BEL_CAT_PWR_MAX_WATTS and
# SIM_BEL_BATTERY_CAPACITY_KWH in catenary_optimization. The BEL pack is not a
# factor in this design -- only the hybrid's is -- so it stays at the value the
# catenary study used, keeping that arm comparable with it.
BEL_CAT_PWR_MAX_WATTS = 5.0e6
BEL_BATTERY_MWH = 9.6

# Uniform catenary power limit, matching COMMON["uniform_power_limit_watts"].
CATENARY_POWER_LIMIT_WATTS = 50.0e6

# Metadata cross-joined onto every output frame, so each row says which
# scenario produced it without needing experiments.csv alongside. Matches the
# milestone 6.2 convention of `pl.DataFrame(case_details).join(metrics, "cross")`.
SCENARIO_COLUMNS = [
    "Sim_Key", "Corridor", "Powertrain", "Label", "Battery_MWh",
    "Architecture", "Layout_Fingerprint", "System_Voltage_V",
    "Electrification_Type", "Network_Path", "Locations_Path", "Consumes_Diesel",
]

# Only names in `metric_calculator.function_mappings` are valid; Diesel_Usage
# and Electricity_Usage are not top-level metrics, they are emitted as part of
# Energy_Costs, which is why that is requested instead.
#
# Dwell_Fuel_Energy and APU_Fuel_Consumption are the two the anti-idling factor
# is built from: the first is fuel actually burned at rest, the second is what
# an APU would burn instead, from each locomotive's real auxiliary-power
# history (0.063 gal/hr per aux kW + 0.4115). Both need saved sim history.
METRICS = [
    ("Meet_Pass_Events", "count"),
    ("Dwell_Time", "seconds"),
    ("Dwell_Fuel_Energy", "MJ"),
    ("APU_Fuel_Consumption", "MJ"),
    # Energy, not Energy_Costs: the cost metric prices diesel from
    # `info.nodal_energy_prices`, which is empty on this consist-plan-driven
    # path. Costs are computed downstream instead, from one harmonized price
    # set applied to catenary and non-catenary cells alike.
    ("Energy", "MJ"),
    ("Freight_Moved", "million tonne-km"),
    ("Freight_Moved", "car-miles"),
    ("Energy_Per_Freight_Moved", "MJ per car-mile"),
    # GHG is computed downstream instead of requested here. `calculate_ghg`
    # cannot run on this code path: `metric()` defaults its Year column to the
    # STRING "All", so in the branch taken when charging was not explicitly
    # modelled the electricity frame carries Year="All" and its join to the
    # emissions factors fails on dtype. Diesel GHG is a linear function of fuel
    # use and grid GHG of electricity use, so both are computed in
    # postprocess_antiidling.py from the factors recorded below -- which is
    # also more transparent than a metric whose factors are implicit.
    ("Count_Locomotives", "assets"),
]


def set_battery_capacity(loco: alt.Locomotive, mwh: float) -> alt.Locomotive:
    """Resize a locomotive's pack, by pydict round-trip.

    There is no direct setter for ``res.energy_capacity_joules`` in this repo --
    ``alt.set_param_from_path`` exists in the parallel-systems fork but not here
    -- so the locomotive goes out to a pydict and back, the same way
    ``catenary_optimization._bel_with_cat_loco`` does it. The pack sits under
    the locomotive-type key, which differs per powertrain.
    """
    d = loco.to_pydict()
    kind = next(iter(d["loco_type"]))
    if "res" not in (d["loco_type"][kind] or {}):
        raise ValueError(f"{kind} has no battery to resize")
    d["loco_type"][kind]["res"]["energy_capacity_joules"] = mwh * 3.6e9
    return alt.Locomotive.from_pydict(d)


def build_locomotive(powertrain: str, battery_mwh: Optional[float]) -> alt.Locomotive:
    """The Rust locomotive for one powertrain level.

    Every powertrain gets the same starting tractive effort, so a difference
    between cells is a difference in the powertrain rather than in how hard the
    locomotive can pull.
    """
    if powertrain == "diesel":
        loco = alt.Locomotive.default()
    elif powertrain == "hybrid":
        loco = alt.Locomotive.default_hybrid_electric_loco()
    elif powertrain == "catenary_bel":
        # Pantograph power and pack size match the catenary analysis this arm
        # replicates, so its cells stay comparable with that study.
        loco = alt.Locomotive.default_battery_electric_loco_with_cat(
            BEL_CAT_PWR_MAX_WATTS
        )
        loco = set_battery_capacity(loco, BEL_BATTERY_MWH)
    elif powertrain == "catenary_dual":
        loco = alt.Locomotive.default_dual_mode_loco()
    else:
        raise ValueError(f"unknown powertrain {powertrain!r}")

    loco.set_force_max_pounds(FORCE_MAX_POUNDS, "UpdateMu")
    if battery_mwh is not None:
        loco = set_battery_capacity(loco, float(battery_mwh))
        loco.set_force_max_pounds(FORCE_MAX_POUNDS, "UpdateMu")
    return loco


def planner_config_for(
    powertrain: str, battery_mwh: Optional[float]
) -> planner_config.TrainPlannerConfig:
    """Planner config whose single locomotive type carries this powertrain."""
    fuel = "Electricity" if powertrain == "catenary_bel" else "Diesel"
    config = planner_config.TrainPlannerConfig(
        loco_info=pd.DataFrame(
            {
                PLAN_LOCO_TYPE: {
                    "Fuel_Type": fuel,
                    "Min_Servicing_Time_Hr": 3.0,
                    "Rust_Loco": build_locomotive(powertrain, battery_mwh),
                    "Cost_USD": alt.defaults.DIESEL_LOCO_COST_USD,
                    "Lifespan_Years": alt.defaults.LOCO_LIFESPAN,
                }
            }
        )
        .transpose()
        .reset_index(names="Locomotive_Type")
    )
    refueler_type = "Diesel_Fueler" if fuel == "Diesel" else "BEL_Charger"
    refuelers = config.refueler_info[
        config.refueler_info["Refueler_Type"] == refueler_type
    ].copy()
    if refuelers.empty:  # fall back rather than hand the planner an empty pool
        refuelers = config.refueler_info.head(1).copy()
    refuelers["Locomotive_Type"] = PLAN_LOCO_TYPE
    config.refueler_info.update(refuelers)
    return config


def network_for(row: dict) -> alt.Network:
    """Corridor network with the observed speed cap, energized if catenary.

    The cap is the single speed-limit modelling choice for the whole study:
    ``min(posted, corridor p99 observed spot speed)``. ``prepare_network`` reads
    it from the corridor metadata written alongside the consist plans.
    """
    corridor = CORRIDORS[row["Corridor"]]
    if row["Architecture"] == "na":
        return prepare_network(corridor)

    # Catenary cells run an optimized, electrically parameterized buildout built
    # by ``build_layouts.py``: substations selected by the architecture's own
    # feeding radius, then ``apply_line_losses`` stamping voltage, resistance
    # and reactance and derating each interval's power limit.
    #
    # This replaced ``energize_uniform``, which sets power limits but leaves
    # voltage at zero -- so no power flowed through any pantograph, catenary
    # energy recorded as zero, and all three architectures gave byte-identical
    # results. Loading a prebuilt layout rather than optimizing here keeps the
    # MILP out of the per-cell path, where it would rerun identically for every
    # powertrain sharing an architecture.
    fingerprint = row.get("Layout_Fingerprint") or "nobuild"
    if fingerprint in ("nobuild", "na", None):
        # The optimizer declined to electrify at this cost cell. That is a
        # result, not a gap: the corridor runs unelectrified and the catenary
        # locomotive simply never sees a wire. Simulating it is what makes the
        # cost cells where electrification does not pay comparable with the
        # ones where it does.
        logger.info("  no buildout at this cost cell -- unelectrified corridor")
        return prepare_network(corridor)

    path = LAYOUTS / f"{fingerprint}.msgpack"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing -- run `python build_layouts.py`, then "
            "`python build_experiments.py`, so the table keys on built layouts"
        )
    network = alt.Network.from_file(str(path))

    # The speed cap still applies: the layout is about electrification, the cap
    # about how fast track allows. `prepare_network` reads the corridor file, so
    # the cap is applied here directly to the layout instead.
    cap = corridor.effective_cap()
    if cap is not None:
        d = alt.utilities.set_network_max_speed(
            network.to_pydict(), max_speed_mph=cap
        )
        network = alt.Network.from_pydict(d)
        logger.info("track limit = min(posted, %.1f mph observed p99)", cap)
    return network


def extend_plan(plan: pl.DataFrame, warmup_days: float, cooldown_days: float):
    """Wrap the representative week into warm-up + measured + cool-down.

    The plan is a *representative week*, calibrated so simulated carload-km
    match observed carload-km. Warm-up traffic therefore has to look like the
    same week, not like extra demand -- inventing new trains would change the
    density the whole study is calibrated to.

    So the week wraps around itself: its final ``warmup_days`` become the days
    before the window, and its first ``cooldown_days`` become the days after.
    Day-of-week structure survives, because the traffic preceding Monday really
    is that corridor's Friday, and the departure profile that drives meet-pass
    conflicts is preserved rather than smeared.

    Returns ``(extended_plan, (window_start_hr, window_end_hr))``.
    """
    if warmup_days <= 0 and cooldown_days <= 0:
        return plan, None

    warm_h = warmup_days * HOURS_PER_DAY
    cool_h = cooldown_days * HOURS_PER_DAY
    dep = pl.col("Departure_Time_Planned_Hr")
    time_cols = [c for c in ("Departure_Time_Planned_Hr", "Arrival_Time_Planned_Hr")
                 if c in plan.columns]

    def shifted(frame: pl.DataFrame, delta: float, id_offset: int) -> pl.DataFrame:
        return frame.with_columns(
            [pl.col(c) + pl.lit(delta) for c in time_cols]
            + [pl.col("Train_ID") + pl.lit(id_offset)]
        )

    span = plan["Train_ID"].max() + 1
    blocks = []
    if warm_h > 0:
        # The tail of the week, moved to sit immediately before it.
        blocks.append(
            shifted(plan.filter(dep >= PLAN_WEEK_HOURS - warm_h), -PLAN_WEEK_HOURS, 0)
        )
    blocks.append(shifted(plan, 0.0, span))
    if cool_h > 0:
        # The head of the week, moved to sit immediately after it.
        blocks.append(shifted(plan.filter(dep < cool_h), PLAN_WEEK_HOURS, 2 * span))

    extended = pl.concat(blocks, how="diagonal_relaxed")
    # Rebase so the earliest departure is at hour 0; the planner and the trim
    # both read absolute hours.
    floor = extended["Departure_Time_Planned_Hr"].min()
    extended = extended.with_columns(
        [pl.col(c) - pl.lit(floor) for c in time_cols]
    )
    return extended, (warm_h, warm_h + PLAN_WEEK_HOURS)


def trim_history(sim):
    """Record only what `HISTORY` names.

    Applied by `sim_manager` after `set_save_interval` and before the walk --
    the only point it can be: `set_save_interval` recurses into every
    sub-component, so anything configured earlier is overwritten by it, and once
    the walk starts the history exists. At this point history is still empty,
    so the pydict round-trip is cheap.
    """
    return history.configure(sim, HISTORY)


def _stop_row(sim_key, train_idx, duration_s, energy_j, terminal):
    return {
        "Sim_Key": sim_key,
        "train_idx": train_idx,
        "duration_seconds": duration_s,
        "fuel_MJ": energy_j / 1e6,
        "terminal": terminal,
    }


def stop_events(sims, sim_key: str) -> pl.DataFrame:
    """Every stop a train made, with its duration and the fuel burned during it.

    ``metric_calculator`` reports dwell time and dwell fuel as corridor totals,
    which is not enough for the anti-idling factor: a shutdown threshold acts on
    *individual* stops, so a 20-minute hold and forty 30-second meets adding to
    the same total are treated completely differently. Applying a threshold to
    an aggregate would silently assume one of those shapes. This walks the same
    saved history metric_calculator uses and emits one row per stop, so the
    threshold can be applied to the real distribution downstream.
    """
    rows = []
    for train_idx, slts in enumerate(sims.to_pydict()):
        history = slts["history"]
        time_s = history.get("time_seconds", [])
        speed = history.get("speed_meters_per_second", [])
        # Cumulative fuel energy, differenced -- not the snapshot rate times the
        # interval. `E[i+1] - E[i]` is exact at any save interval because
        # ALTRIOS accumulates it every one-second step, where the rate form
        # assumes the rate held across the whole 120 s. Matches
        # `metric_calculator._interval_delta`, which is what keeps these rows
        # summing to `Dwell_Fuel_Energy` exactly.
        fuel_j = slts["loco_con"]["history"].get("energy_fuel_joules", [])
        if not time_s or not speed:
            continue
        # `terminal` separates standing at the origin or destination from
        # stopping en route. A train the dispatcher holds stands at its origin
        # from its PLANNED departure -- at zero speed, so indistinguishable
        # from a meet by speed alone -- and one that arrives stands at its
        # destination to the end of its history. Neither is congestion, and an
        # anti-idling policy would treat a locomotive parked at a terminal
        # differently from one holding the main line. `_en_route_flags` uses
        # the monotone `offset_meters`; all-True when offset was not recorded.
        flags = metric_calculator._en_route_flags(history)
        if flags is None:
            flags = [True] * len(time_s)
        # A stop is a maximal run of samples at zero speed. Compare against a
        # tolerance: the solver snaps to the target but can leave a residual.
        #
        # A run also ENDS when the terminal flag changes, not only when the
        # train moves. At a 120 s sampling interval a train can leave its
        # origin and stop again between two samples, both of which read zero
        # speed -- so one run of zero-speed samples can straddle the moment the
        # train left the terminal. Splitting there keeps every emitted stop
        # wholly terminal or wholly en route, which is what makes these rows
        # sum to `metric_calculator`'s per-sample `All` and `En_Route` subsets
        # exactly rather than approximately.
        start = None
        start_idx = 0
        energy_j = 0.0
        for i in range(len(time_s) - 1):
            if speed[i] <= 1e-6:
                if start is not None and flags[i] != flags[start_idx]:
                    rows.append(_stop_row(
                        sim_key, train_idx, time_s[i] - start, energy_j,
                        not flags[start_idx],
                    ))
                    start = None
                if start is None:
                    start, start_idx, energy_j = time_s[i], i, 0.0
                # Only intervals the train was stopped for at BOTH ends carry
                # idle fuel; the interval it pulls away in contains the whole
                # acceleration. Matches `stopped_interval` in metric_calculator.
                if i + 1 < len(fuel_j) and speed[i + 1] <= 1e-6:
                    energy_j += max(0.0, fuel_j[i + 1] - fuel_j[i])
            elif start is not None:
                rows.append(_stop_row(
                    sim_key, train_idx, time_s[i] - start, energy_j,
                    not flags[start_idx],
                ))
                start = None
        if start is not None:
            rows.append(_stop_row(
                sim_key, train_idx, time_s[-1] - start, energy_j,
                not flags[start_idx],
            ))
    schema = {"Sim_Key": pl.Utf8, "train_idx": pl.Int64,
              "duration_seconds": pl.Float64, "fuel_MJ": pl.Float64,
              "terminal": pl.Boolean}
    return pl.DataFrame(rows, schema=schema)


def scenario_frame(row: dict) -> pl.DataFrame:
    """One-row frame of this scenario's metadata, for cross-joining onto output."""
    return pl.DataFrame(
        {c: [row.get(c)] for c in SCENARIO_COLUMNS if c in row}
    )


def run_simulation(row: dict, plan: pl.DataFrame):
    """One simulation cell: returns ``(metrics, stop_events, consist_plan)``."""
    corridor = CORRIDORS[row["Corridor"]]
    config = planner_config_for(row["Powertrain"], row["Battery_MWh"])
    network = network_for(row)
    location_map = alt.import_locations(str(NETWORKS / corridor.locations))

    # Warm-up and cool-down traffic, so the measured week is not bounded by an
    # empty corridor at either end. Supplying a consist plan bypasses
    # sim_manager's own warm-start machinery -- it is gated on
    # `consist_plan_in is None` -- so the window is built here and passed
    # explicitly.
    plan, window = extend_plan(plan, WARMUP_DAYS, COOLDOWN_DAYS)

    started = time.perf_counter()
    (
        train_consist_plan,
        loco_pool,
        refuel_facilities,
        grid_emissions_factors,
        nodal_energy_prices,
        sims,
        _timed_paths,
        _untrimmed,
    ) = sim_manager.main(
        network=network,
        rail_vehicles=load_rail_vehicles(),
        location_map=location_map,
        consist_plan_in=plan,
        train_planner_config=config,
        # Walk inside sim_manager rather than calling run_speed_limit_train_sims
        # afterwards, as the parallel-systems template does. This repo's
        # sim_manager converts its timed paths to pydicts before returning them
        # (`tp.to_pydict()`), so they can no longer be walked from outside --
        # a second walk fails in the Rust extractor. The internal walk also sets
        # save_interval(60) first, which is what makes the dwell and stop-event
        # history available downstream.
        walk_timed_paths=True,
        save_interval=SAVE_INTERVAL,
        configure_sim=trim_history,
        measure_window_hr=window,
        debug=False,
    )

    # Consequence of walking internally: no refuel sessions are produced.
    # An EMPTY frame rather than None, deliberately: `calculate_ghg` contains
    #
    #     elif (info.refuel_sessions is None) and (
    #         info.emissions_factors.select(pl.col("Node").n_unique() > 0))
    #
    # and `and` on a DataFrame raises "truth value of a DataFrame is ambiguous".
    # It only fires when a cell uses electricity -- so diesel passes and every
    # hybrid and catenary cell would crash. Passing a non-None frame makes the
    # left operand False and Python short-circuits before touching the
    # DataFrame. Electricity GHG is then computed without spatial resolution,
    # which is exactly right when the factors cover one region and an
    # approximation if they cover several; the node count is logged below so
    # that is visible rather than assumed.
    # Typed, not bare: consumers filter it on its columns (calculate_electricity_use
    # does `filter(pl.col("Fuel_Type") == "Electricity")`), and a schema-less
    # empty frame fails that lookup. Schema per the ScenarioInfo docstring.
    refuel_sessions = pl.DataFrame(
        schema={
            "Locomotive_ID": pl.UInt32,
            "Locomotive_Type": pl.Utf8,
            "Fuel_Type": pl.Utf8,
            "Node": pl.Utf8,
            "Refuel_Energy_J": pl.Float64,
        }
    )

    # The metric frames stamp Year from `scenario_year` as an integer, but the
    # emissions factors arrive with Year as a string, so the electricity-GHG
    # join fails on dtype. Only cells that actually use electricity reach that
    # join, which is why diesel runs and every hybrid and catenary cell does not.
    if (
        grid_emissions_factors is not None
        and "Year" in grid_emissions_factors.columns
        and grid_emissions_factors.schema["Year"] != pl.Int64
    ):
        grid_emissions_factors = grid_emissions_factors.with_columns(
            pl.col("Year").cast(pl.Int64, strict=False)
        )

    if grid_emissions_factors is not None and "Node" in grid_emissions_factors.columns:
        nodes = grid_emissions_factors["Node"].n_unique()
        if nodes > 1:
            logger.warning(
                "  emissions factors span %d nodes; electricity GHG is not "
                "spatially resolved in this run", nodes,
            )

    info = metric_calculator.ScenarioInfo(
        sims=sims,
        simulation_days=7,
        annualize=False,
        scenario_year=alt.defaults.BASE_ANALYSIS_YEAR,
        loco_pool=loco_pool,
        consist_plan=train_consist_plan,
        refuel_facilities=refuel_facilities,
        refuel_sessions=refuel_sessions,
        emissions_factors=grid_emissions_factors,
        nodal_energy_prices=nodal_energy_prices,
        train_planner_config=config,
    )
    metrics = metric_calculator.main(
        info, annual_metrics=METRICS, calculate_multiyear_metrics=False
    )
    # Catenary energy, emitted here because `metric_calculator` computes it but
    # never reports it. `calculate_energy` uses the pantograph figure only to
    # SPLIT its electricity total into line-loss and stationary-charging parts;
    # it never ADDS it. On this code path the electricity total is net energy at
    # the battery -- near zero for a locomotive fed continuously from the wire --
    # so a fully electrified corridor reports zero electricity consumed while
    # correctly reporting its resistive losses. `_sim_catenary_joules_for` has
    # the number; this puts it where the cost and GHG arithmetic can see it.
    pantograph_j, line_loss_j = metric_calculator._sim_catenary_joules_for(info)
    catenary_rows = [
        ("Catenary_Pantograph_Energy", "MJ", pantograph_j / 1e6),
        ("Catenary_Line_Loss", "MJ", line_loss_j / 1e6),
        # What the substation must supply: delivered plus resistive loss. This
        # is the quantity to price and to charge grid emissions against.
        ("Catenary_Grid_Draw", "MJ", (pantograph_j + line_loss_j) / 1e6),
    ]
    metrics = pl.concat(
        [
            metrics,
            pl.DataFrame({
                "Metric": [r[0] for r in catenary_rows],
                "Units": [r[1] for r in catenary_rows],
                "Value": [r[2] for r in catenary_rows],
                "Subset": ["All"] * len(catenary_rows),
                "Year": ["All"] * len(catenary_rows),
            }),
        ],
        how="diagonal_relaxed",
    )

    # Carry the grid emissions factor with the results so downstream GHG is
    # computed from the factors this run actually used, not a re-lookup.
    grid_kg_per_mwh = None
    if (
        grid_emissions_factors is not None
        and "CO2eq_kg_per_MWh" in grid_emissions_factors.columns
        and grid_emissions_factors.height
    ):
        grid_kg_per_mwh = float(grid_emissions_factors["CO2eq_kg_per_MWh"].mean())
    metrics = pl.concat(
        [
            metrics,
            pl.DataFrame({
                "Metric": ["Grid_Emissions_Factor"],
                "Units": ["kg CO2-eq per MWh"],
                "Value": [grid_kg_per_mwh],
                "Subset": ["All"],
                "Year": ["All"],
            }),
        ],
        how="diagonal_relaxed",
    )
    logger.info(
        "  %s done in %.1f s (%d trains)",
        row["Sim_Key"], time.perf_counter() - started,
        train_consist_plan["Train_ID"].n_unique(),
    )
    meta = scenario_frame(row)
    events = stop_events(sims, row["Sim_Key"])
    return (
        meta.join(metrics, how="cross"),
        meta.join(events.drop("Sim_Key"), how="cross") if events.height else events,
        meta.join(train_consist_plan, how="cross"),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--corridor", choices=sorted(CORRIDORS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--powertrain")
    parser.add_argument(
        "--experiments", type=Path, default=HERE / "experiments.csv"
    )
    parser.add_argument("--out", type=Path, default=OUT / "sim_metrics.csv")
    parser.add_argument("--stops-out", type=Path, default=OUT / "stop_events.csv")
    parser.add_argument("--plans-out", type=Path, default=OUT / "consist_plans.csv")
    parser.add_argument(
        "--dry-run", action="store_true", help="list the simulations, run none"
    )
    parser.add_argument(
        "--max-trains",
        type=int,
        help="cap trains per simulation. For smoke-testing the pipeline only -- "
        "results are not comparable across corridors, since the traffic level "
        "is the calibrated quantity",
    )
    parser.add_argument(
        "--limit", type=int, help="run only the first N simulations"
    )
    parser.add_argument(
        "--sim-key",
        help="run exactly this one simulation. `run_all.sh` uses this to give "
        "each simulation its own process: ALTRIOS holds the est-times DAG and "
        "walked histories in Rust memory, and running several corridors in one "
        "process accumulates enough of it to fail a 1 GB allocation mid-sweep "
        "-- which aborts rather than raising, so the loop dies with it",
    )
    parser.add_argument(
        "--per-sim-dir",
        type=Path,
        default=OUT / "sims",
        help="write one set of files per simulation here, so a failed cell "
        "costs only itself and a rerun can skip what already succeeded",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip simulations whose per-sim output already exists",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cells = pl.read_csv(args.experiments)
    if args.corridor:
        cells = cells.filter(pl.col("Corridor") == args.corridor)
    if args.powertrain:
        cells = cells.filter(pl.col("Powertrain") == args.powertrain)
    if not args.corridor and not args.all:
        parser.error("pass --corridor or --all")

    # One row per distinct simulation; the rest of the design rides on top.
    # Ordered so the diesel baselines run first, then hybrids by battery size,
    # then the catenary cells: a sweep that runs out of time still holds the
    # comparisons the study turns on rather than a third of one corridor.
    order = {"diesel": 0, "hybrid": 1, "catenary_dual": 2, "catenary_bel": 3}
    sims = (
        cells.unique(subset=["Sim_Key"], keep="first")
        .with_columns(
            pl.col("Powertrain").replace_strict(order, default=9).alias("_order")
        )
        .sort("_order", "Battery_MWh", "Corridor", "Architecture", nulls_last=False)
        .drop("_order")
    )
    logger.info(
        "%d analysis cells -> %d simulations", cells.height, sims.height
    )
    for row in sims.iter_rows(named=True):
        logger.info(
            "  %-52s %s battery=%s arch=%s",
            row["Sim_Key"], row["Corridor"],
            row["Battery_MWh"], row["Architecture"],
        )
    if args.dry_run:
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.sim_key:
        sims = sims.filter(pl.col("Sim_Key") == args.sim_key)
        if sims.height == 0:
            parser.error(f"no simulation with Sim_Key {args.sim_key!r}")
    if args.limit:
        sims = sims.head(args.limit)
    args.per_sim_dir.mkdir(parents=True, exist_ok=True)
    results: List[pl.DataFrame] = []
    stops: List[pl.DataFrame] = []
    out_plans: List[pl.DataFrame] = []
    failures: Dict[str, str] = {}
    plans: Dict[str, pl.DataFrame] = {}

    for row in sims.iter_rows(named=True):
        slug = row["Corridor"]
        if slug not in plans:
            plan = pl.read_csv(CORRIDORS[slug].plan_path)
            if args.max_trains:
                keep = plan["Train_ID"].unique().sort()[: args.max_trains]
                plan = plan.filter(pl.col("Train_ID").is_in(keep))
                logger.warning(
                    "SMOKE TEST: %s capped to %d trains -- results are not the "
                    "calibrated traffic level", slug, plan["Train_ID"].n_unique()
                )
            logger.info(
                "%s plan: %d trains, %d locomotive rows, from %s",
                slug, plan["Train_ID"].n_unique(), plan.height,
                CORRIDORS[slug].plan_path.name,
            )
            plans[slug] = plan
        safe = row["Sim_Key"].replace("|", "__").replace(".", "p")
        metrics_path = args.per_sim_dir / f"{safe}_metrics.csv"
        if args.skip_existing and metrics_path.exists():
            logger.info("skipping %s (already done)", row["Sim_Key"])
            continue
        logger.info("=== %s ===", row["Sim_Key"])
        try:
            metrics, events, consist_plan = run_simulation(row, plans[slug])
        except Exception as exc:  # noqa: BLE001 - one cell failing is not fatal
            logger.exception("  FAILED: %s", str(exc)[:200])
            failures[row["Sim_Key"]] = str(exc)[:200]
            continue
        metrics.write_csv(metrics_path)
        if events.height:
            events.write_csv(args.per_sim_dir / f"{safe}_stops.csv")
        consist_plan.write_csv(args.per_sim_dir / f"{safe}_plan.csv")
        results.append(metrics)
        if events.height:
            stops.append(events)
        out_plans.append(consist_plan)
        # Written every cell so an interrupted sweep keeps what it finished.
        pl.concat(results, how="diagonal_relaxed").write_csv(args.out)
        pl.concat(out_plans, how="diagonal_relaxed").write_csv(args.plans_out)
        if stops:
            pl.concat(stops, how="diagonal_relaxed").write_csv(args.stops_out)

    if results:
        logger.info("wrote %s (%d simulations)", args.out, len(results))
        logger.info("wrote %s", args.plans_out)
        if stops:
            logger.info("wrote %s", args.stops_out)
    if failures:
        logger.error("%d simulations failed:", len(failures))
        for key, message in failures.items():
            logger.error("  %s: %s", key, message)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
