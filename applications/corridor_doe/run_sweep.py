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

import hashlib
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import json
import polars as pl

import altrios as alt
from altrios import history, metric_calculator, sim_manager
from altrios.train_planner import data_prep, planner_config

import catenary_policy  # noqa: E402  -- receptive wire flag + export credit
import hybrid_controls_io  # noqa: E402  -- optional tuned hybrid controller
import terminal_charging  # noqa: E402  -- per-locomotive terminal charging sessions

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
# Helper assignments are cached: the test costs a free-running `make_est_times`
# per train and the answer only changes when the plan or the layout does.
HELPERS_DIR = Path(__file__).parent / "out" / "helper_assignments"
PLAN_LOCO_TYPE = "Dual_Mode"
# The battery-electric arm needs a second type in the pool. A pure-BEL consist
# cannot cross this corridor on the heaviest trains -- 19 of Spokane-Lethbridge's
# 52 carry more than ~4,400 tons per locomotive and run the pack to its 5% floor
# on the loaded direction -- and the catenary study's answer to that is not more
# wire but ONE conventional unit swapped into the consist. The wire is sized from
# the fleet-MEAN drain by design (`BEL_AVG_KWH_PER_MILE`, documented as "24 light
# .. 62 heavy"); the heavy tail is meant to be carried by helpers. Sizing the
# wire for the tail instead would over-build every layout to solve a problem the
# architecture already solves.
HELPER_LOCO_TYPE = "Diesel_Helper"
# State of charge a train must still hold at the end of the FREE-RUNNING
# feasibility test to be accepted without a helper. Merely finishing is not
# enough: the test cannot see dispatch, and a train that free-runs to the 5%
# floor with nothing to spare hits it for real once meets add stopped time and
# auxiliary load. Two layouts passed the bare test and then stranded trains
# mid-walk at exactly 5.0%, which is what this reserve exists to prevent.
HELPER_ESCALATION_FRACTIONS = (0.0, 0.25, 0.5, 1.0)
# Ceiling on how much of a battery-electric fleet may be converted to diesel
# before the cell stops being a battery-electric result at all. Without it,
# escalation "succeeds" by turning the arm into a diesel operation: on the
# dc_9kv cost cell where the optimizer declined to electrify, it swapped 71
# units across 48 of 52 trains and reported 22.9 million MJ of diesel, ZERO grid
# draw, as an `ok` catenary_bel row. An electrification a fleet cannot run is a
# RESULT -- it belongs in the analysis as infeasible, not as diesel wearing a
# battery-electric label. The deep-dive's own escalation swaps one unit of
# three, so a third of the fleet is a generous bound.
HELPER_MAX_SHARE = 1.0 / 3.0
# How escalation widens when a cell still strands a train. The base pass gives
# each train the FEWEST helpers that let it cross free-running, which is the
# deep-dive's rule. That test cannot see dispatch, though -- meets add stopped
# time and auxiliary load no single-train run produces -- so a cell that still
# fails is retried with one extra helper given to this fraction of trains,
# lowest finishing state of charge first.
#
# Escalating by a state-of-charge THRESHOLD instead does not work, and failed
# here before this replaced it: a helper REPLACES a battery-electric unit rather
# than adding to the consist, so more helpers can leave the remaining packs
# working harder and finishing LOWER. Demanding a fixed reserve was therefore
# non-monotone -- on the dc_9kv layout it went from 14 helper units at 0.20 to
# ZERO at 0.35, because no helper count met the threshold and the loop fell
# through to "none" on 157 trains. Ranking by margin and adding one unit at a
# time is monotone by construction.

# Catenary battery-electric settings, matching SIM_BEL_CAT_PWR_MAX_WATTS and
# SIM_BEL_BATTERY_CAPACITY_KWH in catenary_optimization. The BEL pack is not a
# factor in this design -- only the hybrid's is -- so it stays at the value the
# catenary study used, keeping that arm comparable with it.
BEL_CAT_PWR_MAX_WATTS = 5.0e6
BEL_BATTERY_MWH = 9.6
# Charge acceptance of the battery-electric pack, as a multiple of capacity.
#
# ALTRIOS rates a pack with ONE number for both directions, and on this
# locomotive that number (3.281 MW) is the TRACTION rating -- 4,400 hp, the
# FLXdrive-class figure this arm models. Reusing it as a charge limit is not a
# battery property: it is a drivetrain specification doing a second job, and it
# caps en-route intake under catenary below what any cell chemistry requires.
#
# 1C is the conventional continuous-charge ceiling, the threshold above which
# the literature calls charging "fast" and lithium plating at the anode becomes
# the binding concern. It holds for both LFP and NMC, so it does not presume a
# chemistry, and it is conservative against the ~3C that lithium-titanium-oxide
# battery trains (Stadler FLIRT Akku / Class 756) sustain from the wire. It is
# deliberately BELOW the 2C this study assumes for hybrid packs, which are sized
# for power where this one is sized for range.
#
# On the 9.6 MWh pack this is 9.6 MW, above the 5 MW pantograph, so current
# collection remains the effective limit; the constraint would bind for packs
# under 5 MWh. Applied via `res.pwr_in_max`, which leaves discharge untouched.
BEL_CHARGE_C_RATE = 1.0


def set_battery_charge_from_c_rate(loco: alt.Locomotive, c_rate: float) -> alt.Locomotive:
    """Set the pack's CHARGE acceptance to ``c_rate`` times capacity.

    Discharge is untouched: it stays on ``res.pwr_out_max``, which carries the
    locomotive's traction rating.
    """
    d = loco.to_pydict()
    kind = next(iter(d["loco_type"]))
    body = d["loco_type"][kind] or {}
    if "res" not in body:
        raise ValueError(f"{kind} has no battery to rate")
    capacity_wh = float(body["res"]["energy_capacity_joules"]) / 3600.0
    body["res"]["pwr_in_max_watts"] = c_rate * capacity_wh
    return alt.Locomotive.from_pydict(d)

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
    # The planner's refuelers (diesel fuelers and battery-electric chargers).
    ("Count_Refuelers", "assets"),
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


def with_fuel_converter_of(loco: alt.Locomotive, donor: alt.Locomotive) -> alt.Locomotive:
    """``loco`` with ``donor``'s fuel converter (engine) in place of its own."""
    d = loco.to_pydict()
    kind = next(iter(d["loco_type"]))
    dd = donor.to_pydict()
    donor_kind = next(iter(dd["loco_type"]))
    if "fc" not in (d["loco_type"][kind] or {}) or "fc" not in (dd["loco_type"][donor_kind] or {}):
        raise ValueError(f"{kind} or {donor_kind} has no fuel converter")
    d["loco_type"][kind]["fc"] = dd["loco_type"][donor_kind]["fc"]
    return alt.Locomotive.from_pydict(d)


def _strip_history(obj):
    """A pydict without its (empty, but schema-bearing) history blocks."""
    if isinstance(obj, dict):
        return {k: _strip_history(v) for k, v in obj.items() if k != "history"}
    if isinstance(obj, list):
        return [_strip_history(v) for v in obj]
    return obj


# Hybrid pack power rating as a multiple of capacity (per hour). The library
# default is 1.09 MW on a 0.3 MWh pack -- 3.6C, an HEV-style buffer -- and
# resizing capacity alone left power fixed, so the DoE's four pack sizes were
# silently four different batteries (4.4C at 0.25 MWh down to 0.55C at 2 MWh)
# and a bigger pack lasted longer down a grade without absorbing any faster.
# A fixed C-rate makes the capacity factor mean what it says.
#
# 2C is the top of what mainstream LFP/NMC cells sustain continuously (LTO goes
# far higher at a price the ATB $/kWh basis does not carry), above the ~1.4C of
# the first FLXdrive (a range-sized BEL pack, the wrong reference for a hybrid)
# and in line with hybrid-vehicle practice for packs holding minutes of prime-
# mover output. On these corridors it lets a 1 MWh pack take a median braking
# event whole (2.5 MW per consist on Spokane; 0.7-3.8 MW per unit); 1C cannot,
# and 3C+ runs into the pack-full limit and the 5 MW drivetrain cap. Not a DoE
# factor by design -- one value, stated. This is the DISCHARGE rating; charge
# acceptance is HYBRID_CHARGE_C_RATE below.
HYBRID_PACK_C_RATE = 2.0

# Hybrid CHARGE acceptance (regen and engine charging), set apart from discharge
# on 2026-10-18. A 0.25-2 MWh pack beside a 3.36 MW engine is a power-sized
# buffer, so power-optimized cells are the coherent specification, and those
# sustain 3C+ charge; braking events here last minutes, not hours. The single-
# train brake diagnostic on loaded Spokane showed 2C refusing ~16 GJ of braking
# per trip that 3C would mostly take. 3C rather than 4C keeps the claim inside
# mainstream power-cell capability and the model's own efficiency map (0.94 at
# 3C). Discharge stays at 2C: the 5 MW drive caps it and it rarely binds. Applied
# via `res.pwr_in_max`. One stated value, not a DoE factor.
HYBRID_CHARGE_C_RATE = 3.0


def set_battery_power_from_c_rate(loco: alt.Locomotive, c_rate: float) -> alt.Locomotive:
    """Rate the pack at ``c_rate`` times its capacity, capped at the electric
    drive's rating (power beyond it is unusable for traction or regen)."""
    d = loco.to_pydict()
    kind = next(iter(d["loco_type"]))
    body = d["loco_type"][kind] or {}
    if "res" not in body:
        raise ValueError(f"{kind} has no battery to rate")
    capacity_wh = float(body["res"]["energy_capacity_joules"]) / 3600.0
    pwr = c_rate * capacity_wh
    edrv_max = float(body["edrv"]["pwr_out_max_watts"])
    body["res"]["pwr_out_max_watts"] = min(pwr, edrv_max)
    return alt.Locomotive.from_pydict(d)


_CONTROLS_CACHE: Dict[str, object] = {}


def hybrid_controls():
    """Tuned hybrid controller settings, or None. Read once per process."""
    if "v" not in _CONTROLS_CACHE:
        _CONTROLS_CACHE["v"] = hybrid_controls_io.load()
    return _CONTROLS_CACHE["v"]


def build_locomotive(powertrain: str, battery_mwh: Optional[float],
                     controls: Optional[Dict[str, float]] = None) -> alt.Locomotive:
    """The Rust locomotive for one powertrain level.

    Every powertrain gets the same starting tractive effort, so a difference
    between cells is a difference in the powertrain rather than in how hard the
    locomotive can pull.
    """
    if powertrain == "diesel":
        loco = alt.Locomotive.default()
    elif powertrain == "hybrid":
        # The library's default hybrid carries HALF the conventional unit's
        # engine (1,678 vs 3,356 kW) with a ~1.1 MW battery beside it, so at
        # equal consist size a hybrid train had half the sustained power once
        # the pack hit its floor: 2-9 h slower at the median, and slow enough
        # to double meet-pass delay on the busy corridors. This study's hybrid
        # is the diesel-electric being modelled PLUS a battery, so it takes
        # that locomotive's fuel converter whole.
        loco = with_fuel_converter_of(
            alt.Locomotive.default_hybrid_electric_loco(), alt.Locomotive.default())
    elif powertrain == "catenary_bel":
        # Pantograph power and pack size match the catenary analysis this arm
        # replicates, so its cells stay comparable with that study.
        loco = alt.Locomotive.default_battery_electric_loco_with_cat(
            BEL_CAT_PWR_MAX_WATTS
        )
        loco = set_battery_capacity(loco, BEL_BATTERY_MWH)
        loco = set_battery_charge_from_c_rate(loco, BEL_CHARGE_C_RATE)
    elif powertrain == "catenary_dual":
        loco = alt.Locomotive.default_dual_mode_loco()
    else:
        raise ValueError(f"unknown powertrain {powertrain!r}")

    # Whether the wire takes braking energy back (policy, not per-cell factor).
    loco = catenary_policy.apply_to_locomotive(loco)
    loco.set_force_max_pounds(FORCE_MAX_POUNDS, "UpdateMu")
    if battery_mwh is not None:
        loco = set_battery_capacity(loco, float(battery_mwh))
        loco = set_battery_power_from_c_rate(loco, HYBRID_PACK_C_RATE)
        if powertrain == "hybrid":
            loco = set_battery_charge_from_c_rate(loco, HYBRID_CHARGE_C_RATE)
        loco.set_force_max_pounds(FORCE_MAX_POUNDS, "UpdateMu")
    # Tuned controller last, so it overrides the library defaults on the fully
    # sized locomotive. A no-op for every non-hybrid and when no tuned settings
    # exist, which is why no caller has to branch on it.
    loco = hybrid_controls_io.apply_to_locomotive(loco, controls or {})
    return loco


# Refuelers per terminal while planning: high enough that the planner's queue
# never binds (charger sizing happens afterwards, from the walked sessions).
UNCONSTRAINED_PORTS_PER_NODE = 500


def planner_config_for(
    powertrain: str, battery_mwh: Optional[float],
    corridor: Optional[str] = None, scope: Optional[str] = None,
) -> planner_config.TrainPlannerConfig:
    """Planner config whose locomotive types cover this powertrain.

    One type for every arm but battery-electric, which also carries a
    conventional `HELPER_LOCO_TYPE`. The pool is keyed by `Locomotive_Type`, and
    a consist plan may name either, so a mixed consist needs nothing more than
    the second row here plus the right value in the plan.
    """
    # Hybrids are declared "Electricity" so the planner HOLDS them at a charger
    # before departure: terminal charging is the binding hold (hours at charger
    # rate) where diesel refuelling is minutes at 300 gal/min, so modelling the
    # charge alone and leaving diesel implicit gets the dwell right. It also
    # routes hybrid refuel sessions into `calculate_electricity_use`, which
    # filters on Fuel_Type == "Electricity" -- without this their charging energy
    # is invisible to the metrics and priced at zero.
    #
    # Trade-off accepted: `require_diesel` matches Fuel_Type on "(?i)diesel", so
    # a hybrid no longer satisfies it. That flag is False by default and unused
    # in this DoE. If it is ever enabled here, match on `Has_Battery` or widen
    # the regex instead of reverting this.
    fuel = ("Electricity" if powertrain in ("catenary_bel", "hybrid") else "Diesel")
    # Tuned hybrid controller, if one has been written by the controls sweep.
    # `corridor` is optional so the diagnostic scripts that call this with two
    # arguments keep working; without it the library defaults stand.
    controls = {}
    if powertrain == "hybrid" and corridor:
        controls = hybrid_controls_io.values_for(
            hybrid_controls(), corridor, scope or "fwd")
        if controls:
            logger.info("  tuned hybrid controller (%s/%s): %s",
                        corridor, scope or "fwd",
                        {k: round(v, 4) for k, v in controls.items()})
    types = {
        PLAN_LOCO_TYPE: {
            "Fuel_Type": fuel,
            "Min_Servicing_Time_Hr": 3.0,
            "Rust_Loco": build_locomotive(powertrain, battery_mwh, controls),
            "Cost_USD": alt.defaults.DIESEL_LOCO_COST_USD,
            "Lifespan_Years": alt.defaults.LOCO_LIFESPAN,
        }
    }
    if powertrain == "catenary_bel":
        types[HELPER_LOCO_TYPE] = {
            "Fuel_Type": "Diesel",
            "Min_Servicing_Time_Hr": 3.0,
            "Rust_Loco": build_locomotive("diesel", None),
            "Cost_USD": alt.defaults.DIESEL_LOCO_COST_USD,
            "Lifespan_Years": alt.defaults.LOCO_LIFESPAN,
        }
    config = planner_config.TrainPlannerConfig(
        loco_info=pd.DataFrame(types).transpose().reset_index(
            names="Locomotive_Type")
    )
    # One refueler row per LOCOMOTIVE TYPE in the pool. `build_refuelers` left-
    # joins `refueler_info` on (Locomotive_Type, Fuel_Type), so a type with no
    # row joins to null, and the null lands in a Series that polars then tries to
    # cast to Categorical -- "cannot cast 'Object' type", raised two modules away
    # from the actual omission. Assigned rather than `update`d: `update` aligns
    # on the DataFrame index, so a row count that differs from the default two
    # silently keeps stale defaults.
    wanted = [(PLAN_LOCO_TYPE, fuel)]
    if HELPER_LOCO_TYPE in set(types):
        wanted.append((HELPER_LOCO_TYPE, "Diesel"))
    rows = []
    for loco_type, fuel_type in wanted:
        template = config.refueler_info[
            config.refueler_info["Fuel_Type"] == fuel_type
        ]
        if template.empty:  # fall back rather than hand the planner nothing
            template = config.refueler_info.head(1)
        row = template.head(1).copy()
        row["Locomotive_Type"] = loco_type
        row["Fuel_Type"] = fuel_type
        rows.append(row)
    config.refueler_info = pd.concat(rows, ignore_index=True)
    # Locomotives are REUSED from train to train: the plan fixes each train's
    # locomotive count and type, and the planner draws units that have arrived,
    # charged/refuelled and been serviced (see `TrainPlannerConfig.
    # reuse_locomotives`). Repositioning between terminals is implicit and not
    # simulated. The imported plans give every locomotive-trip its own ID, so
    # without this the fleet was one unit per locomotive-trip (6-12x too big)
    # and charger queues could never delay anything.
    config.reuse_locomotives = True
    # Ports are unconstrained while planning so no unit waits for a charger;
    # chargers are then SIZED to the peak concurrent charging sessions the walk
    # produces (`terminal_charging.sized_chargers`).
    config.refuelers_per_incoming_corridor = UNCONSTRAINED_PORTS_PER_NODE
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


def span_travel_hours(sims, corridor) -> Optional[float]:
    """Median simulated time across the corridor's WIDEST observed sensor pair.

    The counterpart to `observed_span_median_hr` in `corridor_meta`: the same
    two points on the same corridor, so the difference between them is a
    like-for-like measure of stationary time the simulation does not reproduce.
    Measured on the span and never scaled to the full corridor -- the
    unobserved remainder is precisely where there is no basis for assuming the
    observed behaviour continues, and on Spokane-Lethbridge it demonstrably
    does not, since the queue forms outside the sensor envelope.

    Each train's crossing time is interpolated from `offset_meters`, which the
    history spec already keeps. Reverse-direction trains are measured between
    the mirrored offsets, so both directions cover the same physical track.
    """
    try:
        sensors = pl.read_csv(corridor.sensors_path).sort("corridor_km")
    except Exception:
        return None
    if sensors.height < 2:
        return None
    lo_km = float(sensors["corridor_km"][0])
    hi_km = float(sensors["corridor_km"][-1])

    def crossing(times, offsets, target):
        for i in range(len(offsets) - 1):
            if offsets[i] <= target <= offsets[i + 1]:
                if offsets[i + 1] == offsets[i]:
                    return times[i]
                frac = (target - offsets[i]) / (offsets[i + 1] - offsets[i])
                return times[i] + frac * (times[i + 1] - times[i])
        return None

    hours = []
    for slts in sims.to_pydict():
        hist = slts["history"]
        t = hist.get("time_seconds") or []
        off = hist.get("offset_meters") or []
        if len(t) < 2 or len(off) < 2:
            continue
        t = [x / 3600.0 for x in t]
        off = [x / 1000.0 for x in off]
        total = off[-1]
        # Forward trains pass lo then hi; reverse trains cover the same track
        # between the mirrored offsets. Take whichever ordering resolves.
        for a, b in ((lo_km, hi_km), (total - hi_km, total - lo_km)):
            ta, tb = crossing(t, off, a), crossing(t, off, b)
            if ta is not None and tb is not None and tb > ta:
                hours.append(tb - ta)
                break
    if not hours:
        return None
    return float(pl.Series(hours).median())


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


def _train_sim_for(row: dict, n_helpers: int, rail_vehicles, location_map):
    """One plan train, built the way the planner builds it.

    Mass and length are scaled to the plan's observed figures first. Without
    that the feasibility test is run on a train far lighter and faster than the
    one the sweep simulates, and every train looks feasible.
    """
    car_type = rail_vehicles[0].to_pydict()["car_type"]
    n_cars = {car_type: int(row.get("Cars_Loaded") or 0)
                        + int(row.get("Cars_Empty") or 0)}
    n_locos = int(row["_n_locos"])
    helpers = min(max(n_helpers, 0), n_locos)
    locos = ([build_locomotive("diesel", None)] * helpers
             + [build_locomotive("catenary_bel", None)] * (n_locos - helpers))
    rv = data_prep.scale_rv_mass_and_length(
        [rail_vehicles[0]], n_cars, locos,
        float(row["Train_Weight_tons"]) * alt.utilities.KG_PER_TON,
        float(row["Train_Length_meters"]),
    )
    rv = data_prep.raise_rv_speed_max(
        rv, row.get("Train_Speed_Max_Meters_Per_Second"))
    tsb = alt.TrainSimBuilder(
        train_id=str(row["Train_ID"]),
        origin_id=row["Origin_ID"],
        destination_id=row["Destination_ID"],
        train_config=alt.TrainConfig(rail_vehicles=rv, n_cars_by_type=n_cars,
                                     train_type=alt.TrainType.Freight),
        loco_con=alt.Consist(locos),
    )
    return tsb.make_speed_limit_train_sim(location_map=location_map,
                                          save_interval=None)


def _min_soc(consist) -> float:
    """Lowest battery state of charge left in a consist after a test run.

    `make_est_times` returns the evolved consist, so the margin a train finished
    with costs nothing extra to read -- no second simulation and no history.
    A consist with no battery returns 1.0, which never triggers escalation.
    """
    socs = []
    for loco in consist.to_pydict()["loco_vec"]:
        loco_type = loco["loco_type"]
        inner = loco_type[next(iter(loco_type))]
        res = inner.get("res") if isinstance(inner, dict) else None
        if res:
            socs.append(float(res["state"]["soc"]))
    return min(socs) if socs else 1.0

def _helper_cache_path(plan: pl.DataFrame, row: dict) -> Path:
    """Cache key over everything the assignment depends on."""
    h = hashlib.sha1()
    h.update(plan.write_csv().encode("utf-8"))
    h.update(str(row.get("Layout_Fingerprint")).encode("utf-8"))
    h.update(str(row.get("Network_Path")).encode("utf-8"))
    return HELPERS_DIR / (h.hexdigest()[:16] + ".json")


def _base_helper_pass(plan, row, network, location_map, cache):
    """Fewest helpers each train needs to cross free-running, and its margin.

    One entry per train: the smallest number of conventional units that lets a
    free-running `make_est_times` complete, and the lowest battery state of
    charge the consist finished that run with. The margin is what escalation
    ranks by later, and it costs nothing extra -- `make_est_times` returns the
    evolved consist.
    """
    if cache.exists():
        logger.info("  helper base pass: reusing %s", cache.name)
        return {int(k): tuple(v) for k, v in
                json.loads(cache.read_text(encoding="utf-8")).items()}
    trains = (
        plan.group_by("Train_ID")
        .agg(pl.len().alias("_n_locos"), pl.all().first())
        .sort("Train_ID")
    )
    rail_vehicles = load_rail_vehicles()
    base, started = {}, time.perf_counter()
    for rec in trains.to_dicts():
        n_locos = int(rec["_n_locos"])
        for k in range(n_locos):  # 0 helpers first, then escalate
            try:
                _etn, consist = alt.make_est_times(
                    _train_sim_for(rec, k, rail_vehicles, location_map), network)
                base[int(rec["Train_ID"])] = (k, _min_soc(consist))
                break
            except Exception:  # noqa: BLE001 - infeasible, try one more unit
                continue
        else:
            # No helper count crossed. Record the MOST help available rather
            # than none: falling through to zero is the worst answer, not a
            # conservative one, and that is what an earlier version did.
            logger.error("  train %s could not cross with any helper count",
                         rec["Train_ID"])
            base[int(rec["Train_ID"])] = (n_locos - 1, 0.0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({str(k): list(v) for k, v in base.items()}),
                     encoding="utf-8")
    logger.info("  helper base pass: %d of %d trains need help (%d units) "
                "in %.0f s", sum(1 for k, _ in base.values() if k), len(base),
                sum(k for k, _ in base.values()), time.perf_counter() - started)
    return base


def assign_diesel_helpers(plan: pl.DataFrame, row: dict, network,
                          location_map, extra_fraction: float) -> pl.DataFrame:
    """Swap conventional units into the BEL consists that need them.

    The catenary deep-dive escalates by RETRY: it dispatches one train at a
    time, and on a pure-BEL failure re-runs that train with one more diesel unit
    until it completes. The sweep cannot retry that way -- it dispatches the
    whole plan in one call, which is what produces meet-pass -- so the decision
    is made in advance on the same test the deep-dive's first attempt performs,
    then widened at cell granularity if the dispatch still strands a train.

    Trains that need help have their first `k` locomotive rows retyped, so the
    plan carries the mixed consist and the pool builds it natively.
    """
    base = _base_helper_pass(plan, row, network, location_map,
                             _helper_cache_path(plan, row))
    n_by_train = dict(
        plan.group_by("Train_ID").agg(pl.len().alias("n")).iter_rows()
    )
    assignment = {t: k for t, (k, _) in base.items() if k}
    if extra_fraction > 0:
        # Lowest finishing margin first: those are the trains dispatch is most
        # likely to push past the floor.
        ranked = sorted(base, key=lambda t: base[t][1])
        for train in ranked[:int(round(extra_fraction * len(ranked)))]:
            cap = max(int(n_by_train.get(train, 1)) - 1, 0)
            assignment[train] = min(base[train][0] + 1, cap)
        assignment = {t: k for t, k in assignment.items() if k}
        logger.info("  escalated to fraction %.2f: %d of %d trains helped "
                    "(%d units)", extra_fraction, len(assignment), len(base),
                    sum(assignment.values()))
    units = int(plan.height)
    if sum(assignment.values()) > HELPER_MAX_SHARE * units:
        raise RuntimeError(
            f"battery-electric infeasible: {sum(assignment.values())} of {units} "
            f"locomotive units would have to be diesel "
            f"(>{HELPER_MAX_SHARE:.0%}), which is a diesel operation, not a "
            f"battery-electric one")
    if not assignment:
        return plan
    rank = pl.col("Locomotive_ID").rank("ordinal").over("Train_ID")
    need = pl.col("Train_ID").replace_strict(assignment, default=0)
    return plan.with_columns(
        pl.when(rank <= need)
        .then(pl.lit(HELPER_LOCO_TYPE))
        .otherwise(pl.col("Locomotive_Type"))
        .alias("Locomotive_Type")
    )


def _dispatch_and_walk(plan: pl.DataFrame, corridor, config, network, location_map,
                       stamp=None, charging_guidelines=None):
    """Dispatch and walk ONE sample: its own meet-pass problem, end to end.

    Kept separate from `run_simulation` because a corridor whose calibrated
    volume exceeds its stable traffic level is carried by SEVERAL samples,
    and each has to be dispatched on its own. Merging them before dispatch
    would assert exactly the congestion the split exists to avoid claiming.
    """
    # Warm-up and cool-down traffic, so the measured week is not bounded by an
    # empty corridor at either end. Supplying a consist plan bypasses
    # sim_manager's own warm-start machinery -- it is gated on
    # `consist_plan_in is None` -- so the window is built here and passed
    # explicitly.
    plan, window = extend_plan(plan, WARMUP_DAYS, COOLDOWN_DAYS)
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
        # `stamp` applies per-train tuned hybrid CONTROLLER settings
        # (origin x load class); None for every other arm and when untuned. It
        # no longer touches departure SOC -- that is the planner's, set from
        # `network_charging_guidelines` below, so there is exactly one owner.
        configure_sim=(trim_history if stamp is None
                       else (lambda sim: trim_history(stamp(sim)))),
        # Allowable battery headroom per origin, destination, load class,
        # locomotive type and pack size. The planner caps each departure SOC at
        # `SOC_Max_J - headroom`.
        network_charging_guidelines=charging_guidelines,
        measure_window_hr=window,
        debug=False,
    )
    return {
        "consist_plan": train_consist_plan,
        "loco_pool": loco_pool,
        "refuel_facilities": refuel_facilities,
        "grid_emissions_factors": grid_emissions_factors,
        "nodal_energy_prices": nodal_energy_prices,
        "sims": sims,
    }


# Samples are independent draws from the same trip population and each numbers
# its trains and locomotives from 1, so their IDs collide. Offsetting by a
# stride wider than any plan's ID range keeps them distinct through the joins
# `metric_calculator` makes between the consist plan, the locomotive pool and
# the sim vector -- without it, sample 2's train 7 would be matched against
# sample 1's locomotives.
_SAMPLE_ID_STRIDE = 1_000_000


def _merge_sample_runs(parts: List[dict]) -> dict:
    """Combine independently dispatched samples into one corridor-week.

    The samples are NOT merged before dispatch: no train in one ever met a
    train in another. That carries the corridor's full volume into energy,
    emissions, cost and fleet count while UNDERSTATING congestion, which is
    the honest direction to err -- the alternative, dispatching the whole
    volume as one problem, is a queue that never clears and travel times that
    mean nothing. See `stable_weekly_count` in `railstate_corridor.py`.

    Merging here rather than summing metrics afterwards is deliberate:
    `metric_calculator` then does its own aggregation over the combined week,
    so ratios like `Energy_Per_Freight_Moved` come out as the corridor ratio
    rather than an average of per-sample ratios, and nothing downstream needs
    to know the split happened.
    """
    if len(parts) == 1:
        return parts[0]
    plans, pools, sim_dicts = [], [], []
    for k, part in enumerate(parts):
        offset = k * _SAMPLE_ID_STRIDE
        part_sims = part["sims"].to_pydict()
        plans.append(part["consist_plan"].with_columns(
            (pl.col("Train_ID") + offset).alias("Train_ID"),
            (pl.col("Locomotive_ID") + offset).alias("Locomotive_ID"),
            # Positional, not an ID: it indexes the concatenated vector, so it
            # shifts by how many sims the earlier samples contributed.
            (pl.col("TrainSimVec_Index") + len(sim_dicts))
            .alias("TrainSimVec_Index"),
        ))
        pools.append(part["loco_pool"].with_columns(
            (pl.col("Locomotive_ID") + offset).alias("Locomotive_ID")))
        sim_dicts.extend(part_sims)
    merged = dict(parts[0])
    merged["consist_plan"] = pl.concat(plans, how="diagonal_relaxed")
    merged["loco_pool"] = pl.concat(pools, how="diagonal_relaxed")
    merged["sims"] = alt.SpeedLimitTrainSimVec.from_pydict(sim_dicts)
    return merged


def run_simulation(row: dict, plans):
    """One simulation cell: returns ``(metrics, stop_events, consist_plan)``.

    ``plans`` is one consist plan per sample -- a single frame is accepted and
    treated as one sample, which is what every corridor whose volume fits
    under its stable level uses.
    """
    corridor = CORRIDORS[row["Corridor"]]
    config = planner_config_for(row["Powertrain"], row["Battery_MWh"],
                                corridor=row["Corridor"])
    network = network_for(row)
    location_map = alt.import_locations(str(NETWORKS / corridor.locations))
    stamp = None
    charging_guidelines = None
    if row["Powertrain"] == "hybrid":
        regen_table = hybrid_controls_io.load_regen()
        stamp = hybrid_controls_io.regen_configurer(
            regen_table, row["Corridor"], row["Battery_MWh"])
        # The calibration's departure SOCs, expressed as the planner's allowable
        # headroom. Absent rows leave the pack departing as full as it arrived.
        charging_guidelines = hybrid_controls_io.headroom_guidelines(
            regen_table, row["Corridor"], row["Battery_MWh"])
        if stamp is not None:
            logger.info("  per-group regen controls applied")
        if charging_guidelines is not None:
            logger.info("  departure headroom: %s",
                        dict(charging_guidelines.select(
                            "Origin", "Allowable_Battery_Headroom_MWh").iter_rows()))
    if isinstance(plans, pl.DataFrame):
        plans = [plans]

    started = time.perf_counter()
    # Battery-electric only: every other arm runs one locomotive type, so there
    # is nothing to escalate and the ladder collapses to a single pass.
    ladder = (HELPER_ESCALATION_FRACTIONS if row["Powertrain"] == "catenary_bel"
              else (None,))
    parts = None
    for attempt, fraction in enumerate(ladder, start=1):
        staged = plans if fraction is None else [
            assign_diesel_helpers(p, row, network, location_map, fraction)
            for p in plans
        ]
        try:
            parts = []
            for k, plan in enumerate(staged, start=1):
                if len(staged) > 1:
                    logger.info("  sample %d of %d: %d trains", k, len(staged),
                                plan["Train_ID"].n_unique())
                parts.append(_dispatch_and_walk(plan, corridor, config, network,
                                                location_map, stamp,
                                                charging_guidelines))
            break
        except Exception as exc:  # noqa: BLE001 - escalate and try again
            if attempt >= len(ladder):
                raise
            logger.warning(
                "  a train stranded at escalation %.2f (%s); widening to %.2f",
                fraction, str(exc).splitlines()[0][:80], ladder[attempt])
            parts = None
    merged = _merge_sample_runs(parts)
    train_consist_plan = merged["consist_plan"]
    loco_pool = merged["loco_pool"]
    # The fleet is the units ever dispatched (warm-up and cool-down included);
    # the pool also holds the never-used stock the reuse rule could draw on.
    if "Times_Dispatched" in loco_pool.columns:
        stock = loco_pool.height
        loco_pool = loco_pool.filter(pl.col("Times_Dispatched") > 0)
        logger.info("  fleet: %d units used of %d stock (locomotive reuse)",
                    loco_pool.height, stock)
    refuel_facilities = merged["refuel_facilities"]
    grid_emissions_factors = merged["grid_emissions_factors"]
    nodal_energy_prices = merged["nodal_energy_prices"]
    sims = merged["sims"]

    # Terminal charging. Walking inside sim_manager produces no charging
    # sessions, so they are reconstructed from the walk: each battery-carrying
    # locomotive is charged at every departure from what it arrived with on its
    # previous trip up to what it departs with (`terminal_charging`). Departure
    # SOC comes from the consist plan's `Departure_SOC_J`, which the planner set
    # under its headroom cap -- there is no second source. The sessions replace
    # the old fleet true-up
    # (net pack drawdown / charger efficiency) in `calculate_electricity_use`,
    # and are priced at their terminal in the rollout.
    #
    # The frame is typed and, when there is nothing to charge, EMPTY rather than
    # None, deliberately: `calculate_ghg` contains
    #
    #     elif (info.refuel_sessions is None) and (
    #         info.emissions_factors.select(pl.col("Node").n_unique() > 0))
    #
    # and `and` on a DataFrame raises "truth value of a DataFrame is ambiguous".
    # Consumers also filter it on its columns, which a schema-less frame lacks.
    charger_template = (terminal_charging.bel_charger_rows(refuel_facilities)
                        if refuel_facilities is not None else None)
    charger_w = (float(charger_template["Refueler_J_Per_Hr"][0]) / 3600.0
                 if charger_template is not None and charger_template.height
                 else alt.defaults.BEL_CHARGE_RATE_KW * 1e3)
    if row["Powertrain"] in ("hybrid", "catenary_bel"):
        refuel_sessions = terminal_charging.charging_sessions(
            train_consist_plan, sims.to_pydict(),
            alt.defaults.BEL_CHARGER_EFFICIENCY, charger_w)
    else:
        refuel_sessions = pl.DataFrame(schema=terminal_charging.SESSION_SCHEMA)
    # Facilities the metrics price: battery-electric chargers sized to the peak
    # concurrent sessions at each terminal. Hybrid chargers are scaled from the
    # battery-electric cell's in the rollout (they need its result). Diesel
    # fuelers carry no cost in the defaults and are not sized.
    refuel_facilities = (
        terminal_charging.sized_chargers(charger_template, refuel_sessions)
        if row["Powertrain"] == "catenary_bel" else None)
    if refuel_facilities is not None:
        logger.info("  chargers sized to peak concurrent sessions: %s",
                    dict(refuel_facilities.select("Node", "Port_Count").iter_rows()))
    terminal_j = terminal_charging.energy_by_node(refuel_sessions)
    if terminal_j:
        logger.info("  terminal charging: %s",
                    {n: f"{j / 3.6e9:,.1f} MWh" for n, j in terminal_j.items()})

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
        # The pool is already the fleet (units ever dispatched), so count it whole:
        # the measured week's consist plan alone would miss units that ran only in
        # warm-up or cool-down.
        count_unused_locomotives=True,
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
    pantograph_j, regen_j, line_loss_j = metric_calculator._sim_catenary_energy_for(info)
    grid_draw_j, grid_return_j = catenary_policy.grid_flows(pantograph_j, regen_j, line_loss_j)
    # Simulated time over the observed sensor span, for the idle band. Paired
    # downstream with `observed_span_median_hr`: their difference is stationary
    # time a real train spent that the model never produced, and it bounds the
    # anti-idling opportunity from above.
    span_hr = span_travel_hours(sims, corridor)
    catenary_rows = [
        ("Catenary_Pantograph_Energy", "MJ", pantograph_j / 1e6),
        ("Catenary_Regen_Energy", "MJ", regen_j / 1e6),
        ("Catenary_Line_Loss", "MJ", line_loss_j / 1e6),
        # What the substation must supply: gross draw plus the line loss on the
        # way in. Priced in full and charged grid emissions in full.
        ("Catenary_Grid_Draw", "MJ", grid_draw_j / 1e6),
        # What the substation receives back under a receptive wire: returned
        # energy minus the line loss on the way out. Zero when
        # `catenary_policy.RECEPTIVE` is off. Credits emissions in full and
        # cost at `catenary_policy.EXPORT_CREDIT_FRACTION` (post-processing).
        ("Catenary_Grid_Return", "MJ", grid_return_j / 1e6),
    ]
    if span_hr is not None:
        catenary_rows.append(("Span_Travel_Time", "hours", span_hr))
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
    # The walked sims, loco pool and catenary draw go back to the caller so
    # they can be PERSISTED: the multi-year discounted-cost rollout runs as a
    # separate stage over saved sims, so the economics can be re-run and
    # re-priced without re-simulating. Grid regions come from the emissions
    # factors sim_manager resolved for this corridor's locations.
    regions = (sorted(grid_emissions_factors["Node"].unique().to_list())
               if grid_emissions_factors is not None
               and "Node" in grid_emissions_factors.columns else [])
    sidecar = {
        "sims": sims,
        "loco_pool": loco_pool,
        "pantograph_j": float(pantograph_j),
        "regen_j": float(regen_j),
        "line_loss_j": float(line_loss_j),
        "grid_draw_j": float(grid_draw_j),
        "grid_return_j": float(grid_return_j),
        "grid_regions": regions,
        "n_trains": int(train_consist_plan["Train_ID"].n_unique()),
        # Grid joules delivered by terminal chargers, per terminal (week total),
        # and the planner's refuelers -- the charger capital the rollout prices.
        "terminal_charging_j": terminal_j,
        "refuel_facilities": refuel_facilities,
        "charging_sessions": refuel_sessions,
    }
    return (
        meta.join(metrics, how="cross"),
        meta.join(events.drop("Sim_Key"), how="cross") if events.height else events,
        meta.join(train_consist_plan, how="cross"),
        sidecar,
    )


def _inputs_fingerprint(row: dict) -> str:
    """Hash of everything one cell's result depends on.

    `--skip-existing` used to trust the mere presence of an output file. That
    silently reused results computed against a DIFFERENT consist plan twice in
    this project -- once an 88-train plan after the corridor had been resized to
    52 -- and the failure is invisible: a stale result and a fresh one differ in
    no way a filename can express, and every number downstream still looks
    plausible. So the check is on content: the plan samples, the corridor
    metadata that carries the speed cap and sample count, the optimized layout
    where there is one, and the experiment row itself.
    """
    corridor = CORRIDORS[row["Corridor"]]
    h = hashlib.sha1()
    for path in corridor.sample_plan_paths:
        h.update(Path(path).read_bytes())
    if corridor.meta_path.exists():
        h.update(corridor.meta_path.read_bytes())
    fingerprint = row.get("Layout_Fingerprint") or "nobuild"
    if fingerprint not in ("nobuild", "na", None):
        layout = LAYOUTS / (str(fingerprint) + ".msgpack")
        if layout.exists():
            h.update(layout.read_bytes())
    for key in ("Sim_Key", "Corridor", "Powertrain", "Battery_MWh",
                "Architecture", "Layout_Fingerprint"):
        h.update(str(row.get(key)).encode("utf-8"))
    # The locomotive itself: a redefined powertrain (the hybrid's engine was
    # doubled once) must invalidate the cells that used the old one, and a
    # powertrain name cannot express that.
    loco = build_locomotive(row["Powertrain"], row.get("Battery_MWh"))
    h.update(json.dumps(_strip_history(loco.to_pydict()), sort_keys=True,
                        default=str).encode("utf-8"))
    # The tuned controller lives in a CSV rather than in the locomotive built
    # above, so hash the file itself: retuning must re-run the hybrid cells.
    # HYBRID ONLY -- folding it into every powertrain would change the
    # fingerprint of the diesel and catenary cells too, marking valid results
    # stale and re-running hours of simulation for a file they never read.
    if row["Powertrain"] == "hybrid":
        h.update(hybrid_controls_io.fingerprint_bytes())
        h.update(hybrid_controls_io.regen_fingerprint_bytes())
    return h.hexdigest()


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
    plans: Dict[str, List[pl.DataFrame]] = {}

    for row in sims.iter_rows(named=True):
        slug = row["Corridor"]
        if slug not in plans:
            loaded = []
            for path in CORRIDORS[slug].sample_plan_paths:
                plan = pl.read_csv(path)
                if args.max_trains:
                    keep = plan["Train_ID"].unique().sort()[: args.max_trains]
                    plan = plan.filter(pl.col("Train_ID").is_in(keep))
                    logger.warning(
                        "SMOKE TEST: %s capped to %d trains -- results are not "
                        "the calibrated traffic level", slug,
                        plan["Train_ID"].n_unique())
                logger.info(
                    "%s plan: %d trains, %d locomotive rows, from %s",
                    slug, plan["Train_ID"].n_unique(), plan.height, path.name)
                loaded.append(plan)
            if len(loaded) > 1:
                logger.info(
                    "%s: %d samples, %d trains total -- dispatched as separate "
                    "meet-pass problems, merged for metrics", slug, len(loaded),
                    sum(f["Train_ID"].n_unique() for f in loaded))
            plans[slug] = loaded
        safe = row["Sim_Key"].replace("|", "__").replace(".", "p")
        metrics_path = args.per_sim_dir / f"{safe}_metrics.csv"
        stamp_path = args.per_sim_dir / (safe + "_inputs.txt")
        stamp = _inputs_fingerprint(row)
        # A cell is only done if its sims bundle exists too; the rollout stage
        # needs it, and earlier sweeps never wrote one.
        bundle_path = args.per_sim_dir / f"{safe}_sims.msgpack"
        if args.skip_existing and metrics_path.exists() and bundle_path.exists():
            prior = (stamp_path.read_text(encoding="utf-8").strip()
                     if stamp_path.exists() else "")
            if prior == stamp:
                logger.info("skipping %s (already done)", row["Sim_Key"])
                continue
            logger.warning(
                "  re-running %s: its inputs changed since that output was "
                "written", row["Sim_Key"])
        logger.info("=== %s ===", row["Sim_Key"])
        try:
            metrics, events, consist_plan, sidecar = run_simulation(row, plans[slug])
        except Exception as exc:  # noqa: BLE001 - one cell failing is not fatal
            logger.exception("  FAILED: %s", str(exc)[:200])
            failures[row["Sim_Key"]] = str(exc)[:200]
            continue
        metrics.write_csv(metrics_path)
        stamp_path.write_text(stamp, encoding="utf-8")
        if events.height:
            events.write_csv(args.per_sim_dir / f"{safe}_stops.csv")
        consist_plan.write_csv(args.per_sim_dir / f"{safe}_plan.csv")
        # Persist the walked sims for the rollout stage. Small, because the
        # history spec already trimmed them to the vectors the study reads.
        sidecar["sims"].to_file(str(args.per_sim_dir / f"{safe}_sims.msgpack"))
        sidecar["loco_pool"].write_csv(args.per_sim_dir / f"{safe}_loco_pool.csv")
        if sidecar.get("refuel_facilities") is not None:
            sidecar["refuel_facilities"].with_columns(
                pl.col(pl.Categorical).cast(pl.Utf8)
            ).write_csv(args.per_sim_dir / f"{safe}_refuelers.csv")
        if sidecar.get("charging_sessions") is not None and sidecar["charging_sessions"].height:
            sidecar["charging_sessions"].write_csv(
                args.per_sim_dir / f"{safe}_charging_sessions.csv")
        (args.per_sim_dir / f"{safe}_sidecar.json").write_text(json.dumps({
            k: v for k, v in sidecar.items()
            if k not in ("sims", "loco_pool", "refuel_facilities", "charging_sessions")
        }), encoding="utf-8")
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
