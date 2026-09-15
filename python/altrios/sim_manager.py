"""
Module for getting the output of the Train Consist Planner and Meet Pass Planner to run 3 week simulation.
"""

import polars as pl
from typing import Any, Callable, Optional, Union, Dict, List, Tuple
from pathlib import Path
import time
from altrios import defaults

import altrios as alt
from altrios.train_planner import planner, planner_config
from altrios import metric_calculator as metrics


def main(
    rail_vehicles: List[alt.RailVehicle],
    location_map: Dict[str, List[alt.Location]],
    network: List[alt.Link],
    simulation_days: int = defaults.SIMULATION_DAYS,
    warm_start_days: int = defaults.WARM_START_DAYS,
    scenario_year: int = defaults.BASE_ANALYSIS_YEAR,
    walk_timed_paths = False,
    debug: bool = False,
    loco_pool: Optional[pl.DataFrame] = None,
    refuelers: Optional[pl.DataFrame] = None,
    grid_emissions_factors: Optional[pl.DataFrame] = None,
    nodal_energy_prices: Optional[pl.DataFrame] = None,
    train_planner_config: planner_config.TrainPlannerConfig = planner_config.TrainPlannerConfig(),
    train_type: alt.TrainType = alt.TrainType.Freight,
    demand_file: Union[pl.DataFrame, Path, str] = str(defaults.DEMAND_FILE),
    network_charging_guidelines: Optional[pl.DataFrame] = None,
    consist_plan_in: Optional[pl.DataFrame] = None,
    save_interval: Optional[int] = 60,
    configure_sim: Optional[Callable[[Any], Any]] = None,
    measure_window_hr: Optional[Tuple[float, float]] = None,
) -> Tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    alt.SpeedLimitTrainSimVec,
    List[List[alt.LinkIdxTime]],
]:
    """
    # Return
    ```
    return (
        train_consist_plan,
        loco_pool,
        refuelers,
        grid_emissions_factors,
        nodal_energy_prices,
        train_sims,
        timed_paths,
    )
    ```

    """

    if debug:
        print("Entering `sim_manager` module.")
        alt.utils.print_dt()

    for loc_name in location_map:
        for loc in location_map[loc_name]:
            loc_dict = loc.to_pydict()
            if loc_dict["Link Index"] >= len(network):
                raise ValueError(
                    "Location "
                    + loc_dict["Location ID"]
                    + " with link index "
                    + str(loc_dict["Link Index"])
                    + " is invalid for network!"
                )

    train_planner_config.loco_info = metrics.add_battery_costs(
        train_planner_config.loco_info, scenario_year
    )
    if consist_plan_in is None:
        train_planner_config.simulation_days = simulation_days + 2 * warm_start_days

    t0_ptc = time.perf_counter()
    (
        consist_plan_out,
        loco_pool,
        refuelers,
        speed_limit_train_sims,
        est_time_nets,
    ) = planner.run_train_planner(
        rail_vehicles=rail_vehicles,
        location_map=location_map,
        network=network,
        loco_pool=loco_pool,
        refuelers=refuelers,
        scenario_year=scenario_year,
        config=train_planner_config,
        demand_file=demand_file,
        train_type=train_type,
        network_charging_guidelines=network_charging_guidelines,
        consist_plan_in=consist_plan_in
    )
    t1_ptc = time.perf_counter()

    if grid_emissions_factors is None:
        grid_emissions_factors = metrics.import_emissions_factors_cambium(
            location_map=location_map, scenario_year=scenario_year
        )
    if nodal_energy_prices is None:
        nodal_energy_prices = metrics.import_energy_prices_eia(
            location_map=location_map, scenario_year=scenario_year
        )

    if debug:
        print(
            f"Elapsed time to plan train consist for year {scenario_year}: {t1_ptc - t0_ptc:.3g} s"
        )

    origins = []
    for slts in speed_limit_train_sims:
        origins.append(slts.to_pydict()['origs'][0]['Location ID'])
    
    t0_disp = time.perf_counter()
    # All trains go into a single `run_dispatch` call so meet-pass conflict resolution
    # actually runs. Dispatching one train per call bypasses meet-pass entirely.
    timed_paths = alt.run_dispatch(
        network,
        alt.SpeedLimitTrainSimVec(speed_limit_train_sims),
        est_time_nets,
        False,
        False,
    )
    if walk_timed_paths:
        # `save_interval` and `configure_sim` exist because saved history is the
        # dominant memory cost of a walked plan, and how much of it a study needs
        # varies enormously. History volume scales with each train's SIMULATED
        # trip duration, which itself grows with traffic -- so on a long, busy
        # corridor the cost is far worse than train count alone suggests.
        # Winnipeg-Chicago (162 trains, 1,237 km) reaches ~560 MB/train at the
        # default 60 s and exhausts a 64 GB machine partway through the walk,
        # while the same corridor at 80 trains costs ~27 MB/train.
        #
        # `configure_sim` runs on each sim AFTER the interval is applied and
        # BEFORE it is walked -- the only point where per-component history can
        # be switched off, since `set_save_interval` recurses into every
        # sub-component and would otherwise re-enable it. Called before the
        # walk, a caller's pydict round-trip is cheap: history is still empty.
        # Defaults preserve the previous behaviour exactly.
        # Warm-up and cool-down trains are DISPATCHED but not WALKED. They exist
        # to fill the corridor -- so that a measured train meets realistic
        # traffic at the start and end of its window rather than an empty
        # network -- and nothing reads their physics or energy. Walking them
        # anyway costs time and memory proportional to their number, then throws
        # the result away: the trim below this block used to run AFTER the walk,
        # so a 7/7/7 run walked three weeks to keep one.
        #
        # `measure_window_hr` is (start, end) in planned departure hours. Only
        # trains departing inside it are walked. None walks everything, which is
        # the previous behaviour.
        walk_idxs = set(range(len(speed_limit_train_sims)))
        if measure_window_hr is not None and "Departure_Time_Planned_Hr" in (
            consist_plan_out.columns if consist_plan_out is not None else []
        ):
            lo, hi = measure_window_hr
            measured = consist_plan_out.filter(
                pl.col("Departure_Time_Planned_Hr") >= pl.lit(lo),
                pl.col("Departure_Time_Planned_Hr") < pl.lit(hi),
            )
            # speed_limit_train_sims is 0-indexed; Train_ID starts at 1.
            walk_idxs = {
                int(t) - 1 for t in measured.get_column("Train_ID").unique().to_list()
            }
            if debug:
                print(
                    f"walking {len(walk_idxs)} of {len(speed_limit_train_sims)} trains "
                    f"(departures in [{lo}, {hi}) h); the rest are warm-up/cool-down"
                )

        for i in range(len(speed_limit_train_sims)):
            if i not in walk_idxs:
                continue
            speed_limit_train_sims[i].set_save_interval(save_interval)
            if configure_sim is not None:
                speed_limit_train_sims[i] = configure_sim(speed_limit_train_sims[i])
            speed_limit_train_sims[i].walk_timed_path(
                network=network,
                timed_path=timed_paths[i],
            )

    timed_paths: List[List[alt.LinkIdxTime]] = [  # type: ignore[no-redef]
        tp.to_pydict() for tp in timed_paths
    ]


    t1_disp = time.perf_counter()
    if debug:
        print(
            f"Elapsed time to run dispatch for year {scenario_year}: {t1_disp - t0_disp:.3g} s"
        )

    slts_dicts = [sim.to_pydict() for sim in speed_limit_train_sims]

    train_times = pl.DataFrame(
        {
            "Train_ID": pl.Series(
                [int(sim["train_id"]) for sim in slts_dicts], dtype=pl.Int32
            ).cast(pl.UInt32),
            "Origin_ID": pl.Series(
                [sim["origs"][0]["Location ID"] for sim in slts_dicts],
                dtype=str,
            ),
            "Destination_ID": pl.Series(
                [sim["dests"][0]["Location ID"] for sim in slts_dicts],
                dtype=str,
            ),
            "Departure_Time_Actual_Hr": pl.Series(
                [this[0]["time_seconds"] / 3_600 for this in timed_paths],
                dtype=pl.Float64,
            ),
            "Arrival_Time_Actual_Hr": pl.Series(
                [this[len(this) - 1]["time_seconds"] / 3_600 for this in timed_paths],
                dtype=pl.Float64,
            ),
        }
    )

    consist_plan_out = consist_plan_out.join(
        train_times, on=["Train_ID", "Origin_ID", "Destination_ID"], how="left"
    )
    consist_plan_out_untrimmed = consist_plan_out.clone()
    if (train_planner_config.single_train_mode is False) and (consist_plan_in is None):
        consist_plan_out = consist_plan_out.filter(
            pl.col("Departure_Time_Planned_Hr") >= pl.lit(24 * warm_start_days),
            pl.col("Departure_Time_Planned_Hr")
            < pl.lit(24 * (simulation_days + warm_start_days)),
        )
    elif measure_window_hr is not None:
        # A caller-supplied plan carries its own warm-up and cool-down, so trim
        # to the window it named. This has to happen for the RESULTS as well as
        # the walk: everything downstream -- the returned consist plan, the sim
        # vector, and therefore every metric and its annualization -- must see
        # the measured days only. Warm-up trains were never walked, so leaving
        # them in would put empty histories into the metric calculator and
        # inflate any per-train or per-week total by the warm-up ratio.
        lo, hi = measure_window_hr
        consist_plan_out = consist_plan_out.filter(
            pl.col("Departure_Time_Planned_Hr") >= pl.lit(lo),
            pl.col("Departure_Time_Planned_Hr") < pl.lit(hi),
        )
    
    consist_plan_out = consist_plan_out.with_columns(
        (pl.col("Train_ID").rank("dense") - 1).alias("TrainSimVec_Index")
    )
    # speed_limit_train_sims is 0-indexed but Train_ID starts at 1
    to_keep = consist_plan_out.unique(subset=["Train_ID"]).to_series().sort()
    # Only the kept sims need stamping: the rest are discarded on the next line,
    # and a pydict round-trip on a sim that was never walked is pure waste.
    for i in to_keep:
        sim_dict = speed_limit_train_sims[i - 1].to_pydict()
        sim_dict["simulation_days"] = simulation_days
        speed_limit_train_sims[i - 1] = alt.SpeedLimitTrainSim.from_pydict(sim_dict)

    train_sims = alt.SpeedLimitTrainSimVec(
        [speed_limit_train_sims[i - 1] for i in to_keep]
    )
    timed_paths = [timed_paths[i - 1] for i in to_keep]

    return (
        consist_plan_out,
        loco_pool,
        refuelers,
        grid_emissions_factors,
        nodal_energy_prices,
        train_sims,
        timed_paths,
        consist_plan_out_untrimmed,
    )
