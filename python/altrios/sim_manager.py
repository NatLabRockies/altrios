"""
Module for getting the output of the Train Consist Planner and Meet Pass Planner to run 3 week simulation.
"""

import polars as pl
from typing import Optional, Union, Dict, List, Tuple
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
    debug: bool = False,
    loco_pool: Optional[pl.DataFrame] = None,
    refuelers: Optional[pl.DataFrame] = None,
    grid_emissions_factors: Optional[pl.DataFrame] = None,
    nodal_energy_prices: Optional[pl.DataFrame] = None,
    train_planner_config: planner_config.TrainPlannerConfig = planner_config.TrainPlannerConfig(),
    train_type: alt.TrainType = alt.TrainType.Freight,
    demand_file: Union[pl.DataFrame, Path, str] = str(defaults.DEMAND_FILE),
    network_charging_guidelines: Optional[pl.DataFrame] = None,
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
    train_planner_config.simulation_days = simulation_days + 2 * warm_start_days
    t0_ptc = time.perf_counter()
    (
        train_consist_plan,
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

    t0_disp = time.perf_counter()
    timed_paths = alt.run_dispatch(
        network,
        alt.SpeedLimitTrainSimVec(speed_limit_train_sims),
        est_time_nets,
        False,
        False,
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

    train_consist_plan = train_consist_plan.join(
        train_times, on=["Train_ID", "Origin_ID", "Destination_ID"], how="left"
    )
    train_consist_plan_untrimmed = train_consist_plan.clone()
    if train_planner_config.single_train_mode is False:
        train_consist_plan = train_consist_plan.filter(
            pl.col("Departure_Time_Planned_Hr") >= pl.lit(24 * warm_start_days),
            pl.col("Departure_Time_Planned_Hr")
            < pl.lit(24 * (simulation_days + warm_start_days)),
        )
    train_consist_plan = train_consist_plan.with_columns(
        (pl.col("Train_ID").rank("dense") - 1).alias("TrainSimVec_Index")
    )
    # speed_limit_train_sims is 0-indexed but Train_ID starts at 1
    to_keep = train_consist_plan.unique(subset=["Train_ID"]).to_series().sort()
    for i, sim in enumerate(speed_limit_train_sims):
        sim_dict = sim.to_pydict()
        sim_dict["simulation_days"] = simulation_days
        speed_limit_train_sims[i] = alt.SpeedLimitTrainSim.from_pydict(sim_dict)

    train_sims = alt.SpeedLimitTrainSimVec(
        [speed_limit_train_sims[i - 1] for i in to_keep]
    )
    timed_paths = [timed_paths[i - 1] for i in to_keep]

    return (
        train_consist_plan,  # NOTE: we generate this and therefore don't need it
        loco_pool,
        refuelers,
        grid_emissions_factors,
        nodal_energy_prices,
        train_sims,
        timed_paths,
        train_consist_plan_untrimmed,
    )
