from altrios import sim_manager
from altrios import metric_calculator, defaults
from altrios.metric_calculator import ScenarioInfo

import altrios as alt
from altrios.train_planner import planner, planner_config
import numpy as np
import time
import pandas as pd
import polars as pl
from typing import List, Tuple, Optional, Union
from pathlib import Path
import os

DEBUG = True


def simulate_prescribed_rollout(
    max_bel_share: float,
    number_of_years: int,
    results_folder: Path,
    start_year: int = defaults.BASE_ANALYSIS_YEAR,
    # If you do not have this path, please add it! We are trying to keep confidential documents out of the git repo
    network_filename_path: str = str(alt.resources_root() / "networks/Taconite.yaml"),
    save_interval: Optional[int] = None,
    freight_demand_percent_growth: float = 0.0,
    demand_file: Union[pl.DataFrame, Path, str] = defaults.DEMAND_FILE,
    train_planner_config: planner_config.TrainPlannerConfig = planner_config.TrainPlannerConfig(),
    count_unused_locomotives=False,
    write_complete_results: Optional[bool] = False,
    write_metrics: Optional[bool] = False,
) -> Tuple[List[ScenarioInfo], pl.DataFrame]:
    years = list(range(start_year, start_year + number_of_years))
    target_bel_shares = np.zeros(len(years))
    if len(years) == 0:
        target_bel_shares[0] = max_bel_share
    else:
        for idx, _ in enumerate(target_bel_shares):
            if idx == 0:
                target_bel_shares[idx] = 0.0
            else:
                target_bel_shares[idx] = ((idx) / (len(years) - 1)) * max_bel_share

    loco_type_shares = []
    for target_bel_share in target_bel_shares:
        annual_loco_type_shares = {"BEL": target_bel_share, "Diesel_Large": 1 - target_bel_share}
        loco_type_shares.append(annual_loco_type_shares)

    save_dir = Path(results_folder) # make sure it's a path
    save_dir.mkdir(exist_ok=True, parents=True) 
    with open(save_dir / "README.md", "w") as file:
        file.writelines(
            [
                "This directory contains results from demo files and can usually be safely deleted."
            ]
        )
    with open(save_dir / ".gitignore", "w") as file:
        file.writelines(["*"])

    base_freight_demand_df = pd.read_csv(demand_file)
    demand_paths = []
    for year in years:
        if freight_demand_percent_growth > 0:
            demand_filename = (
                results_folder
                + "/"
                + os.path.basename(demand_file).replace(".csv", str(year) + ".csv")
            )
            base_freight_demand_df.to_csv(demand_filename, float_format="%.0f")
            demand_paths.append(demand_filename)
            base_freight_demand_df.Number_of_Cars = (
                base_freight_demand_df.Number_of_Cars
                * (1 + freight_demand_percent_growth / 100)
            )
            base_freight_demand_df.Number_of_Containers = (
                base_freight_demand_df.Number_of_Containers
                * (1 + freight_demand_percent_growth / 100)
            )
        else:
            demand_paths.append(demand_file)

    rail_vehicles = [
        alt.RailVehicle.from_file(vehicle_file, skip_init=False)
        for vehicle_file in Path(alt.resources_root() / "rolling_stock/").glob("*.yaml")
    ]

    location_map = alt.import_locations(
        str(alt.resources_root() / "networks/default_locations.csv")
    )
    network = alt.Network.from_file(network_filename_path, skip_init=False)
    sim_days = defaults.SIMULATION_DAYS
    scenarios = []
    for idx, scenario_year in enumerate(years):
        train_planner_config.loco_type_shares = loco_type_shares[idx]
        t0 = time.perf_counter()
        (
            train_consist_plan,
            loco_pool,
            refuel_facilities,
            grid_emissions_factors,
            nodal_energy_prices,
            speed_limit_train_sims,
            timed_paths,
            train_consist_plan_untrimmed,
        ) = sim_manager.main(
            network=network,
            rail_vehicles=rail_vehicles,
            location_map=location_map,
            simulation_days=sim_days,
            scenario_year=scenario_year,
            debug=True,
            train_planner_config=train_planner_config,
            demand_file=demand_paths[idx],
        )

        t1 = time.perf_counter()
        if DEBUG:
            print(
                f"Elapsed time to run `sim_manager.main() for year {scenario_year}`: {t1 - t0:.3g} s"
            )
        speed_limit_train_sims.set_save_interval(save_interval)
        used_loco_pool = loco_pool.filter(
            pl.col("Locomotive_ID").is_in(
                train_consist_plan.get_column("Locomotive_ID").unique()
            )
        )
        (sims, refuel_sessions) = alt.run_speed_limit_train_sims(
            speed_limit_train_sims=speed_limit_train_sims,
            network=network,
            train_consist_plan_py=train_consist_plan,
            loco_pool_py=used_loco_pool,
            refuel_facilities_py=refuel_facilities,
            timed_paths=[alt.TimedLinkPath.from_pydict(tp) for tp in timed_paths],
        )
        # gallons = (sims.get_energy_fuel_joules(annualize=True) / 1e3 / 45.6e3) * 1e3 / 3206

        loco_pool = loco_pool.drop(
            [
                "Refueler_Efficiency",
                "Refueler_J_Per_Hr",
                "Port_Count",
                "Battery_Headroom_J",
            ]
        )
        train_consist_plan = train_consist_plan.sort(
            ["Locomotive_Type", "Locomotive_ID", "Train_ID"]
        )
        refuel_sessions = refuel_sessions.sort(
            ["Locomotive_Type", "Locomotive_ID", "Refuel_Start_Time_Hr"]
        ).with_columns(
            train_consist_plan.get_column("TrainSimVec_Index"),
            train_consist_plan.get_column("Train_ID"),
        )
        scenarios.append(
            ScenarioInfo(
                sims,
                sim_days,
                True,
                scenario_year,
                loco_pool,
                train_consist_plan,
                refuel_facilities,
                refuel_sessions,
                grid_emissions_factors,
                nodal_energy_prices,
                count_unused_locomotives,
            )
        )

        t2 = time.perf_counter()

        if DEBUG:
            print(
                f"Elapsed time to run `run_speed_limit_train_sims()` for year {scenario_year}: {t2 - t1:.3g} s"
            )

        if write_complete_results:
            sims.to_file(
                str(
                    results_folder
                    / "RolloutResults_Year {}_Demand {}.json".format(
                        scenario_year, Path(demand_file).name
                    )
                ).replace(".csv", "")
            )

            train_consist_plan.write_csv(
                str(
                    results_folder
                    / "ConsistPlan_Year {}_Demand {}.csv".format(
                        scenario_year, Path(demand_file).name
                    )
                ).replace(".csv", "")
                + ".csv"
            )

            t3 = time.perf_counter()

            if DEBUG:
                print(
                    f"Elapsed time to serialize results for year {scenario_year}: {t3 - t2:.3g} s"
                )

        t2 = time.perf_counter()

    metrics = metric_calculator.main(scenarios)

    if write_metrics:
        print
        (
            results_folder
            / "Metrics_Demand {}_DemandFile {}.xlsx".format(
                scenario_year, os.path.basename(demand_file)
            ).replace(".csv", "")
        )
        metrics.to_pandas().to_excel(
            results_folder
            / "Metrics_Demand {}_DemandFile {}.xlsx".format(
                scenario_year, os.path.basename(demand_file)
            ).replace(".csv", "")
        )

    t3 = time.perf_counter()
    if DEBUG:
        print(
            f"Elapsed time to run `metric_calculator.main()` and serialize metrics: {t3 - t2:.3g} s"
        )

    return scenarios, metrics
