import pandas as pd
import polars as pl
from typing import Dict, Tuple, List
import numpy as np
import polars.selectors as cs
from altrios import defaults
import altrios as alt
from altrios.train_planner.planner_config import TrainPlannerConfig
from altrios.train_planner import (
    data_prep,
    schedulers,
    planner_config,
    train_demand_generators,
)


def manual_train_planner(
    consist_plan: pl.DataFrame,
    loco_map: Dict[str, str],
    rail_vehicles: List[alt.RailVehicle],
    location_map: Dict[str, List[alt.Location]],
    network: List[alt.Link],
    scenario_year: int = defaults.BASE_ANALYSIS_YEAR, 
    config: TrainPlannerConfig = TrainPlannerConfig(),
    

) -> Tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    List[alt.SpeedLimitTrainSim],
    List[alt.EstTimeNet],
]:
    """
    Generate train simulations from historical data

    # Arguments
    - `consist_plan`:
        consist_plan = pl.DataFrame(schema=
            {'Train_ID': pl.Int64,
             'Train_Type': pl.Utf8,
             'Locomotive_ID': pl.UInt32,
             'Locomotive_Type': pl.Categorical,
             'Origin_ID': pl.Utf8,
             'Destination_ID': pl.Utf8,
             'Cars_Loaded': pl.Float64,
             'Cars_Empty': pl.Float64,
             'Containers_Loaded': pl.Float64,
             'Containers_Empty': pl.Float64,
             'Departure_SOC_J': pl.Float64,
             'Departure_Time_Planned_Hr': pl.Float64,
             'Arrival_Time_Planned_Hr': pl.Float64})
    - `loco_map`: mapping of test data locomotive types to ALTRIOS locomotive
        types with real world locomotive types as keys and ALTRIOS models as values
    """

    # Append additional locomotive info to configuration.  Not sure why this is important, but I'm copying the regular train planner.
    config.loco_info = data_prep.append_loco_info(config.loco_info)

    # this is a list of all the unique destinations in the consist plan
    node_list = (
        pl.concat(
            [
                consist_plan.get_column("Origin_ID"),
                consist_plan.get_column("Destination_ID"),
            ]
        )
        .unique()
        .sort()
    )

    loco_pool = build_loco_pool_from_model_list(
        consist_plan, config
    )

    # build refueling or recharging infrastructure at all nodes
    refuelers = data_prep.build_refuelers(
        node_list,
        loco_pool,
        config.refueler_info,
        config.refuelers_per_incoming_corridor,
    )
    
    #TODO figure out I why I am adding this headroom vairable here.  Need to get it working right now.
    refuelers = refuelers.with_columns(pl.lit(0.0).alias("Battery_Headroom_J"))

    # Create mapping from freight types to car types
    freight_type_to_car_type = {}
    for rv in rail_vehicles:
        # Check for duplicate mappings (should not happen)
        rv_dict = rv.to_pydict()

        if rv_dict["freight_type"] in freight_type_to_car_type:
            raise Exception(f"More than one unique rail vehicle car type for freight type {rv_dict['freight_type']}")
        else:
            # Map this freight type to its car type
            freight_type_to_car_type[rv_dict["freight_type"]] = rv_dict["car_type"]


    # initialize lists for simulation results
    speed_limit_train_sims=[]
    est_time_nets=[]
    for this_train in consist_plan.unique(subset=["Train_ID"], maintain_order=True, keep='first').iter_rows(named=True):
    # Create train simulation builder
        # Calculate drag coefficient if configured
        if config.drag_coeff_function is not None:
            # Apply drag coefficient function based on number of cars
            cd_area_vec = config.drag_coeff_function(
                int(this_train["Number_of_Cars"])
            )
        else:
            # No custom drag coefficient
            cd_area_vec = None

        # Configure rail vehicles for this train
        rv_to_use, n_cars_by_type = data_prep.configure_rail_vehicles(
            this_train, rail_vehicles, freight_type_to_car_type
        )

        #get Train type object? from the rust side of things.  Checkout link for enumeration.
        #https://docs.rs/altrios-core/latest/altrios_core/track/enum.TrainType.html
        if this_train['Train_Type'] == 'Intermodal':
            train_type = alt.TrainType.Intermodal
        else:
            train_type = alt.TrainType.Freight

        # Create train configuration
        train_config = alt.TrainConfig(
            rail_vehicles=rv_to_use,
            n_cars_by_type=n_cars_by_type,
            train_type=train_type,
            cd_area_vec=cd_area_vec,
        )

        #getting the locomotives road number out of the consist plan for this train
        consist_locos_road_numbers = consist_plan.filter(Train_ID = this_train["Train_ID"])['Locomotive_ID']
        #pulling locomotives from pool to get fuel type that is needed about a dozen lines down where we are checking for fuel type
        consist_locos = loco_pool.filter(pl.col('Locomotive_ID').is_in(consist_locos_road_numbers))
        # Create list of locomotive objects from configuration
        # TODO get the locos from the consist plan by train
        locos = [
            config.loco_info[
                config.loco_info["Locomotive_Type"] == loco_type
            ]["Rust_Loco"]
            .to_list()[0]
            .copy()
            for loco_type in consist_locos.get_column("Locomotive_Type")
        ]

        #TODO figure out how to set soc to 100% for the time being
        # Set state of charge for electric locomotives
        loco_list = [loco.to_pydict() for loco in locos]
        for i, loco_dict in enumerate(loco_list):
            if consist_locos.get_column("Fuel_Type")[i] == "Electricity":
                loco_type_val = next(iter(loco_dict["loco_type"].values()))
                loco_type_key = next(iter(loco_dict["loco_type"].keys()))                
                #TODO figure out if 1.0 will cause problems
                loco_type_val["res"]["state"]["soc"] = 1.0
                loco_dict["loco_type"][loco_type_key] = loco_type_val
                loco_list[i] = loco_dict

        loco_vec = [alt.Locomotive.from_pydict(loco_dict) for loco_dict in loco_list]      
        # Create locomotive consist from the selected locomotives
        loco_con = alt.Consist(
            loco_vec=loco_vec,
            save_interval=None,
        )

        #TODO grab time from planned departure time
        # Create initial train state with correct time
        init_train_state = alt.InitTrainState(
            time_seconds=this_train["Departure_Time_Planned_Hr"] * 3600  # Convert hours to seconds
        )

        tsb = alt.TrainSimBuilder(
            train_id=str(this_train["Train_ID"]),
            origin_id=str(this_train["Origin_ID"]),
            destination_id=str(this_train["Destination_ID"]),
            train_config=train_config,
            loco_con=loco_con,
            init_train_state=init_train_state,
        )

        # Create speed-limited train simulation
        slts = tsb.make_speed_limit_train_sim(
            location_map=location_map,
            save_interval=None,
            simulation_days=config.simulation_days,
            scenario_year=scenario_year,
        )

        # Generate estimated travel times
        (est_time_net, loco_con_out) = alt.make_est_times(
            slts, network, config.failed_sim_logging_path
        )
        # Store simulation results
        speed_limit_train_sims.append(slts)
        est_time_nets.append(est_time_net)

    #TODO this is call train_consist_plan in the regular planner.  May be worth cleaning up names across the code in several places.
    consist_plan = (
        consist_plan.with_columns(
            # Convert categorical columns to strings
            cs.categorical().cast(str),
            # Ensure ID columns are unsigned integers
            pl.col("Train_ID", "Locomotive_ID").cast(pl.UInt32),
        )
        # Sort by locomotive ID and train ID
        .sort(["Locomotive_ID", "Train_ID"], descending=False)
    )

    loco_pool = (
        loco_pool.with_columns(
            # Convert categorical columns to strings
            cs.categorical().cast(str),
            # Ensure ID columns are unsigned integers
            pl.col("Locomotive_ID").cast(pl.UInt32),
        )
        # Sort by locomotive ID and train ID
        .sort(["Locomotive_ID"], descending=False)
    )
    # loco_pool = loco_pool.with_columns(cs.categorical().cast(str))
    refuelers = refuelers.with_columns(cs.categorical().cast(str))

    return (
        consist_plan,
        loco_pool,
        refuelers,
        speed_limit_train_sims,
        est_time_nets,
    )


def build_loco_pool_from_model_list(
    consist_plan: pl.DataFrame,
    config: TrainPlannerConfig,
) -> pl.DataFrame:

    unique_consist_plan_locos = consist_plan.unique(subset=["Locomotive_ID", "Locomotive_Type"], maintain_order=True, keep='first')
    loco_pool = unique_consist_plan_locos[["Locomotive_ID", "Locomotive_Type", "Origin_ID"]]
    loco_pool = loco_pool.rename({'Origin_ID':'Node'})
    
    # TODO: check this line


    loco_pool = loco_pool.with_columns(pl.lit(0.0).alias("Arrival_Time"), 
                                       pl.lit(0.0).alias("Servicing_Done_Time"),
                                       pl.lit(0.0).alias("Refueling_Done_Time"),
                                       pl.lit("Ready").alias("Status"),
                                       pl.lit(0.0).alias("SOC_Target_J"),
                                       pl.lit(0.0).alias("Refuel_Duration"),
                                       pl.lit(0.0).alias("Refueler_J_Per_Hr"),
                                       pl.lit(0.0).alias("Refueler_Efficiency"),
                                       pl.lit(0.0).alias("Port_Count"),
                                       pl.lit(0.0).alias("Battery_Headroom_J")  #TODO this is a column that gets added in the regular train planner that is how charging strategy gets enforced.  probably need to rethink how I am doing it here long term.
                                       )
    
    loco_pool = loco_pool.cast({"Locomotive_Type" : pl.Categorical,
                                "Node" : pl.Categorical,
                                })
        # [
        #     "": pl.Series(np.zeros(rows), dtype=pl.Float64),
        #     "Servicing_Done_Time": pl.Series(np.zeros(rows), dtype=pl.Float64),
        #     "Refueling_Done_Time": pl.Series(np.tile(0, rows), dtype=pl.Float64),
        #     "Status": pl.Series(np.tile("Ready", rows), dtype=pl.Categorical),
        #     "SOC_Target_J": pl.Series(np.zeros(rows), dtype=pl.Float64),
        #     "Refuel_Duration": pl.Series(np.zeros(rows), dtype=pl.Float64),
        #     "Refueler_J_Per_Hr": pl.Series(np.zeros(rows), dtype=pl.Float64),
        #     "Refueler_Efficiency": pl.Series(np.zeros(rows), dtype=pl.Float64),
        #     "Port_Count": pl.Series(np.zeros(rows), dtype=pl.UInt32),
        # ]
    

    loco_info_pl = pl.from_pandas(
        config.loco_info.drop(labels="Rust_Loco", axis=1),
        schema_overrides={
            "Locomotive_Type": pl.Categorical,
            "Fuel_Type": pl.Categorical,
        },
    )

    loco_pool = loco_pool.join(loco_info_pl, on="Locomotive_Type")

    return loco_pool
