from altrios import sim_manager
from altrios import utilities, defaults
import altrios as alt
from altrios.train_planner import planner_config
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path
ALTRIOS_demo_result_root = alt.resources_root().parent/"demos/results/case study/Traffic Level Sensitivity Result.csv"
result_output = pd.read_csv(ALTRIOS_demo_result_root)
for i in range(1,10):
    num_train_per_day = i
    Nodes = ["Allouez","Hibbing"]
    Node_1 = Nodes[0]
    Node_2 = Nodes[1]
    Train_types = ["Ironore", "Grain"]
    travel_time = 10 #hr
    num_car_per_train = 100
    percent_of_loaded = 0.5
    sets_of_dispatches = 21
    loco_per_train = 4
    trains_per_day_per_node = (len(Nodes) - 1) * len(Train_types)
    num_of_trains = len(Nodes) * (len(Nodes) - 1) * len(Train_types) * num_train_per_day * sets_of_dispatches
    print("Expected number of trains to be simulated: ", num_of_trains)
    loco_consist = "Diesel" # Diesel or mixed # Diesel_Large and BEL are the loco names
    columns = ["Train_ID", 
            "Train_Type", 
            "Locomotive_ID", 
            "Locomotive_Type", 
            "Origin_ID", 
            "Destination_ID", 
            "Cars_Loaded", 
            "Cars_Empty", 
            "Containers_Loaded", 
            "Containers_Empty", 
            "Departure_SOC_J", 
            "Departure_Time_Planned_Hr", 
            "Arrival_Time_Planned_Hr", 
            "Refuel_Start_Time_Planned_Hr", 
            "Refuel_End_Time_Planned_Hr", 
            "Departure_Time_Actual_Hr", 
            "Arrival_Time_Actual_Hr", 
            "TrainSimVec_Index"]
    train_plan = pd.DataFrame(columns=columns)
    train_id = 0
    for iteration in range(sets_of_dispatches):
        print("Iteration: ", iteration)
        for origin in Nodes:
            destination_nodes = [node for node in Nodes if node != origin]
            for destination in destination_nodes:
                for train_type in Train_types:
                    train_id += 1
                    for loco_num in range(loco_per_train):
                        row_number = len(train_plan)
                        loco_id = int(train_id * 10 +loco_num)
                        if (loco_num == loco_per_train - 1) and (loco_consist == "mixed"):
                            loco_type = "BEL"
                        else:
                            loco_type = "Diesel_Large"
                        cars_loaded = np.floor(num_car_per_train * percent_of_loaded) if train_type == "Manifest" else num_car_per_train
                        cars_empty = num_car_per_train - cars_loaded if train_type == "Manifest" else 0
                        containers_loaded = cars_loaded if train_type == "Intermodal" else 0
                        containers_empty = cars_empty if train_type == "Intermodal" else 0
                        departure_soc_j = 8.28691E+11
                        departure_time_planned_hr = (train_id * (24 / trains_per_day_per_node / len(Nodes)) + round(np.random.uniform(-1, 1),1) if train_id != 0 else 0) if loco_num == 0 else train_plan.loc[row_number - 1, "Departure_Time_Planned_Hr"]
                        arrival_time_planned_hr = departure_time_planned_hr + travel_time + round(np.random.uniform(-1, 1),1) if loco_num == 0 else train_plan.loc[row_number - 1, "Arrival_Time_Planned_Hr"]
                        refuel_start_time_planned_hr = arrival_time_planned_hr + np.random.uniform(0, 1) if loco_num == 0 else train_plan.loc[row_number - 1, "Refuel_Start_Time_Planned_Hr"]
                        refuel_end_time_planned_hr = refuel_start_time_planned_hr + np.random.uniform(0, 1) if loco_num == 0 else train_plan.loc[row_number - 1, "Refuel_End_Time_Planned_Hr"]
                        departure_time_actual_hr = departure_time_planned_hr + np.random.uniform(-0.5, 0.5) if loco_num == 0 else train_plan.loc[row_number - 1, "Departure_Time_Actual_Hr"]
                        arrival_time_actual_hr = arrival_time_planned_hr + np.random.uniform(-0.5, 0.5) if loco_num == 0 else train_plan.loc[row_number - 1, "Arrival_Time_Actual_Hr"]
                        train_plan.loc[row_number, :] = [train_id, train_type, loco_id, loco_type, origin, destination, cars_loaded, cars_empty, containers_loaded, containers_empty, departure_soc_j, departure_time_planned_hr, arrival_time_planned_hr, refuel_start_time_planned_hr, refuel_end_time_planned_hr, departure_time_actual_hr, arrival_time_actual_hr, train_id]
                    # print(departure_time_planned_hr)
    print(train_plan)
    cutoff_point_0 = 24 * 7
    cutoff_point_1 = 24 * 14
    train_plan = train_plan.loc[train_plan["Departure_Time_Planned_Hr"] >= cutoff_point_0]
    train_plan = train_plan.loc[train_plan["Arrival_Time_Planned_Hr"] < cutoff_point_1]
    print(train_plan)
    train_plan.to_csv(alt.resources_root() / "demo_data/manual_train_planner_demo/demo_train_consist_plan_test.csv", index=False)



    t0_total = time.perf_counter()

    SHOW_PLOTS = alt.utils.show_plots()

    plot_dir = Path() / "plots"
    # make the dir if it doesn't exist
    plot_dir.mkdir(exist_ok=True)

    # %%
    # Use the same public network, locations, and rolling stock as `sim_manager_demo.py`
    # so this demo does not depend on any proprietary datasets.
    print("Loading `rail_vehicles`")
    rail_vehicles = [
        alt.RailVehicle.from_file(vehicle_file, skip_init=False)
        for vehicle_file in Path(alt.resources_root() / "rolling_stock/").glob("*.yaml")
    ]

    print("Importing `location_map`")
    location_map = alt.import_locations(
        alt.resources_root() / "networks/default_locations.csv"
    )

    print("Loading `network`")
    network = alt.Network.from_file(
        alt.resources_root() / "networks/Taconite-NoBalloon.yaml"
    )

    print("Loading `train_planner_config`")
    train_planner_config = planner_config.TrainPlannerConfig()

    # %%
    # `manual_train_planner_demo` demonstrates replaying an already-planned train
    # consist plan via `sim_manager.main(consist_plan_in=...)`. The example consist
    # plan below was generated with the standard planner on the public Taconite
    # network (the same inputs as `sim_manager_demo.py`) and is committed under
    # `resources/demo_data/` so this demo stays self-contained and free of any
    # proprietary data. In a production workflow `consist_plan_in` would instead come
    # from your own pre-planned source.
    print("Loading example `consist_plan_in`")
    consist_plan_in = pl.read_csv(
        alt.resources_root() / "demo_data/manual_train_planner_demo/demo_train_consist_plan_test.csv"
    )

    # %%
    print("Running `sim_manager.main` in manual mode (replaying `consist_plan_in`)")
    t0_main = time.perf_counter()
    # Because `consist_plan_in` is provided, the planner skips demand generation and
    # scheduling and instead replays the pre-planned trains.
    (
        consist_plan_out,
        loco_pool,
        refuel_facilities,
        grid_emissions_factors,
        nodal_energy_prices,
        speed_limit_train_sims,
        timed_paths,
        consist_plan_out_untrimmed,
    ) = sim_manager.main(
        network=network,
        rail_vehicles=rail_vehicles,
        location_map=location_map,
        consist_plan_in=consist_plan_in,
        train_planner_config=train_planner_config,
        debug=True,
    )


    # %%
    t1_main = time.perf_counter()
    print(f"Elapsed time to run `sim_manager.main()`: {t1_main - t0_main:.3g} s")

    # %%
    t0_train_sims = time.perf_counter()
    speed_limit_train_sims.set_save_interval(100)
    (sims, refuel_sessions) = alt.run_speed_limit_train_sims(
        speed_limit_train_sims=speed_limit_train_sims,
        network=network,
        train_consist_plan_py=consist_plan_out,
        loco_pool_py=loco_pool,
        refuel_facilities_py=refuel_facilities,
        timed_paths=[alt.TimedLinkPath.from_pydict(tp) for tp in timed_paths],
    )
    t1_train_sims = time.perf_counter()
    print(f"Elapsed time to run train sims: {t1_train_sims - t0_train_sims:.3g} s")
    t_train_time = sum([sim["state"]["time_seconds"] for sim in sims.to_pydict()])
    print(f"Total train-seconds simulated: {t_train_time} s")
    print(f"Total train-hours simulated: {t_train_time / 3600} hr")
    print(f"Average train-second per train sim: {t_train_time / len(sims.to_pydict())} s")
    print(f"Average train-hour per train sim: {t_train_time / len(sims.to_pydict()) / 3600} hr")
    row_number = len(result_output)
    result_output.loc[row_number, "num_train_per_day"] = num_train_per_day
    result_output.loc[row_number, "Total Train Time (s)"] = t_train_time
    result_output.loc[row_number, "Total Train Time (hr)"] = t_train_time / 3600
    result_output.loc[row_number, "Average Train Time (s)"] = t_train_time / len(sims.to_pydict())
    result_output.loc[row_number, "Average Train Time (hr)"] = t_train_time / len(sims.to_pydict())/ 3600
    result_output.to_csv(ALTRIOS_demo_result_root, index=False)
    # %%
    t0_summary_sims = time.perf_counter()
    speed_limit_train_sims.set_save_interval(None)
    (summary_sims, summary_refuel_sessions) = alt.run_speed_limit_train_sims(
        speed_limit_train_sims=speed_limit_train_sims,
        network=network,
        train_consist_plan_py=consist_plan_out,
        loco_pool_py=loco_pool,
        refuel_facilities_py=refuel_facilities,
        timed_paths=[alt.TimedLinkPath.from_pydict(tp) for tp in timed_paths],
    )
    t1_summary_sims = time.perf_counter()
    print(
        f"Elapsed time to build and run summary sims: {t1_summary_sims - t0_summary_sims:.3g} s"
    )

    # %%
    t0_tolist = time.perf_counter()
    sims_list = sims.to_pydict()
    t1_tolist = time.perf_counter()
    print(f"Elapsed time to run `to_pydict()`: {t1_tolist - t0_tolist:.3g} s")

    sim0 = sims_list[0]


    # %%

    t0_main = time.perf_counter()
    e_total_fuel_mj = summary_sims.get_energy_fuel_joules(annualize=False) / 1e9
    t1_main = time.perf_counter()

    print(f"Elapsed time to get total fuel energy: {t1_main - t0_main:.3g} s")
    print(f"Total fuel energy used: {e_total_fuel_mj:.3g} GJ")

    v_total_fuel_gal = (
        summary_sims.get_energy_fuel_joules(annualize=False)
        / 1e3
        / defaults.LHV_DIESEL_KJ_PER_KG
        / defaults.RHO_DIESEL_KG_PER_M3
        * utilities.LITER_PER_M3
        * utilities.GALLONS_PER_LITER
    )

    print(f"Total fuel used: {v_total_fuel_gal:.3g} gallons")
    print(f"Total elapsed time: {time.perf_counter() - t0_total} s")
result_output.to_csv(ALTRIOS_demo_result_root, index=False)