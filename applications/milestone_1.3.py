# %%
import polars as pl
import pandas as pd
import seaborn as sns
from pathlib import Path
from copy import copy
import altrios as alt
from altrios import sim_manager
from altrios.train_planner import planner_config

sns.set_theme()
SHOW_PLOTS = alt.utils.show_plots()
SAVE_INTERVAL = 60

rs_dir = Path(__file__).parents[1] / "python" / "altrios" / "resources" / "rolling_stock/"
network_dir = Path(__file__).parents[1] / "python" / "altrios" / "resources" / "networks/"
corridor_data_dir = Path("C:/Users/mbruchon/OneDrive - NREL/Data/in/ALTRIOS_Corridor/RailState/")

print("Loading `rail_vehicles`")
rail_vehicles = [
    alt.RailVehicle.from_file(rv_file, skip_init=False)
    for rv_file in Path(rs_dir).glob("*.yaml")
]

network = alt.Network.from_file(network_dir / "Spokane_Lethbridge" / "spokane_lethbridge.msgpack")
location_map = alt.import_locations(network_dir / "Spokane_Lethbridge" / "spokane_lethbridge_locations.csv")
consist_plan_in = pl.read_csv(Path(__file__).parent / "catenary_optimization" / "train_consist_plan_spokane_lethbridge_clean.csv")
 
network_dict_for_waypoints = network.to_pydict()[1]
waypoint_indices = pl.concat([
    pl.DataFrame({"idx": [entry['idx_curr'] for entry in network_dict_for_waypoints if entry['osm_id']=='1316919787']}).with_columns(pl.lit("Moyie Springs, ID").alias("waypoint")),
    pl.DataFrame({"idx": [entry['idx_curr'] for entry in network_dict_for_waypoints if entry['osm_id'] in ['88748322','88740985']]}).with_columns(pl.lit("Jaffray, BC").alias("waypoint")),
    pl.DataFrame({"idx": [entry['idx_curr'] for entry in network_dict_for_waypoints if entry['osm_id']=='88740159']}).with_columns(pl.lit("Fernie, BC").alias("waypoint")),
    pl.DataFrame({"idx": [entry['idx_curr'] for entry in network_dict_for_waypoints if entry['osm_id'] in ['100087805','100087827']]}).with_columns(pl.lit("Sparwood, BC").alias("waypoint")),
    pl.DataFrame({"idx": [entry['idx_curr'] for entry in network_dict_for_waypoints if entry['osm_id']=='1337667723']}).with_columns(pl.lit("Coalhurst, AB").alias("waypoint"))
])

train_planner_config = planner_config.TrainPlannerConfig(
    loco_info= pd.DataFrame({
        "Dual_Mode": {
            "Fuel_Type": "Diesel",
            "Min_Servicing_Time_Hr": 3.0,
            "Rust_Loco": alt.Locomotive.default_dual_mode_loco(),
            "Cost_USD": alt.defaults.DIESEL_LOCO_COST_USD,
            "Lifespan_Years": alt.defaults.LOCO_LIFESPAN
            },
        }).transpose().reset_index(names='Locomotive_Type'))
refueler_info = train_planner_config.refueler_info[train_planner_config.refueler_info["Refueler_Type"] == "Diesel_Fueler"].copy()
refueler_info["Locomotive_Type"] = "Dual_Mode"
train_planner_config.refueler_info.update(refueler_info)

network_dict = copy(network.to_pydict())
network_dict = alt.utilities.set_network_max_speed(network_dict, max_speed_mph = 40.0)
network_speed_capped = alt.Network.from_pydict(network_dict)

for scaler in [0.595]: #[i / 200 for i in range(120, 99, -1)]:
    scaler_string = f"{int(scaler*100):03d}"

    network_dict_adjusted = copy(network_speed_capped.to_pydict())
    network_dict_adjusted = alt.utilities.scale_network_speed_limits(network_dict_adjusted, scaler)
    network_speed_fixed = alt.Network.from_pydict(network_dict_adjusted)
    network_speed_fixed.to_file(network_dir / f"spokane_lethbridge_with_sidings_speed_reduced_{scaler_string}.yaml")
    x = 5
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
        network=network_speed_fixed,
        rail_vehicles=rail_vehicles,
        location_map=location_map,
        consist_plan_in=consist_plan_in,
        train_planner_config=train_planner_config,
        walk_timed_paths=True,
        debug=True,
    )

    waypoint_stats = []
    for slts in speed_limit_train_sims.to_pydict():
        waypoint_stats_slts = (pl.DataFrame(slts['history'])
            .join(waypoint_indices, left_on="link_idx_front", right_on="idx", how="inner")
            .group_by("waypoint")
            .agg(
                pl.col("time_seconds").mean().alias("Mean_Time_Sec"),
                pl.col("offset_back_meters").mean().alias("Mean_Offset_Back_Meters"),
                pl.col("speed_meters_per_second").mean().alias("Mean_Speed_Meters_Per_Second")
            )
            .with_columns(pl.lit(slts['train_id']).cast(pl.UInt32).alias("Train_ID"))
        )
        waypoint_stats.append(waypoint_stats_slts)

    consist_plan_details_1 = (consist_plan_in
        .select(
            pl.col("Train_ID").alias("Train_ID_Raw"),
            pl.col("Train_ID").rank("dense").alias("Train_ID"),
            pl.col("Origin_ID"),
            pl.col("Destination_ID"),
            pl.col("Arrival_Time_Planned_Hr").sub("Departure_Time_Planned_Hr").alias("moyie_coalhurst_ground_truth_hr"),
        )
        .unique()
    )
    consist_plan_details_2 = (consist_plan_out
        .select(
            pl.col("Train_ID"),
            pl.col("Origin_ID"),
            pl.col("Destination_ID"),
            pl.col("Arrival_Time_Actual_Hr").sub("Departure_Time_Actual_Hr").alias("spokane_lethbridge_travel_time_hr")
        )
        .unique()
    )
    consist_plan_details = (consist_plan_details_1
        .join(consist_plan_details_2, how="inner", on=["Train_ID", "Origin_ID", "Destination_ID"])
        .join(pl.concat(waypoint_stats, how="diagonal_relaxed"), how="inner", on="Train_ID")
        .with_columns(
            pl.lit(scaler).alias("Speed_Scaler"),
            pl.col("Mean_Time_Sec").sub(pl.col("Mean_Time_Sec").min().over("Train_ID")).truediv(3600.0).alias("Elapsed_Time_Hr")
        )
    )
    consist_plan_details.write_csv(network_dir / f"consist_plan_details_speed_scaler_{scaler_string}_4.csv")
