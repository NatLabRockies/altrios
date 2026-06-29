# %%
import time
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import os
from copy import copy

import altrios as alt
from altrios.demos import plot_util

sns.set_theme()

SHOW_PLOTS = alt.utils.show_plots()

SAVE_INTERVAL = 60

# Build the train config
print("Loading rail vehicles")
rail_vehicle_loaded = alt.RailVehicle.from_file(
    alt.resources_root() / "rolling_stock/Manifest_Loaded.yaml"
)
rail_vehicle_empty = alt.RailVehicle.from_file(
    alt.resources_root() / "rolling_stock/Manifest_Empty.yaml"
)

# https://docs.rs/altrios-core/latest/altrios_core/train/struct.TrainConfig.html
print("Loading `TrainConfig`")
train_config = alt.TrainConfig(
    rail_vehicles=[rail_vehicle_loaded, rail_vehicle_empty],
    n_cars_by_type={
        "Manifest_Loaded": 75,
        "Manifest_Empty": 75,
    },
    train_length_meters=None,
    train_mass_kilograms=None,
)

dml: alt.Locomotive = alt.Locomotive.default_dual_mode_loco()
dml.set_save_interval(SAVE_INTERVAL)


# construct a vector of one BEL, one HEL, and several conventional locomotives
loco_vec = (
 [dml.copy(), dml.copy(), dml.copy(), dml.copy(), dml.copy(), dml.copy(), dml.copy(), dml.copy(), dml.copy(), dml.copy(),]
)

# instantiate consist
print("Building `Consist`")
loco_con = alt.Consist(
    loco_vec,
    SAVE_INTERVAL,
)
location_map = alt.import_locations(alt.resources_root() / "networks/spokane_lethbridge_with_sidings_locations.csv")
# Load the network and construct the timed link path through the network.
print("Loading `Network`")
network = alt.Network.from_file(
    alt.resources_root() / "networks/spokane_lethbridge_with_sidings.yaml"
)
network_without_catenary = copy(network)

# Instantiate the intermediate `TrainSimBuilder`
tsb1 = alt.TrainSimBuilder(
    train_id="0",
    origin_id="Spokane",
    destination_id="Lethbridge",
    train_config=train_config,
    loco_con=loco_con,
)
train_sim_without_cat1: alt.SpeedLimitTrainSim = tsb1.make_speed_limit_train_sim(
    location_map=location_map,
    save_interval=SAVE_INTERVAL,
)
train_sim_without_cat1.set_save_interval(SAVE_INTERVAL)
est_time_net_1, _consist = alt.make_est_times(train_sim_without_cat1, network_without_catenary)
tsb2 = alt.TrainSimBuilder(
    train_id="0",
    origin_id="Spokane",
    destination_id="Lethbridge",
    train_config=train_config,
    loco_con=loco_con,
)
train_sim_without_cat2: alt.SpeedLimitTrainSim = tsb2.make_speed_limit_train_sim(
    location_map=location_map,
    save_interval=SAVE_INTERVAL,
)
train_sim_without_cat2.set_save_interval(SAVE_INTERVAL)
est_time_net_2, _consist = alt.make_est_times(train_sim_without_cat2, network_without_catenary)
uniform_power_limit_watts=50_000_000.0
# network_with_catenary = alt.utils.add_catenary_high_grade(
#     network,
#     uniform_power_limit_watts,
#     grade_threshold = 0.007
# )
network_with_catenary = alt.utils.add_catenary_even_spacing(
    network,
    uniform_power_limit_watts,
    spacing_meters = (30_000.00, 30_000.00),
    est_time_nets=[est_time_net_1],# est_time_net_2],
)
timed_link_path = next(
    iter(
        alt.run_dispatch(
            network_without_catenary,
            alt.SpeedLimitTrainSimVec([train_sim_without_cat1]),
            [est_time_net_1],
            False,
            False,
        )
    )
)
print("Running `walk_timed_path`")
train_sim_without_cat1.walk_timed_path(
    network=network_without_catenary,
    timed_path=timed_link_path,
)
ts_dict = train_sim_without_cat1.to_pydict()
df_without_cat = pl.DataFrame({
    "meters": ts_dict['history']['offset_back_meters'],
    "seconds": ts_dict['history']['time_seconds'],
    "grid_input": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['cat_int']['history']['pwr_elec_in_watts'],
    "catenary_output": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['cat_int']['history']['pwr_elec_out_watts'],
    "fc_input": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['fc']['history']['pwr_fuel_watts'],
    "fc_output": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['fc']['history']['pwr_shaft_watts'],
    "edrive_output": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['edrv']['history']['pwr_mech_prop_out_watts'],
})
tsb1_with_cat = alt.TrainSimBuilder(
    train_id="0",
    origin_id="Spokane",
    destination_id="Lethbridge",
    train_config=train_config,
    loco_con=loco_con,
)
train_sim_with_cat: alt.SpeedLimitTrainSim = tsb1_with_cat.make_speed_limit_train_sim(
    location_map=location_map,
    save_interval=SAVE_INTERVAL,
)
train_sim_with_cat.set_save_interval(SAVE_INTERVAL)
est_time_net, _consist = alt.make_est_times(train_sim_with_cat, network_with_catenary)
timed_link_path = next(
    iter(
        alt.run_dispatch(
            network_with_catenary,
            alt.SpeedLimitTrainSimVec([train_sim_with_cat]),
            [est_time_net],
            False,
            False,
        )
    )
)
print("Running `walk_timed_path`")
train_sim_with_cat.walk_timed_path(
    network=network_with_catenary,
    timed_path=timed_link_path,
)
ts_dict = train_sim_with_cat.to_pydict()
df_with_cat = pl.DataFrame({
    "meters": ts_dict['history']['offset_back_meters'],
    "seconds": ts_dict['history']['time_seconds'],
    "grade": ts_dict['history']['grade_back'],
    "elevation": ts_dict['history']['elev_back_meters'],
    "grade_front": ts_dict['history']['grade_front'],
    "elevation_front": ts_dict['history']['elev_front_meters'],
    "grid_input": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['cat_int']['history']['pwr_elec_in_watts'],
    "catenary_output": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['cat_int']['history']['pwr_elec_out_watts'],
    "fc_input": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['fc']['history']['pwr_fuel_watts'],
    "fc_output": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['fc']['history']['pwr_shaft_watts'],
    "edrive_output": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['edrv']['history']['pwr_mech_prop_out_watts'],
})

#df_without_cat.with_columns(
#    pl.col("meters").mul(0.000621371).alias("miles"),
#    pl.selectors.ends_with("put").cum_sum().mul(SAVE_INTERVAL/3600.0/1e6)
#).write_csv(alt.resources_root() / "demo_data/without_cat2.csv")
df_with_cat.with_columns(
    pl.col("meters").mul(alt.utilities.MI_PER_KM * 1.0e-3).alias("miles"),
    pl.selectors.ends_with("put").cum_sum().mul(SAVE_INTERVAL/3600.0/1e6).name.suffix("_cumulative")
).write_csv(alt.resources_root() / "demo_data/with_cat_30km_eastbound.csv")
assert len(ts_dict["history"]) > 1

# %%
