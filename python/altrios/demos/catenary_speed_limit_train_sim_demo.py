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
        "Manifest_Loaded": 30,
        "Manifest_Empty": 30,
    },
    train_length_meters=None,
    train_mass_kilograms=None,
)

dml: alt.Locomotive = alt.Locomotive.default_dual_mode_loco()
dml.set_save_interval(SAVE_INTERVAL)


# construct a vector of one BEL, one HEL, and several conventional locomotives
loco_vec = (
    []
    + [dml.copy()]
)

# instantiate consist
print("Building `Consist`")
loco_con = alt.Consist(
    loco_vec,
    SAVE_INTERVAL,
)

# Instantiate the intermediate `TrainSimBuilder`
tsb = alt.TrainSimBuilder(
    train_id="0",
    origin_id="Superior",
    destination_id="Minneapolis",
    train_config=train_config,
    loco_con=loco_con,
)

location_map = alt.import_locations(alt.resources_root() / "networks/default_locations.csv")

# Load the network and construct the timed link path through the network.
print("Loading `Network`")
network = alt.Network.from_file(
    alt.resources_root() / "networks/Taconite-NoBalloon.msgpack"
)
network_without_catenary = copy(network)

train_sim_without_cat: alt.SpeedLimitTrainSim = tsb.make_speed_limit_train_sim(
    location_map=location_map,
    save_interval=SAVE_INTERVAL,
)
train_sim_without_cat.set_save_interval(SAVE_INTERVAL)
est_time_net, _consist = alt.make_est_times(train_sim_without_cat, network_without_catenary)

df_est_time_net = (
    pl.DataFrame(est_time_net.to_pydict()["val"])
    .lazy()
    .unnest("link_event")
    .with_columns(
        pl.col("est_type").replace({0: "Arrive", 1: "Clear", 2: "Fake"}).cast(pl.Categorical),
        pl.selectors.contains("idx").replace({0: None}).cast(pl.UInt32)
    )
    .filter(pl.col("est_type") == pl.lit("Arrive"))
    .select("link_idx", "time_sched", "speed", "dist_to_next")
    .unique()
    .sort("time_sched")
    .with_columns(
        pl.col("dist_to_next").cum_sum().alias("total_dist_meters")
    )
    .with_columns(
    ((pl.col("total_dist_meters").shift(1, fill_value=0) % 50_000) < 5_000).alias("has_catenary")
    )
).collect()


timed_link_path = next(
    iter(
        alt.run_dispatch(
            network_without_catenary,
            alt.SpeedLimitTrainSimVec([train_sim_without_cat]),
            [est_time_net],
            False,
            False,
        )
    )
)
print("Running `walk_timed_path`")
train_sim_without_cat.walk_timed_path(
    network=network_without_catenary,
    timed_path=timed_link_path,
)
ts_dict = train_sim_without_cat.to_pydict()
df_without_cat = pl.DataFrame({
    "meters": ts_dict['history']['offset_back_meters'],
    "seconds": ts_dict['history']['time_seconds'],
    "grid_input": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['cat_int']['history']['pwr_elec_in_watts'],
    "catenary_output": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['cat_int']['history']['pwr_elec_out_watts'],
    "fc_input": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['fc']['history']['pwr_fuel_watts'],
    "fc_output": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['fc']['history']['pwr_shaft_watts'],
    "edrive_output": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['edrv']['history']['pwr_mech_prop_out_watts'],
})

uniform_power_limit_watts=50_000_000.0
network_with_catenary = alt.utils.add_catenary_to_network(network, uniform_power_limit_watts)
train_sim_with_cat: alt.SpeedLimitTrainSim = tsb.make_speed_limit_train_sim(
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
    "grid_input": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['cat_int']['history']['pwr_elec_in_watts'],
    "catenary_output": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['cat_int']['history']['pwr_elec_out_watts'],
    "fc_input": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['fc']['history']['pwr_fuel_watts'],
    "fc_output": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['fc']['history']['pwr_shaft_watts'],
    "edrive_output": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['edrv']['history']['pwr_mech_prop_out_watts'],
})

df_without_cat.with_columns(
    pl.col("meters").mul(0.000621371).alias("miles"),
    pl.selectors.ends_with("put").cum_sum().mul(SAVE_INTERVAL/3600.0/1e6)
).write_csv(alt.resources_root() / "demo_data/without_cat2.csv")
df_with_cat.with_columns(
    pl.col("meters").mul(0.000621371).alias("miles"),
    pl.selectors.ends_with("put").cum_sum().mul(SAVE_INTERVAL/3600.0/1e6)
).write_csv(alt.resources_root() / "demo_data/with_cat2.csv")
assert len(ts_dict["history"]) > 1

# %%
