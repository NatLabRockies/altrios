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

SAVE_INTERVAL = 1

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
        "Manifest_Loaded": 20,
        "Manifest_Empty": 20,
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
    origin_id="Minneapolis",
    destination_id="Superior",
    train_config=train_config,
    loco_con=loco_con,
)

# Load the network and construct the timed link path through the network.
print("Loading `Network`")
network = alt.Network.from_file(
    alt.resources_root() / "networks/Taconite-NoBalloon.yaml"
)
network_without_catenary = copy(network)
uniform_power_limit_watts=50_000_000.0
network_with_catenary = alt.utils.add_catenary_to_network(network, uniform_power_limit_watts)

location_map = alt.import_locations(
    alt.resources_root() / "networks/default_locations.csv"
)

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
    "grid_input": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['cat_int']['history']['pwr_elec_in_watts'],
    "catenary_output": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['cat_int']['history']['pwr_elec_out_watts'],
    "fc_input": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['fc']['history']['pwr_fuel_watts'],
    "fc_output": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['fc']['history']['pwr_shaft_watts'],
})

train_sim_without_cat: alt.SpeedLimitTrainSim = tsb.make_speed_limit_train_sim(
    location_map=location_map,
    save_interval=SAVE_INTERVAL,
)
train_sim_without_cat.set_save_interval(SAVE_INTERVAL)
est_time_net, _consist = alt.make_est_times(train_sim_without_cat, network_with_catenary)
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
    "grid_input": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['cat_int']['history']['pwr_elec_in_watts'],
    "catenary_output": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['cat_int']['history']['pwr_elec_out_watts'],
    "fc_input": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['fc']['history']['pwr_fuel_watts'],
    "fc_output": ts_dict['loco_con']['loco_vec'][0]['loco_type']['DualModeLoco']['fc']['history']['pwr_shaft_watts'],
})
df_without_cat.write_csv(alt.resources_route() / "demo_data/without_cat.csv")
df_with_cat.write_csv(alt.resources_root() / "demo_data/with_cat.csv")
print(f"Time to simulate: {t1 - t0:.5g}")
raw_fuel_gigajoules = train_sim.get_energy_fuel_joules(False) / 1e9
print(
    f"Total raw fuel used with BEL and HEL buffers active: {raw_fuel_gigajoules:.6g} GJ"
)
corrected_fuel_gigajoules = train_sim.get_energy_fuel_soc_corrected_joules() / 1e9
print(
    f"Total SOC-corrected fuel used with BEL and HEL buffers active: {corrected_fuel_gigajoules:.6g} GJ"
)
assert len(ts_dict["history"]) > 1

# %%
