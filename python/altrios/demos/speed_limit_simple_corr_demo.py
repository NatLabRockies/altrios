# %%
"""
SetSpeedTrainSim over a simple, hypothetical corridor
"""

import time
import matplotlib.pyplot as plt
import seaborn as sns
from altrios.demos import plot_util

import altrios as alt

sns.set_theme()

SHOW_PLOTS = alt.utils.show_plots()
SAVE_INTERVAL = 1

# Build the train config
rail_vehicle_loaded = alt.RailVehicle.from_file(
    alt.resources_root() / "rolling_stock/Manifest_Loaded.yaml"
)
rail_vehicle_empty = alt.RailVehicle.from_file(
    alt.resources_root() / "rolling_stock/Manifest_Empty.yaml"
)

# https://docs.rs/altrios-core/latest/altrios_core/train/struct.TrainConfig.html
train_config = alt.TrainConfig(
    rail_vehicles=[rail_vehicle_loaded, rail_vehicle_empty],
    n_cars_by_type={
        "Manifest_Loaded": 50,
        "Manifest_Empty": 50,
    },
    train_length_meters=None,
    train_mass_kilograms=None,
)

# Build the locomotive consist model
# instantiate battery model
# https://docs.rs/altrios-core/latest/altrios_core/consist/locomotive/powertrain/reversible_energy_storage/struct.ReversibleEnergyStorage.html#
res = alt.ReversibleEnergyStorage.from_file(
    alt.resources_root()
    / "powertrains/reversible_energy_storages/Kokam_NMC_75Ah_flx_drive.yaml"
)

edrv = alt.ElectricDrivetrain(
    pwr_out_frac_interp=[0.0, 1.0],
    eta_interp=[0.98, 0.98],
    pwr_out_max_watts=5e9,
    save_interval=SAVE_INTERVAL,
)

bel: alt.Locomotive = alt.Locomotive.from_pydict(
    {
        "loco_type": {
            "BatteryElectricLoco": {
                "res": res.to_pydict(),
                "edrv": edrv.to_pydict(),
            }
        },
        "pwr_aux_offset_watts": 8.55e3,
        "pwr_aux_traction_coeff": 540.0e-6,
        "force_max_newtons": 667.2e3,
        "mass_kilograms": alt.LocoParams.default().to_pydict()["mass_kilograms"],
        "save_interval": SAVE_INTERVAL,
    }
)

hel: alt.Locomotive = alt.Locomotive.default_hybrid_electric_loco()

# construct a vector of one BEL and several conventional locomotives
loco_vec = [bel.copy()] + [hel.copy()] + [alt.Locomotive.default()] * 7

# instantiate consist
loco_con = alt.Consist(loco_vec)

# Instantiate the intermediate `TrainSimBuilder`
tsb = alt.TrainSimBuilder(
    train_id="0",
    origin_id="A",
    destination_id="B",
    train_config=train_config,
    loco_con=loco_con,
)

# Load the network and construct the timed link path through the network.
network = alt.Network.from_file(
    alt.resources_root() / "networks/simple_corridor_network.yaml"
)

location_map = alt.import_locations(
    alt.resources_root() / "networks/simple_corridor_locations.csv"
)
train_sim: alt.SetSpeedTrainSim = tsb.make_speed_limit_train_sim(
    location_map=location_map,
    save_interval=1,
)
train_sim.set_save_interval(SAVE_INTERVAL)
est_time_net, _consist = alt.make_est_times(train_sim, network)

timed_link_path = alt.run_dispatch(
    network,
    alt.SpeedLimitTrainSimVec([train_sim]),
    [est_time_net],
    False,
    False,
)[0]

t0 = time.perf_counter()
train_sim.walk_timed_path(
    network=network,
    timed_path=timed_link_path,
)
t1 = time.perf_counter()
print(f"Time to simulate: {t1 - t0:.5g}")
ts_dict = train_sim.to_pydict()
assert len(ts_dict["history"]) > 1

# pull out solved locomotive for plotting convenience
loco0: alt.Locomotive = next(iter(ts_dict["loco_con"]["loco_vec"]))
loco0_type = next(iter(loco0["loco_type"].values()))

fig0, ax0 = plot_util.plot_train_level_powers(train_sim, "With Buffers")
fig1, ax1 = plot_util.plot_train_network_info(train_sim, "With Buffers")
fig2, ax2 = plot_util.plot_consist_pwr(train_sim, "With Buffers")
fig3, ax3 = plot_util.plot_hel_pwr_and_soc(train_sim, "With Buffers")
fig4, ax4 = plot_util.plot_bel_pwr_and_soc(train_sim, "With Buffers")

if SHOW_PLOTS:
    plt.show()
