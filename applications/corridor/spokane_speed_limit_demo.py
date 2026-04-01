# %%
# Exact copy of speed_limit_train_sim_demo.py adapted for Spokane–Lethbridge.
# Changes from original (marked with # SPOKANE):
#   1. Network file
#   2. Locations file
#   3. origin_id / destination_id
#   4. More locos (mountain grades need extra traction)
#   5. OVERRIDE_SSTS_INPUTS always True → saves Spokane link_path & speed_trace
#   6. Fixed tolist() → to_pydict() (API change in v0.3)
#   7. ENABLE_ASSERTS=False (no reference data for Spokane)
#   8. Headless matplotlib backend + save figures to networks/

import time
import matplotlib
matplotlib.use("Agg")  # SPOKANE: headless
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import os
import sys
from copy import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import altrios as alt
from altrios.demos import plot_util
sns.set_theme()

SHOW_PLOTS = False  # SPOKANE: no display
SAVE_INTERVAL = 1

# SPOKANE: paths
_REPO = Path(__file__).resolve().parents[2]
_NETWORKS = _REPO / "networks"
SPOKANE_NETWORK    = _NETWORKS / "spokane_lethbridge_with_sidings.yaml"
SPOKANE_LOCATIONS  = _NETWORKS / "spokane_lethbridge_with_sidings_locations.csv"
SPOKANE_LINK_PATH  = _NETWORKS / "spokane_link_path.csv"
SPOKANE_SPEED_TRACE = _NETWORKS / "spokane_speed_trace.csv"
SPOKANE_REF_DIR    = _NETWORKS / "ref_data/speed_limit_train_sim_demo"

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
# instantiate electric drivetrain (motors and any gearboxes)
# https://docs.rs/altrios-core/latest/altrios_core/consist/locomotive/powertrain/electric_drivetrain/struct.ElectricDrivetrain.html
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
bel_dict = bel.to_pydict()
bel_pt_cntrl = bel_dict["loco_type"]["BatteryElectricLoco"]["pt_cntrl"]["RGWDB"]
bel_pt_cntrl["speed_soc_disch_buffer_meters_per_second"] = 10
bel_pt_cntrl["speed_soc_regen_buffer_meters_per_second"] = 15
bel_dict = copy(bel_dict)
bel_dict["loco_type"]["BatteryElectricLoco"]["pt_cntrl"]["RGWDB"] = bel_pt_cntrl
bel = alt.Locomotive.from_pydict(bel_dict)

bel_new_pt_cntrl = copy(bel_pt_cntrl)
# effectively turn off the buffers
bel_new_pt_cntrl["speed_soc_disch_buffer_meters_per_second"] = 0
bel_new_pt_cntrl["speed_soc_regen_buffer_meters_per_second"] = 100
bel_new_dict = copy(bel_dict)
bel_new_dict["loco_type"]["BatteryElectricLoco"]["pt_cntrl"]["RGWDB"] = bel_new_pt_cntrl
bel_sans_buffers = alt.Locomotive.from_pydict(bel_new_dict)

hel: alt.Locomotive = alt.Locomotive.default_hybrid_electric_loco()
hel_dict = hel.to_pydict()
hel_pt_cntrl = hel_dict["loco_type"]["HybridLoco"]["pt_cntrl"]["RGWDB"]
hel_pt_cntrl["speed_soc_disch_buffer_meters_per_second"] = 0
hel_pt_cntrl["speed_soc_regen_buffer_meters_per_second"] = 100
hel_dict["loco_type"]["HybridLoco"]["pt_cntrl"]["RGWDB"] = hel_pt_cntrl
hel = alt.Locomotive.from_pydict(hel_dict)

hel_new_pt_cntrl = copy(hel_pt_cntrl)
# effectively turn off the buffers
hel_new_pt_cntrl["speed_soc_disch_buffer_meters_per_second"] = 15
hel_new_pt_cntrl["speed_soc_regen_buffer_meters_per_second"] = 15
hel_new_dict = copy(hel_dict)
hel_new_dict["loco_type"]["HybridLoco"]["pt_cntrl"]["RGWDB"] = hel_new_pt_cntrl
hel_sans_buffers = alt.Locomotive.from_pydict(hel_new_dict)

# construct a vector of one BEL, one HEL, and several conventional locomotives
# SPOKANE: +2 extra conventional locos for mountain-grade traction
loco_vec = (
    []
    + [hel.copy()]
    + [alt.Locomotive.default()] * 3  # SPOKANE: 3 instead of 1
)

# construct a vector of one BEL, one HEL, and several conventional locomotives
loco_vec_sans_buffers = (
    []
    + [hel_sans_buffers.copy()]
    + [alt.Locomotive.default()] * 3  # SPOKANE: 3 instead of 1
)

# instantiate consist
print("Building `Consist`")
loco_con = alt.Consist(
    loco_vec,
    SAVE_INTERVAL,
)

# instantiate consist
loco_con_sans_buffers = alt.Consist(
    loco_vec_sans_buffers,
    SAVE_INTERVAL,
)

# Instantiate the intermediate `TrainSimBuilder`
tsb = alt.TrainSimBuilder(
    train_id="0",
    origin_id="Spokane",       # SPOKANE
    destination_id="Lethbridge",  # SPOKANE
    train_config=train_config,
    loco_con=loco_con,
)

# Instantiate the intermediate `TrainSimBuilder`
tsb_sans_buffers = alt.TrainSimBuilder(
    train_id="0",
    origin_id="Spokane",       # SPOKANE
    destination_id="Lethbridge",  # SPOKANE
    train_config=train_config,
    loco_con=loco_con_sans_buffers,
)

# Load the network and construct the timed link path through the network.
print("Loading `Network`")
network = alt.Network.from_file(SPOKANE_NETWORK)  # SPOKANE

location_map = alt.import_locations(SPOKANE_LOCATIONS)  # SPOKANE

train_sim: alt.SpeedLimitTrainSim = tsb.make_speed_limit_train_sim(
    location_map=location_map,
    save_interval=SAVE_INTERVAL,
)
train_sim.set_save_interval(SAVE_INTERVAL)

train_sim_sans_buffers: alt.SpeedLimitTrainSim = (
    tsb_sans_buffers.make_speed_limit_train_sim(
        location_map=location_map,
        save_interval=SAVE_INTERVAL,
    )
)
train_sim_sans_buffers.set_save_interval(SAVE_INTERVAL)

print("Running `make_est_times`")
est_time_net, _consist = alt.make_est_times(train_sim, network)

est_time_net_sans_buffers, _consist = alt.make_est_times(
    train_sim_sans_buffers, network
)

print("Running `run_dispatch`")
timed_link_path = next(
    iter(
        alt.run_dispatch(
            network,
            alt.SpeedLimitTrainSimVec([train_sim]),
            [est_time_net],
            False,
            False,
        )
    )
)

timed_link_path_sans_buffers = next(
    iter(
        alt.run_dispatch(
            network,
            alt.SpeedLimitTrainSimVec([train_sim_sans_buffers]),
            [est_time_net_sans_buffers],
            False,
            False,
        )
    )
)


# SPOKANE: always save inputs for set_speed demo (to Spokane-specific paths)
# Fixed: tolist() → to_pydict() (API changed in v0.3)
print("Saving inputs for `set_speed_train_sim_demo.py`")
link_path = alt.LinkPath.from_pydict(
    [entry["link_idx"] for entry in timed_link_path.to_pydict()]  # SPOKANE fix
)
link_path.to_csv_file(SPOKANE_LINK_PATH)  # SPOKANE

t0 = time.perf_counter()
print("Running `walk_timed_path`")
train_sim.walk_timed_path(
    network=network,
    timed_path=timed_link_path,
)
t1 = time.perf_counter()

ts_dict = train_sim.to_pydict()

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

t0 = time.perf_counter()
train_sim_sans_buffers.walk_timed_path(
    network=network,
    timed_path=timed_link_path_sans_buffers,
)
t1 = time.perf_counter()

print(f"\nTime to simulate without buffers: {t1 - t0:.5g}")
raw_fuel_sans_buffers_gigajoules = (
    train_sim_sans_buffers.get_energy_fuel_joules(False) / 1e9
)
print(
    f"Total raw fuel used with BEL and HEL buffers inactive: {raw_fuel_sans_buffers_gigajoules:.6g} GJ"
)
corrected_fuel_sans_buffers_gigajoules = (
    train_sim_sans_buffers.get_energy_fuel_soc_corrected_joules() / 1e9
)
print(
    f"Total SOC-corrected fuel used with BEL and HEL buffers inactive: {corrected_fuel_sans_buffers_gigajoules:.6g} GJ"
)
assert len(train_sim_sans_buffers.to_pydict()["history"]) > 1

savings_raw = (
    -(raw_fuel_gigajoules - raw_fuel_sans_buffers_gigajoules)
    / raw_fuel_sans_buffers_gigajoules
    * 100
)
print(f"\nRaw fuel savings from buffers: {savings_raw:.5g}%")
savings_soc_corrected = (
    -(corrected_fuel_gigajoules - corrected_fuel_sans_buffers_gigajoules)
    / corrected_fuel_sans_buffers_gigajoules
    * 100
)
print(f"SOC-corrected fuel savings from buffers: {savings_soc_corrected:.5g}%")

# SPOKANE: save speed trace for set_speed demo
speed_trace = alt.SpeedTrace(
    ts_dict["history"]["time_seconds"],
    ts_dict["history"]["speed_meters_per_second"],
)
speed_trace.to_csv_file(SPOKANE_SPEED_TRACE)  # SPOKANE
print(f"Saved speed trace → {SPOKANE_SPEED_TRACE}")

fig0, ax0 = plot_util.plot_train_level_powers(train_sim, "With Buffers")
fig1, ax1 = plot_util.plot_train_network_info(train_sim, "With Buffers")
fig2, ax2 = plot_util.plot_consist_pwr(train_sim, "With Buffers")
fig3, ax3 = plot_util.plot_hel_pwr_and_soc(train_sim, "With Buffers")

fig0_sans_buffers, ax0_sans_buffers = plot_util.plot_train_level_powers(
    train_sim_sans_buffers, "Without Buffers"
)
fig1_sans_buffers, ax1_sans_buffers = plot_util.plot_train_network_info(
    train_sim_sans_buffers, "Without Buffers"
)
fig2_sans_buffers, ax2_sans_buffers = plot_util.plot_consist_pwr(
    train_sim_sans_buffers, "Without Buffers"
)
fig3_sans_buffers, ax3_sans_buffers = plot_util.plot_hel_pwr_and_soc(
    train_sim_sans_buffers, "Without Buffers"
)

# SPOKANE: save figures instead of showing
_NETWORKS.mkdir(parents=True, exist_ok=True)
for fig, name in [
    (fig0, "train_level_powers_with_buffers"),
    (fig1, "train_network_info_with_buffers"),
    (fig2, "consist_pwr_with_buffers"),
    (fig3, "hel_pwr_soc_with_buffers"),
    (fig0_sans_buffers, "train_level_powers_without_buffers"),
    (fig1_sans_buffers, "train_network_info_without_buffers"),
    (fig2_sans_buffers, "consist_pwr_without_buffers"),
    (fig3_sans_buffers, "hel_pwr_soc_without_buffers"),
]:
    fig.savefig(_NETWORKS / f"spokane_{name}.png", dpi=150, bbox_inches="tight")
plt.close("all")
print(f"Figures saved to {_NETWORKS}/")

# SPOKANE: skip assertions (no reference data for this route)
# ENABLE_ASSERTS = False
# (reference data would need to be generated with ENABLE_REF_OVERRIDE first)

# %%
