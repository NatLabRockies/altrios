# %%
import altrios as alt
from altrios.demos import plot_util
import time
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import os
from copy import copy
import sys

corridor = 'Amarillo-FortWorth'
origin = 'Amarillo'
destination = 'Fort Worth - BNSF'
location_root = "double_sidings/Garrett/Amarillo_FortWorth -  No Savgol/Network Locations.csv"
network_root = "double_sidings/Garrett/Amarillo_FortWorth -  No Savgol/Network.yaml"

# corridor = 'Amarillo-FortWorth'
# origin = 'Amarillo'
# destination = 'FortWorth'
# location_root = "double_sidings/corridor/Amarillo-FortWorth/locations.csv"
# network_root = "double_sidings/corridor/Amarillo-FortWorth/Network.yaml"

# corridor = 'Clovis-Flagstaff'
# origin = 'Clovis'
# destination = 'Flagstaff'
# location_root = "double_sidings/corridor/Clovis-Flagstaff/locations.csv"
# network_root = "double_sidings/corridor/Clovis-Flagstaff/Network.yaml"

# corridor = 'Clovis-Flagstaff'
# origin = 'Clovis'
# destination = 'FLagstaff'
# location_root = "double_sidings/Garrett/Flagstaff_Clovis - No Savgol/Network Locations.csv"
# network_root = "double_sidings/Garrett/Flagstaff_Clovis - No Savgol/Network.yaml"

# corridor = 'Barstow-LongBeach'
# origin = 'Barstow'
# destination = 'LongBeach'
# location_root = f"double_sidings/corridor/{corridor}/locations.csv"
# network_root = f"double_sidings/corridor/{corridor}/Network.yaml"

# corridor = 'Barstow-LongBeach'
# origin = 'Barstow'
# destination = 'POLA'
# location_root = "double_sidings/Garrett/POLA_Barstow_Eastward/Network Locations.csv"
# network_root = "double_sidings/Garrett/POLA_Barstow_Eastward/Network.yaml"

# define train batch size and locomotives
if len(sys.argv) > 1:
    L = int(sys.argv[1])   # train length (# cars)
    n_locos = int(sys.argv[2])
else:
    L = 20
    n_locos = 1

sns.set_theme()
SHOW_PLOTS = alt.utils.show_plots()
SAVE_INTERVAL = 1

# Build the train config
print("Loading rail vehicles")
rail_vehicle_loaded = alt.RailVehicle.from_file(
    alt.resources_root() / "rolling_stock/Intermodal_Loaded.yaml"
)
rail_vehicle_empty = alt.RailVehicle.from_file(
    alt.resources_root() / "rolling_stock/Intermodal_Empty.yaml"
)

# https://docs.rs/altrios-core/latest/altrios_core/train/struct.TrainConfig.html
print("Loading `TrainConfig`")
train_config = alt.TrainConfig(
    rail_vehicles=[rail_vehicle_loaded, rail_vehicle_empty],
    n_cars_by_type={
        "Intermodal_Loaded": L,
        "Intermodal_Empty": 0,
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
loco_vec = (
    []
    # + [hel.copy()]
    + [alt.Locomotive.default()] * n_locos    # conventional trains
)

# instantiate consist
print("Building `Consist`")
loco_con = alt.Consist(
    loco_vec,
    SAVE_INTERVAL,
)


tsb = alt.TrainSimBuilder(
    train_id="0",
    origin_id=f"{origin}",
    destination_id=f"{destination}",
    train_config=train_config,
    loco_con=loco_con,
)

# Load the network and construct the timed link path through the network.
print("Loading `Network`")
location_map = alt.import_locations(alt.resources_root() / location_root)
network = alt.Network.from_file(alt.resources_root() / network_root)

train_sim: alt.SpeedLimitTrainSim = tsb.make_speed_limit_train_sim(
    location_map=location_map,
    save_interval=SAVE_INTERVAL,
)
train_sim.set_save_interval(SAVE_INTERVAL)

print("Running `make_est_times`")
est_time_net, _consist = alt.make_est_times(train_sim, network)

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

# whether to override files used by set_speed_train_sim_demo.py
OVERRIDE_SSTS_INPUTS = os.environ.get("OVERRIDE_SSTS_INPUTS", "false").lower() == "true"
if OVERRIDE_SSTS_INPUTS:
    print("Overriding files used by `set_speed_train_sim_demo.py`")
    link_path = alt.LinkPath([x.link_idx for x in timed_link_path.tolist()])
    link_path.to_csv_file(alt.resources_root() / "demo_data/link_path.csv")

t0 = time.perf_counter()
print("Running `walk_timed_path`")
train_sim.walk_timed_path(
    network=network,
    timed_path=timed_link_path,
)
t1 = time.perf_counter()

ts_dict = train_sim.to_pydict()
print(f"Travel time: {ts_dict['state']['time_seconds']}s")

print(f"Time to simulate: {t1 - t0:.5g}")
raw_fuel_gigajoules = train_sim.get_energy_fuel_joules(False) / 1e9
print(f"Total raw fuel used with BEL and HEL buffers active: {raw_fuel_gigajoules:.6g} GJ")
corrected_fuel_gigajoules = train_sim.get_energy_fuel_soc_corrected_joules() / 1e9
print(f"Total SOC-corrected fuel used with BEL and HEL buffers active: {corrected_fuel_gigajoules:.6g} GJ")

assert len(ts_dict["history"]) > 1

t1 = time.perf_counter()

# Uncomment the following lines to overwrite `set_speed_train_sim_demo.py` `speed_trace`
if OVERRIDE_SSTS_INPUTS:
    speed_trace = alt.SpeedTrace(
        ts_dict["history"]["time_seconds"],
        ts_dict["history"]["speed_meters_per_second"],
    )
    speed_trace.to_csv_file(alt.resources_root() / "demo_data/speed_trace.csv")

fig0, ax0 = plot_util.plot_train_level_powers(train_sim, "With Buffers")
fig1, ax1 = plot_util.plot_train_network_info(train_sim, "With Buffers")
fig2, ax2 = plot_util.plot_consist_pwr(train_sim, "With Buffers")

# -------------------------------------
# plot the elevations and grades
# -------------------------------------

import matplotlib.pyplot as plt
import pandas as pd

elevation_meters = ts_dict['history']['elev_front_meters']
grade = ts_dict['history']['grade_front']
total_dist_meters = ts_dict['history']['total_dist_meters']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax1.plot(total_dist_meters, elevation_meters, color='blue', label='Elevation (m)')
ax1.set_ylabel("Elevation (m)")
ax1.set_title("Elevation Profile")
ax1.legend()

ax2.plot(total_dist_meters, grade, color='red', label='Grade')
ax2.set_ylabel("Grade")
ax2.set_xlabel("Distance (m)")
ax2.set_title("Grade Profile")
ax2.legend()


# distance vs. elevation & grade
df = pd.DataFrame({
    "total_dist_meters": total_dist_meters,
    "elevation_meters": elevation_meters,
    "grade": grade
})
df.to_excel(f"/Users/qianqiantong/Desktop/elevation_{corridor}_grade_data.xlsx", index=False)
print("saved to desktop!")

# force
max_idx, max_time, max_res = plot_util.plot_total_resistance(train_sim)

# -------------------------------------
# -------------------------------------

if SHOW_PLOTS:
    plt.tight_layout()
    plt.show()

# Impact of sweep of battery capacity
# whether to run assertions, enabled by default
ENABLE_ASSERTS = os.environ.get("ENABLE_ASSERTS", "true").lower() == "true"
# whether to override reference files used in assertions, disabled by default
ENABLE_REF_OVERRIDE = os.environ.get("ENABLE_REF_OVERRIDE", "false").lower() == "true"
# directory for reference files for checking sim results against expected results
ref_dir = alt.resources_root() / "demo_data/speed_limit_train_sim_demo/"

if ENABLE_REF_OVERRIDE:
    ref_dir.mkdir(exist_ok=True, parents=True)
    df: pl.DataFrame = train_sim.to_dataframe().lazy().collect()[-1]
    df.write_csv(ref_dir / "to_dataframe_expected.csv")
if ENABLE_ASSERTS:
    print("Checking output of `to_dataframe`")
    to_dataframe_expected = pl.scan_csv(
        ref_dir / "to_dataframe_expected.csv"
    ).collect()[-1]
    print("Success!")

# %%
