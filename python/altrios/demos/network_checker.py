# %%
import time
import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
import seaborn as sns
import os
from copy import copy
import glob
import altrios as alt
from altrios.demos import plot_util
import re
from pathlib import Path
import sys

sns.set_theme()

env_folder_path = os.path.dirname(sys.executable)
if Path(env_folder_path + "/Library/share/gdal").exists():
    os.environ["GDAL_DATA"] = env_folder_path + "/Library/share/gdal"
    os.environ["GDAL_DRIVER_PATH"] = env_folder_path + "/Library/lib/gdalplugins"
    os.environ["GEOTIFF_CSV"] = env_folder_path + "/Library/share/epsg_csv"
    os.environ["PROJ_LIB"] = env_folder_path + "/Library/share/proj"
    os.environ["PROJ_NETWORK"] = "OFF"
else:

    import pyproj

    pyproj.datadir.set_data_dir(os.path.dirname(env_folder_path) + "/share/proj")


import geopandas as gpd

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
loco_vec = [] + [hel.copy()] + [alt.Locomotive.default()] * 3

# construct a vector of one BEL, one HEL, and several conventional locomotives
loco_vec_sans_buffers = [] + [hel_sans_buffers.copy()] + [alt.Locomotive.default()] * 1

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


networks = glob.glob("../Network Builder Test Small/Generated Networks/*")

# %%
Results = []
for network_path in networks:
    print(network_path)

    locations = pd.read_csv(network_path + "/Network Locations.csv")[
        "Location ID"
    ].unique()

    for origin in locations:
        for dest in locations:
            try:
                if dest != origin:
                    # Instantiate the intermediate `TrainSimBuilder`
                    tsb = alt.TrainSimBuilder(
                        train_id="0",
                        origin_id=origin,
                        destination_id=dest,
                        train_config=train_config,
                        loco_con=loco_con,
                    )

                    # Load the network and construct the timed link path through the network.
                    # print("Loading `Network`")
                    network = alt.Network.from_file(network_path + "/Network.yaml")

                    location_map = alt.import_locations(
                        network_path + "/Network Locations.csv"
                    )

                    train_sim: alt.SpeedLimitTrainSim = tsb.make_speed_limit_train_sim(
                        location_map=location_map,
                        save_interval=SAVE_INTERVAL,
                    )
                    train_sim.set_save_interval(SAVE_INTERVAL)
                    train_sim.to_file(network_path + "/debug_train_sim.yaml")
                    # print("Running `make_est_times`")
                    est_time_net, _consist = alt.make_est_times(train_sim, network)

                    # print("Running `run_dispatch`")
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

                    t0 = time.perf_counter()
                    # print("Running `walk_timed_path`")
                    train_sim.walk_timed_path(
                        network=network,
                        timed_path=timed_link_path,
                    )
                    t1 = time.perf_counter()

                    ts_dict = train_sim.to_pydict()
                    print(
                        "success {}, {}, {}".format(network_path[-25:-1], origin, dest)
                    )
                    network.to_file(network_path + "/Network.msgpack")
                    Results.append(
                        {
                            "network": network_path,
                            "origin": origin,
                            "destination": dest,
                            "Success?": 1,
                        }
                    )
            except Exception as e:
                Results.append(
                    {
                        "network": network_path,
                        "origin": origin,
                        "destination": dest,
                        "Success?": 0,
                    }
                )
                # print(e)
                e_str = str(e)
                split_errors = e_str.split("\\n")
                geopackage = "../Network Builder Test Small/SmallNetworkBuild.gpkg"
                layer = gpd.read_file(
                    geopackage,
                    layer=network_path.split("\\")[-1].split("/")[-1] + "_linked",
                )
                error_points = []
                for error_str in split_errors:
                    if error_str.endswith("do not point back!"):
                        link_numbers = re.findall(r"\d+", error_str)
                        link_numbers = list(map(int, link_numbers))
                        links = layer[layer.yaml_idx.isin(link_numbers)]
                        all_points = []
                        for idx, link in links.iterrows():
                            points = link.geometry.coords
                            all_points.append(points[0])
                            all_points.append(points[-1])

                        missing_switch = pd.Series(all_points).value_counts().index[0]
                        error_points.append(missing_switch)

                unique_points = pd.Series(error_points).value_counts()
                with open(network_path + "/missing_switches.txt", "w") as file:
                    file.writelines([e_str])
                    file.write(
                        "--------------------------------------------------------\n"
                    )
                    for point in unique_points.index:
                        file.write(
                            "https://www.openstreetmap.org/edit#map={}/{}/{}\n".format(
                                20, point[1], point[0]
                            )
                        )

                print("failure: {}, {}, {}".format(network_path[-25:-1], origin, dest))


Results = pd.DataFrame(Results)
Results.to_csv("network_check_results.csv")
# %%
