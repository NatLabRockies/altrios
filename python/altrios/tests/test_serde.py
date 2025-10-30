import time
import altrios as alt

SAVE_INTERVAL = 100


def get_solved_speed_limit_train_sim():
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

    bel = alt.Locomotive.from_pydict(
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

    # construct a vector of one BEL and several conventional locomotives
    loco_vec = [bel.copy()] + [alt.Locomotive.default()] * 7
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

    train_sim.walk_timed_path(
        network=network,
        timed_path=timed_link_path,
    )
    assert len(train_sim.to_pydict()["history"]) > 1

    return train_sim


def test_pydict():
    ts = get_solved_speed_limit_train_sim()

    t0 = time.perf_counter_ns()
    ts_dict_msg = ts.to_pydict(flatten=False, data_fmt="msg_pack")
    ts_msg = alt.SpeedLimitTrainSim.from_pydict(ts_dict_msg, data_fmt="msg_pack")
    t1 = time.perf_counter_ns()
    t_msg = t1 - t0
    print(f"\nElapsed time for MessagePack: {t_msg:.3e} ns ")

    t0 = time.perf_counter_ns()
    ts_dict_yaml = ts.to_pydict(flatten=False, data_fmt="yaml")
    ts_yaml = alt.SpeedLimitTrainSim.from_pydict(ts_dict_yaml, data_fmt="yaml")
    t1 = time.perf_counter_ns()
    t_yaml = t1 - t0
    print(f"Elapsed time for YAML: {t_yaml:.3e} ns ")
    print(f"YAML time per MessagePack time: {(t_yaml / t_msg):.3e} ")

    t0 = time.perf_counter_ns()
    ts_dict_json = ts.to_pydict(flatten=False, data_fmt="json")
    _ts_json = alt.SpeedLimitTrainSim.from_pydict(ts_dict_json, data_fmt="json")
    t1 = time.perf_counter_ns()
    t_json = t1 - t0
    print(f"Elapsed time for json: {t_json:.3e} ns ")
    print(f"JSON time per MessagePack time: {(t_json / t_msg):.3e} ")

    # `to_pydict` is necessary because of some funkiness with direct equality comparison
    assert ts_msg.to_pydict() == ts.to_pydict()
    assert ts_yaml.to_pydict() == ts.to_pydict()


if __name__ == "__main__":
    test_pydict()
