import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import multiprocessing
import matplotlib.pyplot as plt  # noqa: F401
import time
import pickle
import json
from scipy.signal import savgol_filter

from altrios.optimization import cal_and_val as cval
from altrios.optimization.cal_and_val import StarmapParallelization
from altrios import Consist, Locomotive, SetSpeedTrainSim, SpeedTrace
from altrios import InitTrainState, TrainSimBuilder, TrainConfig
import altrios as alt
import utils

CUR_FUEL_LHV_J__KG = 43e6

# speed_column = 'GPS Velocity BNSF 3940'
speed_column = "Locomotive Speed GECX 3000"

ignoredict = utils.TRIP_FILE_IGNORE_DICT
with open("set_speed_train_cal_ignoredict.json", "r") as f:
    ignoredict.update(json.load(f))  # put additional things to ignore list in here

ignore_re_pattern = utils.get_ignore_list_re_pattern(ignoredict)

savgol_window = 100
train_type = "Manifest"


def get_train_sim_df_mods(
    df_raw: pd.DataFrame, idx_end: Optional[int] = None
) -> pd.DataFrame:
    df_train_sim = df_raw.copy()
    if idx_end:
        df_train_sim = df_train_sim.iloc[:idx_end, :].copy()
    df_train_sim["PacificTime_orig"] = df_train_sim["PacificTime"].copy()
    df_train_sim["PacificTime"] = pd.to_datetime(
        df_train_sim["PacificTime_orig"]
    ).dt.tz_convert("UTC")

    df_train_sim.drop_duplicates(subset="PacificTime", inplace=True)

    df_train_sim["time [s]"] = cval.get_delta_seconds(
        df_train_sim["PacificTime"]
    ).cumsum()
    df_train_sim["Total Tractive Force [N]"] = (
        df_train_sim[
            [
                "Tractive Effort Feedback BNSF 3940",
                "Tractive Effort Feedback BNSF 3965",
                "Tractive Effort Feedback GECX 3000",
            ]
        ].sum(axis=1)
        * alt.utils.N_PER_LB
    )
    df_train_sim["Total Tractive Power [W]"] = (
        df_train_sim["Total Tractive Force [N]"]
        * df_train_sim[speed_column]
        * alt.utils.MPS_PER_MPH
    )

    df_train_sim["Total Cumu. Tractive Energy [J]"] = (
        df_train_sim["Total Tractive Power [W]"]
        * df_train_sim["time [s]"].diff().fillna(0.0)
    ).cumsum()

    df_train_sim["Total Pos. Cumu. Tractive Energy [J]"] = (
        (
            df_train_sim["Total Tractive Power [W]"]
            * df_train_sim["time [s]"].diff().fillna(0.0)
        )
        .where(df_train_sim["Total Tractive Power [W]"] > 0, 0.0)
        .cumsum()
    )

    speed = savgol_filter(
        df_train_sim[speed_column].to_numpy() * alt.utils.MPS_PER_MPH,
        savgol_window,
        3,
    )
    speed = np.clip(
        speed,
        a_min=0.0,
        a_max=1e9,
    )
    df_train_sim["Filtered Speed"] = speed

    df_train_sim = df_train_sim[
        [
            "Filtered Speed",
            "PacificTime_orig",
            "PacificTime",
            "time [s]",
            speed_column,
            "Tractive Effort Feedback BNSF 3940",
            "Tractive Effort Feedback BNSF 3940",
            "Tractive Effort Feedback BNSF 3965",
            "Tractive Effort Feedback GECX 3000",
            "Total Tractive Force [N]",
            "Total Tractive Power [W]",
            "Total Cumu. Tractive Energy [J]",
            "Total Pos. Cumu. Tractive Energy [J]",
            "Length",
            "Empties",
            "Loads",
            "Weight",
            "ALTRIOS - BARSTO Distance [m]",
            "ALTRIOS - STOBAR Distance [m]",
        ]
    ].copy()

    return df_train_sim


def get_train_sim_inputs(df: pd.DataFrame, file_path: Path) -> bytes:
    if "Bar to Stock" in file_path.stem:
        link_path = [alt.LinkIdx(1)]
        offset_col = "ALTRIOS - BARSTO Distance [m]"
        origin_id = "Barstow"
        destination_id = "Stockton"
    elif "Stock to Bar" in file_path.stem:
        link_path = [alt.LinkIdx(2)]
        offset_col = "ALTRIOS - STOBAR Distance [m]"
        origin_id = "Stockton"
        destination_id = "Barstow"
    else:
        raise ValueError("Directionality is invalid.")

    network = alt.Network.from_file(
        alt.package_root() / "../../../data/StockToBar_10thPoint_corrected.yaml"
    )

    max_offset = network[link_path[0].idx].length_meters - 1e3

    try:
        first_bad_row = np.where(df[offset_col] >= max_offset)[0][0] - 10
        df.drop(np.arange(first_bad_row - 1, len(df)), inplace=True)
    except:  # noqa: E722
        pass

    # speed_trace = SpeedTrace(
    #     df['time [s]'].iloc[1:].to_numpy(),
    #     df['Filtered Speed'].iloc[1:],
    # )

    speed_trace = SpeedTrace(
        df["time [s]"].to_numpy(),
        df["Filtered Speed"],
    )

    speed_start_mps = df["Locomotive Speed GECX 3000"].iloc[0] * alt.utils.MPS_PER_MPH

    # loco_conventional = Locomotive.default()
    # altpy.set_param_from_path(
    #     loco_conventional, "fc.pwr_ramp_lag_seconds", 0.000001)
    # loco_vec = [
    #     loco_conventional.copy(),
    #     alt.Locomotive.default_battery_electric_loco(),
    #     loco_conventional.copy(),
    #     loco_conventional.copy(),
    #     # loco_con needs to be consistent with whatever is actually in ZANZEFF
    # ]
    loco_vec = [Locomotive.build_dummy_loco()]
    save_interval = 1

    loco_con = Consist(loco_vec, save_interval)
    loco_con.__assert_limits = False

    # replace with some length
    train_length_meters = df["Length"].iloc[0] * alt.utils.M_PER_FT
    # column TRN_GRS_TONS does not include locomotive tons
    # TODO: add locomotive weight, if deemed necessary after careful thought
    # the goal is to match the net force across the draw bar between the
    # last locomotive and the rest of the train
    train_mass_kilograms = df["Weight"].iloc[0] * alt.utils.KG_PER_TON

    train_config = TrainConfig(
        rail_vehicle_type=train_type,
        cars_empty=df["Empties"].iloc[0],
        cars_loaded=df["Loads"].iloc[0],
        train_type=None,  # Defaults to Freight
        train_length_meters=train_length_meters,
        train_mass_kilograms=train_mass_kilograms,
    )

    init_train_state = InitTrainState(
        offset_meters=max(train_length_meters, df[offset_col].iloc[0]),
        speed_meters_per_second=speed_start_mps,
    )

    tsb = TrainSimBuilder(
        file_path.stem,
        origin_id=origin_id,
        destination_id=destination_id,
        train_config=train_config,
        loco_con=loco_con,
        init_train_state=init_train_state,
    )

    rail_vehicle_map = alt.import_rail_vehicles(
        str(alt.resources_root() / "rolling_stock/rail_vehicles.csv")
    )

    return (
        tsb.to_bincode(),
        {k: v.to_bincode() for (k, v) in rail_vehicle_map.items()},
        [link.to_bincode() for link in network],
        [lp.to_bincode() for lp in link_path],
        speed_trace.to_bincode(),
    )


class ModelError(cval.ModelError):
    def update_params(self, xs: List[float]) -> Dict[str, SetSpeedTrainSim]:
        """
        SetSpeedTrainSim specific override of cval.ModelError.update_params
        """
        assert len(xs) == len(self.params)

        t0 = time.perf_counter()

        model_dict = {}

        for key, value in self.bincode_model_dict.items():
            (tsb, rail_vehicle_map, network, link_path, speed_trace) = value
            tsb = alt.TrainSimBuilder.from_bincode(tsb)
            rail_vehicle_map = {
                k: alt.RailVehicle.from_bincode(v)
                for (k, v) in rail_vehicle_map.items()
            }
            network = [alt.Link.from_bincode(link) for link in network]
            link_path = [alt.LinkIdx.from_bincode(lp) for lp in link_path]
            speed_trace = alt.SpeedTrace.from_bincode(speed_trace)

            for path, x in zip(self.params, xs):
                alt.set_param_from_path(rail_vehicle_map[train_type], path, x)

            # set `cd_area_empty_square_meters` to be same as `cd_area_loaded_square_meters`
            alt.set_param_from_path(
                rail_vehicle_map[train_type],
                "cd_area_empty_square_meters",
                rail_vehicle_map[train_type].cd_area_loaded_square_meters,
            )

            train_sim = tsb.make_set_speed_train_sim(
                rail_vehicle_map=rail_vehicle_map,
                network=network,
                link_path=link_path,
                speed_trace=speed_trace,
                save_interval=1,
            )

            model_dict[key] = train_sim

        t1 = time.perf_counter()
        if self.verbose:
            print(f"Time to update params: {t1 - t0:.3g} s")

        return model_dict


def get_mod_err(
    df_and_sim_dict: Dict[str, Tuple[pd.DataFrame, alt.SetSpeedTrainSim]],
    model_objectives: List[Tuple[str, str]],
    model_params: Tuple[str],
    debug: Optional[bool] = False,
) -> ModelError:
    mod_err = ModelError(
        bincode_model_dict={
            key: df_sim_tup[1] for key, df_sim_tup in df_and_sim_dict.items()
        },
        dfs={key: df_sim_tup[0] for key, df_sim_tup in df_and_sim_dict.items()},
        objectives=model_objectives,
        params=model_params,
        model_type="SetSpeedTrainSim",
        verbose=False,
        debug=debug,
    )

    return mod_err


params_and_bounds = (
    ("cd_area_loaded_square_meters", (1, 8)),
    # `cd_area_empty_square_meters` gets manually set equal to
    # `cd_area_loaded_square_meters`
    # ("cd_area_empty_square_meters", (1, 8)),
    # ("davis_b_seconds_per_meter", (0, 0.1)),
    ("rolling_ratio", (0.0003, 0.003)),
    ("bearing_res_per_axle_newtons", (40, 320)),
)
params = [pb[0] for pb in params_and_bounds]
params_bounds = [pb[1] for pb in params_and_bounds]

objectives = [
    ("Total Pos. Cumu. Tractive Energy [J]", "history.energy_whl_out_pos_joules")
]


def_save_path = Path("train_sim_cal")

if __name__ == "__main__":
    from utils import get_parser

    parser = get_parser(description="ALTRIOS Conventional Locomotive Calibration")

    parser.add_argument(
        "--save-path",
        type=str,
        default=str(def_save_path),
        help="Path to folder for saving results.  Creates folder if needed.",
    )
    parser.add_argument(
        "--plotly",
        action="store_true",
        help="If passed, generates and saves plotly plots.",
    )
    parser.add_argument(
        "--pyplot",
        action="store_true",
        help="If passed, generates and saves pyplot plots.",
    )
    parser.add_argument(
        "--show-pyplot",
        action="store_true",
        help="If passed, shows any generated pyplot plots.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="If passed, runs in debug mode."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for randomized trip selection process.",
    )

    # example usage:
    # `python zanzeff_set_speed_train_cal.py --n-max-gen 10 --pop-size 10 --n-proc 10 --make-plots`  # noqa: E501

    args = parser.parse_args()
    # else:
    #     seed = time.time()
    n_proc = args.n_proc
    n_max_gen = args.n_max_gen
    pop_size = args.pop_size
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)
    pyplot = args.pyplot
    plotly = args.plotly
    show_pyplot = args.show_pyplot
    debug = args.debug
    seed = args.seed

    artifact_dir = Path("train_sim_cal")
    artifact_dir.mkdir(exist_ok=True)

    cal_mod_files, val_mod_files = utils.select_cal_and_val_trips(
        save_path=save_path,
        trip_dir=alt.package_root()
        / "../../../data/trips/ZANZEFF Data - v5.1 - cleaned ALTRIOS Confidential",
        ignore_re_pattern=ignore_re_pattern,
        force_rerun=args.repartition,
    )

    # Dict[str, str] keys: reject file key, values: reason for rejecton
    rejects = {}

    cal_mod_raw_dfs = {}
    cal_mod_dfs = {}
    cal_df_and_sims = {}  # Dict[str, (df, sim)]
    for i, file in enumerate(cal_mod_files):
        if debug and (i > 5):
            break
        file = Path(file)
        print(f"Processing: {file.name}")
        raw_df = pd.read_csv(file, low_memory=False)
        if len(raw_df) <= savgol_window:
            ignoredict.update(
                {
                    file.stem: "too short for savgol window",
                }
            )
            with open("set_speed_train_cal_ignoredict.json", "w") as f:
                json.dump(ignoredict, f)
            raise Exception(
                f"{file.stem} has length {len(raw_df)}, which is shorter "
                + "than savgol window: {savgol_window}. Appending to ignoredict."
            )

        if len(raw_df) < 1:
            rejects[file.name] = "len < 1"
            continue
        cal_mod_raw_dfs[file.stem] = raw_df
        cal_mod_dfs[file.stem] = get_train_sim_df_mods(cal_mod_raw_dfs[file.stem])
        cal_df_and_sims[file.stem] = (
            cal_mod_dfs[file.stem],
            get_train_sim_inputs(cal_mod_dfs[file.stem], file),
        )

    cal_mod_err = get_mod_err(
        cal_df_and_sims,
        objectives,
        params,
    )

    val_mod_raw_dfs = {}
    val_mod_dfs = {}
    val_df_and_sims = {}
    for i, file in enumerate(val_mod_files):
        if debug and (i > 5):
            break
        file = Path(file)
        print(f"Processing: {file.name}")
        val_mod_raw_dfs[file.stem] = pd.read_csv(file, low_memory=False)
        val_mod_dfs[file.stem] = get_train_sim_df_mods(val_mod_raw_dfs[file.stem])
        val_df_and_sims[file.stem] = (
            val_mod_dfs[file.stem],
            get_train_sim_inputs(val_mod_dfs[file.stem], file),
        )

    val_mod_err = get_mod_err(
        val_df_and_sims,
        objectives,
        params,
    )

    # force val set to run to check data up front
    val_mod_dict = val_mod_err.update_params(
        [0.5 * (pb[0] + pb[1]) for pb in params_bounds]
    )

    with open(artifact_dir / "cal_mod_err.pickle", "wb") as f:
        pickle.dump(cal_mod_err, f)
    with open(artifact_dir / "val_mod_err.pickle", "wb") as f:
        pickle.dump(val_mod_err, f)

    algorithm = cval.NSGA3(
        ref_dirs=cval.get_reference_directions(
            "energy",
            # must be at least cal_objectives.n_obj
            n_dim=cal_mod_err.n_obj,
            n_points=pop_size,  # must be at least pop_size
        ),
        sampling=cval.LHS(),
        pop_size=pop_size,
    )
    termination = cval.DMOT(n_max_gen=n_max_gen, period=5)

    t0 = time.perf_counter()
    if n_proc == 1:
        problem = cval.CalibrationProblem(
            mod_err=cal_mod_err,
            n_constr=1,
            params_bounds=params_bounds,
        )
        res, res_df = cval.run_minimize(
            problem=problem,
            algorithm=algorithm,
            termination=termination,
            save_path=save_path,
        )
    else:
        assert n_proc > 1
        with multiprocessing.Pool(n_proc) as pool:
            problem = cval.CalibrationProblem(
                mod_err=cal_mod_err,
                n_constr=1,
                params_bounds=params_bounds,
                elementwise_runner=StarmapParallelization(pool.starmap),
            )
            res, res_df = cval.run_minimize(
                problem=problem,
                algorithm=algorithm,
                termination=termination,
                save_path=save_path,
            )

    t1 = time.perf_counter()
    print(f"Number of processes: {n_proc}, Simulation time: {t1 - t0:.5f} seconds")

    optimal_params = cval.min_error_selection(res_df, param_num=len(cal_mod_err.params))

    cal_plot_save_dir = Path(save_path) / "plots/cal"
    cal_plot_save_dir.mkdir(exist_ok=True, parents=True)
    calibration_model_dict = cal_mod_err.update_params(optimal_params)
    cal_err = cal_mod_err.get_errors(
        calibration_model_dict,
        pyplot=pyplot,
        plotly=plotly,
        show_pyplot=show_pyplot,
        plot_save_dir=cal_plot_save_dir,
    )

    val_plot_save_dir = Path(save_path) / "plots/val"
    val_plot_save_dir.mkdir(exist_ok=True, parents=True)
    validation_model_dict = val_mod_err.update_params(optimal_params)
    val_err = val_mod_err.get_errors(
        validation_model_dict,
        pyplot=pyplot,
        plotly=plotly,
        show_pyplot=show_pyplot,
        plot_save_dir=val_plot_save_dir,
    )


# %%

# idx_start = int(18e3)
# idx_end = int(22e3)
# idx_start = 0
# idx_end = -1


# fig, axes = plt.subplots(
#     4, 1, sharex=True, figsize=(15, 15), facecolor='white')
# axes[0].set_title(file.stem, fontsize=18)
# axes[0].plot(
#     np.array(train_sim.history.time_seconds)[idx_start:idx_end] / 3_600,
#     np.array(train_sim.history.energy_whl_out_joules)[
#         idx_start:idx_end] * 1e-9,
#     label='model',
# )
# axes[0].plot(
#     (df['time [s]'] / 3_600)[idx_start:idx_end],
#     (df['Total Cumu. Tractive Energy [J]'] * 1e-9)[idx_start:idx_end],
#     label='test data',
# )
# axes[0].set_ylabel("Cumu. Trac.\nEnergy [GJ]", fontsize=18)
# axes[0].legend()

# axes[1].plot(
#     np.array(train_sim.history.time_seconds)[idx_start:idx_end] / 3_600,
#     np.array(train_sim.history.pwr_whl_out_watts)[idx_start:idx_end] * 1e-6,
#     label='model',
# )
# axes[1].plot(
#     (df['time [s]'] / 3_600)[idx_start:idx_end],
#     (df['Total Tractive Power [W]'] * 1e-6)[idx_start:idx_end],
#     label='test data',
# )
# axes[1].set_ylabel("Trac. Power [MW]", fontsize=18)
# axes[1].legend()

# axes[-2].plot(
#     np.array(
#         train_sim.speed_trace.time_seconds
#     )[:train_sim.history.len()][idx_start:idx_end] / 3_600,
#     np.array(train_sim.history.offset_meters)[
#         :train_sim.history.len()][idx_start:idx_end]
# )
# axes[-2].set_ylabel('Offset [m]', fontsize=18)

# axes[-1].plot(
#     np.array(
#         train_sim.speed_trace.time_seconds
#     )[:train_sim.history.len()][idx_start:idx_end] / 3_600,
#     np.array(train_sim.speed_trace.speed_meters_per_second)[
#         :train_sim.history.len()][idx_start:idx_end]
# )
# axes[-1].set_ylabel('Speed [m/s]', fontsize=18)
# axes[-1].set_xlabel('Time [hr]', fontsize=18)

# for ax in axes:
#     ax.tick_params(labelsize=16)

# plt.tight_layout()

# plt.savefig("plots/tpc_val_v_time.png")
# plt.savefig("plots/tpc_val_v_time.svg")

# plt.show()

# # %%

# fig, axes = plt.subplots(
#     4, 1, sharex=True, figsize=(15, 15), facecolor='white')
# axes[0].set_title(file.stem, fontsize=18)
# axes[0].plot(
#     np.array(train_sim.history.offset_meters)[idx_start:idx_end] / 1e3,
#     np.array(train_sim.history.energy_whl_out_joules)[
#         idx_start:idx_end] * 1e-9,
#     label='model',
# )
# axes[0].plot(
#     (df['ALTRIOS - STOBAR Distance [m]'] / 1_000)[idx_start:idx_end],
#     (df['Total Cumu. Tractive Energy [J]'] * 1e-9)[idx_start:idx_end],
#     label='test data',
# )
# axes[0].set_ylabel("Cumu. Trac.\nEnergy [GJ]", fontsize=18)
# axes[0].legend()

# axes[1].plot(
#     np.array(train_sim.history.offset_meters)[idx_start:idx_end] / 1e3,
#     np.array(train_sim.history.pwr_whl_out_watts)[idx_start:idx_end] * 1e-6,
#     label='model',
# )
# axes[1].plot(
#     (df['ALTRIOS - STOBAR Distance [m]'] / 1e3)[idx_start:idx_end],
#     (df['Total Tractive Power [W]'] * 1e-6)[idx_start:idx_end],
#     label='test data',
# )
# axes[1].set_ylabel("Trac. Power [MW]", fontsize=18)
# axes[1].legend()

# axes[2].plot(
#     np.array(train_sim.history.offset_meters)[
#         :train_sim.history.len()][idx_start:idx_end] / 1e3,
#     np.array(
#         train_sim.speed_trace.time_seconds
#     )[:train_sim.history.len()][idx_start:idx_end] / 3_600,
# )

# axes[2].plot(
#     np.array(train_sim.history.offset_meters)[
#         :train_sim.history.len()][idx_start:idx_end] / 1e3,
#     np.array(
#         train_sim.speed_trace.time_seconds
#     )[:train_sim.history.len()][idx_start:idx_end] / 3_600,
#     label='model'
# )
# axes[2].plot(
#     (df['ALTRIOS - STOBAR Distance [m]'] / 1e3)[idx_start:idx_end],
#     (df['time [s]'] / 3_600)[idx_start:idx_end],
# )

# axes[2].set_ylabel('Time [hr]', fontsize=18)
# axes[2].legend()

# axes[3].plot(
#     np.array(
#         train_sim.history.offset_meters
#     )[:train_sim.history.len()][idx_start:idx_end] / 1_000,
#     np.array(train_sim.speed_trace.speed_meters_per_second)[
#         :train_sim.history.len()][idx_start:idx_end]
# )
# axes[3].set_ylabel('Speed [m/s]', fontsize=18)
# axes[3].set_xlabel('Distance [km]', fontsize=18)

# for ax in axes:
#     ax.tick_params(labelsize=16)

# plt.tight_layout()

# plt.savefig("plots/tpc_val_v_dist.png")
# plt.savefig("plots/tpc_val_v_dist.svg")

# plt.show()

# # %%
