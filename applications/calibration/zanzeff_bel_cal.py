# Per Garrett:
# - how are we doing in modeling Traction Horsepower - AC Locomotive GECX 3000?
# - do we really need to match the SOC behavior of this prototype super well
# - in email, Garrett says there's a note about a bad trip to reject in Feb 3 trip
# - check SOC limits and make sure powertrain model fails when required power
#     conflicts with limits


import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import multiprocessing
import pickle
import time

from altrios.optimization import cal_and_val as cval
from altrios.optimization.cal_and_val import StarmapParallelization
import altrios as alt
import utils


class CalibrationProblem(cval.CalibrationProblem):
    def _evaluate(self, x, out, *args, **kwargs):
        # There may be cases where update_params could fail but we want to keep running: e.g.
        # `PyValueError::new_err("FuelConverter `eta_max` must be between 0 and 1" from
        # fuel_converter: line 69`.  
        #  TODO: figure out how to catch above-described error and keep running but fail for other
        # errors
        try:
            model_dict = self.mod_err.update_params(x)
            err = self.mod_err.get_errors(model_dict)
            out['F'] = np.array([
                val for inner_dict in err.values() for val in inner_dict.values()
            ])
            out['G'] = np.array([-1])
        except:  # might want to look for specific error type here  # noqa: E722
            out['F'] = np.ones(self.n_obj) * 1e12
            out['G'] = np.array([1])


params_and_bounds = (
    ("loco_unit.res.eta_max", (0.85, 0.999)),
    ("loco_unit.res.eta_range", (0.2, 0.5)),
    ("loco_unit.pwr_aux_offset_watts", (10, 5000)),
    ("loco_unit.pwr_aux_traction_coeff", (1e-5, 1e-3))
)
params = [pb[0] for pb in params_and_bounds]
params_bounds = [pb[1] for pb in params_and_bounds]


def get_bel_trip_mods(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dict of trip dfs, return mods
    """
    # df['dt [s]'] = pd.to_datetime(df['PacificTime']).dt.tz_convert('UTC')
    df['timestamp'] = pd.to_datetime(df['PacificTime']).dt.to_pydatetime()
    df['time [s]'] = cval.get_delta_seconds(df['timestamp']).cumsum()

    df3000 = df[['PacificTime', 'time [s]']].copy()
    df3000['Tractive Power [W]'] = (
        df["Tractive Effort Feedback GECX 3000"] * alt.utils.N_PER_LB *
        df["Locomotive Speed GECX 3000"] *
        alt.utils.MPS_PER_MPH
    )
    df3000['SOC'] = df['Propulsion Battery State Of Charge GECX 3000'] / 100.0
    df3000.drop(index=np.where(df3000['SOC'].isna())[0], inplace=True)

    # verify that stuff is left after the drop operation
    assert len(df3000) > 1, f"df3000 is too short in '{file.name}'"

    return df3000


def get_loco_sim(
    df3000: pd.DataFrame
) -> bytes:
    powertrace3000 = alt.PowerTrace(
        df3000['time [s]'].to_numpy(),
        df3000['Tractive Power [W]'].to_numpy(),
        [None] * len(df3000),  
    )

    reves = alt.ReversibleEnergyStorage.default()
    edrv = alt.ElectricDrivetrain.default()

    loco_unit = alt.Locomotive.build_battery_electric_loco(
        reversible_energy_storage=reves,
        drivetrain=edrv,
        pwr_aux_offset_watts=10e3,
        pwr_aux_traction_coeff_ratio=0,
        force_max_newtons=667.2e3,
        save_interval=1,
    )

    alt.utils.set_param_from_path(
        loco_unit, "res.state.soc",
        df3000['SOC'].iloc[0]
    )

    # setting power max 10% higher so calibration cal occur without throwing power error.
    alt.utils.set_param_from_path(
        loco_unit, "res.pwr_out_max_watts",
        3.281e6
    )

    # alt.utils.set_param_from_path(
    #     loco_unit, "edrv.pwr_out_max_watts",
    #     6e15
    # )

    lsb3000 = alt.LocomotiveSimulation(
        loco_unit=loco_unit,
        power_trace=powertrace3000,
        save_interval=1,
    ).to_bincode()

    return lsb3000


def get_mod_err(
    df_and_sim_dict: Dict[str, Tuple[pd.DataFrame, alt.LocomotiveSimulation]],
    model_objectives: List[Tuple[str, str]],
    model_params: Tuple[str],
    debug: Optional[bool] = False,
) -> cval.ModelError:

    mod_err = cval.ModelError(
        bincode_model_dict={
            key: df_sim_tup[1] for key,
            df_sim_tup in df_and_sim_dict.items()
        },
        dfs={
            key: df_sim_tup[0] for key, df_sim_tup in df_and_sim_dict.items()
        },
        objectives=model_objectives,
        params=model_params,
        model_type='LocomotiveSimulation',
        verbose=False,
        debug=debug,
    )

    return mod_err


def_save_path = Path("bel_loco_cal")

if __name__ == '__main__':
    from utils import get_parser
    parser = get_parser(
        description="ALTRIOS Battery Electric Locomotive Calibration")

    # parser.add_argument('--norm', type=int, default=2,
    #                     help="`norm` for minimum distance calculation.")
    # parser.add_argument("--run-prep", action="store_true",
    #                     help="If provided, runs pre-processing of trip data.  This must be done at least once on each machine.")
    parser.add_argument(
        "--save-path", type=str, default=str(def_save_path),
        help="Path to folder for saving results.  Creates folder if needed."
    )
    parser.add_argument(
        "--plotly", action="store_true",
        help="If passed, generates and saves plotly plots."
    )
    parser.add_argument(
        "--pyplot", action="store_true",
        help="If passed, generates and saves pyplot plots."
    )
    parser.add_argument(
        "--show-pyplot", action="store_true",
        help="If passed, shows any generated pyplot plots."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="If passed, runs in debug mode."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for randomized trip selection process."
    )

    args = parser.parse_args()

    n_proc = args.n_proc
    n_max_gen = args.n_max_gen
    pop_size = args.pop_size
    # norm_num = args.norm
    # assert norm_num >= 1
    # run_prep = args.run_prep
    save_path = Path(args.save_path)
    save_path.mkdir(exist_ok=True)
    pyplot = args.pyplot
    plotly = args.plotly
    show_pyplot = args.show_pyplot
    debug = args.debug
    seed = args.seed

    objectives = [
        (
            "SOC",
            "loco_unit.res.history.soc"
        )
    ]

    artifact_dir = Path("bel_loco_cal")
    artifact_dir.mkdir(exist_ok=True)

    # note that "Condfidential" has been consistently misspelled

    ignore_dict = utils.TRIP_FILE_IGNORE_DICT
    ignore_dict.update({
        "2-3 Stock to Bar - ALTRIOS Condfidential 2": 
            "soc and power demand are wack for first ~1,000 seconds",
        "2-3 Stock to Bar - ALTRIOS Condfidential 3": 
            "zero tractive power the whole time",
        "2-7 Stock to Bar - ALTRIOS Condfidential 1": 
            "zero tractive power the whole time",
        "3-16 Bar to Stock - ALTRIOS Condfidential 1": 
            "most of the trip is nothing happening",
        "2-2 Bar to Stock - ALTRIOS Condfidential 2": "big gap in data",
        "1-21 Bar to Stock - ALTRIOS Condfidential 2": "Odd SOC behavior, not typical",
        "2-10 Bar to Stock - ALTRIOS Condfidential 2": "SOC interpolated",
        "1-12 Bar to Stock - ALTRIOS Condfidential 3": "big gap in data",
        "1-23 Stock to Bar - ALTRIOS Condfidential 1": "goofy SOC in the middle probably caused by a power cycle",
        "1-26 Bar to Stock - ALTRIOS Condfidential 2": "SOC jumps because of a string coming in and out",
        "3-3 Stock to Bar - ALTRIOS Condfidential 2": "Short gap in data, could be salvaged easily",
        "2-12 Stock to Bar - ALTRIOS Condfidential 1":  "SOC went negative",
        "2-2 Bar to Stock - ALTRIOS Condfidential 5":  "Odd SOC caused by power cycle",
        "2-7 Stock to Bar - ALTRIOS Condfidential 2":  "Odd SOC caused by power cycle",
        "3-10 Stock to Bar - ALTRIOS Condfidential 1": "Odd SOC caused by power cycle",
        "1-21 Bar to Stock - ALTRIOS Condfidential 3":  "gap in data",
        "2-6 Bar to Stock - ALTRIOS Condfidential 1": "soc went negative when BEL restarted",
        "1-14 Stock to Bar - ALTRIOS Condfidential 4": "Odd SOC caused by power cycle",
        "2-12 Stock to Bar - ALTRIOS Condfidential 2": "Odd SOC cause by power cycle",
        "2-17 Stock to Bar - ALTRIOS Condfidential 2": "gap in data",
        "1-6 Stock to Bar - ALTRIOS Condfidential 4": "small step change in SOC of ~2% near end of file",
        "2-10 Bar to Stock - ALTRIOS Condfidential 1": "small SOC step change of about 1.5%",
        "2-20 Stock to Bar - ALTRIOS Condfidential 4":  "gap in data",
        "3-16 Bar to Stock - ALTRIOS Condfidential 6": "small gap in middle of data; otherwise nice dataset",
        "3-2 Bar to Stock - ALTRIOS Condfidential 3": "gap in data"

    })  # put additional things to ignore list in here
    ignore_re_pattern = utils.get_ignore_list_re_pattern(ignore_dict)

    cal_mod_files, val_mod_files = utils.select_cal_and_val_trips(
        save_path=save_path,
        ignore_re_pattern=ignore_re_pattern,
        force_rerun=args.repartition,
        # fname_re_pattern="(1-\d) (\w+) to (\w+).*\.csv",
    )

    # Dict[str, str] keys: reject file key, values: reason for rejection
    rejects = {}

    cal_mod_raw_dfs = {}
    cal_mod_dfs = {}
    cal_df_and_sims = {}  # Dict[str, (df, sim)]

    for (i, file) in enumerate(cal_mod_files):
        file = Path(file)
        print(f"Processing: {file.name}")
        raw_df = pd.read_csv(
            file, low_memory=False)
        # if anything needs to be trimmed
        # if file.name == '':
        #     do whatever needs to happen here
        if len(raw_df) < 1:
            rejects[file.name] = "len < 1"
            continue
        cal_mod_raw_dfs[file.stem] = raw_df
        cal_mod_dfs[file.stem] = get_bel_trip_mods(
            cal_mod_raw_dfs[file.stem])

        cal_df_and_sims[file.stem] = (
            cal_mod_dfs[file.stem],
            get_loco_sim(cal_mod_dfs[file.stem])
        )

    cal_mod_err = get_mod_err(
        cal_df_and_sims,
        objectives,
        params,
    )

    val_mod_raw_dfs = {}
    val_mod_dfs = {}
    val_df_and_sims = {}

    for (i, file) in enumerate(val_mod_files):
        file = Path(file)
        print(f"Processing: {file.name}")
        val_mod_raw_dfs[file.stem] = pd.read_csv(
            file, low_memory=False)
        val_mod_dfs[file.stem] = get_bel_trip_mods(
            val_mod_raw_dfs[file.stem])
        val_df_and_sims[file.stem] = (
            val_mod_dfs[file.stem],
            get_loco_sim(val_mod_dfs[file.stem])
        )

    val_mod_err = get_mod_err(
        val_df_and_sims,
        objectives,
        params,
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
        problem = CalibrationProblem(
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
    print(
        f"Number of processes: {n_proc}, Simulation time: {t1 - t0:.5f} seconds")

    optimal_params = cval.min_error_selection(
        res_df,
        param_num=len(cal_mod_err.params),
    )

    cal_plot_save_dir = Path(save_path) / 'plots/cal'
    cal_plot_save_dir.mkdir(exist_ok=True, parents=True)
    calibration_model_dict = cal_mod_err.update_params(optimal_params)
    cal_err = cal_mod_err.get_errors(
        calibration_model_dict,
        pyplot=pyplot,
        plotly=plotly,
        show_pyplot=show_pyplot,
        plot_save_dir=cal_plot_save_dir,
    )

    val_plot_save_dir = Path(save_path) / 'plots/val'
    val_plot_save_dir.mkdir(exist_ok=True, parents=True)
    validation_model_dict = val_mod_err.update_params(
        optimal_params)
    val_err = val_mod_err.get_errors(
        validation_model_dict,
        pyplot=pyplot,
        plotly=plotly,
        show_pyplot=show_pyplot,
        plot_save_dir=val_plot_save_dir,
    )
