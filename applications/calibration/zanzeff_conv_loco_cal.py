#%%
# Notes from discussion with Garrett
# do the model deviations w.r.t. locomotive position occur at idle or under load?
# How much does aux load impact discrepancies?

import pandas as pd
import numpy as np
from pathlib import Path
from unittest import TestCase
from typing import Tuple, Optional, List, Dict, Any
import argparse
import matplotlib.pyplot as plt
import time
import pickle
import re


from altrios.optimization import cal_and_val as cval
from altrios.optimization.cal_and_val import StarmapParallelization
import altrios as alt
from altrios import LocomotiveSimulation
import utils
import hashlib
import os

CUR_FUEL_LHV_J__KG = 43e6


def get_conv_trip_mods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given single trip data frame, return dataframe for non-lead locomotive
    """
    df = df.drop_duplicates('PacificTime').copy()
    df['timestamp'] = pd.to_datetime(
        df['PacificTime']).dt.to_pydatetime()
    df['time [s]'] = cval.get_delta_seconds(df['timestamp']).cumsum()

  

    df39xx = df[['timestamp', 'time [s]']].copy()

    lead_loc = None
    for lead_loc_possible in df["Lead Locomotive"].unique():
        try:
            if lead_loc is not None:
                assert lead_loc == int(lead_loc_possible)
            else:
                lead_loc = int(lead_loc_possible)

        except:
            pass

    if lead_loc == 3940:
        trailing_loc = 3965
        
    elif lead_loc == 3965:
        trailing_loc = 3940
       
    else:
        raise ValueError(f"Invalid lead locomotive: {lead_loc}")

    df39xx['Tractive Power [W]'] = (
        df["Tractive Effort Feedback BNSF " + str(trailing_loc)] * alt.utils.N_PER_LB *
        df["Locomotive Speed GECX 3000"] * alt.utils.MPS_PER_MPH).fillna(0.0)

    df39xx["Fuel Power [W]"] = df[
        "Fuel Rate " + str(trailing_loc) + " [lbs/hr]"].fillna(0.0) * alt.utils.KG_PER_LB \
        / 3600 * CUR_FUEL_LHV_J__KG
    df39xx["Fuel Energy [J]"] = (df39xx["Fuel Power [W]"] *
                                    cval.get_delta_seconds(
                                        df39xx['timestamp'])
                                    ).cumsum().copy()
    df39xx["engine_on"] = df['Engine Speed (RPM) BNSF ' + str(trailing_loc)] > 100
    return df39xx

def get_loco_sim(df39xx: pd.DataFrame) -> bytes:
    powertrace = alt.PowerTrace(
        df39xx['time [s]'].to_numpy(),
        df39xx['Tractive Power [W]'].to_numpy(),
        df39xx.engine_on,  # This is 39XX engine state (on/off)
    )
    loco_unit = alt.Locomotive.default()

    alt.set_param_from_path(loco_unit, "pwr_aux_offset_watts", 10e3)
    alt.set_param_from_path(loco_unit, "fc.pwr_ramp_lag_seconds", 0.0000000000000001)
    alt.set_param_from_path(loco_unit, "fc.pwr_out_max_watts", 3255000.0*2)
    alt.set_param_from_path(loco_unit, "edrv.pwr_out_max_watts", 3255000.0*2)
    alt.set_param_from_path(loco_unit, "gen.pwr_out_max_watts", 3255000.0*2)


    loco_sim = LocomotiveSimulation(loco_unit=loco_unit,
                                    power_trace=powertrace,
                                    save_interval=1,
                                    )
    loco_sim_bincode = loco_sim.to_bincode()
    return loco_sim_bincode


class ModelError(cval.ModelError):
    def update_params(
        self, xs: List[float]
    ) -> Dict[str, LocomotiveSimulation]:
        """
        conv loco specific override of cval.ModelError.update_params
        """
        assert (len(xs) == len(self.params))

        t0 = time.perf_counter()

        return_model_dict = {}

        for key, value in self.bincode_model_dict.items():
            return_model_dict[key] = LocomotiveSimulation.from_bincode(value)

        for key in return_model_dict.keys():
            for path, x in zip(self.params, xs):
            
                return_model_dict[key] = alt.set_param_from_path(
                    return_model_dict[key],
                    path,
                    x
                )
            # # override to set gen and edrv to have same peak efficiency
            # alt.set_param_from_path(
            #     return_model_dict[key],
            #     'loco_unit.edrv.eta_max',
            #     return_model_dict[key].loco_unit.gen.eta_max,
            # )
            # # make sure eta_interp is all same value to be consistent
            # alt.set_param_from_path(
            #     return_model_dict[key],
            #     'loco_unit.edrv.eta_range',
            #     0.0,
            # )

        t1 = time.perf_counter()
        if self.verbose:
            print(f"Time to update params: {t1 - t0:.3g} s")

        return return_model_dict


def get_mod_err(
    df_and_sim_dict: Dict[str, Tuple[pd.DataFrame, alt.LocomotiveSimulation]],
    model_objectives: List[Tuple[str, str]],
    model_params: Tuple[str],
    debug: Optional[bool] = False,
) -> ModelError:

    mod_err = ModelError(
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


def_save_path = Path("conv_loco_cal")
cal_plot_save_path = def_save_path / "plots/cal"
cal_plot_save_path.mkdir(exist_ok=True, parents=True)
val_plot_save_path = def_save_path / "plots/val"
val_plot_save_path.mkdir(exist_ok=True, parents=True)

if __name__ == '__main__':
    from utils import get_parser
    parser = get_parser(
        description='ALTRIOS Conventional Locomotive Calibration')

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
    # else:
    #     seed = time.time()
    n_proc = args.n_proc
    n_max_gen = args.n_max_gen
    pop_size = args.pop_size
    save_path = Path(args.save_path)
    file_info_path = Path(args.save_path)
    file_info_path.mkdir(exist_ok=True)
    pyplot = args.pyplot
    plotly = args.plotly
    show_pyplot = args.show_pyplot
    debug = args.debug
    seed = args.seed

    params_and_bounds = (
        # ("loco_unit.fc.eta_max", (0.35, 0.5)),
        # ("loco_unit.fc.eta_range",g (0.3, 0.48)),
        ("loco_unit.fc.pwr_idle_fuel_watts", (0, 20e3)),
        ("loco_unit.gen.eta_max", (0.88, 0.98)),
        # ("loco_unit.gen.eta_range", (0, 0.1)),
        ("loco_unit.pwr_aux_offset_watts", (0.0, 1_000_000)),
        ("loco_unit.pwr_aux_traction_coeff", (0.0, 0.1)),
        ("loco_unit.edrv.eta_max", (0.85, 0.99)),
        # ("loco_unit.edrv.eta_range", (0, 0.1)),
    )
    params = [pb[0] for pb in params_and_bounds]
    params_bounds = [pb[1] for pb in params_and_bounds]

    objectives = [
        (
            "Fuel Energy [J]",
            "loco_unit.fc.history.energy_fuel_joules"
        )
    ]

    artifact_dir = Path("conv_loco_cal")
    artifact_dir.mkdir(exist_ok=True)


    cal_mod_files, val_mod_files = utils.select_cal_and_val_trips(
        save_path=save_path,
        force_rerun=args.repartition,
    )

    # Dict[str, str] keys: reject file key, values: reason for rejecton
    rejects = {}

    cal_mod_raw_dfs = {}
    cal_mod_dfs = {}
    cal_df_and_sims = {}  # Dict[str, (df, sim)]
    for (i, file) in enumerate(cal_mod_files):
        if debug and (i > 5):
            break
        file = Path(file)
        print(f"Processing: {file.name}")
        raw_df = pd.read_csv(
            file, low_memory=False)
        if len(raw_df) < 1:
            rejects[file.name] = "len < 1"
            continue
        cal_mod_raw_dfs[file.stem] = raw_df
        cal_mod_dfs[file.stem] = get_conv_trip_mods(
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
        if debug and (i > 5):
            break
        file = Path(file)
        print(f"Processing: {file.name}")
        val_mod_raw_dfs[file.stem] = pd.read_csv(
            file, low_memory=False)
        val_mod_dfs[file.stem] = get_conv_trip_mods(
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
        problem = cval.CalibrationProblem(
            mod_err=cal_mod_err,
            n_constr=1,
            params_bounds=params_bounds,
        )
        res, res_df = cval.run_minimize(
            problem=problem,
            algorithm=algorithm,
            termination=termination,
            save_path=file_info_path,
        )
    else:
        assert n_proc > 1
        import multiprocessing
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
                save_path=file_info_path,
            )

    t1 = time.perf_counter()
    print(
        f"Number of processes: {n_proc}, Simulation time: {t1 - t0:.5f} seconds")

    optimal_params = cval.min_error_selection(
        res_df, param_num=len(cal_mod_err.params)
    )

    calibration_model_dict = cal_mod_err.update_params(optimal_params)
    cal_err = cal_mod_err.get_errors(
        calibration_model_dict,
        pyplot=pyplot,
        plotly=plotly,
        show_pyplot=show_pyplot,
        plot_save_dir=cal_plot_save_path,
    )

    validation_model_dict = val_mod_err.update_params(
        optimal_params)
    val_err = val_mod_err.get_errors(
        validation_model_dict,
        pyplot=pyplot,
        plotly=plotly,
        show_pyplot=show_pyplot,
        plot_save_dir=val_plot_save_path,
    )
