# Notes from discussion with Garrett
# do the model deviations w.r.t. locomotive position occur at idle or under load?
# How much does aux load impact discrepancies?

import pprint
from pathlib import Path

import pandas as pd
import numpy as np
import numpy.typing as npt

import altrios as alt
from altrios import LocomotiveSimulation, pymoo_api
from altrios.pymoo_api import StarmapParallelization

CUR_FUEL_LHV_J__KG = 43e6


def get_conv_trip_mods(df: pd.DataFrame) -> pd.DataFrame:
    """Given single trip data frame, return dataframe for non-lead locomotive"""
    df = df.drop_duplicates("PacificTime").copy()
    df["timestamp"] = pd.to_datetime(df["PacificTime"]).dt.to_pydatetime()
    df["time [s]"] = pymoo_api.get_delta_seconds(df["timestamp"]).cumsum()

    df39xx = df[["timestamp", "time [s]"]].copy()

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

    df39xx["Tractive Power [W]"] = (
        df["Tractive Effort Feedback BNSF " + str(trailing_loc)]
        * alt.utils.N_PER_LB
        * df["Locomotive Speed GECX 3000"]
        * alt.utils.MPS_PER_MPH
    ).fillna(0.0)

    df39xx["Fuel Power [W]"] = (
        df["Fuel Rate " + str(trailing_loc) + " [lbs/hr]"].fillna(0.0)
        * alt.utils.KG_PER_LB
        / 3600
        * CUR_FUEL_LHV_J__KG
    )
    df39xx["Fuel Energy [J]"] = (
        (df39xx["Fuel Power [W]"] * pymoo_api.get_delta_seconds(df39xx["timestamp"]))
        .cumsum()
        .copy()
    )
    df39xx["engine_on"] = df["Engine Speed (RPM) BNSF " + str(trailing_loc)] > 100
    return df39xx


def get_loco_sim(df39xx: pd.DataFrame) -> bytes:
    powertrace = alt.PowerTrace(
        df39xx["time [s]"].to_numpy(),
        df39xx["Tractive Power [W]"].to_numpy(),
        df39xx.engine_on,  # This is 39XX engine state (on/off)
    )
    loco_unit = alt.Locomotive.default()

    loco_dict = loco_unit.to_pydict()

    loco_dict["pwr_aux_offset_watts"] = 10e3
    loco_dict["fc"]["pwr_ramp_lag_seconds"] = 0.0000000000000001
    loco_dict["fc"]["pwr_out_max_watts"] = 3255000.0 * 2
    loco_dict["edrv"]["pwr_out_max_watts"] = 3255000.0 * 2
    loco_dict["gen"]["pwr_out_max_watts"] = 3255000.0 * 2

    loco_sim = LocomotiveSimulation(
        loco_unit=alt.Locomotive.from_pydict(loco_dict),
        power_trace=powertrace,
        save_interval=1,
    )
    loco_sim_bincode = loco_sim.to_bincode()
    return loco_sim_bincode


def_save_path = Path("conv_loco_cal")


# Objective Functions -- `obj_fns`
def get_mod_fuel_energy(sim_dict: dict) -> npt.NDArray:
    return np.array(
        sim_dict["loco_type"]["ConventionalLocomotive"]["fc"]["history"]["fuel_energy_joules"]
    )


def get_exp_fuel_energy(df: pd.DataFrame) -> pd.DataFrame:
    return df["Fuel Energy [J]"]


# Parameter Functions -- `param_fns_and_bounds`
def new_pwr_idle_fuel_watts(sim_dict: dict, new_val: float) -> dict:
    """
    Set `pwr_idle_fuel_watts` in `FuelConverter`
    """
    sim_dict["loco_type"]["ConventionalLocomotive"]["fc"][""]
    return sim_dict


(("loco_unit.fc.pwr_idle_fuel_watts", (0, 20e3)),)
(("loco_unit.gen.eta_max", (0.88, 0.98)),)
(("loco_unit.pwr_aux_offset_watts", (0.0, 1_000_000)),)
(("loco_unit.pwr_aux_traction_coeff", (0.0, 0.1)),)
(("loco_unit.edrv.eta_max", (0.85, 0.99)),)

# Model Objectives
cal_mod_obj = pymoo_api.ModelObjectives(
    models=sims_for_cal,
    dfs=dfs_for_cal,
    obj_fns=((get_mod_fuel_energy, get_exp_fuel_energy),),
    param_fns_and_bounds=(
        (new_em_eff_max, (0.80, 0.99)),  # new_em_eff_max,
        (new_em_eff_range, (0.1, 0.6)),  # new_em_eff_range,
        # new_cab_shell_htc_w_per_m2_k,
        (new_cab_shell_htc_w_per_m2_k, (10, 350)),
        # new_cab_htc_to_amb_stop_w_per_m2_k,
        (new_cab_htc_to_amb_stop_w_per_m2_k, (10, 250)),
        (new_cab_tm_j_per_k, (50e3, 350e3)),  # new_cab_tm_j_per_k,
        (new_cab_length_m, (1.5, 7)),  # new_cab_length_m,
        (new_res_cndctnc_to_amb, (1, 60)),  # new_res_cndctnc_to_amb,
        (new_res_cndctnc_to_cab, (1, 60)),  # new_res_cndctnc_to_cab,
        (new_res_tm_j_per_k, (30e3, 200e3)),  # new_res_tm_j_per_k,
        (new_hvac_p_res_w_per_k, (5, 1_000)),  # new_hvac_p_res_w_per_k,
        (new_hvac_i_res, (1, 100)),  # new_hvac_i_res,
        # (new_hvac_d_res, (1, 100)),  # new_hvac_d_res,
        (new_hvac_p_cabin_w_per_k, (5, 1_000)),  # new_hvac_p_cabin_w_per_k,
        (new_hvac_i_cabin, (1, 100)),  # new_hvac_i_cabin,
        # (new_hvac_d_cabin, (1, 100)),  # new_hvac_d_cabin,
        # new_hvac_frac_of_ideal_cop,
        (new_hvac_frac_of_ideal_cop, (0.15, 0.35)),
    ),
    sim_type=alt.LocomotiveSimulation,
    constr_fns=(),
    verbose=False,
)

if __name__ == "__main__":
    print("Params and bounds:")
    pprint.pp(cal_mod_obj.param_fns_and_bounds)
    print("")
    perturb_params()
    parser = pymoo_api.get_parser()
    args = parser.parse_args()

    n_processes = args.processes
    n_max_gen = args.n_max_gen
    # should be at least as big as n_processes
    pop_size = args.pop_size
    run_minimize = not (args.skip_minimize)

    print(f"Starting calibration with: {args}.")
    algorithm = pymoo_api.NSGA2(
        # size of each population
        pop_size=pop_size,
        # LatinHyperCube sampling seems to be more effective than the default
        # random sampling
        sampling=pymoo_api.LHS(),
    )
    termination = pymoo_api.DMOT(
        # max number of generations, default of 10 is very small
        n_max_gen=n_max_gen,
        # evaluate tolerance over this interval of generations every
        period=5,
        # parameter variation tolerance
        xtol=args.xtol,
        # objective variation tolerance
        ftol=args.ftol,
    )

    if n_processes == 1:
        print("Running serial evaluation.")
        # series evaluation
        # Setup calibration problem
        cal_prob = pymoo_api.CalibrationProblem(
            mod_obj=cal_mod_obj,
        )

        res, res_df = pymoo_api.run_minimize(
            problem=cal_prob,
            algorithm=algorithm,
            termination=termination,
            save_path=save_path,
        )
    else:
        print(f"Running parallel evaluation with n_processes: {n_processes}.")
        assert n_processes > 1
        # parallel evaluation
        import multiprocessing

        with multiprocessing.Pool(n_processes) as pool:
            problem = pymoo_api.CalibrationProblem(
                mod_obj=cal_mod_obj,
                elementwise_runner=StarmapParallelization(pool.starmap),
            )
            res, res_df = pymoo_api.run_minimize(
                problem=problem,
                algorithm=algorithm,
                termination=termination,
                save_path=save_path,
            )
