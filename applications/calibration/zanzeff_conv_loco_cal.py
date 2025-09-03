"""
Script for calibrating models for non-lead locomotives from Zanzeff data or,
if it's not available, from reasonably contrived data (aspirationally, TODO:
make this happen)
"""
# Notes from discussion with Garrett
# do the model deviations w.r.t. locomotive position occur at idle or under load?
# How much does aux load impact discrepancies?

import pprint
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import utils

import altrios as alt
from altrios import LocomotiveSimulation, pymoo_api
from altrios.pymoo_api import StarmapParallelization

CUR_FUEL_LHV_J__KG = 43e6

# loco = alt.Locomotive.from_file(
#     Path(__file__).parent / "resources/powertrains/fuel_converters/wabtec_tier4.yaml")
veh = alt.Locomotive.default()
veh_dict = veh.to_pydict()


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

        except Exception:
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


def get_loco_sim(df39xx: pd.DataFrame) -> LocomotiveSimulation:
    df39xx = df39xx.drop_duplicates("PacificTime").copy()
    df39xx["timestamp"] = pd.to_datetime(df39xx["PacificTime"]).dt.to_pydatetime()
    df39xx["time [s]"] = pymoo_api.get_delta_seconds(df39xx["timestamp"]).cumsum()
    trailing_loc = 3965
    df39xx["Tractive Power [W]"] = (
        df39xx["Tractive Effort Feedback BNSF " + str(trailing_loc)]
        * alt.utils.N_PER_LB
        * df39xx["Locomotive Speed GECX 3000"]
        * alt.utils.MPS_PER_MPH
    ).fillna(0.0)

    df39xx["Fuel Power [W]"] = (
        df39xx["Fuel Rate " + str(trailing_loc) + " [lbs/hr]"].fillna(0.0)
        * alt.utils.KG_PER_LB
        / 3600
        * CUR_FUEL_LHV_J__KG
    )
    df39xx["Fuel Energy [J]"] = (
        (df39xx["Fuel Power [W]"] * pymoo_api.get_delta_seconds(df39xx["timestamp"]))
        .cumsum()
        .copy()
    )
    df39xx["engine_on"] = df39xx["Engine Speed (RPM) BNSF " + str(trailing_loc)] > 100
    alt.PowerTrace.from_pydict(
        {"time_seconds": [5.0, 5.0], "pwr_watts": [5.0, 5.0], "engine_on": [False, False]},
    )
    powertrace = alt.PowerTrace.from_pydict(
        {
            "time_seconds": df39xx["time [s]"].to_list(),
            "pwr_watts": df39xx["Tractive Power [W]"].to_list(),
            "engine_on": df39xx.engine_on.to_list(),  # This is 39XX engine state (on/off)
        },
    )
    loco_unit = alt.Locomotive.default()

    loco_dict = loco_unit.to_pydict()

    # loco_dict["pwr_aux_offset_watts"] = 10e3
    # loco_dict["fc"]["pwr_ramp_lag_seconds"] = 0.0000000000000001
    # loco_dict["fc"]["pwr_out_max_watts"] = 3255000.0 * 2
    # loco_dict["edrv"]["pwr_out_max_watts"] = 3255000.0 * 2
    # loco_dict["gen"]["pwr_out_max_watts"] = 3255000.0 * 2

    loco_sim = LocomotiveSimulation(
        loco_unit=alt.Locomotive.from_pydict(loco_dict),
        power_trace=powertrace,
        save_interval=1,
    )
    return loco_sim


df_path = "ZANZEFF Data- Corrected GPS Plus Train Build ALTRIOS Confidential v2/"
# df_files = []
# for file in df_path.iterdir():
#     if file.suffix == ".csv":
#         df_files.append(file.name, pd.read_csv(file))
# PATH TO simulation csvs.iterdir(txt file only)
save_path = Path(__file__).parents[1] / "train_sim_cal"
if not save_path.exists():
    save_path.mkdir(parents=True)
# parser = pymoo_api.get_parser()
# args = parser.parse_args()
cal_files, val_files = utils.select_cal_and_val_trips(
    save_path=save_path,
    # force_rerun=args.repartition,
)
# val=30% cal=70%


dfs_for_cal: dict[str, pd.DataFrame] = {cal_file: pd.read_csv(cal_file) for cal_file in cal_files}
def_save_path = Path("conv_loco_cal")


# model pydict
# sims_for_cal: dict[str, alt.loco_sim] = {}
# populate `sims_for_cal`
# for loco_name, loco in train_sim["loco_con"]["loco_vec"].items():
#     loco_name: str
#     loco: alt.Locomotive
#     # NOTE: maybe change `save_interval` to 5
#     sims_for_cal[loco_name] = alt.loco_sim(
#         loco_unit, power_trace, save_interval).to_pydict()


def get_loco_sims(dfs: dict[str, pd.DataFrame]):
    loco_sims = {}
    for trip_name, df in dfs.items():
        loco_sims[trip_name] = get_loco_sim(df).to_pydict()
    return loco_sims


# Objective Functions -- `obj_fns`
def get_mod_fuel_energy(sim_dict: dict) -> npt.NDArray:
    return np.array(
        sim_dict["loco_unit"]["loco_type"]["ConventionalLoco"]["fc"]["history"][
            "energy_fuel_joules"
        ],
    )


def get_exp_fuel_energy(df: pd.DataFrame) -> pd.DataFrame:
    return df["Fuel Energy [J]"]


# Parameter Functions -- `param_fns_and_bounds`
def new_pwr_idle_fuel_watts(sim_dict: dict, new_val: float) -> dict:
    """Set `pwr_idle_fuel_watts` in `FuelConverter`"""
    sim_dict["loco_unit"]["loco_type"]["ConventionalLoco"]["fc"]["pwr_idle_fuel_watts"] = new_val
    return sim_dict


def new_gen_eta_max(sim_dict: dict, new_val: float) -> dict:
    """Set `eta_max` in `Generator`"""
    sim_dict["loco_unit"]["loco_type"]["ConventionalLoco"]["gen"]["eta_max"] = new_val
    return sim_dict


def new_pwr_aux_offset_watts(sim_dict: dict, new_val: float) -> dict:
    """Set `pwr_aux_offset_watts` in `ConventionalLocomotive`"""
    sim_dict["loco_unit"]["loco_type"]["ConventionalLoco"]["pwr_aux_offset_watts"] = new_val
    return sim_dict


def new_pwr_aux_traction_coeff(sim_dict: dict, new_val: float) -> dict:
    """Set `pwr_aux_traction_coeff` in `ConventionalLocomotive`"""
    sim_dict["loco_unit"]["loco_type"]["ConventionalLoco"]["pwr_aux_traction_coeff"] = new_val
    return sim_dict


def new_edrv_eta_max(sim_dict: dict, new_val: float) -> dict:
    """Set `eta_max` in `ElectricDrivetrain`"""
    sim_dict["loco_unit"]["loco_type"]["ConventionalLoco"]["edrv"]["eta_max"] = new_val
    return sim_dict


(("loco_unit.fc.pwr_idle_fuel_watts", (0, 20e3)),)
(("loco_unit.gen.eta_max", (0.88, 0.98)),)
(("loco_unit.pwr_aux_offset_watts", (0.0, 1_000_000)),)
(("loco_unit.pwr_aux_traction_coeff", (0.0, 0.1)),)
(("loco_unit.edrv.eta_max", (0.85, 0.99)),)

# Model Objectives
cal_mod_obj = pymoo_api.ModelObjectives(
    models=get_loco_sims(dfs_for_cal),
    dfs=dfs_for_cal,
    obj_fns=((get_mod_fuel_energy, get_exp_fuel_energy),),
    param_fns_and_bounds=(
        (new_pwr_idle_fuel_watts, (0, 20e3)),
        (new_gen_eta_max, (0.88, 0.98)),
        (new_pwr_aux_offset_watts, (0.0, 100_000)),
        (new_pwr_aux_traction_coeff, (0.0, 0.1)),
        (new_edrv_eta_max, (0.85, 0.99)),
    ),
    sim_type=alt.LocomotiveSimulation,
    constr_fns=(),
    verbose=False,
)


def perturb_params(pos_perturb_dec: float = 0.05, neg_perturb_dec: float = 0.1):
    """
    # Arguments:
    # - `pos_perturb_doc`: perturbation percentage added to all params.  Can be overridden invididually
    # - `neg_perturb_doc`: perturbation percentage subtracted from all params.  Can be overridden invididually
    """
    loco = alt.Locomotive.default()
    loco_dict = loco.to_pydict()
    baseline_params_and_bounds = [
        (loco_dict["loco_type"]["ConventionalLoco"]["fc"]["pwr_idle_fuel_watts"], None),
        (loco_dict["loco_type"]["ConventionalLoco"]["gen"]["eta_interp"][1], None),
        (loco_dict["pwr_aux_offset_watts"], None),
        (loco_dict["pwr_aux_traction_coeff"], None),
        (loco_dict["loco_type"]["ConventionalLoco"]["edrv"]["eta_interp"][1], None),
    ]

    baseline_params = [bpb[0] for bpb in baseline_params_and_bounds]

    print(
        "Verifying that model responds to input parameter changes by individually perturbing parameters",
    )
    baseline_errors = cal_mod_obj.get_errors(
        cal_mod_obj.update_params([param for param in baseline_params]),
    )[0]

    for i, param_and_bounds in enumerate(baseline_params_and_bounds):
        param = param_and_bounds[0]
        bounds = param_and_bounds[1]
        # +5%
        if bounds is not None:
            param_pos_perturb_dec = bounds[0]
            param_neg_perturb_dec = bounds[1]
        else:
            param_pos_perturb_dec = pos_perturb_dec
            param_neg_perturb_dec = neg_perturb_dec

        assert param_pos_perturb_dec >= 0
        assert param_neg_perturb_dec >= 0

        perturbed_params = baseline_params.copy()
        perturbed_params[i] = param * (1 + param_pos_perturb_dec)
        perturbed_errors = cal_mod_obj.get_errors(cal_mod_obj.update_params(perturbed_params))
        if np.all(perturbed_errors == baseline_errors):
            print("\nperturbed_errros:")
            pprint.pp(perturbed_errors)
            print("baseline_errors")
            pprint.pp(baseline_errors)
            print("")
            raise Exception(
                f"+{100 * param_pos_perturb_dec}% perturbation failed for param {cal_mod_obj.param_fns[i].__name__}",
            )

        # -5%
        perturbed_params = baseline_params.copy()
        perturbed_params[i] = param * (1 - param_neg_perturb_dec)
        perturbed_errors = cal_mod_obj.get_errors(cal_mod_obj.update_params(perturbed_params))
        if np.all(perturbed_errors == baseline_errors):
            print("\nperturbed_errros:")
            pprint.pp(perturbed_errors)
            print("baseline_errors")
            pprint.pp(baseline_errors)
            print("")
            raise Exception(
                f"-{100 * param_neg_perturb_dec}% perturbation failed for param {cal_mod_obj.param_fns[i].__name__}",
            )

    print("Success!")


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
