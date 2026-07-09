"""Module containing functions and classes for easy interaction with PyMOO"""

import argparse
import pprint  # noqa: F401
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

# pymoo
try:
    from pymoo.algorithms.base.genetic import GeneticAlgorithm
    from pymoo.algorithms.moo.nsga2 import NSGA2  # noqa: F401
    from pymoo.algorithms.moo.nsga3 import NSGA3  # noqa: F401

    # Imports for convenient use in scripts
    from pymoo.core.problem import (
        ElementwiseProblem,
        LoopedElementwiseEvaluation,
        StarmapParallelization,  # noqa: F401
    )
    from pymoo.core.result import Result
    from pymoo.operators.sampling.lhs import LatinHypercubeSampling as LHS  # noqa: F401
    from pymoo.optimize import minimize
    from pymoo.termination.default import DefaultMultiObjectiveTermination as DMOT
    from pymoo.util.display.column import Column
    from pymoo.util.display.output import Output
    from pymoo.util.ref_dirs import get_reference_directions  # noqa: F401

    PYMOO_AVAILABLE = True
except ModuleNotFoundError as err:
    print(
        f"{err}\nTry running `pip install pymoo==0.6.0.1` to use all features in "
        + "`altrios.pymoo_api`",
    )
    PYMOO_AVAILABLE = False

import altrios as alt


def get_error_val(
    model: npt.NDArray[np.float64],
    test: npt.NDArray[np.float64],
    time_steps: npt.NDArray[np.float64],
) -> float:
    """
    Return time-averaged error for model and test signal.

    # Args:
        - `model`: array of values for signal from model
        - `test`: array of values for signal from test data
        - `time_steps`: array (or scalar for constant) of values for model time steps [s]

    # Returns:
        - `error`: integral of absolute value of difference between model and test per time
    """
    assert len(model) == len(test) == len(time_steps), (
        f"{len(model)}, {len(test)}, {len(time_steps)}"
    )

    err_val: float = np.trapezoid(y=abs(model - test), x=time_steps) / (  # type: ignore[attr-defined]
        time_steps[-1] - time_steps[0]
    )
    return err_val


@dataclass
class ModelObjectives:
    """
    Class for calculating eco-driving objectives

    # Attributes/Fields
    - `models` (Dict[str, Dict]): dictionary of model dicts to be simulated
    - `dfs` (Dict[str, pd.DataFrame]): dictionary of dataframes from test data
      corresponding to `models`
    - `obj_fns` (Tuple[Callable] | Tuple[Tuple[Callable, Callable]]):
      Tuple of functions for extracting objective signal values for either minimizing a
      scalar metric (e.g. fuel economy) or minimizing error relative to test
      data.
        - minimizing error in fuel consumption relative to test data
          ```
          obj_fns = (
              (
                  # model -- note, `lambda` only works for single thread
                  lambda sim_dict: sim_dict['veh']['pt_type']['Conventional']['fc']['history']['energy_fuel_joules'],
                  # test data
                  lambda df: df['fuel_flow_gps'] * ... (conversion factors to get to same unit),
              ), # note that this trailing comma ensures `obj_fns` is interpreted as a tuple
          )
          ```
        - minimizing fuel consumption
          ```
          obj_fns = (
              (
                  # note that a trailing comma ensures `obj_fns` is interpreted as tuple
                  lambda sim_dict: sim_dict['veh']['pt_type']['Conventional']['fc']['state']['energy_fuel_joules'],
              )
          )
          ```
    - `param_fns_and_bounds` Tuple[Tuple[Callable], Tuple[Tuple[float, float]]]:
      tuple containing functions to modify parameters and bounds for optimizer
      Example
      ```
      def new_peak_res_eff (sim_dict, new_peak_eff) -> Dict:
          sim_dict['veh']['pt_type']['HybridElectricVehicle']['res']['peak_eff'] = new_peak_eff
          return sim_dict
      ...
      param_fns_and_bounds = (
          (new_peak_res_eff, (100.0, 200.0)),
      )
      ```
    - `verbose` (bool): print more stuff or not
    """  # noqa: E501

    models: dict[str, dict]
    dfs: dict[str, pd.DataFrame]
    obj_fns: tuple[Callable] | tuple[tuple[Callable, Callable]]
    constr_fns: tuple[Callable]
    param_fns_and_bounds: tuple[tuple[Callable], tuple[tuple[float, float]]]
    sim_type: alt.SerdeAPI
    param_fns: tuple[Callable] = field(init=False)
    bounds: tuple[tuple[float, float]] = field(init=False)

    # if True, prints timing and misc info
    verbose: bool = False

    # calculated in __post_init__
    n_obj: int | None = None
    n_constr: int | None = None

    def __post_init__(self):
        """Initialize special attributes and check for proper setup"""
        assert self.n_obj is None, "`n_obj` is not intended to be user provided"
        assert len(self.dfs) == len(self.models), f"{len(self.dfs)} != {len(self.models)}"
        self.param_fns = tuple(pb[0] for pb in self.param_fns_and_bounds)
        self.bounds = tuple(pb[1] for pb in self.param_fns_and_bounds)
        assert len(self.bounds) == len(self.param_fns)
        self.n_obj = len(self.models) * len(self.obj_fns)
        self.n_constr = len(self.models) * len(self.constr_fns)

    def update_params(self, xs: list[Any]):
        """Update model parameters based on `x`, which must match length of self.param_fns"""
        assert len(xs) == len(self.param_fns), f"({len(xs)} != {len(self.param_fns)}"

        t0 = time.perf_counter()

        # Instantiate SimDrive objects
        sim_drives: dict[str, alt.SerdeAPI | Exception] = {}

        # Update all model parameters
        for key, pydict in self.models.items():
            for param_fn, new_val in zip(self.param_fns, xs):
                pydict = param_fn(pydict, new_val)
            # this assignement may be redundant, but `pydict` is probably **not** mutably modified.
            # If this is correct, then this assignment is necessary
            self.models[key] = pydict

        for key, pydict in self.models.items():
            try:
                sim_drives[key] = alt.SerdeAPI.from_pydict(pydict, skip_init=False)
            except Exception as err:
                sim_drives[key] = err
        t1 = time.perf_counter()
        if self.verbose:
            print(f"Time to update params: {t1 - t0:.3g} s")
        return sim_drives

    def get_errors(
        self,
        sim_drives: dict[str, alt.SerdeAPI],
        return_mods: bool = False,
    ) -> (
        tuple[dict[str, list[float]], dict[str, list[float]]]
        | tuple[
            dict[str, list[float]],
            dict[str, list[float]],
            dict[str, alt.SerdeAPI | Any],
            dict[str, alt.SerdeAPI | Any],
        ]
    ):
        """
        Calculate model errors w.r.t. test data for each element in dfs/models for each objective.

        # Args:
            - `sim_drives`: dictionary with user-defined keys and SimDrive instances
            - `return_mods`: if true, also returns dict of solved models. Defaults to False.

        # Returns:
            Objectives and optionally solved models
        """
        objectives: dict[str, list[float]] = {}
        constraint_violations: dict[str, list[float]] = {}
        solved_mods: dict[str, alt.SerdeAPI | Any] = {}
        unsolved_mods: dict[str, alt.SerdeAPI | Any] = {}

        # loop through all the provided trips
        for (key, df_exp), sim in zip(self.dfs.items(), sim_drives.values()):
            df_exp = self.append_all_objs_for_trip(
                return_mods,
                sim,
                unsolved_mods,
                key,
                df_exp,
                objectives,
                constraint_violations,
                solved_mods,
            )
        # print("\nobjectives:")
        # pprint.pp(objectives)
        # print("")
        if return_mods:
            return objectives, constraint_violations, solved_mods, unsolved_mods
        else:
            return objectives, constraint_violations

    def append_all_objs_for_trip(
        self,
        return_mods: bool,
        sim,
        unsolved_mods,
        key,
        df_exp,
        objectives,
        constraint_violations,
        solved_mods,
    ):
        """
        Append objectives for all trips

        # Arguments
        - `return_mods`: whether to return solved models
        - `sim_dict`: dictionary of solved simulations
        - `df_exp`: dictionary of reference data for calibration
        - `walk_success`: whether the simulation succeeded
        - `objectives`: objectives to mutably append for each trip
        """
        if return_mods:
            unsolved_mods[key] = sim.to_pydict()

        try:
            t0 = time.perf_counter()
            sim.walk_once()  # type: ignore
            t1 = time.perf_counter()
            sim_dict = sim.to_pydict()
            walk_success = True
        except RuntimeError as err:
            t1 = time.perf_counter()
            sim_dict = sim.to_pydict()
            walk_success = True
            print(err)
            if len(sim_dict["veh"]["history"]["time_seconds"]) < np.floor(len(df_exp) / 2):
                walk_success = False

        if self.verbose:
            print(f"Time to simulate {key}: {t1 - t0:.3g}")

        objectives[key] = []
        constraint_violations[key] = []

        if return_mods:
            solved_mods[key] = sim_dict

        # trim dataframe to match length of cycle completed by vehicle
        df_exp = df_exp[: len(sim_dict["veh"]["history"]["time_seconds"])]

        # loop through the objectives for each trip
        for i_obj, obj_fn in enumerate(self.obj_fns):
            obj_fn = cast(Sequence, obj_fn)
            obj_fn = cast(Callable, obj_fn)
            obj_fn = self.append_obj_for_trip(
                obj_fn,
                sim_dict,
                df_exp,
                walk_success,
                objectives,
                key,
            )
        for constr_fn in self.constr_fns:
            constraint_violations[key].append(constr_fn(sim_dict))

        t2 = time.perf_counter()
        if self.verbose:
            print(f"Time to postprocess: {t2 - t1:.3g} s")
        return df_exp

    def append_obj_for_trip(
        self,
        obj_fn: Callable | tuple[Callable, Callable],
        sim_dict,
        df_exp,
        walk_success,
        objectives,
        key,
    ):
        """
        Append objectives for each trip

        # Arguments
        - `obj_fn`: objective function
        - `sim_dict`: dictionary of solved simulations
        - `df_exp`: dictionary of reference data for calibration
        - `walk_success`: whether the simulation succeeded
        - `objectives`: objectives to loop through for each trip


        """
        if isinstance(obj_fn, Sequence):
            if len(obj_fn) == 2:
                # objective and reference passed
                mod_sig = obj_fn[0](sim_dict)
                ref_sig = obj_fn[1](df_exp)
            else:
                raise ValueError("Each element in `self.obj_fns` must have length of 1 or 2")
        else:
            # minimizing scalar objective
            mod_sig = obj_fn(sim_dict)
            ref_sig = None

        if ref_sig is not None:
            time_s = sim_dict["veh"]["history"]["time_seconds"]
            # TODO: provision for incomplete simulation in here somewhere

            if not walk_success:
                objectives[key].append(1.02e12)
            else:
                try:
                    objectives[key].append(
                        get_error_val(
                            mod_sig,
                            ref_sig,
                            time_s,
                        ),
                    )
                except AssertionError:
                    # `get_error_val` checks for length equality with an assertion
                    # If length equality is not satisfied, this design is
                    # invalid because the cycle could not be completed.
                    # NOTE: instead of appending an arbitrarily large
                    # objective value, we could instead either try passing
                    # `np.nan` or trigger a constraint violation.
                    objectives[key].append(1.03e12)
        else:
            raise Exception("this is here for debugging and should be deleted")
            objectives[key].append(mod_sig)
        return obj_fn


if PYMOO_AVAILABLE:

    class CalibrationProblem(ElementwiseProblem):
        """Problem for calibrating models to match test data"""

        def __init__(
            self,
            mod_obj: ModelObjectives,
            elementwise_runner=LoopedElementwiseEvaluation(),
        ):
            self.mod_obj = mod_obj
            super().__init__(
                n_var=len(self.mod_obj.param_fns),
                n_obj=self.mod_obj.n_obj,
                xl=[bounds[0] for bounds in self.mod_obj.bounds],
                xu=[bounds[1] for bounds in self.mod_obj.bounds],
                elementwise_runner=elementwise_runner,
                n_ieq_constr=self.mod_obj.n_constr,
            )

        def _evaluate(self, x, out, *args, **kwargs):  # noqa: ARG002
            sim_drives = self.mod_obj.update_params(x)
            (errs, cvs) = self.mod_obj.get_errors(sim_drives)
            out["F"] = list(errs.values())
            if self.n_ieq_constr > 0:
                out["G"] = list(cvs.values())

    class CustomOutput(Output):
        """Adds custom columns to pymoo output"""

        def __init__(self):
            super().__init__()
            self.t_gen_start = time.perf_counter()
            self.n_nds = Column("n_nds", width=8)
            self.t_s = Column("t [s]", width=10)
            self.euclid_min = Column("euclid min", width=13)
            self.columns += [self.n_nds, self.t_s, self.euclid_min]

        def update(self, algorithm):
            """Update new row"""
            super().update(algorithm)
            self.n_nds.set(len(algorithm.opt))
            self.t_s.set(f"{(time.perf_counter() - self.t_gen_start):.3g}")
            f = algorithm.pop.get("F")
            euclid_min = np.sqrt((np.array(f) ** 2).sum(axis=1)).min()
            self.euclid_min.set(f"{euclid_min:.3g}")

    def run_minimize(
        problem: CalibrationProblem,
        algorithm: GeneticAlgorithm,
        termination: DMOT,
        save_path: Path | str,
        copy_algorithm: bool = False,
        copy_termination: bool = False,
        save_history: bool = False,
    ) -> tuple[Result, pd.DataFrame]:
        """Wrap pymoo.optimize.minimize and add various helpful features"""
        print("`run_minimize` starting at")
        alt.utils.print_dt()

        t0 = time.perf_counter()
        res = minimize(
            problem,
            algorithm,
            termination,
            copy_algorithm=copy_algorithm,
            copy_termination=copy_termination,
            seed=1,
            verbose=True,
            save_history=save_history,
            output=CustomOutput(),
        )

        f_columns = [
            f"{key.split(' ')[0]}: {cast(Sequence, obj)[0].__name__}"
            for key in problem.mod_obj.dfs
            for obj in problem.mod_obj.obj_fns
        ]
        f_df = pd.DataFrame(
            data=list(res.F.tolist()),
            columns=f_columns,
        )

        x_df = pd.DataFrame(
            data=list(res.X.tolist()),
            columns=[param.__name__ for param in problem.mod_obj.param_fns],
        )

        Path(save_path).mkdir(exist_ok=True, parents=True)

        res_df = pd.concat([x_df, f_df], axis=1)
        res_df["euclidean"] = (
            (res_df.iloc[:, len(problem.mod_obj.param_fns) :] ** 2).sum(1).pow(1 / 2)
        )
        res_df.to_csv(Path(save_path) / "pymoo_res_df.csv", index=False)

        t1 = time.perf_counter()
        print(f"Elapsed time to run minimization: {t1 - t0:.5g} s")

        return res, res_df


def get_parser(
    def_description: str = "Program for calibrating altrios models.",
    def_p: int = 4,
    def_n_max_gen: int = 500,
    def_pop_size: int = 12,
) -> argparse.ArgumentParser:
    """
    Generate parser for optimization hyper params and misc. other params

    # Args:
        - `def_p`: default number of processes
        - `def_n_max_gen`: max allowed generations
        - `def_pop_size`: default population size

    # Returns:
        argparse.ArgumentParser: _description_
    """
    parser = argparse.ArgumentParser(description=def_description)
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=def_p,
        help=f"Number of pool processes. Defaults to {def_p}",
    )
    parser.add_argument(
        "--n-max-gen",
        type=int,
        default=def_n_max_gen,
        help=f"PyMOO termination criterion: n_max_gen. Defaults to {def_n_max_gen}",
    )
    parser.add_argument(
        "--xtol",
        type=float,
        default=DMOT().x.termination.tol,
        help=f"PyMOO termination criterion: xtol. Defaluts to {DMOT().x.termination.tol}",
    )
    parser.add_argument(
        "--ftol",
        type=float,
        default=DMOT().f.termination.tol,
        help=f"PyMOO termination criterion: ftol. Defaults to {DMOT().f.termination.tol}",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=def_pop_size,
        help=f"PyMOO population size in each generation. Defaults to {def_pop_size}",
    )
    parser.add_argument(
        "--skip-minimize",
        action="store_true",
        help="If provided, load previous results.",
    )
    # parser.add_argument(
    #     '--show',
    #     action="store_true",
    #     help="If provided, shows plots."
    # )
    # parser.add_argument(
    #     "--make-plots",
    #     action="store_true",
    #     help="Generates plots, if provided."
    # )

    return parser


def get_delta_seconds(ds: pd.Series) -> pd.Series:
    """
    Get time delta in seconds from series

    Arugments:
    ----------
    - ds: pandas.Series; data of the current segment previously passed to to_datetime_from_format
    returns:
    - out: pandas.Series; a pandas.Series data that shows the datetime deltas between rows of the
    segment. Row i has time elasped between `i` and row `i-1`. Row 0 has value 0.
    Returns pd.Series of time delta [s]
    """
    out = (ds - ds.shift(1)) / np.timedelta64(1, "s")
    out.iloc[0] = 0.0
    return out
