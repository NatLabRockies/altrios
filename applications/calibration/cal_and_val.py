"""
Module for running train, locomotive, and/or consist models to calibrate and validate 
against test data.
"""

from dataclasses import dataclass
from typing import Union, Dict, Tuple, List, Any, Optional
from typing_extensions import Self
from pathlib import Path
import pickle

# pymoo
from pymoo.util.display.output import Output
from pymoo.util.display.column import Column
from pymoo.operators.sampling.lhs import LatinHypercubeSampling as LHS
from pymoo.termination.default import DefaultMultiObjectiveTermination as DMOT
from pymoo.core.problem import Problem, ElementwiseProblem, LoopedElementwiseEvaluation, StarmapParallelization
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.optimize import minimize

# misc
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import json
import numpy.typing as npt
import re
sns.set_theme()

# local
from altrios import SetSpeedTrainSim, ConsistSimulation, LocomotiveSimulation
import altrios as alt

def get_delta_seconds(ds: pd.Series) -> pd.Series:
    """
    Arugments:
    ---------- 
    - ds: pandas.Series; data of the current segment previously passed to to_datetime_from_format
    returns: 
    - out: pandas.Series; a pandas.Series data that shows the datetime deltas between rows of the 
    segment. Row i has time elasped between `i` and row `i-1`. Row 0 has value 0.

    Returns pd.Series of time delta [s]
    """
    out = ((ds - ds.shift(1)) / np.timedelta64(1, 's'))
    out.iloc[0] = 0.0
    return out

def get_error(t: np.array, mod: np.array, exp: np.array):
    """
    Return error for model data, `mod`, w.r.t. experimental data, `exp`, over time, `t` 
    """
    err = np.trapz(y=abs(mod - exp), x=t) / (t[-1] - t[0])
    return err

@dataclass
class ModelError(object):
    """
    Dataclass class for calculating model error of various ALTRIOS objects w.r.t. test data.

    Attributes:
    - `ser_model_dict`:  `dict` variable in which:

    - key: a `str` representing trip keyword string

    - value: a `str` converted from Rust locomotive models' serialization method

    - `model_type`: `str` that can only be `'ConsistSimulation'`, `'SetSpeedTrainSim'` or `'LocomotiveSimulation'`;
      indicates which model to instantiate during optimization process

    - `dfs`: a `dict` variable in which:
        - key: `str` representing trip keyword; will be the same keyword as in `models`
        - value: `pandas.DataFrame` variable with trip detailed data to be compared against. each df should 
                 have a `time [s]` column

    - `objectives`: a list of 2-element tuples. For each tuple, element 0 is the name of the reference test 
      data signal in `dfs`; element 1 is a tuple of strings representing a hierarchical path to the 
      corresponding model signal. This field is used for error calculation.

    - `params`: a tuple whose individual element is a `str` containing hierarchical paths to parameters
      to manipulate starting from one of the 3 possible Rust model structs

    - `verbose`: `bool`: if `True`, the verbose of error calculation will be printed 

    - `debug`: `bool`: if `True`, prints more stuff

    - `allow_partial`: whether to allow partial runs, if True, errors out whenever a run can't be completed
    """
    # `ser_model_dict` and `dfs` should have the same keys
    ser_model_dict: Dict[str, str]
    # model_type: tells what model if class instance tries to intantiate when `get_errors()` is called
    model_type: str
    # dictionary of test data
    dfs: Dict[str, pd.DataFrame]
    # list of 2-element tuples of objectives for error calcluation
    objectives: List[Tuple[str, str]]
    # list of tuples hierarchical paths to parameters to manipulate
    params: Tuple[str]
    # if True, prints timing and misc info
    verbose: bool = False
    # if True, runs in debug mode
    debug: Optional[bool] = False
    # whether to allow partial runs, if True, errors out whenever a run can't be completed
    allow_partial: bool = True

    def __post_init__(self):
        assert (len(self.dfs) == len(self.ser_model_dict))
        self.n_obj = len(self.objectives) * len(self.ser_model_dict)
    # current placeholder function; to be over-written to provide constraint violation function

    @classmethod
    def load(cls, save_path: Path) -> Self:
        with open(Path(save_path), 'rb') as file:
            return pickle.load(file)

    def get_constr_viol(self):
        pass

    def get_errors(
        self,
        mod_dict,
        return_mods: Optional[bool] = False,
        pyplot: bool = False,
        plotly: bool = False,
        show_pyplot: bool = False,
        plot_save_dir: Optional[Path] = None,
        plot_perc_err: bool = False,
        font_size: float = 16,
        perc_err_target_for_plot: float = 1.5,
        # raise_errs: bool = False,
    ) -> Tuple[Dict[str, Dict[str, float]],  # error dict
               # if return_mods is True, solved models
               Optional[Tuple[
                   Dict[str, Dict[str, float]],
                   Dict[
                       str,
                       Union[
                           SetSpeedTrainSim,
                           ConsistSimulation,
                           LocomotiveSimulation
                       ],
                   ]
               ]
    ]
    ]:
        """
        Calculate model errors w.r.t. test data for each element in dfs/models for each objective.
        Arugments:
        ----------
            - mod_dict: the dict whose values are generated Rust ALTRIOS models 
            - return_mods: `bool`; if true, also returns dict of solved models
            - pyplot: if true, plots objectives with matplotlib.pyplot
            - plotly: if true, plots with plotly.  `plot_save_dir` must be provided.
            - show_pyplot: if true, shows pyplot plots
            - plot_save_dir: Path for saving plots.  
            - plot_perc_err: Whether to include axes for plotting % error

        Returns:
        ----------
            - errors: `dict` whose values are dicts containing the errors wrt each objective
            - solved_mods Optional; `dict` whose values are the Rust locomotive models; 
              only returned when `return_mods` is True
        """

        errors: Dict[str, Optional[float]] = {}
        solved_mods = {}

        # loop through all the provided trips
        for (i, ((key, df_exp), mod0)) in enumerate(zip(self.dfs.items(), mod_dict.values())):
            if self.verbose:
                print(f"Currently processing file: {key}")
            # Setup boundary conditions appropriately based on model type    
            if self.model_type == 'LocomotiveSimulation':
                bc = np.array((mod0.power_trace.pwr_watts).tolist())
                time_seconds = np.array(mod0.power_trace.time_seconds.tolist())
            elif self.model_type == 'SetSpeedTrainSim':
                bc = np.array(mod0.speed_trace.speed_meters_per_second)
                time_seconds = np.array(mod0.speed_trace.time_seconds.tolist())
            elif self.model_type == 'ConsistSimulation':
                bc = np.array((mod0.power_trace.pwr_watts).tolist())
                time_seconds = np.array(mod0.power_trace.time_seconds.tolist())
            else:
                raise AttributeError("Invalid model type.")

            mod = mod0.clone()

            t0 = time.perf_counter()
            try:
                mod.walk()
            except Exception as err:
                if self.debug:
                    print(key)
                    print(err)
                if self.model_type in ["LocomotiveSimulation", "ConsistSimulation"]:
                    i = mod.i
                    len_trace = len(mod.power_trace)
                else:
                    i = mod.state.i
                    len_trace = len(mod.speed_trace)
                if i <= 1:
                    raise Exception(f"{err}\nModel failed to get past first time step.")
                elif (i < len_trace) and self.allow_partial:
                    # trim off portion of trip that did not get simulated
                    mod.trim_failed_steps()
                else: 
                    raise Exception(
                        f"{err}\nModel failed to run past time step {i}" 
                        + f" of {len_trace}."
                    )

            t1 = time.perf_counter()
            if self.verbose:
                print(f"Simulation time: {t1 - t0:.3g} seconds")

            errors[key] = {}
            if return_mods or show_pyplot:
                solved_mods[key] = mod.clone()

            # matplotlib.pyplot
            plots_per_key = 2 if plot_perc_err else 1
            
            fig, axes, pltly_fig = self.setup_plots(
                key=key,
                mod=mod,
                time_seconds=time_seconds, 
                bc=bc, 
                plots_per_key=plots_per_key,
                pyplot=pyplot,
                plotly=plotly,
                plot_save_dir=plot_save_dir,
            )

            # loop through the objectives for each trip
            for i_obj, obj in enumerate(self.objectives):
                mod_path = obj[1].split(".")
                if self.verbose:
                    print(mod_path)
                # extract signal values fo `obj`

                mod_sig = mod  # placeholder
                for elem in mod_path:
                    if self.verbose:
                        print(
                            f"Iterating through structure {type(mod_sig)}.{elem}")
                    mod_sig = mod_sig.__getattribute__(elem)
                mod_sig = np.array((mod_sig).tolist())
                exp_sig = df_exp[obj[0]].to_numpy()
                errors[key][obj[0]] = get_error(
                    df_exp['time [s]'].to_numpy()[:len(mod_sig)],
                    exp_sig[:len(mod_sig)],
                    mod_sig
                )

                fig, axes, pltly_fig = self.populate_plots(
                    fig=fig,
                    axes=axes,
                    pltly_fig=pltly_fig, 
                    show_pyplot=show_pyplot,
                    plot_save_dir=plot_save_dir, 
                    i_obj=i_obj,
                    plots_per_key=plots_per_key,
                    key=key,
                    time_seconds=time_seconds,
                    plot_perc_err=plot_perc_err,
                    mod_sig=mod_sig,
                    exp_sig=exp_sig,
                    obj=obj, 
                    font_size=font_size,
                    perc_err_target_for_plot=perc_err_target_for_plot,
                )

            if pyplot:
                assert fig is not None
                axes[-1].legend(fontsize=font_size-2)

            # plot_save_dir is only active when either pyplot or plotly are enabled
            if plot_save_dir and (show_pyplot or plotly):
                Path(plot_save_dir).mkdir(exist_ok=True, parents=True)
                if fig is not None:
                    plt.tight_layout()
                    plt.savefig(Path(plot_save_dir) / f"{key}.svg")
                    plt.savefig(Path(plot_save_dir) / f"{key}.png")
                if pltly_fig is not None:
                    pltly_fig.update_layout(showlegend=True)
                    pltly_fig.write_html(str(Path(plot_save_dir) / f"{key}.html"))

            if show_pyplot:
                fig.show()
            plt.close(fig)

            t2 = time.perf_counter()
            if self.verbose:
                print(f"Post-processing time: {t2 - t1:.3g} seconds")
        if return_mods:
            return errors, solved_mods
        else:
            return errors

    def update_params(
        self, xs: List[Any]
    ) -> Dict[
            str,
            Union[
                LocomotiveSimulation, SetSpeedTrainSim, ConsistSimulation
            ]
    ]:
        """
        Updates model parameters based on `xs`, which must match length of self.params
        """
        assert (len(xs) == len(self.params))

        t0 = time.perf_counter()

        return_model_dict = {}
        if self.model_type == 'LocomotiveSimulation':
            model_cls = LocomotiveSimulation
        elif self.model_type == 'SetSpeedTrainSim':
            model_cls = SetSpeedTrainSim
        elif self.model_type == 'ConsistSimulation':
            model_cls = ConsistSimulation
        else:
            raise AttributeError('cannot initialize models')

        for key, value in self.ser_model_dict.items():
            return_model_dict[key] = model_cls.from_json(value)

        for path, x in zip(self.params, xs):
            for key in return_model_dict.keys():
                return_model_dict[key] = alt.set_param_from_path(
                    return_model_dict[key], path, x
                )
        t1 = time.perf_counter()
        if self.verbose:
            print(f"Time to update params: {t1 - t0:.3g} s")

        return return_model_dict

    def setup_plots(
        self, 
        key: str,
        mod: Any,
        time_seconds: List[float],
        bc: List[float],
        plots_per_key: int,
        pyplot: bool = False,
        plotly: bool = False,
        plot_save_dir: Optional[str] = None,
    ) -> Tuple[Optional[Figure], Optional[plt.Axes], Optional[go.Figure]]:
        """
        # Arguments:

        * plot: ...
        * plotly: make and save plotly plots

        """
        # 1 or 2 axes per objective + 1 axis for boundary condition trace (e.g. PowerTrace)
        
        rows = len(self.objectives) * plots_per_key + 1
        if isinstance(mod, SetSpeedTrainSim):
            bc_label = "Speed [m/s]"
        elif isinstance(mod, LocomotiveSimulation):
            bc_label = "Tractive Power [MW]"
        else:
            raise Exception(f"Need to account for {type(mod)} in this if ... else block.")

        if plotly and (plot_save_dir is not None):
            pltly_fig = make_subplots(
                rows=rows,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
            )
            pltly_fig.update_layout(title=f'trip: {key}')
            pltly_fig.add_trace(
                go.Scatter(
                    x=time_seconds / 3_600,
                    y=bc,
                    name=bc_label + " filtered",
                ),
                row=rows,
                col=1,
            )
            try:
                pltly_fig.add_trace(
                    go.Scatter(
                        x=(time_seconds / 3_600.),
                        y=self.dfs[key]['Locomotive Speed GECX 3000'] * alt.utils.MPS_PER_MPH,
                        name="unfiltered",
                    ),
                    row=rows,
                    col=1,
                )
            except Exception as err:  # noqa: E722
                # print(err)
                pass

            pltly_fig.update_xaxes(title_text="Time [hr]", row=rows, col=1)
            pltly_fig.update_yaxes(title_text=bc_label, row=rows, col=1)
        elif plotly:
            raise Exception(
                "`plot_save_dir` must also be provided for `plotly` to have " 
                + "any effect."
            )
        else:
            pltly_fig = None

        if pyplot:
            fig, axes = plt.subplots(
                rows, 
                1, 
                sharex=True, 
                figsize=(12, 8),
            )
            axes[0].set_title(f'trip: {key}')
            axes[-1].plot(
                time_seconds / 3_600,
                bc,
                label="filtered"
            )

            try:
                axes[-1].plot(
                    (time_seconds / 3_600),
                    self.dfs[key]['Locomotive Speed GECX 3000'] * alt.utils.MPS_PER_MPH,
                    linestyle='-.',
                    label="unfiltered",
                )
            except:  # noqa: E722
                pass

            axes[-1].set_xlabel('Time [hr]')
            axes[-1].set_ylabel(bc_label)
            return fig, axes, pltly_fig
        else:
            return None, None, pltly_fig

    def populate_plots(
        self,
        fig: Optional[Figure],
        axes: plt.Axes,
        pltly_fig: Optional[go.Figure],
        show_pyplot: bool, # whether to show pyplot plots
        plot_save_dir: Path,
        i_obj: int,
        plots_per_key: int,
        key: str,
        time_seconds,
        plot_perc_err: bool,
        mod_sig,
        exp_sig,
        obj: str, # not sure about this type
        font_size: int,
        perc_err_target_for_plot: float,
    ) -> Tuple[Optional[Figure], Optional[plt.Axes], Optional[go.Figure]]:
        if plot_perc_err:
            # error
            perc_err = (mod_sig - exp_sig[:len(mod_sig)]) / exp_sig[:len(mod_sig)] * 100
            # clean up inf and nan
            perc_err[np.where(perc_err == np.inf)[0][:]] = 0.0
            # trim off the first few bits of junk
            perc_err[np.where(perc_err > 500)[0][:]] = 0.0
        
        if fig is not None:
            # model value of objective variable
            axes[i_obj * plots_per_key].plot(
                (time_seconds / 3_600.)[:len(mod_sig)], 
                mod_sig, 
                label='mod',
            )  
            # model value of objective variable
            axes[i_obj * plots_per_key].plot(
                (time_seconds / 3_600)[:len(mod_sig)],
                exp_sig[:len(mod_sig)],
                linestyle='--',
                label="exp",
            )
            axes[i_obj * plots_per_key].set_ylabel(obj[0])
            axes[i_obj * plots_per_key].legend(fontsize=font_size-2)

            if plot_perc_err:
                axes[i_obj * plots_per_key + 1].plot(
                    (time_seconds / 3_600)[:len(mod_sig)],
                    perc_err
                )
                axes[i_obj * plots_per_key].set_ylabel(obj[0])
                axes[i_obj * plots_per_key].legend(fontsize=font_size-2)

            for ax in axes:
                ax.set_xlabel(ax.get_xlabel(), fontsize=font_size)
                ax.set_ylabel(ax.get_ylabel(), fontsize=font_size)
                ax.tick_params(labelsize=font_size - 2)
                ax.set_title(ax.get_title(), fontsize=font_size)

        if pltly_fig is not None:
            pltly_fig.add_trace(
                go.Scatter(
                    x=(time_seconds / 3_600.)[:len(mod_sig)],
                    y=exp_sig[:len(mod_sig)],
                    # might want to prepend signal name for this
                    name=obj[0] + ' exp',
                ),
                # add 1 for 1-based indexing in plotly
                row=i_obj * plots_per_key + 1,
                col=1,
            )
            pltly_fig.add_trace(
                go.Scatter(
                    x=(time_seconds / 3_600.)[:len(mod_sig)],
                    y=mod_sig,
                    # might want to prepend signal name for this
                    name=obj[0] + ' mod',
                ),
                # add 1 for 1-based indexing in plotly
                row=i_obj * plots_per_key + 1,
                col=1,
            )
            pltly_fig.update_yaxes(
                title_text=obj[0], 
                row=i_obj * plots_per_key + 1, 
                col=1
            )

            if plot_perc_err:
                pltly_fig.add_trace(
                    go.Scatter(
                        x=(time_seconds / 3_600.)[:len(mod_sig)],
                        y=perc_err[:len(mod_sig)],
                        # might want to prepend signal name for this
                        name=obj[0] + ' % err',
                    ),
                    # add 2 for 1-based indexing and offset for % err plot
                    row=i_obj * plots_per_key + 2,
                    col=1,
                )            
                pltly_fig.update_yaxes(
                    title_text=obj[0] + "%Err", 
                    row=i_obj * plots_per_key + 2, 
                    col=1
                )

        return fig, axes, pltly_fig

eta_range_err_re_prog = re.compile("`eta_range` .* must be between 0.0")


class CalibrationProblem(ElementwiseProblem):
    """
    Problem for calibrating models to match test data
    """

    def __init__(
        self,
        mod_err: ModelError,
        # parameter lower and upper bounds
        # default of None is needed for dataclass inheritance
        # this is actually mandatory!
        params_bounds: List[Tuple[float, float]],
        n_constr: Optional[int] = 1,
        # see https://pymoo.org/problems/parallelization.html?highlight=runner#Starmap-Interface
        elementwise_runner=LoopedElementwiseEvaluation(),
        func_eval=LoopedElementwiseEvaluation(),
    ):
        self.mod_err = mod_err
        self.params_bounds = params_bounds

        assert (len(self.params_bounds) == len(self.mod_err.params))
        super().__init__(
            n_var=len(self.mod_err.params),
            n_obj=self.mod_err.n_obj,
            n_constr=n_constr,
            xl=[bounds[0]
                for bounds in self.params_bounds],
            xu=[bounds[1]
                for bounds in self.params_bounds],
            elementwise_runner=elementwise_runner,
            func_eval=func_eval,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        model_dict = self.mod_err.update_params(x)
        err = self.mod_err.get_errors(model_dict)
        out['F'] = np.array([
            val for inner_dict in err.values() for val in inner_dict.values()
        ])
        out['G'] = np.array([-1])

class CustomOutput(Output):
    def __init__(self):
        super().__init__()
        self.t_gen_start = time.perf_counter()
        self.n_nds = Column("n_nds", width=8)
        self.t_s = Column("t [s]", width=10)
        self.euclid_min = Column("euclid min", width=13)
        self.columns += [self.n_nds, self.t_s, self.euclid_min]

    def update(self, algorithm):
        super().update(algorithm)
        self.n_nds.set(len(algorithm.opt))
        self.t_s.set(f"{(time.perf_counter() - self.t_gen_start):.3g}")
        f = algorithm.pop.get('F')
        euclid_min = np.sqrt((np.array(f) ** 2).sum(axis=1)).min()
        self.euclid_min.set(f"{euclid_min:.3g}")


def run_minimize(
    problem: CalibrationProblem,
    algorithm: GeneticAlgorithm,
    termination: DMOT,
    save_history: bool = False,
    copy_algorithm: bool = False,
    copy_termination: bool = False,
    save_path: Optional[str] = "pymoo_res",
    pickle_res_to_file: bool = False,
):
    """
    Arguments:
        - save_path: filename for results -- will save `res_df` separately by appending
    """

    alt.utils.print_dt()

    res = minimize(
        problem,
        algorithm,
        termination=termination,
        seed=1,
        verbose=True,
        output=CustomOutput(),
        save_history=save_history,
        copy_algorithm=copy_algorithm,
        copy_termination=copy_termination,
    )

    f_columns = [
        f"{key}: {obj[0]}" for key in problem.mod_err.dfs.keys() for obj in problem.mod_err.objectives
    ]

    f_df = pd.DataFrame(
        data=[f for f in res.F.tolist()],
        columns=f_columns
    )
    x_df = pd.DataFrame(
        data=[x for x in res.X.tolist()],
        columns=[param for param in problem.mod_err.params]
    )

    res_df = pd.concat([x_df, f_df], axis=1)
    res_df.to_csv(path_or_buf=Path(save_path) / 'res_df.csv', index=None)
    if pickle_res_to_file:
        with open(str(save_path)+'.pickle', 'wb') as fout:
            pickle.dump(res, fout)
    return res, res_df

def min_error_selection(
    result_df: pd.DataFrame,
    param_num: int,
    norm_num: int = 2
) -> np.ndarray:
    """
    Arguments:
    ----------
    result_df: pd.DataFrame containing pymoo res.X and res.F concatenated
    param_num: number of parameters
    norm_num: norm number -- e.g. 2 would result in RMS error
    """
    result_df['overall_error'] = (
        result_df.iloc[:, param_num:] ** norm_num).sum(1).pow(float(1.0 / norm_num))
    best_row = result_df['overall_error'].argmin()
    best_df = result_df.iloc[best_row, :]
    param_vals = result_df.iloc[0, :param_num].to_numpy()
    error_vals = result_df.iloc[0, param_num:].to_numpy()

    return param_vals
