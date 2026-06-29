"""Module for general functions, classes, and unit conversion factors."""

from __future__ import annotations
import numpy as np
from typing import Tuple, Union, Optional, List, Dict
import pandas as pd
import polars as pl
import datetime
import numpy.typing as npt
from pathlib import Path
import os
import shutil

# local imports
import altrios as alt
from altrios import __version__

MPS_PER_MPH = 1.0 / 2.237
N_PER_LB = 4.448
KG_PER_LB = 1.0 / 2.20462
W_PER_HP = 745.7
KG_PER_TON = KG_PER_LB * 2000.0
CM_PER_IN = 2.54
CM_PER_FT = CM_PER_IN * 12.0
M_PER_FT = CM_PER_FT / 100.0
MI_PER_KM = 0.621371
LITER_PER_M3 = 1.0e3
G_PER_TONNE = 1.0e6
GALLONS_PER_LITER = 1.0 / 3.79
KWH_PER_MJ = 0.277778  # https://www.eia.gov/energyexplained/units-and-calculators/energy-conversion-calculators.php
MWH_PER_J = 2.77778e-10
MWH_PER_MJ = KWH_PER_MJ / 1.0e3


def package_root() -> Path:
    """
    Returns the package root directory.
    """
    path = Path(__file__).parent
    return path


def resources_root() -> Path:
    """
    Returns the resources root directory.
    """
    path = package_root() / "resources"
    return path


def print_dt():
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def cumutrapz(x, y):
    """
    Returns cumulative trapezoidal integral array for:
    Arguments:
    ----------
    x: array of monotonically increasing values to integrate over
    y: array of values being integrated
    """
    assert len(x) == len(y)
    z = np.zeros(len(x))
    z[0] = 0
    for i in np.arange(1, len(x)):
        z[i] = z[i - 1] + 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return z


def range_minmax(self) -> pl.Expr:
    return self.max() - self.min()  # type: ignore[no-any-return]


pl.Expr.range = range_minmax  # type: ignore[attr-defined]
del range_minmax


def cumPctWithinGroup(
    df: Union[pl.DataFrame, pl.LazyFrame], grouping_vars: List[str]
) -> Union[pl.DataFrame, pl.LazyFrame]:
    return df.with_columns(
        (
            (pl.int_range(pl.len(), dtype=pl.UInt32).over(grouping_vars).add(1))
            / pl.count().over(grouping_vars)
        ).alias("Percent_Within_Group_Cumulative")
    )


def allocateIntegerEvenly(
    df: Union[pl.DataFrame, pl.LazyFrame], target: str, grouping_vars: List[str]
) -> Union[pl.DataFrame, pl.LazyFrame]:
    return (
        df.sort(grouping_vars)
        .pipe(cumPctWithinGroup, grouping_vars=grouping_vars)
        .with_columns(
            pl.col(target)
            .mul("Percent_Within_Group_Cumulative")
            .round()
            .alias(f"{target}_Group_Cumulative")
        )
        .with_columns(
            (
                pl.col(f"{target}_Group_Cumulative")
                - pl.col(f"{target}_Group_Cumulative").shift(1).over(grouping_vars)
            )
            .fill_null(pl.col(f"{target}_Group_Cumulative"))
            .alias(f"{target}")
        )
        .drop(f"{target}_Group_Cumulative")
    )


def allocateItems(
    df: Union[pl.DataFrame, pl.LazyFrame], grouping_vars: list[str], count_target: str
) -> Union[pl.DataFrame, pl.LazyFrame]:
    return (
        df.sort(grouping_vars + [count_target], descending=True)
        .with_columns(
            pl.col(count_target)
            .sum()
            .over(grouping_vars)
            .round()
            .alias(f"{count_target}_Group"),
            (
                pl.col(count_target).sum().over(grouping_vars).round()
                * (
                    pl.col(count_target).cum_sum().over(grouping_vars)
                    / pl.col(count_target).sum().over(grouping_vars)
                )
            )
            .round()
            .alias(f"{count_target}_Group_Cumulative"),
        )
        .with_columns(
            (
                pl.col(f"{count_target}_Group_Cumulative")
                - pl.col(f"{count_target}_Group_Cumulative")
                .shift(1)
                .over(grouping_vars)
            )
            .fill_null(pl.col(f"{count_target}_Group_Cumulative"))
            .alias("Count")
        )
    )


def resample(
    df: pd.DataFrame,
    dt_new: Optional[float] = 1.0,
    time_col: Optional[str] = "Time[s]",
    rate_vars: Tuple[str] = [],
    hold_vars: Tuple[str] = [],
) -> pd.DataFrame:
    """
    Resamples dataframe `df`.
    Arguments:
    - df: dataframe to resample
    - dt_new: new time step size, default 1.0 s
    - time_col: column for time in s
    - rate_vars: list of variables that represent rates that need to be time averaged
    - hold_vars: vars that need zero-order hold from previous nearest time step
        (e.g. quantized variables like current gear)
    """

    new_dict = dict()

    new_time = np.arange(
        0, np.floor(df[time_col].to_numpy()[-1] / dt_new) * dt_new + dt_new, dt_new
    )

    for col in df.columns:
        if col in rate_vars:
            # calculate average value over time step
            cumu_vals = (df[time_col].diff().fillna(0) * df[col]).cum_sum()
            new_dict[col] = (
                np.diff(
                    np.interp(x=new_time, xp=df[time_col].to_numpy(), fp=cumu_vals),
                    prepend=0,
                )
                / dt_new
            )

        elif col in hold_vars:
            assert col not in rate_vars
            pass  # this may need to be fleshed out

        else:
            # just interpolate -- i.e. state variables like temperatures
            new_dict[col] = np.interp(
                x=new_time, xp=df[time_col].to_numpy(), fp=df[col].to_numpy()
            )

    return pd.DataFrame(new_dict)


def smoothen(signal: npt.ArrayLike, period: int = 9) -> npt.ArrayLike:
    """
    Apply smoothing to signal, assuming 1 Hz data collection.
    """
    new_signal = np.convolve(
        np.concatenate(
            [
                np.full(((period + 1) // 2) - 1, signal[0]),
                signal,
                np.full(period // 2, signal[-1]),
            ]
        ),
        np.ones(period) / period,
        mode="valid",
    )
    return new_signal


def print_dt():
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def set_network_max_speed(network_dict: dict, max_speed_mph: float):
    max_speed_meters_per_second = max_speed_mph * 1000 / 3600 / alt.utilities.MI_PER_KM
    for link_idx in range(len(network_dict[1])):
        if network_dict[1][link_idx]['speed_set'] is None: 
            continue
        if network_dict[1][link_idx]['speed_set']['speed_limits'] is None: 
            continue
        for speed_limit_idx in range(len(network_dict[1][link_idx]['speed_set']['speed_limits'])):
            current_limit = network_dict[1][link_idx]['speed_set']['speed_limits'][speed_limit_idx]['speed_meters_per_second']
            network_dict[1][link_idx]['speed_set']['speed_limits'][speed_limit_idx]['speed_meters_per_second'] = min(current_limit, max_speed_meters_per_second)

    return network_dict

def scale_network_speed_limits(network_dict: dict, scaler: float):
    for link_idx in range(len(network_dict[1])):
        if network_dict[1][link_idx]['speed_set'] is None: 
            continue
        if network_dict[1][link_idx]['speed_set']['speed_limits'] is None: 
            continue
        for speed_limit_idx in range(len(network_dict[1][link_idx]['speed_set']['speed_limits'])):
            current_limit = network_dict[1][link_idx]['speed_set']['speed_limits'][speed_limit_idx]['speed_meters_per_second']
            network_dict[1][link_idx]['speed_set']['speed_limits'][speed_limit_idx]['speed_meters_per_second'] = current_limit * scaler

    return network_dict

def increase_link_speed_limits(network_dict: dict, link_indices: list[int], new_speed_limit_meters_per_second: float, alt_value: float, overrides: dict = None):
    updated_network_dict = network_dict.copy()
    non_max_vals = {}
    for link_idx in range(len(network_dict[1])):
        if network_dict[1][link_idx]['idx_curr'] in link_indices: 
            val_to_assign = new_speed_limit_meters_per_second
        elif (overrides is not None) and (link_idx in overrides):
            val_to_assign = overrides[link_idx]
        else: 
            val_to_assign = alt_value
        if network_dict[1][link_idx]['speed_set'] is None: continue
        if network_dict[1][link_idx]['speed_set']['speed_limits'] is None: continue
        for speed_limit_idx in range(len(network_dict[1][link_idx]['speed_set']['speed_limits'])):
            if updated_network_dict[1][link_idx]['speed_set']['speed_limits'][speed_limit_idx]['speed_meters_per_second'] is None: continue
            updated_network_dict[1][link_idx]['speed_set']['speed_limits'][speed_limit_idx]['speed_meters_per_second'] = val_to_assign
        
        if val_to_assign != alt_value:
            non_max_vals[link_idx] = val_to_assign
    return updated_network_dict, non_max_vals

def add_catenary_uniform(
    network: alt.Network,
    uniform_power_limit_watts: float,
) -> alt.Network:
    network_dict = network.to_pydict()
    # Iterate through all links in the dictionary
    # network_dict is a tuple: (NetworkErrTol dict, [links])
    # Skip index 0 which is the dummy link
    for i in range(1, len(network_dict[1])):
        link = network_dict[1][i]
        # Add catenary power limit spanning the entire link
        link['cat_power_limits'] = [{
            'offset_start_meters': 0.0,
            'offset_end_meters': link['length_meters'],
            'power_limit_watts': uniform_power_limit_watts,
            'district_id': None
        }]

    return alt.Network.from_pydict(network_dict)

def add_catenary_high_grade(
    network: alt.Network,
    uniform_power_limit_watts: float,
    grade_threshold: float
) -> alt.Network:
    network_dict = network.to_pydict()
    for i in range(1, len(network_dict[1])):
        link = network_dict[1][i]
        if len(link['elevs']) >= 2:
            max_grade = -float('inf')
            for j in range(len(link['elevs']) - 1):
                grade = (link["elevs"][j + 1]['elev_meters'] - link["elevs"][j]['elev_meters']) / (link["elevs"][j + 1]['offset_meters'] - link["elevs"][j]['offset_meters'])
                max_grade = max(max_grade, grade)
            if max_grade >= grade_threshold:
                link['cat_power_limits'] = [{
                    'offset_start_meters': 0.0,
                    'offset_end_meters': link['length_meters'],
                    'power_limit_watts': uniform_power_limit_watts,
                    'district_id': None
                }]

    return alt.Network.from_pydict(network_dict)

def _compute_cat_intervals(
    start_pos_in_cycle: float,
    link_length: float,
    gap_meters: float,
    cat_meters: float,
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Return sub-link intervals that should have catenary power and the
    ending cycle position.

    Walks from `start_pos_in_cycle` through the repeating gap/catenary pattern
    (gap_meters of no catenary, then cat_meters of catenary) across `link_length`
    meters, yielding precise (offset_start, offset_end) pairs relative to the
    start of the link.

    Returns (intervals, ending_pos_in_cycle) so the caller can chain links
    without floating-point drift.
    """
    cycle_length = gap_meters + cat_meters
    intervals: List[Tuple[float, float]] = []
    pos_in_cycle = start_pos_in_cycle % cycle_length
    link_offset = 0.0
    while link_offset < link_length - 1e-9:
        if pos_in_cycle < gap_meters:
            advance = min(gap_meters - pos_in_cycle, link_length - link_offset)
            link_offset += advance
            pos_in_cycle += advance
            if pos_in_cycle >= gap_meters - 1e-9:
                pos_in_cycle = gap_meters
        else:
            advance = min(cycle_length - pos_in_cycle, link_length - link_offset)
            if advance > 1e-9:
                intervals.append((link_offset, link_offset + advance))
            link_offset += advance
            pos_in_cycle += advance
            if pos_in_cycle >= cycle_length - 1e-9:
                pos_in_cycle = 0.0
    return intervals, pos_in_cycle


def add_catenary_even_spacing(
    network: alt.Network,
    uniform_power_limit_watts: float,
    spacing_meters: Tuple[float, float],
    est_time_nets: List[alt.EstTimeNet],
) -> alt.Network:
    """
    Place catenary with precise sub-link offsets.

    spacing_meters[0] is the gap (no catenary), spacing_meters[1] is the
    energized segment.  The mainline path is determined by following idx_next
    pointers through the network, guided by the est_time_net path at junctions.
    Only mainline links advance the spacing cycle; parallel-track and siding
    links inherit the catenary state from the current cycle position without
    advancing it.
    """
    gap_meters, cat_meters = spacing_meters

    network_dict = network.to_pydict()
    # Build lookup: link idx -> link dict (index 0 is the dummy link, skip it)
    link_by_idx: Dict[int, dict] = {
        link['idx_curr']: link for link in network_dict[1]
    }

    # Extract ordered link path from the first est_time_net
    est_time_net = est_time_nets[0]
    ordered_link_idxs: List[int] = (
        pl.DataFrame(est_time_net.to_pydict()["val"])
        .lazy()
        .unnest("link_event")
        .with_columns(
            pl.col("est_type").replace({0: "Arrive", 1: "Clear", 2: "Fake"}).cast(pl.Categorical),
            pl.selectors.contains("idx").replace({0: None}).cast(pl.UInt32)
        )
        .filter(pl.col("est_type") == pl.lit("Clear"))
        .select("link_idx", "time_sched")
        .sort("time_sched")
        .select("link_idx")
        .collect()
        .get_column("link_idx")
        .drop_nulls()
        .to_list()
    )
    # Collapse consecutive runs of the same link
    deduped_link_idxs: List[int] = []
    for idx in ordered_link_idxs:
        if not deduped_link_idxs or deduped_link_idxs[-1] != idx:
            deduped_link_idxs.append(idx)

    # Build mainline link set by following idx_next from the first link
    # AND idx_prev from the last link (bidirectional).  At each step, prefer
    # the pointer that appears in the est_time_net path.  The union of both
    # chains covers junctions where one direction picks the wrong branch.
    path_set = set(deduped_link_idxs)

    # Forward chain: follow idx_next from the first path link
    mainline_fwd: set = set()
    current = deduped_link_idxs[0]
    while current and current != 0 and current not in mainline_fwd:
        mainline_fwd.add(current)
        link = link_by_idx.get(current)
        if link is None:
            break
        idx_next = link.get('idx_next', 0)
        idx_next_alt = link.get('idx_next_alt', 0)
        if idx_next in path_set:
            current = idx_next
        elif idx_next_alt in path_set:
            current = idx_next_alt
        else:
            current = idx_next

    # Backward chain: follow idx_prev from the last path link
    mainline_bwd: set = set()
    current = deduped_link_idxs[-1]
    while current and current != 0 and current not in mainline_bwd:
        mainline_bwd.add(current)
        link = link_by_idx.get(current)
        if link is None:
            break
        idx_prev = link.get('idx_prev', 0)
        idx_prev_alt = link.get('idx_prev_alt', 0)
        if idx_prev in path_set:
            current = idx_prev
        elif idx_prev_alt in path_set:
            current = idx_prev_alt
        else:
            current = idx_prev

    mainline_set = mainline_fwd | mainline_bwd

    # Walk the est_time_net path:
    #  - Mainline links (first visit): compute intervals, advance pos_in_cycle
    #  - Non-mainline / revisited: advance a *local* cycle tracker so
    #    consecutive non-mainline blocks get smooth catenary, but do NOT
    #    advance the mainline pos_in_cycle.
    link_cat_intervals: Dict[int, List[Tuple[float, float]]] = {}
    seen_links: set = set()
    pos_in_cycle = 0.0
    local_pos = 0.0  # tracks cycle position within non-mainline blocks
    for link_idx in deduped_link_idxs:
        if link_idx not in link_by_idx:
            continue
        link_length = link_by_idx[link_idx]['length_meters']
        is_mainline = link_idx in mainline_set and link_idx not in seen_links
        if is_mainline:
            intervals, pos_in_cycle = _compute_cat_intervals(
                pos_in_cycle, link_length, gap_meters, cat_meters)
            seen_links.add(link_idx)
            local_pos = pos_in_cycle  # reset local tracker to mainline state
            if intervals:
                link_cat_intervals[link_idx] = intervals
        elif link_idx not in link_cat_intervals:
            intervals, local_pos = _compute_cat_intervals(
                local_pos, link_length, gap_meters, cat_meters)
            if intervals:
                link_cat_intervals[link_idx] = intervals

    # Apply catenary intervals to network links
    for i in range(1, len(network_dict[1])):
        link = network_dict[1][i]
        link_idx = link['idx_curr']
        if link_idx not in link_cat_intervals:
            continue
        link['cat_power_limits'] = [
            {
                'offset_start_meters': start,
                'offset_end_meters': end,
                'power_limit_watts': uniform_power_limit_watts,
                'district_id': None,
            }
            for start, end in link_cat_intervals[link_idx]
        ]

    return alt.Network.from_pydict(network_dict)

def copy_demo_files(demo_path: Path = Path("demos")):
    """
    Copies demo files from package directory into local directory.

    # Arguments
    demo_path: path (relative or absolute in )

    # Warning
    Running this function will overwrite existing files so make sure any files with
    changes you'd like to keep are renamed.
    """

    v = f"v{__version__}"
    demo_path.mkdir(exist_ok=True)

    for src_file in (package_root() / "demos").iterdir():
        if src_file.suffix != ".py":
            continue
        src_file: Path
        dest_file = demo_path / src_file.name
        shutil.copyfile(src_file, dest_file)

        with open(dest_file, "r+") as file:
            file_content = file.readlines()
            prepend_str = f"# %% Copied from ALTRIOS version '{v}'. Guaranteed compatibility with this version only.\n"
            prepend = [prepend_str]
            file_content = prepend + file_content
            file.seek(0)
            file.writelines(file_content)

    print(f"Saved {dest_file.name} to {dest_file}")


def show_plots() -> bool:
    """
    Returns true if plots should be displayed based on `SHOW_PLOTS` environment variable.
    `SHOW_PLOTS` defaults to true, and to set it false, run `SHOW_PLOTS=false python your_script.py`
    """
    return (
        os.environ.get(
            # name of environment variable
            "SHOW_PLOTS",
            # defaults to true if not provided
            "true",
            # only true if provided input is exactly "true", case insensitive
        ).lower()
        == "true"
    )

def network_to_latlon(network: alt.Network) -> pl.DataFrame:
    return(
        pl.DataFrame(network.to_pydict()[1]).lazy()
            .explode("headings")
            .with_columns(
                pl.col("headings").struct.field("lat").alias("lat"), 
                pl.col("headings").struct.field("lon").alias("lon")
            )
            .drop("cat_power_limits", "link_idxs_lockout", "speed_set", "speed_sets", "elevs", "length_meters", "headings")
    )