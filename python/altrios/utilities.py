"""Module for general functions, classes, and unit conversion factors."""

from __future__ import annotations
import numpy as np
from typing import Tuple, Union, Optional, List
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
MI_PER_M = MI_PER_KM * 1.0e-3
LITER_PER_M3 = 1.0e3
G_PER_TONNE = 1.0e6
GALLONS_PER_LITER = 1.0 / 3.79
KWH_PER_MJ = 0.277778  # https://www.eia.gov/energyexplained/units-and-calculators/energy-conversion-calculators.php
MWH_PER_J = 2.77778e-10
MWH_PER_MJ = KWH_PER_MJ / 1.0e3

# GDP chain-type price index deflator to restate 2022 USD in 2025 USD (the AEO2026
# base year). = AEO2026 Table 20 GDP chain-price index 2025 (1.383637, 2012=1.000)
# / 2022 actual (1.27224).
USD_2022_TO_2025 = 1.088
# CPI-U annual averages (BLS, 1982-84=100): 2025 321.943 / 2024 313.689. The NREL 2025
# battery cost projections are in 2024 USD; NREL itself converts dollar years with CPI.
USD_2024_TO_2025 = 1.026


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


def cumPctWithinGroup(
    df: Union[pl.DataFrame, pl.LazyFrame], grouping_vars: List[str]
) -> Union[pl.DataFrame, pl.LazyFrame]:
    return df.with_columns(
        (
            (pl.int_range(pl.len(), dtype=pl.UInt32).over(grouping_vars).add(1))
            / pl.len().over(grouping_vars)
        ).alias("Percent_Within_Group_Cumulative")
    )


def allocateIntegerEvenly(
    df: Union[pl.DataFrame, pl.LazyFrame], target: str, grouping_vars: List[str]
) -> Union[pl.DataFrame, pl.LazyFrame]:
    return (
        # Stable sort: the rounding remainder below goes to whichever rows come
        # first within a group, so tie order must be the input order, not the
        # engine's.
        df.sort(grouping_vars, maintain_order=True)
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
        df.sort(grouping_vars + [count_target], descending=True, maintain_order=True)
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

def set_network_max_speed(network_dict: list, max_speed_mph: float) -> list:
    """Cap every link speed limit in a network pydict.

    Args:
        network_dict: Network pydict from `Network.to_pydict()`; element 1 holds
            the links, each with `speed_set.speed_limits`.
        max_speed_mph: Speed cap (mph).

    Returns:
        The same pydict, modified in place.
    """
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

def scale_network_speed_limits(network_dict: list, scaler: float) -> list:
    """Multiply every link speed limit in a network pydict.

    Args:
        network_dict: Network pydict from `Network.to_pydict()`; element 1 holds
            the links, each with `speed_set.speed_limits`.
        scaler: Factor applied to each `speed_meters_per_second`.

    Returns:
        The same pydict, modified in place.
    """
    for link_idx in range(len(network_dict[1])):
        if network_dict[1][link_idx]['speed_set'] is None: 
            continue
        if network_dict[1][link_idx]['speed_set']['speed_limits'] is None: 
            continue
        for speed_limit_idx in range(len(network_dict[1][link_idx]['speed_set']['speed_limits'])):
            current_limit = network_dict[1][link_idx]['speed_set']['speed_limits'][speed_limit_idx]['speed_meters_per_second']
            network_dict[1][link_idx]['speed_set']['speed_limits'][speed_limit_idx]['speed_meters_per_second'] = current_limit * scaler

    return network_dict

def increase_link_speed_limits(
    network_dict: list,
    link_indices: list[int],
    new_speed_limit_meters_per_second: float,
    alt_value: float,
    overrides: dict[int, float] | None = None,
) -> tuple[list, dict[int, float]]:
    """Assign per-link speed limits in a network pydict.

    Args:
        network_dict: Network pydict from `Network.to_pydict()`; element 1 holds
            the links, each with `speed_set.speed_limits`.
        link_indices: `idx_curr` values of the links set to the new limit.
        new_speed_limit_meters_per_second: Limit (m/s) for `link_indices`.
        alt_value: Limit (m/s) for every other link without an override.
        overrides: Link list position to limit (m/s) for other specific links.

    Returns:
        The updated pydict, and link list position to assigned limit for every
        link whose limit differs from `alt_value`.
    """
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

def convert_network(
    src: Union[str, Path],
    dest: Optional[Union[str, Path]] = None,
    dest_format: str = "msgpack",
    overwrite: bool = False,
    verify: bool = True,
) -> Path:
    """Convert a rail network file to another format, losslessly.

    `msgpack` suits large generated networks; `yaml` suits small hand-edited ones.

    Args:
        src: Existing network file; the format is inferred from its extension.
        dest: Path to write; defaults to `src` with the suffix of `dest_format`.
        dest_format: Format `Network.to_file` accepts, e.g. "msgpack" or "yaml";
            "json" perturbs floats by up to 1 ULP on reading.
        overwrite: Whether to replace an existing `dest`.
        verify: Whether to read `dest` back and check it matches `src`.

    Returns:
        The path written. `src` is left untouched.

    Raises:
        ValueError: When `dest` resolves to `src`.
        FileExistsError: When `dest` exists and `overwrite` is False.
        RuntimeError: When `verify` is True and the round trip differs.
    """
    src = Path(src)
    if dest is None:
        dest = src.with_suffix(f".{dest_format}")
    dest = Path(dest)

    if dest.resolve() == src.resolve():
        raise ValueError(f"`dest` must differ from `src`: {src}")
    if dest.exists() and not overwrite:
        raise FileExistsError(f"{dest} already exists; pass `overwrite=True` to replace it")

    # `skip_init=True` avoids re-running validation that the source network already passed;
    # this is a format conversion, not an import of untrusted data.
    network = alt.Network.from_file(str(src), skip_init=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    network.to_file(str(dest))

    if verify:
        roundtripped = alt.Network.from_file(str(dest), skip_init=True)
        # Lines are sorted before comparing because `speed_sets` is a hash map, whose serialized
        # key order is not stable from one load to the next.
        if sorted(roundtripped.to_yaml().splitlines()) != sorted(network.to_yaml().splitlines()):
            raise RuntimeError(
                f"round trip through {dest_format!r} did not preserve the network: {src} -> {dest}"
            )

    return dest

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

def network_to_latlon(network: alt.Network) -> pl.LazyFrame:
    """Explode a network's link headings into lat/lon rows.

    Args:
        network: Network whose link `headings` carry `lat` and `lon`.

    Returns:
        LazyFrame with one row per heading point: the link's `idx_*` columns,
        `osm_id`, `lat` and `lon`.
    """
    return(
        pl.DataFrame(network.to_pydict()[1]).lazy()
            .explode("headings")
            .with_columns(
                pl.col("headings").struct.field("lat").alias("lat"), 
                pl.col("headings").struct.field("lon").alias("lon")
            )
            .drop("cat_power_limits", "link_idxs_lockout", "speed_set", "speed_sets", "elevs", "length_meters", "headings")
    )