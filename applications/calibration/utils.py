"""
Module containing various utilities for calibration
"""

import argparse
import hashlib
import os
import random
import re
from pathlib import Path

import cal_and_val as cval
import numpy as np
import pandas as pd

import altrios

# ignore list and reasons
TRIP_FILE_IGNORE_DICT = {
    "3-24 Bar to Stock - ALTRIOS Condfidential 2.csv": "nans in SOC",
    "1-21 Bar to Stock - ALTRIOS Condfidential 1.csv": "nans in SOC",
    "2-20 Stock to Bar - ALTRIOS Condfidential 2.csv": "nans in SOC",
    "2-20 Stock to Bar - ALTRIOS Condfidential 3.csv": "nans in SOC",
    "2-20 Stock to Bar - ALTRIOS Condfidential 1.csv": "nans in SOC",
    "3-24 Bar to Stock - ALTRIOS Condfidential 1": "blank/nan in GECX speed",
}


def get_ignore_list_re_pattern(ignore_dict: dict[str, str] = TRIP_FILE_IGNORE_DICT) -> str:
    """
    Regex pattern for files in TRIP_FILE_BLACKDICT
    """
    return "(" + ")|(".join(ignore_dict.keys()) + ")"


def get_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-proc", type=int, default=4, help="Number of parallel processes.")
    parser.add_argument("--n-max-gen", type=int, default=500, help="PyMOO n_max_gen")
    parser.add_argument("--pop-size", type=int, default=24, help="PyMOO pop_size")
    parser.add_argument(
        "--repartition",
        action="store_true",
        help="Generate new train/test data partition if passed",
    )

    return parser


def get_trip_data_dir() -> Path:
    """Return trip data directory that is guaranteed to exist."""
    possible_trip_data_dirs: list[Path] = [
        Path(
            altrios.package_root()
            / "../../../data/trips/ZANZEFF Data - v5.1 1-27-23 ALTRIOS Confidential",
        ).resolve(),
        Path(
            altrios.package_root()
            / "../../data/trips/ZANZEFF Data - v5 1-27-23 ALTRIOS Confidential",
        ).resolve(),
        Path(
            altrios.package_root()
            / "../../data/trips/ZANZEFF Data - v4 1-18-23 ALTRIOS Confidential",
        ).resolve(),
        Path(
            altrios.package_root().parents[2]
            / "ZANZEFF Data- Corrected GPS Plus Train Build ALTRIOS Confidential v2",
        ),
        Path(
            # note that this path is stored in ALTRIOS as a symlink and the
            # actual data does not live here
            altrios.package_root().parents[1]
            / "ZANZEFF Data- Corrected GPS Plus Train Build ALTRIOS Confidential v2",
        ),
    ]

    for trip_data_dir in possible_trip_data_dirs:
        trip_data_dir = altrios.package_root() / trip_data_dir
        if trip_data_dir.exists():
            break

    if not trip_data_dir.exists():
        print(
            "No folders found in:\n",
            "\n".join([str(d) for d in possible_trip_data_dirs]),
            sep="",
        )
        raise FileNotFoundError()

    print(f"trip_data_dir: {trip_data_dir.resolve()}")

    return trip_data_dir


def get_fname_re_pattern() -> str:
    """
    Default finds trip date, origin, and destination in
    file names like `2-15 Bar to Stock - ALTRIOS Confidential.csv`
    """
    return r"(\d{1,2}-\d{1,2}) (\w+) to (\w+).*\.csv"


def select_cal_and_val_trips(
    save_path: Path,
    trip_dir: Path | None = get_trip_data_dir(),
    force_rerun: bool | None = False,
    random_seed: int | None = 42,
    cal_frac: float | None = 0.7,
    fname_re_pattern: str = get_fname_re_pattern(),
    ignore_re_pattern: str = get_ignore_list_re_pattern(),
) -> tuple[list[Path], list[Path]]:
    file_info_path = Path(save_path / "FileInfo.csv")
    if file_info_path.exists() and not force_rerun:
        print(
            f"Using calibration and validation data partitioning from:\n{file_info_path.resolve()}",
        )
        return load_previous_files(save_path, trip_dir)

    fname_prog = re.compile(fname_re_pattern)
    ignore_list_prog = re.compile(ignore_re_pattern)

    files = []
    for file in trip_dir.iterdir():
        if not fname_prog.search(file.name):
            continue
        if ignore_list_prog.search(file.name):
            continue
        files.append(file)

    cal_sample_size = int(np.floor(cal_frac * len(files)))
    val_sample_size = len(files) - cal_sample_size

    random.seed(random_seed)

    cal_files = sorted(random.sample(files, k=cal_sample_size))
    val_files = sorted(list(set(files) - set(cal_files)))

    save_new_file_info(save_path, cal_files, val_files)

    return (cal_files, val_files)


def FileMD5(FilePath: Path) -> str:
    # https://stackoverflow.com/questions/16874598/how-do-i-calculate-the-md5-checksum-of-a-file-in-python
    hash = hashlib.md5(open(FilePath, "rb").read()).hexdigest()
    return hash


def save_new_file_info(
    save_path: Path,
    cal_files: list[Path],
    val_files: list[Path],
) -> pd.DataFrame:
    df_cal_files = pd.DataFrame()
    df_cal_files["Filename"] = cal_files
    df_cal_files["File Type"] = "Calibration"
    df_val_files = pd.DataFrame()
    df_val_files["Filename"] = val_files
    df_val_files["File Type"] = "Validation"
    df_file_info = pd.concat([df_cal_files, df_val_files], axis=0)
    df_file_info["MD5"] = df_file_info["Filename"].apply(FileMD5)
    df_file_info["Filename"] = df_file_info["Filename"].apply(os.path.basename)
    df_file_info.to_csv(save_path / "FileInfo.csv")


def load_previous_files(save_path: Path, trip_dir: Path) -> tuple[list[Path], list[Path]]:
    df_file_info = pd.read_csv(save_path / "FileInfo.csv")

    for _, row in df_file_info.iterrows():
        assert FileMD5(trip_dir / row["Filename"]) == row["MD5"], (
            f"new hash: {FileMD5(trip_dir / row['Filename'])} != old hash{row['MD5']}"
        )

    cal_mod_files = [
        trip_dir / row["Filename"]
        for _, row in df_file_info[df_file_info["File Type"] == "Calibration"].iterrows()
    ]
    val_mod_files = [
        trip_dir / row["Filename"]
        for _, row in df_file_info[df_file_info["File Type"] == "Validation"].iterrows()
    ]

    return cal_mod_files, val_mod_files


def cal_val_file_check_post(loco_cal_mod_err, loco_val_mod_err, file_info_df):
    cal_file_list = list(loco_cal_mod_err.dfs.keys())
    val_file_list = list(loco_val_mod_err.dfs.keys())

    cal_file_list = [cal_file + ".csv" for cal_file in cal_file_list]
    val_file_list = [val_file + ".csv" for val_file in val_file_list]

    cal_length_check_bool = file_info_df[file_info_df["File Type"] == "Calibration"].shape[
        0
    ] == len(cal_file_list)
    val_length_check_bool = file_info_df[file_info_df["File Type"] == "Validation"].shape[0] == len(
        val_file_list,
    )

    cal_filename_match_bool = (
        file_info_df.loc[file_info_df["File Type"] == "Calibration", "Filename"]
        .isin(cal_file_list)
        .min()
    )
    val_filename_match_bool = (
        file_info_df.loc[file_info_df["File Type"] == "Validation", "Filename"]
        .isin(val_file_list)
        .min()
    )

    if not all(
        [
            cal_length_check_bool,
            val_length_check_bool,
            cal_filename_match_bool,
            val_filename_match_bool,
        ],
    ):
        raise ValueError(
            "Files being used for calibration do not match files that were previously used!",
        )


def get_results(
    mod_err: cval.ModelError,
    params,
    plotly: bool,
    pyplot: bool,
    plot_save_dir: Path,
) -> tuple[dict, dict]:
    mod_dict = mod_err.update_params(params)
    errs = mod_err.get_errors(
        mod_dict,
        pyplot=pyplot,
        plotly=plotly,
        plot_save_dir=plot_save_dir,
    )

    return mod_dict, errs
