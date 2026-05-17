"""Module for general functions, classes, and unit conversion factors."""
from pathlib import Path
import polars as pl
import numpy as np
import numba as nb
import json
import yaml


@nb.njit(cache=True)
def _greedy_match_arrivals_to_departures(
    arr_id: np.ndarray,
    dep_id: np.ndarray,
    arr_time: np.ndarray,
    dep_time: np.ndarray,
    min_processing_time_hours: float,
):
    """Greedy 1-to-1 matching of arriving trains to departing trains.

    Iterates departing trains in order of their earliest possible departure_time
    and, for each, claims the earliest still-available arriving train whose
    arrival_time is at least ``min_processing_time_hours`` before the departure.

    After greedy matching, any arrival or departure id that was never matched
    gets exactly one fallback row whose "other side" id is flagged as bogus —
    the caller is expected to null it out. This avoids returning the same
    ``train_id`` (or ``train_id_departure``) on two different rows.

    Returns three bool arrays, all of length ``n`` (one entry per input row):

    * ``keep`` — rows to retain in the output.
    * ``arrival_needs_fixup`` — subset of ``keep``: row exists only to carry
      an orphan **departure**; the arrival side (``train_id``) on this row is
      bogus and should be nulled.
    * ``departure_needs_fixup`` — subset of ``keep``: row exists only to carry
      an orphan **arrival**; the departure side (``train_id_departure``) on
      this row is bogus and should be nulled.

    A "real" match has ``keep=True`` and both fixup flags ``False``.

    ``arr_id`` and ``dep_id`` must be non-negative integers; they are used to
    index per-id bool flags sized to ``max(id) + 1``.
    """
    n = arr_id.shape[0]
    keep = np.zeros(n, dtype=np.bool_)
    arrival_needs_fixup = np.zeros(n, dtype=np.bool_)
    departure_needs_fixup = np.zeros(n, dtype=np.bool_)

    # Size the per-id "used" flags by the largest id observed in each column.
    max_arr_id = 0
    max_dep_id = 0
    for i in range(n):
        if arr_id[i] > max_arr_id:
            max_arr_id = arr_id[i]
        if dep_id[i] > max_dep_id:
            max_dep_id = dep_id[i]
    arr_used = np.zeros(max_arr_id + 1, dtype=np.bool_)
    dep_used = np.zeros(max_dep_id + 1, dtype=np.bool_)

    # Earliest departure_time observed per departing train (for iteration order).
    # Unused slots stay at +inf and are skipped below.
    dep_min_time = np.full(max_dep_id + 1, np.inf)
    for i in range(n):
        d = dep_id[i]
        if dep_time[i] < dep_min_time[d]:
            dep_min_time[d] = dep_time[i]
    dep_order = np.argsort(dep_min_time)

    for k in range(dep_order.shape[0]):
        di = dep_order[k]
        if dep_min_time[di] == np.inf:
            continue  # id value never appeared in dep_id
        if dep_used[di]:
            continue
        best_row = -1
        best_arr_t = np.inf
        for i in range(n):
            if dep_id[i] != di:
                continue
            ai = arr_id[i]
            if arr_used[ai]:
                continue
            if arr_time[i] > dep_time[i] - min_processing_time_hours:
                continue
            if arr_time[i] < best_arr_t:
                best_arr_t = arr_time[i]
                best_row = i
        if best_row >= 0:
            keep[best_row] = True
            arr_used[arr_id[best_row]] = True
            dep_used[di] = True

    # Fallback pass 1: any arrival id that wasn't matched gets one row. The
    # departure side on that row is bogus, so we mark only ``arr_used``.
    for i in range(n):
        if keep[i]:
            continue
        ai = arr_id[i]
        if arr_used[ai]:
            continue
        keep[i] = True
        departure_needs_fixup[i] = True
        arr_used[ai] = True

    # Fallback pass 2: any departure id that's still uncovered gets one row.
    # The arrival side on that row is bogus.
    for i in range(n):
        if keep[i]:
            continue
        di = dep_id[i]
        if dep_used[di]:
            continue
        keep[i] = True
        arrival_needs_fixup[i] = True
        dep_used[di] = True

    return keep, arrival_needs_fixup, departure_needs_fixup

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

CONFIG_PATH = resources_root() / 'sim_config.json'

def load_config(config_path: Path = CONFIG_PATH):
    suffix = Path(config_path).suffix.lower()
    try:
        with open(config_path, 'r') as f:
            if suffix == '.yaml' or suffix == '.yml':
                return yaml.safe_load(f)
            elif suffix == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: '{suffix}'. Expected .yaml, .yml, or .json.")
    except FileNotFoundError:
        raise FileNotFoundError(f"[Error] Config file not found at: {config_path}")


def build_train_timetable(train_consist_plan, terminal_name, as_dicts, track_count=1, min_processing_time_hours = 5.0):
    if track_count > 1:
        print("More than one track not yet supported!")

    # Fall back to Cars_* counts (1 container per car) when explicit Containers_* columns
    # are absent. TODO: When LIFTS handles double-stacked containers, this should be updated.
    available_cols = set(train_consist_plan.columns)
    if "Containers_Empty" not in available_cols:
        train_consist_plan = train_consist_plan.with_columns(
            pl.col("Cars_Empty").alias("Containers_Empty")
        )
    if "Containers_Loaded" not in available_cols:
        train_consist_plan = train_consist_plan.with_columns(
            pl.col("Cars_Loaded").alias("Containers_Loaded")
        )

    this_terminal_trains = (train_consist_plan
        .filter(
            pl.col("Train_Type").str.starts_with(pl.lit("Intermodal"))
        )
        .select(
            pl.col("Origin_ID", "Destination_ID"),
            pl.col("Train_ID").alias("train_id"),
            pl.col("Departure_Time_Actual_Hr").alias("departure_time"),
            pl.col("Arrival_Time_Actual_Hr").alias("arrival_time"),
            pl.col("Cars_Empty").alias("empty_cars"),
            pl.col("Cars_Loaded").alias("full_cars"),
            pl.col("Containers_Empty").alias("empty_containers"),
            pl.col("Containers_Loaded").alias("full_containers")
        )
        .unique()
    )

    if this_terminal_trains.height == 0:
        raise ValueError(
            f"No Intermodal trains found in train_consist_plan at terminal '{terminal_name}'; "
            "build_train_timetable requires at least one row with Train_Type starting with 'Intermodal'."
        ) 
    arrivals = (this_terminal_trains
        .filter(pl.col("Destination_ID") == pl.lit(terminal_name))
        # TODO: When LIFTS handles double-stacked containers, this should be updated accordingly.
        .select(pl.col("train_id", "arrival_time", "empty_cars"), pl.col("full_containers").alias("full_cars"))
        .sort("arrival_time")
    )
    departures = (this_terminal_trains
        .filter(pl.col("Origin_ID") == pl.lit(terminal_name))
        # TODO: When LIFTS handles double-stacked containers, this should be updated accordingly.
        .select(pl.col("train_id", "departure_time"), pl.col("full_cars").alias("oc_number"))
        .sort("departure_time")
    )

    df = (arrivals
        .join_where(departures,
            pl.col("arrival_time") + min_processing_time_hours <= pl.col("departure_time"),
            suffix="_departure"
        )
        .sort("arrival_time", "departure_time")
    )

    # Greedy 1-to-1 matching: each departing train claims the earliest still-
    # available arriving train that satisfies the min_processing_time_hours gap.
    # Orphan rows (one side has no feasible counterpart) are kept with the
    # bogus side nulled out.
    keep, arrival_needs_fixup, departure_needs_fixup = _greedy_match_arrivals_to_departures(
        df["train_id"].to_numpy(),
        df["train_id_departure"].to_numpy(),
        df["arrival_time"].to_numpy(),
        df["departure_time"].to_numpy(),
        min_processing_time_hours,
    )
    arr_fixup_cols = ["train_id", "arrival_time", "full_cars", "empty_cars"]
    dep_fixup_cols = ["train_id_departure", "departure_time", "oc_number"]
    df = (df
        .with_columns(
            pl.Series("_keep", keep),
            pl.Series("_arr_fixup", arrival_needs_fixup),
            pl.Series("_dep_fixup", departure_needs_fixup),
        )
        .filter(pl.col("_keep"))
        .with_columns(
            *[pl.when("_arr_fixup").then(None).otherwise(c).alias(c) for c in arr_fixup_cols],
            *[pl.when("_dep_fixup").then(None).otherwise(c).alias(c) for c in dep_fixup_cols],
        )
        .drop("_keep", "_arr_fixup", "_dep_fixup")
    )
    
    arrivals_to_add = arrivals.join(df, how="anti", on="train_id")
    departures_to_add = departures.join(df, how="anti", left_on="train_id", right_on="train_id_departure")
    to_add = arrivals_to_add.join(departures_to_add, how="full", on="train_id").rename({"train_id_right": "train_id_departure"})
    df = (pl.concat([df, to_add], how="vertical")
        .with_columns(
            pl.col("arrival_time").fill_null(pl.col("departure_time").sub(min_processing_time_hours)),
            pl.col("departure_time").fill_null(pl.col("arrival_time").add(min_processing_time_hours)),
            pl.col("empty_cars").fill_null(pl.col("empty_cars").mean().round()).cast(pl.UInt32),
            pl.col("full_cars").fill_null(pl.col("full_cars").mean().round()).cast(pl.UInt32),
            pl.col("oc_number").fill_null(pl.col("oc_number").mean().round()).cast(pl.UInt32),
        )
        .sort("arrival_time", "departure_time")
        .with_columns(
            pl.max_horizontal("full_cars", "oc_number").alias("truck_number"),
            # Fill null train_id / train_id_departure with unique synthetic integers
            # starting just above the current max id across both columns, so they can't
            # collide with real ids.
            pl.when(pl.col("train_id").is_null())
                .then(
                    pl.max_horizontal("train_id", "train_id_departure").max()
                    + pl.col("train_id").is_null().cum_sum()
                )
                .otherwise("train_id")
                .cast(pl.Int64)
                .alias("train_id")
        )
        .with_columns(
            # train_id_departure's base is recomputed after
            # train_id is filled so it sees the freshly-assigned synthetic values too.
            pl.when(pl.col("train_id_departure").is_null())
                .then(
                    pl.max_horizontal("train_id", "train_id_departure").max()
                    + pl.col("train_id_departure").is_null().cum_sum()
                )
                .otherwise("train_id_departure")
                .cast(pl.Int64)
                .alias("train_id_departure")
        )
    )

    if as_dicts:
        return df.to_dicts()
    else:
        return df


def record_container_event(terminal, container, event_type, timestamp):
    if type(container) is str:
        container_string = container
    else:
        container_string = container.to_string()

    if container_string not in terminal.state.container_events:
        terminal.state.container_events[container_string] = {}
    terminal.state.container_events[container_string][event_type] = timestamp


def emission_calculation(terminal, status: str, move: str, vehicle: str, energy_type: str, travel_time: float) -> float:
    ems = terminal.ems

    move = move.lower()
    status = status.lower()
    vehicle = vehicle.lower()
    energy_type = energy_type.capitalize()

    # --- load consumption (unit: per lift)
    if move == "load":
        key = "crane_loaded" if status == "loaded" else "crane_idle"
        emission_unit = ems["load_consumption"][key][energy_type]
        emissions = emission_unit

    # --- trip consumption (unit: hr × travel_time)
    elif move == "trip":
        key = f"{vehicle}_{status}"  # e.g. hostler_loaded / truck_empty
        emission_unit = ems["trip_consumption"][key][energy_type]
        emissions = emission_unit * travel_time

    # --- side pick consumption (unit: per lift)
    elif move == "side":
        emission_unit = ems["side_pick_consumption"]["side"][energy_type]
        emissions = emission_unit

    else:
        raise ValueError(f"Unsupported move type '{move}' for vehicle '{vehicle}'.")

    return emissions


def record_emission(emission_records: list, vehicle_type: str, resource_id: str, track_id: str, train_id: str, container_id: str, event_type: str, zone: str, emission_value: float, travel_time: float, env_now: float) -> None:
    emission_records.append({
        "resource_type": vehicle_type.lower(),
        "resource_id": str(resource_id),
        "track_id":str(track_id),
        "train_id": str(train_id),
        "container_id": str(container_id),
        "event_type": event_type,
        "zone": zone,
        "energy_consumption(gal)": float(emission_value),
        "load/travel_time(hr)": float(travel_time),
        "record_timestamp": float(env_now),
    })


def save_emission_results(emission_records: pl.DataFrame, out_path: Path, filetype: str = "csv"):
    if out_path is None:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if filetype == "csv":
        emission_records.write_csv(out_path)
    elif filetype == "xlsx":
        emission_records.to_pandas().to_excel(out_path, index=False)
    else:
        raise ValueError("filetype must be 'csv' or 'xlsx'")

def initialize_train_events(env, terminal, train_id):
    state = terminal.state
    for name in [
        "train_ic_unload_events",
        "train_oc_prepared_events",
        "train_ic_picked_events",
        "train_start_load_events",
        "train_end_load_events",
        "train_departed_events",
    ]:
        if not hasattr(state, name):
            setattr(state, name, {})
        d = getattr(state, name)

        if train_id not in d or d[train_id].triggered:
            d[train_id] = env.event()
