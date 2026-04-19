"""Module for general functions, classes, and unit conversion factors."""
from pathlib import Path
import polars as pl
import json
import yaml

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


def build_train_timetable(train_consist_plan, terminal_name, as_dicts, track_count=1, start_end_padding_hours = 12.0):
    if track_count > 1:
        print("More than one track not yet supported!")

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

    first_departure_needs_fixing = [
        pl.col("departure_time") == pl.col("departure_time").min(),
        pl.col("arrival_time") == pl.col("arrival_time").first(),
        pl.col("departure_time") <= pl.col("arrival_time")
    ]
    last_arrival_needs_fixing = [        
        pl.col("arrival_time") == pl.col("arrival_time").max(),
        pl.col("departure_time") == pl.col("departure_time").last(),
        pl.col("departure_time") <= pl.col("arrival_time")
    ]

    df = (arrivals
        .join_where(departures, 
            (pl.col("arrival_time") < pl.col("departure_time")) | 
            (pl.col("departure_time") < pl.col("arrival_time").min()) |
            (pl.col("arrival_time") < pl.col("departure_time").max()),
            suffix="_departure"
        )
        .sort("arrival_time", "departure_time")
        .with_columns(
            pl.when(first_departure_needs_fixing)
                .then(pl.col("departure_time") - start_end_padding_hours)
                .otherwise("arrival_time")
                .alias("arrival_time"),
            pl.when(first_departure_needs_fixing)
                .then("train_id_departure")
                .otherwise("train_id")
                .alias("train_id"),
            pl.when(first_departure_needs_fixing)
                .then(pl.col("full_cars").median())
                .otherwise("full_cars")
                .round().cast(pl.UInt32)
                .alias("full_cars"),
            pl.when(last_arrival_needs_fixing)
                .then(pl.col("arrival_time") + start_end_padding_hours)
                .otherwise("departure_time")
                .alias("departure_time"),
            pl.when(last_arrival_needs_fixing)
                .then("train_id")
                .otherwise("train_id_departure")
                .alias("train_id_departure"),
            pl.when(last_arrival_needs_fixing)
                .then(pl.col("oc_number").median())
                .otherwise("oc_number")
                .round().cast(pl.UInt32)
                .alias("oc_number"),
        )
        .group_by("train_id", maintain_order=True).first()
        .with_columns(
            pl.max_horizontal("full_cars", "oc_number").alias("truck_number")
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

    if container_string not in terminal.container_events:
        terminal.container_events[container_string] = {}
    terminal.container_events[container_string][event_type] = timestamp


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
    for name in [
        "train_ic_unload_events",
        "train_oc_prepared_events",
        "train_ic_picked_events",
        "train_start_load_events",
        "train_end_load_events",
        "train_departed_events",
    ]:
        if not hasattr(terminal, name):
            setattr(terminal, name, {})
        d = getattr(terminal, name)

        if train_id not in d or d[train_id].triggered:
            d[train_id] = env.event()
