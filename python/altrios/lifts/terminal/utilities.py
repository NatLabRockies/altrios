"""Module for general functions, classes, and unit conversion factors."""
from pathlib import Path
import polars as pl
import numpy as np
import numba as nb
import json
import yaml

# ---------------------------------------------------------------------------
# Run-scoped log gate. Threshold semantics match loggingLevel in classes.py:
# NONE=1 prints nothing, BASIC=2 prints BASIC, DEBUG=3 prints BASIC+DEBUG.
# A message at level L prints iff the current threshold >= L. utilities.py
# uses raw ints here to avoid importing classes.py (which depends on this
# module transitively via distances.py).
# ---------------------------------------------------------------------------
_LOG_LEVEL: int = 2  # BASIC


def set_log_level(level) -> None:
    """Set the run-scoped log threshold. Accepts loggingLevel or int."""
    global _LOG_LEVEL
    _LOG_LEVEL = int(level)


def log(level, msg: str) -> None:
    """Print msg iff its severity level is <= the current threshold."""
    if _LOG_LEVEL >= int(level):
        print(msg)


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
        log(2, "More than one track not yet supported!")  # BASIC

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


# ---------------------------------------------------------------------------
# Vessel and drayage schedule builders.
#
# These produce per-arrival event tables consumed by the ``rail_vessel``,
# ``vessel_truck``, and ``truck_rail`` (drayage-side) flows. They mirror
# ``build_train_timetable`` in returning either a Polars DataFrame or a list
# of dicts (one per arrival).
#
# Output column conventions (kept close to build_train_timetable):
#   build_vessel_schedule  -> vessel_id, vessel_name, arrival_time,
#                             departure_time, inbound_containers,
#                             outbound_containers
#   build_drayage_schedule -> truck_id, arrival_time, action ('dropoff' or
#                             'pickup'), container_id (nullable str)
#
# Drayage rows are independent per-truck visits — no per-train batching.
# A truck doing 'dropoff' delivers an export container (becomes OC on the
# rail/vessel side); a truck doing 'pickup' claims an import container
# (was IC on the rail/vessel side).
# ---------------------------------------------------------------------------

def build_vessel_schedule(vessel_call_list, terminal_name, as_dicts):
    """Build the per-vessel-call schedule for a terminal.

    Parameters
    ----------
    vessel_call_list : polars.DataFrame
        Input table with columns:
          - ``Vessel_ID`` (int)
          - ``Vessel_Name`` (str)
          - ``Origin_ID`` (str): terminal where vessel arrives from
          - ``Destination_ID`` (str): terminal where vessel departs to
          - ``Arrival_Time_Hr`` (float)
          - ``Departure_Time_Hr`` (float)
          - ``Inbound_Containers`` (int): containers landed at this terminal
          - ``Outbound_Containers`` (int): containers loaded at this terminal

        A vessel call appears in the schedule iff this terminal is either
        the origin or the destination of the call.
    terminal_name : str
        Filter rows where ``Origin_ID == terminal_name`` or
        ``Destination_ID == terminal_name``.
    as_dicts : bool
        If True, return a list of dicts; otherwise return a DataFrame.
    """
    required = {
        "Vessel_ID", "Vessel_Name", "Origin_ID", "Destination_ID",
        "Arrival_Time_Hr", "Departure_Time_Hr",
        "Inbound_Containers", "Outbound_Containers",
    }
    missing = required.difference(vessel_call_list.columns)
    if missing:
        raise ValueError(
            f"vessel_call_list missing required column(s): {sorted(missing)}"
        )

    df = (vessel_call_list
        .filter(
            (pl.col("Origin_ID") == pl.lit(terminal_name))
            | (pl.col("Destination_ID") == pl.lit(terminal_name))
        )
        .select(
            pl.col("Vessel_ID").alias("vessel_id"),
            pl.col("Vessel_Name").alias("vessel_name"),
            pl.col("Arrival_Time_Hr").alias("arrival_time"),
            pl.col("Departure_Time_Hr").alias("departure_time"),
            pl.col("Inbound_Containers").cast(pl.UInt32).alias("inbound_containers"),
            pl.col("Outbound_Containers").cast(pl.UInt32).alias("outbound_containers"),
        )
        .sort("arrival_time", "vessel_id")
    )

    if df.height == 0:
        raise ValueError(
            f"No vessel calls found in vessel_call_list at terminal '{terminal_name}'; "
            "build_vessel_schedule requires at least one row with Origin_ID or "
            "Destination_ID matching the terminal."
        )

    bad = df.filter(pl.col("departure_time") <= pl.col("arrival_time"))
    if bad.height > 0:
        raise ValueError(
            f"vessel_call_list has {bad.height} row(s) with "
            "departure_time <= arrival_time (vessel must dwell at berth)."
        )

    if as_dicts:
        return df.to_dicts()
    return df


def build_drayage_schedule(drayage_schedule, terminal_name, as_dicts):
    """Build the per-truck drayage schedule for a terminal.

    Parameters
    ----------
    drayage_schedule : polars.DataFrame
        Input table with columns:
          - ``Terminal_ID`` (str)
          - ``Arrival_Time_Hr`` (float)
          - ``Action`` (str): ``'dropoff'`` (truck delivers export container)
            or ``'pickup'`` (truck claims import container).
          - ``Truck_ID`` (int, optional): if missing, ids are assigned
            sequentially in arrival order starting at 1.
          - ``Container_ID`` (str, optional): nullable. When set, this is
            the explicit container the truck is targeting (used by tests).
    terminal_name : str
        Filter rows where ``Terminal_ID == terminal_name``.
    as_dicts : bool
        If True, return a list of dicts; otherwise return a DataFrame.
    """
    required = {"Terminal_ID", "Arrival_Time_Hr", "Action"}
    missing = required.difference(drayage_schedule.columns)
    if missing:
        raise ValueError(
            f"drayage_schedule missing required column(s): {sorted(missing)}"
        )

    df = (drayage_schedule
        .filter(pl.col("Terminal_ID") == pl.lit(terminal_name))
        .with_columns(pl.col("Action").str.to_lowercase().alias("Action"))
    )

    bad_action = df.filter(~pl.col("Action").is_in(["dropoff", "pickup"]))
    if bad_action.height > 0:
        raise ValueError(
            f"drayage_schedule has {bad_action.height} row(s) with Action "
            "not in {'dropoff', 'pickup'}."
        )

    select_cols = [
        pl.col("Arrival_Time_Hr").alias("arrival_time"),
        pl.col("Action").alias("action"),
    ]
    if "Container_ID" in drayage_schedule.columns:
        select_cols.append(pl.col("Container_ID").cast(pl.Utf8).alias("container_id"))
    else:
        select_cols.append(pl.lit(None, dtype=pl.Utf8).alias("container_id"))

    if "Truck_ID" in drayage_schedule.columns:
        select_cols.insert(0, pl.col("Truck_ID").cast(pl.Int64).alias("truck_id"))
        df = df.select(select_cols).sort("arrival_time", "truck_id")
    else:
        df = (df
            .sort("arrival_time")
            .with_row_index(name="truck_id", offset=1)
            .select(
                pl.col("truck_id").cast(pl.Int64),
                *select_cols,
            )
        )

    if df.height == 0:
        raise ValueError(
            f"No drayage rows found at terminal '{terminal_name}'; "
            "build_drayage_schedule requires at least one row with Terminal_ID "
            "matching the terminal."
        )

    if as_dicts:
        return df.to_dicts()
    return df


def record_container_event(state, container, event_type, timestamp):
    """Record a container-event row.

    Appends to ``state.container_events`` (the freight state_init seeds
    it as an empty list). When ``state.output`` is set by the
    workflow_engine runner, the row is also forwarded to the
    :class:`OutputCollector` for engine-side dual write.
    """
    if type(container) is str:
        container_string = container
    else:
        container_string = container.to_string()

    # Flat append is much cheaper in the hot path than the previous
    # dict-of-dict setdefault+assignment; the consumer pivots at end-of-sim.
    state.container_events.append((container_string, event_type, timestamp))

    output = getattr(state, "output", None)
    if output is not None:
        output.record_event({
            "container_id": container_string,
            "event_type": event_type,
            "timestamp": float(timestamp),
        })


def compute_consumption(energy_use_config, status: str, move: str, vehicle: str, energy_type: str, travel_time: float) -> float:
    """Return the per-event consumption for one resource action.

    The returned value is in the native unit of the configured
    ``*_consumption`` block: gallons for Diesel/Hybrid, kWh for Electric.

    ``energy_use_config`` is the ``config["energy_use"]`` sub-dict
    (consumption-rate table).

    Per-equipment rates: when ``vehicle`` is a specific equipment name
    (e.g. ``"main_stack_rtg"``, ``"sts_crane"``, ``"yard_tractor"``) the
    lookup tries ``<vehicle>_<status>`` first, then falls back to the
    generic key (``crane_<status>`` for loads, ``hostler_<status>``
    for trips). Callers that pass generic vehicle names
    (``vehicle="crane"``, ``vehicle="hostler"``, ``vehicle="truck"``) hit
    the generic key directly.
    """
    cfg = energy_use_config

    move = move.lower()
    status = status.lower()
    vehicle = vehicle.lower()
    energy_type = energy_type.capitalize()

    def _lookup(block_name: str, primary_key: str, fallback_key: str) -> float:
        block = cfg[block_name]
        entry = block.get(primary_key) or block.get(fallback_key)
        if entry is None:
            raise KeyError(
                f"energy_use.{block_name}: neither '{primary_key}' nor "
                f"fallback '{fallback_key}' defined."
            )
        if energy_type not in entry:
            raise KeyError(
                f"energy_use.{block_name}.{primary_key if primary_key in block else fallback_key}: "
                f"no '{energy_type}' rate."
            )
        return float(entry[energy_type])

    # --- load consumption (unit: per lift) ---
    if move == "load":
        primary = f"{vehicle}_{status}"
        fallback = f"crane_{status}"
        unit = _lookup("load_consumption", primary, fallback)
        consumption = unit

    # --- trip consumption (unit: hr × travel_time) ---
    elif move == "trip":
        primary = f"{vehicle}_{status}"
        # generic fallback: hostler_* for tractor-like vehicles, truck_* for trucks
        fallback = (
            f"truck_{status}" if vehicle == "truck" else f"hostler_{status}"
        )
        unit = _lookup("trip_consumption", primary, fallback)
        consumption = unit * travel_time

    # --- side pick consumption (unit: per lift) ---
    elif move == "side":
        unit = cfg["side_pick_consumption"]["side"][energy_type]
        consumption = unit

    else:
        raise ValueError(f"Unsupported move type '{move}' for vehicle '{vehicle}'.")

    return consumption


def record_consumption(consumption_records: list | None, vehicle_type: str, fuel_type: str, resource_id: str, track_id: str, train_id: str, container_id: str, event_type: str, zone: str, consumption_value: float, travel_time: float, env_now: float, role: str, quantity: str = "energy") -> dict:
    """Build a consumption record row.

    If ``consumption_records`` is non-None, the row is also appended to
    it (the module-level buffer in :mod:`altrios.lifts.terminal.consumption`,
    consumed by :func:`altrios.lifts.terminal.python_helpers.assemble_outputs`).
    Always returns the row dict so callers can additionally dual-write
    into a workflow-engine :class:`OutputCollector`.
    """
    row = {
        "resource_type": vehicle_type.lower(),
        "role": role,
        "fuel_type": fuel_type,
        "resource_id": str(resource_id),
        "track_id":str(track_id),
        "train_id": str(train_id),
        "container_id": str(container_id),
        "event_type": event_type,
        "zone": zone,
        "quantity": quantity,
        "consumption_value": float(consumption_value),
        "load/travel_time(hr)": float(travel_time),
        "record_timestamp": float(env_now),
    }
    if consumption_records is not None:
        consumption_records.append(row)
    return row

