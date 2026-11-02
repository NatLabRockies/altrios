from __future__ import annotations
from pathlib import Path
from altrios.train_planner import (
    data_prep,
    schedulers,
    planner_config,
    train_demand_generators,
)
import numpy as np
import polars as pl
import polars.selectors as cs
from typing import Union, Tuple, List, Dict, Optional
from itertools import repeat
import altrios as alt
from altrios import defaults


def dispatch(
    dispatch_time: int,
    origin: str,
    loco_pool: pl.DataFrame,
    train_tonnage: float,
    hp_required: float,
    total_cars: float,
    config: planner_config.TrainPlannerConfig,
) -> pl.Series:
    """
    Identify and select locomotives to dispatch for a train, based on origin, requirements, and availability.

    This function selects the optimal set of locomotives from the available pool at a specified
    origin based on horsepower requirements, tonnage needs, and configuration parameters.
    It implements locomotive selection logic including potential diesel requirements and
    ensures sufficient power for the given train.

    Parameters
    ----------
    dispatch_time : int
        Time (in hours) when the train is scheduled to depart
    origin : str
        Origin node name where the train will depart from
    loco_pool : pl.DataFrame
        DataFrame containing all locomotives in the network with their statuses and properties
    train_tonnage : float
        Total tonnage of the train to be dispatched
    hp_required : float
        Horsepower required for this train type on this origin-destination corridor
    total_cars : float
        Total number of cars (loaded, empty, or otherwise) included on the train
    config : planner_config.TrainPlannerConfig
        Configuration object with dispatch settings and rules

    Returns
    -------
    pl.Series
        Boolean series with same length as loco_pool, with True values indicating
        selected locomotives

    Raises
    ------
    ValueError
        If no locomotives are available at the origin or if requirements cannot be met
    """
    # Calculate horsepower required per ton of train weight
    hp_per_ton = hp_required / train_tonnage

    # Find candidate locomotives at the right place that are ready
    candidates = loco_pool.select(
        (pl.col("Node") == origin) & (pl.col("Status") == "Ready")
    ).to_series()

    # Check if any locomotives are available at the origin
    if not candidates.any():
        # Construct error message with detailed information about locomotive availability
        message = (
            f"""No available locomotives at node {origin} at hour {dispatch_time}."""
        )

        # Check for locomotives that are servicing or refueling at this origin
        waiting_counts = (
            loco_pool.filter(
                pl.col("Status").is_in(["Servicing", "Refuel_Queue"]),
                pl.col("Node") == origin,
            )
            .group_by(["Locomotive_Type"])
            .agg(pl.len())
        )

        # If no locomotives are waiting at the origin, show where all locomotives are located
        if waiting_counts.height == 0:
            message = (
                message
                + f"""\nNo locomotives are currently located there. Instead, they are at:"""
            )
            # Get counts of locomotives at each node
            locations = loco_pool.group_by("Node").agg(pl.len())
            for row in locations.iter_rows(named=True):
                message = (
                    message
                    + f"""
                    {row["Node"]}: {row["count"]}"""
                )
        else:
            # Show counts of locomotives servicing or refueling at the origin
            message = (
                message
                + f"""Count of locomotives refueling or waiting to refuel at {origin} are:"""
            )
            for row in waiting_counts.iter_rows(named=True):
                message = message + f"""\n{row["Locomotive_Type"]}: {row["count"]}"""

        raise ValueError(message)

    # Initialize the list of selected locomotives with all candidates
    selected = candidates

    # Initialize variable to store horsepower of required diesel locomotive
    diesel_to_require_hp = 0

    # Handle diesel requirement if configured
    if config.require_diesel:
        # First available diesel (in order of loco_pool) will be moved from candidates to selected
        # TODO gracefully handle cases when there is no diesel locomotive to be dispatched
        # (ex: hold the train until enough diesels are present)

        # Create filter for diesel locomotives
        diesel_filter = pl.col("Fuel_Type").cast(pl.Utf8).str.contains("(?i)diesel")

        # Find diesel locomotives among candidates
        diesel_candidates = loco_pool.select(
            pl.lit(candidates) & diesel_filter
        ).to_series()

        # Check if any diesel locomotives are available
        if not diesel_candidates.any():
            # Count diesel locomotives that are servicing or refueling at this origin
            refueling_diesel_count = loco_pool.filter(
                pl.col("Node") == origin,
                pl.col("Status").is_in(["Servicing", "Refuel_Queue"]),
                diesel_filter,
            ).select(pl.len())[0, 0]

            # Construct detailed error message about diesel locomotive availability
            message = f"""No available diesel locomotives at node {origin} at hour {dispatch_time}, so
                    the one-diesel-per-consist rule cannot be satisfied. {refueling_diesel_count} diesel locomotives at
                    {origin} are servicing, refueling, or queueing."""

            # Add information about refueling capacity if relevant
            if refueling_diesel_count > 0:
                diesel_port_count = (
                    loco_pool.filter(pl.col("Node") == origin, diesel_filter)
                    .select(pl.col("Port_Count").min())
                    .item()
                )
                message += f""" (queue capacity {diesel_port_count})."""
            else:
                message += "."
            raise ValueError(message)

        # Find index of first diesel locomotive
        diesel_to_require = diesel_candidates.eq(True).cum_sum().eq(1).arg_max()

        # Get horsepower of the required diesel locomotive
        diesel_to_require_hp = loco_pool.filter(diesel_filter).select(pl.first("HP"))

        # Remove the diesel from candidates to avoid double-counting
        candidates[diesel_to_require] = False

    # Initialize error message string
    message = ""

    if config.cars_per_locomotive_fixed:
        # Get as many available locomotives as are needed (in order of loco_pool)
        enough = loco_pool.select(
            (pl.lit(1.0) * pl.lit(candidates)).cum_sum() >= total_cars
        ).to_series()

        # Check if we have enough locomotives
        if not enough.any():
            # Construct error message with information about locomotive shortage
            message = f"""Locomotives needed ({total_cars}) at {origin} at hour {dispatch_time}
                is more than the available locomotives ({candidates.sum()}).
                Count of locomotives servicing, refueling, or queueing at {origin} are:"""
    else:
        # Get running sum, including first diesel, of hp of the candidates (in order of loco_pool)
        enough = loco_pool.select(
            (
                (
                    (pl.col("HP") - (pl.col("Loco_Mass_Tons") * pl.lit(hp_per_ton)))
                    * pl.lit(candidates)
                ).cum_sum()
                + pl.lit(diesel_to_require_hp)
            )
            >= hp_required
        ).to_series()

        # Check if we have enough horsepower
        if not enough.any():
            # Calculate available horsepower
            available_hp = loco_pool.select(
                (
                    (pl.col("HP") - (pl.col("Loco_Mass_Tons") * pl.lit(hp_per_ton)))
                    * pl.lit(candidates)
                )
                .cum_sum()
                .max()
            )[0, 0]

            # Construct error message with information about horsepower shortage
            message = f"""Outbound horsepower needed ({hp_required}) at {origin} at hour {dispatch_time}
                is more than the available horsepower ({available_hp}).
                Count of locomotives servicing, refueling, or queueing at {origin} are:"""

    if not enough.any():
        # Hold the train until enough locomotives are present (future development)
        waiting_counts = (
            loco_pool.filter(
                pl.col("Node") == origin,
                pl.col("Status").is_in(["Servicing", "Refuel_Queue"]),
            )
            .group_by(["Locomotive_Type"])
            .agg(pl.count().alias("count"))
        )

        # Add details about waiting locomotives to the error message
        for row in waiting_counts.iter_rows(named=True):
            message = (
                message
                + f"""
            {row["Locomotive_Type"]}: {row["count"]}"""
            )

        # Raise error when we don't have enough locomotives/horsepower
        raise ValueError(message)

    # Find the index of the first row that satisfies our requirements
    last_row_to_use = enough.eq(True).cum_sum().eq(1).arg_max()

    # Set false all the locomotives that would add unnecessary hp
    selected[np.arange(last_row_to_use + 1, len(selected))] = False

    # If diesel is required, make sure it's included in the selection
    if config.require_diesel:
        # Add first diesel (which could come after last_row_to_use) to selection list
        selected[diesel_to_require] = True

    return selected


def update_refuel_queue(
    loco_pool: pl.DataFrame,
    refuelers: pl.DataFrame,
    current_time: float,
    event_tracker: pl.DataFrame,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Update locomotive refueling status, manage service queues, and track events.

    This function processes arrived locomotives, updates refueling and servicing status, and
    manages the refueling queue across all locations. It tracks when locomotives finish refueling
    or servicing, updates their status appropriately, and records these events.

    Parameters
    ----------
    loco_pool : pl.DataFrame
        DataFrame containing all locomotives in the network with their current status
    refuelers : pl.DataFrame
        DataFrame containing all refueling ports in the network with capacity information
    current_time : float
        Current simulation time in hours
    event_tracker : pl.DataFrame
        DataFrame tracking locomotive events (arrivals, refueling, etc.)

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame]
        loco_pool : Updated locomotive pool DataFrame with new statuses
        event_tracker : Updated event tracker with new refueling/servicing events
    """

    # Identify locomotives that have arrived at their destination
    arrived = loco_pool.select(
        (pl.col("Status") == "Dispatched") & (pl.col("Arrival_Time") <= current_time)
    ).to_series()

    # Process newly arrived locomotives
    if arrived.sum() > 0:
        # Update loco_pool with refueling information and new status
        loco_pool = (
            loco_pool
            # Remove old refueler info columns to avoid duplication
            .drop(["Refueler_J_Per_Hr", "Port_Count", "Battery_Headroom_J"])
            # Join with refuelers data to get updated refueling information at current location
            .join(
                refuelers.select(
                    [
                        "Node",
                        "Locomotive_Type",
                        "Fuel_Type",
                        "Refueler_J_Per_Hr",
                        "Port_Count",
                        "Battery_Headroom_J",
                    ]
                ),
                on=["Node", "Locomotive_Type", "Fuel_Type"],
                how="left",
            )
            # Update status and target SOC for arrived locomotives
            .with_columns(
                # Change status to Refuel_Queue for arrived locomotives
                pl.when(arrived)
                .then(pl.lit("Refuel_Queue"))
                .otherwise(pl.col("Status"))
                .alias("Status"),
                # Set target SOC, respecting battery headroom constraints
                pl.when(arrived)
                .then(
                    pl.max_horizontal(
                        [
                            pl.col("SOC_Max_J") - pl.col("Battery_Headroom_J"),
                            pl.col("SOC_J"),
                        ]
                    )
                )
                .otherwise(pl.col("SOC_Target_J"))
                .alias("SOC_Target_J"),
            )
            # Calculate refueling duration based on energy needed and refueling rate
            .with_columns(
                pl.when(arrived)
                .then(
                    (pl.col("SOC_Target_J") - pl.col("SOC_J"))
                    / pl.col("Refueler_J_Per_Hr")
                )
                .otherwise(pl.col("Refuel_Duration"))
                .alias("Refuel_Duration")
            )
            # Sort locomotives by various criteria for processing
            .sort(
                "Node",
                "Locomotive_Type",
                "Fuel_Type",
                "Arrival_Time",
                "Locomotive_ID",
                descending=False,
                nulls_last=True,
            )
        )

        # Organize locomotives by charger type and node for queue processing
        charger_type_breakouts = (
            loco_pool.filter(
                # Only consider locomotives in refuel queue without scheduled completion times
                pl.col("Status") == "Refuel_Queue",
                (pl.col("Refueling_Done_Time") >= current_time)
                | (pl.col("Refueling_Done_Time").is_null()),
            )
            # Partition by node and locomotive type to process each charger separately
            .partition_by(["Node", "Locomotive_Type"])
        )

        # Initialize list to store processed charger data
        charger_type_list = []

        # Process each charger type group
        for charger_type in charger_type_breakouts:
            # Extract relevant columns for processing
            loco_ids = charger_type.get_column("Locomotive_ID")
            arrival_times = charger_type.get_column("Arrival_Time")
            refueling_done_times = charger_type.get_column("Refueling_Done_Time")
            refueling_durations = charger_type.get_column("Refuel_Duration")
            port_counts = charger_type.get_column("Port_Count")

            # Process each locomotive in this charger group
            for i in range(0, refueling_done_times.len()):
                # Skip locomotives that already have a completion time
                if refueling_done_times[i] is not None:
                    continue

                # Find the next available charging port based on when current locomotives will finish
                next_done = refueling_done_times.filter(
                    (refueling_done_times.is_not_null())
                    & (
                        refueling_done_times.rank(method="ordinal", descending=True).eq(
                            port_counts[i]
                        )
                    )
                )

                # Determine when this locomotive can start charging
                if next_done.len() == 0:
                    next_done = arrival_times[i]  # No wait if ports available
                else:
                    next_done = max(
                        next_done[0], arrival_times[i]
                    )  # Either wait for port or arrival

                # Calculate when refueling will be complete
                refueling_done_times[i] = next_done + refueling_durations[i]

            # Add processed data to our list
            charger_type_list.append(pl.DataFrame([loco_ids, refueling_done_times]))

        # Combine all charger data
        all_queues = pl.concat(charger_type_list, how="diagonal")

        # Update the locomotive pool with new refueling completion times
        loco_pool = (
            loco_pool
            # Join the calculated refueling times with the locomotive pool
            .join(all_queues, on="Locomotive_ID", how="left", suffix="_right")
            # Update refueling done times with newly calculated values
            .with_columns(
                pl.when(pl.col("Refueling_Done_Time_right").is_not_null())
                .then(pl.col("Refueling_Done_Time_right"))
                .otherwise(pl.col("Refueling_Done_Time"))
                .alias("Refueling_Done_Time")
            )
            # Drop temporary column
            .drop("Refueling_Done_Time_right")
        )

    # Identify locomotives that have finished refueling
    refueling_finished = loco_pool.select(
        (pl.col("Status") == "Refuel_Queue")
        & (pl.col("Refueling_Done_Time") <= current_time)
    ).to_series()

    # Count how many locomotives finished refueling
    refueling_finished_count = refueling_finished.sum()

    # Process locomotives that finished refueling
    if refueling_finished_count > 0:
        # Record the refueling events (both start and end times)
        new_rows = pl.DataFrame(
            [
                # Create event types - half 'Refueling_Start', half 'Refueling_End'
                np.concatenate(
                    [
                        np.tile("Refueling_Start", refueling_finished_count),
                        np.tile("Refueling_End", refueling_finished_count),
                    ]
                ),
                # Create time values - start times followed by end times
                np.concatenate(
                    [
                        loco_pool.filter(refueling_finished)
                        .select(
                            pl.col("Refueling_Done_Time") - pl.col("Refuel_Duration")
                        )
                        .to_series(),
                        loco_pool.filter(refueling_finished).get_column(
                            "Refueling_Done_Time"
                        ),
                    ]
                ),
                # Create locomotive IDs - duplicated for start and end events
                np.tile(
                    loco_pool.filter(refueling_finished).get_column("Locomotive_ID"), 2
                ),
            ],
            schema=event_tracker.columns,
            orient="col",
        )

        # Add new events to the tracker
        event_tracker = pl.concat([event_tracker, new_rows])

        # Update locomotive pool with post-refueling states
        loco_pool = loco_pool.with_columns(
            # Update SOC to target value for refueled locomotives
            pl.when(refueling_finished)
            .then(pl.col("SOC_Target_J"))
            .otherwise(pl.col("SOC_J"))
            .alias("SOC_J"),
            # Clear refueling done time for finished locomotives
            pl.when(refueling_finished)
            .then(pl.lit(None))
            .otherwise(pl.col("Refueling_Done_Time"))
            .alias("Refueling_Done_Time"),
            # Update status based on servicing needs
            pl.when(
                pl.lit(refueling_finished)
                & (pl.col("Servicing_Done_Time") <= current_time)
            )
            .then(pl.lit("Ready"))  # Ready if servicing also complete
            .when(
                pl.lit(refueling_finished)
                & (pl.col("Servicing_Done_Time") > current_time)
            )
            .then(pl.lit("Servicing"))  # Move to servicing if still needed
            .otherwise(pl.col("Status"))
            .alias("Status"),
        )

    # Identify locomotives that have finished servicing
    servicing_finished = loco_pool.select(
        (pl.col("Status") == "Servicing")
        & (pl.col("Servicing_Done_Time") <= current_time)
    ).to_series()

    # Process locomotives that finished servicing
    if servicing_finished.sum() > 0:
        # Update locomotive pool with post-servicing states
        loco_pool = loco_pool.with_columns(
            # Mark serviced locomotives as ready
            pl.when(servicing_finished)
            .then(pl.lit("Ready"))
            .otherwise(pl.col("Status"))
            .alias("Status"),
            # Clear servicing done time for finished locomotives
            pl.when(servicing_finished)
            .then(pl.lit(None))
            .otherwise(pl.col("Servicing_Done_Time"))
            .alias("Servicing_Done_Time"),
        )

    # Return sorted locomotive pool and updated event tracker
    return loco_pool.sort("Locomotive_ID"), event_tracker


def run_train_planner(
    rail_vehicles: List[alt.RailVehicle],
    location_map: Dict[str, List[alt.Location]],
    network: List[alt.Link],
    # TODO: figure out why this input, which is not provided anywhere, needs to exist
    loco_pool: Optional[pl.DataFrame],
    refuelers: Optional[pl.DataFrame],
    scenario_year: int,
    train_type: alt.TrainType = alt.TrainType.Freight,
    config: planner_config.TrainPlannerConfig = planner_config.TrainPlannerConfig(),
    demand_file: Union[pl.DataFrame, Path, str] = defaults.DEMAND_FILE,
    network_charging_guidelines: Optional[pl.DataFrame] = None,
) -> Tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    List[alt.SpeedLimitTrainSim],
    List[alt.EstTimeNet],
]:
    """
    Run the train planner to generate consist plans, refueling schedules, and simulations.

    This function is the main entry point for train planning. It processes demand data,
    schedules trains, assigns locomotives, plans refueling, and generates train simulations
    based on the provided configuration. It handles both single-train mode and multi-train
    scheduling across a network.

    Parameters
    ----------
    rail_vehicles : List[alt.RailVehicle]
        List of available rail vehicle types with their properties
    location_map : Dict[str, List[alt.Location]]
        Dictionary mapping location IDs to lists of Location objects
    network : List[alt.Link]
        List of links defining the rail network
    loco_pool : Optional[pl.DataFrame]
        DataFrame containing available locomotives with their properties.
        If None, will be generated based on demand.
    refuelers : Optional[pl.DataFrame]
        DataFrame containing refueling facilities with their capacities.
        If None, will be generated based on configuration.
    scenario_year : int
        The year for which to run the simulation (affects energy prices, etc.)
    train_type : alt.TrainType
        Type of train to simulate (default: Freight)
    config : planner_config.TrainPlannerConfig
        Configuration object with planning parameters
    demand_file : Union[pl.DataFrame, Path, str]
        Source of demand data, either as DataFrame or file path
    network_charging_guidelines : Optional[pl.DataFrame]
        Guidelines for charging infrastructure by location

    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, List[alt.SpeedLimitTrainSim], List[alt.EstTimeNet]]
        train_consist_plan : DataFrame with planned train consists and schedules
        loco_pool : Updated DataFrame of all locomotives and their states
        refuelers : DataFrame of all refueling facilities
        speed_limit_train_sims : List of SpeedLimitTrainSim objects for each planned train
        est_time_nets : List of EstTimeNet objects with estimated timings
    """
    # Append additional locomotive info to configuration
    config.loco_info = data_prep.append_loco_info(config.loco_info)

    # Load freight demand data and get list of nodes
    demand, node_list = data_prep.load_freight_demand(demand_file, config)

    # Set default return demand generators if none provided
    if config.return_demand_generators is None:
        config.return_demand_generators = (
            train_demand_generators.get_default_return_demand_generators()
        )

    # Create mapping from freight types to car types
    freight_type_to_car_type = {}
    for rv in rail_vehicles:
        rv_dict = rv.to_pydict()
        # Check for duplicate mappings (should not happen)
        if rv_dict["freight_type"] in freight_type_to_car_type:
            raise Exception(
                f"More than one rail vehicle car type for freight type {rv_dict['freight_type']}"
            )
        else:
            # Map this freight type to its car type
            freight_type_to_car_type[rv_dict["freight_type"]] = rv_dict["car_type"]

    # Handle single train mode (simple scheduling)
    if config.single_train_mode:
        # Generate demand trains without returns or rebalancing
        demand = train_demand_generators.generate_demand_trains(
            demand,
            demand_returns=pl.DataFrame(),
            demand_rebalancing=pl.DataFrame(),
            rail_vehicles=rail_vehicles,
            config=config,
        )
        # Create a simple dispatch schedule with one train per day
        dispatch_schedule = (
            demand.with_row_index(name="index")
            .with_columns(pl.col("index").mul(24.0).alias("Hour"))  # 24-hour intervals
            .drop("index")
        )
    else:
        # Initialize empty dataframes for return and rebalancing demand
        demand_returns = pl.DataFrame()
        demand_rebalancing = pl.DataFrame()

        # If no dispatch scheduler is set, aggregate demand
        if config.dispatch_scheduler is None:
            # Define variables to group by
            grouping_vars = ["Origin", "Destination", "Train_Type"]
            aggregations = []

            # Add optional grouping variables if present
            if "Number_of_Days" in demand.collect_schema():
                grouping_vars.append("Number_of_Days")

            # Add aggregations for container or car counts if present
            if "Number_of_Containers" in demand.collect_schema():
                # Convert containers to cars
                aggregations.append(
                    pl.col("Number_of_Containers")
                    .sum()
                    .truediv(config.containers_per_car)
                )
            if "Number_of_Cars" in demand.collect_schema():
                aggregations.append(pl.col("Number_of_Cars").sum())

            # Group and aggregate demand
            demand = demand.group_by(grouping_vars).agg(
                pl.max_horizontal(aggregations).ceil().alias("Number_of_Cars")
            )

        # Generate return and rebalancing demand if not already scheduled
        if "Hour" not in demand.schema:
            # Generate return demand (empty cars going back)
            demand_returns = train_demand_generators.generate_return_demand(
                demand, config
            )

            # Generate rebalancing demand for manifest trains if needed
            if demand.filter(pl.col("Train_Type").str.contains("Manifest")).height > 0:
                demand_rebalancing = (
                    train_demand_generators.generate_manifest_rebalancing_demand(
                        demand, node_list, config
                    )
                )

        # Set up dispatch scheduler if not provided
        dispatch_scheduler = config.dispatch_scheduler
        if dispatch_scheduler is None:
            # Generate full demand including returns and rebalancing
            demand = train_demand_generators.generate_demand_trains(
                demand,
                demand_returns,
                demand_rebalancing,
                rail_vehicles,
                freight_type_to_car_type,
                config,
            )

            # Set default scheduler
            dispatch_scheduler = schedulers.dispatch_uniform_demand_uniform_departure

        # Create dispatch schedule using configured scheduler
        dispatch_schedule = dispatch_scheduler(demand, rail_vehicles, freight_type_to_car_type, config)
        # Uncomment to add random jitter to departure times
        # dispatch_schedule = dispatch_schedule.with_columns(pl.col("Hour").add(pl.random.rand()))

    # Create locomotive pool if not provided
    if loco_pool is None:
        loco_pool = data_prep.build_locopool(
            config=config, demand_file=demand, dispatch_schedule=dispatch_schedule
        )

    # Create refuelers if not provided
    if refuelers is None:
        refuelers = data_prep.build_refuelers(
            node_list,
            loco_pool,
            config.refueler_info,
            config.refuelers_per_incoming_corridor,
        )

    # Load network charging guidelines if not provided
    if network_charging_guidelines is None:
        network_charging_guidelines = pl.read_csv(
            alt.resources_root() / "networks" / "network_charging_guidelines.csv"
        )

    # Apply charging guidelines to refuelers and loco_pool
    refuelers, loco_pool = data_prep.append_charging_guidelines(
        refuelers, loco_pool, demand, network_charging_guidelines
    )

    # Get final departure time for simulation end
    final_departure = dispatch_schedule.get_column("Hour").max()

    # Initialize dataframe to store train consist plan
    train_consist_plan = pl.DataFrame(
        schema={
            "Train_ID": pl.Int64,
            "Train_Type": pl.Utf8,
            "Locomotive_ID": pl.UInt32,
            "Locomotive_Type": pl.Categorical,
            "Origin_ID": pl.Utf8,
            "Destination_ID": pl.Utf8,
            "Cars_Loaded": pl.Float64,
            "Cars_Empty": pl.Float64,
            "Containers_Loaded": pl.Float64,
            "Containers_Empty": pl.Float64,
            "Departure_SOC_J": pl.Float64,
            "Departure_Time_Planned_Hr": pl.Float64,
            "Arrival_Time_Planned_Hr": pl.Float64,
        }
    )

    # Initialize dataframe to track locomotive events
    event_tracker = pl.DataFrame(
        schema=[
            ("Event_Type", pl.Utf8),
            ("Time_Hr", pl.Float64),
            ("Locomotive_ID", pl.UInt32),
        ]
    )

    # Initialize train ID counter and result lists
    train_id_counter = 1
    speed_limit_train_sims = []
    est_time_nets = []

    # Initialize simulation state
    done = False

    # Start at first departure time
    current_time = dispatch_schedule.get_column("Hour").min()

    # Simulation loop - process trains in chronological order
    while not done:
        # Find trains scheduled to depart at the current time
        current_dispatches = dispatch_schedule.filter(pl.col("Hour") == current_time)

        # If there are trains to dispatch at this time
        if current_dispatches.height > 0:
            # Update refueling status for all locomotives
            loco_pool, event_tracker = update_refuel_queue(
                loco_pool, refuelers, current_time, event_tracker
            )

            # Process each train scheduled to depart at current time
            for this_train in current_dispatches.iter_rows(named=True):
                # Only process trains with positive tonnage
                if this_train["Tons_Per_Train"] > 0:
                    # Generate unique train ID as string
                    train_id = str(train_id_counter)

                    # Handle locomotive selection differently in single train mode
                    if config.single_train_mode:
                        # In single train mode, select all locomotives
                        selected = loco_pool.select(
                            pl.col("Locomotive_ID").is_not_null().alias("selected")
                        ).to_series()
                        # Use all locomotives
                        dispatched = loco_pool
                    else:
                        # In normal mode, use dispatch function to select appropriate locomotives
                        selected = dispatch(
                            current_time,
                            this_train["Origin"],
                            loco_pool,
                            this_train["Tons_Per_Train"],
                            this_train["HP_Required"],
                            this_train["Cars_Loaded"] + this_train["Cars_Empty"],
                            config,
                        )
                        # Filter to only selected locomotives
                        dispatched = loco_pool.filter(selected)

                    # Calculate drag coefficient if configured
                    if config.drag_coeff_function is not None:
                        # Apply drag coefficient function based on number of cars
                        cd_area_vec = config.drag_coeff_function(
                            int(this_train["Number_of_Cars"])
                        )
                    else:
                        # No custom drag coefficient
                        cd_area_vec = None

                    # Configure rail vehicles for this train
                    rv_to_use, n_cars_by_type = data_prep.configure_rail_vehicles(
                        this_train, rail_vehicles, freight_type_to_car_type
                    )

                    # Create train configuration
                    train_config = alt.TrainConfig(
                        rail_vehicles=rv_to_use,
                        n_cars_by_type=n_cars_by_type,
                        train_type=train_type,
                        cd_area_vec=cd_area_vec,
                    )

                    # Get initial state of charge for selected locomotives
                    loco_start_soc_j = dispatched.get_column("SOC_J")

                    # Create ordering of locomotives for processing
                    dispatch_order = (
                        dispatched.select(
                            pl.col("Locomotive_ID").rank().alias("rank").cast(pl.UInt32)
                        )
                        .with_row_index()
                        .sort("index")
                    )

                    # Sort dispatched locomotives by ID
                    dispatched = dispatched.sort("Locomotive_ID")

                    # Calculate state of charge as percentage of capacity
                    loco_start_soc_pct = dispatched.select(
                        pl.col("SOC_J") / pl.col("Capacity_J")
                    ).to_series()

                    # Create list of locomotive objects from configuration
                    locos = [
                        config.loco_info[
                            config.loco_info["Locomotive_Type"] == loco_type
                        ]["Rust_Loco"]
                        .to_list()[0]
                        .copy()
                        for loco_type in dispatched.get_column("Locomotive_Type")
                    ]

                    # Set state of charge for electric locomotives
                    for i, loco in enumerate(locos):
                        if dispatched.get_column("Fuel_Type")[i] == "Electricity":
                            loco_dict = loco.to_pydict()
                            loco_type = next(iter(loco_dict["loco_type"].keys()))
                            loco_dict["loco_type"][loco_type]["res"]["state"]["soc"] = (
                                float(loco_start_soc_pct[i])
                            )
                            locos[i] = alt.Locomotive.from_pydict(loco_dict)

                    # Create locomotive consist from the selected locomotives
                    loco_con = alt.Consist(
                        loco_vec=locos,
                        save_interval=None,
                    )

                    # Create initial train state with correct time
                    init_train_state = alt.InitTrainState(
                        time_seconds=current_time * 3600  # Convert hours to seconds
                    )

                    # Create train simulation builder
                    tsb = alt.TrainSimBuilder(
                        train_id=train_id,
                        origin_id=this_train["Origin"],
                        destination_id=this_train["Destination"],
                        train_config=train_config,
                        loco_con=loco_con,
                        init_train_state=init_train_state,
                    )

                    # Create speed-limited train simulation
                    slts = tsb.make_speed_limit_train_sim(
                        location_map=location_map,
                        save_interval=None,
                        simulation_days=config.simulation_days,
                        scenario_year=scenario_year,
                    )

                    # Generate estimated travel times
                    (est_time_net, loco_con_out) = alt.make_est_times(
                        slts, network, config.failed_sim_logging_path
                    )

                    # Calculate travel time with scaling factors
                    travel_time = (
                        est_time_net.get_running_time_hours()
                        * config.dispatch_scaling_dict["time_mult_factor"]
                        + config.dispatch_scaling_dict["hours_add"]
                    )

                    # Calculate energy usage for each locomotive
                    locos = loco_con_out.to_pydict()["loco_vec"]
                    # Extract energy consumption from locomotive objects based on type
                    energy_use_locos = []
                    for loco in locos:
                        loco_type = next(iter(loco["loco_type"].values()))
                        if "res" in loco_type.keys():
                            energy_use_locos.append(
                                loco_type["res"]["state"]["energy_out_chemical_joules"]
                            )
                        elif "fc" in loco_type.keys():
                            energy_use_locos.append(
                                loco_type["fc"]["state"]["energy_fuel_joules"]
                            )
                        elif ("fc" in loco_type.keys()) and ("res" in loco_type.keys()):
                            energy_use_locos.append(
                                loco_type["fc"]["state"]["energy_fuel_joules"]
                                + loco_type["res"]["state"][
                                    "energy_out_chemical_joules"
                                ]
                            )
                        else:
                            energy_use_locos.append(0)

                    energy_use_j = np.zeros(len(loco_pool))
                    # Assign energy use to selected locomotives based on rank order
                    energy_use_j[selected] = [
                        energy_use_locos[i - 1]
                        for i in dispatch_order.get_column("rank").to_list()
                    ]
                    # Apply energy scaling factor
                    energy_use_j *= config.dispatch_scaling_dict["energy_mult_factor"]
                    energy_use_j = pl.Series(energy_use_j)

                    # Store simulation results
                    speed_limit_train_sims.append(slts)
                    est_time_nets.append(est_time_net)

                    # Update locomotive pool with post-dispatch state
                    loco_pool = loco_pool.with_columns(
                        # Update location to destination for dispatched locomotives
                        pl.when(selected)
                        .then(pl.lit(this_train["Destination"]))
                        .otherwise(pl.col("Node"))
                        .alias("Node"),
                        # Set arrival time for dispatched locomotives
                        pl.when(selected)
                        .then(pl.lit(current_time + travel_time))
                        .otherwise(pl.col("Arrival_Time"))
                        .alias("Arrival_Time"),
                        # Set servicing completion time for dispatched locomotives
                        pl.when(selected)
                        .then(
                            pl.lit(current_time + travel_time)
                            + pl.col("Min_Servicing_Time_Hr")
                        )
                        .otherwise(pl.col("Servicing_Done_Time"))
                        .alias("Servicing_Done_Time"),
                        # Clear refueling time for dispatched locomotives
                        pl.when(selected)
                        .then(None)
                        .otherwise(pl.col("Refueling_Done_Time"))
                        .alias("Refueling_Done_Time"),
                        # Set status to dispatched for selected locomotives
                        pl.when(selected)
                        .then(pl.lit("Dispatched"))
                        .otherwise(pl.col("Status"))
                        .alias("Status"),
                        # Update SOC based on energy usage, respecting min/max constraints
                        pl.when(selected)
                        .then(
                            pl.max_horizontal(
                                pl.col("SOC_Min_J"),
                                pl.min_horizontal(
                                    pl.col("SOC_J") - pl.lit(energy_use_j),
                                    pl.col("SOC_Max_J"),
                                ),
                            )
                        )
                        .otherwise(pl.col("SOC_J"))
                        .alias("SOC_J"),
                    )

                    # Populate the output dataframe with the dispatched trains
                    # Count how many locomotives were selected
                    new_row_count = selected.sum()

                    # Create new rows for the train consist plan
                    new_rows = pl.DataFrame(
                        [
                            # Train ID (same for all locomotives in this train)
                            pl.Series(repeat(train_id_counter, new_row_count)),
                            # Train type (same for all locomotives)
                            pl.Series(repeat(this_train["Train_Type"], new_row_count)),
                            # Locomotive IDs (unique per locomotive)
                            loco_pool.filter(selected).get_column("Locomotive_ID"),
                            # Locomotive types (unique per locomotive)
                            loco_pool.filter(selected).get_column("Locomotive_Type"),
                            # Origin (same for all locomotives)
                            pl.Series(repeat(this_train["Origin"], new_row_count)),
                            # Destination (same for all locomotives)
                            pl.Series(repeat(this_train["Destination"], new_row_count)),
                            # Number of loaded cars (same for all locomotives)
                            pl.Series(repeat(this_train["Cars_Loaded"], new_row_count)),
                            # Number of empty cars (same for all locomotives)
                            pl.Series(repeat(this_train["Cars_Empty"], new_row_count)),
                            # Number of loaded containers (same for all locomotives)
                            pl.Series(
                                repeat(this_train["Containers_Loaded"], new_row_count)
                            ),
                            # Number of empty containers (same for all locomotives)
                            pl.Series(
                                repeat(this_train["Containers_Empty"], new_row_count)
                            ),
                            # Starting SOC (unique per locomotive)
                            loco_start_soc_j,
                            # Departure time (same for all locomotives)
                            pl.Series(repeat(current_time, new_row_count)),
                            # Estimated arrival time (same for all locomotives)
                            pl.Series(
                                repeat(current_time + travel_time, new_row_count)
                            ),
                        ],
                        schema=train_consist_plan.columns,
                        orient="col",
                    )

                    # Add new rows to the train consist plan
                    train_consist_plan = pl.concat(
                        [train_consist_plan, new_rows], how="diagonal_relaxed"
                    )

                    # Increment train ID counter for next train
                    train_id_counter += 1

        # Check if we've processed all scheduled departures
        if current_time >= final_departure:
            # Set current time to infinity to finish processing any remaining refueling
            current_time = float("inf")
            # Final update of refueling queues
            loco_pool, event_tracker = update_refuel_queue(
                loco_pool, refuelers, current_time, event_tracker
            )
            # Mark simulation as complete
            done = True
        else:
            # Find next scheduled departure time
            current_time = (
                dispatch_schedule.filter(pl.col("Hour").gt(current_time))
                .get_column("Hour")
                .min()
            )

    # Post-process the train consist plan
    train_consist_plan = (
        train_consist_plan.with_columns(
            # Convert categorical columns to strings
            cs.categorical().cast(str),
            # Ensure ID columns are unsigned integers
            pl.col("Train_ID", "Locomotive_ID").cast(pl.UInt32),
        )
        # Sort by locomotive ID and train ID
        .sort(["Locomotive_ID", "Train_ID"], descending=False)
    )

    # Convert categorical columns in loco_pool and refuelers to strings
    loco_pool = loco_pool.with_columns(cs.categorical().cast(str))
    refuelers = refuelers.with_columns(cs.categorical().cast(str))

    # Sort event tracker by locomotive ID, time, and event type
    event_tracker = event_tracker.sort(["Locomotive_ID", "Time_Hr", "Event_Type"])

    # Extract refueling start times from event tracker
    service_starts = (
        event_tracker.filter(pl.col("Event_Type") == "Refueling_Start")
        .get_column("Time_Hr")
        .rename("Refuel_Start_Time_Planned_Hr")
    )

    # Extract refueling end times from event tracker
    service_ends = (
        event_tracker.filter(pl.col("Event_Type") == "Refueling_End")
        .get_column("Time_Hr")
        .rename("Refuel_End_Time_Planned_Hr")
    )

    # Add refueling start and end times to the train consist plan
    train_consist_plan = train_consist_plan.with_columns(service_starts, service_ends)

    # Return results
    return (
        train_consist_plan,
        loco_pool,
        refuelers,
        speed_limit_train_sims,
        est_time_nets,
    )


if __name__ == "__main__":
    # Load rail vehicle definitions from yaml files
    rail_vehicles = [
        alt.RailVehicle.from_file(vehicle_file)
        for vehicle_file in Path(alt.resources_root() / "rolling_stock/").glob("*.yaml")
    ]

    # Import location data
    location_map = alt.import_locations(
        str(alt.resources_root() / "networks/default_locations.csv")
    )

    # Load network definition
    network = alt.Network.from_file(
        str(alt.resources_root() / "networks/Taconite-NoBalloon.yaml")
    )

    # Create planner configuration
    config = planner_config.TrainPlannerConfig()

    # Set simulation days with warm-up period
    config.simulation_days = defaults.SIMULATION_DAYS + 2 * defaults.WARM_START_DAYS

    # Build locomotive pool from demand file
    loco_pool = data_prep.build_locopool(config, defaults.DEMAND_FILE)

    # Load freight demand data
    demand, node_list = data_prep.load_freight_demand(
        defaults.DEMAND_FILE, config=config
    )

    # Build refuelers based on node list and locomotive pool
    refuelers = data_prep.build_refuelers(
        node_list,
        loco_pool,
        config.refueler_info,
        config.refuelers_per_incoming_corridor,
    )

    # Run the train planner
    output = run_train_planner(
        rail_vehicles=rail_vehicles,
        location_map=location_map,
        network=network,
        loco_pool=loco_pool,
        refuelers=refuelers,
        scenario_year=defaults.BASE_ANALYSIS_YEAR,
        config=config,
    )
