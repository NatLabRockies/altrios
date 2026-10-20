from __future__ import annotations
import time
from pathlib import Path
from altrios.train_planner import (
    data_prep,
    schedulers,
    planner_config,
    train_demand_generators,
    refueling,
)
import numpy as np
import polars as pl
import polars.selectors as cs
from typing import Union, Tuple, List, Dict, Optional
from itertools import repeat
import altrios as alt
from altrios import defaults


def _select_reused_locomotives(
    loco_pool: pl.DataFrame,
    origin: str,
    loco_types: list,
    dispatch_time: float,
):
    """Pick units for one train under ``config.reuse_locomotives``.

    For each locomotive type the plan asks for (one entry per unit), in order of
    preference: units already in service and ``Ready`` at ``origin`` (earliest
    arrival first); units in service and ``Ready`` at another node (implicit
    repositioning, counted but not simulated); units never dispatched (at the
    origin first). ``Ready`` means the planner's refueling queue and servicing are
    both complete, so a battery unit is charged to its target.

    Returns ``(selected mask aligned with loco_pool, number of implicit moves)``.
    """
    from collections import Counter

    pool = loco_pool.with_row_index("__row").with_columns(
        pl.col("Locomotive_Type").cast(pl.Utf8).alias("__type"),
        (pl.col("Node").cast(pl.Utf8) == origin).alias("__local"),
        (pl.col("Times_Dispatched") > 0).alias("__used"),
    ).filter(pl.col("Status") == "Ready")
    rows, moved = [], 0
    for loco_type, n in Counter(loco_types).items():
        cands = (pool.filter(pl.col("__type") == loco_type)
                 .with_columns(
                     pl.when(pl.col("__used") & pl.col("__local")).then(0)
                     .when(pl.col("__used")).then(1)
                     .when(pl.col("__local")).then(2)
                     .otherwise(3).alias("__rank"))
                 .sort("__rank", "Arrival_Time_Hr", "Locomotive_ID")
                 .head(n))
        if cands.height < n:
            raise ValueError(
                f"Locomotive reuse: {n} {loco_type} unit(s) needed at {origin} at hour "
                f"{dispatch_time}, only {cands.height} Ready anywhere in the pool")
        moved += cands.filter(pl.col("__rank") == 1).height
        rows += cands.get_column("__row").to_list()
    selected = pl.Series(np.isin(np.arange(loco_pool.height), rows))
    return selected, moved


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
        diesel_to_require_hp = loco_pool.filter(diesel_filter).select(pl.first("HP")).item()

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
    consist_plan_in: Optional[pl.DataFrame] = None,
    loco_map: Optional[dict[str, str]] = None
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
    consist_plan_in : Optional[pl.DataFrame]
        Pre-determined train consist plan, if applicable.

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
    freight_type_to_car_type = data_prep.get_freight_to_car_mapping(rail_vehicles)           

    # Timing accumulators for the heavyweight planning steps. These are reported
    # in a summary at the end so slow phases can be identified.
    timings = {
        "demand_and_train_counts": 0.0,
        "build_locopool": 0.0,
        "build_refuelers": 0.0,
        "update_refuel_queue": 0.0,
        "make_speed_limit_train_sim": 0.0,
        "make_est_times": 0.0,
    }
    planner_start = time.perf_counter()

    if consist_plan_in is None:
        # Time the full demand-building and train-count / dispatch-scheduling phase.
        demand_start = time.perf_counter()
        # Load freight demand data and get list of nodes
        demand, node_list = data_prep.load_freight_demand(demand_file, config)

        # Set default return demand generators if none provided
        if config.return_demand_generators is None:
            config.return_demand_generators = (
                train_demand_generators.get_default_return_demand_generators()
            )

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
            if config.dispatch_scheduler is None:
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
                config.dispatch_scheduler = (
                    schedulers.dispatch_uniform_demand_uniform_departure
                )

            # Create dispatch schedule using configured scheduler
            dispatch_schedule = (config.dispatch_scheduler(
                demand, rail_vehicles, freight_type_to_car_type, config)
                .with_row_index("Train_ID", offset=1)
            )
            # Uncomment to add random jitter to departure times
            # dispatch_schedule = dispatch_schedule.with_columns(pl.col("Hour").add(pl.random.rand()))

        # Record time spent building demand levels, train counts, and the dispatch schedule.
        timings["demand_and_train_counts"] += time.perf_counter() - demand_start

        # Create locomotive pool if not provided
        if loco_pool is None:
            locopool_start = time.perf_counter()
            loco_pool = data_prep.build_locopool(
                config=config, demand_file=demand, dispatch_schedule=dispatch_schedule
            )
            timings["build_locopool"] += time.perf_counter() - locopool_start
        active_ods = demand.select(["Origin", "Destination"]).unique()
    else:
        demand = None
        node_list = (
            pl.concat(
                [consist_plan_in.get_column("Origin_ID"), consist_plan_in.get_column("Destination_ID")]
            )
            .unique()
            .sort()
        )
        active_ods = consist_plan_in.select("Origin_ID", "Destination_ID").unique().rename({"Origin_ID": "Origin", "Destination_ID": "Destination"})
        if loco_pool is None:
            loco_pool = (consist_plan_in
                .drop((cs.starts_with("Train_", "Destination_", "Departure_", "TrainSimVec_Index") | cs.ends_with("_Empty", "_Loaded", "_Hr", "_hr")) & (~cs.by_name("Departure_Time_Planned_Hr")))
                .sort("Locomotive_ID", "Departure_Time_Planned_Hr")
                .group_by(pl.all().exclude("Origin_ID", "Departure_Time_Planned_Hr"), maintain_order=True).first()
                .drop("Departure_Time_Planned_Hr")
                .unique()
                .rename({"Origin_ID": "Node"})
                .with_columns(
                    pl.col("Locomotive_ID").cast(pl.UInt32),
                    pl.lit("Ready").alias("Status"),
                )
                .pipe(data_prep.add_loco_status_columns)
                .pipe(data_prep.loco_info_to_pool, config)
            )
            if loco_map is not None:
                loco_pool = (loco_pool
                    .with_columns(pl.col("Locomotive_Type").replace(loco_map))
            )
            dispatch_schedule = (consist_plan_in
                .rename({"Origin_ID": "Origin", 
                        "Destination_ID": "Destination",
                        "Departure_Time_Planned_Hr": "Hour"})
                .drop(cs.starts_with("Refuel"))
                .group_by(~cs.starts_with("Locomotive_"))
                    .agg(pl.col("Locomotive_ID").unique().alias("Locomotive_IDs"),
                         pl.col("Locomotive_Type").cast(pl.Utf8).alias("Locomotive_Types"))
                .with_columns(pl.sum_horizontal("Cars_Loaded", "Cars_Empty").alias("Number_of_Cars"))
            )            

    # Create refuelers if not provided
    if refuelers is None:
        refuelers_start = time.perf_counter()
        refuelers = data_prep.build_refuelers(
            node_list,
            loco_pool,
            config.refueler_info,
            config.refuelers_per_incoming_corridor,
        )
        timings["build_refuelers"] += time.perf_counter() - refuelers_start

    # Load network charging guidelines if not provided
    if network_charging_guidelines is None:
        network_charging_guidelines = pl.read_csv(
            alt.resources_root() / "networks" / "network_charging_guidelines.csv"
        )

    # Apply charging guidelines to refuelers and loco_pool
    refuelers, loco_pool = data_prep.append_charging_guidelines(
        refuelers, loco_pool, active_ods, network_charging_guidelines
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

    # Accumulator for charging/refueling sessions produced during planning.
    refuel_sessions = refueling.empty_refuel_sessions()

    # Locomotive reuse bookkeeping (``config.reuse_locomotives``).
    reuse_moves = 0
    if "Times_Dispatched" not in loco_pool.columns:
        loco_pool = loco_pool.with_columns(pl.lit(0, dtype=pl.UInt32).alias("Times_Dispatched"))

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
            refuel_queue_start = time.perf_counter()
            loco_pool, refuel_sessions = refueling.update_refuel_queue(
                loco_pool, refuelers, current_time, refuel_sessions
            )
            timings["update_refuel_queue"] += time.perf_counter() - refuel_queue_start

            # Process each train scheduled to depart at current time
            for this_train in current_dispatches.iter_rows(named=True):
                # Only process trains with positive tonnage
                if this_train["Number_of_Cars"] > 0:
                    # Handle locomotive selection differently in single train mode, OR when a consist plan was provided
                    if consist_plan_in is not None and config.reuse_locomotives:
                        selected, n_moved = _select_reused_locomotives(
                            loco_pool, this_train["Origin"],
                            [loco_map.get(t, t) if loco_map else t
                             for t in this_train["Locomotive_Types"]],
                            current_time)
                        reuse_moves += n_moved
                        dispatched = loco_pool.filter(selected)
                    elif consist_plan_in is not None:
                        selected = loco_pool.select(
                            pl.col("Locomotive_ID").is_in(this_train["Locomotive_IDs"])
                        ).to_series()
                        dispatched = loco_pool.filter(selected)
                    elif config.single_train_mode:
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

                    # START SHARED STUFF HERE
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
                        train_type = this_train["Train_Type"],
                        cars_loaded = this_train["Cars_Loaded"],
                        cars_empty = this_train["Cars_Empty"],
                        rail_vehicles = rail_vehicles, 
                        freight_type_to_car_type = freight_type_to_car_type
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

                    # Cap departure SOC at the allowable battery headroom for
                    # this origin-destination pair and load class, so the pack
                    # leaves the terminal with room to absorb the regen the grade
                    # ahead will produce. Only ever LOWERS the SOC the pool
                    # tracked, so a locomotive is never dispatched with more
                    # charge than it actually accumulated.
                    n_dispatched = max(dispatched.height, 1)
                    tons_per_loco = (
                        float(this_train["Tons_Per_Train"]) / n_dispatched
                        if this_train.get("Tons_Per_Train") is not None
                        else None
                    )
                    headroom_j = data_prep.headroom_joules_for_dispatch(
                        network_charging_guidelines,
                        this_train["Origin"],
                        this_train.get("Destination"),
                        tons_per_loco,
                    )
                    # Calculate state of charge as percentage of capacity
                    loco_start_soc_pct = dispatched.select(
                        (
                            pl.max_horizontal(
                                [
                                    pl.min_horizontal(
                                        [
                                            pl.col("SOC_J"),
                                            pl.col("SOC_Max_J") - headroom_j,
                                        ]
                                    ),
                                    pl.col("SOC_Min_J"),
                                ]
                            )
                            / pl.col("Capacity_J")
                        ).alias("Departure_SOC_J")
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

                    train_weight_tons = this_train.get("Train_Weight_tons")
                    train_length_meters = this_train.get("Train_Length_meters")
                    if (train_weight_tons is not None) or (train_length_meters is not None):
                        rv_to_use = data_prep.scale_rv_mass_and_length(
                            rv_to_use,
                            n_cars_by_type,
                            locos,
                            train_weight_tons * alt.utilities.KG_PER_TON
                            if train_weight_tons is not None else None,
                            train_length_meters,
                        )

                    # An observed per-train speed ceiling, where the consist plan
                    # carries one, overrides a car model rated below it.
                    rv_to_use = data_prep.raise_rv_speed_max(
                        rv_to_use,
                        this_train.get("Train_Speed_Max_Meters_Per_Second"),
                    )

                    # Create train configuration (after any mass/length scaling of rv_to_use above)
                    train_config = alt.TrainConfig(
                        rail_vehicles=rv_to_use,
                        n_cars_by_type=n_cars_by_type,
                        train_type=train_type,
                        cd_area_vec=cd_area_vec,
                    )

                    # Set state of charge for every locomotive carrying a pack.
                    # Gated on the battery, not on Fuel_Type: a hybrid is
                    # accounted against diesel but still has a state of charge.
                    has_battery = (
                        dispatched.get_column("Has_Battery").to_list()
                        if "Has_Battery" in dispatched.columns
                        else [
                            ft == "Electricity"
                            for ft in dispatched.get_column("Fuel_Type").to_list()
                        ]
                    )
                    for i, loco in enumerate(locos):
                        if has_battery[i]:
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
                        train_id=str(train_id_counter), # TODO: use train ID and match on it in sim_manager 
                        origin_id=this_train["Origin"],
                        destination_id=this_train["Destination"],
                        train_config=train_config,
                        loco_con=loco_con,
                        init_train_state=init_train_state,
                    )

                    # Create speed-limited train simulation
                    make_slts_start = time.perf_counter()
                    slts = tsb.make_speed_limit_train_sim(
                        location_map=location_map,
                        save_interval=None,
                        simulation_days=config.simulation_days,
                        scenario_year=scenario_year,
                    )
                    timings["make_speed_limit_train_sim"] += time.perf_counter() - make_slts_start

                    # Generate estimated travel times

                    make_est_times_start = time.perf_counter()
                    (est_time_net, loco_con_out) = alt.make_est_times(
                        slts, network, config.failed_sim_logging_path
                    )
                    timings["make_est_times"] += time.perf_counter() - make_est_times_start

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
                        .otherwise(pl.col("Arrival_Time_Hr"))
                        .alias("Arrival_Time_Hr"),
                        # Set servicing completion time for dispatched locomotives
                        pl.when(selected)
                        .then(
                            pl.lit(current_time + travel_time)
                            + pl.col("Min_Servicing_Time_Hr")
                        )
                        .otherwise(pl.col("Servicing_Done_Time_Hr"))
                        .alias("Servicing_Done_Time_Hr"),
                        # Clear refueling time for dispatched locomotives
                        pl.when(selected)
                        .then(None)
                        .otherwise(pl.col("Refueling_Done_Time_Hr"))
                        .alias("Refueling_Done_Time_Hr"),
                        # Set status to dispatched for selected locomotives
                        pl.when(selected)
                        .then(pl.lit("Dispatched"))
                        .otherwise(pl.col("Status"))
                        .alias("Status"),
                        pl.when(selected)
                        .then(pl.col("Times_Dispatched") + 1)
                        .otherwise(pl.col("Times_Dispatched"))
                        .alias("Times_Dispatched"),
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
                        .alias("SOC_J")
                    )

                    # Create new rows for the train consist plan
                    new_rows = (loco_pool
                        .filter(selected)
                        .select("Locomotive_ID", "Locomotive_Type")
                        .with_columns(
                            loco_start_soc_j.alias("Departure_SOC_J"),
                            pl.lit(train_id_counter).alias("Train_ID"),
                            pl.lit(current_time).alias("Departure_Time_Planned_Hr"),
                            pl.lit(current_time + travel_time).alias("Arrival_Time_Planned_Hr")
                        )
                        .join(pl.DataFrame([this_train]), how="cross")
                        .rename({"Origin": "Origin_ID", "Destination": "Destination_ID"})
                        .select(train_consist_plan.columns)
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
            refuel_queue_start = time.perf_counter()
            loco_pool, refuel_sessions = refueling.update_refuel_queue(
                loco_pool, refuelers, current_time, refuel_sessions
            )
            timings["update_refuel_queue"] += time.perf_counter() - refuel_queue_start
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
        .sort(["Train_ID", "Locomotive_ID"], descending=False)
    )

    # Convert categorical columns in loco_pool and refuelers to strings
    loco_pool = loco_pool.with_columns(cs.categorical().cast(str))
    refuelers = refuelers.with_columns(cs.categorical().cast(str))

    # Report timing breakdown for the heavyweight planning steps so slow phases
    # can be identified.
    total_elapsed = time.perf_counter() - planner_start
    tracked = sum(timings.values())
    print("Train planner timing breakdown (seconds):")
    for label, seconds in timings.items():
        share = (seconds / total_elapsed * 100) if total_elapsed > 0 else 0.0
        print(f"    {label:<28} {seconds:9.3f} s ({share:5.1f}%)")
    print(f"    {'other/overhead':<28} {total_elapsed - tracked:9.3f} s")
    print(f"    {'total run_train_planner':<28} {total_elapsed:9.3f} s")
    if consist_plan_in is not None and config.reuse_locomotives:
        print(f"Locomotive reuse: {int((loco_pool['Times_Dispatched'] > 0).sum())} units "
              f"dispatched of {loco_pool.height} in the pool; {reuse_moves} implicit "
              f"repositioning moves (not simulated)")

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
        str(alt.resources_root() / "networks/Taconite-NoBalloon.msgpack")
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
