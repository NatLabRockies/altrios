import math
from typing import List, Union, Dict
from collections import defaultdict
import polars as pl
import polars.selectors as cs
import altrios as alt
from altrios import utilities
from altrios.train_planner import planner_config, data_prep, train_demand_generators


def calculate_waiting_time_single_dispatch(
    cumulative_demand_control: int,
    last_dispatch: int,
    demand_hourly: pl.DataFrame,
    dispatch_hour: int,
    remaining_demand_list: pl.DataFrame,
    remaining_demand_list_control: pl.DataFrame,
    search_range: int,
    od_pair_loop: str,
    min_num_cars_per_train: int,
    target_num_cars_per_train: int,
    config: planner_config.TrainPlannerConfig,
) -> tuple:
    """
    Calculate the waiting time for a single dispatch using Polars DataFrames.
    """
    # Initialize variables
    direction_demand = demand_hourly.filter(pl.col("OD_Pair") == od_pair_loop)
    remaining_demand = 0
    total_waiting_time = 0
    total_waiting_time_before_dispatch = 0
    dispatched = 0
    cumulative_demand = cumulative_demand_control
    remaining_demand_list = (
        remaining_demand_list_control.clone()
    )  # Copy to avoid modifying original
    remaining_demand_tem = []
    # Calculate cumulative demand up to the dispatch hour
    end_hour = min(dispatch_hour + 1, direction_demand.get_column("Hour").max())
    hourly_demand = direction_demand.slice(last_dispatch, end_hour - last_dispatch)
    cumulative_demand += hourly_demand["Number_of_Cars"].sum()

    if remaining_demand_list.is_empty():
        hourly_demand = hourly_demand.with_columns(
            (
                (pl.col("Number_of_Cars") * (dispatch_hour - pl.col("Hour"))).alias(
                    "Waiting_Time"
                )
            )
        )
        total_waiting_time_before_dispatch = hourly_demand["Waiting_Time"].sum()
        total_waiting_time += hourly_demand["Waiting_Time"].sum()

        # If there is remaining demand, calculate waiting time for new and remaining demand
    else:
        hourly_demand = hourly_demand.with_columns(
            (
                (pl.col("Number_of_Cars") * (dispatch_hour - pl.col("Hour"))).alias(
                    "Waiting_Time"
                )
            )
        )
        total_waiting_time_before_dispatch = hourly_demand["Waiting_Time"].sum()
        total_waiting_time += hourly_demand["Waiting_Time"].sum()

        # Calculate waiting time for each entry in the remaining demand list
        remaining_waiting_times = remaining_demand_list.with_columns(
            (pl.col("Remaining_Demand") * (dispatch_hour - pl.col("Hour"))).alias(
                "Remaining_Waiting_Time"
            )
        )
        total_waiting_time_before_dispatch += remaining_waiting_times[
            "Remaining_Waiting_Time"
        ].sum()
        total_waiting_time += remaining_waiting_times["Remaining_Waiting_Time"].sum()

    # Handle remaining demands if cumulative demand exceeds thresholds
    if cumulative_demand >= min_num_cars_per_train:
        if cumulative_demand >= target_num_cars_per_train:
            dispatched = target_num_cars_per_train
            dispatched_split = target_num_cars_per_train
            remaining_demand = (
                cumulative_demand - dispatched_split
            )  # Carry over remaining demand
            # Update remaining demand list if there's no prior remaining demand
            if remaining_demand_list.height == 0:
                # remaining_demand_tem = []
                for row in hourly_demand.iter_rows(named=True):
                    # Number_of_Containers is located at the 4th column
                    if row["Number_of_Cars"] > 0:
                        if dispatched_split >= row["Number_of_Cars"]:
                            dispatched_split -= row["Number_of_Cars"]
                        else:
                            remaining_demand_for_hour = (
                                row["Number_of_Cars"] - dispatched_split
                            )
                            # Hour is located at the 5th column
                            remaining_demand_tem.append(
                                (remaining_demand_for_hour, row["Hour"])
                            )
                            # dispatched_split stop working from this hour to the end of this loop behavior
                            dispatched_split = 0
                # Filter `remaining_demand_tem` to include only positive Remaining_Demand values
                filtered_remaining_demand_tem = [
                    (rd[0], rd[1]) for rd in remaining_demand_tem if rd[0] > 0
                ]

                # Construct the DataFrame directly from the filtered list
                remaining_demand_list = pl.DataFrame(
                    {
                        "Remaining_Demand": [
                            rd[0] for rd in filtered_remaining_demand_tem
                        ],
                        "Hour": [rd[1] for rd in filtered_remaining_demand_tem],
                    }
                )
                # remaining_demand_list = pl.DataFrame({"Remaining_Demand": [rd[0] for rd in remaining_demand_tem],"Hour": [rd[1] for rd in remaining_demand_tem]}).filter(pl.col("Remaining_Demand") > 0)
                cumulative_demand = remaining_demand
            else:
                # Prepare the cumulative transformation approach
                dispatched_split -= min(
                    dispatched_split,
                    remaining_demand_list.get_column("Remaining_Demand").sum(),
                )
                # If there is still dispatched capacity left, apply it to new demand within the range
                if dispatched_split > 0:
                    for row in hourly_demand.iter_rows(named=True):
                        if row["Number_of_Cars"] > 0:
                            if dispatched_split >= row["Number_of_Cars"]:
                                dispatched_split -= row["Number_of_Cars"]
                            else:
                                remaining_demand_for_hour = (
                                    row["Number_of_Cars"] - dispatched_split
                                )
                                remaining_demand_tem.append(
                                    (remaining_demand_for_hour, row["Hour"])
                                )
                                dispatched_split = 0
                remaining_demand_list = pl.DataFrame(
                    {
                        "Remaining_Demand": [rd[0] for rd in remaining_demand_tem],
                        "Hour": [rd[1] for rd in remaining_demand_tem],
                    }
                ).filter(pl.col("Remaining_Demand") > 0)
                cumulative_demand = remaining_demand
        else:
            dispatched = cumulative_demand
            cumulative_demand = 0  # Reset cumulative demand if all is dispatched

    # Accumulate demand if below minimum threshold
    else:
        cumulative_demand = cumulative_demand

    # Filter demand for future hours in the specified search range
    future_demand = direction_demand.filter(
        (pl.col("Hour") > dispatch_hour)
        & (
            pl.col("Hour")
            < min(
                last_dispatch + search_range, direction_demand.get_column("Hour").max()
            )
        )
    )

    # Calculate waiting time for each future hour
    future_demand = future_demand.with_columns(
        (
            (last_dispatch + search_range - 1 - pl.col("Hour"))
            * pl.col("Number_of_Cars")
        ).alias("Waiting_Time")
    )

    # Sum up all waiting times for future demand
    total_waiting_time += future_demand["Waiting_Time"].sum()

    # Calculate waiting time for remaining demand from previous hours, if any
    if not remaining_demand_list.is_empty():
        # Add waiting time for remaining demand entries
        remaining_waiting = remaining_demand_list.with_columns(
            (
                (last_dispatch + search_range - 1 - pl.col("Hour"))
                * pl.col("Remaining_Demand")
            ).alias("Remaining_Waiting_Time")
        )

        # Sum up the waiting times from remaining demand list
        total_waiting_time += remaining_waiting["Remaining_Waiting_Time"].sum()
    # Return results
    return (
        total_waiting_time_before_dispatch,
        total_waiting_time,
        remaining_demand_list,
        cumulative_demand,
        dispatched,
    )


def find_minimum_waiting_time(
    num_iterations: int,
    demand_hourly: pl.DataFrame,
    border_time_list: list,
    min_num_cars_per_train: int,
    target_num_cars_per_train: int,
    config: planner_config.TrainPlannerConfig,
) -> pl.DataFrame:
    """
    Find the minimum waiting time for dispatches using Polars DataFrame.
    """
    group_cols = ["Origin", "Destination", "OD_Pair", "Train_Type"]
    new_accumulated_carloads = get_new_accumulated_carloads(
        demand_hourly, group_cols, containers_per_car=config.containers_per_car
    ).rename({"New_Carloads": "Number_of_Cars"})
    demand_hourly = (
        demand_hourly.join(
            new_accumulated_carloads,
            how="left",
            on=["Origin", "Destination", "Train_Type", "Hour"],
        )
        .with_columns(pl.col("Number_of_Cars").fill_null(0.0))
        .drop(cs.ends_with("_right") | cs.by_name("Number_of_Containers"))
    )
    for i in range(len(border_time_list)):
        od_pair_loop = border_time_list[i][0]
        reverse_pair = "-".join(od_pair_loop.split("-")[::-1])
        directional_total_cars = new_accumulated_carloads.filter(
            pl.col("OD_Pair") == od_pair_loop
        )["Number_of_Cars"].sum()
        reverse_total_cars = new_accumulated_carloads.filter(
            pl.col("OD_Pair") == reverse_pair
        )["Number_of_Cars"].sum()
        if directional_total_cars > reverse_total_cars:
            empty_cars = directional_total_cars - reverse_total_cars
            empty_cars_o_d = reverse_pair
        else:
            empty_cars = reverse_total_cars - directional_total_cars
            empty_cars_o_d = od_pair_loop
        print(f"total cars for {od_pair_loop} is {directional_total_cars}")
        print(f"reverse_total_cars for {reverse_pair} is {reverse_total_cars}")
        # print(f"empty_containers for {empty_cars_o_d} is {empty_containers}")
    print(f"empty_containers for {empty_cars_o_d} is {empty_cars}")
    print(f"There are {len(border_time_list[0]) - 1} trains to dispatch")
    final_dispatch_rows = []
    for j in range(len(border_time_list)):
        start_hour = 0
        total_dispatched = 0
        dispatch_time = []
        waiting_time_total = 0
        waiting_time_total_before_dispatch = 0
        cumulative_demand_control = 0
        last_dispatch = 0
        remaining_demand_list_control = pl.DataFrame(
            {
                "Remaining_Demand": pl.Series([], dtype=pl.Int64),
                "Hour": pl.Series([], dtype=pl.Float64),
            }
        )
        dispatched_list = []
        # print(f"border_time_list[j][0] is {border_time_list[j][0]}")
        od_pair_loop = border_time_list[j][0]
        # print(f"od_pair_loop is {od_pair_loop}")
        origin, destination = border_time_list[j][0].split("-")
        # print(f"origin is {origin}")
        # print(f"destination is {destination}")
        # print(demand)
        total_cars = new_accumulated_carloads.filter(pl.col("OD_Pair") == od_pair_loop)[
            "Number_of_Cars"
        ].sum()
        for i in range(2, num_iterations):
            this_demand_hourly = demand_hourly
            if total_cars - total_dispatched == 0:
                dispatched_list.append(0.0)
            search_range = border_time_list[j][i] - max(0, start_hour - 1)
            # DataFrame to accumulate dispatch hour info
            total_waiting_time_demand_list = pl.DataFrame(
                {
                    "Dispatch_Hour": pl.Series([], dtype=pl.Int64),
                    "Waiting_Before_Dispatch": pl.Series([], dtype=pl.Float64),
                    "Total_Waiting": pl.Series([], dtype=pl.Float64),
                    "Remaining_Demand_List": pl.Series([], dtype=pl.Object),
                    "Cumulative_Demand": pl.Series([], dtype=pl.Float64),
                    "Dispatched": pl.Series([], dtype=pl.Float64),
                }
            )
            x = 5
            for dispatch_hour in range(
                max(0, start_hour - 1), start_hour + search_range
            ):
                (
                    total_waiting_time_before_dispatch,
                    total_waiting_time,
                    remaining_demand_list,
                    cumulative_demand,
                    dispatched,
                ) = calculate_waiting_time_single_dispatch(
                    cumulative_demand_control,
                    last_dispatch,
                    this_demand_hourly,
                    dispatch_hour,
                    remaining_demand_list_control,
                    remaining_demand_list_control.clone(),
                    search_range,
                    od_pair_loop,
                    min_num_cars_per_train,
                    target_num_cars_per_train,
                    config,
                )
                # Append data for each dispatch hour, ensuring consistent types
                new_row = pl.DataFrame(
                    {
                        "Dispatch_Hour": [dispatch_hour],
                        "Waiting_Before_Dispatch": [
                            float(total_waiting_time_before_dispatch)
                        ],
                        "Total_Waiting": [float(total_waiting_time)],
                        "Remaining_Demand_List": [remaining_demand_list],
                        "Cumulative_Demand": [float(cumulative_demand)],
                        "Dispatched": [float(dispatched)],
                    }
                )
                total_waiting_time_demand_list = total_waiting_time_demand_list.vstack(
                    new_row
                )

            # Find the row with the minimum "Total_Waiting"
            min_waiting_row = total_waiting_time_demand_list.sort("Total_Waiting").head(
                1
            )
            min_waiting_time_hour = min_waiting_row[0, "Dispatch_Hour"]
            min_waiting_time_before_dispatch = min_waiting_row[
                0, "Waiting_Before_Dispatch"
            ]
            min_waiting_time = min_waiting_row[0, "Total_Waiting"]
            remaining_demand_list = min_waiting_row[0, "Remaining_Demand_List"]
            cumulative_demand = min_waiting_row[0, "Cumulative_Demand"]
            dispatched = min_waiting_row[0, "Dispatched"]

            # Track dispatched containers for each dispatch hour
            dispatched_list.append(dispatched)
            # Reset remaining demand if cumulative demand is zero
            if cumulative_demand == 0:
                remaining_demand_list = pl.DataFrame(schema=["Remaining_Demand_List"])

            # Update control values for the next iteration
            remaining_demand_list_control = remaining_demand_list
            cumulative_demand_control = cumulative_demand
            if min_waiting_time_hour == 503:
                x = 5
            last_dispatch = min_waiting_time_hour + 1
            start_hour = min_waiting_time_hour + 1
            total_dispatched += dispatched
            waiting_time_total += min_waiting_time

            # Accumulate total waiting time for before dispatch and overall
            if i == num_iterations - 1:
                waiting_time_total_before_dispatch += min_waiting_time
            else:
                waiting_time_total_before_dispatch += min_waiting_time_before_dispatch

            # Add the dispatch hour to the list
            dispatch_time.append(min_waiting_time_hour)

        remaining_to_dispatch = total_cars - total_dispatched
        final_waiting_time = remaining_to_dispatch * (
            demand_hourly.get_column("Hour").max() + 1 - start_hour
        )
        waiting_time_total += final_waiting_time
        dispatch_time.append(
            demand_hourly.get_column("Hour").max()
        )  # Assuming final dispatch at the end of the period
        dispatched_list.append(remaining_to_dispatch)
        dispatch_df_row = []
        for i in range(len(dispatched_list)):
            dispatch_df_row.append(
                {
                    "Origin": origin,
                    "Destination": destination,
                    "Train_Type": "Intermodal",
                    "Cars_Per_Train_Loaded": dispatched_list[i],
                    "Cars_Per_Train_Empty": 0.0,
                    "Target_Cars_Per_Train": float(target_num_cars_per_train),
                    "Number_of_Cars_Total": dispatched_list[i],
                    "Hour": float(dispatch_time[i]),
                }
            )
        if od_pair_loop == empty_cars_o_d:
            dispatch_df_row = []
            for i in range(len(dispatched_list)):
                dispatch_df_row.append(
                    {
                        "Origin": origin,
                        "Destination": destination,
                        "Train_Type": "Intermodal",
                        "Cars_Per_Train_Loaded": dispatched_list[i],
                        "Cars_Per_Train_Empty": 0.0,
                        "Target_Cars_Per_Train": float(target_num_cars_per_train),
                        "Number_of_Cars_Total": dispatched_list[i],
                        "Hour": float(dispatch_time[i]),
                    }
                )
        # print(f"dispatch_df_row is {dispatch_df_row}")
        final_dispatch_rows.extend(dispatch_df_row)
    # print(f"final_dispatch_rows is {final_dispatch_rows}")
    dispatch_times = pl.DataFrame(final_dispatch_rows)
    dispatch_times = dispatch_times.sort("Hour")
    return dispatch_times


def formatScheduleColumns(
    df: Union[pl.DataFrame, pl.LazyFrame], config: planner_config.TrainPlannerConfig
) -> Union[pl.DataFrame, pl.LazyFrame]:
    return (
        df.with_columns(
            (
                pl.col("Tons_Per_Car_Loaded").mul("Number_of_Cars_Loaded")
                + pl.col("Tons_Per_Car_Empty").mul("Number_of_Cars_Empty")
            ).alias("Tons_Per_Train"),
            (
                pl.col("HP_Required_Per_Ton_Loaded")
                .mul("Tons_Per_Car_Loaded")
                .mul("Number_of_Cars_Loaded")
                + pl.col("HP_Required_Per_Ton_Empty")
                .mul("Tons_Per_Car_Empty")
                .mul("Number_of_Cars_Empty")
            ).alias("HP_Required"),
            pl.when(pl.col("Train_Type").str.contains("Intermodal"))
            .then(pl.col("Number_of_Cars_Loaded").mul(config.containers_per_car))
            .otherwise(0)
            .alias("Containers_Loaded"),
            pl.when(pl.col("Train_Type").str.contains("Intermodal"))
            .then(pl.col("Number_of_Cars_Empty").mul(config.containers_per_car))
            .otherwise(0)
            .alias("Containers_Empty"),
        )
        .select(
            "Hour",
            "Origin",
            "Destination",
            "Train_Type",
            "Number_of_Cars",
            "Number_of_Cars_Loaded",
            "Number_of_Cars_Empty",
            "Containers_Empty",
            "Containers_Loaded",
            "Tons_Per_Train",
            "HP_Required",
        )
        .rename(
            {
                "Number_of_Cars_Loaded": "Cars_Loaded",
                "Number_of_Cars_Empty": "Cars_Empty",
            }
        )
        .sort(["Hour", "Origin", "Destination", "Train_Type"])
    )


def get_new_accumulated_carloads(
    demand_hourly: Union[pl.DataFrame, pl.LazyFrame],
    group_cols: List[str],
    containers_per_car: int,
    filter_zero: bool = True,
) -> Union[pl.DataFrame, pl.LazyFrame]:
    if group_cols is None:
        df = (
            demand_hourly.sort("Hour")
            .with_columns(
                pl.col("Number_of_Containers")
                .cum_sum()
                .floordiv(containers_per_car)
                .alias("Accumulated_Carloads")
            )
            .with_columns(
                (
                    pl.col("Accumulated_Carloads")
                    - pl.col("Accumulated_Carloads").shift(1).fill_null(0.0)
                ).alias("New_Carloads")
            )
            .select("Hour", "New_Carloads")
        )
    else:
        df = (
            demand_hourly.sort(group_cols + ["Hour"])
            .with_columns(
                pl.col("Number_of_Containers")
                .cum_sum()
                .over(group_cols)
                .floordiv(containers_per_car)
                .alias("Accumulated_Carloads")
            )
            .with_columns(
                (
                    pl.col("Accumulated_Carloads")
                    - pl.col("Accumulated_Carloads")
                    .shift(1)
                    .over(group_cols)
                    .fill_null(0.0)
                ).alias("New_Carloads")
            )
            .select(group_cols + ["Hour", "New_Carloads"])
        )
    if filter_zero:
        df = df.filter(pl.col("New_Carloads") > 0)
    return df


def calculate_dispatch_data(
    total_containers,
    target_num_cars_per_train,
    od_pair,
    demand_hourly,
    max_min_trains,
    containers_per_car,
):
    remaining_containers = total_containers % (
        target_num_cars_per_train * containers_per_car
    )
    num_trains = total_containers // (
        target_num_cars_per_train * containers_per_car
    ) + (1 if remaining_containers > 0 else 0)
    num_trains = int(max(num_trains, max_min_trains))

    new_accumulated_carloads = get_new_accumulated_carloads(
        demand_hourly, group_cols=None, containers_per_car=containers_per_car
    )

    planned_train_lengths = (
        pl.DataFrame(
            {
                "Group": [1]
                * num_trains,  # Unused, just needed for allocateIntegerEvenly to work,
                "Train_ID": list(range(num_trains)),
                "Cars": [total_containers / containers_per_car] * num_trains,
            }
        )
        # Divide containers into trains as evenly as possible
        .pipe(utilities.allocateIntegerEvenly, target="Cars", grouping_vars=["Group"])
        .sort("Train_ID")
        .get_column("Cars")
        .to_list()
    )

    train_dispatch_times = []
    dispatched_train_lengths = []
    accumulated_demand = 0
    p = 0
    for arrival in new_accumulated_carloads.iter_rows(named=True):
        accumulated_demand += arrival["New_Carloads"]
        if p >= len(planned_train_lengths):
            break
        if p == len(planned_train_lengths) - 1:
            train_dispatch_times.append(
                new_accumulated_carloads.get_column("Hour").max()
            )
            break
        while accumulated_demand >= planned_train_lengths[p]:
            train_dispatch_times.append(arrival["Hour"])
            dispatched_train_lengths.append(planned_train_lengths[p])
            total_containers -= planned_train_lengths[p]
            accumulated_demand -= planned_train_lengths[p]
            p += 1
            if p >= len(planned_train_lengths):
                break

    if len(train_dispatch_times) < len(planned_train_lengths):
        for _ in range(len(planned_train_lengths) - len(train_dispatch_times)):
            train_dispatch_times.append(train_dispatch_times[-1] + _ + 1)
    dispatched_train_lengths.append(total_containers / containers_per_car)
    if len(dispatched_train_lengths) < len(planned_train_lengths):
        for _ in range(len(planned_train_lengths) - len(dispatched_train_lengths)):
            dispatched_train_lengths.append(0)

    return pl.DataFrame(
        {
            "Dispatch_Time": train_dispatch_times,
            "Number_of_Cars_Planned": planned_train_lengths,
            "Number_of_Cars_Dispatched": dispatched_train_lengths,
        }
    ).with_columns(
        (pl.col("Number_of_Cars_Planned") * containers_per_car).alias(
            "Number_of_Containers_Planned"
        ),
        (pl.col("Number_of_Cars_Dispatched") * containers_per_car).alias(
            "Number_of_Containers_Dispatched"
        ),
    )


# Define the main function to generate demand trains with the updated rule
def generate_trains_deterministic_hourly(
    demand_hourly: pl.DataFrame, target_num_cars_per_train: int, containers_per_car: int
) -> pl.DataFrame:
    grouped_data = (
        demand_hourly.group_by("Origin", "Destination", "OD_Pair")
        .agg(
            pl.col("Number_of_Containers").sum().alias("Total_Containers"),
            pl.col("Number_of_Containers")
            .sum()
            .mod(target_num_cars_per_train * containers_per_car)
            .alias("Remaining_Containers"),
        )
        .with_columns(
            pl.col("Total_Containers")
            .floordiv(target_num_cars_per_train * containers_per_car)
            .add(pl.col("Remaining_Containers").gt(0))
            .alias("Min_Trains")
        )
        .with_columns(
            pl.col("Min_Trains")
            .max()
            .over(
                pl.concat_str(
                    pl.min_horizontal(pl.col("Origin", "Destination")),
                    pl.lit("_"),
                    pl.max_horizontal(pl.col("Origin", "Destination")),
                )
            )
            .alias("Max_Min_Trains")
        )
    )

    # Prepare a list to collect the results for all OD pairs
    all_dispatch_data = []
    # Step 4: Loop through each unique OD pair to calculate dispatch data
    for row in grouped_data.iter_rows(named=True):
        # Calculate dispatch data for the current OD pair with the updated rule
        dispatch_data = calculate_dispatch_data(
            total_containers=row["Total_Containers"],
            target_num_cars_per_train=target_num_cars_per_train,
            od_pair=row["OD_Pair"],
            demand_hourly=demand_hourly.filter(
                pl.col("OD_Pair") == row["OD_Pair"]
            ).sort("Hour"),
            max_min_trains=row["Max_Min_Trains"],
            containers_per_car=containers_per_car,
        )
        # Append the result to the list
        all_dispatch_data.append(
            dispatch_data.with_columns(
                pl.lit(row["Origin"]).alias("Origin"),
                pl.lit(row["Destination"]).alias("Destination"),
                pl.lit(row["OD_Pair"]).alias("OD_Pair"),
            )
        )

    return pl.concat(all_dispatch_data, how="diagonal_relaxed")


def dispatch_hourly_demand_optimized_departure(
    demand_hourly: pl.DataFrame,
    rail_vehicles: List[alt.RailVehicle],
    freight_type_to_car_type: Dict[str, str],
    config: planner_config.TrainPlannerConfig,
) -> pl.DataFrame:
    """
    Converts a table of demand into a dispatch plan where trains depart from each origin in uniformly spaced intervals.
    Arguments:
    ----------
    demand: `DataFrame` or `LazyFrame` representing origin-destination demands (number of trains).
    rail_vehicles: List of `altrios.RailVehicle` objects.
    config: `TrainPlannerConfig` object.
    Outputs:
    ----------
    Updated demand `DataFrame` or `LazyFrame` representing dispatches, each defined with an origin, destination, train type, number of (loaded and empty) cars, tonnage, and HP per ton requirement.
    """
    min_num_cars_per_train = config.min_cars_per_train[
        "Intermodal_Loaded"
    ]  # TODO make this flexible
    target_num_cars_per_train = config.target_cars_per_train[
        "Intermodal_Loaded"
    ]  # TODO make this flexible
    demand_hourly = demand_hourly.with_columns(
        (pl.col("Origin") + "-" + pl.col("Destination")).alias("OD_Pair")
    )
    dispatch_df = generate_trains_deterministic_hourly(
        demand_hourly, target_num_cars_per_train, config.containers_per_car
    )
    od_dispatch_times = []
    dispatch_times = dispatch_df["Dispatch_Time"].to_list()
    od_pair_list = dispatch_df["OD_Pair"].to_list()
    for i in range(len(od_pair_list)):
        od_border_list_sub = []
        od_border_list_sub.append(od_pair_list[i])
        od_border_list_sub.append(dispatch_times[i])
        od_dispatch_times.append(od_border_list_sub)
    grouped_data = defaultdict(list)
    for od_pair, value in od_dispatch_times:
        grouped_data[od_pair].append(value)
    border_time_list = [[key] + values for key, values in grouped_data.items()]
    num_iterations = len(border_time_list[0])

    schedule = find_minimum_waiting_time(
        num_iterations=num_iterations,
        demand_hourly=demand_hourly,
        border_time_list=border_time_list,
        min_num_cars_per_train=min_num_cars_per_train,
        target_num_cars_per_train=target_num_cars_per_train,
        config=config,
    )
    return (
        schedule
        # TODO: this doesn't handle tons correctly for train type empty
        .pipe(
            data_prep.appendTonsAndHP, rail_vehicles, freight_type_to_car_type, config
        )
        .rename(
            {
                "Cars_Per_Train_Loaded": "Cars_Loaded",
                "Cars_Per_Train_Empty": "Cars_Empty",
            }
        )
        .with_columns(
            (pl.col("Cars_Loaded") + pl.col("Cars_Empty")).alias("Number_of_Cars"),
            pl.col("Tons_Per_Car").mul("Cars_Loaded").alias("Tons_Per_Train"),
            pl.col("Tons_Per_Car")
            .mul("Cars_Loaded")
            .mul("HP_Required_Per_Ton")
            .alias("HP_Required"),
            pl.when(pl.col("Train_Type").str.contains("Intermodal"))
            .then(pl.col("Cars_Loaded").mul(config.containers_per_car))
            .otherwise(0)
            .alias("Containers_Loaded"),
            pl.when(pl.col("Train_Type").str.contains("Intermodal"))
            .then(pl.col("Cars_Empty").mul(config.containers_per_car))
            .otherwise(0)
            .alias("Containers_Empty"),
        )
        .select(
            "Hour",
            "Origin",
            "Destination",
            "Train_Type",
            "Number_of_Cars",
            "Cars_Loaded",
            "Cars_Empty",
            "Containers_Empty",
            "Containers_Loaded",
            "Tons_Per_Train",
            "HP_Required",
        )
        .sort(["Hour", "Origin", "Destination", "Train_Type"])
    )


def dispatch_uniform_demand_uniform_departure(
    demand: pl.DataFrame,
    rail_vehicles: List[alt.RailVehicle],
    freight_type_to_car_type: Dict[str, str],
    config: planner_config.TrainPlannerConfig,
) -> pl.DataFrame:
    """
    Generate a tabulated demand pair to indicate the expected dispatching interval
    and actual dispatching timesteps after rounding, with departures from each terminal
    spaced as evenly as possible
    Arguments:
    ----------
    demand: `DataFrame` or `LazyFrame` representing origin-destination demands (number of trains).
    rail_vehicles: List of `altrios.RailVehicle` objects.
    config: `TrainPlannerConfig` object.
    Outputs:
    ----------
    schedule: Tabulated dispatching time for each demand pair for each train type
    in hours
    """
    hours = config.simulation_days * 24
    grouping_vars = ["Origin", "Destination", "Train_Type"]
    return (
        demand.select(
            pl.exclude("Number_of_Trains").repeat_by("Number_of_Trains").explode()
        )
        .pipe(
            utilities.allocateIntegerEvenly,
            target="Number_of_Cars_Loaded",
            grouping_vars=grouping_vars,
        )
        .drop("Percent_Within_Group_Cumulative")
        .pipe(
            utilities.allocateIntegerEvenly,
            target="Number_of_Cars_Empty",
            grouping_vars=grouping_vars,
        )
        .drop("Percent_Within_Group_Cumulative")
        .group_by(pl.exclude("Number_of_Cars_Empty", "Number_of_Cars_Loaded"))
        .agg(pl.col("Number_of_Cars_Empty", "Number_of_Cars_Loaded"))
        .with_columns(
            pl.col("Number_of_Cars_Loaded").list.sort(descending=True),
            pl.col("Number_of_Cars_Empty").list.sort(descending=False),
        )
        .explode("Number_of_Cars_Empty", "Number_of_Cars_Loaded")
        .with_columns(
            (pl.col("Number_of_Cars_Empty") + pl.col("Number_of_Cars_Loaded")).alias(
                "Number_of_Cars"
            )
        )
        # TODO: space out trains with slightly more vs less demand, instead of ordering randomly
        .with_columns(pl.int_range(pl.len()).shuffle().alias("random_int"))
        .sort("Origin", "Destination", "Train_Type", "random_int")
        .drop("random_int")
        .with_columns(
            (hours * 1.0 / pl.len().over("Origin", "Destination")).alias("Interval")
        )
        .with_columns(
            (
                (pl.col("Interval").cum_count().over(["Origin", "Destination"]))
                * pl.col("Interval")
            ).alias("Hour")
        )
        .pipe(formatScheduleColumns, config=config)
    )


def dispatch_hourly_demand_uniform_departure(
    demand_hourly: pl.DataFrame,
    rail_vehicles: List[alt.RailVehicle],
    freight_type_to_car_type: Dict[str, str],
    config: planner_config.TrainPlannerConfig,
) -> pl.DataFrame:
    if "Number_of_Containers" in demand_hourly.collect_schema():
        demand_aggregate = (
            demand_hourly.group_by(
                "Origin", "Destination", "Number_of_Days", "Train_Type"
            )
            .agg(pl.col("Number_of_Containers").sum())
            .with_columns(
                pl.col("Number_of_Containers")
                .truediv(config.containers_per_car)
                .ceil()
                .alias("Number_of_Cars")
            )
        )
    else:
        demand_aggregate = demand_hourly.group_by(
            "Origin", "Destination", "Number_of_Days", "Train_Type"
        ).agg(pl.col("Number_of_Cars").sum())

    demand_returns = train_demand_generators.generate_return_demand(
        demand_aggregate, config
    )
    demand_rebalancing = pl.DataFrame()
    if (
        demand_aggregate.filter(pl.col("Train_Type").str.contains("Manifest")).height
        > 0
    ):
        nodes = (
            pl.concat(
                [
                    demand_aggregate.get_column("Origin"),
                    demand_aggregate.get_column("Destination"),
                ]
            )
            .unique()
            .sort()
        )
        demand_rebalancing = (
            train_demand_generators.generate_manifest_rebalancing_demand(
                demand_aggregate, nodes, config
            )
        )

    demand = train_demand_generators.generate_demand_trains(
        demand_aggregate,
        demand_returns,
        demand_rebalancing,
        rail_vehicles,
        freight_type_to_car_type,
        config,
    )

    departure_schedule = dispatch_uniform_demand_uniform_departure(
        demand, rail_vehicles, freight_type_to_car_type, config
    ).select("Hour", "Origin", "Destination", "Train_Type")
    new_carloads = (
        get_new_accumulated_carloads(
            demand_hourly,
            group_cols=["Origin", "Destination", "Train_Type"],
            containers_per_car=config.containers_per_car,
            filter_zero=False,
        )
        .with_columns(
            pl.col("New_Carloads")
            .cum_sum()
            .over("Origin", "Destination", "Train_Type")
            .alias("Cumulative_Carloads")
        )
        .drop("New_Carloads")
        .sort("Origin", "Destination", "Train_Type", "Hour")
    )
    departure_schedule = (
        departure_schedule.sort("Origin", "Destination", "Train_Type", "Hour")
        .join_asof(
            new_carloads.with_columns(pl.col("Hour").cast(pl.Float64)),
            by=["Origin", "Destination", "Train_Type"],
            on="Hour",
            strategy="backward",
        )
        .sort("Origin", "Destination", "Train_Type", "Hour")
        .with_columns(
            (
                pl.col("Cumulative_Carloads")
                - pl.col("Cumulative_Carloads")
                .shift(1)
                .over("Origin", "Destination", "Train_Type")
                .fill_null(0.0)
            ).alias("New_Cumulative_Carloads")
        )
    )

    od_departures_revised = []
    max_train_length = math.floor(
        config.target_cars_per_train["Intermodal_Loaded"] * 1.1
    )
    min_train_length = config.min_cars_per_train["Intermodal_Loaded"]
    for od_departures in departure_schedule.partition_by(
        ["Origin", "Destination", "Train_Type"], maintain_order=True
    ):
        train_lengths = od_departures.get_column("New_Cumulative_Carloads").to_list()
        for i in range(len(train_lengths)):
            if (train_lengths[i] > max_train_length) and (i < len(train_lengths) - 1):
                train_lengths[i + 1] += train_lengths[i] - max_train_length
                train_lengths[i] = max_train_length
            elif (train_lengths[i] < min_train_length) and (i < len(train_lengths) - 1):
                train_lengths[i + 1] += train_lengths[i]
                train_lengths[i] = 0

        if train_lengths[len(train_lengths) - 1] > max_train_length:
            print(
                f"Unsupported case: final train too long ({train_lengths[len(train_lengths) - 1]} cars)"
            )
        elif train_lengths[len(train_lengths) - 1] < min_train_length:
            if (
                train_lengths[len(train_lengths) - 1]
                + train_lengths[len(train_lengths) - 2]
                <= max_train_length
            ):
                train_lengths[len(train_lengths) - 2] += train_lengths[
                    len(train_lengths) - 1
                ]
                train_lengths[len(train_lengths) - 1] = 0
            else:
                new_val = (
                    train_lengths[len(train_lengths) - 2]
                    + train_lengths[len(train_lengths) - 1]
                ) / 2
                train_lengths[len(train_lengths) - 2] = math.ceil(new_val)
                train_lengths[len(train_lengths) - 1] = math.floor(new_val)

        od_departures = od_departures.with_columns(
            pl.Series("Carloads", train_lengths, strict=False)
        )
        od_departures_revised.append(od_departures)

    departure_schedule = (
        pl.concat(od_departures_revised, how="diagonal_relaxed")
        .rename({"Carloads": "Cars_Per_Train_Loaded"})
        .filter(pl.col("Cars_Per_Train_Loaded") > 0)
        .select("Hour", "Origin", "Destination", "Train_Type", "Cars_Per_Train_Loaded")
        .with_columns(pl.lit(0).alias("Cars_Per_Train_Empty"))
        .pipe(
            data_prep.appendTonsAndHP, rail_vehicles, freight_type_to_car_type, config
        )
        .rename(
            {
                "Cars_Per_Train_Loaded": "Cars_Loaded",
                "Cars_Per_Train_Empty": "Cars_Empty",
            }
        )
        .with_columns(
            (pl.col("Cars_Loaded") + pl.col("Cars_Empty")).alias("Number_of_Cars"),
            pl.col("Tons_Per_Car").mul("Cars_Loaded").alias("Tons_Per_Train"),
            pl.col("Tons_Per_Car")
            .mul("Cars_Loaded")
            .mul("HP_Required_Per_Ton")
            .alias("HP_Required"),
            pl.when(pl.col("Train_Type").str.contains("Intermodal"))
            .then(pl.col("Cars_Loaded").mul(config.containers_per_car))
            .otherwise(0)
            .alias("Containers_Loaded"),
            pl.when(pl.col("Train_Type").str.contains("Intermodal"))
            .then(pl.col("Cars_Empty").mul(config.containers_per_car))
            .otherwise(0)
            .alias("Containers_Empty"),
        )
        .select(
            "Hour",
            "Origin",
            "Destination",
            "Train_Type",
            "Number_of_Cars",
            "Cars_Loaded",
            "Cars_Empty",
            "Containers_Empty",
            "Containers_Loaded",
            "Tons_Per_Train",
            "HP_Required",
        )
        .sort(["Hour", "Origin", "Destination", "Train_Type"])
    )
    return departure_schedule
