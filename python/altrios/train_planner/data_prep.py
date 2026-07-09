from typing import Union, List, Tuple, Dict
from pathlib import Path
import polars as pl
import polars.selectors as cs
import pandas as pd
import numpy as np
import math
from scipy.stats import rankdata
import altrios as alt
from altrios import defaults, utilities
from altrios.train_planner import planner_config

day_order_map = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}


def convert_demand_to_sim_days(
    demand_table: Union[pl.DataFrame, pl.LazyFrame], simulation_days: int
) -> Union[pl.DataFrame, pl.LazyFrame]:
    if "Number_of_Days" in demand_table.collect_schema():
        return demand_table.with_columns(
            cs.starts_with("Number_of_").truediv(pl.col("Number_of_Days").truediv(simulation_days))
        )

    else:
        print(
            "`Number_of_Days` not specified in demand file. Assuming demand in the file is expressed per week."
        )
        return demand_table.with_columns(cs.starts_with("Number_of_").mul(simulation_days / 7.0))


def load_freight_demand(
    demand_table: Union[pl.DataFrame, pl.LazyFrame, Path, str],
    config: planner_config.TrainPlannerConfig,
) -> Tuple[pl.DataFrame, pl.Series, int]:
    """
    Load the user input csv file into a dataframe for later processing
    Arguments:
    ----------
    user_input_file: path to the input csv file that user import to the module
    Example Input:
        Origin	Destination	Train_Type	Number_of_Cars	Number_of_Containers
        Barstow	Stockton	Unit	    2394	        0
        Barstow	Stockton	Manifest	2588	        0
        Barstow	Stockton	Intermodal	2221	        2221

    Outputs:
    ----------
    df_annual_demand: dataframe with all pair information including:
    origin, destination, train type, number of cars
    node_list: List of origin or destination demand nodes
    """
    if isinstance(demand_table, (Path, str)):
        demand_table = pl.read_csv(demand_table).pipe(
            convert_demand_to_sim_days, simulation_days=config.simulation_days
        )
    elif "Hour" not in demand_table.collect_schema():
        demand_table = demand_table.pipe(
            convert_demand_to_sim_days, simulation_days=config.simulation_days
        )

    nodes = (
        pl.concat([demand_table.get_column("Origin"), demand_table.get_column("Destination")])
        .unique()
        .sort()
    )
    return demand_table, nodes


def prep_hourly_demand(
    total_demand: Union[pl.DataFrame, pl.LazyFrame],
    hourly_demand_density: Union[pl.DataFrame, pl.LazyFrame],
    daily_demand_density: Union[pl.DataFrame, pl.LazyFrame],
    simulation_weeks=1,
) -> Union[pl.DataFrame, pl.LazyFrame]:
    if "Number_of_Containers" in total_demand.collect_schema():
        demand_col = "Number_of_Containers"
    else:
        demand_col = "Number_of_Cars"

    total_demand = total_demand.pipe(
        convert_demand_to_sim_days, simulation_days=simulation_weeks * 7
    )

    hourly_demand_density = (
        hourly_demand_density.group_by("Terminal_Type", "Hour_Of_Day")
        .agg(pl.col("Share").sum())
        .with_columns(pl.col("Share").truediv(pl.col("Share").sum().over("Terminal_Type")))
    )
    daily_demand_density = (
        daily_demand_density.group_by("Terminal_Type", "Day_Of_Week")
        .agg(pl.col("Share").sum())
        .with_columns(pl.col("Share").truediv(pl.col("Share").sum().over("Terminal_Type")))
    )
    one_week = (
        total_demand.join(daily_demand_density, how="inner", on=["Terminal_Type"])
        .with_columns(
            (pl.col(demand_col) * pl.col("Share")).alias(f"{demand_col}_Daily"),
            pl.col("Day_Of_Week").replace_strict(day_order_map).alias("Day_Order"),
        )
        .pipe(
            utilities.allocateItems,
            grouping_vars=["Origin", "Destination", "Train_Type"],
            count_target=f"{demand_col}_Daily",
        )
        .drop(f"{demand_col}_Daily", "Share")
        .rename({"Count": f"{demand_col}_Daily"})
        .join(hourly_demand_density, how="inner", on=["Terminal_Type"])
        .sort("Origin", "Destination", "Day_Order", "Hour_Of_Day")
        .with_columns(
            (pl.col(f"{demand_col}_Daily") * pl.col("Share")).alias(demand_col),
            pl.concat_str(pl.col("Origin"), pl.lit("-"), pl.col("Destination")).alias("OD_Pair"),
            pl.int_range(0, pl.len()).over("Origin", "Destination").alias("Hour"),
        )
        .pipe(
            utilities.allocateItems,
            grouping_vars=["Origin", "Destination", "Train_Type", "Day_Order"],
            count_target=demand_col,
        )
        .drop(demand_col)
        .rename({"Count": demand_col})
        .select("Origin", "Destination", "Train_Type", "Hour", "Number_of_Days", demand_col)
    )
    return (
        pl.concat(
            [
                one_week,
                one_week.with_columns(pl.col("Hour").add(24 * 7)),
                one_week.with_columns(pl.col("Hour").add(24 * 7 * 2)),
            ]
        )
        .with_columns(pl.col("Number_of_Days").mul(3))
        .sort("Origin", "Destination", "Train_Type", "Hour")
    )


def append_loco_info(loco_info: pd.DataFrame) -> pd.DataFrame:
    if all(
        item in loco_info.columns
        for item in [
            "HP",
            "Loco_Mass_Tons",
            "SOC_J",
            "SOC_Min_J",
            "SOC_Max_J",
            "Capacity_J",
        ]
    ):
        return loco_info

    def get_hp(loco: alt.Locomotive):
        return loco.pwr_rated_kilowatts * 1e3 / alt.utils.W_PER_HP

    def get_mass_ton(loco: alt.Locomotive):
        loco_dict = loco.to_pydict()
        if loco_dict["mass_kilograms"] is None:
            return 0
        else:
            return loco_dict["mass_kilograms"] / alt.utils.KG_PER_TON

    def get_starting_soc(loco: alt.Locomotive):
        loco_dict = loco.to_pydict()
        loco_type = next(iter(loco_dict["loco_type"].values()))
        if "res" not in loco_type.keys():
            return defaults.DIESEL_TANK_CAPACITY_J
        else:
            return loco_type["res"]["state"]["soc"] * loco_type["res"]["energy_capacity_joules"]

    def get_min_soc(loco: alt.Locomotive):
        loco_dict = loco.to_pydict()
        loco_type = next(iter(loco_dict["loco_type"].values()))
        if "res" not in loco_type.keys():
            return 0
        else:
            return loco_type["res"]["min_soc"] * loco_type["res"]["energy_capacity_joules"]

    def get_max_soc(loco: alt.Locomotive):
        loco_dict = loco.to_pydict()
        loco_type = next(iter(loco_dict["loco_type"].values()))
        if "res" not in loco_type.keys():
            return defaults.DIESEL_TANK_CAPACITY_J
        else:
            return loco_type["res"]["max_soc"] * loco_type["res"]["energy_capacity_joules"]

    def get_capacity(loco: alt.Locomotive):
        loco_dict = loco.to_pydict()
        loco_type = next(iter(loco_dict["loco_type"].values()))
        if "res" not in loco_type.keys():
            return defaults.DIESEL_TANK_CAPACITY_J
        else:
            return loco_type["res"]["energy_capacity_joules"]

    loco_info.loc[:, "HP"] = loco_info.loc[:, "Rust_Loco"].apply(get_hp)
    loco_info.loc[:, "Loco_Mass_Tons"] = loco_info.loc[:, "Rust_Loco"].apply(get_mass_ton)
    loco_info.loc[:, "SOC_J"] = loco_info.loc[:, "Rust_Loco"].apply(get_starting_soc)
    loco_info.loc[:, "SOC_Min_J"] = loco_info.loc[:, "Rust_Loco"].apply(get_min_soc)
    loco_info.loc[:, "SOC_Max_J"] = loco_info.loc[:, "Rust_Loco"].apply(get_max_soc)
    loco_info.loc[:, "Capacity_J"] = loco_info.loc[:, "Rust_Loco"].apply(get_capacity)
    return loco_info


def build_locopool(
    config: planner_config.TrainPlannerConfig,
    demand_file: Union[pl.DataFrame, pl.LazyFrame, Path, str],
    dispatch_schedule: Union[pl.DataFrame, pl.LazyFrame] | None = None,
    locomotives_per_node: int | None = None,
) -> pl.DataFrame:
    """
    Generate default locomotive pool
    Arguments:
    ----------
    demand_file: Path to a file with origin-destination demand
    shares: List of shares for each locomotive type in loco_info (implemented for two-way shares only)
    Outputs:
    ----------
    loco_pool: Locomotive pool containing all locomotives' information that are within the system
    """
    config.loco_info = append_loco_info(config.loco_info)
    loco_types = list(config.loco_info.loc[:, "Locomotive_Type"])
    demand, node_list = load_freight_demand(demand_file, config)
    # TODO: handle different train types (or mixed train types?)

    num_nodes = len(node_list)
    if locomotives_per_node is None:
        num_ods = demand.select("Origin", "Destination").unique().height
        if "Number_of_Cars" in demand.collect_schema():
            cars_per_od = (
                demand.group_by("Origin", "Destination")
                .agg(pl.col("Number_of_Cars").sum())
                .get_column("Number_of_Cars")
                .mean()
            )
        elif "Number_of_Containers" in demand.collect_schema():
            cars_per_od = (
                demand.group_by("Origin", "Destination")
                .agg(pl.col("Number_of_Containers").sum())
                .get_column("Number_of_Containers")
                .mean()
            ) / config.containers_per_car
        else:
            raise Exception("No valid columns in demand DataFrame")
        if config.single_train_mode:
            initial_size = math.ceil(cars_per_od / min(config.cars_per_locomotive.values()))
            rows = initial_size
        else:
            num_destinations_per_node = num_ods * 1.0 / num_nodes * 1.0
            initial_size_demand = math.ceil(
                (cars_per_od / min(config.cars_per_locomotive.values())) * num_destinations_per_node
            )  # number of locomotives per node
            initial_size_hp = 0
            if dispatch_schedule is not None:
                # Compute the 24-hour window with the most total locomotives needed
                # (assuming each loco is only dispatched once in a given day)
                loco_mass = config.loco_info["Loco_Mass_Tons"].mean()
                # Average hp_per_ton, weighted by total number of cars.
                # Max hp_per_ton across trains would be more conservative if runs are failing.
                hp_per_ton = dispatch_schedule.select(
                    (pl.col("HP_Required").truediv("Tons_Per_Train"))
                    .mul(pl.col("Number_of_Cars"))
                    .sum()
                    .truediv(pl.col("Number_of_Cars").sum())
                ).item()
                hp_per_loco = config.loco_info["HP"].mean() - loco_mass * hp_per_ton
                initial_size_hp = (
                    dispatch_schedule.with_columns(
                        (pl.col("Hour") // 24).cast(pl.Int32).alias("Day"),
                        pl.col("HP_Required")
                        .truediv(hp_per_loco)
                        .ceil()
                        .mul(config.loco_pool_safety_factor)
                        .alias("Locos_Per_Dispatch"),
                    )
                    .group_by("Day", "Origin")
                    .agg(
                        pl.col("Locos_Per_Dispatch").ceil().sum().alias("Locos_Per_Day_Per_Origin")
                    )
                    .select(pl.col("Locos_Per_Day_Per_Origin").max().cast(pl.Int64))
                    .item()
                )
            initial_size = max(initial_size_demand, initial_size_hp)
            rows = initial_size * num_nodes  # number of locomotives in total
    else:
        initial_size = locomotives_per_node
        rows = locomotives_per_node * num_nodes

    if config.single_train_mode:
        sorted_nodes = np.tile([demand.select(pl.col("Origin").first()).item()], rows).tolist()
        engine_numbers = range(0, rows)
    else:
        sorted_nodes = np.sort(np.tile(node_list, initial_size)).tolist()
        engine_numbers = rankdata(sorted_nodes, method="dense") * 1000 + np.tile(
            range(0, initial_size), num_nodes
        )

    if config.loco_type_shares is not None:
        if not all(key in loco_types for key in config.loco_type_shares.keys()):
            raise ValueError(
                f"""A loco_type was specified in the train planner config's loco_type_shares that was not in its loco_info."""
            )
        if (len(loco_types) != 2) | (len(config.loco_type_shares) != 2):
            raise ValueError(
                f"""2-way prescribed locopool requested but number of locomotive types is not 2."""
            )
        # TODO: this logic can be replaced (and generalized to >2 types) using altrios.utilities.allocateItems
        shares = [config.loco_type_shares.get(type, 0.0) for type in loco_types]

        idx_1 = np.argmin(shares)
        idx_2 = 1 - idx_1
        share_type_one = shares[idx_1]
        label_type_one = loco_types[idx_1]
        label_type_two = loco_types[idx_2]

        num_type_one = round(initial_size * share_type_one)
        if 0 == num_type_one:
            types = pd.Series([label_type_two] * initial_size)
        elif initial_size == num_type_one:
            types = pd.Series([label_type_one] * initial_size)
        else:
            # Arrange repeated sequences of type 1 + {type_two_per_type_one, type_two_per_type_one+1} type 2
            # so as to match the required total counts of each.
            type_two_per_type_one = (initial_size - num_type_one) / num_type_one
            # Number of type 1 + {type_two_per_bel+1} type 2 sequences needed
            num_extra_type_two = round(num_type_one * (type_two_per_type_one % 1.0))
            series_fewer_type_two = pd.Series(
                [label_type_one] + [label_type_two] * math.floor(type_two_per_type_one)
            )
            series_more_type_two = pd.Series(
                [label_type_one] + [label_type_two] * math.ceil(type_two_per_type_one)
            )
            types = np.concatenate(
                (
                    np.tile(series_more_type_two, num_extra_type_two),
                    np.tile(series_fewer_type_two, num_type_one - num_extra_type_two),
                ),
                axis=None,
            )
        types = np.tile(types, num_nodes).tolist()
    else:
        repetitions = math.ceil(rows / len(loco_types))
        types = np.tile(loco_types, repetitions).tolist()[0:rows]

    loco_pool = pl.DataFrame(
        {
            "Locomotive_ID": pl.Series(engine_numbers, dtype=pl.UInt32),
            "Locomotive_Type": pl.Series(types, dtype=pl.Categorical),
            "Node": pl.Series(sorted_nodes, dtype=pl.Categorical),
            "Arrival_Time": pl.Series(np.zeros(rows), dtype=pl.Float64),
            "Servicing_Done_Time": pl.Series(np.zeros(rows), dtype=pl.Float64),
            "Refueling_Done_Time": pl.Series(np.tile(0, rows), dtype=pl.Float64),
            "Status": pl.Series(np.tile("Ready", rows), dtype=pl.Categorical),
            "SOC_Target_J": pl.Series(np.zeros(rows), dtype=pl.Float64),
            "Refuel_Duration": pl.Series(np.zeros(rows), dtype=pl.Float64),
            "Refueler_J_Per_Hr": pl.Series(np.zeros(rows), dtype=pl.Float64),
            "Refueler_Efficiency": pl.Series(np.zeros(rows), dtype=pl.Float64),
            "Port_Count": pl.Series(np.zeros(rows), dtype=pl.UInt32),
        }
    )

    loco_info_pl = pl.from_pandas(
        config.loco_info.drop(labels="Rust_Loco", axis=1),
        schema_overrides={
            "Locomotive_Type": pl.Categorical,
            "Fuel_Type": pl.Categorical,
        },
    )

    loco_pool = loco_pool.join(loco_info_pl, on="Locomotive_Type")
    return loco_pool


def build_refuelers(
    node_list: pd.Series,
    loco_pool: pl.DataFrame,
    refueler_info: pd.DataFrame,
    refuelers_per_incoming_corridor: int,
) -> pl.DataFrame:
    """
    Build the default set of refueling facilities.
    Arguments:
    ----------
    node_list: List of origin or destination demand nodes
    loco_pool: Locomotive pool
    refueler_info: DataFrame with information for each type of refueling infrastructure to use
    refuelers_per_incoming_corridor: Queue size per corridor arriving at each node.
    Outputs:
    ----------
    refuelers: Polars dataframe of facility county by node and type of fuel
    """
    ports_per_node = (
        loco_pool.group_by(pl.col("Locomotive_Type", "Fuel_Type").cast(pl.Utf8))
        .agg(
            [
                (pl.lit(refuelers_per_incoming_corridor) * pl.len() / pl.lit(loco_pool.height))
                .ceil()
                .alias("Ports_Per_Node")
            ]
        )
        .join(
            pl.from_pandas(refueler_info),
            on=["Locomotive_Type", "Fuel_Type"],
            how="left",
        )
    )

    locations = pd.DataFrame(data={"Node": np.tile(node_list, ports_per_node.height)})
    locations = locations.sort_values(by=["Node"]).reset_index(drop=True)

    refuelers = pl.DataFrame(
        {
            "Node": pl.Series(locations["Node"], dtype=pl.Categorical).cast(pl.Categorical),
            "Refueler_Type": pl.Series(
                np.tile(ports_per_node.get_column("Refueler_Type").to_list(), len(node_list)),
                dtype=pl.Categorical,
            ).cast(pl.Categorical),
            "Locomotive_Type": pl.Series(
                np.tile(
                    ports_per_node.get_column("Locomotive_Type").to_list(),
                    len(node_list),
                ),
                dtype=pl.Categorical,
            ).cast(pl.Categorical),
            "Fuel_Type": pl.Series(
                np.tile(ports_per_node.get_column("Fuel_Type").to_list(), len(node_list)),
                dtype=pl.Categorical,
            ).cast(pl.Categorical),
            "Refueler_J_Per_Hr": pl.Series(
                np.tile(
                    ports_per_node.get_column("Refueler_J_Per_Hr").to_list(),
                    len(node_list),
                ),
                dtype=pl.Float64,
            ),
            "Refueler_Efficiency": pl.Series(
                np.tile(
                    ports_per_node.get_column("Refueler_Efficiency").to_list(),
                    len(node_list),
                ),
                dtype=pl.Float64,
            ),
            "Lifespan_Years": pl.Series(
                np.tile(
                    ports_per_node.get_column("Lifespan_Years").to_list(),
                    len(node_list),
                ),
                dtype=pl.Float64,
            ),
            "Cost_USD": pl.Series(
                np.tile(ports_per_node.get_column("Cost_USD").to_list(), len(node_list)),
                dtype=pl.Float64,
            ),
            "Port_Count": pl.Series(
                np.tile(
                    ports_per_node.get_column("Ports_Per_Node").to_list(),
                    len(node_list),
                ),
                dtype=pl.UInt32,
            ),
        }
    )
    return refuelers


def append_charging_guidelines(
    refuelers: pl.DataFrame,
    loco_pool: pl.DataFrame,
    demand: pl.DataFrame,
    network_charging_guidelines: pl.DataFrame,
) -> pl.DataFrame:
    active_ods = demand.select(["Origin", "Destination"]).unique()
    network_charging_guidelines = (
        network_charging_guidelines.join(active_ods, on=["Origin", "Destination"], how="inner")
        .group_by(pl.col("Origin"))
        .agg(pl.col("Allowable_Battery_Headroom_MWh").min() * 1e6 / utilities.MWH_PER_MJ)
        .rename({"Allowable_Battery_Headroom_MWh": "Battery_Headroom_J"})
        .with_columns(pl.col("Origin").cast(pl.Categorical))
    )
    refuelers = refuelers.join(
        network_charging_guidelines, left_on="Node", right_on="Origin", how="left"
    ).with_columns(
        pl.when(pl.col("Fuel_Type") == "Electricity")
        .then(pl.col("Battery_Headroom_J"))
        .otherwise(0)
        .fill_null(0)
        .alias("Battery_Headroom_J")
    )
    loco_pool = (
        loco_pool.join(network_charging_guidelines, left_on="Node", right_on="Origin", how="left")
        .with_columns(
            pl.when(pl.col("Fuel_Type") == "Electricity")
            .then(pl.col("Battery_Headroom_J"))
            .otherwise(0)
            .fill_null(0)
            .alias("Battery_Headroom_J")
        )
        .with_columns(
            pl.max_horizontal(
                [
                    pl.col("SOC_Max_J") - pl.col("Battery_Headroom_J"),
                    pl.col("SOC_Min_J"),
                ]
            ).alias("SOC_J")
        )
    )
    return refuelers, loco_pool


def configure_rail_vehicles(
    single_train_dispatch: Dict,
    available_rail_vehicles: List[alt.RailVehicle],
    freight_type_to_car_type: Dict,
) -> (list[alt.RailVehicle], dict[str, int]):
    freight_types = []
    n_cars_by_type = {}
    this_train_type = single_train_dispatch["Train_Type"]
    if single_train_dispatch["Cars_Loaded"] > 0:
        freight_type = f"{this_train_type}_Loaded"
        freight_types.append(freight_type)
        if freight_type in freight_type_to_car_type:
            car_type = freight_type_to_car_type[freight_type]
        else:
            raise Exception(f"Rail vehicle car type not found for freight type {freight_type}.")
        n_cars_by_type[car_type] = int(single_train_dispatch["Cars_Loaded"])
    if single_train_dispatch["Cars_Empty"] > 0:
        freight_type = f"{this_train_type}_Empty"
        freight_types.append(freight_type)
        if freight_type in freight_type_to_car_type:
            car_type = freight_type_to_car_type[freight_type]
        else:
            raise Exception(f"Rail vehicle car type not found for freight type {freight_type}.")
        n_cars_by_type[car_type] = int(single_train_dispatch["Cars_Empty"])

    assert len(n_cars_by_type) > 0, \
        "`single_train_dispatch[\"Cars_Loaded\"]` and `single_train_dispatch[\"Cars_Empty\"]` are both zero"

    rv_to_use = [
        vehicle
        for vehicle in available_rail_vehicles
        if vehicle.to_pydict()["freight_type"] in freight_types
    ]
    return rv_to_use, n_cars_by_type


def appendTonsAndHP(
    df: Union[pl.DataFrame, pl.LazyFrame],
    rail_vehicles,
    freight_type_to_car_type,
    config,
) -> Union[pl.DataFrame, pl.LazyFrame]:
    hp_per_ton = pl.concat(
        [
            (
                pl.DataFrame(this_dict)
                .melt(variable_name="Train_Type", value_name="HP_Required_Per_Ton")
                .with_columns(pl.lit(this_item).alias("O_D"))
                .with_columns(
                    pl.col("O_D").str.split("_").list.first().alias("Origin"),
                    pl.col("O_D").str.split("_").list.last().alias("Destination"),
                )
            )
            for this_item, this_dict in config.hp_required_per_ton.items()
        ],
        how="horizontal_relaxed",
    )

    def get_kg_empty(veh: alt.RailVehicle):
        veh_dict = veh.to_pydict()
        return (
            veh_dict["mass_static_base_kilograms"]
            + veh_dict["axle_count"] * veh_dict["mass_rot_per_axle_kilograms"]
        )

    def get_kg(veh: alt.RailVehicle):
        veh_dict = veh.to_pydict()
        return (
            veh_dict["mass_static_base_kilograms"]
            + veh_dict["mass_freight_kilograms"]
            + veh_dict["axle_count"] * veh_dict["mass_rot_per_axle_kilograms"]
        )

    tons_per_car = (
        pl.DataFrame(
            {
                "Car_Type": pl.Series([rv.to_pydict()["car_type"] for rv in rail_vehicles]),
                "KG": [get_kg(rv) for rv in rail_vehicles],
                "KG_Empty": [get_kg_empty(rv) for rv in rail_vehicles],
            }
        )
        .with_columns(
            pl.when(pl.col("Car_Type").str.to_lowercase().str.contains("_empty"))
            .then(pl.col("KG_Empty") / utilities.KG_PER_TON)
            .otherwise(pl.col("KG") / utilities.KG_PER_TON)
            .alias("Tons_Per_Car")
        )
        .drop(["KG_Empty", "KG"])
    )

    return (
        df.with_columns(
            pl.when(pl.col("Train_Type").str.contains(pl.lit("_Empty")))
            .then(pl.col("Train_Type"))
            .otherwise(
                pl.concat_str(pl.col("Train_Type").str.strip_suffix("_Loaded"), pl.lit("_Loaded"))
            )
            .replace_strict(freight_type_to_car_type)
            .alias("Car_Type")
        )
        .join(tons_per_car, how="left", on="Car_Type")
        # Merge on OD-specific hp_per_ton if the user specified any
        .join(
            hp_per_ton.filter(pl.col("O_D") != pl.lit("Default")).drop("O_D"),
            on=[
                pl.col("Origin"),
                pl.col("Destination"),
                pl.col("Train_Type").str.strip_suffix("_Empty").str.strip_suffix("_Loaded"),
            ],
            how="left",
        )
        # Second, merge on defaults per train type
        .join(
            hp_per_ton.filter((pl.col("O_D") == "Default")).drop(["O_D", "Origin", "Destination"]),
            on=[pl.col("Train_Type").str.strip_suffix("_Empty").str.strip_suffix("_Loaded")],
            how="left",
            suffix="_Default",
        )
        .with_columns(
            pl.coalesce("HP_Required_Per_Ton", "HP_Required_Per_Ton_Default").alias(
                "HP_Required_Per_Ton"
            )
        )
        .drop(cs.ends_with("_Default") | cs.ends_with("_right"))
    )
