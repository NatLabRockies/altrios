import pandas as pd
import polars as pl
import polars.selectors as cs
import math
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import altrios as alt
from altrios import utilities, defaults
from altrios.train_planner import planner_config

MetricType = pl.DataFrame

emissions_factors_greet = pl.read_csv(defaults.FUEL_EMISSIONS_FILE, null_values="NA")
electricity_prices_eia = (
    pl.read_csv(source=defaults.ELECTRICITY_PRICE_FILE, skip_rows=5)
    .select(["Year", "Industrial"])
    .rename({"Industrial": "Price"})
)
liquid_fuel_prices_eia = (
    pl.read_csv(source=defaults.LIQUID_FUEL_PRICE_FILE, skip_rows=5)
    .select(["Year", "   Diesel Fuel (distillate fuel oil) 6/"])
    .rename({"   Diesel Fuel (distillate fuel oil) 6/": "Price"})
)
battery_prices_nrel_atb = (pl.read_csv(source=defaults.BATTERY_PRICE_FILE)).filter(
    pl.col("Scenario") == "Moderate"
)

metric_columns = ["Subset", "Value", "Metric", "Units", "Year"]


@dataclass
class ScenarioInfo:
    """
    Dataclass class maintaining records of scenario parameters that influence metric calculations.

    Fields:
    - `sims`: `SpeedLimitTrainSim` (single-train sim) or `SpeedLimitTrainSimVec` (multi-train sim)
    including simulation results

    - `simulation_days`: Number of days included in these results (after any warm-start or cool-down
    days were excluded)

    - `annualize`: Whether to scale up output metrics to a full year's equivalent.

    - `scenario_year`: Year that is being considered in this scenario.

    - `loco_pool`: `polars.DataFrame` defining the pool of locomotives that were available
    to potentially be dispatched, each having a `Locomotive_ID`,`Locomotive_Type`,`Cost_USD`,`Lifespan_Years`.
    Not required for single-train sim.

    - `consist_plan`: `polars.DataFrame` defining dispatched train consists, where
    each row includes a `Locomotive_ID` and a `Train_ID`. Not required for single-train sim.

    - `refuel_facilities`: `polars.DataFrame` defining refueling facilities, each with a
    `Refueler_Type`, `Port_Count`, and `Cost_USD`, and `Lifespan_Years`. Not required for single-train sim.

    - `refuel_sessions`: `polars.DataFrame` defining refueling sessions, each with a
    `Locomotive_ID`, `Locomotive_Type`, `Fuel_Type`, `Node`, and `Refuel_Energy_J`. Not required for single-train sim.

    - `emissions_factors`: `polars.DataFrame` with unit `CO2eq_kg_per_MWh` defined for each `Node`.
    Not required for single-train sim.

    - `nodal_energy_prices`: `polars.DataFrame` with unit `Price` defined for each `Node` and `Fuel`.
    Not required for single-train sim.

    - `count_unused_locomotives`: If `True`, fleet composition is defined using all locomotives in
    `loco_pool`; if `False`, fleet composition is defined using only the locomotives dispatched.
    Not required for single-train sim.
    """

    sims: Union[alt.SpeedLimitTrainSim, alt.SpeedLimitTrainSimVec]
    simulation_days: int
    annualize: bool
    scenario_year: int = defaults.BASE_ANALYSIS_YEAR
    loco_pool: pl.DataFrame = None
    consist_plan: pl.DataFrame = None
    refuel_facilities: pl.DataFrame = None
    refuel_sessions: pl.DataFrame = None
    emissions_factors: pl.DataFrame = None
    nodal_energy_prices: pl.DataFrame = None
    count_unused_locomotives: bool = False
    train_planner_config: planner_config = None


def metric(
    name: str, units: str, value: float, subset: str = "All", year: str = "All"
) -> MetricType:
    return MetricType(
        {
            "Metric": [name],
            "Units": [units],
            "Value": [None if value is None else float(value)],
            "Subset": [subset],
            "Year": [year],
        }
    )


def metrics_from_list(metrics: List[MetricType]) -> MetricType:
    return pl.concat(metrics, how="diagonal_relaxed")


def value_from_metrics(
    metrics: MetricType, name: str = None, units: str = None, subset: str = None
) -> float:
    if metrics is None:
        return None

    out = metrics
    if name is not None:
        out = out.filter(pl.col("Metric") == pl.lit(name))

    if units is not None:
        out = out.filter(pl.col("Units") == pl.lit(units))

    if subset is not None:
        out = out.filter(pl.col("Subset") == subset)

    if out.height == 1:
        return out.get_column("Value")[0]
    else:
        return None


def main(
        scenario_infos: Union[ScenarioInfo, List[ScenarioInfo]],
        annual_metrics: Union[Tuple[str, str],
                              List[Tuple[str, str]]] = [
            ('Freight_Moved', 'million tonne-mi'),
            ('Freight_Moved', 'million tonne-km'),
            ('Freight_Moved', 'car-miles'),
            ('Freight_Moved', 'cars'),
            ('Freight_Moved', 'detailed car counts'),
            ('GHG', 'tonne CO2-eq'),
            ('Count_Locomotives', 'assets'),
            ('Count_Refuelers', 'assets'),
            ('Energy_Costs', 'USD'),
            ('Energy_Per_Freight_Moved', 'kWh per car-mile')
        ],
        calculate_multiyear_metrics: bool = True,
) -> pl.DataFrame:
    """
    Given a set of simulation results and the associated consist plans, computes economic and environmental metrics.
    Arguments:
    ----------
    scenario_infos: List (with one entry per scenario year) of Scenario Info objects
    metricsToCalc: List of metrics to calculate, each specified as a tuple consisting of a metric and the desired unit
    calculate_multiyear_metrics: True if multi-year rollout costs (including levelized cost) are to be computed
    Outputs:
    ----------
    values: DataFrame of output and intermediate metrics (metric name, units, value, and scenario year)
    """
    if not isinstance(scenario_infos, list):
        scenario_infos = [scenario_infos]

    if not isinstance(annual_metrics, list):
        annual_metrics = [annual_metrics]

    annual_values = []
    for annual_metric in annual_metrics:
        for scenario_info in scenario_infos:
            annual_values.append(
                calculate_annual_metric(
                    annual_metric[0], annual_metric[1], scenario_info
                )
            )

    values = pl.concat(annual_values, how="diagonal_relaxed").unique()

    if calculate_multiyear_metrics:
        values = calculate_rollout_investments(values)
        values = calculate_rollout_total_costs(values)
        values = calculate_rollout_lcotkm(values)

    values = (
        values.filter(pl.col("Value").is_not_null())
        .unique()
        .sort(
            ["Metric", "Units", "Year", "Subset"],
            descending=[False, False, False, True],
        )
    )

    return values


def calculate_annual_metric(
    metric_name: str, units: str, info: ScenarioInfo
) -> MetricType:
    """
    Given a years' worth of simulation results and the associated consist plan, computes the requested metric.
    Arguments:
    ----------
    thisRow: DataFrame containing the requested metric and requested units
    info: A scenario information object representing parameters and results for a single year
    Outputs:
    ----------
    values: DataFrame of requested output metric + any intermediate metrics (metric name, units, value, and scenario year)
    """

    def return_metric_error(info, units) -> MetricType:
        print(f"Metric calculation not implemented: {metric_name}.")
        return value_from_metrics(metric(metric_name, units, None))

    function_for_metric = function_mappings.get(metric_name, return_metric_error)
    return function_for_metric(info, units).with_columns(
        pl.lit(info.scenario_year).alias("Year")
    )


def calculate_rollout_lcotkm(values: MetricType) -> MetricType:
    """
    Given a DataFrame of each year's costs and gross freight deliveries, computes the multi-year levelized cost per gross tonne-km of freight delivered.
    Arguments:
    ----------
    values: DataFrame containing total costs and gross freight deliveries for each modeled scenario year
    Outputs:
    ----------
    DataFrame of LCOTKM result (metric name, units, value, and scenario year)
    """

    cost_timeseries = (values
        .filter(
            pl.col("Metric")==pl.lit("Cost_Total"),
            pl.col("Subset")=="All"
        )
        .select(["Year","Value"])
        .rename({"Value": "Cost_Total"}))
    tkm_timeseries = (values
        .filter(
            pl.col("Metric")==pl.lit("Freight_Moved"),
            pl.col("Units")==pl.lit("million tonne-km")
        )
        .select(["Year", "Value"])
        .rename({"Value": "Mt-km"}))
    timeseries = cost_timeseries.join(tkm_timeseries, on="Year", how="outer")
    timeseries = (timeseries
                  .with_columns((pl.col("Year") - pl.col("Year").min()).alias("Year_Offset"))
                  .with_columns(((1+defaults.DISCOUNT_RATE)**pl.col("Year_Offset")).alias("Discounting_Factor"))
                  .with_columns(
                    (pl.col("Cost_Total") / pl.col("Discounting_Factor")).alias("Cost_Total_Discounted"),
                    (pl.col("Mt-km") / pl.col("Discounting_Factor")).alias("Mt-km_Discounted"))
                  .with_columns(pl.col("Year").cast(pl.Utf8)))           
    cost_total = timeseries.get_column("Cost_Total_Discounted").sum()
    starting_residual_value_to_subtract = (values
        .filter(pl.col("Metric")==pl.lit("Asset_Value_Initial"))
        .get_column("Value")[0]
    )
    cost_minus_residual_baseline = (
        cost_total
        + starting_residual_value_to_subtract
        / timeseries.get_column("Discounting_Factor").max()
    )
    tkm_total = timeseries.get_column("Mt-km_Discounted").sum()
    lcotkm_all = cost_minus_residual_baseline / tkm_total if tkm_total > 0 else math.nan

    cost_discounted = timeseries.select(
        pl.col("Year"),
        Value=pl.col("Cost_Total_Discounted"),
        Metric=pl.lit("Cost_Total_Discounted"),
        Subset=pl.lit("All"),
        Units=pl.lit("USD (discounted)"),
    )
    tkm_discounted = timeseries.select(
        pl.col("Year"),
        Value=pl.col("Mt-km_Discounted"),
        Metric=pl.lit("Mt-km_Discounted"),
        Subset=pl.lit("All"),
        Units=pl.lit("million tonne-km (discounted)"),
    )
    cotkm_annual = timeseries.select(
        pl.col("Year"),
        Value=pl.col("Cost_Total") / pl.col("Mt-km"),
        Metric=pl.lit("Cost_Per_Mt-km"),
        Subset=pl.lit("All"),
        Units=pl.lit("USD per million tonne-km"),
    )
    discount = timeseries.select(
        pl.col("Year"),
        Value=pl.col("Discounting_Factor"),
        Metric=pl.lit("Discounting_Factor"),
        Subset=pl.lit("All"),
        Units=pl.lit("fraction (0-1)"),
    )
    return metrics_from_list(
        [
            values.with_columns(pl.col("Year").cast(pl.Utf8)),
            cost_discounted,
            tkm_discounted,
            cotkm_annual,
            discount,
            metric("LCOTKM", "USD per million tonne-km (levelized)", lcotkm_all),
        ]
    )


def calculate_energy_per_freight(info: ScenarioInfo, units: str) -> MetricType:
    """
    Given a years' worth of simulation results, computes a single year energy usage per unit of freight moved.
    Arguments:
    ----------
    info: A scenario information object representing parameters and results for a single year
    units: Requested units
    Outputs:
    ----------
    DataFrame of energy usage per freight moved (metric name, units, value, and scenario year)
    """
    if "per car-mile" not in units and "per container-mile" not in units:
        print(f"Units of {units} not supported for energy-per-freight calculation.")
        return metric("Energy_Per_Freight_Moved", units, None)

    conversion_from_megajoule = 0
    if "MJ" in units:
        conversion_from_megajoule = 1
    elif "MWh" in units:
        conversion_from_megajoule = utilities.KWH_PER_MJ / 1e3
    elif "kWh" in units:
        conversion_from_megajoule = utilities.KWH_PER_MJ

    diesel_mj = calculate_diesel_use(info, units="MJ")
    electricity_mj = calculate_electricity_use(info, units="MJ")
    total_mj = value_from_metrics(diesel_mj) + value_from_metrics(
        electricity_mj, subset="All"
    )
    total_energy = total_mj * conversion_from_megajoule
    if "per car-mile" in units:
        freight_moved = calculate_freight_moved(info, units="car-miles")
    elif "per container-mile" in units:
        freight_moved = calculate_freight_moved(info, units="container-miles")
    freight_val = value_from_metrics(freight_moved)
    return metrics_from_list(
        [
            diesel_mj,
            electricity_mj,
            metric("Energy_Use", "MJ", total_mj),
            metric("Energy_Per_Freight_Moved", units, total_energy / freight_val),
        ]
    )


def calculate_energy_cost(info: ScenarioInfo, units: str) -> MetricType:
    """
    Given a years' worth of simulation results, computes a single year energy costs.
    Arguments:
    ----------
    info: A scenario information object representing parameters and results for a single year
    units: Requested units
    Outputs:
    ----------
    DataFrame of energy costs + intermediate metrics (metric name, units, value, and scenario year)
    """
    if info.nodal_energy_prices is None:
        return metric("Cost_Energy", units, None)

    diesel_used = calculate_diesel_use(info, units="gallons")
    electricity_used = calculate_electricity_use(info, units="kWh")
    electricity_costs_disagg = (
        electricity_used.filter(pl.col("Subset") != "All")
        .join(
            info.nodal_energy_prices.filter(pl.col("Fuel") == "Electricity"),
            left_on="Subset",
            right_on="Node",
            how="left",
        )
        .with_columns(
            (pl.col("Value") * pl.col("Price") / 100).alias("Value"),
            pl.lit("Cost_Electricity").alias("Metric"),
            pl.lit("USD").alias("Units"),
        )
        .select(metric_columns)
    )
    electricity_costs_agg = (
        electricity_costs_disagg.group_by(["Metric", "Units", "Year"])
        .agg(pl.col("Value").sum())
        .with_columns(pl.lit("All").alias("Subset"))
    )
    electricity_cost_value = 0
    if electricity_costs_agg.height > 0:
        electricity_cost_value = electricity_costs_agg.get_column("Value")[0]
    # Diesel refueling is not yet tracked spatiotemporally; just use average price across the network.
    diesel_price = (
        info.nodal_energy_prices.filter(pl.col("Fuel") == "Diesel")
        .get_column("Price")
        .mean()
    )
    diesel_cost_value = value_from_metrics(diesel_used, "Diesel_Usage") * diesel_price
    if electricity_cost_value is None:
        electricity_cost_value = 0.0
    if diesel_cost_value is None:
        diesel_cost_value = 0
    return metrics_from_list(
        [
            diesel_used,
            electricity_used,
            metric("Cost_Diesel", "USD", diesel_cost_value),
            electricity_costs_disagg,
            electricity_costs_agg,
            metric("Cost_Energy", units, diesel_cost_value + electricity_cost_value),
        ]
    )


def calculate_diesel_use(info: ScenarioInfo, units: str):
    """
    Given a years' worth of simulation results, computes a single year diesel fuel use.
    Arguments:
    ----------
    info: A scenario information object representing parameters and results for a single year
    units: Requested units.
    Outputs:
    ----------
    DataFrame of diesel use (metric name, units, value, and scenario year)
    """
    diesel_emissions_factors = emissions_factors_greet.filter(
        pl.col("Name") == pl.lit("ultra low sulfur diesel")
    )
    lhv_kj_per_kg = diesel_emissions_factors.get_column("LowHeatingValue")[0] * 1e3
    rho_fuel_g_per_gallon = diesel_emissions_factors.get_column("Density")[0]
    if units == "gallons":
        val = (
            (
                info.sims.get_energy_fuel_joules(annualize=info.annualize)
                / 1e3
                / lhv_kj_per_kg
            )
            * 1e3
            / rho_fuel_g_per_gallon
        )
    elif units == "MJ":
        val = info.sims.get_energy_fuel_joules(annualize=info.annualize) / 1e6
    else:
        print(f"Units of {units} not supported for fuel usage calculation.")
        val = None

    return metric("Diesel_Usage", units, val)


def calculate_electricity_use(info: ScenarioInfo, units: str) -> MetricType:
    """
    Given a years' worth of simulation results, computes a single year grid electricity use.
    Arguments:
    ----------
    info: A scenario information object representing parameters and results for a single year
    units: Requested units
    Outputs:
    ----------
    DataFrame of grid electricity use (metric name, units, value, and scenario year)
    """
    conversion_from_joule = 0
    if units == "J":
        conversion_from_joule = 1.0
    elif units == "MJ":
        conversion_from_joule = 1e-6
    elif units == "MWh":
        conversion_from_joule = utilities.KWH_PER_MJ / 1e9
    elif units == "kWh":
        conversion_from_joule = utilities.KWH_PER_MJ / 1e6
    else:
        print(f"Units of {units} not supported for electricity use calculation.")
        return metric("Electricity_Usage", units, None)

    if (info.refuel_sessions is None) or (
        info.refuel_sessions.filter(pl.col("Fuel_Type") == pl.lit("Electricity")).height
        == 0
    ):
        # No refueling session data: charging was not explicitly modeled,
        # so take total net energy at RES and apply charging efficiency factor
        return metric(
            "Electricity_Usage",
            units,
            info.sims.get_net_energy_res_joules(annualize=info.annualize)
            * conversion_from_joule
            / defaults.BEL_CHARGER_EFFICIENCY,
        )
    else:
        if info.annualize:
            scaler = 365.25 / info.simulation_days
        else:
            scaler = 1
        disagg_energy = (
            info.refuel_sessions.filter(pl.col("Fuel_Type") == pl.lit("Electricity"))
            .group_by(["Node"])
            .agg(
                pl.col("Refuel_Energy_J")
                .mul(conversion_from_joule)
                .mul(scaler)
                .sum()
                .alias(units)
            )
            .with_columns(pl.lit("Electricity_Usage").alias("Metric"))
            .melt(
                id_vars=["Metric", "Node"],
                value_vars=units,
                variable_name="Units",
                value_name="Value",
            )
            .rename({"Node": "Subset"})
            .with_columns(pl.lit(info.scenario_year).alias("Year"))
        )

        agg_energy = (
            disagg_energy.group_by(["Metric", "Units", "Year"])
            .agg(pl.col("Value").sum())
            .with_columns(pl.lit("All").alias("Subset"))
        )
        return metrics_from_list([agg_energy, disagg_energy])


def calculate_freight_moved(info: ScenarioInfo, units: str) -> MetricType:
    """
    Given a years' worth of simulation results, computes a single year quantity of freight moved
    Arguments:
    ----------
    info: A scenario information object representing parameters and results for a single year
    units: Requested units
    Outputs:
    ----------
    DataFrame of quantity of freight (metric name, units, value, and scenario year)
    """
    if "-mi" in units:
        conversion_from_km = utilities.MI_PER_KM
    else:
        conversion_from_km = 1.0

    if units in ["million tonne-km", "million tonne-mi"]:
        return metric(
            "Freight_Moved",
            units,
            info.sims.get_megagram_kilometers(annualize=info.annualize)
            * conversion_from_km
            / 1.0e6,
            year=info.scenario_year,
        )
    elif units in ["car-km", "car-miles"]:
        return metric(
            "Freight_Moved",
            units,
            info.sims.get_car_kilometers(annualize=info.annualize) * conversion_from_km,
            year=info.scenario_year,
        )
    elif units == "cars":
        return metric(
            "Freight_Moved",
            units,
            info.sims.get_cars_moved(annualize=info.annualize),
            year=info.scenario_year,
        )
    elif units in ["container-km", "container-miles"]:
        assert (
            info.consist_plan.filter(
                ~pl.col("Train_Type").str.contains("Intermodal")
            ).height
            == 0
        ), "Can only count containers if the consist plan is all Intermodal"
        car_distance = (
            info.sims.get_car_kilometers(annualize=info.annualize) * conversion_from_km
        )
        return metric(
            "Freight_Moved",
            units,
            car_distance * info.train_planner_config.containers_per_car,
            year=info.scenario_year,
        )

    elif units == "containers":
        container_counts = (
            info.consist_plan.select(
                "Train_ID", "Containers_Loaded", "Containers_Empty"
            )
            .unique()
            .drop("Train_ID")
            .sum()
        )
        if info.annualize:
            annualizer = 365.25 / info.simulation_days
        else:
            annualizer = 1.0
        return metrics_from_list(
            [
                metric(
                    "Freight_Moved",
                    units,
                    container_counts.get_column("Containers_Loaded").item()
                    * annualizer,
                    "Loaded",
                    year=info.scenario_year,
                ),
                metric(
                    "Freight_Moved",
                    units,
                    container_counts.get_column("Containers_Empty").item() * annualizer,
                    "Loaded",
                    year=info.scenario_year,
                ),
            ]
        )
    elif units == "detailed car counts":
        kilometers = (
            pl.DataFrame(
                data={
                    "car-km": [
                        alt.SpeedLimitTrainSim.from_pydict(sim).get_kilometers(annualize=info.annualize)
                        for sim in info.sims.to_pydict()
                    ]
                }
            )
            .with_row_index("idx")
            .with_columns(pl.col("car-km").mul(utilities.MI_PER_KM).alias("car-miles"))
        )
        all_n_cars_by_type = [sim['n_cars_by_type'] for sim in info.sims.to_pydict()]
        car_counts = (
            pl.concat(
                [pl.from_dict(item) for item in all_n_cars_by_type],
                how="diagonal_relaxed",
            )
            .with_row_index("idx")
            .melt(id_vars="idx", value_name="cars", variable_name="Subset")
            .filter(pl.col("cars").is_not_null())
            .join(kilometers, how="left", on="idx")
            .drop("idx")
            .group_by("Subset")
            .agg(pl.col("*").sum())
            .sort("Subset")
            .melt(id_vars="Subset", variable_name="Units", value_name="Value")
            .with_columns(
                pl.lit(info.scenario_year).alias("Year"),
                pl.lit("Freight_Moved").alias("Metric"),
                pl.when(info.annualize, pl.col("Units") == pl.lit("cars"))
                .then(pl.col("Value").mul(365.25 / info.simulation_days))
                .otherwise(pl.col("Value"))
                .alias("Value"),
            )
        )
        return car_counts
    else:
        print(f"Units of {units} not supported for freight movement calculation.")
        return metric("Freight_Moved", units, None)


def calculate_ghg(info: ScenarioInfo, units: str) -> MetricType:
    """
    Given a years' worth of simulation results, computes a single year GHG emissions from energy use
    Arguments:
    ----------
    info: A scenario information object representing parameters and results for a single year
    units: Requested units
    Outputs:
    ----------
    DataFrame of GHG emissions from energy use (metric name, units, value, and scenario year)
    """
    if units != "tonne CO2-eq":
        return metric("GHG_Energy", units, None)

    # GREET table only has a value for 2020; apply that to all years
    diesel_ghg_factor = (
        emissions_factors_greet.filter(
            (pl.col("Name") == "ultra low sulfur diesel")
            & (pl.col("Units") == "gCO2 eq/MJ")
            & (pl.col("Year") == 2020)
        ).get_column("Value")
    )[0]

    diesel_MJ = calculate_diesel_use(info, "MJ")
    electricity_MJ = calculate_electricity_use(info, "MJ")
    electricity_MWh = calculate_electricity_use(info, "MWh")

    diesel_ghg_val = (
        value_from_metrics(diesel_MJ, "Diesel_Usage")
        * diesel_ghg_factor
        / utilities.G_PER_TONNE
    )

    electricity_ghg_val = None
    energy_ghg_val = diesel_ghg_val
    if (electricity_MWh.height > 0) & (
        electricity_MWh.select(pl.col("Value").sum()).item() > 0
    ):
        if info.emissions_factors is None:
            energy_ghg_val = None
            print(
                "No electricity emissions factors provided, so GHGs from electricity were not calculated."
            )
        elif (info.refuel_sessions is None) and (
            info.emissions_factors.select(pl.col("Node").n_unique() > 0)
        ):
            energy_ghg_val = None
            print(
                """No refuel session dataframe was provided (so emissions are not spatially resolved), 
                but the emissions factor dataframe contains multiple regions. Subset emissions factors 
                to the desired region before passing the emissions factor dataframe into the metrics calculator."""
            )
        else:
            if electricity_MWh.filter(pl.col("Subset") != "All").height > 0:
                # Disaggregated results are available
                electricity_ghg_val = (
                    electricity_MWh.filter(pl.col("Subset") != pl.lit("All"))
                    .join(
                        info.emissions_factors,
                        left_on=["Subset", "Year"],
                        right_on=["Node", "Year"],
                        how="inner",
                    )
                    .select(pl.col("CO2eq_kg_per_MWh") * pl.col("Value") / 1000.0)
                    .sum()
                    .to_series()[0]
                )
            else:
                # Disaggregated results are not available but there is only one node with emissions data
                electricity_ghg_val = (
                    electricity_MWh.join(info.emissions_factors, on="Year", how="inner")
                    .select(pl.col("CO2eq_kg_per_MWh") * pl.col("Value") / 1000.0)
                    .sum()
                    .to_series()[0]
                )
            energy_ghg_val = diesel_ghg_val + electricity_ghg_val

    return metrics_from_list(
        [
            diesel_MJ,
            electricity_MJ,
            electricity_MWh,
            metric("GHG_Diesel", units, diesel_ghg_val),
            metric("GHG_Electricity", units, electricity_ghg_val),
            metric("GHG_Energy", units, energy_ghg_val),
        ]
    )


def calculate_locomotive_counts(info: ScenarioInfo, _) -> MetricType:
    """
    Given a single scenario year's locomotive consist plan, computes the year's locomotive fleet composition
    Arguments:
    ----------
    info: A scenario information object representing parameters and results for a single year
    Outputs:
    ----------
    DataFrame of locomotive fleet composition metrics (metric name, units, value, and scenario year)
    """

    def get_locomotive_counts(df: pl.DataFrame) -> pl.DataFrame:
        if (info.consist_plan is None) or (info.loco_pool is None):
            return metric("Count_Locomotives", "assets", None)

        out = (
            df.group_by(["Locomotive_Type"])
            .agg(
                (pl.col("Locomotive_ID").n_unique()).alias("Count_Locomotives"),
                pl.col("Cost_USD").mean().alias("Unit_Cost"),
                pl.col("Lifespan_Years").mean().alias("Lifespan"),
            )
            .with_columns(
                (pl.col("Count_Locomotives") / pl.col("Count_Locomotives").sum()).alias(
                    "Pct_Locomotives"
                )
            )
            .melt(id_vars="Locomotive_Type", variable_name="Metric", value_name="Value")
            .with_columns(
                pl.when(pl.col("Metric") == "Unit_Cost")
                .then(pl.lit("USD"))
                .when(pl.col("Metric") == "Lifespan")
                .then(pl.lit("years"))
                .when(pl.col("Metric") == "Pct_Locomotives")
                .then(pl.lit("fraction (0-1)"))
                .otherwise(pl.lit("assets"))
                .alias("Units"),
                pl.lit("All").alias("Year"),
            )
            .rename({"Locomotive_Type": "Subset"})
        )
        out_agg = (
            out.filter(pl.col("Metric") == ("Count_Locomotives"))
            .group_by(["Metric", "Units", "Year"])
            .agg(pl.col("Value").sum())
            .with_columns(pl.lit("All").alias("Subset"))
        )
        return metrics_from_list([out, out_agg])

    locos_to_use = None
    if info.count_unused_locomotives:
        locos_to_use = info.loco_pool
    else:
        locos_to_use = info.consist_plan.join(
            info.loco_pool.select(["Locomotive_ID", "Cost_USD", "Lifespan_Years"]),
            on="Locomotive_ID",
            how="left",
        )

    return get_locomotive_counts(locos_to_use)


def calculate_refueler_counts(info: ScenarioInfo, _) -> MetricType:
    """
    Given a single scenario year's results, counts how many refuelers were included in the simulation.
    Arguments:
    ----------
    info: A scenario information object representing parameters and results for a single year
    Outputs:
    ----------
    DataFrame of locomotive fleet composition metrics (metric name, units, value, and scenario year)
    """
    if info.refuel_facilities is None:
        return metric("Count_Refuelers", "assets", None)

    num_ports = (
        info.refuel_facilities.group_by(["Refueler_Type"])
        .agg(
            pl.col("Cost_USD").mean().alias("Unit_Cost"),
            pl.col("Port_Count").sum().alias("Count_Refuelers"),
            pl.col("Lifespan_Years").mean().alias("Lifespan"),
        )
        .melt(id_vars="Refueler_Type", variable_name="Metric", value_name="Value")
        .with_columns(
            pl.when(pl.col("Metric") == "Unit_Cost")
            .then(pl.lit("USD"))
            .when(pl.col("Metric") == "Lifespan")
            .then(pl.lit("years"))
            .otherwise(pl.lit("assets"))
            .alias("Units"),
            pl.lit("All").alias("Year"),
        )
        .rename({"Refueler_Type": "Subset"})
    )

    return num_ports


def calculate_rollout_investments(values: MetricType) -> MetricType:
    """
    Given multiple scenario years' locomotive fleet compositions, computes additional across-year metrics
    Arguments:
    ----------
    values: DataFrame with multiple scenario years' locomotive fleet composition metrics
    Outputs:
    ----------
    DataFrame of across-year locomotive fleet composition metrics (metric name, units, value, and scenario year)
    """
    item_id_cols = ["Subset"]
    loco_types = (
        values.filter((pl.col("Units") == "assets") & (pl.col("Subset") != "All"))
        .get_column("Subset")
        .unique()
    )
    costs = (
        values.filter(pl.col("Metric") == "Unit_Cost")
        .rename({"Value": "Unit_Cost"})
        .drop(["Metric", "Units"])
    )
    lifespans = (
        values.filter(pl.col("Metric") == "Lifespan")
        .with_columns(pl.col("Value").cast(pl.Int32).alias("Lifespan"))
        .drop(["Metric", "Value", "Units"])
    )
    stock_values = (
        values.filter(
            (pl.col("Metric").str.contains("Count_")) & (pl.col("Subset") != "All")
        )
        .drop(["Metric", "Units"])
        .rename({"Value": "Count"})
    )

    # TODO: add a check for whether the # of unique combos of subset-lifespan differs from the combos of subset.
    # Calculations are not yet implemented for that case.
    years = values.get_column("Year").unique().sort()
    years = years.extend_constant(years.max() + 1, 1)
    age_values = pl.DataFrame(
        {"Age": range(1, 1 + lifespans.get_column("Lifespan").max())},
        schema=[("Age", pl.Int32)],
    )
    vintage_values = pl.DataFrame(
        {"Age": range(years.min() - age_values.get_column("Age").max(), years.max())},
        schema=[("Vintage", pl.Int32)],
    )

    year_zero_fleet_size = (
        values.filter(
            (pl.col("Metric") == "Count_Locomotives")
            & (pl.col("Subset").is_in(loco_types))
            & (pl.col("Year") == years.min())
        )
        .get_column("Value")
        .sum()
    )
    counts = (
        stock_values.select(item_id_cols)
        .unique()
        .join(stock_values.select("Year").unique(), how="cross")
        .join(stock_values, on=item_id_cols + ["Year"], how="left")
        .with_columns(cs.numeric().fill_null(0))
        .sort(item_id_cols + ["Year"])
    )
    item_list = counts.select(item_id_cols).unique()
    incumbents = (
        counts.filter(
            (pl.col("Subset").str.contains("Diesel")) & (pl.col("Year") == years.min())
        )
        .select(item_id_cols + ["Count"])
        .with_columns(
            pl.when(pl.col("Subset") == "Diesel_Large")
            .then(pl.lit(year_zero_fleet_size))
            .otherwise(pl.col("Count"))
            .alias("Count")
        )
    )

    # Initialize to only include incumbent asset counts; all else = 0
    # Start at year -1: all-incumbent fleet,
    # then immediately increment ages to get year 0 incumbent fleet
    age_tracker = (
        item_list.select(item_id_cols)
        .unique()
        .join(vintage_values, how="cross")
        .join(
            costs,
            how="left",
            left_on=item_id_cols + ["Vintage"],
            right_on=item_id_cols + ["Year"],
        )
        .join(
            lifespans,
            how="left",
            left_on=item_id_cols + ["Vintage"],
            right_on=item_id_cols + ["Year"],
        )
        .with_columns(
            pl.col("*").backward_fill().over(item_id_cols),
            (pl.lit(years.min()) - pl.col("Vintage")).alias("Age"),
        )
        .join(incumbents, how="left", on=item_id_cols)
        .with_columns(
            (pl.col("Count") * (pl.col("Age").truediv(pl.col("Lifespan"))))
            .round()
            .alias("Count"),
            (pl.col("Vintage") + pl.col("Lifespan") - 1).alias("Scheduled_Final_Year"),
        )
        .sort(item_id_cols + ["Age"])
        .with_columns(
            pl.when(pl.col("Age") == 1)
            .then(pl.col("Count"))
            .when(pl.col("Age") <= 0)
            .then(pl.lit(0.0))
            .otherwise(pl.col("Count") - (pl.col("Count").shift().over(item_id_cols)))
            .fill_null(0)
        )
    )

    portfolio_value_initial = (
        age_tracker.select(
            (
                pl.col("Count")
                * 1.0
                * (1.0 - pl.col("Age") * 1.0 / pl.col("Lifespan") * 1.0)
                * pl.col("Unit_Cost")
            )
            .sum()
            .alias("Value")
        )
        .with_columns(
            pl.lit("Asset_Value_Initial").alias("Metric"),
            pl.lit("USD").alias("Units"),
            pl.lit("All").alias("Subset"),
            pl.lit(years.min() - 1).alias("Year"),
        )
        .select(metric_columns)
    )

    # Increment ages to get year 0 incumbent fleet
    age_tracker = (
        age_tracker.with_columns(pl.col("Age") + 1)
        .filter((pl.col("Vintage") >= pl.lit(years.min())) | (pl.col("Count") > 0.0))
        .sort(item_id_cols + ["Scheduled_Final_Year"])
    )

    scheduled_retirements_year_zero = (
        age_tracker.filter(pl.col("Scheduled_Final_Year") == years.min() - 1)
        .group_by(item_id_cols)
        .agg(pl.col("Count").sum().alias("Value"))
        .with_columns(
            pl.lit("Retirements_Scheduled").alias("Metric"),
            pl.lit("assets").alias("Units"),
            pl.lit(years.min()).alias("Year"),
        )
        .select(metric_columns)
    )

    age_tracker = age_tracker.with_columns(
        pl.when(pl.col("Scheduled_Final_Year") < years.min())
        .then(0.0)
        .otherwise(pl.col("Count"))
        .alias("Count")
    )

    purchase_metrics = [portfolio_value_initial, scheduled_retirements_year_zero]
    for year in years:
        # If year 1, this includes incumbents only (prior-year retirements already removed)
        # So, we should do the same for subsequent years
        counts_prior_year_end = age_tracker.group_by(item_id_cols).agg(
            pl.col("Count").sum()
        )
        counts_current_year_start = counts.filter(pl.col("Year") == year).drop("Year")
        changes = (
            item_list.join(counts_prior_year_end, on=item_id_cols, how="left")
            .join(
                counts_current_year_start,
                on=item_id_cols,
                how="left",
                suffix="_Current",
            )
            .with_columns(cs.numeric().fill_null(0))
            .with_columns((pl.col("Count_Current") - pl.col("Count")).alias("Change"))
            .drop(["Count", "Count_Current"])
        )
        purchases = changes.filter(pl.col("Change") > 0).with_columns(
            pl.lit(1).alias("Age")
        )
        early_retirements = changes.filter(pl.col("Change") < 0).with_columns(
            pl.col("Change") * -1
        )
        age_tracker = (
            age_tracker.join(purchases, on=item_id_cols + ["Age"], how="left")
            .with_columns(
                pl.when((pl.col("Change") > 0))
                .then(pl.col("Count") + pl.col("Change"))
                .otherwise(pl.col("Count"))
                .alias("Count")
            )
            .drop("Change")
            .join(early_retirements, on=item_id_cols, how="left")
            .with_columns(
                pl.col("Count")
                .cum_sum()
                .over(item_id_cols)
                .alias("Retirements_Early_Cumsum")
            )
            .with_columns(
                pl.when(pl.col("Change") > 0)
                .then(
                    pl.when(pl.col("Retirements_Early_Cumsum") <= pl.col("Change"))
                    .then(pl.col("Count"))
                    .otherwise(
                        pl.max_horizontal(
                            [
                                0,
                                pl.min_horizontal(
                                    [
                                        pl.col("Count"),
                                        pl.col("Change")
                                        - (
                                            pl.col("Retirements_Early_Cumsum")
                                            - pl.col("Count")
                                        ),
                                    ]
                                ),
                            ]
                        )
                    )
                )
                .otherwise(pl.lit(0))
                .alias("Retirements_Early")
            )
            .with_columns(
                pl.when(pl.col("Retirements_Early") > 0)
                .then(pl.col("Count") - pl.col("Retirements_Early"))
                .otherwise(pl.col("Count"))
                .alias("Count")
            )
            .drop(["Change", "Retirements_Early_Cumsum"])
        )

        recovery_share = defaults.STRANDED_ASSET_RESALE_PCT
        retirement_label = "Retirements_Early"
        if year == years[len(years) - 1]:
            recovery_share = 1.0
            retirement_label = "Retirements_End_Of_Rollout"

        early_retirements = (
            age_tracker.filter(
                (pl.col("Retirements_Early") > 0)
                & (pl.col("Scheduled_Final_Year") >= year)
            )
            .group_by(item_id_cols)
            .agg(pl.col("Retirements_Early").sum().alias("Value"))
            .with_columns(
                pl.lit(retirement_label).alias("Metric"),
                pl.lit("assets").alias("Units"),
                pl.lit(year).alias("Year"),
            )
            .select(metric_columns)
        )

        scheduled_retirements = (
            age_tracker.filter(
                (pl.col("Scheduled_Final_Year") == year) & (pl.col("Count") > 0)
            )
            .group_by(item_id_cols)
            .agg(pl.col("Count").sum().alias("Value"))
            .with_columns(
                pl.lit("Retirements_Scheduled").alias("Metric"),
                pl.lit("assets").alias("Units"),
                pl.lit(year + 1).alias("Year"),
            )
            .select(metric_columns)
        )

        early_retirement_costs = (
            age_tracker.filter(pl.col("Retirements_Early") > 0)
            .with_columns(
                pl.lit("Cost_Retired_Assets").alias("Metric"),
                pl.lit("USD").alias("Units"),
                pl.lit(year).alias("Year"),
                pl.when(
                    (pl.col("Age") <= pl.lit(year - years.min() + 1))
                    | (defaults.INCLUDE_EXISTING_ASSETS)
                )
                .then(
                    pl.col("Retirements_Early")
                    * -1.0
                    * (1.0 - pl.col("Age") * 1.0 / pl.col("Lifespan") * 1.0)
                    * pl.col("Unit_Cost")
                    * pl.lit(recovery_share)
                )
                .otherwise(0.0)
                .alias("Value"),
            )
            .group_by("Metric", "Units", "Year", "Subset")
            .agg(pl.col("Value").sum())
            .select(metric_columns)
        )
        purchases = purchases.with_columns(
            pl.lit("Purchases").alias("Metric"),
            pl.lit("assets").alias("Units"),
            pl.lit(year).alias("Year"),
            pl.col("Change").alias("Value"),
        ).select(metric_columns)
        purchase_costs = (
            purchases.join(
                values.filter(pl.col("Metric") == "Unit_Cost"),
                on=item_id_cols + ["Year"],
                how="left",
            )
            .with_columns(
                (pl.col("Value") * pl.col("Value_right")).alias("Value"),
                pl.lit("Cost_Purchases").alias("Metric"),
                pl.lit("USD").alias("Units"),
            )
            .select(metric_columns)
        )
        # Remove temporary columns and increment item ages to prep for the next year
        age_tracker = age_tracker.with_columns(
            pl.col("Age") + 1,
            pl.when(pl.col("Scheduled_Final_Year") <= year)
            .then(0.0)
            .otherwise(pl.col("Count"))
            .alias("Count"),
        ).drop(["Retirements_Early"])
        # Add metrics for early retirements, purchases, and incurred costs
        purchase_metrics.extend(
            [
                scheduled_retirements,
                early_retirements,
                early_retirement_costs,
                purchases,
                purchase_costs,
            ]
        )

    values = pl.concat([values] + purchase_metrics, how="diagonal")

    total_costs = (
        values.filter(pl.col("Metric").is_in(["Cost_Retired_Assets", "Cost_Purchases"]))
        .group_by(["Metric", "Units", "Year"])
        .agg(pl.col("Value").sum())
        .with_columns(pl.lit("All").alias("Subset"))
    )
    return metrics_from_list([values, total_costs])


def calculate_rollout_total_costs(values: MetricType) -> MetricType:
    """
    Given multiple scenario years' locomotive fleet compositions, computes total per-year costs
    Arguments:
    ----------
    values: DataFrame with annual cost metrics
    Outputs:
    ----------
    DataFrame of across-year locomotive fleet composition metrics (metric name, units, value, and scenario year)
    """

    total_costs = (
        values.filter(
            (
                pl.col("Metric").is_in(
                    ["Cost_Energy", "Cost_Purchases", "Cost_Retired_Assets"]
                )
            )
            & (pl.col("Subset") == "All")
        )
        .group_by(pl.col("Units", "Subset", "Year"))
        .agg(pl.col("Value").sum())
        .with_columns(pl.lit("Cost_Total").alias("Metric"))
    )

    return metrics_from_list([values, total_costs])


def import_emissions_factors_cambium(
    location_map: Dict[str, List[alt.Location]],
    scenario_year: int,
    cambium_scenario: str = "MidCase",
    file_path: Path = defaults.GRID_EMISSIONS_FILE,
) -> pl.DataFrame:
    location_ids = [
        loc.to_pydict()["Location ID"]
        for loc_list in location_map.values()
        for loc in loc_list
    ]
    grid_emissions_regions = [
        loc.to_pydict()["Grid Emissions Region"]
        for loc_list in location_map.values()
        for loc in loc_list
    ]
    region_mappings = pl.DataFrame(
        {"Node": location_ids, "Region": grid_emissions_regions}
    ).unique()

    emissions_factors = (
        pl.scan_csv(source=file_path, skip_rows=5, dtypes={"t": pl.Int32})
        .rename({"t": "Year", "gea": "Region"})
        .filter(pl.col("scenario") == pl.lit(cambium_scenario))
        .select(
            pl.col("Region", "Year"),
            (pl.col("lrmer_co2e_c") + pl.col("lrmer_co2e_p")).alias("CO2eq_kg_per_MWh"),
            ((pl.col("Year") - pl.lit(scenario_year)).alias("Year_Diff")),
        )
        .join(region_mappings.lazy(), how="inner", on="Region")
        .drop("Region")
        .collect()
    )
    if emissions_factors.get_column("Year_Diff").abs().min() == 0:
        emissions_factors = emissions_factors.filter(pl.col("Year_Diff") == 0).drop(
            "Year_Diff"
        )
    else:
        earlier_year = emissions_factors.filter(pl.col("Year_Diff") < 0).filter(
            pl.col("Year_Diff") == pl.col("Year_Diff").max()
        )
        later_year = emissions_factors.filter(pl.col("Year_Diff") > 0).filter(
            pl.col("Year_Diff") == pl.col("Year_Diff").min()
        )
        if later_year.height == 0:
            return earlier_year
        elif earlier_year.height == 0:
            return later_year
        else:
            emissions_factors = earlier_year.join(later_year, on=["Node"], how="left")
            total_diff = (
                emissions_factors.get_column("Year_Diff").abs()
                + emissions_factors.get_column("Year_Diff_right").abs()
            )
            emissions_factors = emissions_factors.with_columns(
                (
                    pl.col("CO2eq_kg_per_MWh")
                    * (total_diff - pl.col("Year_Diff").abs())
                    / total_diff
                    + pl.col("CO2eq_kg_per_MWh_right")
                    * (total_diff - pl.col("Year_Diff_right").abs())
                    / total_diff
                ).alias("CO2eq_kg_per_MWh"),
                (
                    pl.col("Year")
                    * (total_diff - pl.col("Year_Diff").abs())
                    / total_diff
                    + pl.col("Year_right")
                    * (total_diff - pl.col("Year_Diff_right").abs())
                    / total_diff
                )
                .cast(pl.Int64, strict=False)
                .alias("Year"),
            ).drop(
                ["Year_right", "CO2eq_kg_per_MWh_right", "Year_Diff_right", "Year_Diff"]
            )

    return emissions_factors


def import_energy_prices_eia(
    location_map: Dict[str, List[alt.Location]],
    scenario_year: int,
    reference_year: int = 2022,
    reference_year_values: Dict[str, List] = {
        "Fuel": ["Diesel", "Electricity"],
        "Region": ["MN", "MN"],
        "Price": [3.616, 9.19],
    },
) -> pl.DataFrame:
    reference_year_values = pl.DataFrame(reference_year_values)
    location_ids = [
        loc.to_pydict()["Location ID"]
        for loc_list in location_map.values()
        for loc in loc_list
    ]
    electricity_price_regions = [
        loc.to_pydict()["Electricity Price Region"]
        for loc_list in location_map.values()
        for loc in loc_list
    ]
    liquid_fuel_price_regions = [
        loc.to_pydict()["Liquid Fuel Price Region"]
        for loc_list in location_map.values()
        for loc in loc_list
    ]
    nodal_reference_prices = pl.concat(
        [
            pl.DataFrame({"Node": location_ids, "Region": electricity_price_regions})
            .unique()
            .with_columns(pl.lit("Electricity").alias("Fuel")),
            pl.DataFrame({"Node": location_ids, "Region": liquid_fuel_price_regions})
            .unique()
            .with_columns(pl.lit("Diesel").alias("Fuel")),
        ],
        how="diagonal",
    ).join(reference_year_values, on=["Region", "Fuel"])

    eia_reference_electric_price = electricity_prices_eia.filter(
        pl.col("Year") == reference_year
    ).get_column("Price")[0]
    electric = electricity_prices_eia.with_columns(
        (pl.col("Price") / eia_reference_electric_price).alias("Price"),
        (pl.col("Year") - pl.lit(scenario_year)).alias("Year_Diff"),
    )

    eia_reference_liquid_fuel_price = liquid_fuel_prices_eia.filter(
        pl.col("Year") == reference_year
    ).get_column("Price")[0]
    liquid = liquid_fuel_prices_eia.with_columns(
        (pl.col("Price") / eia_reference_liquid_fuel_price).alias("Price"),
        (pl.col("Year") - pl.lit(scenario_year)).alias("Year_Diff"),
    )

    if electric.get_column("Year_Diff").abs().min() == 0:
        electric = electric.filter(pl.col("Year_Diff") == 0).drop("Year_Diff")
    else:
        earlier_year = electric.filter(pl.col("Year_Diff") < 0).filter(
            pl.col("Year_Diff") == pl.col("Year_Diff").max()
        )
        later_year = electric.filter(pl.col("Year_Diff") > 0).filter(
            pl.col("Year_Diff") == pl.col("Year_Diff").min()
        )
        if later_year.height == 0:
            electric = earlier_year
        elif earlier_year.height == 0:
            electric = later_year
        else:
            electric = earlier_year.join(later_year, on=["Node"], how="left")
            total_diff = (
                electric.get_column("Year_Diff").abs()
                + electric.get_column("Year_Diff_right").abs()
            )
            electric = electric.with_columns(
                (
                    pl.col("Price")
                    * (total_diff - pl.col("Year_Diff").abs())
                    / total_diff
                    + pl.col("Price_right")
                    * (total_diff - pl.col("Year_Diff_right").abs())
                    / total_diff
                ).alias("Price"),
                (
                    pl.col("Year")
                    * (total_diff - pl.col("Year_Diff").abs())
                    / total_diff
                    + pl.col("Year_right")
                    * (total_diff - pl.col("Year_Diff_right").abs())
                    / total_diff
                )
                .cast(pl.Int64, strict=False)
                .alias("Year"),
            ).drop(["Year_right", "Price_right", "Year_Diff_right", "Year_Diff"])

    if liquid.get_column("Year_Diff").abs().min() == 0:
        liquid = liquid.filter(pl.col("Year_Diff") == 0).drop("Year_Diff")
    else:
        earlier_year = liquid.filter(pl.col("Year_Diff") < 0).filter(
            pl.col("Year_Diff") == pl.col("Year_Diff").max()
        )
        later_year = liquid.filter(pl.col("Year_Diff") > 0).filter(
            pl.col("Year_Diff") == pl.col("Year_Diff").min()
        )
        if later_year.height == 0:
            liquid = earlier_year
        elif earlier_year.height == 0:
            liquid = later_year
        else:
            liquid = earlier_year.join(later_year, on=["Node"], how="left")
            total_diff = (
                liquid.get_column("Year_Diff").abs()
                + liquid.get_column("Year_Diff_right").abs()
            )
            liquid = liquid.with_columns(
                (
                    pl.col("Price")
                    * (total_diff - pl.col("Year_Diff").abs())
                    / total_diff
                    + pl.col("Price_right")
                    * (total_diff - pl.col("Year_Diff_right").abs())
                    / total_diff
                ).alias("Price"),
                (
                    pl.col("Year")
                    * (total_diff - pl.col("Year_Diff").abs())
                    / total_diff
                    + pl.col("Year_right")
                    * (total_diff - pl.col("Year_Diff_right").abs())
                    / total_diff
                )
                .cast(pl.Int64, strict=False)
                .alias("Year"),
            ).drop(["Year_right", "Price_right", "Year_Diff_right", "Year_Diff"])

    liquid_multiplier = liquid.get_column("Price")[0]
    electric_multiplier = electric.get_column("Price")[0]

    nodal_energy_prices = nodal_reference_prices.with_columns(
        pl.when(pl.col("Fuel") == pl.lit("Electricity"))
        .then(pl.col("Price") * electric_multiplier)
        .otherwise(pl.col("Price") * liquid_multiplier)
        .alias("Price")
    ).select(["Node", "Fuel", "Price"])
    return nodal_energy_prices


def add_battery_costs(loco_info: pd.DataFrame, year: int) -> pd.DataFrame:
    prices_to_use = battery_prices_nrel_atb.filter(
        (pl.col("Year") - pl.lit(year)).abs().min()
        == (pl.col("Year") - pl.lit(year)).abs()
    )
    usd_per_kWh = prices_to_use.filter(pl.col("Metric") == "USD_Per_kWh").get_column(
        "Value"
    )[0]
    usd_per_kW = prices_to_use.filter(pl.col("Metric") == "USD_Per_kW").get_column(
        "Value"
    )[0]
    usd_constant = prices_to_use.filter(pl.col("Metric") == "USD_Constant").get_column(
        "Value"
    )[0]

    for idx, row in loco_info.iterrows():
        loco_dict = row["Rust_Loco"].to_pydict()
        loco_type = next(iter(loco_dict["loco_type"].values()))
        if "res" in loco_type:
            kW = loco_type["res"]["pwr_out_max_watts"] / 1000
            kWh = (
                loco_type["res"]["energy_capacity_joules"] * 1e-6 * utilities.KWH_PER_MJ
            )
            loco_info.at[idx, "Cost_USD"] = (
                defaults.BEL_MINUS_BATTERY_COST_USD
                + usd_constant
                + kW * usd_per_kW
                + kWh * usd_per_kWh
            ) * defaults.RETAIL_PRICE_EQUIVALENT_MULTIPLIER
    return loco_info


def calculate_meet_pass_events(info: ScenarioInfo, units: str) -> MetricType:
    slts_results = []
    i = 0
    for slts in info.sims.to_pydict():
        if len(slts["history"]) == 0:
            print(
                "No SpeedLimitTrainSim history was saved, so meet pass events cannot be counted."
            )
            return metric("Meet_Pass_Events", units, None)

        # NOTE: this code is somewhat clunky and would probably benifit from streamlining
        df = alt.SpeedLimitTrainSim.from_pydict(slts).to_dataframe()
        if ("history.time" not in df.collect_schema()) or (
            "history.speed" not in df.collect_schema()
        ):
            print(
                "SpeedLimitTrainSim history doesn't include time and/or speed, so meet pass events cannot be counted."
            )
            return metric("Meet_Pass_Events", units, None)

        df = df.select("history.time", "history.speed").with_columns(
            pl.lit(i).alias("train_idx")
        )
        if df.height > 0:
            slts_results.append(df)
        i += 1
    all = pl.concat(slts_results, how="diagonal_relaxed")
    val = (
        all.sort("train_idx", "history.speed")
        .with_columns(
            (
                pl.col("history.speed")
                - pl.col("history.speed").shift(1).over("train_idx").fill_null(0.0)
            ).alias("speed_change")
        )
        .with_columns(
            pl.when(pl.col("speed_change") < 0, pl.col("history.speed") == 0)
            .then(1)
            .otherwise(0)
            .alias("stop")
        )
        .group_by("train_idx")
        .agg(
            (pl.max_horizontal([0, pl.col("stop").sum() - 1]).alias("meet_pass_stops"))
        )
        .get_column("meet_pass_stops")
        .sum()
    )
    return metric("Meet_Pass_Events", units, val)


function_mappings = {
    "Meet_Pass_Events": calculate_meet_pass_events,
    "Energy_Costs": calculate_energy_cost,
    "Freight_Moved": calculate_freight_moved,
    "Energy_Per_Freight_Moved": calculate_energy_per_freight,
    "GHG": calculate_ghg,
    "Count_Locomotives": calculate_locomotive_counts,
    "Count_Refuelers": calculate_refueler_counts,
}
