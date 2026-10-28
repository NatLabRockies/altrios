import pandas as pd
import polars as pl
import polars.selectors as cs
import math
from typing import Any, Callable, List, Dict, Tuple, Union, Optional
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
# Every ATB battery-cost scenario. "Moderate" is the central case; "Advanced" is
# the low-cost case (about 8% under Moderate in 2025, widening to ~40% by 2035 and
# holding there) and "Conservative" the high one.
battery_prices_nlr_atb_all = pl.read_csv(source=defaults.BATTERY_PRICE_FILE)
BATTERY_SCENARIOS = sorted(battery_prices_nlr_atb_all["Scenario"].unique().to_list())


def battery_prices(scenario: str = "Moderate") -> pl.DataFrame:
    """Return the ATB battery-cost table for one scenario.

    Args:
        scenario: One of `BATTERY_SCENARIOS`, e.g. "Moderate" or "Advanced".

    Returns:
        Rows of the ATB battery price table for `scenario`.

    Raises:
        ValueError: When `scenario` is not in `BATTERY_SCENARIOS`.
    """
    if scenario not in BATTERY_SCENARIOS:
        raise ValueError(f"unknown battery scenario {scenario!r}; have {BATTERY_SCENARIOS}")
    return battery_prices_nlr_atb_all.filter(pl.col("Scenario") == scenario)


# Backwards-compatible view: the central case, as before.
battery_prices_nlr_atb = battery_prices("Moderate")
# Regional price zones (US states / PADD regions / Canadian provinces): an observed
# reference-year price per zone and the name of the trajectory that carries it through
# time. Canadian trajectories come from Canada's Energy Future 2026 (CER).
energy_price_zones = pl.read_csv(source=defaults.ENERGY_PRICE_ZONE_FILE)
cer_end_use_prices = pl.read_csv(source=defaults.CER_ENERGY_PRICE_FILE)

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

    - `agg_cache`: Optional `SimAggregateCache` shared across scenarios so that
    sims-derived, year-invariant aggregates are computed once even when the same
    `sims` object is reused across many scenario years. Populated automatically by
    `main`; callers normally leave it `None`.
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
    infrastructure_costs: pl.DataFrame = None
    agg_cache: Optional["SimAggregateCache"] = None


class SimAggregateCache:
    """Memoize year-invariant, sims-derived aggregates across scenario years.

    Entries are keyed by the sims object's identity, the `annualize` flag and the
    aggregate name; each sims object is retained so its `id` stays valid while
    the cache lives. `main` attaches one cache to every `ScenarioInfo` it builds.
    """

    def __init__(self) -> None:
        """Create an empty cache."""
        self._values: Dict[tuple, object] = {}
        self._keepalive: Dict[int, object] = {}

    def get(
        self,
        sims: Union[alt.SpeedLimitTrainSim, alt.SpeedLimitTrainSimVec],
        annualize: bool,
        name: str,
        compute: Callable[[], Any],
    ) -> Any:
        """Return a cached aggregate, computing it on first use.

        Args:
            sims: Simulations whose identity forms part of the cache key.
            annualize: Annualize flag that forms part of the cache key.
            name: Aggregate name that forms part of the cache key.
            compute: Zero-argument callable producing the value on a miss.

        Returns:
            The cached or freshly computed value.
        """
        sims_id = id(sims)
        self._keepalive.setdefault(sims_id, sims)
        key = (sims_id, bool(annualize), name)
        if key not in self._values:
            self._values[key] = compute()
        return self._values[key]


def _cached_sim_aggregate(
    info: "ScenarioInfo",
    name: str,
    compute: Callable[[], Any],
    *,
    annualize: Optional[bool] = None,
) -> Any:
    """Return a sims-derived aggregate, via `info.agg_cache` when present.

    Args:
        info: Scenario whose `sims` and `agg_cache` are used.
        name: Aggregate name used in the cache key.
        compute: Zero-argument callable producing the value on a cache miss.
        annualize: Overrides `info.annualize` in the cache key; pass False for
            raw quantities shared across annualize variants.

    Returns:
        The cached or freshly computed value.
    """
    resolved = info.annualize if annualize is None else annualize
    cache = getattr(info, "agg_cache", None)
    if cache is None:
        return compute()
    return cache.get(info.sims, resolved, name, compute)


def annualizer(info: "ScenarioInfo") -> float:
    """Return the scaler from simulated window to one year.

    Sims-derived aggregates are read raw from Rust and scaled by this factor.

    Args:
        info: Scenario with `annualize` and `simulation_days`.

    Returns:
        `365.25 / simulation_days` when annualizing, else 1.0.

    Raises:
        ValueError: When annualizing with `simulation_days` unset or zero.
    """
    if not info.annualize:
        return 1.0
    if not info.simulation_days:
        raise ValueError("annualize=True needs a positive ScenarioInfo.simulation_days")
    return 365.25 / float(info.simulation_days)


def _sim_energy_fuel_joules(info: "ScenarioInfo") -> float:
    """Return annualized fuel energy (J) from the sims.

    Args:
        info: Scenario supplying `sims` and the annualization factor.

    Returns:
        Fuel energy (J), scaled by `annualizer(info)`.
    """
    return annualizer(info) * _cached_sim_aggregate(
        info, "energy_fuel_joules",
        lambda: info.sims.get_energy_fuel_joules(annualize=False), annualize=False)


def _sim_net_energy_res_joules(info: "ScenarioInfo") -> float:
    """Return annualized net battery energy (J) from the sims.

    Args:
        info: Scenario supplying `sims` and the annualization factor.

    Returns:
        Net reversible-energy-storage energy (J), scaled by `annualizer(info)`.
    """
    return annualizer(info) * _cached_sim_aggregate(
        info, "net_energy_res_joules",
        lambda: info.sims.get_net_energy_res_joules(annualize=False), annualize=False)


def _sim_megagram_kilometers(info: "ScenarioInfo") -> float:
    """Return annualized freight megagram-kilometers from the sims.

    Args:
        info: Scenario supplying `sims` and the annualization factor.

    Returns:
        Megagram-kilometers, scaled by `annualizer(info)`.
    """
    return annualizer(info) * _cached_sim_aggregate(
        info, "megagram_kilometers",
        lambda: info.sims.get_megagram_kilometers(annualize=False), annualize=False)


def _sim_car_kilometers(info: "ScenarioInfo") -> float:
    """Return annualized car-kilometers from the sims.

    Args:
        info: Scenario supplying `sims` and the annualization factor.

    Returns:
        Car-kilometers, scaled by `annualizer(info)`.
    """
    return annualizer(info) * _cached_sim_aggregate(
        info, "car_kilometers",
        lambda: info.sims.get_car_kilometers(annualize=False), annualize=False)


def _sim_cars_moved(info: "ScenarioInfo") -> float:
    """Return the annualized count of cars moved.

    Args:
        info: Scenario supplying `sims` and the annualization factor.

    Returns:
        Cars moved, scaled by `annualizer(info)`.
    """
    return annualizer(info) * _cached_sim_aggregate(
        info, "cars_moved",
        lambda: info.sims.get_cars_moved(annualize=False), annualize=False)


def _sim_catenary_joules_for(info: "ScenarioInfo") -> Tuple[float, float]:
    """Return raw catenary draw and line loss (J).

    Args:
        info: Scenario supplying `sims` and an optional `agg_cache`.

    Returns:
        `(pantograph_delivered_J, line_loss_J)`, cached per sims object.
    """
    # Raw joules (annualize-independent); the heavy `to_pydict()` read runs at most
    # once per unique sims object rather than once per scenario year.
    return _cached_sim_aggregate(
        info, "catenary_joules",
        lambda: _sim_catenary_joules(info.sims), annualize=False)


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
    annual_metrics: Union[Tuple[str, str], List[Tuple[str, str]]] = [
        ("Meet_Pass_Events", "count"),
        ("Dwell_Time", "seconds"),
        ("Dwell_Fuel_Energy", "MJ"),
        ("APU_Fuel_Consumption", "MJ"),
        ("Freight_Moved", "million tonne-mi"),
        ("Freight_Moved", "million tonne-km"),
        ("Freight_Moved", "car-miles"),
        ("Freight_Moved", "cars"),
        ("GHG", "tonne CO2-eq"),
        ("Count_Locomotives", "assets"),
        ("Count_Refuelers", "assets"),
        ("Energy_Costs", "USD"),
        ("Energy_Per_Freight_Moved", "kWh per car-mile"),
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

    # Share one aggregate cache across all scenarios so any sims object reused
    # across scenario years (e.g. a tiled multi-year rollout) has its Rust-FFI /
    # full-serialization aggregates computed once, not once per year.
    agg_cache = SimAggregateCache()
    for scenario_info in scenario_infos:
        scenario_info.agg_cache = agg_cache

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
    Given a DataFrame of each year's costs and gross freight deliveries, computes the multi-year levelized cost of freight delivered, per gross tonne-km (LCOTKM) and per gross tonne-mile (LCOTMI).
    Arguments:
    ----------
    values: DataFrame containing total costs and gross freight deliveries for each modeled scenario year
    Outputs:
    ----------
    DataFrame of LCOTKM and LCOTMI results (metric name, units, value, and scenario year)
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
        .rename({"Value": "Cost_Total"})
    )
    tkm_timeseries = (
        values.filter(
            (pl.col("Metric") == pl.lit("Freight_Moved"))
            & (pl.col("Units") == pl.lit("million tonne-km"))
            & (pl.col("Subset") == pl.lit("All"))
        )
        .select(["Year", "Value"])
        .rename({"Value": "Mt-km"})
    )
    tmi_timeseries = (
        values.filter(
            (pl.col("Metric") == pl.lit("Freight_Moved"))
            & (pl.col("Units") == pl.lit("million tonne-mi"))
            & (pl.col("Subset") == pl.lit("All"))
        )
        .select(["Year", "Value"])
        .rename({"Value": "Mt-mi"})
    )
    timeseries = (
        cost_timeseries.join(tkm_timeseries, on="Year", how="full", coalesce=True)
        .join(tmi_timeseries, on="Year", how="full", coalesce=True)
    )
    timeseries = (
        timeseries.with_columns(
            (pl.col("Year") - pl.col("Year").min()).alias("Year_Offset")
        )
        .with_columns(
            ((1 + defaults.DISCOUNT_RATE) ** pl.col("Year_Offset")).alias(
                "Discounting_Factor"
            )
        )
        .with_columns(
            (pl.col("Cost_Total") / pl.col("Discounting_Factor")).alias(
                "Cost_Total_Discounted"
            ),
            (pl.col("Mt-km") / pl.col("Discounting_Factor")).alias("Mt-km_Discounted"),
            (pl.col("Mt-mi") / pl.col("Discounting_Factor")).alias("Mt-mi_Discounted"),
        )
        .with_columns(pl.col("Year").cast(pl.Utf8))
    )

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
    tmi_total = timeseries.get_column("Mt-mi_Discounted").sum()
    lcotmi_all = cost_minus_residual_baseline / tmi_total if tmi_total > 0 else math.nan

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
    tmi_discounted = timeseries.select(
        pl.col("Year"),
        Value=pl.col("Mt-mi_Discounted"),
        Metric=pl.lit("Mt-mi_Discounted"),
        Subset=pl.lit("All"),
        Units=pl.lit("million tonne-mi (discounted)"),
    )
    cotkm_annual = timeseries.select(
        pl.col("Year"),
        Value=pl.col("Cost_Total") / pl.col("Mt-km"),
        Metric=pl.lit("Cost_Per_Mt-km"),
        Subset=pl.lit("All"),
        Units=pl.lit("USD per million tonne-km"),
    )
    cotmi_annual = timeseries.select(
        pl.col("Year"),
        Value=pl.col("Cost_Total") / pl.col("Mt-mi"),
        Metric=pl.lit("Cost_Per_Mt-mi"),
        Subset=pl.lit("All"),
        Units=pl.lit("USD per million tonne-mi"),
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
            tmi_discounted,
            cotkm_annual,
            cotmi_annual,
            discount,
            metric("LCOTKM", "USD per million tonne-km (levelized)", lcotkm_all),
            metric("LCOTMI", "USD per million tonne-mi (levelized)", lcotmi_all),
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
    # Guard against zero / missing freight (e.g. an unwalked or empty sim), which
    # would otherwise raise ZeroDivisionError.
    energy_per_freight = total_energy / freight_val if freight_val else None
    return metrics_from_list(
        [
            diesel_mj,
            electricity_mj,
            metric("Energy_Use", "MJ", total_mj),
            metric("Energy_Per_Freight_Moved", units, energy_per_freight),
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
                _sim_energy_fuel_joules(info)
                / 1e3
                / lhv_kj_per_kg
            )
            * 1e3
            / rho_fuel_g_per_gallon
        )
    elif units == "MJ":
        val = _sim_energy_fuel_joules(info) / 1e6
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
            _sim_net_energy_res_joules(info)
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
            .unpivot(
                index=["Metric", "Node"],
                on=units,
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
            _sim_megagram_kilometers(info)
            * conversion_from_km
            / 1.0e6,
            year=info.scenario_year,
        )
    elif units in ["car-km", "car-miles"]:
        return metric(
            "Freight_Moved",
            units,
            _sim_car_kilometers(info) * conversion_from_km,
            year=info.scenario_year,
        )
    elif units == "cars":
        return metric(
            "Freight_Moved",
            units,
            _sim_cars_moved(info),
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
            _sim_car_kilometers(info) * conversion_from_km
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
                    "Empty",
                    year=info.scenario_year,
                ),
            ]
        )
    elif units == "detailed car counts":
        kilometers = (
            pl.DataFrame(
                data={
                    "car-km": [
                        alt.SpeedLimitTrainSim.from_pydict(sim).get_kilometers(annualize=False)
                        * annualizer(info)
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
            .unpivot(index="idx", value_name="cars", variable_name="Subset")
            .filter(pl.col("cars").is_not_null())
            .join(kilometers, how="left", on="idx")
            .drop("idx")
            .group_by("Subset")
            .agg(pl.col("*").sum())
            .sort("Subset")
            .unpivot(index="Subset", variable_name="Units", value_name="Value")
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


def calculate_line_loss_ghg(
    line_loss_df: pl.DataFrame,
    emissions_factors: pl.DataFrame,
    scenario_year: int = defaults.BASE_ANALYSIS_YEAR,
    units: str = "tonne CO2-eq",
) -> MetricType:
    """Compute GHG from catenary line losses by grid region.

    Losses belong to the substation's grid region, which may differ from the train's.

    Args:
        line_loss_df: Output of `altrios.energization.compute_line_losses`, with
            `district_id`, `grid_emissions_region` and `line_loss_energy_joules`;
            the `"__total__"` row is skipped.
        emissions_factors: `CO2eq_kg_per_MWh` per `Node`, optionally per `Year`.
        scenario_year: Year selecting the emissions factor when `Year` exists.
        units: Only "tonne CO2-eq" is supported; any other value yields None.

    Returns:
        `GHG_Line_Loss` rows, one `Subset` per grid region plus "All"; regions
        lacking a factor contribute None and are reported with a printed warning.
    """
    if units != "tonne CO2-eq":
        return metric("GHG_Line_Loss", units, None)

    per_region = (
        line_loss_df.lazy()
        .filter(
            (pl.col("district_id") != pl.lit("__total__"))
            & pl.col("grid_emissions_region").is_not_null()
        )
        .group_by("grid_emissions_region")
        .agg(
            (pl.col("line_loss_energy_joules").sum() / 3.6e9).alias("MWh")
        )
        .collect()
    )
    if per_region.height == 0:
        return metric("GHG_Line_Loss", units, None)

    ef = emissions_factors
    if "Year" in ef.columns:
        ef = ef.filter(pl.col("Year") == scenario_year)
    joined = per_region.join(
        ef.select(["Node", "CO2eq_kg_per_MWh"]),
        left_on="grid_emissions_region",
        right_on="Node",
        how="left",
    )
    missing = joined.filter(pl.col("CO2eq_kg_per_MWh").is_null())
    if missing.height > 0:
        print(
            "No emissions factor for line-loss grid region(s): "
            f"{missing.get_column('grid_emissions_region').to_list()}; "
            "their line-loss GHGs were not calculated."
        )

    rows: List[MetricType] = []
    total = 0.0
    have_any = False
    for region, mwh, factor in zip(
        joined.get_column("grid_emissions_region"),
        joined.get_column("MWh"),
        joined.get_column("CO2eq_kg_per_MWh"),
    ):
        if factor is None:
            rows.append(metric("GHG_Line_Loss", units, None, subset=region))
            continue
        tonnes = mwh * factor / 1000.0
        total += tonnes
        have_any = True
        rows.append(metric("GHG_Line_Loss", units, tonnes, subset=region))
    rows.append(
        metric("GHG_Line_Loss", units, total if have_any else None, subset="All")
    )
    return metrics_from_list(rows)


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
            .unpivot(index="Locomotive_Type", variable_name="Metric", value_name="Value")
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
        .unpivot(index="Refueler_Type", variable_name="Metric", value_name="Value")
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


def calculate_infrastructure_counts(info: ScenarioInfo, _: str) -> MetricType:
    """Count electrification infrastructure assets with cost and lifespan.

    Args:
        info: Scenario whose `infrastructure_costs` lists one row per asset with
            `Infrastructure_Type`, `Cost_USD` and `Lifespan_Years`.
        _: Units argument, unused.

    Returns:
        Per `Infrastructure_Type`: `Count_Infrastructure` (assets), `Unit_Cost`
        (USD, mean) and `Lifespan` (years, mean); a None metric when
        `infrastructure_costs` is None.
    """
    if info.infrastructure_costs is None:
        return metric("Count_Infrastructure", "assets", None)

    counts = (
        info.infrastructure_costs.group_by(["Infrastructure_Type"])
        .agg(
            pl.col("Cost_USD").mean().alias("Unit_Cost"),
            pl.len().alias("Count_Infrastructure"),
            pl.col("Lifespan_Years").mean().alias("Lifespan"),
        )
        .unpivot(index="Infrastructure_Type", variable_name="Metric", value_name="Value")
        .with_columns(
            pl.when(pl.col("Metric") == "Unit_Cost")
            .then(pl.lit("USD"))
            .when(pl.col("Metric") == "Lifespan")
            .then(pl.lit("years"))
            .otherwise(pl.lit("assets"))
            .alias("Units"),
            pl.lit("All").alias("Year"),
        )
        .rename({"Infrastructure_Type": "Subset"})
    )

    return counts


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
        # backward_fill below takes the next newer vintage's values, so the
        # frame must be in Vintage order within each item; do not rely on the
        # joins above preserving it.
        .sort(item_id_cols + ["Vintage"])
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
            # The cumulative sum allocates early retirements oldest-first, so
            # re-assert the order rather than rely on the joins preserving it.
            .sort(item_id_cols + ["Scheduled_Final_Year"])
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


def interpolate_emissions_factors_cambium(
    regions: List[str],
    years: List[int],
    cambium_scenario: str = "MidCase",
    file_path: Path = defaults.GRID_EMISSIONS_FILE,
) -> pl.DataFrame:
    """Interpolate Cambium marginal CO2e factors by region and year.

    Args:
        regions: Cambium GEA region names.
        years: Years to interpolate to.
        cambium_scenario: Cambium scenario name, e.g. "MidCase".
        file_path: Cambium GEA CSV with `scenario`, `gea`, `t`, `lrmer_co2e_c`
            and `lrmer_co2e_p` columns.

    Returns:
        `Node` (region), `Year` and `CO2eq_kg_per_MWh` (combustion plus
        pre-combustion LRMER), linearly interpolated between sample years and
        clamped at the endpoints; regions absent from the table are omitted.
    """
    src = (
        pl.scan_csv(source=file_path, skip_rows=5)
        .filter(
            (pl.col("scenario") == pl.lit(cambium_scenario))
            & pl.col("gea").is_in(list(regions))
        )
        .select(
            pl.col("gea").alias("Node"),
            pl.col("t").cast(pl.Int64).alias("SampleYear"),
            (pl.col("lrmer_co2e_c") + pl.col("lrmer_co2e_p")).alias("CO2eq_kg_per_MWh"),
        )
        .collect()
    )
    rows = []
    for region in regions:
        sub = src.filter(pl.col("Node") == region).sort("SampleYear")
        if sub.height == 0:
            continue
        sample_years = sub.get_column("SampleYear").to_list()
        factors = sub.get_column("CO2eq_kg_per_MWh").to_list()
        for year in years:
            rows.append({
                "Node": region,
                "Year": int(year),
                "CO2eq_kg_per_MWh": _interp_clamped(sample_years, factors, int(year)),
            })
    return pl.DataFrame(rows)


def _interp_clamped(xs: List[float], ys: List[float], x: float) -> Optional[float]:
    """Interpolate `ys` at `x` linearly, clamped to the endpoints.

    Args:
        xs: Sorted sample points.
        ys: Values at `xs`.
        x: Point to evaluate.

    Returns:
        The interpolated value, or None when `xs` is empty.
    """
    if not xs:
        return None
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


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

    emissions_factors = interpolate_emissions_factors_cambium(
        regions=region_mappings.get_column("Region").unique().to_list(),
        years=[scenario_year],
        cambium_scenario=cambium_scenario,
        file_path=file_path,
    ).rename({"Node": "Region"})

    return (
        region_mappings.join(emissions_factors, how="inner", on="Region")
        .select("Node", "Year", "CO2eq_kg_per_MWh")
    )


def _series_price_at(series: pl.DataFrame, year: float) -> float:
    """Return a Year/Price table's price at `year`.

    Args:
        series: Frame with `Year` and `Price` columns.
        year: Year to evaluate; clamped to the tabulated range.

    Returns:
        Price interpolated linearly between tabulated years.

    Raises:
        ValueError: When the series has no rows after dropping nulls.
    """
    s = series.select("Year", "Price").drop_nulls().sort("Year")
    years = s.get_column("Year").cast(pl.Float64).to_list()
    prices = s.get_column("Price").cast(pl.Float64).to_list()
    if not years:
        raise ValueError("empty price series")
    y = float(year)
    if y <= years[0]:
        return prices[0]
    if y >= years[-1]:
        return prices[-1]
    for y0, p0, y1, p1 in zip(years, prices, years[1:], prices[1:]):
        if y0 <= y <= y1:
            return p0 if y1 == y0 else p0 + (p1 - p0) * (y - y0) / (y1 - y0)
    return prices[-1]


def price_trajectory(trajectory: str, fuel: str) -> pl.DataFrame:
    """Return the Year/Price series a price zone's `Trajectory` names.

    Only ratios of the series are used, so its units and dollar year do not matter.

    Args:
        trajectory: `EIA_AEO...` for the national AEO reference case, or
            `CER_EF2026:<Province>` for Canada's Energy Future 2026 Current
            Measures end-use price (2022 CAD/GJ).
        fuel: "Electricity" (industrial sector) or "Diesel" (transportation).

    Returns:
        Frame with `Year` and `Price` columns.

    Raises:
        ValueError: When `trajectory` is unknown or has no CER series.
    """
    if trajectory.startswith("EIA_AEO"):
        return electricity_prices_eia if fuel == "Electricity" else liquid_fuel_prices_eia
    if trajectory.startswith("CER_EF2026:"):
        province = trajectory.split(":", 1)[1]
        variable, sector = (
            ("Electricity", "Industrial") if fuel == "Electricity"
            else ("Diesel", "Transportation")
        )
        series = cer_end_use_prices.filter(
            (pl.col("Scenario") == "Current Measures")
            & (pl.col("Region") == province)
            & (pl.col("Variable") == variable)
            & (pl.col("Sector") == sector)
        ).select("Year", pl.col("Value").alias("Price"))
        if series.height == 0:
            raise ValueError(
                f"no CER EF2026 {variable}/{sector} series for {province!r} "
                f"in {defaults.CER_ENERGY_PRICE_FILE}"
            )
        return series
    raise ValueError(f"unknown price trajectory {trajectory!r}")


def zone_energy_price(zone: str, fuel: str, year: int) -> float:
    """Return a price zone's `fuel` price at `year`.

    The reference price is scaled by the trajectory ratio between the two years.

    Args:
        zone: `Zone` in `Energy_Price_Zones.csv`.
        fuel: "Diesel" or "Electricity".
        year: Scenario year.

    Returns:
        Diesel in USD/gal after subtracting
        `defaults.RAILROAD_DIESEL_DEDUCTION_USD_PER_GAL` (floored at 1.0);
        electricity in US cents/kWh. Real 2025 USD.

    Raises:
        KeyError: When no row matches `zone` and `fuel`.
    """
    row = energy_price_zones.filter((pl.col("Zone") == zone) & (pl.col("Fuel") == fuel))
    if row.height == 0:
        raise KeyError(f"no {fuel} price zone {zone!r} in {defaults.ENERGY_PRICE_ZONE_FILE}")
    row = row.row(0, named=True)
    trajectory = price_trajectory(row["Trajectory"], fuel)
    reference = _series_price_at(trajectory, int(row["Reference_Year"]))
    price = float(row["Reference_Price"]) * _series_price_at(trajectory, year) / reference
    if fuel == "Diesel":
        # ESCALATE FIRST, THEN STRIP. The reference tables and the AEO/CER
        # trajectories are all retail pump prices; a railroad buys at a delivered
        # price roughly `RAILROAD_DIESEL_DEDUCTION_USD_PER_GAL` below that, and
        # that spread is a per-gallon wedge (fixed excise taxes, station margin,
        # rack-to-pump distribution), not a percentage of the fuel price. Holding
        # it constant in real terms is therefore the defensible assumption, and
        # subtracting it AFTER the trajectory is what does that.
        #
        # It also gets the escalation right. If retail grows 11.5% to 2050 while
        # a fixed wedge does not, the commodity underneath grew faster than
        # 11.5%; the railroad's price, which excludes most of that wedge,
        # escalates faster too (about 17% on the US zones). Stripping BEFORE the
        # trajectory would instead escalate the railroad price at the retail
        # rate, implicitly assuming the spread grows with crude -- which
        # understates late-horizon diesel cost by ~5% on the US zones and
        # flatters diesel against the electric arms.
        #
        # Floored at a dollar a gallon rather than allowed to go negative: a
        # deduction larger than a zone's price means the deduction and the
        # reference disagree about what they measure, and silently pricing fuel
        # at zero would hide that.
        price = max(price - defaults.RAILROAD_DIESEL_DEDUCTION_USD_PER_GAL, 1.0)
    return price


def location_price_zones(location_map: Dict[str, List[alt.Location]]) -> pl.DataFrame:
    """Return each location's electricity and diesel price zones.

    Args:
        location_map: Locations by node, each with `Location ID`,
            `Electricity Price Region` and `Liquid Fuel Price Region`.

    Returns:
        Unique rows of `Node`, `Electricity_Zone` and `Diesel_Zone`.
    """
    rows = []
    for loc_list in location_map.values():
        for loc in loc_list:
            d = loc.to_pydict()
            rows.append({
                "Node": d["Location ID"],
                "Electricity_Zone": d["Electricity Price Region"],
                "Diesel_Zone": d["Liquid Fuel Price Region"],
            })
    return pl.DataFrame(rows).unique(maintain_order=True)


def import_energy_prices_eia(
    location_map: Dict[str, List[alt.Location]],
    scenario_year: int,
    reference_year: int = defaults.EIA_PRICE_REFERENCE_YEAR,
    reference_year_values: Optional[Dict[str, List]] = None,
) -> pl.DataFrame:
    """Nodal energy prices (``Node``, ``Fuel``, ``Price``) for ``scenario_year``.

    Each location's price is its price zone's price at that year (see
    :func:`zone_energy_price`): electricity in US cents/kWh, diesel in USD/gal.
    ``reference_year_values`` is a legacy override -- ``{"Fuel": [...], "Region": [...],
    "Price": [...]}`` pins ``reference_year`` prices for the named regions, which are
    then carried by the national AEO trajectory as before.
    """
    overrides = (
        pl.DataFrame(reference_year_values) if reference_year_values is not None else None
    )
    out = []
    for r in location_price_zones(location_map).iter_rows(named=True):
        for fuel, zone in (("Electricity", r["Electricity_Zone"]), ("Diesel", r["Diesel_Zone"])):
            price = None
            if overrides is not None:
                hit = overrides.filter((pl.col("Fuel") == fuel) & (pl.col("Region") == zone))
                if hit.height:
                    aeo = price_trajectory("EIA_AEO", fuel)
                    price = (
                        float(hit.get_column("Price")[0])
                        * _series_price_at(aeo, scenario_year)
                        / _series_price_at(aeo, reference_year)
                    )
            if price is None:
                price = zone_energy_price(zone, fuel, scenario_year)
            out.append({"Node": r["Node"], "Fuel": fuel, "Price": price})
    return pl.DataFrame(out)


# --------------------------------------------------------------- battery pack
# The ATB's `USD_Per_kWh` is NOT a battery price. It is the duration-regression
# slope of a grid storage PROJECT (NREL 2025, Fig 2), and `USD_Per_kW` prices a
# bi-directional grid inverter, transformer, switchgear, substation and a
# 2.4-mile transmission line -- none of which a locomotive has. Using the ATB
# formula as written made a 2C locomotive pack cost $976/kWh, 97% of it the
# power term. See `Scripts/fixed_cost_basis.md` for the full derivation.
#
#   pack $/kWh = ATB_$per_kWh x ATB_PACK_COST_SHARE x m(P/E)
#
# ATB_PACK_COST_SHARE = 210/241 recovers NREL 2025's battery-CABINET line (Fig 1)
# from the energy component (Fig 2): cells, modules, BMS, thermal management,
# racks, enclosure, fire suppression, manufacturer margin and freight -- a
# delivered pack. What it drops is grid interconnection plus the energy-allocated
# project soft costs (developer profit, EPC overhead, contingency, sales tax),
# which is the residual between the two figures and nothing else.
#
# m(P/E) is the high-rate duty premium, fitted to the BatPaC v5.0 manual's own
# Dashboard series (Fig 3.11: rated power held at 300 kW while energy steps
# 60->120 kWh), R^2 0.9994, normalized to the ATB's 4-hour (0.25C) reference.
# Because it is a RATIO, BatPaC's profit-and-warranty markup and its 500,000-
# packs-per-year volume assumption cancel: no double count against the retail
# multiplier and no import of BatPaC's absolute levels.
ATB_PACK_COST_SHARE = 210.0 / 241.0
_BATPAC_PE_INTERCEPT = 83.34
_BATPAC_PE_SLOPE = 7.912
_ATB_REFERENCE_PE = 0.25


def pack_duty_multiplier(power_to_energy: float) -> float:
    """Return the BatPaC pack-cost multiplier for a duty ratio.

    Normalized to the ATB 4-hour (0.25C) reference: 1.070 at 1C, 1.162 at 2C.

    Args:
        power_to_energy: Pack power-to-energy ratio (1/h, i.e. C-rate).

    Returns:
        Dimensionless multiplier on the ATB energy cost.
    """
    return ((_BATPAC_PE_INTERCEPT + _BATPAC_PE_SLOPE * float(power_to_energy))
            / (_BATPAC_PE_INTERCEPT + _BATPAC_PE_SLOPE * _ATB_REFERENCE_PE))


def pack_cost_usd(res: dict, usd_per_kwh: float) -> float:
    """Return one battery pack's manufacturing cost in 2025 USD.

    The retail-price multiplier is left for callers to apply to pack plus chassis.

    Args:
        res: Locomotive pydict `res` body with `energy_capacity_joules`,
            `pwr_out_max_watts` and `pwr_in_max_watts`.
        usd_per_kwh: ATB energy-component cost for the year (2024 USD/kWh).

    Returns:
        `kWh * usd_per_kwh * ATB_PACK_COST_SHARE * duty multiplier`, restated in
        2025 USD; 0.0 when capacity is zero. Duty uses the larger of charge and
        discharge power.
    """
    kwh = res["energy_capacity_joules"] * 1e-6 * utilities.KWH_PER_MJ
    if kwh <= 0:
        return 0.0
    disch_kw = (res.get("pwr_out_max_watts") or 0.0) / 1000
    charge_kw = (res.get("pwr_in_max_watts") or 0.0) / 1000
    power_to_energy = max(disch_kw, charge_kw) / kwh
    return (kwh * usd_per_kwh * ATB_PACK_COST_SHARE
            * pack_duty_multiplier(power_to_energy)
            # NREL 2025 battery cost projections are 2024 USD; restate in 2025.
            * utilities.USD_2024_TO_2025)


def add_battery_costs(loco_info: pd.DataFrame, year: int,
                      scenario: str = "Moderate") -> pd.DataFrame:
    prices_to_use = battery_prices(scenario).filter(
        (pl.col("Year") - pl.lit(year)).abs().min()
        == (pl.col("Year") - pl.lit(year)).abs()
    )
    usd_per_kWh = prices_to_use.filter(pl.col("Metric") == "USD_Per_kWh").get_column(
        "Value"
    )[0]
    for idx, row in loco_info.iterrows():
        loco_dict = row["Rust_Loco"].to_pydict()
        loco_type = next(iter(loco_dict["loco_type"].values()))
        if "res" in loco_type:
            loco_info.at[idx, "Cost_USD"] = (
                defaults.BEL_MINUS_BATTERY_MANUFACTURE_COST_USD
                + pack_cost_usd(loco_type["res"], usd_per_kWh)
            ) * defaults.RETAIL_PRICE_EQUIVALENT_MULTIPLIER
    return loco_info


def calculate_meet_pass_events(info: ScenarioInfo, units: str) -> MetricType:
    """Count meet/pass events as the number of unscheduled stops each train makes.

    A stop is a sample where speed first becomes zero after having been
    positive; the last such stop per train is its arrival at the destination and
    is not a meet/pass, so one stop per train is discounted.
    """
    slts_results = []
    for i, slts in enumerate(info.sims.to_pydict()):
        history = slts["history"]
        time_seconds = history.get("time_seconds", [])
        speed_meters_per_second = history.get("speed_meters_per_second", [])
        if not time_seconds or not speed_meters_per_second:
            print(
                "No SpeedLimitTrainSim history was saved, so meet pass events cannot be counted."
            )
            return metric("Meet_Pass_Events", units, None)

        slts_results.append(
            pl.LazyFrame(
                {
                    "history.time_seconds": time_seconds,
                    "history.speed_meters_per_second": speed_meters_per_second,
                }
            ).with_columns(pl.lit(i).alias("train_idx"))
        )

    if len(slts_results) == 0:
        return metric("Meet_Pass_Events", units, None)

    # Speed is snapped exactly to the target when the solver brings a train to a
    # stand, but compare against a small tolerance so a residual float value
    # still reads as stopped.
    stopped_tol = 1e-6
    val = (
        pl.concat(slts_results, how="diagonal_relaxed")
        # NOTE: must be sorted by time -- the stop detection below relies on
        # consecutive rows being consecutive in time within each train.
        .sort("train_idx", "history.time_seconds")
        .with_columns(
            (pl.col("history.speed_meters_per_second") <= stopped_tol).alias("stopped")
        )
        .with_columns(
            # A stop is the transition from moving to stopped. The first sample
            # of each train is excluded (`fill_null(True)`) because trains
            # start from rest.
            (
                pl.col("stopped")
                & ~pl.col("stopped").shift(1).over("train_idx").fill_null(True)
            )
            .cast(pl.Int64)
            .alias("stop")
        )
        .group_by("train_idx")
        .agg(pl.max_horizontal([0, pl.col("stop").sum() - 1]).alias("meet_pass_stops"))
        .select(pl.col("meet_pass_stops").sum())
        .collect()
        .item()
    )
    return metric("Meet_Pass_Events", units, val)

def _mj_to_units(units: str) -> Optional[float]:
    """Return the multiplier from MJ to `units`.

    Args:
        units: One of "J", "MJ", "GJ", "TJ", "kWh" or "MWh".

    Returns:
        The conversion factor, or None for an unsupported unit.
    """
    return {
        "J": 1e6, "MJ": 1.0, "GJ": 1e-3, "TJ": 1e-6,
        "kWh": utilities.KWH_PER_MJ, "MWh": utilities.KWH_PER_MJ / 1e3,
    }.get(units)


def _sim_catenary_energy(
    sims: Optional[alt.SpeedLimitTrainSimVec],
) -> Tuple[float, float, float]:
    """Sum catenary draw, return and line loss (J).

    Values are read from each `cat_int` terminal state, exact at any save interval.

    Args:
        sims: Simulations whose dual-mode or battery-electric locomotives carry a
            `cat_int`; None yields zeros.

    Returns:
        `(pantograph_draw_J, pantograph_return_J, line_loss_J)`: gross energy
        drawn at the pantograph (`energy_elec_in_joules` plus
        `energy_elec_regen_joules`), energy returned to the wire, and resistive
        line loss supplied by the substation.
    """
    net = 0.0
    regen = 0.0
    line_loss = 0.0
    if sims is None:
        return 0.0, 0.0, 0.0
    for slts in sims.to_pydict():
        loco_con = slts.get("loco_con") or {}
        for loco in loco_con.get("loco_vec") or []:
            loco_type = loco.get("loco_type") or {}
            for key in ("DualModeLoco", "BatteryElectricLoco"):
                sub = loco_type.get(key)
                cat_int = sub.get("cat_int") if sub else None
                if not cat_int:
                    continue
                state = cat_int.get("state") or {}
                hist = cat_int.get("history") or {}
                e_in = state.get("energy_elec_in_joules")
                e_rg = state.get("energy_elec_regen_joules")
                e_ll = state.get("energy_line_loss_joules")
                if e_in is None:
                    e_in = (hist.get("energy_elec_in_joules") or [None])[-1]
                if e_rg is None:
                    e_rg = (hist.get("energy_elec_regen_joules") or [0.0])[-1]
                if e_ll is None:
                    e_ll = (hist.get("energy_line_loss_joules") or [None])[-1]
                net += float(e_in or 0.0)
                regen += float(e_rg or 0.0)
                line_loss += float(e_ll or 0.0)
    return net + regen, regen, line_loss


def _sim_catenary_energy_for(info: "ScenarioInfo") -> Tuple[float, float, float]:
    """Return raw catenary draw, return and line loss (J).

    Args:
        info: Scenario supplying `sims` and an optional `agg_cache`.

    Returns:
        `(pantograph_draw_J, pantograph_return_J, line_loss_J)`, cached per sims
        object.
    """
    return _cached_sim_aggregate(
        info, "catenary_energy",
        lambda: _sim_catenary_energy(info.sims), annualize=False)


def _sim_catenary_joules(
    sims: Optional[alt.SpeedLimitTrainSimVec],
) -> Tuple[float, float]:
    """Sum gross pantograph draw and line loss (J).

    Args:
        sims: Simulations to read; None yields zeros.

    Returns:
        `(pantograph_delivered_J, line_loss_J)`.
    """
    draw, _regen, line_loss = _sim_catenary_energy(sims)
    return draw, line_loss

def calculate_energized_distance(info: ScenarioInfo, units: str) -> MetricType:
    """Report built energization distance per infrastructure component.

    Args:
        info: Scenario whose `infrastructure_costs` may carry `Distance_Miles`
            and `Distance_Units` per `Infrastructure_Type` row.
        units: Units reported when `Distance_Units` is absent.

    Returns:
        `Distance_Energized` rows, one `Subset` per component with a distance,
        summed within each `Distance_Units`; a None metric when no row carries a
        distance.
    """
    infra = info.infrastructure_costs
    if infra is None or "Distance_Miles" not in infra.columns:
        return metric("Distance_Energized", units, None)
    have_units = "Distance_Units" in infra.columns
    df = infra.filter(pl.col("Distance_Miles").is_not_null())
    if df.height == 0:
        return metric("Distance_Energized", units, None)
    group_cols = ["Infrastructure_Type"] + (["Distance_Units"] if have_units else [])
    agg = df.group_by(group_cols).agg(pl.col("Distance_Miles").sum())
    rows = [
        metric(
            "Distance_Energized",
            r["Distance_Units"] if have_units else units,
            r["Distance_Miles"], subset=r["Infrastructure_Type"],
        )
        for r in agg.iter_rows(named=True)
    ]
    return metrics_from_list(rows)


def calculate_energy(info: ScenarioInfo, units: str) -> MetricType:
    """Report energy consumed by category.

    Args:
        info: Scenario supplying the sims, refuel sessions and annualization.
        units: Energy unit: "J", "MJ", "GJ", "TJ", "kWh" or "MWh".

    Returns:
        `Energy` rows with subsets `Diesel`, `Electricity`,
        `Electricity_Line_Loss` (catenary resistive loss from the sims) and
        `Electricity_Stationary_Charging` (grid electricity beyond the catenary
        grid draw); a None metric for an unsupported unit.
    """
    conv = _mj_to_units(units)
    if conv is None:
        print(f"Units of {units} not supported for energy calculation.")
        return metric("Energy", units, None)

    diesel_mj = value_from_metrics(calculate_diesel_use(info, "MJ"), "Diesel_Usage")
    elec_total_mj = value_from_metrics(
        calculate_electricity_use(info, "MJ"), subset="All"
    )
    # Catenary delivery-chain energy from the SIM: pantograph delivered + line
    # loss (= substation-side catenary grid draw). Annualize to match the diesel
    # and electricity totals (same 365.25/simulation_days scaler).
    scaler = (365.25 / info.simulation_days
              if info.annualize and info.simulation_days else 1.0)
    pantograph_j, line_loss_j = _sim_catenary_joules_for(info)
    line_loss_mj = line_loss_j * scaler / 1.0e6
    catenary_grid_mj = (pantograph_j + line_loss_j) * scaler / 1.0e6

    rows: List[MetricType] = []
    if diesel_mj is not None:
        rows.append(metric("Energy", units, diesel_mj * conv, subset="Diesel"))
    if elec_total_mj is not None:
        # Grid electricity not delivered via the catenary is stationary charging.
        charging_mj = max(elec_total_mj - catenary_grid_mj, 0.0)
        rows.append(metric("Energy", units, elec_total_mj * conv, subset="Electricity"))
        rows.append(metric(
            "Energy", units, line_loss_mj * conv, subset="Electricity_Line_Loss"))
        rows.append(metric(
            "Energy", units, charging_mj * conv,
            subset="Electricity_Stationary_Charging"))
    if not rows:
        return metric("Energy", units, None)
    return metrics_from_list(rows)

EN_ROUTE = "En_Route"


def _en_route_flags(train_history: Dict[str, Any]) -> Optional[List[bool]]:
    """Flag samples where the train is between its terminals.

    En route means `offset_meters` lies strictly between its first and last values.

    Args:
        train_history: Train history dict with an `offset_meters` series.

    Returns:
        One flag per sample, or None when offset was not recorded or did not
        advance.
    """
    offset = train_history.get("offset_meters") or []
    if len(offset) < 2:
        return None
    first, last = offset[0], offset[-1]
    if not last > first:
        return None
    return [(o > first) and (o < last) for o in offset]


def _dwell_frame(
    train_history: Dict[str, Any], train_idx: int, **extra: Any
) -> pl.DataFrame:
    """Build one train's time, speed and `en_route` frame.

    Args:
        train_history: Train history dict with `time_seconds` and
            `speed_meters_per_second`, optionally `offset_meters`.
        train_idx: Index written to the `train_idx` column.
        **extra: Additional equal-length columns to include.

    Returns:
        Frame with `history.time_seconds`, `history.speed_meters_per_second`,
        `en_route` (all True when offset was not recorded), `train_idx` and any
        `extra` columns.
    """
    time_seconds = train_history.get("time_seconds") or []
    flags = _en_route_flags(train_history)
    return pl.DataFrame(
        {
            "history.time_seconds": [float(t) for t in time_seconds],
            "history.speed_meters_per_second": [
                float(s) for s in train_history.get("speed_meters_per_second") or []
            ],
            "en_route": flags if flags is not None else [True] * len(time_seconds),
            **extra,
        }
    ).with_columns(pl.lit(train_idx).alias("train_idx"))


def _with_intervals(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Add each sample's interval and a stopped-throughout flag.

    `stopped_interval` requires zero speed at both ends of the interval.

    Args:
        frame: Samples with `train_idx`, `history.time_seconds` and
            `history.speed_meters_per_second`.

    Returns:
        `frame` sorted by train and time with `interval_seconds` (time to the
        next sample, 0 at the last) and boolean `stopped_interval` added.
    """
    return frame.sort("train_idx", "history.time_seconds").with_columns(
        (
            pl.col("history.time_seconds").shift(-1).over("train_idx")
            - pl.col("history.time_seconds")
        )
        .fill_null(0.0)
        .alias("interval_seconds"),
        (
            (pl.col("history.speed_meters_per_second") == 0)
            & (
                pl.col("history.speed_meters_per_second").shift(-1).over("train_idx")
                == 0
            )
        )
        .fill_null(False)
        .alias("stopped_interval"),
    )


def _interval_delta(column: str, alias: str) -> pl.Expr:
    """Difference a cumulative series over each sample's interval.

    Args:
        column: Cumulative series column, monotone within each train.
        alias: Name of the resulting column.

    Returns:
        Expression for `column[i+1] - column[i]` within `train_idx`, 0 at the
        last sample and clipped at zero.
    """
    return (
        (pl.col(column).shift(-1).over("train_idx") - pl.col(column))
        .fill_null(0.0)
        .clip(lower_bound=0.0)
        .alias(alias)
    )


def _both_subsets(
    frame: pl.LazyFrame, column: str, name: str, units: str, scale: float = 1.0
) -> MetricType:
    """Total a column over all dwell and en-route dwell.

    Args:
        frame: Samples with `column` and a boolean `en_route` column.
        column: Column to sum.
        name: Metric name.
        units: Metric units.
        scale: Factor applied to both totals.

    Returns:
        Two `name` rows: `Subset` "All" and `Subset` `EN_ROUTE`.
    """
    totals = (
        frame.select(
            pl.col(column).sum().alias("all"),
            pl.when(pl.col("en_route"))
            .then(pl.col(column))
            .otherwise(0.0)
            .sum()
            .alias("en_route"),
        )
        .collect()
        .row(0)
    )
    return metrics_from_list(
        [
            metric(name, units, totals[0] * scale),
            metric(name, units, totals[1] * scale, subset=EN_ROUTE),
        ]
    )


def calculate_dwell_time(info: ScenarioInfo, units: str) -> MetricType:
    """Total the time trains spend stationary.

    Args:
        info: Scenario whose sims saved `time_seconds` and
            `speed_meters_per_second` history.
        units: Units label for the metric; values are in seconds.

    Returns:
        `Dwell_Time` for all dwell and for en-route dwell; a None metric when no
        history was saved.
    """
    slts_results = []
    i = 0
    for slts in info.sims.to_pydict():
        history = slts["history"]
        if not history.get("time_seconds") or not history.get(
            "speed_meters_per_second"
        ):
            print(
                "No SpeedLimitTrainSim history was saved, so dwell time cannot be calculated."
            )
            return metric("Dwell_Time", units, None)
        slts_results.append(_dwell_frame(history, i).lazy())
        i += 1
    frame = _with_intervals(pl.concat(slts_results, how="diagonal_relaxed")).with_columns(
        pl.when(pl.col("history.speed_meters_per_second") == 0)
        .then(pl.col("interval_seconds"))
        .otherwise(0.0)
        .alias("dwell_seconds")
    )
    return _both_subsets(frame, "dwell_seconds", "Dwell_Time", units)


def _align_on_step(
    train_history: Dict[str, Any],
    component_history: Dict[str, Any],
    component_columns: Dict[str, List[float]],
) -> Optional[pl.DataFrame]:
    """Align a component's series with train time and speed.

    Rows pair by a backward as-of join on step counter `i` when both sides carry it.

    Args:
        train_history: Train history with `time_seconds`,
            `speed_meters_per_second`, optionally `offset_meters` and `i`.
        component_history: Component history supplying its `i` series.
        component_columns: Output column name to the component series to align.

    Returns:
        Frame with `history.time_seconds`, `history.speed_meters_per_second`,
        `en_route` and the component columns; None when either side is empty or
        no `i` is available and the lengths differ.
    """
    time_seconds = train_history.get("time_seconds") or []
    speed = train_history.get("speed_meters_per_second") or []
    if not time_seconds or not speed or not component_columns:
        return None
    if any(not values for values in component_columns.values()):
        return None

    flags = _en_route_flags(train_history)
    train = pl.DataFrame(
        {
            "history.time_seconds": [float(t) for t in time_seconds],
            "history.speed_meters_per_second": [float(s) for s in speed],
            "en_route": flags if flags is not None else [True] * len(time_seconds),
        }
    )
    component = pl.DataFrame(
        {name: [float(v) for v in values] for name, values in component_columns.items()}
    )

    train_i = train_history.get("i") or []
    component_i = component_history.get("i") or []
    if len(train_i) == train.height and len(component_i) == component.height:
        train = train.with_columns(
            pl.Series("i", [int(v) for v in train_i], dtype=pl.Int64)
        )
        component = component.with_columns(
            pl.Series("i", [int(v) for v in component_i], dtype=pl.Int64)
        )
        return (
            train.sort("i")
            .join_asof(component.sort("i"), on="i", strategy="backward")
            .drop("i")
        )

    # No usable step counter: a build predating non-maskable `i`, or history
    # assembled by hand. Positional pairing is only defensible when the lengths
    # match, which implies a common interval.
    if component.height != train.height:
        return None
    return pl.concat([train, component], how="horizontal")


def calculate_dwell_fuel_energy(info: ScenarioInfo, units: str) -> MetricType:
    """Total consist fuel energy burned while trains are stopped.

    An interval counts when speed is zero at both of its ends.

    Args:
        info: Scenario whose sims saved train history and consist
            `energy_fuel_joules` history.
        units: "J", "MJ", "MWh" or "kWh".

    Returns:
        `Dwell_Fuel_Energy` for all dwell and for en-route dwell; a None metric
        for an unsupported unit or missing history.
    """
    if units == "J":
        conversion_from_joule = 1.0
    elif units == "MJ":
        conversion_from_joule = 1e-6
    elif units == "MWh":
        conversion_from_joule = utilities.KWH_PER_MJ / 1e9
    elif units == "kWh":
        conversion_from_joule = utilities.KWH_PER_MJ / 1e6
    else:
        print(f"Units of {units} not supported for dwell fuel energy calculation.")
        return metric("Dwell_Fuel_Energy", units, None)

    slts_results = []
    i = 0
    for slts in info.sims.to_pydict():
        consist_history = slts["loco_con"]["history"]
        aligned = _align_on_step(
            slts["history"],
            consist_history,
            {
                "loco_con.history.energy_fuel_joules": consist_history.get(
                    "energy_fuel_joules", []
                )
            },
        )
        if aligned is None:
            print(
                "No SpeedLimitTrainSim history was saved, so dwell fuel energy cannot be calculated."
            )
            return metric("Dwell_Fuel_Energy", units, None)

        df = aligned.lazy().with_columns(pl.lit(i).alias("train_idx"))
        slts_results.append(df)
        i += 1
    frame = (
        _with_intervals(pl.concat(slts_results, how="diagonal_relaxed"))
        .with_columns(
            _interval_delta(
                "loco_con.history.energy_fuel_joules", "interval_fuel_joules"
            )
        )
        .with_columns(
            pl.when(pl.col("stopped_interval"))
            .then(pl.col("interval_fuel_joules"))
            .otherwise(0.0)
            .alias("dwell_fuel_energy_joules")
        )
    )
    return _both_subsets(
        frame,
        "dwell_fuel_energy_joules",
        "Dwell_Fuel_Energy",
        units,
        conversion_from_joule,
    )


def calculate_apu_fuel_consumption(info: ScenarioInfo, units: str) -> MetricType:
    """Estimate fuel an auxiliary power unit burns during dwell.

    Mean auxiliary power per stopped interval feeds a linear gal/hr APU fuel curve.

    Args:
        info: Scenario whose sims saved train history and per-locomotive
            `energy_aux_joules` history at a common interval.
        units: "J", "MJ", "MWh" or "kWh".

    Returns:
        `APU_Fuel_Consumption` for all dwell and for en-route dwell, converted
        from gallons with the GREET diesel density and heating value; a None
        metric for an unsupported unit or missing or misaligned history.
    """
    if units == "J":
        conversion_from_joule = 1.0
    elif units == "MJ":
        conversion_from_joule = 1e-6
    elif units == "MWh":
        conversion_from_joule = utilities.KWH_PER_MJ / 1e9
    elif units == "kWh":
        conversion_from_joule = utilities.KWH_PER_MJ / 1e6
    else:
        print(f"Units of {units} not supported for APU fuel consumption calculation.")
        return metric("APU_Fuel_Consumption", units, None)

    diesel_emissions_factors = emissions_factors_greet.filter(
        pl.col("Name") == pl.lit("ultra low sulfur diesel")
    )
    lhv_kj_per_kg = diesel_emissions_factors.get_column("LowHeatingValue")[0] * 1e3
    rho_fuel_g_per_gallon = diesel_emissions_factors.get_column("Density")[0]
    joules_per_gallon = lhv_kj_per_kg * rho_fuel_g_per_gallon

    slts_results = []
    i = 0
    for slts in info.sims.to_pydict():
        locomotives = slts["loco_con"]["loco_vec"]
        locomotive_aux_energy_joules = [
            locomotive["history"].get("energy_aux_joules", [])
            for locomotive in locomotives
        ]
        if not locomotive_aux_energy_joules or any(
            not energy for energy in locomotive_aux_energy_joules
        ):
            print(
                "No SpeedLimitTrainSim history was saved, so APU fuel consumption cannot be calculated."
            )
            return metric("APU_Fuel_Consumption", units, None)

        # Locomotives are configured together and step in lockstep, so their
        # aux series share one interval and sum elementwise; the sum is then
        # aligned to the train off the first locomotive's step counter.
        if len({len(energy) for energy in locomotive_aux_energy_joules}) != 1:
            print(
                "Locomotives recorded auxiliary energy at differing intervals, so APU fuel consumption cannot be calculated."
            )
            return metric("APU_Fuel_Consumption", units, None)
        consist_aux_energy_joules = [
            sum(energy_by_locomotive)
            for energy_by_locomotive in zip(*locomotive_aux_energy_joules)
        ]

        aligned = _align_on_step(
            slts["history"],
            locomotives[0]["history"],
            {"aux_energy_joules": consist_aux_energy_joules},
        )
        if aligned is None:
            print(
                "Locomotive auxiliary-power history does not align with train history, so APU fuel consumption cannot be calculated."
            )
            return metric("APU_Fuel_Consumption", units, None)

        df = aligned.lazy().with_columns(pl.lit(i).alias("train_idx"))
        slts_results.append(df)
        i += 1
    # The APU fuel curve is nonlinear in power, so it needs the MEAN auxiliary
    # power over each interval, not the instantaneous value at its start.
    # Differencing the cumulative series gives exactly that mean; the snapshot
    # rate only equals it where aux power happened to be constant.
    frame = (
        _with_intervals(pl.concat(slts_results, how="diagonal_relaxed"))
        .with_columns(_interval_delta("aux_energy_joules", "interval_aux_joules"))
        .with_columns(
            pl.when(pl.col("interval_seconds") > 0)
            .then(pl.col("interval_aux_joules") / pl.col("interval_seconds"))
            .otherwise(0.0)
            .alias("mean_aux_power_watts")
        )
        .with_columns(
            pl.when(pl.col("stopped_interval"))
            .then(
                (0.063 * pl.col("mean_aux_power_watts") / 1_000.0 + 0.4115)
                * pl.col("interval_seconds")
                / 3_600.0
            )
            .otherwise(0.0)
            .alias("apu_fuel_gallons")
        )
    )
    return _both_subsets(
        frame,
        "apu_fuel_gallons",
        "APU_Fuel_Consumption",
        units,
        joules_per_gallon * conversion_from_joule,
    )

function_mappings = {
    "Meet_Pass_Events": calculate_meet_pass_events,
    "Dwell_Time": calculate_dwell_time,
    "Dwell_Fuel_Energy": calculate_dwell_fuel_energy,
    "APU_Fuel_Consumption": calculate_apu_fuel_consumption,
    "Energy_Costs": calculate_energy_cost,
    "Freight_Moved": calculate_freight_moved,
    "Energy_Per_Freight_Moved": calculate_energy_per_freight,
    "Energy": calculate_energy,
    "Distance_Energized": calculate_energized_distance,
    "GHG": calculate_ghg,
    "Count_Locomotives": calculate_locomotive_counts,
    "Count_Refuelers": calculate_refueler_counts,
    "Count_Infrastructure": calculate_infrastructure_counts,
}
