from __future__ import annotations
import os
import polars as pl
import pandas as pd
from typing import Dict, Callable, Optional, Union
from dataclasses import dataclass, field
import altrios as alt
from altrios import defaults

pl.enable_string_cache()

@dataclass
class TrainPlannerConfig:
    """
    Dataclass class for train planner configuration parameters.

    Attributes:
    ----------
    - `single_train_mode`: `True` to only run one round-trip train and schedule its charging; `False` to plan train consists
    - `min_cars_per_train`: `Dict` of the minimum length in number of cars to form a train for each train type
    - `target_cars_per_train`: `Dict` of the standard train length in number of cars for each train type
    - `manifest_empty_return_ratio`: Desired railcar reuse ratio to calculate the empty manifest car demand, (E_ij+E_ji)/(L_ij+L_ji)
    - `cars_per_locomotive`: Heuristic scaling factor used to size number of locomotives needed based on demand.
    - `cars_per_locomotive_fixed`: If `True`, `cars_per_locomotive` overrides `hp_per_ton` calculations used for dispatching decisions.
    - `refuelers_per_incoming_corridor`: Heuristic scaling factor used to scale number of refuelers needed at each node based on number of incoming corridors.
    - `containers_per_car`: Containers stacked on each car (applicable only for intermodal containers)
    - `require_diesel`: `True` to require each consist to have at least one diesel locomotive.
    - `manifest_empty_return_ratio`: `Dict`
    - `drag_coeff_function`: `Dict`
    - `hp_required_per_ton`: `Dict`
    - `dispatch_scaling_dict`: `Dict`
    - `loco_info`: `Dict`
    - `refueler_info`: `Dict`
    - `return_demand_generators`: `Dict`
    """
    simulation_days: int = 21
    single_train_mode: bool = False
    min_cars_per_train: Dict = field(default_factory = lambda: {
        "Default": 60
    })
    target_cars_per_train: Dict = field(default_factory = lambda: {
        "Default": 180
    })
    cars_per_locomotive: Dict = field(default_factory = lambda: {
        "Default": 70
    })
    cars_per_locomotive_fixed: bool = False
    refuelers_per_incoming_corridor: int = 4
    containers_per_car: int = 2
    require_diesel: bool = False
    manifest_empty_return_ratio: float = 0.6
    loco_pool_safety_factor: float = 1.1
    failed_sim_logging_path: Union[str, os.PathLike] = None
    hp_required_per_ton: Dict = field(default_factory = lambda: {
        "Default": {
        "Unit": 2.0,
        "Manifest": 1.5,
        "Intermodal": 4.0
        }                         
    })
    dispatch_scaling_dict: Dict = field(default_factory = lambda: {
        "time_mult_factor": 1.4,
        "hours_add": 2,
        "energy_mult_factor": 1.25
    })
    loco_info: pd.DataFrame = field(default_factory = lambda: pd.DataFrame({
        "Diesel_Large": {
            "Capacity_Cars": 20,
            "Fuel_Type": "Diesel",
            "Min_Servicing_Time_Hr": 3.0,
            "Rust_Loco": alt.Locomotive.default(),
            "Cost_USD": defaults.DIESEL_LOCO_COST_USD,
            "Lifespan_Years": defaults.LOCO_LIFESPAN
            },
        "BEL": {
            "Capacity_Cars": 20,
            "Fuel_Type": "Electricity",
            "Min_Servicing_Time_Hr": 3.0,
            "Rust_Loco": alt.Locomotive.default_battery_electric_loco(),
            "Cost_USD": defaults.BEL_MINUS_BATTERY_COST_USD,
            "Lifespan_Years": defaults.LOCO_LIFESPAN
            }
        }).transpose().reset_index(names='Locomotive_Type'))
    refueler_info: pd.DataFrame = field(default_factory = lambda: pd.DataFrame({
        "Diesel_Fueler": {
            "Locomotive_Type": "Diesel_Large",
            "Fuel_Type": "Diesel",
            "Refueler_J_Per_Hr": defaults.DIESEL_REFUEL_RATE_J_PER_HR,
            "Refueler_Efficiency": defaults.DIESEL_REFUELER_EFFICIENCY,
            "Cost_USD": defaults.DIESEL_REFUELER_COST_USD,
            "Lifespan_Years": defaults.LOCO_LIFESPAN
        },
        "BEL_Charger": {
            "Locomotive_Type": "BEL",
            "Fuel_Type": "Electricity",
            "Refueler_J_Per_Hr": defaults.BEL_CHARGE_RATE_J_PER_HR,
            "Refueler_Efficiency": defaults.BEL_CHARGER_EFFICIENCY,
            "Cost_USD": defaults.BEL_CHARGER_COST_USD,
            "Lifespan_Years": defaults.LOCO_LIFESPAN
        }
    }).transpose().reset_index(names='Refueler_Type'))
    loco_type_shares: Optional[Dict] = None
    drag_coeff_function: Optional[Callable]= None
    dispatch_scheduler: Optional[Callable] = None
    return_demand_generators: Optional[Dict] = None #default defined in train_demand_generators.py
