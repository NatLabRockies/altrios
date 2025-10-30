"""Module for default modeling assumption constants."""
import altrios as alt
from altrios import utilities

SIMULATION_DAYS = 7
WARM_START_DAYS = 7

LHV_DIESEL_KJ_PER_KG = 45.6e3 # https://www.engineeringtoolbox.com/fuels-higher-calorific-values-d_169.html
RHO_DIESEL_KG_PER_M3 = 959.  # https://www.engineeringtoolbox.com/fuels-densities-specific-volumes-d_166.html

DIESEL_TANK_CAPACITY_J = (5000 / utilities.GALLONS_PER_LITER / utilities.LITER_PER_M3) * \
    RHO_DIESEL_KG_PER_M3 * LHV_DIESEL_KJ_PER_KG * 1e3
DIESEL_REFUEL_RATE_J_PER_HR = (300 * 60 / utilities.GALLONS_PER_LITER / utilities.LITER_PER_M3) * \
    RHO_DIESEL_KG_PER_M3 * LHV_DIESEL_KJ_PER_KG * 1e3 # 300 gallons per minute -> joules per hour
DIESEL_REFUELER_EFFICIENCY = 1.0
DIESEL_REFUELER_COST_USD = 0.0

BEL_CHARGE_RATE_KW = 750.0
BEL_CHARGE_RATE_J_PER_HR = (BEL_CHARGE_RATE_KW / 1000.0 / utilities.MWH_PER_MJ) * 1e6 
BEL_CHARGER_EFFICIENCY = 0.9
BEL_CHARGER_COST_USD = BEL_CHARGE_RATE_KW * 1000 # NREL Cost of Charging (Borlaug) showing ~linear trend on kW; #ICCT report showing little change through 2030
BEL_CHARGER_LIFESPAN = 15

BASE_ANALYSIS_YEAR = 2022
DISCOUNT_RATE = 0.07
RETAIL_PRICE_EQUIVALENT_MULTIPLIER = 1.15
INCLUDE_EXISTING_ASSETS = True
STRANDED_ASSET_RESALE_PCT = 0.5

DIESEL_MANUFACTURE_COST_USD = 2950000
BEL_MINUS_BATTERY_COST_USD = DIESEL_MANUFACTURE_COST_USD * 0.5
DIESEL_LOCO_COST_USD = DIESEL_MANUFACTURE_COST_USD * RETAIL_PRICE_EQUIVALENT_MULTIPLIER # Zenith et al.

LOCO_LIFESPAN = 20
ANNUAL_LOCO_TURNOVER = 1.0/LOCO_LIFESPAN
DEFAULT_GAP_SIZE = 0.604

DEMAND_FILE = alt.resources_root() / "Default Demand.csv"
FUEL_EMISSIONS_FILE = alt.resources_root() / "metrics_inputs" / "GREET-CA_Emissions_Factors.csv"
GRID_EMISSIONS_FILE = alt.resources_root() / "metrics_inputs" / "Cambium23_allScenarios_annual_gea.csv"
ELECTRICITY_PRICE_FILE = alt.resources_root() / "metrics_inputs" / "EIA_Electricity_Prices.csv"
LIQUID_FUEL_PRICE_FILE = alt.resources_root() / "metrics_inputs" / "EIA_Liquid_Fuel_Prices.csv"
BATTERY_PRICE_FILE = alt.resources_root() / "metrics_inputs" / "NREL_ATB_Battery_Cost_Forecasts.csv"
