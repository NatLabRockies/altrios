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
# INSTALLED cost per the source, so no retail multiplier (same convention as
# PANTOGRAPH_COST_USD). Diesel fuelers are priced at zero on the assumption that
# the fueling plant already exists, which is why only chargers carry capital.
BEL_CHARGER_COST_USD = BEL_CHARGE_RATE_KW * 1000 * utilities.USD_2022_TO_2025 # NLR Cost of Charging (Borlaug) showing ~linear trend on kW; #ICCT report showing little change through 2030
BEL_CHARGER_LIFESPAN = 15

BASE_ANALYSIS_YEAR = 2025
DISCOUNT_RATE = 0.07
RETAIL_PRICE_EQUIVALENT_MULTIPLIER = 1.15
INCLUDE_EXISTING_ASSETS = True
STRANDED_ASSET_RESALE_PCT = 0.5

DIESEL_MANUFACTURE_COST_USD = 2950000 * utilities.USD_2022_TO_2025
DIESEL_LOCO_COST_USD = DIESEL_MANUFACTURE_COST_USD * RETAIL_PRICE_EQUIVALENT_MULTIPLIER # Zenith et al.

# Battery-electric locomotive LESS its pack. Two constants because the basis
# matters and the old single name did not state which one it was:
#   * `_MANUFACTURE_` is a manufacturing cost, the right input when the pack is
#     added and the retail multiplier applied to the SUM (see
#     `metric_calculator.add_battery_costs`).
#   * `_LOCO_` is the retail-equivalent, the right value anywhere a locomotive
#     price sits beside `DIESEL_LOCO_COST_USD` -- notably the train planner's
#     default `loco_info`, where the two rows must share a basis or the BEL is
#     silently 15% cheap against the diesel it is compared with.
BEL_MINUS_BATTERY_MANUFACTURE_COST_USD = DIESEL_MANUFACTURE_COST_USD * 0.5
BEL_MINUS_BATTERY_LOCO_COST_USD = (
    BEL_MINUS_BATTERY_MANUFACTURE_COST_USD * RETAIL_PRICE_EQUIVALENT_MULTIPLIER
)
# DEPRECATED: kept so existing applications keep resolving. It is the
# MANUFACTURING basis; use one of the two explicit names above instead.
BEL_MINUS_BATTERY_COST_USD = BEL_MINUS_BATTERY_MANUFACTURE_COST_USD

# Electrified locomotive capital (USD). No public list price exists for either;
# both are derived from the diesel / BEL baselines above.
# Dual-mode = a diesel locomotive plus the catenary-interface premium (pantograph
# + power electronics to draw traction power directly from the overhead system).
DUAL_MODE_LOCO_PREMIUM_FRAC = 0.15
DUAL_MODE_LOCO_COST_USD = DIESEL_LOCO_COST_USD * (1.0 + DUAL_MODE_LOCO_PREMIUM_FRAC)
# Pantograph + roof-line power electronics added to a battery-electric locomotive
# so it can draw from the overhead system (catenary-assist / charge-while-running).
# A BEL-with-catenary's full cost = the usual battery-electric cost (see
# metric_calculator.add_battery_costs) plus this premium.
# INSTALLED cost, so no retail multiplier is applied: unlike the locomotive
# bodies above this is a fitted-and-commissioned figure, not a factory cost.
PANTOGRAPH_COST_USD = 150_000.0 * utilities.USD_2022_TO_2025

LOCO_LIFESPAN = 20
ANNUAL_LOCO_TURNOVER = 1.0/LOCO_LIFESPAN
DEFAULT_GAP_SIZE = 0.604

DEMAND_FILE = alt.resources_root() / "Default Demand.csv"
FUEL_EMISSIONS_FILE = alt.resources_root() / "metrics_inputs" / "GREET-CA_Emissions_Factors.csv"
GRID_EMISSIONS_FILE = alt.resources_root() / "metrics_inputs" / "Cambium24_allScenarios_annual_gea.csv"
ELECTRICITY_PRICE_FILE = alt.resources_root() / "metrics_inputs" / "EIA_Electricity_Prices.csv"
LIQUID_FUEL_PRICE_FILE = alt.resources_root() / "metrics_inputs" / "EIA_Liquid_Fuel_Prices.csv"
BATTERY_PRICE_FILE = alt.resources_root() / "metrics_inputs" / "NLR_ATB_Battery_Cost_Forecasts.csv"

# Regional energy prices. Every location's "Electricity Price Region" and "Liquid Fuel
# Price Region" names a zone in ENERGY_PRICE_ZONE_FILE (US state or PADD region, Canadian
# province). A zone's observed price in EIA_PRICE_REFERENCE_YEAR (real 2025 USD; diesel
# USD/gal, industrial electricity cents/kWh) is scaled through time by the ratio of its
# trajectory: the AEO2026 reference case (ELECTRICITY_PRICE_FILE / LIQUID_FUEL_PRICE_FILE)
# for US zones, and the province's Canada's Energy Future 2026 Current Measures end-use
# price (CER_ENERGY_PRICE_FILE) for Canadian zones. The reference year must exist in the
# AEO tables, which start at 2025. See metrics_inputs/README.md for sources.
EIA_PRICE_REFERENCE_YEAR = 2025
ENERGY_PRICE_ZONE_FILE = alt.resources_root() / "metrics_inputs" / "Energy_Price_Zones.csv"
CER_ENERGY_PRICE_FILE = alt.resources_root() / "metrics_inputs" / "CER_EF2026_End_Use_Prices.csv"
