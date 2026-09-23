# Metrics inputs: prices, costs and emissions factors

Sources and methods for the cost and price tables the metrics calculator reads.
All prices are carried in real 2025 US dollars, the AEO2026 base year.

## Energy prices

Two things are combined for every location: a **zone reference price** observed in
2025, and a **trajectory** that moves it through time as a ratio. The method is the
same on both sides of the border; only the sources differ.

### `Energy_Price_Zones.csv`

One row per (zone, fuel). `Zone` is what a location file's `Electricity Price Region`
or `Liquid Fuel Price Region` column names, and what a corridor's price-zone boxes
(see below) resolve substations to.

| Column | Meaning |
|---|---|
| `Reference_Year`, `Reference_Price` | Observed price in that year, in `Units` (US cents/kWh or USD/gal) |
| `Reference_Price_Native`, `Native_Units`, `CAD_per_USD` | The source value before currency conversion |
| `Trajectory` | `EIA_AEO2026` or `CER_EF2026:<Province>` |
| `Source` | Where the number came from |

United States:

- Industrial electricity by state: EIA *Electric Power Monthly* (February 2026 issue),
  Table 5.6.B, industrial sector, year-to-date December 2025.
- Retail on-highway diesel by PADD region: EIA *Gasoline and Diesel Fuel Update*, 2025
  annual averages (`pet_pri_gnd_dcus_*_a`). State codes for diesel are aliases of their
  PADD region so a location file can name one code for both fuels.

Canada (converted at the Bank of Canada 2025 annual average of 1.3978 CAD per USD):

- Large-power electricity: Hydro-Québec, *Comparison of Electricity Prices in Major
  North American Cities*, rates in effect April 1, 2025, the 5,000 kW / 3,060,000 kWh
  per month category, excluding taxes. Provinces are represented by their surveyed
  cities (Alberta by the mean of Calgary and Edmonton).
- Retail diesel: Statistics Canada table 18-10-0001-01, diesel at self-service filling
  stations, mean of the twelve 2025 monthly values for the province's surveyed cities.

Tax basis: the US series is retail including road taxes, the Canadian series is retail
including GST/HST and provincial road taxes. Railways pay neither road tax in full
(rail diesel carries federal excise plus a provincial locomotive fuel tax in Canada, and
is off-road fuel in the US), so both sides are a consistent, slightly high retail proxy.

### Trajectories

- `EIA_Liquid_Fuel_Prices.csv`, `EIA_Electricity_Prices.csv`: EIA Annual Energy Outlook
  2026 reference case (cb2026, released April 2026), transportation-sector diesel and
  industrial electricity, real 2025 dollars. Used as the trajectory for US zones and,
  through the legacy override path, for any caller that pins its own reference prices.
- `CER_EF2026_End_Use_Prices.csv`: Canada Energy Regulator, *Canada's Energy Future 2026*,
  Current Measures scenario, end-use prices by province, sector and fuel, 2005-2050, in
  2022 CAD per GJ. Only ratios of a series are used, so the currency and dollar year of
  this file never reach a result. Transportation-sector diesel and industrial electricity
  are the series a zone's `CER_EF2026:<Province>` trajectory selects.

`metric_calculator.zone_energy_price(zone, fuel, year)` returns
`Reference_Price * trajectory(year) / trajectory(Reference_Year)`, interpolating linearly
between tabulated years and clamping outside them.

### Corridor price-zone boxes

A corridor whose catenary draw should be priced where it is fed lists a boxes file in
`applications/corridor_doe/in/corridors.csv` (`Price_Zones_Path`), for example
`networks/Spokane_Lethbridge/spokane_lethbridge_price_zones.csv`. Each row is a lat/lon
box (null bounds are open) naming an electricity and a diesel zone; rows are tried in
order and a substation takes the first box that contains it.

## Battery costs: `NLR_ATB_Battery_Cost_Forecasts.csv`

NREL, *Cost Projections for Utility-Scale Battery Storage: 2025 Update* (Cole et al.,
NREL/TP-6A40-93281, work completed February 2025), 2024 US dollars. The 2024 bottom-up
cost of a 4-hour utility-scale system is split into an energy component of 241 USD/kWh
and a power component of 372 USD/kW (report section 2.2). The normalized Low / Mid /
High trajectories of the report's Table 2 (2024 = 1.00) are applied to both components
and stored as the `Advanced` / `Moderate` / `Conservative` scenarios. A locomotive pack
costs `kWh * USD_Per_kWh + kW * USD_Per_kW`, restated to 2025 dollars with
`utilities.USD_2024_TO_2025`. The earlier file was the 2023 update (2022 dollars) and
carried a fixed `USD_Constant` term per pack; the 2025 report has no such term.

Note: the report's text quotes the 2035 mid case as 234 USD/kWh while its Table 2 gives
0.73 x 334 = 243 USD/kWh. The table values are used here.

## Emissions

- `GREET-CA_Emissions_Factors.csv`: GREET well-to-wheel factors for diesel.
- `Cambium24_allScenarios_annual_gea.csv` (default) and earlier Cambium releases: NREL
  Cambium long-run marginal grid emissions by region and year.

### Who prices on the zones

`metric_calculator.import_energy_prices_eia` (sim_manager path), the corridor DoE
(`rollout_costs.py`, `rollout_costs_sim_free.py`, `postprocess_antiidling.py`,
`build_layouts.py`) and the catenary deep-dive (`energization_optimization_corridor.py`)
all price from the zones. The deep-dive and the layout builder hand the optimizer a
levelized electricity price per link from the link's zone, and the terminal-zone mean
for diesel; because the optimizer's objective changes, layouts and their fingerprints
built before this change are not reproducible from the current code.
