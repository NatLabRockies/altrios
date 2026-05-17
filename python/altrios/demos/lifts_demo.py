# %%
from altrios import sim_manager
from altrios import utilities, defaults
import altrios as alt
from altrios.train_planner import planner_config
from altrios.lifts import lifts_simulator
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from pathlib import Path
import polars as pl

sns.set_theme()

SHOW_PLOTS = alt.utils.show_plots()
# %

plot_dir = Path() / "plots"
# make the dir if it doesn't exist
plot_dir.mkdir(exist_ok=True)


# %%

t0_import = time.perf_counter()
t0_total = time.perf_counter()

rail_vehicles = [
    alt.RailVehicle.from_file(vehicle_file, skip_init=False)
    for vehicle_file in Path(alt.resources_root() / "rolling_stock/").glob("*.yaml")
]

location_map = alt.import_locations(
    alt.resources_root() / "networks/default_locations.csv"
)
network = alt.Network.from_file(
    alt.resources_root() / "networks/Taconite-NoBalloon.yaml"
)

demand_file = (pl.read_csv(defaults.DEMAND_FILE)
    .filter(pl.col("Train_Type").str.contains("Intermodal"))
)

t1_import = time.perf_counter()
print(
    f"Elapsed time to import rail vehicles, locations, and network: {t1_import - t0_import:.3g} s"
)

train_planner_config = planner_config.TrainPlannerConfig(
            cars_per_locomotive={"Default": 35},
            target_cars_per_train={"Default": 90},
            loco_type_shares={'BEL': 0.5, 'Diesel_Large': 0.5},
            require_diesel=True)

t0_main = time.perf_counter()

(
    train_consist_plan,
    loco_pool,
    refuel_facilities,
    grid_emissions_factors,
    nodal_energy_prices,
    speed_limit_train_sims,
    timed_paths,
    train_consist_plan_untrimmed,
) = sim_manager.main(
    network=network,
    rail_vehicles=rail_vehicles,
    location_map=location_map,
    train_planner_config=train_planner_config,
    debug=True,
    demand_file=demand_file
)

t1_main = time.perf_counter()
print(f"Elapsed time to run `sim_manager.main()`: {t1_main - t0_main:.3g} s")

summary_df, energy_consumption_df, vehicle_log_df = lifts_simulator.run_simulation(
    train_consist_plan = train_consist_plan,
    terminal = "Allouez")

t2_main = time.perf_counter()
print(f"Elapsed time to run LIFTS simulation: {t2_main - t1_main:.3g} s")
