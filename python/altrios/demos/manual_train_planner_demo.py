from altrios import sim_manager
from altrios import utilities, defaults
import altrios as alt
from altrios.train_planner import planner_config
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import time
from pathlib import Path

t0_total = time.perf_counter()

SHOW_PLOTS = alt.utils.show_plots()

plot_dir = Path() / "plots"
# make the dir if it doesn't exist
plot_dir.mkdir(exist_ok=True)

# %%
# Use the same public network, locations, and rolling stock as `sim_manager_demo.py`
# so this demo does not depend on any proprietary datasets.
print("Loading `rail_vehicles`")
rail_vehicles = [
    alt.RailVehicle.from_file(vehicle_file, skip_init=False)
    for vehicle_file in Path(alt.resources_root() / "rolling_stock/").glob("*.yaml")
]

print("Importing `location_map`")
location_map = alt.import_locations(
    alt.resources_root() / "networks/default_locations.csv"
)

print("Loading `network`")
network = alt.Network.from_file(
    alt.resources_root() / "networks/Taconite-NoBalloon.yaml"
)

print("Loading `train_planner_config`")
train_planner_config = planner_config.TrainPlannerConfig()

# %%
# `manual_train_planner_demo` demonstrates replaying an already-planned train
# consist plan via `sim_manager.main(consist_plan_in=...)`. The example consist
# plan below was generated with the standard planner on the public Taconite
# network (the same inputs as `sim_manager_demo.py`) and is committed under
# `resources/demo_data/` so this demo stays self-contained and free of any
# proprietary data. In a production workflow `consist_plan_in` would instead come
# from your own pre-planned source.
print("Loading example `consist_plan_in`")
consist_plan_in = pl.read_csv(
    alt.resources_root() / "demo_data/manual_train_planner_demo/demo_train_consist_plan.csv"
)

# %%
print("Running `sim_manager.main` in manual mode (replaying `consist_plan_in`)")
t0_main = time.perf_counter()
# Because `consist_plan_in` is provided, the planner skips demand generation and
# scheduling and instead replays the pre-planned trains.
(
    consist_plan_out,
    loco_pool,
    refuel_facilities,
    grid_emissions_factors,
    nodal_energy_prices,
    speed_limit_train_sims,
    timed_paths,
    consist_plan_out_untrimmed,
) = sim_manager.main(
    network=network,
    rail_vehicles=rail_vehicles,
    location_map=location_map,
    consist_plan_in=consist_plan_in,
    train_planner_config=train_planner_config,
    debug=True,
)


# %%
t1_main = time.perf_counter()
print(f"Elapsed time to run `sim_manager.main()`: {t1_main - t0_main:.3g} s")

# %%
t0_train_sims = time.perf_counter()
speed_limit_train_sims.set_save_interval(100)
(sims, refuel_sessions) = alt.run_speed_limit_train_sims(
    speed_limit_train_sims=speed_limit_train_sims,
    network=network,
    train_consist_plan_py=consist_plan_out,
    loco_pool_py=loco_pool,
    refuel_facilities_py=refuel_facilities,
    timed_paths=[alt.TimedLinkPath.from_pydict(tp) for tp in timed_paths],
)
t1_train_sims = time.perf_counter()
print(f"Elapsed time to run train sims: {t1_train_sims - t0_train_sims:.3g} s")
t_train_time = sum([sim["state"]["time_seconds"] for sim in sims.to_pydict()])
print(f"Total train-seconds simulated: {t_train_time} s")

# %%
t0_summary_sims = time.perf_counter()
speed_limit_train_sims.set_save_interval(None)
(summary_sims, summary_refuel_sessions) = alt.run_speed_limit_train_sims(
    speed_limit_train_sims=speed_limit_train_sims,
    network=network,
    train_consist_plan_py=consist_plan_out,
    loco_pool_py=loco_pool,
    refuel_facilities_py=refuel_facilities,
    timed_paths=[alt.TimedLinkPath.from_pydict(tp) for tp in timed_paths],
)
t1_summary_sims = time.perf_counter()
print(
    f"Elapsed time to build and run summary sims: {t1_summary_sims - t0_summary_sims:.3g} s"
)

# %%
t0_tolist = time.perf_counter()
sims_list = sims.to_pydict()
t1_tolist = time.perf_counter()
print(f"Elapsed time to run `to_pydict()`: {t1_tolist - t0_tolist:.3g} s")

sim0 = sims_list[0]


# %%

t0_main = time.perf_counter()
e_total_fuel_mj = summary_sims.get_energy_fuel_joules(annualize=False) / 1e9
t1_main = time.perf_counter()

print(f"Elapsed time to get total fuel energy: {t1_main - t0_main:.3g} s")
print(f"Total fuel energy used: {e_total_fuel_mj:.3g} GJ")

v_total_fuel_gal = (
    summary_sims.get_energy_fuel_joules(annualize=False)
    / 1e3
    / defaults.LHV_DIESEL_KJ_PER_KG
    / defaults.RHO_DIESEL_KG_PER_M3
    * utilities.LITER_PER_M3
    * utilities.GALLONS_PER_LITER
)

print(f"Total fuel used: {v_total_fuel_gal:.3g} gallons")
print(f"Total elapsed time: {time.perf_counter() - t0_total} s")