import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

import altrios as alt
from altrios import sim_manager, utilities, defaults
from altrios.train_planner import planner_config, manual_dispatch_demand

sns.set_theme()

SHOW_PLOTS = alt.utils.show_plots()

# %%

parser = argparse.ArgumentParser()

parser.add_argument("--network", required=True)
parser.add_argument("--locations", required=True)
parser.add_argument("--dispatch", required=True)
parser.add_argument("--output", required=True)

args = parser.parse_args()

network_path = Path(args.network)
locations_path = Path(args.locations)
dispatch_path = Path(args.dispatch)
output_path = Path(args.output)

output_path.mkdir(parents=True, exist_ok=True)


# %%

t0_total = time.perf_counter()
t0_import = time.perf_counter()

rail_vehicles = [
    alt.RailVehicle.from_file(vehicle_file, skip_init=False)
    for vehicle_file in (alt.resources_root() / "rolling_stock").glob("*.yaml")
]

location_map = alt.import_locations(str(locations_path))
network = alt.Network.from_file(str(network_path))

t1_import = time.perf_counter()
print(f"Import time: {t1_import - t0_import:.3g} s")


# %%
dispatch_scheduler = manual_dispatch_demand.manual_dispatch_demand(
    str(dispatch_path)
)

train_planner_config = planner_config.TrainPlannerConfig(
    cars_per_locomotive={"Default": 50},
    target_cars_per_train={"Default": 90},
    loco_type_shares={"BEL": 0, "Diesel_Large": 1},
    require_diesel=True,
    dispatch_scheduler=dispatch_scheduler,
)


# %%
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
    travel_time,
) = sim_manager.main(
    network=network,
    rail_vehicles=rail_vehicles,
    location_map=location_map,
    train_planner_config=train_planner_config,
    debug=False,
)

t1_main = time.perf_counter()
print(f"sim_manager.main() time: {t1_main - t0_main:.3g} s")


# %%

speed_limit_train_sims.set_save_interval(100)

t0_train = time.perf_counter()

sims, refuel_sessions = alt.run_speed_limit_train_sims(
    speed_limit_train_sims=speed_limit_train_sims,
    network=network,
    train_consist_plan_py=train_consist_plan,
    loco_pool_py=loco_pool,
    refuel_facilities_py=refuel_facilities,
    timed_paths=[alt.TimedLinkPath.from_pydict(tp) for tp in timed_paths],
)

t1_train = time.perf_counter()
print(f"Train sims time: {t1_train - t0_train:.3g} s")


# %%

train_times = pl.DataFrame(travel_time)

output_file = output_path / f"travel_time_{dispatch_path.name}"
train_times.write_csv(output_file)

print(f"Travel time saved to {output_file}")


# %%

speed_limit_train_sims.set_save_interval(None)

summary_sims, summary_refuel_sessions = alt.run_speed_limit_train_sims(
    speed_limit_train_sims=speed_limit_train_sims,
    network=network,
    train_consist_plan_py=train_consist_plan,
    loco_pool_py=loco_pool,
    refuel_facilities_py=refuel_facilities,
    timed_paths=[alt.TimedLinkPath.from_pydict(tp) for tp in timed_paths],
)

e_total_fuel_mj = summary_sims.get_energy_fuel_joules(annualize=False) / 1e9

v_total_fuel_gal = (
    summary_sims.get_energy_fuel_joules(annualize=False)
    / 1e3
    / defaults.LHV_DIESEL_KJ_PER_KG
    / defaults.RHO_DIESEL_KG_PER_M3
    * utilities.LITER_PER_M3
    * utilities.GALLONS_PER_LITER
)

print(f"Total fuel energy: {e_total_fuel_mj:.3g} GJ")
print(f"Total fuel used: {v_total_fuel_gal:.3g} gallons")
print(f"Total runtime: {time.perf_counter() - t0_total:.3g} s")


# %%

if SHOW_PLOTS:
    sims_list = sims.to_pydict()

    for idx, sim_dict in enumerate(sims_list[:5]):

        fig, ax = plt.subplots(1, 1)
        ax.plot(
            np.array(sim_dict["history"]["time_seconds"]) / 3600,
            sim_dict["history"]["speed_meters_per_second"],
            label="actual",
        )
        ax.plot(
            np.array(sim_dict["history"]["time_seconds"]) / 3600,
            sim_dict["history"]["speed_limit_meters_per_second"],
            label="limit",
        )
        ax.set_xlabel("Time [hr]")
        ax.set_ylabel("Speed [m/s]")
        ax.legend()
        plt.show()