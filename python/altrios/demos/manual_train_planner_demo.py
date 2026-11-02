# %%
from altrios import sim_manager_manual
from altrios import utilities, defaults
import altrios as alt
from altrios.train_planner import planner_config, manual_train_planner
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import time
import seaborn as sns
from pathlib import Path

sns.set_theme()

t0_total = time.perf_counter()

SHOW_PLOTS = alt.utils.show_plots()
# %

plot_dir = Path() / "plots"
# make the dir if it doesn't exist
plot_dir.mkdir(exist_ok=True)
#%%

consist_plan = pl.read_csv(alt.resources_root() / "demo_data/consist plan for manual planner demo.csv")

location_map = alt.import_locations(
    alt.resources_root() / "networks/default_locations.csv"
)
network = alt.Network.from_file(
    alt.resources_root() / "networks/Taconite-NoBalloon.yaml"
)

loco_map = {'Diesel_Large' : 'Diesel_Large',
            'BEL' : 'BEL'}

rail_vehicles = [
    alt.RailVehicle.from_file(vehicle_file, skip_init=False)
    for vehicle_file in Path(alt.resources_root() / "rolling_stock/").glob("*.yaml")
]

t0_main = time.perf_counter()
#not passing in a trainplanner config here because we do not need to specify train length or any other parameters like
#sim_manager_demo.py.  This example is just replaying trains that have already been planned.
(
    train_consist_plan,
    loco_pool,
    refuel_facilities,
    grid_emissions_factors,
    nodal_energy_prices,
    speed_limit_train_sims,
    timed_paths,
    train_consist_plan_untrimmed,
) = sim_manager_manual.main(
    network=network,
    rail_vehicles=rail_vehicles, #double check this to see if it is actually need in sim_manager_manual.py
    location_map=location_map,
    consist_plan= consist_plan,
    loco_map=loco_map,
    debug=True,
)


# train_consist_plan, loco_pool, refuelers, speed_limit_train_sims, est_time_nets = (
#     manual_train_planner(consist_plan, loco_map)
# )

#%%
t1_main = time.perf_counter()
print(f"Elapsed time to run `sim_manager.main()`: {t1_main-t0_main:.3g} s")

# %%
t0_train_sims = time.perf_counter()
speed_limit_train_sims.set_save_interval(100)
(sims, refuel_sessions) = alt.run_speed_limit_train_sims(
    speed_limit_train_sims=speed_limit_train_sims,
    network=network,
    train_consist_plan_py=train_consist_plan,
    loco_pool_py=loco_pool,
    refuel_facilities_py=refuel_facilities,
    timed_paths=[alt.TimedLinkPath.from_pydict(tp) for tp in timed_paths],
)
t1_train_sims = time.perf_counter()
print(f"Elapsed time to run train sims: {t1_train_sims-t0_train_sims:.3g} s")
t_train_time = sum([sim["state"]["time_seconds"] for sim in sims.to_pydict()])
print(f"Total train-seconds simulated: {t_train_time} s")

# %%
t0_summary_sims = time.perf_counter()
speed_limit_train_sims.set_save_interval(None)
(summary_sims, summary_refuel_sessions) = alt.run_speed_limit_train_sims(
    speed_limit_train_sims=speed_limit_train_sims,
    network=network,
    train_consist_plan_py=train_consist_plan,
    loco_pool_py=loco_pool,
    refuel_facilities_py=refuel_facilities,
    timed_paths=[alt.TimedLinkPath.from_pydict(tp) for tp in timed_paths],
)
t1_summary_sims = time.perf_counter()
print(
    f"Elapsed time to build and run summary sims: {t1_summary_sims-t0_summary_sims:.3g} s"
)

# %%
t0_tolist = time.perf_counter()
sims_list = sims.to_pydict()
t1_tolist = time.perf_counter()
print(f"Elapsed time to run `tolist()`: {t1_tolist-t0_tolist:.3g} s")

sim0 = sims_list[0]
# sim0 = alt.SpeedLimitTrainSim.from_bincode(
#     sim0.to_bincode())  # to support linting


# %%

t0_main = time.perf_counter()
e_total_fuel_mj = summary_sims.get_energy_fuel_joules(annualize=False) / 1e9
t1_main = time.perf_counter()

print(f"Elapsed time to get total fuel energy: {t1_main-t0_main:.3g} s")
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


# %%

if SHOW_PLOTS:
    for idx, sim_dict in enumerate(sims_list[:10]):
        loco0 = next(iter(sim_dict["loco_con"]["loco_vec"]))
        loco0_type = next(iter(loco0["loco_type"].values()))

        if len(sim_dict["loco_con"]["loco_vec"]) > 1:
            loco1 = next(iter(sim_dict["loco_con"]["loco_vec"]))
            loco1_type = next(iter(loco1["loco_type"].values()))
            # plt.suptitle(f"sim #: {idx}")
        number_of_plots = 1
        if "fc" in loco0_type:
            number_of_plots += 1
        if "res" in loco1_type:
            number_of_plots += 1
        fig, ax = plt.subplots(number_of_plots, 1, sharex=True)
        fig.suptitle(f"sim #: {idx + 1}")
        ax_idx = -1
        if "fc" in loco0_type:
            ax_idx += 1
            ax[ax_idx].plot(
                np.array(sim_dict["history"]["time_seconds"]) / 3_600,
                np.array(loco0_type["fc"]["history"]["pwr_fuel_watts"]) / 1e6,
                # label='fuel'
            )

            ax[ax_idx].set_ylabel("Single Loco.\nFuel Power [MW]")

        if "res" in loco1_type:
            ax_idx += 1
            ax[ax_idx].plot(
                np.array(sim_dict["history"]["time_seconds"]) / 3_600,
                loco1_type["res"]["history"]["soc"],
            )
            ax[ax_idx].set_ylabel("SOC")

        ax[-1].plot(
            np.array(sim_dict["history"]["time_seconds"]) / 3_600,
            sim_dict["history"]["speed_meters_per_second"],
            label="actual",
        )
        ax[-1].plot(
            np.array(sim_dict["history"]["time_seconds"]) / 3_600,
            sim_dict["history"]["speed_limit_meters_per_second"],
            label="limit",
        )

        ax[-1].legend()
        ax[-1].set_xlabel("Time [hr]")
        ax[-1].set_ylabel("Speed [m/s]")

        fig.tight_layout()

        if SHOW_PLOTS:
            plt.show()

# %%


