# %%
import altrios as alt
from altrios import rollout, defaults
from altrios.train_planner import planner_config
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme()

SHOW_PLOTS = alt.utils.show_plots()
SAVE_INTERVAL = 1 if SHOW_PLOTS else None
# %

# %%


plot_dir = Path() / "plots"
# make the dir if it doesn't exist
plot_dir.mkdir(exist_ok=True)
File = defaults.DEMAND_FILE
# targets = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75, 0.8]
train_planner_config = planner_config.TrainPlannerConfig(
    cars_per_locomotive={"Default": 50},
    target_cars_per_train={"Default": 90},
    require_diesel=True,
)
targets = [0.5]
for target in targets:
    scenario_infos, metrics = rollout.simulate_prescribed_rollout(
        max_bel_share=target,
        number_of_years=1,
        results_folder=Path(__file__).parent / "results/case study/",
        demand_file=File,
        train_planner_config=train_planner_config,
        count_unused_locomotives=False,
        write_complete_results=False,
        freight_demand_percent_growth=0,
        save_interval=SAVE_INTERVAL,
        write_metrics=True,
        network_filename_path=alt.resources_root() / "networks/Taconite-NoBalloon.yaml",
    )

# %%
# Plotting code currently just plots the first year of a multi-year simulation.
to_plot = scenario_infos[0].sims.to_pydict()

for idx, sim_dict in enumerate(to_plot[:10]):
    loco0 = next(iter(sim_dict["loco_con"]["loco_vec"]))
    loco0_type = next(iter(loco0["loco_type"].values()))

    if len(sim_dict["loco_con"]["loco_vec"]) > 1:
        loco1 = next(iter(sim_dict["loco_con"]["loco_vec"]))
        loco1_type = next(iter(loco1["loco_type"].values()))

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

    plt.tight_layout()

    if SHOW_PLOTS:
        plt.show()


# %%
