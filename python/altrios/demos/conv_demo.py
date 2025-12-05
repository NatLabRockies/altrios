# %%
# Script for running the Wabtech BEL consist for sample data from Barstow to Stockton
# Consist comprises [Tier 4](https://www.wabteccorp.com/media/3641/download?inline)


import altrios as alt
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns

sns.set_theme()

SHOW_PLOTS = alt.utils.show_plots()

# %%
SAVE_INTERVAL = 1
# load hybrid consist
t0 = time.perf_counter()
fc = alt.FuelConverter.default()
gen = alt.Generator.default()
edrv = alt.ElectricDrivetrain.default()

conv = alt.Locomotive.build_conventional_loco(
    fuel_converter=fc,
    generator=gen,
    drivetrain=edrv,
    loco_params=alt.LocoParams(
        pwr_aux_offset_watts=13e3,
        pwr_aux_traction_coeff_ratio=1.1e-3,
        force_max_newtons=667.2e3,
    ),
    save_interval=SAVE_INTERVAL,
)


# %%

pt = alt.PowerTrace.default()

sim = alt.LocomotiveSimulation(conv, pt, SAVE_INTERVAL)
t1 = time.perf_counter()

print(f"Time to load: {t1 - t0:.3g}")

# simulate
t0 = time.perf_counter()
sim.walk()
t1 = time.perf_counter()
print(f"Time to simulate: {t1 - t0:.5g}")


# %%


sim_dict = sim.to_pydict()
conv_rslt = sim_dict["loco_unit"]
t_s = np.array(sim_dict["power_trace"]["time_seconds"])

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 12))

# power
fontsize = 16

i = 0

ax[i].plot(
    t_s,
    np.array(
        conv_rslt["loco_type"]["ConventionalLoco"]["fc"]["history"]["pwr_fuel_watts"]
    )
    * 1e-6,
    label="fc pwr_out_fuel",
)
ax[i].plot(
    t_s,
    np.array(
        conv_rslt["loco_type"]["ConventionalLoco"]["fc"]["history"]["pwr_out_max_watts"]
    )
    * 1e-6,
    label="fc pwr_out_max",
)
ax[i].plot(
    t_s,
    np.array(conv_rslt["history"]["pwr_out_max_watts"]) * 1e-6,
    label="loco pwr_out_max",
)
ax[i].plot(
    t_s,
    np.array(conv_rslt["history"]["pwr_out_watts"]) * 1e-6,
    label="loco pwr_out",
)
ax[i].plot(
    t_s,
    np.array(sim_dict["power_trace"]["pwr_watts"]) * 1e-6,
    linestyle="--",
    label="power_trace",
)

ax[i].tick_params(labelsize=fontsize)

ax[i].set_ylabel("Power [MW]", fontsize=fontsize)
ax[i].legend(fontsize=fontsize)

i += 1
ax[i].plot(
    t_s,
    np.array(sim_dict["loco_unit"]["history"]["pwr_out_watts"]),
)
ax[i].set_ylabel("Total Tractive\nEffort [MW]", fontsize=fontsize)

if SHOW_PLOTS:
    plt.tight_layout()
    plt.show()
# %%
