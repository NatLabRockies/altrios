# %%
# Script for running the Wabtech BEL consist for sample data from Barstow to Stockton
# Consist comprises [2X Tier 4](https://www.wabteccorp.com/media/3641/download?inline)
# + [1x BEL](https://www.wabteccorp.com/media/466/download?inline)


import altrios as alt
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns

sns.set_theme()

SHOW_PLOTS = alt.utils.show_plots()


SAVE_INTERVAL = 1


pt = alt.PowerTrace.default()

hel: alt.Locomotive = alt.Locomotive.default_hybrid_electric_loco()

# instantiate battery model
t0 = time.perf_counter()
sim = alt.LocomotiveSimulation(hel, pt, SAVE_INTERVAL)
t1 = time.perf_counter()
print(f"Time to load: {t1 - t0:.3g}")

# simulate
t0 = time.perf_counter()
sim.walk()
t1 = time.perf_counter()
print(f"Time to simulate: {t1 - t0:.5g}")

sim_dict = sim.to_pydict()
hel_rslt = sim_dict["loco_unit"]
t_s = np.array(sim_dict["power_trace"]["time_seconds"])

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 12))

# power
fontsize = 16

i = 0

ax[i].plot(
    t_s,
    np.array(
        hel_rslt["loco_type"]["HybridLoco"]["res"]["history"]["pwr_out_chemical_watts"]
    )
    * 1e-6,
    label="pwr_out_chem",
)
ax[i].plot(
    t_s,
    np.array(hel_rslt["history"]["pwr_out_watts"]) * 1e-6,
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

i += 1
ax[i].plot(
    t_s,
    np.array(hel_rslt["loco_type"]["HybridLoco"]["res"]["history"]["soc"]),
    label="SOC",
)
ax[i].set_ylabel("SOC", fontsize=fontsize)
ax[i].tick_params(labelsize=fontsize)

if SHOW_PLOTS:
    plt.tight_layout()
    plt.show()
