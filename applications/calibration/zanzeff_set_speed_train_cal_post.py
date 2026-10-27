# %%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from zanzeff_set_speed_train_cal import *

# %%

train_res_df = pd.read_csv(def_save_path / "res_df.csv")

# %%

file_info_df = pd.read_csv(def_save_path / "FileInfo.csv")
train_cal_mod_err = cval.ModelError.load(def_save_path / 'cal_mod_err.pickle')
train_val_mod_err = cval.ModelError.load(def_save_path / 'val_mod_err.pickle')

utils.cal_val_file_check_post(train_cal_mod_err, train_val_mod_err, file_info_df)

# %%

train_optimal_params = cval.min_error_selection(
    train_res_df,
    param_num=len(train_cal_mod_err.params)
)

# %% 
    # ("cd_area_loaded_square_meters", (1, 6)),
    # ("cd_area_empty_square_meters", (1, 8)),
    # # ("davis_b_seconds_per_meter", (0, 0.1)),
    # ("rolling_ratio", (0.0003, 0.003)),
    # ("bearing_res_per_axle_newtons", (40, 320)),

# user-specified params for hand tuning and checking things
train_user_params = [
    3., # drag area loaded
    3., # drag area empty (e.g. empty coal car)
    0.004, # rolling ratio
    100., # bearing
]

# %%

pyplot, plotly, = True, True
show_pyplot = False

# %%

cal_plot_save_dir = def_save_path / "plots/cal"

def get_train_cal_results(train_params) -> Tuple[dict, dict]:
    train_cal_mod_dict = train_cal_mod_err.update_params(train_params)
    train_cal_errs, train_cal_mods = train_cal_mod_err.get_errors(
        train_cal_mod_dict,
        pyplot=pyplot,
        plotly=plotly,
        show_pyplot=show_pyplot,
        plot_save_dir=cal_plot_save_dir,
        return_mods=True,
    )

    return train_cal_mod_dict, train_cal_errs, train_cal_mods


train_cal_mod_dict, train_cal_errs, train_cal_mods = get_train_cal_results(
    train_optimal_params,
    # train_user_params
)


# %%

val_plot_save_dir = def_save_path / "plots/val"

def get_train_val_results(train_params) -> Tuple[dict, dict]:
    train_val_mod_dict = train_val_mod_err.update_params(train_params)
    train_val_errs, train_val_mods = train_val_mod_err.get_errors(
        train_val_mod_dict,
        pyplot=pyplot,
        plotly=plotly,
        show_pyplot=show_pyplot,
        plot_save_dir=val_plot_save_dir,
        return_mods=True,
    )

    return train_val_mod_dict, train_val_errs, train_val_mods


train_val_mod_dict, train_val_errs, train_val_mods = get_train_val_results(
    train_optimal_params,
)

# %%

# %% plot up the aggregate results.  
exp_tractive_col = "Total Pos. Cumu. Tractive Energy [J]"

# %%

cal_exp_tractive = []
cal_mod_tractive = []
for (mod, (key, df)) in zip(train_cal_mods.values(), train_cal_mod_err.dfs.items()):
    cal_exp_tractive.append(    
        df[exp_tractive_col].iloc[-1]
    )
    cal_mod_tractive.append(
        mod.state.energy_whl_out_pos_joules
    )

# %%

val_exp_tractive = []
val_mod_tractive = []
for (mod, (key, df)) in zip(train_val_mods.values(), train_val_mod_err.dfs.items()):
    val_exp_tractive.append(    
        df[exp_tractive_col].iloc[-1]
    )
    val_mod_tractive.append(
        mod.state.energy_whl_out_pos_joules
    )

# %%

cal_rmse = np.sqrt(np.sum([
    ((cmf - cef) * 1e-9) ** 2 for cmf, cef in zip(cal_mod_tractive, cal_exp_tractive)
]) / len(cal_mod_tractive))
val_rmse = np.sqrt(np.sum([
    ((vmf - vef) * 1e-9) ** 2 for vmf, vef in zip(val_mod_tractive, val_exp_tractive)
]) / len(val_mod_tractive))


cal_mean_frac_errs = []
cal_tot_time = 0
for df, err in zip(train_cal_mod_err.dfs.values(), train_cal_errs.values()):
    t_elapsed = df['time [s]'].iloc[-1] - df['time [s]'].iloc[0]
    cal_tot_time += t_elapsed
    if df.iloc[-1][exp_tractive_col] != 0.:
        cal_mean_frac_errs.append(
            err[exp_tractive_col] / df.iloc[-1][exp_tractive_col] * t_elapsed
        )
    else:
        cal_mean_frac_errs.append(0.)
cal_mean_frac_err = np.sum(cal_mean_frac_errs) / cal_tot_time

val_mean_frac_errs = []
val_tot_time = 0
for df, err in zip(train_val_mod_err.dfs.values(), train_val_errs.values()):
    t_elapsed = df['time [s]'].iloc[-1] - df['time [s]'].iloc[0]
    val_tot_time += t_elapsed
    val_mean_frac_errs.append(
        err[exp_tractive_col] / df.iloc[-1][exp_tractive_col] * t_elapsed
    )
val_mean_frac_err = np.sum(val_mean_frac_errs) / val_tot_time
mean_frac_err = (
    val_mean_frac_err * val_tot_time + cal_mean_frac_err * cal_tot_time) / (
    val_tot_time + cal_tot_time       
)


val_rmse = np.sqrt(np.sum([
    ((vmf - vef) * 1e-9) ** 2 for vmf, vef in zip(val_mod_tractive, val_exp_tractive)
]) / len(val_mod_tractive))

fig, ax = plt.subplots()
ax.plot(
    np.array(cal_exp_tractive) / 1e9,
    np.array(cal_mod_tractive) / 1e9,
    marker = 'o',
    linestyle = '',
    label='calibration',
)
ax.plot(
    np.array(val_exp_tractive) / 1e9,
    np.array(val_mod_tractive) / 1e9,
    marker = 's',
    linestyle = '',
    label='validation',
)
ax.plot(
    [0, ax.get_xlim()[1]],
    [0, ax.get_ylim()[1]],
    color='k',
)
ax.set_xlabel('Experimental Data Cumu. Pos.\nTractive Energy [GJ]')
ax.set_ylabel("Model Cumu. Pos.\nTractive Energy [GJ]")
ax.set_title(
    "Train Tractive Effort Calibration and Validation" + 
    f"\nTime-averaged error: {mean_frac_err:.2%} " 
)
ax.legend()
ax_max = 4.1
ax.set_xlim([0, ax_max])
ax.set_ylim([0, ax_max])
plt.tight_layout()
plt.savefig(
    cal_plot_save_dir / f'../model v. exp scatter - ax_max={ax_max}.png'
)
plt.savefig(
    cal_plot_save_dir / f'../model v. exp scatter - ax_max={ax_max}.svg'
)


# %%
