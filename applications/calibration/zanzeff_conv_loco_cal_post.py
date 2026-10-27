# %%

from zanzeff_conv_loco_cal import *
import utils

# %%

cal_plot_save_path = def_save_path / "plots/cal"
cal_plot_save_path.mkdir(exist_ok=True, parents=True)
val_plot_save_path = def_save_path / "plots/val"
val_plot_save_path.mkdir(exist_ok=True, parents=True)

conv_res_df = pd.read_csv(def_save_path / "res_df.csv")
file_info_df = pd.read_csv(def_save_path / "FileInfo.csv")

conv_cal_mod_err = cval.ModelError.load(def_save_path / 'cal_mod_err.pickle')
conv_val_mod_err = cval.ModelError.load(def_save_path / 'val_mod_err.pickle')

utils.cal_val_file_check_post(conv_cal_mod_err, conv_val_mod_err, file_info_df)

conv_optimal_params = cval.min_error_selection(
    conv_res_df,
    param_num=len(conv_cal_mod_err.params)
)

conv_default = altc.Locomotive.default()

conv_default_params = [
    conv_default.fc.pwr_idle_fuel_watts,
    conv_default.gen.eta_max,
    conv_default.pwr_aux_offset_watts,
    conv_default.pwr_aux_traction_coeff,
    conv_default.edrv.eta_max,
]

# %%

pyplot, plotly, = True, True
show_pyplot = False


# %%
conv_cal_mod_dict, conv_cal_errs = utils.get_results(
    conv_cal_mod_err, 
    conv_optimal_params,
    plotly=plotly,
    pyplot=pyplot,
    show_pyplot=show_pyplot,
    cal_plot_save_path,
)
# conv_cal_mod_dict, conv_cal_errs = get_conv_cal_results(conv_default_params, True)

# %%
conv_val_mod_dict, conv_val_errs = utils.get_results(
    conv_val_mod_err,
    conv_optimal_params, 
    plotly=plotly,
    pyplot=pyplot,
    show_pyplot=show_pyplot,
    val_plot_save_path,
)
# conv_val_mod_dict, conv_val_errs = get_conv_val_results(conv_default_params, True)

# %% plot up the aggregate results.  
exp_fuel_col = "Fuel Energy [J]"

cal_exp_fuel = []
cal_mod_fuel = []
for key, df in conv_cal_mod_err.dfs.items():
    cal_exp_fuel.append(    
        df[exp_fuel_col].iloc[-1]
    )
    mod = conv_cal_mod_dict[key]
    cal_mod_fuel.append(
        mod.loco_unit.fc.state.energy_fuel_joules
    )

val_exp_fuel = []
val_mod_fuel = []
for key, df in conv_val_mod_err.dfs.items():
    val_exp_fuel.append(    
        df[exp_fuel_col].iloc[-1]
    )
    mod = conv_val_mod_dict[key]
    val_mod_fuel.append(
        mod.loco_unit.fc.state.energy_fuel_joules
    )

cal_rmse = np.sqrt(np.sum([
    ((cmf - cef) * 1e-9) ** 2 for cmf, cef in zip(cal_mod_fuel, cal_exp_fuel)
]) / len(cal_mod_fuel))
val_rmse = np.sqrt(np.sum([
    ((vmf - vef) * 1e-9) ** 2 for vmf, vef in zip(val_mod_fuel, val_exp_fuel)
]) / len(val_mod_fuel))


cal_mean_frac_errs = []
cal_tot_time = 0
for df, err in zip(conv_cal_mod_err.dfs.values(), conv_cal_errs.values()):
    t_elapsed = df['time [s]'].iloc[-1] - df['time [s]'].iloc[0]
    cal_tot_time += t_elapsed
    if df.iloc[-1]['Fuel Energy [J]'] != 0.:
        cal_mean_frac_errs.append(
            err['Fuel Energy [J]'] / df.iloc[-1]['Fuel Energy [J]'] * t_elapsed
        )
    else:
        cal_mean_frac_errs.append(0.)
cal_mean_frac_err = np.sum(cal_mean_frac_errs) / cal_tot_time

val_mean_frac_errs = []
val_tot_time = 0
for df, err in zip(conv_val_mod_err.dfs.values(), conv_val_errs.values()):
    t_elapsed = df['time [s]'].iloc[-1] - df['time [s]'].iloc[0]
    val_tot_time += t_elapsed
    val_mean_frac_errs.append(
        err['Fuel Energy [J]'] / df.iloc[-1]['Fuel Energy [J]'] * t_elapsed
    )
val_mean_frac_err = np.sum(val_mean_frac_errs) / val_tot_time
mean_frac_err = (
    val_mean_frac_err * val_tot_time + cal_mean_frac_err * cal_tot_time) / (
    val_tot_time + cal_tot_time       
)


val_rmse = np.sqrt(np.sum([
    ((vmf - vef) * 1e-9) ** 2 for vmf, vef in zip(val_mod_fuel, val_exp_fuel)
]) / len(val_mod_fuel))

fig, ax = plt.subplots()
ax.plot(
    np.array(cal_exp_fuel) / 1e9,
    np.array(cal_mod_fuel) / 1e9,
    marker = 'o',
    linestyle = '',
    label='calibration',
)
ax.plot(
    np.array(val_exp_fuel) / 1e9,
    np.array(val_mod_fuel) / 1e9,
    marker = 's',
    linestyle = '',
    label='validation',
)
ax.plot(
    [0, ax.get_xlim()[1]],
    [0, ax.get_ylim()[1]],
    color='k',
)
ax.set_xlabel('Experimental Data Fuel Energy [GJ]')
ax.set_ylabel("Model Fuel Energy [GJ]")
ax.set_title(
    "Diesel Locomotive Calibration and Validation" + 
    f"\nTime-averaged error: {mean_frac_err:.2%} " 
)
ax.legend()
plt.tight_layout()
plt.savefig(def_save_path / 'model v. exp scatter.png')
plt.savefig(def_save_path / 'model v. exp scatter.svg')

# %% save individual models to file

# for key,val in conv_cal_mod_dict.items():
#     val.to_file(str(def_save_path / (key + '.json')))
#     print(key + '.json')

# %%