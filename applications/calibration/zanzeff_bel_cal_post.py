# %%

from zanzeff_bel_cal import *

# %%
cal_plot_save_path = def_save_path / "plots/cal"
cal_plot_save_path.mkdir(exist_ok=True, parents=True)
val_plot_save_path = def_save_path / "plots/val"
val_plot_save_path.mkdir(exist_ok=True, parents=True)

bel_res_df = pd.read_csv(def_save_path / "res_df.csv")
file_info_df = pd.read_csv(def_save_path / "FileInfo.csv")

bel_cal_mod_err = cval.ModelError.load(def_save_path / 'cal_mod_err.pickle')
bel_val_mod_err = cval.ModelError.load(def_save_path / 'val_mod_err.pickle')

utils.cal_val_file_check_post(bel_cal_mod_err, bel_val_mod_err, file_info_df)

bel_optimal_params = cval.min_error_selection(
    bel_res_df,
    param_num=len(bel_cal_mod_err.params)
)

bel_default = altc.Locomotive.default_battery_electric_loco()

# note that this is not guaranteed to be consistent with `params_and_bounds` from zanzeff_bel_cal.py
bel_default_params =  [
    bel_default.res.eta_max,
    bel_default.res.eta_range,
    bel_default.pwr_aux_offset_watts,
    bel_default.pwr_aux_traction_coeff,
]

# %%

pyplot, plotly, = True, True
show_pyplot = False

# %%

bel_cal_mod_dict, bel_cal_errs = utils.get_results(
    bel_cal_mod_err,
    bel_optimal_params,
    plotly=plotly,
    pyplot=pyplot,
    show_pyplot=show_pyplot,
    plot_save_dir=cal_plot_save_path,
)


# %%
bel_val_mod_dict, bel_val_errs = utils.get_results(
    bel_val_mod_err,
    bel_optimal_params,
    plotly=plotly,
    pyplot=pyplot,
    show_pyplot=show_pyplot,
    plot_save_dir=val_plot_save_path,
)
