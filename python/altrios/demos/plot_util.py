
import matplotlib.pyplot as plt
import numpy as np

import altrios as alt


def extract_bel_from_train_sim(ts: alt.SetSpeedTrainSim) -> list:
    ts_dict = ts.to_pydict()
    ts_list = ts_dict["loco_con"]["loco_vec"]
    loco_list = []
    for loco in ts_list:
        if "BatteryElectricLoco" in loco["loco_type"]:
            loco_list.append(loco)
    if not loco_list:
        print("NO BEL IS FOUND IN CONSIST")
        return False
    return loco_list


def extract_conv_from_train_sim(ts: alt.SetSpeedTrainSim) -> list:
    ts_dict = ts.to_pydict()
    ts_list = ts_dict["loco_con"]["loco_vec"]
    loco_list = []
    for loco in ts_list:
        if "ConventionalLoco" in loco["loco_type"]:
            loco_list.append(loco)
    if not loco_list:
        print("NO CONVENTIONAL LOCO IS FOUND IN CONSIST")
        return False
    return loco_list


def extract_hel_from_train_sim(ts: alt.SetSpeedTrainSim) -> list:
    ts_dict = ts.to_pydict()
    ts_list = ts_dict["loco_con"]["loco_vec"]
    loco_list = []
    for loco in ts_list:
        if "HybridLoco" in loco["loco_type"]:
            loco_list.append(loco)
    if not loco_list:
        print("NO HYBRID LOCO IS FOUND IN CONSIST")
        return False
    return loco_list


def plot_locos_from_ts(ts: alt.SetSpeedTrainSim, x: str, show_plots: bool = False):
    """
    Can take in either SetSpeedTrainSim or SpeedLimitTrainSim
    Extracts first instance of each loco_type and plots representative plots
    Offers two plotting options to put on x axis
    ts: train sim
    x: ["time","offset"]
    """
    ts_dict = ts.to_pydict()
    if isinstance(ts, alt.SpeedLimitTrainSim):
        plot_title = "Speed Limit Train Sim"
    if isinstance(ts, alt.SetSpeedTrainSim):
        plot_title = "Set Speed Train Sim"
    if x == "time" or x == "Time":
        x_axis = np.array(ts_dict["history"]["time_seconds"]) / 3_600
        x_label = "Time (hr)"
    if x == "distance" or x == "Distance":
        x_axis = np.array(ts_dict["history"]["offset_back_meters"]) / 1_000
        x_label = "Distance (km)"
    first_bel = []
    first_hel = []

    if extract_bel_from_train_sim(ts):
        first_bel = extract_bel_from_train_sim(ts)[0]
        """
        first fig
        speed vs dist or time
        soc vs dist or time
        various powers along the powertrain vs dist or time
        various cumulative energies along the powertrain vs dist or time
        """
        fig, ax = plt.subplots(4, 1, sharex=True)
        ax[0].plot(
            x_axis,
            np.array(ts_dict["history"]["pwr_whl_out_watts"]) / 1e6,
            label="tract pwr",
        )
        ax[0].set_ylabel("Power [MW]")
        ax[0].legend()

        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_aero_newtons"]) / 1e3,
            label="aero",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_rolling_newtons"]) / 1e3,
            label="rolling",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_curve_newtons"]) / 1e3,
            label="curve",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_bearing_newtons"]) / 1e3,
            label="bearing",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_grade_newtons"]) / 1e3,
            label="grade",
        )
        ax[1].set_ylabel("Force [MN]")
        ax[1].legend()

        ax[2].plot(
            x_axis,
            np.array(
                first_bel["loco_type"]["BatteryElectricLoco"]["res"]["history"]["soc"],
            ),
        )
        ax[2].set_ylabel("SOC")

        ax[-1].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
            label="achieved speed",
        )
        if isinstance(ts, alt.SpeedLimitTrainSim):
            ax[-1].plot(
                x_axis,
                ts_dict["history"]["speed_limit_meters_per_second"],
                label="limit",
            )
        ax[-1].set_xlabel(x_label)
        ax[-1].set_ylabel("Speed [m/s]")
        ax[-1].legend()
        plt.suptitle(plot_title + " " + "Train Resistance, BEL SOC and Train Speed")

        fig1, ax1 = plt.subplots(3, 1, sharex=True)
        ax1[0].plot(
            x_axis,
            np.array(ts_dict["history"]["offset_in_link_meters"]) / 1_000,
            label="current link",
        )
        ax1[0].plot(
            x_axis,
            np.array(ts_dict["history"]["offset_meters"]) / 1_000,
            label="overall",
        )
        ax1[0].legend()
        ax1[0].set_ylabel("Net Dist. [km]")

        ax1[1].plot(
            x_axis,
            ts_dict["history"]["link_idx_front"],
            linestyle="",
            marker=".",
        )
        ax1[1].set_ylabel("Link Idx Front")

        ax1[-1].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
        )
        ax1[-1].set_xlabel(x_label)
        ax1[-1].set_ylabel("Speed [m/s]")

        plt.suptitle(plot_title + " " + "Distance and Link Tracking")
        plt.tight_layout()

        fig2, ax2 = plt.subplots(3, 1, sharex=True)
        ax2[0].plot(
            x_axis,
            np.array(ts_dict["history"]["pwr_whl_out_watts"]) / 1e6,
            label="tract pwr",
        )
        ax2[0].set_ylabel("Power [MW]")
        ax2[0].legend()

        ax2[1].plot(
            x_axis,
            np.array(ts_dict["history"]["grade_front"]) * 100.0,
        )
        ax2[1].set_ylabel("Grade [%] at\nHead End")

        ax2[-1].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
        )
        ax2[-1].set_xlabel(x_label)
        ax2[-1].set_ylabel("Speed [m/s]")

        plt.suptitle(plot_title + " " + "Power and Grade Profile")
        plt.tight_layout()
        if show_plots:
            plt.show()

    if extract_conv_from_train_sim(ts) is not False:
        fig, ax = plt.subplots(3, 1, sharex=True)
        ax[0].plot(
            x_axis,
            np.array(ts_dict["history"]["pwr_whl_out_watts"]) / 1e6,
            label="tract pwr",
        )
        ax[0].set_ylabel("Power [MW]")
        ax[0].legend()

        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_aero_newtons"]) / 1e3,
            label="aero",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_rolling_newtons"]) / 1e3,
            label="rolling",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_curve_newtons"]) / 1e3,
            label="curve",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_bearing_newtons"]) / 1e3,
            label="bearing",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_grade_newtons"]) / 1e3,
            label="grade",
        )
        ax[1].set_ylabel("Force [MN]")
        ax[1].legend()
        ax[-1].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
            label="achieved speed",
        )
        if isinstance(ts, alt.SpeedLimitTrainSim):
            ax[-1].plot(
                x_axis,
                ts_dict["history"]["speed_limit_meters_per_second"],
                label="limit",
            )
        ax[-1].set_xlabel(x_label)
        ax[-1].set_ylabel("Speed [m/s]")
        ax[-1].legend()
        plt.suptitle(plot_title + " " + "Train Resistance, and Train Speed")

        fig1, ax1 = plt.subplots(3, 1, sharex=True)
        ax1[0].plot(
            x_axis,
            np.array(ts_dict["history"]["offset_in_link_meters"]) / 1_000,
            label="current link",
        )
        ax1[0].plot(
            x_axis,
            np.array(ts_dict["history"]["offset_meters"]) / 1_000,
            label="overall",
        )
        ax1[0].legend()
        ax1[0].set_ylabel("Net Dist. [km]")

        ax1[1].plot(
            x_axis,
            ts_dict["history"]["link_idx_front"],
            linestyle="",
            marker=".",
        )
        ax1[1].set_ylabel("Link Idx Front")

        ax1[-1].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
        )
        ax1[-1].set_xlabel(x_label)
        ax1[-1].set_ylabel("Speed [m/s]")

        plt.suptitle(plot_title + " " + "Distance and Link Tracking")
        plt.tight_layout()

        fig2, ax2 = plt.subplots(3, 1, sharex=True)
        ax2[0].plot(
            x_axis,
            np.array(ts_dict["history"]["pwr_whl_out_watts"]) / 1e6,
            label="tract pwr",
        )
        ax2[0].set_ylabel("Power [MW]")
        ax2[0].legend()

        ax2[1].plot(
            x_axis,
            np.array(ts_dict["history"]["grade_front"]) * 100.0,
        )
        ax2[1].set_ylabel("Grade [%] at\nHead End")

        ax2[-1].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
        )
        ax2[-1].set_xlabel(x_label)
        ax2[-1].set_ylabel("Speed [m/s]")

        plt.suptitle(plot_title + " " + "Power and Grade Profile")
        plt.tight_layout()
        if show_plots:
            plt.show()

    if extract_hel_from_train_sim(ts) is not False:
        first_hel = extract_hel_from_train_sim(ts)[0]
        fig, ax = plt.subplots(4, 1, sharex=True)
        ax[0].plot(
            x_axis,
            np.array(ts_dict["history"]["pwr_whl_out_watts"]) / 1e6,
            label="tract pwr",
        )
        ax[0].set_ylabel("Power [MW]")
        ax[0].legend()

        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_aero_newtons"]) / 1e3,
            label="aero",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_rolling_newtons"]) / 1e3,
            label="rolling",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_curve_newtons"]) / 1e3,
            label="curve",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_bearing_newtons"]) / 1e3,
            label="bearing",
        )
        ax[1].plot(
            x_axis,
            np.array(ts_dict["history"]["res_grade_newtons"]) / 1e3,
            label="grade",
        )
        ax[1].set_ylabel("Force [MN]")
        ax[1].legend()

        ax[2].plot(
            x_axis,
            np.array(first_hel["loco_type"]["HybridLoco"]["res"]["history"]["soc"]),
        )
        ax[2].set_ylabel("SOC")

        ax[-1].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
            label="achieved speed",
        )
        if isinstance(ts, alt.SpeedLimitTrainSim):
            ax[-1].plot(
                x_axis,
                ts_dict["history"]["speed_limit_meters_per_second"],
                label="limit",
            )
        ax[-1].set_xlabel(x_label)
        ax[-1].set_ylabel("Speed [m/s]")
        ax[-1].legend()
        plt.suptitle(plot_title + " " + "Train Resistance, BEL SOC and Train Speed")

        fig1, ax1 = plt.subplots(3, 1, sharex=True)
        ax1[0].plot(
            x_axis,
            np.array(ts_dict["history"]["offset_in_link_meters"]) / 1_000,
            label="current link",
        )
        ax1[0].plot(
            x_axis,
            np.array(ts_dict["history"]["offset_meters"]) / 1_000,
            label="overall",
        )
        ax1[0].legend()
        ax1[0].set_ylabel("Net Dist. [km]")

        ax1[1].plot(
            x_axis,
            ts_dict["history"]["link_idx_front"],
            linestyle="",
            marker=".",
        )
        ax1[1].set_ylabel("Link Idx Front")

        ax1[-1].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
        )
        ax1[-1].set_xlabel(x_label)
        ax1[-1].set_ylabel("Speed [m/s]")

        plt.suptitle(plot_title + " " + "Distance and Link Tracking")
        plt.tight_layout()

        fig2, ax2 = plt.subplots(3, 1, sharex=True)
        ax2[0].plot(
            x_axis,
            np.array(ts_dict["history"]["pwr_whl_out_watts"]) / 1e6,
            label="tract pwr",
        )
        ax2[0].set_ylabel("Power [MW]")
        ax2[0].legend()

        ax2[1].plot(
            x_axis,
            np.array(ts_dict["history"]["grade_front"]) * 100.0,
        )
        ax2[1].set_ylabel("Grade [%] at\nHead End")

        ax2[-1].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
        )
        ax2[-1].set_xlabel(x_label)
        ax2[-1].set_ylabel("Speed [m/s]")

        plt.suptitle(plot_title + " " + "Power and Grade Profile")
        plt.tight_layout()
        if show_plots:
            plt.show()

        fig3, ax3 = plt.subplots(3, 1, sharex=True)
        ax3[0].plot(
            x_axis,
            np.array(first_hel["history"]["pwr_out_watts"]) / 1e3,
            label="hybrid tract. pwr.",
        )
        ax3[0].plot(
            x_axis,
            np.array(
                first_hel["loco_type"]["HybridLoco"]["res"]["history"][
                    "pwr_out_electrical_watts"
                ],
            )
            / 1e3,
            label="hybrid batt. elec. pwr.",
        )
        ax3[0].set_ylabel("Power [kW]")
        ax3[0].legend()
        ax3[1].plot(
            x_axis,
            first_hel["loco_type"]["HybridLoco"]["res"]["history"]["soc"],
            label="soc",
        )
        ax3[1].plot(
            x_axis,
            first_hel["loco_type"]["HybridLoco"]["res"]["history"]["soc_chrg_buffer"],
            label="chrg buff",
        )
        ax3[1].plot(
            x_axis,
            first_hel["loco_type"]["HybridLoco"]["res"]["history"]["soc_disch_buffer"],
            label="disch buff",
        )
        ax3[1].set_ylabel("[-]")
        ax3[1].legend()
        ax3[2].plot(
            x_axis,
            ts_dict["history"]["speed_meters_per_second"],
        )
        ax3[2].set_ylabel("Speed [m/s]")
        ax3[2].set_xlabel("Times [s]")
        plt.suptitle(plot_title + " " + "Hybrid Loco Power and Buffer Profile")
        plt.tight_layout()
        if show_plots:
            plt.show()


def plot_train_level_powers(
    ts: alt.SpeedLimitTrainSim, mod_str: str,
) -> tuple[plt.Figure, plt.Axes]:
    ts_dict = ts.to_pydict()
    fig, ax = plt.subplots(3, 1, sharex=True)
    plt.suptitle("Train Power " + mod_str)
    ax[0].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["pwr_whl_out_watts"]) / 1e6,
        label="tract pwr",
    )
    ax[0].set_ylabel("Power [MW]")
    ax[0].legend()

    ax[1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["res_aero_newtons"]) / 1e3,
        label="aero",
    )
    ax[1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["res_rolling_newtons"]) / 1e3,
        label="rolling",
    )
    ax[1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["res_curve_newtons"]) / 1e3,
        label="curve",
    )
    ax[1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["res_bearing_newtons"]) / 1e3,
        label="bearing",
    )
    ax[1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["res_grade_newtons"]) / 1e3,
        label="grade",
    )
    ax[1].set_ylabel("Force [MN]")
    ax[1].legend()

    ax[-1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        ts_dict["history"]["speed_meters_per_second"],
        label="achieved",
    )
    ax[-1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        ts_dict["history"]["speed_limit_meters_per_second"],
        label="limit",
    )
    ax[-1].set_xlabel("Time [hr]")
    ax[-1].set_ylabel("Speed [m/s]")
    ax[-1].legend()

    return fig, ax


def plot_train_network_info(
    ts: alt.SpeedLimitTrainSim, mod_str: str,
) -> tuple[plt.Figure, plt.Axes]:
    ts_dict = ts.to_pydict()

    fig, ax = plt.subplots(3, 1, sharex=True)
    plt.suptitle("Train Position in Network " + mod_str)
    ax[0].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["offset_in_link_meters"]) / 1_000,
        label="current link",
    )
    ax[0].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["offset_meters"]) / 1_000,
        label="overall",
    )
    ax[0].legend()
    ax[0].set_ylabel("Net Dist. [km]")

    ax[1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        ts_dict["history"]["link_idx_front"],
        linestyle="",
        marker=".",
    )
    ax[1].set_ylabel("Link Idx Front")

    ax[-1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        ts_dict["history"]["speed_meters_per_second"],
    )
    ax[-1].set_xlabel("Time [hr]")
    ax[-1].set_ylabel("Speed [m/s]")

    plt.tight_layout()

    return fig, ax


def plot_consist_pwr(
    ts: alt.SpeedLimitTrainSim, mod_str: str,
) -> tuple[plt.Figure, plt.Axes]:
    ts_dict = ts.to_pydict()

    fig, ax = plt.subplots(3, 1, sharex=True)
    plt.suptitle("Loco. Consist " + mod_str)
    ax[0].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["pwr_whl_out_watts"]) / 1e6,
        label="consist tract pwr",
    )
    ax[0].set_ylabel("Power [MW]")
    ax[0].legend()

    ax[1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        np.array(ts_dict["history"]["grade_front"]) * 100.0,
    )
    ax[1].set_ylabel("Grade [%] at\nHead End")

    ax[-1].plot(
        np.array(ts_dict["history"]["time_seconds"]) / 3_600,
        ts_dict["history"]["speed_meters_per_second"],
    )
    ax[-1].set_xlabel("Time [hr]")
    ax[-1].set_ylabel("Speed [m/s]")

    return fig, ax


def plot_hel_pwr_and_soc(
    ts: alt.SpeedLimitTrainSim, mod_str: str, hel_type="HybridLoco",
) -> tuple[plt.Figure, plt.Axes]:
    ts_dict = ts.to_pydict()
    loco_list = []
    for loco in ts_dict["loco_con"]["loco_vec"]:
        if "HybridLoco" in loco["loco_type"]:
            loco_list.append(loco)
    hybrid_loco = loco_list[0]
    fig, ax = plt.subplots(3, 1, sharex=True)
    plt.suptitle("Hybrid Locomotive " + mod_str)

    ax_idx = 0
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.array(hybrid_loco["history"]["pwr_out_watts"]) / 1e3,
        label="tract. pwr.",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.array(
            hybrid_loco["loco_type"][hel_type]["res"]["history"]["pwr_disch_max_watts"],
        )
        / 1e3,
        label="batt. max disch. pwr",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.array(
            hybrid_loco["loco_type"][hel_type]["res"]["history"]["pwr_charge_max_watts"],
        )
        / 1e3,
        label="batt. max chrg. pwr",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.array(
            hybrid_loco["loco_type"][hel_type]["res"]["history"][
                "pwr_out_electrical_watts"
            ],
        )
        / 1e3,
        label="batt. elec. pwr.",
    )
    pwr_gen_elect_out = np.array(
        hybrid_loco["loco_type"][hel_type]["gen"]["history"]["pwr_elec_prop_out_watts"],
    ) + np.array(
        hybrid_loco["loco_type"][hel_type]["gen"]["history"]["pwr_elec_aux_watts"],
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        pwr_gen_elect_out / 1e3,
        label="gen. elec. pwr.",
    )
    y_max = ax[ax_idx].get_ylim()[1]
    ax[ax_idx].set_ylim([-y_max, y_max])
    ax[ax_idx].set_ylabel("Power [kW]")
    ax[ax_idx].legend()

    ax_idx += 1
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        hybrid_loco["loco_type"][hel_type]["res"]["history"]["soc"],
        label="soc",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"][1:],
        hybrid_loco["loco_type"][hel_type]["res"]["history"]["soc_chrg_buffer"][1:],
        label="chrg buff",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"][1:],
        hybrid_loco["loco_type"][hel_type]["res"]["history"]["soc_disch_buffer"][1:],
        label="disch buff",
    )
    # TODO: add static min and max soc bounds to plots
    # TODO: make a plot util for any type of locomotive that will plot all the stuff
    ax[ax_idx].set_ylabel("[-]")
    ax[ax_idx].legend()

    ax_idx += 1
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        ts_dict["history"]["speed_meters_per_second"],
    )
    ax[ax_idx].set_ylabel("Speed [m/s]")
    ax[ax_idx].set_xlabel("Times [s]")
    plt.tight_layout()

    return fig, ax


def plot_bel_pwr_and_soc(
    ts: alt.SpeedLimitTrainSim, mod_str: str, bel_type="BatteryElectricLoco",
) -> tuple[plt.Figure, plt.Axes]:
    ts_dict = ts.to_pydict()
    batt_loco = ts_dict["loco_con"]["loco_vec"][0]
    fig, ax = plt.subplots(3, 1, sharex=True)
    plt.suptitle("Battery Electric Locomotive " + mod_str)

    ax_idx = 0
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.array(batt_loco["history"]["pwr_out_watts"]) / 1e3,
        label="tract. pwr.",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.array(
            batt_loco["loco_type"][bel_type]["res"]["history"]["pwr_disch_max_watts"],
        )
        / 1e3,
        label="batt. max disch. pwr",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.array(
            batt_loco["loco_type"][bel_type]["res"]["history"]["pwr_charge_max_watts"],
        )
        / 1e3,
        label="batt. max chrg. pwr",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        np.array(
            batt_loco["loco_type"][bel_type]["res"]["history"][
                "pwr_out_electrical_watts"
            ],
        )
        / 1e3,
        label="batt. elec. pwr.",
    )
    y_max = ax[ax_idx].get_ylim()[1]
    ax[ax_idx].set_ylim([-y_max, y_max])
    ax[ax_idx].set_ylabel("Power [kW]")
    ax[ax_idx].legend()

    ax_idx += 1
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        batt_loco["loco_type"][bel_type]["res"]["history"]["soc"],
        label="soc",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"][1:],
        batt_loco["loco_type"][bel_type]["res"]["history"]["soc_chrg_buffer"][1:],
        label="chrg buff",
    )
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"][1:],
        batt_loco["loco_type"][bel_type]["res"]["history"]["soc_disch_buffer"][1:],
        label="disch buff",
    )
    ax[ax_idx].set_ylabel("[-]")
    ax[ax_idx].legend()

    ax_idx += 1
    # TODO: add static min and max soc bounds to plots
    # TODO: make a plot util for any type of locomotive that will plot all the stuff
    ax[ax_idx].plot(
        ts_dict["history"]["time_seconds"],
        ts_dict["history"]["speed_meters_per_second"],
    )
    ax[ax_idx].set_ylabel("Speed [m/s]")
    ax[ax_idx].set_xlabel("Times [s]")
    ax[ax_idx].legend()
    plt.tight_layout()

    return fig, ax
