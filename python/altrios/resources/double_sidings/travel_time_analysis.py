import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import os

def process_timetables(folder_path, freeflow_time_hr, length_mi, output_excel):
    results_tab1 = []
    results_tab2 = []

    for file in glob.glob(os.path.join(folder_path, "travel_time_dispatch_schedule_daily_throughput_*.csv")):
        basename = os.path.basename(file)
        daily_volume = int(basename.split("_")[-1].replace(".csv", ""))
        headway = 24 / (daily_volume / 2)
        daily_trains = math.ceil(daily_volume / 100)

        df = pd.read_csv(file)

        df["day"] = (df["Departure_Time_Actual_Hr"] // 24) + 1
        df = df[(df["Departure_Time_Actual_Hr"] >= 7 * 24) &
                (df["Departure_Time_Actual_Hr"] <= 14 * 24)]
        if df.empty:
            continue

        # ---------- tab1 (7_days) ----------
        daily_avg = df.groupby("day")["Travel_Time_Hr"].mean().reset_index(name="avg_travel_time_hr")
        daily_avg["daily_volume"] = daily_volume
        daily_avg["headway"] = headway
        daily_avg["daily_trains"] = daily_trains
        daily_avg["freeflow_time_hr"] = freeflow_time_hr
        daily_avg["avg_delay_hr"] = daily_avg["avg_travel_time_hr"] - freeflow_time_hr
        daily_avg["normalized_delay_min_per_100_train_miles"] = (60 * daily_avg["avg_delay_hr"]) / (length_mi / 100) # min/(100 t-m)
        results_tab1.append(daily_avg)

        # ---------- tab2 (1_week) ----------
        overall_avg = df["Travel_Time_Hr"].mean()
        results_tab2.append({
            "daily_volume": daily_volume,
            "headway": headway,
            "daily_trains": daily_trains,
            "freeflow_time_hr": freeflow_time_hr,
            "avg_travel_time_hr": overall_avg,
            "avg_delay_hr": overall_avg - freeflow_time_hr,
            "normalized_delay_min_per_100_train_miles": (60 * (overall_avg - freeflow_time_hr)) / (length_mi / 100),
        })

    tab1 = pd.concat(results_tab1, ignore_index=True).round(2)
    tab1 = tab1.sort_values(by=["daily_trains", "day"]).reset_index(drop=True)

    tab2 = pd.DataFrame(results_tab2).round(2)
    tab2 = tab2.sort_values(by="daily_trains").reset_index(drop=True)

    with pd.ExcelWriter(output_excel) as writer:
        tab1.to_excel(writer, sheet_name="7_days", index=False)
        tab2.to_excel(writer, sheet_name="1_week", index=False)

    return tab1, tab2


def plot_exponential_fit(tab1, results_folder, output_file="delay_vs_trains.png"):
    grouped = tab1.groupby("daily_trains")["normalized_delay_min_per_100_train_miles"]
    means = grouped.mean()
    stds = grouped.std()

    x = means.index.values
    y = means.values
    yerr = stds.values

    # def exp_func(x, a, b):
    #     return a * np.exp(b * x)
    #
    # # popt, _ = curve_fit(exp_func, x, y, p0=(1, 0.01))  # 初始猜测值
    #
    # plt.figure(figsize=(8, 6))
    # plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, label="Mean ± Std")
    # plt.plot(x, exp_func(x, *popt), 'r-',
    #          label=f"Fit: y = {popt[0]:.2f} * exp({popt[1]:.4f} * x)")

    mask = y > 0
    x_pos = x[mask]
    y_pos = y[mask]

    log_y = np.log(y_pos)
    coeffs = np.polyfit(x_pos, log_y, 1)  # b, ln(a)
    b = coeffs[0]
    ln_a = coeffs[1]
    a = np.exp(ln_a)
    plt.figure(figsize=(8, 6))
    plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, label="Mean ± Std")
    plt.plot(x_pos, a * np.exp(b * x_pos), 'r-',
             label=f"Fit: y = {a:.2f} * exp({b:.4f} * x)")

    plt.xlabel("Daily Trains")
    plt.ylabel("Normalized Delay (hr / 100 train-miles)")
    plt.title("Exponential Fit (log-transform) of Normalized Delay vs Daily Trains")
    plt.legend()
    plt.grid(True)

    plt.xlabel("Daily Trains")
    plt.ylabel("Normalized Delay (hr / 100 train-miles)")
    plt.title("Exponential Fit of Normalized Delay vs Daily Trains")
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(results_folder, output_file)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    # plt.close()
    print(f"The plot saved to {output_path}")


results_folder = "/Users/qianqiantong/PycharmProjects/altrios-private/altrios/python/altrios/resources/double_sidings/results/"
freeflow_time_hr = 5.86 #6.52
length_mi = 337.0
output_excel = os.path.join(results_folder, "results_summary.xlsx")
tab1, tab2 = process_timetables(results_folder, freeflow_time_hr, length_mi, output_excel)
# plot_exponential_fit(tab1, results_folder, "delay_vs_trains.png")