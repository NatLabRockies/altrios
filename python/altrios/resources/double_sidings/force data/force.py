import pandas as pd
import matplotlib.pyplot as plt
import os


corridor_name = "clovis"
file_path = f"/Users/qianqiantong/PycharmProjects/altrios-private/altrios/python/altrios/resources/double_sidings/force data/{corridor_name}.xlsx"
output_dir = "/Users/qianqiantong/PycharmProjects/altrios-private/altrios/python/altrios/resources/double_sidings/force data/corridor_plots"
os.makedirs(output_dir, exist_ok=True)


def plot_energy_or_force(df, x_col, base_col, compare_cols, labels,
                         abs_title, pct_title, abs_ylabel, pct_ylabel,
                         filename):
    # Markers for good visibility
    markers = ["o", "s", "D", "^", "v", "P"]

    x = df[x_col]

    # Prepare percentage data
    pct_data = [(df[col] - df[base_col]) / df[base_col] * 100 for col in compare_cols]

    # Absolute data columns = base + compare
    abs_cols = [base_col] + compare_cols
    abs_labels = ["Track Chart Data"] + labels

    # ============================================================
    # Create 1-row, 2-column subplot
    # ============================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ============================================================
    # LEFT: Absolute Values
    # ============================================================
    ax = axes[0]
    for i, (col, lab) in enumerate(zip(abs_cols, abs_labels)):
        ax.plot(x, df[col], marker=markers[i], label=lab)
    ax.set_title(abs_title)
    ax.set_xlabel("Train Length (cars)")
    ax.set_ylabel(abs_ylabel)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    # ============================================================
    # RIGHT: Percentage Change
    # ============================================================
    ax = axes[1]

    # Track chart baseline = 0%
    ax.plot(x, [0] * len(x), marker=markers[0], label="Track Chart Data (Baseline)")

    for i, (pct, lab) in enumerate(zip(pct_data, labels)):
        ax.plot(x, pct, marker=markers[i + 1], label=lab)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_title(pct_title)
    ax.set_xlabel("Train Length (cars)")
    ax.set_ylabel(pct_ylabel)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()

    # Save figure
    save_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


energy_df = pd.read_excel(file_path, sheet_name="energy")
force_df = pd.read_excel(file_path, sheet_name="force")
energy_df.rename(columns={"train length": "train_length"}, inplace=True)

# ============================================================
# ENERGY SETTINGS
# ============================================================
energy_base = "track_chart_total_energy_GJ"
energy_compare_cols = [
    "usgs_filtered_total_energy_GJ",
    "usgs_no_mean_total_energy_GJ",
    "usgs_no_savgol_total_energy_GJ"
]
energy_labels_usgs = [
    "Filtered USGS Data",
    "USGS Data Without Mean Filter",
    "USGS Data Without Savgol Filter"
]

plot_energy_or_force(
    df=energy_df,
    x_col="train_length",
    base_col=energy_base,
    compare_cols=energy_compare_cols,
    labels=energy_labels_usgs,
    abs_title="Train Length vs. Total Energy Consumption",
    pct_title="Train Length vs. Total Energy % Change",
    abs_ylabel="Energy (GJ)",
    pct_ylabel="Energy Change (%)",
    filename=f"energy_{corridor_name}.png"
)

# ============================================================
# FORCE SETTINGS
# ============================================================
force_base = "track_chart_max_res_newtons"
force_compare_cols = [
    "usgs_filtered_max_res_newtons",
    "usgs_no_mean_max_res_newtons",
    "usgs_no_savgol_max_res_newtons"
]
force_labels_usgs = [
    "Filtered USGS Data",
    "USGS Data Without Median Filter",
    "USGS Data Without Savgol Filter"
]

plot_energy_or_force(
    df=force_df,
    x_col="train_length",
    base_col=force_base,
    compare_cols=force_compare_cols,
    labels=force_labels_usgs,
    abs_title="Train Length vs. Max Resistance Force",
    pct_title="Train Length vs. Max Resistance % Change",
    abs_ylabel="Max Resistance (N)",
    pct_ylabel="Resistance Change (%)",
    filename=f"force_{corridor_name}.png"
)

print("All plots generated successfully (1 row 2 col, custom markers).")
