import os
import glob
import math
import random
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altrios as alt


ROOT = alt.resources_root()
CORRIDOR_ROOT = ROOT / "double_sidings/corridor"
TIMETABLE_ROOT = ROOT / "double_sidings/timetable"
RESULT_ROOT = ROOT / "double_sidings/results"
CORRIDOR_META_FILE = ROOT / "double_sidings/corridor/corridors.xlsx"

THROUGHPUT_LIST = list(range(20, 3001, 200))

SIM_DAYS = 7  # start stats at the end of day 7
SIM_HOURS = SIM_DAYS * 24
MAX_SIM_SEC = 1000

TRAIN_CARS = 100
TRAIN_TYPE = "Intermodal"
TONS_PER_TRAIN = 4930.0
HP_REQUIRED = 14790.0

os.makedirs(TIMETABLE_ROOT, exist_ok=True)
os.makedirs(RESULT_ROOT, exist_ok=True)


def load_corridor_metadata():
    df = pd.read_excel(CORRIDOR_META_FILE)
    df["corridor"] = df["corridor"].astype(str).str.strip().str.replace(" ", "", regex=False)
    return df.set_index("corridor")[["length", "speed"]].to_dict("index")


def generate_freeflow_schedule(corridor):
    timetable_path = TIMETABLE_ROOT / corridor
    os.makedirs(timetable_path, exist_ok=True)

    origin, destination = corridor.split("-")

    rows = [
        [12, origin, destination, TRAIN_TYPE, TRAIN_CARS, 100, 0, 0, 100, TONS_PER_TRAIN, HP_REQUIRED],
        [12, destination, origin, TRAIN_TYPE, TRAIN_CARS, 100, 0, 0, 100, TONS_PER_TRAIN, HP_REQUIRED],
    ]

    df = pd.DataFrame(rows, columns=[
        "Hour", "Origin", "Destination", "Train_Type",
        "Number_of_Cars", "Cars_Loaded", "Cars_Empty",
        "Containers_Empty", "Containers_Loaded",
        "Tons_Per_Train", "HP_Required"
    ])

    file_path = timetable_path / "dispatch_schedule_dailythroughput_freeflow.csv"
    df.to_csv(file_path, index=False)
    return file_path


def generate_throughput_schedule(corridor, daily_volume):
    timetable_path = TIMETABLE_ROOT / corridor
    os.makedirs(timetable_path, exist_ok=True)

    trains_per_day = math.ceil(daily_volume / TRAIN_CARS)
    headway = 24 / trains_per_day

    origin, destination = corridor.split("-")
    rows = []

    for direction in [(origin, destination), (destination, origin)]:
        for day in range(SIM_DAYS):
            for i in range(trains_per_day):
                expected_time = day * 24 + (i + 0.5) * headway
                dispatch_delay = random.uniform(-12, 12)
                actual_time = max(0, min(SIM_HOURS - 1, expected_time + dispatch_delay))

                rows.append([
                    round(expected_time, 2),
                    round(actual_time, 2),
                    direction[0],
                    direction[1],
                    TRAIN_TYPE,
                    TRAIN_CARS,
                    100,
                    0,
                    0,
                    100,
                    TONS_PER_TRAIN,
                    HP_REQUIRED
                ])

    df = pd.DataFrame(rows, columns=[
        "Hour_expected", "Hour",
        "Origin", "Destination", "Train_Type",
        "Number_of_Cars", "Cars_Loaded", "Cars_Empty",
        "Containers_Empty", "Containers_Loaded",
        "Tons_Per_Train", "HP_Required"
    ])

    filename = f"dispatch_schedule_daily_throughput_{daily_volume}.csv"
    file_path = timetable_path / filename
    df.to_csv(file_path, index=False)
    return file_path


def run_simulation(corridor, dispatch_file, timeout_sec):
    network_path = CORRIDOR_ROOT / corridor / "Network.yaml"
    location_path = CORRIDOR_ROOT / corridor / "locations.csv"
    output_folder = RESULT_ROOT / corridor
    os.makedirs(output_folder, exist_ok=True)

    cmd = [
        "python",
        "multi_train_sim.py",
        "--network", str(network_path),
        "--locations", str(location_path),
        "--dispatch", str(dispatch_file),
        "--output", str(output_folder)
    ]

    try:
        subprocess.run(cmd, check=True, timeout=timeout_sec)
        return True

    except subprocess.TimeoutExpired:
        print(f"[Warning] Simulation timeout (> {timeout_sec}s), skipping.")
        return False

    except subprocess.CalledProcessError:
        print("[Warning] Simulation crashed, skipping.")
        return False


def compute_freeflow_time(result_folder):
    file = glob.glob(str(result_folder / "travel_time_dispatch_schedule_dailythroughput_freeflow.csv"))[0]
    df = pd.read_csv(file)

    df["day"] = (df["Departure_Time_Actual_Hr"] // 24) + 1

    if SIM_DAYS >= 21:
        df = df[(df["day"] >= 8) & (df["day"] <= 14)]
    else:
        df = df[(df["day"] >= 1) & (df["day"] <= 7)]

    freeflow_time_hr = float(df["Travel_Time_Hr"].mean())

    print(f"\nFreeflow travel time (simulation) = {freeflow_time_hr:.3f} hr")

    return freeflow_time_hr


def process_results(corridor, freeflow_time_hr, length_mi):
    result_folder = RESULT_ROOT / corridor
    output_excel = result_folder / "results_summary.xlsx"

    summary_rows = []

    for file in glob.glob(str(result_folder / "travel_time_dispatch_schedule_daily_throughput_*.csv")):
        basename = os.path.basename(file)
        daily_volume = int(basename.split("_")[-1].replace(".csv", ""))

        df = pd.read_csv(file)
        df["day"] = (df["Departure_Time_Actual_Hr"] // 24) + 1

        if SIM_DAYS >= 21:
            df = df[(df["day"] >= 8) & (df["day"] <= 14)]
        else:
            df = df[(df["day"] >= 1) & (df["day"] <= 7)]

        if df.empty:
            continue

        avg_travel_time_hr = float(df["Travel_Time_Hr"].mean())
        avg_delay_hr = avg_travel_time_hr - freeflow_time_hr

        daily_trains = math.ceil(daily_volume / 100)
        headway = 24 / (daily_volume / 2)

        normalized_delay = (60 * avg_delay_hr) / (length_mi / 100)

        summary_rows.append({
            "daily_volume": daily_volume,
            "headway": round(headway, 2),
            "daily_trains": daily_trains,
            "freeflow_time_hr": round(freeflow_time_hr, 2),
            "avg_travel_time_hr": round(avg_travel_time_hr, 2),
            "avg_delay_hr": round(avg_delay_hr, 2),
            "normalized_delay_min_per_100_train_miles": round(normalized_delay, 2),
        })

    result_summary = pd.DataFrame(summary_rows).sort_values("daily_volume")
    result_summary.to_excel(output_excel, index=False)

    print(f"Results saved to {output_excel}")
    print(result_summary)

    return result_summary


# ==========================================================
# PIPELINE
# ==========================================================

def run_corridor_pipeline(corridor, meta):

    print(f"\n==============================")
    print(f"Processing corridor: {corridor}")
    print(f"==============================")

    length_mi = float(meta["length"])
    speed_mph = float(meta["speed"])

    print(f"Length = {length_mi} mi | Speed = {speed_mph} mph")

    # freeflow
    freeflow_dispatch = generate_freeflow_schedule(corridor)
    run_simulation(corridor, freeflow_dispatch, timeout_sec=MAX_SIM_SEC)

    result_folder = RESULT_ROOT / corridor
    freeflow_time_hr = compute_freeflow_time(result_folder)

    # throughput runs
    for volume in THROUGHPUT_LIST:
        dispatch_file = generate_throughput_schedule(corridor, volume)
        run_simulation(corridor, dispatch_file, timeout_sec=MAX_SIM_SEC)

    # results
    process_results(corridor, freeflow_time_hr, length_mi)



if __name__ == "__main__":

    corridor_meta = load_corridor_metadata()

    corridors = ["Galveston-Rosenburg", "Rosenburg-Somerville", "Temple-FtWorth", "Somerville-Temple"]  #  \\ list(corridor_meta.keys())

    for corridor in corridors:
        run_corridor_pipeline(corridor, corridor_meta[corridor])

    print("\nAll corridor experiments finished.")