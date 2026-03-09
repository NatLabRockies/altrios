import os
import glob
import random
import subprocess
import pandas as pd
import altrios as alt


ROOT = alt.resources_root()

CORRIDOR_ROOT = ROOT / "double_sidings/output"
TIMETABLE_ROOT = ROOT / "double_sidings/timetable"
RESULT_ROOT = ROOT / "double_sidings/results"


SIM_DAYS = 21
SIM_HOURS = SIM_DAYS * 24
MAX_SIM_SEC = 500

TRAIN_CARS = 100
TRAIN_TYPE = "Intermodal"
TONS_PER_TRAIN = 4930.0
HP_REQUIRED = 14790.0

THROUGHPUT_LIST = list(range(1, 25))
FREEFLOW_ONEWAY_TRAINS = 1

WINDOW_DAY_START = 7
WINDOW_DAY_END = 14


os.makedirs(TIMETABLE_ROOT, exist_ok=True)
os.makedirs(RESULT_ROOT, exist_ok=True)


# Generate dispatch schedule
def generate_schedule(corridor, oneway_trains_per_day):

    timetable_path = TIMETABLE_ROOT / corridor
    os.makedirs(timetable_path, exist_ok=True)

    origin, destination = corridor.split("-")

    rows = []

    # FREEFLOW
    if oneway_trains_per_day == FREEFLOW_ONEWAY_TRAINS:

        for direction in [(origin, destination), (destination, origin)]:

            rows.append([
                12.0,
                12.0,
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
            "Origin", "Destination",
            "Train_Type",
            "Number_of_Cars",
            "Cars_Loaded",
            "Cars_Empty",
            "Containers_Empty",
            "Containers_Loaded",
            "Tons_Per_Train",
            "HP_Required"
        ])

        filename = "dispatch_oneway_freeflow.csv"

    # THROUGHPUT
    else:

        headway = 24 / oneway_trains_per_day

        for direction in [(origin, destination), (destination, origin)]:

            for day in range(SIM_DAYS):

                for i in range(oneway_trains_per_day):

                    expected_time = day * 24 + (i + 0.5) * headway

                    dispatch_delay = random.uniform(-12, 12)

                    actual_time = max(
                        0,
                        min(SIM_HOURS - 1, expected_time + dispatch_delay)
                    )

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
            "Origin", "Destination",
            "Train_Type",
            "Number_of_Cars",
            "Cars_Loaded",
            "Cars_Empty",
            "Containers_Empty",
            "Containers_Loaded",
            "Tons_Per_Train",
            "HP_Required"
        ])

        filename = f"dispatch_oneway_{oneway_trains_per_day}.csv"

    file_path = timetable_path / filename
    df.to_csv(file_path, index=False)

    return file_path



# Run ALTRIOS simulation
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

        print("[Warning] Simulation timeout, skipping.")
        return False

    except subprocess.CalledProcessError:

        print("[Warning] Simulation crashed, skipping.")
        return False


# Compute directional travel times

def compute_directional_travel_times(
    travel_time_csv,
    corridor,
    filter_middle_week=True,
    use_actual_runtime=False
):

    origin, destination = corridor.split("-")

    df = pd.read_csv(travel_time_csv)

    if df.empty:
        return None

    # freeflow runtime
    if use_actual_runtime:
        df["Travel_Time_Hr"] = (
            df["Arrival_Time_Actual_Hr"] -
            df["Departure_Time_Actual_Hr"]
        )

    # window filter
    if filter_middle_week:

        df["day"] = (df["Departure_Time_Planned_Hr"] // 24).astype(int) + 1

        df = df[(df["day"] >= WINDOW_DAY_START) & (df["day"] <= WINDOW_DAY_END)]

        if df.empty:
            return None

    df_od = df[
        (df["Origin_ID"] == origin) &
        (df["Destination_ID"] == destination)
    ]

    df_do = df[
        (df["Origin_ID"] == destination) &
        (df["Destination_ID"] == origin)
    ]

    od_avg = df_od["Travel_Time_Hr"].mean() if not df_od.empty else None
    do_avg = df_do["Travel_Time_Hr"].mean() if not df_do.empty else None

    return {
        "origin": origin,
        "destination": destination,
        "fwd": od_avg,
        "bwd": do_avg,
    }


def run_freeflow(corridor):

    print("\n--- Running FREEFLOW ---")

    dispatch_file = generate_schedule(corridor, FREEFLOW_ONEWAY_TRAINS)

    ok = run_simulation(corridor, dispatch_file, MAX_SIM_SEC)

    if not ok:
        return None

    result_folder = RESULT_ROOT / corridor

    travel_file = glob.glob(
        str(result_folder / "travel_time_dispatch_oneway_freeflow.csv")
    )

    if not travel_file:
        print("Freeflow result file not found.")
        return None

    return compute_directional_travel_times(
        travel_file[0],
        corridor,
        filter_middle_week=False,
        use_actual_runtime=True
    )


def process_results(corridor, freeflow_stats):

    result_folder = RESULT_ROOT / corridor

    rows_fwd = []
    rows_bwd = []
    rows_bi = []

    for file in glob.glob(str(result_folder / "travel_time_dispatch_oneway_*.csv")):

        basename = os.path.basename(file)

        if basename == "travel_time_dispatch_oneway_freeflow.csv":
            continue

        oneway = int(basename.split("_")[-1].replace(".csv", ""))

        scenario_stats = compute_directional_travel_times(
            file,
            corridor,
            filter_middle_week=True
        )

        if scenario_stats is None:
            continue

        total_trains_both = 2 * oneway
        total_volume = total_trains_both * TRAIN_CARS

        ff_fwd = freeflow_stats["fwd"]
        ff_bwd = freeflow_stats["bwd"]

        sc_fwd = scenario_stats["fwd"]
        sc_bwd = scenario_stats["bwd"]

        if sc_fwd is None or sc_bwd is None:
            continue

        ff_bi = (ff_fwd + ff_bwd) / 2
        sc_bi = (sc_fwd + sc_bwd) / 2

        delay_fwd = max(0, sc_fwd - ff_fwd)
        delay_bwd = max(0, sc_bwd - ff_bwd)
        delay_bi = max(0, sc_bi - ff_bi)

        rows_fwd.append({
            "volume_cars_per_day": total_volume,
            "oneway_trains_per_day": oneway,
            "travel_time_hr": sc_fwd,
            "freeflow_time_hr": ff_fwd,
            "delay_hr": delay_fwd,
            "delay_min": delay_fwd * 60
        })

        rows_bwd.append({
            "volume_cars_per_day": total_volume,
            "oneway_trains_per_day": oneway,
            "travel_time_hr": sc_bwd,
            "freeflow_time_hr": ff_bwd,
            "delay_hr": delay_bwd,
            "delay_min": delay_bwd * 60
        })

        rows_bi.append({
            "volume_cars_per_day": total_volume,
            "oneway_trains_per_day": oneway,
            "travel_time_hr": sc_bi,
            "freeflow_time_hr": ff_bi,
            "delay_hr": delay_bi,
            "delay_min": delay_bi * 60
        })

    df_fwd = pd.DataFrame(rows_fwd).sort_values("oneway_trains_per_day")
    df_bwd = pd.DataFrame(rows_bwd).sort_values("oneway_trains_per_day")
    df_bi = pd.DataFrame(rows_bi).sort_values("oneway_trains_per_day")

    output_excel = result_folder / f"{corridor}_results_summary.xlsx"

    with pd.ExcelWriter(output_excel) as writer:

        df_fwd.to_excel(writer, sheet_name="forward", index=False)
        df_bwd.to_excel(writer, sheet_name="backward", index=False)
        df_bi.to_excel(writer, sheet_name="bidirectional", index=False)

    print("\nResults saved to Excel:")
    print(output_excel)

    return df_fwd, df_bwd, df_bi


def run_corridor_pipeline(corridor):

    print(f"Processing corridor: {corridor}")

    freeflow_stats = run_freeflow(corridor)

    if freeflow_stats is None:
        print("Freeflow failed.")
        return

    for trains in THROUGHPUT_LIST:

        dispatch_file = generate_schedule(corridor, trains)

        ok = run_simulation(corridor, dispatch_file, MAX_SIM_SEC)

        if not ok:
            print("Simulation stopped due to timeout/crash.")
            break

    process_results(corridor, freeflow_stats)


if __name__ == "__main__":

    corridors = [
        # "Somerville-Dobbin",
        "Amarillo-FtWorth"
    ]

    for corridor in corridors:
        run_corridor_pipeline(corridor)

    print("\nAll experiments finished.")