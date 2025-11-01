import pandas as pd
from pathlib import Path
import altrios.lifts.utilities
from altrios.lifts.parameters import *
import altrios.lifts.distance_single_track as layout

K, k, M, N, n_t, n_p, n_r= layout.load_layout_config_from_json()

package_root = altrios.lifts.utilities.package_root()
out_path = package_root / 'output'
out_path.mkdir(parents=True, exist_ok=True)

# Dictionary to store vehicle events
vehicle_events = {
    'crane': [],
    'hostler': [],
    'truck': [],
    'side': []
}


def record_vehicle_event(vehicle_category, vehicle, container, state, move, time, distance, speed, density, emission, type, timestamp):
    vehicle_events[vehicle_category].append({
        'vehicle_id': str(vehicle),
        'container_id':str(container),
        'state': state, # loaded or empty(idle)
        'move': move, # load or trip
        'time': time,
        'distance': distance,
        'speed': speed,
        'density': density,
        'emission': emission,
        'type': type,  # 'D'->Diesel, 'E'->Electric, 'H'->Hybrid
        'timestamp': timestamp
    })


def calculate_performance(dataframes, ic_count, oc_count):
    summary = {vehicle: {'IC Emissions': 0, 'OC Emissions': 0, 'Total Emissions': 0}
               for vehicle in dataframes.keys()}

    for vehicle, df in dataframes.items():
        for _, row in df.iterrows():
            emission = row['emission']
            container_id = str(row['container_id'])

            # IC / OC
            if container_id.startswith("OC"):
                summary[vehicle]['OC Emissions'] += emission
            else:
                summary[vehicle]['IC Emissions'] += emission

        summary[vehicle]['Total Emissions'] = (
            summary[vehicle]['IC Emissions'] + summary[vehicle]['OC Emissions']
        )

    summary_df = pd.DataFrame.from_dict(summary, orient='index')
    summary_df.reset_index(inplace=True)
    summary_df.rename(columns={'index': 'Vehicle'}, inplace=True)

    total_row = summary_df.sum(numeric_only=True).to_dict()
    total_row['Vehicle'] = 'Total'
    summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)

    average_row = {
        'Vehicle': 'Average',
        'IC Emissions': total_row['IC Emissions'] / ic_count if ic_count else 0,
        'OC Emissions': total_row['OC Emissions'] / oc_count if oc_count else 0,
        'Total Emissions': total_row['Total Emissions'] / (ic_count + oc_count) if (ic_count + oc_count) else 0,
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([average_row])], ignore_index=True)
    return summary_df


def save_energy_to_excel(state):
    df_logs_pd = {}
    df_logs_pl = []
    for vehicle_type, events in vehicle_events.items():
        df_logs_pd[vehicle_type] = pd.DataFrame(events)
        df_logs_pl.append(pl.DataFrame(events).select(pl.lit(vehicle_type).alias("vehicle_type"), pl.all()))

    df_logs_pl = pl.concat(df_logs_pl, how="diagonal_relaxed")
    ic_count = state.IC_NUM
    oc_count = state.OC_NUM

    # return summary statistics
    performance_df = calculate_performance(df_logs_pd, ic_count, oc_count)
    performance_df.rename(columns={'IC Emissions': 'IC Energy Consumption (kWh)',
                               'OC Emissions': 'OC Energy Consumption (kWh)',
                               'Total Emissions': 'Total Energy Consumption (kWh)'}, inplace=True)

    file_name = f"{state.CRANE_NUMBER}C-{state.HOSTLER_NUMBER}H_vehicle_throughput_{K}_batch_size_{k}.xlsx"
    file_path = out_path / 'single_track_results' / file_name

    with pd.ExcelWriter(file_path) as writer:
        for vehicle_type, df in df_logs_pd.items():
            df.to_excel(writer, sheet_name=vehicle_type, index=False)
        performance_df.to_excel(writer, sheet_name='performance', index=True)

    return pl.from_pandas(performance_df), df_logs_pl


def calculate_container_processing_time(container_excel_path, num_trains, train_batch_size, daily_throughput, ic_delay_time, oc_delay_time):
    df = pd.read_excel(container_excel_path)

    df['type'] = df['container_id'].apply(lambda x: 'IC' if str(x).isdigit() else 'OC' if str(x).startswith('OC-') else 'Unknown')

    if (daily_throughput/2) % train_batch_size == 0:
        full_load_limit = num_trains * train_batch_size
    else:
        full_load_limit = train_batch_size

    ic_df = df[df['type'] == 'IC'].copy()
    ic_df.loc[:, 'numeric_id'] = ic_df['container_id'].astype(int)
    ic_full = ic_df[ic_df['numeric_id'] <= full_load_limit]
    ic_remaining = ic_df[ic_df['numeric_id'] > full_load_limit]
    ic_full_avg = ic_full['container_processing_time'].mean()
    ic_remain_avg = ic_remaining['container_processing_time'].mean() if len(ic_remaining) > 0 else 0
    ic_avg_time = ic_full_avg + ic_remain_avg
    print(f"full_ic_avg_time: {ic_full_avg}, remaining_ic_avg_time: {ic_remain_avg}")

    oc_df = df[df['type'] == 'OC'].copy()
    oc_df.loc[:, 'numeric_id'] = oc_df['container_id'].str.replace('OC-', '', regex=False).astype(int)
    oc_full = oc_df[oc_df['numeric_id'] <= full_load_limit]
    oc_remaining = oc_df[oc_df['numeric_id'] > full_load_limit]
    oc_full_avg = oc_full['container_processing_time'].mean()
    oc_remain_avg = oc_remaining['container_processing_time'].mean() if len(ic_remaining) > 0 else 0
    oc_avg_time = oc_full_avg + oc_remain_avg
    print(f"full_oc_avg_time: {oc_full_avg}, remaining_oc_avg_time: {oc_remain_avg}")

    total_ic_avg_time = ic_avg_time + ic_delay_time
    total_oc_avg_time = oc_avg_time + oc_delay_time

    return ic_avg_time, oc_avg_time, total_ic_avg_time, total_oc_avg_time


def calculate_vehicle_energy(vehicle_excel_path):
    df = pd.read_excel(vehicle_excel_path, sheet_name="performance")
    average_row = df[df['Vehicle'] == 'Average']

    ic_energy = average_row['IC Energy Consumption (kWh)'].values[0]
    oc_energy = average_row['OC Energy Consumption (kWh)'].values[0]
    total_energy = average_row['Total Energy Consumption (kWh)'].values[0]

    return ic_energy, oc_energy, total_energy


def print_and_save_metrics(ic_time, oc_time, total_time, ic_energy, oc_energy, total_energy):
    print("*" * 100)
    print("Intermodal Terminal Performance Matrix")
    print(f"IC average processing time: {ic_time:.4f}")
    print(f"OC average processing time: {oc_time:.4f}")
    print(f"Average container processing time: {total_time:.4f}")
    print(f"IC average energy consumption: {ic_energy:.4f}")
    print(f"OC average energy consumption: {oc_energy:.4f}")
    print(f"Average container energy consumption: {total_energy:.4f}")
    print("*" * 100)

    single_run = [ic_time, oc_time, total_time, ic_energy, oc_energy, total_energy]
    return single_run