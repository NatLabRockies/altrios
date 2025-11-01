import pandas as pd
import altrios.lifts.utilities
from demo_parameters import *
from pathlib import Path

package_root = altrios.lifts.utilities.package_root()
out_path = package_root / 'demos' / 'double_track_results'
out_path.mkdir(parents=True, exist_ok=True)

# Dictionary to store vehicle events
vehicle_events = {
    'crane': [],
    'hostler': [],
    'truck': []
}

def record_vehicle_event(vehicle_category, vehicle, action, state, time, emission, event_type, timestamp):
    vehicle_events[vehicle_category].append({
        'vehicle_id': str(vehicle),
        'action': action,
        'state': state,
        'time': time,
        'emission': emission,
        'event_type': event_type,
        'timestamp': timestamp
    })

def calculate_performance(dataframes, ic_count, oc_count):
    summary = {vehicle: {'IC Emissions': 0, 'OC Emissions': 0, 'Total Emissions': 0,
                         'IC Time': 0, 'OC Time': 0, 'Total Time': 0} for vehicle in dataframes.keys()}

    for vehicle, df in dataframes.items():
        for _, row in df.iterrows():
            emission = row['emission']
            time = row['time']
            action = row['action']

            if 'OC' in action:
                summary[vehicle]['OC Emissions'] += emission
                summary[vehicle]['OC Time'] += time
            else:
                summary[vehicle]['IC Emissions'] += emission
                summary[vehicle]['IC Time'] += time

            summary[vehicle]['Total Emissions'] = summary[vehicle]['IC Emissions'] + summary[vehicle]['OC Emissions']
            summary[vehicle]['Total Time'] = summary[vehicle]['IC Time'] + summary[vehicle]['OC Time']

    summary_df = pd.DataFrame.from_dict(summary, orient='index')
    summary_df.reset_index(inplace=True)
    summary_df.rename(columns={'index': 'Vehicle'}, inplace=True)

    total_row = summary_df.sum(numeric_only=True).to_dict()
    total_row['Vehicle'] = 'Total'
    summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)

    # Append the average row directly below total row
    average_row = {
        'Vehicle': 'Average',
        'IC Emissions': total_row['IC Emissions'] / ic_count if ic_count else 0,
        'OC Emissions': total_row['OC Emissions'] / oc_count if oc_count else 0,
        'Total Emissions': total_row['Total Emissions'] / (ic_count + oc_count) if (ic_count + oc_count) else 0,
        'IC Time': total_row['IC Time'] / ic_count if ic_count else 0,
        'OC Time': total_row['OC Time'] / oc_count if oc_count else 0,
        'Total Time': total_row['Total Time'] / (ic_count + oc_count) if (ic_count + oc_count) else 0
    }

    summary_df = pd.concat([summary_df, pd.DataFrame([average_row])], ignore_index=True)
    return summary_df

def save_energy_to_excel(state):
    df_logs = {}
    for vehicle_type, events in vehicle_events.items():
        df_logs[vehicle_type] = pd.DataFrame(events)

    ic_count = state.IC_NUM  # 获取 IC 数量
    oc_count = state.OC_NUM  # 获取 OC 数量

    # Calculate summary statistics
    performance_df = calculate_performance(df_logs, ic_count, oc_count)
    performance_df.rename(columns={'IC Emissions': 'IC Energy Consumption',
                               'OC Emissions': 'OC Energy Consumption',
                               'Total Emissions': 'Total Energy Consumption',
                               'IC Time': 'IC Processing Time',
                               'OC Time': 'OC Processing Time',
                               'Total Time': 'Total Processing Time'}, inplace=True)

    file_name = f"double_track_vehicle_log_crane_{state.CRANE_NUMBER}_hostler_{state.HOSTLER_NUMBER}.xlsx"
    file_path = out_path / file_name

    with pd.ExcelWriter(file_path) as writer:
        for vehicle_type, df in df_logs.items():
            df.to_excel(writer, sheet_name=vehicle_type, index=False)
        performance_df.to_excel(writer, sheet_name='performance', index=True)


import pandas as pd

def calculate_container_processing_time(container_excel_path):
    df = pd.read_excel(container_excel_path)

    # 根据 container_id 判断 IC 或 OC
    df['type'] = df['container_id'].apply(lambda x: 'IC' if str(x).isdigit() else 'OC' if str(x).startswith('OC-') else 'Unknown')

    # 处理时间列名为 container_processing_time
    ic_df = df[df['type'] == 'IC']
    oc_df = df[df['type'] == 'OC']

    ic_avg_time = ic_df['container_processing_time'].mean() if not ic_df.empty else 0
    oc_avg_time = oc_df['container_processing_time'].mean() if not oc_df.empty else 0
    total_avg_time = df['container_processing_time'].mean() if not df.empty else 0

    return ic_avg_time, oc_avg_time, total_avg_time


def calculate_vehicle_energy(vehicle_excel_path):
    df = pd.read_excel(vehicle_excel_path, sheet_name="performance")
    average_row = df[df['Vehicle'] == 'Average']

    ic_energy = average_row['IC Energy Consumption'].values[0]
    oc_energy = average_row['OC Energy Consumption'].values[0]
    total_energy = average_row['Total Energy Consumption'].values[0]

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
