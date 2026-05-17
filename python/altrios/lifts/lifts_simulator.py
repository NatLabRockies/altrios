"""Top-level LIFTS terminal simulation entrypoint.

The actor/process logic (truck gate, hostlers, cranes, container handoff,
per-train orchestration) lives in dedicated sibling modules. This file only
configures the SimPy environment, kicks off ``process_train_arrival`` for each
scheduled train, and post-processes the resulting container/emission records.
"""
import random

import polars as pl
import simpy

from altrios.lifts import distances, utilities
from altrios.lifts.classes import Terminal
from altrios.lifts.emissions import emission_records
from altrios.lifts.train_flow import process_train_arrival

def run_simulation(
        train_consist_plan: pl.DataFrame,
        terminal: str,
        out_path=None):
    '''
    Run a multi-train LIFTS simulation for the given terminal.
    '''
    # Reset the module-level emissions buffer so repeat invocations are clean
    emission_records.clear()

    terminal_config = utilities.load_config(utilities.resources_root() / "config.yaml")
    terminal_layout = distances.get_layout(terminal_config)

    random.seed(42)

    train_timetable = utilities.build_train_timetable(train_consist_plan, terminal, as_dicts=True)
    truck_number = max([entry['truck_number'] for entry in train_timetable])
    chassis_count = max([entry['empty_cars'] + entry['full_cars'] for entry in train_timetable])
    env = simpy.Environment()

    terminal = Terminal(env, 
        config=terminal_config,
        layout=terminal_layout, 
        truck_capacity=truck_number, 
        chassis_count=chassis_count)

    print("\nTrain timetable:")
    for schedule in train_timetable:
        print(schedule)
        env.process(process_train_arrival(env, terminal, schedule))

    num_tracks = terminal.track_number
    num_cranes = num_tracks * terminal.cranes_per_track
    num_hostlers = terminal.hostler_number

    print("*" * 50)
    print(f"Tracks: {num_tracks}; Cranes: {num_cranes}; Hostlers: {num_hostlers}")
    print("*" * 50)

    # When a train_consist_plan is supplied, simulate the entire plan regardless
    # of the config's simulation length. Otherwise honor the configured horizon.
    if train_consist_plan is not None:
        env.run()
    else:
        env.run(until=terminal_config["simulation"]["length"])

    # Create DataFrame for container events
    container_data = (
        pl.from_dicts(
            [dict(event, **{'container_id': container_id}) for container_id, event in terminal.container_events.items()],
            infer_schema_length=None
        )
        .lazy()
        .sort("container_id")
        .select(pl.col("container_id"), pl.exclude("container_id"))
        .with_columns(
            pl.when(pl.col("truck_exit").is_not_null() & pl.col("train_arrival_expected").is_not_null())
                .then(pl.col("truck_exit") - pl.col("train_arrival_expected"))
                .when(pl.col("train_depart").is_not_null())
                .then(pl.col("crane_load") - pl.col("truck_arrival"))
                .otherwise(None)
                .alias("container_processing_time"),
            pl.col("container_id").str.extract(r"Train-(\d+)").cast(pl.Int64).alias("train_id"),
            pl.col("container_id").str.starts_with("IC").alias("is_ic")
        )
    )

    # OC train actual arrival time
    train_arrival_df = (container_data
        .filter(
            pl.col("is_ic"),
            pl.col("train_arrival_actual").is_not_null()
        )
        .group_by("train_id")
        .agg(pl.col("train_arrival_actual").mean())
    )
    # OC train expected arrival time
    train_arrival_expected_df = (container_data
        .filter(
            pl.col("is_ic"),
            pl.col("train_arrival_expected").is_not_null()
        )
        .group_by("train_id")
        .agg(pl.col("train_arrival_expected").mean())
    )
    container_data = (container_data
        .join(train_arrival_df, on="train_id", how="left")
        .join(train_arrival_expected_df, on="train_id", how="left")
        .rename({
            "train_arrival_actual_right": "train_arrival_actual_oc",
            "train_arrival_expected_right": "train_arrival_expected_oc"
        })
        .drop("is_ic", "train_id")
    ).collect()

    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        daily_throughput = 2 * terminal.train_batch_size * terminal.track_number
        container_data.write_excel(out_path / f"simulation_container_{daily_throughput}_track_{num_tracks}_crane_{num_cranes}_hostler_{num_hostlers}_results.xlsx")
        if emission_records:
            emission_records_df = pl.DataFrame(emission_records)
            utilities.save_emission_results(
                emission_records_df,
                out_path / f"emission_container_{daily_throughput}_track_{num_tracks}_crane_{num_cranes}_hostler_{num_hostlers}_results.xlsx",
                filetype="xlsx",
            )
    return container_data



if __name__ == "__main__":
    consist_plan = (pl.read_csv(utilities.package_root() / 'resources' / 'train_consist_plan.csv')
        .with_columns(pl.lit("Intermodal").alias("Train_Type"))
    )
    run_simulation(
        train_consist_plan=consist_plan,
        terminal = "Allouez",
        out_path = utilities.package_root() / 'demos' / 'lifts' / 'demos' / 'starter_demo' / 'results'
    )
