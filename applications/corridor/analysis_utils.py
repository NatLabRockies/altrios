"""
Utility functions for analyzing sensor rollup data

This module provides helper functions to extract specific insights from the
sensor rollup output.
"""

import polars as pl
from typing import Optional


def load_rollup(file_path: str) -> pl.DataFrame:
    """Load the sensor rollup data."""
    return pl.read_parquet(file_path)


def get_sensor_summary(
    df: pl.DataFrame, sensor_id: Optional[int] = None
) -> pl.DataFrame:
    """
    Get summary statistics for a specific sensor or all sensors.

    Args:
        df: Rollup DataFrame
        sensor_id: Optional sensor ID to filter by

    Returns:
        DataFrame with sensor summary
    """
    if sensor_id:
        df = df.filter(pl.col("sensorId") == sensor_id)

    return df.select(
        [
            "sensorId",
            "sensor",
            "direction",
            "year",
            "month",
            "days_active",
            "daily_avg_total_trains",
            "avg_speed_mph",
            "avg_daily_train_types",
            "avg_daily_operators",
        ]
    ).sort(["sensorId", "direction", "year", "month"])


def get_loaded_empty_summary(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create summary of loaded vs empty cars by category.

    Args:
        df: Rollup DataFrame

    Returns:
        DataFrame summarizing loaded vs empty statistics
    """
    loaded_cols = [
        c for c in df.columns if "_Loaded" in c and "daily_avg_car_count_" in c
    ]
    empty_cols = [
        c for c in df.columns if "_Empty" in c and "daily_avg_car_count_" in c
    ]

    # Sum across all sensors for each category
    results = []

    for col in loaded_cols:
        category = col.replace("daily_avg_car_count_", "").replace("_Loaded", "")
        total = df[col].fill_null(0).sum()
        results.append(
            {"category": category, "status": "Loaded", "total_daily_avg": total}
        )

    for col in empty_cols:
        category = col.replace("daily_avg_car_count_", "").replace("_Empty", "")
        total = df[col].fill_null(0).sum()
        results.append(
            {"category": category, "status": "Empty", "total_daily_avg": total}
        )

    return pl.DataFrame(results).sort(["category", "status"])


def get_car_type_breakdown(
    df: pl.DataFrame, sensor_id: Optional[int] = None
) -> pl.DataFrame:
    """
    Get car type breakdown with loaded/empty/unknown status.

    Args:
        df: Rollup DataFrame
        sensor_id: Optional sensor ID to filter by

    Returns:
        DataFrame with car type breakdown
    """
    if sensor_id:
        df = df.filter(pl.col("sensorId") == sensor_id)

    # Get all car count columns
    car_count_cols = [c for c in df.columns if "daily_avg_car_count_" in c]

    # Calculate totals for each category
    results = []
    for col in car_count_cols:
        category = col.replace("daily_avg_car_count_", "")
        total = df[col].fill_null(0).sum()
        results.append({"category": category, "total_daily_avg_cars": total})

    result_df = pl.DataFrame(results)

    # Split category into car_type and status
    result_df = result_df.with_columns(
        [
            pl.col("category").str.split("_").list.get(-1).alias("status"),
            pl.col("category")
            .str.replace(r"_(?:Loaded|Empty|Unknown)$", "")
            .alias("car_type"),
        ]
    )

    return result_df.select(["car_type", "status", "total_daily_avg_cars"]).sort(
        ["car_type", "status"], descending=[False, False]
    )


def get_busiest_corridors(df: pl.DataFrame, top_n: int = 10) -> pl.DataFrame:
    """
    Identify busiest corridors by daily train count.

    Args:
        df: Rollup DataFrame
        top_n: Number of top corridors to return

    Returns:
        DataFrame with busiest corridors
    """
    return (
        df.select(
            [
                "sensorId",
                "sensor",
                "direction",
                "year",
                "month",
                "daily_avg_total_trains",
                "avg_speed_mph",
                "days_active",
            ]
        )
        .sort("daily_avg_total_trains", descending=True)
        .head(top_n)
    )


def get_speed_statistics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate speed statistics across all sensors.

    Args:
        df: Rollup DataFrame

    Returns:
        DataFrame with speed statistics by direction
    """
    return (
        df.group_by("direction")
        .agg(
            [
                pl.col("avg_speed_mph").mean().alias("mean_speed_mph"),
                pl.col("avg_speed_mph").median().alias("median_speed_mph"),
                pl.col("avg_speed_mph").min().alias("min_speed_mph"),
                pl.col("avg_speed_mph").max().alias("max_speed_mph"),
                pl.col("avg_speed_mph").std().alias("std_speed_mph"),
                pl.count().alias("n_sensor_direction_combos"),
            ]
        )
        .sort("mean_speed_mph", descending=True)
    )


def export_sensor_csv(df: pl.DataFrame, sensor_id: int, output_file: str):
    """
    Export data for a specific sensor to CSV.

    Args:
        df: Rollup DataFrame
        sensor_id: Sensor ID to export
        output_file: Output CSV file path
    """
    sensor_data = df.filter(pl.col("sensorId") == sensor_id)
    sensor_data.write_csv(output_file)
    print(f"Exported {len(sensor_data)} rows for sensor {sensor_id} to {output_file}")


def get_monthly_trends(
    df: pl.DataFrame, sensor_id: Optional[int] = None
) -> pl.DataFrame:
    """
    Get monthly trends for sensor(s).

    Args:
        df: Rollup DataFrame
        sensor_id: Optional sensor ID to filter by

    Returns:
        DataFrame with monthly trends
    """
    if sensor_id:
        df = df.filter(pl.col("sensorId") == sensor_id)

    # For the single month in this dataset, just return the summary
    # In multi-month datasets, this would show trends over time
    return (
        df.group_by(["year", "month"])
        .agg(
            [
                pl.col("daily_avg_total_trains").mean().alias("avg_trains_per_sensor"),
                pl.col("avg_speed_mph").mean().alias("avg_speed"),
                pl.col("sensorId").n_unique().alias("n_sensors"),
                pl.col("days_active").mean().alias("avg_days_active"),
            ]
        )
        .sort(["year", "month"])
    )


def filter_by_car_type(df: pl.DataFrame, car_type: str) -> pl.DataFrame:
    """
    Filter to sensors that have specific car types.

    Args:
        df: Rollup DataFrame
        car_type: Car type to filter by (e.g., "Tank Car", "Stack Car")

    Returns:
        DataFrame filtered to sensors with that car type
    """
    # Find all columns matching this car type
    matching_cols = [
        c for c in df.columns if car_type in c and "daily_avg_car_count_" in c
    ]

    if not matching_cols:
        print(f"No columns found for car type: {car_type}")
        return df.filter(pl.lit(False))

    # Filter to rows where at least one matching column has non-null values
    condition = pl.lit(False)
    for col in matching_cols:
        condition = condition | pl.col(col).is_not_null()

    return df.filter(condition).select(
        [
            "sensorId",
            "sensor",
            "direction",
            "year",
            "month",
            "daily_avg_total_trains",
            "avg_speed_mph",
        ]
        + matching_cols
    )


if __name__ == "__main__":
    # Example usage
    print("Loading rollup data...")
    df = load_rollup(
        "/Users/hpanneer/Downloads/sensor_rollup_tc_export_2024-09.parquet"
    )

    print("\n" + "=" * 80)
    print("BUSIEST CORRIDORS")
    print("=" * 80)
    print(get_busiest_corridors(df, top_n=5))

    print("\n" + "=" * 80)
    print("SPEED STATISTICS BY DIRECTION")
    print("=" * 80)
    print(get_speed_statistics(df))

    print("\n" + "=" * 80)
    print("LOADED VS EMPTY SUMMARY")
    print("=" * 80)
    print(get_loaded_empty_summary(df).head(10))

    print("\n" + "=" * 80)
    print("CAR TYPE BREAKDOWN")
    print("=" * 80)
    print(get_car_type_breakdown(df).head(15))

    print("\n" + "=" * 80)
    print("MONTHLY TRENDS")
    print("=" * 80)
    print(get_monthly_trends(df))
