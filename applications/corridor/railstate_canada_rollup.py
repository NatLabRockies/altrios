"""
RailState Canadian Sensor Rollup Analysis

This script processes train sighting data from RailState sensors in Canada
and creates a comprehensive rollup table with all requested metrics.

The output includes:
- Sensor identification and location
- Direction of travel
- Daily unique trains per railcar category (empty vs loaded)
- Daily car counts per category (empty vs loaded)
- Daily unique car IDs per category (empty vs loaded)
"""

import polars as pl
from pathlib import Path
from typing import Optional, List
import logging
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Canadian provinces/territories abbreviations
CANADIAN_PROVINCES = [
    "AB",
    "BC",
    "MB",
    "NB",
    "NL",
    "NS",
    "NT",
    "NU",
    "ON",
    "PE",
    "QC",
    "SK",
    "YT",
]


def is_canadian_sensor(sensor_name: str) -> bool:
    """
    Determine if a sensor is in Canada based on its name.

    Args:
        sensor_name: Sensor name/location string

    Returns:
        True if sensor appears to be in Canada
    """
    if not sensor_name:
        return False

    # Check if any Canadian province abbreviation appears at the end
    for province in CANADIAN_PROVINCES:
        if sensor_name.strip().endswith(f", {province}"):
            return True

    return False


def parse_jsonl_files(file_paths: List[str]) -> pl.LazyFrame:
    """
    Read multiple JSONL files using Polars lazy mode and combine them.

    Args:
        file_paths: List of paths to JSONL files

    Returns:
        LazyFrame with the combined parsed data
    """
    logger.info(f"Reading {len(file_paths)} JSONL files")

    # Read all files and concatenate with schema alignment
    dfs = []
    for file_path in file_paths:
        logger.info(f"  - {Path(file_path).name}")
        try:
            df = pl.scan_ndjson(file_path)
            # Cast problematic columns to String to ensure compatibility
            df = df.with_columns(
                [
                    pl.col("detectionEndUTC").cast(pl.String, strict=False),
                    pl.col("detectionEndSensorLocal").cast(pl.String, strict=False),
                ]
            )
            dfs.append(df)
        except Exception as e:
            logger.warning(f"  ! Error reading {file_path}: {e}")
            continue

    if not dfs:
        raise ValueError("No files could be read successfully")

    # Concatenate all dataframes with diagonal to align schemas
    logger.info("Combining all files with schema alignment...")
    combined_df = pl.concat(dfs, how="diagonal_relaxed")

    return combined_df


def explode_car_data(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Explode the nested 'cars' array to have one row per car.

    Args:
        df: LazyFrame with nested car data

    Returns:
        LazyFrame with exploded car data
    """
    logger.info("Exploding car data")

    # Explode the cars array
    df_exploded = df.explode("cars")

    # Extract car-level fields from the struct
    df_exploded = df_exploded.with_columns(
        [
            pl.col("cars").struct.field("type").alias("car_type"),
            pl.col("cars").struct.field("carId").alias("car_id"),
            pl.col("cars").struct.field("isLoaded").alias("is_loaded"),
            pl.col("cars").struct.field("owner").alias("car_owner"),
        ]
    )

    return df_exploded


def create_sensor_rollup(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Create sensor-level rollup table with daily metrics.

    Args:
        df: LazyFrame with exploded car data

    Returns:
        LazyFrame with sensor+direction+month rollup
    """
    logger.info("Creating sensor rollup")

    # Parse detection time and extract date components
    logger.info("  - Parsing timestamps...")
    df = df.with_columns(
        [
            pl.col("detectionTimeUTC")
            .str.strptime(pl.Datetime("us", "UTC"), "%Y-%m-%dT%H:%M:%S%.fZ")
            .alias("detection_datetime"),
        ]
    )

    df = df.with_columns(
        [
            pl.col("detection_datetime").dt.date().alias("detection_date"),
            pl.col("detection_datetime").dt.year().alias("year"),
            pl.col("detection_datetime").dt.month().alias("month"),
        ]
    )

    # Create loaded status category (handling nulls)
    logger.info("  - Categorizing loaded status...")
    df = df.with_columns(
        [
            pl.when(pl.col("is_loaded").is_null())
            .then(pl.lit("Unknown"))
            .when(pl.col("is_loaded"))
            .then(pl.lit("Loaded"))
            .otherwise(pl.lit("Empty"))
            .alias("loaded_status")
        ]
    )

    # Create category key combining car type and loaded status
    df = df.with_columns(
        [(pl.col("car_type") + "_" + pl.col("loaded_status")).alias("car_category")]
    )

    # First, create daily aggregations per sensor+direction+date+category
    logger.info("  - Creating daily aggregations...")
    daily_agg = df.group_by(
        [
            "sensorId",
            "sensor",
            "direction",
            "year",
            "month",
            "detection_date",
            "car_category",
        ]
    ).agg(
        [
            pl.col("trainId").n_unique().alias("unique_trains"),
            pl.col("car_id").count().alias("car_count"),
            pl.col("car_id").n_unique().alias("unique_car_ids"),
        ]
    )

    # Then aggregate to monthly level - calculate daily averages
    logger.info("  - Aggregating to monthly level...")
    monthly_rollup = daily_agg.group_by(
        ["sensorId", "sensor", "direction", "year", "month", "car_category"]
    ).agg(
        [
            pl.col("unique_trains").mean().alias("daily_avg_unique_trains"),
            pl.col("car_count").mean().alias("daily_avg_car_count"),
            pl.col("unique_car_ids").mean().alias("daily_avg_unique_car_ids"),
            pl.col("detection_date").n_unique().alias("days_active"),
        ]
    )

    # Also get overall statistics per sensor+direction+month (not split by category)
    logger.info("  - Computing overall statistics...")
    overall_stats = df.group_by(
        ["sensorId", "sensor", "direction", "year", "month", "detection_date"]
    ).agg(
        [
            pl.col("trainId").n_unique().alias("daily_unique_trains"),
            pl.col("speedMph").mean().alias("daily_avg_speed_mph"),
            pl.col("trainType").n_unique().alias("daily_unique_train_types"),
            pl.col("trainOperator").n_unique().alias("daily_unique_operators"),
        ]
    )

    overall_monthly = overall_stats.group_by(
        ["sensorId", "sensor", "direction", "year", "month"]
    ).agg(
        [
            pl.col("daily_unique_trains").mean().alias("daily_avg_total_trains"),
            pl.col("daily_avg_speed_mph").mean().alias("avg_speed_mph"),
            pl.col("daily_unique_train_types").mean().alias("avg_daily_train_types"),
            pl.col("daily_unique_operators").mean().alias("avg_daily_operators"),
            pl.col("detection_date").n_unique().alias("days_active"),
        ]
    )

    # Instead of pivot, keep category data in long format and join with overall stats
    # This is more memory efficient for large datasets
    logger.info("  - Combining category data with overall statistics...")

    # Join category metrics with overall stats
    final_rollup = monthly_rollup.join(
        overall_monthly,
        on=["sensorId", "sensor", "direction", "year", "month"],
        how="left",
    )

    # Sort by sensor, direction, year, month, category
    final_rollup = final_rollup.sort(
        ["sensorId", "direction", "year", "month", "car_category"]
    )

    return final_rollup


def get_sensor_locations(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Extract unique sensor information.

    Args:
        df: LazyFrame with sensor data

    Returns:
        LazyFrame with sensor metadata
    """
    logger.info("Extracting sensor metadata")

    # Get unique sensors
    sensor_meta = df.select(
        [
            "sensorId",
            "sensor",
        ]
    ).unique()

    # Add placeholder for coordinates (would come from external metadata)
    sensor_meta = sensor_meta.with_columns(
        [
            pl.lit(None).cast(pl.Float64).alias("latitude"),
            pl.lit(None).cast(pl.Float64).alias("longitude"),
        ]
    )

    return sensor_meta


def generate_statistics(df: pl.DataFrame) -> dict:
    """
    Generate summary statistics from the rollup data.

    Args:
        df: DataFrame with rollup data

    Returns:
        Dictionary containing summary statistics
    """
    # Get car category columns
    car_count_cols = [c for c in df.columns if "daily_avg_car_count_" in c]
    loaded_cols = [c for c in car_count_cols if "_Loaded" in c]
    empty_cols = [c for c in car_count_cols if "_Empty" in c]

    # Calculate loaded vs empty totals
    loaded_empty_stats = {}
    for col in loaded_cols:
        category = col.replace("daily_avg_car_count_", "").replace("_Loaded", "")
        total = float(df[col].fill_null(0).sum())
        loaded_empty_stats[f"{category}_Loaded"] = round(total, 2)

    for col in empty_cols:
        category = col.replace("daily_avg_car_count_", "").replace("_Empty", "")
        total = float(df[col].fill_null(0).sum())
        loaded_empty_stats[f"{category}_Empty"] = round(total, 2)

    # Direction statistics
    direction_stats = {}
    for direction in df["direction"].unique().to_list():
        dir_df = df.filter(pl.col("direction") == direction)
        direction_stats[direction] = {
            "count": int(dir_df.shape[0]),
            "avg_speed_mph": round(float(dir_df["avg_speed_mph"].mean()), 2),
            "avg_daily_trains": round(
                float(dir_df["daily_avg_total_trains"].mean()), 2
            ),
        }

    # All corridors sorted by daily average trains
    all_corridors = df.select(
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
    ).sort("daily_avg_total_trains", descending=True)

    # Calculate total trains (daily avg × days active)
    all_corridors = all_corridors.with_columns(
        [
            (pl.col("daily_avg_total_trains") * pl.col("days_active")).alias(
                "total_trains_in_period"
            )
        ]
    )

    corridors_list = []
    for row in all_corridors.iter_rows(named=True):
        corridors_list.append(
            {
                "sensorId": int(row["sensorId"]),
                "sensor": row["sensor"],
                "direction": row["direction"],
                "year": int(row["year"]),
                "month": int(row["month"]),
                "days_active": int(row["days_active"]),
                "daily_avg_trains": round(float(row["daily_avg_total_trains"]), 2),
                "total_trains_in_period": round(
                    float(row["total_trains_in_period"]), 2
                ),
                "avg_speed_mph": round(float(row["avg_speed_mph"]), 2),
                "avg_daily_train_types": round(float(row["avg_daily_train_types"]), 2),
                "avg_daily_operators": round(float(row["avg_daily_operators"]), 2),
            }
        )

    # Overall statistics
    stats = {
        "summary": {
            "total_rows": int(df.shape[0]),
            "total_columns": int(df.shape[1]),
            "unique_sensors": int(df["sensorId"].n_unique()),
            "unique_sensor_direction_combos": int(
                df.select(["sensorId", "direction"]).unique().shape[0]
            ),
            "avg_days_active": round(float(df["days_active"].mean()), 2),
            "overall_avg_daily_trains": round(
                float(df["daily_avg_total_trains"].mean()), 2
            ),
            "overall_avg_speed_mph": round(float(df["avg_speed_mph"].mean()), 2),
            "total_trains_all_sensors": round(
                float((df["daily_avg_total_trains"] * df["days_active"]).sum()), 2
            ),
        },
        "direction_statistics": direction_stats,
        "all_corridors_by_daily_avg_trains": corridors_list,
        "top_10_by_daily_avg_trains": corridors_list[:10],
        "top_10_by_total_trains": sorted(
            corridors_list, key=lambda x: x["total_trains_in_period"], reverse=True
        )[:10],
        "loaded_vs_empty_daily_avg": loaded_empty_stats,
        "car_categories": sorted(
            [c.replace("daily_avg_car_count_", "") for c in car_count_cols]
        ),
    }

    return stats


def main(
    input_files: List[str],
    output_file: Optional[str] = None,
    output_format: str = "parquet",
    canada_only: bool = True,
):
    """
    Main processing function.

    Args:
        input_files: List of paths to input JSONL files
        output_file: Path to output file (optional, will auto-generate if not provided)
        output_format: Output format ('parquet', 'csv', or 'json')
        canada_only: If True, filter to only Canadian sensors
    """
    logger.info("Starting Canadian sensor rollup analysis")

    # Read data from all files
    df = parse_jsonl_files(input_files)

    # Explode car data
    df_exploded = explode_car_data(df)

    # Create rollup
    rollup = create_sensor_rollup(df_exploded)

    # Get sensor locations
    sensor_meta = get_sensor_locations(df)

    # Join sensor metadata
    final_result = rollup.join(sensor_meta, on=["sensorId", "sensor"], how="left")

    # Reorder columns to put key identifiers first
    key_cols = [
        "sensorId",
        "sensor",
        "latitude",
        "longitude",
        "direction",
        "year",
        "month",
        "days_active",
        "daily_avg_total_trains",
        "avg_speed_mph",
        "avg_daily_train_types",
        "avg_daily_operators",
    ]

    # Get all other columns
    all_cols = final_result.collect_schema().names()
    other_cols = [col for col in all_cols if col not in key_cols]

    final_result = final_result.select(key_cols + other_cols)

    # Collect results with streaming engine for better memory efficiency
    logger.info("Collecting results (using streaming engine for large datasets)...")
    try:
        result_df = final_result.collect(engine="streaming")
    except Exception as e:
        logger.warning(f"Streaming engine failed, trying GPU fallback: {e}")
        try:
            result_df = final_result.collect(engine="gpu")
        except Exception as e2:
            logger.warning(f"GPU engine failed, using default collect: {e2}")
            result_df = final_result.collect()

    # Filter to Canadian sensors if requested
    if canada_only:
        logger.info("Filtering to Canadian sensors only")
        canadian_mask = result_df["sensor"].map_elements(
            is_canadian_sensor, return_dtype=pl.Boolean
        )
        result_df = result_df.filter(canadian_mask)
        logger.info(
            f"Found {len(result_df)} Canadian sensor+direction+month combinations"
        )

    # Generate output filename if not provided
    if output_file is None:
        output_file = str(
            Path(input_files[0]).parent / f"canada_sensor_rollup.{output_format}"
        )

    # Save based on format
    logger.info(f"Saving results to {output_file}")
    if output_format == "parquet":
        result_df.write_parquet(output_file)
    elif output_format == "csv":
        result_df.write_csv(output_file)
    elif output_format == "json":
        result_df.write_json(output_file)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    # Generate and save statistics
    logger.info("Generating statistics")
    stats = generate_statistics(result_df)

    # Save statistics to JSON
    stats_file = (
        Path(output_file)
        .with_suffix(".json")
        .with_stem(Path(output_file).stem + "_stats")
    )
    logger.info(f"Saving statistics to {stats_file}")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Analysis complete! Output saved to {output_file}")
    logger.info(f"Statistics saved to {stats_file}")
    logger.info(f"Total sensors: {result_df['sensorId'].n_unique()}")
    logger.info(f"Total sensor+direction+month combinations: {len(result_df)}")

    # Print summary
    logger.info("\nCanadian Sensors Summary:")
    canadian_sensors = result_df.select(["sensorId", "sensor"]).unique().sort("sensor")
    print(canadian_sensors)

    return result_df


if __name__ == "__main__":
    import sys

    # List of input files
    input_files = [
        "/Volumes/LaCie/RailState_data/tc_export_2024-09.jsonl",
        "/Volumes/LaCie/RailState_data/tc_export_2024-10.jsonl",
        "/Volumes/LaCie/RailState_data/tc_export_2024-11.jsonl",
        "/Volumes/LaCie/RailState_data/tc_export_2024-12.jsonl",
        "/Volumes/LaCie/RailState_data/tc_export_2025-01.jsonl",
        "/Volumes/LaCie/RailState_data/tc_export_2025-02.jsonl",
        "/Volumes/LaCie/RailState_data/tc_export_2025-03.jsonl",
        "/Volumes/LaCie/RailState_data/tc_export_2025-04.jsonl",
        "/Volumes/LaCie/RailState_data/tc_export_2025-05.jsonl",
        "/Volumes/LaCie/RailState_data/tc_export_2025-06.jsonl",
        "/Volumes/LaCie/RailState_data/tc_export_2025-07.jsonl",
        "/Volumes/LaCie/RailState_data/tc_export_2025-08.jsonl",
    ]

    # Allow command line override
    if len(sys.argv) > 1:
        input_files = sys.argv[1:]

    output_format = "parquet"
    output_file = None

    # Run analysis
    result = main(input_files, output_file, output_format, canada_only=True)
