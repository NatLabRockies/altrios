#!/usr/bin/env python3
"""
RailState Canadian Sensor Rollup - Incremental Processing
Process multiple months of RailState data incrementally, one file at a time.
Filters for Canadian sensors only.
"""

import polars as pl
import logging
from pathlib import Path
import json
from typing import List
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def is_canadian_sensor(sensor_name: str) -> bool:
    """
    Check if a sensor is in Canada based on state/province abbreviation.

    Args:
        sensor_name: Sensor name string (e.g., "Haney, TX Eastbound")

    Returns:
        True if sensor is in a Canadian province/territory
    """
    if not sensor_name or not isinstance(sensor_name, str):
        return False

    # Canadian provinces and territories
    canadian_provinces = {
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
    }

    # Extract state/province from sensor name (format: "City, ST Direction")
    parts = sensor_name.split(",")
    if len(parts) < 2:
        return False

    state_part = parts[1].strip().split()[0]  # Get first word after comma
    return state_part in canadian_provinces


def process_single_file(filepath: str) -> pl.LazyFrame:
    """
    Process a single JSONL file and return Canadian sensors rollup.

    Args:
        filepath: Path to JSONL file

    Returns:
        LazyFrame with sensor rollup for this file
    """
    logger.info(f"Processing {Path(filepath).name}...")

    # Read the JSONL file
    df = pl.scan_ndjson(filepath)

    # Cast problematic columns to string to handle nulls
    df = df.with_columns(
        [
            pl.col("detectionEndUTC").cast(pl.String, strict=False),
            pl.col("detectionEndSensorLocal").cast(pl.String, strict=False),
        ]
    )

    # Filter to Canadian sensors early to reduce data volume
    logger.info("  - Filtering to Canadian sensors...")
    df = df.filter(
        pl.col("sensor").map_elements(is_canadian_sensor, return_dtype=pl.Boolean)
    )

    # Explode car data
    logger.info("  - Exploding car data...")
    df = df.explode("cars")

    # Extract car fields
    df = df.with_columns(
        [
            pl.col("cars").struct.field("carId").alias("car_id"),
            pl.col("cars").struct.field("type").alias("car_type"),
            pl.col("cars").struct.field("isLoaded").alias("is_loaded"),
        ]
    )

    # Parse timestamps
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

    # Create loaded status category
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

    # Create category key
    df = df.with_columns(
        [(pl.col("car_type") + "_" + pl.col("loaded_status")).alias("car_category")]
    )

    # Daily aggregations per sensor+direction+date+category
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

    # Monthly rollup
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

    # Overall statistics
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

    # Combine category data with overall stats
    final_rollup = monthly_rollup.join(
        overall_monthly,
        on=["sensorId", "sensor", "direction", "year", "month"],
        how="left",
    )

    # Get sensor metadata (handle missing lat/long)
    logger.info("  - Extracting sensor metadata...")
    schema_cols = df.collect_schema().names()

    if "latitude" in schema_cols and "longitude" in schema_cols:
        sensor_metadata = df.group_by(["sensorId", "sensor"]).agg(
            [
                pl.col("latitude").mean().alias("latitude"),
                pl.col("longitude").mean().alias("longitude"),
            ]
        )
    else:
        # If lat/long don't exist, create nulls
        sensor_metadata = (
            df.select(["sensorId", "sensor"])
            .unique()
            .with_columns(
                [
                    pl.lit(None).cast(pl.Float64).alias("latitude"),
                    pl.lit(None).cast(pl.Float64).alias("longitude"),
                ]
            )
        )

    # Join with metadata
    final_result = final_rollup.join(
        sensor_metadata, on=["sensorId", "sensor"], how="left"
    )

    # Collect this file's results
    logger.info("  - Collecting results...")
    result_df = final_result.collect()

    logger.info(f"  - Found {len(result_df)} sensor+direction+month+category rows")

    return result_df


def process_all_files(input_files: List[str], output_file: str):
    """
    Process all files incrementally and combine results.

    Args:
        input_files: List of JSONL file paths
        output_file: Output parquet file path
    """
    logger.info(f"Starting incremental processing of {len(input_files)} files")

    all_results = []

    for filepath in input_files:
        try:
            result_df = process_single_file(filepath)
            all_results.append(result_df)
            logger.info(f"✓ Completed {Path(filepath).name}")
        except Exception as e:
            logger.error(f"✗ Failed to process {Path(filepath).name}: {e}")
            continue

    if not all_results:
        logger.error("No files were successfully processed!")
        return None

    # Combine all results
    logger.info(f"Combining results from {len(all_results)} files...")
    combined_df = pl.concat(all_results, how="diagonal_relaxed")

    # Reorder columns
    key_cols = [
        "sensorId",
        "sensor",
        "latitude",
        "longitude",
        "direction",
        "year",
        "month",
        "car_category",
        "days_active",
        "daily_avg_unique_trains",
        "daily_avg_car_count",
        "daily_avg_unique_car_ids",
        "daily_avg_total_trains",
        "avg_speed_mph",
        "avg_daily_train_types",
        "avg_daily_operators",
    ]

    all_cols = combined_df.columns
    other_cols = [col for col in all_cols if col not in key_cols]
    combined_df = combined_df.select(key_cols + other_cols)

    # Sort
    combined_df = combined_df.sort(
        ["sensorId", "direction", "year", "month", "car_category"]
    )

    # Save to parquet
    logger.info(f"Saving combined results to {output_file}")
    combined_df.write_parquet(output_file)

    # Generate statistics
    logger.info("Generating statistics...")
    stats = {
        "total_rows": len(combined_df),
        "unique_sensors": combined_df["sensorId"].n_unique(),
        "unique_corridors": combined_df.select(
            [
                pl.concat_str(
                    [pl.col("sensor"), pl.lit(" "), pl.col("direction")]
                ).alias("corridor")
            ]
        )["corridor"].n_unique(),
        "date_range": {
            "start_year": int(combined_df["year"].min()),
            "start_month": int(combined_df["month"].min()),
            "end_year": int(combined_df["year"].max()),
            "end_month": int(combined_df["month"].max()),
        },
        "car_categories": sorted(combined_df["car_category"].unique().to_list()),
    }

    # Top sensors by traffic
    top_sensors = (
        combined_df.group_by(["sensor", "direction"])
        .agg(
            [
                pl.col("daily_avg_total_trains").mean().alias("avg_daily_trains"),
            ]
        )
        .sort("avg_daily_trains", descending=True)
        .head(20)
    )

    stats["top_20_corridors"] = [
        {
            "corridor": f"{row['sensor']} {row['direction']}",
            "avg_daily_trains": float(row["avg_daily_trains"]),
        }
        for row in top_sensors.iter_rows(named=True)
    ]

    # Save statistics
    stats_file = (
        Path(output_file)
        .with_suffix(".json")
        .with_stem(Path(output_file).stem + "_stats")
    )
    logger.info(f"Saving statistics to {stats_file}")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Processing complete!")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total rows: {stats['total_rows']:,}")
    logger.info(f"Unique sensors: {stats['unique_sensors']}")
    logger.info(f"Unique corridors: {stats['unique_corridors']}")
    logger.info(f"Car categories: {len(stats['car_categories'])}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Stats: {stats_file}")
    logger.info(f"{'=' * 60}\n")

    return combined_df


if __name__ == "__main__":
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

    output_file = "/Users/hpanneer/GitHub/altrios-applied/applications/corridor/canada_sensor_rollup.parquet"

    # Run incremental processing
    result = process_all_files(input_files, output_file)

    if result is not None:
        logger.info("Success!")
    else:
        logger.error("Processing failed!")
        sys.exit(1)
