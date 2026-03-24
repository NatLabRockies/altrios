#!/usr/bin/env python3
"""
RailState North American Sensor Rollup - Incremental Processing with ACTUAL TOTALS
This version processes all US and Canadian sensors.
Calculates both daily averages AND actual total counts from raw data.
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


# Removed is_canadian_sensor filter - now processing all US and Canadian sensors


def process_single_file(filepath: str) -> tuple:
    """
    Process a single JSONL file and return both aggregated AND raw counts.

    Returns:
        Tuple of (monthly_rollup_df, total_counts_df)
    """
    logger.info(f"Processing {Path(filepath).name}...")

    # Read the JSONL file
    df = pl.scan_ndjson(filepath)

    # Cast problematic columns
    df = df.with_columns(
        [
            pl.col("detectionEndUTC").cast(pl.String, strict=False),
            pl.col("detectionEndSensorLocal").cast(pl.String, strict=False),
        ]
    )

    # No filtering - processing all US and Canadian sensors
    logger.info("  - Processing all US and Canadian sensors...")

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

    # === PART 1: Daily aggregations (for averages) ===
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

    # Monthly rollup (averages)
    logger.info("  - Aggregating to monthly level (averages)...")
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

    # === PART 2: Overall statistics (not split by category) ===
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

    # === PART 3: ACTUAL TOTALS across entire period (per sensor+direction+category) ===
    logger.info("  - Computing ACTUAL TOTALS from raw data...")
    total_counts = df.group_by(["sensorId", "sensor", "direction", "car_category"]).agg(
        [
            # Actual total counts
            pl.col("trainId").n_unique().alias("total_unique_trains"),
            pl.col("car_id").count().alias("total_car_count"),
            pl.col("car_id").n_unique().alias("total_unique_car_ids"),
            # Metadata
            pl.col("detection_date").n_unique().alias("total_days_with_sightings"),
            pl.col("year").n_unique().alias("years_covered"),
            pl.col("month").n_unique().alias("months_covered"),
        ]
    )

    # Combine monthly rollup with overall stats
    final_rollup = monthly_rollup.join(
        overall_monthly,
        on=["sensorId", "sensor", "direction", "year", "month"],
        how="left",
    )

    # Get sensor metadata
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

    # Join metadata to both dataframes
    final_rollup = final_rollup.join(
        sensor_metadata, on=["sensorId", "sensor"], how="left"
    )

    total_counts = total_counts.join(
        sensor_metadata, on=["sensorId", "sensor"], how="left"
    )

    # Collect results
    logger.info("  - Collecting results...")
    monthly_df = final_rollup.collect()
    totals_df = total_counts.collect()

    logger.info(f"  - Monthly rollup: {len(monthly_df)} rows")
    logger.info(f"  - Total counts: {len(totals_df)} rows")

    return monthly_df, totals_df


def process_all_files(input_files: List[str], output_dir: str):
    """Process all files and save both monthly and total counts."""
    logger.info(f"Starting incremental processing of {len(input_files)} files")

    all_monthly = []
    all_totals = []

    for filepath in input_files:
        try:
            monthly_df, totals_df = process_single_file(filepath)
            all_monthly.append(monthly_df)
            all_totals.append(totals_df)
            logger.info(f"✓ Completed {Path(filepath).name}")
        except Exception as e:
            logger.error(f"✗ Failed to process {Path(filepath).name}: {e}")
            continue

    if not all_monthly:
        logger.error("No files were successfully processed!")
        return None

    # Combine monthly rollups
    logger.info(f"Combining monthly rollups from {len(all_monthly)} files...")
    monthly_combined = pl.concat(all_monthly, how="diagonal_relaxed")

    # Combine and aggregate totals (sum across months)
    logger.info(f"Combining and aggregating totals...")
    totals_combined = pl.concat(all_totals, how="diagonal_relaxed")

    # Aggregate totals to get final counts per sensor+direction+category
    final_totals = totals_combined.group_by(
        ["sensorId", "sensor", "latitude", "longitude", "direction", "car_category"]
    ).agg(
        [
            pl.col("total_unique_trains").sum(),
            pl.col("total_car_count").sum(),
            pl.col("total_unique_car_ids").max(),  # Use max since unique IDs don't sum
            pl.col("total_days_with_sightings").sum(),
            pl.col("years_covered").max(),
            pl.col("months_covered").sum(),
        ]
    )

    # Save monthly rollup
    monthly_file = Path(output_dir) / "north_america_sensor_rollup_monthly.parquet"
    logger.info(f"Saving monthly rollup to {monthly_file}")
    monthly_combined.write_parquet(monthly_file)

    # Save totals
    totals_file = Path(output_dir) / "north_america_sensor_rollup_totals.parquet"
    logger.info(f"Saving actual totals to {totals_file}")
    final_totals.write_parquet(totals_file)

    logger.info(f"\n{'=' * 70}")
    logger.info("Processing complete!")
    logger.info(f"{'=' * 70}")
    logger.info(f"Monthly rollup: {len(monthly_combined):,} rows")
    logger.info(f"Total counts: {len(final_totals):,} rows")
    logger.info(f"Files saved in: {output_dir}")
    logger.info(f"{'=' * 70}\n")

    return monthly_combined, final_totals


if __name__ == "__main__":
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

    output_dir = "/Users/hpanneer/GitHub/altrios-applied/applications/corridor"

    process_all_files(input_files, output_dir)
