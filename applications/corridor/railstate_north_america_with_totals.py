#!/usr/bin/env python3
"""
RailState North American Sensor Rollup - Incremental Processing with ACTUAL TOTALS
This version processes Canadian provinces and US states (border states by default, all states with --all-states).
Calculates both daily averages AND actual total counts from raw data.

Usage:
    python railstate_north_america_with_totals.py                    # Border states only (default)
    python railstate_north_america_with_totals.py --all-states       # All US states + Canada
    python railstate_north_america_with_totals.py --border-only      # Border states only (explicit)
"""

import polars as pl
import logging
from pathlib import Path
import json
from typing import List
import sys
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_north_american_regions(include_all_us_states: bool = False) -> set:
    """
    Get the set of state/province codes to include.

    Args:
        include_all_us_states: If True, include all US states. If False, only border states.

    Returns:
        Set of state/province abbreviations
    """
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

    if include_all_us_states:
        # All 50 US states
        us_states = {
            "AL",
            "AK",
            "AZ",
            "AR",
            "CA",
            "CO",
            "CT",
            "DE",
            "FL",
            "GA",
            "HI",
            "ID",
            "IL",
            "IN",
            "IA",
            "KS",
            "KY",
            "LA",
            "ME",
            "MD",
            "MA",
            "MI",
            "MN",
            "MS",
            "MO",
            "MT",
            "NE",
            "NV",
            "NH",
            "NJ",
            "NM",
            "NY",
            "NC",
            "ND",
            "OH",
            "OK",
            "OR",
            "PA",
            "RI",
            "SC",
            "SD",
            "TN",
            "TX",
            "UT",
            "VT",
            "VA",
            "WA",
            "WV",
            "WI",
            "WY",
        }
    else:
        # US border states only (northern and southern borders with rail crossings)
        us_states = {
            # Northern border (Canada)
            "WA",
            "ID",
            "MT",
            "ND",
            "MN",
            "WI",
            "MI",
            "OH",
            "PA",
            "NY",
            "VT",
            "NH",
            "ME",
            # Southern border (Mexico) - major rail crossings
            "CA",
            "AZ",
            "NM",
            "TX",
        }

    return canadian_provinces | us_states


def is_north_american_region(
    sensor_name: str, include_all_us_states: bool = False
) -> bool:
    """
    Check if a sensor is in Canada or US states (border states or all states).

    Args:
        sensor_name: Sensor name string (e.g., "Haney, BC Eastbound")
        include_all_us_states: If True, include all US states. If False, only border states.

    Returns:
        True if sensor is in the specified regions
    """
    if not sensor_name or not isinstance(sensor_name, str):
        return False

    north_american_regions = get_north_american_regions(include_all_us_states)

    # Extract state/province from sensor name (format: "City, ST Direction")
    parts = sensor_name.split(",")
    if len(parts) < 2:
        return False

    state_part = parts[1].strip().split()[0]  # Get first word after comma
    return state_part in north_american_regions


def process_single_file(filepath: str, include_all_us_states: bool = False) -> tuple:
    """
    Process a single JSONL file and return both aggregated AND raw counts.

    Returns:
        Tuple of (monthly_rollup_df, total_counts_df)
    """
    logger.info(f"Processing {Path(filepath).name}...")

    # Read the JSONL file with error handling for malformed records
    try:
        df = pl.scan_ndjson(filepath, ignore_errors=True)
    except Exception as e:
        logger.warning(f"  - Issue with scan_ndjson, trying read_ndjson: {e}")
        df = pl.read_ndjson(filepath, ignore_errors=True).lazy()

    # Cast problematic columns
    df = df.with_columns(
        [
            pl.col("detectionEndUTC").cast(pl.String, strict=False),
            pl.col("detectionEndSensorLocal").cast(pl.String, strict=False),
        ]
    )

    # Filter to Canadian provinces and US states
    if include_all_us_states:
        logger.info("  - Filtering to Canada and ALL US states...")
    else:
        logger.info("  - Filtering to Canada and US border states...")

    df = df.filter(
        pl.col("sensor").map_elements(
            lambda x: is_north_american_region(x, include_all_us_states),
            return_dtype=pl.Boolean,
        )
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


def process_all_files(
    input_files: List[str], output_dir: str, include_all_us_states: bool = False
):
    """Process all files and save both monthly and total counts."""
    logger.info(f"Starting incremental processing of {len(input_files)} files")

    all_monthly = []
    all_totals = []

    for filepath in input_files:
        try:
            monthly_df, totals_df = process_single_file(filepath, include_all_us_states)
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
    logger.info("Combining and aggregating totals...")
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
    suffix = "all_states" if include_all_us_states else "border_states"
    monthly_file = (
        Path(output_dir) / f"north_america_sensor_rollup_monthly_{suffix}.parquet"
    )
    logger.info(f"Saving monthly rollup to {monthly_file}")
    monthly_combined.write_parquet(monthly_file)

    # Save totals
    totals_file = (
        Path(output_dir) / f"north_america_sensor_rollup_totals_{suffix}.parquet"
    )
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process RailState data for North American sensors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process border states only (default):
  python railstate_north_america_with_totals.py
  
  # Process all US states + Canada:
  python railstate_north_america_with_totals.py --all-states
  
  # Explicitly specify border states only:
  python railstate_north_america_with_totals.py --border-only
""",
    )
    parser.add_argument(
        "--all-states",
        action="store_true",
        help="Include all 50 US states (not just border states) plus Canada",
    )
    parser.add_argument(
        "--border-only",
        action="store_true",
        help="Include only border states (default behavior)",
    )

    args = parser.parse_args()

    # Determine which mode to use
    include_all_us_states = args.all_states

    if args.all_states and args.border_only:
        logger.error("Cannot specify both --all-states and --border-only")
        sys.exit(1)

    mode_description = (
        "ALL US states + Canada" if include_all_us_states else "Border states + Canada"
    )
    logger.info(f"Running in mode: {mode_description}")
    logger.info("=" * 70)
    logger.info("=" * 70)

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

    process_all_files(input_files, output_dir, include_all_us_states)
