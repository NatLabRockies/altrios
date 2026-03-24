#!/usr/bin/env python3
"""
Add sensor metadata (lat/lng, country, region) to the parquet files
"""

import polars as pl
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_sensor_metadata(sensors_overview_file: str):
    """Load sensor metadata from sensors_overview.jsonl"""
    logger.info(f"Loading sensor metadata from {sensors_overview_file}...")

    with open(sensors_overview_file, "r") as f:
        data = json.load(f)

    sensors = data["sensors"]
    logger.info(f"Loaded {len(sensors)} sensors")

    # Convert to DataFrame
    sensor_metadata = []
    for sensor in sensors:
        sensor_metadata.append(
            {
                "sensorId": sensor["sensorId"],
                "name": sensor["name"],
                "latitude": sensor.get("lat"),
                "longitude": sensor.get("lng"),
                "country": sensor.get("country"),
                "region": sensor.get("region"),
                "timezone": sensor.get("timezone"),
                "railways": ",".join(sensor.get("railways", [])),
                "operatingRailways": ",".join(sensor.get("operatingRailways", [])),
                "isActive": sensor.get("isActive", False),
            }
        )

    df_metadata = pl.DataFrame(sensor_metadata)
    logger.info(f"Created metadata DataFrame with {len(df_metadata)} rows")

    return df_metadata


def update_parquet_with_metadata(
    parquet_file: str, metadata_df: pl.DataFrame, output_file: str
):
    """Update parquet file with sensor metadata"""
    logger.info(f"Loading {parquet_file}...")
    df = pl.read_parquet(parquet_file)

    logger.info(f"Original shape: {df.shape}")
    logger.info(f"Original columns: {df.columns}")

    # Remove old lat/long columns if they exist
    cols_to_drop = [col for col in ["latitude", "longitude"] if col in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop)
        logger.info(f"Dropped old columns: {cols_to_drop}")

    # Join with metadata
    logger.info("Joining with sensor metadata...")
    df_updated = df.join(
        metadata_df.select(
            ["sensorId", "latitude", "longitude", "country", "region", "timezone"]
        ),
        on="sensorId",
        how="left",
    )

    logger.info(f"Updated shape: {df_updated.shape}")

    # Check how many sensors got lat/long
    non_null_count = df_updated.filter(pl.col("latitude").is_not_null()).height
    logger.info(f"Rows with latitude/longitude: {non_null_count} / {len(df_updated)}")

    # Save updated file
    logger.info(f"Saving to {output_file}...")
    df_updated.write_parquet(output_file)

    logger.info("✓ Updated parquet file saved")

    return df_updated


def main():
    """Main function"""
    import sys

    # Determine file suffix from command line args
    if len(sys.argv) > 1:
        monthly_file = sys.argv[1]
        totals_file = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Default to border states files
        monthly_file = "/Users/hpanneer/GitHub/altrios-applied/applications/corridor/north_america_sensor_rollup_monthly.parquet"
        totals_file = "/Users/hpanneer/GitHub/altrios-applied/applications/corridor/north_america_sensor_rollup_totals.parquet"

    # Load sensor metadata
    metadata_df = load_sensor_metadata(
        "/Volumes/LaCie/RailState_data/sensors_overview.jsonl"
    )

    # Show US and Canadian sensors with lat/long
    logger.info("\nChecking sensors in metadata...")
    north_america = metadata_df.filter(
        pl.col("country").is_in(["United States", "Canada", "USA"])
        | pl.col("country").str.contains("(?i)(United States|Canada|USA)")
    )
    logger.info(f"Found {len(north_america)} North American sensors in metadata")

    # Update monthly parquet
    logger.info("\n" + "=" * 70)
    logger.info("Updating monthly parquet file...")
    logger.info("=" * 70)
    monthly_updated = update_parquet_with_metadata(
        monthly_file,
        metadata_df,
        monthly_file,
    )

    # Update totals parquet
    if totals_file:
        logger.info("\n" + "=" * 70)
        logger.info("Updating totals parquet file...")
        logger.info("=" * 70)
        totals_updated = update_parquet_with_metadata(
            totals_file,
            metadata_df,
            totals_file,
        )
    else:
        totals_updated = monthly_updated

    # Show sample with lat/long
    logger.info("\n" + "=" * 70)
    logger.info("Sample sensors with coordinates:")
    logger.info("=" * 70)
    sample = (
        monthly_updated.filter(pl.col("latitude").is_not_null())
        .select(["sensor", "direction", "latitude", "longitude", "region"])
        .unique()
        .head(10)
    )
    print(sample)

    logger.info("\n✓ All parquet files updated with sensor metadata!")


if __name__ == "__main__":
    main()
