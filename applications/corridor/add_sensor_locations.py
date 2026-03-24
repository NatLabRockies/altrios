#!/usr/bin/env python3
"""
Add latitude/longitude to Canadian sensor rollup data from external metadata file.

Usage:
    python add_sensor_locations.py <metadata_file> [--output output.parquet]

Metadata file should be CSV/Parquet with columns:
    - sensorId (or sensor_id, id)
    - latitude (or lat)
    - longitude (or long, lon, lng)
"""

import polars as pl
import argparse
import sys
from pathlib import Path


def standardize_metadata_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Standardize column names in metadata file.

    Args:
        df: Metadata DataFrame

    Returns:
        DataFrame with standardized column names
    """
    col_mapping = {}

    # Find sensor ID column
    for col in df.columns:
        lower = col.lower()
        if lower in ["sensorid", "sensor_id", "id"]:
            col_mapping[col] = "sensorId"
        elif lower in ["latitude", "lat"]:
            col_mapping[col] = "latitude"
        elif lower in ["longitude", "long", "lon", "lng"]:
            col_mapping[col] = "longitude"

    if col_mapping:
        df = df.rename(col_mapping)

    # Verify we have required columns
    required = ["sensorId", "latitude", "longitude"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Metadata file missing columns: {missing}. Available: {df.columns}"
        )

    return df.select(["sensorId", "latitude", "longitude"])


def add_locations(rollup_file: str, metadata_file: str, output_file: str = None):
    """
    Add sensor locations to rollup data.

    Args:
        rollup_file: Path to canada_sensor_rollup.parquet
        metadata_file: Path to sensor metadata CSV/Parquet
        output_file: Output path (optional, defaults to adding _with_locations suffix)
    """
    print(f"Reading rollup data from {rollup_file}...")
    rollup_df = pl.read_parquet(rollup_file)

    print(f"Reading sensor metadata from {metadata_file}...")
    if metadata_file.endswith(".csv"):
        metadata_df = pl.read_csv(metadata_file)
    elif metadata_file.endswith(".parquet"):
        metadata_df = pl.read_parquet(metadata_file)
    else:
        raise ValueError(f"Unsupported metadata file format: {metadata_file}")

    print("Standardizing metadata column names...")
    metadata_df = standardize_metadata_columns(metadata_df)

    # Make sure sensorId types match
    metadata_df = metadata_df.with_columns([pl.col("sensorId").cast(pl.Int64)])

    print(f"Found {len(metadata_df)} sensors with coordinates in metadata")

    # Drop existing lat/long columns from rollup if they exist
    cols_to_drop = []
    if "latitude" in rollup_df.columns:
        cols_to_drop.append("latitude")
    if "longitude" in rollup_df.columns:
        cols_to_drop.append("longitude")

    if cols_to_drop:
        print(f"Removing existing columns: {cols_to_drop}")
        rollup_df = rollup_df.drop(cols_to_drop)

    # Join with metadata
    print("Joining with sensor metadata...")
    enriched_df = rollup_df.join(metadata_df, on="sensorId", how="left")

    # Reorder columns to put lat/long after sensor name
    col_order = ["sensorId", "sensor", "latitude", "longitude", "direction"]
    other_cols = [col for col in enriched_df.columns if col not in col_order]
    enriched_df = enriched_df.select(col_order + other_cols)

    # Report statistics
    total_rows = len(enriched_df)
    rows_with_location = enriched_df.filter(pl.col("latitude").is_not_null()).shape[0]
    unique_sensors = enriched_df["sensorId"].n_unique()
    sensors_with_location = enriched_df.filter(pl.col("latitude").is_not_null())[
        "sensorId"
    ].n_unique()

    print(f"\nResults:")
    print(f"  Total rows: {total_rows:,}")
    print(
        f"  Rows with location: {rows_with_location:,} ({100 * rows_with_location / total_rows:.1f}%)"
    )
    print(f"  Unique sensors: {unique_sensors}")
    print(
        f"  Sensors with location: {sensors_with_location} ({100 * sensors_with_location / unique_sensors:.1f}%)"
    )

    # Save output
    if output_file is None:
        base = Path(rollup_file)
        output_file = str(base.with_stem(base.stem + "_with_locations"))

    print(f"\nSaving enriched data to {output_file}...")
    enriched_df.write_parquet(output_file)

    print("Done!")
    return enriched_df


def main():
    parser = argparse.ArgumentParser(
        description="Add sensor locations to Canadian rollup data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python add_sensor_locations.py sensor_metadata.csv
  python add_sensor_locations.py sensors.parquet --output rollup_with_coords.parquet

Metadata file format:
  CSV or Parquet with columns: sensorId (or id), latitude (or lat), longitude (or long)
        """,
    )

    parser.add_argument(
        "metadata_file", help="Path to sensor metadata CSV/Parquet file"
    )
    parser.add_argument(
        "--rollup",
        default="/Users/hpanneer/GitHub/altrios-applied/applications/corridor/canada_sensor_rollup.parquet",
        help="Path to rollup parquet file (default: canada_sensor_rollup.parquet)",
    )
    parser.add_argument(
        "--output", "-o", help="Output file path (default: adds _with_locations suffix)"
    )

    args = parser.parse_args()

    # Verify files exist
    if not Path(args.rollup).exists():
        print(f"Error: Rollup file not found: {args.rollup}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.metadata_file).exists():
        print(f"Error: Metadata file not found: {args.metadata_file}", file=sys.stderr)
        sys.exit(1)

    try:
        add_locations(args.rollup, args.metadata_file, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
