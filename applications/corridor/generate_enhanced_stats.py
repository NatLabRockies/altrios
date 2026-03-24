#!/usr/bin/env python3
"""
Generate enhanced statistics for Canadian sensor rollup data.
Includes detailed breakdowns by sensor, location, direction, and car category.
"""

import polars as pl
import json
from pathlib import Path


def generate_enhanced_stats(input_file: str, output_file: str):
    """
    Generate comprehensive statistics from Canadian sensor rollup data.

    Args:
        input_file: Path to parquet file
        output_file: Path to output JSON file
    """
    print(f"Reading {input_file}...")
    df = pl.read_parquet(input_file)

    print(f"Generating enhanced statistics...")

    # Overall summary
    stats = {
        "summary": {
            "total_rows": len(df),
            "unique_sensors": df["sensorId"].n_unique(),
            "unique_corridors": df.select(
                [
                    pl.concat_str(
                        [pl.col("sensor"), pl.lit(" "), pl.col("direction")]
                    ).alias("corridor")
                ]
            )["corridor"].n_unique(),
            "date_range": {
                "start_year": int(df["year"].min()),
                "start_month": int(df["month"].min()),
                "end_year": int(df["year"].max()),
                "end_month": int(df["month"].max()),
            },
            "total_car_categories": df["car_category"].n_unique(),
        }
    }

    # 1. Sensor details with location and direction
    print("  - Computing sensor details...")
    sensor_details = (
        df.group_by(["sensorId", "sensor", "latitude", "longitude", "direction"])
        .agg(
            [
                pl.col("daily_avg_total_trains").mean().alias("avg_daily_trains"),
                pl.col("avg_speed_mph").mean().alias("avg_speed_mph"),
                pl.col("days_active").max().alias("total_days_active"),
                pl.col("year").n_unique().alias("years_covered"),
                pl.col("month").n_unique().alias("months_covered"),
            ]
        )
        .sort("avg_daily_trains", descending=True)
    )

    stats["sensors"] = [
        {
            "sensorId": int(row["sensorId"]),
            "sensor_name": row["sensor"],
            "latitude": row["latitude"] if row["latitude"] is not None else None,
            "longitude": row["longitude"] if row["longitude"] is not None else None,
            "direction": row["direction"],
            "avg_daily_trains": round(float(row["avg_daily_trains"]), 2),
            "avg_speed_mph": round(float(row["avg_speed_mph"]), 2),
            "total_days_active": int(row["total_days_active"]),
            "years_covered": int(row["years_covered"]),
            "months_covered": int(row["months_covered"]),
        }
        for row in sensor_details.iter_rows(named=True)
    ]

    # 2. Daily unique trains per category (by loaded status)
    print("  - Computing daily trains by car category...")
    trains_by_category = (
        df.group_by("car_category")
        .agg(
            [
                pl.col("daily_avg_unique_trains").mean().alias("avg_daily_trains"),
                pl.col("daily_avg_unique_trains").sum().alias("total_daily_trains"),
                pl.col("sensorId").n_unique().alias("sensors_with_category"),
            ]
        )
        .sort("avg_daily_trains", descending=True)
    )

    # Split by loaded status
    trains_loaded = trains_by_category.filter(
        pl.col("car_category").str.contains("_Loaded")
    )
    trains_empty = trains_by_category.filter(
        pl.col("car_category").str.contains("_Empty")
    )
    trains_unknown = trains_by_category.filter(
        pl.col("car_category").str.contains("_Unknown")
    )

    stats["daily_trains_by_category"] = {
        "loaded": [
            {
                "car_category": row["car_category"],
                "avg_daily_trains": round(float(row["avg_daily_trains"]), 3),
                "total_daily_trains": round(float(row["total_daily_trains"]), 2),
                "sensors_with_category": int(row["sensors_with_category"]),
            }
            for row in trains_loaded.iter_rows(named=True)
        ],
        "empty": [
            {
                "car_category": row["car_category"],
                "avg_daily_trains": round(float(row["avg_daily_trains"]), 3),
                "total_daily_trains": round(float(row["total_daily_trains"]), 2),
                "sensors_with_category": int(row["sensors_with_category"]),
            }
            for row in trains_empty.iter_rows(named=True)
        ],
        "unknown": [
            {
                "car_category": row["car_category"],
                "avg_daily_trains": round(float(row["avg_daily_trains"]), 3),
                "total_daily_trains": round(float(row["total_daily_trains"]), 2),
                "sensors_with_category": int(row["sensors_with_category"]),
            }
            for row in trains_unknown.iter_rows(named=True)
        ],
    }

    # 3. Daily car counts per category (by loaded status)
    print("  - Computing daily car counts by category...")
    cars_by_category = (
        df.group_by("car_category")
        .agg(
            [
                pl.col("daily_avg_car_count").mean().alias("avg_daily_cars"),
                pl.col("daily_avg_car_count").sum().alias("total_daily_cars"),
                pl.col("sensorId").n_unique().alias("sensors_with_category"),
            ]
        )
        .sort("avg_daily_cars", descending=True)
    )

    cars_loaded = cars_by_category.filter(
        pl.col("car_category").str.contains("_Loaded")
    )
    cars_empty = cars_by_category.filter(pl.col("car_category").str.contains("_Empty"))
    cars_unknown = cars_by_category.filter(
        pl.col("car_category").str.contains("_Unknown")
    )

    stats["daily_car_counts_by_category"] = {
        "loaded": [
            {
                "car_category": row["car_category"],
                "avg_daily_cars": round(float(row["avg_daily_cars"]), 3),
                "total_daily_cars": round(float(row["total_daily_cars"]), 2),
                "sensors_with_category": int(row["sensors_with_category"]),
            }
            for row in cars_loaded.iter_rows(named=True)
        ],
        "empty": [
            {
                "car_category": row["car_category"],
                "avg_daily_cars": round(float(row["avg_daily_cars"]), 3),
                "total_daily_cars": round(float(row["total_daily_cars"]), 2),
                "sensors_with_category": int(row["sensors_with_category"]),
            }
            for row in cars_empty.iter_rows(named=True)
        ],
        "unknown": [
            {
                "car_category": row["car_category"],
                "avg_daily_cars": round(float(row["avg_daily_cars"]), 3),
                "total_daily_cars": round(float(row["total_daily_cars"]), 2),
                "sensors_with_category": int(row["sensors_with_category"]),
            }
            for row in cars_unknown.iter_rows(named=True)
        ],
    }

    # 4. Daily unique car IDs per category (by loaded status)
    print("  - Computing daily unique car IDs by category...")
    unique_cars_by_category = (
        df.group_by("car_category")
        .agg(
            [
                pl.col("daily_avg_unique_car_ids")
                .mean()
                .alias("avg_daily_unique_cars"),
                pl.col("daily_avg_unique_car_ids")
                .sum()
                .alias("total_daily_unique_cars"),
                pl.col("sensorId").n_unique().alias("sensors_with_category"),
            ]
        )
        .sort("avg_daily_unique_cars", descending=True)
    )

    unique_loaded = unique_cars_by_category.filter(
        pl.col("car_category").str.contains("_Loaded")
    )
    unique_empty = unique_cars_by_category.filter(
        pl.col("car_category").str.contains("_Empty")
    )
    unique_unknown = unique_cars_by_category.filter(
        pl.col("car_category").str.contains("_Unknown")
    )

    stats["daily_unique_car_ids_by_category"] = {
        "loaded": [
            {
                "car_category": row["car_category"],
                "avg_daily_unique_cars": round(float(row["avg_daily_unique_cars"]), 3),
                "total_daily_unique_cars": round(
                    float(row["total_daily_unique_cars"]), 2
                ),
                "sensors_with_category": int(row["sensors_with_category"]),
            }
            for row in unique_loaded.iter_rows(named=True)
        ],
        "empty": [
            {
                "car_category": row["car_category"],
                "avg_daily_unique_cars": round(float(row["avg_daily_unique_cars"]), 3),
                "total_daily_unique_cars": round(
                    float(row["total_daily_unique_cars"]), 2
                ),
                "sensors_with_category": int(row["sensors_with_category"]),
            }
            for row in unique_empty.iter_rows(named=True)
        ],
        "unknown": [
            {
                "car_category": row["car_category"],
                "avg_daily_unique_cars": round(float(row["avg_daily_unique_cars"]), 3),
                "total_daily_unique_cars": round(
                    float(row["total_daily_unique_cars"]), 2
                ),
                "sensors_with_category": int(row["sensors_with_category"]),
            }
            for row in unique_unknown.iter_rows(named=True)
        ],
    }

    # 5. Top corridors by direction
    print("  - Computing top corridors by direction...")
    for direction in ["Eastbound", "Westbound", "Northbound", "Southbound"]:
        dir_df = df.filter(pl.col("direction") == direction)
        if len(dir_df) > 0:
            top_by_dir = (
                dir_df.group_by(["sensor", "direction"])
                .agg(
                    [
                        pl.col("daily_avg_total_trains")
                        .mean()
                        .alias("avg_daily_trains"),
                    ]
                )
                .sort("avg_daily_trains", descending=True)
                .head(10)
            )

            stats[f"top_10_{direction.lower()}"] = [
                {
                    "corridor": f"{row['sensor']} {row['direction']}",
                    "avg_daily_trains": round(float(row["avg_daily_trains"]), 2),
                }
                for row in top_by_dir.iter_rows(named=True)
            ]

    # 6. Per-sensor car counts by category
    print("  - Computing per-sensor car counts by category...")
    sensor_category_stats = df.group_by(
        ["sensorId", "sensor", "direction", "car_category"]
    ).agg(
        [
            # Average daily values (mean across months)
            pl.col("daily_avg_car_count").mean().alias("avg_daily_cars"),
            pl.col("daily_avg_unique_trains").mean().alias("avg_daily_trains"),
            pl.col("daily_avg_unique_car_ids").mean().alias("avg_daily_unique_cars"),
            # Total counts in period (sum of daily_avg * days_active)
            (pl.col("daily_avg_car_count") * pl.col("days_active"))
            .sum()
            .alias("total_cars_in_period"),
            (pl.col("daily_avg_unique_trains") * pl.col("days_active"))
            .sum()
            .alias("total_trains_in_period"),
            (pl.col("daily_avg_unique_car_ids") * pl.col("days_active"))
            .sum()
            .alias("total_unique_cars_in_period"),
            # Metadata
            pl.col("year").n_unique().alias("years_covered"),
            pl.col("month").n_unique().alias("months_with_data"),
            pl.col("days_active").sum().alias("total_days_active"),
        ]
    )

    # Create nested structure: sensor -> direction -> categories
    stats["per_sensor_car_counts"] = {}

    for row in sensor_category_stats.iter_rows(named=True):
        sensor_key = f"{row['sensorId']}_{row['sensor']}_{row['direction']}"

        if sensor_key not in stats["per_sensor_car_counts"]:
            stats["per_sensor_car_counts"][sensor_key] = {
                "sensorId": int(row["sensorId"]),
                "sensor_name": row["sensor"],
                "direction": row["direction"],
                "by_category": {"loaded": [], "empty": [], "unknown": []},
            }

        # Determine loaded status
        category = row["car_category"]
        if "_Loaded" in category:
            status = "loaded"
        elif "_Empty" in category:
            status = "empty"
        else:
            status = "unknown"

        stats["per_sensor_car_counts"][sensor_key]["by_category"][status].append(
            {
                "car_category": category,
                "avg_daily_cars": round(float(row["avg_daily_cars"]), 3),
                "avg_daily_trains": round(float(row["avg_daily_trains"]), 3),
                "avg_daily_unique_car_ids": round(
                    float(row["avg_daily_unique_cars"]), 3
                ),
                "total_cars_in_period": round(float(row["total_cars_in_period"]), 1),
                "total_trains_in_period": round(
                    float(row["total_trains_in_period"]), 1
                ),
                "total_unique_cars_in_period": round(
                    float(row["total_unique_cars_in_period"]), 1
                ),
                "months_with_data": int(row["months_with_data"]),
                "years_covered": int(row["years_covered"]),
                "total_days_active": int(row["total_days_active"]),
            }
        )

    # Convert dict to list for JSON serialization
    stats["per_sensor_car_counts"] = list(stats["per_sensor_car_counts"].values())

    # 7. Breakdown by province
    print("  - Computing breakdown by province...")

    def extract_province(sensor_name):
        """Extract province code from sensor name."""
        if not sensor_name or not isinstance(sensor_name, str):
            return "Unknown"
        parts = sensor_name.split(",")
        if len(parts) >= 2:
            state_part = parts[1].strip().split()[0]
            return state_part
        return "Unknown"

    df_with_province = df.with_columns(
        [
            pl.col("sensor")
            .map_elements(extract_province, return_dtype=pl.String)
            .alias("province")
        ]
    )

    province_stats = (
        df_with_province.group_by("province")
        .agg(
            [
                pl.col("sensorId").n_unique().alias("unique_sensors"),
                pl.col("daily_avg_total_trains").mean().alias("avg_daily_trains"),
                pl.col("avg_speed_mph").mean().alias("avg_speed_mph"),
            ]
        )
        .sort("unique_sensors", descending=True)
    )

    stats["by_province"] = [
        {
            "province": row["province"],
            "unique_sensors": int(row["unique_sensors"]),
            "avg_daily_trains": round(float(row["avg_daily_trains"]), 2),
            "avg_speed_mph": round(float(row["avg_speed_mph"]), 2),
        }
        for row in province_stats.iter_rows(named=True)
    ]

    # Save to JSON
    print(f"Saving enhanced statistics to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStatistics saved successfully!")
    print(f"  - Total sensors: {stats['summary']['unique_sensors']}")
    print(f"  - Total corridors: {stats['summary']['unique_corridors']}")
    print(f"  - Car categories: {stats['summary']['total_car_categories']}")
    print(f"  - Sensor details: {len(stats['sensors'])} entries")
    print(f"  - Per-sensor car counts: {len(stats['per_sensor_car_counts'])} sensors")
    print(f"  - Provinces covered: {len(stats['by_province'])}")

    return stats


if __name__ == "__main__":
    input_file = "/Users/hpanneer/GitHub/altrios-applied/applications/corridor/canada_sensor_rollup.parquet"
    output_file = "/Users/hpanneer/GitHub/altrios-applied/applications/corridor/canada_sensor_rollup_enhanced_stats.json"

    generate_enhanced_stats(input_file, output_file)
