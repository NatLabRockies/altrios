#!/usr/bin/env python3
"""
Generate Enhanced Statistics with ACTUAL TOTALS
Uses both monthly averages and actual total counts from separate parquet files
"""

import polars as pl
import json
from pathlib import Path
from collections import defaultdict
import logging
import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def calculate_enhanced_stats(monthly_file: str, totals_file: str, output_file: str):
    """
    Generate comprehensive statistics combining monthly averages and actual totals.
    """
    logger.info("Loading data files...")

    # Load monthly rollup (for averages)
    monthly_df = pl.read_parquet(monthly_file)
    logger.info(f"Loaded monthly rollup: {len(monthly_df):,} rows")

    # Load actual totals
    totals_df = pl.read_parquet(totals_file)
    logger.info(f"Loaded actual totals: {len(totals_df):,} rows")

    stats = {}

    # ===== SENSOR OVERVIEW =====
    logger.info("Calculating sensor overview...")

    unique_sensors = monthly_df.select("sensorId", "sensor").unique()
    total_sensors = len(unique_sensors)

    unique_corridors = monthly_df.select("sensorId", "sensor", "direction").unique()
    total_corridors = len(unique_corridors)

    car_categories = sorted(
        monthly_df.select("car_category").unique()["car_category"].to_list()
    )

    stats["overview"] = {
        "total_sensors": total_sensors,
        "total_corridors": total_corridors,
        "total_car_categories": len(car_categories),
        "car_categories": car_categories,
        "time_period": {
            "start_month": f"{monthly_df['year'].min()}-{monthly_df['month'].min():02d}",
            "end_month": f"{monthly_df['year'].max()}-{monthly_df['month'].max():02d}",
            "total_months": len(monthly_df.select("year", "month").unique()),
        },
    }

    # ===== PER-SENSOR STATISTICS =====
    logger.info("Calculating per-sensor statistics...")

    sensors = []

    for sensor_info in unique_corridors.iter_rows(named=True):
        sensor_id = sensor_info["sensorId"]
        sensor_name = sensor_info["sensor"]
        direction = sensor_info["direction"]

        # Get monthly data for this sensor+direction
        sensor_monthly = monthly_df.filter(
            (pl.col("sensorId") == sensor_id) & (pl.col("direction") == direction)
        )

        # Get totals for this sensor+direction
        sensor_totals = totals_df.filter(
            (pl.col("sensorId") == sensor_id) & (pl.col("direction") == direction)
        )

        # Get date range for this sensor
        date_range = sensor_monthly.select(
            [
                pl.col("year"),
                pl.col("month"),
            ]
        ).sort(["year", "month"])

        first_row = date_range.head(1).to_dicts()[0]
        last_row = date_range.tail(1).to_dicts()[0]

        start_year = first_row["year"]
        start_month = first_row["month"]
        end_year = last_row["year"]
        end_month = last_row["month"]

        start_date = f"{start_year}-{start_month:02d}-01"

        # Calculate end date as last day of the end month
        if end_month == 12:
            # December - use 31st
            end_date = f"{end_year}-{end_month:02d}-31"
        else:
            # Use last day of month (approximate with first day of next month - 1 day)
            last_day = (
                datetime.date(end_year, end_month + 1, 1) - datetime.timedelta(days=1)
            ).day
            end_date = f"{end_year}-{end_month:02d}-{last_day:02d}"

        # Overall averages (across all car categories)
        overall_avg = sensor_monthly.select(
            [
                pl.col("daily_avg_total_trains").mean().alias("avg_daily_trains"),
                pl.col("avg_speed_mph").mean().alias("avg_speed"),
                pl.col("avg_daily_train_types").mean().alias("avg_train_types"),
                pl.col("avg_daily_operators").mean().alias("avg_operators"),
            ]
        ).to_dicts()[0]

        # Per-category statistics
        category_stats = {}

        for category in car_categories:
            # Monthly averages for this category
            cat_monthly = sensor_monthly.filter(pl.col("car_category") == category)

            if len(cat_monthly) == 0:
                continue

            # Averages from monthly data
            avg_stats = cat_monthly.select(
                [
                    pl.col("daily_avg_unique_trains").mean().alias("avg_daily_trains"),
                    pl.col("daily_avg_car_count").mean().alias("avg_daily_cars"),
                    pl.col("daily_avg_unique_car_ids")
                    .mean()
                    .alias("avg_daily_unique_cars"),
                ]
            ).to_dicts()[0]

            # Totals from actual raw data counts
            cat_totals = sensor_totals.filter(pl.col("car_category") == category)

            if len(cat_totals) > 0:
                total_stats = cat_totals.select(
                    [
                        pl.col("total_unique_trains").alias("total_trains"),
                        pl.col("total_car_count").alias("total_cars"),
                        pl.col("total_unique_car_ids").alias("total_unique_car_ids"),
                        pl.col("total_days_with_sightings").alias(
                            "days_with_sightings"
                        ),
                        pl.col("months_covered").alias("months_covered"),
                    ]
                ).to_dicts()[0]
            else:
                total_stats = {
                    "total_trains": 0,
                    "total_cars": 0,
                    "total_unique_car_ids": 0,
                    "days_with_sightings": 0,
                    "months_covered": 0,
                }

            # Combine averages and totals
            category_stats[category] = {**avg_stats, **total_stats}

        # Extract province
        province = (
            sensor_name.split(",")[-1].strip().split()[0]
            if "," in sensor_name
            else "Unknown"
        )

        # Get lat/long and other metadata from first row
        metadata = (
            sensor_monthly.select(
                ["latitude", "longitude", "country", "region", "timezone"]
            )
            .head(1)
            .to_dicts()
        )

        if metadata:
            metadata_dict = metadata[0]
        else:
            metadata_dict = {
                "latitude": None,
                "longitude": None,
                "country": None,
                "region": None,
                "timezone": None,
            }

        sensors.append(
            {
                "sensorId": sensor_id,
                "sensor": sensor_name,
                "province": province,
                "direction": direction,
                "latitude": metadata_dict.get("latitude"),
                "longitude": metadata_dict.get("longitude"),
                "country": metadata_dict.get("country"),
                "region": metadata_dict.get("region"),
                "timezone": metadata_dict.get("timezone"),
                "start_date": start_date,
                "end_date": end_date,
                "overall_statistics": overall_avg,
                "car_category_statistics": category_stats,
            }
        )

    stats["sensors"] = sensors
    logger.info(f"Processed {len(sensors)} sensor corridors")

    # ===== BY PROVINCE =====
    logger.info("Calculating province-level statistics...")

    provinces = defaultdict(
        lambda: {
            "sensors": set(),
            "corridors": [],
            "monthly_data": [],
            "totals_data": [],
        }
    )

    # Collect data by province
    for row in monthly_df.iter_rows(named=True):
        sensor_name = row["sensor"]
        province = (
            sensor_name.split(",")[-1].strip().split()[0]
            if "," in sensor_name
            else "Unknown"
        )

        provinces[province]["sensors"].add(sensor_name)
        provinces[province]["corridors"].append(f"{sensor_name}_{row['direction']}")
        provinces[province]["monthly_data"].append(row)

    for row in totals_df.iter_rows(named=True):
        sensor_name = row["sensor"]
        province = (
            sensor_name.split(",")[-1].strip().split()[0]
            if "," in sensor_name
            else "Unknown"
        )
        provinces[province]["totals_data"].append(row)

    province_stats = {}

    for province, data in provinces.items():
        # Convert to Polars for aggregation
        monthly_pl = pl.DataFrame(data["monthly_data"])
        totals_pl = pl.DataFrame(data["totals_data"])

        # Monthly averages
        monthly_agg = monthly_pl.group_by("car_category").agg(
            [
                pl.col("daily_avg_unique_trains").mean().alias("avg_daily_trains"),
                pl.col("daily_avg_car_count").mean().alias("avg_daily_cars"),
                pl.col("daily_avg_unique_car_ids")
                .mean()
                .alias("avg_daily_unique_cars"),
            ]
        )

        # Actual totals
        totals_agg = totals_pl.group_by("car_category").agg(
            [
                pl.col("total_unique_trains").sum().alias("total_trains"),
                pl.col("total_car_count").sum().alias("total_cars"),
                pl.col("total_unique_car_ids").sum().alias("total_unique_car_ids"),
            ]
        )

        # Combine
        combined = monthly_agg.join(totals_agg, on="car_category", how="outer")

        category_breakdown = {}
        for row in combined.iter_rows(named=True):
            category_breakdown[row["car_category"]] = {
                "avg_daily_trains": row.get("avg_daily_trains"),
                "avg_daily_cars": row.get("avg_daily_cars"),
                "avg_daily_unique_cars": row.get("avg_daily_unique_cars"),
                "total_trains": row.get("total_trains", 0),
                "total_cars": row.get("total_cars", 0),
                "total_unique_car_ids": row.get("total_unique_car_ids", 0),
            }

        province_stats[province] = {
            "sensor_count": len(data["sensors"]),
            "corridor_count": len(set(data["corridors"])),
            "sensors": sorted(list(data["sensors"])),
            "car_category_statistics": category_breakdown,
        }

    stats["by_province"] = province_stats
    logger.info(f"Processed {len(province_stats)} provinces")

    # ===== LOADED VS EMPTY BREAKDOWN =====
    logger.info("Calculating loaded vs empty breakdown...")

    loaded_empty_stats = {}

    for category in car_categories:
        # Monthly averages
        cat_monthly = monthly_df.filter(pl.col("car_category") == category)

        if len(cat_monthly) > 0:
            avg_stats = cat_monthly.select(
                [
                    pl.col("daily_avg_unique_trains").mean().alias("avg_daily_trains"),
                    pl.col("daily_avg_car_count").mean().alias("avg_daily_cars"),
                    pl.col("daily_avg_unique_car_ids")
                    .mean()
                    .alias("avg_daily_unique_cars"),
                ]
            ).to_dicts()[0]
        else:
            avg_stats = {
                "avg_daily_trains": 0,
                "avg_daily_cars": 0,
                "avg_daily_unique_cars": 0,
            }

        # Actual totals
        cat_totals = totals_df.filter(pl.col("car_category") == category)

        if len(cat_totals) > 0:
            total_stats = cat_totals.select(
                [
                    pl.col("total_unique_trains").sum().alias("total_trains"),
                    pl.col("total_car_count").sum().alias("total_cars"),
                    pl.col("total_unique_car_ids").sum().alias("total_unique_car_ids"),
                ]
            ).to_dicts()[0]
        else:
            total_stats = {
                "total_trains": 0,
                "total_cars": 0,
                "total_unique_car_ids": 0,
            }

        loaded_empty_stats[category] = {**avg_stats, **total_stats}

    stats["loaded_vs_empty"] = loaded_empty_stats

    # ===== SAVE RESULTS =====
    logger.info(f"Saving statistics to {output_file}...")

    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    logger.info("✓ Statistics saved successfully!")

    # Print summary
    print("\n" + "=" * 70)
    print("ENHANCED STATISTICS GENERATED")
    print("=" * 70)
    print(f"Total sensors: {stats['overview']['total_sensors']}")
    print(f"Total corridors: {stats['overview']['total_corridors']}")
    print(f"Car categories: {stats['overview']['total_car_categories']}")
    print(
        f"Time period: {stats['overview']['time_period']['start_month']} to {stats['overview']['time_period']['end_month']}"
    )
    print(f"Provinces: {len(stats['by_province'])}")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        monthly_file = sys.argv[1]
        totals_file = sys.argv[2] if len(sys.argv) > 2 else None
        output_file = (
            sys.argv[3]
            if len(sys.argv) > 3
            else "north_america_sensor_rollup_enhanced_stats.json"
        )
    else:
        monthly_file = "/Users/hpanneer/GitHub/altrios-applied/applications/corridor/north_america_sensor_rollup_monthly.parquet"
        totals_file = "/Users/hpanneer/GitHub/altrios-applied/applications/corridor/north_america_sensor_rollup_totals.parquet"
        output_file = "/Users/hpanneer/GitHub/altrios-applied/applications/corridor/north_america_sensor_rollup_enhanced_stats.json"

    calculate_enhanced_stats(monthly_file, totals_file, output_file)
