#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Convert north_america_sensor_rollup_enhanced_stats.json into multiple CSV files
for easier analysis and visualization
"""

import polars as pl
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def convert_stats_to_csv(json_file: str, output_dir: str):
    """
    Convert the enhanced statistics JSON into multiple CSV files.
    """
    logger.info(f"Loading statistics from {json_file}...")

    with open(json_file, "r") as f:
        stats = json.load(f)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ===== 1. SENSOR OVERVIEW =====
    logger.info("Creating sensor overview CSV...")

    sensor_overview = []
    for sensor in stats["sensors"]:
        sensor_overview.append(
            {
                "sensorId": sensor["sensorId"],
                "sensor": sensor["sensor"],
                "province": sensor["province"],
                "direction": sensor["direction"],
                "latitude": sensor.get("latitude"),
                "longitude": sensor.get("longitude"),
                "country": sensor.get("country"),
                "region": sensor.get("region"),
                "timezone": sensor.get("timezone"),
                "start_date": sensor["start_date"],
                "end_date": sensor["end_date"],
                "avg_daily_trains": sensor["overall_statistics"]["avg_daily_trains"],
                "avg_speed_mph": sensor["overall_statistics"]["avg_speed"],
                "avg_train_types": sensor["overall_statistics"]["avg_train_types"],
                "avg_operators": sensor["overall_statistics"]["avg_operators"],
            }
        )

    df_overview = pl.DataFrame(sensor_overview)
    overview_file = output_path / "01_sensor_overview.csv"
    df_overview.write_csv(overview_file)
    logger.info(f"✓ Saved {overview_file} ({len(df_overview)} rows)")

    # ===== 2. PER-SENSOR CAR CATEGORY STATISTICS =====
    logger.info("Creating per-sensor car category statistics CSV...")

    sensor_category_stats = []
    for sensor in stats["sensors"]:
        for category, cat_stats in sensor["car_category_statistics"].items():
            sensor_category_stats.append(
                {
                    "sensorId": sensor["sensorId"],
                    "sensor": sensor["sensor"],
                    "province": sensor["province"],
                    "direction": sensor["direction"],
                    "latitude": sensor.get("latitude"),
                    "longitude": sensor.get("longitude"),
                    "country": sensor.get("country"),
                    "region": sensor.get("region"),
                    "start_date": sensor["start_date"],
                    "end_date": sensor["end_date"],
                    "car_category": category,
                    "avg_daily_trains": cat_stats.get("avg_daily_trains"),
                    "avg_daily_cars": cat_stats.get("avg_daily_cars"),
                    "avg_daily_unique_cars": cat_stats.get("avg_daily_unique_cars"),
                    "total_trains": cat_stats.get("total_trains", 0),
                    "total_cars": cat_stats.get("total_cars", 0),
                    "total_unique_car_ids": cat_stats.get("total_unique_car_ids", 0),
                    "days_with_sightings": cat_stats.get("days_with_sightings", 0),
                    "months_covered": cat_stats.get("months_covered", 0),
                }
            )

    df_category = pl.DataFrame(sensor_category_stats)
    category_file = output_path / "02_sensor_car_category_details.csv"
    df_category.write_csv(category_file)
    logger.info(f"✓ Saved {category_file} ({len(df_category)} rows)")

    # ===== 3. PROVINCE SUMMARY =====
    logger.info("Creating province summary CSV...")

    province_summary = []
    for province, prov_stats in stats["by_province"].items():
        province_summary.append(
            {
                "province": province,
                "sensor_count": prov_stats["sensor_count"],
                "corridor_count": prov_stats["corridor_count"],
                "sensors": ", ".join(prov_stats["sensors"]),
            }
        )

    df_province = pl.DataFrame(province_summary)
    province_file = output_path / "03_province_summary.csv"
    df_province.write_csv(province_file)
    logger.info(f"✓ Saved {province_file} ({len(df_province)} rows)")

    # ===== 4. PROVINCE CAR CATEGORY STATISTICS =====
    logger.info("Creating province car category statistics CSV...")

    province_category_stats = []
    for province, prov_stats in stats["by_province"].items():
        for category, cat_stats in prov_stats["car_category_statistics"].items():
            province_category_stats.append(
                {
                    "province": province,
                    "car_category": category,
                    "avg_daily_trains": cat_stats.get("avg_daily_trains"),
                    "avg_daily_cars": cat_stats.get("avg_daily_cars"),
                    "avg_daily_unique_cars": cat_stats.get("avg_daily_unique_cars"),
                    "total_trains": cat_stats.get("total_trains", 0),
                    "total_cars": cat_stats.get("total_cars", 0),
                    "total_unique_car_ids": cat_stats.get("total_unique_car_ids", 0),
                }
            )

    df_prov_category = pl.DataFrame(province_category_stats)
    prov_category_file = output_path / "04_province_car_category_stats.csv"
    df_prov_category.write_csv(prov_category_file)
    logger.info(f"✓ Saved {prov_category_file} ({len(df_prov_category)} rows)")

    # ===== 5. LOADED VS EMPTY SUMMARY =====
    logger.info("Creating loaded vs empty summary CSV...")

    loaded_empty_stats = []
    for category, cat_stats in stats["loaded_vs_empty"].items():
        # Parse category to get car type and loaded status
        if "_" in category:
            parts = category.rsplit("_", 1)
            car_type = parts[0]
            loaded_status = parts[1] if len(parts) > 1 else "Unknown"
        else:
            car_type = category
            loaded_status = "Unknown"

        loaded_empty_stats.append(
            {
                "car_category": category,
                "car_type": car_type,
                "loaded_status": loaded_status,
                "avg_daily_trains": cat_stats.get("avg_daily_trains"),
                "avg_daily_cars": cat_stats.get("avg_daily_cars"),
                "avg_daily_unique_cars": cat_stats.get("avg_daily_unique_cars"),
                "total_trains": cat_stats.get("total_trains", 0),
                "total_cars": cat_stats.get("total_cars", 0),
                "total_unique_car_ids": cat_stats.get("total_unique_car_ids", 0),
            }
        )

    df_loaded_empty = pl.DataFrame(loaded_empty_stats)
    loaded_empty_file = output_path / "05_loaded_vs_empty_summary.csv"
    df_loaded_empty.write_csv(loaded_empty_file)
    logger.info(f"✓ Saved {loaded_empty_file} ({len(df_loaded_empty)} rows)")

    # ===== 6. STACK CAR SPECIFIC ANALYSIS =====
    logger.info("Creating stack car specific CSV...")

    stack_car_stats = []
    for sensor in stats["sensors"]:
        for category, cat_stats in sensor["car_category_statistics"].items():
            if "Stack" in category:
                stack_car_stats.append(
                    {
                        "sensorId": sensor["sensorId"],
                        "sensor": sensor["sensor"],
                        "province": sensor["province"],
                        "direction": sensor["direction"],
                        "latitude": sensor.get("latitude"),
                        "longitude": sensor.get("longitude"),
                        "country": sensor.get("country"),
                        "region": sensor.get("region"),
                        "start_date": sensor["start_date"],
                        "end_date": sensor["end_date"],
                        "car_category": category,
                        "loaded_status": category.split("_")[-1]
                        if "_" in category
                        else "Unknown",
                        "avg_daily_trains": cat_stats.get("avg_daily_trains"),
                        "avg_daily_cars": cat_stats.get("avg_daily_cars"),
                        "avg_daily_unique_cars": cat_stats.get("avg_daily_unique_cars"),
                        "total_trains": cat_stats.get("total_trains", 0),
                        "total_cars": cat_stats.get("total_cars", 0),
                        "total_unique_car_ids": cat_stats.get(
                            "total_unique_car_ids", 0
                        ),
                        "days_with_sightings": cat_stats.get("days_with_sightings", 0),
                        "months_covered": cat_stats.get("months_covered", 0),
                    }
                )

    if stack_car_stats:
        df_stack = pl.DataFrame(stack_car_stats)
        stack_file = output_path / "06_stack_car_analysis.csv"
        df_stack.write_csv(stack_file)
        logger.info(f"✓ Saved {stack_file} ({len(df_stack)} rows)")

    # ===== 7. TOP SENSORS BY VOLUME =====
    logger.info("Creating top sensors by volume CSV...")

    # Calculate total volume per sensor
    sensor_volumes = []
    for sensor in stats["sensors"]:
        total_cars = sum(
            cat_stats.get("total_cars", 0)
            for cat_stats in sensor["car_category_statistics"].values()
        )
        total_trains = sum(
            cat_stats.get("total_trains", 0)
            for cat_stats in sensor["car_category_statistics"].values()
        )

        sensor_volumes.append(
            {
                "sensorId": sensor["sensorId"],
                "sensor": sensor["sensor"],
                "province": sensor["province"],
                "direction": sensor["direction"],
                "latitude": sensor.get("latitude"),
                "longitude": sensor.get("longitude"),
                "country": sensor.get("country"),
                "region": sensor.get("region"),
                "start_date": sensor["start_date"],
                "end_date": sensor["end_date"],
                "total_cars": total_cars,
                "total_trains": total_trains,
                "avg_speed_mph": sensor["overall_statistics"]["avg_speed"],
            }
        )

    df_volumes = pl.DataFrame(sensor_volumes).sort("total_cars", descending=True)
    volumes_file = output_path / "07_sensor_volume_ranking.csv"
    df_volumes.write_csv(volumes_file)
    logger.info(f"✓ Saved {volumes_file} ({len(df_volumes)} rows)")

    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("CSV CONVERSION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  1. 01_sensor_overview.csv - {len(df_overview)} sensors")
    print(
        f"  2. 02_sensor_car_category_details.csv - {len(df_category)} sensor+category combinations"
    )
    print(f"  3. 03_province_summary.csv - {len(df_province)} provinces")
    print(
        f"  4. 04_province_car_category_stats.csv - {len(df_prov_category)} province+category combinations"
    )
    print(
        f"  5. 05_loaded_vs_empty_summary.csv - {len(df_loaded_empty)} car categories"
    )
    if stack_car_stats:
        print(f"  6. 06_stack_car_analysis.csv - {len(df_stack)} stack car entries")
    print(
        f"  7. 07_sensor_volume_ranking.csv - {len(df_volumes)} sensors ranked by volume"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "csv_exports"
    else:
        json_file = "/Users/hpanneer/GitHub/altrios-applied/applications/corridor/north_america_sensor_rollup_enhanced_stats.json"
        output_dir = (
            "/Users/hpanneer/GitHub/altrios-applied/applications/corridor/csv_exports"
        )

    convert_stats_to_csv(json_file, output_dir)
