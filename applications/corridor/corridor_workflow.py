#!/usr/bin/env python3
"""
Unified Corridor Workflow

Single script that processes RailState sensor data through the corridor-selection
pipeline:
  1. Rollup:   Parse JSONL sightings → monthly averages + actual totals (parquet)
  2. Metadata: Enrich parquet files with lat/lng, country, region from sensors_overview
  3. Stats:    Generate enhanced JSON statistics (per-sensor, per-province, loaded/empty)
  4. CSV:      Export statistics to multiple CSV files for analysis

Once a corridor is selected from these statistics, build a routable ALTRIOS
network for it with the network builder
(applications/network_builder/network_builder_demo.py).

Drop an enhanced-stats JSON under applications/corridor/demo_data/ (gitignored,
so real RailState data stays local) and you can regenerate the selection CSVs
from it directly — no rollup/metadata/stats rerun needed.

Usage (uv-managed environment):
    # One-time setup: sync deps + build altrios (auto-selects Python 3.10/3.11):
    uv sync

    # Build the selection CSVs straight from an existing enhanced-stats JSON
    # (e.g. one kept locally under demo_data/, no RailState input needed):
    uv run python applications/corridor/corridor_workflow.py --steps csv \
        --stats-json applications/corridor/demo_data/<your_enhanced_stats>.json

    # Full pipeline on real RailState data (rollup + metadata + stats + csv):
    uv run python applications/corridor/corridor_workflow.py

    # All US states (not just border states):
    uv run python applications/corridor/corridor_workflow.py --all-states

    # Custom input/output:
    uv run python applications/corridor/corridor_workflow.py \
        --input-dir /path/to/jsonl --output-dir /path/to/output

    # Run only specific steps:
    uv run python applications/corridor/corridor_workflow.py \
        --steps rollup metadata stats csv
"""

import polars as pl
import logging
import json
import argparse
import sys
import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Region filtering
# ---------------------------------------------------------------------------

CANADIAN_PROVINCES = {
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

US_BORDER_STATES = {
    # Northern border
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
    # Southern border
    "CA",
    "AZ",
    "NM",
    "TX",
}

US_ALL_STATES = {
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


def get_regions(include_all_us_states: bool = False) -> set:
    us = US_ALL_STATES if include_all_us_states else US_BORDER_STATES
    return CANADIAN_PROVINCES | us


def is_in_region(sensor_name: str, regions: set) -> bool:
    if not sensor_name or not isinstance(sensor_name, str):
        return False
    parts = sensor_name.split(",")
    if len(parts) < 2:
        return False
    state_part = parts[1].strip().split()[0]
    return state_part in regions


# ---------------------------------------------------------------------------
# Step 1: Rollup – JSONL → monthly parquet + totals parquet
# ---------------------------------------------------------------------------


def step_rollup(
    input_files: List[str],
    output_dir: str,
    include_all_us_states: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Parse JSONL files and produce monthly rollup + actual totals."""
    logger.info("=" * 70)
    logger.info("STEP 1: Sensor rollup")
    logger.info("=" * 70)

    regions = get_regions(include_all_us_states)
    all_monthly: list[pl.DataFrame] = []
    all_totals: list[pl.DataFrame] = []

    for filepath in input_files:
        try:
            monthly_df, totals_df = _process_single_file(filepath, regions)
            all_monthly.append(monthly_df)
            all_totals.append(totals_df)
            logger.info(f"  ✓ {Path(filepath).name}")
        except Exception as e:
            logger.error(f"  ✗ {Path(filepath).name}: {e}")

    if not all_monthly:
        raise RuntimeError("No files were successfully processed")

    monthly_combined = pl.concat(all_monthly, how="diagonal_relaxed")

    totals_combined = pl.concat(all_totals, how="diagonal_relaxed")
    final_totals = totals_combined.group_by(
        ["sensorId", "sensor", "latitude", "longitude", "direction", "car_category"]
    ).agg(
        [
            pl.col("total_unique_trains").sum(),
            pl.col("total_car_count").sum(),
            pl.col("total_unique_car_ids").max(),
            pl.col("total_days_with_sightings").sum(),
            pl.col("years_covered").max(),
            pl.col("months_covered").sum(),
        ]
    )

    suffix = "all_states" if include_all_us_states else "border_states"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    monthly_path = out / f"sensor_rollup_monthly_{suffix}.parquet"
    totals_path = out / f"sensor_rollup_totals_{suffix}.parquet"

    monthly_combined.write_parquet(monthly_path)
    final_totals.write_parquet(totals_path)

    logger.info(f"  Monthly rollup : {len(monthly_combined):,} rows → {monthly_path}")
    logger.info(f"  Actual totals  : {len(final_totals):,} rows → {totals_path}")
    return monthly_combined, final_totals


def _process_single_file(
    filepath: str, regions: set
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Process one JSONL file → (monthly_df, totals_df)."""
    logger.info(f"  Processing {Path(filepath).name} ...")

    try:
        df = pl.scan_ndjson(filepath, ignore_errors=True)
    except Exception:
        df = pl.read_ndjson(filepath, ignore_errors=True).lazy()

    df = df.with_columns(
        [
            pl.col("detectionEndUTC").cast(pl.String, strict=False),
            pl.col("detectionEndSensorLocal").cast(pl.String, strict=False),
        ]
    )

    # Region filter
    df = df.filter(
        pl.col("sensor").map_elements(
            lambda x: is_in_region(x, regions), return_dtype=pl.Boolean
        )
    )

    # Explode cars
    df = df.explode("cars").with_columns(
        [
            pl.col("cars").struct.field("carId").alias("car_id"),
            pl.col("cars").struct.field("type").alias("car_type"),
            pl.col("cars").struct.field("isLoaded").alias("is_loaded"),
        ]
    )

    # Timestamps
    df = df.with_columns(
        [
            pl.col("detectionTimeUTC")
            .str.strptime(pl.Datetime("us", "UTC"), "%Y-%m-%dT%H:%M:%S%.fZ")
            .alias("detection_datetime"),
        ]
    ).with_columns(
        [
            pl.col("detection_datetime").dt.date().alias("detection_date"),
            pl.col("detection_datetime").dt.year().alias("year"),
            pl.col("detection_datetime").dt.month().alias("month"),
        ]
    )

    # Category
    df = df.with_columns(
        [
            pl.when(pl.col("is_loaded").is_null())
            .then(pl.lit("Unknown"))
            .when(pl.col("is_loaded"))
            .then(pl.lit("Loaded"))
            .otherwise(pl.lit("Empty"))
            .alias("loaded_status"),
        ]
    ).with_columns(
        [
            (pl.col("car_type") + "_" + pl.col("loaded_status")).alias("car_category"),
        ]
    )

    # --- Daily → monthly averages ---
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

    monthly_rollup = daily_agg.group_by(
        [
            "sensorId",
            "sensor",
            "direction",
            "year",
            "month",
            "car_category",
        ]
    ).agg(
        [
            pl.col("unique_trains").mean().alias("daily_avg_unique_trains"),
            pl.col("car_count").mean().alias("daily_avg_car_count"),
            pl.col("unique_car_ids").mean().alias("daily_avg_unique_car_ids"),
            pl.col("detection_date").n_unique().alias("days_active"),
        ]
    )

    # --- Overall stats (not split by category) ---
    overall_daily = df.group_by(
        [
            "sensorId",
            "sensor",
            "direction",
            "year",
            "month",
            "detection_date",
        ]
    ).agg(
        [
            pl.col("trainId").n_unique().alias("daily_unique_trains"),
            pl.col("speedMph").mean().alias("daily_avg_speed_mph"),
            pl.col("trainType").n_unique().alias("daily_unique_train_types"),
            pl.col("trainOperator").n_unique().alias("daily_unique_operators"),
        ]
    )

    overall_monthly = overall_daily.group_by(
        [
            "sensorId",
            "sensor",
            "direction",
            "year",
            "month",
        ]
    ).agg(
        [
            pl.col("daily_unique_trains").mean().alias("daily_avg_total_trains"),
            pl.col("daily_avg_speed_mph").mean().alias("avg_speed_mph"),
            pl.col("daily_unique_train_types").mean().alias("avg_daily_train_types"),
            pl.col("daily_unique_operators").mean().alias("avg_daily_operators"),
            pl.col("detection_date").n_unique().alias("days_active"),
        ]
    )

    # --- Actual totals ---
    total_counts = df.group_by(
        [
            "sensorId",
            "sensor",
            "direction",
            "car_category",
        ]
    ).agg(
        [
            pl.col("trainId").n_unique().alias("total_unique_trains"),
            pl.col("car_id").count().alias("total_car_count"),
            pl.col("car_id").n_unique().alias("total_unique_car_ids"),
            pl.col("detection_date").n_unique().alias("total_days_with_sightings"),
            pl.col("year").n_unique().alias("years_covered"),
            pl.col("month").n_unique().alias("months_covered"),
        ]
    )

    # Combine monthly with overall
    final_rollup = monthly_rollup.join(
        overall_monthly,
        on=["sensorId", "sensor", "direction", "year", "month"],
        how="left",
    )

    # Sensor lat/lng placeholders
    schema_cols = df.collect_schema().names()
    if "latitude" in schema_cols and "longitude" in schema_cols:
        sensor_meta = df.group_by(["sensorId", "sensor"]).agg(
            [
                pl.col("latitude").mean().alias("latitude"),
                pl.col("longitude").mean().alias("longitude"),
            ]
        )
    else:
        sensor_meta = (
            df.select(["sensorId", "sensor"])
            .unique()
            .with_columns(
                [
                    pl.lit(None).cast(pl.Float64).alias("latitude"),
                    pl.lit(None).cast(pl.Float64).alias("longitude"),
                ]
            )
        )

    final_rollup = final_rollup.join(sensor_meta, on=["sensorId", "sensor"], how="left")
    total_counts = total_counts.join(sensor_meta, on=["sensorId", "sensor"], how="left")

    return final_rollup.collect(), total_counts.collect()


# ---------------------------------------------------------------------------
# Step 2: Metadata – enrich parquet with sensors_overview.jsonl
# ---------------------------------------------------------------------------


def step_metadata(
    monthly_path: str,
    totals_path: str,
    sensors_overview_file: str,
) -> None:
    """Add lat/lng, country, region, timezone from sensors_overview.jsonl."""
    logger.info("=" * 70)
    logger.info("STEP 2: Enrich with sensor metadata")
    logger.info("=" * 70)

    metadata_df = _load_sensor_metadata(sensors_overview_file)

    for path in [monthly_path, totals_path]:
        if not Path(path).exists():
            logger.warning(f"  Skipping missing file: {path}")
            continue
        _update_parquet_metadata(path, metadata_df, path)

    logger.info("  ✓ Metadata enrichment complete")


def _load_sensor_metadata(sensors_file: str) -> pl.DataFrame:
    with open(sensors_file, "r") as f:
        data = json.load(f)

    rows = []
    for s in data["sensors"]:
        rows.append(
            {
                "sensorId": s["sensorId"],
                "name": s["name"],
                "latitude": s.get("lat"),
                "longitude": s.get("lng"),
                "country": s.get("country"),
                "region": s.get("region"),
                "timezone": s.get("timezone"),
                "railways": ",".join(s.get("railways", [])),
                "operatingRailways": ",".join(s.get("operatingRailways", [])),
                "isActive": s.get("isActive", False),
            }
        )
    df = pl.DataFrame(rows)
    logger.info(f"  Loaded metadata for {len(df)} sensors")
    return df


def _update_parquet_metadata(
    parquet_path: str, metadata_df: pl.DataFrame, output_path: str
) -> None:
    df = pl.read_parquet(parquet_path)
    cols_to_drop = [
        c
        for c in ["latitude", "longitude", "country", "region", "timezone"]
        if c in df.columns
    ]
    if cols_to_drop:
        df = df.drop(cols_to_drop)

    df = df.join(
        metadata_df.select(
            ["sensorId", "latitude", "longitude", "country", "region", "timezone"]
        ),
        on="sensorId",
        how="left",
    )
    df.write_parquet(output_path)
    matched = df.filter(pl.col("latitude").is_not_null()).height
    logger.info(
        f"  {Path(parquet_path).name}: {matched}/{len(df)} rows with coordinates"
    )


# ---------------------------------------------------------------------------
# Step 3: Stats – JSON statistics from parquet
# ---------------------------------------------------------------------------


def step_stats(monthly_path: str, totals_path: str, output_path: str) -> dict:
    """Generate enhanced JSON statistics combining averages and totals."""
    logger.info("=" * 70)
    logger.info("STEP 3: Generate enhanced statistics")
    logger.info("=" * 70)

    monthly_df = pl.read_parquet(monthly_path)
    totals_df = pl.read_parquet(totals_path)

    car_categories = sorted(
        monthly_df.select("car_category").unique()["car_category"].to_list()
    )

    # --- Overview ---
    stats: dict = {
        "overview": {
            "total_sensors": len(monthly_df.select("sensorId", "sensor").unique()),
            "total_corridors": len(
                monthly_df.select("sensorId", "sensor", "direction").unique()
            ),
            "total_car_categories": len(car_categories),
            "car_categories": car_categories,
            "time_period": {
                "start_month": f"{monthly_df['year'].min()}-{monthly_df['month'].min():02d}",
                "end_month": f"{monthly_df['year'].max()}-{monthly_df['month'].max():02d}",
                "total_months": len(monthly_df.select("year", "month").unique()),
            },
        },
    }

    # --- Per-sensor ---
    unique_corridors = monthly_df.select("sensorId", "sensor", "direction").unique()
    sensors = []

    for row in unique_corridors.iter_rows(named=True):
        sid, sname, direction = row["sensorId"], row["sensor"], row["direction"]

        sm = monthly_df.filter(
            (pl.col("sensorId") == sid) & (pl.col("direction") == direction)
        )
        st = totals_df.filter(
            (pl.col("sensorId") == sid) & (pl.col("direction") == direction)
        )

        date_range = sm.select("year", "month").sort("year", "month")
        first, last = date_range.head(1).to_dicts()[0], date_range.tail(1).to_dicts()[0]
        start_date = f"{first['year']}-{first['month']:02d}-01"
        ey, em = last["year"], last["month"]
        last_day = (
            (datetime.date(ey, em % 12 + 1, 1) - datetime.timedelta(days=1)).day
            if em < 12
            else 31
        )
        end_date = f"{ey}-{em:02d}-{last_day:02d}"

        overall_avg = sm.select(
            [
                pl.col("daily_avg_total_trains").mean().alias("avg_daily_trains"),
                pl.col("avg_speed_mph").mean().alias("avg_speed"),
                pl.col("avg_daily_train_types").mean().alias("avg_train_types"),
                pl.col("avg_daily_operators").mean().alias("avg_operators"),
            ]
        ).to_dicts()[0]

        cat_stats = {}
        for cat in car_categories:
            cm = sm.filter(pl.col("car_category") == cat)
            if len(cm) == 0:
                continue
            avg = cm.select(
                [
                    pl.col("daily_avg_unique_trains").mean().alias("avg_daily_trains"),
                    pl.col("daily_avg_car_count").mean().alias("avg_daily_cars"),
                    pl.col("daily_avg_unique_car_ids")
                    .mean()
                    .alias("avg_daily_unique_cars"),
                ]
            ).to_dicts()[0]

            ct = st.filter(pl.col("car_category") == cat)
            if len(ct) > 0:
                tots = ct.select(
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
                tots = {
                    "total_trains": 0,
                    "total_cars": 0,
                    "total_unique_car_ids": 0,
                    "days_with_sightings": 0,
                    "months_covered": 0,
                }
            cat_stats[cat] = {**avg, **tots}

        province = (
            sname.split(",")[-1].strip().split()[0] if "," in sname else "Unknown"
        )

        meta_row = (
            sm.select(["latitude", "longitude", "country", "region", "timezone"])
            .head(1)
            .to_dicts()
        )
        meta = (
            meta_row[0]
            if meta_row
            else {
                "latitude": None,
                "longitude": None,
                "country": None,
                "region": None,
                "timezone": None,
            }
        )

        sensors.append(
            {
                "sensorId": sid,
                "sensor": sname,
                "province": province,
                "direction": direction,
                "latitude": meta.get("latitude"),
                "longitude": meta.get("longitude"),
                "country": meta.get("country"),
                "region": meta.get("region"),
                "timezone": meta.get("timezone"),
                "start_date": start_date,
                "end_date": end_date,
                "overall_statistics": overall_avg,
                "car_category_statistics": cat_stats,
            }
        )

    stats["sensors"] = sensors

    # --- By province ---
    provinces: dict = defaultdict(
        lambda: {
            "sensors": set(),
            "corridors": [],
            "monthly_data": [],
            "totals_data": [],
        }
    )
    for r in monthly_df.iter_rows(named=True):
        prov = (
            r["sensor"].split(",")[-1].strip().split()[0]
            if "," in r["sensor"]
            else "Unknown"
        )
        provinces[prov]["sensors"].add(r["sensor"])
        provinces[prov]["corridors"].append(f"{r['sensor']}_{r['direction']}")
        provinces[prov]["monthly_data"].append(r)
    for r in totals_df.iter_rows(named=True):
        prov = (
            r["sensor"].split(",")[-1].strip().split()[0]
            if "," in r["sensor"]
            else "Unknown"
        )
        provinces[prov]["totals_data"].append(r)

    prov_stats = {}
    for prov, data in provinces.items():
        mp = pl.DataFrame(data["monthly_data"])
        tp = pl.DataFrame(data["totals_data"])
        ma = mp.group_by("car_category").agg(
            [
                pl.col("daily_avg_unique_trains").mean().alias("avg_daily_trains"),
                pl.col("daily_avg_car_count").mean().alias("avg_daily_cars"),
                pl.col("daily_avg_unique_car_ids")
                .mean()
                .alias("avg_daily_unique_cars"),
            ]
        )
        ta = tp.group_by("car_category").agg(
            [
                pl.col("total_unique_trains").sum().alias("total_trains"),
                pl.col("total_car_count").sum().alias("total_cars"),
                pl.col("total_unique_car_ids").sum().alias("total_unique_car_ids"),
            ]
        )
        combined = ma.join(ta, on="car_category", how="outer")
        cat_breakdown = {}
        for cr in combined.iter_rows(named=True):
            cat_breakdown[cr["car_category"]] = {
                "avg_daily_trains": cr.get("avg_daily_trains"),
                "avg_daily_cars": cr.get("avg_daily_cars"),
                "avg_daily_unique_cars": cr.get("avg_daily_unique_cars"),
                "total_trains": cr.get("total_trains", 0),
                "total_cars": cr.get("total_cars", 0),
                "total_unique_car_ids": cr.get("total_unique_car_ids", 0),
            }
        prov_stats[prov] = {
            "sensor_count": len(data["sensors"]),
            "corridor_count": len(set(data["corridors"])),
            "sensors": sorted(list(data["sensors"])),
            "car_category_statistics": cat_breakdown,
        }
    stats["by_province"] = prov_stats

    # --- Loaded vs empty ---
    le_stats = {}
    for cat in car_categories:
        cm = monthly_df.filter(pl.col("car_category") == cat)
        avg = (
            cm.select(
                [
                    pl.col("daily_avg_unique_trains").mean().alias("avg_daily_trains"),
                    pl.col("daily_avg_car_count").mean().alias("avg_daily_cars"),
                    pl.col("daily_avg_unique_car_ids")
                    .mean()
                    .alias("avg_daily_unique_cars"),
                ]
            ).to_dicts()[0]
            if len(cm) > 0
            else {
                "avg_daily_trains": 0,
                "avg_daily_cars": 0,
                "avg_daily_unique_cars": 0,
            }
        )

        ct = totals_df.filter(pl.col("car_category") == cat)
        tots = (
            ct.select(
                [
                    pl.col("total_unique_trains").sum().alias("total_trains"),
                    pl.col("total_car_count").sum().alias("total_cars"),
                    pl.col("total_unique_car_ids").sum().alias("total_unique_car_ids"),
                ]
            ).to_dicts()[0]
            if len(ct) > 0
            else {"total_trains": 0, "total_cars": 0, "total_unique_car_ids": 0}
        )

        le_stats[cat] = {**avg, **tots}
    stats["loaded_vs_empty"] = le_stats

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    logger.info(f"  Sensors: {len(sensors)}, Provinces: {len(prov_stats)}")
    logger.info(f"  → {output_path}")
    return stats


# ---------------------------------------------------------------------------
# Step 4: CSV export
# ---------------------------------------------------------------------------


def step_csv(stats_json_path: str, output_dir: str) -> None:
    """Convert JSON statistics into multiple CSV files."""
    logger.info("=" * 70)
    logger.info("STEP 4: Export to CSV")
    logger.info("=" * 70)

    with open(stats_json_path, "r") as f:
        stats = json.load(f)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1 – Sensor overview
    rows = []
    for s in stats["sensors"]:
        rows.append(
            {
                "sensorId": s["sensorId"],
                "sensor": s["sensor"],
                "province": s["province"],
                "direction": s["direction"],
                "latitude": s.get("latitude"),
                "longitude": s.get("longitude"),
                "country": s.get("country"),
                "region": s.get("region"),
                "timezone": s.get("timezone"),
                "start_date": s["start_date"],
                "end_date": s["end_date"],
                "avg_daily_trains": s["overall_statistics"]["avg_daily_trains"],
                "avg_speed_mph": s["overall_statistics"]["avg_speed"],
                "avg_train_types": s["overall_statistics"]["avg_train_types"],
                "avg_operators": s["overall_statistics"]["avg_operators"],
            }
        )
    _write_csv(out / "01_sensor_overview.csv", rows)

    # 2 – Per-sensor car category
    rows = []
    for s in stats["sensors"]:
        for cat, cs in s["car_category_statistics"].items():
            rows.append(
                {
                    "sensorId": s["sensorId"],
                    "sensor": s["sensor"],
                    "province": s["province"],
                    "direction": s["direction"],
                    "latitude": s.get("latitude"),
                    "longitude": s.get("longitude"),
                    "country": s.get("country"),
                    "region": s.get("region"),
                    "start_date": s["start_date"],
                    "end_date": s["end_date"],
                    "car_category": cat,
                    **cs,
                }
            )
    _write_csv(out / "02_sensor_car_category_details.csv", rows)

    # 3 – Province summary
    rows = []
    for prov, ps in stats["by_province"].items():
        rows.append(
            {
                "province": prov,
                "sensor_count": ps["sensor_count"],
                "corridor_count": ps["corridor_count"],
                "sensors": ", ".join(ps["sensors"]),
            }
        )
    _write_csv(out / "03_province_summary.csv", rows)

    # 4 – Province car category
    rows = []
    for prov, ps in stats["by_province"].items():
        for cat, cs in ps["car_category_statistics"].items():
            rows.append({"province": prov, "car_category": cat, **cs})
    _write_csv(out / "04_province_car_category_stats.csv", rows)

    # 5 – Loaded vs empty
    rows = []
    for cat, cs in stats["loaded_vs_empty"].items():
        parts = cat.rsplit("_", 1)
        rows.append(
            {
                "car_category": cat,
                "car_type": parts[0] if len(parts) > 1 else cat,
                "loaded_status": parts[1] if len(parts) > 1 else "Unknown",
                **cs,
            }
        )
    _write_csv(out / "05_loaded_vs_empty_summary.csv", rows)

    logger.info("  ✓ CSV export complete")


def _write_csv(path: Path, rows: list) -> None:
    df = pl.DataFrame(rows)
    df.write_csv(path)
    logger.info(f"  {path.name}: {len(df)} rows")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_STEPS = ["rollup", "metadata", "stats", "csv"]


def main():
    parser = argparse.ArgumentParser(
        description="Unified corridor workflow for RailState sensor data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        default="/Volumes/LaCie/RailState_data",
        help="Directory containing tc_export_*.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory for output files (default: applications/corridor/results/)",
    )
    parser.add_argument(
        "--sensors-overview",
        default="/Volumes/LaCie/RailState_data/sensors_overview.jsonl",
        help="Path to sensors_overview.jsonl for metadata enrichment",
    )
    parser.add_argument(
        "--all-states",
        action="store_true",
        help="Include all 50 US states (default: border states only)",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=None,
        help="Run only specific steps (default: rollup metadata stats csv)",
    )
    parser.add_argument(
        "--stats-json",
        default=None,
        help="Feed the csv step an existing enhanced-stats JSON instead of the "
        "one produced this run — rebuilds the selection CSVs without rerunning "
        "rollup/metadata/stats (e.g. from a JSON kept locally under "
        "applications/corridor/demo_data/, which is gitignored).",
    )

    args = parser.parse_args()

    steps = set(args.steps) if args.steps else {"rollup", "metadata", "stats", "csv"}

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    suffix = "all_states" if args.all_states else "border_states"
    monthly_path = str(
        Path(args.output_dir) / f"sensor_rollup_monthly_{suffix}.parquet"
    )
    totals_path = str(Path(args.output_dir) / f"sensor_rollup_totals_{suffix}.parquet")
    stats_path = str(
        Path(args.output_dir) / f"sensor_rollup_enhanced_stats_{suffix}.json"
    )
    csv_dir = str(Path(args.output_dir) / f"csv_export_{suffix}")

    # Discover input files
    input_dir = Path(args.input_dir)
    input_files = sorted(str(p) for p in input_dir.glob("tc_export_*.jsonl"))
    if not input_files and "rollup" in steps:
        logger.error(f"No tc_export_*.jsonl files found in {input_dir}")
        sys.exit(1)

    logger.info(
        f"Mode: {'all US states' if args.all_states else 'border states'} + Canada"
    )
    logger.info(f"Steps: {', '.join(sorted(steps))}")
    if input_files:
        logger.info(f"Input files: {len(input_files)}")

    # Execute steps
    if "rollup" in steps:
        step_rollup(input_files, args.output_dir, args.all_states)

    if "metadata" in steps:
        step_metadata(monthly_path, totals_path, args.sensors_overview)

    if "stats" in steps:
        step_stats(monthly_path, totals_path, stats_path)

    if "csv" in steps:
        step_csv(args.stats_json or stats_path, csv_dir)

    logger.info("=" * 70)
    logger.info("Workflow complete!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
