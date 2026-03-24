"""
Example script to explore the sensor rollup data

This demonstrates how to load and analyze the sensor rollup output.
"""

import polars as pl

# Load the data
data_file = "/Users/hpanneer/Downloads/sensor_rollup_tc_export_2024-09.parquet"
df = pl.read_parquet(data_file)

print("=" * 80)
print("SENSOR ROLLUP DATA SUMMARY")
print("=" * 80)

print(f"\nData shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Number of unique sensors: {df['sensorId'].n_unique()}")
print(
    f"Number of unique sensor+direction combinations: {df.select(['sensorId', 'direction']).unique().shape[0]}"
)

print("\n" + "=" * 80)
print("SENSOR LOCATIONS")
print("=" * 80)
print("\nUnique sensors:")
print(df.select(["sensorId", "sensor"]).unique().sort("sensorId").head(20))

print("\n" + "=" * 80)
print("DIRECTION DISTRIBUTION")
print("=" * 80)
print(df.group_by("direction").agg(pl.count()).sort("count", descending=True))

print("\n" + "=" * 80)
print("TOP 10 BUSIEST SENSORS (by daily average trains)")
print("=" * 80)
busiest = (
    df.select(
        [
            "sensorId",
            "sensor",
            "direction",
            "year",
            "month",
            "daily_avg_total_trains",
            "avg_speed_mph",
            "days_active",
        ]
    )
    .sort("daily_avg_total_trains", descending=True)
    .head(10)
)
print(busiest)

print("\n" + "=" * 80)
print("CAR CATEGORIES FOUND IN DATA")
print("=" * 80)
# Extract unique car categories from column names
cols = df.columns
car_categories = sorted(
    set(
        [
            c.replace("daily_avg_unique_trains_", "")
            for c in cols
            if c.startswith("daily_avg_unique_trains_")
        ]
    )
)
print(f"\nTotal categories: {len(car_categories)}\n")
for cat in car_categories:
    print(f"  - {cat}")

print("\n" + "=" * 80)
print("SAMPLE DATA - First sensor")
print("=" * 80)
sample = df.head(1)
print("\nKey metrics:")
for col in [
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
]:
    print(f"  {col}: {sample[col][0]}")

print("\n" + "=" * 80)
print("LOADED VS EMPTY ANALYSIS")
print("=" * 80)

# Get columns for loaded vs empty for major car types
loaded_cols = [c for c in cols if "_Loaded" in c and "daily_avg_car_count_" in c]
empty_cols = [c for c in cols if "_Empty" in c and "daily_avg_car_count_" in c]

if loaded_cols:
    print(f"\nColumns tracking loaded cars: {len(loaded_cols)}")
    total_loaded = df.select(
        [pl.col(c).fill_null(0).sum() for c in loaded_cols]
    ).sum_horizontal()[0]
    print(f"Total daily avg loaded cars across all sensors: {total_loaded:.2f}")

if empty_cols:
    print(f"\nColumns tracking empty cars: {len(empty_cols)}")
    total_empty = df.select(
        [pl.col(c).fill_null(0).sum() for c in empty_cols]
    ).sum_horizontal()[0]
    print(f"Total daily avg empty cars across all sensors: {total_empty:.2f}")

print("\n" + "=" * 80)
print("EXPORT OPTIONS")
print("=" * 80)
print("""
To export subsets of this data:

# Export to CSV
df.write_csv('sensor_rollup.csv')

# Export specific sensors
df.filter(pl.col('sensor').str.contains('Barstow')).write_csv('barstow_sensors.csv')

# Export specific columns only
df.select(['sensorId', 'sensor', 'direction', 'year', 'month', 
           'daily_avg_total_trains', 'avg_speed_mph']).write_csv('summary.csv')

# Export aggregated by sensor (average across directions and months)
df.group_by('sensorId', 'sensor').agg([
    pl.col('daily_avg_total_trains').mean(),
    pl.col('avg_speed_mph').mean(),
]).write_csv('sensor_averages.csv')
""")

print("\n" + "=" * 80)
print("Done!")
print("=" * 80)
