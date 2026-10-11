
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, expr, window, avg, min, max, udf
)
from pyspark.sql.types import StringType

# Project folders
BASE = Path(__file__).resolve().parent
SOURCE = str(BASE / "sensor_source")
OUTPUT = str(BASE / "output")
CHECKPOINT = str(BASE / "checkpoints")

# Start Spark
spark = (
    SparkSession.builder
    .appName("IoT Sensor Monitoring")
    .master("local[2]")
    .config("spark.sql.shuffle.partitions", "2")
    .config("spark.hadoop.fs.defaultFS", "file:///")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


# Temperature classification UDF
def classify_temperature(temp):
    if temp < 35:
        return "NORMAL"
    elif temp <= 40:
        return "WARNING"
    return "CRITICAL"


classify_udf = udf(classify_temperature, StringType())


# CSV schema
schema = """
event_id STRING,
device_id STRING,
location STRING,
event_time STRING,
temperature STRING,
humidity STRING
"""


# Read new CSV files incrementally
raw = (
    spark.readStream
    .format("csv")
    .option("header", "true")
    .option("recursiveFileLookup", "true")
    .option("maxFilesPerTrigger", 1)
    .schema(schema)
    .load(SOURCE)
)


# Parse timestamps and numeric values safely
parsed = (
    raw
    .withColumn(
        "event_time",
        expr("try_to_timestamp(event_time, 'yyyy-MM-dd HH:mm:ss')")
    )
    .withColumn(
        "temperature",
        expr("try_cast(temperature AS DOUBLE)")
    )
    .withColumn(
        "humidity",
        expr("try_cast(humidity AS DOUBLE)")
    )
)


# Keep only valid readings
valid = parsed.filter(
    col("event_id").isNotNull()
    & (col("event_id") != "")
    & col("device_id").isNotNull()
    & (col("device_id") != "")
    & col("location").isNotNull()
    & (col("location") != "")
    & col("event_time").isNotNull()
    & col("temperature").between(-50, 100)
    & col("humidity").between(0, 100)
)


# Event-time watermark and duplicate handling
unique = (
    valid
    .withWatermark("event_time", "10 minutes")
    .dropDuplicatesWithinWatermark(["event_id"])
)


# Classify individual sensor readings
events = unique.withColumn(
    "temperature_status",
    classify_udf(col("temperature"))
)


# Query 1: Save individual events
events_query = (
    events.writeStream
    .queryName("sensor_events")
    .format("parquet")
    .outputMode("append")
    .option("path", OUTPUT + "/sensor_events")
    .option("checkpointLocation", CHECKPOINT + "/events")
    .start()
)


# Calculate 5-minute temperature statistics by location
windows = (
    unique
    .groupBy(
        window(col("event_time"), "5 minutes"),
        col("location")
    )
    .agg(
        avg("temperature").alias("avg_temperature"),
        min("temperature").alias("min_temperature"),
        max("temperature").alias("max_temperature")
    )
    .withColumn(
        "temperature_status",
        classify_udf(col("max_temperature"))
    )
)


# Query 2: Save completed windows to Parquet
windows_query = (
    windows.writeStream
    .queryName("temperature_windows")
    .format("parquet")
    .outputMode("append")
    .option("path", OUTPUT + "/temperature_windows")
    .option("checkpointLocation", CHECKPOINT + "/windows")
    .start()
)


# Query 3: Print completed windows in the terminal
def print_window_report(batch_df, batch_id):
    print(
        f"\n===== TEMPERATURE REPORT | Batch {batch_id} =====",
        flush=True
    )

    if batch_df.isEmpty():
        print("No completed windows in this batch.", flush=True)
    else:
        batch_df.orderBy("window", "location").show(
            truncate=False
        )


report_query = (
    windows.writeStream
    .queryName("terminal_temperature_report")
    .outputMode("append")
    .foreachBatch(print_window_report)
    .option(
        "checkpointLocation",
        CHECKPOINT + "/terminal_report"
    )
    .start()
)


# Startup information
print("\n" + "=" * 55, flush=True)
print("       IOT SENSOR MONITORING STARTED", flush=True)
print("=" * 55, flush=True)
print(f"Input folder: {SOURCE}", flush=True)
print(f"Event output: {OUTPUT}/sensor_events", flush=True)
print(
    f"Window output: {OUTPUT}/temperature_windows",
    flush=True
)
print("Window duration: 5 minutes", flush=True)
print("Watermark: 10 minutes", flush=True)
print("NORMAL: temperature < 35 C", flush=True)
print("WARNING: temperature 35-40 C", flush=True)
print("CRITICAL: temperature > 40 C", flush=True)
print("Monitoring reports: terminal + Parquet", flush=True)
print("Press Ctrl+C to stop.", flush=True)
print("=" * 55, flush=True)


# Keep all streaming queries alive
try:
    spark.streams.awaitAnyTermination()
finally:
    for query in spark.streams.active:
        query.stop()
    spark.stop()

