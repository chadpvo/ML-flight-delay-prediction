# Databricks notebook source
# MAGIC %skip
# MAGIC root = "dbfs:/mnt/mids-w261/"
# MAGIC
# MAGIC # Directories only
# MAGIC def tree_dirs_only(path, indent=""):
# MAGIC     for item in dbutils.fs.ls(path):
# MAGIC         if item.name.endswith("/"):                # only print directories
# MAGIC             name = item.name.rstrip("/")
# MAGIC             print(f"{indent}📁 {name}")
# MAGIC             tree_dirs_only(item.path, indent + "  ")
# MAGIC
# MAGIC tree_dirs_only(root)

# COMMAND ----------

!pip install matplotlib-venn

# COMMAND ----------

# MAGIC %md
# MAGIC ## Standard Imports

# COMMAND ----------

# DBTITLE 1,Standard Imports for Visualization
# PySpark aggregation -> Small Pandas result for plotting
from pyspark.sql.functions import (col, sum as spark_sum, count as spark_count, min as spark_min, max as spark_max, lit)
import pyspark.sql.functions as F
from pyspark.sql.window import Window
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from matplotlib_venn import venn2
from pyspark.sql.types import DoubleType, NumericType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Path Setup

# COMMAND ----------

# DBTITLE 1,Path to full joined dataset
# Base directory
data_BASE_DIR = "dbfs:/student-groups/Group_02_01/"
df_3m_path = f"{data_BASE_DIR}_checkpoints/Airport_Weather__3m"
df_1y_path = f"{data_BASE_DIR}_checkpoints/Airport_Weather__1y"
df_full_path = f"{data_BASE_DIR}_checkpoints/Airport_Weather_"

df_3m = spark.read.parquet(df_3m_path)
df_1y = spark.read.parquet(df_1y_path)
df_full = spark.read.parquet(df_full_path)

# COMMAND ----------

# DBTITLE 1,Path to Graph Features Datasets
blind_test_set_path = f"dbfs:/student-groups/Group_02_01/fasw/processed_train_test/test"
blind_train_set_path = f"dbfs:/student-groups/Group_02_01/fasw/processed_train_test/train"

print(f"Blind Test set path: {blind_test_set_path}")
print(f"Blind Train set path: {blind_train_set_path}")

df_test = spark.read.parquet(blind_test_set_path)
df_train = spark.read.parquet(blind_train_set_path)
df_full_graph = df_train.unionByName(df_test)
display(df_full_graph.limit(10))

# COMMAND ----------

# DBTITLE 1,DFs Comparison
datasets = {
    "df_3m": df_3m,
    "df_1y": df_1y,
    "df_full": df_full
}

results = []

for name, df in datasets.items():

    # Total rows
    total_rows = df.count()

    # Class distribution
    class_1 = df.filter(col("ARR_DEL15") == 1).count()
    class_0 = df.filter(col("ARR_DEL15") == 0).count()

    # Time range
    time_range = (
        df.select(
            spark_min("FL_DATE").alias("start"),
            spark_max("FL_DATE").alias("end")
        ).collect()[0]
    )

    results.append({
        "dataset_name": name,
        "class_0_count": class_0,
        "class_1_count": class_1,
        "total_rows": total_rows,
        "class_1_pct": round(class_1 * 100 / total_rows, 2),
        "start_date": time_range["start"],
        "end_date": time_range["end"]
    })

# Convert to Pandas table for easy display
EDA_summary = pd.DataFrame(results)
display(EDA_summary)

# COMMAND ----------

# DBTITLE 1,Quick 5-yr Check

flight_by_year = (df_full.groupBy("YEAR").agg(spark_count("*").alias("num_flights")).orderBy("YEAR"))
display(flight_by_year)

# COMMAND ----------

# DBTITLE 1,2015-2019 Filter
df_full = df_full.filter((col("YEAR") >= 2015) & (col("YEAR") <= 2019))

# COMMAND ----------

# DBTITLE 1,Recheck Df_full
filtered_year = (df_full.groupBy("YEAR").agg(spark_count("*").alias("num_flights")).orderBy("YEAR"))
display(filtered_year)

# COMMAND ----------

# DBTITLE 1,EDA DF Comparison Post-Filter
datasets = {
    "df_3m": df_3m,
    "df_1y": df_1y,
    "df_full": df_full
}

results = []

for name, df in datasets.items():

    # Total rows
    total_rows = df.count()

    # Class distribution
    class_1 = df.filter(col("ARR_DEL15") == 1).count()
    class_0 = df.filter(col("ARR_DEL15") == 0).count()

    # Time range
    time_range = (
        df.select(
            spark_min("FL_DATE").alias("start"),
            spark_max("FL_DATE").alias("end")
        ).collect()[0]
    )

    results.append({
        "dataset_name": name,
        "class_0_count": class_0,
        "class_1_count": class_1,
        "total_rows": total_rows,
        "class_1_pct": round(class_1 * 100 / total_rows, 2),
        "start_date": time_range["start"],
        "end_date": time_range["end"]
    })

# Convert to Pandas table for easy display
EDA_summary = pd.DataFrame(results)
display(EDA_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC # FILE SIZE CALCULATIONS

# COMMAND ----------

# MAGIC %skip
# MAGIC file_path = "dbfs:/FileStore/airport_codes.csv"
# MAGIC
# MAGIC # Read the data (as you already have)
# MAGIC df_airport_codes = spark.read.csv(file_path, header=True, inferSchema=True)
# MAGIC # display(df_airport_codes)
# MAGIC
# MAGIC # Use dbutils.fs.ls() to get file metadata
# MAGIC # It returns a list of FileInfo objects, which for a single file is a list with one item
# MAGIC file_info_list = dbutils.fs.ls(file_path)
# MAGIC
# MAGIC if file_info_list:
# MAGIC     # Extract the size in bytes from the first (and only) item in the list
# MAGIC     size_bytes = file_info_list[0].size
# MAGIC     size_mb = size_bytes / (1024 * 1024)
# MAGIC
# MAGIC     print(f"The file size in MB is approximately: {size_mb:.4f} MB")
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # EDA on New Joined Data

# COMMAND ----------

# DBTITLE 1,Data Types Summary
display(df_full.dtypes)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Class Distribution Check

# COMMAND ----------

# DBTITLE 1,Class Imbalances Comparison 3m/12m/5y
# Class percentages calculated using ARR_DEL15
summary = {
    "Dataset": ["3-Month", "1-Year", "5-Year (2015-2019)"],
    "Class 0 (Non-Delayed)": [
        df_3m.filter("ARR_DEL15 = 0").count() / df_3m.count() * 100,
        df_1y.filter("ARR_DEL15 = 0").count() / df_1y.count() * 100,
        df_full.filter("ARR_DEL15 = 0").count() / df_full.count() * 100
    ],
    "Class 1 (Delayed)": [
        df_3m.filter("ARR_DEL15 = 1").count() / df_3m.count() * 100,
        df_1y.filter("ARR_DEL15 = 1").count() / df_1y.count() * 100,
        df_full.filter("ARR_DEL15 = 1").count() / df_full.count() * 100
    ]
}
df_plot = pd.DataFrame(summary)
df_melt = df_plot.melt(id_vars="Dataset", var_name="Class", value_name="Percent")
fig = px.bar(
    df_melt,
    x="Dataset",
    y="Percent",
    color="Class",
    barmode="group",
    text="Percent",
    color_discrete_sequence=["steelblue", "lightsteelblue"],
    template="plotly_white",
    title="Class Imbalance Distribution Across Datasets<br><sup>Class labels calculated using ARR_DEL15</sup>"
)

fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig.update_layout(
    yaxis_title="Percentage (%)",
    xaxis_title="Dataset",
    font=dict(color="black"),
    plot_bgcolor="white",
    paper_bgcolor="white",
    height=600
)

fig.show()

# COMMAND ----------

# DBTITLE 1,Null Analysis by Data Tapes & Columns
# Single Spark operation to get null counts
agg_exprs = [spark_sum(col(c).isNull().cast("int")).alias(c) for c in df_1y.columns]
agg_exprs.append(spark_count("*").alias("_total_rows"))  # Fix: use spark_count instead of count

null_counts_row = df_1y.agg(*agg_exprs).collect()[0]
num_rows_test = null_counts_row["_total_rows"]

# Function to simplify data type names
def simplify_dtype(dtype_str):
    """Simplify Spark data type strings for readability"""
    if "ArrayType(StructType" in dtype_str:
        return "Array[Struct]"
    elif "ArrayType(StringType" in dtype_str:
        return "Array[String]"
    elif "ArrayType(IntegerType" in dtype_str:
        return "Array[Integer]"
    elif "ArrayType" in dtype_str:
        return "Array"
    elif "StructType" in dtype_str:
        return "Struct"
    elif "StringType" in dtype_str:
        return "String"
    elif "IntegerType" in dtype_str:
        return "Integer"
    elif "DoubleType" in dtype_str or "FloatType" in dtype_str:
        return "Double"
    elif "TimestampNTZType" in dtype_str:
        return "TimestampNTZ"
    elif "TimestampType" in dtype_str:
        return "Timestamp"
    elif "DateType" in dtype_str:
        return "Date"
    elif "BooleanType" in dtype_str:
        return "Boolean"
    elif "LongType" in dtype_str:
        return "Long"
    elif "BinaryType" in dtype_str:
        return "Binary"
    else:
        # Extract just the type name before "Type"
        return dtype_str.split("Type")[0] if "Type" in dtype_str else dtype_str

# Get data types for each column (simplified)
column_types = {field.name: simplify_dtype(str(field.dataType)) for field in df_1y.schema.fields}

# Build pandas dataframe with column info
null_data = [
    {
        "column": c, 
        "null_count": null_counts_row[c], 
        "null_percent": (null_counts_row[c] / num_rows_test) * 100,
        "data_type": column_types[c]
    } 
    for c in df_1y.columns
]

pdf_null_test = pd.DataFrame(null_data)

# Aggregate by data type
type_summary = pdf_null_test.groupby("data_type").agg({
    "null_count": "sum",
    "column": "count"
}).reset_index()
type_summary.columns = ["data_type", "total_null_count", "num_columns"]
type_summary["null_percent"] = (type_summary["total_null_count"] / num_rows_test) * 100
type_summary = type_summary.sort_values("total_null_count", ascending=False)

# Get top 30 columns
pdf_top30 = pdf_null_test.sort_values("null_count", ascending=False).head(30)

# Create subplots with more spacing
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Top 30 Columns by Null Count", "Null Count by Data Type"),
    horizontal_spacing=0.20,
    specs=[[{"type": "bar"}, {"type": "bar"}]]
)

# Left plot: Top 30 columns
fig.add_trace(
    go.Bar(
        y=pdf_top30["column"][::-1],
        x=pdf_top30["null_count"][::-1],
        orientation="h",
        text=[f"{p:.1f}%" for p in pdf_top30["null_percent"][::-1]],
        textposition="outside",
        marker_color="indianred",
        name="Columns",
        hovertemplate="<b>%{y}</b><br>Null Count: %{x:,}<br>%{text}<extra></extra>",
        cliponaxis=False
    ),
    row=1, col=1
)

# Right plot: By data type (without % in label)
fig.add_trace(
    go.Bar(
        y=type_summary["data_type"][::-1],
        x=type_summary["total_null_count"][::-1],
        orientation="h",
        text=[f"({n} cols)" for n in type_summary["num_columns"][::-1]],
        textposition="outside",
        marker_color="steelblue",
        name="Data Types",
        hovertemplate="<b>%{y}</b><br>Total Null Count: %{x:,}<br>Columns: %{text}<extra></extra>",
        cliponaxis=False
    ),
    row=1, col=2
)

# Update axes with better margins
fig.update_xaxes(title_text="Null Count", row=1, col=1)
fig.update_xaxes(title_text="Total Null Count", row=1, col=2)
fig.update_yaxes(
    title_text="Column", 
    row=1, col=1,
    automargin=True
)
fig.update_yaxes(
    title_text="Data Type", 
    row=1, col=2,
    automargin=True
)

# Update layout
fig.update_layout(
    title_text="Null Value Analysis - 1-Year FASW Dataset",
    height=600,
    width=1800,
    showlegend=False,
    template="plotly_white",
    margin=dict(l=200, r=150, t=80, b=50)
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA on Airlines / Flight Features Family

# COMMAND ----------

# DBTITLE 1,Flights Delay Aggregated by Airlines Table
window_spec = Window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

agg = (
    df_full
    .groupBy("OP_UNIQUE_CARRIER")
    .agg(
        F.count("*").alias("num_flights"),
        F.sum(F.col("ARR_DEL15").cast("int")).alias("num_delayed_flights")
    )
    .withColumn("delay_pct", F.round((F.col("num_delayed_flights") / F.col("num_flights")) * 100, 2))
    .withColumn("pct_total", F.round((F.col("num_flights") / F.sum("num_flights").over(window_spec)) * 100, 2))
    .withColumn("pct_total_delayed", F.round((F.col("num_delayed_flights") / F.sum("num_delayed_flights").over(window_spec)) * 100, 2))
)

display(
    agg.orderBy(F.desc("num_flights"))
    .withColumnRenamed("delay_pct", "Delay %")
    .withColumnRenamed("pct_total", "% of Total Flights")
    .withColumnRenamed("pct_total_delayed", "% of Total Delayed Flights")
)

# COMMAND ----------

# DBTITLE 1,Delay % by Airline Bar Graph
# Convert to pandas
pdf_carrier = agg.toPandas()

# Sort by delay_pct for better visualization
pdf_carrier = pdf_carrier.sort_values("delay_pct", ascending=False)

fig = px.bar(
    pdf_carrier,
    x="OP_UNIQUE_CARRIER",
    y="delay_pct",
    text="delay_pct",
    hover_data=["num_flights", "num_delayed_flights", "pct_total", "pct_total_delayed"],
    title="Delay Percentage by Airline (ARR_DEL15)",
    template="plotly_white"
)

fig.update_traces(
    marker_color="steelblue",
    texttemplate="%{text:.1f}%",
    textangle=0
)

fig.update_layout(
    xaxis_title="Airline",
    yaxis_title="Delay Percentage (%)",
    height=600,
    yaxis=dict(rangemode="tozero"),
    xaxis=dict(tickangle=0)  # Set x-axis tick labels to horizontal
)

fig.show()

# COMMAND ----------

# DBTITLE 1,Aggregated Data Delays
TOP_N = 10

# Process ORIGIN airports
agg_origin = (df_full
    .groupBy("ORIGIN")
    .agg(
        spark_count("*").alias("total_flights"),
        spark_sum(col("ARR_DEL15").cast("int")).alias("delayed_flights")
    )
    .withColumn("delay_pct", (col("delayed_flights") / col("total_flights") * 100))
)

# Process DEST airports
agg_dest = (df_full
    .groupBy("DEST")
    .agg(
        spark_count("*").alias("total_flights"),
        spark_sum(col("ARR_DEL15").cast("int")).alias("delayed_flights")
    )
    .withColumn("delay_pct", (col("delayed_flights") / col("total_flights") * 100))
)

# Convert to pandas
pd_origin = agg_origin.orderBy(col("delayed_flights").desc()).limit(TOP_N).toPandas()
pd_dest = agg_dest.orderBy(col("delayed_flights").desc()).limit(TOP_N).toPandas()

# Sort for better visualization (ascending for horizontal bars)
pd_origin = pd_origin.sort_values("delayed_flights", ascending=True)
pd_dest = pd_dest.sort_values("delayed_flights", ascending=True)

# Create subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(f"Top {TOP_N} ORIGIN Airports by Delayed Flights", 
                    f"Top {TOP_N} DESTINATION Airports by Delayed Flights"),
    horizontal_spacing=0.15
)

# Add ORIGIN trace
fig.add_trace(
    go.Bar(
        x=pd_origin["delayed_flights"],
        y=pd_origin["ORIGIN"],
        orientation="h",
        text=pd_origin["delayed_flights"],
        textposition="inside",
        insidetextanchor="end",
        marker_color="steelblue",
        hovertemplate="<b>%{y}</b><br>" +
                      "Delayed Flights: %{x:,}<br>" +
                      "Total Flights: %{customdata[0]:,}<br>" +
                      "Delay %: %{customdata[1]:.1f}%<extra></extra>",
        customdata=pd_origin[["total_flights", "delay_pct"]].values
    ),
    row=1, col=1
)

# Add DEST trace
fig.add_trace(
    go.Bar(
        x=pd_dest["delayed_flights"],
        y=pd_dest["DEST"],
        orientation="h",
        text=pd_dest["delayed_flights"],
        textposition="inside",
        insidetextanchor="end",
        marker_color="coral",
        hovertemplate="<b>%{y}</b><br>" +
                      "Delayed Flights: %{x:,}<br>" +
                      "Total Flights: %{customdata[0]:,}<br>" +
                      "Delay %: %{customdata[1]:.1f}%<extra></extra>",
        customdata=pd_dest[["total_flights", "delay_pct"]].values
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=500,
    showlegend=False,
    template="plotly_white",
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=12, color="black")
)

# Update axes
fig.update_xaxes(title_text="Number of Delayed Flights", row=1, col=1)
fig.update_xaxes(title_text="Number of Delayed Flights", row=1, col=2)
fig.update_yaxes(title_text="Airport Code", row=1, col=1)
fig.update_yaxes(title_text="Airport Code", row=1, col=2)

fig.show()

# COMMAND ----------

# DBTITLE 1,Avg Distance Total Flights
avg_dist_by_airline = (
    df_full
    .groupBy("OP_UNIQUE_CARRIER")
    .agg(
        F.avg("DISTANCE").alias("avg_distance"),
        F.count("*").alias("total_flights")
    )
    .orderBy(F.col("avg_distance").desc())
)

pdf_avg_dist = avg_dist_by_airline.toPandas()
pdf_avg_dist["avg_distance_rounded"] = pdf_avg_dist["avg_distance"].round(0).astype(int)

fig = go.Figure()

# Bar for average distance (left y-axis)
fig.add_bar(
    x=pdf_avg_dist["OP_UNIQUE_CARRIER"],
    y=pdf_avg_dist["avg_distance"],
    name="Avg Distance",
    marker_color="teal",
    text=pdf_avg_dist["avg_distance_rounded"],
    textposition="outside",
    textfont_size=16,
    yaxis="y1"
)

# Line for total flights (right y-axis)
fig.add_trace(
    go.Scatter(
        x=pdf_avg_dist["OP_UNIQUE_CARRIER"],
        y=pdf_avg_dist["total_flights"],
        name="Total Flights",
        mode="lines+markers",
        marker=dict(color="orange"),
        yaxis="y2"
    )
)

fig.update_layout(
    title="Average Flight Distance and Total Flights by Airline",
    xaxis_title="Airline",
    yaxis=dict(
        title="Average Distance (miles)",
        side="left"
    ),
    yaxis2=dict(
        title="Total Flights",
        overlaying="y",
        side="right"
    ),
    template="plotly_white",
    height=500
)

fig.show()

# COMMAND ----------

# DBTITLE 1,QoQ Delay % by Airline
df_q = (
    df_full
    .withColumn("year", F.year("FL_DATE"))
    .withColumn("quarter", F.quarter("FL_DATE"))
    .groupBy("year", "quarter", "OP_UNIQUE_CARRIER")
    .agg(F.mean("ARR_DEL15").alias("delay_pct"))
    .orderBy("year", "quarter", "OP_UNIQUE_CARRIER")
)

pdf = df_q.toPandas()
pdf["period"] = pdf["year"].astype(str) + " Q" + pdf["quarter"].astype(str)

# Find the row with the highest and lowest delay_pct
max_idx = pdf["delay_pct"].idxmax()
min_idx = pdf["delay_pct"].idxmin()
max_row = pdf.loc[max_idx]
min_row = pdf.loc[min_idx]

fig = go.Figure()

for carrier in pdf["OP_UNIQUE_CARRIER"].unique():
    carrier_df = pdf[pdf["OP_UNIQUE_CARRIER"] == carrier]
    fig.add_trace(go.Scatter(
        x=carrier_df["period"],
        y=carrier_df["delay_pct"] * 100,
        mode="lines+markers",
        name=carrier
    ))

# Add data labels for highest and lowest percentage points
fig.add_trace(go.Scatter(
    x=[max_row["period"], min_row["period"]],
    y=[max_row["delay_pct"] * 100, min_row["delay_pct"] * 100],
    mode="markers+text",
    text=[
        f"{max_row['OP_UNIQUE_CARRIER']} - {max_row['delay_pct']*100:.1f}%",
        f"{min_row['OP_UNIQUE_CARRIER']} - {min_row['delay_pct']*100:.1f}%"
    ],
    textposition="top right",
    marker=dict(color="red", size=12),
    showlegend=False,
    textfont=dict(size=13, color="black")
))

fig.update_layout(
    title="Quarter-Over-Quarter Delay Trend by Airline (2015–2019)",
    xaxis_title="Quarter",
    yaxis_title="Delay Percentage (%)",
    template="plotly_white",
    hovermode="x unified",
    height=600
)

fig.show()

# COMMAND ----------

# DBTITLE 1,Heatmap - Delay % Airline/Airport
# Calculate delay percentage for each airline-airport combination
df_airline_airport_delay = (
    df_full
    .groupBy("OP_UNIQUE_CARRIER", "ORIGIN")
    .agg(
        F.count("*").alias("total_flights"),
        F.sum(F.col("ARR_DEL15").cast("int")).alias("delayed_flights")
    )
    .withColumn("delay_pct", (F.col("delayed_flights") / F.col("total_flights")) * 100)
    .orderBy("OP_UNIQUE_CARRIER", "ORIGIN")
)

# Get top 20 airports by total flights
top_airports_df = (
    df_full.groupBy("ORIGIN")
    .agg(F.count("*").alias("total_flights"))
    .orderBy(F.col("total_flights").desc())
    .limit(20)
)
top_airports = [row["ORIGIN"] for row in top_airports_df.collect()]

# Convert to pandas and pivot
pdf_airline_airport = df_airline_airport_delay.toPandas()
pivot_airline_airport = pdf_airline_airport.pivot(
    index="OP_UNIQUE_CARRIER", 
    columns="ORIGIN", 
    values="delay_pct"
)

# Filter to only top 20 airports by flight count
pivot_filtered = pivot_airline_airport[top_airports]

# Fill NaN with -1 for airports where airline doesn't operate
pivot_filled = pivot_filtered.fillna(-1)

fig = px.imshow(
    pivot_filled,
    labels=dict(x="Origin Airport", y="Airline", color="Delay %"),
    color_continuous_scale="YlOrRd",
    title="Heatmap of Delay Percentage by Airline and Top 20 Origin Airports",
    aspect="auto"
)

fig.update_layout(
    height=700,
    width=1000,  # Wider to accommodate airport codes
    xaxis=dict(tickangle=0)  # Angle airport codes for readability
)

fig.show()

# COMMAND ----------

# DBTITLE 1,Bubble Map Delay Minutes w/ percent delay for ARR_DELAY by Origin Airport
df_airport_delays = (
    df_full
    .groupBy("ORIGIN", "ORIGIN_STATE_ABR", "ORIGIN_SIZE")  # Added ORIGIN_SIZE
    .agg(
        F.count("*").alias("total_flights"),
        F.sum(F.col("ARR_DEL15").cast("int")).alias("delayed_flights"),
        F.avg(F.when(F.col("ARR_DELAY") > 0, F.col("ARR_DELAY"))).alias("avg_delay_minutes")
    )
    .withColumn("delay_pct", (F.col("delayed_flights") / F.col("total_flights")) * 100)
    .filter(F.col("total_flights") >= 50)
    .filter(F.col("avg_delay_minutes").isNotNull())
)

pdf_airport_delays = df_airport_delays.toPandas()

# Clean up airport size values (handle nulls and standardize)
pdf_airport_delays['airport_size'] = pdf_airport_delays['ORIGIN_SIZE'].fillna('Unknown')

# Calculate dynamic ranges based on data with padding
x_min = pdf_airport_delays['avg_delay_minutes'].min()
x_max = pdf_airport_delays['avg_delay_minutes'].max()
y_min = pdf_airport_delays['delay_pct'].min()
y_max = pdf_airport_delays['delay_pct'].max()

# Add padding (10% on each side)
x_padding = (x_max - x_min) * 0.1
y_padding = (y_max - y_min) * 0.1

x_range = [x_min - x_padding, x_max + x_padding]
y_range = [y_min - y_padding, y_max + y_padding]

# Calculate dynamic size based on data range
# Base dimensions with scaling factor
width = max(800, min(1600, int(400 + (x_max - x_min) * 15)))
height = max(600, min(1200, int(400 + (y_max - y_min) * 20)))

# Create the bubble chart with airport size coloring
fig = px.scatter(
    pdf_airport_delays,
    x="avg_delay_minutes",
    y="delay_pct",
    size="total_flights",
    color="airport_size",
    hover_name="ORIGIN",
    hover_data={
        "total_flights": ":,",
        "delayed_flights": ":,",
        "delay_pct": ":.2f",
        "avg_delay_minutes": ":.2f",
        "airport_size": True,
        "ORIGIN_STATE_ABR": True
    },
    title="Airport Delay Analysis: % of Flights Delayed vs. Average Delay Duration",
    labels={
        "avg_delay_minutes": "",  # Remove default x-axis title
        "delay_pct": "% of flights delayed 15 mins+",
        "airport_size": "Airport Size"
    },
    color_discrete_map={
        'large': '#d62728',      # Bold red for large airports
        'medium': '#ff7f0e',     # Bold orange for medium airports
        'small': '#1f77b4',      # Bold blue for small airports
        'Unknown': '#7f7f7f'     # Gray for unknown
    },
    category_orders={"airport_size": ["large", "medium", "small"]},  # Order in legend
    size_max=80
)

# Create list to store all annotations
annotations_list = []

# Add labels for top 15 busiest airports (no background)
top_airports = pdf_airport_delays.nlargest(15, 'total_flights')
for idx, row in top_airports.iterrows():
    annotations_list.append(
        dict(
            x=row['avg_delay_minutes'],
            y=row['delay_pct'],
            text=row['ORIGIN'],
            showarrow=False,
            yshift=15,
            font=dict(size=11, color='white', family='Arial Black'),
            bgcolor='rgba(0,0,0,0)',
            borderpad=0
        )
    )

# Add x-axis title annotation on the right
annotations_list.append(
    dict(
        text="Avg. flight delay in minutes",
        xref="paper",
        yref="paper",
        x=1.0,
        y=-0.08,
        xanchor="right",
        yanchor="top",
        showarrow=False,
        font=dict(size=12)
    )
)

fig.update_layout(
    height=600,  # Dynamic height
    width=800,    # Dynamic width
    xaxis=dict(
        range=x_range,  # Dynamic x range
        gridcolor='lightgray',
        title=""
    ),
    yaxis=dict(
        range=y_range,  # Dynamic y range
        gridcolor='light    gray'
    ),
    plot_bgcolor='white',
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.15,
        xanchor="center",
        x=0.5,
        title="Airport Size"
    ),
    annotations=annotations_list
)

# Make bubbles more defined with borders
fig.update_traces(
    marker=dict(
        line=dict(width=0.9, color='grey'),
        opacity=0.75
    )
)

fig.show()

# COMMAND ----------

# DBTITLE 1,Agg Analysis by Airport Size
# Print summary by airport size
print("\n=== Delay Summary by Airport Size ===")
size_summary = pdf_airport_delays.groupby('airport_size').agg({
    'ORIGIN': 'count',
    'delay_pct': 'mean',
    'avg_delay_minutes': 'mean',
    'total_flights': 'sum'
}).round(2)
size_summary.columns = ['num_airports', 'avg_delay_pct', 'avg_delay_minutes', 'total_flights']
print(size_summary.sort_values('total_flights', ascending=False))

# COMMAND ----------

# DBTITLE 1,Map - Top Delay Routes by Dep Location
# Focus on routes from a specific airport (e.g., ORD)
focus_airport = 'ORD'

df_routes_focused = (
    df_full
    .filter(F.col("CANCELLED") == 0)
    .filter(F.col("ORIGIN") == focus_airport)
    .filter(F.col("ORIGIN_STATE_ABR").isNotNull())  # Domestic only
    .filter(F.col("DEST_STATE_ABR").isNotNull())    # Domestic only
    .groupBy("ORIGIN", "DEST", "ORIGIN_LAT", "ORIGIN_LONG", "DEST_LAT", "DEST_LON")
    .agg(
        F.count("*").alias("total_flights"),
        F.sum(F.col("ARR_DEL15").cast("int")).alias("delayed_flights")
    )
    .withColumn("delay_pct", (F.col("delayed_flights") / F.col("total_flights")) * 100)
    .filter(F.col("total_flights") >= 50)  # Higher threshold for more reliable data
)

pdf_routes_focused = df_routes_focused.toPandas()

# Convert coordinates
pdf_routes_focused['ORIGIN_LAT'] = pd.to_numeric(pdf_routes_focused['ORIGIN_LAT'], errors='coerce')
pdf_routes_focused['ORIGIN_LONG'] = pd.to_numeric(pdf_routes_focused['ORIGIN_LONG'], errors='coerce')
pdf_routes_focused['DEST_LAT'] = pd.to_numeric(pdf_routes_focused['DEST_LAT'], errors='coerce')
pdf_routes_focused['DEST_LON'] = pd.to_numeric(pdf_routes_focused['DEST_LON'], errors='coerce')
pdf_routes_focused = pdf_routes_focused.dropna()

# Filter domestic flights (lat/lon within continental US bounds)
pdf_routes_focused = pdf_routes_focused[
    (pdf_routes_focused['DEST_LAT'] >= 24) & 
    (pdf_routes_focused['DEST_LAT'] <= 50) &
    (pdf_routes_focused['DEST_LON'] >= -125) & 
    (pdf_routes_focused['DEST_LON'] <= -65)
]

# Filter for top destinations with high delay percentage
top_delayed = pdf_routes_focused.nlargest(20, 'delay_pct')

# Function to create subtle curved path between two points
def create_curved_path(lon1, lat1, lon2, lat2, num_points=100):
    """Create a gently curved path similar to great circle routes"""
    lons = []
    lats = []
    
    for i in range(num_points):
        t = i / (num_points - 1)
        
        # Linear interpolation
        lon = lon1 + (lon2 - lon1) * t
        lat = lat1 + (lat2 - lat1) * t
        
        # Add gentle arc (simulating great circle)
        # Peak of arc at midpoint
        arc_height = np.sin(np.pi * t) * 2  # Gentle curve
        lat += arc_height
        
        lons.append(lon)
        lats.append(lat)
    
    return lons, lats

# Create the map
fig_focused = go.Figure()

# Add curved routes from focus airport (only top delayed routes)
for idx, row in top_delayed.iterrows():
    # Create curved path
    lons, lats = create_curved_path(
        row['ORIGIN_LONG'], row['ORIGIN_LAT'],
        row['DEST_LON'], row['DEST_LAT']
    )
    
    # Color based on delay percentage (darker = more delay)
    delay = row['delay_pct']
    if delay < 40:
        color = 'rgba(255, 180, 100, 0.6)'  # Light orange
        width = 1.5
    elif delay < 45:
        color = 'rgba(255, 120, 80, 0.7)'   # Orange
        width = 2
    elif delay < 50:
        color = 'rgba(230, 70, 60, 0.8)'    # Red
        width = 2.5
    else:
        color = 'rgba(180, 30, 50, 0.9)'    # Dark red
        width = 3
    
    fig_focused.add_trace(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='lines',
        line=dict(width=width, color=color),
        hoverinfo='skip',
        showlegend=False
    ))

# Add destination airports
fig_focused.add_trace(go.Scattergeo(
    lon=top_delayed['DEST_LON'],
    lat=top_delayed['DEST_LAT'],
    mode='markers',
    marker=dict(
        size=12,
        color=top_delayed['delay_pct'],
        colorscale=[
            [0, 'rgb(255, 180, 100)'],      # Light orange (lower delay)
            [0.3, 'rgb(255, 120, 80)'],     # Orange
            [0.6, 'rgb(230, 70, 60)'],      # Red
            [1, 'rgb(180, 30, 50)']         # Dark red (higher delay)
        ],
        cmin=top_delayed['delay_pct'].min(),
        cmax=top_delayed['delay_pct'].max(),
        colorbar=dict(
            title="Delay %",
            x=1.02,
            len=0.5
        ),
        line=dict(width=2, color='white')
    ),
    text=top_delayed['DEST'],
    hovertemplate='<b>%{text}</b><br>Delay: %{marker.color:.1f}%<br><extra></extra>',
    showlegend=False
))

# Add origin airport (larger marker)
origin_data = top_delayed.iloc[0]
fig_focused.add_trace(go.Scattergeo(
    lon=[origin_data['ORIGIN_LONG']],
    lat=[origin_data['ORIGIN_LAT']],
    mode='markers+text',
    marker=dict(size=30, color='rgb(120, 20, 40)', line=dict(width=3, color='white')),
    text=[focus_airport],
    textposition='top center',
    textfont=dict(size=18, color='rgb(120, 20, 40)', family='Arial Black'),
    hovertemplate=f'<b>{focus_airport}</b><br>Origin Airport<extra></extra>',
    showlegend=False
))

fig_focused.update_layout(
    title=f'Top High-Delay Routes from {focus_airport} (Domestic Flights)',
    geo=dict(
        scope='usa',
        projection_type='albers usa',
        showland=True,
        landcolor='rgb(230, 230, 230)',
        coastlinecolor='rgb(150, 150, 150)',
        bgcolor='rgba(245, 248, 250, 1)',
        showlakes=True,
        lakecolor='rgb(240, 245, 255)',
        lonaxis=dict(range=[-125, -65]),
        lataxis=dict(range=[24, 50])
    ),
    height=500,
    width=1000,
    margin=dict(l=0, r=0, t=50, b=0),  # Reduced margins
    paper_bgcolor='white'
)

fig_focused.show()

# Print top delayed destinations
print(f"\n=== Top 20 High-Delay Domestic Destinations from {focus_airport} ===")
print(top_delayed[['DEST', 'delay_pct', 'total_flights', 'delayed_flights']].sort_values('delay_pct', ascending=False).to_string())

# COMMAND ----------

# DBTITLE 1,Map - Top 5 Origins w/ most frequent delayed route
# Find top 5 origin airports with most delayed flights (domestic only)
top_airports_df = (
    df_full
    .filter(F.col("CANCELLED") == 0)
    .filter(F.col("ORIGIN_STATE_ABR").isNotNull())
    .filter(F.col("DEST_STATE_ABR").isNotNull())
    .groupBy("ORIGIN", "ORIGIN_LAT", "ORIGIN_LONG")
    .agg(F.sum(F.col("ARR_DEL15").cast("int")).alias("delayed_flights"))
    .orderBy(F.col("delayed_flights").desc())
    .limit(5)
)

top_airports = [row['ORIGIN'] for row in top_airports_df.collect()]

df_routes_focused = (
    df_full
    .filter(F.col("CANCELLED") == 0)
    .filter(F.col("ORIGIN").isin(top_airports))
    .filter(F.col("ORIGIN_STATE_ABR").isNotNull())
    .filter(F.col("DEST_STATE_ABR").isNotNull())
    .groupBy("ORIGIN", "DEST", "ORIGIN_LAT", "ORIGIN_LONG", "DEST_LAT", "DEST_LON")
    .agg(
        F.count("*").alias("total_flights"),
        F.sum(F.col("ARR_DEL15").cast("int")).alias("delayed_flights")
    )
    .withColumn("delay_pct", (F.col("delayed_flights") / F.col("total_flights")) * 100)
    .filter(F.col("total_flights") >= 50)
)

pdf_routes_focused = df_routes_focused.toPandas()

# Convert coordinates
pdf_routes_focused['ORIGIN_LAT'] = pd.to_numeric(pdf_routes_focused['ORIGIN_LAT'], errors='coerce')
pdf_routes_focused['ORIGIN_LONG'] = pd.to_numeric(pdf_routes_focused['ORIGIN_LONG'], errors='coerce')
pdf_routes_focused['DEST_LAT'] = pd.to_numeric(pdf_routes_focused['DEST_LAT'], errors='coerce')
pdf_routes_focused['DEST_LON'] = pd.to_numeric(pdf_routes_focused['DEST_LON'], errors='coerce')
pdf_routes_focused = pdf_routes_focused.dropna()

# Filter domestic flights (lat/lon within continental US bounds)
pdf_routes_focused = pdf_routes_focused[
    (pdf_routes_focused['DEST_LAT'] >= 24) & 
    (pdf_routes_focused['DEST_LAT'] <= 50) &
    (pdf_routes_focused['DEST_LON'] >= -125) & 
    (pdf_routes_focused['DEST_LON'] <= -65)
]

# For each airport, get top 12 delayed routes (expand from 4)
top_delayed_routes = []
for airport in top_airports:
    sub = pdf_routes_focused[pdf_routes_focused['ORIGIN'] == airport]
    topN = sub.nlargest(12, 'delay_pct')
    top_delayed_routes.append(topN)
top_delayed = pd.concat(top_delayed_routes, ignore_index=True)

# Function to create subtle curved path between two points
def create_curved_path(lon1, lat1, lon2, lat2, num_points=100):
    lons = []
    lats = []
    for i in range(num_points):
        t = i / (num_points - 1)
        lon = lon1 + (lon2 - lon1) * t
        lat = lat1 + (lat2 - lat1) * t
        arc_height = np.sin(np.pi * t) * 2
        lat += arc_height
        lons.append(lon)
        lats.append(lat)
    return lons, lats

fig_focused = go.Figure()

# Add curved routes for each airport
for idx, row in top_delayed.iterrows():
    lons, lats = create_curved_path(
        row['ORIGIN_LONG'], row['ORIGIN_LAT'],
        row['DEST_LON'], row['DEST_LAT']
    )
    delay = row['delay_pct']
    if delay < 40:
        color = 'rgba(255, 180, 100, 0.6)'
        width = 1.5
    elif delay < 45:
        color = 'rgba(255, 120, 80, 0.7)'
        width = 2
    elif delay < 50:
        color = 'rgba(230, 70, 60, 0.8)'
        width = 2.5
    else:
        color = 'rgba(180, 30, 50, 0.9)'
        width = 3
    fig_focused.add_trace(go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='lines',
        line=dict(width=width, color=color),
        hoverinfo='skip',
        showlegend=False
    ))

# Add destination airports
fig_focused.add_trace(go.Scattergeo(
    lon=top_delayed['DEST_LON'],
    lat=top_delayed['DEST_LAT'],
    mode='markers',
    marker=dict(
        size=12,
        color=top_delayed['delay_pct'],
        colorscale=[
            [0, 'rgb(255, 180, 100)'],
            [0.3, 'rgb(255, 120, 80)'],
            [0.6, 'rgb(230, 70, 60)'],
            [1, 'rgb(180, 30, 50)']
        ],
        cmin=top_delayed['delay_pct'].min(),
        cmax=top_delayed['delay_pct'].max(),
        colorbar=dict(
            title="Delay %",
            x=1.02,
            len=0.5
        ),
        line=dict(width=2, color='white')
    ),
    text=top_delayed['DEST'],
    hovertemplate='<b>%{text}</b><br>Delay: %{marker.color:.1f}%<br><extra></extra>',
    showlegend=False
))

# Add origin airports (larger marker, one for each)
for airport in top_airports:
    airport_row = pdf_routes_focused[pdf_routes_focused['ORIGIN'] == airport].iloc[0]
    fig_focused.add_trace(go.Scattergeo(
        lon=[airport_row['ORIGIN_LONG']],
        lat=[airport_row['ORIGIN_LAT']],
        mode='markers+text',
        marker=dict(size=30, color='rgb(120, 20, 40)', line=dict(width=3, color='white')),
        text=[airport],
        textposition='top center',
        textfont=dict(size=18, color='rgb(120, 20, 40)', family='Arial Black'),
        hovertemplate=f'<b>{airport}</b><br>Origin Airport<extra></extra>',
        showlegend=False
    ))

fig_focused.update_layout(
    title=f'Top High-Delay Routes from Top 5 Airports (Domestic Flights)',
    geo=dict(
        scope='usa',
        projection_type='albers usa',
        showland=True,
        landcolor='rgb(230, 230, 230)',
        coastlinecolor='rgb(150, 150, 150)',
        bgcolor='rgba(245, 248, 250, 1)',
        showlakes=True,
        lakecolor='rgb(240, 245, 255)',
        lonaxis=dict(range=[-125, -65]),
        lataxis=dict(range=[24, 50])
    ),
    height=600,
    width=1600,  # Expanded width
    margin=dict(l=0, r=0, t=50, b=0),
    paper_bgcolor='white'
)

fig_focused.show()

# Print top delayed destinations for each airport
for airport in top_airports:
    print(f"\n=== Top High-Delay Domestic Destinations from {airport} ===")
    print(top_delayed[top_delayed['ORIGIN'] == airport][['DEST', 'delay_pct', 'total_flights', 'delayed_flights']].sort_values('delay_pct', ascending=False).to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Dual Horizontal Map by Cities
# Calculate delays by origin city
delay_by_origin_city = (
    df_full
    .groupBy("ORIGIN_CITY_NAME")
    .agg(
        F.count("*").alias("total_flights"),
        F.sum(F.col("ARR_DEL15").cast("int")).alias("delayed_flights")
    )
    .withColumn("delay_pct", (F.col("delayed_flights") / F.col("total_flights")) * 100)
)

# Calculate delays by destination city
delay_by_dest_city = (
    df_full
    .groupBy("DEST_CITY_NAME")
    .agg(
        F.count("*").alias("total_flights"),
        F.sum(F.col("ARR_DEL15").cast("int")).alias("delayed_flights")
    )
    .withColumn("delay_pct", (F.col("delayed_flights") / F.col("total_flights")) * 100)
)

# Convert to pandas
pdf_origin = delay_by_origin_city.toPandas()
pdf_dest = delay_by_dest_city.toPandas()

# Rename columns for merging
pdf_origin = pdf_origin.rename(columns={
    'ORIGIN_CITY_NAME': 'city',
    'delayed_flights': 'origin_delayed',
    'total_flights': 'origin_total',
    'delay_pct': 'origin_delay_pct'
})

pdf_dest = pdf_dest.rename(columns={
    'DEST_CITY_NAME': 'city',
    'delayed_flights': 'dest_delayed',
    'total_flights': 'dest_total',
    'delay_pct': 'dest_delay_pct'
})

# Merge on city name
pdf_combined = pdf_origin.merge(pdf_dest, on='city', how='outer').fillna(0)

# Calculate total delays for sorting
pdf_combined['total_delayed'] = pdf_combined['origin_delayed'] + pdf_combined['dest_delayed']

# Get top 20 cities by total delays
TOP_N = 20
pdf_top = pdf_combined.nlargest(TOP_N, 'total_delayed').sort_values('total_delayed', ascending=True)

# Create diverging bar chart
fig = go.Figure()

# Origin delays (left side - negative values)
fig.add_trace(go.Bar(
    y=pdf_top['city'],
    x=-pdf_top['origin_delayed'],  # Negative for left side
    name='Origin Delays',
    orientation='h',
    marker_color='steelblue',
    text=pdf_top['origin_delayed'].astype(int),
    textposition='outside',
    hovertemplate="<b>%{y}</b><br>" +
                  "Origin Delayed: %{text:,}<br>" +
                  "Origin Total: %{customdata[0]:,}<br>" +
                  "Origin Delay %: %{customdata[1]:.1f}%<extra></extra>",
    customdata=pdf_top[['origin_total', 'origin_delay_pct']].values
))

# Destination delays (right side - positive values)
fig.add_trace(go.Bar(
    y=pdf_top['city'],
    x=pdf_top['dest_delayed'],
    name='Destination Delays',
    orientation='h',
    marker_color='coral',
    text=pdf_top['dest_delayed'].astype(int),
    textposition='outside',
    hovertemplate="<b>%{y}</b><br>" +
                  "Dest Delayed: %{x:,}<br>" +
                  "Dest Total: %{customdata[0]:,}<br>" +
                  "Dest Delay %: %{customdata[1]:.1f}%<extra></extra>",
    customdata=pdf_top[['dest_total', 'dest_delay_pct']].values
))

for i, row in pdf_top.iterrows():
    fig.add_annotation(
        x=0,
        y=row['city'],
        text=f"<b>{row['city']}</b>",
        showarrow=False,
        font=dict(size=10, color="black"),
        xanchor="center",
        bgcolor="rgba(255, 255, 255, 0.5)",  # Semi-transparent white
        borderpad=4,
        xref='x',  # Keep in data coordinates
        yref='y'
    )

# Calculate the maximum value for better scaling
max_val = max(pdf_top['origin_delayed'].max(), pdf_top['dest_delayed'].max())
axis_range = max_val * 1.15  # Add 15% padding for text labels

fig.update_layout(
    title=f'Top {TOP_N} Cities by Delayed Flights: Origin (Left) vs Destination (Right)',
    xaxis_title='Number of Delayed Flights',
    yaxis_title='',
    barmode='overlay',
    height=500,
    template='plotly_white',
    xaxis=dict(
        tickformat=',d',
        range=[-axis_range, axis_range]  # Dynamically set based on data
    ),
    yaxis=dict(showticklabels=False),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Add a vertical line at x=0
fig.add_vline(x=0, line_width=2, line_color="black")

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA on Time-Based Features

# COMMAND ----------

# DBTITLE 1,Venn Diagram Analysis
# Look at Delayed_Flights Only
delay_analysis = df_full.select("DEP_DEL15", "ARR_DEL15").filter(col("DEP_DEL15").isNotNull() & col("ARR_DEL15").isNotNull())

# Create crosstab analysis
contingency = (delay_analysis.groupBy("DEP_DEL15", "ARR_DEL15").agg(spark_count("*").alias("count")).orderBy("DEP_DEL15", "ARR_DEL15"))

pd_contingency = contingency.toPandas()
print("\nContingency Table:")
print(pd_contingency)

# Calculate percentages and metrics
total_flights = delay_analysis.count()

# Departure Delayed Filter Only
dep_delayed = delay_analysis.filter(col("DEP_DEL15") == 1).count()
# Within the cohort, how many became delayed on arrival?
dep_delayed_arr_delayed = delay_analysis.filter((col("DEP_DEL15") == 1) & (col("ARR_DEL15") == 1)).count()
# Within the cohort, how many became recovered?
dep_delayed_arr_ontime = delay_analysis.filter((col("DEP_DEL15") == 1) & (col("ARR_DEL15") == 0)).count()

# Flights on time on departure filter
dep_ontime = delay_analysis.filter(col("DEP_DEL15") == 0).count()
# How many became delayed upon arrival within this cohort
dep_ontime_arr_delayed = delay_analysis.filter((col("DEP_DEL15") == 0) & (col("ARR_DEL15") == 1)).count()
# How many remained ontime within this cohort
dep_ontime_arr_ontime = delay_analysis.filter((col("DEP_DEL15") == 0) & (col("ARR_DEL15") == 0)).count()

print(f"Total flights analyzed: {total_flights:,}")
print(f"\nDeparture Delayed (DEP_DEL15=1): {dep_delayed:,} ({dep_delayed/total_flights*100:.2f}%)")
print(f"Stayed delayed (ARR_DEL15=1): {dep_delayed_arr_delayed:,} ({dep_delayed_arr_delayed/dep_delayed*100:.2f}%)")
print(f"Recovered (ARR_DEL15=0): {dep_delayed_arr_ontime:,} ({dep_delayed_arr_ontime/dep_delayed*100:.2f}%)")

print(f"\nDeparture On-Time (DEP_DEL15=0): {dep_ontime:,} ({dep_ontime/total_flights*100:.2f}%)")
print(f"Became delayed (ARR_DEL15=1): {dep_ontime_arr_delayed:,} ({dep_ontime_arr_delayed/dep_ontime*100:.2f}%)")
print(f"Stayed on-time (ARR_DEL15=0): {dep_ontime_arr_ontime:,} ({dep_ontime_arr_ontime/dep_ontime*100:.2f}%)")

# COMMAND ----------

# DBTITLE 1,Venn Diagram - Delays
import matplotlib.pyplot as plt

total_flights = delay_analysis.count()
arr_delayed = delay_analysis.filter(col("ARR_DEL15") == 1).count()

# Intersection: Both delayed
both_delayed = delay_analysis.filter(
    (col("DEP_DEL15") == 1) & (col("ARR_DEL15") == 1)
).count()

# Only departure delayed (recovered)
only_dep_delayed = delay_analysis.filter((col("DEP_DEL15") == 1) & (col("ARR_DEL15") == 0)).count()
# Only arrival delayed (new delays)
only_arr_delayed = delay_analysis.filter((col("DEP_DEL15") == 0) & (col("ARR_DEL15") == 1)).count()
# Neither delayed
neither_delayed = delay_analysis.filter((col("DEP_DEL15") == 0) & (col("ARR_DEL15") == 0)).count()
# Create the Venn diagram
fig, ax = plt.subplots(figsize=(18, 5))  
venn = venn2(
    subsets=(only_dep_delayed, only_arr_delayed, both_delayed),
    set_labels=('Departure Delayed\n(DEP_DEL15=1)', 'Arrival Delayed\n(ARR_DEL15=1)'),
    set_colors=('#ff9999', '#9999ff'),
    alpha=0.4,  
    ax=ax
)

# Customize the labels with counts and percentages
if venn.get_label_by_id('10'):
    venn.get_label_by_id('10').set_text(
        f'{only_dep_delayed:,}\n({only_dep_delayed/total_flights*100:.1f}%)\n\n'
        f'Recovered:\nDeparted late but\narrived on time'
    )
    venn.get_label_by_id('10').set_fontsize(10)

if venn.get_label_by_id('01'):
    venn.get_label_by_id('01').set_text(
        f'{only_arr_delayed:,}\n({only_arr_delayed/total_flights*100:.1f}%)\n\n'
        f'New Delays:\nDeparted on time but\narrived late'
    )
    venn.get_label_by_id('01').set_fontsize(10)

if venn.get_label_by_id('11'):
    venn.get_label_by_id('11').set_text(
        f'{both_delayed:,}\n({both_delayed/total_flights*100:.1f}%)\n\n'
        f'Persistent Delays:\nDelayed on both\ndeparture & arrival'
    )
    venn.get_label_by_id('11').set_fontsize(10)
    venn.get_label_by_id('11').set_color('darkblue')

# Update set labels to include totals
if venn.get_label_by_id('A'):
    venn.get_label_by_id('A').set_text(
        f'Departure Delayed\n'
        f'Total: {dep_delayed:,} ({dep_delayed/total_flights*100:.1f}%)'
    )
    venn.get_label_by_id('A').set_fontsize(12)
    venn.get_label_by_id('A').set_fontweight('bold')

if venn.get_label_by_id('B'):
    venn.get_label_by_id('B').set_text(
        f'Arrival Delayed\n'
        f'Total: {arr_delayed:,} ({arr_delayed/total_flights*100:.1f}%)'
    )
    venn.get_label_by_id('B').set_fontsize(12)
    venn.get_label_by_id('B').set_fontweight('bold')

plt.title('Flight Delay Relationship: Departure vs Arrival Delays',fontsize=16, fontweight='bold')

plt.show()

# COMMAND ----------

# DBTITLE 1,Heatmap - Delay % by Hour/Airline
df_hourly_delay = (
    df_full
    .groupBy("OP_UNIQUE_CARRIER", "CRS_DEP_HOUR")
    .agg(
        F.count("*").alias("total_flights"),
        F.sum(F.col("ARR_DEL15").cast("int")).alias("delayed_flights")
    )
    .withColumn("delay_pct", (F.col("delayed_flights") / F.col("total_flights")) * 100)
    .orderBy("OP_UNIQUE_CARRIER", "CRS_DEP_HOUR")
)

pdf_hourly_delay = df_hourly_delay.toPandas()
pivot_delay = pdf_hourly_delay.pivot(index="OP_UNIQUE_CARRIER", columns="CRS_DEP_HOUR", values="delay_pct")

# Ensure all hours 0-23 are present as columns
all_hours = np.arange(0, 24)
pivot_delay = pivot_delay.reindex(columns=all_hours)

# Replace NaN with a placeholder value, e.g., -1
pivot_delay_filled = pivot_delay.fillna(-1)

fig = px.imshow(
    pivot_delay_filled,
    labels=dict(x="Hour of Day", y="Airline", color="Delay %"),
    color_continuous_scale="YlOrRd",
    title="Heatmap of Delay Percentage by Hour of Day and Airline",
    x=all_hours  # Explicitly set x-axis labels
)

fig.update_layout(
    height=700,
    xaxis=dict(
        tickmode='linear',  # Show every tick
        tick0=0,            # Start at 0
        dtick=1             # Increment by 1
    )
)
fig.show()

# COMMAND ----------

# DBTITLE 1,1. Flight Delay Rate by Departure Hour & Day of Week (% of Flights Delayed > 15mins)
df_hour_dow = (
    df_full
    .groupBy("CRS_DEP_HOUR", "DAY_OF_WEEK")
    .agg(
        F.count("*").alias("total_flights"),
        F.sum(F.col("ARR_DEL15").cast("int")).alias("delayed_flights")
    )
    .withColumn("delay_rate", (F.col("delayed_flights") / F.col("total_flights")) * 100)
)

pdf_hour_dow = df_hour_dow.toPandas()

# Pivot for heatmap (swap axes: hours as rows, days as columns)
pivot_hour_dow = pdf_hour_dow.pivot(index="CRS_DEP_HOUR", columns="DAY_OF_WEEK", values="delay_rate")

# Map day of week numbers to names for columns
day_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 
             5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
pivot_hour_dow.columns = pivot_hour_dow.columns.map(day_names)

# Reorder columns to start with Monday
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pivot_hour_dow = pivot_hour_dow[day_order]

# Fill any missing values
pivot_hour_dow = pivot_hour_dow.fillna(0)

# Create heatmap with swapped axes
fig1 = go.Figure(data=go.Heatmap(
    z=pivot_hour_dow.values,
    x=pivot_hour_dow.columns,
    y=pivot_hour_dow.index,
    colorscale='YlOrRd',
    text=np.round(pivot_hour_dow.values, 1),
    texttemplate='%{text}',
    textfont={"size": 9},
    colorbar=dict(title="Delay Rate (%)"),
    hovertemplate='<b>Hour: %{y}</b><br>Day: %{x}<br>Delay Rate: %{z:.1f}%<extra></extra>'
))

fig1.update_layout(
    title='Flight Delay Rate by Day of Week & Departure Hour (% of Flights Delayed > 15mins)',
    xaxis_title='Day of Week',
    yaxis_title='Departure Hour (24-hour)',
    height=500,
    width=1400,
    xaxis=dict(
        tickmode='array',
        tickvals=day_order,
        ticktext=day_order
    ),
    yaxis=dict(
        tickmode='linear',
        tick0=0,
        dtick=1
    )
)

fig1.show()

# COMMAND ----------

# DBTITLE 1,Polar Bar Chart for Cyclical Week Pattern (Pris)
import plotly.graph_objects as go
import numpy as np

# --- Aggregate by Day of Week ---
df_dow = (
    df_full
    .groupBy("DAY_OF_WEEK")
    .agg(
        F.count("*").alias("total_flights"),
        F.sum(F.col("ARR_DEL15").cast("int")).alias("delayed_flights")
    )
    .withColumn("delay_rate", (F.col("delayed_flights") / F.col("total_flights")) * 100)
)

# Convert to Pandas
pdf_dow = df_dow.toPandas()

# Map day numbers to names
day_names = {
    1: "Monday", 
    2: "Tuesday", 
    3: "Wednesday", 
    4: "Thursday",
    5: "Friday", 
    6: "Saturday", 
    7: "Sunday"
}

pdf_dow["day_name"] = pdf_dow["DAY_OF_WEEK"].map(day_names)

# Order days
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
pdf_dow["day_name"] = pd.Categorical(pdf_dow["day_name"], categories=day_order, ordered=True)

# Sort to preserve order for polar plot
pdf_dow = pdf_dow.sort_values("day_name").reset_index(drop=True)


# theta angles evenly spaced
angles = np.linspace(0, 360, len(pdf_dow), endpoint=False)

fig3 = go.Figure()

# radial bars drawn as scatterpolar "filled" shapes
fig3.add_trace(go.Scatterpolar(
    r=pdf_dow["delay_rate"],
    theta=angles,
    mode="lines+markers+text",
    fill="toself",
    text=np.round(pdf_dow["delay_rate"], 1),
    textposition="top center",
    hovertemplate='<b>%{text}% Delay</b><br>Angle: %{theta}°<extra></extra>',
    marker=dict(size=4),
    line=dict(width=2)
))

# Replace angle ticks with day names
fig3.update_layout(
    title="Day-of-Week Flight Delay Cycle",
    polar=dict(
        angularaxis=dict(
            tickmode="array",
            tickvals=angles,
            ticktext=pdf_dow["day_name"],
            direction="clockwise",
        ),
        radialaxis=dict(title="Delay Rate")
    ),
    showlegend=False,
    height=650,
    width=650
)

fig3.show()


# COMMAND ----------

# DBTITLE 1,Heatmap Polar Chart
fig2 = go.Figure()

fig2.add_trace(go.Barpolar(
    r=pdf_dow["delay_rate"],
    theta=pdf_dow["day_name"],
    hovertemplate='<b>%{theta}</b><br>Delay Rate: %{r:.1f}%<extra></extra>',
    marker=dict(
        colorscale="YlOrRd",
        color=pdf_dow["delay_rate"],
        line=dict(color="black", width=1)
    )
))

fig2.update_layout(
    title="Weekly Flight Delay Cycle (Average Delay Rate by Day of Week)",
    polar=dict(
        radialaxis=dict(title="Delay Rate (%)", tickangle=45, gridcolor="lightgray"),
        angularaxis=dict(direction="clockwise", period=7),
    ),
    height=600,
    width=600,
    showlegend=False
)

fig2.show()


# COMMAND ----------

from pyspark.sql import functions as F

# Aggregate by Day of Week and Hour
df_hour_dow = (
    df_full
    .groupBy("DAY_OF_WEEK", "CRS_DEP_HOUR")
    .agg(
        F.count("*").alias("total_flights"),
        F.sum(F.col("ARR_DEL15").cast("int")).alias("delayed_flights")
    )
    .withColumn("delay_rate", (F.col("delayed_flights") / F.col("total_flights")) * 100)
)

# Convert to Pandas
pdf_hour_dow = df_hour_dow.toPandas()

# Map day numbers to names
day_names = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 
             5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
pdf_hour_dow["day_name"] = pdf_hour_dow["DAY_OF_WEEK"].map(day_names)

# Order days for plotting
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
pdf_hour_dow["day_name"] = pd.Categorical(pdf_hour_dow["day_name"], categories=day_order, ordered=True)

fig = px.bar_polar(
    pdf_hour_dow,
    r="CRS_DEP_HOUR",       # radial axis = hour of day
    theta="day_name",       # angular axis = day of week
    color="delay_rate",     # color = delay rate
    color_continuous_scale="YlOrRd",
    hover_data=["CRS_DEP_HOUR", "day_name", "delay_rate"]
)

fig.update_layout(
    title="Weekly Flight Delay Cycle by Hour (Color = Delay Rate)",
    polar=dict(
        radialaxis=dict(title="Hour of Day", tickmode="array",    # use custom tickvals
            tickvals=[],         # remove all labels
            ticktext=[]),
        angularaxis=dict(direction="clockwise", period=7)
    ),
    height=650,
    width=650
)

fig.show()


# COMMAND ----------

df_month = (
    df_full
    .groupBy("MONTH")
    .agg(
        F.count("*").alias("total_flights"),
        F.sum(F.col("ARR_DEL15").cast("int")).alias("delayed_flights")
    )
    .withColumn("delay_rate", (F.col("delayed_flights") / F.col("total_flights")) * 100)
)

pdf_month = df_month.toPandas()

month_names = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

pdf_month["month_name"] = pdf_month["MONTH"].map(month_names)
pdf_month = pdf_month.sort_values("MONTH")

import numpy as np
import plotly.graph_objects as go

# 12 angles evenly spaced for 12 months
angles = np.linspace(0, 360, 12, endpoint=False)

fig2 = go.Figure()

fig2.add_trace(go.Scatterpolar(
    r=pdf_month["delay_rate"],
    theta=angles,
    mode="lines+markers+text",
    fill="toself",
    text=np.round(pdf_month["delay_rate"], 1),
    textposition="top center",
    hovertemplate='<b>%{text}% Delay</b><br>Month: %{theta}°<extra></extra>',
    marker=dict(size=6),
    line=dict(width=2)
))

fig2.update_layout(
    title="Monthly Seasonal Flight Delay Pattern (Labeled Polar Plot)",
    polar=dict(
        angularaxis=dict(
            tickmode="array",
            tickvals=angles,
            ticktext=pdf_month["month_name"],
            direction="clockwise",
        ),
        radialaxis=dict(title="Delay Rate (%)")
    ),
    showlegend=False,
    height=700,
    width=700
)

fig2.show()



# COMMAND ----------

#Seasonal Decomposition Plot (STL from stats model) -- show seasonal cycles 
import matplotlib.dates as mdates

from statsmodels.tsa.seasonal import STL

#aggregate by daily delay rate
df_daily = (
    df_full
    .groupBy("FL_DATE")
    .agg(
        F.count("*").alias("total_flights"),
        F.sum(F.col("ARR_DEL15").cast("int")).alias("delayed_flights")
    )
    .withColumn("delay_rate", (F.col("delayed_flights") / F.col("total_flights")) * 100)
    .orderBy("FL_DATE")
)

#daily rate
pdf_daily = df_daily.toPandas()
pdf_daily["FL_DATE"] = pd.to_datetime(pdf_daily["FL_DATE"])
pdf_daily = pdf_daily.set_index("FL_DATE")

#monthly
pdf_monthly = pdf_daily["delay_rate"].resample("M").mean()

stl = STL(
    pdf_daily["delay_rate"],
    period=7,            # weekly seasonality
    robust=True          # outlier-resistant (important for weather spikes)
)

result = stl.fit()

# #plot
# fig = result.plot()
# fig.set_size_inches(12, 8)
# plt.suptitle("STL Seasonal Decomposition of Flight Delay Rate", fontsize=16)
# plt.show()

# Extract components
observed = result.observed
trend = result.trend
seasonal = result.seasonal
resid = result.resid

idx = observed.index

# -------------------------------
# Identify extreme values (top/bottom 5%)
# -------------------------------
upper_thresh = observed.quantile(0.95)
lower_thresh = observed.quantile(0.05)
extreme_mask = (observed >= upper_thresh) | (observed <= lower_thresh)

# -------------------------------
# Create custom plotting
# -------------------------------
fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

# ---- Observed ----
axes[0].plot(idx, observed, label="Observed", color="steelblue")
axes[0].scatter(idx[extreme_mask], observed[extreme_mask], s=20, color='red', label='Extreme')
axes[0].set_title("Observed")
axes[0].legend()

# ---- Trend ----
axes[1].plot(idx, trend, color="tab:orange")
axes[1].set_title("Trend")

# ---- Seasonal ----
axes[2].plot(idx, seasonal, color="tab:green")
axes[2].set_title("Seasonal")

# ---- Residual ----
axes[3].plot(idx, resid, color="tab:purple")
axes[3].scatter(idx[extreme_mask], resid[extreme_mask], s=20, color='red', label='Extreme Residuals')
axes[3].set_title("Residual")
axes[3].legend()

# -------------------------------
# Format x-axis as Year-Month
# -------------------------------
for ax in axes:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # major tick every month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # format as YYYY-MM
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')      # rotate labels

plt.suptitle("STL Seasonal Decomposition of Flight Delay Rate", fontsize=16)
plt.tight_layout()
plt.show()


# COMMAND ----------

stl_month = STL(
    pdf_monthly,
    period=12,     # 12 months in a seasonal cycle
    robust=True
)

result_month = stl_month.fit()

import matplotlib.pyplot as plt
import numpy as np

# Define extreme threshold (top 10%)
threshold = pdf_monthly.quantile(0.90)

# Identify extremes
extreme_mask = pdf_monthly > threshold

# Create figure layout
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# ----- 1. Observed with extreme points -----
axes[0].plot(pdf_monthly.index, pdf_monthly.values, label="Observed", color="steelblue")
axes[0].scatter(
    pdf_monthly.index[extreme_mask],
    pdf_monthly[extreme_mask],
    color="red",
    s=60,
    label="Extreme Delay Months"
)
axes[0].set_title("Observed Delay Rate (Monthly)")
axes[0].legend()

# ----- 2. Trend -----
axes[1].plot(result_month.trend, color="darkgreen")
axes[1].set_title("Trend")

# ----- 3. Seasonal -----
axes[2].plot(result_month.seasonal, color="purple")
axes[2].set_title("Seasonality (Annual)")

# ----- 4. Residual -----
axes[3].plot(result_month.resid, color="gray")
axes[3].axhline(0, color="black", linewidth=1)
axes[3].set_title("Residual")

plt.suptitle("STL Seasonal Decomposition (Monthly Flight Delay Rate)", fontsize=16)
plt.tight_layout()
plt.show()


# COMMAND ----------

# DBTITLE 1,Arr Delay by Minutes Distribution (Redrop for Report)
# Arrival Delay Distribution (only for delayed flights)
df_arr_delay_dist = (
    df_full
    .filter(F.col("CANCELLED") == 0)
    .filter(F.col("ARR_DEL15") == 1)  # Only delayed flights
    .filter(F.col("ARR_DELAY").isNotNull())
    .filter(F.col("ARR_DELAY") <= 300)  # Filter extreme outliers
    .select("ARR_DELAY")
    .sample(fraction=0.3, seed=42)  # Sample to reduce data size
)

pdf_arr_delay = df_arr_delay_dist.toPandas()

# Create simple histogram
fig = go.Figure()

fig.add_trace(go.Histogram(
    x=pdf_arr_delay['ARR_DELAY'],
    nbinsx=60,
    marker_color='#FF6B6B',
    marker_line_color='black',
    marker_line_width=1
))

fig.update_layout(
    title="Distribution of Arrival Delays",
    xaxis_title="Arrival Delay (minutes)",
    yaxis_title="Number of Flights",
    height=500,
    width=900,
    plot_bgcolor='white',
    xaxis=dict(gridcolor='lightgray'),
    yaxis=dict(gridcolor='lightgray')
)

fig.show()

# COMMAND ----------

# DBTITLE 1,Flight Status by Day of Month
#make status col
df_status = (
    df_full
    .withColumn(
        "flight_status",
        F.when(F.col("CANCELLED") == 1, "Canceled")
         .when(F.col("ARR_DEL15") == 1, "Delayed")
         .otherwise("On Time")
    )
)

#agg counts by day of month x status
status_by_day = (
    df_status
    .groupBy("DAY_OF_MONTH", "flight_status")
    .agg(F.count("*").alias("num_flights"))
    .orderBy("DAY_OF_MONTH", "flight_status")
)

display(status_by_day)

pdf_status = status_by_day.toPandas()
#pivot so that each status becomes a separate col
fig = px.line(
    pdf_status,
    x="DAY_OF_MONTH",
    y="num_flights",
    color="flight_status",   # separate line per status
    markers=True,
    title="On-Time, Delayed, and Canceled Flights by Day of Month",
    labels={
        "DAY_OF_MONTH": "Day of Month",
        "num_flights": "Number of Flights",
        "flight_status": "Flight Status"
    },
    template="plotly_white"
)

fig.update_layout(
    xaxis=dict(dtick=1),
    height=550
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA on Weather Features Family

# COMMAND ----------

# DBTITLE 1,Distribution of Weather Delay (Redrop for Report)
# Distribution of Arrival-Weather Diff (only for delayed flights)
df_arr_weather_diff = (
    df_full
    .filter(F.col("CANCELLED") == 0)
    .filter(F.col("ARR_DEL15") == 1)  # Only delayed flights
    .filter(F.col("ARR_DELAY").isNotNull())
    .filter(F.col("WEATHER_DELAY").isNotNull())
    .withColumn("arr_weather_diff", F.col("ARR_DELAY") - F.col("WEATHER_DELAY"))
    .filter(F.col("arr_weather_diff").between(-100, 300))  # Filter extreme outliers
    .select("arr_weather_diff")
    .sample(fraction=0.3, seed=42)
)

pdf_diff = df_arr_weather_diff.toPandas()

# Create simple histogram
fig2 = go.Figure()

fig2.add_trace(go.Histogram(
    x=pdf_diff['arr_weather_diff'],
    nbinsx=60,
    marker_color='#FF6B6B',
    marker_line_color='black',
    marker_line_width=1
))

fig2.update_layout(
    title="Distribution of Arrival-Weather Delay Difference",
    xaxis_title="Arrival - Weather Delay (minutes)",
    yaxis_title="Number of Flights",
    height=500,
    width=1000,
    plot_bgcolor='white',
    xaxis=dict(gridcolor='lightgray'),
    yaxis=dict(gridcolor='lightgray')
)

fig2.show()

# COMMAND ----------

# DBTITLE 1,Distribution of Departure-Weather Delay Difference
# Distribution of Departure-Weather Diff (only for delayed flights)
df_dep_weather_diff = (
    df_full
    .filter(F.col("CANCELLED") == 0)
    .filter(F.col("DEP_DEL15") == 1)  # Only delayed flights
    .filter(F.col("DEP_DELAY").isNotNull())
    .filter(F.col("WEATHER_DELAY").isNotNull())
    .withColumn("dep_weather_diff", F.col("DEP_DELAY") - F.col("WEATHER_DELAY"))
    .filter(F.col("dep_weather_diff").between(0, 300))  # Filter extreme outliers
    .select("dep_weather_diff")
    .sample(fraction=0.3, seed=42)
)

pdf_diff = df_dep_weather_diff.toPandas()

# Create simple histogram
fig2 = go.Figure()

fig2.add_trace(go.Histogram(
    x=pdf_diff['dep_weather_diff'],
    nbinsx=60,
    marker_color='#FF6B6B',
    marker_line_color='black',
    marker_line_width=1
))

fig2.update_layout(
    title="Distribution of Departure-Weather Delay Difference",
    xaxis_title="Departure - Weather Delay (minutes)",
    yaxis_title="Number of Flights",
    height=500,
    width=1000,
    plot_bgcolor='white',
    xaxis=dict(gridcolor='lightgray'),
    yaxis=dict(gridcolor='lightgray')
)

fig2.show()

# COMMAND ----------

# DBTITLE 1,Dep/Arr Weather Delay Distribution Combination
# Sample and prepare Departure-Weather difference
df_dep_weather_diff = (
    df_full
    .filter(F.col("CANCELLED") == 0)
    .filter(F.col("DEP_DEL15") == 1)
    .filter(F.col("DEP_DELAY").isNotNull())
    .filter(F.col("WEATHER_DELAY").isNotNull())
    .withColumn("dep_weather_diff", F.col("DEP_DELAY") - F.col("WEATHER_DELAY"))
    .filter(F.col("dep_weather_diff").between(0, 300))
    .select("dep_weather_diff")
    .sample(fraction=0.3, seed=42)
)

pdf_dep = df_dep_weather_diff.toPandas()

# Sample and prepare Arrival-Weather difference
df_arr_weather_diff = (
    df_full
    .filter(F.col("CANCELLED") == 0)
    .filter(F.col("ARR_DEL15") == 1)
    .filter(F.col("ARR_DELAY").isNotNull())
    .filter(F.col("WEATHER_DELAY").isNotNull())
    .withColumn("arr_weather_diff", F.col("ARR_DELAY") - F.col("WEATHER_DELAY"))
    .filter(F.col("arr_weather_diff").between(-100, 300))
    .select("arr_weather_diff")
    .sample(fraction=0.3, seed=42)
)

pdf_arr = df_arr_weather_diff.toPandas()

# Create side-by-side histograms
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Departure - Weather Delay Difference", "Arrival - Weather Delay Difference")
)

# Departure histogram
fig.add_trace(go.Histogram(
    x=pdf_dep['dep_weather_diff'],
    nbinsx=60,
    marker_color='#FF6B6B',
    marker_line_color='black',
    marker_line_width=1
), row=1, col=1)

# Arrival histogram
fig.add_trace(go.Histogram(
    x=pdf_arr['arr_weather_diff'],
    nbinsx=60,
    marker_color='#4E79A7',
    marker_line_color='black',
    marker_line_width=1
), row=1, col=2)

# Set fixed y-axis max to 100K
fig.update_layout(
    height=500,
    width=1600,
    plot_bgcolor='white',
    yaxis=dict(range=[0, 100000], gridcolor='lightgray'),
    yaxis2=dict(range=[0, 100000], gridcolor='lightgray'),
    xaxis=dict(gridcolor='lightgray'),
    xaxis2=dict(gridcolor='lightgray'),
    showlegend=False
)

fig.show()

# COMMAND ----------

# Create precipitation indicator
df_weather = (
    df_full.withColumn(
        "ANY_PRECIP",
        (F.col("rain") + F.col("drizzle") + F.col("snow") + 
         F.col("ice_pellets") + F.col("hail") + F.col("freezing_rain") +
         F.col("rain_showers") + F.col("snow_showers")) > 0
    )
)

# Calculate delay rate
precip_stats = (
    df_weather.groupBy("ANY_PRECIP")
    .agg(F.mean("ARR_DEL15").alias("pct_delayed"))
    .orderBy("ANY_PRECIP")
).toPandas()

import plotly.express as px
fig = px.bar(
    precip_stats,
    x="ANY_PRECIP",
    y="pct_delayed",
    title="Delay Rates: Flights With vs Without Precipitation",
    labels={"ANY_PRECIP": "Precipitation?", "pct_delayed": "Pct Delayed (ARR_DEL15)"},
    text="pct_delayed"
)
fig.show()

# COMMAND ----------

# DBTITLE 1,Precipitation and Cloud Ceilings
# --- Precipitation ---
df_weather = (
    df_full.withColumn(
        "ANY_PRECIP",
        (F.col("rain") + F.col("drizzle") + F.col("snow") + 
         F.col("ice_pellets") + F.col("hail") + F.col("freezing_rain") +
         F.col("rain_showers") + F.col("snow_showers")) > 0
    )
)

precip_stats = (
    df_weather.groupBy("ANY_PRECIP")
    .agg(F.mean("ARR_DEL15").alias("pct_delayed"))
    .orderBy("ANY_PRECIP")
).toPandas()

# --- Ceiling ---
ceiling_order = ["<500 ft (Very low)", "500–1000 ft", "1000–2000 ft", ">2000 ft"]
df_ceiling = df_full.withColumn(
    "CEILING_BIN",
    F.when(F.col("Ceiling_ft_agl") < 500, "<500 ft (Very low)")
     .when(F.col("Ceiling_ft_agl") < 1000, "500–1000 ft")
     .when(F.col("Ceiling_ft_agl") < 2000, "1000–2000 ft")
     .otherwise(">2000 ft")
)

ceiling_stats = (
    df_ceiling.groupBy("CEILING_BIN")
    .agg(F.mean("ARR_DEL15").alias("pct_delayed"))
    .toPandas()
)
ceiling_stats["CEILING_BIN"] = pd.Categorical(ceiling_stats["CEILING_BIN"], categories=ceiling_order, ordered=True)
ceiling_stats = ceiling_stats.sort_values("CEILING_BIN")

# --- Combine 2 charts ---
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Delay Rates: Flights With vs Without Precipitation",
                    "Arrival Delay Percentage by Ceiling Height")
)

# Precipitation chart
fig.add_trace(
    go.Bar(
        x=[False, True],
        y=precip_stats.sort_values("ANY_PRECIP")["pct_delayed"],
        text=precip_stats.sort_values("ANY_PRECIP")["pct_delayed"].round(2),
        textposition="outside",
        name="Precipitation",
        marker_color="orange"
    ),
    row=1, col=1
)

# Ceiling chart
fig.add_trace(
    go.Bar(
        x=ceiling_order,
        y=ceiling_stats["pct_delayed"],
        text=ceiling_stats["pct_delayed"].round(2),
        textposition="outside",
        name="Ceiling",
        marker_color="green"
    ),
    row=1, col=2
)

# Update layout
fig.update_layout(
    showlegend=False,
    height=600,
    width=1700,
    yaxis=dict(title="Pct Delayed (ARR_DEL15)"),
    yaxis2=dict(title="Pct Delayed (ARR_DEL15)")
)

fig.show()

# COMMAND ----------

df_weather = df_full.withColumn("precip_amt", F.coalesce(F.col("HourlyPrecipitation").cast("double"), F.lit(0.0))) \
                    .withColumn("precip_flag", F.when(F.col("precip_amt") > 0, 1).otherwise(0))

agg = (
    df_weather
    .filter((F.col("LATITUDE").isNotNull()) & (F.col("LONGITUDE").isNotNull()))
    .groupBy("ORIGIN", "NAME", "LATITUDE", "LONGITUDE")
    .agg(
        F.count("*").alias("obs_count"),
        F.mean("precip_amt").alias("avg_precip"),
        F.sum("precip_flag").alias("n_precip")
    )
    .withColumn("pct_precip", F.col("n_precip") / F.col("obs_count") * 100.0)
)

pd_agg = agg.toPandas()
pd_agg = pd_agg.dropna(subset=["LATITUDE", "LONGITUDE"])
pd_agg["lat"] = pd_agg["LATITUDE"].astype(float)
pd_agg["lon"] = pd_agg["LONGITUDE"].astype(float)
pd_agg["avg_precip"] = pd_agg["avg_precip"].astype(float)
pd_agg["pct_precip"] = pd_agg["pct_precip"].astype(float)

# optionally filter stations with small obs_count
pd_agg = pd_agg[pd_agg["obs_count"] >= 50]  # adjust threshold
fig = px.scatter_mapbox(
    pd_agg,
    lat="lat",
    lon="lon",
    size="avg_precip",               
    color="pct_precip",              
    hover_name="NAME",
    hover_data={"avg_precip":":.2f", "pct_precip":":.1f", "obs_count":True},
    size_max=20,
    zoom=3,
    mapbox_style="open-street-map",
    title="Station-level Precipitation: Avg Precip (size) and % Observations with Precip (color)",
    range_color=[0, 20]
)
fig.update_layout(height=600,width=1000)
fig.show()

# COMMAND ----------

# DBTITLE 1,Windspeed
df_corr = (
    df_full
    .groupBy("ORIGIN", "ORIGIN_SIZE")
    .agg(
        F.mean("HourlyWindSpeed").alias("avg_wind_speed"),
        F.mean("ARR_DEL15").alias("delay_rate")
    )
    .dropna(subset=["avg_wind_speed", "delay_rate", "ORIGIN_SIZE"])
)

pdf_corr = df_corr.toPandas()
pdf_corr["delay_rate"] = (pdf_corr["delay_rate"] * 100).clip(upper=60)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Arrival Delay Percentage by Wind Speed",
        "Correlation Between Airport Size, Wind Speed, and Delay Rate"
    ),
    column_widths=[0.55, 0.45],
    horizontal_spacing=0.08
)

# Scatter plot: wind speed vs delay rate by airport size
fig.add_trace(
    go.Scatter(
        x=pdf_corr["avg_wind_speed"],
        y=pdf_corr["delay_rate"],
        mode="markers",
        marker=dict(size=10, opacity=0.7),
        text=pdf_corr["ORIGIN"],
        hovertemplate="<b>%{text}</b><br>Avg Wind Speed: %{x:.1f} mph<br>Delay Rate: %{y:.1f}%<extra></extra>",
        showlegend=False
    ),
    row=1, col=2
)

# Color by ORIGIN_SIZE
for size in pdf_corr["ORIGIN_SIZE"].unique():
    mask = pdf_corr["ORIGIN_SIZE"] == size
    fig.add_trace(
        go.Scatter(
            x=pdf_corr.loc[mask, "avg_wind_speed"],
            y=pdf_corr.loc[mask, "delay_rate"],
            mode="markers",
            marker=dict(size=10, opacity=0.7),
            name=str(size),
            text=pdf_corr.loc[mask, "ORIGIN"],
            hovertemplate="<b>%{text}</b><br>Avg Wind Speed: %{x:.1f} mph<br>Delay Rate: %{y:.1f}%<extra></extra>"
        ),
        row=1, col=2
    )

# ----- CORRECT LABELS FOR SUBPLOT 1 (left) -----
fig.update_xaxes(title_text="Wind Speed Bin", row=1, col=1)
fig.update_yaxes(title_text="Pct Delayed (%)", row=1, col=1)

# Wind speed bin analysis
df_wind = df_full.withColumn(
    "WIND_BIN",
    F.when(F.col("HourlyWindSpeed") < 5, "0–5 mph")
     .when(F.col("HourlyWindSpeed") < 15, "5–15 mph")
     .when(F.col("HourlyWindSpeed") < 25, "15–25 mph")
     .otherwise("25+ mph")
)

wind_stats = (
    df_wind.groupBy("WIND_BIN")
    .agg(F.mean("ARR_DEL15").alias("pct_delayed"))
    .toPandas()
)

bin_order = ["0–5 mph", "5–15 mph", "15–25 mph", "25+ mph"]
wind_stats["WIND_BIN"] = pd.Categorical(wind_stats["WIND_BIN"], categories=bin_order, ordered=True)
wind_stats = wind_stats.sort_values("WIND_BIN")

fig.add_trace(
    go.Bar(
        x=wind_stats["WIND_BIN"],
        y=wind_stats["pct_delayed"] * 100,
        marker_color="steelblue",
        text=(wind_stats["pct_delayed"] * 100).round(1),
        textposition="outside",
        showlegend=False
    ),
    row=1, col=1
)

# ----- CORRECT LABELS FOR SUBPLOT 2 (right) -----
fig.update_xaxes(title_text="Avg Wind Speed (mph)", row=1, col=2)
fig.update_yaxes(title_text="Delay Rate (%)", row=1, col=2)

fig.update_layout(
    height=600,
    width=1700,
    template="plotly_white"
)

fig.show()

# COMMAND ----------

df_corr = (
    df_full
    .groupBy("ORIGIN", "ORIGIN_SIZE")
    .agg(
        F.mean("HourlyWindSpeed").alias("avg_wind_speed"),
        F.mean("ARR_DEL15").alias("delay_rate")
    )
    .dropna(subset=["avg_wind_speed", "delay_rate", "ORIGIN_SIZE"])
)

pdf_corr = df_corr.toPandas()
pdf_corr["delay_rate"] = (pdf_corr["delay_rate"] * 100).clip(upper=60)

# Wind speed bin analysis
df_wind = df_full.withColumn(
    "WIND_BIN",
    F.when(F.col("HourlyWindSpeed") < 5, "0–5 mph")
     .when(F.col("HourlyWindSpeed") < 15, "5–15 mph")
     .when(F.col("HourlyWindSpeed") < 25, "15–25 mph")
     .otherwise("25+ mph")
)

wind_stats = (
    df_wind.groupBy("WIND_BIN")
    .agg(F.mean("ARR_DEL15").alias("pct_delayed"))
    .toPandas()
)

bin_order = ["0–5 mph", "5–15 mph", "15–25 mph", "25+ mph"]
wind_stats["WIND_BIN"] = pd.Categorical(wind_stats["WIND_BIN"], categories=bin_order, ordered=True)
wind_stats = wind_stats.sort_values("WIND_BIN")

# Wind speed by hour of day
df_wind_hour = (
    df_full
    .withColumn("DEP_HOUR", (F.col("CRS_DEP_TIME") / 100).cast("int"))
    .groupBy("DEP_HOUR")
    .agg(F.mean("HourlyWindSpeed").alias("avg_wind_speed"))
    .orderBy("DEP_HOUR")
)
pdf_wind_hour = df_wind_hour.toPandas()

# Wind speed by month
df_wind_month = (
    df_full
    .groupBy("MONTH")
    .agg(F.mean("HourlyWindSpeed").alias("avg_wind_speed"))
    .orderBy("MONTH")
)
pdf_wind_month = df_wind_month.toPandas()

# Adjustable font sizes
title_font_size = 11
axis_title_font_size = 10
axis_tick_font_size = 10
data_label_font_size = 9

fig = make_subplots(
    rows=1, cols=4,
    subplot_titles=(
        "Arrival Delay Percentage by Wind Speed",
        "Avg Wind Speed by Hour of Day",
        "Avg Wind Speed by Month",
        "Correlation - Airport Size, Wind Speed, and Delay %"
    ),
    column_widths=[0.25, 0.25, 0.25, 0.25],
    horizontal_spacing=0.06
)

# Bar chart: wind speed bin vs delay rate
fig.add_trace(
    go.Bar(
        x=wind_stats["WIND_BIN"],
        y=wind_stats["pct_delayed"] * 100,
        marker_color="steelblue",
        text=(wind_stats["pct_delayed"] * 100).round(1),
        textposition="outside",
        textfont_size=data_label_font_size,
        showlegend=False
    ),
    row=1, col=1
)

# Line chart: windspeed by hour of day
fig.add_trace(
    go.Scatter(
        x=pdf_wind_hour["DEP_HOUR"],
        y=pdf_wind_hour["avg_wind_speed"],
        mode="lines+markers",
        marker=dict(color="darkcyan", size=7),
        line=dict(width=2, color="darkcyan"),
        name="Avg Wind Speed by Hour",
        showlegend=False
    ),
    row=1, col=2
)

# Line chart: windspeed by month
fig.add_trace(
    go.Scatter(
        x=pdf_wind_month["MONTH"],
        y=pdf_wind_month["avg_wind_speed"],
        mode="lines+markers",
        marker=dict(color="darkorange", size=7),
        line=dict(width=2, color="darkorange"),
        name="Avg Wind Speed by Month",
        showlegend=False
    ),
    row=1, col=3
)

# Scatter plot: wind speed vs delay rate by airport size (moved to col=4)
fig.add_trace(
    go.Scatter(
        x=pdf_corr["avg_wind_speed"],
        y=pdf_corr["delay_rate"],
        mode="markers",
        marker=dict(size=10, opacity=0.7),
        text=pdf_corr["ORIGIN"],
        hovertemplate="<b>%{text}</b><br>Avg Wind Speed: %{x:.1f} mph<br>Delay Rate: %{y:.1f}%<extra></extra>",
        showlegend=False
    ),
    row=1, col=4
)
for size in pdf_corr["ORIGIN_SIZE"].unique():
    mask = pdf_corr["ORIGIN_SIZE"] == size
    fig.add_trace(
        go.Scatter(
            x=pdf_corr.loc[mask, "avg_wind_speed"],
            y=pdf_corr.loc[mask, "delay_rate"],
            mode="markers",
            marker=dict(size=10, opacity=0.7),
            name=str(size),
            text=pdf_corr.loc[mask, "ORIGIN"],
            hovertemplate="<b>%{text}</b><br>Avg Wind Speed: %{x:.1f} mph<br>Delay Rate: %{y:.1f}%<extra></extra>",
            textfont=dict(size=data_label_font_size)
        ),
        row=1, col=4
    )

fig.update_xaxes(title_text="Wind Speed Bin", title_font=dict(size=axis_title_font_size), tickfont=dict(size=axis_tick_font_size), row=1, col=1)
fig.update_yaxes(title_text="Pct Delayed (%)", title_font=dict(size=axis_title_font_size), tickfont=dict(size=axis_tick_font_size), row=1, col=1)
fig.update_xaxes(title_text="Hour of Day", title_font=dict(size=axis_title_font_size), tickfont=dict(size=axis_tick_font_size), row=1, col=2)
fig.update_yaxes(title_text="Avg Wind Speed (mph)", title_font=dict(size=axis_title_font_size), tickfont=dict(size=axis_tick_font_size), row=1, col=2)
fig.update_xaxes(title_text="Month", title_font=dict(size=axis_title_font_size), tickfont=dict(size=axis_tick_font_size), row=1, col=3)
fig.update_yaxes(title_text="Avg Wind Speed (mph)", title_font=dict(size=axis_title_font_size), tickfont=dict(size=axis_tick_font_size), row=1, col=3)
fig.update_xaxes(title_text="Avg Wind Speed (mph)", title_font=dict(size=axis_title_font_size), tickfont=dict(size=axis_tick_font_size), row=1, col=4)
fig.update_yaxes(title_text="Delay Rate (%)", title_font=dict(size=axis_title_font_size), tickfont=dict(size=axis_tick_font_size), row=1, col=4)

fig.update_layout(
    height=500,
    template="plotly_white",
    font=dict(size=axis_tick_font_size),
    title_font=dict(size=title_font_size),
    title=dict(font=dict(size=title_font_size)),
    legend=dict(
        orientation="v",
        yanchor="top",
        y=1,
        xanchor="right",
        x=1
    )
)

fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Miscellaneous EDA

# COMMAND ----------

delay_cols = [
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY"
]


def build_delay_combo_expr():
    exprs = []
    for col in delay_cols:
        exprs.append(
            F.when(F.col(col) > 0, col.replace("_DELAY","").replace("_", " ").title())
        )
    return exprs

df_delay_combos = (
    df_1y
    .withColumn("delay_combo_list", F.array(*build_delay_combo_expr()))
    .withColumn("delay_combo", F.concat_ws(" & ", F.expr("filter(delay_combo_list, x -> x is not null)")))
    .drop("delay_combo_list")
)
df_delay_combos = df_delay_combos.filter(F.col("delay_combo") != "")

combo_counts = (
    df_delay_combos
    .groupBy("delay_combo")
    .agg(F.count("*").alias("num_flights"))
    .orderBy(F.col("num_flights").desc())
)

pdf_combo = combo_counts.toPandas()
pdf_combo



# COMMAND ----------

fig = px.bar(
    pdf_combo,
    x="delay_combo",
    y="num_flights",
    text="num_flights",
    color_discrete_sequence=["steelblue"],
    title="Count of Flights by Exclusive Delay Type Combinations"
)

fig.update_traces(
    textposition="outside",
    textfont_size=12,
    texttemplate="%{text:,}"  # Add data label formatting
)
fig.update_layout(
    xaxis_title="Delay Combination",
    yaxis_title="Number of Flights",
    xaxis_tickangle=45,
    height=600
)
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ###On-time, Delayed, and Canceled flights

# COMMAND ----------

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pyspark.sql import functions as F

# ============== PART 1: Average Distance by Airline ==============
avg_dist_by_airline = (
    df_1y
    .groupBy("OP_UNIQUE_CARRIER")
    .agg(
        F.avg("DISTANCE").alias("avg_distance"),
        F.count("*").alias("total_flights")
    )
    .orderBy(F.col("avg_distance").desc())
)

pdf_avg_dist = avg_dist_by_airline.toPandas()
pdf_avg_dist["avg_distance_rounded"] = pdf_avg_dist["avg_distance"].round(0).astype(int)

# ============== PART 2: Flight Routes Map ==============
# Find top 5 origin airports with most delayed flights (domestic only)
top_airports_df = (
    df_1y
    .filter(F.col("CANCELLED") == 0)
    .filter(F.col("ORIGIN_STATE_ABR").isNotNull())
    .filter(F.col("DEST_STATE_ABR").isNotNull())
    .groupBy("ORIGIN", "ORIGIN_LAT", "ORIGIN_LONG")
    .agg(F.sum(F.col("ARR_DEL15").cast("int")).alias("delayed_flights"))
    .orderBy(F.col("delayed_flights").desc())
    .limit(5)
)

top_airports = [row['ORIGIN'] for row in top_airports_df.collect()]

df_routes_focused = (
    df_1y
    .filter(F.col("CANCELLED") == 0)
    .filter(F.col("ORIGIN").isin(top_airports))
    .filter(F.col("ORIGIN_STATE_ABR").isNotNull())
    .filter(F.col("DEST_STATE_ABR").isNotNull())
    .groupBy("ORIGIN", "DEST", "ORIGIN_LAT", "ORIGIN_LONG", "DEST_LAT", "DEST_LON")
    .agg(
        F.count("*").alias("total_flights"),
        F.sum(F.col("ARR_DEL15").cast("int")).alias("delayed_flights")
    )
    .withColumn("delay_pct", (F.col("delayed_flights") / F.col("total_flights")) * 100)
    .filter(F.col("total_flights") >= 50)
)

pdf_routes_focused = df_routes_focused.toPandas()

# Convert coordinates
pdf_routes_focused['ORIGIN_LAT'] = pd.to_numeric(pdf_routes_focused['ORIGIN_LAT'], errors='coerce')
pdf_routes_focused['ORIGIN_LONG'] = pd.to_numeric(pdf_routes_focused['ORIGIN_LONG'], errors='coerce')
pdf_routes_focused['DEST_LAT'] = pd.to_numeric(pdf_routes_focused['DEST_LAT'], errors='coerce')
pdf_routes_focused['DEST_LON'] = pd.to_numeric(pdf_routes_focused['DEST_LON'], errors='coerce')
pdf_routes_focused = pdf_routes_focused.dropna()

# Filter domestic flights (lat/lon within continental US bounds)
pdf_routes_focused = pdf_routes_focused[
    (pdf_routes_focused['DEST_LAT'] >= 24) & 
    (pdf_routes_focused['DEST_LAT'] <= 50) &
    (pdf_routes_focused['DEST_LON'] >= -125) & 
    (pdf_routes_focused['DEST_LON'] <= -65)
]

# For each airport, get top 12 delayed routes
top_delayed_routes = []
for airport in top_airports:
    sub = pdf_routes_focused[pdf_routes_focused['ORIGIN'] == airport]
    topN = sub.nlargest(12, 'delay_pct')
    top_delayed_routes.append(topN)
top_delayed = pd.concat(top_delayed_routes, ignore_index=True)

# Function to create subtle curved path between two points
def create_curved_path(lon1, lat1, lon2, lat2, num_points=100):
    lons = []
    lats = []
    for i in range(num_points):
        t = i / (num_points - 1)
        lon = lon1 + (lon2 - lon1) * t
        lat = lat1 + (lat2 - lat1) * t
        arc_height = np.sin(np.pi * t) * 2
        lat += arc_height
        lons.append(lon)
        lats.append(lat)
    return lons, lats

# ============== CREATE COMBINED FIGURE ==============
fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "xy", "secondary_y": True}, {"type": "geo"}]],
    subplot_titles=("Average Flight Distance and Total Flights by Airline", 
                    "Top High-Delay Routes from Top 5 Airports (Domestic Flights)"),
    column_widths=[0.45, 0.55],
    horizontal_spacing=0.03
)

# Add bar chart for average distance
fig.add_trace(
    go.Bar(
        x=pdf_avg_dist["OP_UNIQUE_CARRIER"],
        y=pdf_avg_dist["avg_distance"],
        name="Avg Distance",
        marker_color="teal",
        text=pdf_avg_dist["avg_distance_rounded"],
        textposition="outside",
        textfont_size=16,
        showlegend=True
    ),
    row=1, col=1,
    secondary_y=False
)

# Add line for total flights
fig.add_trace(
    go.Scatter(
        x=pdf_avg_dist["OP_UNIQUE_CARRIER"],
        y=pdf_avg_dist["total_flights"],
        name="Total Flights",
        mode="lines+markers",
        marker=dict(color="orange", size=8),
        line=dict(width=2),
        showlegend=True
    ),
    row=1, col=1,
    secondary_y=True
)

# Add curved routes to map
for idx, row in top_delayed.iterrows():
    lons, lats = create_curved_path(
        row['ORIGIN_LONG'], row['ORIGIN_LAT'],
        row['DEST_LON'], row['DEST_LAT']
    )
    delay = row['delay_pct']
    if delay < 40:
        color = 'rgba(255, 180, 100, 0.6)'
        width = 1.5
    elif delay < 45:
        color = 'rgba(255, 120, 80, 0.7)'
        width = 2
    elif delay < 50:
        color = 'rgba(230, 70, 60, 0.8)'
        width = 2.5
    else:
        color = 'rgba(180, 30, 50, 0.9)'
        width = 3
    fig.add_trace(
        go.Scattergeo(
            lon=lons,
            lat=lats,
            mode='lines',
            line=dict(width=width, color=color),
            hoverinfo='skip',
            showlegend=False
        ),
        row=1, col=2
    )

# Add destination airports
fig.add_trace(
    go.Scattergeo(
        lon=top_delayed['DEST_LON'],
        lat=top_delayed['DEST_LAT'],
        mode='markers',
        marker=dict(
            size=12,
            color=top_delayed['delay_pct'],
            colorscale=[
                [0, 'rgb(255, 180, 100)'],
                [0.3, 'rgb(255, 120, 80)'],
                [0.6, 'rgb(230, 70, 60)'],
                [1, 'rgb(180, 30, 50)']
            ],
            cmin=top_delayed['delay_pct'].min(),
            cmax=top_delayed['delay_pct'].max(),
            colorbar=dict(
                title=dict(
                    text="Delay %",
                    font=dict(size=10)
                ),
                thickness=12,
                len=0.35,
                x=0.98,
                y=0.12,
                xanchor='right',
                yanchor='bottom',
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(150, 150, 150, 0.4)',
                borderwidth=1,
                tickfont=dict(size=9),
                outlinewidth=0
            ),
            line=dict(width=2, color='white')
        ),
        text=top_delayed['DEST'],
        hovertemplate='<b>%{text}</b><br>Delay: %{marker.color:.1f}%<br><extra></extra>',
        showlegend=False
    ),
    row=1, col=2
)

# Add origin airports
for airport in top_airports:
    airport_row = pdf_routes_focused[pdf_routes_focused['ORIGIN'] == airport].iloc[0]
    fig.add_trace(
        go.Scattergeo(
            lon=[airport_row['ORIGIN_LONG']],
            lat=[airport_row['ORIGIN_LAT']],
            mode='markers+text',
            marker=dict(size=30, color='rgb(120, 20, 40)', line=dict(width=3, color='white')),
            text=[airport],
            textposition='top center',
            textfont=dict(size=18, color='rgb(120, 20, 40)', family='Arial Black'),
            hovertemplate=f'<b>{airport}</b><br>Origin Airport<extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )

# Update layout
fig.update_xaxes(title_text="Airline", row=1, col=1)
fig.update_yaxes(title_text="Average Distance (miles)", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="Total Flights", row=1, col=1, secondary_y=True)

fig.update_geos(
    scope='usa',
    projection_type='albers usa',
    showland=True,
    landcolor='rgb(230, 230, 230)',
    coastlinecolor='rgb(150, 150, 150)',
    bgcolor='rgba(245, 248, 250, 1)',
    showlakes=True,
    lakecolor='rgb(240, 245, 255)',
    lonaxis=dict(range=[-125, -65]),
    lataxis=dict(range=[24, 50]),
    row=1, col=2
)

fig.update_layout(
    height=600,
    width=1800,
    template="plotly_white",
    showlegend=True,
    legend=dict(x=0.2, y=1.0, orientation="h"),
    margin=dict(l=50, r=30, t=80, b=50),
    paper_bgcolor='white'
)

fig.show()

# Print top delayed destinations for each airport
for airport in top_airports:
    print(f"\n=== Top High-Delay Domestic Destinations from {airport} ===")
    print(top_delayed[top_delayed['ORIGIN'] == airport][['DEST', 'delay_pct', 'total_flights', 'delayed_flights']].sort_values('delay_pct', ascending=False).to_string(index=False))

# COMMAND ----------

# DBTITLE 1,Combination Graph
# ============== PART 1: Delay Combinations ==============
delay_cols = [
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY"
]

def build_delay_combo_expr():
    exprs = []
    for col in delay_cols:
        exprs.append(
            F.when(F.col(col) > 0, col.replace("_DELAY","").replace("_", " ").title())
        )
    return exprs

df_delay_combos = (
    df_1y
    .withColumn("delay_combo_list", F.array(*build_delay_combo_expr()))
    .withColumn("delay_combo", F.concat_ws(" & ", F.expr("filter(delay_combo_list, x -> x is not null)")))
    .drop("delay_combo_list")
)
df_delay_combos = df_delay_combos.filter(F.col("delay_combo") != "")

combo_counts = (
    df_delay_combos
    .groupBy("delay_combo")
    .agg(F.count("*").alias("num_flights"))
    .orderBy(F.col("num_flights").desc())
)

pdf_combo = combo_counts.toPandas()

# Format num_flights for display: 200000 -> 200K, 1500 -> 1.5K, <1000 stays as is
def format_k(val):
    if val >= 1000:
        if val % 1000 == 0:
            return f"{int(val/1000)}K"
        else:
            return f"{val/1000:.1f}K"
    else:
        return str(val)

pdf_combo["num_flights_label"] = pdf_combo["num_flights"].apply(format_k)

# ============== PART 2: Arrival Delay Distribution ==============
df_arr_delay_dist = (
    df_1y
    .filter(F.col("CANCELLED") == 0)
    .filter(F.col("ARR_DEL15") == 1)  # Only delayed flights
    .filter(F.col("ARR_DELAY").isNotNull())
    .filter(F.col("ARR_DELAY") <= 300)  # Filter extreme outliers
    .select("ARR_DELAY")
    .sample(fraction=0.3, seed=42)  # Sample to reduce data size
)

pdf_arr_delay = df_arr_delay_dist.toPandas()

# ============== CREATE COMBINED FIGURE ==============
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Count of Flights by Exclusive Delay Type Combinations", 
                    "Distribution of Arrival Delays"),
    column_widths=[0.5, 0.5],
    horizontal_spacing=0.1
)

# Add delay combinations bar chart (no data label)
fig.add_trace(
    go.Bar(
        x=pdf_combo["delay_combo"],
        y=pdf_combo["num_flights"],
        marker_color="steelblue",
        showlegend=False
    ),
    row=1, col=1
)

# Add arrival delay histogram
fig.add_trace(
    go.Histogram(
        x=pdf_arr_delay['ARR_DELAY'],
        nbinsx=60,
        marker_color='#FF6B6B',
        marker_line_color='black',
        marker_line_width=1,
        showlegend=False
    ),
    row=1, col=2
)

# Update layout
fig.update_xaxes(title_text="Delay Combination", tickangle=-60, row=1, col=1)
fig.update_yaxes(title_text="Number of Flights", row=1, col=1)

fig.update_xaxes(title_text="Arrival Delay (minutes)", gridcolor='lightgray', row=1, col=2)
fig.update_yaxes(title_text="Number of Flights", gridcolor='lightgray', row=1, col=2)

fig.update_layout(
    height=700,
    width=2000,
    plot_bgcolor='white',
    paper_bgcolor='white',
    showlegend=False
)

fig.show()

# COMMAND ----------

# DBTITLE 1,Combination Graph 2
#make status col
df_status = (
    df_1y
    .withColumn(
        "flight_status",
        F.when(F.col("ARR_DEL15") == 1, "Delayed")
         .otherwise("On Time")
    )
)

#agg counts by day of month x status
status_by_day = (
    df_status
    .groupBy("DAY_OF_MONTH", "flight_status")
    .agg(F.count("*").alias("num_flights"))
    .orderBy("DAY_OF_MONTH", "flight_status")
)

# ============== PART 1: Heatmap - Delay Rate by Hour and Day ==============
df_delay_hour_dow = (
    df_1y
    .withColumn("DEP_HOUR", (F.col("CRS_DEP_TIME") / 100).cast("int"))
    .groupBy("DEP_HOUR", "DAY_OF_WEEK")
    .agg(
        F.count("*").alias("total_flights"),
        F.sum(F.col("ARR_DEL15").cast("int")).alias("delayed_flights")
    )
    .withColumn("delay_pct", (F.col("delayed_flights") / F.col("total_flights")) * 100)
)
pdf_heatmap = df_delay_hour_dow.toPandas()

# hour axis: 0–23
hours = list(range(24))

# day-of-week axis: 1–7 (Mon–Sun)
days = [1, 2, 3, 4, 5, 6, 7]

# pretty labels
day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# initialize matrix with None
z_matrix = [[None for _ in days] for _ in hours]

# fill matrix
for _, row in pdf_heatmap.iterrows():
    h = int(row["DEP_HOUR"])
    d = int(row["DAY_OF_WEEK"])

    if h in hours and d in days:
        hi = hours.index(h)
        di = days.index(d)
        z_matrix[hi][di] = float(row["delay_pct"])

text_matrix = [
    [f"{v:.1f}%" if v is not None else "" for v in row]
    for row in z_matrix
]

# Reverse for top-down (hour 0 at top, hour 23 at bottom)
hours_rev = hours[::-1]
z_matrix_rev = z_matrix[::-1]
text_matrix_rev = text_matrix[::-1]

# ============== PART 2: Line Chart - On-Time and Delayed Flights by Day ==============
pdf_status = status_by_day.toPandas()

# ============== CREATE COMBINED FIGURE ==============
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "On-Time and Delayed Flights by Day of Month",
        "Flight Delay Rate by Departure Hour and Day of Week<br>(% of flights delayed > 15 min)"
    ),
    column_widths=[0.5, 0.5],
    horizontal_spacing=0.08,
    specs=[[{"type": "xy"}, {"type": "heatmap"}]]
)

# Add line chart - get unique statuses and colors (now col=1)
statuses = pdf_status['flight_status'].unique()
colors = px.colors.qualitative.Plotly

for i, status in enumerate(statuses):
    data = pdf_status[pdf_status['flight_status'] == status]
    fig.add_trace(
        go.Scatter(
            x=data['DAY_OF_MONTH'],
            y=data['num_flights'],
            mode='lines+markers',
            name=status,
            line=dict(color=colors[i % len(colors)]),
            marker=dict(size=6),
            showlegend=True
        ),
        row=1, col=1
    )

# Add heatmap (now col=2) - Custom color scale optimized for your data range
fig.add_trace(
    go.Heatmap(
        z=z_matrix_rev,
        x=day_labels,
        y=hours_rev,
        colorscale=[
            [0, "#ffffcc"],      
            [0.15, "#ffeda0"],   
            [0.3, "#fed976"],    
            [0.5, "#feb24c"],    
            [0.7, "#fd8d3c"],    
            [0.85, "#fc4e2a"],  
            [1, "#bd0026"]       
        ],
        zmin=15,  # Set minimum to focus on your data range
        zmax=45,  # Set maximum to better show variation (outliers will be capped at this color)
        colorbar=dict(
            title="Delay Rate (%)",
            x=1.02,
            xanchor='left',
            len=0.8,
            thickness=15
        ),
        text=text_matrix_rev,
        texttemplate="%{text}",
        textfont=dict(color="black", size=10),
        hovertemplate="Hour: %{y}:00<br>Day: %{x}<br>Delay: %{z:.1f}%<extra></extra>",
        showlegend=False
    ),
    row=1, col=2
)

# Update axes
fig.update_xaxes(title_text="Day of Month", dtick=1, row=1, col=1)
fig.update_yaxes(title_text="Number of Flights", row=1, col=1)

fig.update_xaxes(title_text="Day of Week", row=1, col=2)
fig.update_yaxes(
    title_text="Scheduled Departure Hour (24h)", 
    tickmode='linear',
    tick0=0,
    dtick=1,
    autorange='reversed',
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=650,
    width=1700,
    template="plotly_white",
    showlegend=True,
    legend=dict(
        orientation="v",
        yanchor="top",
        y=0.98,
        xanchor="left",
        x=0.02,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(200, 200, 200, 0.5)',
        borderwidth=1
    ),
    margin=dict(l=70, r=90, t=100, b=70)
)

fig.show()

# COMMAND ----------

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Create combined figure with two subplots in a single row
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        f'Top {TOP_N} Cities by Delayed Flights: Origin (Left) vs Destination (Right)',
        "Airport Delay Analysis: % of Flights Delayed vs. Average Delay Duration"
    ),
    column_widths=[0.48, 0.52],
    horizontal_spacing=0.08
)

# --- Subplot 1: Diverging bar chart for city delays ---
fig.add_trace(
    go.Bar(
        y=pdf_top['city'],
        x=-pdf_top['origin_delayed'],
        name='Origin Delays',
        orientation='h',
        marker_color='steelblue',
        text=pdf_top['origin_delayed'].astype(int),
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>" +
                      "Origin Delayed: %{text:,}<br>" +
                      "Origin Total: %{customdata[0]:,}<br>" +
                      "Origin Delay %: %{customdata[1]:.1f}%<extra></extra>",
        customdata=pdf_top[['origin_total', 'origin_delay_pct']].values,
        legendgroup='group1',
        showlegend=True
    ),
    row=1, col=1
)

fig.add_trace(
    go.Bar(
        y=pdf_top['city'],
        x=pdf_top['dest_delayed'],
        name='Destination Delays',
        orientation='h',
        marker_color='coral',
        text=pdf_top['dest_delayed'].astype(int),
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>" +
                      "Dest Delayed: %{x:,}<br>" +
                      "Dest Total: %{customdata[0]:,}<br>" +
                      "Dest Delay %: %{customdata[1]:.1f}%<extra></extra>",
        customdata=pdf_top[['dest_total', 'dest_delay_pct']].values,
        legendgroup='group1',
        showlegend=True
    ),
    row=1, col=1
)

# Add city name annotations in the center
for i, row_ in pdf_top.iterrows():
    fig.add_annotation(
        x=0,
        y=row_['city'],
        text=f"<b>{row_['city']}</b>",
        showarrow=False,
        font=dict(size=10, color="black"),
        xanchor="center",
        bgcolor="rgba(255, 255, 255, 0.5)",
        borderpad=4,
        xref='x',
        yref='y',
        row=1, col=1
    )

# Add vertical line at x=0
fig.add_vline(x=0, line_width=2, line_color="black", row=1, col=1)

# Update axes for subplot 1
fig.update_xaxes(
    title_text='Number of Delayed Flights',
    tickformat=',d',
    range=[-axis_range, axis_range],
    row=1, col=1
)
fig.update_yaxes(showticklabels=False, row=1, col=1)

# --- Subplot 2: Bubble chart for airport delays ---
# Updated color map to match your image
color_map = {
    'large_airport': '#1f77b4',      # Blue for large airports
    'medium_airport': '#ff7f0e',     # Orange for medium airports
    'small_airport': '#e377c2',      # Pink for small airports
    'Unknown': '#7f7f7f'             # Gray for unknown
}

# Define the order for plotting (bottom to top layering)
# Plot large airports first so they appear at the bottom, then medium on top
size_order = ['large_airport', 'small_airport', 'Unknown', 'medium_airport']

# But keep legend order as desired
legend_order = ['medium_airport', 'large_airport', 'small_airport', 'Unknown']

for size in size_order:
    if size not in pdf_airport_delays['airport_size'].unique():
        continue
    
    df_size = pdf_airport_delays[pdf_airport_delays['airport_size'] == size]
    fig.add_trace(
        go.Scatter(
            x=df_size["avg_delay_minutes"],
            y=df_size["delay_pct"],
            mode="markers",
            name=size.replace('_', ' ').replace(' airport', '_airport'),
            marker=dict(
                size=df_size["total_flights"] / pdf_airport_delays["total_flights"].max() * 80,
                color=color_map.get(size, '#7f7f7f'),
                line=dict(width=0.9, color='grey'),
                opacity=0.75
            ),
            text=df_size["ORIGIN"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Total Flights: %{customdata[0]:,}<br>"
                "Delayed Flights: %{customdata[1]:,}<br>"
                "Delay %: %{customdata[2]:.2f}%<br>"
                "Avg Delay: %{customdata[3]:.2f} min<br>"
                "Airport Size: %{customdata[4]}<br>"
                "State: %{customdata[5]}<extra></extra>"
            ),
            customdata=df_size[[
                "total_flights", "delayed_flights", "delay_pct",
                "avg_delay_minutes", "airport_size", "ORIGIN_STATE_ABR"
            ]].values,
            legendgroup='group2',
            showlegend=True,
            # Control legend order
            legendrank=legend_order.index(size) if size in legend_order else 999
        ),
        row=1, col=2
    )

# Add airport code annotations for top airports
if isinstance(top_airports, list):
    top_airports_df = pdf_airport_delays[pdf_airport_delays['ORIGIN'].isin(top_airports)]
    for idx, row_ in top_airports_df.iterrows():
        fig.add_annotation(
            x=row_['avg_delay_minutes'],
            y=row_['delay_pct'],
            text=row_['ORIGIN'],
            showarrow=False,
            yshift=15,
            font=dict(size=11, color='white', family='Arial Black'),
            bgcolor='rgba(0,0,0,0)',
            borderpad=0,
            xref='x2',
            yref='y2',
            row=1, col=2
        )
else:
    for idx, row_ in top_airports.iterrows():
        fig.add_annotation(
            x=row_['avg_delay_minutes'],
            y=row_['delay_pct'],
            text=row_['ORIGIN'],
            showarrow=False,
            yshift=15,
            font=dict(size=11, color='white', family='Arial Black'),
            bgcolor='rgba(0,0,0,0)',
            borderpad=0,
            xref='x2',
            yref='y2',
            row=1, col=2
        )

# Update axes for subplot 2
fig.update_xaxes(
    title_text="Avg Delay (minutes)",
    range=x_range,
    gridcolor='lightgray',
    row=1, col=2
)
fig.update_yaxes(
    title_text="% of flights delayed 15 mins+",
    range=y_range,
    gridcolor='lightgray',
    row=1, col=2
)

# Update overall layout with unified legend styling
fig.update_layout(
    height=500,
    template='plotly_white',
    barmode='overlay',
    margin=dict(l=50, r=50, t=80, b=80),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.15,
        xanchor="center",
        x=0.5,
        font=dict(size=12),
        tracegroupgap=180,
        itemsizing='constant',
        traceorder='normal'  # Respect legendrank ordering
    )
)

fig.show()

# COMMAND ----------

from pyspark.sql.functions import col, when, array, concat_ws, expr, count

delay_cols = [
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY"
]

def build_delay_combo_expr():
    exprs = []
    for col_name in delay_cols:
        exprs.append(
            when(col(col_name) > 0, col_name.replace("_DELAY", "").replace("_", " ").title())
        )
    return exprs

# Create route node
df_delay_routes = (
    df_full
    .withColumn("delay_combo_list", array(*build_delay_combo_expr()))
    .withColumn("delay_reason", expr("filter(delay_combo_list, x -> x is not null)"))
    .withColumn("route", concat_ws(" → ", col("ORIGIN"), col("DEST")))
    .withColumn("delay_reason_exploded", expr("explode(delay_reason)"))
    .groupBy("route", "delay_reason_exploded")
    .agg(count("*").alias("num_delayed_flights"))
)

# For each reason, get top 10 routes
from pyspark.sql.window import Window
import pyspark.sql.functions as F

window = Window.partitionBy("delay_reason_exploded").orderBy(F.col("num_delayed_flights").desc())
df_top10_per_reason = (
    df_delay_routes
    .withColumn("rank", F.row_number().over(window))
    .filter(F.col("rank") <= 10)
    .orderBy("delay_reason_exploded", F.col("num_delayed_flights").desc())
    .drop("rank")
)

# Order delay reasons by total counts (desc)
reason_totals = (
    df_top10_per_reason
    .groupBy("delay_reason_exploded")
    .agg(F.sum("num_delayed_flights").alias("total_count"))
    .orderBy(F.col("total_count").desc())
    .toPandas()
)
ordered_reasons = reason_totals["delay_reason_exploded"].tolist()

pdf_sankey = df_top10_per_reason.toPandas()

# Order routes by total delayed flights (desc)
route_totals = (
    pdf_sankey.groupby("route")["num_delayed_flights"].sum().sort_values(ascending=False)
)
ordered_routes = route_totals.index.tolist()

# Add total data label for each route
route_labels = [
    f"{route} ({route_totals[route]:,})" for route in ordered_routes
]

# Add total data label for each reason
reason_labels = []
for reason in ordered_reasons:
    total = reason_totals.loc[reason_totals["delay_reason_exploded"] == reason, "total_count"].values[0]
    reason_labels.append(f"{reason} ({total:,})")

# Final node order: routes (desc) + reasons (desc)
nodes = route_labels + reason_labels
node_indices = {name: i for i, name in enumerate(nodes)}

# Update links to use new route/reason labels
sankey_links = []
for _, row in pdf_sankey.iterrows():
    route_label = f"{row['route']} ({int(route_totals[row['route']]):,})"
    reason_label = f"{row['delay_reason_exploded']} ({int(reason_totals.loc[reason_totals['delay_reason_exploded'] == row['delay_reason_exploded'], 'total_count'].values[0]):,})"
    sankey_links.append({
        'source': node_indices[route_label],
        'target': node_indices[reason_label],
        'value': row['num_delayed_flights']
    })

import plotly.graph_objects as go

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=18,
        line=dict(color="black", width=0.5),
        label=nodes
    ),
    link=dict(
        source=[l['source'] for l in sankey_links],
        target=[l['target'] for l in sankey_links],
        value=[l['value'] for l in sankey_links]
    )
)])

fig.update_layout(
    title_text="Top Delayed Routes and Delay Reasons",
    height=650,
    width=1700
)
fig.show()

# COMMAND ----------

import pandas as pd
import plotly.graph_objects as go

# Create the comparison dataframe with dummy data
model_results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Decision Tree', 'XGBoost', 'MLP'],
    'Recall_Train': [0.72, 0.78, 0.75, 0.80, 0.76],
    'Recall_Val': [0.70, 0.76, 0.72, 0.78, 0.74],
    'Recall_Test': [0.68, 0.74, 0.69, 0.76, 0.71],
    'Precision_Train': [0.65, 0.70, 0.68, 0.73, 0.69],
    'Precision_Val': [0.64, 0.69, 0.66, 0.71, 0.67],
    'Precision_Test': [0.62, 0.67, 0.64, 0.69, 0.65],
    'F2_Train': [0.70, 0.75, 0.73, 0.78, 0.74],
    'F2_Val': [0.68, 0.73, 0.70, 0.76, 0.72],
    'F2_Test': [0.66, 0.71, 0.67, 0.73, 0.69],
    'PR_AUC_Train': [0.68, 0.74, 0.71, 0.77, 0.72],
    'PR_AUC_Val': [0.66, 0.72, 0.69, 0.75, 0.70],
    'PR_AUC_Test': [0.64, 0.70, 0.66, 0.72, 0.68]
})

# Display the dataframe
print("Model Performance Comparison")
print("="*80)
display(model_results)

# Define color palette for each model
model_colors = {
    'Logistic Regression': '#FF6B6B',  # Red
    'Random Forest': '#4ECDC4',        # Teal
    'Decision Tree': '#FFE66D',        # Yellow
    'XGBoost': '#95E1D3',              # Mint
    'MLP': '#C7CEEA'                   # Purple
}

def plot_validation_performance(df):
    """
    Phase 1: Training/Validation (Model Development Phase)
    """
    
    metrics = ['Recall', 'Precision', 'F2', 'PR_AUC']
    models = df['Model'].tolist()
    
    fig = go.Figure()
    
    # Add bars for each model
    for model in models:
        model_data = df[df['Model'] == model].iloc[0]
        
        validation_scores = [
            model_data['Recall_Val'],
            model_data['Precision_Val'],
            model_data['F2_Val'],
            model_data['PR_AUC_Val']
        ]
        
        fig.add_trace(
            go.Bar(
                name=model,
                x=metrics,
                y=validation_scores,
                marker_color=model_colors[model],
                text=[f'{score:.2f}' for score in validation_scores],
                textposition='outside',
                hovertemplate=f'<b>{model}</b><br>' +
                             'Metric: %{x}<br>' +
                             'Validation Score: %{y:.2f}<extra></extra>'
            )
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': '<b>Rolling Windows Training (Final Evaluation Phase)</b><br>' +
                    '<sub>Training vs Validation Performance</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=500,
        width=900,
        barmode='group',
        legend=dict(
            title=dict(text='<b>Models</b>'),
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title='<b>Evaluation Metrics</b>',
            showgrid=False
        ),
        yaxis=dict(
            visible=False,
            range=[0, 1]
        )
    )
    
    fig.show()

def plot_test_performance(df):
    """
    Rolling Windows Training (Final Evaluation Phase)
    """
    
    metrics = ['Recall', 'Precision', 'F2', 'PR_AUC']
    models = df['Model'].tolist()
    
    fig = go.Figure()
    
    # Add bars for each model
    for model in models:
        model_data = df[df['Model'] == model].iloc[0]
        
        test_scores = [
            model_data['Recall_Test'],
            model_data['Precision_Test'],
            model_data['F2_Test'],
            model_data['PR_AUC_Test']
        ]
        
        fig.add_trace(
            go.Bar(
                name=model,
                x=metrics,
                y=test_scores,
                marker_color=model_colors[model],
                text=[f'{score:.2f}' for score in test_scores],
                textposition='outside',
                hovertemplate=f'<b>{model}</b><br>' +
                             'Metric: %{x}<br>' +
                             'Test Score: %{y:.2f}<extra></extra>'
            )
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': '<b> Final Blind Test</b><br>' +
                    '<sub>Training vs Test (Blind) Performance</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=500,
        width=900,
        barmode='group',
        legend=dict(
            title=dict(text='<b>Models</b>'),
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title='<b>Evaluation Metrics</b>',
            showgrid=False
        ),
        yaxis=dict(
            visible=False,
            range=[0, 1]
        )
    )
    
    fig.show()

def plot_performance_summary_table(df):
    """
    Create a summary table showing Train/Val/Test for each model
    """
    
    # Prepare data for table
    summary_data = []
    for _, row in df.iterrows():
        summary_data.append([
            row['Model'],
            f"{row['F2_Train']:.2f}",
            f"{row['F2_Val']:.2f}",
            f"{row['F2_Test']:.2f}",
            f"{row['F2_Train'] - row['F2_Test']:.2f}"  # Overfitting gap
        ])
    
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Model</b>', '<b>F2 (Train)</b>', '<b>F2 (Val)</b>', 
                    '<b>F2 (Test)</b>', '<b>Train-Test Gap</b>'],
            fill_color='#4A5568',
            align='left',
            font=dict(color='white', size=13)
        ),
        cells=dict(
            values=list(zip(*summary_data)),
            fill_color=[['#F7FAFC', '#EDF2F7'] * 3],
            align='left',
            font=dict(size=12),
            height=30
        )
    )])
    
    fig.update_layout(
        title='<b>F2 Score Summary: Train → Validation → Test</b>',
        height=300,
        width=900,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    fig.show()

# Create the visualizations
print("\n" + "="*80)
print("PHASE 1: VALIDATION PERFORMANCE")
print("="*80)
plot_validation_performance(model_results)

print("\n" + "="*80)
print("PHASE 2: TEST PERFORMANCE")
print("="*80)
plot_test_performance(model_results)

print("\n" + "="*80)
print("PERFORMANCE SUMMARY TABLE")
print("="*80)
plot_performance_summary_table(model_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## EDA Graph Features

# COMMAND ----------

# MAGIC %md
# MAGIC #### Graph Features
# MAGIC | Feature Name                                       | Description                                                     |
# MAGIC |----------------------------------------------------|-----------------------------------------------------------------|
# MAGIC | origin_pagerank                                    | Rank importance of origin airport; higher = major hub          |
# MAGIC | dest_pagerank                                      | Rank importance of destination airport; higher = major hub     |
# MAGIC | origin_out_degree                                  | Total number of departing flights from origin airport          |
# MAGIC | dest_in_degree                                     | Total number of arriving flights to destination airport        |
# MAGIC | prev_flight_arr_delay_clean                        | Previous flight’s arrival delay in minutes                     |
# MAGIC | actual_to_crs_time_to_next_flight_diff_mins_clean  | Actual turnaround time in minutes (current scheduled - previous actual arrival) |
# MAGIC | prev_crs_arr_time_utc                              | Previous flight’s scheduled arrival time in UTC                |
# MAGIC | prev_crs_arr_time_utc_unix                         | Previous flight’s scheduled arrival time (UNIX timestamp)      |
# MAGIC | crs_time_to_next_flight_diff_mins_log              | Log-transformed scheduled turnaround time (optional/log version) |
# MAGIC | actual_to_crs_time_to_next_flight_diff_mins_clean_log | Log-transformed actual turnaround time (optional/log version) |
# MAGIC

# COMMAND ----------

# DBTITLE 1,check all columns after graph add
display(spark.createDataFrame([(col,) for col in df_full_graph.columns], schema=['column_name']))

# COMMAND ----------

# DBTITLE 1,check graph columns
from pyspark.sql import functions as F

# Graph columns you want to inspect
graph_cols = [
    "origin_pagerank",
    "dest_pagerank",
    "origin_out_degree",
    "dest_in_degree",
    "prev_flight_arr_delay_clean",
    "actual_to_crs_time_to_next_flight_diff_mins_clean",
    "crs_time_to_next_flight_diff_mins"
]

# Total rows for null percentage
total_rows = df_full_graph.count()

# Build summary dataframe
summary_rows = []
for col in graph_cols:
    null_count = df_full_graph.filter(F.col(col).isNull()).count()
    null_pct = null_count / total_rows if total_rows > 0 else None
    summary_rows.append((col, null_count, null_pct))

graph_summary_df = spark.createDataFrame(
    summary_rows, 
    schema=["Column", "NullCount", "NullPercent"]
)

display(graph_summary_df)


# COMMAND ----------

#Visual of PageRank for DEST and ARR 
# If Spark DF, convert to Pandas for plotting
df_graph_pd = df_full_graph.select(graph_cols + ["ORIGIN", "DEST", "ORIGIN_LAT", "ORIGIN_LONG", "DEST_LAT", "DEST_LON"]).toPandas()

display(df_graph_pd)


# COMMAND ----------

#most popular flight routes
# Count number of flights per route
route_counts = (
    df_graph_pd
    .groupby(["ORIGIN", "DEST"])
    .size()  # count rows per group
    .reset_index(name="num_flights")
    .sort_values("num_flights", ascending=False)
)

# Show top 10 most common routes
top_routes = route_counts.head(10)
print(top_routes)


# COMMAND ----------

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from matplotlib import cm

# --- Take top 10 routes, ordered by number of flights descending ---
top_routes = route_counts.sort_values("num_flights", ascending=False).head(10).copy()

# --- Map lat/lon from df_graph_pd ---
origin_coords = df_graph_pd.drop_duplicates("ORIGIN").set_index("ORIGIN")[["ORIGIN_LAT","ORIGIN_LONG"]].to_dict('index')
dest_coords   = df_graph_pd.drop_duplicates("DEST").set_index("DEST")[["DEST_LAT","DEST_LON"]].to_dict('index')

top_routes["ORIGIN_LAT"] = top_routes["ORIGIN"].map(lambda x: origin_coords[x]["ORIGIN_LAT"])
top_routes["ORIGIN_LONG"] = top_routes["ORIGIN"].map(lambda x: origin_coords[x]["ORIGIN_LONG"])
top_routes["DEST_LAT"] = top_routes["DEST"].map(lambda x: dest_coords[x]["DEST_LAT"])
top_routes["DEST_LON"] = top_routes["DEST"].map(lambda x: dest_coords[x]["DEST_LON"])

# --- Color scale based on num_flights ---
max_flights = top_routes["num_flights"].max()
min_flights = top_routes["num_flights"].min()

def get_color(val):
    # normalize between 0 and 1
    norm = (val - min_flights) / (max_flights - min_flights)
    cmap = cm.get_cmap("viridis")
    rgba = cmap(norm)
    return f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'

# --- Plot figure ---
fig = go.Figure()
line_width = 2

# Plot routes
for _, row in top_routes.iterrows():
    color = get_color(row["num_flights"])
    
    # Draw route line
    fig.add_trace(go.Scattergeo(
        lon=[row["ORIGIN_LONG"], row["DEST_LON"]],
        lat=[row["ORIGIN_LAT"], row["DEST_LAT"]],
        mode='lines',
        line=dict(width=line_width, color=color),
        opacity=0.8,
        hoverinfo='text',
        text=f"{row['ORIGIN']} → {row['DEST']}<br>Flights: {row['num_flights']}",
        showlegend=False  # hide auto legend for lines
    ))
    
    # Midpoint label
    mid_lon = (row["ORIGIN_LONG"] + row["DEST_LON"]) / 2
    mid_lat = (row["ORIGIN_LAT"] + row["DEST_LAT"]) / 2
    fig.add_trace(go.Scattergeo(
        lon=[mid_lon],
        lat=[mid_lat],
        mode='text',
        text=[str(row["num_flights"])],
        textfont=dict(size=10, color='black'),
        showlegend=False
    ))

# --- Add airport markers ---
airports = pd.concat([
    top_routes[["ORIGIN","ORIGIN_LAT","ORIGIN_LONG"]].rename(columns={"ORIGIN":"Airport","ORIGIN_LAT":"lat","ORIGIN_LONG":"lon"}),
    top_routes[["DEST","DEST_LAT","DEST_LON"]].rename(columns={"DEST":"Airport","DEST_LAT":"lat","DEST_LON":"lon"})
]).drop_duplicates()

fig.add_trace(go.Scattergeo(
    lon=airports['lon'],
    lat=airports['lat'],
    mode='markers+text',
    text=airports['Airport'],
    textposition='top center',
    marker=dict(size=6, color='red', line=dict(width=1, color='black')),
    hoverinfo='text',
    name='Airport'
))

# --- Add legend for route frequency, ordered descending ---
unique_counts = sorted(top_routes["num_flights"].unique(), reverse=True)
for val in unique_counts:
    fig.add_trace(go.Scattergeo(
        lon=[None],
        lat=[None],
        mode='lines',
        line=dict(width=line_width, color=get_color(val)),
        name=f'Flights: {val}'
    ))

# --- Layout ---
fig.update_layout(
    title='Top 10 Flight Routes Colored by Number of Flights',
    showlegend=True,
    legend_title="Number of Flights",
    geo=dict(
        projection_type='albers usa',
        showland=True,
        landcolor='lightgray',
        countrycolor='white',
        lakecolor='lightblue',
        showocean=True,
        oceancolor='lightblue',
    ),
    height=700,
    width=1000
)

fig.show()


# COMMAND ----------

import plotly.graph_objects as go
import numpy as np
from matplotlib import cm

fig = go.Figure()
line_width = 2

# Color map based on num_flights
max_flights = top_routes["num_flights"].max()
min_flights = top_routes["num_flights"].min()

def get_color(val):
    norm = (val - min_flights) / (max_flights - min_flights)
    cmap = cm.get_cmap("viridis")
    rgba = cmap(norm)
    return f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'

# --- Plot each route ---
for i, row in top_routes.iterrows():
    color = get_color(row["num_flights"])
    lon_line = [row["ORIGIN_LONG"], row["DEST_LON"]]
    lat_line = [row["ORIGIN_LAT"], row["DEST_LAT"]]
    
    fig.add_trace(go.Scattergeo(
        lon=lon_line,
        lat=lat_line,
        mode='lines',
        line=dict(width=line_width, color=color),
        opacity=0.8,
        hoverinfo='text',
        text=f"{row['ORIGIN']} → {row['DEST']}<br>Flights: {row['num_flights']}",
        name=f"{row['ORIGIN']} → {row['DEST']} ({row['num_flights']} flights)"
    ))

# --- Add airport markers ---
airports = pd.concat([
    top_routes[["ORIGIN","ORIGIN_LAT","ORIGIN_LONG"]].rename(columns={"ORIGIN":"Airport","ORIGIN_LAT":"lat","ORIGIN_LONG":"lon"}),
    top_routes[["DEST","DEST_LAT","DEST_LON"]].rename(columns={"DEST":"Airport","DEST_LAT":"lat","DEST_LON":"lon"})
]).drop_duplicates()

fig.add_trace(go.Scattergeo(
    lon=airports['lon'],
    lat=airports['lat'],
    mode='markers',
    marker=dict(size=6, color='red', line=dict(width=1, color='black')),
    hoverinfo='text',
    name='Airport'
))

# --- Add airport labels with manual jitter for JFK and LGA ---
label_lons = airports['lon'].copy()
label_lats = airports['lat'].copy()
label_text = airports['Airport'].copy()

# Apply manual jitter for JFK and LGA only
for i, airport in enumerate(label_text):
    if airport == 'JFK':
        label_lons.iloc[i] += 0.3  # shift right
        label_lats.iloc[i] += 0.3  # shift up
    elif airport == 'LGA':
        label_lons.iloc[i] -= 0.3  # shift left
        label_lats.iloc[i] -= 0.3  # shift down

fig.add_trace(go.Scattergeo(
    lon=label_lons,
    lat=label_lats,
    mode='text',
    text=label_text,
    textposition='top center',
    textfont=dict(size=10),
    showlegend=False
))

# --- Layout ---
fig.update_layout(
    title='Top 10 Flight Routes with Number of Flights in Legend',
    showlegend=True,
    legend_title="Flight Routes",
    geo=dict(
        projection_type='albers usa',
        showland=True,
        landcolor='lightgray',
        countrycolor='white',
        lakecolor='lightblue',
        showocean=True,
        oceancolor='lightblue',
    ),
    height=700,
    width=1000
)

fig.show()


# COMMAND ----------

# Calculate importance score for each airport (example: sum of pagerank + degree)
origin_summary = df_graph_pd.groupby("ORIGIN").agg({
    "origin_pagerank": "mean",
    "origin_out_degree": "sum"
}).reset_index().rename(columns={"ORIGIN":"Airport", "origin_pagerank":"Pagerank", "origin_out_degree":"OutDegree"})

dest_summary = df_graph_pd.groupby("DEST").agg({
    "dest_pagerank": "mean",
    "dest_in_degree": "sum"
}).reset_index().rename(columns={"DEST":"Airport", "dest_pagerank":"Pagerank", "dest_in_degree":"InDegree"})

print(origin_summary)
print(dest_summary)

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Create figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

# --- Top 10 Origin Airports ---
sns.barplot(
    x="Pagerank",
    y="Airport",
    data=top_origin,
    palette=sns.color_palette("Blues_r", n_colors=len(top_origin)),  # reversed so high Pagerank = darkest
    ax=axes[0]
)
axes[0].set_title("Top 10 Origin Airports by PageRank")

# Add labels on bars
for p in axes[0].patches:
    width = p.get_width()
    axes[0].text(width + 0.0005,
                 p.get_y() + p.get_height() / 2,
                 f'{width:.4f}',
                 ha='left', va='center')

# --- Top 10 Destination Airports ---
sns.barplot(
    x="Pagerank",
    y="Airport",
    data=top_dest,
    palette=sns.color_palette("Greens_r", n_colors=len(top_dest)),  # reversed palette
    ax=axes[1]
)
axes[1].set_title("Top 10 Destination Airports by PageRank")

# Add labels on bars
for p in axes[1].patches:
    width = p.get_width()
    axes[1].text(width + 0.0005,
                 p.get_y() + p.get_height() / 2,
                 f'{width:.4f}',
                 ha='left', va='center')

plt.tight_layout()
plt.show()


# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,op 10
#top 10 network graph
import pandas as pd

# --- ORIGIN summary ---
origin_summary = df_graph_pd.groupby("ORIGIN").agg({
    "origin_pagerank": "mean",
    "ORIGIN_LAT": "first",
    "ORIGIN_LONG": "first"
}).reset_index().rename(columns={
    "ORIGIN": "Airport",
    "origin_pagerank": "Pagerank",
    "ORIGIN_LAT": "lat",
    "ORIGIN_LONG": "lon"
})

# --- DEST summary ---
dest_summary = df_graph_pd.groupby("DEST").agg({
    "dest_pagerank": "mean",
    "DEST_LAT": "first",
    "DEST_LON": "first"
}).reset_index().rename(columns={
    "DEST": "Airport",
    "dest_pagerank": "Pagerank",
    "DEST_LAT": "lat",
    "DEST_LON": "lon"
})

# Top 10 by PageRank
top_origin = origin_summary.sort_values("Pagerank", ascending=False).head(10)
top_dest = dest_summary.sort_values("Pagerank", ascending=False).head(10)

# Combined nodes
nodes = pd.concat([
    top_origin.assign(Type="Origin"),
    top_dest.assign(Type="Dest")
])

selected_airports = set(nodes["Airport"])

edges = df_graph_pd[df_graph_pd["ORIGIN"].isin(selected_airports) &
                    df_graph_pd["DEST"].isin(selected_airports)][[
    "ORIGIN", "DEST", "ORIGIN_LAT", "ORIGIN_LONG", "DEST_LAT", "DEST_LON"
]]

# COMMAND ----------

import pandas as pd

# Pick top 10 from each group
top_origin = origin_summary.sort_values("Pagerank", ascending=False).head(10)
top_dest = dest_summary.sort_values("Pagerank", ascending=False).head(10)

# Combine & label type
nodes = pd.concat([
    top_origin.assign(Type="Origin"),
    top_dest.assign(Type="Dest")
])


# COMMAND ----------

print(dest_nodes)
print(dest_nodes.shape)
print(dest_nodes.columns)

# COMMAND ----------

import plotly.graph_objects as go

# --- Split into two datasets ---
origin_nodes = nodes[nodes["Type"] == "Origin"].copy()
dest_nodes   = nodes[nodes["Type"] == "Dest"].copy()

# --- Helper function to build a map ---
def build_map(df, title, color):
    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lon=df["lon"],
        lat=df["lat"],
        text=df["Airport"] + "<br>Pagerank: " + df["Pagerank"].round(4).astype(str),
        mode="markers+text",
        textposition="top center",
        marker=dict(
            size=(df["Pagerank"] ** 0.5) * 40,     # scaled marker size
            color=color,
            opacity=0.9,
            line=dict(width=1, color="black")
        ),
        hovertemplate="<b>%{text}</b><extra></extra>"
    ))

    fig.update_layout(
        title=title,
        height=650,
        geo=dict(
            projection_type="albers usa",
            showland=True,
            landcolor="lightgray",
            countrycolor="white",
            coastlinecolor="gray",
            lakecolor="lightblue",
            showocean=True,
            oceancolor="lightblue"
        )
    )

    return fig

# --- Build the two maps ---
fig_origin = build_map(origin_nodes, "Top Origin and Destination Airports by PageRank", "blue")
fig_dest   = build_map(dest_nodes, "Top Destination Airports by PageRank", "green")

# --- Show them separately ---
fig_origin.show()
fig_dest.show()


# COMMAND ----------

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# --- Step 0: define top 10 origins ---
top_origin_airports = origin_summary.sort_values('Pagerank', ascending=False).head(10)['Airport'].tolist()

# --- Step 1: Filter top flights for only top origins ---
top_flights_small = top_flights[top_flights['ORIGIN'].isin(top_origin_airports)]

# Optionally, keep only top 3 destinations per origin
top_flights_small = top_flights_small.groupby('ORIGIN').apply(lambda x: x.nlargest(3, 'num_flights')).reset_index(drop=True)

print(top_origin_airports)
print(top_flights_small)

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

# --- Function to shorten line only at destination ---
def shorten_line_for_arrow_fixed(src, dst, dst_node_radius=1.1): #adj dst_node_radius for arrows to show
    """
    Keep start at src, shorten line at dst by dst_node_radius (same for all arrows)
    """
    x0, y0 = src
    x1, y1 = dst
    dx, dy = x1 - x0, y1 - y0
    total_length = np.hypot(dx, dy)
    if total_length == 0:
        factor_end = 1.0
    else:
        factor_end = max(0.0, 1 - dst_node_radius / total_length)
    x_end = x0 + dx * factor_end
    y_end = y0 + dy * factor_end
    return (x0, y0), (x_end, y_end)

# --- Manual jitter for overlapping nodes ---
pos_adjusted_jitter = pos_adjusted.copy()
if 'LGA' in pos_adjusted_jitter:
    pos_adjusted_jitter['LGA'] = (pos_adjusted_jitter['LGA'][0] - 0.3, pos_adjusted_jitter['LGA'][1] + 0.3)
if 'JFK' in pos_adjusted_jitter:
    pos_adjusted_jitter['JFK'] = (pos_adjusted_jitter['JFK'][0] + 0.3, pos_adjusted_jitter['JFK'][1] - 0.3)

plt.figure(figsize=(12,8), facecolor='white')

# Node sizes
node_sizes = [G.nodes[node]['pagerank']*50000 for node in G.nodes()]

# Node colors
node_colors = ['skyblue' if node in top_origin_airports else 'lightgreen' for node in G.nodes()]

# Draw edges with arrows (fixed arrow size)
for edge in G.edges():
    src, dst = edge
    (x0, y0), (x1, y1) = shorten_line_for_arrow_fixed(pos_adjusted_jitter[src],
                                                       pos_adjusted_jitter[dst],
                                                       dst_node_radius=1.1)
    arrow = FancyArrowPatch((x0, y0), (x1, y1),
                            arrowstyle='-|>', mutation_scale=15,  # fixed size
                            color='gray', linewidth=max(0.5, G.edges[edge]['num_flights']*0.5e-5),
                            alpha=0.7, zorder=1)
    plt.gca().add_patch(arrow)

# Draw nodes
for node, (x, y) in pos_adjusted_jitter.items():
    plt.scatter(x, y, s=G.nodes[node]['pagerank']*50000,
                c='skyblue' if node in top_origin_airports else 'lightgreen', zorder=2)

# Draw labels
for node, (x, y) in pos_adjusted_jitter.items():
    plt.text(x, y, node, fontsize=10, ha='center', va='center', zorder=3)

plt.title('Network of Top 10 Origin Airports by PageRank and Their Top Destinations')
plt.axis('off')
plt.show()


# COMMAND ----------

# MAGIC %skip
# MAGIC #NETWORK GRAPH IS INTENSIVE GRRRRR MUST SIMPLIFY (WIP)
# MAGIC # Unique airports
# MAGIC unique_airports = pd.concat([
# MAGIC     df_filtered[['ORIGIN','ORIGIN_LAT','ORIGIN_LONG','origin_pagerank']].rename(
# MAGIC         columns={'ORIGIN':'airport','ORIGIN_LAT':'lat','ORIGIN_LONG':'lon'}
# MAGIC     ),
# MAGIC     df_filtered[['DEST','DEST_LAT','DEST_LON','origin_pagerank']].rename(
# MAGIC         columns={'DEST':'airport','DEST_LAT':'lat','DEST_LON':'lon'}
# MAGIC     )
# MAGIC ]).drop_duplicates(subset='airport')
# MAGIC
# MAGIC airport_scatter = go.Scattergeo(
# MAGIC     lon=unique_airports['lon'],
# MAGIC     lat=unique_airports['lat'],
# MAGIC     text=unique_airports['airport'],
# MAGIC     marker=dict(
# MAGIC         size=unique_airports['origin_pagerank']*20,
# MAGIC         color='blue',
# MAGIC         line_color='black',
# MAGIC         line_width=0.5,
# MAGIC         sizemode='area'
# MAGIC     ),
# MAGIC     name='Airports'
# MAGIC )
# MAGIC
# MAGIC # Flight lines
# MAGIC flight_lines = [
# MAGIC     go.Scattergeo(
# MAGIC         lon=[row['ORIGIN_LONG'], row['DEST_LON']],
# MAGIC         lat=[row['ORIGIN_LAT'], row['DEST_LAT']],
# MAGIC         mode='lines',
# MAGIC         line=dict(width=0.5, color='red'),
# MAGIC         opacity=0.5,
# MAGIC         showlegend=False
# MAGIC     )
# MAGIC     for _, row in df_filtered.iterrows()
# MAGIC ]
# MAGIC
# MAGIC fig = go.Figure(data=[airport_scatter]+flight_lines)
# MAGIC fig.update_layout(
# MAGIC     title_text='Top 10 Airports by Pagerank and Flight Connections',
# MAGIC     showlegend=True,
# MAGIC     geo=dict(
# MAGIC         scope='north america',
# MAGIC         projection_type='natural earth',
# MAGIC         showland=True,
# MAGIC         landcolor="rgb(243, 243, 243)",
# MAGIC         countrycolor="rgb(204, 204, 204)",
# MAGIC     )
# MAGIC )
# MAGIC fig.show()
# MAGIC

# COMMAND ----------

#need to fix, but supposed to be SIMPLE PLOT WITH MAP OF ORIGIN BY PAGERANK
import matplotlib.pyplot as plt
import pandas as pd

# df_graph_pd: pandas DataFrame with columns:
# ORIGIN, DEST, ORIGIN_LAT, ORIGIN_LONG, DEST_LAT, DEST_LON, origin_pagerank

# -------------------------
# Top N airports by PageRank
# -------------------------
N = 10

# Top ORIGIN airports
top_origins = df_graph_pd.groupby(['ORIGIN','ORIGIN_LAT','ORIGIN_LONG'])['origin_pagerank']\
    .max().nlargest(N).reset_index()

# Top DEST airports
top_dests = df_graph_pd.groupby(['DEST','DEST_LAT','DEST_LON'])['origin_pagerank']\
    .max().nlargest(N).reset_index()

# -------------------------
# Create subplots
# -------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Common map bounds for North America
lon_min, lon_max = -130, -60
lat_min, lat_max = 20, 55

# -------------------------
# Plot ORIGIN airports
# -------------------------
axes[0].scatter(
    top_origins['ORIGIN_LONG'], top_origins['ORIGIN_LAT'],
    s=top_origins['origin_pagerank']*2000,  # bubble size
    color='skyblue', edgecolor='black', alpha=0.7
)
for i, row in top_origins.iterrows():
    axes[0].text(row['ORIGIN_LONG'], row['ORIGIN_LAT']+0.5, row['ORIGIN'], 
                 fontsize=10, ha='center', va='bottom', fontweight='bold')
axes[0].set_xlim(lon_min, lon_max)
axes[0].set_ylim(lat_min, lat_max)
axes[0].set_title("Top ORIGIN Airports by PageRank", fontsize=14)
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].grid(True, linestyle='--', alpha=0.5)

# -------------------------
# Plot DEST airports
# -------------------------
axes[1].scatter(
    top_dests['DEST_LON'], top_dests['DEST_LAT'],
    s=top_dests['origin_pagerank']*2000,
    color='salmon', edgecolor='black', alpha=0.7
)
for i, row in top_dests.iterrows():
    axes[1].text(row['DEST_LON'], row['DEST_LAT']+0.5, row['DEST'], 
                 fontsize=10, ha='center', va='bottom', fontweight='bold')
axes[1].set_xlim(lon_min, lon_max)
axes[1].set_ylim(lat_min, lat_max)
axes[1].set_title("Top DEST Airports by PageRank", fontsize=14)
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC
# MAGIC %skip
# MAGIC #network graph
# MAGIC import plotly.graph_objects as go
# MAGIC import numpy as np
# MAGIC
# MAGIC # Build vectors for all edges
# MAGIC edge_lats = []
# MAGIC edge_lons = []
# MAGIC
# MAGIC for _, row in edges.iterrows():
# MAGIC     edge_lats += [row["ORIGIN_LAT"], row["DEST_LAT"], None]  # None = break
# MAGIC     edge_lons += [row["ORIGIN_LONG"], row["DEST_LON"], None]
# MAGIC
# MAGIC fig = go.Figure()
# MAGIC
# MAGIC # ---- EDGES (single trace instead of thousands!) ----
# MAGIC fig.add_trace(go.Scattergeo(
# MAGIC     lat=edge_lats,
# MAGIC     lon=edge_lons,
# MAGIC     mode="lines",
# MAGIC     line=dict(width=1, color="gray"),
# MAGIC     opacity=0.25,
# MAGIC     hoverinfo="none"
# MAGIC ))
# MAGIC
# MAGIC # ---- NODES ----
# MAGIC fig.add_trace(go.Scattergeo(
# MAGIC     lat=nodes["lat"],
# MAGIC     lon=nodes["lon"],
# MAGIC     mode="markers+text",
# MAGIC     text=nodes["Airport"],
# MAGIC     textposition="top center",
# MAGIC     marker=dict(
# MAGIC         size=nodes["Pagerank"] * 8000,
# MAGIC         color=nodes["Type"].map({"Origin": "blue", "Dest": "green"}),
# MAGIC         opacity=0.85,
# MAGIC         line=dict(width=0.5, color="black")
# MAGIC     ),
# MAGIC     hovertemplate="<b>%{text}</b><br>Pagerank: %{marker.size}<extra></extra>"
# MAGIC ))
# MAGIC
# MAGIC fig.update_layout(
# MAGIC     title="Network Graph: Top ORIGIN and DEST Airports",
# MAGIC     height=650,
# MAGIC     geo=dict(
# MAGIC         projection_type="albers usa",
# MAGIC         showland=True,
# MAGIC         landcolor="lightgray",
# MAGIC         coastlinecolor="gray"
# MAGIC     )
# MAGIC )
# MAGIC
# MAGIC fig.show()
# MAGIC