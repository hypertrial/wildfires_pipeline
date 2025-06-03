#!/usr/bin/env python3
"""
Data Loader Helper Module

Provides functionality for loading and validating wildfire data from parquet files
for use in forecasting models. Handles data validation and ensures required
columns are present for forecasting operations.
"""

import polars as pl
import os


def load_wildfire_data(data_path="../data/wildfire_integrated.parquet"):
    """
    Load wildfire data from parquet file using polars.

    Args:
        data_path (str): Path to the parquet file

    Returns:
        pl.DataFrame: Loaded wildfire data

    Raises:
        FileNotFoundError: If parquet file doesn't exist
        ValueError: If required columns are missing
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Parquet file not found at: {data_path}")

    # Load with polars
    df = pl.read_parquet(data_path)

    # Validate required columns
    required_columns = [
        "fire_cluster_id",
        "cluster_center_lat",
        "cluster_center_lon",
        "current_wind_speed_10m",
        "current_wind_direction_10m",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"Loaded {len(df)} rows from {data_path}")
    print(f"Found {df['fire_cluster_id'].n_unique()} unique fire clusters")

    return df


def validate_dataframe(df):
    """
    Validate that the DataFrame has the expected structure and data.

    Args:
        df (pl.DataFrame): The wildfire DataFrame to validate

    Returns:
        bool: True if validation passes

    Raises:
        ValueError: If validation fails
    """
    if len(df) == 0:
        raise ValueError("DataFrame is empty")

    # Check for required columns
    required_columns = [
        "fire_cluster_id",
        "cluster_center_lat",
        "cluster_center_lon",
        "current_wind_speed_10m",
        "current_wind_direction_10m",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

        # Check for null values in critical columns
        null_count = df.select(pl.col(col).is_null().sum()).item()
        if null_count > 0:
            print(f"Warning: Found {null_count} null values in column '{col}'")

    # Check data ranges
    lat_out_of_range = df.filter(
        (pl.col("cluster_center_lat") < -90) | (pl.col("cluster_center_lat") > 90)
    ).height
    if lat_out_of_range > 0:
        raise ValueError("Invalid latitude values found")

    lon_out_of_range = df.filter(
        (pl.col("cluster_center_lon") < -180) | (pl.col("cluster_center_lon") > 180)
    ).height
    if lon_out_of_range > 0:
        raise ValueError("Invalid longitude values found")

    wind_dir_out_of_range = df.filter(
        (pl.col("current_wind_direction_10m") < 0)
        | (pl.col("current_wind_direction_10m") > 360)
    ).height
    if wind_dir_out_of_range > 0:
        print("Warning: Wind direction values outside 0-360 range found")

    negative_wind_speed = df.filter(pl.col("current_wind_speed_10m") < 0).height
    if negative_wind_speed > 0:
        print("Warning: Negative wind speed values found")

    return True
