"""
Data utility functions for wildfire analysis.

Provides common data cleaning, type conversion, and file saving utilities.
"""

import polars as pl
from typing import Dict


def clean_data_types(df: pl.DataFrame) -> pl.DataFrame:
    """
    Clean and standardize data types for parquet compatibility.

    Args:
        df: DataFrame to clean

    Returns:
        DataFrame with cleaned data types
    """
    # Convert string columns
    string_cols = ["confidence", "satellite", "instrument", "daynight", "version"]
    for col in string_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Utf8))

    # Ensure numeric columns are properly typed
    for col in df.columns:
        if df[col].dtype in [pl.Object, pl.Utf8]:
            # Try to convert to numeric if it looks numeric
            try:
                df = df.with_columns(
                    pl.when(pl.col(col).str.contains(r"^\d+(\.\d+)?$", literal=False))
                    .then(pl.col(col).cast(pl.Float64))
                    .otherwise(pl.col(col))
                    .alias(col)
                )
            except:
                pass  # Keep original type if conversion fails

    return df


def save_debug_files(
    fires_df: pl.DataFrame,
    cluster_centers: pl.DataFrame,
    weather_locations: pl.DataFrame,
    current_weather: pl.DataFrame,
    forecast_weather: pl.DataFrame,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Save intermediate data files for debugging purposes.

    Args:
        fires_df: Fire detection data
        cluster_centers: Fire cluster centers
        weather_locations: Weather collection locations
        current_weather: Current weather data
        forecast_weather: Forecast weather data
        verbose: Whether to print file save information

    Returns:
        Dictionary mapping data type to filename
    """

    saved_files = {}

    # Save fires data
    if not fires_df.is_empty():
        fires_file = "fires_data.parquet"
        clean_data_types(fires_df).write_parquet(fires_file)
        saved_files["fires"] = fires_file
        if verbose:
            print(f"   Saved fire data to: {fires_file}")

    # Save cluster centers
    if not cluster_centers.is_empty():
        cluster_file = "cluster_centers.parquet"
        cluster_centers.write_parquet(cluster_file)
        saved_files["clusters"] = cluster_file
        if verbose:
            print(f"   Saved cluster centers to: {cluster_file}")

    # Save weather locations
    if not weather_locations.is_empty():
        weather_locations_file = "weather_locations.parquet"
        weather_locations.write_parquet(weather_locations_file)
        saved_files["weather_locations"] = weather_locations_file
        if verbose:
            print(f"   Saved weather locations to: {weather_locations_file}")

    # Save current weather data
    if not current_weather.is_empty():
        current_weather_file = "current_weather.parquet"
        clean_data_types(current_weather).write_parquet(current_weather_file)
        saved_files["current_weather"] = current_weather_file
        if verbose:
            print(f"   Saved current weather data to: {current_weather_file}")
            lat_min = current_weather["latitude"].min()
            lat_max = current_weather["latitude"].max()
            lon_min = current_weather["longitude"].min()
            lon_max = current_weather["longitude"].max()
            print(
                f"   Current weather coordinates range: lat {lat_min:.5f} to {lat_max:.5f}, lon {lon_min:.5f} to {lon_max:.5f}"
            )

    # Save forecast weather data
    if not forecast_weather.is_empty():
        forecast_weather_file = "forecast_weather.parquet"
        clean_data_types(forecast_weather).write_parquet(forecast_weather_file)
        saved_files["forecast_weather"] = forecast_weather_file
        if verbose:
            print(f"   Saved forecast weather data to: {forecast_weather_file}")
            lat_min = forecast_weather["latitude"].min()
            lat_max = forecast_weather["latitude"].max()
            lon_min = forecast_weather["longitude"].min()
            lon_max = forecast_weather["longitude"].max()
            print(
                f"   Forecast weather coordinates range: lat {lat_min:.5f} to {lat_max:.5f}, lon {lon_min:.5f} to {lon_max:.5f}"
            )

    return saved_files


def print_summary_statistics(
    fires_df: pl.DataFrame,
    cluster_centers: pl.DataFrame,
    weather_locations: pl.DataFrame,
    combined_df: pl.DataFrame,
    merge_stats: Dict[str, int],
) -> None:
    """Print comprehensive summary statistics for the integrated dataset."""

    print("\n=== Summary Statistics ===")
    print(f"High-confidence fire detections: {len(fires_df)}")

    if "fire_cluster_id" in fires_df.columns:
        print(f"Fire clusters: {fires_df['fire_cluster_id'].n_unique()}")
        print(f"Weather data collected for: {len(weather_locations)} cluster centers")

    # Check weather data availability in final dataset
    weather_columns = [
        col for col in combined_df.columns if col.startswith(("current_", "forecast_"))
    ]
    if weather_columns:
        print(f"Weather columns in final dataset: {len(weather_columns)}")
        for col in weather_columns[:5]:  # Show first 5 weather columns
            non_null_count = len(combined_df) - combined_df[col].null_count()
            total_count = len(combined_df)
            print(
                f"  {col}: {non_null_count}/{total_count} non-null values ({non_null_count / total_count * 100:.1f}%)"
            )


def print_sample_data(
    combined_df: pl.DataFrame,
    current_weather: pl.DataFrame,
    forecast_weather: pl.DataFrame,
) -> None:
    """Print sample of the integrated data."""

    print("\n=== Sample Data ===")
    key_columns = ["latitude", "longitude", "acq_date", "acq_time", "satellite"]

    if "fire_cluster_id" in combined_df.columns:
        key_columns.extend(["fire_cluster_id", "cluster_size"])

    if len(current_weather) > 0:
        current_weather_cols = [
            col for col in combined_df.columns if col.startswith("current_")
        ][:2]
        key_columns.extend(current_weather_cols)

    if len(forecast_weather) > 0:
        forecast_weather_cols = [
            col for col in combined_df.columns if col.startswith("forecast_")
        ][:2]
        key_columns.extend(forecast_weather_cols)

    # Only show columns that exist
    available_columns = [col for col in key_columns if col in combined_df.columns]

    try:
        print(combined_df.select(available_columns).head())
    except UnicodeEncodeError:
        # Handle Unicode encoding issues on Windows
        print(
            "Sample data contains special characters that cannot be displayed in this terminal."
        )
        print("Showing basic info instead:")
        print(f"Columns: {available_columns}")
        print(f"Number of rows: {len(combined_df)}")
        print(f"Data types: {combined_df.select(available_columns).dtypes}")
    except Exception as e:
        print(f"Error displaying sample data: {e}")
        print(
            f"Dataset has {len(combined_df)} rows and {len(combined_df.columns)} columns"
        )
