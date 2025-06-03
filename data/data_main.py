"""
Wildfire Spread Analysis - Main Integration Module

Integrates NASA FIRMS fire detection data with Open-Meteo weather data
to create a comprehensive dataset for wildfire analysis and prediction.

Process:
1. Collects high-confidence active fire detections from satellite sources
2. Gathers current and forecast weather data directly at fire locations
3. Merges weather conditions with fire locations
4. Outputs combined dataset as wildfire_integrated.parquet

Features derived meteorological indicators for fire behavior modeling.
"""

import os

import polars as pl
from firms import collect_fire_data, get_cluster_summary
from merging import merge_weather_data
from utils import (
    clean_data_types,
    print_sample_data,
    print_summary_statistics,
)
from validation import validate_and_report
from weather import meteo_current_collect, meteo_forecast_collect


def main():
    """
    Main integration function that combines fire and weather data.

    This function orchestrates the entire data collection and integration process:
    1. Collects high-confidence fire detection data from satellite sources
    2. Gathers current and forecast meteorological data at fire cluster centers
    3. Merges weather conditions with fire locations
    4. Creates a comprehensive dataset for wildfire analysis
    """
    print("=== Wildfire Spread Analysis - Data Integration ===\n")

    # Create output directory if it doesn't exist
    output_dir = "../output"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Collect high-confidence fire detection data
    print("1. Collecting high-confidence fire detection data...")
    fires_df = collect_fire_data()

    if fires_df.is_empty():
        print("No high-confidence fires detected. Exiting.")
        return

    print(f"   Found {len(fires_df)} high-confidence fire detections")

    # Step 1.5: Get cluster summary for weather collection
    cluster_centers = pl.DataFrame()
    weather_locations = pl.DataFrame()

    if "fire_cluster_id" in fires_df.columns:
        print("\n1.5. Preparing cluster centers for weather collection...")
        cluster_centers = get_cluster_summary(fires_df)
        weather_locations = cluster_centers.select(
            ["cluster_center_lat", "cluster_center_lon"]
        ).rename({"cluster_center_lat": "latitude", "cluster_center_lon": "longitude"})
        print(f"   Using {len(weather_locations)} cluster centers for weather data")
    else:
        # Fallback if clustering is disabled
        weather_locations = fires_df.select(["latitude", "longitude"]).unique()
        print(
            f"   Using {len(weather_locations)} unique fire locations for weather data"
        )

    # Step 2: Collect current meteorological data
    print("\n2. Collecting current meteorological data at fire cluster centers...")
    current_weather = meteo_current_collect.collect_current_weather(weather_locations)
    print(f"   Collected current weather for {len(current_weather)} cluster locations")

    # Step 3: Collect forecast meteorological data
    print("\n3. Collecting forecast meteorological data at fire cluster centers...")
    forecast_weather = meteo_forecast_collect.collect_forecast_weather(
        weather_locations
    )
    print(
        f"   Collected forecast weather for {len(forecast_weather)} cluster locations"
    )

    # Prepare final dataset
    merge_stats = {}

    # Check if we have sufficient weather data
    if len(current_weather) == 0 and len(forecast_weather) == 0:
        print("\nNo weather data available. Using fire data only.")
        final_df = fires_df
    else:
        # Step 4: Merge weather data with fire locations
        print("\n4. Merging weather data with fire locations...")
        final_df, merge_stats = merge_weather_data(
            fires_df, cluster_centers, current_weather, forecast_weather
        )

    # Step 5: Clean data types and save the integrated dataset
    print("\n5. Cleaning data and saving final dataset...")
    final_df = clean_data_types(final_df)

    output_file = os.path.join(output_dir, "wildfire_integrated.parquet")
    final_df.write_parquet(output_file)

    print("\n=== Integration Complete ===")
    print(f"Combined dataset saved to: {output_file}")
    print(f"Total records: {len(final_df)}")
    print(f"Columns: {len(final_df.columns)}")

    # Display summary statistics
    print_summary_statistics(
        fires_df, cluster_centers, weather_locations, final_df, merge_stats
    )

    # Display sample of the integrated data
    print_sample_data(final_df, current_weather, forecast_weather)

    # Step 6: Validate the output file
    print("\n=== Validating Output File ===")
    validation_passed = validate_and_report(output_file)

    if validation_passed:
        print("[SUCCESS] All validation checks passed!")
    else:
        print(
            "[WARNING] Some validation checks failed - please review the issues above"
        )

    return final_df


if __name__ == "__main__":
    main()
