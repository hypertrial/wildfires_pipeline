"""
Coordinate-based merging utilities for wildfire weather data integration.

Handles proximity-based matching between fire cluster centers and weather data
to account for coordinate precision differences from weather APIs.
"""

import numpy as np
import polars as pl
from scipy.spatial.distance import cdist
from typing import Dict, Tuple


def merge_weather_data(
    fires_df: pl.DataFrame,
    cluster_centers: pl.DataFrame,
    current_weather: pl.DataFrame,
    forecast_weather: pl.DataFrame,
    max_distance_km: float = 20.0,
    verbose: bool = True,
) -> Tuple[pl.DataFrame, Dict[str, int]]:
    """
    Merge weather data with fire locations using proximity-based matching.

    Args:
        fires_df: DataFrame with fire detection data
        cluster_centers: DataFrame with fire cluster centers
        current_weather: DataFrame with current weather data
        forecast_weather: DataFrame with forecast weather data
        max_distance_km: Maximum distance in km for matching weather data to fire clusters
        verbose: Whether to print detailed merge information

    Returns:
        Tuple of (merged_dataframe, statistics_dict)
    """
    stats = {
        "fire_records": len(fires_df),
        "cluster_centers": len(cluster_centers),
        "current_matches": 0,
        "forecast_matches": 0,
        "fires_with_current": 0,
        "fires_with_forecast": 0,
    }

    # Start with fire detection data
    combined_df = fires_df.clone()

    if verbose:
        print("4. Merging weather data with fire locations...")
        print(f"   Starting with {len(combined_df)} fire detections")

    # For clustered data, use cluster-based merging
    if "fire_cluster_id" in fires_df.columns and not cluster_centers.is_empty():
        if verbose:
            print("   Using cluster-based merging...")

        cluster_weather_mapping = _merge_cluster_weather(
            cluster_centers, current_weather, forecast_weather, max_distance_km, verbose
        )

        # Update statistics
        if "current_timestamp" in cluster_weather_mapping.columns:
            stats["current_matches"] = (
                len(cluster_weather_mapping)
                - cluster_weather_mapping["current_timestamp"].null_count()
            )
        if "forecast_timestamp" in cluster_weather_mapping.columns:
            stats["forecast_matches"] = (
                len(cluster_weather_mapping)
                - cluster_weather_mapping["forecast_timestamp"].null_count()
            )

        if verbose:
            print("   Cluster weather mapping coverage:")
            print(
                f"     Current weather matches: {stats['current_matches']}/{len(cluster_centers)} clusters"
            )
            print(
                f"     Forecast weather matches: {stats['forecast_matches']}/{len(cluster_centers)} clusters"
            )

        # Merge weather data with fire detections based on cluster ID
        weather_cols = [
            col
            for col in cluster_weather_mapping.columns
            if col
            not in ["fire_cluster_id", "cluster_center_lat", "cluster_center_lon"]
        ]

        if verbose:
            print(f"   Merging {len(weather_cols)} weather columns with fire data...")

        combined_df = combined_df.join(
            cluster_weather_mapping.select(["fire_cluster_id"] + weather_cols),
            on="fire_cluster_id",
            how="left",
        )

        # Update fire statistics
        if len(current_weather) > 0 and "current_timestamp" in combined_df.columns:
            stats["fires_with_current"] = (
                len(combined_df) - combined_df["current_timestamp"].null_count()
            )
        if len(forecast_weather) > 0 and "forecast_timestamp" in combined_df.columns:
            stats["fires_with_forecast"] = (
                len(combined_df) - combined_df["forecast_timestamp"].null_count()
            )

        if verbose:
            print(
                f"   Fire records with current weather: {stats['fires_with_current']}/{len(combined_df)}"
            )
            print(
                f"   Fire records with forecast weather: {stats['fires_with_forecast']}/{len(combined_df)}"
            )

    else:
        # Fallback for non-clustered data - direct coordinate merging
        if verbose:
            print("   Using direct coordinate merging with proximity matching...")

        combined_df = _merge_direct_coordinates(
            combined_df, current_weather, forecast_weather, max_distance_km, verbose
        )

        # Update statistics
        if len(current_weather) > 0 and "current_timestamp" in combined_df.columns:
            stats["fires_with_current"] = (
                len(combined_df) - combined_df["current_timestamp"].null_count()
            )
        if len(forecast_weather) > 0 and "forecast_timestamp" in combined_df.columns:
            stats["fires_with_forecast"] = (
                len(combined_df) - combined_df["forecast_timestamp"].null_count()
            )

    return combined_df, stats


def _merge_cluster_weather(
    cluster_centers: pl.DataFrame,
    current_weather: pl.DataFrame,
    forecast_weather: pl.DataFrame,
    max_distance_km: float,
    verbose: bool,
) -> pl.DataFrame:
    """Merge weather data with cluster centers using proximity matching."""

    cluster_weather_mapping = cluster_centers.select(
        ["fire_cluster_id", "cluster_center_lat", "cluster_center_lon"]
    ).clone()

    if verbose:
        print(f"   Cluster mapping has {len(cluster_weather_mapping)} entries")

    # Merge current weather data if available
    if len(current_weather) > 0:
        if verbose:
            print("   Merging current weather conditions...")
            print(f"   Current weather has {len(current_weather)} records")

        cluster_weather_mapping = _proximity_merge_weather(
            cluster_weather_mapping, current_weather, "current", max_distance_km
        )

        if "current_timestamp" in cluster_weather_mapping.columns:
            successful_matches = (
                len(cluster_weather_mapping)
                - cluster_weather_mapping["current_timestamp"].null_count()
            )
            if verbose:
                print(
                    f"   Proximity-based merge result: {successful_matches} successful current weather matches (within {max_distance_km}km)"
                )

    # Merge forecast weather data if available
    if len(forecast_weather) > 0:
        if verbose:
            print("   Merging forecast weather conditions...")
            print(f"   Forecast weather has {len(forecast_weather)} records")

        cluster_weather_mapping = _proximity_merge_weather(
            cluster_weather_mapping, forecast_weather, "forecast", max_distance_km
        )

        if "forecast_timestamp" in cluster_weather_mapping.columns:
            successful_matches = (
                len(cluster_weather_mapping)
                - cluster_weather_mapping["forecast_timestamp"].null_count()
            )
            if verbose:
                print(
                    f"   Proximity-based merge result: {successful_matches} successful forecast weather matches (within {max_distance_km}km)"
                )

    return cluster_weather_mapping


def _merge_direct_coordinates(
    combined_df: pl.DataFrame,
    current_weather: pl.DataFrame,
    forecast_weather: pl.DataFrame,
    max_distance_km: float,
    verbose: bool,
) -> pl.DataFrame:
    """Merge weather data directly with fire coordinates using proximity matching."""

    if len(current_weather) > 0:
        if verbose:
            print("   Merging current weather conditions...")

        combined_df = _proximity_merge_fire_weather(
            combined_df, current_weather, "current", max_distance_km
        )

        if "current_timestamp" in combined_df.columns:
            successful_matches = (
                len(combined_df) - combined_df["current_timestamp"].null_count()
            )
            if verbose:
                print(
                    f"   Current weather merge: {len(combined_df)} rows, {successful_matches} matches"
                )

    if len(forecast_weather) > 0:
        if verbose:
            print("   Merging forecast weather conditions...")

        combined_df = _proximity_merge_fire_weather(
            combined_df, forecast_weather, "forecast", max_distance_km
        )

        if "forecast_timestamp" in combined_df.columns:
            successful_matches = (
                len(combined_df) - combined_df["forecast_timestamp"].null_count()
            )
            if verbose:
                print(
                    f"   Forecast weather merge: {len(combined_df)} rows, {successful_matches} matches"
                )

    return combined_df


def _proximity_merge_weather(
    cluster_mapping: pl.DataFrame,
    weather_data: pl.DataFrame,
    prefix: str,
    max_distance_km: float,
) -> pl.DataFrame:
    """Perform proximity-based merge between cluster centers and weather data."""

    # Convert to numpy for distance calculations
    cluster_coords = cluster_mapping.select(
        ["cluster_center_lat", "cluster_center_lon"]
    ).to_numpy()
    weather_coords = weather_data.select(["latitude", "longitude"]).to_numpy()

    # Calculate distances between cluster centers and weather locations (in km)
    # Using haversine-like approximation for small distances
    distances = (
        cdist(cluster_coords, weather_coords, metric="euclidean") * 111.32
    )  # Rough km conversion

    # Find closest weather station for each cluster within max distance
    closest_indices = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)

    # Create mapping for clusters with weather data within max distance
    valid_matches = min_distances <= max_distance_km

    # Create weather data to merge
    weather_columns = [
        col for col in weather_data.columns if col not in ["latitude", "longitude"]
    ]

    # Initialize new columns with nulls
    new_cols = {}
    for col in weather_columns:
        if col == "timestamp":
            new_cols[f"{prefix}_timestamp"] = pl.lit(None, dtype=pl.Datetime)
        else:
            new_cols[f"{prefix}_{col}"] = pl.lit(None, dtype=weather_data[col].dtype)

    result = cluster_mapping.with_columns(**new_cols)

    # Fill in matched weather data
    if valid_matches.any():
        valid_cluster_indices = np.where(valid_matches)[0]
        valid_weather_indices = closest_indices[valid_matches]

        for i, (cluster_idx, weather_idx) in enumerate(
            zip(valid_cluster_indices, valid_weather_indices)
        ):
            for col in weather_columns:
                new_col_name = (
                    f"{prefix}_timestamp" if col == "timestamp" else f"{prefix}_{col}"
                )
                weather_value = weather_data[col][int(weather_idx)]
                result = result.with_columns(
                    pl.when(pl.int_range(pl.len()) == int(cluster_idx))
                    .then(pl.lit(weather_value))
                    .otherwise(pl.col(new_col_name))
                    .alias(new_col_name)
                )

    return result


def _proximity_merge_fire_weather(
    fire_df: pl.DataFrame,
    weather_data: pl.DataFrame,
    prefix: str,
    max_distance_km: float,
) -> pl.DataFrame:
    """Perform proximity-based merge between fire locations and weather data."""

    # Convert to numpy for distance calculations
    fire_coords = fire_df.select(["latitude", "longitude"]).to_numpy()
    weather_coords = weather_data.select(["latitude", "longitude"]).to_numpy()

    # Calculate distances (rough km approximation)
    distances = cdist(fire_coords, weather_coords, metric="euclidean") * 111.32

    # Find closest weather station for each fire within max distance
    closest_indices = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)

    # Create mapping for fires with weather data within max distance
    valid_matches = min_distances <= max_distance_km

    # Create weather data to merge
    weather_columns = [
        col for col in weather_data.columns if col not in ["latitude", "longitude"]
    ]

    # Initialize new columns with nulls
    new_cols = {}
    for col in weather_columns:
        if col == "timestamp":
            new_cols[f"{prefix}_timestamp"] = pl.lit(None, dtype=pl.Datetime)
        else:
            new_cols[f"{prefix}_{col}"] = pl.lit(None, dtype=weather_data[col].dtype)

    result = fire_df.with_columns(**new_cols)

    # Fill in matched weather data
    if valid_matches.any():
        valid_fire_indices = np.where(valid_matches)[0]
        valid_weather_indices = closest_indices[valid_matches]

        for i, (fire_idx, weather_idx) in enumerate(
            zip(valid_fire_indices, valid_weather_indices)
        ):
            for col in weather_columns:
                new_col_name = (
                    f"{prefix}_timestamp" if col == "timestamp" else f"{prefix}_{col}"
                )
                weather_value = weather_data[col][
                    int(weather_idx)
                ]  # Convert to Python int
                result = result.with_columns(
                    pl.when(
                        pl.int_range(pl.len()) == int(fire_idx)
                    )  # Convert to Python int
                    .then(pl.lit(weather_value))
                    .otherwise(pl.col(new_col_name))
                    .alias(new_col_name)
                )

    return result
