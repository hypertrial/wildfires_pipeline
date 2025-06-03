"""
Fire Clustering Module

Provides spatial clustering functionality for grouping nearby fire detections
into logical fire events using DBSCAN clustering with haversine distance.
"""

import numpy as np
import polars as pl
from sklearn.cluster import DBSCAN
from sklearn.metrics import DistanceMetric

# Fire clustering configuration
DEFAULT_CLUSTER_DISTANCE_KM = 1.0  # Group fires within 1km of each other


def _haversine_distance_matrix(
    coords_rad: np.ndarray, distance_km: float
) -> np.ndarray:
    """Optimized haversine distance calculation for DBSCAN."""
    # Use sklearn's haversine metric for better performance
    haversine_metric = DistanceMetric.get_metric("haversine")
    return haversine_metric.pairwise(coords_rad) * 6371.0  # Earth radius in km


def calculate_fire_clusters(
    df: pl.DataFrame, distance_km: float = DEFAULT_CLUSTER_DISTANCE_KM
) -> pl.DataFrame:
    """
    Group nearby fire detections into clusters using spatial clustering.

    Args:
        df: DataFrame with fire detections containing 'latitude' and 'longitude' columns
        distance_km: Maximum distance in kilometers to group fires together

    Returns:
        DataFrame with additional 'fire_cluster_id' column and aggregated cluster statistics
    """
    # Early exit for empty or single-point datasets
    if df.is_empty():
        return df

    if len(df) == 1:
        return df.with_columns(
            [
                pl.lit(0).alias("fire_cluster_id"),
                pl.lit(1).alias("cluster_size"),
                pl.col("latitude").alias("cluster_center_lat"),
                pl.col("longitude").alias("cluster_center_lon"),
                pl.col("frp").alias("cluster_max_frp"),
                pl.lit(None).alias("cluster_avg_confidence"),
            ]
        )

    # Check required columns early
    required_cols = ["latitude", "longitude"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing required columns for clustering: {missing_cols}")
        # Add default cluster columns even if clustering fails
        return df.with_columns(
            [
                pl.int_range(pl.len()).alias("fire_cluster_id"),
                pl.lit(1).alias("cluster_size"),
                pl.col("latitude").alias("cluster_center_lat"),
                pl.col("longitude").alias("cluster_center_lon"),
                pl.col("frp").alias("cluster_max_frp"),
                pl.lit(None).alias("cluster_avg_confidence"),
            ]
        )

    # Early filtering and validation using lazy evaluation
    df_lazy = df.lazy()

    # Filter valid coordinates and add row index in one operation
    valid_coords_df = (
        df_lazy.with_row_index("original_idx")
        .filter(
            pl.col("latitude").is_not_null()
            & pl.col("longitude").is_not_null()
            & pl.col("latitude").is_between(-90, 90)
            & pl.col("longitude").is_between(-180, 180)
        )
        .collect()
    )

    if valid_coords_df.is_empty():
        print("[ERROR] No valid coordinates found for clustering")
        # Add default cluster columns even if clustering fails
        return df.with_columns(
            [
                pl.int_range(pl.len()).alias("fire_cluster_id"),
                pl.lit(1).alias("cluster_size"),
                pl.col("latitude").alias("cluster_center_lat"),
                pl.col("longitude").alias("cluster_center_lon"),
                pl.col("frp").alias("cluster_max_frp"),
                pl.lit(None).alias("cluster_avg_confidence"),
            ]
        )

    # Extract coordinates efficiently
    coords = valid_coords_df.select(["latitude", "longitude"]).to_numpy()

    # Convert to radians for haversine calculation
    coords_rad = np.radians(coords)

    # Perform clustering with optimized approach
    try:
        if len(coords) < 1000:
            # For smaller datasets, use precomputed distance matrix
            distance_matrix = _haversine_distance_matrix(coords_rad, distance_km)
            clustering = DBSCAN(eps=distance_km, min_samples=1, metric="precomputed")
            cluster_labels = clustering.fit_predict(distance_matrix)
        else:
            # For larger datasets, use haversine metric directly (more memory efficient)
            # Note: sklearn's DBSCAN doesn't directly support haversine, so we use a workaround
            # by scaling the coordinates and using euclidean approximation for efficiency
            # This is an approximation but much faster for large datasets

            # Convert to approximate cartesian coordinates for efficiency
            lat_mean = np.mean(coords[:, 0])
            cos_lat = np.cos(np.radians(lat_mean))

            # Scale coordinates to approximate distance in km
            coords_scaled = coords.copy()
            coords_scaled[:, 0] *= 111.0  # 1 degree lat ≈ 111 km
            coords_scaled[:, 1] *= 111.0 * cos_lat  # 1 degree lon ≈ 111*cos(lat) km

            clustering = DBSCAN(eps=distance_km, min_samples=1, metric="euclidean")
            cluster_labels = clustering.fit_predict(coords_scaled)

    except Exception as e:
        print(f"[ERROR] DBSCAN clustering failed: {e}")
        # Add default cluster columns even if clustering fails
        return df.with_columns(
            [
                pl.int_range(pl.len()).alias("fire_cluster_id"),
                pl.lit(1).alias("cluster_size"),
                pl.col("latitude").alias("cluster_center_lat"),
                pl.col("longitude").alias("cluster_center_lon"),
                pl.col("frp").alias("cluster_max_frp"),
                pl.lit(None).alias("cluster_avg_confidence"),
            ]
        )

    # Create cluster mapping efficiently
    cluster_mapping = pl.DataFrame(
        {
            "original_idx": valid_coords_df["original_idx"].to_list(),
            "fire_cluster_id": cluster_labels,
        }
    )

    # Add row index to original dataframe and join cluster results
    df_with_clusters = (
        df.lazy()
        .with_row_index("original_idx")
        .join(cluster_mapping.lazy(), on="original_idx", how="left")
        .drop("original_idx")
        .collect()
    )

    # Handle null cluster IDs efficiently - fix the cumsum issue
    max_cluster_id = cluster_mapping["fire_cluster_id"].max() or -1

    # Count null values and create sequential IDs for them
    null_count = df_with_clusters["fire_cluster_id"].null_count()
    if null_count > 0:
        # Create a sequence of new cluster IDs for null values
        null_ids = list(range(max_cluster_id + 1, max_cluster_id + 1 + null_count))

        df_with_clusters = (
            df_with_clusters.with_row_index("temp_idx")
            .with_columns(
                [
                    pl.when(pl.col("fire_cluster_id").is_null())
                    .then(
                        pl.col("temp_idx").map_elements(
                            lambda x: null_ids.pop(0)
                            if null_ids
                            else max_cluster_id + 1,
                            return_dtype=pl.Int64,
                        )
                    )
                    .otherwise(pl.col("fire_cluster_id"))
                    .alias("fire_cluster_id")
                ]
            )
            .drop("temp_idx")
        )

    # Calculate cluster statistics in one efficient operation
    cluster_stats = (
        df_with_clusters.lazy()
        .group_by("fire_cluster_id")
        .agg(
            [
                pl.col("latitude")
                .filter(pl.col("latitude").is_not_null())
                .mean()
                .alias("cluster_center_lat"),
                pl.col("longitude")
                .filter(pl.col("longitude").is_not_null())
                .mean()
                .alias("cluster_center_lon"),
                pl.len().alias("cluster_size"),
                pl.col("frp").max().alias("cluster_max_frp"),
                pl.lit(None).alias("cluster_avg_confidence"),
            ]
        )
        .collect()
    )

    # Final join to add cluster statistics
    try:
        result = df_with_clusters.join(cluster_stats, on="fire_cluster_id", how="left")
        return result

    except Exception as e:
        print(f"[ERROR] Failed to merge cluster statistics: {e}")
        # Ensure we still have the fire_cluster_id column
        if "fire_cluster_id" not in df_with_clusters.columns:
            df_with_clusters = df_with_clusters.with_columns(
                [
                    pl.int_range(pl.len()).alias("fire_cluster_id"),
                    pl.lit(1).alias("cluster_size"),
                    pl.col("latitude").alias("cluster_center_lat"),
                    pl.col("longitude").alias("cluster_center_lon"),
                    pl.col("frp").alias("cluster_max_frp"),
                    pl.lit(None).alias("cluster_avg_confidence"),
                ]
            )
        return df_with_clusters


def get_cluster_summary(df: pl.DataFrame) -> pl.DataFrame:
    """
    Get a summary of fire clusters with one row per cluster.

    Args:
        df: DataFrame with clustered fire data

    Returns:
        DataFrame with one row per fire cluster containing cluster statistics
    """
    if df.is_empty() or "fire_cluster_id" not in df.columns:
        return pl.DataFrame()

    # Optimized cluster summary using lazy evaluation
    try:
        cluster_summary = (
            df.lazy()
            .group_by("fire_cluster_id")
            .agg(
                [
                    # Use first non-null values for cluster statistics
                    pl.col("cluster_center_lat")
                    .filter(pl.col("cluster_center_lat").is_not_null())
                    .first()
                    .alias("cluster_center_lat"),
                    pl.col("cluster_center_lon")
                    .filter(pl.col("cluster_center_lon").is_not_null())
                    .first()
                    .alias("cluster_center_lon"),
                    pl.col("cluster_size").first().alias("cluster_size"),
                    pl.col("cluster_max_frp").first().alias("cluster_max_frp"),
                    pl.col("cluster_avg_confidence")
                    .first()
                    .alias("cluster_avg_confidence"),
                    pl.col("acq_date").first().alias("acq_date"),
                    # Efficiently concatenate unique values
                    pl.col("country_id")
                    .filter(pl.col("country_id").is_not_null())
                    .unique()
                    .str.concat(", ")
                    .alias("country_id"),
                    pl.col("satellite")
                    .filter(pl.col("satellite").is_not_null())
                    .unique()
                    .str.concat(", ")
                    .alias("satellite"),
                    pl.col("instrument")
                    .filter(pl.col("instrument").is_not_null())
                    .unique()
                    .str.concat(", ")
                    .alias("instrument"),
                ]
            )
            .sort("fire_cluster_id")
            .collect()
        )

        return cluster_summary

    except Exception as e:
        print(f"[ERROR] Failed to create cluster summary: {e}")
        return pl.DataFrame()


def add_clustering_columns_to_schema(cols: list, has_clusters: bool) -> list:
    """Add clustering-related columns to the schema if clustering was performed."""
    if has_clusters:
        clustering_cols = [
            "fire_cluster_id",
            "cluster_size",
            "cluster_center_lat",
            "cluster_center_lon",
            "cluster_max_frp",
            "cluster_avg_confidence",
        ]

        # Use set operations for efficient deduplication
        existing_cols = set(cols)
        cols.extend([col for col in clustering_cols if col not in existing_cols])

    return cols
