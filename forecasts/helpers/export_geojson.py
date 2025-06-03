#!/usr/bin/env python3
"""
GeoJSON Export Helper Module

Provides functionality for creating and exporting GeoJSON files from fire model results.
Handles the conversion of cellular automata grid results into GeoJSON format
for visualization and further analysis.
"""

import json
import numpy as np
from typing import List
from shapely.geometry import Polygon
from shapely.ops import unary_union


def extract_fire_cells(
    grid, lats, lons, cluster_id: int, forecast_minutes: int
) -> List[dict]:
    """
    Extract fire cells from grid and convert to GeoJSON features.

    Args:
        grid: CA grid with fire states (0=unburned, 1=burning, 2=burned)
        lats: Latitude array
        lons: Longitude array
        cluster_id (int): Cluster ID
        forecast_minutes (int): Forecast time in minutes

    Returns:
        list: List of GeoJSON features
    """
    # Extract cells with state 1 (burning) or 2 (burned)
    mask = (grid == 1) | (grid == 2)
    i_idx, j_idx = np.where(mask)
    cell_lats = lats[i_idx]
    cell_lons = lons[j_idx]

    features = []
    for lat, lon in zip(cell_lats, cell_lons):
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        float(lon),
                        float(lat),
                    ],  # [lon, lat] per GeoJSON spec
                },
                "properties": {
                    "cluster_id": int(cluster_id),
                    "forecast_minutes": forecast_minutes,
                },
            }
        )

    return features


def create_fire_spread_polygon(
    grid, lats, lons, cluster_id: int, area_type: str, forecast_minutes: int = None
) -> dict:
    """
    Create a polygon feature representing the fire spread area boundary.

    Args:
        grid: CA grid with fire states (0=unburned, 1=burning, 2=burned)
        lats: Latitude array for grid
        lons: Longitude array for grid
        cluster_id (int): Cluster ID
        area_type (str): Type of area ('current' or 'forecast')
        forecast_minutes (int, optional): Forecast time in minutes

    Returns:
        dict: GeoJSON feature with polygon geometry or None if no fire area
    """
    # For current areas, only include actively burning cells (state 1)
    # For forecast areas, include both burning and burned cells (state 1 and 2)
    if area_type == "current":
        mask = grid == 1  # Only actively burning cells for current
    else:
        mask = (grid == 1) | (grid == 2)  # Both burning and burned cells for forecast

    if not np.any(mask):
        return None

    i_idx, j_idx = np.where(mask)

    # Get grid spacing (assuming uniform)
    if len(lats) > 1 and len(lons) > 1:
        lat_step = lats[1] - lats[0]
        lon_step = lons[1] - lons[0]
    else:
        # Fallback for single cell
        lat_step = 0.009  # ~1km
        lon_step = 0.009

    # Create grid cell polygons and union them
    polygons = []
    for i, j in zip(i_idx, j_idx):
        lat = lats[i]
        lon = lons[j]

        # Create polygon for this grid cell
        half_lat = lat_step / 2
        half_lon = lon_step / 2

        cell_polygon = Polygon(
            [
                (lon - half_lon, lat - half_lat),  # SW
                (lon + half_lon, lat - half_lat),  # SE
                (lon + half_lon, lat + half_lat),  # NE
                (lon - half_lon, lat + half_lat),  # NW
                (lon - half_lon, lat - half_lat),  # Close
            ]
        )
        polygons.append(cell_polygon)

    if not polygons:
        return None

    # Union all polygons to create boundary
    if len(polygons) == 1:
        merged_polygon = polygons[0]
    else:
        merged_polygon = unary_union(polygons)

    # Handle MultiPolygon case by taking the largest polygon
    if hasattr(merged_polygon, "geoms"):
        merged_polygon = max(merged_polygon.geoms, key=lambda p: p.area)

    # Convert to GeoJSON coordinates format
    if merged_polygon.is_empty:
        return None

    coords = list(merged_polygon.exterior.coords)

    # Build properties
    properties = {
        "cluster_id": int(cluster_id),
        "area_type": area_type,
    }

    if forecast_minutes is not None:
        properties["forecast_minutes"] = forecast_minutes

    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": properties,
    }


def create_cluster_center_feature(cluster_id: int, lat: float, lon: float) -> dict:
    """
    Create a point feature for the original fire cluster center.

    Args:
        cluster_id (int): Cluster ID
        lat (float): Cluster center latitude
        lon (float): Cluster center longitude

    Returns:
        dict: GeoJSON feature with point geometry
    """
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
        "properties": {"cluster_id": int(cluster_id), "feature_type": "cluster_center"},
    }


def create_combined_forecast_geojson(
    clusters_data: List[dict], output_path: str
) -> int:
    """
    Create a combined GeoJSON containing cluster centers, current fire areas, and forecasted areas.

    Args:
        clusters_data (list): List of cluster data dictionaries containing:
            - cluster_id: int
            - center_lat: float
            - center_lon: float
            - current_grid: numpy array
            - forecast_grid: numpy array
            - lats: numpy array
            - lons: numpy array
            - forecast_minutes: int
        output_path (str): Path to save the combined GeoJSON

    Returns:
        int: Total number of features created
    """
    all_features = []

    for cluster_data in clusters_data:
        cluster_id = cluster_data["cluster_id"]

        # Add cluster center point
        center_feature = create_cluster_center_feature(
            cluster_id, cluster_data["center_lat"], cluster_data["center_lon"]
        )
        all_features.append(center_feature)

        # Add current fire spread area
        current_area = create_fire_spread_polygon(
            cluster_data["current_grid"],
            cluster_data["lats"],
            cluster_data["lons"],
            cluster_id,
            "current",
        )
        if current_area:
            all_features.append(current_area)

        # Add forecasted fire spread area
        forecast_area = create_fire_spread_polygon(
            cluster_data["forecast_grid"],
            cluster_data["lats"],
            cluster_data["lons"],
            cluster_id,
            "forecast",
            cluster_data["forecast_minutes"],
        )
        if forecast_area:
            all_features.append(forecast_area)

    # Create GeoJSON FeatureCollection
    geojson = create_geojson_featurecollection(all_features)

    # Save to file
    save_geojson(geojson, output_path)

    return len(all_features)


def create_geojson_featurecollection(features: List[dict]) -> dict:
    """
    Create a GeoJSON FeatureCollection from a list of features.

    Args:
        features (list): List of GeoJSON features

    Returns:
        dict: GeoJSON FeatureCollection
    """
    return {"type": "FeatureCollection", "features": features}


def save_geojson(geojson: dict, filepath: str):
    """
    Save GeoJSON to file.

    Args:
        geojson (dict): GeoJSON object
        filepath (str): Output file path
    """
    with open(filepath, "w") as f:
        json.dump(geojson, f, indent=2)


def create_and_save_forecast_geojson(
    grid, lats, lons, cluster_id: int, forecast_minutes: int, output_path: str
) -> int:
    """
    Complete pipeline to extract fire cells, create GeoJSON, and save to file.

    Args:
        grid: CA grid with fire states
        lats: Latitude array
        lons: Longitude array
        cluster_id (int): Cluster ID
        forecast_minutes (int): Forecast time in minutes
        output_path (str): Path to save the GeoJSON file

    Returns:
        int: Number of features created
    """
    # Extract fire cells as features
    features = extract_fire_cells(grid, lats, lons, cluster_id, forecast_minutes)

    # Create GeoJSON FeatureCollection
    geojson = create_geojson_featurecollection(features)

    # Save to file
    save_geojson(geojson, output_path)

    return len(features)
