#!/usr/bin/env python3
"""
Baseline Fire Forecasting Model

Baseline fire forecasting model implementation using Cellular Automata.
Provides spatial fire spread simulation with wind influence for short-term
wildfire spread prediction.

Features:
- 1 km × 1 km spatial grid resolution
- Cellular Automata fire spread simulation
- Wind speed and direction influence
- GeoJSON output for visualization

This model:
1) Reads FIRMS + weather data from parquet using polars
2) Builds 1 km×1 km grids around fire cluster centers
3) Runs Cellular Automata (CA) for fire spread simulation
4) Forecasts fire spread 15 minutes into the future
5) Exports results as GeoJSON files

This model is designed to be run through forecasts_main.py:
    python forecasts_main.py --model baseline --forecast_minutes 15
"""

import os
from typing import Tuple

import numpy as np
import polars as pl

# Handle both relative imports (when used as module) and absolute imports (when run as script)
try:
    from ..helpers.data_loader import load_wildfire_data
    from ..helpers.export_geojson import (
        create_combined_forecast_geojson,
    )
except ImportError:
    # Fallback for when running as script
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from helpers.data_loader import load_wildfire_data
    from helpers.export_geojson import create_combined_forecast_geojson


class FireForecastModel:
    """Baseline fire forecasting model using Cellular Automata."""

    def __init__(self, data_path="../output/wildfire_integrated.parquet"):
        """
        Initialize the baseline model.

        Args:
            data_path (str): Path to the parquet file
        """
        self.data_path = data_path
        self.df = None
        self.clusters = None
        self.cluster_grids = {}
        self.model_name = "baseline"

    def load_data(self):
        """Load wildfire data from parquet file using polars."""
        self.df = load_wildfire_data(self.data_path)
        return self.df

    def generate_cluster_grids(self):
        """
        Generate 1 km × 1 km grids covering ±10 km around each cluster center.

        Returns:
            dict: cluster_grids mapping cluster_id → dict with lat/lon arrays
        """
        # Get unique clusters with polars
        clusters_df = self.df.select(
            ["fire_cluster_id", "cluster_center_lat", "cluster_center_lon"]
        ).unique()

        self.cluster_grids = {}

        for row in clusters_df.iter_rows(named=True):
            cid = row["fire_cluster_id"]
            lat0 = row["cluster_center_lat"]
            lon0 = row["cluster_center_lon"]

            # Degrees per km
            deg_per_km_lat = 1.0 / 111.0
            deg_per_km_lon = 1.0 / (111.0 * np.cos(np.deg2rad(lat0)))

            # ±10 km box in degrees
            dlat_half = 10 * deg_per_km_lat
            dlon_half = 10 * deg_per_km_lon

            lat_min = lat0 - dlat_half
            lat_max = lat0 + dlat_half
            lon_min = lon0 - dlon_half
            lon_max = lon0 + dlon_half

            # Step sizes for 1 km grid
            lat_step = deg_per_km_lat
            lon_step = deg_per_km_lon

            # Build arrays of cell-center coordinates
            lats = np.arange(lat_min + lat_step / 2, lat_max, lat_step)
            lons = np.arange(lon_min + lon_step / 2, lon_max, lon_step)

            mesh_lon, mesh_lat = np.meshgrid(lons, lats)

            self.cluster_grids[cid] = {
                "lat_centers": mesh_lat.ravel(),
                "lon_centers": mesh_lon.ravel(),
                "lats": lats,
                "lons": lons,
                "cluster_center": (lat0, lon0),
            }

        return self.cluster_grids

    def get_cluster_wind_data(self, cluster_id: int) -> Tuple[float, float]:
        """
        Get wind speed and direction for a specific cluster.

        Args:
            cluster_id (int): The fire cluster ID

        Returns:
            tuple: (wind_speed_m_s, wind_direction_radians)
        """
        cluster_data = (
            self.df.filter(pl.col("fire_cluster_id") == cluster_id)
            .select(["current_wind_speed_10m", "current_wind_direction_10m"])
            .head(1)
        )

        if len(cluster_data) == 0:
            raise ValueError(f"No data found for cluster {cluster_id}")

        wind_speed = cluster_data["current_wind_speed_10m"][0]
        wind_direction_deg = cluster_data["current_wind_direction_10m"][0]
        wind_direction_rad = np.deg2rad(wind_direction_deg)

        return wind_speed, wind_direction_rad

    def simulate_cluster_fire(
        self,
        cluster_id: int,
        p0: float = 0.1,
        alpha: float = 0.02,
        max_steps: int = 100,
    ):
        """
        Simulate fire spread for a specific cluster using Cellular Automata.

        Args:
            cluster_id (int): The fire cluster ID
            p0 (float): Base ignition probability
            alpha (float): Wind influence factor
            max_steps (int): Maximum CA iterations

        Returns:
            tuple: (final_grid, lats, lons, wind_u, wind_theta)
        """
        if cluster_id not in self.cluster_grids:
            raise ValueError(f"Cluster {cluster_id} not found in grids")

        grid_data = self.cluster_grids[cluster_id]
        lats = grid_data["lats"]
        lons = grid_data["lons"]
        lat0, lon0 = grid_data["cluster_center"]

        # Get wind data for this cluster
        U_cluster, θw_cluster = self.get_cluster_wind_data(cluster_id)

        nlat, nlon = len(lats), len(lons)

        # Initialize CA grid (0=unburned, 1=burning, 2=burned)
        grid = np.zeros((nlat, nlon), dtype=np.uint8)

        # Ignite cell closest to cluster center
        i0 = np.argmin(np.abs(lats - lat0))
        j0 = np.argmin(np.abs(lons - lon0))
        grid[i0, j0] = 1  # burning

        # Build uniform wind arrays
        wind_u = np.full((nlat, nlon), U_cluster, dtype=float)
        wind_theta = np.full((nlat, nlon), θw_cluster, dtype=float)

        # Precompute neighbor offsets & bearings
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        bearings = [
            135 * np.pi / 180,
            180 * np.pi / 180,
            225 * np.pi / 180,
            90 * np.pi / 180,
            0 * np.pi / 180,
            45 * np.pi / 180,
            270 * np.pi / 180,
            315 * np.pi / 180,
        ]

        def step_CA(current_grid):
            """Perform one CA iteration."""
            new_grid = current_grid.copy()
            for i in range(nlat):
                for j in range(nlon):
                    if current_grid[i, j] != 1:
                        continue
                    U = wind_u[i, j]
                    θw = wind_theta[i, j]
                    for (dx, dy), θi in zip(offsets, bearings):
                        ni, nj = i + dx, j + dy
                        if (0 <= ni < nlat) and (0 <= nj < nlon):
                            if current_grid[ni, nj] == 0:
                                φ = np.cos(θi - θw)
                                P = p0 * max(0, 1 + alpha * U * φ)
                                if np.random.rand() < P:
                                    new_grid[ni, nj] = 1
                    new_grid[i, j] = 2
            return new_grid

        # Run CA simulation
        for _ in range(max_steps):
            if not (grid == 1).any():
                break
            grid = step_CA(grid)

        return grid, lats, lons, wind_u, wind_theta

    def forecast_grid(
        self,
        current_grid,
        wind_u,
        wind_theta,
        steps_to_forecast: int,
        p0: float = 0.1,
        alpha: float = 0.02,
    ):
        """
        Forecast fire spread for a given number of steps into the future.

        Args:
            current_grid: Current CA grid state
            wind_u: Wind speed array
            wind_theta: Wind direction array
            steps_to_forecast (int): Number of steps to forecast
            p0 (float): Base ignition probability
            alpha (float): Wind influence factor

        Returns:
            np.ndarray: Future grid state
        """
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        bearings = [
            135 * np.pi / 180,
            180 * np.pi / 180,
            225 * np.pi / 180,
            90 * np.pi / 180,
            0 * np.pi / 180,
            45 * np.pi / 180,
            270 * np.pi / 180,
            315 * np.pi / 180,
        ]
        nlat, nlon = current_grid.shape

        def step_CA_forecast(grid):
            new_grid = grid.copy()
            for i in range(nlat):
                for j in range(nlon):
                    if grid[i, j] != 1:
                        continue
                    U = wind_u[i, j]
                    θw = wind_theta[i, j]
                    for (dx, dy), θi in zip(offsets, bearings):
                        ni, nj = i + dx, j + dy
                        if (0 <= ni < nlat) and (0 <= nj < nlon):
                            if grid[ni, nj] == 0:
                                φ = np.cos(θi - θw)
                                P = p0 * max(0, 1 + alpha * U * φ)
                                if np.random.rand() < P:
                                    new_grid[ni, nj] = 1
                    new_grid[i, j] = 2
            return new_grid

        grid = current_grid.copy()
        for _ in range(steps_to_forecast):
            if not (grid == 1).any():
                break
            grid = step_CA_forecast(grid)

        return grid

    def run_forecast(
        self, forecast_minutes: int = 15, output_dir: str = "forecast_geojson"
    ):
        """
        Run the complete forecasting pipeline for all clusters.

        Args:
            forecast_minutes (int): Minutes to forecast into the future
            output_dir (str): Directory to save GeoJSON files
        """
        if self.df is None:
            self.load_data()

        if not self.cluster_grids:
            self.generate_cluster_grids()

        # Assuming 1 CA step = 1 minute
        steps_to_forecast = forecast_minutes

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get unique cluster IDs
        cluster_ids = self.df["fire_cluster_id"].unique().to_list()

        # Collect data for all clusters
        clusters_data = []

        for cid in cluster_ids:
            try:
                # Simulate initial fire state (first few steps only for "current")
                initial_steps = min(
                    5, forecast_minutes // 4
                )  # Use 25% of forecast time for current
                current_grid, lats, lons, wind_u, wind_theta = (
                    self.simulate_cluster_fire(
                        cluster_id=cid, p0=0.1, alpha=0.02, max_steps=initial_steps
                    )
                )

                # For forecast, continue from current state with more aggressive spread
                forecast_grid = self.forecast_grid(
                    current_grid=current_grid,
                    wind_u=wind_u,
                    wind_theta=wind_theta,
                    steps_to_forecast=steps_to_forecast,
                    p0=0.15,  # Increased base probability for more spread
                    alpha=0.03,  # Increased wind influence
                )

                # Get cluster center coordinates
                grid_data = self.cluster_grids[cid]
                center_lat, center_lon = grid_data["cluster_center"]

                # Store cluster data for combined output
                cluster_data = {
                    "cluster_id": cid,
                    "center_lat": center_lat,
                    "center_lon": center_lon,
                    "current_grid": current_grid,
                    "forecast_grid": forecast_grid,
                    "lats": lats,
                    "lons": lons,
                    "forecast_minutes": forecast_minutes,
                }
                clusters_data.append(cluster_data)

                print(f"Processed cluster {cid}")

            except Exception as e:
                print(f"Error processing cluster {cid}: {e}")
                continue

        # Create combined GeoJSON with all cluster data
        combined_output_path = os.path.join(
            output_dir, f"wildfire_forecast_{forecast_minutes}min.geojson"
        )

        total_features = create_combined_forecast_geojson(
            clusters_data, combined_output_path
        )

        print(
            f"Created combined forecast with {total_features} features: {combined_output_path}"
        )
        print(
            f"Processed {len(clusters_data)} clusters. GeoJSON file saved to: {output_dir}"
        )

        return clusters_data

    def get_model_info(self):
        """Get information about this model implementation."""
        return {
            "name": self.model_name,
            "description": "Baseline Cellular Automata fire forecasting model",
            "algorithm": "Cellular Automata with wind influence",
            "parameters": {
                "p0": "Base ignition probability",
                "alpha": "Wind influence factor",
                "max_steps": "Maximum CA simulation steps",
            },
        }
