#!/usr/bin/env python3
"""
Wildfire Forecast Visualization

Creates interactive maps showing wildfire forecast results from GeoJSON data.
Displays cluster centers, current fire areas, and forecasted fire spread areas.

Usage:
    python plots_main.py --input forecast_geojson/wildfire_forecast_15min.geojson
    python plots_main.py --input forecast_geojson/wildfire_forecast_15min.geojson --output plots/
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict

import folium
import geopandas as gpd


def load_geojson_data(filepath: str) -> gpd.GeoDataFrame:
    """
    Load GeoJSON forecast data into a GeoDataFrame.

    Args:
        filepath (str): Path to the GeoJSON file

    Returns:
        gpd.GeoDataFrame: Loaded forecast data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"GeoJSON file not found: {filepath}")

    print(f"Loading forecast data from: {filepath}")
    gdf = gpd.read_file(filepath)
    print(f"Loaded {len(gdf)} features")

    # Print summary of feature types
    if "feature_type" in gdf.columns:
        centers = len(gdf[gdf["feature_type"] == "cluster_center"])
        print(f"  - {centers} cluster centers")

    if "area_type" in gdf.columns:
        current_areas = len(gdf[gdf["area_type"] == "current"])
        forecast_areas = len(gdf[gdf["area_type"] == "forecast"])
        print(f"  - {current_areas} current fire areas")
        print(f"  - {forecast_areas} forecast fire areas")

    return gdf


def create_interactive_map(
    gdf: gpd.GeoDataFrame, output_path: str = "forecast_map.html"
) -> folium.Map:
    """
    Create an interactive Folium map showing the wildfire forecast.

    Args:
        gdf (gpd.GeoDataFrame): Forecast data
        output_path (str): Path to save the HTML map

    Returns:
        folium.Map: Interactive map object
    """
    print("Creating interactive map...")

    # Calculate map center from all geometries
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=6, tiles="OpenStreetMap"
    )

    # Add different tile layers
    folium.TileLayer(
        "Stamen Terrain",
        name="Terrain",
        attr="Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.",
    ).add_to(m)
    folium.TileLayer(
        "CartoDB positron", name="Light", attr="© CartoDB, © OpenStreetMap contributors"
    ).add_to(m)

    # Color scheme
    colors = {
        "cluster_center": "#FF0000",  # Red for cluster centers
        "current": "#FFA500",  # Orange for current fire areas
        "forecast": "#FF9999",  # Light red for forecast areas
    }

    # Add cluster centers
    if "feature_type" in gdf.columns:
        centers = gdf[gdf["feature_type"] == "cluster_center"]
        for idx, row in centers.iterrows():
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=8,
                popup=f"Cluster {row['cluster_id']} Center",
                color="darkred",
                fillColor=colors["cluster_center"],
                fillOpacity=0.8,
                weight=2,
            ).add_to(m)

    # Add fire areas in order: forecast first, then current on top
    if "area_type" in gdf.columns:
        # Add forecast fire areas first (background)
        forecast_areas = gdf[gdf["area_type"] == "forecast"]
        for idx, row in forecast_areas.iterrows():
            geom = row.geometry.__geo_interface__
            forecast_minutes = row.get("forecast_minutes", "Unknown")
            folium.GeoJson(
                geom,
                style_function=lambda x: {
                    "fillColor": colors["forecast"],
                    "color": "red",
                    "weight": 1,
                    "fillOpacity": 0.4,
                    "dashArray": "5, 5",
                },
                popup=f"Cluster {row['cluster_id']} - {forecast_minutes}min Forecast",
                tooltip=f"Forecast Fire (Cluster {row['cluster_id']}, {forecast_minutes}min)",
            ).add_to(m)

        # Add current fire areas on top (foreground) with distinct styling
        current_areas = gdf[gdf["area_type"] == "current"]
        for idx, row in current_areas.iterrows():
            # Convert to GeoJSON for folium
            geom = row.geometry.__geo_interface__
            folium.GeoJson(
                geom,
                style_function=lambda x: {
                    "fillColor": colors["current"],
                    "color": "darkorange",
                    "weight": 3,
                    "fillOpacity": 0.8,
                    "dashArray": None,  # Solid line for current areas
                },
                popup=f"Cluster {row['cluster_id']} - Current Fire Area",
                tooltip=f"Current Fire (Cluster {row['cluster_id']})",
            ).add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 220px; height: 140px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>Wildfire Forecast Legend</h4>
    <p><i class="fa fa-circle" style="color:red"></i> Cluster Centers</p>
    <p><i class="fa fa-square" style="color:orange; border: 2px solid darkorange;"></i> Current Fire Areas (Solid)</p>
    <p><i class="fa fa-square" style="color:lightcoral; border: 1px dashed red;"></i> Forecast Areas (Dashed)</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save map
    print(f"Saving interactive map to: {output_path}")
    m.save(output_path)

    return m


def get_forecast_summary(gdf: gpd.GeoDataFrame) -> Dict:
    """
    Generate a summary of the forecast data.

    Args:
        gdf (gpd.GeoDataFrame): Forecast data

    Returns:
        dict: Summary statistics
    """
    summary = {
        "total_features": len(gdf),
        "total_clusters": 0,
        "current_fire_areas": 0,
        "forecast_areas": 0,
        "forecast_minutes": None,
        "geographic_bounds": None,
    }

    # Count cluster centers
    if "cluster_id" in gdf.columns:
        summary["total_clusters"] = gdf["cluster_id"].nunique()

    # Count area types
    if "area_type" in gdf.columns:
        summary["current_fire_areas"] = len(gdf[gdf["area_type"] == "current"])
        summary["forecast_areas"] = len(gdf[gdf["area_type"] == "forecast"])

    # Get forecast duration
    if "forecast_minutes" in gdf.columns:
        forecast_minutes = gdf["forecast_minutes"].dropna()
        if not forecast_minutes.empty:
            summary["forecast_minutes"] = forecast_minutes.iloc[0]

    # Get geographic bounds
    bounds = gdf.total_bounds
    summary["geographic_bounds"] = {
        "min_lon": bounds[0],
        "min_lat": bounds[1],
        "max_lon": bounds[2],
        "max_lat": bounds[3],
    }

    return summary


def main():
    """Main function for the plotting script."""
    parser = argparse.ArgumentParser(
        description="Visualize wildfire forecast data on interactive maps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plots_main.py --input forecast_geojson/wildfire_forecast_15min.geojson
  python plots_main.py --input forecast_geojson/wildfire_forecast_15min.geojson --output plots/
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        default="../forecasts/forecast_geojson/wildfire_forecast_15min.geojson",
        help="Path to input GeoJSON file (default: ../forecasts/forecast_geojson/wildfire_forecast_15min.geojson)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="plots/",
        help="Output directory for interactive HTML map (default: plots/)",
    )

    args = parser.parse_args()

    try:
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load the GeoJSON data
        gdf = load_geojson_data(args.input)

        # Print summary
        summary = get_forecast_summary(gdf)
        print("\n=== Forecast Data Summary ===")
        print(f"Total features: {summary['total_features']}")
        print(f"Total clusters: {summary['total_clusters']}")
        print(f"Current fire areas: {summary['current_fire_areas']}")
        print(f"Forecast areas: {summary['forecast_areas']}")
        if summary["forecast_minutes"]:
            print(f"Forecast duration: {summary['forecast_minutes']} minutes")

        bounds = summary["geographic_bounds"]
        print(
            f"Geographic extent: {bounds['min_lat']:.2f}°N to {bounds['max_lat']:.2f}°N, "
            f"{bounds['min_lon']:.2f}°E to {bounds['max_lon']:.2f}°E"
        )

        # Create interactive map
        print("\n=== Creating Interactive Map ===")
        interactive_path = output_dir / "wildfire_forecast_interactive.html"
        create_interactive_map(gdf, str(interactive_path))

        print("\n=== Visualization Complete ===")
        print(f"Interactive map saved to: {interactive_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
