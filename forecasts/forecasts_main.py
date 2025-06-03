#!/usr/bin/env python3
"""
Wildfire Forecasting - Main Entry Point

Provides the main interface for running wildfire spread forecasting models.
This module integrates with the data collection pipeline and provides
forecasting capabilities for wildfire spread prediction.

Usage:
    python forecasts_main.py --model baseline --forecast_minutes 15
    python forecasts_main.py --model baseline --forecast_minutes 30 --output_dir custom_output
"""

import argparse
import os
import sys
from typing import Optional

# Handle both relative imports (when used as module) and absolute imports (when run as script)
try:
    from .models.baseline import FireForecastModel
except ImportError:
    from models.baseline import FireForecastModel


def create_model(model_name: str, data_path: Optional[str] = None) -> FireForecastModel:
    """
    Create and initialize a forecasting model.

    Args:
        model_name (str): Name of the model to create ('baseline')
        data_path (str, optional): Path to the wildfire data file

    Returns:
        FireForecastModel: Initialized model instance

    Raises:
        ValueError: If model_name is not supported
    """
    if model_name.lower() == "baseline":
        if data_path:
            return FireForecastModel(data_path=data_path)
        else:
            return FireForecastModel()
    else:
        raise ValueError(f"Unsupported model: {model_name}. Available models: baseline")


def run_forecast(
    model_name: str = "baseline",
    forecast_minutes: int = 15,
    data_path: Optional[str] = None,
    output_dir: str = "../output",
):
    """
    Run wildfire forecasting for the specified model and parameters.

    Args:
        model_name (str): Name of the forecasting model to use
        forecast_minutes (int): Minutes to forecast into the future
        data_path (str, optional): Path to wildfire data file
        output_dir (str): Directory to save forecast results
    """
    print("=== Wildfire Forecasting System ===\n")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create and initialize the model
    print(f"1. Initializing {model_name} forecasting model...")
    model = create_model(model_name, data_path)

    # Load the data
    print("2. Loading wildfire data...")
    df = model.load_data()
    print(f"   Loaded {len(df)} fire detection records")

    # Generate spatial grids
    print("3. Generating spatial grids for fire clusters...")
    cluster_grids = model.generate_cluster_grids()
    print(f"   Created grids for {len(cluster_grids)} fire clusters")

    # Run forecasting
    print(f"4. Running {forecast_minutes}-minute fire spread forecast...")
    clusters_data = model.run_forecast(
        forecast_minutes=forecast_minutes, output_dir=output_dir
    )

    # Display model information
    model_info = model.get_model_info()
    print("\n=== Forecast Complete ===")
    print(f"Model: {model_info['name']} - {model_info['description']}")
    print(f"Forecast duration: {forecast_minutes} minutes")
    print(f"Processed {len(clusters_data)} fire clusters")
    print(
        f"Combined results saved to: {output_dir}/wildfire_forecast_{forecast_minutes}min.geojson"
    )

    return model


def main():
    """Main command-line interface for the forecasting system."""
    parser = argparse.ArgumentParser(
        description="Wildfire spread forecasting system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python forecasts_main.py --model baseline --forecast_minutes 15
  python forecasts_main.py --model baseline --forecast_minutes 30 --output_dir results
  python forecasts_main.py --model baseline --data_path /path/to/data.parquet
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        help="Forecasting model to use (default: baseline)",
    )

    parser.add_argument(
        "--forecast_minutes",
        type=int,
        default=15,
        help="Minutes to forecast into the future (default: 15)",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to wildfire data parquet file (optional)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output",
        help="Directory to save forecast results (default: ../output)",
    )

    args = parser.parse_args()

    try:
        run_forecast(
            model_name=args.model,
            forecast_minutes=args.forecast_minutes,
            data_path=args.data_path,
            output_dir=args.output_dir,
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
