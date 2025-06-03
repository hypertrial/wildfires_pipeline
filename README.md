# Wildfire Models Private

A comprehensive wildfire spread analysis and forecasting system that integrates NASA FIRMS fire detection data with Open-Meteo weather data to predict wildfire behavior using Cellular Automata models.

## Overview

This repository provides a complete pipeline for:

- **Data Collection**: Gathering fire detection and weather data
- **Data Integration**: Merging fire locations with meteorological conditions
- **Forecasting**: Predicting fire spread using Cellular Automata models
- **Visualization**: Creating interactive maps of forecast results
- **Exploration**: Interactive data analysis using Marimo notebooks

## Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd wildfires_models_private
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Main Components

### 1. Data Collection & Integration

**Purpose**: Collects NASA FIRMS fire detection data and integrates it with Open-Meteo weather data.

**Command**:

```bash
python data/data_main.py
```

**Output**:

- `wildfire_integrated.parquet` - Combined dataset with fire locations and weather conditions
- Validation report and summary statistics

**What it does**:

- Collects high-confidence fire detection data from satellite sources
- Gathers current and forecast meteorological data at fire cluster centers
- Merges weather conditions with fire locations
- Validates and saves the integrated dataset

### 2. Wildfire Forecasting

**Purpose**: Runs wildfire spread forecasting using Cellular Automata models.

**Basic Command**:

```bash
python forecasts/forecasts_main.py
```

**Advanced Commands**:

```bash
# 15-minute forecast (default)
python forecasts/forecasts_main.py --model baseline --forecast_minutes 15

# 30-minute forecast
python forecasts/forecasts_main.py --model baseline --forecast_minutes 30

# Custom output directory
python forecasts/forecasts_main.py --model baseline --forecast_minutes 15 --output_dir results

# Custom data file
python forecasts/forecasts_main.py --model baseline --data_path /path/to/data.parquet
```

**Output**:

- `forecast_geojson/wildfire_forecast_{minutes}min.geojson` - GeoJSON file with forecast results
- Console output with forecast summary

**What it does**:

- Loads wildfire data from parquet file
- Generates 1km×1km spatial grids around fire cluster centers
- Runs Cellular Automata simulation with wind influence
- Exports forecast results as GeoJSON for visualization

### 3. Visualization & Plotting

**Purpose**: Creates interactive maps showing wildfire forecast results.

**Basic Command**:

```bash
python plots/plots_main.py
```

**Advanced Commands**:

```bash
# Specify input GeoJSON file
python plots/plots_main.py --input forecast_geojson/wildfire_forecast_15min.geojson

# Custom output directory
python plots/plots_main.py --input forecast_geojson/wildfire_forecast_15min.geojson --output plots/
```

**Output**:

- `plots/wildfire_forecast_interactive.html` - Interactive Folium map

**What it does**:

- Loads GeoJSON forecast data
- Creates interactive maps with cluster centers, current fire areas, and forecast areas
- Provides legend and summary statistics

### 4. Interactive Data Explorer

**Purpose**: Interactive Jupyter-like notebook for data exploration and analysis using Marimo.

**Command**:

```bash
python explorer.py
```

**Alternative (if Marimo is installed globally)**:

```bash
marimo run explorer.py
```

**What it does**:

- Loads and explores the integrated wildfire dataset
- Provides interactive data analysis capabilities
- Demonstrates Cellular Automata fire spread simulation
- Shows cluster analysis and grid generation

## Complete Workflow

To run the complete analysis pipeline:

1. **Collect and integrate data**:

   ```bash
   python data/data_main.py
   ```

2. **Generate forecasts**:

   ```bash
   python forecasts/forecasts_main.py --model baseline --forecast_minutes 15
   ```

3. **Create visualizations**:

   ```bash
   python plots/plots_main.py --input forecast_geojson/wildfire_forecast_15min.geojson
   ```

4. **Explore data interactively**:
   ```bash
   python explorer.py
   ```

## Directory Structure

```
├── data/                          # Data collection and integration
│   ├── data_main.py              # Main data integration script
│   ├── wildfire_integrated.parquet # Integrated dataset
│   ├── weather/                  # Weather data collection modules
│   ├── firms/                    # Fire detection data modules
│   ├── merging/                  # Data merging utilities
│   ├── utils/                    # Data processing utilities
│   └── validation/               # Data validation modules
├── forecasts/                     # Forecasting models and scripts
│   ├── forecasts_main.py         # Main forecasting script
│   ├── models/                   # Forecasting model implementations
│   ├── helpers/                  # Forecasting utilities
│   └── forecast_geojson/         # Output GeoJSON files
├── plots/                         # Visualization and plotting
│   ├── plots_main.py             # Main plotting script
│   └── wildfire_forecast_interactive.html # Interactive map
├── explorer.py                    # Interactive Marimo notebook
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Key Features

- **Real-time Data**: Integrates live NASA FIRMS fire detection data
- **Weather Integration**: Incorporates current and forecast meteorological conditions
- **Cellular Automata**: Uses CA models for realistic fire spread simulation
- **Wind Influence**: Accounts for wind speed and direction in spread predictions
- **High Resolution**: 1km×1km spatial grid resolution
- **Interactive Visualization**: Folium-based interactive maps
- **Flexible Forecasting**: Configurable forecast duration and parameters

## Dependencies

Key packages (see `requirements.txt` for complete list):

- `polars` - Fast data processing
- `numpy` - Numerical computations
- `marimo` - Interactive notebooks
- `scikit-learn` - Machine learning utilities
- `folium` - Interactive mapping
- `geopandas` - Geospatial data processing
- `matplotlib` - Static plotting
- `requests` - API data collection
