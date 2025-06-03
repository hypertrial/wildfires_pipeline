# Wildfires Pipeline 

A wildfire spread analysis and forecasting system developed by **Hypertrial.ai**. This project is currently in its early phase.

Integrates NASA FIRMS fire detection data with Open-Meteo weather data to predict wildfire behavior using Cellular Automata models.

## Interactive Map 

The forecasting results can be visualized using our interactive web application: **[Wildfire Spread Interactive Map](https://github.com/hypertrial/wildfires_map)**

This companion product provides:
- Interactive Leaflet.js-based mapping
- Real-time wildfire forecast visualization
- Modern web interface for exploring forecast data

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the complete pipeline**:
   ```bash
   # Collect and integrate data
   python data/data_main.py
   
   # Generate forecasts
   python forecasts/forecasts_main.py
   
   # Create visualizations
   python plots/plots_main.py
   
   # Explore data interactively
   python explorer.py
   ```

## Components

- **Data Collection**: Gathers fire detection and weather data
- **Forecasting**: Predicts fire spread using Cellular Automata models  
- **Visualization**: Creates interactive maps of forecast results
- **Exploration**: Interactive data analysis using Marimo notebooks

## Key Features

- Real-time NASA FIRMS fire detection data
- Weather integration with current and forecast conditions
- Cellular Automata fire spread simulation with wind influence
- 1km×1km spatial grid resolution
- Interactive Folium-based maps

## Directory Structure

```
├── data/                    # Data collection and integration
├── forecasts/              # Forecasting models and scripts
├── plots/                  # Visualization and plotting
├── explorer.py             # Interactive Marimo notebook
└── requirements.txt        # Python dependencies
```

---
*Developed by [Hypertrial.ai](https://hypertrial.ai) - Early phase project*
