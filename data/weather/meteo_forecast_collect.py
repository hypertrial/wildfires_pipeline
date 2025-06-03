"""
Weather Forecast Data Collection Helper Module

Collects hourly forecast meteorological data at active fire locations
using the Open-Meteo API for next 24-168 hours.
"""

import random
import time
from datetime import datetime, timezone

import openmeteo_requests
import polars as pl
import requests_cache
from retry_requests import retry

# Configure Open-Meteo client with caching and retry capability
cache = requests_cache.CachedSession(".cache", expire_after=3600)  # 1-hour cache
session = retry(cache, retries=5, backoff_factor=0.5)  # Exponential backoff
om = openmeteo_requests.Client(session=session)

# Open-Meteo API configuration
URL = "https://api.open-meteo.com/v1/forecast"

# Comprehensive list of meteorological variables to collect for forecasting
FORECAST_LIST = [
    "temperature_2m",  # Air temperature at 2m above ground (degrees C)
    "apparent_temperature",  # Feels-like temperature (degrees C)
    "relative_humidity_2m",  # Relative humidity at 2m (%)
    "dew_point_2m",  # Dew point temperature (degrees C)
    "vapour_pressure_deficit",  # Vapor pressure deficit (kPa)
    "wind_speed_10m",  # Wind speed at 10m (km/h)
    "wind_direction_10m",  # Wind direction at 10m (degrees)
    "wind_gusts_10m",  # Wind gusts at 10m (km/h)
    "precipitation",  # Precipitation amount (mm)
    "shortwave_radiation",  # Solar radiation (W/m^2)
    "cloud_cover",  # Total cloud cover (%)
    "surface_pressure",  # Surface pressure (hPa)
    "visibility",  # Visibility (m)
    "is_day",  # Daylight (1) or night (0)
    "weather_code",  # WMO weather code
]
FORECAST = ",".join(FORECAST_LIST)
CHUNK = 30  # Reduced chunk size to avoid rate limits
RATE_S = 5.0  # Increased pause between API calls (seconds)
MAX_RETRIES = 3  # Maximum number of retries for failed requests


def fetch(chunk, retry_count=0):
    """Fetch weather forecast data for a chunk of fire locations with retry logic."""
    lats, lons = zip(*chunk)
    params = {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, lons)),
        "hourly": FORECAST,
        "timezone": "auto",
        "forecast_days": 1,  # 24 hours forecast
    }

    try:
        return om.weather_api(URL, params=params)
    except Exception as e:
        error_str = str(e)
        print(f"[ERR] {e} for chunk starting {chunk[0]}")

        # Check if it's a rate limit error
        if "limit exceeded" in error_str.lower() or "rate limit" in error_str.lower():
            if retry_count < MAX_RETRIES:
                # Exponential backoff with jitter
                wait_time = (2**retry_count) * 30 + random.uniform(
                    0, 30
                )  # 60s, 120s, 240s + jitter
                print(
                    f"[RETRY] Rate limit hit. Waiting {wait_time:.1f} seconds before retry {retry_count + 1}/{MAX_RETRIES}"
                )
                time.sleep(wait_time)
                return fetch(chunk, retry_count + 1)
            else:
                print(f"[SKIP] Max retries exceeded for chunk starting {chunk[0]}")
                return []
        else:
            # For other errors, don't retry
            return []


def collect_forecast_weather(fire_locations: pl.DataFrame) -> pl.DataFrame:
    """Collect and process hourly forecast weather data at fire locations."""
    if fire_locations.is_empty():
        return pl.DataFrame()

    # Extract unique fire coordinates
    fire_coords = fire_locations.select(["latitude", "longitude"]).unique()
    fire_coords_list = [(row[0], row[1]) for row in fire_coords.to_numpy()]

    print(
        f"   Processing {len(fire_coords_list)} unique fire locations in chunks of {CHUNK}"
    )
    print(
        f"   Estimated time: {len(fire_coords_list) // CHUNK * RATE_S / 60:.1f} minutes"
    )

    # Collect forecast data for all fire locations
    records = []
    total_chunks = (len(fire_coords_list) + CHUNK - 1) // CHUNK

    for i in range(0, len(fire_coords_list), CHUNK):
        chunk_num = i // CHUNK + 1
        chunk = fire_coords_list[i : i + CHUNK]

        print(
            f"   Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} locations)..."
        )

        responses = fetch(chunk)
        for r in responses:
            try:
                forecast = r.Hourly()

                # Get forecast timestamps
                forecast_times = forecast.Time()
                if forecast_times is None:
                    continue

                # Check if we got the expected number of variables
                if forecast.VariablesLength() != len(FORECAST_LIST):
                    continue

                # Handle different types of forecast_times data
                try:
                    # Check if forecast_times is iterable (array/list) or single value
                    if hasattr(forecast_times, "__iter__") and not isinstance(
                        forecast_times, (str, bytes)
                    ):
                        # It's an array/list of timestamps
                        forecast_times_list = list(forecast_times)
                    else:
                        # It's a single timestamp - this might be the start time
                        # We need to generate the hourly timestamps
                        start_time = int(forecast_times)
                        # Generate 24 hourly timestamps (3600 seconds = 1 hour)
                        forecast_times_list = [
                            start_time + (i * 3600) for i in range(24)
                        ]

                    forecast_ts = pl.from_epoch(forecast_times_list, time_unit="s")
                except Exception as e:
                    print(f"[WARN] Error converting forecast times: {e}")
                    continue

                # Filter forecast data to next 24 hours only
                now = datetime.now(timezone.utc)
                next_24h_timestamp = now.timestamp() + (24 * 3600)  # 24 hours from now

                # For efficiency, only include forecast hours in next 24 hours
                for t in range(len(forecast_ts)):
                    ts = forecast_ts[t]
                    ts_timestamp = ts.timestamp() if hasattr(ts, "timestamp") else ts

                    # Check if timestamp is within next 24 hours
                    if now.timestamp() <= ts_timestamp <= next_24h_timestamp:
                        # Extract all weather variables for this forecast time
                        try:
                            hour_vals = []
                            for j in range(forecast.VariablesLength()):
                                var_data = forecast.Variables(j).ValuesAsNumpy()
                                # Handle both single values and arrays
                                if hasattr(var_data, "__len__") and len(var_data) > t:
                                    hour_vals.append(float(var_data[t]))
                                elif (
                                    hasattr(var_data, "__len__") and len(var_data) == 1
                                ):
                                    hour_vals.append(float(var_data[0]))
                                else:
                                    # Single value for all hours
                                    hour_vals.append(float(var_data))
                        except Exception as e:
                            print(f"[WARN] Error extracting weather values: {e}")
                            continue

                        # Only add record if we have the right number of values
                        if len(hour_vals) == len(FORECAST_LIST):
                            # Create a structured record with all forecast variables
                            records.append(
                                {
                                    "latitude": r.Latitude(),
                                    "longitude": r.Longitude(),
                                    "forecast_timestamp": ts,
                                    "forecast_hour": t,
                                    **dict(zip(FORECAST_LIST, hour_vals)),
                                }
                            )

            except Exception as e:
                print(f"[WARN] Error processing response: {e}")
                continue

        # Implement rate limiting between chunks with some randomization
        if i + CHUNK < len(fire_coords_list):
            wait_time = RATE_S + random.uniform(0, 2)  # Add some jitter
            print(f"   Waiting {wait_time:.1f} seconds before next chunk...")
            time.sleep(wait_time)

    # Create DataFrame from collected records
    weather_df = pl.DataFrame(records)

    print(
        f"   Successfully collected forecast data for {len(weather_df)} time-location combinations"
    )
    return weather_df
