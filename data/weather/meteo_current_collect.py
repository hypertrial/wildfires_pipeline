"""
Weather Current Data Collection Helper Module

Collects current meteorological data at active fire locations
using the Open-Meteo API.
"""

import random
import time

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

# Comprehensive list of meteorological variables to collect
CURRENT_LIST = [
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
CURRENT = ",".join(CURRENT_LIST)
CHUNK = 30  # Reduced chunk size to avoid rate limits
RATE_S = 5.0  # Increased pause between API calls (seconds)
MAX_RETRIES = 3  # Maximum number of retries for failed requests


def fetch(chunk, retry_count=0):
    """Fetch weather current data for a chunk of fire locations with retry logic."""
    lats, lons = zip(*chunk)
    params = {
        "latitude": ",".join(map(str, lats)),
        "longitude": ",".join(map(str, lons)),
        "current": CURRENT,
        "timezone": "auto",
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


def collect_current_weather(fire_locations: pl.DataFrame) -> pl.DataFrame:
    """Collect and process current weather data directly at fire locations."""
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

    # Collect current data for all fire locations
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
            current = r.Current()

            # Get current timestamp and weather data
            current_ts = pl.from_epoch([current.Time()], time_unit="s")[0]

            # Extract all weather variables for the current time
            vals = [
                current.Variables(j).Value() for j in range(current.VariablesLength())
            ]
            if len(vals) != len(CURRENT_LIST):
                continue  # Skip if we didn't get all expected variables

            # Create a structured record with all current variables
            records.append(
                {
                    "latitude": r.Latitude(),
                    "longitude": r.Longitude(),
                    "timestamp": current_ts,
                    **dict(zip(CURRENT_LIST, vals)),
                }
            )

        # Implement rate limiting between chunks with some randomization
        if i + CHUNK < len(fire_coords_list):
            wait_time = RATE_S + random.uniform(0, 2)  # Add some jitter
            print(f"   Waiting {wait_time:.1f} seconds before next chunk...")
            time.sleep(wait_time)

    # Create DataFrame from collected records
    weather_df = pl.DataFrame(records)

    print(
        f"   Successfully collected current data for {len(weather_df)} fire locations"
    )
    return weather_df
