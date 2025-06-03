"""
FIRMS Fire Detection Data Collection Helper Module

Collects active fire detections from NASA's FIRMS API for North America.
Returns high-confidence fire detection data from multiple satellite sources.
"""

import concurrent.futures as cf
import io
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import polars as pl
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .clustering import (
    DEFAULT_CLUSTER_DISTANCE_KM,
    add_clustering_columns_to_schema,
    calculate_fire_clusters,
)

# Load environment variables from .env file
load_dotenv()

# API configuration
API_KEY = os.getenv("FIRMS_MAP_KEY")
if not API_KEY:
    raise ValueError(
        "FIRMS_MAP_KEY environment variable is required. Please copy .env-example to .env and configure your API key."
    )
DAY = "2"  # Fetch detections from last 24 hours (most recent data)
BASE = "https://firms.modaps.eosdis.nasa.gov/usfs/api/country/csv"

# Countries to query
COUNTRIES = ["USA", "CAN"]  # United States and Canada

# Satellite data sources to query
SOURCES = [
    "VIIRS_SNPP_NRT",  # VIIRS instrument on Suomi NPP satellite
    "VIIRS_NOAA20_NRT",  # VIIRS instrument on NOAA-20 satellite
    "VIIRS_NOAA21_NRT",  # VIIRS instrument on NOAA-21 satellite
    "LANDSAT_NRT",  # Landsat satellites (US/Canada only)
    "MODIS_NRT",  # MODIS instrument (lower resolution)
]


def create_session() -> requests.Session:
    """Create a requests session with retry strategy and optimized settings."""
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=3,  # Total number of retries
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry
        backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
        raise_on_status=False,
    )

    # Mount adapter with retry strategy
    adapter = HTTPAdapter(
        max_retries=retry_strategy, pool_connections=10, pool_maxsize=20
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # Set default headers
    session.headers.update(
        {
            "User-Agent": "Wildfire-Models/1.0",
            "Accept": "text/csv",
            "Connection": "keep-alive",
        }
    )

    return session


def fetch_with_retry(
    src: str,
    country: str,
    session: Optional[requests.Session] = None,
    max_retries: int = 3,
) -> pl.DataFrame:
    """Fetch fire detection data with retry logic and improved timeout handling."""
    url = f"{BASE}/{API_KEY}/{src}/{country}/{DAY}"

    if session is None:
        session = create_session()

    for attempt in range(max_retries):
        try:
            # Use separate connect and read timeouts: (connect_timeout, read_timeout)
            # Connect timeout: time to establish connection
            # Read timeout: time to wait for server response
            timeout = (10, 60)  # 10s to connect, 60s to read response

            if attempt > 0:
                # Add exponential backoff delay for retries
                delay = 2**attempt
                print(
                    f"   Retrying {src} ({country}) in {delay}s... (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)

            r = session.get(url, timeout=timeout)
            r.raise_for_status()

            # Check response content
            response_text = r.text
            line_count = response_text.count("\n")

            if line_count <= 1:  # header only → no detections
                print(f"   {src} ({country}): 0 detections (no data)")
                return pl.DataFrame()

            # Parse CSV
            df = pl.read_csv(io.StringIO(response_text))
            print(f"   {src} ({country}): {len(df)} detections")

            return df

        except requests.exceptions.Timeout:
            error_msg = f"Timeout after {timeout[0]}s connect + {timeout[1]}s read"
            if attempt < max_retries - 1:
                print(f"   [TIMEOUT] {src} ({country}): {error_msg} - retrying...")
                continue
            else:
                print(
                    f"   [ERROR] {src} ({country}): {error_msg} - max retries exceeded"
                )
                return pl.DataFrame()

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection failed - {str(e)[:100]}..."
            if attempt < max_retries - 1:
                print(f"   [CONNECTION] {src} ({country}): {error_msg} - retrying...")
                continue
            else:
                print(
                    f"   [ERROR] {src} ({country}): {error_msg} - max retries exceeded"
                )
                return pl.DataFrame()

        except requests.exceptions.HTTPError as e:
            # Don't retry HTTP errors (4xx, 5xx that aren't in retry list)
            print(f"   [ERROR] {src} ({country}): HTTP {e.response.status_code} - {e}")
            return pl.DataFrame()

        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed - {str(e)[:100]}..."
            if attempt < max_retries - 1:
                print(f"   [REQUEST] {src} ({country}): {error_msg} - retrying...")
                continue
            else:
                print(
                    f"   [ERROR] {src} ({country}): {error_msg} - max retries exceeded"
                )
                return pl.DataFrame()

        except Exception as e:
            print(f"   [ERROR] {src} ({country}): Parse error - {e}")
            return pl.DataFrame()

    return pl.DataFrame()


def fetch(src: str, country: str) -> pl.DataFrame:
    """Legacy fetch function - now uses the improved fetch_with_retry."""
    return fetch_with_retry(src, country)


def collect_fire_data(
    cluster_fires: bool = True, cluster_distance_km: float = DEFAULT_CLUSTER_DISTANCE_KM
) -> pl.DataFrame:
    """
    Collect and process high-confidence fire detection data from all satellite sources and countries.
    Only keeps detections from the last 6 hours.

    Args:
        cluster_fires: Whether to group nearby fires into clusters
        cluster_distance_km: Maximum distance in kilometers to group fires together
    """
    print(
        f"Collecting fire data from {len(SOURCES)} sources across {len(COUNTRIES)} countries..."
    )

    # Create a shared session for connection reuse
    session = create_session()

    # Create list of (source, country) combinations for concurrent fetching
    fetch_tasks = [(src, country) for src in SOURCES for country in COUNTRIES]

    # Fetch data from all source-country combinations concurrently with shared session
    with cf.ThreadPoolExecutor(max_workers=6) as pool:  # Limit concurrent requests
        futures = [
            pool.submit(fetch_with_retry, src, country, session)
            for src, country in fetch_tasks
        ]
        dataframes = [future.result() for future in cf.as_completed(futures)]

    # Close the session
    session.close()

    # Filter out empty dataframes
    non_empty_dfs = [df for df in dataframes if not df.is_empty()]

    if not non_empty_dfs:
        print("[ERROR] No fire detection data available!")
        return pl.DataFrame()

    print(f"Consolidating data from {len(non_empty_dfs)} successful fetches...")

    # Use diagonal concatenation to handle different schemas
    df = pl.concat(non_empty_dfs, how="diagonal_relaxed")

    if df.is_empty():
        print("[ERROR] Final consolidated dataframe is empty!")
        return pl.DataFrame()

    # Filter for detections from the last 6 hours only
    initial_count = len(df)
    print(f"Processing {initial_count} raw detections...")

    # Calculate the cutoff time (6 hours ago)
    cutoff_time = datetime.utcnow() - timedelta(hours=6)
    cutoff_date = cutoff_time.strftime("%Y-%m-%d")
    cutoff_hhmm_int = int(
        cutoff_time.strftime("%H%M")
    )  # Convert to integer for comparison

    print(
        f"Filtering for detections after {cutoff_time.strftime('%Y-%m-%d %H:%M')} UTC..."
    )

    # Filter for last 6 hours
    if "acq_date" in df.columns and "acq_time" in df.columns:
        # Ensure acq_time is treated as integer for comparison
        # acq_date is string format YYYY-MM-DD, acq_time is integer HHMM
        df = df.filter(
            (pl.col("acq_date") > cutoff_date)
            | (
                (pl.col("acq_date") == cutoff_date)
                & (pl.col("acq_time").cast(pl.Int64) >= cutoff_hhmm_int)
            )
        )
        six_hour_count = len(df)
        print(
            f"After 6-hour filtering: {six_hour_count} detections (removed {initial_count - six_hour_count} older detections)"
        )
    else:
        print(
            "[WARNING] No acq_date or acq_time columns found - skipping time filtering!"
        )

    if df.is_empty():
        print("[ERROR] No detections found within the last 6 hours!")
        return pl.DataFrame()

    # Filter for high-confidence detections only
    print(f"Applying confidence filtering to {len(df)} detections...")

    # Handle different confidence formats from different satellites
    if "confidence" in df.columns:
        # Create a mask for high-confidence detections
        categorical_mask = (
            df["confidence"]
            .cast(pl.Utf8)
            .str.to_lowercase()
            .is_in(["high", "h", "nominal", "n"])
        )

        # Handle numeric confidence values - first filter to only numeric rows, then convert and check
        is_numeric = (
            df["confidence"].cast(pl.Utf8).str.contains(r"^\d+(\.\d+)?$", literal=False)
        )

        # For numeric values, create a separate dataframe to do the conversion safely
        numeric_rows = df.filter(is_numeric)
        if not numeric_rows.is_empty():
            numeric_high_conf_ids = (
                numeric_rows.with_row_count()
                .filter(
                    numeric_rows["confidence"].cast(pl.Utf8).cast(pl.Float64) >= 90
                )["row_nr"]
                .to_list()
            )
        else:
            numeric_high_conf_ids = []

        # Create numeric mask using row indices
        df = df.with_row_count()
        numeric_mask = df["row_nr"].is_in(numeric_high_conf_ids)

        # Combine masks and filter
        high_confidence_mask = categorical_mask | numeric_mask
        df = df.filter(high_confidence_mask).drop("row_nr")
        print(f"After confidence filtering: {len(df)} detections")
    else:
        print("[WARNING] No confidence column found in data!")

    if df.is_empty():
        print("[ERROR] No high-confidence detections remaining!")
        return pl.DataFrame()

    # Remove duplicate detections of the same fire from different satellites
    duplicate_cols = ["latitude", "longitude", "acq_date", "acq_time"]
    missing_cols = [col for col in duplicate_cols if col not in df.columns]
    if not missing_cols:
        before_dedup = len(df)
        df = df.unique(subset=duplicate_cols)
        after_dedup = len(df)
        if before_dedup != after_dedup:
            print(f"Removed {before_dedup - after_dedup} duplicate detections")

    # Group nearby fires into clusters if requested
    if cluster_fires and not df.is_empty():
        print(f"Clustering fires within {cluster_distance_km}km...")
        try:
            df = calculate_fire_clusters(df, cluster_distance_km)
            unique_clusters = df["fire_cluster_id"].n_unique()
            print(f"Grouped {len(df)} detections into {unique_clusters} fire clusters")
        except Exception as e:
            print(f"[ERROR] Clustering failed: {e}")

    # Ensure consistent columns across all source datasets
    cols = [
        "country_id",
        "latitude",
        "longitude",
        "bright_ti4",
        "scan",
        "track",
        "acq_date",
        "acq_time",
        "satellite",
        "instrument",
        "confidence",
        "version",
        "bright_ti5",
        "frp",
        "daynight",
    ]

    # Add clustering columns if they exist
    cols = add_clustering_columns_to_schema(
        cols, cluster_fires and "fire_cluster_id" in df.columns
    )

    # Check which columns are missing and add them with null values
    for c in cols:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))

    df = df.select(cols)

    # Add standard brightness column for validation compatibility
    brightness_data_available = {}
    for col in ["brightness", "bright_ti4", "bright_ti5"]:
        if col in df.columns:
            non_null_count = len(df) - df[col].null_count()
            brightness_data_available[col] = non_null_count

    # Use the column with the most data as primary brightness
    if "brightness" not in df.columns:
        if brightness_data_available:
            best_col = max(brightness_data_available.items(), key=lambda x: x[1])
            if best_col[1] > 0:
                df = df.with_columns(df[best_col[0]].alias("brightness"))
            else:
                df = df.with_columns(pl.lit(None).alias("brightness"))
        else:
            df = df.with_columns(pl.lit(None).alias("brightness"))

    # Ensure brightness column is included in final schema
    if "brightness" not in cols:
        cols.append("brightness")
        df = df.select(cols)

    if cluster_fires and not df.is_empty():
        print(
            f"Final result: {len(df)} detections in {df['fire_cluster_id'].n_unique()} clusters"
        )
    else:
        print(f"Final result: {len(df)} high-confidence fire detections")

    return df
