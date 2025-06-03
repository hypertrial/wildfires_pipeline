"""
Data merging utilities for wildfire analysis.

Handles coordinate-based merging between fire locations and weather data,
including proximity-based matching for coordinate precision mismatches.
"""

from .coordinate_merger import merge_weather_data

__all__ = ['merge_weather_data'] 