"""
FIRMS (Fire Information for Resource Management System) Package

This package provides functionality for collecting and processing NASA FIRMS
fire detection data, including spatial clustering of nearby fire detections.
"""

from .firms_collect import collect_fire_data
from .clustering import (
    calculate_fire_clusters,
    get_cluster_summary,
    DEFAULT_CLUSTER_DISTANCE_KM
)

__all__ = [
    'collect_fire_data',
    'calculate_fire_clusters', 
    'get_cluster_summary',
    'DEFAULT_CLUSTER_DISTANCE_KM'
] 