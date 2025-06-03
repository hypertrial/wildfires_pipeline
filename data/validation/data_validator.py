"""
Data validation utilities for wildfire analysis output.

Provides validation of integrated fire and weather datasets to ensure
data quality and completeness.
"""

import os
import time
from typing import Dict, Any, List, Optional

import polars as pl


def validate_output_file(
    file_path: str = "wildfire_integrated.parquet",
    max_age_minutes: int = 10,
    required_columns: Optional[List[str]] = None,
    min_records: int = 1,
) -> Dict[str, Any]:
    """
    Validate the integrated wildfire dataset output file.

    Args:
        file_path: Path to the output file to validate
        max_age_minutes: Maximum age of file in minutes to be considered fresh
        required_columns: List of required column names. If None, uses default.
        min_records: Minimum number of records required

    Returns:
        Dictionary containing validation results and statistics
    """
    # Default required columns for fire detection data
    if required_columns is None:
        required_columns = [
            "latitude",
            "longitude",
            "acq_date",
            "acq_time",
            "satellite",
            "instrument",
            "confidence",
        ]

    # Expected brightness columns (at least one should be present)
    brightness_columns = ["brightness", "bright_ti4", "bright_ti5"]

    validation_result = {
        "file_exists": False,
        "file_size_mb": 0,
        "file_age_minutes": float("inf"),
        "is_fresh": False,
        "record_count": 0,
        "column_count": 0,
        "missing_columns": [],
        "has_null_coordinates": False,
        "data_types_valid": True,
        "errors": [],
        "warnings": [],
    }

    # Step 1: Check file existence
    if not os.path.exists(file_path):
        validation_result["errors"].append(f"File does not exist: {file_path}")
        print(f"[ERROR] File does not exist: {file_path}")
        return validation_result

    validation_result["file_exists"] = True

    # Step 2: Check file age
    file_stat = os.stat(file_path)
    file_age_seconds = time.time() - file_stat.st_mtime
    file_age_minutes = file_age_seconds / 60
    validation_result["file_age_minutes"] = file_age_minutes

    if file_age_minutes <= max_age_minutes:
        validation_result["is_fresh"] = True
    else:
        validation_result["warnings"].append(
            f"File is {file_age_minutes:.1f} minutes old (>{max_age_minutes} minutes)"
        )
        print(f"[WARNING] File is older than {max_age_minutes} minutes")

    # Step 3: Check file size
    file_size_bytes = file_stat.st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    validation_result["file_size_mb"] = file_size_mb

    if file_size_mb == 0:
        validation_result["errors"].append("File is empty (0 bytes)")
        print("[ERROR] File is empty")
        return validation_result

    # Step 4: Try to read the file
    try:
        if file_path.endswith(".parquet"):
            df = pl.read_parquet(file_path)
        elif file_path.endswith(".csv"):
            df = pl.read_csv(file_path)
        else:
            validation_result["errors"].append(f"Unsupported file format: {file_path}")
            print("[ERROR] Unsupported file format")
            return validation_result
    except Exception as e:
        validation_result["errors"].append(f"Failed to read file: {str(e)}")
        print(f"[ERROR] Failed to read file: {e}")
        return validation_result

    # Step 5: Basic dataframe validation
    validation_result["record_count"] = len(df)
    validation_result["column_count"] = len(df.columns)

    if len(df) < min_records:
        validation_result["errors"].append(
            f"Insufficient records: {len(df)} (minimum required: {min_records})"
        )
        print(f"[ERROR] Insufficient records: {len(df)} < {min_records}")

    # Step 6: Check for required columns
    missing_columns = []
    for col in required_columns:
        if col not in df.columns:
            missing_columns.append(col)

    # Check for brightness columns (flexible requirement)
    has_brightness = any(col in df.columns for col in brightness_columns)
    if not has_brightness:
        missing_columns.append("brightness (or bright_ti4/bright_ti5)")

    validation_result["missing_columns"] = missing_columns

    if missing_columns:
        validation_result["errors"].append(
            f"Missing required columns: {missing_columns}"
        )
        print(f"[ERROR] Missing required columns: {missing_columns}")

    # Step 7: Validate data types for key columns
    data_type_issues = []

    # Check coordinate columns
    if "latitude" in df.columns:
        if not df["latitude"].dtype.is_numeric():
            data_type_issues.append("latitude column is not numeric")
            print("[ERROR] Latitude column is not numeric")
        elif df["latitude"].null_count() > 0:
            validation_result["has_null_coordinates"] = True
            null_count = df["latitude"].null_count()
            validation_result["warnings"].append(
                f"Found {null_count} null values in latitude column"
            )
            print(f"[WARNING] Found {null_count} null latitude values")
        elif not df["latitude"].is_between(-90, 90).all():
            invalid_count = (~df["latitude"].is_between(-90, 90)).sum()
            validation_result["warnings"].append(
                f"Found {invalid_count} latitude values outside valid range [-90, 90]"
            )
            print(f"[WARNING] Found {invalid_count} invalid latitude values")

    if "longitude" in df.columns:
        if not df["longitude"].dtype.is_numeric():
            data_type_issues.append("longitude column is not numeric")
            print("[ERROR] Longitude column is not numeric")
        elif df["longitude"].null_count() > 0:
            validation_result["has_null_coordinates"] = True
            null_count = df["longitude"].null_count()
            validation_result["warnings"].append(
                f"Found {null_count} null values in longitude column"
            )
            print(f"[WARNING] Found {null_count} null longitude values")
        elif not df["longitude"].is_between(-180, 180).all():
            invalid_count = (~df["longitude"].is_between(-180, 180)).sum()
            validation_result["warnings"].append(
                f"Found {invalid_count} longitude values outside valid range [-180, 180]"
            )
            print(f"[WARNING] Found {invalid_count} invalid longitude values")

    # Check date/time columns
    if "acq_date" in df.columns:
        try:
            df.with_columns(pl.col("acq_date").str.to_date())
        except:
            data_type_issues.append("acq_date column cannot be converted to datetime")
            print("[ERROR] acq_date column is not valid datetime format")

    # Check brightness values - DETAILED ANALYSIS
    for brightness_col in brightness_columns:
        if brightness_col in df.columns:
            if not df[brightness_col].dtype.is_numeric():
                data_type_issues.append(f"{brightness_col} column is not numeric")
                print(f"[ERROR] {brightness_col} column is not numeric")
                continue

            # Detailed null analysis
            null_count = df[brightness_col].null_count()
            total_count = len(df)
            null_percentage = (null_count / total_count * 100) if total_count > 0 else 0

            if null_count == total_count:
                validation_result["warnings"].append(
                    f"All {brightness_col} values are null"
                )
                print(f"[WARNING] All {brightness_col} values are null")
            elif null_count > 0:
                validation_result["warnings"].append(
                    f"{null_count}/{total_count} ({null_percentage:.1f}%) {brightness_col} values are null"
                )
                print(f"[WARNING] {null_count} {brightness_col} values are null")

            # Analyze non-null values
            valid_values = df[brightness_col].drop_nulls()
            if len(valid_values) > 0:
                min_val = valid_values.min()
                max_val = valid_values.max()

                # Check for reasonable brightness temperature ranges (should be in Kelvin)
                if min_val < 200 or max_val > 2000:
                    validation_result["warnings"].append(
                        f"{brightness_col} has unusual values: range {min_val:.1f} - {max_val:.1f} K"
                    )
                    print(
                        f"[WARNING] {brightness_col} has unusual range: {min_val:.1f} - {max_val:.1f} K"
                    )

                # Check for potential unit issues (values that look like Celsius instead of Kelvin)
                if max_val < 100:
                    validation_result["warnings"].append(
                        f"{brightness_col} values may be in wrong units (max: {max_val:.1f}, expected Kelvin)"
                    )
                    print(
                        f"[WARNING] {brightness_col} may be in wrong units (max: {max_val:.1f})"
                    )

    # Check confidence values
    if "confidence" in df.columns:
        if df["confidence"].null_count() == len(df):
            validation_result["warnings"].append("All confidence values are null")
            print("[WARNING] All confidence values are null")

    if data_type_issues:
        validation_result["data_types_valid"] = False
        validation_result["errors"].extend(data_type_issues)

    # Step 8: Additional data quality checks

    # Check for duplicate records
    if (
        "latitude" in df.columns
        and "longitude" in df.columns
        and "acq_date" in df.columns
    ):
        duplicate_count = len(df) - len(
            df.unique(subset=["latitude", "longitude", "acq_date"])
        )
        if duplicate_count > 0:
            validation_result["warnings"].append(
                f"Found {duplicate_count} potential duplicate records"
            )
            print(f"[WARNING] Found {duplicate_count} duplicate records")

    # Check weather data availability if present
    weather_columns = [
        col for col in df.columns if col.startswith(("current_", "forecast_"))
    ]
    if weather_columns:
        # Check rows with all null weather values
        null_weather_count = (
            df.select(weather_columns).null_count().sum_horizontal().sum()
        )
        total_weather_values = len(df) * len(weather_columns)
        weather_coverage = (
            (total_weather_values - null_weather_count) / total_weather_values
        ) * 100

        if weather_coverage < 50:
            validation_result["warnings"].append(
                f"Low weather data coverage: {weather_coverage:.1f}%"
            )
            print(f"[WARNING] Low weather data coverage: {weather_coverage:.1f}%")

    # Check clustering data if present
    if "fire_cluster_id" in df.columns:
        cluster_count = df["fire_cluster_id"].n_unique()

        # Check cluster statistics columns
        cluster_stat_cols = ["cluster_size", "cluster_center_lat", "cluster_center_lon"]
        missing_cluster_cols = [
            col for col in cluster_stat_cols if col not in df.columns
        ]
        if missing_cluster_cols:
            validation_result["warnings"].append(
                f"Missing cluster statistics columns: {missing_cluster_cols}"
            )
            print(f"[WARNING] Missing cluster columns: {missing_cluster_cols}")

    # Step 9: Final validation status
    validation_result["is_valid"] = len(validation_result["errors"]) == 0

    return validation_result


def print_validation_report(validation_result: Dict[str, Any]) -> None:
    """Print a formatted validation report."""

    print("\n" + "=" * 50)
    print("VALIDATION REPORT")
    print("=" * 50)

    # File status
    print("[FILE] File Status:")
    print(f"   Exists: {'OK' if validation_result['file_exists'] else 'FAIL'}")
    print(f"   Size: {validation_result['file_size_mb']:.2f} MB")
    print(f"   Age: {validation_result['file_age_minutes']:.1f} minutes")
    print(f"   Fresh: {'OK' if validation_result['is_fresh'] else 'WARN'}")

    # Data status
    print("\n[DATA] Data Status:")
    print(f"   Records: {validation_result['record_count']:,}")
    print(f"   Columns: {validation_result['column_count']}")
    print(
        f"   Data types valid: {'OK' if validation_result['data_types_valid'] else 'FAIL'}"
    )

    # Issues
    if validation_result["errors"]:
        print(f"\n[ERROR] ERRORS ({len(validation_result['errors'])}):")
        for error in validation_result["errors"]:
            print(f"   - {error}")

    if validation_result["warnings"]:
        print(f"\n[WARN] WARNINGS ({len(validation_result['warnings'])}):")
        for warning in validation_result["warnings"]:
            print(f"   - {warning}")

    # Overall status
    overall_status = "PASSED" if validation_result.get("is_valid", False) else "FAILED"
    print(f"\n[RESULT] Overall Status: {overall_status}")
    print("=" * 50)


def validate_and_report(
    file_path: str = "wildfire_integrated.parquet",
    max_age_minutes: int = 10,
    required_columns: Optional[List[str]] = None,
    min_records: int = 1,
    print_report: bool = True,
) -> bool:
    """
    Validate a file and optionally print a formatted report.

    Args:
        file_path: Path to file to validate
        max_age_minutes: Maximum acceptable file age in minutes
        required_columns: List of required column names
        min_records: Minimum number of records required
        print_report: Whether to print the validation report

    Returns:
        True if validation passed, False otherwise
    """
    validation_result = validate_output_file(
        file_path=file_path,
        max_age_minutes=max_age_minutes,
        required_columns=required_columns,
        min_records=min_records,
    )

    if print_report:
        print_validation_report(validation_result)

    return validation_result.get("is_valid", False)
