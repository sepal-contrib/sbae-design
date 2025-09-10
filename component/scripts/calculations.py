"""SBAE Statistical Calculations Module.

Contains all mathematical functions for sampling-based area estimation.
"""

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


def get_z_score(confidence_level: float) -> float:
    """Calculate Z-score for given confidence level.

    Args:
        confidence_level: Confidence level (0.80 to 0.99)

    Returns:
        Z-score value
    """
    if confidence_level == 0.90:
        return 1.645
    elif confidence_level == 0.95:
        return 1.960
    elif confidence_level == 0.99:
        return 2.576
    else:
        # For custom confidence levels
        p_value = (1 + confidence_level) / 2.0
        return stats.norm.ppf(p_value)


def calculate_overall_accuracy_sample_size(
    target_oa: float, allowable_error: float, confidence_level: float
) -> int:
    """Calculate sample size for overall accuracy objective.

    Formula: N = (Z² x OA x (1 - OA)) / E²

    Args:
        target_oa: Target overall accuracy (0-1)
        allowable_error: Allowable error (0-1)
        confidence_level: Confidence level (0-1)

    Returns:
        Required sample size
    """
    z_score = get_z_score(confidence_level)
    numerator = z_score**2 * target_oa * (1 - target_oa)
    denominator = allowable_error**2
    return math.ceil(numerator / denominator)


def calculate_target_class_sample_size(
    target_class_accuracy: float, target_class_error: float, confidence_level: float
) -> int:
    """Calculate sample size for target class precision objective.

    Formula: n_target = (Z² x UA x (1 - UA)) / Ej²

    Args:
        target_class_accuracy: Expected user's accuracy for target class (0-1)
        target_class_error: Allowable error for target class (0-1)
        confidence_level: Confidence level (0-1)

    Returns:
        Required sample size for target class
    """
    z_score = get_z_score(confidence_level)
    numerator = z_score**2 * target_class_accuracy * (1 - target_class_accuracy)
    denominator = target_class_error**2
    return math.ceil(numerator / denominator)


def allocate_samples_proportional(
    area_df: pd.DataFrame, total_samples: int
) -> pd.Series:
    """Allocate samples proportionally to class areas.

    Formula: n_j = N x (Area_j / Total_Area)

    Args:
        area_df: DataFrame with map_code and map_area columns
        total_samples: Total number of samples to allocate

    Returns:
        Series with sample allocation per class
    """
    # Check for valid area data
    if area_df["map_area"].sum() == 0 or area_df["map_area"].isna().all():
        raise ValueError("Invalid area data: all areas are zero or NaN")

    # Replace any NaN values with 0
    area_values = area_df["map_area"].fillna(0)

    # Calculate proportions
    total_area = area_values.sum()
    if total_area == 0:
        raise ValueError("Total area is zero")

    area_proportions = area_values / total_area

    return total_samples * area_proportions


def allocate_samples_neyman(
    area_df: pd.DataFrame, expected_accuracies: Dict[int, float], total_samples: int
) -> Dict[int, float]:
    """Allocate samples using Neyman allocation.

    Formula: n_j = N x (W_j x S_j) / Σ(W_k x S_k)
    Where: W_j = area proportion, S_j = standard deviation

    Args:
        area_df: DataFrame with map_code and map_area columns
        expected_accuracies: Dictionary of expected accuracies per class
        total_samples: Total number of samples to allocate

    Returns:
        Dictionary with sample allocation per class
    """
    # Calculate standard deviations
    std_devs = {}
    for _, row in area_df.iterrows():
        code = int(row["map_code"])
        ua = expected_accuracies.get(code, 0.5)
        std_devs[code] = math.sqrt(ua * (1 - ua))

    # Calculate allocation ratios
    area_proportions = area_df["map_area"] / area_df["map_area"].sum()
    wj_sj_products = {}

    for _, row in area_df.iterrows():
        code = int(row["map_code"])
        idx = area_df[area_df["map_code"] == code].index[0]
        wj_sj_products[code] = area_proportions.iloc[idx] * std_devs[code]

    sum_wj_sj = sum(wj_sj_products.values())
    allocation_ratios = {k: v / sum_wj_sj for k, v in wj_sj_products.items()}

    return {k: total_samples * v for k, v in allocation_ratios.items()}


def allocate_samples_equal(
    area_df: pd.DataFrame, total_samples: int
) -> Dict[int, float]:
    """Allocate samples equally among all classes.

    Formula: n_j = N / number_of_classes

    Args:
        area_df: DataFrame with map_code and map_area columns
        total_samples: Total number of samples to allocate

    Returns:
        Dictionary with equal sample allocation per class
    """
    num_classes = len(area_df)
    samples_per_class = total_samples / num_classes

    return {int(row["map_code"]): samples_per_class for _, row in area_df.iterrows()}


def apply_minimum_constraints(
    allocation: Dict[int, float], min_samples_per_class: int
) -> Dict[int, int]:
    """Apply minimum samples per class constraint and convert to integers.

    Args:
        allocation: Dictionary with float sample allocations
        min_samples_per_class: Minimum samples required per class

    Returns:
        Dictionary with integer sample allocations meeting constraints
    """
    final_allocation = {}
    for class_code, n_float in allocation.items():
        # Handle NaN values
        if pd.isna(n_float) or not np.isfinite(n_float):
            n_int = min_samples_per_class
        else:
            n_int = max(math.ceil(n_float), min_samples_per_class)
        final_allocation[class_code] = n_int

    return final_allocation


def calculate_sample_design(
    area_df: pd.DataFrame,
    objective: str,
    target_oa: float,
    allowable_error: float,
    confidence_level: float,
    min_samples_per_class: int,
    allocation_method: str,
    expected_accuracies: Optional[Dict[int, float]] = None,
    target_class: Optional[int] = None,
    target_class_error: float = 0.05,
) -> Dict[int, int]:
    """Main function to calculate optimal sample allocation.

    Args:
        area_df: DataFrame with class areas
        objective: "Overall Accuracy" or "Target Class Precision"
        target_oa: Target overall accuracy (0-1)
        allowable_error: Allowable error (0-1)
        confidence_level: Confidence level (0-1)
        min_samples_per_class: Minimum samples per class
        allocation_method: "Proportional", "Neyman", or "Equal"
        expected_accuracies: Expected accuracies per class (for Neyman)
        target_class: Target class code (for precision objective)
        target_class_error: Target class allowable error

    Returns:
        Dictionary with final sample allocation per class

    Raises:
        ValueError: If required parameters are missing for selected options
    """
    # Calculate baseline total sample size
    if objective == "Overall Accuracy":
        n_total = calculate_overall_accuracy_sample_size(
            target_oa, allowable_error, confidence_level
        )
    else:  # Target Class Precision
        if not target_class or not expected_accuracies:
            raise ValueError(
                "Target class precision requires target class and expected accuracies"
            )

        target_class_accuracy = expected_accuracies[target_class]
        n_total = calculate_target_class_sample_size(
            target_class_accuracy, target_class_error, confidence_level
        )

    # Adjust for minimum samples constraint
    num_classes = len(area_df)
    n_total_adjusted = max(n_total, num_classes * min_samples_per_class)

    # Allocate samples by method
    if allocation_method == "Proportional":
        sample_allocation_series = allocate_samples_proportional(
            area_df, n_total_adjusted
        )
        # Convert series to dictionary for consistency
        allocation_dict = {}
        for _, row in area_df.iterrows():
            code = int(row["map_code"])
            idx = area_df[area_df["map_code"] == code].index[0]
            allocation_dict[code] = sample_allocation_series.iloc[idx]

    elif allocation_method == "Neyman":
        if not expected_accuracies:
            raise ValueError("Neyman allocation requires expected accuracies")
        allocation_dict = allocate_samples_neyman(
            area_df, expected_accuracies, n_total_adjusted
        )

    elif allocation_method == "Equal":
        allocation_dict = allocate_samples_equal(area_df, n_total_adjusted)

    else:
        raise ValueError(f"Unknown allocation method: {allocation_method}")

    # Apply constraints and convert to integers
    final_allocation = apply_minimum_constraints(allocation_dict, min_samples_per_class)

    # Handle target class precision objective
    if objective == "Target Class Precision" and target_class and expected_accuracies:
        target_class_accuracy = expected_accuracies[target_class]
        n_required_target = calculate_target_class_sample_size(
            target_class_accuracy, target_class_error, confidence_level
        )

        final_allocation[target_class] = max(
            final_allocation[target_class], n_required_target
        )

    return final_allocation


def validate_parameters(
    target_oa: float,
    allowable_error: float,
    confidence_level: float,
    min_samples_per_class: int,
    expected_accuracies: Optional[Dict[int, float]] = None,
) -> list:
    """Validate input parameters and return list of errors.

    Args:
        target_oa: Target overall accuracy
        allowable_error: Allowable error
        confidence_level: Confidence level
        min_samples_per_class: Minimum samples per class
        expected_accuracies: Expected accuracies per class

    Returns:
        List of validation error messages
    """
    errors = []

    # Range checks
    if not (0.5 <= target_oa <= 0.99):
        errors.append("Target overall accuracy must be between 0.5 and 0.99")

    if not (0.01 <= allowable_error <= 0.2):
        errors.append("Allowable error must be between 0.01 and 0.2")

    if not (0.8 <= confidence_level <= 0.99):
        errors.append("Confidence level must be between 0.8 and 0.99")

    if min_samples_per_class <= 0:
        errors.append("Minimum samples per class must be positive")

    # Expected accuracies validation
    if expected_accuracies:
        for class_code, ua in expected_accuracies.items():
            if not (0.3 <= ua <= 1.0):
                errors.append(
                    f"Expected accuracy for class {class_code} must be between 0.3 and 1.0"
                )

    return errors


def calculate_allocation_summary(
    area_df: pd.DataFrame, allocation: Dict[int, int], class_lookup: Dict[int, str]
) -> pd.DataFrame:
    """Create a summary DataFrame of the sample allocation results.

    Args:
        area_df: DataFrame with class areas
        allocation: Sample allocation per class
        class_lookup: Class code to name mapping

    Returns:
        DataFrame with allocation summary
    """
    results_data = []
    total_area = area_df["map_area"].sum()
    total_samples = sum(allocation.values())

    for _, row in area_df.iterrows():
        code = int(row["map_code"])
        class_name = class_lookup.get(code, f"Class {code}")
        area = row["map_area"]
        area_percent = (area / total_area) * 100
        samples = allocation.get(code, 0)
        sample_percent = (samples / total_samples) * 100 if total_samples > 0 else 0
        samples_per_area = (samples / area * 1000) if area > 0 else 0

        results_data.append(
            {
                "Class Code": code,
                "Class Name": class_name,
                "Area": f"{area:.2f}",
                "Area %": f"{area_percent:.1f}%",
                "Samples": samples,
                "Sample %": f"{sample_percent:.1f}%",
                "Samples/1000 area units": f"{samples_per_area:.2f}",
            }
        )

    return pd.DataFrame(results_data)
