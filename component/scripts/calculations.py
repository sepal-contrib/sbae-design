"""SBAE Statistical Calculations Module.

Contains all mathematical functions for sampling-based area estimation.
"""

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from component.scripts.simple_random import calculate_overall_accuracy_sample_size
from component.scripts.stratified import (
    allocate_samples_equal,
    allocate_samples_neyman,
    apply_adjusted_allocation,
    calculate_stratified_sample_size,
    calculate_target_class_sample_size,
)


def apply_minimum_constraints(
    allocation: Dict[int, float], min_samples_per_class: int
) -> Dict[int, int]:
    """Apply minimum samples per class constraint and convert to integers.

    This is a simple version that just enforces minimums without redistribution.
    For proper adjusted proportional allocation, use apply_adjusted_allocation().

    Args:
        allocation: Dictionary with float sample allocations
        min_samples_per_class: Minimum samples required per class

    Returns:
        Dictionary with integer sample allocations meeting constraints
    """
    final_allocation = {}
    for class_code, n_float in allocation.items():
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
    total_samples_override: Optional[int] = None,
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
        total_samples_override: Fixed total for simple/systematic sampling (no per-class allocation)
        expected_accuracies: Expected accuracies per class (for Neyman and stratified calculation)
        target_class: Target class code (for precision objective)
        target_class_error: Target class allowable error

    Returns:
        Dictionary with final sample allocation per class

    Raises:
        ValueError: If required parameters are missing for selected options
    """
    if total_samples_override is not None and allocation_method in [
        "Simple",
        "Systematic",
    ]:
        return {}

    # Calculate baseline total sample size (or use override for stratified with fixed total)
    if total_samples_override is not None:
        n_total = int(total_samples_override)
    elif objective == "Overall Accuracy":
        # Use stratified formula if we have expected accuracies, otherwise fall back to simple formula
        if expected_accuracies and allocation_method not in ["Simple", "Systematic"]:
            n_total = calculate_stratified_sample_size(
                area_df=area_df,
                expected_accuracies=expected_accuracies,
                target_standard_error=allowable_error,
            )
        else:
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

    # For proportional allocation, use adjusted allocation with proper redistribution
    if allocation_method == "Proportional":
        final_allocation = apply_adjusted_allocation(
            area_df, n_total, min_samples_per_class
        )

    elif allocation_method == "Neyman":
        if not expected_accuracies:
            raise ValueError("Neyman allocation requires expected accuracies")

        # Calculate Neyman allocation
        allocation_dict = allocate_samples_neyman(area_df, expected_accuracies, n_total)

        # Then apply adjusted allocation to handle minimums properly
        final_allocation = {}
        total_area = area_df["map_area"].sum()
        classes_below_min = []
        classes_above_min = []
        samples_reserved = 0

        # First pass: identify classes below minimum
        for code, n_samples in allocation_dict.items():
            if n_samples < min_samples_per_class:
                final_allocation[code] = min_samples_per_class
                classes_below_min.append(code)
                samples_reserved += min_samples_per_class
            else:
                classes_above_min.append(code)

        # Second pass: redistribute to classes above minimum
        samples_remaining = n_total - samples_reserved
        if samples_remaining < 0:
            samples_remaining = 0

        # Recalculate Neyman for classes above minimum
        sum_wh_sh_above_min = 0
        for _, row in area_df.iterrows():
            code = int(row["map_code"])
            if code in classes_above_min:
                W_h = row["map_area"] / total_area
                eua = expected_accuracies.get(code, 0.5)
                S_h = math.sqrt(eua * (1 - eua))
                sum_wh_sh_above_min += W_h * S_h

        if sum_wh_sh_above_min > 0:
            for _, row in area_df.iterrows():
                code = int(row["map_code"])
                if code in classes_above_min:
                    W_h = row["map_area"] / total_area
                    eua = expected_accuracies.get(code, 0.5)
                    S_h = math.sqrt(eua * (1 - eua))
                    allocation_ratio = (W_h * S_h) / sum_wh_sh_above_min
                    final_allocation[code] = max(
                        min_samples_per_class,
                        math.ceil(samples_remaining * allocation_ratio),
                    )

    elif allocation_method == "Equal":
        allocation_dict = allocate_samples_equal(area_df, n_total)
        final_allocation = apply_minimum_constraints(
            allocation_dict, min_samples_per_class
        )

    else:
        raise ValueError(f"Unknown allocation method: {allocation_method}")

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
