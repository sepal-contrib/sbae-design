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
    """Calculate sample size for overall accuracy objective (Simple Random Sampling).

    Formula: N = (Z² x OA x (1 - OA)) / E²

    Note: This is for simple random sampling. For stratified sampling,
    use calculate_stratified_sample_size() instead.

    Args:
        target_oa: Target overall accuracy (0-1)
        allowable_error: Allowable error (0-1)
        confidence_level: Confidence level (0-1)

    Returns:
        Required sample size

    Raises:
        ValueError: If allowable_error is zero or negative
    """
    if allowable_error <= 0:
        raise ValueError("Allowable error must be greater than 0")

    z_score = get_z_score(confidence_level)
    numerator = z_score**2 * target_oa * (1 - target_oa)
    denominator = allowable_error**2

    result = numerator / denominator
    if not math.isfinite(result):
        raise ValueError(f"Sample size calculation resulted in invalid value: {result}")

    return math.ceil(result)


def calculate_stratified_sample_size(
    area_df: pd.DataFrame,
    expected_accuracies: Dict[int, float],
    target_standard_error: float,
) -> int:
    """Calculate sample size for stratified random sampling.

    Formula from methodology Step 6:
    n = (Σ(W_h x S_h) / S(Ô))²

    Where:
    - W_h = stratum weight (proportion of area)
    - S_h = standard deviation = √(EUA_h x (1 - EUA_h))
    - S(Ô) = target standard error of overall accuracy

    Args:
        area_df: DataFrame with map_code and map_area columns
        expected_accuracies: Dictionary of expected user's accuracies per class
        target_standard_error: Target standard error (e.g., 0.01 for 1%)

    Returns:
        Required total sample size

    Raises:
        ValueError: If parameters are invalid
    """
    if target_standard_error <= 0:
        raise ValueError("Target standard error must be greater than 0")

    total_area = area_df["map_area"].sum()
    if total_area == 0:
        raise ValueError("Total area is zero")

    sum_wh_sh = 0.0

    for _, row in area_df.iterrows():
        code = int(row["map_code"])
        area = row["map_area"]

        W_h = area / total_area
        eua = expected_accuracies.get(code, 0.5)
        S_h = math.sqrt(eua * (1 - eua))

        sum_wh_sh += W_h * S_h

    ratio = sum_wh_sh / target_standard_error
    n = ratio**2

    if not math.isfinite(n):
        raise ValueError(f"Sample size calculation resulted in invalid value: {n}")

    return math.ceil(n)


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


def apply_adjusted_allocation(
    area_df: pd.DataFrame,
    total_samples: int,
    min_samples_per_class: int,
) -> Dict[int, int]:
    """Apply adjusted proportional allocation with minimum constraints.

    Implementation of methodology Step 7 - Adjusted Proportional Allocation:
    1. Calculate proportional allocation
    2. Identify classes below minimum
    3. Set those to minimum
    4. Redistribute remaining samples proportionally among other classes

    Args:
        area_df: DataFrame with map_code and map_area columns
        total_samples: Total number of samples to allocate
        min_samples_per_class: Minimum samples per class

    Returns:
        Dictionary with final sample allocation per class
    """
    total_area = area_df["map_area"].sum()
    if total_area == 0:
        raise ValueError("Total area is zero")

    allocation = {}
    classes_below_min = []
    classes_above_min = []
    samples_reserved = 0
    area_above_min = 0

    for _, row in area_df.iterrows():
        code = int(row["map_code"])
        area = row["map_area"]
        proportion = area / total_area
        proportional_samples = total_samples * proportion

        if proportional_samples < min_samples_per_class:
            allocation[code] = min_samples_per_class
            classes_below_min.append(code)
            samples_reserved += min_samples_per_class
        else:
            classes_above_min.append(code)
            area_above_min += area

    samples_remaining = total_samples - samples_reserved

    if samples_remaining < 0:
        total_samples_adjusted = len(area_df) * min_samples_per_class
        samples_remaining = total_samples_adjusted - samples_reserved

    for _, row in area_df.iterrows():
        code = int(row["map_code"])
        if code in classes_above_min:
            area = row["map_area"]
            if area_above_min > 0:
                proportion = area / area_above_min
                allocation[code] = math.ceil(samples_remaining * proportion)
            else:
                allocation[code] = min_samples_per_class

    return allocation


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


def calculate_precision_curve(
    target_oa: float,
    confidence_level: float,
    min_sample_size: int = 30,
    max_sample_size: int = 1000,
    num_points: int = 50,
) -> pd.DataFrame:
    """Calculate precision curve showing MOE vs sample size relationship.

    The margin of error (MOE) decreases as sample size increases following:
    MOE = Z * sqrt(OA * (1 - OA) / n)

    This is derived from rearranging the sample size formula:
    n = (Z² * OA * (1 - OA)) / MOE²
    Therefore: MOE = sqrt((Z² * OA * (1 - OA)) / n)

    Args:
        target_oa: Target overall accuracy (0-1)
        confidence_level: Confidence level (0-1)
        min_sample_size: Minimum sample size for curve
        max_sample_size: Maximum sample size for curve
        num_points: Number of points to calculate in curve

    Returns:
        DataFrame with columns: sample_size, moe_percent, moe_decimal
    """
    z_score = get_z_score(confidence_level)

    sample_sizes = np.linspace(min_sample_size, max_sample_size, num_points)

    moe_values = []
    for n in sample_sizes:
        moe = z_score * math.sqrt((target_oa * (1 - target_oa)) / n)
        moe_values.append(moe)

    precision_df = pd.DataFrame(
        {
            "sample_size": sample_sizes.astype(int),
            "moe_decimal": moe_values,
            "moe_percent": [moe * 100 for moe in moe_values],
        }
    )

    return precision_df


def cochran_sample_size(
    target_oa: float, allowable_error: float, confidence_level: float
) -> int:
    """Alias for Cochran (1977) sample size formula for proportions.

    This uses the same formula as calculate_overall_accuracy_sample_size but
    is provided with the historical citation name for clarity in the UI.
    """
    return calculate_overall_accuracy_sample_size(
        target_oa, allowable_error, confidence_level
    )


def calculate_confidence_interval(
    p_hat: float, n: int, confidence_level: float, method: str = "wilson"
):
    """Calculate confidence interval for a proportion and return (lower, upper, moe_decimal).

    Methods supported: 'normal' (Wald), 'wilson' (recommended)

    Returns:
        (lower_bound, upper_bound, moe_decimal)
    """
    if n <= 0:
        return (0.0, 1.0, 1.0)

    z = get_z_score(confidence_level)
    p = p_hat

    if method == "normal":
        se = math.sqrt(p * (1 - p) / n)
        moe = z * se
        lower = max(0.0, p - moe)
        upper = min(1.0, p + moe)
        return (lower, upper, moe)

    # Wilson score interval
    denom = 1 + (z**2) / n
    centre = (p + (z**2) / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))) / denom
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    moe = (upper - lower) / 2.0
    return (lower, upper, moe)


def calculate_sample_size_wilson(
    target_oa: float,
    allowable_error: float,
    confidence_level: float,
    max_n: int = 1000000,
) -> int:
    """Find the smallest n such that the Wilson interval half-width <= allowable_error.

    This implements a numeric search. It provides an alternative to the normal-approximation
    Cochran formula and can be used where Stehman & Foody (2019) recommend more robust
    interval-based planning.
    """
    if allowable_error <= 0:
        raise ValueError("Allowable error must be greater than 0")

    get_z_score(confidence_level)

    # Start from Cochran estimate as initial guess
    n = calculate_overall_accuracy_sample_size(
        target_oa, allowable_error, confidence_level
    )
    # Ensure at least 2
    n = max(2, n)

    # If Wilson at this n is already small enough, return. Otherwise increment until satisfied.
    while n <= max_n:
        _, _, moe = calculate_confidence_interval(
            target_oa, n, confidence_level, method="wilson"
        )
        if moe <= allowable_error:
            return n
        n += 1

    raise ValueError(f"Required sample size exceeds maximum search limit ({max_n})")


def calculate_current_moe(
    current_sample_size: int, target_oa: float, confidence_level: float
) -> float:
    """Calculate current margin of error for a given sample size.

    MOE = Z * sqrt(OA * (1 - OA) / n)

    Args:
        current_sample_size: Current sample size
        target_oa: Target overall accuracy (0-1)
        confidence_level: Confidence level (0-1)

    Returns:
        Margin of error as decimal (0-1)
    """
    if current_sample_size <= 0:
        return 1.0

    z_score = get_z_score(confidence_level)
    moe = z_score * math.sqrt((target_oa * (1 - target_oa)) / current_sample_size)
    return moe


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


def calculate_per_class_moe(
    n_h: int,
    p_h: float,
    confidence_level: float,
    N_h: Optional[int] = None,
    deff: float = 1.0,
) -> float:
    """Calculate per-class Margin of Error (MOE) given current allocation.

    Formula (binomial normal approximation):
    MOE_h = Z * sqrt(DEFF * (p_h * (1 - p_h) / n_h)) * sqrt((N_h - n_h) / (N_h - 1)) * 100

    Args:
        n_h: Samples allocated to the class (from allocation table)
        p_h: Expected accuracy for that class (pilot/prior; if unknown use 0.5)
        confidence_level: Confidence level (0-1, e.g., 0.95 for 95%)
        N_h: Finite population size for the class (optional, number of pixels/features)
        deff: Design effect if clustering inflates variance (≈1.0 for spatially balanced; >1 if clustered)

    Returns:
        Margin of Error as percentage (0-100)
    """
    if n_h <= 0:
        return 100.0

    z_score = get_z_score(confidence_level)

    variance_term = math.sqrt(deff * (p_h * (1 - p_h) / n_h))

    if N_h is not None and N_h > n_h:
        fpc = math.sqrt((N_h - n_h) / (N_h - 1))
    else:
        fpc = 1.0

    moe = z_score * variance_term * fpc * 100

    return moe


def calculate_per_class_moe_for_allocation(
    allocation: Dict[int, int],
    area_df: pd.DataFrame,
    confidence_level: float,
    expected_accuracies: Optional[Dict[int, float]] = None,
    population_sizes: Optional[Dict[int, int]] = None,
    deff: float = 1.0,
) -> pd.DataFrame:
    """Calculate per-class MOE for all classes in an allocation.

    Args:
        allocation: Dictionary mapping class codes to sample sizes
        area_df: DataFrame with class information
        confidence_level: Confidence level (0-1)
        expected_accuracies: Expected accuracies per class (default 0.5 if not provided)
        population_sizes: Population sizes per class (optional for FPC)
        deff: Design effect (default 1.0)

    Returns:
        DataFrame with class_code, class_name, samples, expected_accuracy, moe_percent
    """
    results = []

    for _, row in area_df.iterrows():
        code = int(row["map_code"])
        class_name = row.get("map_edited_class", f"Class {code}")

        n_h = allocation.get(code, 0)
        p_h = expected_accuracies.get(code, 0.5) if expected_accuracies else 0.5
        N_h = population_sizes.get(code) if population_sizes else None

        moe = calculate_per_class_moe(
            n_h=n_h,
            p_h=p_h,
            confidence_level=confidence_level,
            N_h=N_h,
            deff=deff,
        )

        results.append(
            {
                "class_code": code,
                "class_name": class_name,
                "samples": n_h,
                "expected_accuracy": p_h,
                "moe_percent": moe,
            }
        )

    return pd.DataFrame(results)
