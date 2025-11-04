import math
from typing import Dict, Optional

import pandas as pd

from component.scripts.calc_utils import get_z_score


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
