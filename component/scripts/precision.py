import math

import numpy as np
import pandas as pd

from component.scripts.calc_utils import get_z_score


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
