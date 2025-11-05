import math

from component.scripts.calc_utils import get_z_score


def calculate_overall_accuracy_sample_size(
    target_oa: float, allowable_error: float, confidence_level: float
) -> int:
    """Calculate sample size for overall accuracy objective (Simple Random Sampling).

    Based on Cochran's formula for estimating a population proportion.
    See sampling-theory/02 - Sampling Intuition.ipynb for theoretical background.

    Formula: n = (Z² x p x (1 - p)) / e²

    Where:
        - Z = z-score for the desired confidence level
        - p = expected proportion (target overall accuracy)
        - e = allowable absolute error (margin of error)

    Args:
        target_oa: Target overall accuracy (0-1), serves as expected proportion
        allowable_error: Allowable absolute error (0-1), NOT relative error
        confidence_level: Confidence level (0-1)

    Returns:
        Required sample size (rounded up)

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
