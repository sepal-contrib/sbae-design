import math

from component.scripts.calc_utils import get_z_score


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
