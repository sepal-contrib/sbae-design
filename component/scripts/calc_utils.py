import math

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
