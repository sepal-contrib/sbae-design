import math
from typing import Optional

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
        p_value = (1 + confidence_level) / 2.0
        return stats.norm.ppf(p_value)


def calculate_confidence_interval(
    p_hat: float,
    n: int,
    confidence_level: float,
    method: str = "wilson",
    N: Optional[int] = None,
):
    """Calculate confidence interval for a proportion.

    Implements standard confidence interval methods with optional Finite Population
    Correction (FPC) and Bessel's correction for consistency with sampling theory.

    Args:
        p_hat: Sample proportion (0-1)
        n: Sample size
        confidence_level: Confidence level (0-1)
        method: 'normal' (Wald with corrections) or 'wilson' (recommended)
        N: Population size for Finite Population Correction (optional)

    Returns:
        Tuple of (lower_bound, upper_bound, moe_decimal)

    Note:
        - Normal method uses Bessel's correction (n-1) for unbiased variance estimation
        - FPC is applied when N is provided and n is non-negligible fraction of N
        - Wilson interval is more robust for extreme proportions or small samples
    """
    if n <= 0:
        return (0.0, 1.0, 1.0)

    z = get_z_score(confidence_level)
    p = p_hat

    if method == "normal":
        if n == 1:
            return (0.0, 1.0, 1.0)

        se = math.sqrt(p * (1 - p) / (n - 1))

        if N is not None and N > n:
            fpc = math.sqrt((N - n) / (N - 1))
            se *= fpc

        moe = z * se
        lower = max(0.0, p - moe)
        upper = min(1.0, p + moe)
        return (lower, upper, moe)

    denom = 1 + (z**2) / n
    centre = (p + (z**2) / (2 * n)) / denom
    margin = (z * math.sqrt((p * (1 - p) / n) + (z**2) / (4 * n**2))) / denom
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    moe = (upper - lower) / 2.0
    return (lower, upper, moe)
