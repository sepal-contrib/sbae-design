"""Test stratified sampling implementation against methodology examples.

This test suite verifies that our implementation follows the stratified
sampling methodology from the documentation.
"""

import math

import pandas as pd
import pytest

from component.sampling.stratified import StratifiedSamplingStrategy
from component.sampling.types import AllocationMethod, SamplingInputs, SamplingMethod
from component.scripts.calc_utils import get_z_score
from component.scripts.calculations import (
    apply_adjusted_allocation,
    calculate_stratified_sample_size,
)
from component.scripts.stratified import (
    calculate_openforis_stratified_design,
    calculate_per_class_moe_for_allocation,
)


def test_z_score_values():
    """Test Z-score calculation for standard confidence levels."""
    assert abs(get_z_score(0.90) - 1.645) < 0.001
    assert abs(get_z_score(0.95) - 1.960) < 0.001
    assert abs(get_z_score(0.99) - 2.576) < 0.001


def test_standard_deviation_calculation():
    """Test S_h = sqrt(EUA_h * (1 - EUA_h)) calculation."""
    # From methodology examples
    test_cases = [
        (0.90, 0.300),  # Forest
        (0.85, 0.357),  # Agriculture
        (0.70, 0.458),  # Deforestation
        (0.65, 0.477),  # Degradation
    ]

    for eua, expected_sh in test_cases:
        calculated_sh = math.sqrt(eua * (1 - eua))
        assert (
            abs(calculated_sh - expected_sh) < 0.001
        ), f"S_h for EUA={eua} should be {expected_sh}, got {calculated_sh}"


def test_stratified_sample_size_methodology_example():
    """Test stratified sample size calculation against methodology Step 6 example.

    From the methodology document example:
    - Forest: W_h=0.50, EUA=0.90, S_h=0.300 -> W_h*S_h=0.1500
    - Agriculture: W_h=0.30, EUA=0.85, S_h=0.357 -> W_h*S_h=0.1071
    - Urban: W_h=0.15, EUA=0.90, S_h=0.300 -> W_h*S_h=0.0450
    - Deforestation: W_h=0.008, EUA=0.70, S_h=0.458 -> W_h*S_h=0.0037
    - Degradation: W_h=0.012, EUA=0.65, S_h=0.477 -> W_h*S_h=0.0057
    - Water: W_h=0.030, EUA=0.90, S_h=0.300 -> W_h*S_h=0.0090
    Sum = 0.3205

    With target S(Ô) = 0.01:
    n = (0.3205 / 0.01)² = 1,027
    """
    # Create test area data (total area = 1,000,000 ha)
    area_data = pd.DataFrame(
        {
            "map_code": [1, 2, 3, 4, 5, 6],
            "map_area": [500000, 300000, 150000, 8000, 12000, 30000],  # in hectares
            "map_edited_class": [
                "Forest",
                "Agriculture",
                "Urban",
                "Deforestation",
                "Degradation",
                "Water",
            ],
        }
    )

    expected_accuracies = {
        1: 0.90,  # Forest
        2: 0.85,  # Agriculture
        3: 0.90,  # Urban (assuming same as Forest)
        4: 0.70,  # Deforestation
        5: 0.65,  # Degradation
        6: 0.90,  # Water
    }

    target_se = 0.01  # 1% standard error

    n = calculate_stratified_sample_size(
        area_df=area_data,
        expected_accuracies=expected_accuracies,
        target_standard_error=target_se,
    )

    # Expected: 1,027 samples
    assert abs(n - 1027) <= 2, f"Expected ~1027 samples, got {n}"


def test_adjusted_proportional_allocation():
    """Test adjusted proportional allocation with minimum constraints.

    This tests the algorithm from Step 7 of the methodology.
    """
    # Simple test case: 4 classes, total 100 samples, minimum 10 per class
    area_data = pd.DataFrame(
        {
            "map_code": [1, 2, 3, 4],
            "map_area": [5000, 3000, 1500, 500],  # Small class (4) is only 5% of area
            "map_edited_class": ["Class 1", "Class 2", "Class 3", "Class 4"],
        }
    )

    total_samples = 100
    min_samples = 10

    allocation = apply_adjusted_allocation(area_data, total_samples, min_samples)

    # Verify minimums are enforced
    for samples in allocation.values():
        assert (
            samples >= min_samples
        ), f"Sample count {samples} below minimum {min_samples}"

    # Verify total is reasonable (may exceed original total due to minimums)
    total_allocated = sum(allocation.values())
    assert total_allocated >= total_samples, "Total should be at least requested amount"

    # Class 4 (smallest) should get minimum
    assert allocation[4] == min_samples, "Smallest class should get exactly minimum"


def test_adjusted_allocation_redistribution():
    """Test that samples are redistributed correctly when minimums are enforced."""
    # Extreme test case: very small rare class
    area_data = pd.DataFrame(
        {
            "map_code": [1, 2],
            "map_area": [99000, 1000],  # Rare class is only 1% of area
            "map_edited_class": ["Common", "Rare"],
        }
    )

    total_samples = 100
    min_samples = 30  # Rare class would get ~1 sample without minimum

    allocation = apply_adjusted_allocation(area_data, total_samples, min_samples)

    # Rare class should get minimum
    assert allocation[2] == min_samples

    # Common class should get the rest (redistributed)
    # With minimum enforcement, total may exceed 100
    assert allocation[1] >= (total_samples - min_samples)


def test_stratum_weights_sum_to_one():
    """Test that stratum weights sum to 1.0 (Step 4 of methodology)."""
    area_data = pd.DataFrame(
        {
            "map_code": [1, 2, 3],
            "map_area": [5000, 3000, 2000],
            "map_edited_class": ["A", "B", "C"],
        }
    )

    total_area = area_data["map_area"].sum()
    weights = area_data["map_area"] / total_area

    assert abs(weights.sum() - 1.0) < 1e-10, "Weights should sum to 1.0"


def test_maximum_variance_at_50_percent():
    """Test that maximum variance occurs at EUA = 0.5 (from methodology)."""
    eua_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    variances = [eua * (1 - eua) for eua in eua_values]

    max_variance = max(variances)
    max_index = variances.index(max_variance)

    assert eua_values[max_index] == 0.5, "Maximum variance should be at EUA=0.5"
    assert abs(max_variance - 0.25) < 1e-10, "Maximum variance should be 0.25"


def test_stratified_formula_components():
    """Test individual components of the stratified sample size formula."""
    # Simple 2-class example for manual verification
    area_data = pd.DataFrame(
        {
            "map_code": [1, 2],
            "map_area": [6000, 4000],  # 60% and 40%
            "map_edited_class": ["A", "B"],
        }
    )

    expected_accuracies = {
        1: 0.8,  # S_h = sqrt(0.8 * 0.2) = 0.4
        2: 0.6,  # S_h = sqrt(0.6 * 0.4) = 0.49
    }

    # Calculate components manually
    total_area = 10000
    W1 = 6000 / total_area  # 0.6
    W2 = 4000 / total_area  # 0.4
    S1 = math.sqrt(0.8 * 0.2)  # 0.4
    S2 = math.sqrt(0.6 * 0.4)  # 0.49

    sum_wh_sh = W1 * S1 + W2 * S2  # 0.6*0.4 + 0.4*0.49 = 0.24 + 0.196 = 0.436

    target_se = 0.05
    expected_n = (sum_wh_sh / target_se) ** 2  # (0.436/0.05)^2 = 76.05

    # Test our function
    n = calculate_stratified_sample_size(
        area_df=area_data,
        expected_accuracies=expected_accuracies,
        target_standard_error=target_se,
    )

    assert abs(n - expected_n) <= 1, f"Expected {expected_n}, got {n}"


def test_openforis_stratified_design_columns_and_final_allocation():
    """Open Foris parity design should expose equal/proportional/adjusted/final."""
    area_data = pd.DataFrame(
        {
            "map_code": [1, 2],
            "map_area": [9000, 1000],
            "map_edited_class": ["Stable", "Rare"],
        }
    )

    design = calculate_openforis_stratified_design(
        area_df=area_data,
        expected_accuracies={1: 0.9, 2: 0.7},
        target_standard_error=0.05,
        min_samples_per_class=5,
    ).set_index("map_code")

    sum_wi_si = (0.9 * math.sqrt(0.9 * 0.1)) + (0.1 * math.sqrt(0.7 * 0.3))
    overall_sample = (sum_wi_si / 0.05) ** 2

    assert design.loc[1, "equal"] == math.floor(overall_sample / 2)
    assert design.loc[2, "equal"] == math.floor(overall_sample / 2)
    assert design.loc[1, "proportional"] == math.floor(0.9 * overall_sample)
    assert design.loc[2, "proportional"] == math.floor(0.1 * overall_sample)
    assert design.loc[1, "adjusted"] == math.floor(overall_sample - 5)
    assert design.loc[2, "adjusted"] == 5
    assert design.loc[1, "final"] == design.loc[1, "adjusted"]
    assert design.loc[2, "final"] == design.loc[2, "adjusted"]


def test_openforis_stratified_design_requires_eua_for_each_class():
    area_data = pd.DataFrame(
        {
            "map_code": [1, 2],
            "map_area": [9000, 1000],
            "map_edited_class": ["Stable", "Rare"],
        }
    )

    with pytest.raises(ValueError, match="missing for class code\\(s\\): 2"):
        calculate_openforis_stratified_design(
            area_df=area_data,
            expected_accuracies={1: 0.9},
            target_standard_error=0.05,
            min_samples_per_class=5,
        )


def test_per_class_moe_returns_map_code_and_expected_accuracy():
    """Per-class MOE output should expose map_code and preserve class-specific EUA."""
    area_data = pd.DataFrame(
        {
            "map_code": [1, 2],
            "map_area": [7000, 3000],
            "map_edited_class": ["Forest", "Loss"],
        }
    )

    moe_df = calculate_per_class_moe_for_allocation(
        allocation={1: 100, 2: 50},
        area_df=area_data,
        confidence_level=0.95,
        expected_accuracies={1: 0.9, 2: 0.7},
    )

    assert "map_code" in moe_df.columns
    assert "class_code" in moe_df.columns
    assert moe_df.loc[moe_df["map_code"] == 1, "expected_accuracy"].iloc[0] == 0.9
    assert moe_df.loc[moe_df["map_code"] == 2, "expected_accuracy"].iloc[0] == 0.7


def test_stratified_strategy_uses_openforis_adjusted_allocation():
    area_data = pd.DataFrame(
        {
            "map_code": [1, 2],
            "map_area": [9000, 1000],
            "map_edited_class": ["Stable", "Rare"],
        }
    )

    inputs = SamplingInputs(
        sampling_method=SamplingMethod.STRATIFIED,
        target_error=5.0,
        confidence_level=95.0,
        expected_accuracy=85.0,
        area_data=area_data,
        allocation_method=AllocationMethod.PROPORTIONAL,
        min_samples_per_class=5,
        expected_accuracies={1: 0.9, 2: 0.7},
    )

    results = StratifiedSamplingStrategy().calculate(inputs)

    assert results.success is True
    assert results.allocation_method == "adjusted_proportional"
    assert results.allocation_dict == {1: 34, 2: 5}
    assert results.total_samples == 39
    rare_class = next(item for item in results.samples_per_class if item.map_code == 2)
    assert rare_class.proportional_samples == 3
    assert rare_class.adjusted_samples == 5
    assert rare_class.samples == 5
    assert rare_class.expected_accuracy == 0.7
    assert rare_class.moe_percent is not None


def test_stratified_strategy_errors_when_class_eua_is_missing():
    area_data = pd.DataFrame(
        {
            "map_code": [1, 2],
            "map_area": [9000, 1000],
            "map_edited_class": ["Stable", "Rare"],
        }
    )

    inputs = SamplingInputs(
        sampling_method=SamplingMethod.STRATIFIED,
        target_error=5.0,
        confidence_level=95.0,
        expected_accuracy=85.0,
        area_data=area_data,
        allocation_method=AllocationMethod.PROPORTIONAL,
        min_samples_per_class=5,
        expected_accuracies={1: 0.9},
    )

    results = StratifiedSamplingStrategy().calculate(inputs)

    assert results.success is False
    assert "Expected user accuracy is missing for class code(s): 2" in (
        results.error_message or ""
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
