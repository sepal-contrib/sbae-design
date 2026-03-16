"""Tests for statistical calculations."""

import pandas as pd
import pytest

from component.scripts.calc_utils import calculate_confidence_interval
from component.scripts.calculations import (
    calculate_overall_accuracy_sample_size,
    calculate_sample_design,
)
from component.scripts.stratified import allocate_samples_balanced


def test_calculate_overall_accuracy_sample_size_valid():
    """Test sample size calculation with valid parameters."""
    result = calculate_overall_accuracy_sample_size(
        target_oa=0.85, allowable_error=0.05, confidence_level=0.95
    )
    assert result > 0
    assert isinstance(result, int)


def test_calculate_overall_accuracy_sample_size_zero_error():
    """Test that zero allowable error raises ValueError."""
    with pytest.raises(ValueError, match="Allowable error must be greater than 0"):
        calculate_overall_accuracy_sample_size(
            target_oa=0.85, allowable_error=0.0, confidence_level=0.95
        )


def test_calculate_overall_accuracy_sample_size_negative_error():
    """Test that negative allowable error raises ValueError."""
    with pytest.raises(ValueError, match="Allowable error must be greater than 0"):
        calculate_overall_accuracy_sample_size(
            target_oa=0.85, allowable_error=-0.05, confidence_level=0.95
        )


def test_calculate_sample_design_with_valid_data():
    """Test sample design calculation with valid area data."""
    area_df = pd.DataFrame(
        {
            "map_code": [1, 2, 3],
            "map_area": [10000, 20000, 30000],
            "map_edited_class": ["Forest", "Water", "Urban"],
        }
    )

    result = calculate_sample_design(
        area_df=area_df,
        objective="Overall Accuracy",
        target_oa=0.85,
        allowable_error=0.05,
        confidence_level=0.95,
        min_samples_per_class=5,
        allocation_method="Proportional",
    )

    assert isinstance(result, dict)
    assert len(result) == 3
    assert all(v >= 5 for v in result.values())
    assert sum(result.values()) > 0


def test_confidence_interval_wilson():
    lower, upper, moe = calculate_confidence_interval(0.85, 200, 0.95, method="wilson")
    assert 0.0 <= lower <= 1.0
    assert 0.0 <= upper <= 1.0
    assert moe >= 0


def test_calculate_sample_design_override():
    """When passing total_samples_override, allocation should sum to that value after constraints."""
    area_df = pd.DataFrame(
        {
            "map_code": [1, 2, 3],
            "map_area": [10000, 20000, 30000],
            "map_edited_class": ["Forest", "Water", "Urban"],
        }
    )
    alloc = calculate_sample_design(
        area_df=area_df,
        objective="Overall Accuracy",
        target_oa=0.85,
        allowable_error=0.05,
        confidence_level=0.95,
        min_samples_per_class=5,
        allocation_method="Proportional",
        total_samples_override=90,
    )
    total = sum(alloc.values())
    assert total >= 90


def test_calculate_sample_design_zero_error():
    """Test that zero error in sample design raises appropriate error."""
    area_df = pd.DataFrame(
        {
            "map_code": [1, 2],
            "map_area": [10000, 20000],
            "map_edited_class": ["Forest", "Water"],
        }
    )

    with pytest.raises(ValueError):
        calculate_sample_design(
            area_df=area_df,
            objective="Overall Accuracy",
            target_oa=0.85,
            allowable_error=0.0,
            confidence_level=0.95,
            min_samples_per_class=5,
            allocation_method="Proportional",
        )


def test_balanced_allocation():
    """Test balanced allocation is between equal and proportional."""
    area_df = pd.DataFrame(
        {
            "map_code": [1, 2, 3],
            "map_area": [1000, 5000, 10000],
        }
    )

    total_samples = 1000
    balanced = allocate_samples_balanced(area_df, total_samples)

    # Should have 3 classes
    assert len(balanced) == 3

    # Total should sum to approximately total_samples (may vary due to rounding)
    assert abs(sum(balanced.values()) - total_samples) < 10

    # For class with largest area (code 3), balanced should be between:
    # - Equal: 1000/3 = 333.33
    # - Proportional: 1000 * (10000/16000) = 625
    # - Balanced: (333.33 + 625) / 2 = 479.16
    assert 450 < balanced[3] < 510

    # For class with smallest area (code 1), balanced should be between:
    # - Equal: 333.33
    # - Proportional: 1000 * (1000/16000) = 62.5
    # - Balanced: (333.33 + 62.5) / 2 = 197.92
    assert 180 < balanced[1] < 220


def test_balanced_allocation_in_sample_design():
    """Test that balanced allocation works in calculate_sample_design."""
    area_df = pd.DataFrame(
        {
            "map_code": [1, 2, 3],
            "map_area": [1000, 5000, 10000],
            "map_edited_class": ["Rare", "Medium", "Common"],
        }
    )

    result = calculate_sample_design(
        area_df=area_df,
        objective="Overall Accuracy",
        target_oa=0.85,
        allowable_error=0.05,
        confidence_level=0.95,
        min_samples_per_class=10,
        allocation_method="Balanced",
    )

    # Should have allocation for all classes
    assert len(result) == 3

    # All should meet minimum
    assert all(v >= 10 for v in result.values())

    # Balanced should give more samples to larger classes but not as extreme as proportional
    assert result[3] > result[2] > result[1]
