# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_economist

from coreason_economist.rates import ToolRate


def test_tool_rate_creation():
    """Test creating a ToolRate."""
    rate = ToolRate(cost_per_call=0.01)
    assert rate.cost_per_call == 0.01


def test_tool_rate_immutability():
    """Test that ToolRate is frozen."""
    rate = ToolRate(cost_per_call=0.01)
    try:
        rate.cost_per_call = 0.02
        raise AssertionError("Should not be able to modify frozen model")
    except Exception:
        pass
